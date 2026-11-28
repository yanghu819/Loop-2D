"""Exact Autoregressive Math Benchmark Evaluator for Dense 2D-AttnRes (`AttnRes2DDense`).

Supports:
  - Benchmarks: `aime24`, `aime25`, `aime26`, `amc23`, `math500`, `gsm8k`.
  - Exact FlashDecoding (`flash_attn_with_kvcache`) 1-token recurrence with zero extra KV.
  - `--carry-cut` causal diagnostic (sets `H_t <- 0` on the exact same trained weights).
  - Exact `math_verify` symbolic answer grading.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
import time

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

try:
    from flash_attn import flash_attn_with_kvcache
except ImportError:
    flash_attn_with_kvcache = None  # type: ignore

try:
    from math_verify import parse as mv_parse, verify as mv_verify
except ImportError:
    mv_parse = None  # type: ignore
    mv_verify = None  # type: ignore

import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from loop2d.dense_qwen3 import AttnRes2DDense
from loop2d.pool import sample_next_token


def _rotate_half_1tok(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


@torch.compile(mode="default", dynamic=False)
def _rms_norm_fp32(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)).to(x.dtype) * w


@torch.compile(mode="default", dynamic=False)
def _qkv_norm_rope_compiled(
    x_norm: torch.Tensor,
    w_qkv: torch.Tensor,
    q_norm_w: torch.Tensor,
    k_norm_w: torch.Tensor,
    cos_t: torch.Tensor,
    sin_t: torch.Tensor,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
    eps: float,
):
    qkv = F.linear(x_norm, w_qkv)
    q, k, v = qkv.split((n_heads * head_dim, n_kv_heads * head_dim, n_kv_heads * head_dim), dim=-1)
    bsz = x_norm.shape[0]
    q = F.rms_norm(q.view(bsz, 1, n_heads, head_dim), (head_dim,), q_norm_w, eps)
    k = F.rms_norm(k.view(bsz, 1, n_kv_heads, head_dim), (head_dim,), k_norm_w, eps)
    v = v.view(bsz, 1, n_kv_heads, head_dim)
    q = (q * cos_t) + (_rotate_half_1tok(q) * sin_t)
    k = (k * cos_t) + (_rotate_half_1tok(k) * sin_t)
    return q, k, v


@torch.compile(mode="default", dynamic=False)
def _mlp_out_compiled(x_norm: torch.Tensor, w_gu: torch.Tensor, w_d: torch.Tensor) -> torch.Tensor:
    gate, up = F.linear(x_norm, w_gu).chunk(2, dim=-1)
    return F.linear(F.silu(gate) * up, w_d)


def extract_packed_weights(model: AttnRes2DDense) -> dict:
    base = model.base
    cfg = base.config
    dtype = torch.bfloat16
    layers = []
    for blk in base.model.layers:
        att, mlp = blk.self_attn, blk.mlp
        layers.append({
            "norm1": blk.input_layernorm.weight.detach().to(dtype).contiguous(),
            "norm2": blk.post_attention_layernorm.weight.detach().to(dtype).contiguous(),
            "q_norm": att.q_norm.weight.detach().to(dtype).contiguous(),
            "k_norm": att.k_norm.weight.detach().to(dtype).contiguous(),
            "w_qkv": torch.cat([att.q_proj.weight.detach(), att.k_proj.weight.detach(), att.v_proj.weight.detach()], dim=0).to(dtype).contiguous(),
            "w_o": att.o_proj.weight.detach().to(dtype).contiguous(),
            "w_gu": torch.cat([mlp.gate_proj.weight.detach(), mlp.up_proj.weight.detach()], dim=0).to(dtype).contiguous(),
            "w_d": mlp.down_proj.weight.detach().to(dtype).contiguous(),
        })
    return {
        "layers": layers,
        "emb_w": base.model.embed_tokens.weight.detach().to(dtype).contiguous(),
        "final_norm_w": base.model.norm.weight.detach().to(dtype).contiguous(),
        "head_w": base.lm_head.weight.detach().to(dtype).contiguous(),
        "n_layers": int(cfg.num_hidden_layers),
        "dim": int(cfg.hidden_size),
        "n_heads": int(cfg.num_attention_heads),
        "n_kv_heads": int(cfg.num_key_value_heads),
        "head_dim": int(getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)),
        "eps": float(cfg.rms_norm_eps),
        "vocab_size": int(cfg.vocab_size),
    }


def load_benchmark(name: str, bench_dir: Path) -> list[dict]:
    items: list[dict] = []
    jsonl_p = bench_dir / f"{name}.jsonl"
    parquet_p = bench_dir / f"{name}.parquet"
    if jsonl_p.exists():
        with jsonl_p.open() as f:
            for idx, line in enumerate(f):
                r = json.loads(line)
                q = r.get("problem") or r.get("question") or r.get("completion_input") or ""
                q = re.sub(r"^Question:\s*", "", q).strip()
                q = re.sub(r"\n?Answer:\s*$", "", q).strip()
                if "\\boxed" not in q and "Answer:" not in q:
                    q = q + "\nPlease reason step by step, and put your final answer within \\boxed{}."
                gt = str(r.get("ground_truth") or r.get("answer") or "").strip()
                if name == "gsm8k" and "####" in gt:
                    gt = gt.split("####")[-1].strip()
                items.append({"id": f"{name}_{idx}", "bench": name, "prompt": q, "gt": gt})
    elif parquet_p.exists():
        tbl = pq.read_table(parquet_p)
        for idx, row in enumerate(tbl.to_pylist()):
            pr = row["prompt"]
            q = pr[-1]["content"] if isinstance(pr, list) else str(pr)
            rm = row.get("reward_model") or {}
            gt = str(rm.get("ground_truth") if isinstance(rm, dict) else "").strip()
            items.append({"id": f"{name}_{idx}", "bench": name, "prompt": q.strip(), "gt": gt})
    else:
        raise FileNotFoundError(f"Benchmark file not found in {bench_dir}: expected {name}.jsonl or {name}.parquet")
    return items


def extract_answer_candidate(text: str) -> str:
    if "</think>" in text:
        text = text.split("</think>")[-1]
    idx = text.rfind("\\boxed{")
    if idx != -1:
        depth, start = 0, idx + 7
        for j in range(start, len(text)):
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                if depth == 0:
                    return text[start:j].strip()
                depth -= 1
    m = re.findall(r"(?i)Answer:\s*\$?([^\n$]+)", text)
    if m:
        return m[-1].strip()
    return text.strip()[-256:]


def grade_answer(pred_text: str, gt_str: str, finished: bool) -> bool:
    if not finished:
        return False
    cand = extract_answer_candidate(pred_text)
    if cand == gt_str:
        return True
    if mv_parse is None or mv_verify is None:
        return cand.replace(" ", "") == gt_str.replace(" ", "")
    try:
        if mv_verify(mv_parse(f"${gt_str}$"), mv_parse(f"${cand}$")):
            return True
    except Exception:
        pass
    try:
        return bool(mv_verify(mv_parse(gt_str), mv_parse(cand)))
    except Exception:
        return False


@torch.no_grad()
def generate_batch_dense_exact(
    model: AttnRes2DDense,
    packed: dict,
    prompt_token_lists: list[list[int]],
    eos_ids: set[int],
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    presence_penalty: float,
    carry_cut: bool,
    device: torch.device,
) -> tuple[list[list[int]], list[bool]]:
    if flash_attn_with_kvcache is None:
        raise ImportError("flash-attn is required for fast FlashDecoding in eval_dense_math.py (`pip install flash-attn`).")
    B = len(prompt_token_lists)
    dtype = torch.bfloat16
    D, n_heads, n_kv_heads, head_dim = packed["dim"], packed["n_heads"], packed["n_kv_heads"], packed["head_dim"]
    eps, layers, vocab_size = packed["eps"], packed["layers"], packed["vocab_size"]

    p_lens = [len(p) for p in prompt_token_lists]
    max_prompt = max(p_lens)
    max_len = max_prompt + max_new_tokens + 8

    kv_cap = min(max_len, max(4096, ((max_prompt + 2047) // 2048) * 2048))
    k_caches = [torch.zeros(B, kv_cap, n_kv_heads, head_dim, dtype=dtype, device=device) for _ in layers]
    v_caches = [torch.zeros(B, kv_cap, n_kv_heads, head_dim, dtype=dtype, device=device) for _ in layers]

    def _ensure_kv_cap(needed_len: int) -> None:
        nonlocal kv_cap
        if needed_len < kv_cap:
            return
        new_cap = min(max_len, max(needed_len + 16, kv_cap + 4096))
        for idx in range(len(k_caches)):
            nk = torch.zeros(B, new_cap, n_kv_heads, head_dim, dtype=dtype, device=device)
            nv = torch.zeros(B, new_cap, n_kv_heads, head_dim, dtype=dtype, device=device)
            nk[:, :kv_cap].copy_(k_caches[idx])
            nv[:, :kv_cap].copy_(v_caches[idx])
            k_caches[idx], v_caches[idx] = nk, nv
        kv_cap = new_cap

    pos_all = torch.arange(max_len, device=device, dtype=torch.long).unsqueeze(0)
    cos_all, sin_all = model.base.model.rotary_emb(torch.empty(1, max_len, D, dtype=dtype, device=device), pos_all)
    cos_all, sin_all = cos_all[0].to(dtype).contiguous(), sin_all[0].to(dtype).contiguous()

    h_carry = torch.zeros(B, 1, D, dtype=dtype, device=device)
    zero_carry = torch.zeros_like(h_carry)
    seqlens = torch.zeros(B, dtype=torch.int32, device=device)

    def decode_one_step(tok_ids: torch.Tensor, pos_idx: torch.Tensor) -> torch.Tensor:
        e_t = F.embedding(tok_ids.view(B, 1), packed["emb_w"])
        cos_t = cos_all.index_select(0, pos_idx).view(B, 1, 1, head_dim)
        sin_t = sin_all.index_select(0, pos_idx).view(B, 1, 1, head_dim)

        a = e_t
        if not model.has_carry:
            for ly, kc, vc in zip(layers, k_caches, v_caches):
                x1 = _rms_norm_fp32(a, ly["norm1"], eps)
                q, kn, vn = _qkv_norm_rope_compiled(x1, ly["w_qkv"], ly["q_norm"], ly["k_norm"], cos_t, sin_t, n_heads, n_kv_heads, head_dim, eps)
                ao = flash_attn_with_kvcache(q, kc, vc, k=kn, v=vn, cache_seqlens=seqlens, causal=True)
                a = a + F.linear(ao.reshape(B, 1, -1), ly["w_o"])
                a = a + _mlp_out_compiled(_rms_norm_fp32(a, ly["norm2"], eps), ly["w_gu"], ly["w_d"])
            return F.linear(_rms_norm_fp32(a, packed["final_norm_w"], eps), packed["head_w"])[:, 0, :]

        v_blocks = e_t.unsqueeze(-2)
        block_start = e_t
        cur_h = zero_carry if carry_cut else h_carry
        h_next = cur_h
        for l_idx, (ly, kc, vc) in enumerate(zip(layers, k_caches, v_caches)):
            u1 = model.read_pool(2 * l_idx, a, v_blocks, cur_h)
            x1 = _rms_norm_fp32(u1, ly["norm1"], eps)
            q, kn, vn = _qkv_norm_rope_compiled(x1, ly["w_qkv"], ly["q_norm"], ly["k_norm"], cos_t, sin_t, n_heads, n_kv_heads, head_dim, eps)
            ao = flash_attn_with_kvcache(q, kc, vc, k=kn, v=vn, cache_seqlens=seqlens, causal=True)
            a = a + F.linear(ao.reshape(B, 1, -1), ly["w_o"])

            u2 = model.read_pool(2 * l_idx + 1, a, v_blocks, cur_h)
            x2 = _rms_norm_fp32(u2, ly["norm2"], eps)
            a = a + _mlp_out_compiled(x2, ly["w_gu"], ly["w_d"])

            if (l_idx + 1) == model.l_out and (not carry_cut):
                h_next = model.update_carry(cur_h, a, v_blocks, block_start)
            if ((l_idx + 1) % model.block_size == 0) and ((l_idx + 1) < len(layers)):
                v_blocks = torch.cat([v_blocks, (a - block_start).unsqueeze(-2)], dim=-2)
                block_start = a
        a = model.read_pool(2 * len(layers), a, v_blocks, cur_h)
        if not carry_cut:
            h_carry.copy_(h_next)
        return F.linear(_rms_norm_fp32(a, packed["final_norm_w"], eps), packed["head_w"])[:, 0, :]

    last_logits = torch.zeros(B, vocab_size, dtype=dtype, device=device)
    for t in range(max_prompt):
        tok_t = torch.tensor([prompt_token_lists[i][t] if t < p_lens[i] else 0 for i in range(B)], dtype=torch.long, device=device)
        saved_h = h_carry.clone()
        logits_t = decode_one_step(tok_t, seqlens.long())
        for i in range(B):
            if t < p_lens[i]:
                seqlens[i] += 1
                if t == p_lens[i] - 1:
                    last_logits[i] = logits_t[i]
            else:
                h_carry[i] = saved_h[i]

    seen_mask = torch.zeros(B, vocab_size, dtype=torch.bool, device=device)
    gen_tokens: list[list[int]] = [[] for _ in range(B)]
    finished = [False] * B

    next_tok = sample_next_token(last_logits, seen_mask, temperature, top_k, top_p, presence_penalty)
    for i in range(B):
        tid = int(next_tok[i].item())
        gen_tokens[i].append(tid)
        seen_mask[i, tid] = True
        if tid in eos_ids:
            finished[i] = True

    for _ in range(1, max_new_tokens):
        if all(finished):
            break
        _ensure_kv_cap(int(seqlens.max().item()) + 2)
        logits_t = decode_one_step(next_tok, seqlens.long())
        seqlens.add_(1)
        next_tok = sample_next_token(logits_t, seen_mask, temperature, top_k, top_p, presence_penalty)
        for i in range(B):
            if not finished[i]:
                tid = int(next_tok[i].item())
                gen_tokens[i].append(tid)
                seen_mask[i, tid] = True
                if tid in eos_ids:
                    finished[i] = True

    del k_caches, v_caches
    torch.cuda.empty_cache()
    return gen_tokens, finished


def main() -> None:
    p = argparse.ArgumentParser(description="Exact Math Reasoning Evaluator for Dense 2D-AttnRes")
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--bench-dir", type=Path, required=True)
    p.add_argument("--benchmarks", default="aime24,aime25,amc23")
    p.add_argument("--samples", type=int, default=8)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--max-new-tokens", type=int, default=32000)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--presence-penalty", type=float, default=1.5)
    p.add_argument("--carry-cut", action="store_true", help="Zero out horizontal carry H_t <- 0 on the same weights")
    p.add_argument("--rank", type=int, default=0)
    p.add_argument("--world", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    device = torch.device("cuda:0")
    torch.manual_seed(args.seed + args.rank)
    torch._dynamo.config.cache_size_limit = 256

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    eos_ids = {tok.eos_token_id}
    for extra_eos in ("<|im_end|>", "<|endoftext|>"):
        eid = tok.convert_tokens_to_ids(extra_eos)
        if isinstance(eid, int) and eid >= 0:
            eos_ids.add(eid)

    base = AutoModelForCausalLM.from_pretrained(str(args.ckpt), torch_dtype=torch.bfloat16, attn_implementation="sdpa").to(device).eval()
    has_extra = (args.ckpt / "attnres_2d_extra.pt").exists()
    model = AttnRes2DDense(base, mode=("attnres_2d" if has_extra else "std")).to(device).eval()
    if has_extra:
        model.load_extra(args.ckpt, map_location="cpu")
    packed = extract_packed_weights(model)

    all_tasks: list[dict] = []
    for bname in [b.strip() for b in args.benchmarks.split(",") if b.strip()]:
        for item in load_benchmark(bname, args.bench_dir):
            for s_idx in range(args.samples):
                all_tasks.append({**item, "sample_idx": s_idx})
    my_tasks = all_tasks[args.rank :: args.world]

    bos_user = tok.encode("<|im_start|>user\n", add_special_tokens=False)
    mid_asst = tok.encode("<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False)

    records = []
    t0 = time.perf_counter()
    for i in range(0, len(my_tasks), args.batch):
        chunk = my_tasks[i : i + args.batch]
        p_lists = [bos_user + tok.encode(c["prompt"], add_special_tokens=False) + mid_asst for c in chunk]
        gens, fins = generate_batch_dense_exact(
            model, packed, p_lists, eos_ids,
            max_new_tokens=args.max_new_tokens, temperature=args.temperature,
            top_k=args.top_k, top_p=args.top_p, presence_penalty=args.presence_penalty,
            carry_cut=args.carry_cut, device=device,
        )
        for c, g_ids, fin in zip(chunk, gens, fins):
            txt = tok.decode(g_ids, skip_special_tokens=True)
            ok = grade_answer(txt, c["gt"], fin)
            records.append({
                "id": c["id"], "bench": c["bench"], "sample_idx": c["sample_idx"],
                "correct": bool(ok), "finished": bool(fin), "gen_tokens": len(g_ids),
            })
        acc = 100.0 * sum(r["correct"] for r in records) / len(records)
        avg_l = sum(r["gen_tokens"] for r in records) / len(records)
        print(f"  [rank {args.rank}] {len(records)}/{len(my_tasks)} | acc={acc:.1f}% | avg_len={avg_l:.0f} tok | {time.perf_counter()-t0:.0f}s", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"args": vars(args), "mode": model.mode, "records": records}, indent=2, default=str))
    print(f"[eval_dense_math] SAVED {args.out}", flush=True)


if __name__ == "__main__":
    main()
