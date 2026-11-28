"""Single-Root FSDP2 Trainer for 26.4B Looped MoE (`TML-d24-varloop`) + Unified 2D-AttnRes.

Architecture highlights:
  - Only 2,448 added scalars (9.8 KB fp32 at P=1; 2,520 = 10.1 KB at P=2), +0 B KV cache.
  - 100% standard AdamW optimization (zero custom optimizers, zero extra matrices).
  - Zero-CPU-RAM meta-init -> checkpoint_wrapper inside fully_shard -> single-root FSDP2.
  - Checkpointed ChunkedSoftcapLMHead computes 200k-vocab softcap=30 CE in ~0.8 GB VRAM.
  - Exact sub-readout K-pass Jacobi (K=4, backprop_last=2): passes 1..K-1 stop at layer 18
    (3L/4), skipping Postlude 18..23 and LM head with zero approximation error.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from safetensors.torch import load_file, save_file
import torch
import torch.distributed as dist
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict, set_model_state_dict
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from loop2d.looped_moe import AttnRes2DLoopedMoE, Inkling2Config, _softcap
from loop2d.pool import init_multiscale_lambda_logits
from scripts.train_dense import PackedSFTDataset


def init_dist() -> tuple[int, int, int, torch.device]:
    if "RANK" in os.environ:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        world = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
    else:
        rank, world, local_rank = 0, 1, 0
    torch.cuda.set_device(local_rank)
    return rank, world, local_rank, torch.device("cuda", local_rank)


def is_main(rank: int) -> bool:
    return rank == 0


def clean_to_wrapped_key(k: str) -> str:
    return f"m.{k}"


def wrapped_to_clean_key(k: str) -> str:
    if k.startswith("m."):
        k = k[2:]
    return k.replace("._checkpoint_wrapped_module.", ".")


class ChunkedSoftcapLMHead(nn.Module):
    """Exact shifted CE with softcap=30.0 in O(chunk_size * V) memory, FSDP2-safe."""

    def __init__(self, lm_head: nn.Linear, softcap: float, chunk_size: int = 1024):
        super().__init__()
        self.weight = lm_head.weight
        self.softcap = softcap
        self.chunk_size = chunk_size

    def forward(self, h_final: torch.Tensor, labels: torch.LongTensor) -> tuple[torch.Tensor, int]:
        h_flat = h_final[:, :-1, :].reshape(-1, h_final.shape[-1])
        y_flat = labels[:, 1:].reshape(-1)
        valid = y_flat != -100
        n_valid = int(valid.sum().item())
        if n_valid == 0:
            return h_flat.sum() * 0.0, 0
        h_v, y_v = h_flat[valid], y_flat[valid]
        w, sc = self.weight, self.softcap

        def _ce_chunk(hc: torch.Tensor, yc: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
            return F.cross_entropy(_softcap(F.linear(hc, weight).float(), sc), yc, reduction="sum")

        loss_sum = h_v.new_zeros((), dtype=torch.float32)
        use_ckpt = torch.is_grad_enabled()
        for c0 in range(0, n_valid, self.chunk_size):
            hc, yc = h_v[c0 : c0 + self.chunk_size], y_v[c0 : c0 + self.chunk_size]
            if use_ckpt:
                loss_sum = loss_sum + checkpoint(_ce_chunk, hc, yc, w, use_reentrant=False)
            else:
                loss_sum = loss_sum + _ce_chunk(hc, yc, w)
        return loss_sum / float(n_valid), n_valid


class LoopedMoEFSDPWrapper(nn.Module):
    def __init__(self, m: AttnRes2DLoopedMoE, chunk_size: int = 1024):
        super().__init__()
        self.m = m
        self.m.lm_head = ChunkedSoftcapLMHead(m.lm_head, m.cfg.final_logit_softcapping, chunk_size=chunk_size)

    def forward(
        self,
        input_ids: torch.LongTensor,
        labels: torch.LongTensor,
        doc_start_mask: torch.Tensor | None,
        passes: int = 4,
        backprop_last: int = 2,
    ) -> tuple[torch.Tensor, int]:
        h = self.m.forward_multipass(
            input_ids, passes=passes, backprop_last=backprop_last,
            doc_start_mask=doc_start_mask, return_hidden=True,
        )["hidden"]
        return self.m.lm_head(h, labels)


@torch.no_grad()
def evaluate_multipass_fsdp(
    wrap: LoopedMoEFSDPWrapper,
    val_ds: PackedSFTDataset,
    k_list: list[int],
    n_seqs: int,
    rank: int,
    world: int,
    device: torch.device,
) -> dict[str, float]:
    wrap.eval()
    ks = [1] if not wrap.m.has_carry else k_list
    n_eval = max(world, (min(n_seqs, val_ds.n_seqs) // world) * world)
    per_rank = n_eval // world

    nll_sums = {k: 0.0 for k in ks}
    tok_count = 0
    for j in range(per_rank):
        idx = (rank * per_rank + j) % val_ds.n_seqs
        ids, lbl, doc = val_ds.get_batch([idx], device)
        for k in ks:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                loss_k, n_tok = wrap(ids, lbl, doc, passes=k, backprop_last=1)
            nll_sums[k] += float(loss_k.item()) * n_tok
            if k == ks[0]:
                tok_count += n_tok

    vec = torch.tensor([nll_sums[k] for k in ks] + [float(tok_count)], dtype=torch.float64, device=device)
    if dist.is_initialized():
        dist.all_reduce(vec, op=dist.ReduceOp.SUM)
    tot_tok = max(1.0, float(vec[-1].item()))
    res = {f"nll@K{k}": float(vec[i].item() / tot_tok) for i, k in enumerate(ks)}
    res["val_asst_tokens"] = int(tot_tok)
    if wrap.m.has_carry and 4 in ks and 8 in ks:
        res["flatness_K8_minus_K4"] = res["nll@K8"] - res["nll@K4"]
    wrap.train()
    return res


def save_fsdp_bf16_checkpoint(
    out_dir: Path,
    wrap: LoopedMoEFSDPWrapper,
    src_ckpt_dir: Path,
    rank: int,
    step: int,
    tag: str,
) -> None:
    opts = StateDictOptions(full_state_dict=True, cpu_offload=True)
    full_sd = get_model_state_dict(wrap, options=opts)
    if dist.is_initialized():
        dist.barrier()
    if not is_main(rank):
        return

    ckpt_dir = out_dir / f"ckpt_{tag}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    extra_state: dict[str, torch.Tensor] = {}
    base_tensors: dict[str, torch.Tensor] = {}
    fp32_sub = (
        "e_score_correction_bias", "mlp.gate.weight", "post_attention_convolution",
        "post_mlp_convolution", "sconv.conv1d", "attention_input", "mlp_input",
        "final_input", "message_input", "message_gate_proj", "forget_gate_proj",
    )
    for raw_k, v in full_sd.items():
        k = wrapped_to_clean_key(raw_k)
        if k.startswith(AttnRes2DLoopedMoE.EXTRA_PREFIXES):
            extra_state[k] = v.detach().float().cpu().contiguous()
        elif any(s in k for s in fp32_sub):
            base_tensors[k] = v.detach().float().cpu().contiguous()
        else:
            base_tensors[k] = v.detach().to(torch.bfloat16).cpu().contiguous()

    max_bytes = int(4.5 * (1024 ** 3))
    shards: list[dict[str, torch.Tensor]] = []
    cur: dict[str, torch.Tensor] = {}
    cur_b, tot_b = 0, 0
    for k in sorted(base_tensors.keys()):
        t = base_tensors[k]
        nb = t.numel() * t.element_size()
        tot_b += nb
        if cur and cur_b + nb > max_bytes:
            shards.append(cur)
            cur, cur_b = {}, 0
        cur[k] = t
        cur_b += nb
    if cur:
        shards.append(cur)

    wmap: dict[str, str] = {}
    for idx, sd in enumerate(shards, start=1):
        fn = f"model-{idx:05d}-of-{len(shards):05d}.safetensors"
        save_file(sd, str(ckpt_dir / fn))
        for k in sd:
            wmap[k] = fn
    (ckpt_dir / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": tot_b}, "weight_map": wmap}, indent=2)
    )
    torch.save(
        {
            "mode": wrap.m.mode,
            "loop_num_passes": wrap.m.loop_passes,
            "step": step,
            "attnres_2d_state": extra_state,
        },
        ckpt_dir / "attnres_2d_extra.pt",
    )
    for fn in ["config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja", "generation_config.json"]:
        src_f, dst_f = src_ckpt_dir / fn, ckpt_dir / fn
        if src_f.exists() and not dst_f.exists():
            shutil.copy2(src_f, dst_f)
    if (src_ckpt_dir / "tiktoken").exists() and not (ckpt_dir / "tiktoken").exists():
        shutil.copytree(src_ckpt_dir / "tiktoken", ckpt_dir / "tiktoken")


def main() -> None:
    p = argparse.ArgumentParser(description="Train 26.4B Looped MoE with 2D-AttnRes via FSDP2")
    p.add_argument("--model", type=Path, required=True, help="BF16 TML-d24-varloop checkpoint dir")
    p.add_argument("--data-dir", type=Path, required=True, help="Packed SFT dataset dir")
    p.add_argument("--out-dir", type=Path, required=True, help="Output directory")
    p.add_argument("--mode", choices=list(AttnRes2DLoopedMoE.VALID_MODES), default="attnres_2d")
    p.add_argument("--loop-passes", type=int, default=1, help="Vertical loop passes P (1..5)")
    p.add_argument("--passes", type=int, default=4, help="Jacobi passes K")
    p.add_argument("--backprop-last", type=int, default=2)
    p.add_argument("--lr-base", type=float, default=5e-6)
    p.add_argument("--lr-attnres-q", type=float, default=1e-4)
    p.add_argument("--lr-lambda", type=float, default=3e-4)
    p.add_argument("--lr-scalar", type=float, default=1e-2)
    p.add_argument("--msg0-pool-init", type=float, default=0.35)
    p.add_argument("--freeze-base-steps", type=int, default=10)
    p.add_argument("--total-tokens", type=int, default=26_000_000)
    p.add_argument("--global-seqs-per-step", type=int, default=16)
    p.add_argument("--eval-every", type=int, default=30)
    p.add_argument("--save-every", type=int, default=100)
    p.add_argument("--val-seqs", type=int, default=64)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rank, world, local_rank, device = init_dist()
    torch.manual_seed(args.seed + rank)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    train_ds = PackedSFTDataset(args.data_dir, "train")
    val_ds = PackedSFTDataset(args.data_dir, "val")
    toks_per_step = args.global_seqs_per_step * train_ds.seq_len
    total_steps = max(1, round(args.total_tokens / toks_per_step))
    assert args.global_seqs_per_step % world == 0
    grad_accum = args.global_seqs_per_step // world

    if is_main(rank):
        print(
            f"[train_looped_moe] mode={args.mode} P={args.loop_passes} K={args.passes}(last {args.backprop_last}) "
            f"world={world} grad_accum={grad_accum} total_steps={total_steps}",
            flush=True,
        )
        (args.out_dir / "run_args.json").write_text(json.dumps(vars(args), indent=2, default=str))

    t_load = time.perf_counter()
    cfg = Inkling2Config.from_json(args.model / "config.json", loop_num_passes=args.loop_passes)
    with torch.device("meta"):
        raw_model = AttnRes2DLoopedMoE(cfg, mode=args.mode, msg0_pool_init=args.msg0_pool_init, grad_checkpoint=False, attn_backend="flex")
        wrap = LoopedMoEFSDPWrapper(raw_model, chunk_size=1024)

    mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    extra_params = {
        param for name, param in wrap.named_parameters()
        if wrapped_to_clean_key(name).startswith(AttnRes2DLoopedMoE.EXTRA_PREFIXES)
    }

    for i in range(len(wrap.m.model.layers)):
        wrap.m.model.layers[i] = checkpoint_wrapper(wrap.m.model.layers[i])
        fully_shard(wrap.m.model.layers[i], mp_policy=mp, reshard_after_forward=True)
    fully_shard(wrap.m.model.embed_tokens, mp_policy=mp, reshard_after_forward=True)
    fully_shard(wrap.m.lm_head, mp_policy=mp, reshard_after_forward=True)
    fully_shard(wrap, mp_policy=mp, reshard_after_forward=True, ignored_params=extra_params)
    wrap.to_empty(device=device)

    full_sd: dict[str, torch.Tensor] = {}
    if is_main(rank):
        idx = json.loads((args.model / "model.safetensors.index.json").read_text())
        for sf in sorted(set(idx["weight_map"].values())):
            shard = load_file(str(args.model / sf), device="cpu")
            for ck, t in shard.items():
                full_sd[clean_to_wrapped_key(ck)] = t
        if wrap.m.has_carry:
            ep = args.model / "attnres_2d_extra.pt"
            if ep.exists():
                extra = torch.load(ep, map_location="cpu", weights_only=False)
                for ck, t in (extra.get("attnres_2d_state") or {}).items():
                    full_sd[clean_to_wrapped_key(ck)] = t
            else:
                init_extra = {
                    "g_h_pre_att": torch.zeros(wrap.m.loop_start - 1, dtype=torch.float32),
                    "g_h_pre_mlp": torch.zeros(wrap.m.loop_start, dtype=torch.float32),
                    "g_h_core_att": torch.zeros(wrap.m.core_len, wrap.m.loop_passes, dtype=torch.float32),
                    "g_h_core_mlp": torch.zeros(wrap.m.core_len, wrap.m.loop_passes, dtype=torch.float32),
                    "g_h_post_att": torch.zeros(wrap.m.post_len, dtype=torch.float32),
                    "g_h_post_mlp": torch.zeros(wrap.m.post_len, dtype=torch.float32),
                    "g_h_final": torch.zeros((), dtype=torch.float32),
                    "g_h_msg0": torch.zeros(wrap.m.core_len, 4, dtype=torch.float32),
                    "g_h_msg0_pool": torch.full((wrap.m.core_len, 4), float(args.msg0_pool_init), dtype=torch.float32),
                    "theta_lambda": init_multiscale_lambda_logits(cfg.hidden_size, 0.25, 0.95),
                }
                if wrap.m.loop_passes > 1:
                    init_extra["g_h_msg_pool"] = torch.zeros(wrap.m.core_len, wrap.m.loop_passes - 1, 4, dtype=torch.float32)
                for ck, t in init_extra.items():
                    full_sd[clean_to_wrapped_key(ck)] = t

    set_model_state_dict(
        wrap,
        model_state_dict=full_sd,
        options=StateDictOptions(full_state_dict=True, broadcast_from_rank0=True, strict=True),
    )
    del full_sd
    if dist.is_initialized():
        dist.barrier()
    if is_main(rank):
        print(f"[FSDP2] broadcast 26.4B model across {world} GPUs in {time.perf_counter()-t_load:.1f}s | added_params={wrap.m.count_added_params():,}", flush=True)

    replicated_params: list[nn.Parameter] = []
    base_decay, base_nodecay, attnres_q_p, lambda_p, scalar_p = [], [], [], [], []
    attnres_q_sub = ("attention_input.query", "mlp_input.query", "final_input.query", "message_input.query")
    for raw_name, param in wrap.named_parameters():
        if not param.requires_grad:
            continue
        name = wrapped_to_clean_key(raw_name)
        if param in extra_params:
            replicated_params.append(param)
        if name.startswith("g_h_"):
            scalar_p.append(param)
        elif name == "theta_lambda":
            lambda_p.append(param)
        elif args.lr_attnres_q > 0.0 and any(s in name for s in attnres_q_sub):
            attnres_q_p.append(param)
        elif param.ndim >= 2 and "norm" not in name and "embed" not in name:
            base_decay.append(param)
        else:
            base_nodecay.append(param)

    adam_base_groups = [
        {"params": base_decay, "lr": args.lr_base, "weight_decay": 0.01, "kind": "base"},
        {"params": base_nodecay, "lr": args.lr_base, "weight_decay": 0.0, "kind": "base"},
    ]
    if attnres_q_p:
        adam_base_groups.append({"params": attnres_q_p, "lr": args.lr_attnres_q, "weight_decay": 0.0, "kind": "attnres_q"})
    adam_extra_groups = []
    if lambda_p:
        adam_extra_groups.append({"params": lambda_p, "lr": args.lr_lambda, "weight_decay": 0.0, "kind": "lambda"})
    if scalar_p:
        adam_extra_groups.append({"params": scalar_p, "lr": args.lr_scalar, "weight_decay": 0.0, "kind": "scalar"})

    opt_base = torch.optim.AdamW(adam_base_groups, betas=(0.9, 0.95), eps=1e-8)
    opt_extra = torch.optim.AdamW(adam_extra_groups, betas=(0.9, 0.95), eps=1e-8) if adam_extra_groups else None

    ev0 = evaluate_multipass_fsdp(wrap, val_ds, [1, 2, 4, 8], args.val_seqs, rank, world, device)
    log_path = args.out_dir / "metrics.jsonl"
    if is_main(rank):
        ev0["step"], ev0["tokens"] = 0, 0
        print(f"[eval step 0] {json.dumps(ev0)}", flush=True)
        with log_path.open("a") as f:
            f.write(json.dumps({"eval": ev0, "telemetry": wrap.m.collect_telemetry()}) + "\n")

    warmup = max(10, int(0.05 * total_steps))
    decay_start = int(0.80 * total_steps)

    def lr_mult(step: int) -> float:
        if step < warmup:
            return float(step + 1) / float(warmup)
        if step < decay_start:
            return 1.0
        prog = (step - decay_start) / max(1, total_steps - decay_start)
        return 0.01 + 0.99 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))

    t_run = time.perf_counter()
    wrap.train()
    ev = ev0

    for step in range(total_steps):
        t0 = time.perf_counter()
        step_1 = step + 1
        mult = lr_mult(step)
        freeze_base = wrap.m.has_carry and (step < args.freeze_base_steps)

        for g in opt_base.param_groups:
            blr = args.lr_attnres_q if g["kind"] == "attnres_q" else args.lr_base
            g["lr"] = 0.0 if freeze_base else (blr * mult)
        if opt_extra is not None:
            for g in opt_extra.param_groups:
                blr = args.lr_lambda if g["kind"] == "lambda" else args.lr_scalar
                g["lr"] = blr * mult

        opt_base.zero_grad(set_to_none=True)
        if opt_extra is not None:
            opt_extra.zero_grad(set_to_none=True)

        step_loss = 0.0
        for ga in range(grad_accum):
            s_idx = (step * args.global_seqs_per_step + ga * world + rank) % train_ds.n_seqs
            ids, lbl, doc = train_ds.get_batch([s_idx], device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
                loss, _ = wrap(
                    ids, lbl, doc,
                    passes=args.passes if wrap.m.has_carry else 1,
                    backprop_last=args.backprop_last if wrap.m.has_carry else 1,
                )
            (loss / float(grad_accum)).backward()
            step_loss += float(loss.item()) / float(grad_accum)

        if dist.is_initialized():
            for rp in replicated_params:
                if rp.grad is None:
                    rp.grad = torch.zeros_like(rp)
                dist.all_reduce(rp.grad, op=dist.ReduceOp.AVG)

        if opt_extra is not None:
            eps_ps = [p for g in opt_extra.param_groups for p in g["params"] if p.grad is not None]
            if eps_ps:
                torch.nn.utils.clip_grad_norm_(eps_ps, 1.0)
            opt_extra.step()
        if not freeze_base:
            base_ps = [p for g in opt_base.param_groups for p in g["params"] if p.grad is not None]
            gnorm = float(torch.nn.utils.clip_grad_norm_(base_ps, 1.0).item()) if base_ps else 0.0
            opt_base.step()
        else:
            opt_base.zero_grad(set_to_none=True)
            gnorm = 0.0

        dt = time.perf_counter() - t0
        if step_1 % args.log_every == 0 or step_1 == 1 or step_1 == total_steps:
            lt = torch.tensor([step_loss], device=device)
            if dist.is_initialized():
                dist.all_reduce(lt, op=dist.ReduceOp.AVG)
            avg_loss = float(lt.item())
            if is_main(rank):
                mem_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
                tps = toks_per_step / max(1e-6, dt)
                rec = {
                    "step": step_1, "tokens": step_1 * toks_per_step, "loss": round(avg_loss, 5),
                    "freeze_base": freeze_base, "gnorm": round(gnorm, 4), "lr_mult": round(mult, 4),
                    "step_s": round(dt, 2), "tok_per_s": round(tps, 1), "mem_gib": round(mem_gb, 2),
                }
                print(
                    f"step {step_1:4d}/{total_steps} | tok={step_1*toks_per_step/1e6:6.1f}M | loss={avg_loss:.4f} "
                    f"| frz={int(freeze_base)} gn={gnorm:.3f} | {dt:.1f}s ({tps/1e3:.1f}k tok/s) | mem={mem_gb:.1f}GiB",
                    flush=True,
                )
                with log_path.open("a") as f:
                    f.write(json.dumps(rec) + "\n")

        if step_1 % args.eval_every == 0 or step_1 == total_steps:
            ev = evaluate_multipass_fsdp(wrap, val_ds, [1, 2, 4, 8], args.val_seqs, rank, world, device)
            if is_main(rank):
                tel = wrap.m.collect_telemetry()
                ev["step"], ev["tokens"] = step_1, step_1 * toks_per_step
                print(f"[eval step {step_1}] {json.dumps(ev)} | tel={json.dumps(tel)}", flush=True)
                with log_path.open("a") as f:
                    f.write(json.dumps({"eval": ev, "telemetry": tel}) + "\n")

        if step_1 % args.save_every == 0 or step_1 == total_steps:
            tag = "final" if step_1 == total_steps else f"step{step_1}"
            save_fsdp_bf16_checkpoint(args.out_dir, wrap, args.model, rank, step_1, tag=tag)
            if is_main(rank):
                print(f"[ckpt] saved ckpt_{tag}", flush=True)

    if is_main(rank):
        summary = {
            "mode": args.mode, "loop_passes": args.loop_passes, "passes": args.passes,
            "total_steps": total_steps, "total_tokens": total_steps * toks_per_step,
            "wall_s": round(time.perf_counter() - t_run, 1),
            "final_eval": ev, "telemetry": wrap.m.collect_telemetry(),
        }
        (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        print(f"=== TRAIN COMPLETE ({summary['wall_s']}s) ===", flush=True)

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
