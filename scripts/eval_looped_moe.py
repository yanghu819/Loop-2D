"""Exact Distributed Math & Science Evaluator for 26.4B Looped MoE (`AttnRes2DLoopedMoE`).

Evaluates AIME 24/25/26, AMC 23, MATH-500, and OOD PhD Science (GPQA-Diamond) at 32k context
using exact chunked KV + Conv1D caching, active-slot batch compaction, and optional
`--carry-cut` (`H_t <- 0`) causal ablation on the exact same checkpoint weights.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import re
import sys
import time

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch
from transformers import AutoTokenizer

from loop2d.looped_moe import AttnRes2DLoopedMoE, generate_batch_looped_moe
from scripts.eval_dense_math import grade_answer, load_benchmark


def load_gpqa_diamond(path: Path) -> list[dict]:
    """Load official GPQA-Diamond (198 PhD science questions) with deterministic A-D shuffle."""
    rng = random.Random(20261126)
    letters = ["A", "B", "C", "D"]
    items: list[dict] = []
    with path.open() as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            r = json.loads(line)
            opts = [
                (r["Correct Answer"].strip(), True),
                (r["Incorrect Answer 1"].strip(), False),
                (r["Incorrect Answer 2"].strip(), False),
                (r["Incorrect Answer 3"].strip(), False),
            ]
            rng.shuffle(opts)
            gt = [letters[k] for k, (_, ok) in enumerate(opts) if ok][0]
            opt_str = "\n".join(f"({letters[k]}) {txt}" for k, (txt, _) in enumerate(opts))
            prompt = (
                f"{r['Question'].strip()}\n\n{opt_str}\n\n"
                "Please reason step by step, and put your final choice letter (A, B, C, or D) within \\boxed{}."
            )
            items.append({"id": f"gpqa_diamond_{i}", "bench": "gpqa_diamond", "prompt": prompt, "gt": gt})
    return items


def grade_mcq(text: str, gt: str) -> bool:
    gt = gt.strip().upper()
    boxed = re.findall(r"\\boxed\{([^}]*)\}", text)
    if boxed:
        inner = re.sub(r"[^A-Da-d]", "", boxed[-1]).upper()
        if inner:
            return inner[-1] == gt
    m = re.findall(r"(?:FINAL|Answer|answer|Option|option)\s*[:is]*\s*\(?([A-D])\)?", text)
    if m:
        return m[-1].upper() == gt
    m2 = re.findall(r"\b([A-D])\b", text[-120:])
    return bool(m2) and m2[-1].upper() == gt


def extract_answer_section(decoded_text: str) -> str:
    marker = "<|content_text|>"
    if marker in decoded_text:
        after = decoded_text.rsplit(marker, 1)[-1]
        after = after.replace("<|end_message|>", "").replace("<|content_model_end_sampling|>", "").strip()
        if after:
            return after
    return decoded_text


def main() -> None:
    p = argparse.ArgumentParser(description="Exact Math & GPQA Evaluator for 26.4B Looped MoE 2D-AttnRes")
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--bench-dir", type=Path, required=True, help="Directory containing benchmark jsonl/parquet files")
    p.add_argument("--gpqa-path", type=Path, default=None, help="Path to gpqa_diamond train.jsonl (required for gpqa_diamond)")
    p.add_argument("--mode", choices=list(AttnRes2DLoopedMoE.VALID_MODES), default="attnres_2d")
    p.add_argument("--loop-passes", type=int, default=1)
    p.add_argument("--benchmarks", default="aime24,aime25,aime26,amc23")
    p.add_argument("--samples-per-prob", type=int, default=8)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--max-new-tokens", type=int, default=32768)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--top-k", type=int, default=0)
    p.add_argument("--reasoning-effort", default="high")
    p.add_argument("--carry-cut", action="store_true", help="Zero out horizontal carry H_t <- 0 on the same weights")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--rank", type=int, default=int(os.environ.get("RANK", 0)))
    p.add_argument("--world", type=int, default=int(os.environ.get("WORLD_SIZE", 1)))
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_f = args.out_dir / f"rank{args.rank}.json"
    if out_f.exists() and out_f.stat().st_size > 128:
        print(f"[eval_looped_moe] rank={args.rank} SKIP (already complete: {out_f})", flush=True)
        return

    device = torch.device(f"cuda:{args.rank % torch.cuda.device_count()}")
    torch.cuda.set_device(device)

    tok = AutoTokenizer.from_pretrained(str(args.ckpt))
    eos_ids = {200006, 199999}
    if tok.eos_token_id is not None:
        eos_ids.add(int(tok.eos_token_id))

    t_load = time.perf_counter()
    model = AttnRes2DLoopedMoE.from_bf16_dir(
        args.ckpt, mode=args.mode, loop_num_passes=args.loop_passes,
        grad_checkpoint=False, attn_backend="eager", dtype=torch.bfloat16, device=device,
    ).eval()
    print(f"[eval_looped_moe] rank={args.rank}/{args.world} mode={args.mode} P={args.loop_passes} cut={args.carry_cut} loaded in {time.perf_counter()-t_load:.1f}s", flush=True)

    benches = [b.strip() for b in args.benchmarks.split(",") if b.strip()]
    tasks = []
    for b in benches:
        if b == "gpqa_diamond":
            if args.gpqa_path is None:
                raise ValueError("--gpqa-path is required when evaluating gpqa_diamond")
            items = load_gpqa_diamond(args.gpqa_path)
        else:
            items = load_benchmark(b, args.bench_dir)
        for it in items:
            q = it["prompt"].strip()
            if "\\boxed" not in q:
                q = q + "\n\nPlease reason step by step, and put your final answer within \\boxed{}."
            rendered = tok.apply_chat_template(
                [{"role": "user", "content": q}],
                tokenize=False, add_generation_prompt=True, reasoning_effort=args.reasoning_effort,
            )
            p_toks = tok.encode(rendered, add_special_tokens=False)
            for s_idx in range(args.samples_per_prob):
                tasks.append({"id": it["id"], "bench": b, "sample_idx": s_idx, "gt": it["gt"], "prompt_toks": p_toks})

    my_tasks = tasks[args.rank :: args.world]
    eff_batch = min(args.batch, 48)
    partial_f = args.out_dir / f"rank{args.rank}.partial.json"
    records: list[dict] = []
    start_b0 = 0
    if partial_f.exists():
        try:
            prev = json.loads(partial_f.read_text()).get("records", [])
            if isinstance(prev, list) and len(prev) > 0 and len(prev) % eff_batch == 0:
                records, start_b0 = prev, len(prev)
        except Exception:
            records, start_b0 = [], 0

    t0 = time.perf_counter()
    for b0 in range(start_b0, len(my_tasks), eff_batch):
        chunk = my_tasks[b0 : b0 + eff_batch]
        torch.manual_seed(args.seed + args.rank * 100000 + b0)
        gens, fins = generate_batch_looped_moe(
            model, [c["prompt_toks"] for c in chunk], eos_ids,
            max_new_tokens=args.max_new_tokens, temperature=args.temperature,
            top_k=args.top_k, top_p=args.top_p, carry_cut=args.carry_cut, device=device,
        )
        for item, g_ids, fin in zip(chunk, gens, fins):
            raw_txt = tok.decode(g_ids, skip_special_tokens=False)
            ans_txt = extract_answer_section(raw_txt)
            if item["bench"] == "gpqa_diamond":
                ok = grade_mcq(ans_txt, item["gt"]) if fin else False
                if (not ok) and fin and (ans_txt != raw_txt):
                    ok = grade_mcq(raw_txt, item["gt"])
            else:
                ok = grade_answer(ans_txt, item["gt"], fin)
                if (not ok) and (ans_txt != raw_txt):
                    ok = grade_answer(raw_txt, item["gt"], fin)
            records.append({
                "id": item["id"], "bench": item["bench"], "sample_idx": item["sample_idx"],
                "correct": bool(ok), "finished": bool(fin), "gen_tokens": len(g_ids),
            })
        n_done = len(records)
        acc = 100.0 * sum(r["correct"] for r in records) / n_done
        avg_l = sum(r["gen_tokens"] for r in records) / n_done
        tmp_p = args.out_dir / f"rank{args.rank}.partial.tmp"
        tmp_p.write_text(json.dumps({"records": records}))
        tmp_p.replace(partial_f)
        print(f"  [rank {args.rank}] {n_done}/{len(my_tasks)} | acc={acc:.1f}% | avg_len={avg_l:.0f} tok | {time.perf_counter()-t0:.0f}s", flush=True)

    tmp_f = args.out_dir / f"rank{args.rank}.json.tmp"
    tmp_f.write_text(json.dumps({"records": records}))
    tmp_f.replace(out_f)
    if partial_f.exists():
        partial_f.unlink()
    print(f"[eval_looped_moe] rank={args.rank} DONE -> {out_f}", flush=True)


if __name__ == "__main__":
    main()
