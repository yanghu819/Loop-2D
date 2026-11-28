"""Multi-GPU DDP SFT Trainer for Dense 2D-AttnRes (`AttnRes2DDense`, e.g. Qwen3-4B-Base).

Highlights:
  - Unified 2D-AttnRes Pool + 1-Line Contractive Gate λ at 3L/4 (+0 B KV, 0.0047% params).
  - 100% standard AdamW optimization (zero custom optimizers, zero extra matrices).
  - Exact sub-readout K-pass Jacobi forward (K=4, backprop_last=2): passes 1..K-1 stop at
    3L/4, skipping the top L/4 readout layers and LM head with zero approximation error.
  - Mandatory `doc_start_mask` carry reset across packed 16k sequences.
  - Built-in K in {1, 2, 3, 4, 6, 8} held-out convergence validation + exact sequential
    parity check + automatic `--resume auto` preemption safety.
"""
from __future__ import annotations

import argparse
from datetime import timedelta
import gc
import json
import math
import os
from pathlib import Path
import sys
import time

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoModelForCausalLM

from loop2d.dense_qwen3 import AttnRes2DDense

try:
    from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
    _HAS_LIGER = True
except Exception:
    LigerFusedLinearCrossEntropyLoss = None
    _HAS_LIGER = False


def init_dist() -> tuple[int, int, int, torch.device]:
    if "RANK" in os.environ:
        dist.init_process_group("nccl", timeout=timedelta(minutes=30))
        rank = dist.get_rank()
        world = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
    else:
        rank, world, local_rank = 0, 1, 0
    torch.cuda.set_device(local_rank)
    return rank, world, local_rank, torch.device("cuda", local_rank)


def is_main(rank: int) -> bool:
    return rank == 0


def lr_mult_sqrt_decay(
    step: int,
    total_steps: int,
    warmup_frac: float = 0.03,
    decay_frac: float = 0.20,
    min_ratio: float = 0.01,
) -> float:
    """3% linear warmup, 77% constant plateau, 20% square-root decay to 1% peak LR."""
    warm = max(1, int(total_steps * warmup_frac))
    dec = max(1, int(total_steps * decay_frac))
    dec_start = max(warm, total_steps - dec)
    if step < warm:
        return (step + 1) / warm
    if step < dec_start:
        return 1.0
    progress = min(1.0, (step - dec_start) / max(1, total_steps - dec_start))
    return min_ratio + (1.0 - min_ratio) * (1.0 - math.sqrt(progress))


class PackedSFTDataset:
    """Memory-mapped packed token dataset (`{split}_ids.npy`, `_labels.npy`, `_docstarts.npy`)."""

    def __init__(self, data_dir: Path, split: str) -> None:
        self.ids = np.load(data_dir / f"{split}_ids.npy", mmap_mode="r")
        self.labels = np.load(data_dir / f"{split}_labels.npy", mmap_mode="r")
        self.docstarts = np.load(data_dir / f"{split}_docstarts.npy", mmap_mode="r")
        self.n_seqs, self.seq_len = self.ids.shape

    def get_batch(self, idxs: list[int], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ids = torch.from_numpy(np.array(self.ids[idxs], dtype=np.int64)).to(device, non_blocking=True)
        lbl = torch.from_numpy(np.array(self.labels[idxs], dtype=np.int64)).to(device, non_blocking=True)
        doc = torch.from_numpy(np.array(self.docstarts[idxs], dtype=np.bool_)).to(device, non_blocking=True)
        return ids, lbl, doc


class ChunkedLinearCELoss(nn.Module):
    """Fallback exact chunked linear cross-entropy when Liger kernel is unavailable."""

    def __init__(self, ignore_index: int = -100, chunk_size: int = 2048) -> None:
        super().__init__()
        self.ignore_index = ignore_index
        self.chunk_size = chunk_size

    def forward(self, weight: torch.Tensor, hidden: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        valid = targets != self.ignore_index
        n_valid = int(valid.sum().item())
        if n_valid == 0:
            return hidden.sum() * 0.0
        hv = hidden[valid]
        yv = targets[valid]
        loss_sum = hv.new_zeros((), dtype=torch.float32)
        for c0 in range(0, n_valid, self.chunk_size):
            logits = F.linear(hv[c0 : c0 + self.chunk_size], weight).float()
            loss_sum = loss_sum + F.cross_entropy(logits, yv[c0 : c0 + self.chunk_size], reduction="sum")
        return loss_sum / float(n_valid)


@torch.no_grad()
def evaluate_multipass(
    model: AttnRes2DDense,
    val_ds: PackedSFTDataset,
    passes_list: list[int],
    n_seqs: int,
    rank: int,
    world: int,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    lm_head_w = model.base.lm_head.weight
    n_use = min(n_seqs, val_ds.n_seqs)
    my_idxs = list(range(rank, n_use, world))

    eval_passes = [1] if not model.has_carry else passes_list
    sums = {k: 0.0 for k in eval_passes}
    toks = {k: 0 for k in eval_passes}

    for idx in my_idxs:
        ids, lbl, doc = val_ds.get_batch([idx], device)
        shift_lbl = lbl[:, 1:].contiguous()
        valid_mask = shift_lbl != -100
        n_valid = int(valid_mask.sum().item())
        if n_valid == 0:
            continue
        for k in eval_passes:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                h = model.forward_multipass(ids, passes=k, backprop_last=1, doc_start_mask=doc, return_hidden=True)["hidden"]
                h_shift = h[:, :-1, :].contiguous()
                flat_h = h_shift.view(-1, h_shift.shape[-1])[valid_mask.view(-1)]
                flat_y = shift_lbl.view(-1)[valid_mask.view(-1)]
                nll_sum = 0.0
                for c0 in range(0, flat_y.numel(), 2048):
                    logits_c = F.linear(flat_h[c0 : c0 + 2048], lm_head_w).float()
                    nll_sum += float(F.cross_entropy(logits_c, flat_y[c0 : c0 + 2048], reduction="sum").item())
            sums[k] += nll_sum
            toks[k] += n_valid

    res: dict[str, float] = {}
    tot_t = 0
    for k in eval_passes:
        t = torch.tensor([sums[k], float(toks[k])], dtype=torch.float64, device=device)
        if dist.is_initialized():
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
        res[f"nll@K{k}"] = float(t[0].item() / max(1.0, t[1].item()))
        tot_t = int(t[1].item())
    res["val_asst_tokens"] = tot_t
    if model.has_carry:
        res["min_nll"] = min(res[f"nll@K{k}"] for k in eval_passes)
        if "nll@K8" in res and "nll@K4" in res:
            res["flatness_K8_minus_K4"] = res["nll@K8"] - res["nll@K4"]
        if "nll@K8" in res and "nll@K6" in res:
            res["flatness_K8_minus_K6"] = res["nll@K8"] - res["nll@K6"]
    model.train()
    return res


@torch.no_grad()
def evaluate_exact_parity(
    model: AttnRes2DDense,
    val_ds: PackedSFTDataset,
    n_seqs: int,
    seq_len_eval: int,
    device: torch.device,
) -> dict[str, Any]:
    """Compare token-by-token exact recurrence vs K=8 parallel Jacobi forward."""
    if not model.has_carry:
        return {}
    model.eval()
    n_use = min(n_seqs, val_ds.n_seqs)
    nll_ex_sum, nll_k8_sum, n_tok = 0.0, 0.0, 0
    for idx in range(n_use):
        ids, lbl, doc = val_ds.get_batch([idx], device)
        ids_s, lbl_s, doc_s = ids[:, :seq_len_eval], lbl[:, :seq_len_eval], doc[:, :seq_len_eval].clone()
        bnds = torch.nonzero(doc_s[0, 1:], as_tuple=False)
        if bnds.numel() > 0:
            cut = int(bnds[0].item()) + 1
            if cut < 128:
                continue
            ids_s, lbl_s, doc_s = ids_s[:, :cut], lbl_s[:, :cut], doc_s[:, :cut]
        y = lbl_s[:, 1:]
        mask = y != -100
        if int(mask.sum().item()) == 0:
            continue
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            l_ex = model.forward_exact_sequential(ids_s)[:, :-1]
            l_k8 = model.forward_multipass(ids_s, passes=8, backprop_last=1, doc_start_mask=doc_s)["logits"][:, :-1]
        y_flat = y[mask]
        nll_ex_sum += float(F.cross_entropy(l_ex[mask].float(), y_flat, reduction="sum").item())
        nll_k8_sum += float(F.cross_entropy(l_k8[mask].float(), y_flat, reduction="sum").item())
        n_tok += int(y_flat.numel())
    model.train()
    if n_tok == 0:
        return {"ex_tokens": 0}
    nll_ex, nll_k8 = nll_ex_sum / n_tok, nll_k8_sum / n_tok
    return {
        "ex_tokens": n_tok,
        "nll_exact": nll_ex,
        "nll_K8_slice": nll_k8,
        "abs_diff": abs(nll_ex - nll_k8),
    }


def save_checkpoint(
    out_dir: Path,
    raw_model: AttnRes2DDense,
    optimizer: torch.optim.Optimizer,
    step: int,
    tag: str = "latest",
) -> None:
    ckpt_dir = out_dir / f"ckpt_{tag}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    if tag == "latest":
        raw_model.base.save_pretrained(ckpt_dir)
    else:
        bf16_sd = {k: v.detach().to(torch.bfloat16).cpu() for k, v in raw_model.base.state_dict().items()}
        raw_model.base.save_pretrained(ckpt_dir, state_dict=bf16_sd)
    raw_model.save_extra(ckpt_dir, step=step)
    if tag == "latest":
        tmp_path = out_dir / "train_state_latest.pt.tmp"
        torch.save({"step": step, "optimizer": optimizer.state_dict()}, tmp_path)
        os.replace(tmp_path, out_dir / "train_state_latest.pt")


def main() -> None:
    p = argparse.ArgumentParser(description="Train Dense 2D-AttnRes (e.g. Qwen3-4B) via K-pass Jacobi SFT")
    p.add_argument("--model", required=True, help="HF model name or local checkpoint directory")
    p.add_argument("--data-dir", type=Path, required=True, help="Packed SFT dataset directory")
    p.add_argument("--out-dir", type=Path, required=True, help="Output directory for logs & checkpoints")
    p.add_argument("--mode", choices=list(AttnRes2DDense.VALID_MODES), default="attnres_2d")
    p.add_argument("--attn-impl", choices=["flash_attention_2", "sdpa"], default="flash_attention_2")
    p.add_argument("--passes", type=int, default=4, help="Jacobi passes K (default: 4)")
    p.add_argument("--backprop-last", type=int, default=2, help="Backprop through last M passes (default: 2)")
    p.add_argument("--lr-base", type=float, default=8e-6)
    p.add_argument("--lr-query", type=float, default=3e-4)
    p.add_argument("--lr-scalar", type=float, default=1e-2)
    p.add_argument("--total-tokens", type=int, default=500_000_000)
    p.add_argument("--global-seqs-per-step", type=int, default=16)
    p.add_argument("--micro-seqs", type=int, default=1)
    p.add_argument("--eval-every", type=int, default=200)
    p.add_argument("--val-seqs", type=int, default=128)
    p.add_argument("--parity-seqs", type=int, default=8)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", choices=["none", "auto"], default="auto")
    args = p.parse_args()

    rank, world, local_rank, device = init_dist()
    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_ds = PackedSFTDataset(args.data_dir, "train")
    val_ds = PackedSFTDataset(args.data_dir, "val")
    seq_len = train_ds.seq_len
    toks_per_step = args.global_seqs_per_step * seq_len
    total_steps = max(1, round(args.total_tokens / toks_per_step))
    assert args.global_seqs_per_step % (world * args.micro_seqs) == 0
    grad_accum = args.global_seqs_per_step // (world * args.micro_seqs)

    resume_ckpt = args.out_dir / "ckpt_latest"
    resuming = (
        args.resume == "auto"
        and (resume_ckpt / "attnres_2d_extra.pt").exists()
        and (args.out_dir / "train_state_latest.pt").exists()
    )
    load_path = str(resume_ckpt) if resuming else args.model

    if is_main(rank):
        print(
            f"[train_dense] mode={args.mode} K={args.passes}(last {args.backprop_last}) "
            f"world={world} grad_accum={grad_accum} total_steps={total_steps} ({args.total_tokens/1e9:.3f}B tok)",
            flush=True,
        )
        (args.out_dir / "run_args.json").write_text(json.dumps(vars(args), indent=2, default=str))

    base = AutoModelForCausalLM.from_pretrained(load_path, torch_dtype=torch.float32, attn_implementation=args.attn_impl).to(device)
    raw_model = AttnRes2DDense(base, mode=args.mode, grad_checkpoint=True).to(device)
    if resuming:
        raw_model.load_extra(resume_ckpt, map_location="cpu")

    if is_main(rank):
        n_base = sum(p.numel() for p in raw_model.base.parameters())
        n_add = raw_model.count_added_params()
        print(
            f"[train_dense] base_params={n_base:,} added_params={n_add:,} ({100*n_add/n_base:.4f}%) "
            f"L_out={raw_model.l_out}/{raw_model.n_layers}",
            flush=True,
        )

    ddp_model = DDP(raw_model, device_ids=[local_rank], find_unused_parameters=False) if world > 1 else raw_model
    param_groups = raw_model.get_param_groups(lr_base=args.lr_base, lr_query=args.lr_query, lr_scalar=args.lr_scalar)
    optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.95), eps=1e-8)

    start_step = 0
    if resuming:
        local_world = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
        for lr_i in range(local_world):
            if local_rank == lr_i:
                ts = torch.load(args.out_dir / "train_state_latest.pt", map_location="cpu", weights_only=False)
                start_step = int(ts["step"])
                optimizer.load_state_dict(ts["optimizer"])
                del ts
                gc.collect()
            if dist.is_initialized():
                dist.barrier()

    base_lrs = [g["lr"] for g in optimizer.param_groups]
    loss_fn = LigerFusedLinearCrossEntropyLoss(ignore_index=-100) if _HAS_LIGER else ChunkedLinearCELoss(ignore_index=-100)

    if start_step == 0:
        v0 = evaluate_multipass(raw_model, val_ds, [1, 2, 4, 6, 8], args.val_seqs, rank, world, device)
        if is_main(rank):
            print(f"[eval step 0] {json.dumps(v0)}", flush=True)

    ddp_model.train()
    t_run = time.perf_counter()
    log_path = args.out_dir / "metrics.jsonl"

    for step in range(start_step, total_steps):
        t_st = time.perf_counter()
        mult = lr_mult_sqrt_decay(step, total_steps)
        for g, blr in zip(optimizer.param_groups, base_lrs):
            g["lr"] = blr * mult
        optimizer.zero_grad(set_to_none=True)

        step_loss_sum = 0.0
        base_seq_idx = step * args.global_seqs_per_step
        for acc in range(grad_accum):
            seq_start = (base_seq_idx + (acc * world + rank) * args.micro_seqs) % train_ds.n_seqs
            idxs = [(seq_start + j) % train_ds.n_seqs for j in range(args.micro_seqs)]
            ids, lbl, doc = train_ds.get_batch(idxs, device)

            sync_ctx = ddp_model.no_sync() if (world > 1 and acc < grad_accum - 1) else torch.enable_grad()
            with sync_ctx:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
                    out = ddp_model(
                        ids, passes=args.passes, backprop_last=args.backprop_last,
                        doc_start_mask=doc, return_hidden=True, labels=lbl, loss_fn=loss_fn,
                    )
                    loss = out["loss"] / grad_accum
                loss.backward()
                step_loss_sum += float(loss.detach().item())

        all_params = [p for g in optimizer.param_groups for p in g["params"] if p.grad is not None]
        gnorm = float(torch.nn.utils.clip_grad_norm_(all_params, 1.0).item())
        optimizer.step()

        dt = time.perf_counter() - t_st
        step_1 = step + 1
        if (step_1 % args.log_every == 0 or step_1 == total_steps) and is_main(rank):
            tps = toks_per_step / max(dt, 1e-6)
            mem_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
            rec = {
                "step": step_1, "tokens": step_1 * toks_per_step, "loss": round(step_loss_sum, 5),
                "gnorm": round(gnorm, 4), "lr_mult": round(mult, 4), "step_s": round(dt, 3),
                "tok_per_s": round(tps, 1), "mem_gib": round(mem_gb, 2),
            }
            print(
                f"step {step_1:5d}/{total_steps} | tok={step_1*toks_per_step/1e6:7.1f}M | "
                f"loss={step_loss_sum:.4f} | gnorm={gnorm:.3f} | {dt:.2f}s ({tps/1e3:.1f}k tok/s) | mem={mem_gb:.1f}GiB",
                flush=True,
            )
            with log_path.open("a") as f:
                f.write(json.dumps(rec) + "\n")

        if step_1 % args.eval_every == 0 or step_1 == total_steps:
            ev = evaluate_multipass(raw_model, val_ds, [1, 2, 3, 4, 6, 8], args.val_seqs, rank, world, device)
            if is_main(rank):
                tel = raw_model.collect_telemetry()
                ev["step"] = step_1
                ev["tokens"] = step_1 * toks_per_step
                print(f"[eval step {step_1}] {json.dumps(ev)}", flush=True)
                with log_path.open("a") as f:
                    f.write(json.dumps({"eval": ev, "telemetry": tel}) + "\n")
                save_checkpoint(args.out_dir, raw_model, optimizer, step_1, tag="latest")
            if dist.is_initialized():
                dist.barrier()

    if dist.is_initialized():
        dist.barrier()
    if is_main(rank):
        save_checkpoint(args.out_dir, raw_model, optimizer, total_steps, tag="final")
    final_ev = evaluate_multipass(raw_model, val_ds, [1, 2, 3, 4, 6, 8], min(256, val_ds.n_seqs), rank, world, device)
    if is_main(rank):
        ex = evaluate_exact_parity(raw_model, val_ds, n_seqs=args.parity_seqs, seq_len_eval=2048, device=device) if args.parity_seqs > 0 else {}
        summary = {
            "mode": args.mode, "passes": args.passes, "backprop_last": args.backprop_last,
            "total_steps": total_steps, "total_tokens": total_steps * toks_per_step,
            "wall_s": round(time.perf_counter() - t_run, 1),
            "final_eval": final_ev, "exact_parity": ex, "telemetry": raw_model.collect_telemetry(),
        }
        (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        print(f"=== TRAIN COMPLETE ({summary['wall_s']}s) ===\n{json.dumps(summary, indent=2)}", flush=True)

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
