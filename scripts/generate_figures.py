"""Generate all 5 publication-grade 2D-AttnRes visualizations (300 DPI, standalone reproducible).

Usage:
    python3 scripts/generate_figures.py --out-dir assets
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Global clean publication typography & color palette
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11.5,
    "axes.labelweight": "bold",
    "xtick.labelsize": 10.5,
    "ytick.labelsize": 10.5,
    "legend.fontsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#CBD5E1",
    "axes.linewidth": 1.1,
    "figure.facecolor": "#FFFFFF",
    "axes.facecolor": "#FFFFFF",
    "savefig.facecolor": "#FFFFFF",
    "savefig.dpi": 300,
})

C_HERO = "#2563EB"       # Royal Indigo: 2D-AttnRes (Best Practice)
C_HERO_DARK = "#1E40AF"  # Deep Indigo
C_TEAL = "#0D9488"       # Vivid Teal: 2D-AttnRes P=2 / 55M SFT
C_VIOLET = "#7C3AED"     # Deep Violet: 3L/4 boundary & MessageGate
C_STOCK = "#64748B"      # Slate Gray: Stock Baseline
C_STD = "#94A3B8"        # Light Steel: Matched Std SFT
C_ORIGTR = "#D97706"     # Warm Amber: Original TR (Cai et al., 2026)
C_CUT = "#DC2626"        # Coral Red: Causal Wire-Cut (H_t <- 0)
C_GRID = "#E2E8F0"       # Subtle Grid
C_TEXT = "#0F172A"       # Rich Dark Slate Text
C_MUTED = "#475569"      # Muted Annotation Text


# ---------------------------------------------------------------------------
# 100% Disk-Verified Benchmark & Checkpoint Telemetry Data
# ---------------------------------------------------------------------------
VERIFIED_DATA = {
    "moe_pareto": {
        "stock_kv": [1.0, 1.5, 2.0],
        "stock_aime": [66.81, 84.44, 85.97],
        "stock_pooled": [75.00, 89.04, 90.29],
        "stock_tok": [16565, 12038, 11291],
        "matched_std_p1": {"kv": 1.0, "aime": 69.44, "pooled": 77.60, "tok": 14802},
        "attnres2d_p1_hv1": {"kv": 1.0, "aime": 78.33, "pooled": 84.52, "tok": 13385, "kb": 9.8},
        "attnres2d_p1_hv2": {"kv": 1.0, "aime": 80.28, "pooled": 85.87, "tok": 12522, "kb": 9.8},
        "attnres2d_p2_hv1": {"kv": 1.5, "aime": 86.11, "pooled": 90.29, "tok": 11494, "kb": 10.1},
    },
    "gpqa_p1": {
        "labels": ["Stock P=1\n(1.0× KV)", "Matched Std SFT\n(1.0× KV)", "P=1 + 2D-AttnRes\n(+9.8 KB, 1.0× KV)"],
        "acc": [52.78, 57.32, 66.92],
        "tok": [9145, 8234, 7587],
    },
    "wire_cut": {
        "benches": ["AIME 24", "AIME 25", "AIME 26", "AMC 23", "Comp\nPooled"],
        "on_acc": [84.58, 76.67, 73.75, 98.44, 84.52],
        "cut_acc": [73.75, 62.50, 61.25, 91.56, 73.75],
        "stock_acc": [73.75, 62.08, 64.58, 93.44, 75.00],
        "on_tok": [11784, 13868, 14501, 3367],
        "cut_tok": [15948, 16532, 17512, 5942],
        "tok_labels": ["AIME 24", "AIME 25", "AIME 26", "AMC 23"],
    },
    "qwen3_4b": {
        "std_toks_m": [104.9, 209.7, 314.6, 419.4, 499.9],
        "std_nll": [1.00582, 0.99356, 0.98330, 0.97075, 0.95603],
        "origtr_toks_m": [104.9, 209.7, 314.6, 419.4, 499.9],
        "origtr_nll": [1.00535, 0.99205, 0.98008, 0.96606, 0.95112],
        "ours_toks_m": [52.4, 104.9, 157.3, 209.7, 262.1, 314.6, 367.0, 419.4, 471.9, 499.9],
        "ours_nll": [1.01592, 1.00026, 0.99193, 0.98686, 0.98013, 0.97547, 0.97303, 0.96255, 0.95061, 0.94793],
        "k_vals": [1, 2, 3, 4, 6, 8],
        "k_nll": [0.98443, 0.95054, 0.94808, 0.94793, 0.94798, 0.94800],
    },
    "telemetry": {
        "tml_att_abs_24": [
            0.0000, 0.0084, 0.0409, 0.1119, 0.1026, 0.1064,
            0.3579, 0.1982, 0.2020, 0.4280, 1.0920, 0.2130,
            0.2005, 0.0115, 0.0015, 0.0016, 0.0133, 0.0098,
            0.0064, 0.0061, 0.0123, 0.0043, 0.0180, 0.0074,
        ],
        "tml_mlp_abs_24": [
            0.0099, 0.0246, 0.0130, 0.1517, 0.0707, 0.0295,
            0.0524, 0.0655, 0.1248, 0.0920, 0.1894, 0.3849,
            0.0226, 0.0246, 0.0272, 0.0080, 0.0151, 0.0475,
            0.0046, 0.0035, 0.0001, 0.0045, 0.0285, 0.0620,
        ],
        "tml_msg0_abs_core12": [
            0.2618, 0.2988, 0.6827, 0.5941, 0.2674, 0.3032,
            0.2399, 0.2407, 0.1617, 0.1299, 0.0505, 0.0862,
        ],
        "qwen_w_write": [-0.4320, 0.1598, 1.8900, 1.5529, 0.0681],
        "qwen_w_blocks": ["B0\nL0–5", "B1\nL6–11", "B2\nL12–17", "B3\nL18–23", "B4\nL24–26"],
        "tml_lam_hist": [0, 0, 0, 0, 2, 28, 124, 132, 146, 125, 127, 157, 148, 122, 133, 150, 119, 136, 142, 141, 129, 133, 109, 1],
        "qwen_lam_hist": [15, 64, 68, 56, 53, 44, 69, 67, 89, 109, 122, 169, 200, 226, 247, 261, 215, 191, 135, 92, 44, 22, 2, 0],
        "lam_edges": [i / 24.0 for i in range(25)],
    },
}


def _box(ax, x, y, w, h, text, fc, ec, tc=C_TEXT, fs=9.5, fw="bold", lw=1.4, r=0.025, alpha=1.0, zorder=3):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.008,rounding_size={r}",
        facecolor=fc, edgecolor=ec, linewidth=lw, alpha=alpha, zorder=zorder,
    )
    ax.add_patch(patch)
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", color=tc, fontsize=fs, fontweight=fw, zorder=zorder + 1)
    return patch


def _arrow(ax, p1, p2, color=C_MUTED, lw=1.8, rad=0.0, ls="-", ms=12, zorder=4):
    arr = FancyArrowPatch(
        p1, p2,
        connectionstyle=f"arc3,rad={rad}",
        arrowstyle=f"-|>,head_length={ms*0.6},head_width={ms*0.35}",
        color=color, linewidth=lw, linestyle=ls, zorder=zorder,
    )
    ax.add_patch(arr)


# ---------------------------------------------------------------------------
# Figure 1: Unified 2D-AttnRes Architecture vs 1D Vertical Loop & Original TR
# ---------------------------------------------------------------------------
def make_fig1(out_dir: Path) -> None:
    fig = plt.figure(figsize=(16.2, 6.8))
    fig.suptitle(
        "2D-AttnRes: Unifying Vertical Depth Routing & Zero-KV Horizontal Recurrence in One Shared Pool",
        fontsize=15.5, fontweight="bold", color=C_TEXT, y=0.965,
    )
    fig.text(
        0.5, 0.915,
        "Zero dual streams · Zero extra matrices · +0 B KV cache · 1-line contractive gate λ at the 3L/4 boundary",
        ha="center", fontsize=11, color=C_MUTED,
    )

    gs = fig.add_gridspec(1, 3, left=0.02, right=0.98, top=0.865, bottom=0.04, wspace=0.07)

    # ---- Panel 1: 1D Vertical Loop ----
    ax = fig.add_subplot(gs[0, 0])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    _box(ax, 0.02, 0.02, 0.96, 0.96, "", "#F8FAFC", "#CBD5E1", lw=1.3, r=0.03, zorder=1)
    ax.text(0.50, 0.935, "(A)  1D Vertical Depth Loop", ha="center", va="center", fontsize=12.5, fontweight="bold", color=C_MUTED)
    ax.text(0.50, 0.882, "Repeated Core passes reset every token", ha="center", va="center", fontsize=9.8, color=C_STOCK)

    for tx, lbl in [(0.23, "Token t−1"), (0.77, "Token t")]:
        ax.text(tx, 0.132, lbl, ha="center", va="center", fontsize=10, fontweight="bold", color=C_TEXT)
        _box(ax, tx - 0.15, 0.185, 0.30, 0.11, "Prelude\n0 → L/4", "#FFFFFF", "#94A3B8", fs=8.8)
        _box(ax, tx - 0.15, 0.37, 0.30, 0.24, "Looped Core\nL/4 → 3L/4\n(P = 2–3 passes)", "#E2E8F0", "#64748B", fs=8.6)
        _box(ax, tx - 0.15, 0.695, 0.30, 0.11, "Postlude\n3L/4 → L", "#FFFFFF", "#94A3B8", fs=8.8)
        _arrow(ax, (tx, 0.30), (tx, 0.365), color=C_STOCK, lw=1.7)
        _arrow(ax, (tx, 0.615), (tx, 0.69), color=C_STOCK, lw=1.7)
        # Put vertical loop arrow on outer sides so middle stays 100% open!
        if tx < 0.5:
            _arrow(ax, (tx - 0.155, 0.57), (tx - 0.155, 0.41), color=C_STOCK, lw=1.8, rad=0.52)
        else:
            _arrow(ax, (tx + 0.155, 0.57), (tx + 0.155, 0.41), color=C_STOCK, lw=1.8, rad=-0.52)

    ax.plot([0.40, 0.60], [0.49, 0.49], color="#CBD5E1", ls=":", lw=2.0, zorder=2)
    ax.text(0.50, 0.52, "✕", ha="center", va="center", fontsize=16, fontweight="bold", color=C_CUT, zorder=5)
    ax.text(0.50, 0.435, "No deep\ncarry", ha="center", va="center", fontsize=8.8, color=C_CUT, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.18", fc="#FFFFFF", ec="none", alpha=0.9), zorder=5)
    _box(ax, 0.06, 0.038, 0.88, 0.056, "Cost: 1.5×–2.0× KV Cache & Core FLOPs", "#FEF2F2", "#FCA5A5", tc="#991B1B", fs=9.2, r=0.015)

    # ---- Panel 2: Original TR (Cai et al., 2026) ----
    ax = fig.add_subplot(gs[0, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    _box(ax, 0.02, 0.02, 0.96, 0.96, "", "#FFFBEB", "#FCD34D", lw=1.3, r=0.03, zorder=1)
    ax.text(0.50, 0.935, "(B)  Original TR (Cai et al., 2026)", ha="center", va="center", fontsize=12.5, fontweight="bold", color="#92400E")
    ax.text(0.50, 0.882, "Separate dual a/b streams · writes to layer L", ha="center", va="center", fontsize=9.8, color="#B45309")

    for tx, lbl in [(0.23, "Token t−1"), (0.77, "Token t")]:
        ax.text(tx, 0.132, lbl, ha="center", va="center", fontsize=10, fontweight="bold", color=C_TEXT)
        _box(ax, tx - 0.155, 0.185, 0.31, 0.58, "", "#FFFFFF", "#F59E0B", lw=1.3)
        ax.text(tx - 0.075, 0.475, "Stream a\n(residual)\n\n0 → L", ha="center", va="center", fontsize=8.4, color=C_TEXT, fontweight="bold")
        ax.text(tx + 0.075, 0.475, "Stream b\n(static\n+r_b,+s_b)\n0 → L", ha="center", va="center", fontsize=8.0, color="#B45309", fontweight="bold")
        ax.plot([tx + 0.002, tx + 0.002], [0.205, 0.745], color="#FDE68A", lw=1.4, ls="--")

    _arrow(ax, (0.39, 0.74), (0.61, 0.22), color=C_ORIGTR, lw=2.3, rad=-0.16)
    _box(ax, 0.422, 0.445, 0.156, 0.125, "B_t\n(L_out = L)\nλ = 1", "#FEF3C7", C_ORIGTR, tc="#92400E", fs=8.3, r=0.015, zorder=5)
    ax.text(0.50, 0.815, "Jacobi must run ALL 100% layers every pass", ha="center", va="center", fontsize=9.0, color="#B45309", fontweight="bold")
    _box(ax, 0.06, 0.038, 0.88, 0.056, "Overhead: Dual a/b stream · No 3L/4 cutoff · No λ", "#FEF3C7", "#F59E0B", tc="#92400E", fs=9.0, r=0.015)

    # ---- Panel 3: 2D-AttnRes (Ours — Best Practice) ----
    ax = fig.add_subplot(gs[0, 2])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    _box(ax, 0.02, 0.02, 0.96, 0.96, "", "#EFF6FF", C_HERO, lw=2.2, r=0.03, zorder=1)
    ax.text(0.50, 0.935, "(C)  2D-AttnRes  (Ours — Best Practice)", ha="center", va="center", fontsize=12.5, fontweight="bold", color=C_HERO_DARK)
    ax.text(0.50, 0.882, "One shared 2D AttnRes pool + 1-line λ at 3L/4", ha="center", va="center", fontsize=9.8, color=C_HERO)

    # Token t-1
    _box(ax, 0.06, 0.185, 0.27, 0.115, "Prelude\n0 → L/4", "#FFFFFF", C_HERO, fs=8.6)
    _box(ax, 0.06, 0.355, 0.27, 0.225, "Core\nL/4 → 3L/4\n(Write u_{3L/4})", "#DBEAFE", C_HERO, tc=C_HERO_DARK, fs=8.4)
    _box(ax, 0.06, 0.695, 0.27, 0.12, "Readout\n3L/4 → L\n(Skipped K−1×)", "#F1F5F9", "#94A3B8", tc=C_MUTED, fs=8.1)
    ax.text(0.195, 0.132, "Token t−1", ha="center", va="center", fontsize=10, fontweight="bold", color=C_TEXT)
    _arrow(ax, (0.195, 0.305), (0.195, 0.35), color=C_HERO, lw=1.7)
    _arrow(ax, (0.195, 0.585), (0.195, 0.69), color="#94A3B8", lw=1.5, ls="--")

    # 3L/4 boundary line with crisp pill label
    ax.plot([0.05, 0.95], [0.635, 0.635], color=C_VIOLET, ls="--", lw=1.7, zorder=2)
    ax.text(0.50, 0.635, "3L/4 Sub-Readout Cutoff", ha="center", va="center", fontsize=8.4, fontweight="bold", color=C_VIOLET,
            bbox=dict(boxstyle="round,pad=0.20", fc="#FFFFFF", ec=C_VIOLET, lw=0.9), zorder=6)

    # Token t
    _box(ax, 0.67, 0.185, 0.27, 0.115, "Prelude\n0 → L/4", "#FFFFFF", C_HERO, fs=8.6)
    _box(ax, 0.67, 0.355, 0.27, 0.225, "Unified 2D Pool\nSoftmax([V_blk(t),\n             H_{t−1}])", "#2563EB", C_HERO_DARK, tc="#FFFFFF", fs=8.2)
    _box(ax, 0.67, 0.695, 0.27, 0.12, "Readout\n3L/4 → L\n+ LM Head", "#FFFFFF", C_HERO, tc=C_TEXT, fs=8.3)
    ax.text(0.805, 0.132, "Token t", ha="center", va="center", fontsize=10, fontweight="bold", color=C_TEXT)
    _arrow(ax, (0.805, 0.305), (0.805, 0.35), color=C_HERO, lw=1.7)
    _arrow(ax, (0.805, 0.585), (0.805, 0.69), color=C_HERO, lw=1.7)

    # 1-Line Contractive Gate H_{t-1} bridge
    _box(ax, 0.375, 0.385, 0.25, 0.175, "1-Line Gate λ\nH_t = Norm(\n  λ·H_{t−1} + û)", "#F3E8FF", C_VIOLET, tc="#5B21B6", fs=8.2, lw=1.6, r=0.02, zorder=5)
    _arrow(ax, (0.335, 0.53), (0.373, 0.50), color=C_VIOLET, lw=2.2)
    _arrow(ax, (0.628, 0.47), (0.668, 0.47), color=C_HERO, lw=2.2)
    _arrow(ax, (0.628, 0.41), (0.668, 0.25), color=C_HERO, lw=1.7, rad=0.14)

    _box(ax, 0.055, 0.038, 0.89, 0.056, "+0 B KV · 9.8 KB Params (MoE) · Pure AdamW · 0.00 Step-0 ID", "#DCFCE7", "#16A34A", tc="#14532D", fs=8.7, r=0.015)

    out_path = out_dir / "fig1_2d_attnres_architecture.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


# ---------------------------------------------------------------------------
# Figure 2: 26.4B MoE KV/Compute Pareto + Zero-Shot OOD PhD Science (GPQA)
# ---------------------------------------------------------------------------
def make_fig2(out_dir: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.8, 6.4), gridspec_kw={"width_ratios": [1.24, 1.0]})
    fig.subplots_adjust(wspace=0.23, top=0.84, bottom=0.14, left=0.06, right=0.97)
    fig.suptitle(
        "26.4B Looped MoE: Breaking the Vertical KV Pareto & +14.1% Zero-Shot PhD Science",
        fontsize=15, fontweight="bold", color=C_TEXT, y=0.96,
    )
    fig.text(
        0.5, 0.905,
        "Exact n=8 evaluation at 32k CoT · 2D-AttnRes adds only 2,448 scalars (9.8 KB) at P=1 / 2,520 scalars (10.1 KB) at P=2, +0 B KV",
        ha="center", fontsize=10.5, color=C_MUTED,
    )

    # ---- Left Panel: KV / Compute Pareto ----
    d = VERIFIED_DATA["moe_pareto"]
    ax1.grid(True, axis="y", color=C_GRID, linewidth=0.9, zorder=0)
    ax1.set_axisbelow(True)

    ax1.plot(d["stock_kv"], d["stock_aime"], "--o", color=C_STOCK, lw=2.3, ms=8.5, zorder=3, label="Stock TML-d24 (1D Vertical Loop P=1,2,3)")
    # Stock annotations placed cleanly in 100% open whitespace
    ax1.annotate("Stock P=1\n66.81%", (1.0, 66.81), xytext=(1.02, 63.3), ha="center", va="center", fontsize=9.5, fontweight="bold", color=C_STOCK)
    ax1.annotate("Stock P=2\n84.44%", (1.5, 84.44), xytext=(1.62, 81.2), ha="left", va="center", fontsize=9.5, fontweight="bold", color=C_STOCK,
                 arrowprops=dict(arrowstyle="-", color="#94A3B8", lw=1.0))
    ax1.annotate("Stock P=3\n85.97%", (2.0, 85.97), xytext=(1.99, 82.6), ha="center", va="center", fontsize=9.5, fontweight="bold", color=C_STOCK)

    # Matched Std SFT at P=1
    mstd = d["matched_std_p1"]
    ax1.scatter([mstd["kv"]], [mstd["aime"]], s=105, marker="s", color=C_STD, edgecolor="#475569", lw=1.2, zorder=4, label="Matched Std SFT (P=1, 0 added params)")
    ax1.annotate("Matched Std\n69.44%", (1.0, 69.44), xytext=(1.18, 69.44), ha="left", va="center", fontsize=9.3, fontweight="bold", color="#475569",
                 arrowprops=dict(arrowstyle="-", color="#94A3B8", lw=1.1))

    # 2D-AttnRes P=1 (26M & 55M SFT) — placed cleanly in open whitespace
    p1_hv1 = d["attnres2d_p1_hv1"]
    p1_hv2 = d["attnres2d_p1_hv2"]
    ax1.scatter([1.0], [p1_hv1["aime"]], s=155, marker="D", color=C_HERO, edgecolor="#FFFFFF", lw=1.5, zorder=5, label="P=1 + 2D-AttnRes (26M SFT, +9.8 KB, +0 B KV)")
    ax1.scatter([1.0], [p1_hv2["aime"]], s=185, marker="D", color=C_TEAL, edgecolor="#FFFFFF", lw=1.5, zorder=6, label="P=1 + 2D-AttnRes (55M SFT, +9.8 KB, +0 B KV)")

    ax1.annotate("P=1 + 2D-AttnRes (55M SFT)\n80.28% (+13.5%, 4.0k fewer tok)", (1.0, p1_hv2["aime"]), xytext=(1.03, 84.8),
                 ha="center", va="center", fontsize=9.3, fontweight="bold", color=C_TEAL,
                 bbox=dict(boxstyle="round,pad=0.22", fc="#F0FDFA", ec=C_TEAL, lw=0.9),
                 arrowprops=dict(arrowstyle="->", color=C_TEAL, lw=1.4))
    ax1.annotate("P=1 + 2D-AttnRes (26M SFT)\n78.33% (+8.9% vs Matched Std)", (1.0, p1_hv1["aime"]), xytext=(1.43, 73.2),
                 ha="center", va="center", fontsize=9.2, fontweight="bold", color=C_HERO_DARK,
                 bbox=dict(boxstyle="round,pad=0.20", fc="#EFF6FF", ec=C_HERO, lw=0.9),
                 arrowprops=dict(arrowstyle="->", color=C_HERO, lw=1.4))

    # Vertical gain arrow at 1.0x KV
    ax1.annotate("", xy=(0.945, p1_hv2["aime"]), xytext=(0.945, 66.81),
                 arrowprops=dict(arrowstyle="<->", color=C_HERO, lw=2.0))
    ax1.text(0.925, 73.6, "+13.5%\n+0 B KV", ha="right", va="center", fontsize=9.4, fontweight="bold", color=C_HERO)

    # 2D-AttnRes P=2 (beats Stock P=3)
    p2 = d["attnres2d_p2_hv1"]
    ax1.scatter([1.5], [p2["aime"]], s=320, marker="*", color=C_HERO_DARK, edgecolor="#FFFFFF", lw=1.3, zorder=7, label="P=2 + 2D-AttnRes (26M SFT, +10.1 KB, matches Stock P=3)")
    ax1.axhline(85.97, xmin=0.51, xmax=0.92, color=C_VIOLET, ls=":", lw=1.7, zorder=2)
    ax1.annotate("", xy=(1.52, 87.6), xytext=(1.98, 87.6),
                 arrowprops=dict(arrowstyle="<->", color=C_VIOLET, lw=2.0))
    ax1.text(1.75, 88.8, "Matches Stock P=3 (90.29% Pooled)\nat −25% KV & −33% Core FLOPs", ha="center", va="bottom",
             fontsize=9.2, fontweight="bold", color=C_VIOLET,
             bbox=dict(boxstyle="round,pad=0.25", fc="#F5F3FF", ec=C_VIOLET, lw=1.0))
    ax1.annotate("P=2 + 2D-AttnRes\n86.11% AIME", (1.5, p2["aime"]), xytext=(1.26, 89.2),
                 ha="center", va="center", fontsize=9.8, fontweight="bold", color=C_HERO_DARK,
                 arrowprops=dict(arrowstyle="->", color=C_HERO_DARK, lw=1.5))

    ax1.set_title("AIME 24/25/26 Accuracy vs. KV Cache & Core Compute", pad=12, color=C_TEXT)
    ax1.set_xlabel("Relative KV Cache & Core Compute Budget")
    ax1.set_ylabel("AIME 24 / 25 / 26 Mean Accuracy (%)")
    ax1.set_xlim(0.82, 2.13)
    ax1.set_ylim(61.0, 93.2)
    ax1.set_xticks([1.0, 1.5, 2.0])
    ax1.set_xticklabels(["1.0× KV\n(P = 1)", "1.5× KV\n(P = 2)", "2.0× KV\n(P = 3)"])
    ax1.legend(loc="lower right", frameon=True, facecolor="#FFFFFF", edgecolor=C_GRID, fontsize=8.6)

    # ---- Right Panel: Zero-Shot OOD GPQA-Diamond ----
    g = VERIFIED_DATA["gpqa_p1"]
    x = np.arange(len(g["labels"]))
    colors = [C_STOCK, C_STD, C_HERO]
    bars = ax2.bar(x, g["acc"], width=0.56, color=colors, edgecolor="#FFFFFF", linewidth=1.5, zorder=3)
    ax2.grid(True, axis="y", color=C_GRID, linewidth=0.9, zorder=0)
    ax2.set_axisbelow(True)

    for xi, acc, tok in zip(x, g["acc"], g["tok"]):
        ax2.text(xi, acc + 0.9, f"{acc:.2f}%", ha="center", va="bottom", fontsize=12, fontweight="bold",
                 color=(C_HERO_DARK if xi == 2 else C_TEXT))
        ax2.text(xi, acc - 4.5, f"Mean CoT:\n{tok:,} tok", ha="center", va="center", fontsize=9.2,
                 fontweight="bold", color="#FFFFFF")

    ax2.annotate("", xy=(2, 72.0), xytext=(0, 72.0), arrowprops=dict(arrowstyle="<->", color=C_HERO, lw=1.8))
    ax2.text(1.0, 72.9, "+14.14% vs Stock P=1  (1,558 fewer tokens)", ha="center", va="bottom",
             fontsize=9.8, fontweight="bold", color=C_HERO_DARK,
             bbox=dict(boxstyle="round,pad=0.25", fc="#EFF6FF", ec=C_HERO, lw=1.0))

    ax2.annotate("+9.60% pure 2D-AttnRes lift\n(zero science training data)", xy=(2, 66.92), xytext=(1.02, 65.2),
                 ha="center", va="center", fontsize=9.2, fontweight="bold", color=C_TEAL,
                 arrowprops=dict(arrowstyle="->", color=C_TEAL, lw=1.6),
                 bbox=dict(boxstyle="round,pad=0.25", fc="#F0FDFA", ec=C_TEAL, lw=1.0))

    ax2.set_title("Zero-Shot OOD Transfer: GPQA-Diamond (PhD Science)", pad=12, color=C_TEXT)
    ax2.set_ylabel("GPQA-Diamond Accuracy (%)  [792/792 Sealed]")
    ax2.set_xticks(x)
    ax2.set_xticklabels(g["labels"], fontsize=10)
    ax2.set_ylim(38, 79.0)

    out_path = out_dir / "fig2_moe_pareto_and_ood.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


# ---------------------------------------------------------------------------
# Figure 3: Causal Wire-Cut Proof (H_t ON vs H_t <- 0 on Identical Weights)
# ---------------------------------------------------------------------------
def make_fig3(out_dir: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.2, 5.8), gridspec_kw={"width_ratios": [1.32, 1.0]})
    fig.subplots_adjust(wspace=0.22, top=0.82, bottom=0.14, left=0.06, right=0.97)
    fig.suptitle(
        "Causal Wire-Cut Proof: Zeroing H_t ← 0 on Identical Weights Collapses Accuracy & Bloats CoT",
        fontsize=14.8, fontweight="bold", color=C_TEXT, y=0.96,
    )
    fig.text(
        0.5, 0.90,
        "Exact same P=1 + 2D-AttnRes (26M SFT) checkpoint (2,448 scalars = 9.8 KB) · 1,040/1,040 sealed rollouts ON vs 1,040/1,040 sealed OFF",
        ha="center", fontsize=10.3, color=C_MUTED,
    )

    wc = VERIFIED_DATA["wire_cut"]
    x = np.arange(len(wc["benches"]))
    w = 0.37

    # ---- Left: Accuracy collapse (clean labels inside bars -> zero collision with Stock tick!) ----
    ax1.grid(True, axis="y", color=C_GRID, linewidth=0.9, zorder=0)
    ax1.set_axisbelow(True)
    ax1.bar(x - w / 2, wc["on_acc"], width=w, color=C_HERO, edgecolor="#FFFFFF", lw=1.2, zorder=3, label="2D-AttnRes ON (H_t active)")
    ax1.bar(x + w / 2, wc["cut_acc"], width=w, color=C_CUT, edgecolor="#FFFFFF", lw=1.2, zorder=3, label="Wire-Cut OFF (H_t ← 0, same weights)")
    ax1.scatter(x + w / 2, wc["stock_acc"], color="#0F172A", marker="_", s=300, lw=2.8, zorder=5, label="Stock P=1 Baseline")

    for xi, on, cut in zip(x, wc["on_acc"], wc["cut_acc"]):
        ax1.text(xi - w / 2, on - 2.6, f"{on:.1f}%", ha="center", va="center", fontsize=9.3, fontweight="bold", color="#FFFFFF", zorder=4)
        ax1.text(xi + w / 2, cut - 2.6, f"{cut:.1f}%", ha="center", va="center", fontsize=9.3, fontweight="bold", color="#FFFFFF", zorder=4)
        diff = cut - on
        ax1.text(xi, on + 2.3, f"{diff:+.1f}%", ha="center", va="bottom", fontsize=9.4, fontweight="bold",
                 color="#991B1B", bbox=dict(boxstyle="round,pad=0.18", fc="#FEF2F2", ec="#FCA5A5", lw=0.9), zorder=6)

    ax1.set_title("Accuracy Collapse Across Every Competition Benchmark", pad=10, color=C_TEXT)
    ax1.set_ylabel("Exact n=8 Accuracy (%)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(wc["benches"], fontsize=10.3, fontweight="bold")
    ax1.set_ylim(48, 109.5)
    ax1.legend(loc="upper left", frameon=True, facecolor="#FFFFFF", edgecolor=C_GRID, fontsize=9.1)

    # ---- Right: Token length bloat (generous headroom for top callout banner!) ----
    xt = np.arange(len(wc["tok_labels"]))
    ax2.grid(True, axis="y", color=C_GRID, linewidth=0.9, zorder=0)
    ax2.set_axisbelow(True)
    ax2.bar(xt - w / 2, [t / 1000.0 for t in wc["on_tok"]], width=w, color=C_HERO, edgecolor="#FFFFFF", lw=1.2, zorder=3, label="2D-AttnRes ON")
    ax2.bar(xt + w / 2, [t / 1000.0 for t in wc["cut_tok"]], width=w, color=C_CUT, edgecolor="#FFFFFF", lw=1.2, zorder=3, label="Wire-Cut OFF (H_t ← 0)")

    for xi, t_on, t_cut in zip(xt, wc["on_tok"], wc["cut_tok"]):
        pct = 100.0 * (t_cut - t_on) / t_on
        ax2.text(xi - w / 2, t_on / 1000.0 + 0.35, f"{t_on/1000:.1f}k", ha="center", va="bottom", fontsize=9.2, fontweight="bold", color=C_HERO_DARK)
        ax2.text(xi + w / 2, t_cut / 1000.0 + 0.35, f"{t_cut/1000:.1f}k", ha="center", va="bottom", fontsize=9.2, fontweight="bold", color=C_CUT)
        ax2.text(xi, max(t_on, t_cut) / 1000.0 + 1.85, f"+{pct:.0f}%", ha="center", va="bottom", fontsize=9.2, fontweight="bold",
                 color="#991B1B", bbox=dict(boxstyle="round,pad=0.17", fc="#FEF2F2", ec="#FCA5A5", lw=0.85))

    ax2.set_title("CoT Token Bloat When Cross-Token Carry Is Cut", pad=10, color=C_TEXT)
    ax2.set_ylabel("Mean Generated CoT Length (Thousands of Tokens)")
    ax2.set_xticks(xt)
    ax2.set_xticklabels(wc["tok_labels"], fontsize=10.3, fontweight="bold")
    ax2.set_ylim(0, 24.5)
    ax2.text(0.50, 0.935, "AIME Mean: 13.4k → 16.7k tok (+3,279 tok bloat)", transform=ax2.transAxes,
             ha="center", va="center", fontsize=9.6, fontweight="bold", color="#991B1B",
             bbox=dict(boxstyle="round,pad=0.28", fc="#FEF2F2", ec=C_CUT, lw=1.0))

    out_path = out_dir / "fig3_causal_wire_cut.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


# ---------------------------------------------------------------------------
# Figure 4: Dense Qwen3-4B Scaling Curve & K=1..8 Contractive Convergence
# ---------------------------------------------------------------------------
def make_fig4(out_dir: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.4, 5.9), gridspec_kw={"width_ratios": [1.22, 1.0]})
    fig.subplots_adjust(wspace=0.22, top=0.82, bottom=0.14, left=0.065, right=0.97)
    fig.suptitle(
        "Dense Qwen3-4B (0.0047% Params): 1.65× Original TR's Gain & Dead-Flat K=4→8 Convergence",
        fontsize=14.8, fontweight="bold", color=C_TEXT, y=0.96,
    )
    fig.text(
        0.5, 0.90,
        "36-layer Qwen3-4B-Base · 1.65M held-out assistant val tokens · 189,956 added params (+0 B KV, pure AdamW)",
        ha="center", fontsize=10.3, color=C_MUTED,
    )

    q = VERIFIED_DATA["qwen3_4b"]

    # ---- Left: 0 -> 0.50B Val NLL Trajectory ----
    ax1.grid(True, color=C_GRID, linewidth=0.9, zorder=0)
    ax1.plot(q["std_toks_m"], q["std_nll"], "--o", color=C_STOCK, lw=2.3, ms=6.5, label="Standard Qwen3-4B (0 added params)")
    ax1.plot(q["origtr_toks_m"], q["origtr_nll"], "-.^", color=C_ORIGTR, lw=2.4, ms=7.0, label="Original TR (Cai et al., 2026)")
    ax1.plot(q["ours_toks_m"], q["ours_nll"], "-o", color=C_HERO, lw=3.0, ms=7.0, zorder=5, label="2D-AttnRes (Ours, 190K params, +0 B KV)")

    # End-of-run delta callouts placed in 100% open whitespace!
    ax1.annotate("Std: 0.95603", (499.9, 0.95603), xytext=(488, 0.977),
                 ha="center", va="center", fontsize=9.5, fontweight="bold", color=C_STOCK,
                 bbox=dict(boxstyle="round,pad=0.20", fc="#F8FAFC", ec="#CBD5E1", lw=0.9),
                 arrowprops=dict(arrowstyle="->", color=C_STOCK, lw=1.3))
    ax1.annotate("Original TR: 0.95112\n(−4.91 mnat vs Std)", (499.9, 0.95112), xytext=(362, 0.9545),
                 ha="right", va="center", fontsize=9.3, fontweight="bold", color="#B45309",
                 bbox=dict(boxstyle="round,pad=0.20", fc="#FFFBEB", ec="#F59E0B", lw=0.9),
                 arrowprops=dict(arrowstyle="->", color=C_ORIGTR, lw=1.3))
    ax1.annotate("2D-AttnRes: 0.94793\n(−8.10 mnat = 1.65× OrigTR)", (499.9, 0.94793), xytext=(362, 0.9442),
                 ha="right", va="center", fontsize=9.7, fontweight="bold", color=C_HERO_DARK,
                 bbox=dict(boxstyle="round,pad=0.24", fc="#EFF6FF", ec=C_HERO, lw=1.1),
                 arrowprops=dict(arrowstyle="->", color=C_HERO, lw=1.6))

    ax1.set_title("Held-Out Validation NLL Trajectory (0 → 0.50B Tokens)", pad=10, color=C_TEXT)
    ax1.set_xlabel("Training Tokens (Millions)")
    ax1.set_ylabel("Held-Out Assistant Val NLL (nats, lower is better)")
    ax1.set_xlim(35, 530)
    ax1.set_ylim(0.9405, 1.021)
    ax1.legend(loc="upper right", frameon=True, facecolor="#FFFFFF", edgecolor=C_GRID, fontsize=9.3)

    # ---- Right: Exact K=1..8 Fixed-Point Convergence ----
    ax2.grid(True, color=C_GRID, linewidth=0.9, zorder=0)
    ax2.axhline(0.95603, color=C_STOCK, ls="--", lw=1.8, label="Standard Qwen3-4B (0.95603)")
    ax2.axhline(0.95112, color=C_ORIGTR, ls="-.", lw=1.8, label="Original TR @ K=4 (0.95112)")
    ax2.plot(q["k_vals"], q["k_nll"], "-o", color=C_HERO, lw=3.0, ms=8.5, zorder=5, label="2D-AttnRes (Ours) across K passes")

    ax2.axvspan(4.0, 8.0, color="#DCFCE7", alpha=0.45, zorder=1)
    ax2.text(6.0, 0.9835, "Dead-Flat Fixed Point\nK4→K8: +0.076 mnat\nK6→K8: +0.020 mnat",
             ha="center", va="top", fontsize=9.4, fontweight="bold", color="#166534",
             bbox=dict(boxstyle="round,pad=0.28", fc="#F0FDF4", ec="#16A34A", lw=1.1))

    # Clean vertical arrow & label cleanly to the RIGHT of the K=1->K=2 drop!
    ax2.annotate("", xy=(2.35, 0.94793), xytext=(2.35, 0.98443),
                 arrowprops=dict(arrowstyle="<->", color=C_VIOLET, lw=2.0))
    ax2.plot([1.05, 2.38], [0.98443, 0.98443], color=C_VIOLET, ls=":", lw=1.1, alpha=0.7)
    ax2.text(2.50, 0.9695, "−36.5 mnat\nhorizontal carry\nlift (K1 → K4)", ha="left", va="center",
             fontsize=9.3, fontweight="bold", color=C_VIOLET,
             bbox=dict(boxstyle="round,pad=0.22", fc="#F5F3FF", ec=C_VIOLET, lw=0.85))

    for kv, nv, dy in [(1, 0.98443, 0.0022), (2, 0.95054, -0.0033), (4, 0.94793, -0.0033), (8, 0.94800, -0.0033)]:
        ax2.text(kv, nv + dy, f"{nv:.5f}", ha="center", va="center", fontsize=8.8, fontweight="bold", color=C_HERO_DARK)

    ax2.set_title("Exact Jacobi Fixed-Point Stability (K = 1 → 8 at 0.50B)", pad=10, color=C_TEXT)
    ax2.set_xlabel("Jacobi Evaluation Passes (K)")
    ax2.set_ylabel("Held-Out Assistant Val NLL (nats)")
    ax2.set_xticks(q["k_vals"])
    ax2.set_xlim(0.55, 8.55)
    ax2.set_ylim(0.942, 0.9905)
    ax2.legend(loc="center right", bbox_to_anchor=(0.99, 0.45), frameon=True, facecolor="#FFFFFF", edgecolor=C_GRID, fontsize=8.8)

    out_path = out_dir / "fig4_dense_4b_scaling.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


# ---------------------------------------------------------------------------
# Figure 5: Mechanistic Anatomy of Trained 2D-AttnRes Weights
# ---------------------------------------------------------------------------
def make_fig5(out_dir: Path) -> None:
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(17.0, 5.5), gridspec_kw={"width_ratios": [1.38, 0.95, 1.12]})
    fig.subplots_adjust(wspace=0.25, top=0.80, bottom=0.16, left=0.05, right=0.98)
    fig.suptitle(
        "Inside Trained 2D-AttnRes: Learned Depth Reading, Mid-Deep Writing & Multi-Timescale Memory",
        fontsize=14.8, fontweight="bold", color=C_TEXT, y=0.96,
    )
    fig.text(
        0.5, 0.895,
        "Direct X-ray of trained checkpoint weights (zero synthetic data): 26.4B MoE (9.8 KB) & Dense Qwen3-4B (190K params)",
        ha="center", fontsize=10.3, color=C_MUTED,
    )

    t = VERIFIED_DATA["telemetry"]

    # ---- Panel A: Layer-wise |g_h| across 24 layers of 26.4B MoE ----
    layers = np.arange(24)
    ax1.axvspan(-0.5, 5.5, color="#F1F5F9", alpha=0.75, zorder=0)
    ax1.axvspan(5.5, 17.5, color="#EFF6FF", alpha=0.80, zorder=0)
    ax1.axvspan(17.5, 23.5, color="#FEF2F2", alpha=0.45, zorder=0)
    ax1.axvline(17.5, color=C_VIOLET, ls="--", lw=1.8, zorder=2)

    ax1.grid(True, axis="y", color=C_GRID, linewidth=0.8, zorder=1)
    ax1.bar(layers - 0.19, t["tml_att_abs_24"], width=0.38, color=C_HERO, label="|g_h| Attention Pool", zorder=3)
    ax1.bar(layers + 0.19, t["tml_mlp_abs_24"], width=0.38, color=C_TEAL, label="|g_h| MLP Pool", zorder=3)
    ax1.plot(np.arange(6, 18), t["tml_msg0_abs_core12"], "-o", color=C_VIOLET, lw=2.2, ms=5.0, zorder=4, label="|tanh(g_h_msg0)| Core MsgGate")

    ax1.text(2.5, 1.29, "Prelude\n(L0–5)", ha="center", va="top", fontsize=8.8, fontweight="bold", color=C_MUTED)
    ax1.text(11.5, 1.29, "Core Reasoning (L6–17)\nPeak |g_h| = 1.09 @ L10", ha="center", va="top", fontsize=9.0, fontweight="bold", color=C_HERO_DARK)
    ax1.text(20.5, 1.29, "Readout\n(L18–23)\n≈ 0", ha="center", va="top", fontsize=8.8, fontweight="bold", color="#991B1B")
    ax1.text(17.9, 0.36, "3L/4\nWrite", ha="left", va="center", fontsize=8.8, fontweight="bold", color=C_VIOLET,
             bbox=dict(boxstyle="round,pad=0.18", fc="#F5F3FF", ec=C_VIOLET, lw=0.85))

    ax1.set_title("(A) Cross-Token Read Gates |g_h| (26.4B MoE)", pad=10, color=C_TEXT)
    ax1.set_xlabel("Transformer Layer Index (0–23)")
    ax1.set_ylabel("Learned Gate Magnitude")
    ax1.set_xlim(-0.7, 23.7)
    ax1.set_ylim(0, 1.35)
    ax1.set_xticks([0, 5, 6, 10, 14, 17, 18, 23])
    # Place legend cleanly in wide-open upper-right region over layers 13–23 (bars < 0.06, zero overlap with L10 peak!)
    ax1.legend(loc="upper right", bbox_to_anchor=(0.99, 0.78), frameon=True, facecolor="#FFFFFF", edgecolor=C_GRID, fontsize=8.1)

    # ---- Panel B: Auto-discovered write depth w_write (Dense Qwen3-4B) ----
    bx = np.arange(len(t["qwen_w_write"]))
    b_colors = [C_CUT if v < 0 else (C_HERO if v > 1.0 else C_STD) for v in t["qwen_w_write"]]
    ax2.grid(True, axis="y", color=C_GRID, linewidth=0.8, zorder=0)
    ax2.axhline(0.0, color=C_TEXT, lw=1.0, zorder=2)
    ax2.axhline(1.0, color=C_MUTED, ls=":", lw=1.4, zorder=2, label="Init = 1.0 (uniform)")
    ax2.bar(bx, t["qwen_w_write"], width=0.58, color=b_colors, edgecolor="#FFFFFF", lw=1.2, zorder=3)

    for xi, v in zip(bx, t["qwen_w_write"]):
        dy = 0.08 if v >= 0 else -0.09
        ax2.text(xi, v + dy, f"{v:+.2f}", ha="center", va="bottom" if v >= 0 else "top",
                 fontsize=9.6, fontweight="bold", color=(C_HERO_DARK if v > 1.0 else (C_CUT if v < 0 else C_MUTED)))

    ax2.annotate("Mid-deep reasoning band\n(L/3 → 2L/3 peak)", xy=(2.5, 1.92), xytext=(2.55, 2.34),
                 ha="center", va="center", fontsize=8.9, fontweight="bold", color=C_HERO_DARK,
                 bbox=dict(boxstyle="round,pad=0.22", fc="#EFF6FF", ec=C_HERO, lw=1.0))

    ax2.set_title("(B) Auto-Discovered Write Depth (4B)", pad=10, color=C_TEXT)
    ax2.set_xlabel("Depth Block (≤ 3L/4)")
    ax2.set_ylabel("Learned Write Weight w_write")
    ax2.set_xticks(bx)
    ax2.set_xticklabels(t["qwen_w_blocks"], fontsize=9.2)
    ax2.set_ylim(-0.78, 2.62)
    # Place legend in open upper-left above B0/B1!
    ax2.legend(loc="upper left", bbox_to_anchor=(0.01, 0.67), frameon=True, facecolor="#FFFFFF", edgecolor=C_GRID, fontsize=8.4)

    # ---- Panel C: Learned Multi-Timescale λ Distribution ----
    edges = np.array(t["lam_edges"])
    centers = 0.5 * (edges[:-1] + edges[1:])
    width = edges[1] - edges[0]
    tml_p = np.array(t["tml_lam_hist"], dtype=float)
    tml_p = 100.0 * tml_p / tml_p.sum()
    qwen_p = np.array(t["qwen_lam_hist"], dtype=float)
    qwen_p = 100.0 * qwen_p / qwen_p.sum()

    ax3.grid(True, axis="y", color=C_GRID, linewidth=0.8, zorder=0)
    ax3.bar(centers, tml_p, width=width * 0.92, color=C_TEAL, alpha=0.45, edgecolor=C_TEAL, lw=1.0, zorder=3,
            label="26.4B MoE (mean=0.60, p10–90: 0.31–0.88)")
    ax3.step(centers, qwen_p, where="mid", color=C_HERO, lw=2.6, zorder=4,
             label="Dense Qwen3-4B (mean=0.53, p10–90: 0.21–0.76)")
    ax3.fill_between(centers, qwen_p, step="mid", color=C_HERO, alpha=0.14, zorder=2)

    for lam_v, hl_txt in [(0.25, "t½≈0.5t\n(syntax)"), (0.60, "t½≈1.4t\n(clause)"), (0.88, "t½≈5.4t\n(state)")]:
        ax3.axvline(lam_v, color=C_MUTED, ls=":", lw=1.1, zorder=2)
        ax3.text(lam_v, 12.1, hl_txt, ha="center", va="top", fontsize=8.1, fontweight="bold", color=C_MUTED)

    ax3.set_title("(C) Learned Multi-Timescale Gate λ = σ(θ_λ)", pad=10, color=C_TEXT)
    ax3.set_xlabel("Channel Retention Factor λ  (Contractive: λ < 1)")
    ax3.set_ylabel("Percentage of Hidden Channels (%)")
    ax3.set_xlim(0.0, 1.0)
    ax3.set_ylim(0, 15.2)
    ax3.legend(loc="upper left", bbox_to_anchor=(0.01, 0.99), frameon=True, facecolor="#FFFFFF", edgecolor=C_GRID, fontsize=8.0)

    out_path = out_dir / "fig5_mechanistic_anatomy.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def main() -> None:
    p = argparse.ArgumentParser(description="Generate all 5 publication figures for 2D-AttnRes")
    p.add_argument("--out-dir", type=Path, default=Path("assets"))
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    make_fig1(args.out_dir)
    make_fig2(args.out_dir)
    make_fig3(args.out_dir)
    make_fig4(args.out_dir)
    make_fig5(args.out_dir)
    print("ALL 5 FIGURES GENERATED SUCCESSFULLY.")


if __name__ == "__main__":
    main()
