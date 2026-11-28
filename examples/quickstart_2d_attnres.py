"""Self-contained CPU/GPU quickstart verifying all core 2D-AttnRes invariants in <5 seconds.

Demonstrates:
  1. Exact 0.00e+00 step-0 bit-identity with stock pretrained Transformer.
  2. Full O(1) gradient flow into unified pool gates & 1-line contractive gate λ on step 0.
  3. Exact K=T Jacobi parallel training == token-by-token sequential recurrence parity.
  4. Exact zero cross-document carry leakage across packed document boundaries.
  5. Instant causal `--carry-cut` (`H_t <- 0`) ablation on identical trained weights.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from transformers import Qwen3Config as DenseConfig, Qwen3ForCausalLM as DenseForCausalLM
except ImportError:
    from transformers import LlamaConfig as DenseConfig, LlamaForCausalLM as DenseForCausalLM

from loop2d import AttnRes2DDense, AttnRes2DLoopedMoE, Inkling2Config


def demo_dense_qwen3() -> None:
    print("=== 1. Dense Transformer (Qwen3-style) + 2D-AttnRes ===")
    torch.manual_seed(0)
    cfg = DenseConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=12,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
    )
    base = DenseForCausalLM(cfg).eval()
    model = AttnRes2DDense(base, mode="attnres_2d").eval()
    x = torch.randint(0, cfg.vocab_size, (2, 10))

    # 1. Step-0 exact bit-identity
    with torch.no_grad():
        logits_base = base(x).logits
        logits_2d = model.forward_multipass(x, passes=4, backprop_last=1)["logits"]
    diff_s0 = float((logits_2d - logits_base).abs().max().item())
    print(f"  Added params: {model.count_added_params():,} | L_out={model.l_out}/{model.n_layers} | +0 B KV")
    print(f"  Step-0 max |logits_2D - logits_stock| = {diff_s0:.2e}  (exact bit-identity!)")
    assert diff_s0 == 0.0

    # 2. One standard AdamW step activates the 2D-AttnRes pool & contractive gate λ
    model.train()
    opt = torch.optim.AdamW(model.get_param_groups(lr_base=1e-4, lr_query=1e-2, lr_scalar=5e-2))
    out = model.forward_multipass(x, passes=4, backprop_last=2)
    loss = F.cross_entropy(out["logits"][:, :-1].reshape(-1, cfg.vocab_size), x[:, 1:].reshape(-1))
    loss.backward()
    print(f"  Step-0 grad norms: |dL/dg_h|={model.g_h.grad.norm():.4f}, |dL/dg_v|={model.g_v.grad.norm():.4f}")
    assert model.g_h.grad.norm().item() > 0.0
    opt.step()

    # 3. Exact K=T Jacobi parallel == token-by-token sequential parity
    model.eval()
    with torch.no_grad():
        logits_par = model.forward_multipass(x, passes=x.shape[1], backprop_last=1)["logits"]
        logits_seq = model.forward_exact_sequential(x)
        logits_cut = model.forward_exact_sequential(x, carry_cut=True)
    parity = float((logits_par - logits_seq).abs().max().item())
    cut_delta = float((logits_seq - logits_cut).abs().max().item())
    print(f"  Exact K=T parallel vs 1-tok sequential max diff = {parity:.2e}")
    print(f"  Causal wire-cut (H_t <- 0) max logit shift      = {cut_delta:.2e}")
    assert parity < 1e-4
    assert cut_delta > 0.0

    # 4. Exact zero cross-document carry bleed under doc_start_mask + block-diagonal causal mask
    x2 = x.clone()
    x2[:, :5] = (x2[:, :5] + 17) % cfg.vocab_size
    doc_start = torch.zeros_like(x, dtype=torch.bool)
    doc_start[:, 0] = True
    doc_start[:, 5] = True
    T = x.shape[1]
    doc_id = doc_start.long().cumsum(dim=1)
    idx = torch.arange(T)
    same_doc = (doc_id[:, :, None] == doc_id[:, None, :]) & (idx[None, :, None] >= idx[None, None, :])
    attn_mask_4d = torch.where(same_doc[:, None], 0.0, torch.finfo(torch.float32).min)
    with torch.no_grad():
        h_out1 = model.forward_multipass(x, passes=4, backprop_last=1, doc_start_mask=doc_start, attention_mask=attn_mask_4d)["h_out"]
        h_out2 = model.forward_multipass(x2, passes=4, backprop_last=1, doc_start_mask=doc_start, attention_mask=attn_mask_4d)["h_out"]
    bleed = float((h_out1[:, 5:] - h_out2[:, 5:]).abs().max().item())
    print(f"  Cross-doc carry bleed on doc #2 (tokens 5..9)   = {bleed:.2e}")
    assert bleed < 1e-5
    print(f"  Telemetry: {model.collect_telemetry()}\n")


def demo_looped_moe() -> None:
    print("=== 2. Looped MoE (TML-d24 style) + 2D-AttnRes ===")
    torch.manual_seed(1)
    cfg = Inkling2Config(
        vocab_size=128,
        unpadded_vocab_size=128,
        hidden_size=48,
        num_hidden_layers=12,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=12,
        swa_num_attention_heads=4,
        swa_num_key_value_heads=2,
        swa_head_dim=12,
        sliding_window_size=16,
        d_rel=8,
        rel_extent=16,
        intermediate_size=32,
        n_routed_experts=4,
        num_experts_per_tok=2,
        n_layers_per_attn_res_block=3,
        loop_layer_index_start=3,
        loop_layer_index_end=9,
        loop_num_passes=2,
        loop_gate_rank=8,
        eos_token_id=1,
        pad_token_id=0,
    )
    model = AttnRes2DLoopedMoE(cfg, mode="attnres_2d", attn_backend="eager")
    for p in model.parameters():
        if p.ndim >= 2 and not any(p is q for n, q in model.named_parameters() if n.startswith(AttnRes2DLoopedMoE.EXTRA_PREFIXES)):
            nn.init.normal_(p, std=0.02)
    x = torch.randint(0, cfg.vocab_size, (1, 8))
    model.eval()
    with torch.no_grad():
        logits_par = model.forward_multipass(x, passes=x.shape[1], backprop_last=1)["logits"]
        logits_seq = model.forward_exact_sequential(x)
    diff = float((logits_par - logits_seq).abs().max().item())
    print(f"  Added params (toy P=2): {model.count_added_params():,} | L_out={model.loop_end}/{cfg.num_hidden_layers}")
    print(f"  Exact K=T parallel vs 1-tok sequential max diff = {diff:.2e}")
    assert diff < 1e-4

    # Report exact 26.4B production parameter count
    cfg_26b_p1 = Inkling2Config(loop_num_passes=1)
    cfg_26b_p2 = Inkling2Config(loop_num_passes=2)
    with torch.device("meta"):
        m_p1 = AttnRes2DLoopedMoE(cfg_26b_p1, mode="attnres_2d", attn_backend="eager")
        m_p2 = AttnRes2DLoopedMoE(cfg_26b_p2, mode="attnres_2d", attn_backend="eager")
    p1_n, p2_n = m_p1.count_added_params(), m_p2.count_added_params()
    print(f"  Production 26.4B MoE exact added scalars: P=1 -> {p1_n:,} ({p1_n*4/1000:.1f} KB fp32) | P=2 -> {p2_n:,} ({p2_n*4/1000:.1f} KB fp32)")


if __name__ == "__main__":
    demo_dense_qwen3()
    demo_looped_moe()
    print("ALL QUICKSTART CHECKS PASSED!")
