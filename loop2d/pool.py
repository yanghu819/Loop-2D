"""Core mathematical primitives for 2D-AttnRes (Unified 2D Pool + 1-Line Contractive Gate λ).

Everything in 2D-AttnRes reduces to two clean operations at zero extra KV cache (+0 B KV),
zero extra projection matrices, and standard AdamW optimization:

1. **Unified 2D-AttnRes Read** (`attnres_2d_pool` / `attnres_2d_anchored_pool`):
   Every sublayer attends via one shared RMSNorm-key softmax over:
       [vertical depth blocks of token t]  ∪  [H_{t-1} from 3L/4 of token t-1]
   with a zero-initialized scalar gate `g_h` guaranteeing exact 0.00e+00 bit-identity
   at step 0 and O(1) non-vanishing gradients from step 1.

2. **1-Line Contractive Write at 3L/4** (`update_contractive_carry`):
       H_t = RMSNorm(sigmoid(theta_lambda) * H_{t-1} + RMSNorm(u_{3L/4}(t)))
   Bounded spectral contraction guarantees dead-flat K=4 -> K=8 -> K=inf fixed-point
   convergence during parallel Jacobi training and unbounded autoregressive decoding.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def rmsnorm_unweighted(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Unit-RMS normalization computed in fp32 and cast back to input dtype."""
    dt = x.dtype
    xf = x.float()
    return (xf * torch.rsqrt(xf.square().mean(dim=-1, keepdim=True) + eps)).to(dt)


def attnres_2d_pool(
    q: torch.Tensor,
    vertical: torch.Tensor,
    h_carry: torch.Tensor | None,
    gamma_h: torch.Tensor | None,
    eps: float = 1e-6,
    center_ref: bool = True,
) -> torch.Tensor:
    """Unified 2D-AttnRes softmax pool for models with native Attention Residuals (e.g. Looped MoE).

    Appends token (t-1)'s 3L/4 horizontal carry `h_carry` directly into the native vertical
    RMSNorm-key softmax pool `[vertical_0, ..., vertical_{S-1}, h_carry]`:

      - For `gamma_h > 0`, algebraically exact unified softmax with log-gate offset.
      - At `gamma_h == 0`, exact `0.00e+00` bit-identity with vertical-only AttnRes AND
        full `O(1)` gradient flow into `gamma_h` on the first backward pass.
      - With `center_ref=True`, references `logit_h` against `LSE_v - log(S)` (mean-centered
        vertical log-partition) so deep layers with many vertical blocks never exponentially
        starve horizontal gate gradients.
      - Automatically masks `h_carry` to 0 at `t = 0`, document boundaries, or `--carry-cut`.

    Args:
        q: Query vectors of shape `(Q, D)` in float32.
        vertical: Stacked vertical depth blocks of shape `(B, T, S, D)`.
        h_carry: Previous token's 3L/4 contractive carry `H_{t-1}` of shape `(B, T, D)` or `None`.
        gamma_h: Zero-initialized scalar gate of shape `()`, `(Q,)`, or `None`.
        eps: RMSNorm numerical stability epsilon.
        center_ref: Mean-center vertical LSE reference (recommended `True`).

    Returns:
        Pooled states of shape `(B, T, Q, D)` in float32.
    """
    normed = rmsnorm_unweighted(vertical.float(), eps)
    dim = normed.shape[-1]
    logits = torch.einsum("qd,btsd->btqs", q, normed) * (dim ** -0.5)
    probs = F.softmax(logits, dim=-1)
    out = torch.einsum("btqs,btsd->btqd", probs, normed)

    if h_carry is not None and gamma_h is not None:
        hf = h_carry.float()
        lse_v = torch.logsumexp(logits, dim=-1)
        ref_v = (lse_v - math.log(logits.shape[-1])) if center_ref else lse_v

        valid = (hf.square().mean(dim=-1, keepdim=True) > 1e-12).float()  # (B, T, 1)
        h_norm = rmsnorm_unweighted(hf, eps) * valid                      # (B, T, D)
        logit_h = torch.einsum("qd,btd->btq", q, h_norm) * (dim ** -0.5)  # (B, T, Q)
        ratio = torch.exp((logit_h - ref_v).clamp(-15.0, 15.0)) * valid   # (B, T, Q)

        g = gamma_h.float()
        if g.ndim == 1:
            g = g.view(1, 1, -1)
        denom = 1.0 + g.abs() * ratio
        alpha_v = 1.0 / denom
        alpha_h = (g * ratio) / denom
        out = alpha_v.unsqueeze(-1) * out + alpha_h.unsqueeze(-1) * h_norm.unsqueeze(-2)
    return out


def attnres_2d_anchored_pool(
    q: torch.Tensor,
    anchor: torch.Tensor,
    v_blocks: torch.Tensor,
    h_carry: torch.Tensor | None,
    g_v: torch.Tensor,
    g_h: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Anchored unified 2D-AttnRes pool for dense pretrained Transformers (e.g. Qwen3).

    Treats the pretrained residual stream `anchor` as the unit-weight anchor of the shared
    RMSNorm-key softmax pool `[anchor, v_blocks_0..S-1, h_carry]`, scaling the pooled mixture
    back to `anchor`'s RMS magnitude:

      - Exact `0.00e+00` bit-identity with the stock pretrained model when `g_v = 0, g_h = 0`.
      - Full `O(1)` gradient flow into `g_v`, `g_h`, and `q` from step 0.
      - Automatically masks `h_carry` to 0 at `t = 0`, document starts, or `--carry-cut`.

    Args:
        q: Per-sublayer query vector of shape `(D,)`.
        anchor: Current residual stream `h` of shape `(B, T, D)`.
        v_blocks: Earlier vertical depth blocks of shape `(B, T, S_v, D)`.
        h_carry: Previous token's 3L/4 contractive carry `H_{t-1}` of shape `(B, T, D)` or `None`.
        g_v: Per-block vertical gates of shape `(S_v,)`.
        g_h: Scalar horizontal carry gate `()`.
        eps: RMSNorm epsilon.

    Returns:
        Pooled sublayer input tensor of shape `(B, T, D)` in `anchor.dtype`.
    """
    hf = anchor.float()
    dim = hf.shape[-1]
    rms_h = (hf.square().mean(dim=-1, keepdim=True) + eps).sqrt()
    h_hat = hf / rms_h
    qf = q.float()
    z_anc = torch.einsum("d,btd->bt", qf, h_hat) * (dim ** -0.5)

    num_delta = torch.zeros_like(hf)
    denom = torch.ones_like(z_anc)

    n_v = v_blocks.shape[-2]
    if n_v > 0:
        vf = v_blocks.float()
        v_hat = vf * torch.rsqrt(vf.square().mean(dim=-1, keepdim=True) + eps)
        z_v = torch.einsum("d,btsd->bts", qf, v_hat) * (dim ** -0.5)
        ratio_v = torch.exp((z_v - z_anc.unsqueeze(-1)).clamp(-15.0, 15.0))
        gv = g_v[:n_v].float().view(1, 1, n_v)
        denom = denom + (gv.abs() * ratio_v).sum(dim=-1)
        num_delta = num_delta + torch.einsum("bts,btsd->btd", gv * ratio_v, v_hat)

    if h_carry is not None:
        bf = h_carry.float()
        valid = (bf.square().mean(dim=-1, keepdim=True) > 1e-12).float()
        b_hat = bf * torch.rsqrt(bf.square().mean(dim=-1, keepdim=True) + eps) * valid
        z_h = torch.einsum("d,btd->bt", qf, b_hat) * (dim ** -0.5)
        ratio_h = torch.exp((z_h - z_anc).clamp(-15.0, 15.0)) * valid.squeeze(-1)
        gh = g_h.float()
        denom = denom + gh.abs() * ratio_h
        num_delta = num_delta + (gh * ratio_h).unsqueeze(-1) * b_hat

    return ((hf + rms_h * num_delta) / denom.unsqueeze(-1)).to(anchor.dtype)


def update_contractive_carry(
    h_prev: torch.Tensor,
    write_signal: torch.Tensor,
    theta_lambda: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """1-line contractive forgetting gate at the 3L/4 sub-readout boundary (+0 B KV).

    Computes in fp32 for guaranteed fixed-point stability:
        H_t = RMSNorm(sigmoid(theta_lambda) * H_{t-1} + RMSNorm(write_signal))
    """
    with torch.amp.autocast(device_type=h_prev.device.type, enabled=False):
        u_hat = rmsnorm_unweighted(write_signal.float(), eps)
        lam = torch.sigmoid(theta_lambda.float())
        z = lam * h_prev.float() + u_hat
        out = rmsnorm_unweighted(z, eps)
        return out.to(h_prev.dtype)


def init_multiscale_lambda_logits(
    dim: int,
    low: float = 0.25,
    high: float = 0.95,
) -> torch.Tensor:
    """Initialize per-channel forgetting gate logits across half-lives ~0.5 to ~14 tokens."""
    lams = torch.linspace(low, high, dim, dtype=torch.float32)
    return torch.log(lams / (1.0 - lams))


def shift_carry(
    h_out: torch.Tensor,
    doc_start_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Shift carry right by 1 token: `h_in[:, t] = h_out[:, t-1]`, zero at `t=0` and doc starts."""
    h_in = torch.zeros_like(h_out)
    if h_out.shape[1] > 1:
        h_in[:, 1:] = h_out[:, :-1]
    if doc_start_mask is not None:
        m = doc_start_mask[:, :, None, None] if h_out.ndim == 4 else doc_start_mask.unsqueeze(-1)
        h_in = h_in.masked_fill(m, 0.0)
    return h_in


def shift_state_pair(
    ws_out: tuple[torch.Tensor, torch.Tensor] | None,
    doc_start_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Shift a `(block_states, hidden)` Core state pair right by 1 token with doc-boundary reset."""
    if ws_out is None:
        return None
    ws_bs, ws_h = ws_out
    n_bs = torch.zeros_like(ws_bs)
    n_h = torch.zeros_like(ws_h)
    if ws_h.shape[1] > 1:
        n_bs[:, 1:] = ws_bs[:, :-1]
        n_h[:, 1:] = ws_h[:, :-1]
    if doc_start_mask is not None:
        n_bs = n_bs.masked_fill(doc_start_mask[:, :, None, None], 0.0)
        n_h = n_h.masked_fill(doc_start_mask[:, :, None], 0.0)
    return (n_bs, n_h)


def docstarts_to_position_ids(doc_start_mask: torch.Tensor) -> torch.LongTensor:
    """Convert `(B, T)` boolean `doc_start_mask` into per-document 0-indexed `position_ids`."""
    bsz, seq_len = doc_start_mask.shape
    idx = torch.arange(seq_len, device=doc_start_mask.device, dtype=torch.long).unsqueeze(0).expand(bsz, seq_len)
    starts = doc_start_mask.bool().clone()
    starts[:, 0] = True
    doc_anchor = torch.where(starts, idx, torch.zeros_like(idx)).cummax(dim=1).values
    return idx - doc_anchor


def sample_next_token(
    logits: torch.Tensor,
    seen_mask: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    presence_penalty: float = 0.0,
) -> torch.Tensor:
    """Vectorized sampling with exact presence penalty, top-k, and nucleus (top-p) filtering."""
    logits = logits.float()
    if presence_penalty > 0.0:
        logits = logits - presence_penalty * seen_mask.float()
    if temperature <= 0.0:
        return logits.argmax(dim=-1)
    logits = logits / temperature
    if top_k > 0 and top_k < logits.shape[-1]:
        topv, _ = torch.topk(logits, top_k, dim=-1)
        min_v = topv[:, -1:].contiguous()
        logits = torch.where(logits < min_v, torch.full_like(logits, -float("inf")), logits)
    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        remove = cum_probs > top_p
        remove[:, 1:] = remove[:, :-1].clone()
        remove[:, 0] = False
        sorted_logits = sorted_logits.masked_fill(remove, -float("inf"))
        logits = torch.empty_like(logits).scatter_(-1, sorted_idx, sorted_logits)
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)
