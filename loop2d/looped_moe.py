"""Standalone 26.4B Looped MoE (`TML-d24-varloop-research`) + Unified 2D-AttnRes.

Best-practice architecture: Unified 2D-AttnRes Pool + 1-Line Contractive Gate λ at 3L/4.
  - Only 2,448 added scalars (9.8 KB fp32 / 0.000009% of 26.4B) at P=1 (2,520 at P=2).
  - Zero extra KV cache (+0 B KV), zero extra matrices, 100% standard AdamW.
  - Exact 0.00e+00 step-0 bit-identity with stock TML-d24-varloop.
  - Natural 3L/4 sub-readout cutoff at Core exit (layer 18 / 24): passes 1..K-1 of Jacobi
    parallel training skip Postlude layers 18..23 and the 200k-vocab LM head with zero error.
  - Exact algebraic LSE-scaled attention sinks (FlexAttention O(T) + eager reference),
    Hopper grouped-GEMM 128-expert MoE (`torch._grouped_mm`), and chunked KV + 4-Conv1D
    cache with active-slot batch compaction for fast exact 32k decoding.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

from safetensors.torch import load_file
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from loop2d.pool import (
    attnres_2d_pool,
    init_multiscale_lambda_logits,
    rmsnorm_unweighted,
    sample_next_token,
    shift_carry,
    shift_state_pair,
    update_contractive_carry,
)

try:
    from torch.nn.attention.flex_attention import AuxRequest, BlockMask, create_block_mask, flex_attention
    _HAS_FLEX = True
except Exception:
    try:
        from torch.nn.attention.flex_attention import BlockMask, create_block_mask, flex_attention
        AuxRequest = None  # type: ignore
        _HAS_FLEX = True
    except Exception:
        AuxRequest = None  # type: ignore
        BlockMask = Any  # type: ignore
        create_block_mask = None  # type: ignore
        flex_attention = None  # type: ignore
        _HAS_FLEX = False


@dataclass
class Inkling2Config:
    vocab_size: int = 201024
    unpadded_vocab_size: int = 200058
    hidden_size: int = 2304
    num_hidden_layers: int = 24
    num_attention_heads: int = 24
    num_key_value_heads: int = 8
    head_dim: int = 128
    swa_num_attention_heads: int = 24
    swa_num_key_value_heads: int = 8
    swa_head_dim: int = 128
    sliding_window_size: int = 1024
    d_rel: int = 16
    rel_extent: int = 1024
    log_scaling_n_floor: int = 8192
    log_scaling_alpha: float = 0.1
    rms_norm_eps: float = 1e-6
    sconv_kernel_size: int = 4
    intermediate_size: int = 1152
    n_routed_experts: int = 128
    num_experts_per_tok: int = 8
    route_scale: float = 8.0
    qk_l2_clip_threshold: float = 22.627416997969522
    rel_l2_clip_threshold: float = 8.0
    n_layers_per_attn_res_block: int = 6
    mlp_gate_softcap: float = 4.0
    mlp_up_softcap: float = 25.0
    final_logit_softcapping: float = 30.0
    loop_layer_index_start: int = 6
    loop_layer_index_end: int = 18
    loop_num_passes: int = 1
    loop_gate_rank: int = 64
    eos_token_id: int = 200006
    pad_token_id: int = 199999

    @classmethod
    def from_json(cls, path: Path, loop_num_passes: int | None = None) -> "Inkling2Config":
        raw = json.loads(path.read_text())
        if "text_config" in raw:
            tc = dict(raw["text_config"])
            tc["eos_token_id"] = raw.get("eos_token_id", 200006)
        else:
            tc = dict(raw)
        lc = tc.get("loop_config") or {}
        return cls(
            vocab_size=int(tc.get("vocab_size", 201024)),
            unpadded_vocab_size=int(tc.get("unpadded_vocab_size", 200058)),
            hidden_size=int(tc.get("hidden_size", 2304)),
            num_hidden_layers=int(tc.get("num_hidden_layers", 24)),
            num_attention_heads=int(tc.get("num_attention_heads", 24)),
            num_key_value_heads=int(tc.get("num_key_value_heads", 8)),
            head_dim=int(tc.get("head_dim", 128)),
            swa_num_attention_heads=int(tc.get("swa_num_attention_heads", 24)),
            swa_num_key_value_heads=int(tc.get("swa_num_key_value_heads", 8)),
            swa_head_dim=int(tc.get("swa_head_dim", 128)),
            sliding_window_size=int(tc.get("sliding_window_size", 1024)),
            d_rel=int(tc.get("d_rel", 16)),
            rel_extent=int(tc.get("rel_extent", 1024)),
            log_scaling_n_floor=int(tc.get("log_scaling_n_floor", 8192)),
            log_scaling_alpha=float(tc.get("log_scaling_alpha", 0.1)),
            rms_norm_eps=float(tc.get("rms_norm_eps", 1e-6)),
            sconv_kernel_size=int(tc.get("sconv_kernel_size", tc.get("conv_kernel_size", 4))),
            intermediate_size=int(tc.get("intermediate_size", tc.get("moe_intermediate_size", 1152))),
            n_routed_experts=int(tc.get("n_routed_experts", 128)),
            num_experts_per_tok=int(tc.get("num_experts_per_tok", 8)),
            route_scale=float(tc.get("route_scale", 8.0)),
            qk_l2_clip_threshold=float(tc.get("qk_l2_clip_threshold", 22.627416997969522)),
            rel_l2_clip_threshold=float(tc.get("rel_l2_clip_threshold", 8.0)),
            n_layers_per_attn_res_block=int(tc.get("n_layers_per_attn_res_block", 6)),
            mlp_gate_softcap=float(tc.get("mlp_gate_softcap", 4.0)),
            mlp_up_softcap=float(tc.get("mlp_up_softcap", 25.0)),
            final_logit_softcapping=float(tc.get("final_logit_softcapping", 30.0)),
            loop_layer_index_start=int(lc.get("core_start", tc.get("loop_layer_index_start", 6))),
            loop_layer_index_end=int(lc.get("core_end", tc.get("loop_layer_index_end", 18))),
            loop_num_passes=int(loop_num_passes if loop_num_passes is not None else lc.get("num_passes", tc.get("loop_num_passes", 3))),
            loop_gate_rank=int(lc.get("gate_rank", tc.get("loop_gate_rank", 64))),
            eos_token_id=int(tc.get("eos_token_id", 200006)),
            pad_token_id=int(tc.get("pad_token_id", 199999)),
        )

    def is_sliding_layer(self, layer_idx: int) -> bool:
        return (layer_idx + 1) % self.n_layers_per_attn_res_block != 0

    @property
    def total_visited_layers(self) -> int:
        core_len = self.loop_layer_index_end - self.loop_layer_index_start
        return self.num_hidden_layers + (self.loop_num_passes - 1) * core_len


def _softcap(x: torch.Tensor, cap: float) -> torch.Tensor:
    return torch.tanh(x / cap) * cap


def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return x
    b, h, t, d = x.shape
    return x[:, :, None, :, :].expand(b, h, n_rep, t, d).reshape(b, h * n_rep, t, d)


def _eager_attn_with_sinks(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    position_bias: torch.Tensor | None,
    sinks: torch.Tensor | None,
    attn_mask: torch.Tensor | None,
    scaling: float,
    n_rep: int,
) -> torch.Tensor:
    ks = _repeat_kv(k, n_rep)
    vs = _repeat_kv(v, n_rep)
    w = torch.matmul(q, ks.transpose(2, 3)) * scaling
    if position_bias is not None:
        w = w + position_bias
    if attn_mask is not None:
        w = w + attn_mask
    if sinks is not None:
        comb = torch.cat([w, sinks], dim=-1).float()
        probs = F.softmax(comb, dim=-1)[..., :-1].to(q.dtype)
    else:
        probs = F.softmax(w, dim=-1, dtype=torch.float32).to(q.dtype)
    return torch.matmul(probs, vs).transpose(1, 2).contiguous()


_flex_attn_compiled = torch.compile(flex_attention, dynamic=True) if _HAS_FLEX else None


def _flex_attn_with_sinks(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    rel_logits: torch.Tensor,
    sinks: torch.Tensor,
    block_mask: Any,
    scaling: float,
    rel_extent: int,
) -> torch.Tensor:
    """O(T)-memory FlexAttention with exact algebraic LSE sink scaling."""
    assert _HAS_FLEX and _flex_attn_compiled is not None
    rl = rel_logits.to(q.dtype).contiguous()

    def score_mod(score, b, h, q_idx, kv_idx):
        dist = q_idx - kv_idx
        in_rel = (dist >= 0) & (dist < rel_extent)
        idx = torch.clamp(dist, 0, rel_extent - 1)
        return score + torch.where(in_rel, rl[b, h, q_idx, idx], score.new_zeros(()))

    if AuxRequest is not None:
        out, aux = _flex_attn_compiled(
            q, k, v,
            score_mod=score_mod,
            block_mask=block_mask,
            scale=scaling,
            enable_gqa=(q.shape[1] != k.shape[1]),
            return_aux=AuxRequest(lse=True),
        )
        lse = aux.lse
    else:
        out, lse = _flex_attn_compiled(
            q, k, v,
            score_mod=score_mod,
            block_mask=block_mask,
            scale=scaling,
            enable_gqa=(q.shape[1] != k.shape[1]),
            return_lse=True,
        )
    sink_scale = torch.sigmoid(lse.float().unsqueeze(-1) - sinks.float()).to(out.dtype)
    return (out * sink_scale).transpose(1, 2).contiguous()


class Inkling2Cache:
    """Chunk-growing KV + 4-Conv1D cache with sliding-window cap and active-slot compaction."""

    def __init__(
        self,
        cfg: Inkling2Config,
        batch_size: int,
        init_cap: int,
        max_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ):
        self.cfg = cfg
        self.batch_size = batch_size
        self.max_len = max_len
        self.device = device
        self.dtype = dtype
        self.n_vis = cfg.total_visited_layers
        self.kv_heads = cfg.num_key_value_heads
        self.head_dim = cfg.head_dim
        self.win = cfg.sliding_window_size
        self.k_sz = cfg.sconv_kernel_size
        self.seqlens = torch.zeros(batch_size, dtype=torch.long, device=device)

        self.is_sliding_vis: list[bool] = []
        for l_idx in range(cfg.num_hidden_layers):
            self.is_sliding_vis.append(cfg.is_sliding_layer(l_idx))
        for _ in range(1, cfg.loop_num_passes):
            for l_idx in range(cfg.loop_layer_index_start, cfg.loop_layer_index_end):
                self.is_sliding_vis.append(cfg.is_sliding_layer(l_idx))

        self.full_cap = min(max_len, max(2048, init_cap))
        self.k_cache: list[torch.Tensor] = []
        self.v_cache: list[torch.Tensor] = []
        for is_sl in self.is_sliding_vis:
            cap = min(max_len, self.win + 256) if is_sl else self.full_cap
            self.k_cache.append(torch.zeros(batch_size, self.kv_heads, cap, self.head_dim, dtype=dtype, device=device))
            self.v_cache.append(torch.zeros(batch_size, self.kv_heads, cap, self.head_dim, dtype=dtype, device=device))

        kv_dim = self.kv_heads * self.head_dim
        dim = cfg.hidden_size
        self.conv_states: list[list[torch.Tensor]] = [
            [
                torch.zeros(batch_size, kv_dim, self.k_sz - 1, dtype=torch.float32, device=device),
                torch.zeros(batch_size, kv_dim, self.k_sz - 1, dtype=torch.float32, device=device),
                torch.zeros(batch_size, dim, self.k_sz - 1, dtype=torch.float32, device=device),
                torch.zeros(batch_size, dim, self.k_sz - 1, dtype=torch.float32, device=device),
            ]
            for _ in range(self.n_vis)
        ]
        self.layer_seen = [0] * self.n_vis
        self.global_seen = [0] * self.n_vis
        self.pad_lens: torch.Tensor | None = None
        self.cur_conv_mask: torch.Tensor | None = None

    def compact(self, keep_idx: torch.Tensor) -> None:
        self.batch_size = int(keep_idx.numel())
        self.seqlens = self.seqlens.index_select(0, keep_idx)
        for idx in range(self.n_vis):
            self.k_cache[idx] = self.k_cache[idx].index_select(0, keep_idx)
            self.v_cache[idx] = self.v_cache[idx].index_select(0, keep_idx)
            self.conv_states[idx] = [cs.index_select(0, keep_idx) for cs in self.conv_states[idx]]
        if self.pad_lens is not None:
            self.pad_lens = self.pad_lens.index_select(0, keep_idx)
            if bool((self.pad_lens == 0).all().item()):
                self.pad_lens = None

    def ensure_full_cap(self, needed: int) -> None:
        if needed <= self.full_cap:
            return
        new_cap = min(self.max_len, max(needed + 64, self.full_cap + 4096))
        for idx, is_sl in enumerate(self.is_sliding_vis):
            if is_sl:
                continue
            nk = torch.zeros(self.batch_size, self.kv_heads, new_cap, self.head_dim, dtype=self.dtype, device=self.device)
            nv = torch.zeros(self.batch_size, self.kv_heads, new_cap, self.head_dim, dtype=self.dtype, device=self.device)
            nk[:, :, : self.full_cap].copy_(self.k_cache[idx])
            nv[:, :, : self.full_cap].copy_(self.v_cache[idx])
            self.k_cache[idx], self.v_cache[idx] = nk, nv
        self.full_cap = new_cap

    def update_kv(self, vis_idx: int, k_new: torch.Tensor, v_new: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
        q_len = k_new.shape[2]
        seen = self.layer_seen[vis_idx]
        g_end = self.global_seen[vis_idx] + q_len
        self.global_seen[vis_idx] = g_end
        if not self.is_sliding_vis[vis_idx]:
            self.ensure_full_cap(seen + q_len)
            self.k_cache[vis_idx][:, :, seen : seen + q_len].copy_(k_new)
            self.v_cache[vis_idx][:, :, seen : seen + q_len].copy_(v_new)
            self.layer_seen[vis_idx] = seen + q_len
            return self.k_cache[vis_idx][:, :, : seen + q_len], self.v_cache[vis_idx][:, :, : seen + q_len], g_end

        cap = self.k_cache[vis_idx].shape[2]
        if seen + q_len <= cap:
            self.k_cache[vis_idx][:, :, seen : seen + q_len].copy_(k_new)
            self.v_cache[vis_idx][:, :, seen : seen + q_len].copy_(v_new)
            end = seen + q_len
            self.layer_seen[vis_idx] = end
            start = max(0, end - self.win) if q_len == 1 else 0
            return self.k_cache[vis_idx][:, :, start:end], self.v_cache[vis_idx][:, :, start:end], g_end

        keep = min(seen, self.win - 1) if q_len == 1 else min(seen, max(0, cap - q_len))
        if keep > 0:
            self.k_cache[vis_idx][:, :, :keep].copy_(self.k_cache[vis_idx][:, :, seen - keep : seen])
            self.v_cache[vis_idx][:, :, :keep].copy_(self.v_cache[vis_idx][:, :, seen - keep : seen])
        if keep + q_len > cap:
            new_cap = keep + q_len + 128
            nk = torch.zeros(self.batch_size, self.kv_heads, new_cap, self.head_dim, dtype=self.dtype, device=self.device)
            nv = torch.zeros(self.batch_size, self.kv_heads, new_cap, self.head_dim, dtype=self.dtype, device=self.device)
            if keep > 0:
                nk[:, :, :keep].copy_(self.k_cache[vis_idx][:, :, :keep])
                nv[:, :, :keep].copy_(self.v_cache[vis_idx][:, :, :keep])
            self.k_cache[vis_idx], self.v_cache[vis_idx] = nk, nv
        self.k_cache[vis_idx][:, :, keep : keep + q_len].copy_(k_new)
        self.v_cache[vis_idx][:, :, keep : keep + q_len].copy_(v_new)
        self.layer_seen[vis_idx] = keep + q_len
        return self.k_cache[vis_idx][:, :, : keep + q_len], self.v_cache[vis_idx][:, :, : keep + q_len], g_end


class Inkling2RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dt = x.dtype
        xf = x.float()
        out = xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + self.eps)
        return (self.weight.float() * out).to(dt)


class Inkling2NormedEmbedding(nn.Embedding):
    def __init__(self, vocab_size: int, dim: int, pad_id: int | None, eps: float):
        super().__init__(vocab_size, dim, padding_idx=pad_id)
        self.embed_norm = Inkling2RMSNorm(dim, eps)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_norm(super().forward(input_ids))


class Inkling2ScaledL2Clip(nn.Module):
    def __init__(self, num_heads: int, head_dim: int, threshold: float, log_scaling: bool = False, n_floor: int = 8192, alpha: float = 0.1):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(num_heads, head_dim))
        self.threshold = threshold
        self.log_scaling = log_scaling
        self.n_floor = n_floor
        self.alpha = alpha

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        dt = x.dtype
        xf = x.float() * self.scale.float()
        norm = torch.linalg.vector_norm(xf, dim=-1, keepdim=True)
        xf = xf * (self.threshold / (norm + 1e-6)).clamp(max=1.0)
        if self.log_scaling:
            tau = 1.0 + self.alpha * torch.log(((position_ids + 1).float() / self.n_floor).clamp(min=1.0))
            xf = xf * tau[..., None, None]
        return xf.to(dt)


class Inkling2RelativePositionBias(nn.Module):
    def __init__(self, cfg: Inkling2Config, num_heads: int, rel_extent: int, log_scaling: bool):
        super().__init__()
        self.num_heads = num_heads
        self.rel_extent = rel_extent
        self.linear = nn.Linear(cfg.hidden_size, num_heads * cfg.d_rel, bias=False)
        self.norm = Inkling2ScaledL2Clip(
            num_heads, cfg.d_rel, cfg.rel_l2_clip_threshold,
            log_scaling=log_scaling, n_floor=cfg.log_scaling_n_floor, alpha=cfg.log_scaling_alpha,
        )
        self.proj = nn.Parameter(torch.empty(cfg.d_rel, rel_extent))

    def compute_rel_logits(self, x: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        rs = self.linear(x).view(*x.shape[:-1], self.num_heads, -1)
        rs = self.norm(rs, position_ids)
        return (rs.float() @ self.proj.float()).transpose(1, 2)

    def forward_eager(self, x: torch.Tensor, relative_distance: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        rel_logits = self.compute_rel_logits(x, position_ids)
        idx = relative_distance.clamp(0, self.rel_extent - 1).expand(*rel_logits.shape[:2], -1, -1)
        bias = rel_logits.gather(-1, idx)
        return bias.masked_fill((relative_distance < 0) | (relative_distance >= self.rel_extent), 0.0)


class Inkling2ShortConv(nn.Module):
    def __init__(self, dim: int, k_sz: int, layer_idx: int, conv_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.conv_idx = conv_idx
        self.k_sz = k_sz
        self.conv1d = nn.Conv1d(dim, dim, kernel_size=k_sz, groups=dim, padding=0, bias=False)
        self.conv1d.float()

    def forward(
        self,
        x: torch.Tensor,
        dynamic_input: torch.Tensor | None = None,
        cache: Inkling2Cache | None = None,
        vis_idx: int = 0,
        seq_idx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        dt = x.dtype
        xf = x.float()
        res = xf
        if cache is not None and cache.cur_conv_mask is not None:
            xf = xf * cache.cur_conv_mask.unsqueeze(-1)
        B, T, C = xf.shape
        xt = xf.transpose(1, 2)
        w = self.conv1d.weight.float()

        if cache is not None:
            st = cache.conv_states[vis_idx][self.conv_idx]
            cat = torch.cat([st, xt], dim=-1)
            out = F.conv1d(cat, w, padding=0, groups=C)
            cache.conv_states[vis_idx][self.conv_idx].copy_(cat[:, :, -(self.k_sz - 1) :])
        else:
            if seq_idx is not None:
                pad = F.pad(xt, (self.k_sz - 1, 0))
                win = pad.unfold(-1, self.k_sz, 1)
                s_pad = torch.cat([seq_idx[:, :1].expand(-1, self.k_sz - 1), seq_idx], dim=1)
                s_win = s_pad.unfold(-1, self.k_sz, 1)
                mask = (s_win == seq_idx.unsqueeze(-1)).unsqueeze(1).to(win.dtype)
                out = (win * mask * w.squeeze(1)[:, None, :]).sum(-1)
            else:
                pad = F.pad(xt, (self.k_sz - 1, 0))
                out = F.conv1d(pad, w, padding=0, groups=C)
        out = (out.transpose(1, 2) + res) / math.sqrt(self.k_sz + 1)
        return out.to(dt)


class Inkling2DynamicShortConv(nn.Module):
    def __init__(self, dim: int, input_dim: int, num_heads: int, k_sz: int, layer_idx: int, conv_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.conv_idx = conv_idx
        self.k_sz = k_sz
        self.num_heads = num_heads
        self.static_weight = nn.Parameter(torch.zeros(k_sz, dim))
        self.weight_proj = nn.Linear(input_dim, num_heads * k_sz, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        dynamic_input: torch.Tensor,
        cache: Inkling2Cache | None = None,
        vis_idx: int = 0,
        seq_idx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        dt = x.dtype
        B, T, C = x.shape
        dw = self.weight_proj(dynamic_input).float().view(B, T, self.num_heads, 1, self.k_sz)
        w = self.static_weight.float().T.view(self.num_heads, -1, self.k_sz) + dw

        xf = x.float()
        res = xf
        if cache is not None and cache.cur_conv_mask is not None:
            xf = xf * cache.cur_conv_mask.unsqueeze(-1)
        xt = xf.transpose(1, 2)
        if cache is not None:
            st = cache.conv_states[vis_idx][self.conv_idx]
            cat = torch.cat([st, xt], dim=-1)
            cache.conv_states[vis_idx][self.conv_idx].copy_(cat[:, :, -(self.k_sz - 1) :])
        else:
            cat = F.pad(xt, (self.k_sz - 1, 0))

        win = cat.unfold(-1, self.k_sz, 1).flip(-1).transpose(1, 2).reshape(B, T, self.num_heads, -1, self.k_sz)
        if seq_idx is not None and cache is None:
            s_pad = torch.cat([seq_idx[:, :1].expand(-1, self.k_sz - 1), seq_idx], dim=1)
            s_win = s_pad.unfold(-1, self.k_sz, 1).flip(-1)
            win = win * (s_win == seq_idx.unsqueeze(-1))[:, :, None, None, :]
        out = (win * w).sum(-1).reshape(B, T, C)
        out = (out + res) / math.sqrt(2 * self.k_sz + 1)
        return out.to(dt)


class Inkling2QueryProj(nn.Module):
    def __init__(self, cfg: Inkling2Config, num_heads: int, head_dim: int, log_scaling: bool):
        super().__init__()
        self.head_dim = head_dim
        self.linear = nn.Linear(cfg.hidden_size, num_heads * head_dim, bias=False)
        self.norm = Inkling2ScaledL2Clip(
            num_heads, head_dim, cfg.qk_l2_clip_threshold,
            log_scaling=log_scaling, n_floor=cfg.log_scaling_n_floor, alpha=cfg.log_scaling_alpha,
        )

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        s = self.linear(x).view(*x.shape[:-1], -1, self.head_dim)
        return self.norm(s, position_ids).transpose(1, 2)


class Inkling2KVProj(nn.Module):
    def __init__(self, cfg: Inkling2Config, num_heads: int, head_dim: int, layer_idx: int, conv_idx: int, is_key: bool, dynamic: bool):
        super().__init__()
        self.head_dim = head_dim
        self.is_key = is_key
        self.threshold = cfg.qk_l2_clip_threshold
        self.eps = cfg.rms_norm_eps
        self.linear = nn.Linear(cfg.hidden_size, num_heads * head_dim, bias=False)
        self.sconv = (
            Inkling2DynamicShortConv(num_heads * head_dim, cfg.hidden_size, num_heads, cfg.sconv_kernel_size, layer_idx, conv_idx)
            if dynamic
            else Inkling2ShortConv(num_heads * head_dim, cfg.sconv_kernel_size, layer_idx, conv_idx)
        )

    def forward(self, x: torch.Tensor, cache: Inkling2Cache | None, vis_idx: int, seq_idx: torch.Tensor | None) -> torch.Tensor:
        s = self.linear(x)
        s = self.sconv(s, x, cache=cache, vis_idx=vis_idx, seq_idx=seq_idx)
        s = s.view(*s.shape[:-1], -1, self.head_dim)
        if self.is_key:
            dt = s.dtype
            sf = s.float()
            n = torch.linalg.vector_norm(sf, dim=-1, keepdim=True)
            s = (sf * (self.threshold / (n + 1e-6)).clamp(max=1.0)).to(dt)
        else:
            s = rmsnorm_unweighted(s, self.eps)
        return s.transpose(1, 2)


class Inkling2GatedOutProj(nn.Module):
    def __init__(self, cfg: Inkling2Config, num_heads: int, head_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, num_heads * head_dim, bias=False)
        self.linear = nn.Linear(num_heads * head_dim, cfg.hidden_size, bias=False)

    def forward(self, attn_out: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        g = torch.sigmoid(self.gate_proj(x).float())
        return self.linear((attn_out.float() * g).to(attn_out.dtype))


class Inkling2Attention(nn.Module):
    def __init__(self, cfg: Inkling2Config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.is_sliding = cfg.is_sliding_layer(layer_idx)
        self.num_heads = cfg.swa_num_attention_heads if self.is_sliding else cfg.num_attention_heads
        self.num_kv_heads = cfg.swa_num_key_value_heads if self.is_sliding else cfg.num_key_value_heads
        self.head_dim = cfg.swa_head_dim if self.is_sliding else cfg.head_dim
        self.n_rep = self.num_heads // self.num_kv_heads
        self.rel_extent = cfg.sliding_window_size if self.is_sliding else cfg.rel_extent
        self.scaling = self.head_dim ** -0.5

        self.q_proj = Inkling2QueryProj(cfg, self.num_heads, self.head_dim, log_scaling=not self.is_sliding)
        self.k_proj = Inkling2KVProj(cfg, self.num_kv_heads, self.head_dim, layer_idx, 0, is_key=True, dynamic=not self.is_sliding)
        self.v_proj = Inkling2KVProj(cfg, self.num_kv_heads, self.head_dim, layer_idx, 1, is_key=False, dynamic=not self.is_sliding)
        self.o_proj = Inkling2GatedOutProj(cfg, self.num_heads, self.head_dim)
        self.sink_proj = nn.Linear(cfg.hidden_size, self.num_heads, bias=False)
        self.relative_position_bias = Inkling2RelativePositionBias(cfg, self.num_heads, self.rel_extent, log_scaling=not self.is_sliding)

    def forward(
        self,
        q_in: torch.Tensor,
        k_in: torch.Tensor,
        v_in: torch.Tensor,
        position_ids: torch.Tensor,
        cache: Inkling2Cache | None,
        vis_idx: int,
        seq_idx: torch.Tensor | None,
        attn_backend: str,
        eager_mask: torch.Tensor | None,
        rel_dist: torch.Tensor | None,
        block_mask: Any,
    ) -> torch.Tensor:
        B, T, _ = q_in.shape
        q = self.q_proj(q_in, position_ids)
        k = self.k_proj(k_in, cache=cache, vis_idx=vis_idx, seq_idx=seq_idx)
        v = self.v_proj(v_in, cache=cache, vis_idx=vis_idx, seq_idx=seq_idx)

        if cache is not None:
            k, v, g_end = cache.update_kv(vis_idx, k, v)
            kv_len = k.shape[2]
            q_g = torch.arange(g_end - T, g_end, device=q.device)
            kv_g = torch.arange(g_end - kv_len, g_end, device=q.device)
            rel_dist_eff = q_g[:, None] - kv_g[None, :]
            valid_c = ((rel_dist_eff >= 0) & ((not self.is_sliding) | (rel_dist_eff < self.rel_extent))) if T > 1 else None
            if cache.pad_lens is not None:
                valid_kv = kv_g[None, :] >= cache.pad_lens[:, None]
                valid_all = (valid_c[None, None, :, :] & valid_kv[:, None, None, :]) if valid_c is not None else valid_kv[:, None, None, :]
                c_mask = torch.where(valid_all, 0.0, float("-inf")).to(q.dtype)
            elif valid_c is not None:
                c_mask = torch.where(valid_c, 0.0, float("-inf")).to(q.dtype)[None, None]
            else:
                c_mask = None
            pos_bias = self.relative_position_bias.forward_eager(q_in, rel_dist_eff, position_ids)
            sinks = self.sink_proj(q_in).transpose(1, 2).unsqueeze(-1)
            ao = _eager_attn_with_sinks(q, k, v, pos_bias, sinks, c_mask, self.scaling, self.n_rep)
        elif attn_backend == "flex" and _HAS_FLEX and block_mask is not None:
            rel_logits = self.relative_position_bias.compute_rel_logits(q_in, position_ids)
            sinks = self.sink_proj(q_in).transpose(1, 2).unsqueeze(-1)
            ao = _flex_attn_with_sinks(q, k, v, rel_logits, sinks, block_mask, self.scaling, self.rel_extent)
        else:
            assert rel_dist is not None
            pos_bias = self.relative_position_bias.forward_eager(q_in, rel_dist, position_ids)
            sinks = self.sink_proj(q_in).transpose(1, 2).unsqueeze(-1)
            ao = _eager_attn_with_sinks(q, k, v, pos_bias, sinks, eager_mask, self.scaling, self.n_rep)

        return self.o_proj(ao.reshape(B, T, -1), q_in)


class Inkling2TopKRouter(nn.Module):
    def __init__(self, cfg: Inkling2Config):
        super().__init__()
        self.top_k = cfg.num_experts_per_tok
        self.num_experts = cfg.n_routed_experts
        self.route_scale = cfg.route_scale
        self.weight = nn.Parameter(torch.empty(cfg.n_routed_experts, cfg.hidden_size, dtype=torch.float32))
        self.register_buffer("e_score_correction_bias", torch.zeros(cfg.n_routed_experts, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = F.linear(x.float(), self.weight.float())
        scores = torch.sigmoid(logits)
        idx = torch.topk(scores + self.e_score_correction_bias.float(), self.top_k, dim=-1, sorted=False).indices
        w = scores.gather(1, idx)
        w = (w / (w.sum(dim=-1, keepdim=True) + 1e-20)) * self.route_scale
        return w, idx


class Inkling2Experts(nn.Module):
    def __init__(self, cfg: Inkling2Config):
        super().__init__()
        self.num_experts = cfg.n_routed_experts
        self.gate_cap = cfg.mlp_gate_softcap
        self.up_cap = cfg.mlp_up_softcap
        self.gate_up_proj = nn.Parameter(torch.empty(cfg.n_routed_experts, 2 * cfg.intermediate_size, cfg.hidden_size))
        self.down_proj = nn.Parameter(torch.empty(cfg.n_routed_experts, cfg.hidden_size, cfg.intermediate_size))

    def forward(self, x: torch.Tensor, top_k_idx: torch.Tensor, top_k_w: torch.Tensor) -> torch.Tensor:
        N, D = x.shape
        K = top_k_idx.shape[1]
        if N == 0:
            return x

        flat_exp = top_k_idx.reshape(-1)
        flat_tok = torch.arange(N, device=x.device).unsqueeze(1).expand(N, K).reshape(-1)
        flat_w = top_k_w.reshape(-1)

        if x.is_cuda and hasattr(torch, "_grouped_mm"):
            order = torch.argsort(flat_exp)
            s_exp = flat_exp[order]
            s_tok = flat_tok[order]
            s_w = flat_w[order]

            counts = torch.bincount(s_exp, minlength=self.num_experts)
            offs = counts.cumsum(0, dtype=torch.int32)
            xs = x[s_tok]
            w13 = self.gate_up_proj.to(xs.dtype)
            w2 = self.down_proj.to(xs.dtype)

            gu = torch._grouped_mm(xs, w13.transpose(1, 2), offs=offs)
            g, u = gu.chunk(2, dim=-1)
            gf, uf = g.float(), u.float()
            act = (_softcap(gf, self.gate_cap) * torch.sigmoid(gf) * _softcap(uf, self.up_cap)).to(xs.dtype)
            y = torch._grouped_mm(act, w2.transpose(1, 2), offs=offs).float() * s_w.unsqueeze(-1)

            out = torch.zeros(N, D, dtype=torch.float32, device=x.device)
            out.index_add_(0, s_tok, y)
            return out.to(x.dtype)

        # Portable CPU/fallback reference for unit testing
        out = torch.zeros(N, D, dtype=torch.float32, device=x.device)
        w13 = self.gate_up_proj.to(x.dtype)
        w2 = self.down_proj.to(x.dtype)
        for e in range(self.num_experts):
            mask = flat_exp == e
            if not bool(mask.any()):
                continue
            toks = flat_tok[mask]
            ws = flat_w[mask]
            gu = F.linear(x[toks], w13[e])
            g, u = gu.chunk(2, dim=-1)
            gf, uf = g.float(), u.float()
            act = (_softcap(gf, self.gate_cap) * torch.sigmoid(gf) * _softcap(uf, self.up_cap)).to(x.dtype)
            y = F.linear(act, w2[e]).float() * ws.unsqueeze(-1)
            out.index_add_(0, toks, y)
        return out.to(x.dtype)


class Inkling2SparseMoeBlock(nn.Module):
    def __init__(self, cfg: Inkling2Config):
        super().__init__()
        self.gate = Inkling2TopKRouter(cfg)
        self.experts = Inkling2Experts(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        flat = x.reshape(-1, D)
        w, idx = self.gate(flat)
        return self.experts(flat, idx, w).reshape(B, T, D)


class Inkling2BlockAttnRes(nn.Module):
    def __init__(self, dim: int, num_queries: int = 1, eps: float = 1e-6):
        super().__init__()
        shape = (dim,) if num_queries == 1 else (num_queries, dim)
        self.query = nn.Parameter(torch.zeros(*shape, dtype=torch.float32))
        self.eps = eps
        self.center_ref = True

    def forward(
        self,
        block_states: torch.Tensor,
        partial: torch.Tensor,
        h_carry: torch.Tensor | None = None,
        gamma_h: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...]:
        stacked = torch.cat([block_states, partial.unsqueeze(-2)], dim=-2)
        q = self.query.float().view(-1, stacked.shape[-1])
        out = attnres_2d_pool(q, stacked, h_carry, gamma_h, eps=self.eps, center_ref=self.center_ref)
        return out.unbind(-2)


class Inkling2MessageAttnRes(nn.Module):
    def __init__(self, num_loop_layers: int, dim: int, eps: float = 1e-6):
        super().__init__()
        self.query = nn.Parameter(torch.zeros(num_loop_layers, 4, dim, dtype=torch.float32))
        self.eps = eps
        self.center_ref = True

    def forward(
        self,
        block_states: torch.Tensor,
        partial: torch.Tensor,
        loop_layer_idx: int,
        h_carry: torch.Tensor | None = None,
        gamma_h: torch.Tensor | None = None,
    ) -> torch.Tensor:
        stacked = torch.cat([block_states, partial.unsqueeze(-2)], dim=-2)
        q = self.query[loop_layer_idx].float()
        return attnres_2d_pool(q, stacked, h_carry, gamma_h, eps=self.eps, center_ref=self.center_ref)


class Inkling2MessageGate(nn.Module):
    def __init__(self, dim: int, rank: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.down_proj = nn.Linear(2 * dim, rank, bias=False)
        self.forget_gate_proj = nn.Linear(rank, dim, bias=False, dtype=torch.float32)
        self.message_gate_proj = nn.Linear(rank, dim, bias=False, dtype=torch.float32)

    def forward(self, h: torch.Tensor, msg: torch.Tensor) -> torch.Tensor:
        hf, mf = h.float(), msg.float()
        cat = torch.cat([rmsnorm_unweighted(hf, self.eps), rmsnorm_unweighted(mf, self.eps)], dim=-1)
        g = self.down_proj(cat.to(self.down_proj.weight.dtype)).float()
        fg = torch.tanh(self.forget_gate_proj(g))
        mg = torch.tanh(self.message_gate_proj(g))
        return (1.0 - fg) * hf + mg * mf


class Inkling2DecoderLayer(nn.Module):
    def __init__(self, cfg: Inkling2Config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.is_sliding = cfg.is_sliding_layer(layer_idx)
        self.is_loop = cfg.loop_layer_index_start <= layer_idx < cfg.loop_layer_index_end
        self.is_block_boundary = (layer_idx % cfg.n_layers_per_attn_res_block) == 0

        if layer_idx > 0:
            self.attention_input = Inkling2BlockAttnRes(cfg.hidden_size, num_queries=(1 if self.is_sliding else 3), eps=cfg.rms_norm_eps)
        else:
            self.attention_input = None
        self.mlp_input = Inkling2BlockAttnRes(cfg.hidden_size, num_queries=1, eps=cfg.rms_norm_eps)

        self.self_attn = Inkling2Attention(cfg, layer_idx)
        self.mlp = Inkling2SparseMoeBlock(cfg)

        if self.is_sliding:
            self.input_layernorm = Inkling2RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        else:
            self.q_input_layernorm = Inkling2RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
            self.k_input_layernorm = Inkling2RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
            self.v_input_layernorm = Inkling2RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.post_attention_layernorm = Inkling2RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)

        self.post_attention_convolution = Inkling2ShortConv(cfg.hidden_size, cfg.sconv_kernel_size, layer_idx, 2)
        self.post_mlp_convolution = Inkling2ShortConv(cfg.hidden_size, cfg.sconv_kernel_size, layer_idx, 3)

        if self.is_loop:
            self.q_message_gate = Inkling2MessageGate(cfg.hidden_size, cfg.loop_gate_rank, cfg.rms_norm_eps)
            self.k_message_gate = Inkling2MessageGate(cfg.hidden_size, cfg.loop_gate_rank, cfg.rms_norm_eps)
            self.v_message_gate = Inkling2MessageGate(cfg.hidden_size, cfg.loop_gate_rank, cfg.rms_norm_eps)
            self.mlp_message_gate = Inkling2MessageGate(cfg.hidden_size, cfg.loop_gate_rank, cfg.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        block_states: torch.Tensor,
        position_ids: torch.Tensor,
        messages: torch.Tensor | None = None,
        ws_messages: torch.Tensor | None = None,
        gamma_ws: torch.Tensor | None = None,
        cache: Inkling2Cache | None = None,
        vis_idx: int = 0,
        seq_idx: torch.Tensor | None = None,
        attn_backend: str = "eager",
        eager_mask: torch.Tensor | None = None,
        rel_dist: torch.Tensor | None = None,
        block_mask: Any = None,
        h_carry: torch.Tensor | None = None,
        g_h_att: torch.Tensor | None = None,
        g_h_mlp: torch.Tensor | None = None,
    ) -> torch.Tensor:
        dt = hidden_states.dtype
        block_res = hidden_states
        use_ws = (messages is None) and (ws_messages is not None) and (gamma_ws is not None) and self.is_loop
        if use_ws:
            ws_valid = (ws_messages.float().square().mean(dim=(-1, -2)) > 1e-12).float().unsqueeze(-1)
            gq0, gk0, gv0, gm0 = torch.tanh(gamma_ws.float()).unbind(0)
            gq, gk, gv, gm = gq0 * ws_valid, gk0 * ws_valid, gv0 * ws_valid, gm0 * ws_valid
        else:
            gq = gk = gv = gm = None

        if self.is_sliding:
            h_in = (
                self.attention_input(block_states, hidden_states, h_carry=h_carry, gamma_h=g_h_att)[0]
                if self.attention_input is not None
                else hidden_states
            )
            if messages is None and not use_ws:
                qh = kh = vh = self.input_layernorm(h_in).to(dt)
            elif use_ws:
                qm, km, vm, _ = ws_messages.unbind(-2)
                hf = h_in.float()
                qh = self.input_layernorm(hf + gq * (self.q_message_gate(h_in, qm) - hf)).to(dt)
                kh = self.input_layernorm(hf + gk * (self.k_message_gate(h_in, km) - hf)).to(dt)
                vh = self.input_layernorm(hf + gv * (self.v_message_gate(h_in, vm) - hf)).to(dt)
            else:
                qm, km, vm, _ = messages.unbind(-2)
                qh = self.input_layernorm(self.q_message_gate(h_in, qm)).to(dt)
                kh = self.input_layernorm(self.k_message_gate(h_in, km)).to(dt)
                vh = self.input_layernorm(self.v_message_gate(h_in, vm)).to(dt)
        else:
            if self.attention_input is None:
                qh0 = kh0 = vh0 = hidden_states
            else:
                qh0, kh0, vh0 = self.attention_input(block_states, hidden_states, h_carry=h_carry, gamma_h=g_h_att)
            if messages is not None:
                qm, km, vm, _ = messages.unbind(-2)
                qh0 = self.q_message_gate(qh0, qm)
                kh0 = self.k_message_gate(kh0, km)
                vh0 = self.v_message_gate(vh0, vm)
            elif use_ws:
                qm, km, vm, _ = ws_messages.unbind(-2)
                qf, kf, vf = qh0.float(), kh0.float(), vh0.float()
                qh0 = qf + gq * (self.q_message_gate(qh0, qm) - qf)
                kh0 = kf + gk * (self.k_message_gate(kh0, km) - kf)
                vh0 = vf + gv * (self.v_message_gate(vh0, vm) - vf)
            qh = self.q_input_layernorm(qh0).to(dt)
            kh = self.k_input_layernorm(kh0).to(dt)
            vh = self.v_input_layernorm(vh0).to(dt)

        y_att = self.self_attn(
            qh, kh, vh, position_ids=position_ids, cache=cache, vis_idx=vis_idx, seq_idx=seq_idx,
            attn_backend=attn_backend, eager_mask=eager_mask, rel_dist=rel_dist, block_mask=block_mask,
        )
        y_att = self.post_attention_convolution(y_att, cache=cache, vis_idx=vis_idx, seq_idx=seq_idx)

        res = y_att.float()
        if self.is_block_boundary:
            block_states = torch.cat([block_states, block_res.unsqueeze(-2)], dim=-2)
        else:
            res = block_res.float() + res

        m_in = self.mlp_input(block_states, res, h_carry=h_carry, gamma_h=g_h_mlp)[0]
        if messages is not None:
            m_in = self.mlp_message_gate(m_in, messages[..., 3, :])
        elif use_ws:
            mf = m_in.float()
            m_in = mf + gm * (self.mlp_message_gate(m_in, ws_messages[..., 3, :]) - mf)
        y_mlp = self.mlp(self.post_attention_layernorm(m_in).to(dt))
        y_mlp = self.post_mlp_convolution(y_mlp, cache=cache, vis_idx=vis_idx, seq_idx=seq_idx)
        return (res + y_mlp.float()).to(dt)


class Inkling2TextModel(nn.Module):
    def __init__(self, cfg: Inkling2Config):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = Inkling2NormedEmbedding(cfg.vocab_size, cfg.hidden_size, cfg.pad_token_id, cfg.rms_norm_eps)
        self.layers = nn.ModuleList([Inkling2DecoderLayer(cfg, i) for i in range(cfg.num_hidden_layers)])
        self.norm = Inkling2RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.final_input = Inkling2BlockAttnRes(cfg.hidden_size, num_queries=1, eps=cfg.rms_norm_eps)
        self.message_input = Inkling2MessageAttnRes(cfg.loop_layer_index_end - cfg.loop_layer_index_start, cfg.hidden_size, cfg.rms_norm_eps)


class AttnRes2DLoopedMoE(nn.Module):
    """26.4B Looped MoE (`TML-d24-varloop-research`) retrofitted with Unified 2D-AttnRes.

    Modes:
      - `"std"`: Stock TML-d24-varloop (0 added params, 1 pass).
      - `"attnres_2d"`: Unified 2D-AttnRes Pool + 1-Line Contractive Gate λ at `3L/4`
        (2,448 scalars = 9.8 KB fp32 at P=1; 2,520 scalars = 10.1 KB at P=2; +0 B KV).
    """

    VALID_MODES = ("std", "attnres_2d")
    EXTRA_PREFIXES = ("g_h_", "theta_lambda")

    def __init__(
        self,
        cfg: Inkling2Config,
        mode: str = "attnres_2d",
        msg0_pool_init: float = 0.35,
        grad_checkpoint: bool = False,
        attn_backend: str = "flex",
    ):
        super().__init__()
        if mode not in self.VALID_MODES:
            raise ValueError(f"Unknown mode={mode!r}; expected one of {self.VALID_MODES}")
        self.cfg = cfg
        self.mode = mode
        self.has_carry = (mode == "attnres_2d")
        self.msg0_pool_init = float(msg0_pool_init)
        self.grad_checkpoint = grad_checkpoint
        self.attn_backend = attn_backend if (attn_backend == "flex" and _HAS_FLEX) else "eager"

        self.model = Inkling2TextModel(cfg)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.unpadded_vocab_size, bias=False)

        self.loop_start = cfg.loop_layer_index_start      # 6
        self.loop_end = cfg.loop_layer_index_end          # 18 == 3L/4
        self.core_len = self.loop_end - self.loop_start   # 12
        self.post_len = cfg.num_hidden_layers - self.loop_end  # 6
        self.loop_passes = cfg.loop_num_passes

        if self.has_carry:
            dim = cfg.hidden_size
            # Zero-initialized scalar gates into every native BlockAttnRes & MessageAttnRes pool
            self.g_h_pre_att = nn.Parameter(torch.zeros(self.loop_start - 1, dtype=torch.float32))
            self.g_h_pre_mlp = nn.Parameter(torch.zeros(self.loop_start, dtype=torch.float32))
            self.g_h_core_att = nn.Parameter(torch.zeros(self.core_len, self.loop_passes, dtype=torch.float32))
            self.g_h_core_mlp = nn.Parameter(torch.zeros(self.core_len, self.loop_passes, dtype=torch.float32))
            self.g_h_post_att = nn.Parameter(torch.zeros(self.post_len, dtype=torch.float32))
            self.g_h_post_mlp = nn.Parameter(torch.zeros(self.post_len, dtype=torch.float32))
            self.g_h_final = nn.Parameter(torch.zeros((), dtype=torch.float32))

            # Pass-0 MessageGate cross-token read (g_h_msg0=0 preserves exact 0.0 step-0 identity;
            # g_h_msg0_pool=0.35 breaks the 0x0 saddle so H_{t-1} mixes cleanly into pass-0 MessageAttnRes!)
            self.g_h_msg0 = nn.Parameter(torch.zeros(self.core_len, 4, dtype=torch.float32))
            self.g_h_msg0_pool = nn.Parameter(torch.full((self.core_len, 4), self.msg0_pool_init, dtype=torch.float32))
            if self.loop_passes > 1:
                self.g_h_msg_pool = nn.Parameter(torch.zeros(self.core_len, self.loop_passes - 1, 4, dtype=torch.float32))
            else:
                self.register_parameter("g_h_msg_pool", None)

            # 1-line multi-timescale contractive forgetting gate λ in (0, 1)^D
            self.theta_lambda = nn.Parameter(init_multiscale_lambda_logits(dim, 0.25, 0.95))
        else:
            for pname in (
                "g_h_pre_att", "g_h_pre_mlp", "g_h_core_att", "g_h_core_mlp",
                "g_h_post_att", "g_h_post_mlp", "g_h_final",
                "g_h_msg0", "g_h_msg0_pool", "g_h_msg_pool", "theta_lambda",
            ):
                self.register_parameter(pname, None)

    def count_added_params(self) -> int:
        if not self.has_carry:
            return 0
        return int(sum(p.numel() for n, p in self.named_parameters() if n.startswith(self.EXTRA_PREFIXES)))

    @classmethod
    def from_bf16_dir(
        cls,
        ckpt_dir: Path,
        mode: str = "attnres_2d",
        loop_num_passes: int | None = None,
        msg0_pool_init: float = 0.35,
        grad_checkpoint: bool = False,
        attn_backend: str = "flex",
        dtype: torch.dtype = torch.bfloat16,
        device: str | torch.device = "cpu",
    ) -> "AttnRes2DLoopedMoE":
        cfg = Inkling2Config.from_json(ckpt_dir / "config.json", loop_num_passes=loop_num_passes)
        fp32_sub = (
            "e_score_correction_bias", "mlp.gate.weight", "post_attention_convolution",
            "post_mlp_convolution", "sconv.conv1d", "attention_input", "mlp_input",
            "final_input", "message_input", "message_gate_proj", "forget_gate_proj",
            "theta_lambda", "g_h_",
        )
        with torch.device("meta"):
            inst = cls(cfg, mode=mode, msg0_pool_init=msg0_pool_init, grad_checkpoint=grad_checkpoint, attn_backend=attn_backend)
        for name, p in inst.named_parameters():
            p.data = p.data.to(torch.float32 if any(s in name for s in fp32_sub) else dtype)
        for _, b in inst.named_buffers():
            b.data = b.data.float()
        inst.to_empty(device=device)

        if inst.has_carry:
            with torch.no_grad():
                inst.g_h_pre_att.zero_()
                inst.g_h_pre_mlp.zero_()
                inst.g_h_core_att.zero_()
                inst.g_h_core_mlp.zero_()
                inst.g_h_post_att.zero_()
                inst.g_h_post_mlp.zero_()
                inst.g_h_final.zero_()
                inst.g_h_msg0.zero_()
                inst.g_h_msg0_pool.fill_(msg0_pool_init)
                if inst.g_h_msg_pool is not None:
                    inst.g_h_msg_pool.zero_()
                inst.theta_lambda.copy_(init_multiscale_lambda_logits(cfg.hidden_size, 0.25, 0.95).to(device))

        idx = json.loads((ckpt_dir / "model.safetensors.index.json").read_text())
        shard_files = sorted(set(idx["weight_map"].values()))
        dev_str = str(device)
        expected_keys = {k for k in inst.state_dict().keys() if not k.startswith(cls.EXTRA_PREFIXES)}
        loaded_keys: set[str] = set()
        for sf in shard_files:
            sd = load_file(str(ckpt_dir / sf), device=dev_str)
            loaded_keys.update(sd.keys())
            _, unexpected = inst.load_state_dict(sd, strict=False)
            if unexpected:
                raise RuntimeError(f"Unexpected keys in {sf}: {unexpected[:10]}")
            del sd
        missing = sorted(expected_keys - loaded_keys)
        if missing:
            raise RuntimeError(f"Missing keys in {ckpt_dir}: {missing[:10]}")

        if inst.has_carry:
            extra_p = ckpt_dir / "attnres_2d_extra.pt"
            if extra_p.exists():
                extra = torch.load(extra_p, map_location=device, weights_only=False)
                inst.load_state_dict(extra.get("attnres_2d_state", {}), strict=False)
        return inst

    def _build_masks(
        self,
        B: int,
        T: int,
        device: torch.device,
        dtype: torch.dtype,
        doc_start_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, Any, Any, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        if doc_start_mask is not None:
            starts = doc_start_mask.bool().clone()
            starts[:, 0] = True
            seq_idx = starts.long().cumsum(dim=1) - 1
            pos = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
            doc_base = torch.where(starts, pos, torch.zeros_like(pos)).cummax(dim=1).values
            position_ids = pos - doc_base
        else:
            seq_idx = None
            position_ids = torch.arange(T, device=device).unsqueeze(0).expand(B, T)

        if self.attn_backend == "flex" and _HAS_FLEX and create_block_mask is not None:
            win = self.cfg.sliding_window_size
            if seq_idx is not None:
                s_id = seq_idx
                def full_mod(b, h, q_i, kv_i):
                    return (q_i >= kv_i) & (s_id[b, q_i] == s_id[b, kv_i])
                def swa_mod(b, h, q_i, kv_i):
                    return (q_i >= kv_i) & ((q_i - kv_i) < win) & (s_id[b, q_i] == s_id[b, kv_i])
            else:
                def full_mod(b, h, q_i, kv_i):
                    return q_i >= kv_i
                def swa_mod(b, h, q_i, kv_i):
                    return (q_i >= kv_i) & ((q_i - kv_i) < win)
            bm_full = create_block_mask(full_mod, B=B if seq_idx is not None else None, H=None, Q_LEN=T, KV_LEN=T, device=device)
            bm_swa = create_block_mask(swa_mod, B=B if seq_idx is not None else None, H=None, Q_LEN=T, KV_LEN=T, device=device)
            return position_ids, seq_idx, bm_full, bm_swa, None, None, None

        p_idx = torch.arange(T, device=device)
        rel_dist = p_idx[:, None] - p_idx[None, :]
        causal = rel_dist >= 0
        swa = causal & (rel_dist < self.cfg.sliding_window_size)
        if seq_idx is not None:
            same_doc = seq_idx[:, None, :, None] == seq_idx[:, None, None, :]
            m_full = torch.where(causal[None, None] & same_doc, 0.0, float("-inf")).to(dtype)
            m_swa = torch.where(swa[None, None] & same_doc, 0.0, float("-inf")).to(dtype)
        else:
            m_full = torch.where(causal, 0.0, float("-inf")).to(dtype)[None, None]
            m_swa = torch.where(swa, 0.0, float("-inf")).to(dtype)[None, None]
        return position_ids, seq_idx, None, None, m_full, m_swa, rel_dist

    def _run_layer(
        self,
        layer: Inkling2DecoderLayer,
        h: torch.Tensor,
        bs: torch.Tensor,
        position_ids: torch.Tensor,
        messages: torch.Tensor | None,
        cache: Inkling2Cache | None,
        vis_idx: int,
        seq_idx: torch.Tensor | None,
        bm_full: Any,
        bm_swa: Any,
        m_full: torch.Tensor | None,
        m_swa: torch.Tensor | None,
        rel_dist: torch.Tensor | None,
        ws_messages: torch.Tensor | None = None,
        gamma_ws: torch.Tensor | None = None,
        h_carry: torch.Tensor | None = None,
        g_h_att: torch.Tensor | None = None,
        g_h_mlp: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bm = bm_swa if layer.is_sliding else bm_full
        em = m_swa if layer.is_sliding else m_full
        if self.grad_checkpoint and self.training and cache is None:
            return checkpoint(
                layer, h, bs, position_ids, messages, ws_messages, gamma_ws,
                None, vis_idx, seq_idx, self.attn_backend, em, rel_dist, bm,
                h_carry, g_h_att, g_h_mlp,
                use_reentrant=False,
            )
        return layer(
            h, bs, position_ids, messages=messages, ws_messages=ws_messages, gamma_ws=gamma_ws,
            cache=cache, vis_idx=vis_idx, seq_idx=seq_idx, attn_backend=self.attn_backend,
            eager_mask=em, rel_dist=rel_dist, block_mask=bm,
            h_carry=h_carry, g_h_att=g_h_att, g_h_mlp=g_h_mlp,
        )

    def _forward_one_pass(
        self,
        input_ids: torch.LongTensor,
        h_carry: torch.Tensor | None,
        position_ids: torch.Tensor,
        seq_idx: torch.Tensor | None,
        bm_full: Any,
        bm_swa: Any,
        m_full: torch.Tensor | None,
        m_swa: torch.Tensor | None,
        rel_dist: torch.Tensor | None,
        need_postlude: bool,
        cache: Inkling2Cache | None = None,
        ws_state: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, tuple[torch.Tensor, torch.Tensor] | None]:
        """Run one pass through Prelude -> Looped Core -> (optional) Postlude."""
        h = self.model.embed_tokens(input_ids)
        bs = h.new_zeros(*h.shape[:-1], 0, h.shape[-1])
        pool_h = h_carry if (self.has_carry and h_carry is not None) else None

        # 1. Prelude: layers 0..5
        for l_idx in range(self.loop_start):
            layer = self.model.layers[l_idx]
            gh_att = self.g_h_pre_att[l_idx - 1] if (pool_h is not None and l_idx > 0) else None
            gh_mlp = self.g_h_pre_mlp[l_idx] if pool_h is not None else None
            new_h = self._run_layer(
                layer, h, bs, position_ids, None, cache, l_idx, seq_idx,
                bm_full, bm_swa, m_full, m_swa, rel_dist,
                h_carry=pool_h, g_h_att=gh_att, g_h_mlp=gh_mlp,
            )
            if layer.is_block_boundary:
                bs = torch.cat([bs, h.unsqueeze(-2)], dim=-2)
            h = new_h

        pre_h, pre_bs = h, bs
        num_pre_blocks = pre_bs.shape[-2]

        # 2. Looped Core: layers 6..17 repeated loop_passes times (P = 1..5)
        for p_idx in range(self.loop_passes):
            if p_idx == 0:
                prev_bs = prev_h = None
            else:
                prev_bs = bs[..., num_pre_blocks + 1 :, :]
                prev_h = h
                h, bs = pre_h, pre_bs

            for local_idx in range(self.core_len):
                l_idx = self.loop_start + local_idx
                layer = self.model.layers[l_idx]
                vis_idx = l_idx if p_idx == 0 else (self.cfg.num_hidden_layers + (p_idx - 1) * self.core_len + local_idx)

                if p_idx > 0:
                    gm_pool = (
                        self.g_h_msg_pool[local_idx, p_idx - 1]
                        if (self.has_carry and self.g_h_msg_pool is not None and pool_h is not None)
                        else None
                    )
                    msgs = self.model.message_input(prev_bs, prev_h, local_idx, h_carry=pool_h, gamma_h=gm_pool)
                else:
                    msgs = None

                ws_msgs = None
                g_ws = None
                if self.has_carry and p_idx == 0 and ws_state is not None:
                    gm0p = self.g_h_msg0_pool[local_idx] if pool_h is not None else None
                    ws_msgs = self.model.message_input(ws_state[0], ws_state[1], local_idx, h_carry=pool_h, gamma_h=gm0p)
                    g_ws = self.g_h_msg0[local_idx]

                gh_att = self.g_h_core_att[local_idx, p_idx] if pool_h is not None else None
                gh_mlp = self.g_h_core_mlp[local_idx, p_idx] if pool_h is not None else None
                new_h = self._run_layer(
                    layer, h, bs, position_ids, msgs, cache, vis_idx, seq_idx,
                    bm_full, bm_swa, m_full, m_swa, rel_dist,
                    ws_messages=ws_msgs, gamma_ws=g_ws,
                    h_carry=pool_h, g_h_att=gh_att, g_h_mlp=gh_mlp,
                )
                if layer.is_block_boundary:
                    bs = torch.cat([bs, h.unsqueeze(-2)], dim=-2)
                h = new_h

        # 1-Line Contractive Carry Update strictly at Core exit (layer 18 == 3L/4)!
        h_next: torch.Tensor | None = None
        ws_next: tuple[torch.Tensor, torch.Tensor] | None = None
        if self.has_carry and h_carry is not None:
            h_next = update_contractive_carry(h_carry, h.float() - pre_h.float(), self.theta_lambda, eps=self.cfg.rms_norm_eps)
            ws_next = (bs[..., num_pre_blocks + 1 :, :], h)

        if not need_postlude:
            return None, h_next, ws_next

        # 3. Postlude (Readout Zone): layers 18..23 (ZERO carry writes!)
        for post_i in range(self.post_len):
            l_idx = self.loop_end + post_i
            layer = self.model.layers[l_idx]
            gh_att = self.g_h_post_att[post_i] if pool_h is not None else None
            gh_mlp = self.g_h_post_mlp[post_i] if pool_h is not None else None
            new_h = self._run_layer(
                layer, h, bs, position_ids, None, cache, l_idx, seq_idx,
                bm_full, bm_swa, m_full, m_swa, rel_dist,
                h_carry=pool_h, g_h_att=gh_att, g_h_mlp=gh_mlp,
            )
            if layer.is_block_boundary:
                bs = torch.cat([bs, h.unsqueeze(-2)], dim=-2)
            h = new_h

        gh_fin = self.g_h_final if pool_h is not None else None
        h = self.model.final_input(bs, h, h_carry=pool_h, gamma_h=gh_fin)[0]
        h = self.model.norm(h).to(self.lm_head.weight.dtype)
        return h, h_next, ws_next

    def compute_logits(self, h: torch.Tensor) -> torch.Tensor:
        return _softcap(self.lm_head(h).float(), self.cfg.final_logit_softcapping)

    def forward_multipass(
        self,
        input_ids: torch.LongTensor,
        passes: int = 4,
        backprop_last: int = 2,
        doc_start_mask: torch.Tensor | None = None,
        return_hidden: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Exact sub-readout K-pass Jacobi parallel forward (passes 1..K-1 skip Postlude & LM head)."""
        B, T = input_ids.shape
        device = input_ids.device
        dtype = self.lm_head.weight.dtype
        position_ids, seq_idx, bm_full, bm_swa, m_full, m_swa, rel_dist = self._build_masks(B, T, device, dtype, doc_start_mask)

        if not self.has_carry:
            h, _, _ = self._forward_one_pass(
                input_ids, None, position_ids, seq_idx, bm_full, bm_swa, m_full, m_swa, rel_dist, need_postlude=True,
            )
            assert h is not None
            return {"hidden": h} if return_hidden else {"logits": self.compute_logits(h)}

        h_carry = torch.zeros(B, T, self.cfg.hidden_size, dtype=dtype, device=device)
        ws_in: tuple[torch.Tensor, torch.Tensor] | None = None
        no_grad_passes = max(0, passes - backprop_last)
        h_final: torch.Tensor | None = None

        for p in range(passes):
            is_last = (p == passes - 1)
            if p < no_grad_passes:
                with torch.no_grad():
                    _, h_out, ws_out = self._forward_one_pass(
                        input_ids, h_carry, position_ids, seq_idx, bm_full, bm_swa, m_full, m_swa, rel_dist,
                        need_postlude=False, ws_state=ws_in,
                    )
                assert h_out is not None
                h_carry = shift_carry(h_out, doc_start_mask).detach()
                ws_s = shift_state_pair(ws_out, doc_start_mask)
                ws_in = (ws_s[0].detach(), ws_s[1].detach()) if ws_s is not None else None
            else:
                h_final, h_out, ws_out = self._forward_one_pass(
                    input_ids, h_carry, position_ids, seq_idx, bm_full, bm_swa, m_full, m_swa, rel_dist,
                    need_postlude=is_last, ws_state=ws_in,
                )
                if not is_last:
                    assert h_out is not None
                    h_carry = shift_carry(h_out, doc_start_mask)
                    ws_in = shift_state_pair(ws_out, doc_start_mask)
        assert h_final is not None
        return {"hidden": h_final} if return_hidden else {"logits": self.compute_logits(h_final)}

    @torch.no_grad()
    def forward_exact_sequential(
        self,
        input_ids: torch.LongTensor,
        carry_cut: bool = False,
    ) -> torch.Tensor:
        """Exact token-by-token sequential recurrence using Inkling2Cache."""
        B, T = input_ids.shape
        device = input_ids.device
        dtype = self.lm_head.weight.dtype
        cache = Inkling2Cache(self.cfg, B, init_cap=max(2048, T + 8), max_len=T + 8, device=device, dtype=dtype)
        use_carry = self.has_carry and (not carry_cut)
        h_carry = torch.zeros(B, 1, self.cfg.hidden_size, dtype=dtype, device=device) if use_carry else None
        ws_in: tuple[torch.Tensor, torch.Tensor] | None = None

        logits_list = []
        for t in range(T):
            tok = input_ids[:, t : t + 1]
            pos = torch.full((B, 1), t, dtype=torch.long, device=device)
            h, h_next, ws_next = self._forward_one_pass(
                tok, h_carry, pos, None, None, None, None, None, None,
                need_postlude=True, cache=cache, ws_state=ws_in,
            )
            assert h is not None
            logits_list.append(self.compute_logits(h))
            if use_carry and h_next is not None:
                h_carry = h_next
                ws_in = ws_next
        return torch.cat(logits_list, dim=1)

    def get_param_groups(
        self,
        lr_base: float = 5e-6,
        lr_attnres_q: float = 0.0,
        lr_lambda: float = 3e-4,
        lr_scalar: float = 1e-2,
        weight_decay: float = 0.01,
    ) -> list[dict[str, Any]]:
        """Return standard AdamW parameter groups (no custom optimizer required)."""
        attnres_q_sub = ("attention_input.query", "mlp_input.query", "final_input.query", "message_input.query")
        base_decay, base_nodecay, attnres_q_p, lambda_p, scalar_p = [], [], [], [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if name.startswith("g_h_"):
                scalar_p.append(p)
            elif name == "theta_lambda":
                lambda_p.append(p)
            elif lr_attnres_q > 0.0 and any(s in name for s in attnres_q_sub):
                attnres_q_p.append(p)
            elif p.ndim >= 2 and "norm" not in name and "embed" not in name:
                base_decay.append(p)
            else:
                base_nodecay.append(p)
        groups = []
        if base_decay:
            groups.append({"params": base_decay, "lr": lr_base, "weight_decay": weight_decay, "group_name": "base_decay"})
        if base_nodecay:
            groups.append({"params": base_nodecay, "lr": lr_base, "weight_decay": 0.0, "group_name": "base_nodecay"})
        if attnres_q_p:
            groups.append({"params": attnres_q_p, "lr": lr_attnres_q, "weight_decay": 0.0, "group_name": "attnres_q"})
        if lambda_p:
            groups.append({"params": lambda_p, "lr": lr_lambda, "weight_decay": 0.0, "group_name": "lambda"})
        if scalar_p:
            groups.append({"params": scalar_p, "lr": lr_scalar, "weight_decay": 0.0, "group_name": "scalars"})
        return groups

    def collect_telemetry(self) -> dict[str, Any]:
        if not self.has_carry:
            return {}
        lams = torch.sigmoid(self.theta_lambda.detach().float().cpu())
        gm0 = torch.tanh(self.g_h_msg0.detach().float().cpu())
        tel: dict[str, Any] = {
            "g_h_pre_att": [round(float(x), 4) for x in self.g_h_pre_att.detach().float().cpu().tolist()],
            "g_h_pre_mlp": [round(float(x), 4) for x in self.g_h_pre_mlp.detach().float().cpu().tolist()],
            "g_h_core_att_by_pass": [round(float(x), 4) for x in self.g_h_core_att.detach().float().abs().mean(dim=0).cpu().tolist()],
            "g_h_core_mlp_by_pass": [round(float(x), 4) for x in self.g_h_core_mlp.detach().float().abs().mean(dim=0).cpu().tolist()],
            "g_h_post_att_max": round(float(self.g_h_post_att.detach().float().abs().max().item()), 4),
            "g_h_post_mlp_max": round(float(self.g_h_post_mlp.detach().float().abs().max().item()), 4),
            "g_h_final": round(float(self.g_h_final.detach().float().item()), 4),
            "g_h_msg0_mean": round(float(gm0.abs().mean().item()), 4),
            "g_h_msg0_max": round(float(gm0.abs().max().item()), 4),
            "g_h_msg0_qkv_mlp": [round(float(x), 4) for x in gm0.abs().mean(dim=0).tolist()],
            "lambda_mean": round(float(lams.mean().item()), 4),
            "lambda_p10_p90": [round(float(lams.quantile(0.1).item()), 4), round(float(lams.quantile(0.9).item()), 4)],
        }
        if self.g_h_msg_pool is not None:
            gmp = self.g_h_msg_pool.detach().float().cpu()
            tel["g_h_msg_pool_mean"] = round(float(gmp.abs().mean().item()), 4)
            tel["g_h_msg_pool_max"] = round(float(gmp.abs().max().item()), 4)
        return tel


@torch.no_grad()
def generate_batch_looped_moe(
    model: AttnRes2DLoopedMoE,
    prompt_token_lists: list[list[int]],
    eos_ids: set[int],
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 0.95,
    presence_penalty: float = 0.0,
    carry_cut: bool = False,
    prefill_passes: int = 4,
    device: torch.device | None = None,
) -> tuple[list[list[int]], list[bool]]:
    """Fast exact left-padded batched autoregressive generation with active-slot compaction."""
    if device is None:
        device = next(model.parameters()).device
    dtype = model.lm_head.weight.dtype
    B = len(prompt_token_lists)
    p_lens = [len(p) for p in prompt_token_lists]
    max_p = max(p_lens)
    pad_lens = [max_p - l for l in p_lens]
    max_len = max_p + max_new_tokens + 8

    cache = Inkling2Cache(model.cfg, B, init_cap=min(max_len, max(2048, max_p + 512)), max_len=max_len, device=device, dtype=dtype)
    if any(pl > 0 for pl in pad_lens):
        cache.pad_lens = torch.tensor(pad_lens, dtype=torch.long, device=device)

    pad_id = model.cfg.pad_token_id
    p_ids = torch.full((B, max_p), pad_id, dtype=torch.long, device=device)
    for i, p in enumerate(prompt_token_lists):
        p_ids[i, pad_lens[i] :] = torch.tensor(p, dtype=torch.long, device=device)

    t_grid = torch.arange(max_p, device=device).unsqueeze(0).expand(B, max_p)
    pad_t = torch.tensor(pad_lens, dtype=torch.long, device=device).unsqueeze(1)
    pos_p = (t_grid - pad_t).clamp(min=0)
    valid_p = (t_grid >= pad_t).float()

    use_carry = model.has_carry and (not carry_cut)
    dim = model.cfg.hidden_size
    h_carry = torch.zeros(B, 1, dim, dtype=dtype, device=device) if use_carry else None
    ws_cur: tuple[torch.Tensor, torch.Tensor] | None = None

    # 1. Prompt Prefill
    if not use_carry:
        cache.cur_conv_mask = valid_p if cache.pad_lens is not None else None
        h_p, _, _ = model._forward_one_pass(p_ids, None, pos_p, None, None, None, None, None, None, need_postlude=True, cache=cache)
        cache.cur_conv_mask = None
        assert h_p is not None
        last_logits = model.compute_logits(h_p[:, -1, :])
    elif prefill_passes >= 1:
        doc_start_p = torch.zeros(B, max_p, dtype=torch.bool, device=device)
        for i, pl in enumerate(pad_lens):
            doc_start_p[i, pl] = True
        pos_m, seq_m, bm_full, bm_swa, m_full, m_swa, rel_dist = model._build_masks(B, max_p, device, dtype, doc_start_p)
        h_seq = torch.zeros(B, max_p, dim, dtype=dtype, device=device)
        ws_seq: tuple[torch.Tensor, torch.Tensor] | None = None
        v_mask = valid_p.unsqueeze(-1).to(dtype)
        for _ in range(prefill_passes - 1):
            _, h_out, ws_out = model._forward_one_pass(
                p_ids, h_seq, pos_m, seq_m, bm_full, bm_swa, m_full, m_swa, rel_dist,
                need_postlude=False, cache=None, ws_state=ws_seq,
            )
            assert h_out is not None
            h_seq = shift_carry(h_out, doc_start_p) * v_mask
            ws_s = shift_state_pair(ws_out, doc_start_p)
            ws_seq = (ws_s[0] * v_mask.unsqueeze(-2), ws_s[1] * v_mask) if ws_s is not None else None
        cache.cur_conv_mask = valid_p if cache.pad_lens is not None else None
        h_p, h_next_seq, ws_next_seq = model._forward_one_pass(
            p_ids, h_seq, pos_p, None, None, None, None, None, None,
            need_postlude=True, cache=cache, ws_state=ws_seq,
        )
        cache.cur_conv_mask = None
        assert h_p is not None and h_next_seq is not None
        last_logits = model.compute_logits(h_p[:, -1, :])
        h_carry = h_next_seq[:, -1:]
        ws_cur = (ws_next_seq[0][:, -1:, :, :], ws_next_seq[1][:, -1:, :]) if ws_next_seq is not None else None
    else:
        assert h_carry is not None
        last_logits = None
        for t in range(max_p):
            v_t = valid_p[:, t : t + 1]
            cache.cur_conv_mask = v_t if cache.pad_lens is not None else None
            h_t, h_next, ws_next = model._forward_one_pass(
                p_ids[:, t : t + 1], h_carry, pos_p[:, t : t + 1],
                None, None, None, None, None, None,
                need_postlude=True, cache=cache, ws_state=ws_cur,
            )
            assert h_next is not None
            vm_t = v_t.unsqueeze(-1).to(dtype)
            h_carry = h_next * vm_t
            if ws_next is not None:
                ws_cur = (ws_next[0] * vm_t.unsqueeze(-2), ws_next[1] * vm_t)
            if t == max_p - 1:
                assert h_t is not None
                last_logits = model.compute_logits(h_t[:, 0, :])
        cache.cur_conv_mask = None
        assert last_logits is not None

    # 2. Batched Decode with active-slot compaction
    vocab = last_logits.shape[-1]
    use_pp = (presence_penalty != 0.0)
    seen_mask = torch.zeros(B, vocab, dtype=torch.bool, device=device)
    gen_tokens: list[list[int]] = [[] for _ in range(B)]
    finished = [False] * B
    active_orig = list(range(B))

    next_tok = sample_next_token(last_logits, seen_mask, temperature, top_k, top_p, presence_penalty)
    for r, orig_i in enumerate(active_orig):
        tid = int(next_tok[r].item())
        gen_tokens[orig_i].append(tid)
        if use_pp:
            seen_mask[r, tid] = True
        if tid in eos_ids:
            finished[orig_i] = True

    cur_pos = (max_p - pad_t).clone()
    for _ in range(1, max_new_tokens):
        keep_rows = [r for r, orig_i in enumerate(active_orig) if not finished[orig_i]]
        if not keep_rows:
            break
        cur_B = len(active_orig)
        if (cur_B - len(keep_rows)) >= max(1, cur_B // 4):
            keep_t = torch.tensor(keep_rows, dtype=torch.long, device=device)
            cache.compact(keep_t)
            next_tok = next_tok.index_select(0, keep_t)
            cur_pos = cur_pos.index_select(0, keep_t)
            seen_mask = seen_mask.index_select(0, keep_t)
            if h_carry is not None:
                h_carry = h_carry.index_select(0, keep_t)
            if ws_cur is not None:
                ws_cur = (ws_cur[0].index_select(0, keep_t), ws_cur[1].index_select(0, keep_t))
            active_orig = [active_orig[r] for r in keep_rows]
            cur_B = len(active_orig)

        h_t, h_next, ws_next = model._forward_one_pass(
            next_tok.view(cur_B, 1), h_carry, cur_pos, None, None, None, None, None, None,
            need_postlude=True, cache=cache, ws_state=ws_cur,
        )
        assert h_t is not None
        if use_carry and h_next is not None:
            h_carry = h_next
            ws_cur = ws_next
        cur_pos.add_(1)
        logits_t = model.compute_logits(h_t[:, 0, :])
        next_tok = sample_next_token(logits_t, seen_mask, temperature, top_k, top_p, presence_penalty)
        for r, orig_i in enumerate(active_orig):
            if not finished[orig_i]:
                tid = int(next_tok[r].item())
                gen_tokens[orig_i].append(tid)
                if use_pp:
                    seen_mask[r, tid] = True
                if tid in eos_ids:
                    finished[orig_i] = True

    del cache
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return gen_tokens, finished
