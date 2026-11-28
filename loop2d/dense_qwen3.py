"""2D-AttnRes wrapper for dense pretrained Transformers (e.g. Qwen3-4B-Base).

Best-practice architecture: Unified 2D-AttnRes Pool + 1-Line Contractive Gate λ at 3L/4.
  - Zero extra KV cache (+0 B KV).
  - Zero extra projection matrices (only 189,956 added params on 4.02B Qwen3-4B = 0.0047%).
  - 100% standard AdamW optimization.
  - Exact 0.00e+00 bit-identity with the pretrained backbone at step 0.
  - Exact sub-readout K-pass Jacobi parallel training (passes 1..K-1 stop at 3L/4, skipping
    the top L/4 readout layers and LM head with zero approximation error).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from loop2d.pool import (
    attnres_2d_anchored_pool,
    docstarts_to_position_ids,
    init_multiscale_lambda_logits,
    shift_carry,
    update_contractive_carry,
)


class AttnRes2DDense(nn.Module):
    """Wrap a pretrained HuggingFace Qwen3/Llama-style CausalLM with 2D-AttnRes.

    Modes:
      - `"std"`: Standard unmodified dense backbone (0 added params, 1 pass).
      - `"attnres_2d"`: Unified 2D-AttnRes Pool + 1-Line Contractive Gate λ at `3L/4`.
    """

    VALID_MODES = ("std", "attnres_2d")

    def __init__(
        self,
        base_model: nn.Module,
        mode: str = "attnres_2d",
        l_out: int | None = None,
        grad_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        if mode not in self.VALID_MODES:
            raise ValueError(f"Unknown mode={mode!r}; expected one of {self.VALID_MODES}")
        self.base = base_model
        self.config = base_model.config
        self.mode = mode
        self.grad_checkpoint = grad_checkpoint

        self.n_layers = int(self.config.num_hidden_layers)
        self.dim = int(self.config.hidden_size)
        self.eps = float(self.config.rms_norm_eps)
        self.has_carry = (mode == "attnres_2d")

        # 6 vertical blocks across depth (block_size = L // 6) + 3L/4 sub-readout boundary
        self.block_size = max(1, self.n_layers // 6)
        self.max_v_blocks = 1 + (self.n_layers - 1) // self.block_size
        self.l_out = (int(l_out) if l_out is not None else (3 * self.n_layers) // 4) if self.has_carry else 0

        if self.has_carry:
            # 2L sublayers (Attn + MLP) + 1 final 2D-AttnRes readout before lm_head
            self.n_sub = 2 * self.n_layers + 1
            self.n_write_blocks = (self.l_out - 1) // self.block_size + 1

            # Zero-init queries & gates guarantee exact 0.00e+00 bit-identity at step 0
            self.q_2d = nn.Parameter(torch.zeros(self.n_sub, self.dim, dtype=torch.float32))
            self.g_v = nn.Parameter(torch.zeros(self.n_sub, self.max_v_blocks, dtype=torch.float32))
            self.g_h = nn.Parameter(torch.zeros(self.n_sub, dtype=torch.float32))

            # Learnable write-depth weights over the <=3L/4 vertical blocks
            self.w_write = nn.Parameter(torch.ones(self.n_write_blocks, dtype=torch.float32))

            # Multi-timescale contractive forgetting gate λ in (0, 1)^D, init [0.25, 0.95]
            self.theta_lambda = nn.Parameter(init_multiscale_lambda_logits(self.dim, 0.25, 0.95))
        else:
            self.n_sub = 0
            self.n_write_blocks = 0
            for pname in ("q_2d", "g_v", "g_h", "w_write", "theta_lambda"):
                self.register_parameter(pname, None)

    def count_added_params(self) -> int:
        if not self.has_carry:
            return 0
        return int(
            self.q_2d.numel()
            + self.g_v.numel()
            + self.g_h.numel()
            + self.w_write.numel()
            + self.theta_lambda.numel()
        )

    def get_param_groups(
        self,
        lr_base: float = 8e-6,
        lr_query: float = 3e-4,
        lr_scalar: float = 1e-2,
        weight_decay: float = 0.1,
    ) -> list[dict[str, Any]]:
        """Return standard AdamW parameter groups (no custom optimizer required)."""
        base_decay, base_nodecay = [], []
        for name, p in self.base.named_parameters():
            if not p.requires_grad:
                continue
            if p.ndim < 2 or "norm" in name or "embed" in name:
                base_nodecay.append(p)
            else:
                base_decay.append(p)

        groups: list[dict[str, Any]] = [
            {"params": base_decay, "lr": lr_base, "weight_decay": weight_decay, "group_name": "base_decay"},
            {"params": base_nodecay, "lr": lr_base, "weight_decay": 0.0, "group_name": "base_nodecay"},
        ]
        if self.has_carry:
            groups.append({"params": [self.q_2d], "lr": lr_query, "weight_decay": 0.0, "group_name": "attnres_q"})
            groups.append({
                "params": [self.g_v, self.g_h, self.w_write, self.theta_lambda],
                "lr": lr_scalar,
                "weight_decay": 0.0,
                "group_name": "attnres_scalars",
            })
        return groups

    def _prepare_pos_emb(
        self,
        h: torch.Tensor,
        position_ids: torch.LongTensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        if not hasattr(self.base.model, "rotary_emb"):
            return None
        bsz, seq_len, _ = h.shape
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=h.device, dtype=torch.long).unsqueeze(0).expand(bsz, -1)
        return self.base.model.rotary_emb(h, position_ids)

    def read_pool(
        self,
        sub_idx: int,
        anchor: torch.Tensor,
        v_blocks: torch.Tensor,
        h_carry: torch.Tensor | None,
    ) -> torch.Tensor:
        """Read from the unified 2D-AttnRes pool for sublayer `sub_idx`."""
        return attnres_2d_anchored_pool(
            q=self.q_2d[sub_idx],
            anchor=anchor,
            v_blocks=v_blocks,
            h_carry=h_carry,
            g_v=self.g_v[sub_idx],
            g_h=self.g_h[sub_idx],
            eps=self.eps,
        )

    def update_carry(
        self,
        h_prev: torch.Tensor,
        h_3l4: torch.Tensor,
        v_blocks: torch.Tensor,
        block_start: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the 1-line contractive carry update strictly at layer `l_out = 3L/4`."""
        slices = torch.cat([v_blocks[:, :, 1:], (h_3l4 - block_start).unsqueeze(-2)], dim=-2)
        write_signal = torch.einsum("s,btsd->btd", self.w_write.float(), slices.float())
        return update_contractive_carry(h_prev, write_signal, self.theta_lambda, eps=self.eps)

    def _run_layer(
        self,
        layer_idx: int,
        a: torch.Tensor,
        v_blocks: torch.Tensor,
        h_carry: torch.Tensor | None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        position_ids: torch.LongTensor | None,
        past_key_values: Any = None,
        use_cache: bool = False,
        **attn_kwargs: Any,
    ) -> torch.Tensor:
        layer = self.base.model.layers[layer_idx]
        m_att = 2 * layer_idx
        m_mlp = m_att + 1

        u1 = self.read_pool(m_att, a, v_blocks, h_carry)
        h1 = layer.input_layernorm(u1)
        sa_kwargs: dict[str, Any] = dict(
            hidden_states=h1,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **attn_kwargs,
        )
        if position_embeddings is not None:
            sa_kwargs["position_embeddings"] = position_embeddings
        attn_out = layer.self_attn(**sa_kwargs)
        y1 = attn_out[0] if isinstance(attn_out, tuple) else attn_out
        a = a + y1

        u2 = self.read_pool(m_mlp, a, v_blocks, h_carry)
        h2 = layer.post_attention_layernorm(u2)
        y2 = layer.mlp(h2)
        return a + y2

    def _run_pass(
        self,
        h_emb: torch.Tensor,
        h_carry: torch.Tensor | None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        position_ids: torch.LongTensor | None,
        attention_mask: torch.Tensor | None,
        stop_layer: int,
        use_ckpt: bool,
        **attn_kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run layers `0 .. stop_layer - 1`. Returns `(h_out, carry_next_unshifted)`."""
        a = h_emb
        if not self.has_carry:
            for l_idx in range(stop_layer):
                layer = self.base.model.layers[l_idx]
                if use_ckpt and torch.is_grad_enabled():
                    a = checkpoint(
                        lambda x, _l=layer: _l(
                            x,
                            attention_mask=attention_mask,
                            position_ids=position_ids,
                            position_embeddings=position_embeddings,
                            **attn_kwargs,
                        ),
                        a,
                        use_reentrant=False,
                    )
                else:
                    a = layer(
                        a,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        position_embeddings=position_embeddings,
                        **attn_kwargs,
                    )
            return a, None

        assert h_carry is not None
        v_blocks = h_emb.unsqueeze(-2)  # (B, T, 1, D) = [e_t]
        block_start = h_emb
        h_next: torch.Tensor | None = None

        for l_idx in range(stop_layer):
            if use_ckpt and torch.is_grad_enabled():
                a = checkpoint(
                    lambda _a, _vb, _hc, _li=l_idx: self._run_layer(
                        _li, _a, _vb, _hc, position_embeddings, attention_mask, position_ids, **attn_kwargs
                    ),
                    a, v_blocks, h_carry,
                    use_reentrant=False,
                )
            else:
                a = self._run_layer(
                    l_idx, a, v_blocks, h_carry, position_embeddings, attention_mask, position_ids, **attn_kwargs
                )
            if (l_idx + 1) == self.l_out:
                h_next = self.update_carry(h_carry, a, v_blocks, block_start)
            if ((l_idx + 1) % self.block_size == 0) and ((l_idx + 1) < self.n_layers):
                v_blocks = torch.cat([v_blocks, (a - block_start).unsqueeze(-2)], dim=-2)
                block_start = a

        if stop_layer == self.n_layers:
            a = self.read_pool(2 * self.n_layers, a, v_blocks, h_carry)
        return a, h_next

    def forward(
        self,
        input_ids: torch.LongTensor,
        passes: int = 4,
        backprop_last: int = 2,
        doc_start_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        return_hidden: bool = False,
        labels: torch.LongTensor | None = None,
        loss_fn: nn.Module | None = None,
        **attn_kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        return self.forward_multipass(
            input_ids=input_ids,
            passes=passes,
            backprop_last=backprop_last,
            doc_start_mask=doc_start_mask,
            position_ids=position_ids,
            attention_mask=attention_mask,
            return_hidden=return_hidden,
            labels=labels,
            loss_fn=loss_fn,
            **attn_kwargs,
        )

    def forward_multipass(
        self,
        input_ids: torch.LongTensor,
        passes: int = 4,
        backprop_last: int = 2,
        doc_start_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        return_hidden: bool = False,
        labels: torch.LongTensor | None = None,
        loss_fn: nn.Module | None = None,
        **attn_kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """Exact sub-readout K-pass Jacobi parallel forward.

        Passes `1 .. K-1` stop strictly at `l_out = 3L/4`, skipping the top `L/4` readout
        layers and the vocabulary LM head with zero approximation error.
        """
        if position_ids is None and doc_start_mask is not None:
            position_ids = docstarts_to_position_ids(doc_start_mask)
        h_emb = self.base.model.embed_tokens(input_ids)
        if h_emb.is_cuda and torch.is_autocast_enabled("cuda"):
            h_emb = h_emb.to(torch.get_autocast_dtype("cuda"))
        pos_emb = self._prepare_pos_emb(h_emb, position_ids)

        if not self.has_carry:
            a, _ = self._run_pass(
                h_emb, None, pos_emb, position_ids, attention_mask,
                stop_layer=self.n_layers, use_ckpt=self.grad_checkpoint, **attn_kwargs,
            )
            h_final = self.base.model.norm(a)
            out: dict[str, torch.Tensor] = {"hidden": h_final}
            if labels is not None and loss_fn is not None:
                h_shift = h_final[:, :-1, :].contiguous().view(-1, h_final.shape[-1])
                y_shift = labels[:, 1:].contiguous().view(-1)
                out["loss"] = loss_fn(self.base.lm_head.weight.to(h_shift.dtype), h_shift, y_shift)
            elif not return_hidden:
                out["logits"] = self.base.lm_head(h_final)
            return out

        k_total = max(1, int(passes))
        k_grad = max(1, min(k_total, int(backprop_last)))
        grad_start = k_total - k_grad

        h_carry = torch.zeros_like(h_emb)
        a_final = h_emb
        h_out_last = h_carry

        for p_idx in range(k_total):
            is_last = (p_idx == k_total - 1)
            need_grad = (p_idx >= grad_start) and torch.is_grad_enabled()
            stop_layer = self.n_layers if is_last else self.l_out

            if need_grad:
                a_p, h_out = self._run_pass(
                    h_emb, h_carry, pos_emb, position_ids, attention_mask,
                    stop_layer=stop_layer, use_ckpt=self.grad_checkpoint, **attn_kwargs,
                )
            else:
                with torch.no_grad():
                    a_p, h_out = self._run_pass(
                        h_emb.detach(), h_carry.detach(), pos_emb, position_ids, attention_mask,
                        stop_layer=stop_layer, use_ckpt=False, **attn_kwargs,
                    )
            assert h_out is not None
            h_out_last = h_out
            if not is_last:
                h_carry = shift_carry(h_out, doc_start_mask)
            else:
                a_final = a_p

        h_final = self.base.model.norm(a_final)
        out = {"hidden": h_final, "h_out": h_out_last}
        if labels is not None and loss_fn is not None:
            h_shift = h_final[:, :-1, :].contiguous().view(-1, h_final.shape[-1])
            y_shift = labels[:, 1:].contiguous().view(-1)
            out["loss"] = loss_fn(self.base.lm_head.weight.to(h_shift.dtype), h_shift, y_shift)
        elif not return_hidden:
            out["logits"] = self.base.lm_head(h_final)
        return out

    @torch.no_grad()
    def forward_exact_sequential(
        self,
        input_ids: torch.LongTensor,
        carry_cut: bool = False,
    ) -> torch.Tensor:
        """Exact token-by-token sequential recurrence using HuggingFace DynamicCache."""
        from transformers.cache_utils import DynamicCache

        bsz, seq_len = input_ids.shape
        device = input_ids.device
        dtype = self.base.model.embed_tokens.weight.dtype
        cache = DynamicCache()
        h_carry = torch.zeros(bsz, 1, self.dim, device=device, dtype=dtype)

        all_logits = []
        for t in range(seq_len):
            tok = input_ids[:, t : t + 1]
            pos = torch.full((bsz, 1), t, device=device, dtype=torch.long)
            h = self.base.model.embed_tokens(tok)
            pos_emb = self._prepare_pos_emb(h, pos)

            if not self.has_carry:
                a = h
                for l_idx in range(self.n_layers):
                    l_kw: dict[str, Any] = dict(
                        attention_mask=None, position_ids=pos, past_key_values=cache, use_cache=True,
                    )
                    if pos_emb is not None:
                        l_kw["position_embeddings"] = pos_emb
                    lo = self.base.model.layers[l_idx](a, **l_kw)
                    a = lo[0] if isinstance(lo, tuple) else lo
                all_logits.append(self.base.lm_head(self.base.model.norm(a)))
                continue

            cur_carry = torch.zeros_like(h_carry) if carry_cut else h_carry
            a = h
            v_blocks = h.unsqueeze(-2)
            block_start = h
            h_next = cur_carry
            for l_idx in range(self.n_layers):
                a = self._run_layer(
                    l_idx, a, v_blocks, cur_carry, pos_emb,
                    attention_mask=None, position_ids=pos, past_key_values=cache, use_cache=True,
                )
                if (l_idx + 1) == self.l_out:
                    h_next = self.update_carry(cur_carry, a, v_blocks, block_start)
                if ((l_idx + 1) % self.block_size == 0) and ((l_idx + 1) < self.n_layers):
                    v_blocks = torch.cat([v_blocks, (a - block_start).unsqueeze(-2)], dim=-2)
                    block_start = a
            a = self.read_pool(2 * self.n_layers, a, v_blocks, cur_carry)
            h_carry = h_next
            all_logits.append(self.base.lm_head(self.base.model.norm(a)))

        return torch.cat(all_logits, dim=1)

    def collect_telemetry(self) -> dict[str, Any]:
        """Collect lightweight mechanistic diagnostics (gates, write depth, λ distribution)."""
        if not self.has_carry:
            return {}
        gh = self.g_h.detach().float().cpu()
        gv = self.g_v.detach().float().cpu()
        lam = torch.sigmoid(self.theta_lambda.detach().float().cpu())
        return {
            "g_h_abs_mean": round(float(gh.abs().mean().item()), 4),
            "g_h_max": round(float(gh.abs().max().item()), 4),
            "g_h_final": round(float(gh[-1].item()), 4),
            "g_v_abs_mean": round(float(gv.abs().mean().item()), 4),
            "q_2d_rms": round(float(self.q_2d.detach().float().pow(2).mean().sqrt().item()), 5),
            "w_write": [round(float(x), 4) for x in self.w_write.detach().float().cpu().tolist()],
            "lambda_mean": round(float(lam.mean().item()), 4),
            "lambda_p10_p90": [round(float(lam.quantile(0.1).item()), 4), round(float(lam.quantile(0.9).item()), 4)],
        }

    def extra_state_dict(self) -> dict[str, torch.Tensor]:
        return {k: v.detach().cpu() for k, v in self.state_dict().items() if not k.startswith("base.")}

    def save_extra(self, ckpt_dir: Path, step: int = 0) -> None:
        torch.save(
            {
                "mode": self.mode,
                "l_out": self.l_out,
                "step": step,
                "attnres_2d_state": self.extra_state_dict(),
            },
            ckpt_dir / "attnres_2d_extra.pt",
        )

    def load_extra(self, ckpt_dir: Path, map_location: str | torch.device = "cpu") -> bool:
        p = ckpt_dir / "attnres_2d_extra.pt"
        if p.exists():
            blob = torch.load(p, map_location=map_location, weights_only=False)
            st = blob.get("attnres_2d_state", {})
            self.load_state_dict(st, strict=False)
            return True
        return False
