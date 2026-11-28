"""2D-AttnRes: Unified 2D Attention Residual Pool + 1-Line Contractive Gate λ.

Unifies vertical depth routing and zero-KV (+0 B) horizontal cross-token recurrence
in one shared RMSNorm-key softmax residual pool at the 3L/4 sub-readout boundary.
"""
from __future__ import annotations

from loop2d.dense_qwen3 import AttnRes2DDense
from loop2d.looped_moe import (
    AttnRes2DLoopedMoE,
    Inkling2Cache,
    Inkling2Config,
    generate_batch_looped_moe,
)
from loop2d.pool import (
    attnres_2d_anchored_pool,
    attnres_2d_pool,
    docstarts_to_position_ids,
    rmsnorm_unweighted,
    sample_next_token,
    shift_carry,
    shift_state_pair,
    update_contractive_carry,
)

__all__ = [
    "AttnRes2DDense",
    "AttnRes2DLoopedMoE",
    "Inkling2Cache",
    "Inkling2Config",
    "attnres_2d_anchored_pool",
    "attnres_2d_pool",
    "docstarts_to_position_ids",
    "generate_batch_looped_moe",
    "rmsnorm_unweighted",
    "sample_next_token",
    "shift_carry",
    "shift_state_pair",
    "update_contractive_carry",
]
