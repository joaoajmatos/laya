"""Dense attention with `scaled_dot_product_attention`, as the native path runs it (T032).

`dense_attention` is the baseline. `dense_masked_attention` applies a sparse pattern as a mask over
full-length attention, which is what the native local layers do (audit verdict ``dense_masked``):
same results as the sparse kernel, same work as dense. The benchmark uses it to show that a mask
alone is not a saving (FR-023).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .reference import MaskSpec, zero_empty_rows


def _sdpa_with(spec_mask: torch.Tensor, q, k, v):
    has_key = spec_mask.any(-1)
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=spec_mask)
    return zero_empty_rows(out, has_key)


def dense_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, spec: MaskSpec) -> torch.Tensor:
    """Full attention with key padding. The mask is built inside the call (timed)."""
    if spec.pattern != "full":
        raise ValueError("dense_attention runs the full pattern; use dense_masked_attention for others")
    L = q.shape[-2]
    key_ok = torch.arange(L)[None, :] < spec.lengths_tensor()[:, None]      # [B, L]
    mask = key_ok[:, None, None, :].expand(-1, 1, L, L)
    return _sdpa_with(mask, q, k, v)


def dense_masked_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, spec: MaskSpec) -> torch.Tensor:
    """Any pattern as an L x L mask over full attention: same work as dense."""
    return _sdpa_with(spec.allowed(q.shape[-2]), q, k, v)
