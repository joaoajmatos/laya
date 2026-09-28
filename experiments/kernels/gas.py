"""Gather-attend-scatter attention (T034, research.md R8).

For each query block, gather a fixed set of key/value blocks (`reference.gather_blocks`: block 0,
the neighbours, and evenly spaced blocks up to `spec.selection_blocks`), run dense attention on the
gathered keys only, and scatter each block's output back into the sequence. Index building, the
gather, the attention and the scatter are all inside the call and so inside the timing.
Work is ``n_blocks x block x selection_blocks*block``: linear in length for a fixed selection.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .reference import MaskSpec, gather_blocks, zero_empty_rows


def gas_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, spec: MaskSpec) -> torch.Tensor:
    if spec.pattern != "gather":
        raise ValueError("gas_attention runs the gather pattern")
    B, H, L, D = q.shape
    bs, s = spec.block, spec.selection_blocks
    nb = spec.n_blocks(L)
    Lp = nb * bs
    pad = Lp - L
    if pad:
        q = F.pad(q, (0, 0, 0, pad))
        k = F.pad(k, (0, 0, 0, pad))
        v = F.pad(v, (0, 0, 0, pad))
    idx, slot_ok = gather_blocks(nb, s)                              # [nb, s]
    kb = k.view(B, H, nb, bs, D)
    vb = v.view(B, H, nb, bs, D)
    # Gather, with blocks moved into the batch dimension ([B*nb, H, ., D]) for SDPA's fused path.
    kg = kb[:, :, idx].reshape(B, H, nb, s * bs, D).transpose(1, 2).reshape(B * nb, H, s * bs, D)
    vg = vb[:, :, idx].reshape(B, H, nb, s * bs, D).transpose(1, 2).reshape(B * nb, H, s * bs, D)
    key_pos = (idx[:, :, None] * bs + torch.arange(bs)[None, None, :]).reshape(nb, s * bs)
    slot = slot_ok[:, :, None].expand(nb, s, bs).reshape(nb, s * bs)
    lengths = spec.lengths_tensor()
    key_ok = slot[None] & (key_pos[None] < lengths[:, None, None])  # [B, nb, s*bs]
    mask = key_ok.reshape(B * nb, 1, 1, s * bs)
    qb = q.view(B, H, nb, bs, D).transpose(1, 2).reshape(B * nb, H, bs, D)
    out_blocks = F.scaled_dot_product_attention(qb, kg, vg, attn_mask=mask)
    out_blocks = zero_empty_rows(out_blocks, mask.expand(B * nb, 1, bs, s * bs).any(-1))
    out_blocks = out_blocks.view(B, nb, H, bs, D).transpose(1, 2)    # [B, H, nb, bs, D]
    out = torch.empty(B, H, nb, bs, D, dtype=out_blocks.dtype)
    out.index_copy_(2, torch.arange(nb), out_blocks)                 # scatter back by query block
    return out.reshape(B, H, Lp, D)[:, :, :L]
