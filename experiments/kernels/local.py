"""Block-local attention that skips out-of-window work (T033, research.md R8).

Queries are grouped into blocks of `spec.block`; each block attends to its own and the two
neighbouring key blocks. Keys are laid out with `unfold` as ``[batch, heads, n_blocks, 3*block, d]``
so the score matmul is ``n_blocks x block x 3*block`` instead of ``L x L`` (blocks are moved into the
batch dimension so SDPA uses its fused 4-D kernel): work grows linearly
with length. A short last block and lengths that do not divide by the block size are handled by
padding to whole blocks and masking the padded keys. Mask building, padding and copies are inside
the call and so inside the timing (FR-021).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .reference import MaskSpec, zero_empty_rows


def local_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, spec: MaskSpec) -> torch.Tensor:
    if spec.pattern != "block_local":
        raise ValueError("local_attention runs the block_local pattern")
    B, H, L, D = q.shape
    bs = spec.block
    nb = spec.n_blocks(L)
    Lp = nb * bs
    pad = Lp - L
    if pad:
        q = F.pad(q, (0, 0, 0, pad))
        k = F.pad(k, (0, 0, 0, pad))
        v = F.pad(v, (0, 0, 0, pad))
    # One extra block on each side, then a sliding window of 3 blocks with stride 1 block.
    kp = F.pad(k, (0, 0, bs, bs))
    vp = F.pad(v, (0, 0, bs, bs))
    # Blocks go into the batch dimension ([B*nb, H, ., D]) so SDPA takes its fused 4-D path;
    # a 5-D call falls back to the slow reference path (10-20x slower in testing).
    kn = kp.unfold(2, 3 * bs, bs).permute(0, 2, 1, 4, 3).reshape(B * nb, H, 3 * bs, D)
    vn = vp.unfold(2, 3 * bs, bs).permute(0, 2, 1, 4, 3).reshape(B * nb, H, 3 * bs, D)
    qb = q.view(B, H, nb, bs, D).permute(0, 2, 1, 3, 4).reshape(B * nb, H, bs, D)
    # Validity of each neighbour key: inside the sequence and inside the row's length.
    key_pos = (torch.arange(nb)[:, None] - 1) * bs + torch.arange(3 * bs)[None, :]   # [nb, 3bs]
    lengths = spec.lengths_tensor()
    key_ok = (key_pos[None] >= 0) & (key_pos[None] < lengths[:, None, None])         # [B, nb, 3bs]
    mask = key_ok.reshape(B * nb, 1, 1, 3 * bs)
    out = F.scaled_dot_product_attention(qb, kn, vn, attn_mask=mask)
    out = zero_empty_rows(out, mask.expand(B * nb, 1, bs, 3 * bs).any(-1))
    out = out.view(B, nb, H, bs, D).permute(0, 2, 1, 3, 4)
    return out.reshape(B, H, Lp, D)[:, :, :L]
