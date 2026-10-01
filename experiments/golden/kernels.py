"""`golden kernels`: random q/k/v with the reference output and the repo's PyTorch kernels (laya:004 US3).

The "native-layer reference" is `experiments/kernels/reference.py:reference_attention`: `laya/layers/attention.py`
is empty, so that file is the repo's definition of the semantics (see the spec's clarifications).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from . import common

LENGTHS = (256, 1000, 2048)
HEADS, HEAD_DIM, BLOCK, SELECTION_BLOCKS = 16, 64, 128, 4
TOLERANCE = 1e-5
#: implementation name -> MaskSpec pattern
IMPLS = {"block_local": "block_local", "gas": "gather"}


def kernel_golden(impl: str, length: int, seed: int) -> Dict[str, Any]:
    import torch
    from ..kernels import gas, local
    from ..kernels.reference import MaskSpec, reference_attention
    fn = {"block_local": local.local_attention, "gas": gas.gas_attention}[impl]
    gen = torch.Generator().manual_seed(seed)
    q, k, v = (torch.randn(1, HEADS, length, HEAD_DIM, generator=gen) for _ in range(3))
    spec = MaskSpec(lengths=[length], pattern=IMPLS[impl], block=BLOCK, selection_blocks=SELECTION_BLOCKS)
    with torch.no_grad():
        ref = reference_attention(q, k, v, spec)
        out = fn(q, k, v, spec)
    diff = float((ref - out).abs().max())
    if not diff <= TOLERANCE:
        raise RuntimeError("%s at length %d: max abs diff %.3g > %g" % (impl, length, diff, TOLERANCE))
    return {"q": q, "k": k, "v": v, "reference": ref, "kernel": out, "max_abs_diff": diff, "spec": spec}


def export_kernels(out: Path, lengths=LENGTHS) -> Dict[str, Any]:
    import torch
    out = Path(out)
    common.make_deterministic()
    records = {}
    for impl in IMPLS:
        for length in lengths:
            seed = common.SEED + length
            g = kernel_golden(impl, length, seed)
            name = "%s_L%d" % (impl, length)
            common.save_tensors(out / "kernels" / name / "tensors.safetensors",
                                {"q": g["q"], "k": g["k"], "v": g["v"], "reference": g["reference"],
                                 "kernel": g["kernel"], "max_abs_diff": torch.tensor(g["max_abs_diff"])})
            spec = g["spec"]
            meta = common.base_meta(
                "kernel-golden", case=name, impl=impl, length=length, heads=HEADS, head_dim=HEAD_DIM,
                pattern=spec.pattern, block=spec.block, selection_blocks=spec.selection_blocks,
                lengths=list(spec.lengths), input_seed=seed, scale="1/sqrt(head_dim)",
                reference="experiments/kernels/reference.py:reference_attention",
                kernel="experiments/kernels/%s.py" % ("local" if impl == "block_local" else "gas"),
                max_abs_diff=g["max_abs_diff"], tolerance=TOLERANCE,
                empty_row_convention="a query with no allowed key returns zeros",
                note="block_local is block-granular (query block i sees key blocks i-1, i, i+1), not +/-64 keys")
            common.write_json(out / "kernels" / name / "meta.json", meta)
            records[name] = meta
    return records
