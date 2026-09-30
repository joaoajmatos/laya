"""Optimized-system variants: the decision head's fast-path switch and CPU int8 quantization (T042).

Specs: specs/002-decision-benchmark-baselines (research.md R14; FR-017). These are reported apart
from the architectural baselines and applied outside ``laya/``:

* ``fastpath_off``   PyTorch's `TransformerEncoderLayer` inference fast path disabled (Phase 1 R17).
* ``int8_encoder``   dynamic int8 quantization of the encoder's `Linear` layers only; the decision
                     head and its fast path stay native.
* ``int8_all_nofast`` dynamic int8 quantization of every `Linear` layer, with the fast path off.
                     With the fast path on, the head's fast-path check reads ``weight`` from a
                     quantized layer, which is a method, and fails (``'function' object has no
                     attribute 'device'``, 2026-09-29 spike), so that combination is refused.

Quantization is CPU-only. On another device the variant is ``unsupported`` with that reason.
"""
from __future__ import annotations

import contextlib
import copy
import warnings
from typing import Any, Dict, Optional

VARIANTS = ("none", "fastpath_off", "int8_encoder", "int8_all_nofast")
QUANT_API = "torch.ao.quantization.quantize_dynamic(qint8)"


class VariantError(RuntimeError):
    """A variant cannot be applied as asked."""


def unsupported_reason(variant: str, device: str) -> Optional[str]:
    """Why `variant` cannot run on `device`, or None when it can."""
    if variant not in VARIANTS:
        return "unknown variant %r" % variant
    if variant == "none":
        return None
    if device != "cpu":
        what = "int8 dynamic quantization" if variant.startswith("int8") else "the mha fast-path switch"
        return "%s is a CPU-only variant; not run on %s" % (what, device)
    return None


def _quantize(module) -> None:
    import torch
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")        # PyTorch's notice that quantize_dynamic is moving to torchao
        torch.ao.quantization.quantize_dynamic(module, {torch.nn.Linear}, dtype=torch.qint8, inplace=True)


def count_quantized(module) -> int:
    """Number of dynamically quantized Linear layers in `module`."""
    import torch.ao.nn.quantized.dynamic as qdyn
    return sum(1 for m in module.modules() if isinstance(m, qdyn.Linear))


def apply_variant(agent, variant: str, reversible: bool = False) -> Dict[str, Any]:
    """Apply `variant` to the loaded agent; returns a record (and ``restore`` when `reversible`).

    Non-reversible application (the default) is for per-condition subprocesses. `reversible` keeps a copy
    of the model so tests can undo it.
    """
    import torch
    if variant not in VARIANTS:
        raise VariantError("unknown variant %r; choose from %s" % (variant, VARIANTS))
    prev_fast = bool(torch.backends.mha.get_fastpath_enabled())
    record: Dict[str, Any] = {"variant": variant, "quantization_api": None, "quantized_layers": 0,
                              "fastpath_before": prev_fast}
    saved = copy.deepcopy(agent.model) if reversible and variant.startswith("int8") else None

    def restore() -> None:
        torch.backends.mha.set_fastpath_enabled(prev_fast)
        if saved is not None:
            agent.model = saved

    if variant == "none":
        pass
    elif variant == "fastpath_off":
        torch.backends.mha.set_fastpath_enabled(False)
    elif variant == "int8_encoder":
        _quantize(agent.model.encoder)
        record.update(quantization_api=QUANT_API, quantized_layers=count_quantized(agent.model.encoder))
    else:   # int8_all_nofast
        if prev_fast:
            raise VariantError("int8_all_nofast needs the mha fast path off: quantized weights break the "
                               "TransformerEncoderLayer fast-path check (research.md R14); "
                               "run with --mha-fastpath off or use int8_encoder")
        _quantize(agent.model)
        record.update(quantization_api=QUANT_API, quantized_layers=count_quantized(agent.model))
    record["fastpath_after"] = bool(torch.backends.mha.get_fastpath_enabled())
    if reversible:
        record["restore"] = restore
    return record


@contextlib.contextmanager
def fastpath(enabled: bool):
    """Set PyTorch's mha fast path for a block and restore it after."""
    import torch
    prev = bool(torch.backends.mha.get_fastpath_enabled())
    torch.backends.mha.set_fastpath_enabled(bool(enabled))
    try:
        yield
    finally:
        torch.backends.mha.set_fastpath_enabled(prev)


def load_for_variant(model: str, revision: Optional[str], threads: Optional[int], variant: str):
    """Load the agent on CPU with `variant` applied; returns ``(agent, load_info, variant_record)``.

    ``fastpath_off`` and ``int8_all_nofast`` load with the fast path off (a labelled runtime setting, as in
    Phase 1). ``int8_encoder`` keeps the native fast path.
    """
    from .runner import load_agent
    agent, info = load_agent(model, revision, threads, mha_fastpath=variant not in ("fastpath_off", "int8_all_nofast"))
    rec = apply_variant(agent, variant if variant != "fastpath_off" else "none")
    rec["variant"] = variant
    rec["fastpath_after"] = info["mha_fastpath"]
    return agent, info, rec
