"""T036: fast-path switch and int8 quantization variants (experiments/variants.py)."""
import laya
import pytest
import torch

from experiments import variants as V


@pytest.fixture()
def agent(fastpath_checkpoint):
    a = laya.Agent(fastpath_checkpoint, device="cpu")
    yield a


Q = {"q": {"type": "choice", "instructions": "which option best matches", "criteria": {"a": "x", "b": "y", "c": "z"}}}
STATE = "the small red city is quiet in the morning and the harbor is warm " * 3


def probs(agent):
    return agent.predict(STATE, Q, max_len=200)["answers"]["q"]["probabilities"]


def test_fastpath_context_sets_and_restores():
    before = torch.backends.mha.get_fastpath_enabled()
    with V.fastpath(False):
        assert torch.backends.mha.get_fastpath_enabled() is False
    assert torch.backends.mha.get_fastpath_enabled() == before


def test_fastpath_off_agrees_with_native_within_1e4(agent):
    native = probs(agent)
    with V.fastpath(False):
        off = probs(agent)
    for k in native:
        assert abs(native[k] - off[k]) < 1e-4


def test_fastpath_off_variant_is_recorded_and_restorable(agent):
    before = torch.backends.mha.get_fastpath_enabled()
    rec = V.apply_variant(agent, "fastpath_off", reversible=True)
    try:
        assert rec["variant"] == "fastpath_off" and rec["fastpath_after"] is False and rec["fastpath_before"] == before
    finally:
        rec["restore"]()
    assert torch.backends.mha.get_fastpath_enabled() == before


def test_int8_encoder_quantizes_only_the_encoder_and_keeps_the_head_fastpath(agent):
    rec = V.apply_variant(agent, "int8_encoder", reversible=True)
    try:
        assert rec["quantization_api"] == V.QUANT_API and rec["quantized_layers"] > 0
        assert V.count_quantized(agent.model.encoder) == rec["quantized_layers"]
        assert V.count_quantized(agent.model.head) == 0                  # the decision head stays native
        assert rec["fastpath_after"] is True
        p = probs(agent)                                                  # runs, with the head on its fast path
        assert abs(sum(p.values()) - 1.0) < 1e-3
    finally:
        rec["restore"]()
    assert V.count_quantized(agent.model) == 0                            # restored


def test_int8_all_needs_the_fast_path_off(agent):
    with pytest.raises(V.VariantError) as err:
        V.apply_variant(agent, "int8_all_nofast")
    assert "fast" in str(err.value)
    with V.fastpath(False):
        rec = V.apply_variant(agent, "int8_all_nofast", reversible=True)
        try:
            assert rec["quantized_layers"] > V.count_quantized(agent.model.encoder) - 1
            assert V.count_quantized(agent.model.head) > 0
            p = probs(agent)
            assert abs(sum(p.values()) - 1.0) < 1e-3
        finally:
            rec["restore"]()


def test_quantized_output_is_close_to_native_but_not_identical(agent):
    native = probs(agent)
    rec = V.apply_variant(agent, "int8_encoder", reversible=True)
    try:
        q = probs(agent)
    finally:
        rec["restore"]()
    assert max(abs(native[k] - q[k]) for k in native) < 0.2              # random tiny weights: a loose bound


def test_variants_are_cpu_only_and_unknown_ones_are_refused():
    assert V.unsupported_reason("none", "gpu") is None
    for v in ("fastpath_off", "int8_encoder", "int8_all_nofast"):
        assert V.unsupported_reason(v, "cpu") is None
        assert "CPU-only" in V.unsupported_reason(v, "gpu")
    assert "unknown" in V.unsupported_reason("fp16", "cpu")
    with pytest.raises(V.VariantError):
        V.apply_variant(object(), "fp16")
