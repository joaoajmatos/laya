"""T012: context audit (experiments/audit.py)."""
import pytest

from experiments import audit as A
from experiments.results import validate_item
from .conftest import FIXTURE_AGENT_CFG, FIXTURE_ENCODER


def test_layer_schedule_matches_the_fixture(tiny_agent):
    a = A.build_audit(tiny_agent)
    n, every = FIXTURE_ENCODER["num_hidden_layers"], FIXTURE_ENCODER["global_attn_every_n_layers"]
    assert [l["index"] for l in a["layers"]] == list(range(n))
    assert [l["attention_type"] for l in a["layers"]] == [
        "global" if i % every == 0 else "local" for i in range(n)]
    for l in a["layers"]:
        if l["attention_type"] == "local":
            assert l["window"] == FIXTURE_ENCODER["local_attention"]
        else:
            assert "window" not in l
        assert l["rope_theta"] and l["rope_theta"] > 0
    e = a["encoder"]
    assert e["hidden_size"] == 64 and e["num_heads"] == 2 and e["head_dim"] == 32
    assert e["num_layers"] == 3 and e["attention_implementation"] == "sdpa"


def test_positional_limits_are_separate_fields(tiny_agent):
    p = A.build_audit(tiny_agent)["positional"]
    assert p["max_position_embeddings"] == FIXTURE_ENCODER["max_position_embeddings"]
    assert p["configured_max_len"] == FIXTURE_AGENT_CFG["max_len"]
    assert p["max_position_embeddings"] != p["configured_max_len"]
    assert p["effective_supported_max"] == p["max_position_embeddings"]


def test_decision_head_matches_config(tiny_agent):
    h = A.build_audit(tiny_agent)["decision_head"]
    assert h["layers"] == FIXTURE_AGENT_CFG["head_layers"] == h["configured_head_layers"]
    assert h["hidden"] == 64 and h["heads"] == 1
    assert "full sequence" in h["note"]


def test_lengths_above_capacity_are_unsupported(tiny_agent):
    cap = FIXTURE_ENCODER["max_position_embeddings"]
    a = A.build_audit(tiny_agent, lengths=[128, 512, cap, cap + 1, 16384])
    by = {l["length"]: l for l in a["lengths"]}
    assert by[128]["supported"] and not by[128]["beyond_configured_max_len"]
    assert by[512]["supported"] and by[512]["beyond_configured_max_len"]  # above max_len=256
    assert by[cap]["supported"]
    for L in (cap + 1, 16384):
        assert by[L]["supported"] is False and by[L]["status"] == "unsupported"
        assert "positional capacity" in by[L]["reason"]
        validate_item(by[L])
    assert [l["length"] for l in a["lengths"]] == [128, 512, cap, cap + 1, 16384]  # none dropped


def test_executed_work_starts_undetermined(tiny_agent):
    note = A.build_audit(tiny_agent)["executed_work_note"]
    assert note["verdict"] == "not_yet_determined" and note["verdict"] in A.EXECUTED_WORK_VERDICTS


def test_implicit_fallback_appears(tiny_agent, monkeypatch):
    assert A.build_audit(tiny_agent)["fallbacks"] == []
    monkeypatch.setattr(tiny_agent, "cpu_fallback_count", 2, raising=False)
    monkeypatch.setattr(tiny_agent, "last_fallback_reason", "CUDA out of memory", raising=False)
    f = A.build_audit(tiny_agent)["fallbacks"]
    assert f and f[0]["event"] == "cpu_fallback" and f[0]["count"] == 2
    assert "out of memory" in f[0]["reason"]


def test_research_plan_differences_are_reported_not_failed(tiny_agent):
    a = A.build_audit(tiny_agent)
    assert "max_position_embeddings" in a["discrepancies"]  # fixture has 2048, plan says 8192
    c = a["research_plan_comparison"]["max_position_embeddings"]
    assert c == {"research_plan": 8192, "observed": 2048, "matches": False}


def test_schedule_printout(tiny_agent):
    text = A.format_schedule(A.build_audit(tiny_agent, lengths=[100, 99999]))
    assert "layer  1  local" in text and "UNSUPPORTED" in text
