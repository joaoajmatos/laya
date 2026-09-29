"""research.md R17: the decision head's TransformerEncoderLayer fast path."""
import pytest
import torch

from experiments import abtest, cli, profile, results
from experiments.runner import load_agent
from experiments.timing import assert_clean

STATE = "the customer asked for a refund on a late delivery " * 20
Q = {"q": {"type": "choice", "instructions": "which option", "criteria": ["red", "blue", "green"]}}


def test_fast_path_eligibility(fastpath_agent, tiny_agent):
    assert profile.head_takes_fastpath(fastpath_agent) is True
    assert profile.head_takes_fastpath(tiny_agent) is False  # one head: odd, so never the fast path


def test_fixture_head_takes_the_fast_path(fastpath_agent):
    assert fastpath_agent.model.head.layers[0].self_attn.num_heads % 2 == 0
    with torch.profiler.profile() as prof, torch.no_grad():
        fastpath_agent.predict(STATE, Q)
    assert any(e.name == profile.FASTPATH_OP for e in prof.events())


def test_fast_path_off_gives_the_same_answers(fastpath_agent):
    on = fastpath_agent.predict(STATE, Q)["answers"]["q"]["probabilities"]
    prev = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    try:
        off = fastpath_agent.predict(STATE, Q)["answers"]["q"]["probabilities"]
    finally:
        torch.backends.mha.set_fastpath_enabled(prev)
    for k in on:
        assert off[k] == pytest.approx(on[k], abs=1e-3)


def test_profile_keeps_the_fast_path_and_attributes_it(fastpath_agent):
    r = profile.profile_in_process(fastpath_agent, {"total_tokens": 256, "repeats": 2, "warmup": 1})
    assert r["head_path"] == "fastpath" and r["mha_fastpath"] is True
    assert r["components"]["decision_head"]["ms"] > 0
    assert any(k.startswith("decision_head.") for k in r["subcomponents_ms"])
    by = r["score_value_by_layer_type"]
    assert by["head"]["n_layers"] == len(fastpath_agent.model.head.layers)
    assert_clean(fastpath_agent)


def test_profile_with_fast_path_off_labels_the_head(fastpath_agent):
    prev = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    try:
        r = profile.profile_in_process(fastpath_agent, {"total_tokens": 256, "repeats": 2, "warmup": 1})
    finally:
        torch.backends.mha.set_fastpath_enabled(prev)
    assert r["head_path"] == "modules" and r["mha_fastpath"] is False
    assert r["components"]["decision_head"]["ms"] > 0


def test_abtest_new_modes(fastpath_agent):
    r = abtest.abtest_in_process(fastpath_agent, {"total_tokens": 256, "repeats": 2,
                                                  "modes": ["clean", "fastpath_off", "labels_encoder", "labels"]})
    assert r["status"] == "measured" and set(r["ratio_to_clean"]) == {"clean", "fastpath_off", "labels_encoder", "labels"}
    assert torch.backends.mha.get_fastpath_enabled() is True  # restored
    assert_clean(fastpath_agent)


def test_load_agent_records_the_setting(fastpath_checkpoint):
    try:
        _, info = load_agent(fastpath_checkpoint, threads=1, mha_fastpath=False)
        assert info["mha_fastpath"] is False
    finally:
        torch.backends.mha.set_fastpath_enabled(True)


def test_one_setting_per_run(fastpath_checkpoint, tmp_path, monkeypatch):
    monkeypatch.setattr(results, "RESULTS_ROOT", tmp_path)
    try:
        assert cli.main(["manifest", "--run-id", "v", "--model", fastpath_checkpoint, "--threads", "1",
                         "--mha-fastpath", "off"]) == 0
        man = results.read_json(tmp_path / "v", "manifest.json")
        assert man["runtime"]["mha_fastpath"] is False and man["runtime"]["variant"] == "mha_fastpath_off"
        assert cli.main(["manifest", "--run-id", "v", "--model", fastpath_checkpoint, "--threads", "1",
                         "--mha-fastpath", "on"]) == cli.EXIT_TOOL_ERROR
        # without the flag the run's setting is inherited, so commands like `report` just work
        assert cli.main(["report", "--run-id", "v", "--model", fastpath_checkpoint, "--threads", "1"]) \
            == cli.EXIT_TOOL_ERROR  # no audit.json yet: a report error, not a setting error
        assert cli.main(["audit", "--run-id", "v", "--model", fastpath_checkpoint, "--threads", "1"]) == 0
        assert results.read_json(tmp_path / "v", "manifest.json")["runtime"]["mha_fastpath"] is False
        assert cli.main(["report", "--run-id", "v", "--model", fastpath_checkpoint, "--threads", "1"]) == 0
        text = (tmp_path / "v" / "report.md").read_text(encoding="utf-8")
        assert "variant mha_fastpath_off, not native Laya" in text
    finally:
        torch.backends.mha.set_fastpath_enabled(True)
