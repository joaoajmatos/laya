"""T022: profile runs (experiments/profile.py)."""
import pytest

from experiments import profile as P


QUESTIONS = {"q": {"type": "choice", "instructions": "which option", "criteria": ["red", "blue"]}}
STATE = "the customer asked for a refund on a late delivery"


def test_stage_wrappers_call_through(tiny_agent):
    before = tiny_agent.predict(STATE, QUESTIONS)
    with P.stage_wrappers(tiny_agent) as stages:
        during = tiny_agent.predict(STATE, QUESTIONS)
    after = tiny_agent.predict(STATE, QUESTIONS)
    assert before["answers"] == during["answers"] == after["answers"]
    assert {"_encode_state", "_forward", "_decode_answers", "collate_items"} <= set(stages)
    import laya.agent as agent_mod
    from laya.common import collate_items
    assert agent_mod.collate_items is collate_items
    assert not {"_encode_state", "_forward", "_decode_answers"} & set(vars(tiny_agent))


def test_labels_do_not_change_outputs(tiny_agent):
    before = tiny_agent.predict(STATE, QUESTIONS)
    with P.label_modules(tiny_agent):
        during = tiny_agent.predict(STATE, QUESTIONS)
    assert before["answers"] == during["answers"]
    assert sum(len(m._forward_hooks) + len(m._forward_pre_hooks) for m in tiny_agent.model.modules()) == 0


@pytest.fixture(scope="module")
def profiled(tiny_agent):
    return [P.profile_in_process(tiny_agent, {"total_tokens": L, "repeats": 2, "warmup": 1})
            for L in (128, 256, 512, 1024)]


def test_shares_sum_to_one(profiled):
    for it in profiled:
        assert it["status"] == "measured" and it["profile_run"] is True and it["kind"] == "profile"
        assert set(it["components"]) == set(P.COMPONENTS)
        assert sum(c["share"] for c in it["components"].values()) == pytest.approx(1.0, abs=1e-6)


def test_explained_fraction_definition(profiled):
    for it in profiled:
        total = sum(c["ms"] for c in it["components"].values())
        named = sum(it["components"][k]["ms"] for k in P.NAMED)
        assert it["explained_fraction"] == pytest.approx(named / total, abs=1e-9)
        assert it["explained_ok"] == (it["explained_fraction"] >= 0.90)
        assert it["explained_ok"] or "below the 90% target" in it["explained_note"]


def test_attention_is_split_by_layer_type(profiled):
    by = profiled[-1]["score_value_by_layer_type"]
    assert by["local"]["n_layers"] == 1 and by["global"]["n_layers"] == 2 and by["head"]["n_layers"] == 1
    assert any(k.endswith("mask_construction") for k in profiled[-1]["subcomponents_ms"])


def test_scaling_block(profiled):
    s = P.scaling(profiled)
    assert s["lengths"] == [128, 256, 512, 1024]
    assert set(s["components"]) == set(P.COMPONENTS)
    assert s["components"]["attention_score_value"] is not None
    assert P.scaling(profiled[:2])["components"]["attention_score_value"] is None  # < 3 lengths


def test_fit_exponent():
    assert P.fit_exponent([1, 2, 4, 8], [3, 12, 48, 192]) == pytest.approx(2.0)
    assert P.fit_exponent([1, 2], [1, 2]) is None


@pytest.mark.parametrize("local,glob,expected", [
    (1.05, 1.95, "verified"), (1.9, 1.95, "dense_masked"), (1.6, 1.95, "not_yet_determined"),
    (1.0, 1.1, "not_yet_determined")])
def test_executed_work_verdict(local, glob, expected):
    s = {"lengths": [512, 1024, 2048, 4096], "score_value_per_layer": {"local": local, "global": glob}}
    assert P.executed_work_verdict(s, window=128)["verdict"] == expected


def test_verdict_needs_lengths_beyond_window():
    s = {"lengths": [64, 128, 256], "score_value_per_layer": {"local": 1.0, "global": 2.0}}
    assert P.executed_work_verdict(s, window=128)["verdict"] == "not_yet_determined"


def test_profile_command_merges_clean_and_updates_audit(tiny_checkpoint, tmp_path, monkeypatch):
    from experiments import cli, results
    monkeypatch.setattr(results, "RESULTS_ROOT", tmp_path)
    assert cli.main(["audit", "--run-id", "p", "--model", tiny_checkpoint, "--threads", "1"]) == 0
    assert cli.main(["sweep", "--run-id", "p", "--model", tiny_checkpoint, "--threads", "1",
                     "--lengths", "128,256,512", "--questions", "1", "--batch-sizes", "1",
                     "--repeats", "3", "--warmup", "1", "--canary-length", "0"]) == 0
    assert cli.main(["profile", "--run-id", "p", "--model", tiny_checkpoint, "--threads", "1",
                     "--lengths", "128,256,512", "--repeats", "2", "--overhead-length", "128",
                     "--canary-length", "128", "--canary-repeats", "2"]) == 0
    prof = results.read_json(tmp_path / "p", "profile.json")
    assert prof["canary"]["enabled"] and len(prof["canary"]["runs"]) == 4  # 3 lengths + end
    assert all("drift" in it for it in prof["items"])
    for it in prof["items"]:
        assert it["total_clean_ms"] > 0 and it["profiled_to_clean_ratio"] > 0
        assert it["derived_from"][0].startswith("sweep.json#")
    assert prof["overhead_check"]["overhead_ratio"] > 0
    audit = results.read_json(tmp_path / "p", "audit.json")
    assert audit["executed_work_note"]["verdict"] in ("verified", "dense_masked", "not_yet_determined")
    assert audit["executed_work_note"]["derived_from"] == ["profile.json#scaling"]


def test_abtest_interleaves_modes_and_reports_ratios(tiny_agent):
    from experiments import abtest
    n = len(abtest.MODES)
    r = abtest.abtest_in_process(tiny_agent, {"total_tokens": 256, "repeats": n})
    assert r["status"] == "measured" and r["kind"] == "abtest"
    assert set(r["timings_ms"]) == set(abtest.MODES)
    assert r["ratio_to_clean"]["clean"] == 1.0
    assert all(len(v) == n for v in r["samples_ms"].values())
    assert [o[0] for o in r["order"]] == list(abtest.MODES)  # every mode leads once
    assert all(t >= 1 for ts in r["threads_after_call"].values() for t in ts)
    from experiments.timing import assert_clean
    assert_clean(tiny_agent)  # nothing left attached


def test_abtest_command(tiny_checkpoint, tmp_path, monkeypatch):
    from experiments import cli, results
    monkeypatch.setattr(results, "RESULTS_ROOT", tmp_path)
    assert cli.main(["abtest", "--run-id", "ab", "--model", tiny_checkpoint, "--threads", "1",
                     "--lengths", "128", "--repeats", "2", "--modes", "clean,all"]) == 0
    body = results.read_json(tmp_path / "ab", "abtest.json")
    assert body["items"][0]["status"] == "measured" and set(body["items"][0]["ratio_to_clean"]) == {"clean", "all"}
    assert cli.main(["abtest", "--run-id", "ab", "--model", tiny_checkpoint, "--threads", "1",
                     "--modes", "all"]) == cli.EXIT_TOOL_ERROR  # clean is required
