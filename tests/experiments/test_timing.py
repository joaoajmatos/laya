"""T021: clean latency measurement (experiments/timing.py) and the sweep command."""
import pytest

from experiments import cli, profile, results, timing
from experiments.results import validate_item
from experiments.runner import run_condition


class Spy:
    def __init__(self, agent, monkeypatch):
        self.calls = {"predict": 0, "predict_batch": 0}
        for name in self.calls:
            real = getattr(type(agent), name)

            def make(real=real, name=name):
                def spy(*a, **k):
                    self.calls[name] += 1
                    return real(*a, **k)
                return spy
            monkeypatch.setattr(type(agent), name, make())


def test_single_request_measurement(tiny_agent):
    r = timing.measure_in_process(tiny_agent, {"total_tokens": 200, "repeats": 5, "warmup": 2},
                                  {"load_seconds": 1.5, "threads": 1})
    validate_item(r)
    assert r["status"] == "measured" and r["kind"] == "single" and r["call"] == "predict"
    assert r["repeats"] == {"requested": 5, "completed": 5, "warmup": 2}
    assert len(r["samples_ms"]) == 5
    assert set(r["timings_ms"]) >= {"min", "p50", "p95", "mean", "std", "n"}
    assert r["timings_ms"]["low_sample_p95"] is True
    assert r["first_vs_median"]["ratio"] > 0
    assert r["load_seconds"] == 1.5  # separate from timings
    assert r["token_accounting"]["rows"][0]["final_length"] == 200
    assert r["condition"]["options_per_question"] == 2
    assert r["clean_path"]["profiler_active"] is False


def test_warmup_is_excluded(tiny_agent, monkeypatch):
    spy = Spy(tiny_agent, monkeypatch)
    r = timing.measure_in_process(tiny_agent, {"total_tokens": 150, "repeats": 4, "warmup": 3})
    assert spy.calls["predict"] == 7 and len(r["samples_ms"]) == 4


def test_clean_path_refuses_wrappers_hooks_and_profiler(tiny_agent):
    import torch
    timing.assert_clean(tiny_agent)
    with profile.stage_wrappers(tiny_agent):
        with pytest.raises(timing.CleanPathError, match="wrapped"):
            timing.measure_in_process(tiny_agent, {"total_tokens": 150, "repeats": 1, "warmup": 0})
    with profile.label_modules(tiny_agent):
        with pytest.raises(timing.CleanPathError, match="module forward hooks"):
            timing.assert_clean(tiny_agent)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]):
        with pytest.raises(timing.CleanPathError, match="profiler"):
            timing.assert_clean(tiny_agent)
    timing.assert_clean(tiny_agent)  # everything was removed again
    assert not {"_encode_state", "_forward", "_decode_answers"} & set(vars(tiny_agent))


def test_above_capacity_is_unsupported_without_model_call(tiny_agent, monkeypatch):
    spy = Spy(tiny_agent, monkeypatch)
    r = timing.measure_in_process(tiny_agent, {"total_tokens": 4096, "repeats": 3})
    validate_item(r)
    assert r["status"] == "unsupported" and "positional capacity" in r["reason"]
    assert spy.calls == {"predict": 0, "predict_batch": 0}


def test_kinds_and_calls(tiny_agent, monkeypatch):
    spy = Spy(tiny_agent, monkeypatch)
    multi = timing.measure_in_process(tiny_agent, {"total_tokens": 150, "questions": 3, "repeats": 2, "warmup": 0})
    assert multi["kind"] == "multi_question" and multi["call"] == "predict"
    # predict() delegates to predict_batch() inside Laya, so predict_batch also counts those
    assert spy.calls == {"predict": 2, "predict_batch": 2}
    batch = timing.measure_in_process(tiny_agent, {"total_tokens": 150, "batch_size": 4, "repeats": 2, "warmup": 0})
    assert batch["kind"] == "batch" and batch["call"] == "predict_batch"
    assert spy.calls == {"predict": 2, "predict_batch": 4}  # no predict() call for the batch kind
    assert len(batch["token_accounting"]["rows"]) == 4
    single = timing.measure_in_process(tiny_agent, {"total_tokens": 150, "options": 5, "repeats": 1, "warmup": 0})
    assert len({single["kind"], multi["kind"], batch["kind"]}) == 3
    assert single["condition"]["options_per_question"] == 5


def test_beyond_configured_max_len_flag(tiny_agent):
    cap = tiny_agent.cfg["max_len"]
    over = timing.measure_in_process(tiny_agent, {"total_tokens": cap + 44, "repeats": 1, "warmup": 0})
    under = timing.measure_in_process(tiny_agent, {"total_tokens": cap, "repeats": 1, "warmup": 0})
    assert over["beyond_configured_max_len"] is True and under["beyond_configured_max_len"] is False


def test_time_cap_gives_partial(tiny_agent):
    r = timing.measure_in_process(tiny_agent, {"total_tokens": 150, "repeats": 50, "warmup": 1, "time_cap": 0.0})
    validate_item(r)
    assert r["status"] == "partial" and "time cap" in r["reason"]


def test_default_repeats():
    assert [timing.default_repeats(L) for L in (128, 512, 1024, 2048, 4096, 8192)] == [30, 30, 20, 20, 10, 10]


def test_measure_condition_in_subprocess(tiny_checkpoint):
    item = run_condition("experiments.timing:measure_condition",
                         {"model": tiny_checkpoint, "threads": 1, "total_tokens": 128, "repeats": 3, "warmup": 1},
                         time_cap=300)
    validate_item(item)
    assert item["status"] == "measured"
    assert item["peak_rss_bytes"] > 0 and item["load_seconds"] > 0
    assert item["model"]["revision"] == "local" and item["fixture_model"] is True


def test_sweep_conditions_axes_and_full():
    axes = cli.sweep_conditions([128, 512], [1, 2, 5], [1, 4])
    assert axes == [(128, 1, 1), (128, 2, 1), (128, 5, 1), (128, 1, 4),
                    (512, 1, 1), (512, 2, 1), (512, 5, 1), (512, 1, 4)]
    assert len(cli.sweep_conditions([128], [1, 2], [1, 4], "full")) == 4


def test_sweep_command_records_everything(tiny_checkpoint, tmp_path, monkeypatch):
    monkeypatch.setattr(results, "RESULTS_ROOT", tmp_path)
    run = results.run_dir("sw", root=tmp_path)
    results.write_json(run, "audit.json", {"positional": {"max_position_embeddings": 2048}})
    code = cli.main(["sweep", "--run-id", "sw", "--model", tiny_checkpoint, "--threads", "1",
                     "--lengths", "128,4096", "--questions", "1,2", "--batch-sizes", "1",
                     "--repeats", "2", "--warmup", "0"])
    assert code == 0
    items = results.read_json(run, "sweep.json")["items"]
    got = {(i["condition"]["total_tokens"], i["condition"]["questions"]): i["status"] for i in items}
    assert got == {(128, 1): "measured", (128, 2): "measured", (4096, 1): "unsupported", (4096, 2): "unsupported"}
    for i in items:
        validate_item(i)
    single = [i for i in items if i.get("kind") == "single" and i["status"] == "measured"]
    assert len(single) == 1  # multi-question never lands in the single-request curve


def test_sweep_canary_block_and_drift_annotation(tiny_checkpoint, tmp_path, monkeypatch, in_process_children):
    # Canary bookkeeping is orchestration; real child processes are exercised by the sweep tests around it.
    monkeypatch.setattr(results, "RESULTS_ROOT", tmp_path)
    code = cli.main(["sweep", "--run-id", "c", "--model", tiny_checkpoint, "--threads", "1",
                     "--lengths", "128,256", "--questions", "1", "--batch-sizes", "1",
                     "--repeats", "2", "--warmup", "0", "--canary-length", "128", "--canary-repeats", "2"])
    assert code == 0
    body = results.read_json(tmp_path / "c", "sweep.json")
    can = body["canary"]
    assert can["enabled"] and can["length"] == 128
    assert [r["before_item"] for r in can["runs"]] == [0, 1, None]  # before each length, and at the end
    assert can["runs"][0]["ratio_to_first"] == 1.0
    assert len(body["items"]) == 2  # canaries never enter the measurement curves
    for i, it in enumerate(body["items"]):
        d = it["drift"]
        assert d["canary_before"] == i and d["canary_after"] == i + 1
        assert isinstance(d["flagged"], bool) and d["derived_from"]


def test_canary_flags_drift_beyond_threshold():
    import argparse
    args = argparse.Namespace(canary_length=512, canary_repeats=5, drift_threshold=0.05, _results_file="sweep.json")
    c = cli.Canary(args)
    c.runs = [{"index": 0, "before_item": 0, "p50_ms": 100.0, "ratio_to_first": 1.0},
              {"index": 1, "before_item": 1, "p50_ms": 103.0, "ratio_to_first": 1.03},
              {"index": 2, "before_item": 2, "p50_ms": 112.0, "ratio_to_first": 1.12},
              {"index": 3, "before_item": None, "p50_ms": 101.0, "ratio_to_first": 1.01}]
    items = [{}, {}, {}]
    c.annotate(items)
    assert [it["drift"]["flagged"] for it in items] == [False, True, True]
    assert items[1]["drift"]["max_deviation"] == pytest.approx(0.12)
    assert c.block()["max_deviation"] == pytest.approx(0.12)


def test_canary_can_be_disabled(tiny_checkpoint, tmp_path, monkeypatch):
    monkeypatch.setattr(results, "RESULTS_ROOT", tmp_path)
    assert cli.main(["sweep", "--run-id", "n", "--model", tiny_checkpoint, "--threads", "1",
                     "--lengths", "128", "--questions", "1", "--batch-sizes", "1",
                     "--repeats", "1", "--warmup", "0", "--canary-length", "0"]) == 0
    body = results.read_json(tmp_path / "n", "sweep.json")
    assert body["canary"] == {"enabled": False} and "drift" not in body["items"][0]
