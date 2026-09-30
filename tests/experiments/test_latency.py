"""T039: latency of baseline conditions (experiments/latency.py)."""
import json
import time

import pytest

from experiments import baselines as B
from experiments import data as D
from experiments import evalrun as E
from experiments import families as F
from experiments import latency as L
from experiments import timing
from experiments.results import Refusal, read_json


@pytest.fixture(scope="module")
def fam(tiny_agent, tiny_upstream, tiny_checkpoint, tmp_path_factory):
    root = tmp_path_factory.mktemp("latency-data")
    D.import_dataset(root=root, download=tiny_upstream.download, with_checkpoints=False)
    D.make_splits(seed=3, root=root)
    meta = F.build_split(tiny_agent, "dev", lengths=(256, 512), variants=("distractor@mid",), seed=0, root=root,
                         tokenizer_name="tiny")
    return {"root": root, "fid": meta["families_id"], "model": tiny_checkpoint}


# The tiny dataset has 3 dev cases per workflow, so its half-sample (where the sample is drawn) holds 2:
# 8 items here, 12 (three per workflow) at full size (see test_data.py).
N = 8


def spec(fam, cond, **kw):
    return dict({"condition": cond, "model": fam["model"], "revision": None, "threads": 1, "families_id": fam["fid"],
                 "data_root": str(fam["root"]), "repeats": 1, "warmup": 1, "time_cap": None}, **kw)


def cond(name="native", length=512, **kw):
    return E.make_condition(name, length, **kw)


def test_sample_is_the_fixed_dev_sample(fam):
    splits = D.read_data_json("splits.json", fam["root"])
    items = L.latency_items(fam["fid"], splits, 512, fam["root"])
    assert len(items) == N == len(splits["latency_sample"]) and {i["variant"] for i in items} == {"distractor@mid"} and {i["split"] for i in items} == {"dev"}
    per_wf = {}
    for i in items:
        per_wf[i["workflow"]] = per_wf.get(i["workflow"], 0) + 1
    assert per_wf == {wf: 2 for wf in D.WORKFLOWS}
    assert [i["item_id"] for i in items] == [i["item_id"] for i in L.latency_items(fam["fid"], splits, 512, fam["root"])]


def test_repeats_default_by_length():
    assert [L.default_repeats(n) for n in (512, 1024, 2048, 4096, 8192)] == [3, 3, 2, 1, 1]


def test_cpu_result_has_the_cost_result_fields_and_uses_the_clean_path(fam, monkeypatch):
    calls = []
    real = timing.assert_clean
    monkeypatch.setattr(timing, "assert_clean", lambda a: (calls.append(1), real(a))[1])
    rec = L.latency_condition(spec(fam, cond("native", 512), repeats=2))
    assert rec["status"] == "measured" and rec["device"] == "cpu" and rec["kind"] == "single"
    assert len(rec["sample"]) == N and rec["repeats"] == {"requested": 2, "completed": 2 * N, "warmup": 1}
    assert rec["timings_ms"]["p50"] > 0 and rec["timings_ms"]["p95"] >= rec["timings_ms"]["p50"]
    assert len(rec["samples_ms"]) == 2 * N and len(calls) == 2 * N                       # the guard ran before every timed call
    assert {"fallbacks", "position_limit", "configured_max_len", "beyond_configured_max_len", "mha_fastpath",
            "tokens_seen", "clean_path", "drift_within_condition"} <= set(rec)
    assert rec["fallbacks"]["device"] == "cpu" and rec["beyond_configured_max_len"] is True
    assert rec["tokens_seen"]["min"] == rec["tokens_seen"]["max"] == 512


def test_a_hook_makes_clean_timing_refuse(fam, monkeypatch, tiny_agent):
    handle = tiny_agent.model.encoder.register_forward_hook(lambda *a: None)
    try:
        with pytest.raises(timing.CleanPathError):
            timing.assert_clean(tiny_agent)
    finally:
        handle.remove()


def test_selection_is_inside_the_timed_call(fam, monkeypatch):
    real = B.RetrieveRunner.select

    def slow(self, agent, item):
        time.sleep(0.05)
        return real(self, agent, item)

    monkeypatch.setattr(B.RetrieveRunner, "select", slow)
    rec = L.latency_condition(spec(fam, cond("retrieve256", 512)))
    assert rec["status"] == "measured" and rec["timings_ms"]["p50"] >= 50.0


def test_windowing_is_inside_the_timed_call_and_accounting_is_not(fam, monkeypatch):
    from experiments import tokens
    monkeypatch.setattr(tokens, "account", lambda *a, **k: (_ for _ in ()).throw(AssertionError("timed path")))
    # annotate() uses tokens.account only for native-style runners; the window runner tokenizes on its own.
    rec = L.latency_condition(spec(fam, cond("window", 512, params={"size": 100})))
    assert rec["status"] == "measured" and rec["deployable"] is True


def test_int8_and_fastpath_variants_are_unsupported_on_gpu_without_loading(fam):
    for v in ("int8_encoder", "int8_all_nofast", "fastpath_off"):
        rec = L.latency_condition(spec(fam, cond("native", 512, variant=v, device="gpu")))
        assert rec["status"] == "unsupported" and "CPU-only" in rec["reason"] and rec["device"] == "gpu"


def test_a_length_with_no_runnable_sample_item_is_unsupported(fam):
    rec = L.latency_condition(spec(fam, cond("native", 1024)))               # families were built for 256 and 512 only
    assert rec["status"] == "unsupported" and "no" in rec["reason"]


def test_gpu_requests_without_cuda_are_refused_and_never_fall_back_to_cpu(fam, tmp_path):
    import torch
    if torch.cuda.is_available():
        pytest.skip("this environment has CUDA")
    with pytest.raises(Refusal) as err:
        L.run_latency(tmp_path, [cond("native", 512, device="gpu")], fam["model"], fam["fid"], revision=None,
                      data_root=fam["root"], runner_fn=lambda *a, **k: {"status": "measured"})
    assert err.value.reason == "device_mismatch" and "CPU results are not substituted" in str(err.value)
    assert not (tmp_path / "gpu_latency").exists() and not (tmp_path / "latency").exists()


def test_gpu_files_are_written_apart_from_cpu_files(fam, tmp_path):
    fake = lambda fn, s, time_cap=None: {"status": "measured", "timings_ms": {"p50": 12.0}}      # noqa: E731
    L.run_latency(tmp_path, [cond("native", 512, device="gpu")], fam["model"], fam["fid"], revision=None,
                  data_root=fam["root"], require_cuda=False, runner_fn=fake)
    L.run_latency(tmp_path, [cond("native", 512)], fam["model"], fam["fid"], revision=None,
                  data_root=fam["root"], runner_fn=fake)
    assert (tmp_path / "gpu_latency" / "native.none.gpu.L512.json").exists()
    assert (tmp_path / "latency" / "native.none.cpu.L512.json").exists()
    body = read_json(tmp_path / "gpu_latency", "native.none.gpu.L512.json")
    assert body["device"] == "gpu" and body["condition"]["device"] == "gpu"
    assert L.latency_dir(tmp_path, "cpu").name == "latency" and L.latency_dir(tmp_path, "gpu").name == "gpu_latency"


def test_the_time_cap_yields_a_partial_result_with_a_reason(fam):
    rec = L.latency_condition(spec(fam, cond("native", 512), repeats=50, time_cap=0.5))
    assert rec["status"] == "partial" and "time cap" in rec["reason"] and rec["repeats"]["completed"] < 50 * N
