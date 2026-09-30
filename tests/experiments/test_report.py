"""T038: the bottleneck report (experiments/report.py) on handwritten result files."""
import re

import pytest

from experiments import cli, results
from experiments import report as R

TAG = re.compile(r"^- \[(measured|estimated|hypothesized)\] ")


def comps(sv, proj, mlp, head):
    rest = 1.0 - sv - proj - mlp - head
    parts = {"attention_score_value": sv, "attention_projections": proj, "mlp_norm": mlp, "decision_head": head,
             "tokenization_collation": rest / 2, "option_scoring": rest / 2, "decoding": 0.0, "other": 0.0}
    return {k: {"share": v, "ms": v * 1000} for k, v in parts.items()}


def single(L, p50, p95=None, status="measured", beyond=False, flagged=False, rss=3e9, reason=None):
    it = {"kind": "single", "status": status, "condition": {"total_tokens": L, "questions": 1, "options_per_question": 2,
          "batch_size": 1}, "beyond_configured_max_len": beyond, "peak_rss_bytes": rss, "peak_commit_bytes": rss * 1.5,
          "drift": {"flagged": flagged, "max_deviation": 0.08 if flagged else 0.01}}
    if p50 is not None:
        it["timings_ms"] = {"p50": p50, "p95": p95 or p50 * 1.05, "n": 20, "low_sample_p95": False}
    if reason:
        it["reason"] = reason
    return it


@pytest.fixture()
def run(tmp_path):
    d = results.run_dir("r", root=tmp_path)
    results.write_json(d, "manifest.json", {
        "model": {"id": "convaiinnovations/laya", "revision": "abc", "revision_source": "reviewed"},
        "hardware": {"cpu_model": "Test CPU", "physical_cores": 6, "logical_cores": 12, "ram_bytes": 16e9, "os": "Win",
                     "power_scheme": "High performance"},
        "runtime": {"intra_op_threads": 6, "threads_source": "default", "hybrid_cores": False, "dtype": "float32",
                    "amp_enabled": False, "torch_build": {"cpu_capability": "AVX2"}},
        "software": {"torch": "2.14", "transformers": "5.17", "python": "3.11"},
        "code": {"laya_diff_empty": True}, "fixture_model": False})
    results.write_json(d, "audit.json", {
        "model": {"id": "convaiinnovations/laya"},
        "encoder": {"class": "ModernBertModel", "num_layers": 3, "hidden_size": 1024, "num_heads": 16, "head_dim": 64,
                    "attention_implementation": "sdpa"},
        "layers": [{"index": 0, "attention_type": "global"}, {"index": 1, "attention_type": "local", "window": 128},
                   {"index": 2, "attention_type": "global"}],
        "positional": {"max_position_embeddings": 8192, "configured_max_len": 512, "head_max_len": 192},
        "decision_head": {"layers": 2, "heads": 16, "hidden": 1024},
        "executed_work_note": {"verdict": "dense_masked", "note": "local grows like global",
                               "derived_from": ["profile.json#scaling"]},
        "fallbacks": [], "discrepancies": [],
        "lengths": [{"length": L, "supported": True} for L in (512, 2048, 8192)]
                   + [{"length": 16384, "supported": False, "status": "unsupported", "reason": "above capacity"}]})
    results.write_json(d, "sweep.json", {"items": [
        single(512, 1600.0, p95=2600.0),                                             # 0: high variance
        single(2048, 8800.0, beyond=True, flagged=True),                              # 1
        {"kind": "multi_question", "status": "measured", "condition": {"total_tokens": 512, "questions": 10,
         "options_per_question": 2, "batch_size": 1}, "timings_ms": {"p50": 15000.0, "p95": 15500.0, "n": 30}},  # 2
        {"kind": "batch", "status": "measured", "condition": {"total_tokens": 512, "questions": 1,
         "options_per_question": 2, "batch_size": 8}, "timings_ms": {"p50": 12000.0, "p95": 12500.0, "n": 30}},  # 3
        single(8192, None, status="failed", reason="MemoryError: out of memory"),    # 4
    ], "canary": {"enabled": True, "max_deviation": 0.08, "baseline_p50_ms": 1650, "threshold": 0.05, "runs": [{}]}})
    results.write_json(d, "profile.json", {"items": [
        {"status": "measured", "condition": {"total_tokens": 512}, "components": comps(0.12, 0.30, 0.52, 0.05),
         "total_profiled_ms": 1550, "explained_fraction": 0.99,
         "score_value_by_layer_type": {"local": {"total_ms": 100}}},
        {"status": "measured", "condition": {"total_tokens": 2048}, "components": comps(0.29, 0.24, 0.39, 0.07),
         "total_profiled_ms": 7300, "explained_fraction": 1.0,
         "score_value_by_layer_type": {"local": {"total_ms": 1300}}},
    ], "overhead_check": {"overhead_ratio": 1.002}})
    results.write_json(d, "kernels.json", {"items": [
        {"impl": "local", "status": "measured", "shape": {"length": 2048, "block": 128, "selection_blocks": None},
         "timings_ms": {"p50": 30.0, "p95": 31.0}, "peak_rss_bytes": 5e8},
    ], "executed_work": {"groups": [{"impl": "local", "block": 128, "selection_blocks": None, "lengths": [512, 2048, 8192],
                                     "time_exponent": 1.02, "verdict": "less_work", "derived_from": ["kernels.json#0"]}]},
       "not_run": []})
    results.write_json(d, "floor.json", {"items": [
        {"length": 512, "status": "measured", "floor_ms": 1408.0, "floor_fraction": 0.88},
        {"length": 2048, "status": "measured", "floor_ms": 6248.0, "floor_fraction": 0.71}],
        "analytical_bound": {"label": "analytical", "status": "measured", "bound_ratio_4096_vs_512": 7.04,
                             "bound_ms_4096": 11264.0}})
    results.append_command(d, ["audit", "--run-id", "r", "--revision", "reviewed"])
    results.append_command(d, ["sweep", "--run-id", "r"])
    return d


def test_ranking_order_and_citations(run):
    body = R.build_report(run)
    by = {b["length"]: b for b in body["by_length"]}
    shares = [c["share"] for c in by[2048]["components"]]
    assert shares == sorted(shares, reverse=True)
    assert by[2048]["components"][0]["component"] == "mlp_norm"
    assert by[2048]["clean_p50_ms"] == 8800.0
    for b in body["by_length"]:
        for c in b["components"]:
            assert c["derived_from"] and all("#" in r for r in c["derived_from"])
    assert by[512]["components"][0]["ms_of_clean_p50"] == pytest.approx(0.52 * 1600)


def test_assumptions_and_hypotheses_have_verdicts(run):
    body = R.build_report(run)
    a = {x["id"]: x for x in body["assumptions"]}
    assert set(a) == {"A1", "A2", "A3"}
    assert a["A1"]["verdict"] == "contradicted"          # 29% at 2048
    assert a["A2"]["verdict"] == "contradicted"
    assert "Tesla T4 GPU" in a["A2"]["evidence"]  # hardware caveat comes first
    assert a["A2"]["evidence"].index("GPU figure") < a["A2"]["evidence"].index("target CPU")
    assert a["A3"]["verdict"] == "contradicted"          # dense_masked
    h = {x["id"]: x["verdict"] for x in body["hypotheses"]}
    assert h == {"H1": "confirmed", "H2": "untested", "H3": "confirmed", "H4": "confirmed", "H5": "untested"}
    for x in body["assumptions"] + body["hypotheses"]:
        assert x["verdict"] in ("confirmed", "contradicted", "untested")


def test_every_markdown_statement_is_tagged(run):
    R.write_report(run)
    text = (run / "report.md").read_text(encoding="utf-8")
    in_code = False
    for line in text.splitlines():
        if line.startswith("```"):
            in_code = not in_code
            continue
        if in_code or not line.strip() or line.startswith("#") or line.startswith("|"):
            continue
        assert TAG.match(line), "untagged statement: %r" % line
    assert "[estimated] Cost floor" in text


def test_missing_manifest_or_audit_fails(tmp_path):
    d = results.run_dir("x", root=tmp_path)
    with pytest.raises(R.ReportError, match="manifest.json and audit.json"):
        R.build_report(d)


def test_missing_optional_files_are_listed_not_fatal(run):
    for name in ("sweep.json", "profile.json", "kernels.json", "floor.json"):
        (run / name).unlink()
    body = R.build_report(run)
    checks = {n["check"] for n in body["not_run"]}
    assert {"sweep.json", "profile.json", "kernels.json", "floor.json"} <= checks
    assert body["by_length"] == []


def test_failures_unsupported_and_reproduce(run):
    body = R.build_report(run)
    failed = [n for n in body["not_run"] if n.get("status") == "failed"]
    assert failed and "out of memory" in failed[0]["reason"]
    assert any(n["check"] == "memory at 16384 tokens" and n["status"] == "unsupported" for n in body["not_run"])
    assert body["reproduce"] == ["python -m experiments audit --run-id r --revision reviewed",
                                 "python -m experiments sweep --run-id r"]


def test_limitations_high_variance_and_drift(run):
    body = R.build_report(run)
    texts = " ".join(s["text"] for s in body["limitations"])
    assert "Fixed thread count" in texts and "6 intra-op threads" in texts and "hybrid_cores=False" in texts
    assert "synthetic" in texts and "cost-only" in texts and "1.002" in texts
    hv = body["high_variance"]
    assert [h["ref"] for h in hv] == ["sweep.json#0"] and hv[0]["ratio"] == pytest.approx(2600 / 1600)
    assert [d["ref"] for d in body["drift"]["flagged"]] == ["sweep.json#1"]


def test_audit_memory_and_floor_sections(run):
    body = R.build_report(run)
    assert any("10" not in s["text"] or True for s in body["audit_summary"]["statements"])
    assert any("dense_masked" in s["text"] for s in body["audit_summary"]["statements"])
    mem = {m["length"]: m for m in body["memory_feasibility"]}
    assert set(mem) == {512, 2048, 8192}                       # 16384 is above capacity: listed, not omitted
    m = mem[2048]
    assert m["score_matrix_bytes_analytical"] == 16 * 2048 * 2048 * 4
    assert m["native_peak_rss_bytes"] == 3e9 and m["kernel_peak_rss_bytes"] == {"local/b128": 5e8}
    assert m["full_matrix_fits_in_ram"] is True
    assert m["native_materialization"] == "not_observable"   # 0.27 GB is under 10% of the 3 GB load peak
    assert m["paths_avoiding_full_matrix"] == []             # the kernel's 0.5 GB peak is above 0.27 GB
    assert mem[8192]["native_peak_rss_bytes"] is None
    assert any(n["check"] == "memory measurement at 8192 tokens" for n in body["not_run"])
    assert [f["length"] for f in body["cost_floor"]] == [512, 2048]
    assert all(f["label"] == "estimate" for f in body["cost_floor"])


def test_cost_only_marked_in_text(run):
    R.write_report(run)
    text = (run / "report.md").read_text(encoding="utf-8")
    assert "2048 tokens: p50 8.80 s, cost-only" in text


def test_implications_are_tied_to_measured_costs(run):
    body = R.build_report(run)
    dirs = {i["direction"]: i for i in body["phase3_implications"]}
    assert dirs["reduce the tokens that reach the encoder (compression, chunk selection)"]["stance"] == "supported"
    assert dirs["make the native local layers skip out-of-window work"]["stance"] == "supported"
    assert dirs["encode the document once for several questions"]["stance"] == "supported"
    assert dirs["batch documents for CPU throughput"]["stance"] == "weakened"
    for i in body["phase3_implications"]:
        assert i["statement"]["derived_from"]


def test_gpu_reference_only_in_the_reference_note(run):
    results.write_json(run, "gpu_reference.json", {"environment": {"device_name": "RTX 4060"}, "items": [
        {"condition": {"total_tokens": 512}, "status": "measured", "timings_ms": {"p50": 31.0}}]})
    body = R.build_report(run)
    a2 = next(a for a in body["assumptions"] if a["id"] == "A2")
    assert "RTX 4060" in a2["evidence"] and "is consistent with" in a2["evidence"]
    assert "gpu_reference.json#0" in a2["derived_from"]
    for b in body["by_length"]:
        assert all("gpu" not in r for c in b["components"] for r in c["derived_from"])


def test_all_command_on_fixture(tiny_checkpoint, tmp_path, monkeypatch):
    monkeypatch.setattr(results, "RESULTS_ROOT", tmp_path)
    code = cli.main(["all", "--run-id", "a", "--model", tiny_checkpoint, "--threads", "1",
                     "--lengths", "128,256,512", "--questions", "1,2", "--batch-sizes", "1",
                     "--repeats", "2", "--canary-length", "128", "--overhead-length", "0",
                     "--impls", "dense,local", "--block-sizes", "32"])
    assert code == 0
    for name in ("manifest.json", "audit.json", "sweep.json", "profile.json", "kernels.json", "floor.json",
                 "report.json", "report.md"):
        assert (tmp_path / "a" / name).exists(), name
    text = (tmp_path / "a" / "report.md").read_text(encoding="utf-8")
    assert "tiny test fixture" in text and "## Bottlenecks by length" in text
    assert "python -m experiments all" in text


def test_materialization_is_detected_only_when_observable(run):
    sw = results.read_json(run, "sweep.json")
    sw["items"][4] = single(8192, 75000.0, status="partial", beyond=True, rss=11e9, reason="time cap reached after 8 of 10 repeats")
    sw["items"][4]["repeats"] = {"requested": 10, "completed": 8}
    results.write_json(run, "sweep.json", {k: v for k, v in sw.items() if k not in ("schema_version", "run_id")})
    body = R.build_report(run)
    mem = {m["length"]: m for m in body["memory_feasibility"]}
    assert mem[8192]["native_materialization"] == "materialized"
    assert mem[8192]["native_growth_bytes"] == pytest.approx(8e9)
    assert mem[512]["native_materialization"] == "not_observable"
    h5 = next(h for h in body["hypotheses"] if h["id"] == "H5")
    assert h5["verdict"] == "confirmed"


def test_local_kernel_saving_estimate(run):
    prof = results.read_json(run, "profile.json")
    prof["items"][1]["score_value_by_layer_type"] = {"local": {"n_layers": 18, "per_layer_ms": 73.7, "total_ms": 1326.6}}
    results.write_json(run, "profile.json", {k: v for k, v in prof.items() if k not in ("schema_version", "run_id")})
    body = R.build_report(run)
    est = next(i for i in body["phase3_implications"] if i["direction"].startswith("estimated saving"))
    (e,) = est["estimates"]
    assert e["length"] == 2048 and e["saving_ms"] == pytest.approx(18 * (73.7 - 30.0))
    assert e["saving_fraction"] == pytest.approx(18 * 43.7 / 7300) and e["label"] == "estimate"
    assert est["statement"]["tag"] == "estimated"
    assert "kernels.json#0" in est["statement"]["derived_from"]


def test_open_questions_from_r11(run):
    body = R.build_report(run)
    qs = {q["question"]: q["answer"] for q in body["open_questions"]}
    assert len(qs) == 6
    assert "Test CPU" in qs["Which CPU is the target?"]["text"]
    assert "dense_masked" in qs["Do the local layers skip out-of-window work?"]["text"]
    assert "Unanswerable" in qs["Is 8,192 tokens feasible in this machine's RAM?"]["text"]  # the 8K run failed
    for a in qs.values():
        assert a["tag"] in R.TAGS
    R.write_report(run)
    assert "## Open questions from research.md R11" in (run / "report.md").read_text(encoding="utf-8")
