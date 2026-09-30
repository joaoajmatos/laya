"""T053: the Phase 2 report (experiments/report2.py)."""
import json
import re

import pytest

from experiments import data as D
from experiments import evalrun as E
from experiments import lengths as Lg
from experiments import report2 as R
from experiments import summary as S
from experiments.results import append_jsonl, write_json


def rec(item, case, correct, split="dev", p=0.8):
    return {"item_id": item, "case_id": case, "split": split, "workflow": "customer_service", "question_type": "choice",
            "status": "measured", "correct": correct, "predicted": "a", "gold": "a" if correct else "b",
            "probabilities": {"a": p, "b": 1 - p}, "prob_predicted": p, "level_error": None, "evidence_visible": "full",
            "low_confidence": False, "argmax_agree": True, "tv_top_quartile": False}


def put(run, cid, rows):
    for r in rows:
        append_jsonl(run / "quality" / cid / "predictions.jsonl", dict(r, condition_id=cid))


def latency(run, cid, p50, device="cpu"):
    sub = "gpu_latency" if device == "gpu" else "latency"
    write_json(run / sub, "%s.json" % cid, {"status": "measured", "device": device, "condition": {},
                                             "timings_ms": {"p50": p50, "p95": p50 * 1.2},
                                             "peak_rss_bytes": 3_000_000_000 if device == "cpu" else None})


ACC = {"native": 1.0, "trunc512": 0.6, "truncCap": 0.7, "retrieve1024": 0.995, "window": 0.5, "oracle": 1.0}
COST = {"native": 5000.0, "trunc512": 800.0, "truncCap": 1500.0, "retrieve1024": 2500.0, "window": 4000.0, "oracle": 500.0}


@pytest.fixture(scope="module")
def world(tiny_agent, tiny_upstream, tmp_path_factory):
    base = tmp_path_factory.mktemp("report")
    root = base / "data"
    D.import_dataset(root=root, download=tiny_upstream.download, with_checkpoints=False)
    D.make_splits(seed=3, root=root)
    Lg.write_length_profile(tiny_agent, "tiny", root)
    run = base / "results" / "run1"
    n_cases = 60
    for L in (2048, 4096, 8192):
        for name, acc in ACC.items():
            rows = []
            for c in range(n_cases):
                for q in range(5):
                    idx = c * 5 + q
                    rows.append(rec("c%d|q%d|%d" % (c, q, L), "c%d" % c, (idx % 200) < int(acc * 200)))
            put(run, "%s.none.cpu.L%d" % (name, L), rows)
            latency(run, "%s.none.cpu.L%d" % (name, L), COST[name] * (L / 2048.0))
    latency(run, "native.none.gpu.L2048", 90.0, "gpu")
    # an optimized variant: the same answers as native on 20 items
    put(run, "native.int8_encoder.cpu.L2048", [rec("c%d|q%d|2048" % (c, q), "c%d" % c, True) for c in range(4) for q in range(5)])
    latency(run, "native.int8_encoder.cpu.L2048", 3800.0)
    write_json(run, "solvability.json", E.solvability_report(
        {"fine_tuned": {"solves": True, "summary": {"accuracy": {"overall": 0.75, "n": 600}}, "majority_accuracy": 0.46,
                        "paired_vs_majority": {"diff": 0.29, "lo": 0.25, "hi": 0.33}, "status": "measured"},
         "base": {"solves": False, "summary": {"accuracy": {"overall": 0.36, "n": 600}}, "majority_accuracy": 0.46,
                  "paired_vs_majority": {"diff": -0.1, "lo": -0.14, "hi": -0.06}, "status": "measured"}}, ["fine_tuned", "base"]))
    (root / "audit_result.json").write_text(json.dumps({"families_id": "abc", "n_audited": 60, "share_answer_changed": 0.0,
                                                         "share_ambiguous": 0.0, "all_evidence_intact": True, "passed": True}), encoding="utf-8")
    (run / "commands.txt").write_text("python -m experiments eval --split dev --run-id run1\n", encoding="utf-8")
    S.build_summary(run, n_boot=500)
    return {"run": run, "root": root}


def test_every_statement_is_labeled_and_names_its_sources(world):
    body = R.build_report(world["run"], world["root"])
    n = 0
    for name, sec in body["sections"].items():
        for st in sec.get("statements", []):
            n += 1
            assert st["label"] in R.LABELS and st["text"]
    assert n > 15
    md = R.render_markdown(body)
    bullets = [ln for ln in md.splitlines() if ln.startswith("- [")]
    assert bullets and all(re.match(r"- \[(measured|estimated|hypothesized)\] ", b) for b in bullets)
    assert "`summary.json`" in md


def test_verdicts_follow_the_paired_intervals_at_2k_4k_and_8k(world):
    v = R.build_report(world["run"], world["root"])["sections"]["verdicts"]
    by = {(t["length"], t["reference"]): t for t in v["table"]}
    for L in (2048, 4096, 8192):
        assert by[(L, "trunc512")]["verdict"] == "better" and by[(L, "trunc512")]["lo"] > 0
        assert by[(L, "truncCap")]["verdict"] == "better"
    texts = " ".join(s["text"] for s in v["statements"])
    assert "is better than truncation to 512 tokens" in texts and "case-clustered" in texts


def test_a_baseline_within_the_margin_at_lower_cost_is_named(world):
    m = R.build_report(world["run"], world["root"])["sections"]["margin"]
    names = {e["condition_id"] for e in m["qualifying"]}
    assert "retrieve1024.none.cpu.L2048" in names
    assert not names & {"trunc512.none.cpu.L2048", "window.none.cpu.L2048", "truncCap.none.cpu.L2048"}
    assert "retrieve1024.none.cpu.L2048" in m["statements"][0]["text"]
    assert all(e["cpu_p50_ms"] < e["native_cpu_p50_ms"] for e in m["qualifying"])


def test_oracle_is_a_diagnostic_control_and_never_ranked_as_deployable(world):
    body = R.build_report(world["run"], world["root"])
    rows = [r for r in body["sections"]["curves"]["rows"] if r["name"] == "oracle"]
    assert rows and all(r["deployable"] is False for r in rows)
    assert "diagnostic control, not deployable" in R.render_markdown(body)
    names = {e["condition_id"] for e in body["sections"]["margin"]["qualifying"]}
    assert not any(n.startswith("oracle") for n in names)


def test_cpu_tables_never_read_gpu_latency_and_gpu_has_its_own_section(world):
    body = R.build_report(world["run"], world["root"])
    cpu = body["sections"]["curves"]["curves"]["cpu"]["native"]
    assert {p["length"]: p["latency_p50_ms"] for p in cpu}[2048] == 5000.0          # from latency/, not the 90 ms GPU file
    gpu = body["sections"]["curves"]["curves"]["gpu"]["native"]
    assert gpu[0]["latency_p50_ms"] == 90.0
    md = R.render_markdown(body)
    assert "## GPU latency (GPU only; never substituted for CPU)" in md
    cpu_part = md.split("## GPU latency")[0]
    assert "90" not in cpu_part.split("## Quality against cost (latency on CPU)")[1].split("## Native versus")[0].replace("900", "")


def test_optimized_variants_are_apart_and_state_a_zero_change(world):
    v = R.build_report(world["run"], world["root"])["sections"]["variants"]
    assert len(v["rows"]) == 1 and v["rows"][0]["variant"] == "int8_encoder"
    assert "+0.0 points" in v["statements"][0]["text"] and "optimized-system variant" in v["statements"][0]["text"]
    assert not any("int8" in r["condition_id"] for r in R.build_report(world["run"], world["root"])["sections"]["curves"]["rows"])


def test_phase3_table_lists_each_direction_with_status_and_label(world):
    rows = {r["direction"]: r for r in R.build_report(world["run"], world["root"])["sections"]["phase3"]["rows"]}
    assert set(rows) == {"sparse attention", "selection", "compression", "none (truncate)"}
    assert rows["selection"]["status"] == "supported"                # retrieval matches native at lower cost
    assert rows["sparse attention"]["status"] == "weakened"          # a simple baseline substitutes
    assert rows["compression"]["status"] == "open" and rows["compression"]["label"] == "hypothesized"
    assert rows["none (truncate)"]["status"] == "weakened"           # native beats truncation


def test_limits_reproduce_commands_and_final_count(world):
    body = R.build_report(world["run"], world["root"])
    text = " ".join(s["text"] for s in body["sections"]["limits"]["statements"]).lower()
    for word in ("synthetic", "public", "trained", "realistic", "decoder"):
        assert word in text
    assert body["sections"]["reproduce"]["commands"] == ["python -m experiments eval --split dev --run-id run1"]
    assert body["sections"]["plan"]["final_scored_items"] == 0
    assert "Final-split items scored in this phase: 0." in R.render_markdown(body)


def test_missing_pieces_are_stated_not_invented(tmp_path):
    (tmp_path / "empty").mkdir()
    body = R.build_report(tmp_path / "empty", tmp_path / "nodata")
    md = R.render_markdown(body)
    assert "has not been imported" in md and "solvability check has not been run" in md
    assert "was not measured" in md and "has not been frozen" in md
    rows = {r["direction"]: r for r in body["sections"]["phase3"]["rows"]}
    assert rows["all"]["status"] == "open"
    with pytest.raises(R.ReportError):
        R.build_report(tmp_path / "nope")


def test_write_report_writes_json_and_markdown(world):
    j, m = R.write_report(world["run"], world["root"])
    assert j.exists() and m.exists() and json.loads(j.read_text(encoding="utf-8"))["sections"]["data"]["statements"]


# --------------------------------------------------------------------------- GPU-scored quality in the report (FR-024)

def _gpu_world(tmp_path, gpu_flips=0):
    """A run whose headline quality was scored on the GPU, with a CPU parity subset."""
    run = tmp_path / "gpuresults" / "run2"
    for L in (2048, 4096, 8192):
        for name, acc in ACC.items():
            rows = [rec("c%d|q%d|%d" % (c, q, L), "c%d" % c, ((c * 5 + q) % 200) < int(acc * 200))
                    for c in range(60) for q in range(5)]
            put(run, "%s.none.gpu.L%d" % (name, L), rows)
            latency(run, "%s.none.cpu.L%d" % (name, L), COST[name] * (L / 2048.0))
    # the CPU parity subset: native and truncCap at 2048 on the first 20 items, GPU flips `gpu_flips` of them
    for name in ("native", "truncCap"):
        gpu = list(S.read_jsonl(run / "quality" / ("%s.none.gpu.L2048" % name) / "predictions.jsonl"))[:20]
        cpu = [dict(r, device="cpu") for r in gpu]
        for r in gpu[:gpu_flips]:
            pass
        put(run, "%s.none.cpu.L2048" % name, cpu)
    if gpu_flips:
        path = run / "quality" / "native.none.gpu.L2048" / "predictions.jsonl"
        lines = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]
        for r in lines[:gpu_flips]:
            r["predicted"] = "b" if r["predicted"] == "a" else "a"
            r["correct"] = not r["correct"]
        path.write_text("\n".join(json.dumps(x) for x in lines) + "\n", encoding="utf-8")
    latency(run, "native.none.gpu.L2048", 90.0, "gpu")
    S.build_summary(run, n_boot=300)
    return run


def test_gpu_scored_quality_is_labeled_and_the_parity_result_is_stated(tmp_path):
    run = _gpu_world(tmp_path)
    body = R.build_report(run)
    assert body["sections"]["curves"]["quality_device"] == "gpu"
    assert all(r["quality_device"] == "gpu" for r in body["sections"]["curves"]["rows"])
    par = " ".join(s["text"] for s in body["sections"]["parity"]["statements"])
    assert "stands in for CPU quality" in par and "passed" in par
    curves = " ".join(s["text"] for s in body["sections"]["curves"]["statements"])
    assert "scored on the GPU" in curves and "latency and memory are CPU" in curves
    # CPU cost still comes from latency/, joined through the CPU twin of each GPU-scored cell
    row = next(r for r in body["sections"]["curves"]["rows"] if r["condition_id"] == "native.none.gpu.L2048")
    assert row["cpu_p50_ms"] == 5000.0 and "latency/native.none.cpu.L2048.json" in row["sources"]


def test_verdicts_and_the_margin_question_use_the_gpu_scored_cells(tmp_path):
    body = R.build_report(_gpu_world(tmp_path))
    by = {(t["length"], t["reference"]): t for t in body["sections"]["verdicts"]["table"]}
    assert by[(2048, "trunc512")]["verdict"] == "better"
    names = {e["condition_id"] for e in body["sections"]["margin"]["qualifying"]}
    assert "retrieve1024.none.gpu.L2048" in names and all(".gpu." in n for n in names)


def test_a_failed_parity_check_leaves_gpu_quality_unverified(tmp_path):
    body = R.build_report(_gpu_world(tmp_path, gpu_flips=3))          # 3 of 20 predictions differ: 85% identical
    par = " ".join(s["text"] for s in body["sections"]["parity"]["statements"])
    assert body["sections"]["parity"]["passed"] is False and "UNVERIFIED" in par and "FAILED" in par


def test_gpu_scored_quality_without_a_parity_subset_is_unverified(tmp_path):
    run = tmp_path / "noparity" / "run3"
    put(run, "native.none.gpu.L2048", [rec("i%d" % i, "c%d" % i, "dev", True) for i in range(10)])
    S.build_summary(run, n_boot=50)
    par = R.build_report(run)["sections"]["parity"]["statements"][0]["text"]
    assert "no CPU parity subset exists yet" in par and "unverified" in par


def test_cpu_scored_runs_say_no_parity_applies(world):
    par = R.build_report(world["run"], world["root"])["sections"]["parity"]["statements"][0]["text"]
    assert "scored on CPU" in par
