"""T038: calibration and evaluation summary (experiments/summary.py)."""
import numpy as np
import pytest

from experiments import metrics
from experiments import summary as S
from experiments.results import append_jsonl, read_json


def rec(item, case, split, correct, qtype="choice", p=0.9, cid="x", wf="customer_service", vis="full", **kw):
    probs = {"a": p, "b": 1 - p} if qtype != "noul" else {"false": 1 - p, "true": p}
    pred = "a" if qtype != "noul" else "true"
    gold = pred if correct else ("b" if qtype != "noul" else "false")
    r = {"item_id": item, "case_id": case, "split": split, "workflow": wf, "question_type": qtype,
         "status": "measured", "correct": correct, "predicted": pred, "gold": gold, "probabilities": probs,
         "prob_predicted": p, "level_error": None, "evidence_visible": vis, "low_confidence": False,
         "argmax_agree": True, "tv_top_quartile": False, "condition_id": cid}
    r.update(kw)
    return r


def write(run, cid, rows):
    for r in rows:
        r = dict(r, condition_id=cid)
        append_jsonl(run / "quality" / cid / "predictions.jsonl", r)


def test_condition_id_round_trips():
    assert S.parse_condition_id("native.none.cpu.L512") == {"name": "native", "params": "", "variant": "none",
                                                            "device": "cpu", "length": 512}
    p = S.parse_condition_id("window.size512.none.cpu.L2048")
    assert p["name"] == "window" and p["params"] == "size512" and p["length"] == 2048
    assert S.parse_condition_id("native.int8_encoder.cpu.Loriginal")["length"] == "original"
    with pytest.raises(ValueError):
        S.parse_condition_id("nonsense")


def test_temperatures_are_fitted_on_calibration_records_only():
    cal = [rec("i%d" % i, "c%d" % (i // 5), "calibration", i % 5 != 0, p=0.99) for i in range(100)]
    fit = S.fit_condition(cal)
    assert fit["choice"]["n"] == 100 and fit["choice"]["temperature"] > 1.5          # overconfident: T above one
    with pytest.raises(ValueError) as err:
        S.fit_condition(cal + [rec("d", "cd", "dev", True)])
    assert "calibration split only" in str(err.value)


def test_calibrate_writes_only_from_calibration_split_and_summary_applies_it(tmp_path):
    cid = "native.none.cpu.L512"
    cal = [rec("cal%d" % i, "cc%d" % (i // 5), "calibration", i % 5 != 0, p=0.99) for i in range(100)]
    dev = [rec("dev%d" % i, "dc%d" % (i // 5), "dev", i % 5 != 0, p=0.99) for i in range(100)]
    write(tmp_path, cid, cal + dev)
    body = S.calibrate(tmp_path)
    assert list(body["conditions"]) == [cid] and body["fitted_on"] == "calibration"
    assert body["conditions"][cid]["choice"]["n"] == 100                              # the 100 calibration items, not 200
    summ = S.build_summary(tmp_path)
    c = summ["conditions"][cid]
    assert c["ece_raw"]["overall"] == pytest.approx(0.19, abs=1e-6)
    assert c["ece_scaled"]["overall"] < c["ece_raw"]["overall"]                       # raw and scaled are both reported
    assert c["accuracy"]["n"] == 100                                                  # dev only
    assert c["temperatures"]["choice"]["temperature"] > 1.5


def test_window_scaled_ece_is_labeled_non_comparable(tmp_path):
    cid = "window.size256.none.cpu.L1024"
    write(tmp_path, cid, [rec("w%d" % i, "c%d" % (i // 5), "calibration", i % 2 == 0) for i in range(20)]
          + [rec("v%d" % i, "d%d" % (i // 5), "dev", i % 2 == 0) for i in range(20)])
    S.calibrate(tmp_path)
    c = S.build_summary(tmp_path)["conditions"][cid]
    assert "non-comparable" in c["ece_scaled_note"]


def test_summary_has_strata_counts_by_status_and_visibility(tmp_path):
    cid = "trunc512.none.cpu.L2048"
    rows = [rec("a%d" % i, "c%d" % (i // 5), "dev", i % 2 == 0, vis="full" if i % 3 else "none",
                low_confidence=(i % 4 == 0)) for i in range(40)]
    rows.append(dict(rec("z", "cz", "dev", True), status="failed", reason="oom"))
    rows.append(dict(rec("y", "cy", "dev", True), status="unsupported", reason="target_exceeds_length"))
    write(tmp_path, cid, rows)
    c = S.build_summary(tmp_path)["conditions"][cid]
    assert c["counts_by_status"] == {"failed": 1, "measured": 40, "unsupported": 1}
    assert c["n_measured"] == 40 and c["accuracy"]["n"] == 40                        # denominator over measured only
    vis = c["accuracy"]["by_evidence_visible"]
    assert set(vis) == {"full", "none"} and vis["full"]["n"] + vis["none"]["n"] == 40
    assert c["accuracy"]["by_stratum"]["low_confidence"]["n"] == 10
    cell = next(x for x in S.build_summary(tmp_path)["cells"] if x["condition_id"] == cid)
    assert cell["status"] == "partial" and cell["n_measured"] == 40


def test_paired_comparison_against_references_uses_the_same_items_and_case_clusters(tmp_path):
    n = 120
    native = [rec("i%d" % i, "c%d" % (i // 5), "dev", True) for i in range(n)]
    trunc = [rec("i%d" % i, "c%d" % (i // 5), "dev", i % 2 == 0) for i in range(n)]       # 50% right vs 100%
    write(tmp_path, "native.none.cpu.L2048", native)
    write(tmp_path, "trunc512.none.cpu.L2048", trunc)
    write(tmp_path, "oracle.none.cpu.L2048", native)
    summ = S.build_summary(tmp_path, n_boot=500)
    p = summ["conditions"]["native.none.cpu.L2048"]["paired"]["trunc512.none.cpu.L2048"]
    assert p["diff"] == pytest.approx(0.5) and p["verdict"] == "better" and p["n_cases"] == 24
    q = summ["conditions"]["native.none.cpu.L2048"]["paired"]["oracle.none.cpu.L2048"]
    assert q["diff"] == 0.0 and q["verdict"] == "equal"
    back = summ["conditions"]["trunc512.none.cpu.L2048"]["paired"]["native.none.cpu.L2048"]
    assert back["verdict"] == "worse"


def test_underpowered_comparisons_are_inconclusive(tmp_path):
    rng = np.random.default_rng(1)
    a = rng.random(10) < 0.6
    b = rng.random(10) < 0.5
    write(tmp_path, "native.none.cpu.L512", [rec("i%d" % i, "c%d" % i, "dev", bool(a[i])) for i in range(10)])
    write(tmp_path, "trunc512.none.cpu.L512", [rec("i%d" % i, "c%d" % i, "dev", bool(b[i])) for i in range(10)])
    p = S.build_summary(tmp_path, n_boot=500)["conditions"]["native.none.cpu.L512"]["paired"]["trunc512.none.cpu.L512"]
    assert p["verdict"] == "inconclusive" and p["n_cases"] == 10


def test_missing_expected_cells_are_listed_and_optimized_variants_stay_out_of_the_pairing(tmp_path):
    write(tmp_path, "native.none.cpu.L512", [rec("i%d" % i, "c%d" % i, "dev", True) for i in range(6)])
    write(tmp_path, "native.int8_encoder.cpu.L512", [rec("i%d" % i, "c%d" % i, "dev", True) for i in range(6)])
    summ = S.build_summary(tmp_path, expected=["native.none.cpu.L512", "window.sizedefault.none.cpu.L512"], n_boot=100)
    status = {c["condition_id"]: c["status"] for c in summ["cells"]}
    assert status["window.sizedefault.none.cpu.L512"] == "missing" and status["native.none.cpu.L512"] == "measured"
    assert summ["conditions"]["native.int8_encoder.cpu.L512"]["paired"] == {}
    assert read_json(tmp_path, "summary.json")["split"] == "dev"


def test_retrieval_budget_selection_uses_dev_accuracy(tmp_path):
    for budget, acc_n in (("retrieve512", 2), ("retrieve1024", 5), ("retrieve2048", 5)):
        rows = [rec("i%d" % i, "c%d" % i, "dev", i < acc_n) for i in range(6)]
        write(tmp_path, "%s.none.cpu.L2048" % budget, rows)
    sel = S.build_summary(tmp_path, n_boot=50)["retrieval_budget"]
    assert sel["selected"] == 1024 and sel["tuned_on"] == "dev"          # tie with 2048: the smaller budget
    assert sel["mean_accuracy"]["512"] == pytest.approx(2 / 6)


# --------------------------------------------------------------------------- CPU/GPU parity (FR-024, 2026-09-30)

def _pair(tmp_path, n=100, flips=0, acc_shift=0):
    """The same items scored on CPU and GPU; `flips` of them change their predicted answer on GPU."""
    cpu = [rec("i%d" % i, "c%d" % (i // 5), "dev", i % 2 == 0, p=0.8) for i in range(n)]
    gpu = []
    for i, r in enumerate(cpu):
        g = dict(r)
        if i < flips:
            g["predicted"] = "b" if r["predicted"] == "a" else "a"
            g["correct"] = not r["correct"]
            g["probabilities"] = {"a": 0.4, "b": 0.6} if g["predicted"] == "b" else {"a": 0.6, "b": 0.4}
        else:
            g["probabilities"] = {"a": 0.8005, "b": 0.1995}
        gpu.append(g)
    write(tmp_path, "native.none.cpu.L512", cpu)
    write(tmp_path, "native.none.gpu.L512", gpu)


def test_parity_passes_when_predictions_agree_and_accuracy_matches(tmp_path):
    _pair(tmp_path, flips=0)
    par = S.parity(tmp_path)
    r = par["rows"][0]
    assert par["passed"] is True and r["n_items"] == 100 and r["same_prediction_share"] == 1.0
    assert r["accuracy_difference"] == 0.0 and r["max_abs_probability_difference"] == pytest.approx(0.0005, abs=1e-6)
    assert r["condition_id"] == "native.none.cpu.L512" and r["gpu_condition_id"] == "native.none.gpu.L512"
    assert par["criteria"] == {"min_same_prediction_share": 0.98, "max_accuracy_difference": 0.005}


def test_parity_fails_when_too_many_answers_flip(tmp_path):
    _pair(tmp_path, flips=5)                       # 95% identical: below the 98% criterion
    par = S.parity(tmp_path)
    assert par["passed"] is False and par["rows"][0]["same_prediction_share"] == 0.95
    assert par["rows"][0]["passed"] is False


def test_parity_fails_on_an_accuracy_gap_even_with_98_percent_identical_predictions(tmp_path):
    cpu = [rec("i%d" % i, "c%d" % (i // 5), "dev", True) for i in range(100)]
    gpu = [dict(r) for r in cpu]
    for g in gpu[:2]:                              # two right answers become wrong on the GPU
        g["predicted"], g["correct"] = "b", False
    write(tmp_path, "native.none.cpu.L512", cpu)
    write(tmp_path, "native.none.gpu.L512", gpu)
    r = S.parity(tmp_path)["rows"][0]
    assert r["same_prediction_share"] == 0.98 and r["accuracy_difference"] == pytest.approx(-0.02)
    assert r["passed"] is False                    # identical share is enough, but accuracy moved 2 points (limit 0.5)


def test_parity_is_undetermined_without_a_cpu_twin_or_items(tmp_path):
    write(tmp_path, "native.none.gpu.L512", [rec("i%d" % i, "c%d" % i, "dev", True) for i in range(10)])
    assert S.parity(tmp_path) == {"criteria": {"min_same_prediction_share": 0.98, "max_accuracy_difference": 0.005},
                                  "rows": [], "passed": None}


def test_summary_carries_the_parity_block_and_both_devices_as_separate_cells(tmp_path):
    _pair(tmp_path)
    summ = S.build_summary(tmp_path, n_boot=100)
    devices = {c["condition_id"]: c["device"] for c in summ["cells"]}
    assert devices == {"native.none.cpu.L512": "cpu", "native.none.gpu.L512": "gpu"}
    assert summ["parity"]["passed"] is True
