"""T010: data import, fingerprint, low-confidence cutoff and splits (experiments/data.py), offline."""
import json
import os
import shutil

import numpy as np
import pytest

from experiments import data as D
from experiments.results import Refusal
from tests.experiments.conftest import (TINY_TEST_PER_WORKFLOW, TINY_TRAIN_PER_WORKFLOW, UPSTREAM_WORKFLOWS,
                                        write_tiny_upstream)


@pytest.fixture()
def imported(tiny_upstream, tmp_path):
    root = tmp_path / "data"
    man = D.import_dataset(root=root, download=tiny_upstream.download, with_checkpoints=False)
    return root, man


def test_manifest_records_source_counts_and_fingerprint(imported):
    root, man = imported
    assert man["source"] == "LocalLLaMA/typed-decisions" and man["license"] == "apache-2.0"
    assert man["revision"] == D.DATASET_REVISION
    assert man["splits"]["train"]["cases"] == 4 * TINY_TRAIN_PER_WORKFLOW
    assert man["splits"]["test"]["cases"] == 4 * TINY_TEST_PER_WORKFLOW
    assert man["splits"]["test"]["questions"] == 5 * 4 * TINY_TEST_PER_WORKFLOW
    assert man["splits"]["test"]["questions_by_type"] == {"choice": 80, "noul": 40, "score": 80}
    assert man["workflows"]["test"] == {wf: TINY_TEST_PER_WORKFLOW for wf in UPSTREAM_WORKFLOWS}
    assert set(man["fingerprint"]) == {"train", "test", "combined"}
    assert all(len(v) == 64 for v in man["fingerprint"].values())
    assert (root / "manifest.json").exists()


def test_fingerprint_is_deterministic_and_independent_of_row_order(tiny_upstream):
    rows = [D.parse_row(r) for r in tiny_upstream.rows["test"]]
    assert D.fingerprint_rows(rows) == D.fingerprint_rows(list(reversed(rows)))
    changed = [dict(r) for r in rows]
    changed[0] = dict(changed[0], state={"task": "different"})
    assert D.fingerprint_rows(changed) != D.fingerprint_rows(rows)


def test_reimport_matches_and_a_modified_cache_is_refused(imported, tiny_upstream):
    root, man = imported
    again = D.import_dataset(root=root, download=tiny_upstream.download, with_checkpoints=False)
    assert again["fingerprint"] == man["fingerprint"]
    # Tamper with the cached test parquet: one state changes.
    import pyarrow as pa
    import pyarrow.parquet as pq
    target = "agent_trace_observability_000000"
    for cfg in ("all", "agent_trace_observability"):
        path = root / "cache" / cfg / "test-00000-of-00001.parquet"
        rows = pq.read_table(str(path)).to_pylist()
        for r in rows:
            if r["id"] == target:
                r["state"] = json.dumps({"task": "tampered"})
        pq.write_table(pa.Table.from_pylist(rows), str(path))
    with pytest.raises(Refusal) as err:
        D.import_dataset(root=root, download=tiny_upstream.download, with_checkpoints=False)
    assert err.value.reason == "fingerprint_mismatch"


def test_per_workflow_configs_are_cross_checked(tiny_upstream, tmp_path):
    src = tmp_path / "src"
    write_tiny_upstream(str(src))
    import pyarrow as pa
    import pyarrow.parquet as pq
    path = src / "invoice_processing" / "test-00000-of-00001.parquet"
    rows = pq.read_table(str(path)).to_pylist()
    rows[0]["state"] = json.dumps({"task": "disagree"})
    pq.write_table(pa.Table.from_pylist(rows), str(path))
    with pytest.raises(D.DataError):
        D.import_dataset(root=tmp_path / "d", download=lambda rel: str(src / rel), with_checkpoints=False)


def test_low_confidence_cutoff_is_train_bottom_quartile(imported):
    root, man = imported
    conf = [float(q["confidence"]) for r in D.load_cases("train", root) for q in r["gold"].values()]
    lc = man["low_confidence"]
    assert lc["basis"] == "train" and lc["n_questions"] == len(conf)
    assert lc["cutoff"] == pytest.approx(float(np.percentile(conf, 25)))
    cases = D.load_cases("test", root)
    for c in cases:
        for qid, m in c["qmeta"].items():
            assert m["low_confidence"] == (float(c["gold"][qid]["confidence"]) <= lc["cutoff"])
    share = np.mean([m["low_confidence"] for c in cases for m in c["qmeta"].values()])
    assert 0.05 < share < 0.6                        # roughly a quartile of the test questions


def test_extra_agreement_strata_and_metadata(imported):
    root, _ = imported
    c = D.load_cases("test", root)[0]
    m = c["qmeta"]["action"]
    assert m["type"] == "choice" and m["n_options"] == 3
    assert {"argmax_agree", "tv_top_quartile", "low_confidence"} <= set(m)
    assert c["qmeta"]["risk"]["n_options"] == 3 and c["qmeta"]["needs_review"]["n_options"] == 2


def test_factors_never_reach_the_model_input(imported):
    root, _ = imported
    for it in D.original_items("dev", root=root, cases=D.load_cases("test", root)[:3]):
        assert "seed" not in json.loads(it["state_text"])          # the fixture's only `factors` key


def test_nothing_is_written_outside_the_data_root(tiny_upstream, tmp_path):
    before = set(os.listdir(tmp_path))
    root = tmp_path / "only-here"
    D.import_dataset(root=root, download=tiny_upstream.download, with_checkpoints=False)
    assert set(os.listdir(tmp_path)) == before | {"only-here"}


# --------------------------------------------------------------------------- splits

def test_splits_are_stratified_disjoint_and_cover_the_test_split(imported):
    root, _ = imported
    sp = D.make_splits(seed=7, root=root)
    dev, cal, fin = (set(sp["splits"][s]["case_ids"]) for s in D.EVAL_SPLITS)
    assert not (dev & cal) and not (dev & fin) and not (cal & fin)
    all_test = {c["id"] for c in D.load_cases("test", root)}
    assert dev | cal | fin == all_test
    # 10 cases per workflow -> 3 / 2 / 5 (30% / 20% / 50%)
    for s, n in (("dev", 3), ("calibration", 2), ("final", 5)):
        assert sp["splits"][s]["n_per_workflow"] == {wf: n for wf in UPSTREAM_WORKFLOWS}
        assert sp["splits"][s]["n_cases"] == 4 * n
    assert sp["rule"] == "stratified_by_workflow_30_20_50"
    assert all(len(sp["splits"][s]["fingerprint"]) == 64 for s in D.EVAL_SPLITS)


def test_full_size_split_counts():
    cases = [{"id": "%s_%03d" % (wf, i), "workflow": wf, "questions": {"q1": {}, "q2": {}}}
             for wf in UPSTREAM_WORKFLOWS for i in range(100)]
    sp = D.build_splits(cases, 1, "f" * 64)
    assert [sp["splits"][s]["n_cases"] for s in D.EVAL_SPLITS] == [120, 80, 200]
    assert sp["splits"]["dev"]["n_per_workflow"] == {wf: 30 for wf in UPSTREAM_WORKFLOWS}
    assert [len(sp["half_sample"][s]) for s in D.EVAL_SPLITS] == [60, 40, 100]
    assert len(sp["latency_sample"]) == 12 and len(sp["variant_sample"]) == 20
    per_wf = {}
    for it in sp["latency_sample"]:
        per_wf[it["case_id"].rsplit("_", 1)[0]] = per_wf.get(it["case_id"].rsplit("_", 1)[0], 0) + 1
    assert set(per_wf.values()) == {3}


def test_splits_reproduce_and_refuse_to_be_overwritten_by_a_different_result(imported):
    root, _ = imported
    first = D.make_splits(seed=7, root=root)
    again = D.make_splits(seed=7, root=root)
    assert again == first
    with pytest.raises(Refusal):
        D.make_splits(seed=8, root=root)
    assert D.check_splits(root)["ok"] is True


def test_samples_are_subsets_stratified_by_workflow(imported):
    root, _ = imported
    sp = D.make_splits(seed=7, root=root)
    for s in D.EVAL_SPLITS:
        assert set(sp["half_sample"][s]) <= set(sp["splits"][s]["case_ids"])
        wf = [i.rsplit("_", 1)[0] for i in sp["half_sample"][s]]
        assert len(set(wf)) == 4
    assert set(sp["variant_sample"]) <= set(sp["half_sample"]["dev"])
    assert {i["case_id"] for i in sp["latency_sample"]} <= set(sp["half_sample"]["dev"])


def test_split_cases_and_original_items(imported):
    root, _ = imported
    D.make_splits(seed=7, root=root)
    dev = D.split_cases("dev", root)
    assert len(dev) == 12
    items = D.original_items("dev", root=root)
    assert len(items) == 12 * 5
    it = items[0]
    assert it["variant"] == "original" and it["length"] == "original" and it["split"] == "dev"
    assert it["status"] == "ok" and it["option_order"] is None
    assert json.loads(it["state_text"]) == dev[0]["state"]
    assert it["gold"]["label"] == dev[0]["gold"][it["question_id"]]["label"]
    with pytest.raises(ValueError):
        D.split_cases("train", root)
