"""End-to-end tests of the Phase 2 CLI commands (experiments/cli.py, specs/002 contracts/cli.md).

Every test calls ``cli.main([...])`` on the tiny fixture checkpoint and the synthetic upstream dataset. The results
root and the data root are temp directories, the dataset download is the fixture's local copy, and measuring
children run in-process (one test uses a real subprocess) so nothing needs the network, a dataset or a GPU.

Timings and accuracies here exercise the tooling only; they are never measurements.
"""
import contextlib
import importlib
import io
import json
import shutil

import pytest

from experiments import cli, data, evalrun, families, results, runner

REAL_RUN_CONDITION = runner.run_condition

LENGTHS = "256,512,1024"
VARIANTS = "neutral@mid,distractor@begin,distractor@mid,distractor@end"
MAX_CASES = "3"


def in_process(function, spec, time_cap=None, grace=None):
    """Stand-in for `runner.run_condition`: the same child function, no subprocess."""
    module, name = function.split(":")
    return getattr(importlib.import_module(module), name)(dict(spec, time_cap=spec.get("time_cap", time_cap)))


class Env:
    """One results root, one data root, one model: a CLI session."""

    def __init__(self, ckpt, results_root, data_root):
        self.ckpt, self.results_root, self.data_root = ckpt, results_root, data_root

    def run(self, name, *extra, data_arg=True, run_id=None):
        argv = [name, "--model", str(self.ckpt), "--threads", "1"]
        if data_arg:
            argv += ["--data-root", str(self.data_root)]
        if run_id:
            argv += ["--run-id", run_id]
        argv += [str(a) for a in extra]
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = cli.main(argv)
        return Result(code, out.getvalue(), err.getvalue())

    def run_dir(self, run_id):
        return self.results_root / run_id

    def fork(self, tmp_path):
        """A copy of this session's data in a fresh location (for tests that change state). The results root
        is the module's patched one; tests pick their own run ids."""
        root = tmp_path / "data"
        shutil.copytree(self.data_root, root)
        return Env(self.ckpt, self.results_root, root)


class Result:
    def __init__(self, code, out, err):
        self.code, self.out, self.err = code, out, err

    def __repr__(self):
        return "Result(code=%s, out=%r, err=%r)" % (self.code, self.out[-400:], self.err[-400:])


@pytest.fixture(scope="module")
def patched(tiny_upstream, tmp_path_factory):
    """Results root, dataset download and measuring children redirected for the whole module."""
    mp = pytest.MonkeyPatch()
    res = tmp_path_factory.mktemp("cli-results")
    mp.setattr(results, "RESULTS_ROOT", res)
    mp.setattr(data, "_hf_download", lambda revision: tiny_upstream.download)
    mp.setattr(runner, "run_condition", in_process)
    yield res
    mp.undo()


def _new_env(ckpt, patched, tmp_path_factory, name):
    root = tmp_path_factory.mktemp(name + "-data")
    return Env(ckpt, patched, root)


def _write_draft(env, tmp_dir, **patch):
    sheet = json.loads((env.data_root / "audit_sheet.json").read_text(encoding="utf-8"))
    rows = [{"item_id": e["item_id"], "changes_answer": False, "ambiguous": False, "evidence_intact": True, "note": ""}
            for e in sheet["items"]]
    if patch:
        rows[0].update(patch)
    path = tmp_dir / "audit_result_draft.json"
    path.write_text(json.dumps({"items": rows, "auditor": "test fixture", "method": "synthetic"}), encoding="utf-8")
    return path


@pytest.fixture(scope="module")
def built(tiny_checkpoint, patched, tmp_path_factory):
    """data-import, splits, length-profile, families (all three splits) and audit-sample, through the CLI."""
    env = _new_env(tiny_checkpoint, patched, tmp_path_factory, "built")
    steps = {"data-import": env.run("data-import"),
             "splits": env.run("splits", "--split-seed", 3),
             "length-profile": env.run("length-profile")}
    for split in data.EVAL_SPLITS:
        steps["families-" + split] = env.run("families", "--split", split, "--lengths", LENGTHS, "--variants", VARIANTS)
    steps["audit-sample"] = env.run("audit-sample", "--n", 50)
    return env, steps


@pytest.fixture(scope="module")
def audited(built, tmp_path_factory):
    env, _ = built
    env = env.fork(tmp_path_factory.mktemp("audited"))
    rec = env.run("audit-record", "--from", _write_draft(env, env.data_root))
    return env, rec


@pytest.fixture(scope="module")
def evaluated(audited):
    """The whole evaluation chain on dev and calibration through the CLI, ending at the report."""
    env, _ = audited
    rid = "p2"
    common = ["--conditions", "native,trunc512,window", "--lengths", "256,512", "--max-cases", MAX_CASES]
    steps = {"tune": env.run("eval", "--tune", "--run-id", rid, "--lengths", "256")}
    for split in ("dev", "calibration"):
        steps["eval-" + split] = env.run("eval", "--split", split, *common, run_id=rid)
    steps["calibrate"] = env.run("calibrate", run_id=rid, data_arg=False)
    steps["latency"] = env.run("latency", "--conditions", "native,trunc512", "--lengths", "512", "--repeats", 1,
                               "--warmup", 0, run_id=rid)
    steps["eval-summary"] = env.run("eval-summary", "--n-boot", 50, run_id=rid, data_arg=False)
    steps["freeze-plan"] = env.run("freeze-plan", run_id=rid, data_arg=False)
    steps["phase2-report"] = env.run("phase2-report", run_id=rid)
    return env, steps, env.run_dir(rid)


# --------------------------------------------------------------------------- data commands

def test_data_import_writes_the_manifest_and_prints_the_fingerprint(built):
    env, steps = built
    r = steps["data-import"]
    assert r.code == 0, r
    man = json.loads((env.data_root / "manifest.json").read_text(encoding="utf-8"))
    assert man["revision"] == data.DATASET_REVISION and man["source"] == data.DATASET_ID
    assert man["splits"]["train"]["cases"] == 48 and man["splits"]["test"]["cases"] == 40
    assert "fingerprint %s" % man["fingerprint"]["combined"] in r.out
    assert "low-confidence cutoff" in r.out and str(env.data_root / "manifest.json") in r.out
    assert (env.data_root / "cache" / data.DATA_FILES[0]).exists()


def test_every_command_appends_its_command_line_to_commands_txt(built):
    env, _ = built
    text = (env.run_dir("data") / "commands.txt").read_text(encoding="utf-8")
    for name in ("data-import", "splits", "length-profile", "families", "audit-sample"):
        assert "python -m experiments %s" % name in text


def test_data_import_is_repeatable_and_refuses_a_changed_fingerprint(built, tmp_path):
    env, _ = built
    fork = env.fork(tmp_path)
    again = fork.run("data-import")
    assert again.code == 0, again                          # served from the cache, same fingerprint
    man_path = fork.data_root / "manifest.json"
    man = json.loads(man_path.read_text(encoding="utf-8"))
    man["fingerprint"]["combined"] = "0" * 64
    man_path.write_text(json.dumps(man), encoding="utf-8")
    bad = fork.run("data-import")
    assert bad.code == cli.EXIT_TOOL_ERROR
    assert "fingerprint_mismatch" in bad.err


def test_data_import_rejects_a_cache_whose_configs_disagree(built, tmp_path):
    import pyarrow.parquet as pq
    fork = built[0].fork(tmp_path)
    path = fork.data_root / "cache" / "all" / "test-00000-of-00001.parquet"
    table = pq.read_table(str(path), memory_map=False)
    pq.write_table(table.slice(0, table.num_rows - 1), str(path))
    r = fork.run("data-import")
    assert r.code == cli.EXIT_TOOL_ERROR and r.err.startswith("error:")


def test_splits_writes_the_manifest_and_check_verifies_it(built, tmp_path):
    env, steps = built
    r = steps["splits"]
    assert r.code == 0, r
    sp = json.loads((env.data_root / "splits.json").read_text(encoding="utf-8"))
    assert sp["seed"] == 3 and set(sp["splits"]) == set(data.EVAL_SPLITS)
    for s in data.EVAL_SPLITS:
        assert "%s: %d cases" % (s, sp["splits"][s]["n_cases"]) in r.out
    fork = env.fork(tmp_path)
    ok = fork.run("splits", "--check")
    assert ok.code == 0 and "'ok': True" in ok.out
    assert fork.run("splits", "--split-seed", 3).code == 0          # identical result is not an overwrite


def test_splits_refuse_a_different_seed_and_a_manifest_that_does_not_match(built, tmp_path):
    env, _ = built
    fork = env.fork(tmp_path)
    other = fork.run("splits", "--split-seed", 4)
    assert other.code == cli.EXIT_TOOL_ERROR and "fingerprint_mismatch" in other.err
    path = fork.data_root / "splits.json"
    sp = json.loads(path.read_text(encoding="utf-8"))
    sp["splits"]["dev"]["case_ids"] = list(reversed(sp["splits"]["dev"]["case_ids"]))
    path.write_text(json.dumps(sp), encoding="utf-8")
    bad = fork.run("splits", "--check")
    assert bad.code == cli.EXIT_TOOL_ERROR and "fingerprint_mismatch" in bad.err


def test_splits_without_an_import_is_a_tool_error(tiny_checkpoint, patched, tmp_path):
    env = Env(tiny_checkpoint, patched, tmp_path / "empty")
    for argv in (["splits"], ["splits", "--check"], ["length-profile"]):
        r = env.run(*argv)
        assert r.code == cli.EXIT_TOOL_ERROR and r.err.startswith("error:"), r


def test_length_profile_covers_both_upstream_splits(built):
    env, steps = built
    r = steps["length-profile"]
    assert r.code == 0, r
    prof = json.loads((env.data_root / "length_profile.json").read_text(encoding="utf-8"))
    assert set(prof["by_split"]) == {"train", "test"} and prof["model"] == str(env.ckpt)
    assert prof["by_split"]["test"]["overall"]["rows"]["max"] >= prof["by_split"]["test"]["overall"]["rows"]["min"]
    assert "test rows:" in r.out and "share of test cases within each length" in r.out


# --------------------------------------------------------------------------- families and the audit

def test_families_build_items_for_every_split_and_fingerprint_final(built):
    env, steps = built
    fam = None
    for split in data.EVAL_SPLITS:
        r = steps["families-" + split]
        assert r.code == 0, r
        assert "split %s:" % split in r.out and "fingerprint" in r.out
        fam = fam or r.out.split("families ")[1].split(",")[0]
    base = env.data_root / "items" / fam
    assert {p.name for p in base.glob("*.jsonl.gz")} == {"dev.jsonl.gz", "calibration.jsonl.gz", "final.jsonl.gz"}
    meta = json.loads((base / "families.json").read_text(encoding="utf-8"))
    assert set(meta["splits"]) == set(data.EVAL_SPLITS) and meta["lengths"] == [256, 512, 1024]


def test_families_reject_unknown_variants_and_a_missing_import(built, tiny_checkpoint, patched, tmp_path):
    env, _ = built
    bad = env.run("families", "--split", "dev", "--variants", "nonsense")
    assert bad.code == cli.EXIT_TOOL_ERROR and "unknown variants" in bad.err
    empty = Env(tiny_checkpoint, patched, tmp_path / "none")
    assert empty.run("families", "--split", "dev").code == cli.EXIT_TOOL_ERROR


def test_bad_arguments_exit_with_a_usage_error(built):
    env, _ = built
    with pytest.raises(SystemExit) as err:
        env.run("families", "--split", "everything")
    assert err.value.code == cli.EXIT_USAGE


def test_audit_sample_writes_the_sheet_and_says_what_to_do_next(built):
    env, steps = built
    r = steps["audit-sample"]
    assert r.code == 0, r
    sheet = json.loads((env.data_root / "audit_sheet.json").read_text(encoding="utf-8"))
    assert sheet["n"] == 50 and len(sheet["items"]) == 50
    assert (env.data_root / "audit_sheet.md").exists()
    assert "audit-record --from" in r.out and "audit sheet for families %s: 50 items" % sheet["families_id"] in r.out


def test_audit_sample_refuses_fewer_than_fifty_items(built):
    env, _ = built
    r = env.run("audit-sample", "--n", 49)
    assert r.code == cli.EXIT_TOOL_ERROR and "at least 50" in r.err


def test_audit_sample_needs_built_families(tiny_checkpoint, patched, tmp_path):
    r = Env(tiny_checkpoint, patched, tmp_path / "x").run("audit-sample")
    assert r.code == cli.EXIT_TOOL_ERROR and "families" in r.err


def test_audit_record_passes_a_clean_draft_and_binds_it_to_the_families(audited, built):
    env, r = audited
    assert r.code == 0, r
    assert "answer changed 0.0%" in r.out and "PASSED" in r.out and "NOT PASSED" not in r.out
    res = json.loads((env.data_root / "audit_result.json").read_text(encoding="utf-8"))
    sheet = json.loads((env.data_root / "audit_sheet.json").read_text(encoding="utf-8"))
    assert res["passed"] and res["families_id"] == sheet["families_id"] and res["n_audited"] == 50
    assert res["auditor"] == "test fixture"


def test_audit_record_reports_a_failed_audit_without_a_tool_error(built, tmp_path):
    env = built[0].fork(tmp_path)
    r = env.run("audit-record", "--from", _write_draft(env, tmp_path, changes_answer=True))
    assert r.code == 0 and "NOT PASSED" in r.out
    assert json.loads((env.data_root / "audit_result.json").read_text(encoding="utf-8"))["passed"] is False


def test_audit_record_rejects_a_missing_or_incomplete_draft(built, tmp_path):
    env = built[0].fork(tmp_path)
    assert env.run("audit-record", "--from", str(tmp_path / "nope.json")).code == cli.EXIT_TOOL_ERROR
    short = tmp_path / "short.json"
    short.write_text(json.dumps({"items": []}), encoding="utf-8")
    r = env.run("audit-record", "--from", short)
    assert r.code == cli.EXIT_TOOL_ERROR and "no verdict" in r.err
    typo = tmp_path / "typo.json"
    draft = json.loads(_write_draft(env, tmp_path).read_text(encoding="utf-8"))
    del draft["items"][0]["evidence_intact"]
    typo.write_text(json.dumps(draft), encoding="utf-8")
    assert env.run("audit-record", "--from", typo).code == cli.EXIT_TOOL_ERROR


# --------------------------------------------------------------------------- refusals of the evaluation commands

def test_eval_refuses_the_final_split(audited):
    env, _ = audited
    r = env.run("eval", "--split", "final", run_id="refuse")
    assert r.code == cli.EXIT_TOOL_ERROR and "split_locked" in r.err
    assert not (env.run_dir("refuse") / "quality").exists()


def test_eval_and_tune_require_a_passing_audit(built, tmp_path):
    env = built[0].fork(tmp_path)
    for argv in (["eval", "--split", "dev"], ["eval", "--split", "calibration"], ["eval", "--tune"]):
        r = env.run(*argv, run_id="gate")
        assert r.code == cli.EXIT_TOOL_ERROR and "audit_required" in r.err, (argv, r)
    assert not (env.run_dir("gate") / "quality").exists()
    failed = env.run("audit-record", "--from", _write_draft(env, tmp_path, ambiguous=True))
    assert failed.code == 0
    again = env.run("eval", "--split", "dev", run_id="gate")
    assert again.code == cli.EXIT_TOOL_ERROR and "audit_required" in again.err and "did not pass" in again.err


def test_eval_refuses_families_changed_after_the_audit(audited, tmp_path):
    env = audited[0].fork(tmp_path)
    rebuilt = env.run("families", "--split", "dev", "--lengths", "256", "--variants", "distractor@mid", "--seed", 9)
    assert rebuilt.code == 0, rebuilt
    r = env.run("eval", "--split", "dev", run_id="stale")
    assert r.code == cli.EXIT_TOOL_ERROR and "audit_stale" in r.err
    assert env.run("eval", "--split", "dev", "--families-id", "0123456789ab", run_id="stale").code == cli.EXIT_TOOL_ERROR


def test_eval_without_built_families_is_a_tool_error(tiny_checkpoint, patched, tmp_path):
    r = Env(tiny_checkpoint, patched, tmp_path / "x").run("eval", "--split", "dev", run_id="nofam")
    assert r.code == cli.EXIT_TOOL_ERROR and "no families built" in r.err


def test_eval_rejects_unknown_conditions_and_variants(audited):
    env, _ = audited
    for flag, value, text in (("--conditions", "native,bogus", "unknown conditions"),
                              ("--variants", "bogus", "unknown variants")):
        r = env.run("eval", flag, value, run_id="bad")
        assert r.code == cli.EXIT_TOOL_ERROR and text in r.err


@pytest.mark.skipif(__import__("torch").cuda.is_available(), reason="needs a machine without CUDA")
def test_gpu_scoring_without_cuda_is_refused_as_a_device_mismatch(audited):
    env, _ = audited
    for argv in (["eval", "--split", "dev", "--device", "gpu", "--conditions", "native", "--lengths", "256"],
                 ["eval", "--tune", "--device", "gpu"],
                 ["latency", "--device", "gpu", "--conditions", "native", "--lengths", "512"]):
        r = env.run(*argv, run_id="gpu")
        assert r.code == cli.EXIT_TOOL_ERROR and "device_mismatch" in r.err, (argv, r)
    assert not list(env.run_dir("gpu").glob("gpu_latency/*")) and not (env.run_dir("gpu") / "quality").exists()


# --------------------------------------------------------------------------- the evaluation chain

def test_tune_chooses_the_window_size_on_dev_and_stops(evaluated):
    env, steps, run = evaluated
    r = steps["tune"]
    assert r.code == 0, r
    body = json.loads((run / "conditions.json").read_text(encoding="utf-8"))
    assert body["tuned_on"] == "dev" and body["scored_on"] == "cpu" and set(body["grid"]) == {"default", "256", "512"}
    assert "window tuned on dev (scored on cpu)" in r.out
    tuning = sorted(p.name for p in (run / "tuning").iterdir())            # tuning results live apart from quality/
    assert tuning and all(t.startswith("window") for t in tuning)
    assert not any(p.name in tuning for p in (run / "quality").iterdir())


def test_eval_scores_dev_and_calibration_conditions_and_uses_the_tuned_window(evaluated):
    env, steps, run = evaluated
    for split in ("dev", "calibration"):
        r = steps["eval-" + split]
        assert r.code == 0, r
        assert "6 conditions run, 0 not complete" in r.out
    ids = sorted(p.name for p in (run / "quality").iterdir())
    assert len(ids) == 6 and all(i.endswith(("L256", "L512")) and ".cpu." in i for i in ids)
    tuned = json.loads((run / "conditions.json").read_text(encoding="utf-8"))["window"]["size"]
    windows = [i for i in ids if i.startswith("window")]
    assert windows and all(str(tuned) in i or tuned == "default" for i in windows)
    recs = results.read_jsonl(run / "quality" / ids[0] / "predictions.jsonl")
    assert recs and {r["split"] for r in recs} == {"dev", "calibration"} and all(r["device"] == "cpu" for r in recs)
    assert len({r["case_id"] for r in recs}) <= 2 * int(MAX_CASES)      # the pilot cap, per split
    assert not any(r["split"] == "final" for i in ids for r in results.read_jsonl(run / "quality" / i / "predictions.jsonl"))


def test_eval_resumes_without_repeating_finished_items(evaluated):
    env, _, run = evaluated
    cid = next(p for p in (run / "quality").iterdir() if p.name.startswith("native")).name
    path = run / "quality" / cid / "predictions.jsonl"
    before = path.read_bytes()
    r = env.run("eval", "--split", "dev", "--conditions", "native", "--lengths", "256,512", "--max-cases", MAX_CASES,
                run_id="p2")
    assert r.code == 0 and "2 conditions run, 0 not complete" in r.out
    assert path.read_bytes().startswith(before) and path.read_bytes() == before


def test_eval_variants_run_the_restricted_sample(audited, tmp_path):
    env = audited[0].fork(tmp_path)
    r = env.run("eval", "--variants", "fastpath_off", "--lengths", "512", "--max-cases", 2, run_id="var")
    assert r.code == 0, r
    dirs = [p.name for p in (env.run_dir("var") / "quality").iterdir()]
    assert dirs and all("fastpath_off" in d for d in dirs)
    recs = results.read_jsonl(env.run_dir("var") / "quality" / dirs[0] / "predictions.jsonl")
    assert recs and all(x["status"] == "measured" for x in recs)


def test_calibrate_fits_temperatures_from_calibration_results_only(evaluated):
    _, steps, run = evaluated
    r = steps["calibrate"]
    assert r.code == 0, r
    body = json.loads((run / "calibration.json").read_text(encoding="utf-8"))
    assert body["conditions"] and str(run / "calibration.json") in r.out
    for cid, by_type in body["conditions"].items():
        assert cid in r.out and all(v["temperature"] > 0 for v in by_type.values())


def test_calibrate_with_no_results_says_so(tiny_checkpoint, patched, tmp_path):
    r = Env(tiny_checkpoint, patched, tmp_path).run("calibrate", run_id="empty-cal", data_arg=False)
    assert r.code == 0 and "no calibration-split results yet" in r.out
    assert json.loads((patched / "empty-cal" / "calibration.json").read_text(encoding="utf-8"))["conditions"] == {}


def test_latency_writes_one_cpu_file_per_condition(evaluated):
    _, steps, run = evaluated
    r = steps["latency"]
    assert r.code == 0, r
    files = sorted(p.name for p in (run / "latency").glob("*.json"))
    assert len(files) == 2 and all(f.endswith("L512.json") for f in files)
    assert "2 latency conditions written under" in r.out
    item = json.loads((run / "latency" / files[0]).read_text(encoding="utf-8"))
    assert item["device"] == "cpu" and item["status"] in ("measured", "partial", "failed")
    assert not (run / "gpu_latency").exists()                     # CPU results never land in the GPU directory


def test_eval_summary_prints_the_cells_and_writes_summary_json(evaluated):
    _, steps, run = evaluated
    r = steps["eval-summary"]
    assert r.code == 0, r
    body = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    assert {c["condition_id"] for c in body["cells"]} >= {p.name for p in (run / "quality").iterdir() if ".cpu." in p.name}
    assert all(c["condition_id"] in r.out for c in body["cells"])
    assert "selected retrieval budget on dev" in r.out
    assert all(c["n_items"] > 0 for c in body["cells"] if c["status"] == "measured")


def test_freeze_plan_writes_the_plan_and_refuses_to_overwrite_it(evaluated):
    env, steps, run = evaluated
    r = steps["freeze-plan"]
    assert r.code == 0, r
    plan = json.loads((run / "evaluation_plan.json").read_text(encoding="utf-8"))
    assert plan["version"] == 1 and "plan version 1 frozen" in r.out
    assert (run / "evaluation_plan.md").exists()
    again = env.run("freeze-plan", run_id="p2", data_arg=False)
    assert again.code == cli.EXIT_TOOL_ERROR and "frozen" in again.err
    assert json.loads((run / "evaluation_plan.json").read_text(encoding="utf-8"))["fingerprint"] == plan["fingerprint"]
    no_reason = env.run("freeze-plan", "--new-version", run_id="p2", data_arg=False)
    assert no_reason.code == cli.EXIT_TOOL_ERROR
    v2 = env.run("freeze-plan", "--new-version", "--reason", "test", run_id="p2", data_arg=False)
    assert v2.code == 0 and "plan version 2" in v2.out


def test_freeze_plan_refuses_a_run_that_names_a_final_split_item(evaluated):
    env, _, run = evaluated
    leak = env.run_dir("p2-leak")
    shutil.copytree(run, leak)
    cid = next(p for p in (leak / "quality").iterdir()).name
    path = leak / "quality" / cid / "predictions.jsonl"
    rec = results.read_jsonl(path)[0]
    results.append_jsonl(path, dict(rec, item_id="leak", split="final"))
    for name in ("evaluation_plan.json", "evaluation_plan.md"):
        (leak / name).unlink(missing_ok=True)
    r = env.run("freeze-plan", run_id="p2-leak", data_arg=False)
    assert r.code == cli.EXIT_TOOL_ERROR and "split_locked" in r.err


def test_phase2_report_writes_json_and_markdown(evaluated):
    _, steps, run = evaluated
    r = steps["phase2-report"]
    assert r.code == 0, r
    assert (run / "report2.json").exists() and (run / "report2.md").exists()
    assert str(run / "report2.json") in r.out and str(run / "report2.md") in r.out
    md = (run / "report2.md").read_text(encoding="utf-8")
    assert "[measured]" in md or "[estimated]" in md or "[hypothesized]" in md


def test_phase2_report_on_an_empty_run_states_what_is_missing_or_refuses(tiny_checkpoint, patched, tmp_path):
    r = Env(tiny_checkpoint, patched, tmp_path).run("phase2-report", run_id="empty-report")
    assert r.code in (cli.EXIT_OK, cli.EXIT_TOOL_ERROR)
    if r.code:
        assert r.err.startswith("error:")


def test_the_real_subprocess_path_scores_a_condition(audited, tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "run_condition", REAL_RUN_CONDITION)
    env = audited[0].fork(tmp_path)
    r = env.run("eval", "--split", "dev", "--conditions", "native", "--lengths", "256", "--max-cases", 1, run_id="sub")
    assert r.code == 0 and "1 conditions run, 0 not complete" in r.out, r
    recs = results.read_jsonl(next((env.run_dir("sub") / "quality").iterdir()) / "predictions.jsonl")
    assert recs and all(x["status"] == "measured" for x in recs)


# --------------------------------------------------------------------------- phase2-all

def test_phase2_all_dry_run_prints_the_ordered_plan_and_runs_nothing(tiny_checkpoint, patched, tmp_path):
    env = Env(tiny_checkpoint, patched, tmp_path / "dry")
    r = env.run("phase2-all", "--dry-run", run_id="dry")
    assert r.code == 0, r
    lines = [l for l in r.out.splitlines() if l.startswith("  ") and "." in l[:6]]
    assert len(lines) == 21
    order = [l for l in lines if "audit gate" in l or "audit-sample" in l or "eval --tune" in l or "phase2-report" in l]
    assert [("audit-sample" in l, "audit gate" in l) for l in order[:2]] == [(True, False), (False, True)]
    assert "phase2-report" in lines[-1] and "data-import" in lines[0]
    assert lines.index(next(l for l in lines if "audit gate" in l)) < lines.index(next(l for l in lines if "eval --tune" in l))
    assert "Estimated wall time (estimated" in r.out and ".venv-gpu" in r.out
    assert not (tmp_path / "dry").exists()                       # nothing was imported or written
    assert not (patched / "dry" / "manifest.json").exists()


def test_phase2_all_stops_at_the_audit_gate(tiny_checkpoint, patched, tmp_path, monkeypatch):
    """With the pinned checkpoints redirected to the tiny one, the pipeline runs to the audit gate and stops."""
    monkeypatch.setattr(evalrun, "CHECKPOINTS", {"fine_tuned": str(tiny_checkpoint), "base": str(tiny_checkpoint)})
    monkeypatch.setattr(cli, "_gpu_python", lambda: None)
    monkeypatch.setattr(families, "DEFAULT_LENGTHS", (256, 512))          # the tiny model, and a short test
    env = Env(tiny_checkpoint, patched, tmp_path / "all")
    r = env.run("phase2-all", run_id="all")
    assert r.code == 0, r
    assert "== data-import ==" in r.out and "stopped at the audit gate" in r.out and "audit_required" in r.out
    assert "== eval" not in r.out
    assert (tmp_path / "all" / "audit_sheet.json").exists() and (tmp_path / "all" / "splits.json").exists()


def test_solvability_runs_the_named_checkpoints_and_rejects_unknown_ones(built, monkeypatch):
    env, _ = built
    monkeypatch.setattr(evalrun, "CHECKPOINTS", {"fine_tuned": str(env.ckpt), "base": str(env.ckpt)})
    bad = env.run("solvability", "--models", "nope", run_id="solv")
    assert bad.code == cli.EXIT_TOOL_ERROR and "unknown checkpoint names" in bad.err
    r = env.run("solvability", "--models", "fine_tuned", run_id="solv")
    assert r.code == 0, r
    body = json.loads((env.run_dir("solv") / "solvability.json").read_text(encoding="utf-8"))
    assert "fine_tuned" in body["models"] and body["split"] == "dev"
    assert "reference checkpoint:" in r.out and "fine_tuned: accuracy" in r.out
