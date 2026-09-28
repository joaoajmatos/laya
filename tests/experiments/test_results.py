"""T004: result files, command log and timing summaries (experiments/results.py)."""
import json

import pytest

from experiments import results as R


def test_run_dir_is_created_under_root(tmp_path):
    d = R.run_dir("smoke", root=tmp_path)
    assert d == tmp_path / "smoke"
    assert d.is_dir()
    # reusing the id reuses the directory
    assert R.run_dir("smoke", root=tmp_path) == d


def test_default_run_id_is_a_timestamp(tmp_path):
    d = R.run_dir(root=tmp_path)
    assert d.parent == tmp_path
    assert d.name.endswith("Z") and "T" in d.name


@pytest.mark.parametrize("bad", ["../escape", "a/b", "a\\b", "", ".hidden", "x" * 200])
def test_unsafe_run_ids_are_rejected(tmp_path, bad):
    with pytest.raises(ValueError):
        R.run_dir(bad, root=tmp_path)


def test_every_file_carries_schema_version_and_run_id(tmp_path):
    d = R.run_dir("r1", root=tmp_path)
    R.write_json(d, "manifest.json", {"device": {"requested": "cpu"}})
    R.write_json(d, "sweep.json", [{"status": "measured", "x": 1}])
    for name in ("manifest.json", "sweep.json"):
        body = json.loads((d / name).read_text(encoding="utf-8"))
        assert body["schema_version"] == 1
        assert isinstance(body["schema_version"], int)
        assert body["run_id"] == "r1"
    assert json.loads((d / "sweep.json").read_text())["items"][0]["x"] == 1


def test_payload_cannot_override_contract_fields(tmp_path):
    d = R.run_dir("r2", root=tmp_path)
    R.write_json(d, "a.json", {"schema_version": 99, "run_id": "other", "k": 1})
    body = R.read_json(d, "a.json")
    assert body["schema_version"] == R.SCHEMA_VERSION and body["run_id"] == "r2"


def test_commands_are_appended_not_overwritten(tmp_path):
    d = R.run_dir("r3", root=tmp_path)
    R.append_command(d, ["manifest", "--run-id", "r3"])
    R.append_command(d, ["audit", "--lengths", "512,4096"])
    lines = (d / "commands.txt").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert "manifest" in lines[0] and "audit" in lines[1]
    assert lines[0].startswith("python -m experiments")


def test_summarize_fields_and_values():
    s = R.summarize([1.0, 2.0, 3.0, 4.0, 5.0])
    assert set(s) >= {"min", "p50", "p95", "mean", "std", "n", "low_sample_p95"}
    assert s["min"] == 1.0 and s["p50"] == 3.0 and s["mean"] == 3.0 and s["n"] == 5
    assert s["p95"] == pytest.approx(4.8)  # linear interpolation, numpy default
    assert s["std"] == pytest.approx(1.5811388, rel=1e-6)  # sample std


def test_low_sample_p95_flag():
    assert R.summarize([1.0] * 19)["low_sample_p95"] is True
    assert R.summarize([1.0] * 20)["low_sample_p95"] is False


def test_summarize_rejects_empty_and_non_finite():
    with pytest.raises(ValueError):
        R.summarize([])
    with pytest.raises(ValueError):
        R.summarize([1.0, float("nan")])


def test_unknown_status_is_rejected(tmp_path):
    d = R.run_dir("r4", root=tmp_path)
    with pytest.raises(R.ResultValidationError):
        R.write_json(d, "sweep.json", [{"status": "skipped", "reason": "x"}])
    assert not (d / "sweep.json").exists()


def test_status_enum_has_exactly_four_values():
    assert {s.value for s in R.Status} == {"measured", "unsupported", "failed", "partial"}
    R.validate_item({"status": R.Status.MEASURED})


@pytest.mark.parametrize("status", ["unsupported", "failed", "partial"])
def test_non_measured_item_needs_reason(tmp_path, status):
    d = R.run_dir("r5", root=tmp_path)
    with pytest.raises(R.ResultValidationError):
        R.write_json(d, "sweep.json", [{"status": status}])
    with pytest.raises(R.ResultValidationError):
        R.write_json(d, "sweep.json", [{"status": status, "reason": "  "}])
    R.write_json(d, "sweep.json", [{"status": status, "reason": "above positional capacity"}])


def test_top_level_status_is_validated(tmp_path):
    d = R.run_dir("r6", root=tmp_path)
    with pytest.raises(R.ResultValidationError):
        R.write_json(d, "floor.json", {"status": "failed"})


def test_nan_is_written_as_null(tmp_path):
    d = R.run_dir("r7", root=tmp_path)
    R.write_json(d, "x.json", {"v": float("nan")})
    assert R.read_json(d, "x.json")["v"] is None
