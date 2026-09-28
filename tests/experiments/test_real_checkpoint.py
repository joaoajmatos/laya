"""T020 (slow): audit of the pinned English checkpoint. Needs network once, then the HF cache."""
import pytest

from experiments import audit as A
from experiments import cli, results

pytestmark = pytest.mark.slow


def test_audit_of_pinned_checkpoint(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(results, "RESULTS_ROOT", tmp_path)
    code = cli.main(["audit", "--run-id", "real", "--revision", "reviewed",
                     "--lengths", "512,4096,8192,16384"])
    assert code == 0
    run = tmp_path / "real"
    man = results.read_json(run, "manifest.json")
    aud = results.read_json(run, "audit.json")
    from laya.revisions import PINNED_REVISIONS
    assert man["model"]["revision"] == PINNED_REVISIONS["convaiinnovations/laya"]
    assert man["device"]["effective"] == "cpu" and man["fixture_model"] is False
    assert aud["layers"] and {l["attention_type"] for l in aud["layers"]} <= {"local", "global"}
    # Differences from docs/research-plan.md are recorded, never a failure.
    assert set(aud["research_plan_comparison"]) == set(A.RESEARCH_PLAN_EXPECTED)
    assert set(aud["discrepancies"]) == {k for k, v in aud["research_plan_comparison"].items() if not v["matches"]}
    over = [l for l in aud["lengths"] if l["length"] > aud["positional"]["max_position_embeddings"]]
    assert all(l["status"] == "unsupported" for l in over)
    print(A.format_schedule(aud))
