"""phase2-all: how each step receives the options given to the pipeline (experiments/cli.py)."""
import pytest

from experiments import audit_items, cli, families, results


@pytest.fixture
def pipeline(monkeypatch, tmp_path):
    """Run `phase2-all` with every step replaced by a recorder; returns the namespaces the steps saw."""
    cli._load_commands()
    seen = []
    for name, cmd in cli.COMMANDS.items():
        monkeypatch.setattr(cmd, "run", lambda ns, _n=name: (seen.append(ns), 0)[1])
    monkeypatch.setattr(results, "RESULTS_ROOT", tmp_path / "results")
    monkeypatch.setattr(audit_items, "require_audit", lambda *a, **k: {})
    monkeypatch.setattr(families, "current_families_id", lambda *a, **k: "fid")
    monkeypatch.setattr(cli, "_gpu_python", lambda: None)             # stop at the first GPU step

    def run(*argv):
        parser = cli.build_parser()
        args = parser.parse_args(["phase2-all"] + list(argv))
        cli.resolve_common(args)
        with pytest.raises(cli.ToolError, match="venv-gpu"):
            cli._cmd_phase2_all(args)
        return seen
    return run


def test_steps_receive_the_seed_threads_and_model_given_to_phase2_all(pipeline):
    seen = pipeline("--run-id", "p2", "--seed", "7", "--threads", "3", "--model", "some/model")
    assert seen and {ns.seed for ns in seen} == {7}
    assert {ns.threads for ns in seen} == {3}
    assert {ns.model for ns in seen} == {"some/model"}
