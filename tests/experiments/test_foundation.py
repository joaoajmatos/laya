"""Phase 2 checks beyond T004/T007: the offline fixture (T006), load_agent (T009), the CLI (T010)."""
import pytest

from experiments import cli, runner


def test_fixture_loads_offline_and_predicts(tiny_agent):
    assert tiny_agent.device.type == "cpu"
    assert tiny_agent.cfg["laya_sparse_fixture"] is True
    out = tiny_agent.predict(
        "the customer asked for a refund on a late delivery",
        {"q": {"type": "choice", "instructions": "which option is best",
               "criteria": {"a": "refund", "b": "delivery"}}},
    )
    probs = out["answers"]["q"]["probabilities"]
    assert set(probs) == {"a", "b"}
    assert sum(probs.values()) == pytest.approx(1.0, abs=1e-3)


def test_fixture_has_alternating_attention(tiny_agent):
    ecfg = tiny_agent.model.encoder.config
    assert ecfg.num_hidden_layers == 3
    assert ecfg.max_position_embeddings == 2048


def test_load_agent_cpu_threads_and_local_revision(tiny_checkpoint):
    import torch
    before = torch.get_num_threads()
    try:
        agent, info = runner.load_agent(tiny_checkpoint, revision="reviewed", threads=1)
        assert agent.device.type == "cpu"
        assert info["threads"] == 1
        assert info["revision"] == "local" and info["revision_source"] == "local"
        assert info["load_seconds"] > 0
        assert info["cpu_fallback_count"] == 0 and info["fallback_messages"] == []
        assert info["dtype"] == "float32"
    finally:
        torch.set_num_threads(before)


def test_reviewed_revision_resolves_through_pinned_table():
    from laya.revisions import PINNED_REVISIONS
    rev, src = runner.resolve_model_revision("convaiinnovations/laya", "reviewed")
    assert rev == PINNED_REVISIONS["convaiinnovations/laya"] and src == "reviewed"
    with pytest.raises(ValueError):
        runner.resolve_model_revision("someone/unknown-model", "reviewed")
    assert runner.resolve_model_revision("someone/unknown-model", None) == (None, "hub_default")


def test_cli_help_and_missing_command(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["--help"])
    assert exc.value.code == 0
    assert "python -m experiments" in capsys.readouterr().out
    assert cli.main([]) == cli.EXIT_USAGE


def test_cli_registry_logs_command_and_maps_exit_codes(tmp_path, monkeypatch):
    from experiments import results
    monkeypatch.setattr(results, "RESULTS_ROOT", tmp_path)
    saved = dict(cli.COMMANDS)
    try:
        cli.COMMANDS.clear()
        seen = {}

        @cli.command("probe", "test command")
        def _probe(args):
            seen.update(threads=args.threads, source=args.threads_source, run_id=args.run_id)

        @cli.command("broken", "fails as a tool error")
        def _broken(args):
            raise cli.ToolError("device is not CPU")

        assert cli.main(["probe", "--run-id", "t1", "--threads", "2"]) == 0
        assert seen == {"threads": 2, "source": "user", "run_id": "t1"}
        assert cli.main(["probe", "--run-id", "t1"]) == 0
        assert seen["source"] == "default" and seen["threads"] >= 1
        lines = (tmp_path / "t1" / "commands.txt").read_text().splitlines()
        assert len(lines) == 2 and "probe" in lines[0]
        assert cli.main(["broken", "--run-id", "t2"]) == cli.EXIT_TOOL_ERROR
        with pytest.raises(ValueError):
            cli.command("probe", "dup")(lambda a: None)
    finally:
        cli.COMMANDS.clear()
        cli.COMMANDS.update(saved)


def test_model_that_cannot_load_is_a_tool_error(tmp_path, monkeypatch):
    from experiments import results
    monkeypatch.setattr(results, "RESULTS_ROOT", tmp_path)
    missing = str(tmp_path / "no-such-checkpoint")
    for cmd in (["sweep", "--lengths", "128", "--questions", "1", "--batch-sizes", "1", "--repeats", "1"],
                ["profile", "--lengths", "128", "--repeats", "1", "--overhead-length", "0"]):
        code = cli.main(cmd + ["--run-id", "bad", "--model", missing, "--threads", "1"])
        assert code == cli.EXIT_TOOL_ERROR


def test_run_keeps_one_model_and_revision(tmp_path, monkeypatch, capsys):
    from experiments import results
    monkeypatch.setattr(results, "RESULTS_ROOT", tmp_path)
    run = results.run_dir("r", root=tmp_path)
    sha = "55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851"
    results.write_json(run, "manifest.json", {"model": {"id": "convaiinnovations/laya", "revision": sha,
                                                         "revision_source": "reviewed"}})
    seen = {}
    saved = dict(cli.COMMANDS)
    try:
        cli.COMMANDS.clear()

        @cli.command("probe", "t")
        def _probe(args):
            seen["revision"] = args.revision

        assert cli.main(["probe", "--run-id", "r", "--threads", "1"]) == 0
        assert seen["revision"] == sha  # inherited, so offline runs find the cached snapshot
        assert cli.main(["probe", "--run-id", "r", "--threads", "1", "--revision", "reviewed"]) == 0
        assert cli.main(["probe", "--run-id", "r", "--threads", "1", "--revision", "main"]) == cli.EXIT_TOOL_ERROR
        assert cli.main(["probe", "--run-id", "r", "--threads", "1", "--model", "other/model"]) == cli.EXIT_TOOL_ERROR
    finally:
        cli.COMMANDS.clear()
        cli.COMMANDS.update(saved)
