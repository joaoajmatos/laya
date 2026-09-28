"""T011: run manifest (experiments/manifest.py)."""
import subprocess

import pytest

from experiments import manifest as M
from experiments.runner import load_agent

REQUIRED = {"run_id", "created_at", "code", "model", "software", "hardware", "runtime",
            "device", "cache_policy", "seeds"}


@pytest.fixture(scope="module")
def loaded(tiny_checkpoint):
    return load_agent(tiny_checkpoint, threads=1)


def test_every_data_model_field_is_present(loaded):
    agent, info = loaded
    m = M.build_manifest(agent, info, run_id="t", seed=7, threads_source="user")
    assert REQUIRED <= set(m)
    assert {"git_sha", "git_dirty", "laya_diff_empty"} <= set(m["code"])
    assert {"id", "revision", "config_sha256"} <= set(m["model"])
    assert len(m["model"]["config_sha256"]) == 64
    assert {"python", "torch", "transformers", "numpy", "safetensors"} <= set(m["software"])
    assert {"cpu_model", "architecture", "physical_cores", "logical_cores", "ram_bytes", "os"} <= set(m["hardware"])
    rt = m["runtime"]
    assert {"intra_op_threads", "inter_op_threads", "threads_source", "hybrid_cores", "torch_build",
            "laya_cpu_amp", "dtype", "amp_enabled", "compile", "concurrency"} <= set(rt)
    assert rt["intra_op_threads"] == 1 and rt["threads_source"] == "user"
    assert m["device"] == {"requested": "cpu", "effective": "cpu", "fallbacks": []}
    assert m["cache_policy"]["document_caches"] == "none"
    assert m["seeds"]["base_seed"] == 7 and "base_seed + r" in m["seeds"]["rule"]
    assert m["non_comparable"] is False
    assert m["fixture_model"] is True


def test_local_path_revision_is_recorded_as_local(loaded):
    agent, info = loaded
    assert info["revision_source"] == "local"
    assert M.build_manifest(agent, info, run_id="t")["model"]["revision"] == "local"


def test_hub_load_needs_a_revision(loaded):
    agent, info = loaded
    hub = dict(info, revision_source="hub_default", revision=None, model="convaiinnovations/laya")
    with pytest.raises(M.ManifestError):
        M.build_manifest(agent, hub, run_id="t")
    hub["revision"] = "55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851"
    assert M.build_manifest(agent, hub, run_id="t")["model"]["revision"] == hub["revision"]


def test_non_cpu_request_marks_run_non_comparable(loaded):
    agent, info = loaded
    m = M.build_manifest(agent, info, run_id="t", requested_device="cuda")
    assert m["non_comparable"] is True and m["non_comparable_reasons"]


def test_laya_cpu_amp_is_recorded(loaded, monkeypatch):
    agent, info = loaded
    monkeypatch.setenv("LAYA_CPU_AMP", "bf16")
    assert M.build_manifest(agent, info, run_id="t")["runtime"]["laya_cpu_amp"] == "bf16"
    monkeypatch.delenv("LAYA_CPU_AMP")
    assert M.build_manifest(agent, info, run_id="t")["runtime"]["laya_cpu_amp"] is None


def _git(cwd, *args):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


def test_laya_diff_empty_on_clean_tree(tmp_path):
    try:
        _git(tmp_path, "init", "-q")
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("git not available")
    (tmp_path / "laya").mkdir()
    (tmp_path / "laya" / "x.py").write_text("a = 1\n")
    (tmp_path / "other.txt").write_text("x\n")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "init")
    c = M.code_info(tmp_path)
    assert len(c["git_sha"]) == 40 and c["git_dirty"] is False and c["laya_diff_empty"] is True
    (tmp_path / "other.txt").write_text("y\n")
    c = M.code_info(tmp_path)
    assert c["git_dirty"] is True and c["laya_diff_empty"] is True
    (tmp_path / "laya" / "x.py").write_text("a = 2\n")
    assert M.code_info(tmp_path)["laya_diff_empty"] is False
    _git(tmp_path, "checkout", "--", "laya/x.py")
    (tmp_path / "laya" / "new.py").write_text("")
    assert M.code_info(tmp_path)["laya_diff_empty"] is False  # untracked file in laya/ counts


def test_hardware_values_are_real_or_unknown():
    hw = M.hardware_info()
    assert hw["logical_cores"] == "unknown" or hw["logical_cores"] >= 1
    assert hw["ram_bytes"] == "unknown" or hw["ram_bytes"] > 0
    assert hw["hybrid_cores"] in (True, False, "unknown")
