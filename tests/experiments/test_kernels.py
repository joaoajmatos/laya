"""T030: attention kernels, correctness, memory fields and executed work (experiments/kernels/)."""
import pytest
import torch

from experiments import cli, results
from experiments.floor import build_floor
from experiments.kernels import bench
from experiments.kernels.reference import MaskSpec, reference_attention, score_matrix_bytes

CASE_NAMES = {"padding", "boundary_chunk", "non_divisible_length", "fully_masked_row", "mixed_batch"}


@pytest.mark.parametrize("impl", ["dense", "dense_masked", "local", "gas"])
def test_every_case_matches_the_reference(impl):
    checks = bench.correctness(impl)
    assert {c["case"] for c in checks} == CASE_NAMES
    for c in checks:
        assert c["tolerance"] == 1e-5
        assert c["finite"], "%s produced NaN/inf on %s" % (impl, c["case"])
        assert c["passed"], "%s: %s error %g" % (impl, c["case"], c["max_abs_error"])


def test_fully_masked_row_is_zero_not_nan():
    q, k, v = (torch.randn(1, 2, 32, 8) for _ in range(3))
    for pattern in ("full", "block_local", "gather"):
        out = reference_attention(q, k, v, MaskSpec([0], pattern, 8, 4))
        assert torch.equal(out, torch.zeros_like(out))


def test_patterns_really_differ():
    q, k, v = (torch.randn(1, 2, 64, 8) for _ in range(3))
    full = reference_attention(q, k, v, MaskSpec([64], "full", 16))
    local = reference_attention(q, k, v, MaskSpec([64], "block_local", 16))
    assert float((full - local).abs().max()) > 1e-2


def test_memory_fields_are_separate_and_analytical_size_is_exact():
    rec = bench.bench_condition({"impl": "local", "length": 256, "heads": 4, "head_dim": 16,
                                 "block": 64, "repeats": 2, "warmup": 0})
    assert rec["status"] == "measured"
    assert rec["score_matrix_bytes_analytical"] == 1 * 4 * 256 * 256 * 4
    assert score_matrix_bytes(2, 16, 8192) == 2 * 16 * 8192 * 8192 * 4
    assert "peak_rss_bytes" not in rec  # measured memory comes from the process, set by the runner
    assert rec["timings_ms"]["n"] == 2 and "mask construction" in rec["timing_includes"]


def test_issued_work_is_quadratic_for_dense_and_masked_but_linear_for_sparse():
    from experiments.profile import fit_exponent
    Ls = [512, 1024, 2048, 4096]
    exps = {impl: fit_exponent(Ls, [bench.issued_score_elements(impl, 1, 16, L, 128, 4) for L in Ls])
            for impl in bench.IMPLS}
    assert exps["dense"] == pytest.approx(2.0) and exps["dense_masked"] == pytest.approx(2.0)
    assert exps["local"] == pytest.approx(1.0) and exps["gas"] == pytest.approx(1.0)


def _item(impl, L, ms, i=0):
    return {"impl": impl, "status": "measured", "shape": {"length": L, "block": 128, "selection_blocks": None},
            "timings_ms": {"p50": ms}, "issued_score_elements": L * L}


def test_masked_dense_is_same_work_not_less_work():
    Ls = [512, 1024, 2048, 4096]
    items = ([_item("dense", L, (L / 512) ** 2) for L in Ls]
             + [_item("dense_masked", L, 1.3 * (L / 512) ** 2) for L in Ls]
             + [_item("local", L, 0.8 * L / 512) for L in Ls])
    verdicts = {g["impl"]: g["verdict"] for g in bench.executed_work(items)["groups"]}
    assert verdicts == {"dense": "baseline", "dense_masked": "same_work", "local": "less_work"}


def test_classify_needs_visible_quadratic_dense():
    assert bench.classify("local", 1.0, 1.1) == "not_determined"
    assert bench.classify("local", 1.6, 2.0) == "not_determined"


def test_gas_selection_rules():
    conds = cli.kernel_conditions(["gas"], [1024], [128, 512], [512])
    by = {c["block"]: c["selection_blocks"] for c in conds}
    assert by == {128: 4, 512: 1}


def test_kernels_command_and_floor(tiny_checkpoint, tmp_path, monkeypatch):
    monkeypatch.setattr(results, "RESULTS_ROOT", tmp_path)
    assert cli.main(["kernels", "--run-id", "k", "--threads", "1"]) == cli.EXIT_TOOL_ERROR  # needs audit.json
    assert cli.main(["audit", "--run-id", "k", "--model", tiny_checkpoint, "--threads", "1"]) == 0
    code = cli.main(["kernels", "--run-id", "k", "--model", tiny_checkpoint, "--threads", "1",
                     "--lengths", "128,256,4096", "--impls", "dense,local,gas", "--block-sizes", "32",
                     "--selection-sizes", "64,128", "--repeats", "2", "--warmup", "0"])
    assert code == 0
    body = results.read_json(tmp_path / "k", "kernels.json")
    status = {(i["impl"], i["shape"]["length"], i["shape"].get("selection_blocks")): i["status"] for i in body["items"]}
    assert status[("gas", 128, 2)] == "unsupported"          # 64 tokens = 2 blocks of 32 < 3
    assert status[("local", 4096, None)] == "unsupported"    # above the fixture's 2048 positions
    assert status[("dense", 256, None)] == "measured"
    for it in body["items"]:
        results.validate_item(it)
        if it["status"] == "measured":
            assert it["peak_rss_bytes"] > 0 and it["score_matrix_bytes_analytical"] > 0
            assert all(c["passed"] for c in it["correctness"])
    assert body["executed_work"]["groups"]
    assert body["not_run"] and "profile" in body["not_run"][0]["reason"]
    assert not (tmp_path / "k" / "floor.json").exists()


def test_floor_estimate_and_analytical_bound(tmp_path):
    run = results.run_dir("f", root=tmp_path)
    comps = lambda s: {"attention_score_value": {"share": s, "ms": 0}}
    results.write_json(run, "profile.json", {"items": [
        {"status": "measured", "condition": {"total_tokens": 512}, "components": comps(0.12), "total_profiled_ms": 1500},
        {"status": "measured", "condition": {"total_tokens": 2048}, "components": comps(0.30), "total_profiled_ms": 7000},
        {"status": "failed", "reason": "oom", "condition": {"total_tokens": 8192}}]})
    results.write_json(run, "sweep.json", {"items": [
        {"kind": "single", "status": "measured", "condition": {"total_tokens": 512}, "timings_ms": {"p50": 1600.0}}]})
    fl = build_floor(run)
    by = {i["length"]: i for i in fl["items"]}
    assert by[512]["floor_ms"] == pytest.approx(1600 * 0.88) and by[512]["total_basis"] == "clean p50"
    assert by[2048]["floor_ms"] == pytest.approx(7000 * 0.70) and by[2048]["total_basis"] == "profiled total"
    assert by[512]["label"] == "estimate" and by[8192]["status"] == "failed"
    a = fl["analytical_bound"]
    assert a["label"] == "analytical" and a["bound_ratio_4096_vs_512"] == pytest.approx(8 * 0.88)
    assert a["bound_ms_4096"] == pytest.approx(8 * 0.88 * 1600)
    assert "floor_ms" not in a  # estimate and analytical bound are never merged


def test_floor_without_profile_fails_clearly(tmp_path):
    with pytest.raises(FileNotFoundError, match="profile.json"):
        build_floor(results.run_dir("g", root=tmp_path))


def test_gpu_reference_refuses_without_cuda(tiny_checkpoint, tmp_path, monkeypatch):
    if torch.cuda.is_available():
        pytest.skip("CUDA present; this test covers CPU-only environments")
    monkeypatch.setattr(results, "RESULTS_ROOT", tmp_path)
    assert cli.main(["gpu-reference", "--run-id", "g", "--model", tiny_checkpoint, "--threads", "1"]) == cli.EXIT_TOOL_ERROR
    assert not (tmp_path / "g" / "gpu_reference.json").exists()
