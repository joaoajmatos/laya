# Implementation Plan: CPU Path Audit and Measurement (Research Phase 1)

**Branch**: `001-cpu-path-audit` (spec directory only; work stays on `main` unless the user branches) | **Date**: 2026-09-28 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `/specs/001-cpu-path-audit/spec.md`

## Summary

Build the measurement tooling that `experiments/` lacks, and use it on unmodified native Laya
(English checkpoint) to produce the Phase 1 deliverable: a pinned run manifest, a context audit,
component cost curves, memory limits, and a ranked bottleneck list.

The approach keeps `laya/` untouched. Every measurement point the spec needs is reachable from
outside the library: `predict_batch` accepts per-call `max_len` overrides, the agent exposes
`device`, `dtype`, `amp_enabled`, `cpu_fallback_count` and `_compiled`, and the stages
(`_encode_state`, `collate_items`, `_forward`, `_decode_answers`) can be wrapped on the instance
without editing them. Operator-level cost attribution comes from `torch.profiler` plus module
hooks. The spec's permission to edit `laya/` (FR-017 to FR-019) stays available as a contingency
and is not used by this plan.

Each measurement condition runs in its own subprocess. That gives per-condition peak memory, since
peak RSS is monotonic within a process. It also lets an out-of-memory kill be recorded as a
failure, since the kernel can end the process on macOS and Linux without raising an exception.

## Technical Context

**Language/Version**: Python 3.10+ (constitution); development machine has 3.12.8

**Primary Dependencies**: existing `torch>=2.0`, `transformers>=4.48`, `safetensors`, `huggingface_hub`, `numpy`. No new runtime dependency. `pytest` is added as an optional `experiments` extra for tests. Peak memory uses the standard-library `resource` module (units differ on macOS and Linux, handled in one place).

**Storage**: JSON result files under the git-ignored `experiments/results/`. A run directory per session holds the manifest, audit, sweep, profile, kernel, and report outputs.

**Testing**: `pytest`. Unit tests run offline against a tiny randomly initialized ModernBERT-based checkpoint written to a temp directory. Integration tests against the real pinned checkpoint are marked `slow` and need network on first run.

**Target Platform**: The measuring machine, recorded in the manifest. The development machine is an Apple M1 (arm64, 8 cores, 16 GB, no native bf16). Nothing in the tooling is specific to it. Results from a different machine class are separate runs and are never merged.

**Project Type**: Research tooling. A CLI package (`python -m experiments`) that imports `laya` as a library.

**Performance Goals**: None for the tooling itself. The tooling must not perturb what it measures (FR-013): clean latency runs attach no hooks or profilers.

**Constraints**: Sequences up to 8,192 tokens on 16 GB RAM. Per-condition subprocess isolation. Synthetic inputs must reach an exact total token count. No network use after the first checkpoint download.

**Scale/Scope**: 7 lengths, about 3 question counts, 1 batch scenario, 3 kernel families, about 4 shapes each. Tens of minutes for a full sweep at short lengths, and possibly hours at 8K depending on the CPU. Repeat counts are parameters with documented defaults (see research.md R6).

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-checked after Phase 1 design.*

| Principle | Assessment | Status |
|---|---|---|
| I. Preserve the Inference Contract | No change to `laya/` planned. Native path, typed outputs and marker semantics are untouched. If the contingency is used, instrumentation is off by default and output equivalence is tested (FR-019). | Pass |
| II. Measure Research Claims | Each report statement carries a status of measured, estimated, or hypothesized (FR-028). Analytical operation counts are kept separate from timings. Planned APIs are labeled as planned. | Pass |
| III. Valid Native Baseline | The reference is the pinned, unmodified checkpoint, audited from the loaded model (FR-003). "Native" is not assumed to mean full attention; research.md R3 flags that the loaded local layers may still execute dense masked attention. Quality comparison is out of scope for this phase. | Pass |
| IV. Keep the Runtime Focused | All tooling and kernels live in `experiments/`. `laya/layers/attention.py` stays empty. | Pass |
| V. Reproducible Experiments | The manifest records revisions, software, hardware, threads, precision, seeds and cache policy. Each result has its reproduction command. Results are git-ignored. | Pass |
| VI. Fresh-Document CPU Decisions | Primary latency is one fresh document, one question, tokenization through postprocessing. Multi-question and batch runs are separate. No GPU result substitutes for CPU. | Pass |
| VII. Evaluation Independence | Not engaged: no quality evaluation, splits or test data in this phase. Inputs are synthetic and labeled as such. | N/A |

Technical Constraints check: the full forward pass is accounted for, including decision-head attention and data preparation (component profile). Dense masks require executed-work verification before a speed claim (FR-023, research.md R5). Sparse kernels are checked against a reference with identical mask semantics (FR-022). Total context counts task, option and special tokens (FR-005).

**Result**: no violations. Complexity Tracking is empty.

## Project Structure

### Documentation (this feature)

```text
specs/001-cpu-path-audit/
├── plan.md              # This file
├── research.md          # Phase 0: decisions and their rationale
├── data-model.md        # Phase 1: entities and result record shapes
├── quickstart.md        # Phase 1: end-to-end validation guide
├── contracts/
│   ├── cli.md           # Command-line contract
│   └── results.md       # Result file contract
├── checklists/
│   └── requirements.md
└── tasks.md             # Produced later by /speckit-tasks
```

### Source Code (repository root)

```text
experiments/
├── __init__.py
├── __main__.py            # `python -m experiments <command>` entry point
├── cli.py                 # Argument parsing; dispatch to the modules below
├── results.py             # Run directory, JSON writing with schema_version/run_id, commands log, percentile summaries
├── manifest.py            # Environment capture: revisions, versions, hardware, threads, precision
├── audit.py               # Loaded-model audit: layer types, windows, RoPE, limits, head depth, fallbacks
├── tokens.py              # Token accounting record and length validation
├── inputs.py              # Synthetic request builder that hits an exact total token count
├── runner.py              # Subprocess-per-condition executor; failure and kill capture; RSS
├── timing.py              # Clean end-to-end latency measurement and percentile summaries
├── profile.py             # Stage wrappers, torch.profiler, operator-to-component attribution
├── kernels/
│   ├── __init__.py
│   ├── reference.py       # Dense same-mask reference with defined fully-masked-row behavior
│   ├── dense.py           # Dense SDPA baseline
│   ├── local.py           # Local / block attention
│   ├── gas.py             # Gather-attend-scatter attention
│   └── bench.py           # Shape sweep, correctness checks, executed-work checks, memory
├── floor.py               # Attention-free cost-floor estimate from component costs
├── report.py              # Bottleneck ranking and Markdown/JSON report assembly
└── results/               # Git-ignored outputs

tests/
└── experiments/
    ├── conftest.py        # Tiny offline checkpoint fixture
    ├── test_results.py
    ├── test_runner.py
    ├── test_manifest.py
    ├── test_audit.py
    ├── test_tokens.py
    ├── test_inputs.py
    ├── test_timing.py
    ├── test_profile.py
    ├── test_kernels.py
    └── test_report.py
```

**Structure Decision**: one package, `experiments/`, importing `laya` as a library, as constitution principle IV requires. It is deliberately flat. The README sketch (`suite/`, `datasets/`, `scripts/`) describes later phases: `datasets/` belongs to Phase 2, so it is not created here, and the CLI in `cli.py` replaces a `scripts/` directory. `experiments/` stays out of the installed package (`pyproject.toml` lists only `laya`), and tools run from the repo root. `tests/` is new at the top level because none exists.

Two small repository changes accompany the code: `experiments/results/` is added to `.gitignore`, and `pyproject.toml` gains an optional `experiments` extra (`pytest`). Neither touches `laya/`.

## Complexity Tracking

No constitution violations. Nothing to justify.
