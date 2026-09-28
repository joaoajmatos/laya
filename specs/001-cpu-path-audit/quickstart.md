# Quickstart: Validating the Phase 1 Tooling

A guide to running and checking the feature end to end. It does not contain implementation code.
Commands are the contract in [contracts/cli.md](contracts/cli.md); result shapes are in
[contracts/results.md](contracts/results.md) and [data-model.md](data-model.md).

## Prerequisites

- Python 3.10+; work from the repository root.
- `pip install -e ".[experiments]"` (installs `laya-sparse` with torch, transformers, and pytest).
- Network access once, to download the pinned English checkpoint. Afterwards use `HF_HUB_OFFLINE=1`.
- A quiet machine: close heavy applications, keep the machine on power, and note anything else running.

## 1. Offline checks (no download)

```bash
pytest tests/experiments -m "not slow"
```

Expected: all pass. These cover manifest fields, audit values on a tiny model, token
reconciliation, exact-length inputs, percentile math, profiler attribution sums, kernel-vs-reference
agreement (including fully masked rows), and report ranking.

## 2. Manifest and audit

```bash
python -m experiments manifest --revision reviewed --run-id smoke
python -m experiments audit    --revision reviewed --run-id smoke --lengths 512,4096,8192,16384
```

Expected:
- `manifest.json` has every field in the data model, a resolved model revision, and `device.effective: cpu`.
- `audit.json` lists each layer as local or global with its window, states positional capacity, and marks 16384 as `unsupported` with a reason.

## 3. Short sweep and profile

```bash
python -m experiments sweep   --run-id smoke --lengths 128,512 --repeats 10
python -m experiments profile --run-id smoke --lengths 128,512 --repeats 5
```

Expected:
- `sweep.json` has p50/p95, `low_sample_p95: true` (fewer than 20 repeats), peak memory, and a token accounting record whose parts sum to `final_length`.
- `profile.json` has `explained_fraction >= 0.90` or a stated shortfall, and a profiled-to-clean ratio.
- Confirm the 512 result is compared with the ~33 ms historical figure and the difference is stated, not assumed.

## 4. Failure handling

```bash
python -m experiments sweep --run-id smoke-fail --lengths 8192 --time-cap 5
```

Expected: an entry with `status: partial` or `failed` and its `reason`; the command still exits 0.
Confirm no other length was substituted.

## 5. Kernels

```bash
python -m experiments kernels --run-id smoke --lengths 512,2048 --impls dense,local,gas
```

Expected: `kernels.json` lists all correctness cases as pass within the stated tolerance;
each timing includes mask/routing; `executed_work` has a verdict; analytical and measured
memory are separate fields. `floor.json` is produced once `profile.json` exists.

## 6. Full run and report

```bash
python -m experiments all --run-id full --revision reviewed
```

Expected: `report.md` ranks bottlenecks per length with links to items, gives a verdict for each
plan assumption, lists everything not run, and includes the commands from `commands.txt`.

## 7. Integrity checks against the success criteria

- SC-005: `git diff --stat main -- laya/` is empty (or the equivalence test passes if the contingency was used).
- SC-002: no result in `sweep.json` has `truncated: true` unless its `reason` says so.
- SC-008: each `assumptions[]` entry has a verdict and evidence link.
- Constitution V: `git status` shows no files under `experiments/results/`.
