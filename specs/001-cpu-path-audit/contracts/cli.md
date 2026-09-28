# Contract: Command-Line Interface

Entry point: `python -m experiments <command> [options]`, run from the repository root.
Every command writes to `experiments/results/<run_id>/` and prints the path. Output files are
described in [results.md](results.md). Exit code 0 means the command completed, including when
some conditions are recorded as `unsupported` or `failed` (they are results, not errors).
A non-zero exit means the tool itself could not run (bad arguments, model cannot load, device is not CPU).

Common options:

| Option | Meaning |
|---|---|
| `--run-id ID` | Reuse or name a results directory (default: timestamp) |
| `--model ID_OR_PATH` | Checkpoint (default `convaiinnovations/laya`) |
| `--revision REV` | Commit to pin; `reviewed` uses `laya.revisions.PINNED_REVISIONS` |
| `--threads N` | Torch intra-op threads, fixed for the whole run and passed to every child (research.md R12). Default: physical core count. On a hybrid CPU, pass the performance-core count. Recorded in the manifest with its source (`default` or `user`) |
| `--seed N` | Base seed |

## `manifest`
Write `manifest.json` (environment capture). Fails if the effective device is not CPU.

## `audit`
Load the model, write `audit.json`. Also prints the per-layer attention schedule.
Option `--lengths L1,L2,...` reports which lengths are supported.

## `sweep`
Latency sweep. Options: `--lengths` (default `128,256,512,1024,2048,4096,8192`),
`--repeats auto|N`, `--warmup N` (default 3), `--time-cap SECONDS` per condition,
`--questions 1,2,5,10` (1 is the primary single-request curve),
`--options N` (options per question, default 2; recorded per condition),
`--batch-sizes 1,4,8`. Lengths above the configured input cap but within positional capacity are
measured and flagged `beyond_configured_max_len`. Each condition runs in its own subprocess. Writes `sweep.json`.

## `profile`
Component breakdown for single requests at each length. Options: `--lengths`, `--repeats`.
Runs are labeled as profile runs and are not used for headline latency. Writes `profile.json`.

## `kernels`
Microbenchmarks and correctness checks. Options: `--lengths`, `--impls dense,local,gas`,
`--block-sizes 128,256,512`, `--selection-sizes`. Shapes are taken from `audit.json`.
Writes `kernels.json` and `floor.json` (the latter needs `profile.json`).

## `report`
Assemble `report.md` and `report.json` from the other files in the run directory. Fails if
`manifest.json` or `audit.json` is missing. Missing sweep/profile/kernel files are listed
under `not_run` and not treated as errors.

## `all`
Run `manifest`, `audit`, `sweep`, `profile`, `kernels`, `report` in order with shared options.
