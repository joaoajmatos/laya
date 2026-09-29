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
| `--mha-fastpath on\|off` | PyTorch's `TransformerEncoderLayer` inference fast path, used by Laya's decision head. `on` (default) is native Laya; `off` is a labelled variant (research.md R17). Recorded in the manifest; one setting per run directory |

One model and revision per run: when the run directory already has `manifest.json`, a command that
gives no `--revision` uses the manifest's resolved commit (so later commands measure the same weights
and work offline with `HF_HUB_OFFLINE=1`), and a different `--model` or `--revision` is refused. A model
that cannot load inside a measurement process stops `sweep` or `profile` with a non-zero exit; it is
not recorded as a failed condition.

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
`--batch-sizes 1,4,8`, `--grid axes|full`. `axes` (default) varies one factor at a time around the
primary curve: at every length it runs questions=1/batch=1, then each other question count at batch 1,
then each other batch size at one question. `full` runs every combination. The default keeps the run
to 6 conditions per length (42 in all) instead of 12 (84); the tuples run are recorded in `sweep.json`.
`--time-cap` defaults to 900 s per condition, model load included: the child stops starting repeats and
reports `partial`, and the parent kills it 30 s later (`failed`, cause `timeout`). Lengths above positional
capacity (read from `audit.json` in the same run, when present) are recorded `unsupported` without starting
a process. Lengths above the configured input cap but within positional capacity are
measured and flagged `beyond_configured_max_len`. Each condition runs in its own subprocess. `sweep.json` is
rewritten after every condition, so an interrupted sweep keeps what it measured.
Drift canary (research.md R14): `--canary-length` (default 512; 0 disables), `--canary-repeats` (default 5),
`--drift-threshold` (default 0.05). The canary, a single-question request at that length, runs in its own process
before each new length and once at the end. It is stored in `sweep.json` under `canary`, never in `items`.
Each item gets a `drift` record with the canaries measured just before and after it, and `flagged: true`
when either differs from the first canary by more than the threshold.

## `profile`
Component breakdown for single requests at each length. Options: `--lengths`, `--repeats` (default 5),
`--warmup` (default 1), `--time-cap`, `--overhead-length` (default 512; 0 skips the wrappers on/off check).
Runs are labeled as profile runs and are not used for headline latency. Writes `profile.json`, fills
`total_clean_ms` from `sweep.json` when present, and writes the executed-work verdict into `audit.json`
when it exists in the same run. `profile` takes the same canary options and records drift the same way.

## `kernels`
Microbenchmarks and correctness checks. Options: `--lengths`, `--impls dense,dense_masked,local,gas`
(default all four; `dense_masked` applies the block-local pattern as a mask over full attention, which is
what the native local layers do), `--block-sizes 128,256,512`, `--selection-sizes 512,1024` (tokens gathered
per query block by `gas`; a size that is not a whole number of at least 3 blocks is recorded `unsupported`),
`--repeats auto|N` (20 up to 2,048 tokens, 10 above), `--warmup 2`, `--time-cap 300`. Shapes (heads, head
dimension, positional capacity) are taken from `audit.json`; without it the command stops with an error.
Each condition runs in its own process. Writes `kernels.json` and `floor.json` (the latter needs
`profile.json`; otherwise it is listed under `not_run` in `kernels.json`). Run it after `sweep` and
`profile`, not during them.

## `gpu-reference`
GPU-only check of the historical ~33 ms figure (research.md R15). Needs a CUDA build of PyTorch: run it from
`.venv-gpu` (`experiments/setup_gpu.ps1`), after the CPU runs. Options: `--lengths` (default `512,2048`),
`--repeats` (default 30), `--warmup` (default 5). Uses Laya's own CUDA precision defaults. Writes
`gpu_reference.json`. Stops with an error when CUDA is unavailable or Laya falls back to another device.

## `abtest`
Instrumentation A/B test (research.md R16). Options: `--lengths` (default `2048,8192`), `--repeats` (default 4),
`--modes` (default `clean,wrappers,labels,profiler,all`; also `fastpath_off` and `labels_encoder`, research.md R17; `clean` is required), `--time-cap` (default 2400 s).
Each length runs in its own process; within it the same documents run under every mode, interleaved with a
rotating order. Writes `abtest.json` with per-mode timings, the ratio of each mode's p50 to clean, and the
thread count after each call.

## `report`
Assemble `report.md` and `report.json` from the other files in the run directory. Fails if
`manifest.json` or `audit.json` is missing. Missing sweep/profile/kernel/floor files are listed
under `not_run` and not treated as errors. Can be rerun at any time; it only reads result files.

## `all`
Run `audit` (which also writes `manifest.json`), `sweep`, `profile`, `kernels` and `report` in order with the
common options. Optional pass-through options, given only to the steps that take them: `--lengths`,
`--questions`, `--batch-sizes`, `--repeats`, `--time-cap`, `--canary-length`, `--overhead-length`, `--impls`,
`--block-sizes`, `--selection-sizes`. `commands.txt` records the single `all` line.

`sweep`, `profile` and `kernels` also record a `session` block (start time, power scheme, AC power), since the
power plan can change after the manifest was written.
