---

description: "Task list for CPU Path Audit and Measurement (Research Phase 1)"
---

# Tasks: CPU Path Audit and Measurement (Research Phase 1)

**Input**: Design documents from `/specs/001-cpu-path-audit/`

**Prerequisites**: [plan.md](plan.md), [spec.md](spec.md), [research.md](research.md), [data-model.md](data-model.md), [contracts/cli.md](contracts/cli.md), [contracts/results.md](contracts/results.md)

**Tests**: Included. The plan specifies offline unit tests (quickstart step 1 depends on them). Test tasks come first within each story and should fail before the implementation they cover.

**Organization**: Grouped by user story. Paths are relative to the repository root. `laya/` is not edited by any task (research.md R1); T045 verifies that.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependency on an incomplete task)
- **[Story]**: US1 to US4, matching spec.md
- Statuses used in result files are exactly: `measured`, `unsupported`, `failed`, `partial`

---

## Phase 1: Setup

**Purpose**: Package skeleton and repository housekeeping

- [X] T001 Add `experiments/results/` to `.gitignore` so generated results are never committed (constitution V)
- [X] T002 Add an optional dependency group `experiments = ["pytest>=7"]` to `pyproject.toml` without changing `dependencies` or `[tool.setuptools] packages`; register a `slow` pytest marker in a `[tool.pytest.ini_options]` table with `testpaths = ["tests"]`
- [X] T003 Create the empty package skeleton from plan.md: `experiments/__main__.py` (calls `experiments.cli.main()`), `experiments/kernels/__init__.py`, `tests/experiments/__init__.py`

---

## Phase 2: Foundational (blocks all user stories)

**Purpose**: Shared result I/O, subprocess isolation, CPU-only model loading, CLI dispatch, and the offline test checkpoint

**CRITICAL**: No user story starts until this phase is complete

- [X] T004 [P] Write tests in `tests/experiments/test_results.py` for `experiments/results.py`: results directory creation under `experiments/results/<run_id>/`; every JSON file carries `schema_version` (int, starts at 1) and `run_id` at the top level; `commands.txt` is appended, not overwritten; `summarize(samples_ms)` returns `min`, `p50`, `p95`, `mean`, `std`, `n` and sets `low_sample_p95` true when `n < 20`; a status outside `measured`/`unsupported`/`failed`/`partial` is rejected; a non-`measured` item without `reason` is rejected
- [X] T005 Implement `experiments/results.py` per T004: run directory helper, `write_json(run_dir, name, payload)`, `append_command(run_dir, argv)`, `summarize()`, and a `Status` enum with the four statuses. Times are milliseconds, memory is bytes (contracts/results.md rule 3)
- [X] T006 [P] Create the offline fixture in `tests/experiments/conftest.py`: a session-scoped temp checkpoint directory that `laya.Agent(path)` loads without network. Contents: a small randomly initialized ModernBERT encoder config (about 3 layers, hidden 64, 2 heads, alternating local/global `layer_types`, local window small, `max_position_embeddings` a few thousand) saved under `encoder/`; a `DecisionModel` state dict saved as `model.safetensors` (build with `laya.common.build_model(cfg, encoder_dir, pretrained=False)`); a tokenizer trained from scratch with the `tokenizers` package and saved under `tokenizer/` with `[CLS]`, `[SEP]`, `[MASK]`, `[PAD]`, `[UNK]`; and `rl_agent_config.json` with the keys `laya/agent.py` requires (`encoder`, `head_layers`, `max_len`, `head_max_len`, `temperature`, `act_costs`). Read `laya/agent.py` `_verify_compatibility` and `_load_tokenizer`, and `laya/common.py` `build_model`, before writing it. Timings from this tiny model are never reported as measurements
- [X] T007 [P] Write tests in `tests/experiments/test_runner.py` for `experiments/runner.py`: a child that succeeds returns its JSON payload and `peak_rss_bytes` > 0; a child that raises records `status: failed`, `cause: exception` and the message; a child that raises an allocator out-of-memory error (simulate with `RuntimeError("DefaultCPUAllocator: not enough memory")`) or `MemoryError` records `cause: oom`; a child that exits abnormally without writing a result records `cause: oom` for exit code 0xC0000017 on Windows or SIGKILL on POSIX, `cause: signal` for other POSIX signals, and `cause: crash` otherwise, always with `exit_code` (`signal` is null on Windows); the platform-specific cases are skipped on the other platform, and the Windows cases must run on the measuring machine; a child that sleeps past `time_cap` is terminated and recorded `cause: timeout`; the parent process is unaffected in every case
- [X] T008 Implement the subprocess executor in `experiments/runner.py`: `run_condition(module_function, spec, time_cap)` starts a fresh Python child (`python -m experiments.runner --child`), passes the spec as JSON, reads the child's JSON result from a temp file, and converts exit status into a result item per data-model.md. Peak memory comes from one platform function in the child: on Windows `ctypes` + `GetProcessMemoryInfo` (`PeakWorkingSetSize` -> `peak_rss_bytes`, `PeakPagefileUsage` -> `peak_commit_bytes`), on macOS/Linux `resource.getrusage(RUSAGE_SELF).ru_maxrss` (bytes on macOS, kilobytes on Linux; `peak_commit_bytes` null); set `paging_suspected` when peak commit exceeds physical RAM. Import `resource` only on POSIX. Use a temp file path the child opens itself (Windows cannot reopen an open `NamedTemporaryFile`), and end timed-out children with `Popen.kill()`. Depends on T005
- [X] T009 Add `load_agent(model, revision, threads)` to `experiments/runner.py`: resolves `revision="reviewed"` through `laya.revisions.PINNED_REVISIONS`, sets `torch.set_num_threads` to the run's fixed thread count (research.md R12), loads `laya.Agent(..., device="cpu")`, and raises if `agent.device.type != "cpu"`. It also records model load seconds and whether `agent.cpu_fallback_count` or a device fallback message occurred. Does not edit `laya/`. Depends on T008
- [X] T010 Implement `experiments/cli.py` with the common options from contracts/cli.md (`--run-id`, `--model`, `--revision`, `--threads` defaulting to the physical core count and recorded with its source, `--seed`), a command registry that later tasks add to, `commands.txt` logging via `append_command`, and exit codes (0 when conditions are recorded as `unsupported`/`failed`; non-zero only for tool-level errors). Wire `experiments/__main__.py` to `main()`. Depends on T005

**Checkpoint**: `pytest tests/experiments/test_results.py tests/experiments/test_runner.py` passes; `python -m experiments --help` lists the commands registered so far

---

## Phase 3: User Story 1 - Pinned Run Manifest and Context Audit (Priority: P1) MVP

**Goal**: A complete manifest of the environment and an audit of what the loaded model actually is, with exact token accounting and explicit handling of unsupported lengths.

**Independent Test**: Run `manifest` and `audit` on the tiny fixture (offline) and on the pinned English checkpoint (`slow`). Fields complete; layer schedule matches the loaded config; token parts sum to the final length; a length above positional capacity is `unsupported`.

### Tests for User Story 1

- [X] T011 [P] [US1] Write `tests/experiments/test_manifest.py`: every field in data-model.md "Run Manifest" is present; `model.revision` is not null for a Hub-style load and is null-tolerant only for a local path (recorded as `local`); `git.laya_diff_empty` is true on a clean tree; a requested device other than `cpu` marks the run `non_comparable`; `LAYA_CPU_AMP` is recorded when set
- [X] T012 [P] [US1] Write `tests/experiments/test_audit.py`: on the tiny fixture, `layers[]` matches the fixture's configured local/global schedule and window; `positional.max_position_embeddings` and `configured_max_len` are reported as different fields; `decision_head.layers` matches `head_layers`; `effective_supported_max` equals `max_position_embeddings`; `--lengths` above it are `unsupported` with a reason; an implicit fallback (simulated by setting `agent.cpu_fallback_count`) appears in `fallbacks[]`
- [X] T013 [P] [US1] Write `tests/experiments/test_tokens.py`: the record satisfies `special + task_option + state_retained + padding == final_length` exactly; `truncated` is true iff `state_retained < state_original`; a batch of two different-length rows reports nonzero `padding_tokens` for the shorter row; a record whose parts do not sum is rejected; task, option and special tokens are counted against `position_limit`; the record carries `n_questions` and `options_per_question`, and a multi-row request holds one entry per row
- [X] T014 [P] [US1] Write `tests/experiments/test_inputs.py`: for each of several targets (for example 128, 512, 1500), `build_request(agent, total_tokens, seed, n_questions, options_per_question, n_states)` yields a request whose accounting `final_length == total_tokens` exactly for **every question row**, including `n_questions` of 2, 5 and 10, `options_per_question` of 2 and 8, and `n_states` of 4; batches contain no padding tokens; two different seeds give different state text; the same seed gives the same text; a target smaller than the fixed question and option overhead raises with a clear message; a row that misses the target makes the build fail

### Implementation for User Story 1

- [X] T015 [P] [US1] Implement `experiments/manifest.py`: capture code (`git_sha`, `git_dirty`, `laya_diff_empty` via `git diff --quiet <sha> -- laya/`), model (`id`, resolved `revision` from `agent.revision`, `config_sha256`), software versions (Python, torch, transformers, numpy, safetensors), hardware (CPU model, arch, physical/logical cores, RAM bytes, OS; on Windows read the CPU name from the registry via `winreg`, RAM via `GlobalMemoryStatusEx`, physical cores via `Get-CimInstance Win32_Processor` in PowerShell, and the power scheme via `powercfg /getactivescheme` plus AC status via `GetSystemPowerStatus`; use `sysctl` on macOS and `/proc` on Linux; degrade to `unknown` with a note), runtime (torch intra/inter-op threads, thread-count source, `hybrid_cores` read best-effort on Windows from `GetLogicalProcessorInformationEx` core efficiency classes and `unknown` otherwise, BLAS/MKL/OpenMP from `torch.__config__`, `LAYA_CPU_AMP`, `agent.dtype`, `agent.amp_enabled`, compile flag), device (requested, effective), cache policy (question-token reuse on; document caches none) and seeds. Depends on T005, T009
- [X] T016 [P] [US1] Implement `experiments/tokens.py`: `account(agent, state, question, max_len, batch_rows=None)` returns the Token Accounting Record from data-model.md using `laya.common.build_sequence(..., return_stats=True)`, an independent full tokenization of the state (`laya.common.encode_text` with `serialize_state`), and `laya.common.collate_items` for padding. Include `n_questions` and `options_per_question`, and return one entry per question row for multi-row requests. Enforce: "`special + task_option + state_retained + padding == final_length` (exact); `final_length <= position_limit`; `requested_total == final_length` for every question row". Raise `TokenAccountingError` if they do not hold. Depends on T005
- [X] T017 [US1] Implement `experiments/inputs.py`: `build_request(agent, total_tokens, seed, n_questions=1, options_per_question=2, n_states=1)` returns `(states, questions, accounting)`. Build every question and its options from templates of equal token length, so all question rows share one head length; compute the fixed overhead with `build_sequence`; grow a seeded English-word filler state per document by binary search on word count until the tokenized state fills the remaining budget; set the per-call `max_len` to `total_tokens`; verify with `tokens.account` and fail if any row's `final_length != total_tokens` (research.md R4). Batches then hold rows of one length and carry no padding. Depends on T016
- [X] T018 [US1] Implement `experiments/audit.py`: read the loaded model, not documentation. Report encoder class, hidden size, heads, head dim, layer count, attention implementation; per-layer `attention_type` (`local`/`global`) with window and RoPE theta per type (use the encoder config's `layer_types` and `local_attention` where present, else derive from `global_attn_every_n_layers`); `positional` block; `decision_head` block (layers, heads, hidden; note it runs over the full sequence, `laya/common.py:313`); `executed_work_note` initialized to `not_yet_determined`; `fallbacks[]` from `agent.cpu_fallback_count`, `agent.last_fallback_reason`, `agent.device`, `agent.dtype`, `agent.amp_enabled`. Given `--lengths`, mark any above `max_position_embeddings` as `unsupported` with the reason and never truncate or pad to a different length (FR-006). Depends on T009
- [X] T019 [US1] Register the `manifest` and `audit` commands in `experiments/cli.py` per contracts/cli.md: `manifest` fails if the effective device is not CPU; `audit` accepts `--lengths`, prints the per-layer schedule, and writes `manifest.json` and `audit.json` through `results.write_json`. Depends on T015, T018, T010
- [X] T020 [P] [US1] Add an integration test marked `slow` in `tests/experiments/test_real_checkpoint.py`: on `convaiinnovations/laya` at `revision="reviewed"`, `audit` reports the layer schedule and limits from the loaded model, and records any difference from the values in `docs/research-plan.md` (local window 128, global every third layer, 8,192 positions, `head_layers=2`, `max_len=512`) as a stated discrepancy, not a failure. Depends on T019

**Checkpoint**: User Story 1 works alone. `python -m experiments manifest` and `audit` run on the fixture and produce valid files; T011 to T014 pass.

---

## Phase 4: User Story 2 - Native Laya Latency Profile on CPU (Priority: P2)

**Goal**: Clean end-to-end latency across lengths and question counts, plus a separate profiled component breakdown, with failures and memory recorded.

**Independent Test**: Run `sweep` and `profile` on the fixture for two small lengths: repeated timings with p50/p95, component shares that explain at least 90% of profiled time (or state the shortfall), an induced failure recorded and not dropped.

### Tests for User Story 2

- [X] T021 [P] [US2] Write `tests/experiments/test_timing.py`: a single-request measurement returns `repeats.completed` timings, `timings_ms` with `min/p50/p95/mean/std`, `first_vs_median`, `load_seconds` separate from timings, and `peak_rss_bytes`; the clean timing path attaches no hooks and no profiler (assert the instance methods are unwrapped and `torch.profiler` is not active during timing); warmup requests are excluded from `timings_ms`; a request whose `total_tokens` exceeds positional capacity yields `status: unsupported` with a reason and no model call; `multi_question` and `batch` conditions have distinct `kind` values and never appear in the `single` curve; `single` and `multi_question` call the public `predict` while `batch` calls `predict_batch`; a length above the configured input cap but within positional capacity sets `beyond_configured_max_len: true`; the option count is stored with each condition
- [X] T022 [P] [US2] Write `tests/experiments/test_profile.py`: stage wrappers call through to the originals and outputs are identical with and without wrappers on the fixture; the component shares sum to 1 within 1e-6; `explained_fraction` equals the sum of named components (excluding `other`) over the profiled total; a profile record states "`explained_fraction >= 0.90` or the shortfall is stated"; the `scaling` block contains a fitted time exponent per component when at least three lengths are given; the profiled-to-clean ratio is recorded

### Implementation for User Story 2

- [X] T023 [US2] Implement `experiments/timing.py`: the child-side function `measure_condition(spec)` that loads the agent (`runner.load_agent`, recording `load_seconds` and, if `compile` is set, `compile_seconds`), builds a fresh request per repeat with a new seed (`inputs.build_request`), runs `warmup` discarded requests (default 3), then times `repeats` calls (tokenization through postprocessing) with `time.perf_counter`: the public `agent.predict` (`system_one`) for `single` and `multi_question`, and `agent.predict_batch` for `batch`. Records `first_vs_median`, fallback events read from the agent after the run, the token accounting of a representative request (one entry per row), the condition's `options_per_question`, `beyond_configured_max_len` (true when `total_tokens` exceeds `agent.cfg["max_len"]` but not positional capacity), and `kind` (`single`, `multi_question`, `batch`). Repeat defaults from research.md R6: 30 at 512 tokens and below, 20 at 1,024 to 2,048, 10 at 4,096 and 8,192. Stops early and returns `status: partial` when `time_cap` is reached. Depends on T017, T009
- [X] T024 [US2] Implement the stage wrappers in `experiments/profile.py`: instance-level wrappers around `Agent._encode_state`, `Agent._forward`, `Agent._decode_answers`, and the `collate_items` name in `laya.agent`, each recording elapsed time and calling through to the original; a context manager that installs and removes them so nothing stays patched. Map them to `tokenization_collation`, `forward`, `decoding`. Never used by `timing.py` (FR-013). Depends on T009
- [X] T025 [US2] In `experiments/profile.py`, add operator-level attribution: run under `torch.profiler` (CPU activities, `record_shapes=True`); install forward pre-hooks on encoder layers, attention modules, MLPs, norms and decision-head layers that push `torch.profiler.record_function` labels; classify operators into `attention_projections` (linear/matmul under an attention module), `attention_score_value` (`aten::scaled_dot_product_attention`), `mlp_norm`, `decision_head`, `option_scoring` and `other`. Compute shares, `explained_fraction`, the profiled-to-clean ratio, and a per-component `scaling` exponent from a log-log fit across lengths. Depends on T024
- [X] T026 [US2] In `experiments/profile.py`, add the executed-work verification for native layers (research.md R3, R5): measure attention score/value time per layer type (local versus global) across lengths and fit the scaling exponent; write the verdict (`verified` if local-layer time grows clearly slower than global, `dense_masked` if it grows like global, otherwise `not_yet_determined`) into `audit.json`'s `executed_work_note` with the supporting exponents. Depends on T025
- [X] T027 [US2] Register the `sweep` command in `experiments/cli.py` per contracts/cli.md: for every condition (`--grid axes`, the default: one factor varied at a time around questions=1, batch=1; `--grid full`: length x questions x batch size; see contracts/cli.md) call `runner.run_condition` with `timing.measure_condition`, defaults `--lengths 128,256,512,1024,2048,4096,8192`, `--questions 1,2,5,10`, `--options 2`, `--batch-sizes 1,4,8`, `--repeats auto`, `--warmup 3`, `--time-cap`; the primary curve is `questions=1, batch=1`. Lengths above positional capacity become `unsupported` without a model call; killed or timed-out children become `failed` items with their cause; nothing is dropped or substituted (FR-015). Writes `sweep.json`. Depends on T023, T010
- [X] T028 [US2] Register the `profile` command in `experiments/cli.py`: run single-request profile conditions in subprocesses through `runner.run_condition`, using `profile.py`, labeled as profile runs, then merge with the clean p50 from `sweep.json` (if present) to fill `total_clean_ms` and the ratio. Writes `profile.json`, and updates `audit.json` with T026's verdict. Depends on T026, T027
- [X] T048 [US2] Drift canary in `experiments/cli.py` for `sweep` and `profile` (research.md R14, added after the smoke run): re-measure 512 tokens before each length and at the end, store it under `canary`, and annotate each item with `drift` and `flagged`. Tests in `tests/experiments/test_timing.py` and `test_profile.py`
- [X] T029 [US2] Add a run-to-run comparison inside `experiments/timing.py`: `compare_paths(spec)` times the same fixed requests with the stage wrappers off and on, and reports the overhead ratio, so the report can state that instrumentation did not distort clean latency (SC-005). Depends on T024, T023

**Checkpoint**: User Stories 1 and 2 both work. `sweep` and `profile` run on the fixture; T021 and T022 pass.

---

## Phase 5: User Story 3 - Attention Microbenchmarks and Savings Check (Priority: P3)

**Goal**: Standalone dense, local/block, and gather-attend-scatter kernels, checked against a same-mask dense reference and timed on realistic shapes, with memory feasibility and a cost-floor estimate.

**Independent Test**: Run `kernels` on small shapes: all correctness cases pass within the stated tolerance, timings include mask/routing, analytical and measured memory are separate fields, and each implementation carries an executed-work verdict.

### Tests for User Story 3

- [X] T030 [P] [US3] Write `tests/experiments/test_kernels.py`: each of `dense`, `local`, `gas` agrees with `reference` within absolute 1e-5 (fp32) for the cases `padding` (padding at the end of a row), `boundary_chunk` (last block shorter than the block size), `non_divisible_length`, `fully_masked_row` (output is zeros by definition and never NaN), and `mixed_batch`; the reported tolerance is present in each result; the score-matrix analytical size equals `batch * heads * L * L * dtype_size` and is stored apart from `peak_rss_bytes`; a "sparse" call that only masks a dense computation is classified `same_work`, not `less_work`

### Implementation for User Story 3

- [X] T031 [P] [US3] Implement `experiments/kernels/reference.py`: the shared mask specification (padding lengths, local window or block size, gather indices) with mask builders used by all kernels, and `reference_attention(q, k, v, mask)` in fp32 with an explicit score matrix, explicit boolean mask, softmax and value product. "A fully masked row returns zeros" (defined convention; never NaN)
- [X] T032 [P] [US3] Implement `experiments/kernels/dense.py`: `dense_attention(q, k, v, mask)` using `torch.nn.functional.scaled_dot_product_attention` with the same mask semantics, matching the native path (`laya/common.py` `_DynamicMultiheadAttention` shows the call)
- [X] T033 [P] [US3] Implement `experiments/kernels/local.py`: `local_attention(q, k, v, mask_spec, block)` grouping queries into blocks that attend to their own and neighboring key blocks, shaped as a batched matmul over blocks so out-of-window work is not executed; handles a short last block and non-divisible lengths; block sizes 128, 256 and 512 supported
- [X] T034 [P] [US3] Implement `experiments/kernels/gas.py`: `gas_attention(q, k, v, mask_spec, block, selection)` that, per query block, gathers a fixed set of key/value indices (fixed pattern, not learned; research.md R8), runs dense attention on the gathered set, and scatters results back; index building, gather and scatter all count toward timing
- [X] T035 [US3] Implement `experiments/kernels/bench.py`: read shapes (heads, head dim, hidden) from `audit.json`; for each implementation and length run the correctness cases from T030 against `reference`, then time the implementation with mask construction, routing, padding, copies and allocations inside the timed region using `results.summarize`; record `score_matrix_bytes_analytical` and, per condition through `runner.run_condition`, `peak_rss_bytes` as separate fields; compute the executed-work exponent by fitting time against length and give the verdict `less_work`, `same_work` or `not_determined`. Depends on T031, T032, T033, T034, T008, T005
- [X] T036 [P] [US3] Implement `experiments/floor.py`: from `profile.json`, per length `floor_ms = total - attention_score_value` and `floor_fraction`, plus the analytical `8 * (1 - f)` bound for 4K versus 512, labeled `estimate` and `analytical` respectively and stored in separate fields (research.md R9). Fail with a clear message if `profile.json` is absent
- [X] T037 [US3] Register the `kernels` command in `experiments/cli.py` per contracts/cli.md (`--lengths`, `--impls`, `--block-sizes`, `--selection-sizes`): write `kernels.json`, and `floor.json` when `profile.json` exists (otherwise list the floor under `not_run`). Depends on T035, T036, T010

- [X] T049 [US3] GPU reference tooling (research.md R15): `experiments/gpu.py`, the `gpu-reference` command, `experiments/setup_gpu.ps1` (separate `.venv-gpu`, CUDA build tried newest first), `.venv-gpu/` in `.gitignore`, results rule 7 amended so `report` may cite `gpu_reference.json` only in the reference-hardware note
- [X] T051 Instrumentation A/B test (research.md R16, added after the full run): `experiments/abtest.py`, the `abtest` command, tests in `tests/experiments/test_profile.py`; report states the profiled/clean gap, the A/B ratios, the load-peak memory floor and `native_materialization`
- [ ] T052 Run `abtest --run-id full` on the measuring machine after `kernels`, then `report --run-id full`; if a mode other than clean changes p50 by more than the canary spread, document which one and how the component shares are affected
- [X] T053 Decision-head fast path (research.md R17): `profile` keeps the head's fast path and attributes its operators, `head_path` per item; `--mha-fastpath on|off` variant option with one setting per run; `abtest` modes `fastpath_off` and `labels_encoder`; report states the head path, variant runs and the fast-path implication; tests in `tests/experiments/test_fastpath.py` with an even-head fixture
- [ ] T054 On the measuring machine: `abtest --run-id full --modes clean,fastpath_off,labels_encoder,labels` (keep the first abtest.json as abtest-r16.json), re-run `profile --run-id full` with the fixed profiler, run the variant (`audit`, then `sweep --lengths 512,2048,4096,8192 --questions 1,10 --batch-sizes 1,8`, then `report`) as `--run-id full-nofastpath --mha-fastpath off`, then `report --run-id full`
- [ ] T050 Run `setup_gpu.ps1` and `gpu-reference --run-id full` on the RTX 4060 after the CPU sweep and profile finish; apply the interpretation rule in research.md R15

**Checkpoint**: User Story 3 works with only US1 outputs (`audit.json`) available; `floor.json` needs US2 outputs. T030 passes.

---

## Phase 6: User Story 4 - Ranked Bottleneck Report (Priority: P4)

**Goal**: One report combining every result, ranking bottlenecks by measured share, giving a verdict on each plan assumption, and listing everything that was not run.

**Independent Test**: Build a report from hand-written result files: rankings trace to items, every statement is tagged, missing inputs appear under `not_run`, and each assumption has a verdict.

### Tests for User Story 4

- [X] T038 [P] [US4] Write `tests/experiments/test_report.py` using small handwritten `manifest.json`, `audit.json`, `sweep.json`, `profile.json`, `kernels.json` and `floor.json`: bottlenecks are ordered by share of clean end-to-end p50 at each length; every ranking entry has a non-empty `derived_from`; each of the plan's assumptions (attention dominates near 2K; historical ~33 ms at 512 tokens; local layers skip out-of-window work) gets `confirmed`, `contradicted` or `untested`; every statement in `report.md` begins with `[measured]`, `[estimated]` or `[hypothesized]`; `report` fails when `manifest.json` or `audit.json` is missing but lists absent sweep/profile/kernel files under `not_run`; `unsupported` and `failed` items are listed with reasons; commands from `commands.txt` appear under `reproduce`; the report contains a `limitations[]` entry for the thread-count limit and lists a condition with `p95 / p50 > 1.5` under `high_variance[]`; the report contains an `audit_summary`, a `memory_feasibility[]` entry for every length (analytical and measured bytes in separate fields, whether the full score matrix fits, and the path that avoids it), and a `cost_floor[]` entry beside each ranking tagged `[estimated]`; a length with no memory measurement is listed under `not_run` and not omitted; results with `beyond_configured_max_len: true` are marked as cost-only in the text

### Implementation for User Story 4

- [X] T039 [US4] Implement `experiments/report.py`: load the run directory; rank components by measured share of clean p50 per length using only `measured` items; compute assumption verdicts (attention dominance: compare `attention_score_value` share to the rest across lengths; the 512-token latency against ~33 ms with the difference stated, preceded by the statement that the ~33 ms figure's hardware is unstated and most likely GPU (so a larger CPU number is not reported as a regression); pre-registered hypotheses from research.md R13 each get `confirmed`, `contradicted` or `untested`; executed-work verdict from `audit.json`); tag every statement `[measured]`, `[estimated]` (floor, analytical memory) or `[hypothesized]`; list `not_run`, unsupported lengths and failures with reasons; copy `commands.txt` into `reproduce`; write `phase3_implications` stating which prototype directions the data supports, weakens or leaves open, each tied to a measured cost. Also assemble the three deliverable sections FR-026 names: `audit_summary` (layer schedule, positional capacity, input cap, head depth, fallbacks, from `audit.json`), `memory_feasibility[]` (per length: analytical score-matrix bytes and measured peak RSS in separate fields, whether the full score matrix fits in available RAM, and which measured path avoids materializing it, from `sweep.json` and `kernels.json`), and `cost_floor[]` (from `floor.json`, tagged `[estimated]`, shown next to the ranking). Mark results with `beyond_configured_max_len: true` as cost-only. List conditions with `drift.flagged` and state the canary's maximum deviation (research.md R14); state that `profiled_to_clean_ratio` includes drift and that the T029 overhead check measures instrumentation overhead. Write `limitations[]`, always including the fixed-thread-count limit from research.md R12 with the recorded thread count and `hybrid_cores`, and `high_variance[]` for conditions with `p95 / p50 > 1.5`; both appear in `report.md`. Depends on T005
- [X] T040 [US4] Emit `report.json` and `report.md` from `experiments/report.py` per contracts/results.md rules 5 and 6 (`derived_from` references as `file#index`), and register the `report` and `all` commands in `experiments/cli.py` (`all` runs manifest, audit, sweep, profile, kernels, report in order with shared options). Depends on T039, T019, T027, T028, T037

**Checkpoint**: All four stories work together; T038 passes; `all` produces `report.md` on the fixture.

---

## Phase 7: Polish & Cross-Cutting

- [X] T041 [P] Add a short "Running the Phase 1 measurements" section to `README.md` linking `specs/001-cpu-path-audit/quickstart.md`; keep the README's claims consistent with the audit and research.md R13 (the ~33 ms figure carries its hardware caveat); state that the `sliding`, `chunk_cls`, `mosa` and `sqa` variants in the README table remain planned
- [ ] T042 Run `pytest tests/experiments -m "not slow"` and fix failures; confirm no test writes outside a temp directory or `experiments/results/`
- [ ] T043 Run quickstart.md steps 2 to 5 on the pinned English checkpoint on the measuring machine (network needed once); record any deviation from the expected results in the run's `report.md` and fix tooling problems found
- [ ] T044 Run `python -m experiments all --run-id full --revision reviewed` on the measuring machine, including 4,096 and 8,192 tokens; confirm each length is `measured`, `partial`, `unsupported` or `failed` with a reason, none silently substituted (SC-003)
- [ ] T045 Verify SC-005: `git diff --stat main -- laya/` is empty; if not, add the output-equivalence test required by FR-019 and document the instrumentation overhead
- [ ] T046 Verify constitution V: `git status --short` shows nothing under `experiments/results/`, no model weights, caches or credentials, and the review diff touches only `.gitignore`, `pyproject.toml`, `README.md`, `experiments/`, `tests/` and `specs/`
- [ ] T047 Resolve the open items in research.md R11 in the final `report.md` (target CPU identity, 8K feasibility, 512-token latency versus ~33 ms (hardware of the reference stated), attention crossover length, decision-head contribution), or state each as unanswerable with the reason

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: none. **Foundational (Phase 2)**: needs Setup; blocks everything after.
- **US1 (P1)**: needs Foundational. **US2 (P2)**: needs Foundational and US1's `inputs.py` (T017) and `tokens.py` (T016). **US3 (P3)**: needs Foundational and US1's `audit.json`; its floor needs US2's `profile.json`. **US4 (P4)**: needs Foundational; full value needs US1 to US3 outputs.
- **Polish (Phase 7)**: after the stories it covers.

### Within Foundational

T004, T006, T007 in parallel; T005 after T004; T008 after T005 and T007; T009 after T008; T010 after T005.

### Within Each Story

Tests before implementation. Library modules before the CLI registration task. In `experiments/profile.py` tasks T024, T025, T026 run in order (same file). In `experiments/cli.py`, registration tasks (T019, T027, T028, T037, T040) run in order.

### Parallel Opportunities

- US1: T011 to T014 together; T015 and T016 together.
- US2: T021 and T022 together.
- US3: T031 to T034 together (four kernel files); T036 in parallel with T035.
- After Foundational, a second contributor can start US3's kernel files (T031 to T034) without waiting for US1 or US2, since they need only reference shapes.

## Parallel Example: User Story 3

```text
Task: "Implement experiments/kernels/reference.py (T031)"
Task: "Implement experiments/kernels/dense.py (T032)"
Task: "Implement experiments/kernels/local.py (T033)"
Task: "Implement experiments/kernels/gas.py (T034)"
```

## Implementation Strategy

### MVP First (User Story 1 only)

1. Phases 1 and 2. 2. Phase 3 (US1). 3. Stop and validate: `manifest` and `audit` on the fixture, then on the real checkpoint (T020). This alone answers what the loaded model actually is and how many tokens it sees, which the research plan calls out as the first trap.

### Incremental Delivery

Add US2 for latency and cost curves, then US3 for the savings check, then US4 to turn results into the ranked decision. Each increment produces usable result files without the next one.

## Notes

- `tests/__init__.py` exists (added in Phase 2) so the test package imports as `tests.experiments` and never shadows the `experiments` package; `pyproject.toml` sets pytest `pythonpath = ["."]` for the same reason.
- No task edits `laya/`. If a needed probe cannot be reached by wrapping, stop and record the finding, then add it off by default with an equivalence test (plan contingency, FR-017 to FR-019).
- Synthetic-model timings from the fixture only exercise the tooling; only runs on the pinned checkpoint on the measuring machine are reported.
- Commit after each task or logical group. Do not commit anything under `experiments/results/`.
