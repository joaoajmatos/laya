# Research: CPU Path Audit and Measurement

Decisions for the open technical questions in the plan. No `NEEDS CLARIFICATION` markers remain.
Items marked *to verify* are hypotheses the tooling itself must confirm or contradict, and the
final report must label them as such.

## R1. Can `laya/` stay untouched?

**Decision**: Yes. Use zero edits to `laya/` in the initial implementation.

**Rationale**: Every needed measurement point exists outside the library.

- Length control: `predict_batch(..., max_len=N)` overrides the configured cap per call (`laya/agent.py:934`). `max_len` is a ceiling, not a target, so the input builder (R4) supplies text that fills it.
- Fallback detection: after loading and after each run, read `agent.device`, `agent.dtype`, `agent.amp_enabled`, `agent.cpu_fallback_count`, `agent.last_fallback_reason`, `agent._compiled`. The `CPU AMP` mode is chosen by the `LAYA_CPU_AMP` environment variable, which the manifest records.
- Stage timing: wrap `Agent._encode_state`, `Agent._forward`, `Agent._decode_answers` on the instance, and `collate_items` in the `laya.agent` namespace, with timers that call through to the originals. The real `predict_batch` still runs end to end, so there is no re-implemented copy of the logic to drift from it.
- Operator costs: `torch.profiler` and module forward hooks (R5).

**Alternatives considered**:
- Edit `laya/agent.py` to add timing points. Rejected as unnecessary. It adds diff surface and a burden to prove the native path is unchanged (FR-019).
- Re-implement the request path in the harness to time stages. Rejected: it could diverge from the real path and measure something else.

**Contingency**: if a later finding needs a probe that wrapping cannot reach, add it to `laya/` off by default, document its overhead, and test output equivalence (FR-017 to FR-019). Success criterion SC-005 is then checked by `git diff` on `laya/` being empty, or by the equivalence test.

## R2. How to run and isolate each measurement condition

**Decision**: Run each condition (length x question count x batch size) in a fresh subprocess started by `runner.py`. The parent collects a JSON result from the child's stdout or a temp file, and records exit status.

**Rationale**:
- Peak memory never decreases within a process (`PeakWorkingSetSize` / `PeakPagefileUsage` on Windows, `resource.getrusage` on macOS and Linux), so per-condition peaks need per-condition processes.
- An out-of-memory condition on macOS or Linux usually ends the process with a signal instead of a catchable exception. Windows has no OOM killer: the commit limit makes allocation fail (a catchable allocator error), the process can die with `STATUS_NO_MEMORY` (0xC0000017), or the pagefile grows and the condition slows into the time cap. A parent process is the only place all of these can be recorded (FR-015).
- On Windows, a working set can be trimmed under memory pressure, so peak working set alone can understate demand. Peak commit (`PeakPagefileUsage`) is recorded beside it, and a condition whose peak commit exceeds physical RAM is flagged `paging_suspected`; its timings are cost-only.
- It isolates allocator state, thread pools and cached kernels between conditions.

**Consequence**: model load happens once per subprocess, so model-load time is measured naturally and separately for each (FR-010). Steady-state timings start after warmup inside the child.

**Alternatives considered**: one long-lived process with a memory tracker. Rejected because it cannot give per-condition peaks or survive a kill.

## R3. What "native attention" actually executes

**Decision**: The audit reads the loaded encoder's configuration and modules instead of assuming. It reports layer types, the local window, global attention cadence, RoPE settings per layer type, positional capacity, attention implementation, decision-head layer count, and hidden and head sizes.

**Rationale**: the plan says the English encoder uses local attention of 128 with global layers every third layer, `max_position_embeddings=8192`, and `head_layers=2`. Those figures are background and must be confirmed from the loaded model (constitution III). `laya/common.py` builds the encoder with `attn_implementation="sdpa"` and `_apply_rope_config` normalizes RoPE across transformers versions, so the audit also records the transformers version that produced the settings.

**To verify**: in the Hugging Face ModernBERT SDPA path, the local layers may pass a full-sequence sliding-window mask to `scaled_dot_product_attention` instead of skipping out-of-window work. If so, native "local" layers still do dense-shaped computation at run time. The profiler's scaling behavior across lengths (R5) decides this, not the source code alone. It matters because it means the native model may not be cheaper than an all-global one by as much as its configuration suggests, and it changes what Phase 3 must beat.

**Decision-head observation** (from `laya/common.py:313`): the head's transformer layers run over the full sequence before marker positions are gathered, so head cost scales with sequence length. The component profile therefore reports head layers separately (FR-012), and the cost-floor estimate keeps them (FR-025).

## R4. Synthetic inputs with an exact total token count

**Decision**: `inputs.py` builds a request (one state string, one question, options) whose final model input has exactly the requested total length. Method: fix the question and options, compute the fixed token overhead with `build_sequence`, then grow a deterministic filler state until the tokenized state fills the remaining budget. Size it by tokenizing (binary search on filler units), then set `max_len` to the target. The token accounting record (R7) verifies the result, and the run fails if it misses the target.

**Rationale**: the total includes question, options, and special tokens (FR-005), and `max_len` alone does not force length. Filler comes from a seeded generator over ordinary English words, so it tokenizes like real text and not like a degenerate repeated token. A different seed gives a different document per repeat, so no state is reused across timed requests (the fresh-document condition).

**Several questions or documents**: `build_request(agent, total_tokens, seed, n_questions, options_per_question, n_states)` gives every question the same head length (question text and options are built from templates of equal token length) and gives all states the same filler budget. Every question row therefore reaches exactly `total_tokens`, batches have no padding by design, and a condition names one length. The accounting record holds one entry per row, and the build fails if any row misses the target.

**Alternatives considered**: random token IDs. Rejected because they bypass tokenization, which the timed path includes. Repeating one sentence. Rejected because tokenization cost and tokenizer merge behavior differ from natural text.

**Disclosure**: the cache policy for question tokens is recorded in the manifest. Laya's `_reuse_question_tokens` decorator caches fixed-schema question tokens during a call. The plan allows this if disclosed consistently; every run uses the same policy.

## R5. Cost attribution: components and executed work

**Decision**: two run types, kept apart (FR-013).

1. **Clean latency runs**: no hooks, no profiler. Only the outer timer around the public `predict` call.
2. **Profile runs**: `torch.profiler` (CPU activities, operator shapes) plus stage wrappers (R1). Module forward pre-hooks push `record_function` labels for encoder layers, attention modules, MLPs, norms, and head layers, so each profiled operator is attributed to a component. Operator names classify the rest: linear and matmul under an attention module are projections; `scaled_dot_product_attention` is score/value work; other modules map to MLP and normalization; wrapper timers give tokenization/collation, option scoring and decoding.

**Rationale**: forward hooks on a module give only that module's total, so projections and the SDPA op inside one attention module cannot be separated by hooks alone. The profiler sees operators. The report states what fraction of total profiled time the components explain and lists the unexplained remainder (SC-004).

**Executed-work check** (FR-023): for each implementation (native layers and the kernels in R8), run at several lengths and fit the scaling exponent of time and of profiler-reported FLOPs or bytes against length. A claimed sparse saving must show a lower exponent or lower measured cost at long lengths. A dense mask or zeroed head does not qualify. This is the test for the *to verify* item in R3.

**Overhead check**: profile runs are slower. The report states the ratio of profiled to clean total time per length, and uses the clean number for latency.

## R6. Repeats, percentiles, and warmup

**Decision** (defaults, all CLI parameters):

- Warmup: 3 requests, discarded. Detect and record whether the first timed request still differs from the median.
- Timed repeats: 30 per condition at 512 tokens and below, 20 at 1,024 to 2,048, 10 at 4,096 and 8,192.
- Every repeat uses a new seed, so a new fresh document.
- Report min, p50, p95, mean, standard deviation and the number of repeats. With 10 repeats a p95 is close to the maximum, so results with fewer than 20 repeats carry a `low_sample_p95` label.
- Long-length time limit: a per-condition wall-clock cap. A condition that cannot complete the minimum repeats within the cap is recorded as `partial` with the count it reached.

**Rationale**: p95 needs enough samples to mean something (spec: "repeated-run uncertainty"). At 8K on a small CPU a single request may take seconds, so fewer repeats are the only feasible choice, and the label keeps that visible. Thread count is held fixed within a comparison (manifest).

## R7. Token accounting

**Decision**: `tokens.py` returns, per request, `state_tokens_original`, `task_option_tokens`, `special_tokens`, `state_tokens_retained`, `padding_tokens`, `final_length`, plus the position limit and `truncated` flag. The fields are computed from the same functions Laya uses (`build_sequence` with `return_stats=True`, and `collate_items`) plus a separate tokenization of the full state, since `build_sequence` only returns the truncated ids. The record is rejected unless `special + task_option + state_retained + padding` equals the final tensor length.

**Rationale**: `build_sequence` budgets question and options before the state and truncates the state to `max_len - len(head) - 1` (`laya/common.py:187`). Only a comparison with an independent full tokenization shows how much state was dropped. "Padding" is reported apart from evidence: padding only appears when a batch holds sequences of different lengths, so single requests have none.

**Unsupported length rule**: a requested total above `max_position_embeddings` (from the audit) is refused before any model call, with the reason recorded. A total between the configured cap and positional capacity is allowed only because the harness passes an explicit `max_len`, and is labeled as extrapolating past the checkpoint's configured input cap. Whether long positions are usable is a quality question for later phases, and this phase measures cost only. Such results carry `beyond_configured_max_len: true` so the report cannot present them as validated context.

## R8. Kernel reference implementations

**Decision**: pure PyTorch implementations in `experiments/kernels/`, all with the same signature (`q, k, v`, mask spec, returns output), and one dense reference that defines the semantics:

- **reference**: fp32, explicit score matrix, explicit boolean mask, softmax, value product. A fully masked row returns zeros (a defined convention, since raw softmax over all `-inf` gives NaN). Every other kernel is compared to this.
- **dense**: `scaled_dot_product_attention` with the same mask, as in the native path.
- **local**: block-local attention. Queries are grouped into blocks and attend to their own and neighboring key blocks, shaped as a batched matmul over blocks so out-of-window work is not executed.
- **gas** (gather-attend-scatter): for each query block, gather a chosen set of key/value indices (selected by a fixed pattern in Phase 1, not learned), run dense attention on the gathered set, and scatter the results back.

Sequence shapes come from the audit (heads, head dimension, hidden size), across the swept lengths. Timings include mask construction, index building, padding, copies and allocations (FR-021).

**Correctness cases** (FR-022): padding at the end of a row, a boundary chunk shorter than the block size, chunk boundaries that do not divide the length, a query with no allowed keys, and a batch mixing all of these. Tolerance: absolute 1e-5 in fp32 against the reference, stated in the result.

**Rationale**: the goal is to learn whether a sparse implementation *can* be faster on this CPU (plan item 3), not to ship one. The selection rule in `gas` is deliberately fixed so that Phase 1 measures data movement cost, not selection quality.

**Implementation note (Phase 5)**: the sparse kernels move blocks into the batch dimension
(`[batch x blocks, heads, block, head_dim]`) before calling `scaled_dot_product_attention`. With an extra
fifth dimension, PyTorch used its slow reference path, and `local` was no faster than dense at 2,048 tokens
despite 5x less work; the 4-D layout was 10-20x faster with identical results. A `dense_masked`
implementation (the block-local pattern as a mask over full attention, as the native local layers run it)
is benchmarked beside them, so the report can show a mask alone is not a saving.

**Alternatives considered**: `torch.nn.attention.flex_attention` (CPU support and stability vary by torch version and would add a version dependency); custom C++ or Triton kernels (out of scope, and would not be portable).

## R9. Memory feasibility and the cost floor

**Decision**:
- **Memory feasibility**: two things. (a) Analytical: score-matrix bytes as `batch x heads x L x L x dtype-size`, computed and printed for each swept length, labeled analytical. (b) Measured: peak RSS from the per-condition subprocess for the real native run and for each kernel. The report states which measured path avoids materializing the full score matrix and how it knows (peak RSS versus the analytical size).
- **Cost floor**: from the component profile at each length, `floor = total - attention_score_value_time`. This is the time that remains if attention score/value work were free. Projections, MLPs, norms, head layers and data preparation stay. It is labeled an estimate. The plan's operation-count bound (`8 x (1 - f)` at 4K versus 512) is reported next to it as an analytical figure, and the two are not merged.

**Rationale**: constitution II requires keeping analytical counts, kernel timings and end-to-end numbers distinct.

## R10. Testing offline

**Decision**: `tests/experiments/conftest.py` writes a tiny checkpoint to a temp directory: a small randomly initialized ModernBERT config (few layers, small hidden size, a local/global schedule), a matching `DecisionModel`, a small trained-from-nothing tokenizer, and `rl_agent_config.json`, so `laya.Agent(path)` loads it without network access. Unit tests cover manifest fields, audit values against the tiny config, token accounting reconciliation, exact-length input building, percentile summaries, profiler attribution sums, kernel-vs-reference agreement, and report ranking. Integration tests on the pinned checkpoint are marked `slow`.

**Rationale**: the real checkpoint needs a download, and the tooling logic does not depend on trained weights. Timings on a tiny model are not measurements and are never reported.

**Risk**: the installed `transformers` version changes ModernBERT internals (`laya/common.py` already contains version shims). The audit records the version, and tests pin behavior of the tiny model on the installed version.

## R11. Open items carried to the report

These are questions the run must answer, listed so they are not forgotten.

- The target CPU is the researcher's Windows PC; its exact CPU model, core layout (including any performance/efficiency core split) and power scheme come from the manifest. If deployment targets another CPU class, the same commands rerun there and produce a separate result set.
- Does the loaded local attention skip out-of-window work, or run dense masked SDPA? (R3, R5)
- At what length, if any, does attention score/value time overtake the rest of the forward pass? The plan assumes near 2K; the operation count in R13 predicts it does not happen by 2K.
- Is 8,192 feasible on this machine's RAM for the native model?
- What is the measured 512-token latency, compared with the ~33 ms historical reference? That reference's hardware is not stated; Laya's own runtime message gives ~35 ms on GPU and ~200-500 ms on CPU, so it is most likely a GPU figure and the comparison says so first.
- Does the decision head's full-sequence pass contribute enough cost to change which prototype is worth building first?

## R12. Thread count

**Decision**: One fixed intra-op thread count for the whole run, passed to every condition's child process. The default is the physical core count, which is PyTorch's own default. On a CPU with performance and efficiency cores, the researcher may pass `--threads` with the performance-core count. There is no CPU affinity pinning and no thread-count sweep.

**Rationale**: How latency scales with threads is not a Phase 1 question. The bottleneck ranking and the attention crossover only need every condition measured under the same setting, and a fixed, recorded value gives that. A thread sweep would multiply run time for an answer that does not change which prototype to build.

**Limitation (stated in the report)**: On a hybrid CPU the Windows scheduler can place threads on efficiency cores, which mostly inflates p95 and run-to-run spread. Absolute latencies hold for the recorded thread count on this machine only and may be pessimistic compared with a tuned deployment. Relative results (component shares, scaling exponents, crossover length) are less affected, because every condition shares the same setting.

**Guard**: The report flags any condition with `p95 / p50 > 1.5` as `high_variance`, so scheduling noise is visible and not mistaken for a trend.

## R13. Pre-registered hypotheses from the audited shapes

Written on 2026-09-28 after the audit of the pinned checkpoint (run `smoke`) and before any
timing run, so the report can mark each `confirmed`, `contradicted` or `untested` instead of
explaining results after the fact. All numbers here are **analytical estimates**, not measurements.

**Shapes used (from `audit.json`)**: encoder 28 layers, hidden 1024, 16 heads of 64, MLP
intermediate 2624 (gated, so the input projection is 1024 x 5248); 10 global layers, 18 local
layers with a 128-token window; decision head 2 layers, 16 heads, hidden 1024, FFN 4096, over the
full sequence. Matrix-multiply parameters: about 12.2M per encoder layer (341M for 28) plus 25M in
the head, about 366M in total (embeddings do no matrix work).

**Operation count per request of L tokens**: linear layers about 2 x 366M x L FLOPs; attention
scores and value product about 4 x L^2 x 1024 FLOPs per full-sequence layer.

| L | Linear layers | Attention share if local layers skip out-of-window work (12 full layers) | Attention share if local layers run dense with a mask (30 full layers) |
|---|---|---|---|
| 512 | ~375 GFLOP | ~3% | ~8% |
| 2,048 | ~1.5 TFLOP | ~12% | ~26% |
| 8,192 | ~6.0 TFLOP | ~35% | ~58% |

**Hypotheses**

- **H1**: attention score/value work does not dominate end-to-end latency at 2,048 tokens (share below 50%). Contradicts the plan's "attention dominates near 2K".
- **H2**: if attention overtakes the rest of the forward pass at any length up to 8,192, it happens only when local layers run dense masked attention (`executed_work_note = dense_masked`).
- **H3**: at 512 tokens on the i7-8700 (6 threads, fp32, AVX2), end-to-end latency is on the order of one second, far above the ~33 ms reference, because that reference is most likely a GPU figure.
- **H4**: projections and MLPs are the largest measured component at 512 and 2,048 tokens.
- **H5 (memory)**: if a full score matrix is materialized, one full-sequence layer at 8,192 tokens needs about 16 x 8192^2 x 4 B = 4.3 GB, about 8.6 GB with its softmax output, on top of about 1.7 GB of fp32 weights. 8,192 tokens at batch 1 then fits in 16 GB only with little else running, and batch 4 or 8 at 8,192 fails or pages.

**Implication if H1-H4 hold**: sparse attention alone offers limited gains up to 8K on this CPU.
Phase 3 would need to reduce linear-layer work (tokens processed, for example by compression or
chunk selection before the encoder) to change latency substantially.

## R14. Drift canary

**Evidence (smoke run, 2026-09-28)**: within each condition timings were flat (512 tokens: std 46 ms around a
1,555 ms median). Across the session the machine was not: the same 512-token requests measured 1,758 ms in the
overhead check that ran last, about 13% slower, and the 2,048-token profile run (7.3 s) was faster than the clean
2,048 sweep (8.4 s) measured minutes earlier. The likely causes are heat or the CPU's turbo power budget; the tool
cannot read either on Windows without administrator rights.

**Decision**: re-measure a fixed short condition (512 tokens, one question, 5 repeats) in its own process before
each new length and once at the end of `sweep` and `profile`. Conditions whose surrounding canaries differ from the
first canary by more than 5% are flagged. Nothing is rescaled: flagged results stay as measured, and the report
lists them and treats their curves with that caveat.

**Rationale**: a multi-hour sweep compares lengths measured hours apart, so drift between conditions can bend
scaling curves and fake or hide a crossover. The canary costs about 15 s per length. The `profiled_to_clean_ratio`
also mixes instrumentation overhead with drift; the interleaved wrappers on/off check (T029) is the measure of
instrumentation overhead, and the report says so.

## R15. GPU reference check

**Question**: is the historical ~33 ms at 512 tokens a GPU figure (H3's explanation)?

**Decision**: a separate `gpu-reference` command times native Laya on the researcher's RTX 4060 at 512 and
2,048 tokens, one question, with Laya's own CUDA defaults, and writes `gpu_reference.json`. It runs from a
separate environment, `.venv-gpu`, with a CUDA build of PyTorch, so the CPU environment stays CPU-only and a
GPU can never be used by accident in CPU runs. It runs after the CPU sweep, never during it.

**Scope**: FR-016 and constitution VI forbid GPU numbers standing in for the target CPU. The report cites this
file only in the note on the reference's hardware. It does not rank, profile or extrapolate from it.

**Interpretation rule, fixed before running**: a 512-token GPU p50 within 2x of 33 ms (under about 66 ms) is
consistent with the reference being a GPU figure; far above that leaves the reference's hardware unexplained
and the report says so.
