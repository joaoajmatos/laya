# Feature Specification: CPU Path Audit and Measurement (Research Phase 1)

**Feature Branch**: `001-cpu-path-audit` (specification directory only; no branch created)

**Created**: 2026-09-28

**Status**: Draft

**Input**: User description: "Phase 1 of docs/research-plan.md: Audit and measure the complete CPU path. No experiment tooling exists yet, so build the measurement tooling and use it on native Laya: pinned run manifest and context audit, native latency profiler on the target CPU, and attention microbenchmarks with a savings check."

## User Scenarios & Testing *(mandatory)*

The user is the researcher deciding which sparse-attention, selection, or compression prototypes deserve Phase 3 effort. Nothing in `experiments/` exists yet, and `laya/layers/attention.py` is empty and unused by native inference, so this phase builds the measurement tooling and uses it on unmodified native Laya (English checkpoint). Changes to `laya/` are allowed for minimal instrumentation, but the native inference path stays behaviorally unchanged.

### User Story 1 - Pinned Run Manifest and Context Audit (Priority: P1)

The researcher runs an audit against the English checkpoint on the target CPU. It produces a manifest recording the exact environment and a report of what the loaded model actually is: which layers use local versus global attention, positional limits, decision-head depth, input cap, and any implicit device or precision fallback. It also records how every token in a request is accounted for, so later results can be trusted and reproduced.

**Why this priority**: Every later measurement and claim depends on knowing exactly what was run and how much context the model really processed. The research plan warns that input cap, positional capacity, and padding are easily conflated. Without this, latency numbers cannot be interpreted.

**Independent Test**: Run the audit on the target CPU with the pinned checkpoint. Check that the manifest is complete and that the reported layer schedule and limits match the loaded configuration. Feed requests of known composition and confirm the token accounting sums to the final model input length.

**Acceptance Scenarios**:

1. **Given** the English checkpoint on the target CPU, **When** the audit runs, **Then** the manifest records model revision, code revision, software versions, CPU model, available RAM, thread counts, numeric backend, precision, and concurrency, and states that the device is CPU explicitly.
2. **Given** a loaded model, **When** the audit inspects it, **Then** it reports per-layer attention type (local or global), local window size, positional encoding settings, positional capacity, configured input cap, decision-head layer count, and any implicit device or precision fallback observed.
3. **Given** a request with known state, question, and option text, **When** it is prepared for the model, **Then** the log gives original state tokens, task and option tokens, special tokens, retained tokens, padded length, and final input length, and the parts reconcile to the total.
4. **Given** a requested total length above what the model supports, **When** the request is prepared, **Then** the tooling fails or labels the run as unsupported with the reason, and never silently truncates or falls back to another device.
5. **Given** a run at 8,192 tokens, **When** token accounting is checked, **Then** task tokens, options, and special tokens are counted against the position limit.

---

### User Story 2 - Native Laya Latency Profile on CPU (Priority: P2)

The researcher runs a latency sweep of unmodified native Laya across total input lengths from 128 to 8,192 tokens. The primary case is one fresh document and one question, timed from tokenization through postprocessing. The researcher also gets a per-component cost breakdown, multi-question and batch results reported separately, and memory use. Together these show where CPU time actually goes.

**Why this priority**: This is the central measurement of the phase. It sets the baseline that later variants are compared against, and it tests the assumption that attention dominates at long lengths. It depends on User Story 1 for trustworthy token accounting and run metadata.

**Independent Test**: Run the sweep on synthetic inputs of controlled length. Confirm each length yields repeated end-to-end timings, a component breakdown that accounts for the measured total, and recorded memory. Confirm that induced failures are recorded and not dropped.

**Acceptance Scenarios**:

1. **Given** synthetic single-document, single-question requests at 128, 256, 512, 1,024, 2,048, 4,096, and 8,192 total tokens, **When** the sweep runs, **Then** each supported length reports repeated end-to-end timings with p50 and p95, and unsupported lengths are labeled as such.
2. **Given** a timed request, **When** it runs, **Then** the timing covers tokenization, encoding, scoring, and postprocessing, starts with no reusable document representations, and excludes warmup and model loading. Model load and any one-time compilation are reported separately.
3. **Given** the steady-state latency measurement, **When** it is collected, **Then** it comes from runs without profiling hooks that would perturb timing. Component breakdowns come from separate profiling runs, and the report identifies which is which.
4. **Given** a profiled request, **When** the breakdown is produced, **Then** it reports time for tokenization and collation, attention projections, attention score and value operations, MLPs and normalization, decision-head layers, option scoring, and decoding, and states how much of the total the components explain.
5. **Given** multi-question requests (2, 5, and 10 questions) and batched requests, **When** they are measured, **Then** they are reported separately from single-question latency and never substituted for it.
6. **Given** any run, **When** it finishes or fails, **Then** peak process memory is recorded, and out-of-memory or other failures at long lengths appear in the results with their length and cause.

---

### User Story 3 - Attention Microbenchmarks and Savings Check (Priority: P3)

The researcher benchmarks standalone reference implementations of dense, local or block, and gather-attend-scatter attention on realistic shapes. Each is checked for correctness against a dense reference with identical masking rules. The researcher also gets a memory feasibility estimate and a cost-floor estimate for the work that sparse attention alone cannot remove. This shows whether a sparse implementation can save real CPU time before any training effort.

**Why this priority**: It answers whether sparse attention can be faster on this CPU at all. It is more useful once the real cost profile from User Story 2 is known, and it only uses standalone code, so it does not block the other stories.

**Independent Test**: Run each reference implementation on the same inputs, compare it against the dense reference under the same mask, and record timings across sequence lengths. Confirm the report separates measured timings from analytical operation counts.

**Acceptance Scenarios**:

1. **Given** dense, local or block, and gather-attend-scatter reference implementations, **When** they run on realistic model shapes, **Then** their timings include mask construction, routing, padding, copies, and allocations, across the tested lengths.
2. **Given** a sparse reference implementation and a dense reference using the same mask rules, **When** they run on identical inputs, **Then** outputs agree within a stated tolerance, including cases with padding, chunk boundaries, and fully masked rows.
3. **Given** a dense mask or a zeroed head, **When** its speed is evaluated, **Then** the report checks that less work was actually executed, and does not credit a saving that only appears in the mask.
4. **Given** the tested lengths up to 8,192 tokens, **When** memory feasibility is assessed, **Then** the report states memory use, including whether materialized attention scores would exceed available memory, and which measured path avoids that.
5. **Given** the component costs from User Story 2, **When** the cost floor is estimated, **Then** the report gives the cost that remains if attention were free, covering retained projections, MLPs, and all decision-head layers, and labels it an estimate and not a prediction.

---

### User Story 4 - Ranked Bottleneck Report (Priority: P4)

The researcher receives a single deliverable combining the manifest, context audit, component cost curves, memory limits, and a ranked list of bottlenecks. From it the researcher decides which prototypes to build first, and whether any of the plan's assumptions (such as attention dominating near 2,000 tokens) are contradicted.

**Why this priority**: It turns the measurements into the decision the phase exists to support. It depends on the previous stories.

**Independent Test**: Review the report against the raw results. Every ranked bottleneck must trace to a measurement, and every claim must be labeled as measured, estimated, or hypothesized.

**Acceptance Scenarios**:

1. **Given** completed results from the previous stories, **When** the report is assembled, **Then** it ranks bottlenecks by measured share of end-to-end CPU time at each tested length, and each ranking cites its measurements.
2. **Given** the research plan's assumptions about where time goes (attention dominance near 2,000 tokens, the 512-token historical reference), **When** the report is written, **Then** it states whether each was confirmed, contradicted, or left untested.
3. **Given** the whole run, **When** the report is written, **Then** it lists every check that could not be run, every unsupported length, and every failure, with reasons, and includes the commands needed to reproduce the results.
4. **Given** the ranked list, **When** the researcher reads it, **Then** it says which Phase 3 prototype directions the measurements support, weaken, or leave open.

---

### Edge Cases

- A requested length exceeds positional capacity or memory: the run is labeled unsupported or failed with its reason, and is never replaced by a truncated or different-device result.
- Padding raises the input length without adding evidence: the accounting reports padded length and evidence length separately.
- Another process or thermal throttling perturbs timings: the manifest records concurrency, and repeated-run spread is reported so noisy results are visible.
- The model silently uses a different device or precision than requested: the audit detects and records it, and the run is marked non-comparable.
- A profiling run and a clean timing run disagree on total time: both are reported and the discrepancy is stated, not averaged away.
- Warmup or one-time compilation contaminates the first timed request: it is excluded from steady-state results and reported separately.
- A masked attention row has no allowed positions: reference implementations handle it with defined behavior, and the correctness check covers it.
- The target machine differs from the machine that produced the results: results carry hardware details so they cannot be mistaken for target-CPU numbers.

## Requirements *(mandatory)*

### Functional Requirements

**Manifest and context audit**

- **FR-001**: The tooling MUST record, for every run, model revision, code revision, Python, PyTorch, and Transformers versions, CPU model, available RAM, thread counts, numeric backend, precision, compiler settings, and concurrency.
- **FR-002**: The tooling MUST require and record an explicit CPU device, and MUST detect and record any implicit device or precision fallback.
- **FR-003**: The audit MUST report, for the loaded English checkpoint, each layer's attention type, local window size, positional encoding settings, positional capacity, configured input cap, and decision-head layer count, taken from the loaded model and not from documentation.
- **FR-004**: The tooling MUST log, per request, original state tokens, task and option tokens, special tokens, retained tokens, padded length, final input length, the number of questions, and the number of options per question, and the token counts MUST reconcile to the final model input length.
- **FR-005**: The tooling MUST count all tokens, including task, option, special, and any added tokens, against position limits, up to 8,192 total tokens.
- **FR-006**: The tooling MUST fail or explicitly label requests whose length is unsupported, and MUST NOT silently truncate, pad to a different length, or switch devices. A length is *unsupported* when it exceeds the positional capacity read from the loaded model. A length above the checkpoint's configured input cap but within positional capacity is measured and MUST be labeled as beyond the configured cap, because this phase measures cost only and says nothing about decision quality at that length.
- **FR-007**: Runs MUST be scoped to the English checkpoint. A second checkpoint is out of scope for this phase.

**Latency profiling**

- **FR-008**: The tooling MUST measure native, unmodified Laya at 128, 256, 512, 1,024, 2,048, 4,096, and 8,192 total tokens where supported, using synthetic or representative inputs. "Total tokens" is the final input length of each question row. In requests with several questions or several documents, every row MUST be built to that same length, so a condition names one length.
- **FR-009**: The primary measurement MUST be one fresh document with one question, timed from tokenization through postprocessing, with no reusable document representations at the start of each request, matching constitution principle VI.
- **FR-010**: Model loading and one-time compilation MUST be measured and reported separately from steady-state request latency, and warmup MUST be excluded from steady-state results.
- **FR-011**: The tooling MUST measure multi-question requests (2, 5, and 10 questions) and batched throughput separately from single-question latency, and reports MUST NOT use them as a substitute for it.
- **FR-012**: The tooling MUST produce a per-component time breakdown covering tokenization and collation, attention projections, attention score and value operations, MLPs and normalization, decision-head layers, option scoring, and decoding, and MUST report the fraction of total time the components explain.
- **FR-013**: The tooling MUST collect final latency without intrusive profiling hooks, and MUST collect component breakdowns in separate profiling runs identified as such.
- **FR-014**: The tooling MUST record repeated timings per condition, p50 and p95, spread across repeats, and peak process memory.
- **FR-015**: The tooling MUST record failures, out-of-memory events, and runs cut short by a time limit at long lengths with their length and cause, and MUST NOT drop them or replace them with results from a different configuration.
- **FR-016**: Any GPU measurement MUST be separately labeled and MUST NOT appear in, or substitute for, target-CPU results.

**Changes to the model code**

- **FR-017**: Changes to `laya/` MUST be limited to what measurement needs (for example, timing or inspection points), MUST leave native inference outputs unchanged, and MUST keep default behavior and public interfaces backward compatible.
- **FR-018**: Any instrumentation added to `laya/` MUST be off by default, and MUST have a measured overhead documented so that timing runs can show it does not distort steady-state latency.
- **FR-019**: The tooling MUST verify that outputs from the instrumented code path match the uninstrumented native path on the same inputs.

**Attention microbenchmarks**

- **FR-020**: The tooling MUST provide standalone reference implementations of dense, local or block, and gather-attend-scatter attention under `experiments/`, and MUST NOT modify `laya/layers/attention.py` or integrate sparse attention into the model.
- **FR-021**: Microbenchmarks MUST use realistic model shapes, and MUST include mask construction, routing, padding, copies, and allocations in measured time.
- **FR-022**: Each sparse reference implementation MUST be checked against a dense reference using identical mask semantics, covering padding, chunk boundaries, and fully masked rows, and results MUST state the tolerance used.
- **FR-023**: The tooling MUST verify, using scaling behavior and profiler evidence, that a claimed saving comes from less executed work, and MUST NOT credit a dense mask or zeroed head with a speed saving.
- **FR-024**: The report MUST quantify memory feasibility for materialized attention scores at the tested lengths, and MUST state which measured path avoids it.
- **FR-025**: The report MUST estimate the cost floor that remains if attention were free, from retained projections, MLPs, and all decision-head layers, and MUST label it as an estimate.

**Deliverable and reporting**

- **FR-026**: The phase MUST produce a pinned run manifest, a context audit, component cost curves, memory limits, and a ranked list of bottlenecks, and the final report MUST contain a summary of each: the audit, a per-length memory-feasibility statement, the cost floor beside the ranking, and the ranking itself.
- **FR-027**: Ranked bottlenecks MUST be based on measured shares of end-to-end CPU time at each tested length, and each ranking MUST cite its supporting measurements.
- **FR-028**: The report MUST label each statement as measured, analytically estimated, or hypothesized, and MUST state whether the plan's assumptions (attention dominance near 2,000 tokens, the historical 512-token reference) were confirmed, contradicted, or untested.
- **FR-029**: The report MUST list unsupported lengths, failed runs, and checks that could not be run, and MUST include the commands needed to reproduce each result.
- **FR-030**: Generated results, model weights, and caches MUST NOT be committed, and MUST be written to ignored locations, per constitution principle V.

### Key Entities

- **Run Manifest**: A record of one measurement session: model and code revisions, software versions, hardware, thread counts, numeric backend, precision, concurrency, and seeds. It is attached to every result.
- **Context Audit**: The report of the loaded model's actual structure: per-layer attention type, window size, positional settings and capacity, input cap, decision-head depth, and detected fallbacks.
- **Token Accounting Record**: A per-request breakdown of the input into state, task and option, special, retained, and padding tokens, reconciled to the final input length.
- **Latency Measurement**: Repeated timings for one condition (length, question count, options per question, batch size), with p50, p95, spread, peak memory, a flag for lengths beyond the configured input cap, and status (measured, unsupported, failed, partial).
- **Component Profile**: A breakdown of time by model component for one condition, with the fraction of total explained.
- **Microbenchmark Result**: Timing, memory, and correctness outcome for one attention implementation at one shape.
- **Bottleneck Ranking**: An ordered list of cost sources by measured share of CPU time at each length, with links to supporting measurements and a status of confirmed, contradicted, or untested for each plan assumption.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A second researcher can reproduce the manifest and audit from the recorded commands on the same machine class, and every field required by FR-001 and FR-003 is present with no missing values.
- **SC-002**: For every audited request, the token counts reconcile exactly to the final input length, and 100% of requests above supported limits are labeled as unsupported or failed, with none silently truncated.
- **SC-003**: The latency sweep gives a result for each of the seven target lengths, either measured with repeated timings and p50/p95, or explicitly marked unsupported, failed, or partial (time limit reached before the minimum repeats) with a reason.
- **SC-004**: Component profiles explain at least 90% of profiled end-to-end time at each measured length, or state the unexplained share.
- **SC-005**: Instrumentation changes leave native outputs identical to the uninstrumented path on all audit inputs, and steady-state latency with instrumentation off is not distinguishable from the unmodified code within the measured run-to-run spread.
- **SC-006**: Every sparse reference implementation agrees with its dense same-mask reference within the stated tolerance on all tested edge cases (padding, chunk boundaries, fully masked rows).
- **SC-007**: For each tested length, the report states measured peak memory and whether materialized attention scores fit in available memory.
- **SC-008**: The final report ranks bottlenecks at every measured length, each ranking traces to a measurement, and the plan's stated assumptions each have a confirmed, contradicted, or untested verdict.
- **SC-009**: The researcher can use the report alone to choose which Phase 3 prototype directions to build first, with each recommendation tied to a measured cost, and no recommendation resting on an unlabeled estimate.

## Assumptions

- The user is a single researcher working on the target CPU machine, and "target CPU" means the machine used for these measurements, recorded in the manifest.
- Model weights are resident in memory during timed requests, per the research workload.
- Only the English checkpoint is in scope. The multilingual checkpoint (mmBERT) is a separate later replication.
- Synthetic or representative inputs are acceptable in this phase, because the Phase 2 datasets do not exist yet. Measurements need controlled lengths, not task-quality labels, and quality is not measured here.
- Native Laya's current behavior and outputs are the reference. This phase does not change them.
- Lengths that native Laya cannot support (positional or memory limits) are reported as such, without being worked around.
- GPU measurements are optional, separately labeled, and not required for this phase.
- The standalone attention reference implementations are measurement tools. They are not model integrations, and selecting or wiring them into inference is Phase 3.
- The 33 ms at 512 tokens in the README is a historical reference to be re-measured, not an established result for this fork.
- Statistical comparison of quality between variants, calibration, and baselines (windowing, retrieval, decoder scoring) belong to Phase 2 and later, and are out of scope here.
- Dependencies: the pinned English checkpoint must be obtainable, and the constitution (v1.1.0) governs how measurements are reported.
