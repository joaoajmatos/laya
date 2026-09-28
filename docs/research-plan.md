# Laya Context Extension Research Plan

Extending Laya's encoder-only models (ModernBERT-large, mmBERT) to larger context windows
while preserving single-pass scoring speed and decision quality.

## Motivation

- Laya does 33 ms / decision at 512 tokens with full attention
- Beyond ~2K tokens, encoder O(n²) attention dominates runtime
- Decoder models (Qwen3, Jev) win at >4K but are slower at small sizes
- Can we push the encoder further with sparse attention?

## Hypothesis

Chunk-based sparse attention with learnable content sparsity can reach 4K–8K context at
roughly the same cost as 512-token full attention, with minimal accuracy loss on scoring tasks.

## Phase 1 — Instrument Laya's cost profile

_GPU required: No. CPU is fine for timing breakdowns at all lengths._

1. **Benchmark attention cost across context sizes**
   - Model: Laya's English checkpoint (ModernBERT-large, native 512 context)
   - Vary input length: 128, 256, 512, 1024, 2048, 4096, 8192
   - Measure: total forward time, time per attention layer, time in rest of model
   - Use Laya's `predict` and `predict_batch`, on CPU and GPU
   - Key question: at what n does attention dominate? 512? 1024? earlier?

2. **Profile layer-level spend**
   - Hook into the ModernBERT forward pass to time each block
   - Separate attention cost from MLP, layer norm, and other ops
   - Separately time the option-head projection per question

3. **Find the ceiling**
   - Measure Laya multilingual with `max_len=8192`: throughput at 512 vs 4096 vs 8192
   - Identify the inflection point where latency becomes unacceptable

**Deliverable:** ms vs context length chart, broken down by component.

## Phase 2 — Build benchmark tasks

_GPU required: No. Data creation and labeling are offline work._

Create tasks the fork will be measured against. These live in the new repo.

4. **Short context baseline** (1–2 days)
   - Classification dataset: 5–10 questions, short states (<300 tokens each)
   - Source: email triage, support ticket routing, sentiment on short reviews
   - 200+ examples, ground-truth labels
   - Run through Laya full attention to establish baseline accuracy + calibration

5. **Medium context task** (1–2 days)
   - States in the 1000–3000 token range
   - Example: procurement records with criteria (same pattern as C17/C21 but owned data)
   - 3–5 questions per state
   - 100+ examples with gold labels

6. **Long context task** (1–2 days)
   - States in the 4000–8000 token range
   - Example: concatenated documents, multi-page reports, or full chat transcripts
   - 3–5 questions per state
   - 50+ examples with gold labels

## Phase 3 — Sparse attention prototypes

Implement in the Laya fork, starting with the ModernBERT attention module.

### Baseline: Fixed sparse (Longformer-style)

7. **Sliding window + global tokens** (2–3 days)
   _GPU: No. Attention mask changes only, no weight updates._
   - Replace full bidirectional attention with a fixed-size local window (128, 256, 512)
   - Keep first k heads global (attend to everything) — e.g. 2 of 16 heads
   - Prepend learnable global token embeddings (k=16–32) for cross-chunk communication
   - State content read through global summary; question attends to global tokens
   - Verify against full attention on random inputs

### Primary experiment: Content-based dynamic sparsity

8. **Chunk/CLS pattern** (3–4 days)
   _GPU: Required. CLS token embeddings are randomly initialized and need training._
   - Split the state into fixed-size chunks (256 tokens each)
   - Add a CLS token per chunk that encodes a summary of that chunk
   - Question tokens attend only to the chunk CLS tokens (not the raw state tokens)
   - This is the architecture from the length generalization paper (2510.17196)
   - Compare: accuracy at 2K, 4K, 8K vs full-attention baseline at 512

9. **MoSA-style dynamic token selection** (3–4 days, optional)
   _GPU: Required. The routing centroids need training, and mask computation is expensive on CPU at scale._
   - Each head independently selects top-k tokens from the state to attend to
   - Uses expert-choice routing (k-means clustering of keys)
   - Different heads can specialize on different state regions
   - Compare against chunk/CLS on accuracy and speed

10. **SQA: Reduce query heads** (1 day)
    _GPU: No. Masking out query heads is architectural, no training needed._
    - Test with 1, 2, 4 query heads vs the default 16
    - Measure accuracy vs speed trade-off
    - Hypothesis: scoring needs fewer heads than full-text modeling

11. **ASEntmax: Learnable sparse softmax** (1–2 days, optional)
    _GPU: Optional. Activation swap is inference-only; fine-tuning the α parameter would need GPU._
    - Replace softmax with α-entmax that learns to assign exact zeros
    - Can be combined with any of the above patterns
    - Potentially better calibration for downstream scoring

## Phase 4 — Evaluate

_GPU required: Recommended but not mandatory. CPU inference at larger contexts is slow but possible for final scoring runs on small benchmarks (50–100 rows). The decoder comparison (step 13) requires Qwen3-0.6B which runs on CPU._

12. **Run all variants against all benchmarks**
    - Variants:
      - Full-attention baseline (native Laya, 512 context)
      - Sliding window + global (128/256/512 window)
      - Chunk/CLS (256-token chunks)
      - Chunk/CLS + sliding (hybrid)
      - MoSA (if implemented)
      - SQA variants (if implemented)
    - At context sizes: 512, 1024, 2048, 4096, 8192
    - Metrics per cell:
      - Classification accuracy
      - Expected calibration error (ECE)
      - Latency (ms per batch of N)
      - Peak memory

13. **Compare against decoder with KV cache**
    - Same tasks through Qwen3-0.6B with `score_shared` (state prefilled once)
    - At which context size does the decoder become faster than the best sparse encoder?
    - Which gives better calibration at each context size?

14. **Optional: Lightweight fine-tuning** (2–3 days)
    _GPU: Required. LoRA requires torch autograd on 421M parameters._
    - If accuracy drops >5% at 4K+: LoRA-tune attention projections
    - Dataset: ~200 long-context scoring examples from benchmarks above
    - Single GPU, ~2 hours
    - Compare fine-tuned sparse vs full attention baseline

## Phase 5 — Decision

15. **Does any sparse variant achieve all of:**
    - ≤10% accuracy drop at 4K context vs 512-token full attention?
    - ≤2x latency at 4K vs 512-token full attention?
    - Better latency than the decoder (Qwen3 + `score_shared`) at the same context size?

    If yes → publish the fork as a research artifact with benchmarks.  
    If no → document the ceiling and conclude that decoder-with-KV-cache is the correct
    architectural choice for long-context scoring.

## Reference

### Central to the experiment (architecture inspiration)

- **Chunk-based sparse + CLS (2026)** — Understanding and Improving Length Generalization in Hierarchical Sparse Attention Models. https://arxiv.org/abs/2510.17196
  - *Relevance: Primary. Chunk the state, encode each chunk via a CLS token, question reads from CLS summaries. This maps directly to the scoring use case and is the main experiment.*

### Secondary (alternative patterns to test)

- **MoSA (2025)** — Mixture of Sparse Attention: Content-Based Learnable Sparse Attention via Expert-Choice Routing. https://arxiv.org/abs/2505.00315
  - *Relevance: Secondary. Dynamic token selection per head instead of fixed chunk boundaries. If chunk/CLS works, MoSA may improve it further by letting heads specialize on different state regions.*
- **SQA (2025)** — Sparse Query Attention: Computationally Efficient Attention with Query Heads Reduction. https://arxiv.org/abs/2510.01817
  - *Relevance: Quick experiment. Test if we can reduce query heads from 16 to 2–4 with negligible accuracy loss. Scoring probably needs fewer heads than language modeling.*
- **ASEntmax (2026)** — Long-Context Generalization with Sparse Attention (adaptive α-entmax). https://arxiv.org/abs/2506.16640
  - *Relevance: Optional. Replace softmax with learnable sparsity — the model decides what to zero out. Composable with any pattern above. Potentially better calibration.*

### Context (background, not directly implemented)

- **NSA (2025)** — Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention. https://arxiv.org/abs/2502.11089
  - *Relevance: Reference. DeepSeek's approach combines coarse compression + fine selection with CUDA kernels. Validates that hybrid hierarchical sparsity works at scale, but the custom kernel dependency makes it impractical for our fork.*
- **Sparse Frontier (2025)** — The Sparse Frontier: Sparse Attention Trade-offs in Transformer LLMs. https://arxiv.org/abs/2504.17768
  - *Relevance: Reference. Systematic study confirming no single sparse strategy wins everywhere. Use to justify testing multiple patterns rather than betting on one.*
- **Token Sparse Attention (2025)** — Efficient Long-Context Inference with Interleaved Token Selection. https://arxiv.org/abs/2602.03216
  - *Relevance: Reference. Interleaved token selection (compress → attend → decompress per layer). Interesting but adds complexity per layer; chunk/CLS is simpler for our case.*

### Prior art

- **Longformer (2020)** — https://arxiv.org/abs/2004.05150
  - *Relevance: Baseline. Sliding window + global attention. We implement this as the fixed-sparse baseline to compare against.*
- **BigBird (2020)** — https://arxiv.org/abs/2007.14062
  - *Relevance: Baseline. Random + window + global attention. Reference for mask computation patterns.*
- **ModernBERT (2024)** — https://arxiv.org/abs/2412.13663
  - *Relevance: Architectural target. The model we're modifying. Understand its attention implementation before modifying it.*
- **Laya repo** — https://github.com/NandhaKishorM/laya
  - *Relevance: Fork source. We're building from this codebase.*
