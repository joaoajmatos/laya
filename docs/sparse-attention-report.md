# Sparse Attention for Encoder Scoring at Long Context

## Extending Laya's decision models from 512 to 8K tokens

**Project:** Laya-Sparse

**Date:** 2026-09-28

**Status:** Draft

---

## Abstract

Laya achieves ~33 ms / decision at 512 tokens using full O(n²) bidirectional attention
in encoder-only models (ModernBERT-large, 421M; mmBERT, 322M). Beyond ~2K tokens,
attention cost dominates runtime, making long-context scoring impractical. This work
evaluates five sparse attention strategies to extend context to 4K–8K tokens while
preserving single-pass scoring speed and decision quality. The primary experiment
replaces full cross-attention with a chunk-based compression scheme: state tokens are
split into fixed-size chunks, each compressed into a CLS token, and question tokens
attend only to CLS tokens. We compare against fixed sparse baselines, full attention
at native context, and decoder-based scorers with KV cache reuse (Qwen3-0.6B).

---

## 1. Introduction

### 1.1 Problem

Decision models in the Laya family are encoder-only Transformers trained via RLCD to
produce calibrated probability distributions over decision options. Their architecture
is well-suited for System 1 scoring tasks: a single forward pass, no autoregressive
generation, bidirectional context, and ~33 ms latency at native context (512 tokens).

However, bidirectional attention scales as O(n²) with context length. For a 4K-token
state, the attention matrix is 64× larger than at 512 tokens. The model's maximum
position embedding (8192 for ModernBERT) supports longer input, but full O(n²)
attention at 4K–8K is ~1–4 seconds per decision — too slow for real-time scoring.

Decoder-based alternatives (Qwen3-0.6B, Jev) scale better asymptotically thanks to
KV caches, but underperform at small context sizes and introduce autoregressive
generation overhead per question.

### 1.2 Hypothesis

Sparse attention — where each token attends to a subset of the context rather than
all tokens — can reduce the asymptotic complexity to O(n · k) with k ≪ n, while
retaining sufficient information for decision scoring tasks. Specifically:

**Primary hypothesis.** A chunk/CLS compression scheme, where state tokens are
grouped into fixed-size chunks and each chunk is represented by a CLS token, reduces
cross-attention cost by O(state_len / chunk_size) while maintaining scoring accuracy
within 5% of full attention at 512 tokens.

**Secondary hypothesis.** Among training-free sparse alternatives, sliding window
with a small number of global attention heads provides the best accuracy–speed
trade-off for scoring tasks.

**Tertiary hypothesis.** Below a context-dependent crossover point (estimated at
~2K–4K tokens), the best sparse encoder is faster and better calibrated than an
equivalent decoder with KV cache; above that point, the decoder wins.

### 1.3 Contributions

1. A systematic comparison of five sparse attention patterns for encoder-only scoring
2. A pluggable attention framework for ModernBERT-family models
3. A benchmark suite spanning 512–8K token contexts with scoring tasks
4. A comparison against decoder-based scoring with KV cache reuse

---

## 2. Background

### 2.1 Encoder-only scoring with Laya

Laya uses ModernBERT-large (421M params, 16 heads, 8192 max positions, 64-token
sliding window in the base BERT, extended to full attention at inference) or mmBERT
(322M, multilingual). The model processes state + questions in a single forward pass
and produces calibrated scores via an option head projection.

### 2.2 Attention complexity

| Mechanism | Complexity | 512 tokens | 4096 tokens |
|-----------|-----------|-----------|-------------|
| Full bidirectional | O(n² · h) | 1× | ~64× |
| Sliding window (k=256) | O(n · k · h) | ~1× | ~7× |
| Chunk/CLS (c=256) | O(n · n_cls · h) | ~1× | ~4× |
| Decoder + KV cache | O(n² / 2 + m · n) | ~0.5× | ~1× per question |

### 2.3 Related work

**Longformer** (Beltagy et al., 2020) — Sliding window + global tokens. The fixed
sparse baseline for this work.

**BigBird** (Zaheer et al., 2020) — Sliding window + random + global attention.
Adds random connections for better information flow.

**Chunk-based length generalization** (Xiao et al., 2025, 2510.17196) — State split
into chunks with CLS tokens; question attends only to CLS. The primary inspiration
for this work.

**Mixture of Sparse Attention (MoSA)** (Journal et al., 2025) — Each head
dynamically selects top-k tokens via expert-choice routing (k-means on keys).
Content-based sparsity without fixed patterns.

**ASEntmax** (Peters et al., 2019) — Learnable sparse softmax that assigns exact
zeros. Can be combined with any attention pattern.

---

## 3. Research Design

### 3.1 Attention variants

| Variant | Description | Training | Window | Global | GPU |
|---------|-------------|----------|--------|--------|-----|
| Full (baseline) | Standard full bidirectional attention | None | n/a | n/a | No |
| Sliding window | Fixed-size local window + global heads | None | 128/256/512 | 1–2 heads | No |
| Chunk/CLS | State chunked, each chunk→CLS, question→CLS | LoRA | 256/chunk | n_cls | Yes |
| Chunk/CLS + sliding | Chunk/CLS + intra-chunk sliding window | LoRA | 256/chunk | n_cls | Yes |
| MoSA | Dynamic top-k token selection per head | Full | k=64/128 | dynamic | Yes |
| SQA | Reduced query heads (1, 2, 4 vs 16) | None | n/a | n/a | No |
| ASEntmax | Learnable sparse softmax (combined with above) | Optional | n/a | n/a | Optional |

### 3.2 Benchmark tasks

Short context (≤300 tokens): Email triage, support ticket routing, sentiment on short
reviews. 200+ examples, 5–10 questions per state.

Medium context (1000–3000 tokens): Procurement records with decision criteria (pattern:
C17/C21-style). 100+ examples, 3–5 questions per state.

Long context (4000–8000 tokens): Concatenated documents, multi-page reports, full chat
transcripts. 50+ examples, 3–5 questions per state.

### 3.3 Evaluation metrics

- Classification accuracy
- Expected calibration error (ECE)
- Latency (ms per batch of N)
- Peak memory (GB)
- Speedup vs full attention at same context length

### 3.4 Decoder comparison

Same benchmarks through Qwen3-0.6B with `score_shared` (state prefilled once, KV
cache reused across questions). Measures crossover point where decoder becomes
faster / better calibrated than sparse encoder.

---

## 4. Experiments

### Phase 1 — Cost profiling

Instrument Laya's forward pass to measure attention cost vs context length.

| Context | Total time | Attention | MLP | Other | Memory |
|---------|-----------|-----------|-----|-------|--------|
| 128     | —         | —         | —   | —     | —      |
| 512     | —         | —         | —   | —     | —      |
| 1024    | —         | —         | —   | —     | —      |
| 2048    | —         | —         | —   | —     | —      |
| 4096    | —         | —         | —   | —     | —      |
| 8192    | —         | —         | —   | —     | —      |

### Phase 2 — Full attention baselines

| Variant | Context | Accuracy | ECE | Latency | Memory |
|---------|---------|----------|-----|---------|--------|
| Full     | 512     | —        | —   | —       | —      |
| Full     | 1024    | —        | —   | —       | —      |
| Full     | 2048    | —        | —   | —       | —      |
| Full     | 4096    | —        | —   | —       | —      |
| Full     | 8192    | —        | —   | —       | —      |

### Phase 3 — Sparse attention experiments

#### 3a. Sliding window + global tokens

| Window | Global heads | Context | Accuracy | ECE | Latency | Memory |
|--------|-------------|---------|----------|-----|---------|--------|
| 128    | 1           | 2048    | —        | —   | —       | —      |
| 128    | 1           | 4096    | —        | —   | —       | —      |
| 128    | 1           | 8192    | —        | —   | —       | —      |
| 256    | 2           | 2048    | —        | —   | —       | —      |
| 256    | 2           | 4096    | —        | —   | —       | —      |
| 256    | 2           | 8192    | —        | —   | —       | —      |
| 512    | 2           | 2048    | —        | —   | —       | —      |
| 512    | 2           | 4096    | —        | —   | —       | —      |
| 512    | 2           | 8192    | —        | —   | —       | —      |

#### 3b. Chunk/CLS compression

| Chunk size | Shuffle | Context | Accuracy | ECE | Latency | Memory |
|------------|---------|---------|----------|-----|---------|--------|
| 128        | No      | 2048    | —        | —   | —       | —      |
| 128        | No      | 4096    | —        | —   | —       | —      |
| 128        | No      | 8192    | —        | —   | —       | —      |
| 256        | No      | 2048    | —        | —   | —       | —      |
| 256        | No      | 4096    | —        | —   | —       | —      |
| 256        | No      | 8192    | —        | —   | —       | —      |
| 512        | No      | 2048    | —        | —   | —       | —      |
| 512        | No      | 4096    | —        | —   | —       | —      |
| 512        | No      | 8192    | —        | —   | —       | —      |

#### 3c. Chunk/CLS + sliding window (hybrid)

| Chunk | Window | Context | Accuracy | ECE | Latency | Memory |
|-------|--------|---------|----------|-----|---------|--------|
| 256   | 128    | 4096    | —        | —   | —       | —      |
| 256   | 128    | 8192    | —        | —   | —       | —      |
| 512   | 256    | 4096    | —        | —   | —       | —      |
| 512   | 256    | 8192    | —        | —   | —       | —      |

#### 3d. MoSA (optional)

| k | Heads | Context | Accuracy | ECE | Latency | Memory |
|---|-------|---------|----------|-----|---------|--------|
| 64| 16    | 4096    | —        | —   | —       | —      |
| 64| 16    | 8192    | —        | —   | —       | —      |
| 128| 16   | 4096    | —        | —   | —       | —      |
| 128| 16   | 8192    | —        | —   | —       | —      |

#### 3e. SQA (optional)

| Query heads | Context | Accuracy | ECE | Latency | Memory |
|-------------|---------|----------|-----|---------|--------|
| 1           | 4096    | —        | —   | —       | —      |
| 2           | 4096    | —        | —   | —       | —      |
| 4           | 4096    | —        | —   | —       | —      |

#### 3f. ASEntmax (optional, combined with best variant above)

| α | Context | Accuracy | ECE | Latency | Memory |
|---|---------|----------|-----|---------|--------|
| 1.5| 4096   | —        | —   | —       | —      |
| 2.0| 4096   | —        | —   | —       | —      |

### Phase 4 — Decoder comparison

| Model | Context | Accuracy | ECE | Latency (1st Q) | Latency (subseq Q) | Memory |
|-------|---------|----------|-----|-----------------|-------------------|--------|
| Laya full 512 | 512 | — | — | — | — | — |
| Laya sparse (best) | 4096 | — | — | — | — | — |
| Laya sparse (best) | 8192 | — | — | — | — | — |
| Qwen3-0.6B KV | 4096 | — | — | — | — | — |
| Qwen3-0.6B KV | 8192 | — | — | — | — | — |

### Phase 5 — Fine-tuning (conditional)

If accuracy drop > 5% at 4K+: LoRA-tune attention projections on ~200 long-context
examples. Compare fine-tuned sparse vs full-attention baseline.

| Variant | Pre-tune Acc | Post-tune Acc | Δ | ECE Δ |
|---------|-------------|--------------|---|-------|
| Chunk/CLS 256 | — | — | — | — |
| Sliding 256+2G | — | — | — | — |

---

## 5. Results

*Main results will be added here as experiments complete.*

### 5.1 Attention cost profile

### 5.2 Sparse attention comparison

### 5.3 Best variant vs decoder

### 5.4 Ablation studies

---

## 6. Discussion

### 6.1 Interpretation

### 6.2 Limitations

### 6.3 Generalizability

---

## 7. Conclusion

---

## References

1. Beltagy, I., Peters, M. E., & Cohan, A. (2020). Longformer: The Long-Document
   Transformer. arXiv:2004.05150.
2. Zaheer, M., et al. (2020). Big Bird: Transformers for Longer Sequences.
   arXiv:2007.14062.
3. Xiao, G., et al. (2025). Length Generalization in Encoder-Decision Models.
   arXiv:2510.17196.
4. Zhang, Z., et al. (2025). Mixture of Sparse Attention: Dynamic Token Selection
   for Efficient Transformers.
5. Peters, B., Niculae, V., & Martins, A. F. T. (2019). Sparse Sequence-to-Sequence
   Models with α-entmax. arXiv:1904.02615.
6. Laya: Fast, non-autoregressive System 1 decision engine. https://huggingface.co/convaiinnovations/laya

---

## A. Appendix: Experiment environments

### A.1 Hardware

**Encoder experiments:**
- CPU: —
- GPU: —

**Decoder experiments:**
- CPU: —
- GPU: —

### A.2 Software versions

| Component | Version |
|-----------|---------|
| torch | — |
| transformers | — |
| Laya-Sparse | 0.1.0 |

### A.3 Benchmark datasets

*Datasets will be described here with sources, licenses, and statistics.*
