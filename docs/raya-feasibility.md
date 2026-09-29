# Raya: Rust Native Decision Model — Feasibility Evaluation

**Date:** 2026-09-29
**Status:** Draft — pre-implementation analysis, revised against Phase 1 measurements
**Scope:** Whether a from-scratch Rust implementation of a ModernBERT-like encoder plus decision head
(Raya) can outperform Laya on CPU for 4K-8K token decisions. Laya stays the experimental sandbox:
architectures, attention patterns and compression are validated there before anything is ported.

Numbers marked **measured** come from Phase 1 run `full` (and variant run `full-nofastpath`) on the
measuring machine: Intel i7-8700 (6 cores, AVX2, no native bf16), 16 GB RAM, Windows 11, torch 2.14 CPU,
fp32, 6 threads, English checkpoint at revision `55cf4c4`. Sources: `experiments/results/full/report.md`
and `specs/001-cpu-path-audit/research.md` (R13-R17). Numbers marked **estimated** are derived from
those measurements. Numbers marked **unverified** have no source in this repository yet.

---

## 1. Current bottlenecks (native Laya on CPU)

### 1.1 Latency (measured, one document, one question)

| Tokens | p50 | Note |
|---|---|---|
| 512 | 1.84 s | configured input cap |
| 2,048 | 8.82 s | cost-only beyond the cap |
| 4,096 | 21.2 s | cost-only |
| 8,192 | 75.2 s | 8 of 10 repeats before the time cap |

Multi-question requests re-encode the document per question: 10 questions take 8.6-9.2x one question.
Batching barely helps on this CPU: 8 documents take 6.6-9x one.

### 1.2 Where the time goes (measured, profile with the decision head on its native path)

| Component | 512 | 2,048 | 8,192 |
|---|---|---|---|
| MLP and norms (28 gated MLPs: 1024->5248, 2624->1024) | 48% | 32% | 15% |
| Attention projections (QKV, output, RoPE) | 28% | 20% | 9% |
| Encoder attention score/value (10 global + 18 local layers) | 12% | 24% | 39% |
| Decision head (2 layers over the full sequence) | 12% | 24% | 37% |
| Tokenization, collation, decoding, everything else | <1% | <1% | <1% |

- **The local layers do full-length work.** The 18 local layers (128-token window, +/-64 per side) hand
  SDPA a full L x L mask; their attention time grows like the global layers' (exponent 1.84 vs 1.85).
- **The decision head is the hidden quadratic, and it is on the slow path.** Its `nn.TransformerEncoderLayer`
  takes PyTorch's inference fast path, which on CPU runs native multi-head attention and materializes the
  full score matrix. At 8,192 tokens each head layer takes ~14.8 s against ~1 s for an encoder layer.
- **Python is not the bottleneck.** Everything outside PyTorch operators is under 1% of time at 512 and above.

### 1.3 Operation counts (estimated from the audited shapes)

| | 2,048 tokens | 8,192 tokens |
|---|---|---|
| Linear layers (projections, MLPs, head FFN): 2 x ~366M x L | ~1.5 TFLOP | ~6.0 TFLOP |
| Attention score/value, 30 full-length layers (28 encoder as run today + 2 head): 4 x L^2 x 1024 each | ~0.5 TFLOP | ~8.2 TFLOP |

The machine sustains ~270 GFLOP/s here (about a third of fp32 peak), which is normal for PyTorch GEMMs on
this CPU: the runtime is not wasting a large factor.

### 1.4 Memory (measured)

- One full fp32 score matrix at 8,192 tokens, all 16 heads, is 4.3 GB (one head: 268 MB).
- Native 8K, one document: peak 10.9 GB, about 8 GB above the model-load peak. With the head's fast path
  off (variant run): **3.2 GB**. The materialized matrices are the head's.
- Native 8K with 10 questions or batch 8 ran out of memory (asked for 43 GB and 34 GB). With the fast path
  off, the same requests stayed at 8.9 GB and 7.6 GB and only ran into the time cap.
- MLP GEMMs at these lengths are compute-bound; the 1.7 GB of weights is read roughly once per pass.

### 1.5 The opportunity is algorithmic, not the language

| Change | 8K latency | Evidence | Needs retraining |
|---|---|---|---|
| Today (native) | 75 s | measured | — |
| Decision head off PyTorch's fast path | 58 s | measured (variant run; same-process A/B: 0.69x at 8K, 0.81x at 2K) | no, outputs equal within 5e-7 |
| Real +/-64 local attention in the 18 local layers | ~40 s | estimated from the block-local kernel (85 ms vs ~1.1 s per layer at 8K) | no, if an exact kernel matches native |
| Sparse global and head attention | ~25 s | estimated | yes |
| Attention free (floor: projections, MLPs, head FFN) | ~20 s | estimated from the profile | — |

A Rust rewrite of the same architecture gains about what the first row gains, and PyTorch gets that
from one flag (`torch.backends.mha.set_fastpath_enabled(False)`). Everything below the first row is
algorithm, and all of it can be prototyped in Laya first.

---

## 2. Where a from-scratch Rust implementation wins

### 2.1 Architectural freedom (biggest lever)

Raya can choose depth, width and heads per token budget, make block-local or gather-attend-scatter the
native pattern, compress tokens before the MLP-heavy layers, and give the decision head only what the
task needs. **Every one of these changes the model and needs training and a quality evaluation.** The
freedom is real, but it is not specific to Rust: the same architectures can be trained and run in PyTorch.

### 2.2 Fused attention kernels (real, but already partly available)

A fused block kernel (load Q block, gather K/V blocks, scores, online softmax and value sum in registers,
write the output block) never materializes a score matrix and avoids `unfold`/`permute`/`reshape` traffic.

Measured reference points for such a kernel on the i7-8700, one layer, 16 heads, 8,192 tokens:

| Implementation | Time |
|---|---|
| Dense SDPA | 1,198 ms |
| Dense with the local mask (what native local layers do) | 1,482 ms |
| PyTorch block-local, block 128 (`experiments/kernels/local.py`) | **85 ms** |
| PyTorch gather-attend-scatter, 4 blocks of 128 | 124 ms |

A Rust kernel has to beat 85 ms to matter. The draft's inner loop (one 64-dim dot product per score,
~8 cycles each) comes out near 128 ms, slower than what PyTorch already does. A competitive kernel
needs FlashAttention-style tiling (small GEMM tiles per block pair), not per-score dot products.

### 2.3 Custom kernels for odd shapes (medium)

Small, odd-shaped matmuls (`[128, 384] x [384, 64]` per block) are where hand-tuned micro-kernels can
beat generic BLAS. Large GEMMs (the MLPs) are not: use MKL/Accelerate/OpenBLAS there, as PyTorch does.

### 2.4 Framework overhead (small)

Measured: under 1% of time outside operators at 512 tokens and above. A Rust runtime removes it, but
it is not a meaningful speed argument. The real benefits are footprint (no ~300 MB PyTorch install),
startup, and simpler deployment.

---

## 3. Headwinds

1. **Quality is the gate.** Compression between segments, a compressed head, and smaller models all
   need training (Laya's RLCD pipeline; this fork removed the training and evaluation harnesses) and a
   quality and calibration evaluation on independent benchmarks (Phase 2 of the research plan). Until a
   compressed model is shown to keep decision quality, there is nothing proven to port.
2. **Large GEMMs are hard to beat.** The MLPs and projections stay on a BLAS library; parity at best.
3. **Weights.** Same architecture: load Laya's safetensors directly. Any architectural change: retrain.
4. **No GPU path.** Training stays in PyTorch; Raya would be inference only.
5. **Target CPU is not settled.** The draft says Apple Silicon first; every measurement so far is on an
   x86 AVX2 desktop with no native bf16. Kernel choices (NEON vs AVX2/AVX-512, fp16/bf16 viability)
   depend on this and must be fixed before kernel work.

---

## 4. Proposed Raya architecture (to be validated in Laya first)

### 4.1 Principles

1. Sparse attention is native: block-local by default, dense only for short sequences.
2. Compress before you spend: reduce tokens between encoder segments so later layers see fewer tokens.
3. Fuse the hot path: scores, softmax and value sum in one kernel; norms and activations fused where possible.
4. BLAS for dense GEMMs, custom SIMD for attention.
5. Target CPU chosen first (see 3.5).

### 4.2 Configurable shape

```
Input: embeddings, vocab, max positions
Encoder segments (each configurable):
  Segment 1: N1 layers, no compression
  Compression 1: pooling / selection to a target token count
  Segment 2: N2 layers on compressed tokens
  (optional further compression and segments)
Decision head: attention over the compressed sequence + scorer
Attention per segment: block_local(W) | gas(S) | hybrid(global tokens + local) | full (short inputs)
```

### 4.3 Crate layout (proposed)

Unchanged from the first draft in substance: `model/` (config, forward, segments), `layers/` (embedding,
attention dispatcher, gated MLP, norm, head, compression), `kernels/` (BLAS wrappers, fused block-local
and gather-attend-scatter attention, SIMD dispatch for NEON/AVX2/portable), `weights/` (safetensors load,
Laya key mapping, export), `tokenizer/` (wrap Hugging Face `tokenizers`, which is already Rust),
`pipeline/` (sequence builder with the same marker layout as Laya), `cli/`, plus Criterion benches and
equivalence tests against Laya. Consider building on an existing Rust tensor runtime (for example
`candle`) and writing only the fused attention kernel, rather than a full runtime from scratch.

### 4.4 Compression, corrected arithmetic

Example: 10 layers on all tokens, compress 8,192 to ~64 pooled + selected tokens, 10 more layers, head.

- Linear work: the second segment runs on ~1% of the tokens, so its MLP and projection cost is almost
  zero. Total linear work is about **10/28 of today's (~64% less)**, not 28% less as the first draft said.
- Removing 8 of 28 layers and adding a compression step is a different model: it needs training.
- Estimated 8K latency on this CPU for such a model: ~7-8 s of linear work plus ~1 s of block-local
  attention in the first segment, i.e. **on the order of 10 s**, if quality holds.

| Method | Tokens out (from 8K) | Linear work saved in later layers | Quality risk |
|---|---|---|---|
| Mean-pool blocks (128 -> 1) | 64 + global | ~all of it | high |
| Learned weighted pool | 64 + global | ~all of it | medium, trained |
| Gather-attend-scatter selection | 64 + S x 128 | depends on S | lower |
| Hybrid: pool + selected tokens | 64 + S x 128 + pooled | most | low-medium |

---

## 5. Expected performance vs Laya (estimated, i7-8700 class CPU)

| Tokens | Laya today | Laya, head fixed (PyTorch) | Laya, head + real local attention (PyTorch) | Compressed architecture (PyTorch or Rust) |
|---|---|---|---|---|
| 512 | 1.8 s | 1.7 s | ~1.7 s | ~1.7 s (nothing to compress) |
| 2,048 | 8.8 s | 7.6 s | ~6.5 s | ~3-4 s |
| 8,192 | 75 s | 58 s | ~40 s | ~10 s |

The Rust-specific gain on top of each column is a few percent. What Rust adds is footprint, startup,
deployment without PyTorch, and room for hand-tuned fused kernels, which must beat the PyTorch block
kernel's 85 ms per layer at 8K to count.

**Feasibility on this hardware:** interactive (sub-second) 8K decisions are not reachable on a 6-core
AVX2 CPU with this model size by any of the above; even 512 tokens takes 1.7 s. Sub-second long-document
decisions need fewer tokens reaching the model (selection before the encoder), a much smaller model, or
different hardware (CPUs with bf16 matrix units, or a GPU).

---

## 6. Validation pipeline

Laya validates, Raya implements only what has been proven:

| Question | Status |
|---|---|
| Component cost at each length on the target CPU | **answered** (Phase 1) |
| Does SDPA on CPU materialize the score matrix? | **answered**: the materialization is the decision head's PyTorch fast path (R17) |
| Block-local and gather-attend-scatter kernel correctness and speed | **answered** (Phase 1 kernels: exact vs reference, 10-14x faster than dense at 8K) |
| Head off the fast path: speed and equivalence | **answered**: 0.81x at 2K, 0.69x at 8K, outputs equal within 5e-7 |
| Exact +/-64 local kernel equivalent to native | open (Phase 3 engineering) |
| Compression methods and their quality | open (Phase 2; needs benchmarks and training) |
| Compressed decision head quality | open (Phase 2) |
| Minimum viable model size | open (Phase 2-3) |
| Target CPU and precision (fp16/bf16/int8) | open (decision) |

---

## 7. Model sizing to test in Laya (all need training)

| Variant | Layers | Hidden | Heads | Notes |
|---|---|---|---|---|
| Raya-L (Laya shape) | 28 | 1024 | 16 | loads Laya weights; speed = Laya with fixes |
| Raya-M | ~20 | 768 | 12 | ModernBERT-base class; ~2.5x less linear work |
| Raya-S | ~14 | 512 | 8 | quality risk high |
| Raya-C | 10 + 10 compressed | 1024 | 16 | highest expected quality per unit of speed |

Parameter counts in the first draft (366M, 140M, 40M, 260M) should be recomputed from the chosen shapes
before being quoted; 366M is the matrix-multiply share of today's model, not its total.

---

## 8. Roadmap (revised)

1. **Finish Phase 1** (in progress): report, GPU reference check, review.
2. **Cheap wins in Laya, no retraining:** head off the fast path; exact +/-64 local kernel with an
   equivalence test against native.
3. **Phase 2:** independent benchmarks; compression and compressed-head ablations for quality and
   calibration; model-size sweep.
4. **Decide the runtime** only if a compressed variant holds quality: PyTorch, ONNX Runtime, or Rust
   (Raya), for a fixed target CPU.
5. **Raya build** (if chosen): runtime and weights, then fused kernels, then compression, each verified
   against the Laya reference.

The first draft's "Phase 0: 4-6 weeks" is largely done: Phase 1 answered the profiling, materialization
and kernel questions in two days of runs.

---

## 9. Open questions

| Question | Why it matters |
|---|---|
| Which compression keeps decision quality and calibration? | Decides whether Raya-C exists at all |
| Where should the compression boundary go? | Trade-off between quality and linear work |
| Can the head work on a compressed sequence? | Removes the head's quadratic cost entirely |
| Target CPU and precision | NEON vs AVX2/AVX-512; bf16/fp16 or int8 viability |
| Latency requirement per use case | Interactive vs batch changes what "feasible" means |
| Training pipeline | This fork removed Laya's training harnesses |

---

## 10. Market context (unverified)

The following claims come from the first draft and have no source in this repository. Each needs a
citation before it is relied on:

- Upstream Laya: ~28.3k GitHub stars, ~2.5k forks, PyPI v0.3.21, three production checkpoints, and
  integrations with LangChain, LlamaIndex, CrewAI, MCP, ONNX, Docker, NixOS, Spark and a TypeScript package.
- A proprietary competitor ("TypeSafe Jev") at $0.042 per 1M tokens, 236-276 ms latency, ECE 0.246.
- **Laya at 32.8 ms on a T4 GPU, ECE 0.081.** If upstream documents this, it settles the hardware of the
  ~33 ms reference (a T4 GPU) and the README, research plan and Phase 1 report should cite it.
- Users passing `max_len=8192` with the multilingual checkpoint.

The first draft's CPU latency figures (200-500 ms at 512, 1.5-8 s at 4K, "maybe OOM" at 8K) are
superseded by the measurements in section 1.1.

**Candidate segments** (unchanged, to be validated): self-hosters without GPUs, privacy-sensitive on-premise
deployments, edge and low-power devices, high-throughput CPU batch scoring, and existing Laya users in
agent frameworks. For all of them the product surface stays Laya's: `predict()` with state and typed
questions (choice, score, noul), marker-token scoring, and temperature-calibrated probabilities.

---

## 11. Conclusion

Raya is worth considering **only as the runtime for an architecture that has already proven itself in
Laya**. The measured costs say:

1. The biggest immediate win (the decision head's fast path, 19-31%) needs no new model and no Rust.
2. The next (real local attention, ~a further quarter at 8K) needs a kernel, which PyTorch can host.
3. The large win (compression between segments, ~an order of magnitude at 8K) is an architecture change
   whose risk is decision quality, and only training and evaluation in Laya can retire it.

Rust's own contribution is footprint, startup and deployment simplicity, plus fused kernels that would
need to beat 85 ms per layer at 8K. Build Raya after step 3 succeeds and the target CPU is fixed.
