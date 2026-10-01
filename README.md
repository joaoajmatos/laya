# Laya-Sparse

Research fork of [Laya](https://huggingface.co/convaiinnovations/laya) that explores sparse attention for running encoder-only decision models on **4K–8K token contexts on CPU**.

> **Status:** early research. Phase 1 (CPU path audit) is complete. Phase 2 (decision-quality benchmarks) is run on the development and calibration splits (synthetic data; GPU-scored quality is unverified against CPU, and the audit was by an AI assistant); see `docs/research-plan.md` for the findings. Sparse attention variants are not implemented yet.

## Why

Laya scores typed decisions over a state document with an encoder, without generating tokens. Upstream reports 32.8 ms (multilingual) and 39.5 ms (English) per single-question decision at 512 tokens, but on a Tesla T4 GPU, so that figure says little about CPU serving.

Phase 1 measured the unmodified English checkpoint at 512 tokens: **45 ms on an RTX 4060 and 1.84 s on the target CPU** (i7-8700). It also found that attention, including the 2-layer decision head, takes 45% of profiled CPU time near 2K tokens, 59% at 4K and 74% at 8K. That is the cost this fork tries to cut.

Planned approaches: chunked encoding with CLS tokens, sliding windows with global tokens, and content-based token selection. The goal is to reach 4K–8K context at latency comparable to the 512-token baseline. That target is a stretch goal, not a promise.

## Install

Requires Python 3.10+.

```bash
git clone https://github.com/joaoajmatos/laya-sparse.git
cd laya-sparse
pip install -e .
```

On Windows, `experiments/setup_windows.ps1` builds a CPU-only environment for the measurements.

## Usage

The public inference API is unchanged from upstream Laya:

```python
from laya import load

agent = load("convaiinnovations/laya")
result = agent.predict(
    "short state",
    {"q1": {"type": "noul", "instructions": "test?"}},
)
print(result["answers"])
```

## Attention variants

| Variant | Description | Status |
|---|---|---|
| `native` | Unmodified Laya: 10 global and 18 local (128-token window) encoder layers, plus a 2-layer head over the full sequence | Available |
| `sliding` | Fixed local window plus global tokens | Planned |
| `chunk_cls` | State split into chunks, each summarised by a CLS token | Planned (primary experiment) |
| `mosa` | MoSA-style dynamic token selection per head | Planned |
| `sqa` | Fewer query heads | Planned |

Only `native` runs today. The variant-selection API is not finalised.

## Reproducing the Phase 1 measurements

```powershell
python -m experiments all --run-id full --revision reviewed
```

Results are written to `experiments/results/<run-id>/` (git-ignored) and end in `report.md`. Full setup and expected results are in [`specs/001-cpu-path-audit/quickstart.md`](specs/001-cpu-path-audit/quickstart.md).

Timings from the tiny fixture model used in tests are not measurements, and GPU numbers never stand in for CPU results.

## Tests

```bash
pip install -e ".[experiments]"
pytest -m "not slow"
```

Tests marked `slow` need the real checkpoint and network access.

## Documentation

- [Research plan](docs/research-plan.md): objectives, workload definition and roadmap
- [Phase 1 spec](specs/001-cpu-path-audit/spec.md): the CPU path audit
- [Phase 2 spec](specs/002-decision-benchmark-baselines/spec.md): decision benchmarks against truncation, windowing and retrieval baselines
- [Sparse attention report](docs/sparse-attention-report.md): draft; claims are hypotheses until measured

## Contributing

Issues and pull requests are welcome. This project follows a spec-driven workflow; see [`AGENTS.md`](AGENTS.md) for the layout, conventions and process, and the [project constitution](.specify/memory/constitution.md) for the research and reproducibility rules.

## Acknowledgements

Built on [Laya](https://huggingface.co/convaiinnovations/laya) by ConvAI Innovations. This fork strips it to the core single-model inference path (no router, MCP, HTTP server, framework integrations, ONNX/TileLang export, or packaging extras) and adds experiment tooling.

## License

Apache 2.0, inherited from upstream Laya. See [LICENSE](LICENSE).
