"""Shared plumbing for the golden generators: determinism, the fixture model, byte-stable writers.

Everything written here must be byte-identical across repeated runs, so no timestamps, hostnames or paths go
into a file. The laya-sparse commit SHA does (the vendoring rule needs it).
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

DEFAULT_OUT = Path("artifacts") / "raya-golden"
SEED = 0
REPO_ROOT = Path(__file__).resolve().parents[2]
#: Each case dir and each `meta.json` carries this, so a consumer can tell the layout generation.
SCHEMA_VERSION = 1


def make_deterministic(seed: int = SEED) -> None:
    """fp32, one thread, fixed seed, deterministic kernels where PyTorch supports it."""
    import torch
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.set_default_dtype(torch.float32)


def commit_info() -> Dict[str, Any]:
    from ..manifest import code_info
    info = code_info(REPO_ROOT)
    return {"laya_sparse_commit": info["git_sha"], "git_dirty": info.get("git_dirty"),
            "laya_diff_empty": info.get("laya_diff_empty")}


def versions() -> Dict[str, str]:
    import safetensors
    import tokenizers
    import torch
    import transformers
    return {"torch": torch.__version__, "transformers": transformers.__version__,
            "safetensors": safetensors.__version__, "tokenizers": tokenizers.__version__,
            "python": "%d.%d.%d" % sys.version_info[:3]}


def base_meta(kind: str, seed: int = SEED, **extra: Any) -> Dict[str, Any]:
    import torch
    meta = {"schema_version": SCHEMA_VERSION, "kind": kind, "seed": seed, "dtype": "float32",
            "mode": "eval", "threads": torch.get_num_threads(), "device": "cpu"}
    meta.update(commit_info())
    meta["versions"] = versions()
    meta.update(extra)
    return meta


def encoder_config(model) -> Dict[str, Any]:
    """The encoder config as a dict, without the machine-specific temp path transformers records in it."""
    cfg = model.encoder.config.to_dict()
    cfg.pop("_name_or_path", None)
    return cfg


def write_json(path: Path, obj: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    return path


def save_tensors(path: Path, tensors: Dict[str, Any]) -> Path:
    from safetensors.torch import save_file
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file({k: v.detach().contiguous().cpu() for k, v in sorted(tensors.items())}, str(path))
    return path


def _deterministic_tokenizer(out_dir):
    """Stand-in for the fixture's `_build_tokenizer`: the same WordPiece pipeline with an explicit vocabulary.

    The conftest trains its tokenizer, and the trainer's merge order breaks frequency ties differently from run to
    run (two builds gave vocabularies that differ in ~9% of the entries), so ids would not be reproducible. The
    word list, specials, normalizer and pre-tokenizer are the conftest's; only the vocabulary is built directly.
    """
    import string

    from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    from tests.experiments import conftest as fx

    punct = list(": , . ; ? ! ( ) [ ] { } \" ' - _ /".replace(" ", ""))
    letters = list(string.ascii_lowercase) + list(string.digits)
    vocab, seen = [], set()
    for tok in (list(fx.SPECIALS) + punct + letters + ["##" + c for c in letters] + sorted(set(fx._WORDS))):
        if tok not in seen:
            seen.add(tok)
            vocab.append(tok)
    tok = Tokenizer(models.WordPiece(vocab={t: i for i, t in enumerate(vocab)}, unk_token="[UNK]"))
    tok.normalizer = normalizers.BertNormalizer(lowercase=True)
    tok.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    tok.decoder = decoders.WordPiece()
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok, unk_token="[UNK]", pad_token="[PAD]", cls_token="[CLS]", sep_token="[SEP]",
        mask_token="[MASK]", model_max_length=fx.FIXTURE_ENCODER["max_position_embeddings"])
    fast.save_pretrained(out_dir)
    return fast


def fixture_checkpoint(root: str, hidden: int = 64) -> str:
    """Build the tiny offline checkpoint of tests/experiments/conftest.py (seeded) under `root`.

    Same model, config and weight seed as the conftest; only the tokenizer vocabulary is built deterministically.
    """
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from tests.experiments import conftest as fx
    original = fx._build_tokenizer
    fx._build_tokenizer = _deterministic_tokenizer
    try:
        return fx.build_fixture_checkpoint(root, hidden=hidden)
    finally:
        fx._build_tokenizer = original


class FixtureAgent:
    """Context manager: a `laya.Agent` on the fixture checkpoint in a temp dir, plus the dir path."""

    def __init__(self, hidden: int = 64):
        self.hidden = hidden

    def __enter__(self):
        import laya
        make_deterministic()
        self._tmp = tempfile.mkdtemp(prefix="laya-golden-")
        self.path = fixture_checkpoint(self._tmp, hidden=self.hidden)
        make_deterministic()
        self.agent = laya.Agent(self.path, device="cpu")
        self.agent.model.eval()
        return self

    def __exit__(self, *exc):
        shutil.rmtree(self._tmp, ignore_errors=True)
        return False


def total_bytes(root: Path) -> int:
    return sum(p.stat().st_size for p in Path(root).rglob("*") if p.is_file())


def iter_files(root: Path) -> Iterator[Path]:
    for p in sorted(Path(root).rglob("*")):
        if p.is_file():
            yield p
