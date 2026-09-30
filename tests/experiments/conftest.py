"""T006: a tiny, offline Laya checkpoint for the tooling tests.

The checkpoint is a randomly initialised ModernBERT encoder (3 layers, hidden 64, 2 heads,
alternating local/global attention) plus a `DecisionModel` head, with a WordPiece tokenizer
trained from scratch. `laya.Agent(path)` loads it with no network access.

Timings from this model only exercise the tooling. They are never reported as measurements;
the config carries ``"laya_sparse_fixture": true`` so result files can say so.
"""
import json
import os
import random

import pytest

FIXTURE_ENCODER = {
    "num_hidden_layers": 3,
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_attention_heads": 2,
    "global_attn_every_n_layers": 2,   # layers 0 and 2 global, layer 1 local
    "local_attention": 16,             # small window so short inputs already exceed it
    "max_position_embeddings": 2048,
}
FIXTURE_AGENT_CFG = {
    "encoder": "tiny-modernbert-fixture",
    "head_layers": 1,
    "max_len": 256,
    "head_max_len": 96,
    "temperature": [1.0, 1.0, 1.0],
    "act_costs": {"abstain": 0.1},
    "laya_sparse_fixture": True,
}

SPECIALS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]

_WORDS = (
    "the a an of to in and or is are was were be been it this that these those with for on at by "
    "from as not no yes true false question choice score noul answer option label level state "
    "document report customer order request refund delivery account payment invoice support "
    "ticket issue problem status update new old high low urgent normal late early open closed "
    "which what how why when who does should can will would could may might must shall "
    "red blue green yellow north south east west river mountain city village market harbor "
    "morning evening winter summer quiet loud small large quick slow bright dark warm cold "
    "statement holds describe select best matching rate from zero four one two three five six "
    "seven eight nine ten hundred thousand first second third last next previous every each"
).split()


def _corpus(n_lines=4000, seed=0):
    rng = random.Random(seed)
    for _ in range(n_lines):
        yield " ".join(rng.choice(_WORDS) for _ in range(rng.randint(4, 20)))
    yield ": , . ; ? ! ( ) [ ] { } \" ' - _ / 0 1 2 3 4 5 6 7 8 9"


def _build_tokenizer(out_dir):
    from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    tok = Tokenizer(models.WordPiece(unk_token="[UNK]"))
    tok.normalizer = normalizers.BertNormalizer(lowercase=True)
    tok.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    tok.decoder = decoders.WordPiece()
    trainer = trainers.WordPieceTrainer(vocab_size=600, special_tokens=SPECIALS, min_frequency=1)
    tok.train_from_iterator(_corpus(), trainer=trainer)
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok, unk_token="[UNK]", pad_token="[PAD]", cls_token="[CLS]",
        sep_token="[SEP]", mask_token="[MASK]", model_max_length=FIXTURE_ENCODER["max_position_embeddings"],
    )
    fast.save_pretrained(out_dir)
    return fast


def build_fixture_checkpoint(root, hidden: int = 64):
    """Write the tiny checkpoint into `root` and return its path.

    ``hidden=128`` gives the decision head 2 attention heads (Laya uses hidden // 64), an even count,
    so PyTorch's TransformerEncoderLayer fast path is taken as on the real model (research.md R17).
    """
    import torch
    from safetensors.torch import save_file
    from transformers import ModernBertConfig

    from laya.common import build_model

    os.makedirs(root, exist_ok=True)
    tok = _build_tokenizer(os.path.join(root, "tokenizer"))

    ecfg = ModernBertConfig(
        vocab_size=len(tok),
        pad_token_id=tok.pad_token_id,
        cls_token_id=tok.cls_token_id,
        sep_token_id=tok.sep_token_id,
        bos_token_id=tok.cls_token_id,
        eos_token_id=tok.sep_token_id,
        **dict(FIXTURE_ENCODER, hidden_size=hidden, intermediate_size=2 * hidden),
    )
    enc_dir = os.path.join(root, "encoder")
    ecfg.save_pretrained(enc_dir)

    with open(os.path.join(root, "rl_agent_config.json"), "w") as f:
        json.dump(FIXTURE_AGENT_CFG, f, indent=2)

    # build_model(pretrained=False) skips initialisation, so every tensor is filled here.
    model = build_model(FIXTURE_AGENT_CFG, encoder_dir=enc_dir, pretrained=False)
    gen = torch.Generator().manual_seed(1234)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.dim() == 1:
                p.fill_(1.0 if name.endswith("weight") else 0.0)
            else:
                p.copy_(torch.randn(p.shape, generator=gen) * 0.02)
        for name, b in model.named_buffers():
            if name.endswith("temperature"):
                b.fill_(1.0)
    state = {k: v.detach().clone().contiguous() for k, v in model.state_dict().items()}
    save_file(state, os.path.join(root, "model.safetensors"))
    return root


@pytest.fixture(scope="session")
def tiny_checkpoint(tmp_path_factory):
    """Path to the offline fixture checkpoint (built once per test session)."""
    return build_fixture_checkpoint(str(tmp_path_factory.mktemp("tiny-laya-fixture")))


@pytest.fixture(scope="session")
def tiny_agent(tiny_checkpoint):
    """A `laya.Agent` on the fixture, CPU only."""
    import laya
    return laya.Agent(tiny_checkpoint, device="cpu")


@pytest.fixture(scope="session")
def fastpath_checkpoint(tmp_path_factory):
    """Fixture whose decision head takes PyTorch's fast path (an even number of heads)."""
    return build_fixture_checkpoint(str(tmp_path_factory.mktemp("tiny-laya-fastpath")), hidden=128)


@pytest.fixture(scope="session")
def fastpath_agent(fastpath_checkpoint):
    import laya
    return laya.Agent(fastpath_checkpoint, device="cpu")


# --------------------------------------------------------------------------- T003: synthetic upstream dataset

UPSTREAM_WORKFLOWS = ["agent_trace_observability", "customer_service", "invoice_processing", "security_incidents"]
#: This workflow gets long states, so some question rows exceed a small named length.
LONG_WORKFLOW = "security_incidents"
TINY_TRAIN_PER_WORKFLOW = 12
TINY_TEST_PER_WORKFLOW = 10

_TINY_QUESTIONS = {
    "action": {"type": "choice", "instructions": "which option best matches the document",
               "criteria": {"continue": "proceed with the request", "review": "open a review",
                            "stop": "halt the request"}},
    "outcome": {"type": "choice", "instructions": "how did the request end",
                "criteria": {"success": "the request was done", "failure": "the request failed"}},
    "needs_review": {"type": "noul", "instructions": "this document needs review",
                     "criteria": {"false": "no review is needed", "true": "a review is needed"}},
    "risk": {"type": "score", "instructions": "how risky is the request",
             "criteria": ["zero risk", "low risk", "high risk"]},
    "urgency": {"type": "score", "instructions": "how urgent is the request",
                "criteria": ["not urgent", "normal", "urgent", "very urgent"]},
}


def _tiny_probs(rng, k):
    raw = [rng.random() ** 2 + 0.02 for _ in range(k)]
    total = sum(raw)
    return [round(v / total, 6) for v in raw]


def make_upstream_case(workflow, split, index, seed=0):
    """One case in the upstream schema (JSON columns as strings), fully seeded."""
    rng = random.Random("%s|%s|%d|%d" % (workflow, split, index, seed))

    def phrase(n):
        return " ".join(rng.choice(_WORDS) for _ in range(n))

    long = workflow == LONG_WORKFLOW
    state = {"task": phrase(140 if long else 10), "customer": phrase(3),
             "status": rng.choice(["open", "closed", "late"]), "amount": rng.randint(1, 9),
             "note": phrase(60 if long else 6)}
    questions = json.loads(json.dumps(_TINY_QUESTIONS))
    gold, agreement = {}, {}
    for qid, q in questions.items():
        if q["type"] == "choice":
            keys = list(q["criteria"])
            probs = _tiny_probs(rng, len(keys))
            label = keys[probs.index(max(probs))]
            gold[qid] = {"type": "choice", "label": label, "probabilities": dict(zip(keys, probs))}
        elif q["type"] == "noul":
            p = _tiny_probs(rng, 2)
            gold[qid] = {"type": "noul", "label": "true" if p[1] > p[0] else "false",
                         "noul": p[1], "probabilities": {"false": p[0], "true": p[1]}}
        else:
            k = len(q["criteria"])
            probs = _tiny_probs(rng, k)
            label = str(probs.index(max(probs)))
            gold[qid] = {"type": "score", "label": label, "score": round(sum(i * v for i, v in enumerate(probs)), 6),
                         "probabilities": {str(i): v for i, v in enumerate(probs)}}
        gold[qid]["confidence"] = round(0.3 + 0.6 * rng.random(), 6)
        agreement[qid] = {"argmax_agree": rng.random() < 0.6, "argmax_majority": gold[qid]["label"],
                          "total_variation": round(rng.random() * 0.6, 6)}
    return {"id": "%s_%06d" % (workflow, index), "workflow": workflow, "split": split,
            "state": json.dumps(state), "questions": json.dumps(questions), "gold": json.dumps(gold),
            "factors": json.dumps({"seed": index}), "label_agreement": json.dumps(agreement),
            "n_questions": len(questions)}


def write_tiny_upstream(root, n_train=TINY_TRAIN_PER_WORKFLOW, n_test=TINY_TEST_PER_WORKFLOW, seed=0):
    """Write the upstream repository layout (parquet) under `root`; returns the rows by split."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows = {"train": [], "test": []}
    for wf in UPSTREAM_WORKFLOWS:
        for split, n in (("train", n_train), ("test", n_test)):
            rows[split] += [make_upstream_case(wf, split, i, seed) for i in range(n)]

    def dump(rel, subset):
        path = os.path.join(root, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        pq.write_table(pa.Table.from_pylist(subset), path)

    for split in ("train", "test"):
        dump("all/%s-00000-of-00001.parquet" % split, rows[split])
        for wf in UPSTREAM_WORKFLOWS:
            dump("%s/%s-00000-of-00001.parquet" % (wf, split), [r for r in rows[split] if r["workflow"] == wf])
    return rows


class TinyUpstream:
    """The synthetic dataset, with a `download(relpath)` that never touches the network."""

    def __init__(self, root, rows):
        self.root, self.rows = root, rows

    def download(self, relpath):
        return os.path.join(self.root, relpath)


@pytest.fixture(scope="session")
def tiny_upstream(tmp_path_factory):
    root = str(tmp_path_factory.mktemp("tiny-upstream"))
    return TinyUpstream(root, write_tiny_upstream(root))


# --------------------------------------------------------------------------- in-process measuring children

def run_condition_in_process(function, spec, time_cap=None, grace=None):
    """`runner.run_condition` without the subprocess: same child function, same item shape.

    Starting a Python child costs a fresh ``import torch`` (about 10 s on a cold Windows machine), which dominates
    the CLI tests. The subprocess path itself stays covered by test_runner.py and by one real-child test per command
    family; tests that only check what a command *orchestrates* use this instead (`in_process_children`).
    """
    import time
    import traceback
    from experiments import runner
    from experiments.results import Status

    spec = dict(spec)
    if time_cap is not None:
        spec["time_cap"] = float(time_cap)
    base = {"runner": {"function": function}}
    started = time.perf_counter()
    try:
        payload = runner.resolve_function(function)(spec)
    except Exception as exc:
        item = dict(base, status=Status.FAILED.value, cause="exception", signal=None, exit_code=1,
                    exception_type=type(exc).__name__, reason="%s: %s" % (type(exc).__name__, exc),
                    traceback=traceback.format_exc(limit=20))
        item.update(runner._memory_fields(runner.peak_memory()))
        item["runner"]["elapsed_s"] = time.perf_counter() - started
        return item
    item = dict(base)
    item.update(payload)
    item.setdefault("status", Status.MEASURED.value)
    item.update(runner._memory_fields(runner.peak_memory()))
    item["exit_code"] = 0
    item["runner"]["elapsed_s"] = time.perf_counter() - started
    return item


@pytest.fixture()
def in_process_children(monkeypatch):
    """Run measuring children inside the test process (see `run_condition_in_process`)."""
    from experiments import runner
    monkeypatch.setattr(runner, "run_condition", run_condition_in_process)
