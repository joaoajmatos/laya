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


def build_fixture_checkpoint(root):
    """Write the tiny checkpoint into `root` and return its path."""
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
        **FIXTURE_ENCODER,
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
