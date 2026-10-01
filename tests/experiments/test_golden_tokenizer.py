"""laya:005 (implements raya:004): tokenizer goldens from Laya's real sequence builder. Offline, fixture tokenizer."""
import json
import re

import pytest

from experiments.golden import common, tokenizer

REQUIRED = {"empty_document", "primary_one_question", "several_questions", "state_dict", "state_conversation_list",
            "unicode", "marker_like_strings", "long_document", "exact_max_len"}


def _files(root):
    return {str(p.relative_to(root)): p.read_bytes() for p in common.iter_files(root)}


def _record(root, name):
    return json.loads((root / "records" / ("%s.json" % name)).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def exports(tmp_path_factory):
    a, b = tmp_path_factory.mktemp("tok-a"), tmp_path_factory.mktemp("tok-b")
    tokenizer.export_tokenizer(a)
    tokenizer.export_tokenizer(b)
    return a, b


def test_records_exist_for_every_listed_input(exports):
    root, _ = exports
    names = {p.stem for p in (root / "records").glob("*.json")}
    assert REQUIRED <= names


def test_meta_and_tokenizer_json_saved_beside_the_records(exports):
    root, _ = exports
    meta = json.loads((root / "meta.json").read_text(encoding="utf-8"))
    assert re.fullmatch(r"[0-9a-f]{40}", meta["laya_sparse_commit"])
    assert meta["special_tokens"]["mask"] != meta["special_tokens"]["pad"] and meta["model"] == "fixture"
    tok_json = json.loads((root / "tokenizer.json").read_text(encoding="utf-8"))
    assert len(tok_json["model"]["vocab"]) == meta["vocab_size"]


def test_runs_are_byte_identical(exports):
    a, b = exports
    fa, fb = _files(a), _files(b)
    assert fa.keys() == fb.keys()
    assert all(fa[k] == fb[k] for k in fa), [k for k in fa if fa[k] != fb[k]]


def test_item_shape_positions_markers_and_specials(exports):
    root, _ = exports
    meta = json.loads((root / "meta.json").read_text(encoding="utf-8"))
    cls, sep, mask = (meta["special_tokens"][k] for k in ("cls", "sep", "mask"))
    for name in REQUIRED:
        rec = _record(root, name)
        for item in rec["items"]:
            ids = item["input_ids"]
            assert ids[0] == cls and (ids[-1] == sep or item["sequence_clipped"]) and len(ids) <= rec["max_len"]
            assert item["position_ids"] == list(range(len(ids)))
            assert all(ids[m] == mask for m in item["markers"])
            assert item["markers"] == sorted(item["markers"])
            assert item["qtype_name"] in ("choice", "score", "noul")


def test_several_questions_cover_all_types_and_stats(exports):
    rec = _record(exports[0], "several_questions")
    assert {i["qtype_name"] for i in rec["items"]} == {"choice", "score", "noul"}
    assert len(rec["items"]) == 4
    for item in rec["items"]:
        assert item["options_stats"]["options"] == len(item["markers"])


def test_truncation_flags(exports):
    root, _ = exports
    exact = _record(root, "exact_max_len")["items"][0]
    assert exact["length"] == 96 and not exact["state_truncated"] and not exact["sequence_clipped"]
    plus = _record(root, "max_len_plus_one")["items"][0]
    assert plus["length"] == 96 and plus["state_truncated"] and plus["state_tokens_total"] == exact["state_tokens_total"] + 1
    long = _record(root, "long_document")["items"][0]
    assert long["state_truncated"] and long["state_tokens_kept"] < long["state_tokens_total"] and long["length"] == 96
    empty = _record(root, "empty_document")["items"][0]
    assert empty["state_tokens_total"] == 0 and not empty["state_truncated"]
    conv = _record(root, "state_conversation_list")
    assert conv["truncate_left"] is True and conv["items"][0]["state_truncated"]
    wide = _record(root, "options_over_head_budget")["items"][0]
    assert wide["options_stats"]["tokens_per_option"] is not None
    assert _record(root, "primary_one_question")["items"][0]["options_stats"]["tokens_per_option"] is None


def test_no_laya_file_is_modified():
    assert common.commit_info()["laya_diff_empty"] is True
