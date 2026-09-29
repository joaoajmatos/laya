"""Profile runs: component breakdown and executed-work check (T024-T026; research.md R3, R5).

Never used for headline latency (FR-013). Two kinds of instrumentation, both removed afterwards:

* **Stage wrappers** (`stage_wrappers`): instance-level wrappers around `Agent._encode_state`,
  `Agent._forward`, `Agent._decode_answers` and the `collate_items` name in `laya.agent`. They
  time Python-side stages and call straight through, so outputs are unchanged.
* **Operator attribution** (`label_modules` + `torch.profiler`): forward pre/post hooks open a
  `record_function` label around each encoder layer's attention, MLP and norms, the embeddings,
  rotary embedding, final norm, each decision-head layer and the scoring heads. Every profiled
  operator is attributed to its nearest label. Inside an attention label, operators under
  ``scaled_dot_product_attention`` are score/value work, and the conversion of the boolean mask
  (``aten::where`` before the kernel) is mask work; everything else there is projections and RoPE.
  Operators directly under the encoder label (outside every layer) build the attention mask.

Components (data-model "Component Profile"): tokenization_collation, attention_projections,
attention_score_value, mlp_norm, decision_head, option_scoring, decoding, other.
``attention_score_value`` includes attention-mask construction and conversion, matching the kernel
benchmarks' rule that mask work counts as attention cost; the split is kept in ``subcomponents``.
"""
from __future__ import annotations

import contextlib
import math
import time
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .results import Status, summarize

FASTPATH_OP = "aten::_transformer_encoder_layer_fwd"
FASTPATH_ATTENTION_OPS = ("aten::_native_multi_head_attention", "aten::_masked_softmax")

COMPONENTS = ("tokenization_collation", "attention_projections", "attention_score_value",
              "mlp_norm", "decision_head", "option_scoring", "decoding", "other")
NAMED = COMPONENTS[:-1]
EXPLAINED_TARGET = 0.90
PREFIX = "laya::"

# Executed-work verdict thresholds (research.md R3/R5). Exponents are log-log slopes of
# per-layer score/value time against length.
MIN_LENGTHS_FOR_VERDICT = 3
MIN_GLOBAL_EXPONENT = 1.3     # below this, quadratic attention is not yet visible at these lengths
VERIFIED_MARGIN = 0.5         # local grows at least this much slower than global
DENSE_MARGIN = 0.25           # local within this of global: same work, only masked


# --------------------------------------------------------------------------- stage wrappers (T024)

STAGE_MAP = {"_encode_state": "tokenization_collation", "collate_items": "tokenization_collation",
             "_forward": "forward", "_decode_answers": "decoding"}


@contextlib.contextmanager
def stage_wrappers(agent, sink: Optional[Dict[str, List[float]]] = None):
    """Install timing wrappers; yields ``{stage: [ms, ...]}``. Always removed on exit."""
    import laya.agent as agent_mod

    sink = sink if sink is not None else defaultdict(list)

    def wrap(name, fn):
        def wrapper(*args, **kwargs):
            t0 = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                sink[name].append((time.perf_counter() - t0) * 1000.0)
        wrapper.__wrapped__ = fn
        return wrapper

    installed = []
    original_collate = agent_mod.collate_items
    try:
        for name in ("_encode_state", "_forward", "_decode_answers"):
            setattr(agent, name, wrap(name, getattr(agent, name)))
            installed.append(name)
        agent_mod.collate_items = wrap("collate_items", original_collate)
        yield sink
    finally:
        for name in installed:
            with contextlib.suppress(AttributeError):
                delattr(agent, name)
        agent_mod.collate_items = original_collate


# --------------------------------------------------------------------------- module labels (T025)

def _layer_kind(i, layer, ecfg) -> str:
    from .audit import _layer_type
    return _layer_type(i, layer, ecfg)[0]


def head_takes_fastpath(agent) -> bool:
    """Whether the decision head's layers would take PyTorch's inference fast path when unhooked.

    Mirrors the static conditions in `nn.TransformerEncoderLayer.forward` (torch 2.x); the per-call
    ones (no grad, eval, 3-D input, no autocast) hold for Laya's CPU inference.
    """
    import torch
    head = getattr(agent.model, "head", None)
    layers = list(getattr(head, "layers", []) or [])
    if not layers or not torch.backends.mha.get_fastpath_enabled():
        return False
    for l in layers:
        a = getattr(l, "self_attn", None)
        if a is None or not getattr(a, "batch_first", False) or getattr(a, "in_proj_bias", None) is None \
                or not getattr(a, "_qkv_same_embed_dim", False) or a.num_heads % 2 == 1 \
                or not getattr(l, "activation_relu_or_gelu", False) or l.norm1.eps != l.norm2.eps:
            return False
    return True


@contextlib.contextmanager
def label_modules(agent):
    """Forward pre/post hooks that open `record_function` labels. Always removed on exit."""
    import torch

    handles, stack = [], []

    def add(module, label):
        def pre(_m, _inp):
            rf = torch.autograd.profiler.record_function(label)
            rf.__enter__()
            stack.append(rf)

        def post(_m, _inp, _out):
            stack.pop().__exit__(None, None, None)

        handles.append(module.register_forward_pre_hook(pre))
        handles.append(module.register_forward_hook(post))

    model = agent.model
    enc = model.encoder
    try:
        add(model, PREFIX + "model")
        add(enc, PREFIX + "enc")
        for name in ("embeddings", "final_norm", "rotary_emb"):
            sub = getattr(enc, name, None)
            if sub is not None:
                add(sub, PREFIX + "enc." + name)
        for i, layer in enumerate(getattr(enc, "layers", [])):
            kind = _layer_kind(i, layer, enc.config)
            for child_name, child in layer.named_children():
                if child_name == "attn":
                    add(child, "%senc.L%d.%s.attn" % (PREFIX, i, kind))
                elif child_name == "mlp":
                    add(child, "%senc.L%d.%s.mlp" % (PREFIX, i, kind))
                elif "norm" in child_name:
                    add(child, "%senc.L%d.%s.norm" % (PREFIX, i, kind))
        head = getattr(model, "head", None)
        # A hook anywhere in an nn.TransformerEncoderLayer disables PyTorch's inference fast path
        # (research.md R17), so the head is only labelled when the fast path is already off.
        # With it on, the head's operators are recognized by the fast-path op instead.
        if not head_takes_fastpath(agent):
            for i, layer in enumerate(getattr(head, "layers", []) if head is not None else []):
                add(layer, "%shead.L%d" % (PREFIX, i))
                if getattr(layer, "self_attn", None) is not None:
                    add(layer.self_attn, "%shead.L%d.attn" % (PREFIX, i))
        for name in ("scorer", "act_head", "type_emb"):
            sub = getattr(model, name, None)
            if sub is not None:
                add(sub, PREFIX + name)
        yield
    finally:
        for h in handles:
            h.remove()


def _ancestors(event):
    out = []
    p = event.cpu_parent
    while p is not None:
        out.append(p)
        p = p.cpu_parent
    return out


def classify_event(event) -> Optional[Dict[str, Any]]:
    """(component, subcomponent, layer label) for one profiler event, or None outside the model."""
    anc = _ancestors(event)
    chain = [event] + anc
    label = next((e.name for e in chain if e.name.startswith(PREFIX)), None)
    if label is None:
        return None
    if any(e.name == FASTPATH_OP for e in chain):
        in_attn = any(e.name in FASTPATH_ATTENTION_OPS for e in chain)
        return {"component": "decision_head", "sub": "head_attention" if in_attn else "head_other",
                "label": "head.fastpath"}
    lab = label[len(PREFIX):]
    in_sdpa = any("scaled_dot_product" in e.name for e in chain)
    if lab.startswith("head."):
        sub = "head_attention" if ".attn" in lab and in_sdpa else "head_other"
        return {"component": "decision_head", "sub": sub, "label": lab}
    if lab.startswith("enc.L"):
        part = lab.rsplit(".", 1)[-1]
        if part == "attn":
            if in_sdpa:
                return {"component": "attention_score_value", "sub": "sdpa_kernel", "label": lab}
            if event.name == "aten::where":
                return {"component": "attention_score_value", "sub": "mask_conversion", "label": lab}
            return {"component": "attention_projections", "sub": "projections_rope", "label": lab}
        return {"component": "mlp_norm", "sub": part, "label": lab}
    if lab == "enc":
        return {"component": "attention_score_value", "sub": "mask_construction", "label": lab}
    if lab == "enc.rotary_emb":
        return {"component": "attention_projections", "sub": "rotary_tables", "label": lab}
    if lab == "enc.final_norm":
        return {"component": "mlp_norm", "sub": "final_norm", "label": lab}
    if lab == "enc.embeddings":
        return {"component": "other", "sub": "embeddings", "label": lab}
    return {"component": "option_scoring", "sub": lab, "label": lab}  # model, scorer, act_head, type_emb


def attribute(events, total_ms: float, stages: Dict[str, List[float]]) -> Dict[str, Any]:
    """Build components, shares and the explained fraction from one profiled call."""
    comp = defaultdict(float)
    sub = defaultdict(float)
    per_layer = defaultdict(float)
    for e in events:
        c = classify_event(e)
        if c is None:
            continue
        ms = e.self_cpu_time_total / 1000.0
        comp[c["component"]] += ms
        sub["%s.%s" % (c["component"], c["sub"])] += ms
        if c["sub"] in ("sdpa_kernel", "mask_conversion", "head_attention"):
            per_layer[c["label"]] += ms
    comp["tokenization_collation"] += sum(stages.get("_encode_state", [])) + sum(stages.get("collate_items", []))
    comp["decoding"] += sum(stages.get("_decode_answers", []))
    named = sum(comp[k] for k in NAMED)
    other = comp["other"] + max(0.0, total_ms - named - comp["other"])
    denom = named + other
    components = {k: {"ms": comp[k] if k != "other" else other} for k in COMPONENTS}
    for k in COMPONENTS:
        components[k]["share"] = components[k]["ms"] / denom if denom else 0.0
    return {
        "components": components,
        "subcomponents": dict(sorted(sub.items())),
        "score_value_by_layer": dict(sorted(per_layer.items())),
        "explained_fraction": named / denom if denom else 0.0,
        "attributed_total_ms": denom,
    }


def _layer_type_summary(by_layer: Dict[str, float], head_layers: Optional[int] = None) -> Dict[str, Any]:
    groups = defaultdict(list)
    for label, ms in by_layer.items():
        if label.startswith("head."):
            groups["head"].append(ms)
        else:
            groups[label.split(".")[2]].append(ms)  # enc.L{i}.{kind}.attn
    out = {k: {"n_layers": len(v), "total_ms": sum(v), "per_layer_ms": sum(v) / len(v)} for k, v in groups.items()}
    if "head" in out and "head.fastpath" in by_layer and head_layers:
        # one fast-path label covers every head layer
        out["head"]["n_layers"] = head_layers
        out["head"]["per_layer_ms"] = out["head"]["total_ms"] / head_layers
    return out


def profile_in_process(agent, spec: Dict[str, Any], load_info: Optional[Dict[str, Any]] = None,
                       started: Optional[float] = None) -> Dict[str, Any]:
    """Profile `repeats` single requests at one length; median per component across repeats."""
    import torch
    from torch.profiler import ProfilerActivity, profile as torch_profile

    from .inputs import build_request
    from .timing import WARMUP_SEED_OFFSET, _call, base_record, condition_kind

    started = time.perf_counter() if started is None else started
    total = int(spec["total_tokens"])
    nq, opts = int(spec.get("questions", 1)), int(spec.get("options", 2))
    repeats = int(spec.get("repeats") or 5)
    warmup = int(spec.get("warmup", 1))
    seed = int(spec.get("seed", 0))
    cap = spec.get("time_cap")
    rec = base_record(dict(spec, batch_size=1), agent)
    rec["kind"] = "profile"
    rec["profile_run"] = True
    rec["load_seconds"] = (load_info or {}).get("load_seconds")
    rec["repeats"] = {"requested": repeats, "completed": 0, "warmup": warmup}
    if total > rec["position_limit"]:
        rec.update(status=Status.UNSUPPORTED.value,
                   reason="%d tokens exceeds positional capacity %d; no model call made"
                          % (total, rec["position_limit"]))
        return rec
    kind = condition_kind(nq, 1)
    last = 0.0
    for w in range(warmup):
        st, q, _ = build_request(agent, total, seed + WARMUP_SEED_OFFSET + w, nq, opts, 1)
        t0 = time.perf_counter()
        _call(agent, kind, st, q, total)
        last = time.perf_counter() - t0

    runs = []
    for r in range(repeats):
        if cap is not None and runs and (time.perf_counter() - started) + 2 * last > float(cap):
            break
        st, q, acc = build_request(agent, total, seed + r, nq, opts, 1)
        stages: Dict[str, List[float]] = defaultdict(list)
        with stage_wrappers(agent, stages), label_modules(agent):
            with torch_profile(activities=[ProfilerActivity.CPU], record_shapes=False) as prof:
                t0 = time.perf_counter()
                _call(agent, kind, st, q, total)
                total_ms = (time.perf_counter() - t0) * 1000.0
        last = total_ms / 1000.0
        events = prof.events()
        att = attribute(events, total_ms, stages)
        att["total_profiled_ms"] = total_ms
        att["head_fastpath"] = any(e.name == FASTPATH_OP for e in events)
        att["stage_ms"] = {k: sum(v) for k, v in stages.items()}
        runs.append(att)
        if r == 0:
            rec["token_accounting"] = acc

    rec["repeats"]["completed"] = len(runs)
    if not runs:
        rec.update(status=Status.PARTIAL.value, reason="time cap reached before any profiled request")
        return rec

    def med(values):
        return summarize(values)["p50"]

    total_med = med([r["total_profiled_ms"] for r in runs])
    comps = {k: med([r["components"][k]["ms"] for r in runs]) for k in COMPONENTS}
    denom = sum(comps.values())
    rec["total_profiled_ms"] = total_med
    rec["total_profiled_ms_all"] = [r["total_profiled_ms"] for r in runs]
    rec["components"] = {k: {"ms": comps[k], "share": comps[k] / denom if denom else 0.0} for k in COMPONENTS}
    rec["explained_fraction"] = sum(comps[k] for k in NAMED) / denom if denom else 0.0
    subkeys = sorted({k for r in runs for k in r["subcomponents"]})
    rec["subcomponents_ms"] = {k: med([r["subcomponents"].get(k, 0.0) for r in runs]) for k in subkeys}
    layer_keys = sorted({k for r in runs for k in r["score_value_by_layer"]})
    by_layer = {k: med([r["score_value_by_layer"].get(k, 0.0) for r in runs]) for k in layer_keys}
    rec["score_value_by_layer_ms"] = by_layer
    head_layers = len(getattr(getattr(agent.model, "head", None), "layers", []) or [])
    rec["score_value_by_layer_type"] = _layer_type_summary(by_layer, head_layers)
    rec["head_path"] = "fastpath" if all(r["head_fastpath"] for r in runs) else (
        "modules" if not any(r["head_fastpath"] for r in runs) else "mixed")
    rec["mha_fastpath"] = bool(torch.backends.mha.get_fastpath_enabled())
    rec["stage_ms"] = {k: med([r["stage_ms"].get(k, 0.0) for r in runs])
                       for k in sorted({k for r in runs for k in r["stage_ms"]})}
    ok = rec["explained_fraction"] >= EXPLAINED_TARGET
    rec["explained_ok"] = ok
    rec["explained_note"] = (None if ok else
                             "named components explain %.1f%% of profiled time, below the %.0f%% target; "
                             "the remaining %.1f%% is in 'other' (Python dispatch, embeddings, transfers)"
                             % (100 * rec["explained_fraction"], 100 * EXPLAINED_TARGET,
                                100 * (1 - rec["explained_fraction"])))
    rec["total_clean_ms"] = None
    rec["profiled_to_clean_ratio"] = None
    if len(runs) < repeats:
        rec.update(status=Status.PARTIAL.value,
                   reason="time cap reached after %d of %d profiled requests" % (len(runs), repeats))
    else:
        rec["status"] = Status.MEASURED.value
    return rec


def profile_condition(spec: Dict[str, Any]) -> Dict[str, Any]:
    """Child-side entry point for `runner.run_condition`."""
    from .runner import load_agent
    started = time.perf_counter()
    agent, info = load_agent(spec["model"], spec.get("revision"), spec.get("threads"),
                             mha_fastpath=bool(spec.get("mha_fastpath", True)))
    rec = profile_in_process(agent, spec, info, started=started)
    rec["model"] = {"id": info["model"], "revision": info["revision"]}
    return rec


# --------------------------------------------------------------------------- scaling and verdict (T025, T026)

def fit_exponent(lengths: Sequence[float], values: Sequence[float]) -> Optional[float]:
    """Least-squares slope of log(value) against log(length); None with fewer than 3 usable points."""
    pts = [(math.log(L), math.log(v)) for L, v in zip(lengths, values) if L > 0 and v and v > 0]
    if len(pts) < 3:
        return None
    mx = sum(x for x, _ in pts) / len(pts)
    my = sum(y for _, y in pts) / len(pts)
    sxx = sum((x - mx) ** 2 for x, _ in pts)
    if sxx == 0:
        return None
    return sum((x - mx) * (y - my) for x, y in pts) / sxx


def scaling(items: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """Per-component and per-layer-type time exponents across measured profile items."""
    usable = sorted((it for it in items if it.get("status") in ("measured", "partial") and it.get("components")),
                    key=lambda it: it["condition"]["total_tokens"])
    lengths = [it["condition"]["total_tokens"] for it in usable]
    out: Dict[str, Any] = {"lengths": lengths, "components": {}, "score_value_per_layer": {}}
    for k in COMPONENTS:
        out["components"][k] = fit_exponent(lengths, [it["components"][k]["ms"] for it in usable])
    out["total"] = fit_exponent(lengths, [it["total_profiled_ms"] for it in usable])
    for kind in ("local", "global", "head"):
        vals = [it.get("score_value_by_layer_type", {}).get(kind, {}).get("per_layer_ms") for it in usable]
        out["score_value_per_layer"][kind] = fit_exponent(lengths, vals)
    mask = [it.get("subcomponents_ms", {}).get("attention_score_value.mask_construction") for it in usable]
    out["mask_construction"] = fit_exponent(lengths, mask)
    return out


def executed_work_verdict(scal: Dict[str, Any], window: Optional[int] = None) -> Dict[str, Any]:
    """research.md R3/R5: do local layers do less work than global layers as length grows?"""
    local = scal.get("score_value_per_layer", {}).get("local")
    glob = scal.get("score_value_per_layer", {}).get("global")
    lengths = scal.get("lengths", [])
    evidence = {"local_exponent": local, "global_exponent": glob, "lengths": lengths,
                "window": window, "thresholds": {"min_global_exponent": MIN_GLOBAL_EXPONENT,
                                                 "verified_margin": VERIFIED_MARGIN,
                                                 "dense_margin": DENSE_MARGIN}}
    if len(lengths) < MIN_LENGTHS_FOR_VERDICT or local is None or glob is None:
        return {"verdict": "not_yet_determined", "evidence": evidence,
                "note": "needs local and global score/value times at %d or more lengths" % MIN_LENGTHS_FOR_VERDICT}
    if window and max(lengths) <= 2 * window:
        return {"verdict": "not_yet_determined", "evidence": evidence,
                "note": "longest length %d is not beyond twice the local window %d" % (max(lengths), window)}
    if glob < MIN_GLOBAL_EXPONENT:
        return {"verdict": "not_yet_determined", "evidence": evidence,
                "note": "global attention time grows with exponent %.2f; quadratic cost is not yet "
                        "visible at these lengths" % glob}
    if local <= glob - VERIFIED_MARGIN:
        return {"verdict": "verified", "evidence": evidence,
                "note": "local-layer score/value time grows clearly slower than global (%.2f vs %.2f)" % (local, glob)}
    if local >= glob - DENSE_MARGIN:
        return {"verdict": "dense_masked", "evidence": evidence,
                "note": "local-layer score/value time grows like global (%.2f vs %.2f): the window is "
                        "applied as a mask over full-length attention" % (local, glob)}
    return {"verdict": "not_yet_determined", "evidence": evidence,
            "note": "local exponent %.2f is between the thresholds for global %.2f" % (local, glob)}
