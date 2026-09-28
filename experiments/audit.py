"""Context audit: what the loaded model actually is (T018, data-model "Context Audit", research R3).

Everything is read from the loaded modules and configs, not from documentation. Lengths above
positional capacity are reported ``unsupported`` and are never truncated or padded to another
length (FR-006).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

#: Figures `docs/research-plan.md` gives for the English checkpoint. Background only: the audit
#: reports any difference from the loaded model as a discrepancy, not as a failure.
RESEARCH_PLAN_EXPECTED = {
    "local_window": 128,
    "global_every_n_layers": 3,
    "max_position_embeddings": 8192,
    "head_layers": 2,
    "configured_max_len": 512,
}

EXECUTED_WORK_VERDICTS = ("verified", "dense_masked", "not_yet_determined")

_LOCAL_TYPES = {"sliding_attention", "local", "local_attention", "sliding"}
_GLOBAL_TYPES = {"full_attention", "global", "global_attention", "full"}


def _layer_type(i: int, layer, ecfg) -> (str, str):
    """('local' | 'global', source of the answer)."""
    t = getattr(layer, "attention_type", None)
    if isinstance(t, str):
        if t in _LOCAL_TYPES:
            return "local", "module.attention_type"
        if t in _GLOBAL_TYPES:
            return "global", "module.attention_type"
    attn = getattr(layer, "attn", None)
    local = getattr(attn, "local_attention", None)  # transformers 4.x: (-1, -1) means global
    if isinstance(local, (tuple, list)) and len(local) == 2:
        return ("global" if tuple(local) == (-1, -1) else "local"), "module.attn.local_attention"
    types = getattr(ecfg, "layer_types", None)
    if isinstance(types, (list, tuple)) and i < len(types):
        return ("local" if types[i] in _LOCAL_TYPES else "global"), "config.layer_types"
    every = getattr(ecfg, "global_attn_every_n_layers", None)
    if every:
        return ("global" if i % int(every) == 0 else "local"), "config.global_attn_every_n_layers"
    return "global", "default (no local-attention settings found)"


def _rope_theta(kind: str, ecfg) -> Optional[float]:
    rope = getattr(ecfg, "rope_parameters", None)
    key = "sliding_attention" if kind == "local" else "full_attention"
    if isinstance(rope, dict):
        entry = rope.get(key)
        if isinstance(entry, dict) and entry.get("rope_theta") is not None:
            return float(entry["rope_theta"])
        if rope.get("rope_theta") is not None:
            return float(rope["rope_theta"])
    attr = "local_rope_theta" if kind == "local" else "global_rope_theta"
    val = getattr(ecfg, attr, None)
    if val is None:
        val = getattr(ecfg, "rope_theta", None)
    return None if val is None else float(val)


def _global_every(layers: List[Dict[str, Any]]) -> Optional[int]:
    idx = [l["index"] for l in layers if l["attention_type"] == "global"]
    if len(idx) < 2:
        return None
    gaps = {b - a for a, b in zip(idx, idx[1:])}
    return gaps.pop() if len(gaps) == 1 else None


def fallback_events(agent, load_info: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """Device and precision fallbacks, from the agent and the load record."""
    load_info = load_info or {}
    events = []
    count = int(getattr(agent, "cpu_fallback_count", 0) or 0)
    if count:
        events.append({"event": "cpu_fallback", "count": count,
                       "reason": getattr(agent, "last_fallback_reason", None) or "not recorded"})
    for msg in load_info.get("fallback_messages") or []:
        events.append({"event": "device_fallback_message", "reason": msg})
    device = str(getattr(agent, "device", "unknown"))
    if device != "cpu":
        events.append({"event": "device_not_cpu", "reason": "effective device is %s" % device})
    dtype = str(getattr(agent, "dtype", "")).replace("torch.", "")
    if (dtype and dtype != "float32") or getattr(agent, "amp_enabled", False):
        events.append({"event": "reduced_precision",
                       "reason": "dtype %s, amp_enabled %s" % (dtype, getattr(agent, "amp_enabled", False))})
    return events


def length_support(lengths: Sequence[int], max_positions: int, configured_max_len: int) -> List[Dict[str, Any]]:
    out = []
    for L in lengths:
        L = int(L)
        if L > max_positions:
            out.append({"length": L, "supported": False, "status": "unsupported",
                        "reason": "%d tokens exceeds the encoder's positional capacity "
                                  "(max_position_embeddings=%d); not truncated or padded to "
                                  "another length" % (L, max_positions)})
        else:
            out.append({"length": L, "supported": True,
                        "beyond_configured_max_len": L > configured_max_len})
    return out


def build_audit(agent, load_info: Optional[Dict[str, Any]] = None,
                lengths: Optional[Sequence[int]] = None) -> Dict[str, Any]:
    import transformers

    enc = agent.model.encoder
    ecfg = enc.config
    heads = int(ecfg.num_attention_heads)
    hidden = int(ecfg.hidden_size)
    layers = []
    for i, layer in enumerate(getattr(enc, "layers", [])):
        kind, source = _layer_type(i, layer, ecfg)
        entry = {"index": i, "attention_type": kind, "type_source": source,
                 "rope_theta": _rope_theta(kind, ecfg)}
        if kind == "local":
            entry["window"] = getattr(ecfg, "local_attention", None)
            raw = getattr(getattr(layer, "attn", None), "sliding_window", None)
            if raw is not None:
                entry["module_sliding_window"] = raw  # transformers' own parameterisation
        layers.append(entry)

    max_pos = int(ecfg.max_position_embeddings)
    configured = int(agent.cfg.get("max_len", 512))
    head = getattr(agent.model, "head", None)
    head_layers = list(getattr(head, "layers", [])) if head is not None else []
    first = head_layers[0] if head_layers else None
    decision_head = {
        "layers": len(head_layers),
        "configured_head_layers": agent.cfg.get("head_layers"),
        "heads": getattr(getattr(first, "self_attn", None), "num_heads", None),
        "hidden": getattr(getattr(first, "self_attn", None), "embed_dim", None),
        "ffn": getattr(getattr(first, "linear1", None), "out_features", None),
        "note": "the head's transformer layers run over the full sequence before marker positions "
                "are gathered (laya/common.py DecisionModel.forward), so head cost scales with length",
    }

    local = [l for l in layers if l["attention_type"] == "local"]
    observed = {
        "local_window": local[0].get("window") if local else None,
        "global_every_n_layers": _global_every(layers),
        "max_position_embeddings": max_pos,
        "head_layers": decision_head["layers"],
        "configured_max_len": configured,
    }
    comparison = {k: {"research_plan": v, "observed": observed[k], "matches": observed[k] == v}
                  for k, v in RESEARCH_PLAN_EXPECTED.items()}

    audit = {
        "model": {"id": (load_info or {}).get("model", getattr(agent, "model_id", None)),
                  "revision": (load_info or {}).get("revision", getattr(agent, "revision", None)),
                  "fixture_model": bool(agent.cfg.get("laya_sparse_fixture", False))},
        "transformers_version": transformers.__version__,
        "encoder": {
            "class": type(enc).__name__,
            "model_type": getattr(ecfg, "model_type", None),
            "hidden_size": hidden,
            "num_heads": heads,
            "head_dim": hidden // heads,
            "num_layers": len(layers),
            "intermediate_size": getattr(ecfg, "intermediate_size", None),
            "vocab_size": getattr(ecfg, "vocab_size", None),
            "attention_implementation": getattr(ecfg, "_attn_implementation", None),
            "local_attention": getattr(ecfg, "local_attention", None),
            "global_attn_every_n_layers": getattr(ecfg, "global_attn_every_n_layers", None),
        },
        "layers": layers,
        "positional": {
            "max_position_embeddings": max_pos,
            "configured_max_len": configured,
            "head_max_len": int(agent.cfg.get("head_max_len", 192)),
            "effective_supported_max": max_pos,
        },
        "decision_head": decision_head,
        "executed_work_note": {"verdict": "not_yet_determined", "evidence": None,
                               "note": "set by the profile command (T026) from measured scaling"},
        "fallbacks": fallback_events(agent, load_info),
        "research_plan_comparison": comparison,
        "discrepancies": [k for k, v in comparison.items() if not v["matches"]],
    }
    if lengths:
        audit["lengths"] = length_support(lengths, max_pos, configured)
    return audit


def format_schedule(audit: Dict[str, Any]) -> str:
    """Human-readable layer schedule and limits for the terminal."""
    e, p, h = audit["encoder"], audit["positional"], audit["decision_head"]
    lines = ["%s: %d layers, hidden %d, %d heads x %d, attention %s"
             % (e["class"], e["num_layers"], e["hidden_size"], e["num_heads"], e["head_dim"],
                e["attention_implementation"])]
    for l in audit["layers"]:
        extra = ", window %s" % l.get("window") if l["attention_type"] == "local" else ""
        lines.append("  layer %2d  %-6s rope_theta %s%s"
                     % (l["index"], l["attention_type"], l.get("rope_theta"), extra))
    lines.append("positions: %d (configured input cap %d, head budget %d)"
                 % (p["max_position_embeddings"], p["configured_max_len"], p["head_max_len"]))
    lines.append("decision head: %s layers, %s heads, hidden %s (runs over the full sequence)"
                 % (h["layers"], h["heads"], h["hidden"]))
    for L in audit.get("lengths", []):
        lines.append("  length %6d: %s" % (L["length"], "supported" + (" (beyond configured cap)" if L.get("beyond_configured_max_len") else "")
                                          if L["supported"] else "UNSUPPORTED - " + L["reason"]))
    for d in audit.get("discrepancies", []):
        c = audit["research_plan_comparison"][d]
        lines.append("  differs from research plan: %s = %s (plan says %s)" % (d, c["observed"], c["research_plan"]))
    for f in audit.get("fallbacks", []):
        lines.append("  fallback: %s (%s)" % (f["event"], f["reason"]))
    return "\n".join(lines)
