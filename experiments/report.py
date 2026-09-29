"""Bottleneck report: one deliverable from every result file of a run (T039, T040; FR-026 to FR-029).

`build_report(run_path)` returns the body of ``report.json``; `render_markdown(body)` returns
``report.md``. Rules the report enforces:

* Every claim is a *statement* tagged ``measured``, ``estimated`` (cost floor, analytical memory,
  extrapolations) or ``hypothesized`` (forward-looking implications), and cites the items it rests on
  as ``file#index`` (contracts/results.md rules 5 and 6).
* Rankings use only ``measured`` items; ``partial``, ``failed`` and ``unsupported`` items are listed,
  never ranked and never dropped (FR-015, FR-029).
* Results beyond the checkpoint's configured input cap are cost-only; no quality claim is made from
  any result, since the documents are synthetic filler (research.md R4).
* GPU numbers appear only in the note on the ~33 ms reference's hardware (results rule 7, R15).
"""
from __future__ import annotations

import datetime as _dt
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from . import results

TAGS = ("measured", "estimated", "hypothesized")
HISTORICAL_MS = 33.0
GPU_CONSISTENT_MS = 66.0          # research.md R15: within 2x of 33 ms
ORDER_OF_SECOND = (300.0, 3000.0)  # H3: "on the order of one second"
HIGH_VARIANCE_RATIO = 1.5          # research.md R12
OBSERVABLE_FRACTION = 0.1          # a matrix under 10% of the load peak is not observable in peak memory
MIN_PARTIAL_REPEATS = 5            # a partial clean item with this many repeats may anchor a ranking, labelled
LINEAR = ("attention_projections", "mlp_norm")
COMPONENT_NAMES = {
    "tokenization_collation": "tokenization and collation",
    "attention_projections": "attention projections (QKV, output, RoPE)",
    "attention_score_value": "attention score/value (incl. mask building)",
    "mlp_norm": "MLP and norms",
    "decision_head": "decision head (2 layers over the full sequence)",
    "option_scoring": "option scoring",
    "decoding": "decoding",
    "other": "other (embeddings, unattributed)",
}


class ReportError(ValueError):
    """The report cannot be built (manifest or audit missing)."""


# --------------------------------------------------------------------------- helpers

def stmt(tag: str, text: str, refs: Iterable[str] = ()) -> Dict[str, Any]:
    if tag not in TAGS:
        raise ValueError("statement tag must be one of %s" % (TAGS,))
    return {"tag": tag, "text": text, "derived_from": [r for r in refs if r]}


def _ms(v: Optional[float]) -> str:
    if v is None:
        return "n/a"
    return "%.2f s" % (v / 1000.0) if v >= 1000 else "%.0f ms" % v


def _gb(b: Optional[float]) -> str:
    return "n/a" if b is None else "%.2f GB" % (b / 1e9)


def _pct(x: Optional[float]) -> str:
    return "n/a" if x is None else "%.0f%%" % (100 * x)


def _load(run: Path, name: str) -> Optional[Dict[str, Any]]:
    if not (run / name).exists():
        return None
    return results.read_json(run, name)


def _items(body: Optional[Dict[str, Any]]) -> List[Tuple[int, Dict[str, Any]]]:
    return list(enumerate((body or {}).get("items", [])))


def _single(sweep) -> Dict[int, Tuple[int, Dict[str, Any]]]:
    """Primary curve: kind single, measured or partial, keyed by length."""
    out = {}
    for i, it in _items(sweep):
        if it.get("kind") == "single" and it.get("condition", {}).get("batch_size", 1) == 1:
            out[it["condition"]["total_tokens"]] = (i, it)
    return out


def _profiles(profile) -> Dict[int, Tuple[int, Dict[str, Any]]]:
    return {it["condition"]["total_tokens"]: (i, it) for i, it in _items(profile) if "condition" in it}


# --------------------------------------------------------------------------- sections

def audit_summary(audit: Dict[str, Any], manifest: Dict[str, Any]) -> Dict[str, Any]:
    layers = audit.get("layers", [])
    glob = [l["index"] for l in layers if l["attention_type"] == "global"]
    loc = [l for l in layers if l["attention_type"] == "local"]
    window = loc[0].get("window") if loc else None
    pos, head, enc = audit["positional"], audit["decision_head"], audit["encoder"]
    ew = audit.get("executed_work_note") or {}
    fallbacks = audit.get("fallbacks") or []
    s = [
        stmt("measured", "Encoder %s: %d layers, hidden %d, %d heads of %d, attention implementation %s."
             % (enc["class"], enc["num_layers"], enc["hidden_size"], enc["num_heads"], enc["head_dim"],
                enc.get("attention_implementation")), ["audit.json#encoder"]),
        stmt("measured", "%d global layers (%s) and %d local layers with a %s-token window (about +/-%s tokens per side)."
             % (len(glob), ", ".join(map(str, glob)), len(loc), window, (window // 2) if window else "?"),
             ["audit.json#layers"]),
        stmt("measured", "Positional capacity %d tokens; configured input cap (max_len) %d; head budget %d."
             % (pos["max_position_embeddings"], pos["configured_max_len"], pos["head_max_len"]),
             ["audit.json#positional"]),
        stmt("measured", "Decision head: %s transformer layers, %s heads, hidden %s, run over the full sequence before option markers are gathered."
             % (head["layers"], head["heads"], head["hidden"]), ["audit.json#decision_head"]),
        stmt("measured", "Device and precision fallbacks: %s." % (
            "none observed" if not fallbacks else "; ".join("%s (%s)" % (f["event"], f["reason"]) for f in fallbacks)),
             ["audit.json#fallbacks"]),
    ]
    if ew.get("verdict"):
        s.append(stmt("measured", "Executed work in the local layers: %s. %s" % (ew["verdict"], ew.get("note", "")),
                      ["audit.json#executed_work_note"] + list(ew.get("derived_from", []))))
    disc = audit.get("discrepancies") or []
    s.append(stmt("measured", "Differences from docs/research-plan.md figures: %s." % (", ".join(disc) if disc else "none"),
                  ["audit.json#research_plan_comparison"]))
    return {"model": audit.get("model"), "statements": s}


def environment(manifest: Dict[str, Any]) -> List[Dict[str, Any]]:
    hw, rt, sw, m = manifest["hardware"], manifest["runtime"], manifest["software"], manifest["model"]
    return [
        stmt("measured", "Model %s at revision %s (%s); fp32=%s, autocast %s."
             % (m.get("id"), m.get("revision"), m.get("revision_source"), rt.get("dtype") == "float32",
                "on" if rt.get("amp_enabled") else "off"), ["manifest.json#model"]),
        stmt("measured", "CPU %s, %s physical / %s logical cores, %s RAM, %s; power scheme when the manifest was written: %s."
             % (hw.get("cpu_model"), hw.get("physical_cores"), hw.get("logical_cores"),
                _gb(hw.get("ram_bytes")) if isinstance(hw.get("ram_bytes"), (int, float)) else "unknown",
                hw.get("os"), hw.get("power_scheme", "not recorded")), ["manifest.json#hardware"]),
        stmt("measured", "torch %s (CPU capability %s), transformers %s, Python %s; %s intra-op threads (%s)."
             % (sw.get("torch"), (rt.get("torch_build") or {}).get("cpu_capability"), sw.get("transformers"),
                sw.get("python"), rt.get("intra_op_threads"), rt.get("threads_source")), ["manifest.json#runtime"]),
    ]


def sessions(files: Dict[str, Optional[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Machine state recorded when each measuring command started (power plan can change after audit)."""
    out = []
    for name in ("sweep.json", "profile.json", "kernels.json"):
        ses = (files.get(name) or {}).get("session")
        if ses:
            out.append(stmt("measured", "%s started %s with power scheme %s, on AC power: %s."
                            % (name, ses.get("started_at"), ses.get("power_scheme", "not recorded"),
                               ses.get("on_ac_power", "not recorded")), ["%s#session" % name]))
    return out


def latency_tables(sweep) -> Dict[str, Any]:
    primary, other = [], []
    singles = {L: it for L, (i, it) in _single(sweep).items()}
    for i, it in _items(sweep):
        c, t = it.get("condition", {}), it.get("timings_ms") or {}
        if not it.get("kind") and c:
            it = dict(it, kind="batch" if c.get("batch_size", 1) > 1 else
                      ("multi_question" if c.get("questions", 1) > 1 else "single"))
        row = {"ref": "sweep.json#%d" % i, "length": c.get("total_tokens"), "questions": c.get("questions"),
               "batch_size": c.get("batch_size"), "kind": it.get("kind"), "status": it.get("status"),
               "p50_ms": t.get("p50"), "p95_ms": t.get("p95"), "n": t.get("n"),
               "low_sample_p95": t.get("low_sample_p95"), "cost_only": bool(it.get("beyond_configured_max_len")),
               "drift_flagged": bool((it.get("drift") or {}).get("flagged")),
               "peak_rss_bytes": it.get("peak_rss_bytes"), "reason": it.get("reason")}
        if it.get("kind") == "single":
            primary.append(row)
        else:
            base = singles.get(c.get("total_tokens"))
            bp = ((base or {}).get("timings_ms") or {}).get("p50")
            row["multiplier_vs_single"] = (t["p50"] / bp) if (t.get("p50") and bp) else None
            row["units"] = c.get("questions") if it.get("kind") == "multi_question" else c.get("batch_size")
            other.append(row)
    primary.sort(key=lambda r: r["length"] or 0)
    other.sort(key=lambda r: (r["length"] or 0, r["kind"] or "", r["units"] or 0))
    return {"primary": primary, "multi_and_batch": other}


def ranking(sweep, profile, floor) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """by_length ranking, cost_floor list, and not_run entries for lengths without a ranking."""
    singles, profs = _single(sweep), _profiles(profile)
    floors = {f["length"]: (i, f) for i, f in _items(floor)}
    by_length, cost_floor, missing = [], [], []
    for L in sorted(set(singles) | set(profs)):
        si, s = singles.get(L, (None, None))
        pi, p = profs.get(L, (None, None))
        s_ok = s is not None and (s.get("status") == "measured" or (
            s.get("status") == "partial" and (s.get("repeats") or {}).get("completed", 0) >= MIN_PARTIAL_REPEATS
            and (s.get("timings_ms") or {}).get("p50")))
        p_ok = p is not None and p.get("status") == "measured" and p.get("components")
        if not (s_ok and p_ok):
            why = []
            if not s_ok:
                why.append("clean sweep %s" % ((s or {}).get("status") or "missing"))
            if not p_ok:
                why.append("profile %s" % ((p or {}).get("status") or "missing"))
            missing.append({"check": "component ranking at %d tokens" % L,
                            "reason": "needs measured clean and profile items (%s)" % ", ".join(why)})
            continue
        p50 = s["timings_ms"]["p50"]
        comps = sorted(p["components"].items(), key=lambda kv: -kv[1]["share"])
        refs = ["sweep.json#%d" % si, "profile.json#%d" % pi]
        entries = [{"component": k, "share": v["share"], "ms_of_clean_p50": v["share"] * p50, "derived_from": refs}
                   for k, v in comps]
        aa = _attn_share(profs, L)
        by_length.append({"length": L, "clean_p50_ms": p50, "cost_only": bool(s.get("beyond_configured_max_len")),
                          "all_attention_share": aa[0] if aa else None,
                          "clean_status": s.get("status"), "clean_repeats": s.get("repeats"),
                          "drift_flagged": bool((s.get("drift") or {}).get("flagged") or (p.get("drift") or {}).get("flagged")),
                          "explained_fraction": p.get("explained_fraction"), "components": entries,
                          "derived_from": refs})
        fi, f = floors.get(L, (None, None))
        if f and f.get("status") == "measured":
            cost_floor.append({"length": L, "floor_ms": f["floor_ms"], "floor_fraction": f["floor_fraction"],
                               "label": "estimate", "derived_from": ["floor.json#%d" % fi]})
    return by_length, cost_floor, missing


def _share(profs, L, comp) -> Optional[Tuple[float, str]]:
    x = profs.get(L)
    if not x or x[1].get("status") != "measured" or not x[1].get("components"):
        return None
    return x[1]["components"][comp]["share"], "profile.json#%d" % x[0]


def _attn_share(profs, L) -> Optional[Tuple[float, str, float, float]]:
    """All attention score/value work (encoder plus decision head) as a share of profiled time.

    R13 counts the head's two full-sequence layers as attention; with the head on PyTorch's fast path
    its attention is `decision_head.head_attention`, not `attention_score_value` (research.md R17).
    """
    x = profs.get(L)
    if not x or x[1].get("status") != "measured" or not x[1].get("components"):
        return None
    it = x[1]
    enc = it["components"]["attention_score_value"]["share"]
    total = sum(c["ms"] for c in it["components"].values()) or it.get("total_profiled_ms") or 0
    head_ms = (it.get("subcomponents_ms") or {}).get("decision_head.head_attention", 0.0)
    head = head_ms / total if total else 0.0
    return enc + head, "profile.json#%d" % x[0], enc, head


def _nearest(profs, target, lo, hi):
    cands = [L for L, (i, p) in profs.items() if lo <= L <= hi and p.get("status") == "measured" and p.get("components")]
    return min(cands, key=lambda L: abs(math.log(L / target))) if cands else None


def assumptions(audit, sweep, profile, gpu) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """The plan's three assumptions and the pre-registered hypotheses H1-H5 (research.md R13)."""
    profs, singles = _profiles(profile), _single(sweep)
    out, hyp = [], []

    # A1: attention dominates near 2K
    L = _nearest(profs, 2048, 1536, 3072)
    if L is None:
        out.append({"id": "A1", "assumption": "attention dominates near 2,000 tokens", "verdict": "untested",
                    "evidence": "no measured profile between 1,536 and 3,072 tokens", "derived_from": []})
    else:
        sh, ref, enc, head = _attn_share(profs, L)
        out.append({"id": "A1", "assumption": "attention dominates near 2,000 tokens",
                    "verdict": "confirmed" if sh >= 0.5 else "contradicted",
                    "evidence": "attention score/value is %s of profiled time at %d tokens (encoder %s, decision head %s)"
                                % (_pct(sh), L, _pct(enc), _pct(head)),
                    "derived_from": [ref]})

    # A2: 512 tokens near 33 ms
    caveat = ("The ~33 ms figure's hardware is not stated; Laya's own runtime message gives ~35 ms on GPU "
              "and ~200-500 ms on CPU, so it is most likely a GPU figure.")
    s512 = singles.get(512)
    gpu512 = None
    for i, it in _items(gpu):
        if it.get("condition", {}).get("total_tokens") == 512 and it.get("timings_ms"):
            gpu512 = (i, it)
    if not s512 or not (s512[1].get("timings_ms") or {}).get("p50"):
        out.append({"id": "A2", "assumption": "512-token latency near ~33 ms", "verdict": "untested",
                    "evidence": caveat + " No CPU measurement at 512 tokens.", "derived_from": []})
    else:
        p50 = s512[1]["timings_ms"]["p50"]
        ev = caveat + " On the target CPU the 512-token p50 is %s, %.0fx the reference." % (_ms(p50), p50 / HISTORICAL_MS)
        refs = ["sweep.json#%d" % s512[0]]
        if gpu512:
            g = gpu512[1]["timings_ms"]["p50"]
            dev = (gpu or {}).get("environment", {}).get("device_name", "GPU")
            ev += (" GPU-only check (not a CPU result): %s measures %s at 512 tokens, which %s a GPU origin for "
                   "the reference (rule: under %.0f ms)." % (dev, _ms(g), "is consistent with" if g <= GPU_CONSISTENT_MS
                                                              else "does not support", GPU_CONSISTENT_MS))
            refs.append("gpu_reference.json#%d" % gpu512[0])
        out.append({"id": "A2", "assumption": "512-token latency near ~33 ms",
                    "verdict": "confirmed" if p50 <= 2 * HISTORICAL_MS else "contradicted",
                    "evidence": ev, "derived_from": refs})

    # A3: local layers skip out-of-window work
    ew = (audit.get("executed_work_note") or {})
    v = ew.get("verdict", "not_yet_determined")
    out.append({"id": "A3", "assumption": "local layers skip out-of-window work",
                "verdict": {"verified": "confirmed", "dense_masked": "contradicted"}.get(v, "untested"),
                "evidence": "executed-work verdict %s. %s" % (v, ew.get("note", "")),
                "derived_from": ["audit.json#executed_work_note"] + list(ew.get("derived_from", []))})

    # H1: attention below 50% at 2,048
    x = _attn_share(profs, 2048)
    hyp.append({"id": "H1", "hypothesis": "attention score/value is below 50% of time at 2,048 tokens",
                "verdict": "untested" if x is None else ("confirmed" if x[0] < 0.5 else "contradicted"),
                "evidence": "no measured profile at 2,048" if x is None else
                "share %s (encoder %s, decision head %s)" % (_pct(x[0]), _pct(x[2]), _pct(x[3])),
                "derived_from": [x[1]] if x else []})

    # H2: attention overtakes the rest only if local layers run dense masked
    over = [(L, _attn_share(profs, L)) for L in sorted(profs)]
    over = [(L, (s[0], s[1])) for L, s in over if s and s[0] >= 0.5]
    if not over:
        hyp.append({"id": "H2", "hypothesis": "attention overtakes the rest only when local layers run dense masked",
                    "verdict": "untested", "evidence": "attention did not reach 50% at any profiled length, so the "
                    "condition never arose", "derived_from": []})
    else:
        L, (sh, ref) = over[0]
        hyp.append({"id": "H2", "hypothesis": "attention overtakes the rest only when local layers run dense masked",
                    "verdict": "confirmed" if v == "dense_masked" else "contradicted",
                    "evidence": "attention reaches %s at %d tokens; executed-work verdict %s" % (_pct(sh), L, v),
                    "derived_from": [ref, "audit.json#executed_work_note"]})

    # H3: 512 tokens on the order of one second
    if s512 and (s512[1].get("timings_ms") or {}).get("p50"):
        p50 = s512[1]["timings_ms"]["p50"]
        ok = ORDER_OF_SECOND[0] <= p50 <= ORDER_OF_SECOND[1]
        hyp.append({"id": "H3", "hypothesis": "512-token latency is on the order of one second on this CPU",
                    "verdict": "confirmed" if ok else "contradicted", "evidence": "p50 %s" % _ms(p50),
                    "derived_from": ["sweep.json#%d" % s512[0]]})
    else:
        hyp.append({"id": "H3", "hypothesis": "512-token latency is on the order of one second on this CPU",
                    "verdict": "untested", "evidence": "no measurement at 512", "derived_from": []})

    # H4: projections or MLP are the largest component at 512 and 2,048
    tops, refs, missing = [], [], []
    for L in (512, 2048):
        x = profs.get(L)
        if not x or x[1].get("status") != "measured" or not x[1].get("components"):
            missing.append(L)
            continue
        top = max(x[1]["components"].items(), key=lambda kv: kv[1]["share"])[0]
        tops.append((L, top))
        refs.append("profile.json#%d" % x[0])
    if missing:
        hyp.append({"id": "H4", "hypothesis": "projections or MLP are the largest component at 512 and 2,048",
                    "verdict": "untested", "evidence": "no measured profile at %s" % missing, "derived_from": refs})
    else:
        ok = all(t in LINEAR for _, t in tops)
        hyp.append({"id": "H4", "hypothesis": "projections or MLP are the largest component at 512 and 2,048",
                    "verdict": "confirmed" if ok else "contradicted",
                    "evidence": "; ".join("%d tokens: %s" % (L, t) for L, t in tops), "derived_from": refs})

    # H5: memory at 8,192
    s8 = singles.get(8192)
    heads = audit["encoder"]["num_heads"]
    need = heads * 8192 * 8192 * 4
    if s8 and s8[1].get("status") == "failed" and s8[1].get("cause") == "oom":
        hyp.append({"id": "H5", "hypothesis": "a materialized score matrix makes 8,192 tokens memory-bound",
                    "verdict": "confirmed", "evidence": "the 8,192-token request ran out of memory (%s)"
                    % s8[1].get("reason", "oom"), "derived_from": ["sweep.json#%d" % s8[0]]})
    elif not s8 or s8[1].get("status") not in ("measured", "partial") or not s8[1].get("peak_rss_bytes"):
        hyp.append({"id": "H5", "hypothesis": "a materialized score matrix makes 8,192 tokens memory-bound",
                    "verdict": "untested", "evidence": "no completed 8,192-token memory measurement",
                    "derived_from": ["sweep.json#%d" % s8[0]] if s8 else []})
    else:
        rss = s8[1]["peak_rss_bytes"]
        base = min((it.get("peak_rss_bytes") or rss) for L, (i, it) in singles.items()
                   if it.get("peak_rss_bytes") and it.get("status") in ("measured", "partial"))
        grew = rss - base
        materialized = grew >= 0.5 * need
        hyp.append({"id": "H5", "hypothesis": "a materialized score matrix makes 8,192 tokens memory-bound",
                    "verdict": "confirmed" if materialized else "contradicted",
                    "evidence": "peak working set %s at 8,192 vs %s at the smallest length: grew %s, against %s for "
                                "one full score matrix (analytical)%s" % (_gb(rss), _gb(base), _gb(grew), _gb(need),
                                "; with the decision head on PyTorch's fast path the materialized matrices are most "
                                "likely the head's (research.md R17)" if materialized else ""),
                    "derived_from": ["sweep.json#%d" % s8[0]]})
    return out, hyp


def memory_feasibility(audit, manifest, sweep, kernels, lengths: Iterable[int]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    heads = audit["encoder"]["num_heads"]
    ram = manifest["hardware"].get("ram_bytes")
    ram = ram if isinstance(ram, (int, float)) else None
    singles = _single(sweep)
    base = min((it.get("peak_rss_bytes") for _, (i, it) in singles.items()
                if it.get("peak_rss_bytes") and it.get("status") in ("measured", "partial")), default=None)
    kern: Dict[int, List[Tuple[int, Dict[str, Any]]]] = {}
    for i, it in _items(kernels):
        if it.get("peak_rss_bytes") and it.get("status") in ("measured", "partial"):
            kern.setdefault(it["shape"]["length"], []).append((i, it))
    out, missing = [], []
    cap = audit["positional"]["max_position_embeddings"]
    for L in sorted(set(lengths)):
        if L > cap:
            missing.append({"check": "memory at %d tokens" % L, "status": "unsupported",
                            "reason": "above positional capacity %d; not run" % cap})
            continue
        analytical = heads * L * L * 4
        entry = {"length": L, "score_matrix_bytes_analytical": analytical, "label_analytical": "estimated",
                 "ram_bytes": ram, "native_peak_rss_bytes": None, "native_peak_commit_bytes": None,
                 "kernel_peak_rss_bytes": {}, "derived_from": []}
        entry["full_matrix_fits_in_ram"] = (None if ram is None else
                                            bool(analytical + (base or 0) <= ram))
        avoid = []
        s = singles.get(L)
        if s and s[1].get("peak_rss_bytes") and s[1].get("status") in ("measured", "partial"):
            rss = s[1]["peak_rss_bytes"]
            entry["native_peak_rss_bytes"] = rss
            entry["native_peak_commit_bytes"] = s[1].get("peak_commit_bytes")
            entry["paging_suspected"] = bool(s[1].get("paging_suspected"))
            entry["derived_from"].append("sweep.json#%d" % s[0])
            grew = rss - base if base is not None else None
            entry["native_growth_bytes"] = grew
            observable = base is not None and analytical >= OBSERVABLE_FRACTION * base
            if not observable:
                entry["native_materialization"] = "not_observable"
            elif grew >= 0.5 * analytical:
                entry["native_materialization"] = "materialized"
            elif analytical >= 0.5 * base:
                entry["native_materialization"] = "avoided"
                avoid.append("native Laya (peak grew %s above its smallest-length peak, under half of %s)"
                             % (_gb(grew), _gb(analytical)))
            else:
                entry["native_materialization"] = "not_observable"
        for i, it in kern.get(L, []):
            key = it["impl"] + ("" if it["shape"].get("block") is None else "/b%s" % it["shape"]["block"]) + \
                  ("" if it["shape"].get("selection_blocks") is None else "/s%s" % it["shape"]["selection_blocks"])
            entry["kernel_peak_rss_bytes"][key] = it["peak_rss_bytes"]
            entry["derived_from"].append("kernels.json#%d" % i)
            if it["peak_rss_bytes"] < analytical:
                avoid.append("kernel %s (peak %s)" % (key, _gb(it["peak_rss_bytes"])))
        entry["paths_avoiding_full_matrix"] = avoid
        if entry["native_peak_rss_bytes"] is None and not entry["kernel_peak_rss_bytes"]:
            missing.append({"check": "memory measurement at %d tokens" % L,
                            "reason": "no native or kernel process measured at this length"})
        out.append(entry)
    return out, missing


def not_run_items(files: Dict[str, Optional[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    out = []
    for name in ("sweep.json", "profile.json", "kernels.json", "floor.json"):
        if files.get(name) is None:
            out.append({"check": name, "reason": "file not present in this run"})
    for name in ("sweep.json", "profile.json", "kernels.json"):
        for i, it in _items(files.get(name)):
            if it.get("status") in ("unsupported", "failed", "partial"):
                what = it.get("condition") or it.get("shape") or {}
                out.append({"check": "%s#%d" % (name, i), "status": it["status"],
                            "condition": what, "impl": it.get("impl"),
                            "cause": it.get("cause"), "reason": it.get("reason", "")})
    for n in (files.get("kernels.json") or {}).get("not_run", []):
        out.append(dict(n))
    fb = (files.get("floor.json") or {}).get("analytical_bound") or {}
    if fb.get("status") == "unsupported":
        out.append({"check": "analytical 4K bound", "reason": fb.get("reason")})
    return out


def high_variance(files) -> List[Dict[str, Any]]:
    out = []
    for name in ("sweep.json", "kernels.json"):
        for i, it in _items(files.get(name)):
            t = it.get("timings_ms") or {}
            if t.get("p50") and t.get("p95") and t["p95"] / t["p50"] > HIGH_VARIANCE_RATIO:
                out.append({"ref": "%s#%d" % (name, i), "p50_ms": t["p50"], "p95_ms": t["p95"],
                            "ratio": t["p95"] / t["p50"], "condition": it.get("condition") or it.get("shape")})
    return out


def drift_section(files) -> Dict[str, Any]:
    out = {"flagged": [], "canary": {}}
    for name in ("sweep.json", "profile.json"):
        body = files.get(name) or {}
        can = body.get("canary") or {}
        if can.get("enabled"):
            out["canary"][name] = {"max_deviation": can.get("max_deviation"), "baseline_p50_ms": can.get("baseline_p50_ms"),
                                   "threshold": can.get("threshold"), "runs": len(can.get("runs", []))}
        for i, it in _items(body):
            d = it.get("drift") or {}
            if d.get("flagged"):
                out["flagged"].append({"ref": "%s#%d" % (name, i), "max_deviation": d.get("max_deviation"),
                                       "condition": it.get("condition")})
    return out


def limitations(manifest, files, drift, beyond) -> List[Dict[str, Any]]:
    rt, hw = manifest["runtime"], manifest["hardware"]
    s = [
        stmt("measured", "Fixed thread count (research.md R12): %s intra-op threads (%s), hybrid_cores=%s. Absolute "
             "latencies hold for this setting on this machine; no thread sweep and no core pinning were done."
             % (rt.get("intra_op_threads"), rt.get("threads_source"), rt.get("hybrid_cores")), ["manifest.json#runtime"]),
        stmt("measured", "Documents are synthetic English-word filler built to exact token counts (research.md R4). "
             "These runs measure cost only and support no claim about answer quality.", []),
    ]
    if beyond:
        s.append(stmt("measured", "Lengths above the configured input cap (%s) were run with an explicit max_len and are "
                      "cost-only: the checkpoint was not validated there." % ", ".join(map(str, sorted(beyond))), []))
    can = drift["canary"]
    if can:
        s.append(stmt("measured", "Machine drift during the run (research.md R14): canary maximum deviation %s; %d "
                      "condition(s) flagged. Absolute times can differ by about that much between sessions; ratios "
                      "within the run are more stable." % (
                          ", ".join("%s %s" % (k, _pct(v["max_deviation"])) for k, v in can.items()),
                          len(drift["flagged"])), ["%s#canary" % k for k in can]))
    for name in ("sweep.json", "profile.json"):
        runs = [r for r in (((files.get(name) or {}).get("canary") or {}).get("runs") or []) if r.get("ratio_to_first")]
        if len(runs) > 1:
            lo, hi = min(r["ratio_to_first"] for r in runs), max(r["ratio_to_first"] for r in runs)
            s.append(stmt("measured", "%s canary spread: %.2f to %.2f of the first canary across the run. Treat "
                          "differences between conditions smaller than that as within the machine's own variation."
                          % (name, lo, hi), ["%s#canary" % name]))
    singles = {L: it for L, (i, it) in _single(files.get("sweep.json")).items()}
    base = min((it["peak_rss_bytes"] for it in singles.values()
                if it.get("peak_rss_bytes") and it.get("status") in ("measured", "partial")), default=None)
    if base:
        s.append(stmt("measured", "Peak memory is a process-lifetime maximum and includes loading the model (smallest "
                      "measured peak %s). Allocations smaller than that during a request cannot be seen, so memory "
                      "conclusions hold only where the growth is clearly above it." % _gb(base), ["sweep.json"]))
    gaps = []
    for i, it in _items(files.get("profile.json")):
        c = singles.get((it.get("condition") or {}).get("total_tokens"))
        if c and c.get("timings_ms") and it.get("total_profiled_ms"):
            gaps.append((it["condition"]["total_tokens"], it["total_profiled_ms"] / c["timings_ms"]["p50"], i))
    if gaps:
        s.append(stmt("measured", "Profiled total / clean p50 by length: %s. Values well below 1 mean profiled requests ran "
                      "faster than clean ones; component shares come from profiled runs, so this is checked by the "
                      "instrumentation A/B test (research.md R16)." % ", ".join("%d: %.2f" % (L, r) for L, r, _ in gaps),
                      ["profile.json#%d" % i for _, _, i in gaps]))
    heads = {it.get("head_path") for _, it in _items(files.get("profile.json")) if it.get("status") == "measured"}
    if files.get("profile.json") is not None:
        if heads == {None}:
            s.append(stmt("measured", "These profile runs predate research.md R17: their module hooks switched off PyTorch's "
                          "TransformerEncoderLayer fast path in the decision head, so the head measured there is not the "
                          "head the clean runs use, and the profiled total is lower than the clean one. Encoder components "
                          "are unaffected; the decision-head share is understated. Re-run `profile` to fix this.",
                          ["profile.json"]))
        elif heads:
            s.append(stmt("measured", "Decision-head path in the profile runs: %s (research.md R17)."
                          % ", ".join(sorted(h for h in heads if h)), ["profile.json"]))
    for i, it in _items(files.get("abtest.json")):
        rt = it.get("ratio_to_clean") or {}
        if rt:
            s.append(stmt("measured", "Instrumentation A/B at %s tokens, same documents in one process (p50 relative to "
                          "clean): %s." % (it["condition"]["total_tokens"],
                                           ", ".join("%s %.3f" % (k, v) for k, v in rt.items() if v)),
                          ["abtest.json#%d" % i]))
            if rt.get("fastpath_off"):
                s.append(stmt("measured", "At %s tokens, turning off only the decision head's fast path changes p50 by a "
                              "factor of %.3f with identical answers (research.md R17)."
                              % (it["condition"]["total_tokens"], rt["fastpath_off"]), ["abtest.json#%d" % i]))
    ov = (files.get("profile.json") or {}).get("overhead_check") or {}
    if ov.get("overhead_ratio"):
        s.append(stmt("measured", "Instrumentation overhead (T029, same requests, interleaved): stage wrappers change "
                      "p50 by a factor of %.3f. The profiled-to-clean ratio per length also contains machine drift "
                      "and is not an overhead measure." % ov["overhead_ratio"], ["profile.json#overhead_check"]))
    if manifest.get("fixture_model"):
        s.append(stmt("measured", "This run used the tiny test fixture model: its timings exercise the tooling only "
                      "and are not measurements of Laya.", ["manifest.json#fixture_model"]))
    if (manifest.get("code") or {}).get("laya_diff_empty") is False:
        s.append(stmt("measured", "laya/ differed from the recorded commit when the manifest was written (SC-005 not met).",
                      ["manifest.json#code"]))
    return s


def local_kernel_saving(profile, kernels, block: int = 128) -> List[Dict[str, Any]]:
    """Estimated saving per length from swapping native local-layer attention for the block-local kernel."""
    kt = {}
    for i, it in _items(kernels):
        if it.get("impl") == "local" and it.get("shape", {}).get("block") == block and \
                it.get("status") == "measured" and it.get("timings_ms"):
            kt[it["shape"]["length"]] = (i, it["timings_ms"]["p50"])
    out = []
    for L, (i, p) in sorted(_profiles(profile).items()):
        loc = (p.get("score_value_by_layer_type") or {}).get("local")
        if p.get("status") != "measured" or not loc or "per_layer_ms" not in loc or L not in kt \
                or not p.get("total_profiled_ms"):
            continue
        ki, kms = kt[L]
        saving = loc["n_layers"] * max(0.0, loc["per_layer_ms"] - kms)
        out.append({"length": L, "n_local_layers": loc["n_layers"], "native_per_layer_ms": loc["per_layer_ms"],
                    "kernel_ms": kms, "saving_ms": saving, "profiled_ms": p["total_profiled_ms"],
                    "saving_fraction": saving / p["total_profiled_ms"], "label": "estimate",
                    "derived_from": ["profile.json#%d" % i, "kernels.json#%d" % ki]})
    return [e for e in out if e["length"] >= 1024] or out


def implications(by_length, lat, profile, audit, floor, kernels, abtest=None) -> List[Dict[str, Any]]:
    """Which Phase 3 directions the data supports, weakens or leaves open, each tied to a measured cost."""
    out = []
    fp = [(it["condition"]["total_tokens"], it["ratio_to_clean"]["fastpath_off"], i)
          for i, it in _items(abtest) if (it.get("ratio_to_clean") or {}).get("fastpath_off")]
    if fp:
        best = min(fp, key=lambda x: x[1])
        out.append({"direction": "run the decision head through Laya's SDPA attention instead of PyTorch's fast path",
                    "stance": "supported" if best[1] < 0.95 else "open",
                    "statement": stmt("hypothesized", "Turning the fast path off gave %s with identical answers (measured); "
                                      "it needs no retraining and no change to the model's outputs."
                                      % "; ".join("%.0f%% of clean p50 at %d tokens" % (100 * r, L) for L, r, _ in sorted(fp)),
                                      ["abtest.json#%d" % i for _, _, i in fp])})
    profs = _profiles(profile)
    if by_length:
        last = by_length[-1]
        mid = min(by_length, key=lambda b: abs(math.log(b["length"] / 2048)))
        lin_mid = sum(c["share"] for c in mid["components"] if c["component"] in LINEAR)
        lin_last = sum(c["share"] for c in last["components"] if c["component"] in LINEAR)
        out.append({"direction": "reduce the tokens that reach the encoder (compression, chunk selection)",
                    "stance": "supported" if lin_mid >= 0.5 else "open",
                    "statement": stmt("hypothesized", "Projections and MLP take %s of time at %d tokens and %s at %d "
                                      "tokens (measured). Faster attention cannot touch that part; only fewer tokens "
                                      "through the encoder reduces it." % (_pct(lin_mid), mid["length"], _pct(lin_last),
                                                                         last["length"]),
                                      mid["derived_from"] + last["derived_from"])})
        lin = lin_last
        head = next((c["share"] for c in last["components"] if c["component"] == "decision_head"), 0.0)
        out.append({"direction": "restrict or shorten the decision head's full-sequence pass",
                    "stance": "supported" if head >= 0.05 else "weakened",
                    "statement": stmt("hypothesized", "The 2-layer head takes %s at %d tokens (measured)."
                                      % (_pct(head), last["length"]), last["derived_from"])})
    ew = (audit.get("executed_work_note") or {}).get("verdict")
    loc = []
    for L, (i, p) in sorted(profs.items()):
        bt = (p.get("score_value_by_layer_type") or {}).get("local")
        tot = p.get("total_profiled_ms")
        if bt and tot and p.get("status") == "measured":
            loc.append((L, bt["total_ms"] / tot, "profile.json#%d" % i))
    if ew == "dense_masked" and loc:
        L, sh, ref = loc[-1]
        speed = ""
        kgroups = {(g["impl"], g["block"]): g for g in ((kernels or {}).get("executed_work") or {}).get("groups", [])}
        if kgroups.get(("local", 128), {}).get("verdict") == "less_work":
            speed = " The block-local kernel shows less executed work (time exponent %.2f)." % kgroups[("local", 128)]["time_exponent"]
        out.append({"direction": "make the native local layers skip out-of-window work",
                    "stance": "supported",
                    "statement": stmt("hypothesized", "Local layers run dense masked attention; their score/value work is %s "
                                      "of profiled time at %d tokens, most of which a real block-local kernel would remove.%s"
                                      % (_pct(sh), L, speed), [ref, "audit.json#executed_work_note"])})
        est = local_kernel_saving(profile, kernels)
        if est:
            out.append({"direction": "estimated saving from a block-local kernel in the native local layers",
                        "stance": "supported" if max(e["saving_fraction"] for e in est) >= 0.1 else "open",
                        "estimates": est,
                        "statement": stmt("estimated", "Replacing each native local layer's attention with the measured "
                                          "block-local kernel (block 128) would save about %s. Each figure is n_local x "
                                          "(native per-layer local time - kernel time), from separate processes, against "
                                          "the profiled total. The kernel's pattern (own and neighbouring 128-token "
                                          "blocks) covers more than the native +/-64 window, so an exact kernel needs "
                                          "its own correctness check."
                                          % "; ".join("%s of %s at %d tokens (%s)" % (_ms(e["saving_ms"]), _ms(e["profiled_ms"]),
                                                                                     e["length"], _pct(e["saving_fraction"]))
                                                      for e in est),
                                          sorted({r for e in est for r in e["derived_from"]}))})
    multi = [r for r in lat["multi_and_batch"] if r["kind"] == "multi_question" and r.get("multiplier_vs_single")
             and r["status"] == "measured"]
    if multi:
        eff = [r["multiplier_vs_single"] / r["units"] for r in multi]
        worst = max(multi, key=lambda r: (r["units"], r["length"]))
        out.append({"direction": "encode the document once for several questions",
                    "stance": "supported" if min(eff) >= 0.7 else "open",
                    "statement": stmt("hypothesized", "Each extra question costs about a full forward pass: %d questions take "
                                      "%.1fx one question at %d tokens (measured), so shared document encoding could "
                                      "save close to that factor on multi-question requests."
                                      % (worst["units"], worst["multiplier_vs_single"], worst["length"]), [worst["ref"]])})
    batch = [r for r in lat["multi_and_batch"] if r["kind"] == "batch" and r.get("multiplier_vs_single")
             and r["status"] == "measured"]
    if batch:
        b = max(batch, key=lambda r: (r["units"], r["length"]))
        out.append({"direction": "batch documents for CPU throughput",
                    "stance": "weakened" if b["multiplier_vs_single"] >= 0.75 * b["units"] else "supported",
                    "statement": stmt("hypothesized", "A batch of %d documents takes %.1fx one document at %d tokens "
                                      "(measured): the CPU is already busy with one request."
                                      % (b["units"], b["multiplier_vs_single"], b["length"]), [b["ref"]])})
    fb = (floor or {}).get("analytical_bound") or {}
    if fb.get("bound_ratio_4096_vs_512"):
        out.append({"direction": "sparse attention alone reaching 4K at no more than 2x the 512-token latency",
                    "stance": "weakened" if fb["bound_ratio_4096_vs_512"] > 2 else "open",
                    "statement": stmt("estimated", "Even with free attention, linear work puts 4,096 tokens at about %.1fx "
                                      "the 512-token latency (analytical bound from the measured 512 profile)."
                                      % fb["bound_ratio_4096_vs_512"], ["floor.json#analytical_bound"])})
    return out


# --------------------------------------------------------------------------- assembly

def summary(lat, by_length, plan_assumptions, hypotheses, audit, manifest=None) -> List[Dict[str, Any]]:
    out = []
    variant = ((manifest or {}).get("runtime") or {}).get("variant")
    if variant:
        out.append(stmt("measured", "This run is the variant %s, not native Laya: its numbers never stand in for the "
                        "native results (research.md R17)." % variant, ["manifest.json#runtime"]))
    meas = [r for r in lat["primary"] if r["status"] in ("measured", "partial") and r["p50_ms"]]
    if meas:
        lo, hi = meas[0], meas[-1]
        out.append(stmt("measured", "One document, one question: p50 %s at %d tokens and %s at %d tokens%s%s."
                        % (_ms(lo["p50_ms"]), lo["length"], _ms(hi["p50_ms"]), hi["length"],
                           " (partial: %s repeats)" % hi["n"] if hi["status"] == "partial" else "",
                           ", cost-only above the configured cap" if hi["cost_only"] else ""), [lo["ref"], hi["ref"]]))
    if by_length:
        b = by_length[-1]
        top = b["components"][0]
        out.append(stmt("measured", "Largest cost at %d tokens: %s, %s of the clean p50."
                        % (b["length"], COMPONENT_NAMES.get(top["component"], top["component"]), _pct(top["share"])),
                        b["derived_from"]))
        if b.get("all_attention_share") is not None:
            out.append(stmt("measured", "All attention score/value at %d tokens, encoder plus decision head: %s."
                            % (b["length"], _pct(b["all_attention_share"])), b["derived_from"]))
    v = (audit.get("executed_work_note") or {}).get("verdict")
    if v:
        out.append(stmt("measured", "Local-layer executed work: %s." % v, ["audit.json#executed_work_note"]))
    counts = {}
    for a in plan_assumptions + hypotheses:
        counts[a["verdict"]] = counts.get(a["verdict"], 0) + 1
    out.append(stmt("measured", "Plan assumptions and pre-registered hypotheses: %s."
                    % ", ".join("%d %s" % (n, k) for k, n in sorted(counts.items())), []))
    return out


def _read_commands(run: Path) -> List[str]:
    p = run / "commands.txt"
    return [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()] if p.exists() else []


def build_report(run_path: Union[str, Path]) -> Dict[str, Any]:
    run = Path(run_path)
    manifest, audit = _load(run, "manifest.json"), _load(run, "audit.json")
    missing = [n for n, b in (("manifest.json", manifest), ("audit.json", audit)) if b is None]
    if missing:
        raise ReportError("report needs %s in %s; run `audit` for this run first" % (" and ".join(missing), run))
    files = {n: _load(run, n) for n in ("sweep.json", "profile.json", "kernels.json", "floor.json",
                                         "gpu_reference.json", "abtest.json")}
    if files["profile.json"] is not None:
        # floor.json is written by `kernels`; rebuild it so it always matches the current profile
        from .floor import build_floor
        files["floor.json"] = build_floor(run)
        results.write_json(run, "floor.json", files["floor.json"])
    sweep, profile, kernels, floor, gpu = (files[n] for n in ("sweep.json", "profile.json", "kernels.json",
                                                              "floor.json", "gpu_reference.json"))
    lat = latency_tables(sweep)
    by_length, cost_floor, rank_missing = ranking(sweep, profile, floor)
    plan_assumptions, hypotheses = assumptions(audit, sweep, profile, gpu)
    lengths = set(r["length"] for r in lat["primary"]) | {it["shape"]["length"] for _, it in _items(kernels)} | \
              {l["length"] for l in audit.get("lengths", [])}
    mem, mem_missing = memory_feasibility(audit, manifest, sweep, kernels, lengths)
    drift = drift_section(files)
    beyond = {r["length"] for r in lat["primary"] if r["cost_only"]}
    not_run = not_run_items(files) + rank_missing + mem_missing
    if gpu is None:
        not_run.append({"check": "gpu_reference.json", "reason": "GPU reference check not run (research.md R15)"})
    body = {
        "summary": summary(lat, by_length, plan_assumptions, hypotheses, audit, manifest),
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "audit_summary": audit_summary(audit, manifest),
        "environment": environment(manifest) + sessions(files),
        "latency": lat,
        "by_length": by_length,
        "cost_floor": cost_floor,
        "floor_analytical": (floor or {}).get("analytical_bound"),
        "assumptions": plan_assumptions,
        "hypotheses": hypotheses,
        "memory_feasibility": mem,
        "limitations": limitations(manifest, files, drift, beyond),
        "high_variance": high_variance(files),
        "drift": drift,
        "not_run": not_run,
        "reproduce": _read_commands(run),
        "phase3_implications": implications(by_length, lat, profile, audit, floor, kernels, files.get("abtest.json")),
        "kernels_executed_work": ((kernels or {}).get("executed_work") or {}).get("groups", []),
        "statement_status": {"tags": list(TAGS), "rule": "every statement in report.md begins with its tag"},
    }
    return body


# --------------------------------------------------------------------------- markdown

def _line(s: Dict[str, Any]) -> str:
    refs = s.get("derived_from") or []
    return "- [%s] %s%s" % (s["tag"], s["text"], (" (" + ", ".join(refs) + ")") if refs else "")


def render_markdown(body: Dict[str, Any], run_id: str) -> str:
    L: List[str] = []
    add = L.append
    add("# Phase 1 CPU path report: run `%s`" % run_id)
    add("")
    add("- [measured] Generated %s from the result files of this run; every figure cites `file#index`." % body["generated_at"])
    add("")
    add("## Summary")
    add("")
    L.extend(_line(s) for s in body["summary"])
    add("")
    add("## Audit summary")
    add("")
    L.extend(_line(s) for s in body["audit_summary"]["statements"])
    add("")
    add("## Environment")
    add("")
    L.extend(_line(s) for s in body["environment"])
    add("")
    add("## Latency: one document, one question")
    add("")
    add("| Tokens | Status | p50 | p95 | n | Notes |")
    add("|---|---|---|---|---|---|")
    for r in body["latency"]["primary"]:
        notes = [n for n, on in (("cost-only (beyond configured cap)", r["cost_only"]), ("drift flagged", r["drift_flagged"]),
                                 ("low-sample p95", r["low_sample_p95"])) if on]
        if r["status"] != "measured" and r.get("reason"):
            notes.append(r["reason"])
        add("| %s | %s | %s | %s | %s | %s |" % (r["length"], r["status"], _ms(r["p50_ms"]), _ms(r["p95_ms"]),
                                              r["n"] or "", "; ".join(notes) + " (%s)" % r["ref"]))
    add("")
    for r in body["latency"]["primary"]:
        if r["p50_ms"]:
            add("- [measured] %d tokens: p50 %s%s. (%s)" % (r["length"], _ms(r["p50_ms"]),
                                                          ", cost-only beyond the configured input cap" if r["cost_only"] else "",
                                                          r["ref"]))
    add("")
    add("## Multi-question and batch (never part of the single-request curve)")
    add("")
    add("| Tokens | Kind | Questions or docs | Status | p50 | x one question | Ref |")
    add("|---|---|---|---|---|---|---|")
    for r in body["latency"]["multi_and_batch"]:
        m = r.get("multiplier_vs_single")
        add("| %s | %s | %s | %s | %s | %s | %s |" % (r["length"], r["kind"], r["units"], r["status"], _ms(r["p50_ms"]),
                                                   "%.1fx" % m if m else "n/a", r["ref"]))
    add("")
    add("## Bottlenecks by length")
    add("")
    floors = {f["length"]: f for f in body["cost_floor"]}
    for b in body["by_length"]:
        add("### %d tokens (clean p50 %s)%s" % (b["length"], _ms(b["clean_p50_ms"]),
                                                "  - cost-only" if b["cost_only"] else ""))
        add("")
        if b.get("clean_status") == "partial":
            rp = b.get("clean_repeats") or {}
            add("- [measured] The clean run here is partial (%s of %s repeats before the time cap); its p50 is "
                "used with that caveat." % (rp.get("completed"), rp.get("requested")))
        for rank, c in enumerate(b["components"], 1):
            if c["share"] < 0.005:
                continue
            add("- [measured] %d. %s: %s, about %s of the clean p50. (%s)" % (
                rank, COMPONENT_NAMES.get(c["component"], c["component"]), _pct(c["share"]), _ms(c["ms_of_clean_p50"]),
                ", ".join(c["derived_from"])))
        if b.get("all_attention_share") is not None:
            add("- [measured] All attention score/value, encoder plus decision head: %s of profiled time. (%s)"
                % (_pct(b["all_attention_share"]), b["derived_from"][1]))
        f = floors.get(b["length"])
        if f:
            add("- [estimated] Cost floor with free attention: %s (%s of the total). (%s)" % (
                _ms(f["floor_ms"]), _pct(f["floor_fraction"]), ", ".join(f["derived_from"])))
        if b["drift_flagged"]:
            add("- [measured] Machine drift was flagged around this length; compare its absolute times with care.")
        add("")
    fa = body.get("floor_analytical") or {}
    if fa.get("bound_ratio_4096_vs_512"):
        add("- [estimated] Analytical bound: with free attention, 4,096 tokens still take at least %.1fx the 512-token "
            "latency%s. (floor.json#analytical_bound)" % (fa["bound_ratio_4096_vs_512"],
                                                          " (about %s)" % _ms(fa["bound_ms_4096"]) if fa.get("bound_ms_4096") else ""))
        add("")
    add("## Plan assumptions")
    add("")
    for a in body["assumptions"]:
        add("- [measured] %s: **%s**. %s (%s)" % (a["assumption"], a["verdict"], a["evidence"], ", ".join(a["derived_from"]) or "no data"))
    add("")
    add("## Pre-registered hypotheses (research.md R13)")
    add("")
    for h in body["hypotheses"]:
        add("- [measured] %s %s: **%s**. %s (%s)" % (h["id"], h["hypothesis"], h["verdict"], h["evidence"],
                                                     ", ".join(h["derived_from"]) or "no data"))
    add("")
    add("## Memory feasibility")
    add("")
    add("| Tokens | One score matrix (analytical) | Native peak (measured) | Fits in RAM | Paths that avoid it |")
    add("|---|---|---|---|---|")
    for m in body["memory_feasibility"]:
        add("| %s | %s | %s | %s | %s |" % (m["length"], _gb(m["score_matrix_bytes_analytical"]), _gb(m["native_peak_rss_bytes"]),
                                            {True: "yes", False: "no", None: "unknown"}[m["full_matrix_fits_in_ram"]],
                                            "; ".join(m["paths_avoiding_full_matrix"]) or "none measured"))
    add("")
    for m in body["memory_feasibility"]:
        add("- [estimated] %d tokens: one full fp32 score matrix (batch 1, all heads) needs %s; together with the "
            "smallest measured native peak it %s in RAM. (analytical)" % (
            m["length"], _gb(m["score_matrix_bytes_analytical"]),
            {True: "would fit", False: "would not fit", None: "cannot be compared"}[m["full_matrix_fits_in_ram"]]))
        if m["native_peak_rss_bytes"] is not None or m["kernel_peak_rss_bytes"]:
            mat = {"materialized": "; the peak grew %s above the smallest-length peak, consistent with full score "
                                   "matrices being materialized" % _gb(m.get("native_growth_bytes")),
                   "avoided": "; no growth of that size, so native Laya does not materialize it",
                   "not_observable": "; a matrix this small is hidden under the model-load peak, so this cannot "
                                     "tell whether it is materialized"}.get(m.get("native_materialization"), "")
            add("- [measured] %d tokens: native peak %s%s%s. Paths that avoid materializing it: %s. (%s)" % (
                m["length"], _gb(m["native_peak_rss_bytes"]), mat,
                ", paging suspected" if m.get("paging_suspected") else "",
                "; ".join(m["paths_avoiding_full_matrix"]) or "none measured", ", ".join(m["derived_from"])))
    add("")
    if body["kernels_executed_work"]:
        add("## Kernel executed work")
        add("")
        for g in body["kernels_executed_work"]:
            add("- [measured] %s (block %s, selection %s): time exponent %s across %s tokens, verdict %s. (%s)" % (
                g["impl"], g["block"], g["selection_blocks"],
                "%.2f" % g["time_exponent"] if g["time_exponent"] is not None else "n/a",
                ", ".join(map(str, g["lengths"])), g["verdict"], ", ".join(g["derived_from"][:3]) + (" ..." if len(g["derived_from"]) > 3 else "")))
        add("")
    add("## Implications for Phase 3")
    add("")
    for i in body["phase3_implications"]:
        s = i["statement"]
        add("- [%s] %s: **%s**. %s%s" % (s["tag"], i["direction"], i["stance"], s["text"],
                                         (" (" + ", ".join(s["derived_from"]) + ")") if s["derived_from"] else ""))
    add("")
    add("## Limitations")
    add("")
    L.extend(_line(s) for s in body["limitations"])
    add("")
    add("## High variance (p95 / p50 > %.1f)" % HIGH_VARIANCE_RATIO)
    add("")
    if body["high_variance"]:
        for h in body["high_variance"]:
            add("- [measured] %s: p95/p50 = %.2f (p50 %s, p95 %s)." % (h["ref"], h["ratio"], _ms(h["p50_ms"]), _ms(h["p95_ms"])))
    else:
        add("- [measured] No condition exceeded the threshold.")
    add("")
    add("## Drift-flagged conditions")
    add("")
    if body["drift"]["flagged"]:
        groups: List[List[Dict[str, Any]]] = []
        for d in body["drift"]["flagged"]:
            f, n = d["ref"].split("#")
            if groups and groups[-1][-1]["ref"].split("#")[0] == f and \
                    int(groups[-1][-1]["ref"].split("#")[1]) == int(n) - 1 and \
                    round(groups[-1][-1]["max_deviation"] or 0, 3) == round(d["max_deviation"] or 0, 3):
                groups[-1].append(d)
            else:
                groups.append([d])
        for g in groups:
            ref = g[0]["ref"] if len(g) == 1 else "%s-%s" % (g[0]["ref"], g[-1]["ref"].split("#")[1])
            add("- [measured] %s: canary deviation %s." % (ref, _pct(g[0]["max_deviation"])))
    else:
        add("- [measured] None flagged%s." % ("" if body["drift"]["canary"] else " (no canary in this run)"))
    add("")
    add("## Not run, cut short, or unsupported")
    add("")
    for n in body["not_run"]:
        add("- [measured] %s%s: %s" % (n["check"], " (%s)" % n["status"] if n.get("status") else "", n.get("reason") or n.get("cause") or ""))
    add("")
    add("## Reproduce")
    add("")
    add("```")
    L.extend(body["reproduce"])
    add("```")
    add("")
    return "\n".join(L)


def write_report(run_path: Union[str, Path]) -> Tuple[Path, Path]:
    run = Path(run_path)
    body = build_report(run)
    jpath = results.write_json(run, "report.json", body)
    mpath = run / "report.md"
    mpath.write_text(render_markdown(body, run.name), encoding="utf-8", newline="\n")
    return jpath, mpath
