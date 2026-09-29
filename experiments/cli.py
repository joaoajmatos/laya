"""Command-line entry point: ``python -m experiments <command> [options]``.

Contract: specs/001-cpu-path-audit/contracts/cli.md.

Commands register themselves with `command(...)`; later tasks add manifest, audit, sweep,
profile, kernels, report and all. Exit codes: 0 when the command completed, including when some
conditions were recorded as ``unsupported`` or ``failed`` (those are results). Non-zero only when
the tool itself could not run (bad arguments, model cannot load, device is not CPU).
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

from . import results

DEFAULT_MODEL = "convaiinnovations/laya"
EXIT_OK = 0
EXIT_TOOL_ERROR = 1
EXIT_USAGE = 2


class ToolError(Exception):
    """The tool cannot run: reported on stderr and turned into a non-zero exit."""


@dataclass
class Command:
    name: str
    help: str
    run: Callable[[argparse.Namespace], Optional[int]]
    add_arguments: Optional[Callable[[argparse.ArgumentParser], None]] = None


#: name -> Command, in registration order.
COMMANDS: Dict[str, Command] = {}


def command(name: str, help: str, add_arguments: Optional[Callable[[argparse.ArgumentParser], None]] = None):
    """Decorator registering `run(args)` as a subcommand."""
    def register(run):
        if name in COMMANDS:
            raise ValueError("command %r registered twice" % name)
        COMMANDS[name] = Command(name, help, run, add_arguments)
        return run
    return register


def _positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return value


def int_list(text: str) -> List[int]:
    """Parse ``"128,512,4096"``."""
    try:
        values = [int(v) for v in text.split(",") if v.strip()]
    except ValueError:
        raise argparse.ArgumentTypeError("expected comma-separated integers, got %r" % text)
    if not values or any(v < 1 for v in values):
        raise argparse.ArgumentTypeError("expected positive integers, got %r" % text)
    return values


def common_parser() -> argparse.ArgumentParser:
    """Options every command accepts (contracts/cli.md, "Common options")."""
    p = argparse.ArgumentParser(add_help=False)
    g = p.add_argument_group("common options")
    g.add_argument("--run-id", default=None,
                   help="results directory name under experiments/results/ (default: UTC timestamp)")
    g.add_argument("--model", default=DEFAULT_MODEL, help="checkpoint id or local path (default: %(default)s)")
    g.add_argument("--revision", default=None,
                   help="commit to pin; 'reviewed' uses laya.revisions.PINNED_REVISIONS")
    g.add_argument("--threads", type=_positive_int, default=None,
                   help="torch intra-op threads, fixed for the whole run (default: physical core count)")
    g.add_argument("--seed", type=int, default=0, help="base seed (default: %(default)s)")
    g.add_argument("--mha-fastpath", choices=["on", "off"], default=None,
                   help="PyTorch's TransformerEncoderLayer inference fast path, used by Laya's decision head. "
                        "'on' (default) is native Laya; 'off' is a labelled variant (research.md R17). One setting per run.")
    return p


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m experiments",
        description="Phase 1 CPU path audit and measurement for Laya (specs/001-cpu-path-audit).",
    )
    sub = parser.add_subparsers(dest="command", metavar="<command>")
    parent = common_parser()
    for cmd in COMMANDS.values():
        sp = sub.add_parser(cmd.name, help=cmd.help, description=cmd.help, parents=[parent])
        if cmd.add_arguments:
            cmd.add_arguments(sp)
    if not COMMANDS:
        parser.epilog = "No commands are registered yet."
    return parser


def resolve_common(args: argparse.Namespace) -> argparse.Namespace:
    """Fill defaults that need torch or the filesystem, and create the run directory.

    The thread count is resolved once here and passed to every child, so every condition in a
    run uses the same value (research.md R12). Its source is recorded.
    """
    if args.threads is None:
        from .runner import default_threads
        args.threads = default_threads()
        args.threads_source = "default"
    else:
        args.threads_source = "user"
    args.run_path = results.run_dir(args.run_id)
    args.run_id = args.run_path.name
    _match_run_manifest(args)
    if getattr(args, "mha_fastpath", "on") is None:
        args.mha_fastpath = "on"
    return args


def _match_run_manifest(args: argparse.Namespace) -> None:
    """Keep one model and revision per run directory.

    When the run already has a manifest, a command that names no revision uses the manifest's
    resolved commit (so `sweep` after `audit --revision reviewed` measures the same weights, and
    works offline). A different model or revision in the same run is refused.
    """
    try:
        man = results.read_json(args.run_path, "manifest.json")
    except (OSError, ValueError):
        return
    model = man.get("model") or {}
    recorded_fp = (man.get("runtime") or {}).get("mha_fastpath", True)
    if getattr(args, "mha_fastpath", "on") is None:
        args.mha_fastpath = "on" if recorded_fp else "off"   # inherit, like the revision
    elif (args.mha_fastpath == "on") != bool(recorded_fp):
        raise ToolError("run %r was recorded with --mha-fastpath %s; use another --run-id for the other setting"
                        % (args.run_id, "on" if recorded_fp else "off"))
    if model.get("id") and model["id"] != args.model:
        raise ToolError("run %r was recorded with model %r; this command asks for %r. Use another --run-id."
                        % (args.run_id, model["id"], args.model))
    recorded = model.get("revision")
    if not recorded or recorded == "local" or model.get("revision_source") == "local":
        return
    if args.revision in (None, ""):
        args.revision = recorded
        print("using revision %s from this run's manifest.json" % recorded)
        return
    from .runner import resolve_model_revision
    try:
        wanted, _ = resolve_model_revision(args.model, args.revision)
    except ValueError as exc:
        raise ToolError(str(exc))
    if wanted != recorded:
        raise ToolError("run %r was recorded at revision %s; this command asks for %s. Use another --run-id."
                        % (args.run_id, recorded, wanted))


def main(argv: Optional[Sequence[str]] = None) -> int:
    _load_commands()
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help(sys.stderr)
        return EXIT_USAGE
    cmd = COMMANDS[args.command]
    try:
        resolve_common(args)
        results.append_command(args.run_path, argv)
        code = cmd.run(args)
    except ToolError as exc:
        print("error: %s" % exc, file=sys.stderr)
        return EXIT_TOOL_ERROR
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
    return EXIT_OK if code is None else int(code)


def _load_commands() -> None:
    """Commands are registered below in this module; nothing else to import yet."""
    return None


# --------------------------------------------------------------------------- shared helpers

def load_model(args: argparse.Namespace):
    """Load the agent on CPU for a command; tool-level failures become `ToolError`."""
    from .runner import load_agent
    try:
        return load_agent(args.model, revision=args.revision, threads=args.threads,
                          mha_fastpath=args.mha_fastpath == "on")
    except Exception as exc:  # model cannot load, or device is not CPU
        raise ToolError("could not load %r on CPU: %s: %s" % (args.model, type(exc).__name__, exc))


def write_manifest(args: argparse.Namespace, agent, info) -> str:
    from .manifest import ManifestError, build_manifest
    try:
        body = build_manifest(agent, info, run_id=args.run_id, seed=args.seed,
                              threads_source=args.threads_source)
    except ManifestError as exc:
        raise ToolError(str(exc))
    if body["device"]["effective"] != "cpu":
        raise ToolError("effective device is %s, not cpu" % body["device"]["effective"])
    return str(results.write_json(args.run_path, "manifest.json", body))


# --------------------------------------------------------------------------- manifest / audit (T019)

@command("manifest", "Write manifest.json: the pinned environment of this run.")
def _cmd_manifest(args: argparse.Namespace) -> int:
    agent, info = load_model(args)
    path = write_manifest(args, agent, info)
    print(path)
    return EXIT_OK


def _audit_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--lengths", type=int_list, default=None,
                   help="comma-separated total lengths to check against positional capacity")


@command("audit", "Write audit.json (and manifest.json): what the loaded model actually is.", _audit_args)
def _cmd_audit(args: argparse.Namespace) -> int:
    from .audit import build_audit, format_schedule
    agent, info = load_model(args)
    print(write_manifest(args, agent, info))
    body = build_audit(agent, info, lengths=args.lengths)
    print(format_schedule(body))
    print(results.write_json(args.run_path, "audit.json", body))
    return EXIT_OK


# --------------------------------------------------------------------------- sweep / profile (T027, T028)

DEFAULT_LENGTHS = [128, 256, 512, 1024, 2048, 4096, 8192]
DEFAULT_TIME_CAP = 900.0


def _repeats_arg(text: str):
    if text == "auto":
        return "auto"
    try:
        return _positive_int(text)
    except (ValueError, argparse.ArgumentTypeError):
        raise argparse.ArgumentTypeError("expected 'auto' or a positive integer")


def _known_position_limit(run_path) -> Optional[int]:
    try:
        return int(results.read_json(run_path, "audit.json")["positional"]["max_position_embeddings"])
    except (OSError, KeyError, ValueError, TypeError):
        return None


def session_info() -> Dict[str, Any]:
    """Machine state when a measuring command starts: the power plan may differ from the manifest's."""
    import datetime as _dt
    from .manifest import hardware_info
    hw = hardware_info()
    return {"started_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
            "power_scheme": hw.get("power_scheme", "not recorded"),
            "on_ac_power": hw.get("on_ac_power", "not recorded")}


def sweep_conditions(lengths, questions, batch_sizes, grid: str = "axes"):
    """(length, questions, batch) tuples. ``axes`` varies one factor at a time around the primary
    curve (questions=1, batch=1); ``full`` is the complete product."""
    out = []
    for L in lengths:
        if grid == "full":
            out.extend((L, q, b) for q in questions for b in batch_sizes)
            continue
        out.append((L, 1, 1))
        out.extend((L, q, 1) for q in questions if q != 1)
        out.extend((L, 1, b) for b in batch_sizes if b != 1)
    seen, uniq = set(), []
    for c in out:
        if c not in seen:
            seen.add(c)
            uniq.append(c)
    return uniq


def _abort_on_load_failure(item: Dict[str, Any], args: argparse.Namespace, name: str, body) -> None:
    """A model that cannot load is a tool error (contracts/cli.md), not a failed measurement."""
    if item.get("exception_type") != "ModelLoadError":
        return
    results.write_json(args.run_path, name, body)
    raise ToolError("the model could not be loaded in the measurement process, so nothing was measured "
                    "(%s). Partial file: %s" % (item.get("reason"), args.run_path / name))


def _unsupported(L, q, opts, b, limit):
    from .timing import condition_kind
    kind = condition_kind(q, b)
    return {"status": "unsupported",
            "reason": "%d tokens exceeds positional capacity %d (from audit.json); no model call made" % (L, limit),
            "condition": {"total_tokens": L, "questions": q, "options_per_question": opts, "batch_size": b},
            "kind": kind, "call": "predict_batch" if kind == "batch" else "predict"}


def _sweep_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--lengths", type=int_list, default=DEFAULT_LENGTHS)
    p.add_argument("--questions", type=int_list, default=[1, 2, 5, 10])
    p.add_argument("--options", type=_positive_int, default=2, help="options per question (default 2)")
    p.add_argument("--batch-sizes", type=int_list, default=[1, 4, 8])
    p.add_argument("--grid", choices=["axes", "full"], default="axes",
                   help="axes: vary questions and batch size one at a time around questions=1, batch=1 "
                        "(default); full: every combination")
    p.add_argument("--repeats", type=_repeats_arg, default="auto")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--time-cap", type=float, default=DEFAULT_TIME_CAP,
                   help="seconds per condition, model load included (default %(default)s)")
    _canary_args(p)


def _canary_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--canary-length", type=int, default=512,
                   help="drift canary: single-question length re-measured before each length and at "
                        "the end (0 disables; default %(default)s)")
    p.add_argument("--canary-repeats", type=_positive_int, default=5)
    p.add_argument("--drift-threshold", type=float, default=0.05,
                   help="flag conditions whose surrounding canaries differ from the first by more than "
                        "this fraction (default %(default)s)")


class Canary:
    """Re-measures one fixed short condition to detect machine drift (heat, turbo budget, other
    programs) during a long run. Canary results never enter the measurement curves."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.enabled = bool(args.canary_length)
        self.runs: List[Dict[str, Any]] = []

    def measure(self, before_item: Optional[int]) -> Optional[Dict[str, Any]]:
        if not self.enabled:
            return None
        from .runner import run_condition
        a = self.args
        spec = {"model": a.model, "revision": a.revision, "threads": a.threads, "mha_fastpath": a.mha_fastpath == "on", "seed": a.seed + 7_000_000,
                "total_tokens": a.canary_length, "questions": 1, "options": 2, "batch_size": 1,
                "repeats": a.canary_repeats, "warmup": 2}
        item = run_condition("experiments.timing:measure_condition", spec, time_cap=a.time_cap)
        p50 = (item.get("timings_ms") or {}).get("p50")
        base = self.baseline
        run = {"index": len(self.runs), "before_item": before_item, "status": item["status"],
               "p50_ms": p50, "samples_ms": item.get("samples_ms"),
               "ratio_to_first": (p50 / base) if (p50 and base) else (1.0 if p50 and not self.runs else None)}
        if item.get("reason"):
            run["reason"] = item["reason"]
        if item.get("exception_type"):
            run["exception_type"] = item["exception_type"]
        self.runs.append(run)
        print("  canary L=%d: %s" % (a.canary_length, "p50 %.1f ms (x%.3f of first)" % (p50, run["ratio_to_first"])
                                     if p50 else item["status"]), flush=True)
        return item

    @property
    def baseline(self) -> Optional[float]:
        for r in self.runs:
            if r["p50_ms"]:
                return r["p50_ms"]
        return None

    def annotate(self, items: List[Dict[str, Any]]) -> None:
        """Give every item the canaries measured just before and just after it."""
        if not self.enabled:
            return
        thr = self.args.drift_threshold
        for i, it in enumerate(items):
            before = [r for r in self.runs if r["before_item"] is not None and r["before_item"] <= i]
            after = [r for r in self.runs if r["before_item"] is None or r["before_item"] > i]
            b = before[-1] if before else None
            a = after[0] if after else None
            ratios = [r["ratio_to_first"] for r in (b, a) if r and r.get("ratio_to_first")]
            worst = max((abs(x - 1.0) for x in ratios), default=None)
            it["drift"] = {
                "canary_before": b["index"] if b else None,
                "canary_after": a["index"] if a else None,
                "ratios_to_first": ratios,
                "max_deviation": worst,
                "flagged": bool(worst is not None and worst > thr),
                "derived_from": ["%s#canary.runs.%d" % (self.args._results_file, r["index"]) for r in (b, a) if r],
            }

    def block(self) -> Dict[str, Any]:
        if not self.enabled:
            return {"enabled": False}
        dev = [abs(r["ratio_to_first"] - 1.0) for r in self.runs if r.get("ratio_to_first")]
        return {"enabled": True, "length": self.args.canary_length, "repeats": self.args.canary_repeats,
                "threshold": self.args.drift_threshold, "baseline_p50_ms": self.baseline,
                "max_deviation": max(dev) if dev else None, "runs": self.runs}


@command("sweep", "Clean latency sweep, one subprocess per condition; writes sweep.json.", _sweep_args)
def _cmd_sweep(args: argparse.Namespace) -> int:
    from .runner import run_condition
    limit = _known_position_limit(args.run_path)
    items = []
    conds = sweep_conditions(args.lengths, args.questions, args.batch_sizes, args.grid)
    args._results_file = "sweep.json"
    canary = Canary(args)
    last_len = None

    session = session_info()

    def body():
        canary.annotate(items)
        return {"grid": args.grid, "session": session, "items": items, "canary": canary.block()}

    for n, (L, q, b) in enumerate(conds, 1):
        if L != last_len and not (limit is not None and L > limit):
            _abort_on_load_failure(canary.measure(len(items)) or {}, args, "sweep.json", body())
            last_len = L
        if limit is not None and L > limit:
            item = _unsupported(L, q, args.options, b, limit)
        else:
            spec = {"model": args.model, "revision": args.revision, "threads": args.threads,
                    "mha_fastpath": args.mha_fastpath == "on",
                    "seed": args.seed, "total_tokens": L, "questions": q, "options": args.options,
                    "batch_size": b, "repeats": args.repeats, "warmup": args.warmup}
            item = run_condition("experiments.timing:measure_condition", spec, time_cap=args.time_cap)
            item.setdefault("condition", {"total_tokens": L, "questions": q,
                                          "options_per_question": args.options, "batch_size": b})
        items.append(item)
        _abort_on_load_failure(item, args, "sweep.json", body())
        p50 = (item.get("timings_ms") or {}).get("p50")
        print("[%d/%d] L=%d q=%d b=%d: %s%s" % (n, len(conds), L, q, b, item["status"],
                                              " p50 %.1f ms" % p50 if p50 is not None else
                                              " (%s)" % item.get("reason", "")), flush=True)
        results.write_json(args.run_path, "sweep.json", body())
    canary.measure(None)
    results.write_json(args.run_path, "sweep.json", body())
    flagged = [i for i, it in enumerate(items) if (it.get("drift") or {}).get("flagged")]
    if flagged:
        print("drift above %.0f%% around %d condition(s): %s" % (100 * args.drift_threshold, len(flagged), flagged))
    print(args.run_path / "sweep.json")
    return EXIT_OK


def _profile_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--lengths", type=int_list, default=DEFAULT_LENGTHS)
    p.add_argument("--repeats", type=_positive_int, default=5)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--time-cap", type=float, default=DEFAULT_TIME_CAP)
    p.add_argument("--overhead-length", type=int, default=512,
                   help="length for the wrappers on/off overhead check (0 to skip)")
    _canary_args(p)


def _clean_p50(run_path) -> Dict[int, Dict[str, Any]]:
    try:
        sweep = results.read_json(run_path, "sweep.json")
    except OSError:
        return {}
    out = {}
    for i, it in enumerate(sweep.get("items", [])):
        c = it.get("condition", {})
        if it.get("kind") == "single" and it.get("status") == "measured" and it.get("timings_ms"):
            out[c["total_tokens"]] = {"p50": it["timings_ms"]["p50"], "ref": "sweep.json#%d" % i}
    return out


@command("profile", "Profiled component breakdown per length (not headline latency); writes profile.json.",
         _profile_args)
def _cmd_profile(args: argparse.Namespace) -> int:
    from . import profile as P
    from .runner import run_condition
    limit = _known_position_limit(args.run_path)
    items = []
    args._results_file = "profile.json"
    canary = Canary(args)
    session = session_info()
    for n, L in enumerate(args.lengths, 1):
        if limit is not None and L > limit:
            item = _unsupported(L, 1, 2, 1, limit)
            item["kind"] = "profile"
        else:
            _abort_on_load_failure(canary.measure(len(items)) or {}, args, "profile.json", {"items": items})
            spec = {"model": args.model, "revision": args.revision, "threads": args.threads,
                    "mha_fastpath": args.mha_fastpath == "on",
                    "seed": args.seed, "total_tokens": L, "repeats": args.repeats, "warmup": args.warmup}
            item = run_condition("experiments.profile:profile_condition", spec, time_cap=args.time_cap)
            item.setdefault("condition", {"total_tokens": L, "questions": 1,
                                          "options_per_question": 2, "batch_size": 1})
            item["profile_run"] = True
        items.append(item)
        _abort_on_load_failure(item, args, "profile.json", {"items": items})
        print("[%d/%d] profile L=%d: %s" % (n, len(args.lengths), L, item["status"]), flush=True)
    canary.measure(None)
    canary.annotate(items)
    clean = _clean_p50(args.run_path)
    for it in items:
        c = clean.get(it["condition"]["total_tokens"])
        if c and it.get("total_profiled_ms"):
            it["total_clean_ms"] = c["p50"]
            it["profiled_to_clean_ratio"] = it["total_profiled_ms"] / c["p50"]
            it["derived_from"] = [c["ref"]]
    scal = P.scaling(items)
    window = None
    try:
        audit = results.read_json(args.run_path, "audit.json")
        window = audit["encoder"].get("local_attention")
    except OSError:
        audit = None
    verdict = P.executed_work_verdict(scal, window)
    body = {"session": session, "items": items, "scaling": scal, "executed_work": verdict, "canary": canary.block()}
    if args.overhead_length:
        spec = {"model": args.model, "revision": args.revision, "threads": args.threads,
                    "mha_fastpath": args.mha_fastpath == "on",
                "seed": args.seed, "total_tokens": args.overhead_length, "repeats": 10, "warmup": 2}
        body["overhead_check"] = run_condition("experiments.timing:compare_paths", spec,
                                               time_cap=args.time_cap)
    print(results.write_json(args.run_path, "profile.json", body))
    if audit is not None:
        audit["executed_work_note"] = dict(verdict, derived_from=["profile.json#scaling"])
        print(results.write_json(args.run_path, "audit.json", audit))
    else:
        print("audit.json not found in this run; the executed-work verdict is only in profile.json")
    print("executed work: %s - %s" % (verdict["verdict"], verdict["note"]))
    return EXIT_OK


# --------------------------------------------------------------------------- kernels (T037)

def _kernels_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--lengths", type=int_list, default=DEFAULT_LENGTHS)
    p.add_argument("--impls", default="dense,dense_masked,local,gas",
                   help="comma-separated from dense, dense_masked, local, gas (default: all)")
    p.add_argument("--block-sizes", type=int_list, default=[128, 256, 512])
    p.add_argument("--selection-sizes", type=int_list, default=[512, 1024],
                   help="tokens gathered per query block by gas; must be a multiple of the block "
                        "size and at least 3 blocks (default 512,1024)")
    p.add_argument("--repeats", type=_repeats_arg, default="auto")
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--time-cap", type=float, default=300.0)


def kernel_conditions(impls, lengths, blocks, selections):
    out = []
    for L in lengths:
        for impl in impls:
            if impl == "dense":
                out.append({"impl": impl, "length": L, "block": None, "selection_blocks": None})
            elif impl in ("dense_masked", "local"):
                out.extend({"impl": impl, "length": L, "block": b, "selection_blocks": None} for b in blocks)
            else:
                for b in blocks:
                    for s in selections:
                        out.append({"impl": impl, "length": L, "block": b, "selection_tokens": s,
                                    "selection_blocks": s // b if s % b == 0 else None})
    return out


@command("kernels", "Attention microbenchmarks with correctness checks; writes kernels.json and floor.json.",
         _kernels_args)
def _cmd_kernels(args: argparse.Namespace) -> int:
    from .floor import build_floor
    from .kernels.bench import IMPLS, executed_work
    from .kernels.reference import score_matrix_bytes
    from .runner import run_condition
    impls = [s.strip() for s in args.impls.split(",") if s.strip()]
    bad = [i for i in impls if i not in IMPLS]
    if bad:
        raise ToolError("unknown implementation(s) %s; choose from %s" % (bad, sorted(IMPLS)))
    try:
        audit = results.read_json(args.run_path, "audit.json")
    except OSError:
        raise ToolError("kernels takes its shapes from audit.json; run `audit` for run %r first" % args.run_id)
    heads, head_dim = audit["encoder"]["num_heads"], audit["encoder"]["head_dim"]
    limit = audit["positional"]["max_position_embeddings"]
    items = []
    session = session_info()
    conds = kernel_conditions(impls, args.lengths, args.block_sizes, args.selection_sizes)
    for n, c in enumerate(conds, 1):
        shape = {"batch": 1, "heads": heads, "length": c["length"], "head_dim": head_dim,
                 "block": c["block"], "selection_blocks": c.get("selection_blocks"),
                 "selection_tokens": c.get("selection_tokens")}
        if c["impl"] == "gas" and (c["selection_blocks"] is None or c["selection_blocks"] < 3):
            item = {"impl": c["impl"], "shape": shape, "status": "unsupported",
                    "reason": "selection of %s tokens is not a whole number of at least 3 blocks of %d"
                              % (c.get("selection_tokens"), c["block"])}
        elif c["length"] > limit:
            item = {"impl": c["impl"], "shape": shape, "status": "unsupported",
                    "reason": "%d exceeds the model's positional capacity %d" % (c["length"], limit)}
        else:
            spec = {"impl": c["impl"], "length": c["length"], "heads": heads, "head_dim": head_dim,
                    "block": c["block"], "selection_blocks": c.get("selection_blocks"),
                    "repeats": args.repeats, "warmup": args.warmup, "threads": args.threads,
                    "seed": args.seed}
            item = run_condition("experiments.kernels.bench:bench_condition", spec, time_cap=args.time_cap)
            item.setdefault("impl", c["impl"])
            item.setdefault("shape", shape)
        item.setdefault("score_matrix_bytes_analytical", score_matrix_bytes(1, heads, c["length"]))
        items.append(item)
        p50 = (item.get("timings_ms") or {}).get("p50")
        print("[%d/%d] %s L=%d block=%s sel=%s: %s%s" % (
            n, len(conds), c["impl"], c["length"], c["block"], c.get("selection_blocks"), item["status"],
            " p50 %.1f ms" % p50 if p50 is not None else " (%s)" % item.get("reason", "")), flush=True)
        results.write_json(args.run_path, "kernels.json", {"shapes_from": "audit.json", "items": items})
    body = {"shapes_from": "audit.json", "session": session, "heads": heads, "head_dim": head_dim, "items": items,
            "executed_work": executed_work(items), "not_run": []}
    try:
        floor = build_floor(args.run_path)
        print(results.write_json(args.run_path, "floor.json", floor))
    except FileNotFoundError as exc:
        body["not_run"].append({"check": "cost floor (floor.json)", "reason": str(exc)})
        print("floor.json not written: %s" % exc)
    print(results.write_json(args.run_path, "kernels.json", body))
    for g in body["executed_work"]["groups"]:
        print("  %-12s block=%-4s sel=%-4s exponent=%s -> %s" % (
            g["impl"], g["block"], g["selection_blocks"],
            "%.2f" % g["time_exponent"] if g["time_exponent"] is not None else "n/a", g["verdict"]))
    return EXIT_OK


# --------------------------------------------------------------------------- GPU reference (R15)

def _gpu_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--lengths", type=int_list, default=[512, 2048])
    p.add_argument("--repeats", type=_positive_int, default=30)
    p.add_argument("--warmup", type=int, default=5)


@command("gpu-reference",
         "GPU-only check of the historical ~33 ms figure; writes gpu_reference.json (never a CPU result).",
         _gpu_args)
def _cmd_gpu_reference(args: argparse.Namespace) -> int:
    from .gpu import GpuUnavailable, measure_gpu
    try:
        body = measure_gpu(args.model, args.revision, args.lengths, args.repeats, args.warmup, args.seed)
    except GpuUnavailable as exc:
        raise ToolError(str(exc))
    for it in body["items"]:
        p50 = (it.get("timings_ms") or {}).get("p50")
        print("GPU L=%d: %s" % (it["condition"]["total_tokens"],
                                "p50 %.1f ms" % p50 if p50 is not None else it.get("reason")))
    print(results.write_json(args.run_path, "gpu_reference.json", body))
    return EXIT_OK


# --------------------------------------------------------------------------- report / all (T040)

@command("report", "Assemble report.json and report.md from this run's result files.")
def _cmd_report(args: argparse.Namespace) -> int:
    from .report import ReportError, write_report
    try:
        j, m = write_report(args.run_path)
    except ReportError as exc:
        raise ToolError(str(exc))
    print(j)
    print(m)
    return EXIT_OK


#: step -> options of `all` it receives (only when given)
_ALL_STEPS = [
    ("audit", ["--lengths"]),
    ("sweep", ["--lengths", "--questions", "--batch-sizes", "--repeats", "--time-cap", "--canary-length"]),
    ("profile", ["--lengths", "--repeats", "--time-cap", "--canary-length", "--overhead-length"]),
    ("kernels", ["--lengths", "--impls", "--block-sizes", "--selection-sizes", "--repeats"]),
    ("report", []),
]


def _all_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--lengths", default=None)
    p.add_argument("--questions", default=None)
    p.add_argument("--batch-sizes", default=None)
    p.add_argument("--repeats", default=None)
    p.add_argument("--time-cap", default=None)
    p.add_argument("--canary-length", default=None)
    p.add_argument("--overhead-length", default=None)
    p.add_argument("--impls", default=None)
    p.add_argument("--block-sizes", default=None)
    p.add_argument("--selection-sizes", default=None)


@command("all", "Run audit (with manifest), sweep, profile, kernels and report in order.", _all_args)
def _cmd_all(args: argparse.Namespace) -> int:
    parser = build_parser()
    common = ["--run-id", args.run_id, "--model", args.model, "--threads", str(args.threads), "--seed", str(args.seed)]
    common += ["--mha-fastpath", args.mha_fastpath]
    if args.revision:
        common += ["--revision", args.revision]
    for step, passed in _ALL_STEPS:
        extra = []
        for opt in passed:
            val = getattr(args, opt.lstrip("-").replace("-", "_"))
            if val is not None:
                extra += [opt, str(val)]
        ns = parser.parse_args([step] + common + extra)
        ns.threads_source = args.threads_source
        ns.run_path = args.run_path
        _match_run_manifest(ns)
        print("== %s ==" % step, flush=True)
        code = COMMANDS[step].run(ns)
        if code:
            return code
    return EXIT_OK


# --------------------------------------------------------------------------- instrumentation A/B (R16)

def _abtest_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--lengths", type=int_list, default=[2048, 8192])
    p.add_argument("--repeats", type=_positive_int, default=4)
    p.add_argument("--modes", default="clean,wrappers,labels,profiler,all")
    p.add_argument("--time-cap", type=float, default=2400.0)


@command("abtest", "Same documents in one process with and without each kind of instrumentation; "
                   "writes abtest.json.", _abtest_args)
def _cmd_abtest(args: argparse.Namespace) -> int:
    from .abtest import MODES
    from .runner import run_condition
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    bad = [m for m in modes if m not in MODES]
    if bad or "clean" not in modes:
        raise ToolError("modes must include clean and come from %s" % (MODES,))
    session = session_info()
    items = []
    for L in args.lengths:
        spec = {"model": args.model, "revision": args.revision, "threads": args.threads,
                    "mha_fastpath": args.mha_fastpath == "on", "seed": args.seed,
                "total_tokens": L, "repeats": args.repeats, "modes": modes}
        item = run_condition("experiments.abtest:abtest_condition", spec, time_cap=args.time_cap)
        item.setdefault("condition", {"total_tokens": L, "questions": 1, "options_per_question": 2, "batch_size": 1})
        items.append(item)
        _abort_on_load_failure(item, args, "abtest.json", {"session": session, "items": items})
        ratios = item.get("ratio_to_clean") or {}
        print("L=%d %s: %s" % (L, item["status"], ", ".join("%s x%.3f" % (m, r) for m, r in ratios.items() if r)
                               or item.get("reason", "")), flush=True)
        results.write_json(args.run_path, "abtest.json", {"session": session, "items": items})
    print(args.run_path / "abtest.json")
    return EXIT_OK
