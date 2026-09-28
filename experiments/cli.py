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
from typing import Callable, Dict, List, Optional, Sequence

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
    return args


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
    """Import the modules that register commands. Later tasks add their modules here."""
    # e.g. from . import manifest, audit  (T019)
    return None
