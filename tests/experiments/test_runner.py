"""T007: the subprocess executor (experiments/runner.py).

The Windows cases only run on Windows and must pass on the measuring machine; the POSIX signal
cases only run on Linux/macOS.
"""
import os
import sys
import time

import pytest

from experiments import runner
from experiments.results import validate_item

CHILD = "tests.experiments.runner_children:"
WINDOWS = sys.platform == "win32"


def test_success_returns_payload_and_peak_memory():
    item = runner.run_condition(CHILD + "ok", {"x": 21}, time_cap=60)
    validate_item(item)
    assert item["status"] == "measured"
    assert item["answer"] == 42
    assert item["got_time_cap"] == 60.0  # the child sees the cap
    assert item["peak_rss_bytes"] > 8 * 1024 * 1024
    assert item["exit_code"] == 0
    if WINDOWS:
        assert item["peak_commit_bytes"] > 0
    else:
        assert item["peak_commit_bytes"] is None
    assert item["paging_suspected"] is False


def test_exception_is_recorded_as_failed():
    item = runner.run_condition(CHILD + "raises", {}, time_cap=60)
    validate_item(item)
    assert item["status"] == "failed"
    assert item["cause"] == "exception"
    assert "boom from child" in item["reason"]
    assert item["signal"] is None
    assert item["exit_code"] != 0
    assert item["peak_rss_bytes"] > 0


@pytest.mark.parametrize("func", ["allocator_oom", "memory_error"])
def test_allocator_failure_is_oom(func):
    item = runner.run_condition(CHILD + func, {}, time_cap=60)
    validate_item(item)
    assert item["status"] == "failed"
    assert item["cause"] == "oom"


def test_non_dict_payload_is_a_failure():
    item = runner.run_condition(CHILD + "not_a_dict", {}, time_cap=60)
    assert item["status"] == "failed" and item["cause"] == "exception"


@pytest.mark.skipif(WINDOWS, reason="POSIX signals")
def test_sigkill_is_recorded_as_oom():
    item = runner.run_condition(CHILD + "kill_self", {"signal": "SIGKILL"}, time_cap=60)
    validate_item(item)
    assert item["status"] == "failed"
    assert item["cause"] == "oom"
    assert item["signal"] == 9
    assert item["exit_code"] == -9


@pytest.mark.skipif(WINDOWS, reason="POSIX signals")
def test_other_signal_is_recorded_as_signal():
    item = runner.run_condition(CHILD + "kill_self", {"signal": "SIGTERM"}, time_cap=60)
    validate_item(item)
    assert item["cause"] == "signal"
    assert item["signal"] == 15
    assert item["exit_code"] == -15


@pytest.mark.skipif(not WINDOWS, reason="Windows exit codes; must pass on the measuring machine")
@pytest.mark.parametrize("code", [0xC0000017, 0xC000012D])
def test_windows_no_memory_exit_is_oom(code):
    item = runner.run_condition(CHILD + "windows_exit", {"code": code}, time_cap=60)
    validate_item(item)
    assert item["cause"] == "oom"
    assert item["signal"] is None
    assert item["exit_code"] == code


@pytest.mark.skipif(not WINDOWS, reason="Windows exit codes; must pass on the measuring machine")
def test_windows_access_violation_is_crash():
    item = runner.run_condition(CHILD + "windows_exit", {"code": 0xC0000005}, time_cap=60)
    assert item["cause"] == "crash"
    assert item["signal"] is None


def test_exit_without_result_is_crash():
    item = runner.run_condition(CHILD + "plain_exit", {"code": 3}, time_cap=60)
    validate_item(item)
    assert item["status"] == "failed"
    assert item["cause"] == "crash"
    assert item["exit_code"] == 3


def test_timeout_terminates_the_child():
    started = time.perf_counter()
    item = runner.run_condition(CHILD + "sleeps", {"seconds": 60}, time_cap=1, grace=1)
    elapsed = time.perf_counter() - started
    validate_item(item)
    assert item["status"] == "failed"
    assert item["cause"] == "timeout"
    assert elapsed < 30


def test_parent_is_unaffected():
    pid = os.getpid()
    for func, spec in [("raises", {}), ("plain_exit", {"code": 7}), ("allocator_oom", {})]:
        runner.run_condition(CHILD + func, spec, time_cap=60)
    assert os.getpid() == pid
    assert runner.run_condition(CHILD + "ok", {"x": 1}, time_cap=60)["answer"] == 2


def test_ru_maxrss_units():
    assert runner._ru_maxrss_to_bytes(100, "darwin") == 100
    assert runner._ru_maxrss_to_bytes(100, "linux") == 102400


def test_classify_exit_table():
    assert runner.classify_exit(0xC0000017, "win32")["cause"] == "oom"
    assert runner.classify_exit(-0x3FFFFFE9, "win32")["cause"] == "oom"  # same code, signed
    assert runner.classify_exit(1, "win32")["cause"] == "crash"
    assert runner.classify_exit(-9, "linux")["cause"] == "oom"
    assert runner.classify_exit(-11, "linux")["cause"] == "signal"
    assert runner.classify_exit(2, "linux")["cause"] == "crash"


def test_peak_memory_in_this_process():
    mem = runner.peak_memory()
    assert mem["peak_rss_bytes"] > 0
    assert (mem["peak_commit_bytes"] is not None) == WINDOWS
