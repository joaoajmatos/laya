"""Child functions for test_runner.py. Run inside `python -m experiments.runner --child`."""
import os
import signal
import sys
import time


def ok(spec):
    buf = bytearray(8 * 1024 * 1024)  # touch some memory so the peak is clearly above zero
    buf[::4096] = b"x" * len(buf[::4096])
    return {"answer": spec.get("x", 0) * 2, "got_time_cap": spec.get("time_cap")}


def raises(spec):
    raise ValueError("boom from child")


def allocator_oom(spec):
    raise RuntimeError("[enforce fail at alloc_cpu.cpp:117] . DefaultCPUAllocator: "
                       "not enough memory: you tried to allocate 3221225472 bytes.")


def memory_error(spec):
    raise MemoryError()


def kill_self(spec):
    sys.stdout.flush()
    os.kill(os.getpid(), getattr(signal, spec["signal"]))
    time.sleep(10)


def windows_exit(spec):
    import ctypes
    ctypes.windll.kernel32.ExitProcess(int(spec["code"]))


def plain_exit(spec):
    os._exit(int(spec["code"]))


def sleeps(spec):
    time.sleep(float(spec["seconds"]))
    return {"slept": True}


def not_a_dict(spec):
    return [1, 2, 3]
