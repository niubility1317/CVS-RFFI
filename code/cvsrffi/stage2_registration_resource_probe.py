"""Registration-only wall, CPU, and incremental working-set measurement."""

from __future__ import annotations

import ctypes
import os
import sys
import threading
import time
from pathlib import Path
from typing import Callable, TypeVar


T = TypeVar("T")
RESOURCE_SCHEMA = "cvs.phase2.registration_resource_receipt.v1"
SAMPLER_THREAD_NAME = "d92-registration-rss-sampler"


class RegistrationResourceProbeError(RuntimeError):
    """Raised when the current process working set cannot be sampled."""


def _windows_rss_bytes() -> int:
    from ctypes import wintypes

    class ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    counters = ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel32.GetCurrentProcess.argtypes = []
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(ProcessMemoryCounters),
        wintypes.DWORD,
    ]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    process = kernel32.GetCurrentProcess()
    succeeded = psapi.GetProcessMemoryInfo(
        process, ctypes.byref(counters), counters.cb
    )
    if not succeeded:
        raise OSError(ctypes.get_last_error(), "GetProcessMemoryInfo failed")
    return int(counters.WorkingSetSize)


def _linux_rss_bytes() -> int:
    fields = Path("/proc/self/statm").read_text(encoding="ascii").split()
    if len(fields) < 2:
        raise RegistrationResourceProbeError("/proc/self/statm schema drift")
    return int(fields[1]) * int(os.sysconf("SC_PAGE_SIZE"))


def current_rss_bytes() -> int:
    """Return the current process resident working set in bytes."""

    try:
        if os.name == "nt":
            value = _windows_rss_bytes()
        elif sys.platform.startswith("linux"):
            value = _linux_rss_bytes()
        else:
            raise RegistrationResourceProbeError(
                f"unsupported RSS platform: {sys.platform}"
            )
    except (OSError, ValueError) as error:
        raise RegistrationResourceProbeError(
            "current process RSS sampling failed"
        ) from error
    if value <= 0:
        raise RegistrationResourceProbeError("current process RSS is not positive")
    return value


def measure_registration_call(
    call: Callable[[], T],
    *,
    rss_reader: Callable[[], int] = current_rss_bytes,
    perf_counter_ns: Callable[[], int] = time.perf_counter_ns,
    process_time_ns: Callable[[], int] = time.process_time_ns,
    sample_interval_seconds: float = 0.001,
) -> tuple[T, dict[str, int | str]]:
    """Measure one fit call without including encoder or query execution."""

    interval = float(sample_interval_seconds)
    if interval <= 0.0:
        raise ValueError("sample interval must be positive")
    baseline = int(rss_reader())
    peak = baseline
    peak_lock = threading.Lock()
    stop = threading.Event()
    sampler_errors: list[BaseException] = []

    def sample() -> None:
        nonlocal peak
        while not stop.wait(interval):
            try:
                current = int(rss_reader())
            except BaseException as error:  # preserve failure for main thread
                sampler_errors.append(error)
                stop.set()
                return
            with peak_lock:
                peak = max(peak, current)

    sampler = threading.Thread(
        target=sample,
        name=SAMPLER_THREAD_NAME,
        daemon=True,
    )
    wall_start = int(perf_counter_ns())
    cpu_start = int(process_time_ns())
    sampler.start()
    try:
        result = call()
    finally:
        cpu_end = int(process_time_ns())
        wall_end = int(perf_counter_ns())
        stop.set()
        sampler.join()
        try:
            final_rss = int(rss_reader())
        except BaseException as error:
            sampler_errors.append(error)
        else:
            with peak_lock:
                peak = max(peak, final_rss)
    if sampler_errors:
        raise RegistrationResourceProbeError("RSS sampler failed") from sampler_errors[0]
    receipt: dict[str, int | str] = {
        "schema": RESOURCE_SCHEMA,
        "registration_wall_time_ns": wall_end - wall_start,
        "registration_process_cpu_time_ns": cpu_end - cpu_start,
        "registration_baseline_rss_bytes": baseline,
        "registration_peak_rss_bytes": peak,
        "registration_incremental_peak_working_set_bytes": max(0, peak - baseline),
        "rss_sampler": "current_process_working_set_1ms",
    }
    return result, receipt


__all__ = [
    "RegistrationResourceProbeError",
    "current_rss_bytes",
    "measure_registration_call",
]
