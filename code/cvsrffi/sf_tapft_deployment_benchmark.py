"""Isolated repeat benchmark for SF-TAPFT deployment adaptation."""

from __future__ import annotations

import math
import statistics
import threading
import time
import ctypes
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch


BenchmarkRun = Callable[[str, int, Path], Mapping[str, Any]]
_LINUX_PROC_STATUS = Path("/proc/self/status")


def _summary(values: Sequence[float | int]) -> dict[str, float]:
    if not values:
        raise ValueError("benchmark summary requires at least one value")
    ordered = sorted(float(value) for value in values)
    p90_index = max(0, math.ceil(0.9 * len(ordered)) - 1)
    return {
        "median": float(statistics.median(ordered)),
        "p90": ordered[p90_index],
        "max": ordered[-1],
    }


def _current_rss_bytes() -> int:
    if sys.platform == "win32":
        class _ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = _ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        psapi.GetProcessMemoryInfo.argtypes = (
            ctypes.c_void_p,
            ctypes.POINTER(_ProcessMemoryCounters),
            ctypes.c_ulong,
        )
        psapi.GetProcessMemoryInfo.restype = ctypes.c_int
        process = kernel32.GetCurrentProcess()
        succeeded = psapi.GetProcessMemoryInfo(
            process, ctypes.byref(counters), counters.cb
        )
        if not succeeded:
            raise OSError("GetProcessMemoryInfo failed")
        return int(counters.WorkingSetSize)
    if sys.platform.startswith("linux"):
        for line in _LINUX_PROC_STATUS.read_text(encoding="ascii").splitlines():
            if line.startswith("VmRSS:"):
                fields = line.split()
                if len(fields) != 3 or fields[2] != "kB":
                    raise RuntimeError("unexpected VmRSS format in /proc/self/status")
                return int(fields[1]) * 1024
        raise RuntimeError("VmRSS is missing from /proc/self/status")
    import resource

    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def _process_lifetime_maxrss_bytes() -> int:
    if sys.platform == "win32":
        class _ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = _ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        psapi.GetProcessMemoryInfo.argtypes = (
            ctypes.c_void_p,
            ctypes.POINTER(_ProcessMemoryCounters),
            ctypes.c_ulong,
        )
        psapi.GetProcessMemoryInfo.restype = ctypes.c_int
        if not psapi.GetProcessMemoryInfo(
            kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb
        ):
            raise OSError("GetProcessMemoryInfo failed")
        return int(counters.PeakWorkingSetSize)
    import resource

    usage = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return usage if sys.platform == "darwin" else usage * 1024


class _RSSPeakSampler:
    def __init__(self, interval_seconds: float = 0.01) -> None:
        self._interval_seconds = float(interval_seconds)
        self._stop = threading.Event()
        self._peak = _current_rss_bytes()
        self._thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self) -> None:
        while not self._stop.wait(self._interval_seconds):
            self._peak = max(self._peak, _current_rss_bytes())

    def __enter__(self) -> "_RSSPeakSampler":
        self._thread.start()
        return self

    def __exit__(self, *_args: object) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)
        self._peak = max(self._peak, _current_rss_bytes())

    @property
    def peak_bytes(self) -> int:
        return self._peak


class _CudaFreeSampler:
    def __init__(self, interval_seconds: float = 0.01) -> None:
        self._interval_seconds = float(interval_seconds)
        self._stop = threading.Event()
        self._minimum_free = self._read_free()
        self._thread = threading.Thread(target=self._sample, daemon=True)

    @staticmethod
    def _read_free() -> int:
        return int(torch.cuda.mem_get_info()[0]) if torch.cuda.is_available() else 0

    def _sample(self) -> None:
        while not self._stop.wait(self._interval_seconds):
            self._minimum_free = min(self._minimum_free, self._read_free())

    def __enter__(self) -> "_CudaFreeSampler":
        self._thread.start()
        return self

    def __exit__(self, *_args: object) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)
        self._minimum_free = min(self._minimum_free, self._read_free())

    @property
    def minimum_free_bytes(self) -> int:
        return self._minimum_free


def _cuda_synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def benchmark_deployment_runs(
    run_once: BenchmarkRun,
    output_root: str | Path,
    *,
    warmup_runs: int = 3,
    measured_runs: int = 10,
    clock_values_ms: Sequence[float] | None = None,
    rss_samples_bytes: Sequence[int] | None = None,
    cuda_allocated_samples_bytes: Sequence[int] | None = None,
    cuda_reserved_samples_bytes: Sequence[int] | None = None,
    cuda_free_start_samples_bytes: Sequence[int] | None = None,
    cuda_free_min_samples_bytes: Sequence[int] | None = None,
    cuda_free_end_samples_bytes: Sequence[int] | None = None,
    execution_mode: str = "resident_process",
) -> dict[str, Any]:
    """Run immutable warmups and measurements, returning resource summaries."""

    if warmup_runs < 0 or measured_runs <= 0:
        raise ValueError("benchmark requires non-negative warmups and positive measurements")
    if execution_mode != "resident_process":
        raise ValueError(
            "only resident_process is implemented; cold_start requires a fresh subprocess per sample"
        )
    injected = (
        clock_values_ms,
        rss_samples_bytes,
        cuda_allocated_samples_bytes,
        cuda_reserved_samples_bytes,
        cuda_free_start_samples_bytes,
        cuda_free_min_samples_bytes,
        cuda_free_end_samples_bytes,
    )
    for values in injected:
        if values is not None and len(values) != measured_runs:
            raise ValueError("injected benchmark samples must match measured_runs")
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=False)
    for index in range(warmup_runs):
        _cuda_synchronize()
        run_once("warmup", index, root / f"warmup_{index:02d}")
        _cuda_synchronize()

    samples: list[dict[str, Any]] = []
    for index in range(measured_runs):
        rss_start = 0 if rss_samples_bytes is not None else _current_rss_bytes()
        if rss_samples_bytes is not None:
            rss_start = _current_rss_bytes()
        cuda_allocated_start = (
            int(torch.cuda.memory_allocated()) if torch.cuda.is_available() else 0
        )
        cuda_reserved_start = (
            int(torch.cuda.memory_reserved()) if torch.cuda.is_available() else 0
        )
        cuda_free_start = (
            int(cuda_free_start_samples_bytes[index])
            if cuda_free_start_samples_bytes is not None
            else (int(torch.cuda.mem_get_info()[0]) if torch.cuda.is_available() else 0)
        )
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        _cuda_synchronize()
        started = time.perf_counter()
        with _RSSPeakSampler() as rss_sampler, _CudaFreeSampler() as cuda_free_sampler:
            receipt = dict(run_once("measure", index, root / f"measure_{index:02d}"))
            _cuda_synchronize()
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        wall_ms = float(clock_values_ms[index]) if clock_values_ms is not None else elapsed_ms
        rss_peak = (
            int(rss_samples_bytes[index])
            if rss_samples_bytes is not None
            else rss_sampler.peak_bytes
        )
        allocated = (
            int(cuda_allocated_samples_bytes[index])
            if cuda_allocated_samples_bytes is not None
            else (int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0)
        )
        reserved = (
            int(cuda_reserved_samples_bytes[index])
            if cuda_reserved_samples_bytes is not None
            else (int(torch.cuda.max_memory_reserved()) if torch.cuda.is_available() else 0)
        )
        cuda_free_min = (
            int(cuda_free_min_samples_bytes[index])
            if cuda_free_min_samples_bytes is not None
            else cuda_free_sampler.minimum_free_bytes
        )
        cuda_free_end = (
            int(cuda_free_end_samples_bytes[index])
            if cuda_free_end_samples_bytes is not None
            else (int(torch.cuda.mem_get_info()[0]) if torch.cuda.is_available() else 0)
        )
        resource_audit = receipt.get("resource_audit", {})
        samples.append(
            {
                "index": index,
                "wall_clock_ms": wall_ms,
                "cpu_rss_peak_bytes": rss_peak,
                "cpu_rss_current_start_bytes": rss_start,
                "cpu_rss_adaptation_extra_peak_bytes": max(0, rss_peak - rss_start),
                "process_lifetime_maxrss_bytes": _process_lifetime_maxrss_bytes(),
                "cuda_allocated_peak_bytes": allocated,
                "cuda_allocated_adaptation_extra_peak_bytes": max(
                    0, allocated - cuda_allocated_start
                ),
                "cuda_reserved_peak_bytes": reserved,
                "cuda_reserved_adaptation_extra_peak_bytes": max(
                    0, reserved - cuda_reserved_start
                ),
                "cuda_free_start_bytes": cuda_free_start,
                "cuda_free_min_bytes": cuda_free_min,
                "cuda_free_end_bytes": cuda_free_end,
                "cuda_free_consumed_peak_bytes": max(0, cuda_free_start - cuda_free_min),
                "resident_model_tensor_bytes": int(
                    resource_audit.get("resident_model_tensor_bytes", 0)
                ),
                "cache_tensor_bytes": int(resource_audit.get("prefix_cache_tensor_bytes", 0)),
                "delta_bundle_bytes": int(receipt["delta_bundle_bytes"]),
            }
        )

    cache_sizes = {sample["cache_tensor_bytes"] for sample in samples}
    if len(cache_sizes) != 1:
        raise RuntimeError("cache tensor bytes drifted across measured runs")
    return {
        "execution_mode": execution_mode,
        "warmup_runs": warmup_runs,
        "measured_runs": measured_runs,
        "samples": samples,
        "wall_clock_ms": _summary([row["wall_clock_ms"] for row in samples]),
        "cpu_rss_peak_bytes": _summary([row["cpu_rss_peak_bytes"] for row in samples]),
        "process_lifetime_maxrss_bytes": _summary(
            [row["process_lifetime_maxrss_bytes"] for row in samples]
        ),
        "cpu_rss_adaptation_extra_peak_bytes": _summary(
            [row["cpu_rss_adaptation_extra_peak_bytes"] for row in samples]
        ),
        "cuda_allocated_peak_bytes": _summary(
            [row["cuda_allocated_peak_bytes"] for row in samples]
        ),
        "cuda_reserved_peak_bytes": _summary(
            [row["cuda_reserved_peak_bytes"] for row in samples]
        ),
        "cuda_allocated_adaptation_extra_peak_bytes": _summary(
            [row["cuda_allocated_adaptation_extra_peak_bytes"] for row in samples]
        ),
        "cuda_reserved_adaptation_extra_peak_bytes": _summary(
            [row["cuda_reserved_adaptation_extra_peak_bytes"] for row in samples]
        ),
        "cuda_free_min_bytes": _summary([row["cuda_free_min_bytes"] for row in samples]),
        "cuda_free_consumed_peak_bytes": _summary(
            [row["cuda_free_consumed_peak_bytes"] for row in samples]
        ),
        "resident_model_tensor_bytes": _summary(
            [row["resident_model_tensor_bytes"] for row in samples]
        ),
        "cache_tensor_bytes": cache_sizes.pop(),
        "delta_bundle_bytes": _summary([row["delta_bundle_bytes"] for row in samples]),
    }
