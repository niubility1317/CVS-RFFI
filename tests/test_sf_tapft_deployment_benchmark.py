from __future__ import annotations

from pathlib import Path

import pytest

import cvsrffi.sf_tapft_deployment_benchmark as benchmark
from cvsrffi.sf_tapft_deployment_benchmark import benchmark_deployment_runs


def test_linux_current_rss_reads_vmrss_instead_of_lifetime_peak(
    monkeypatch, tmp_path: Path
) -> None:
    status = tmp_path / "status"
    status.write_text("Name:\tpython\nVmRSS:\t1234 kB\n", encoding="ascii")
    monkeypatch.setattr(benchmark.sys, "platform", "linux")
    monkeypatch.setattr(benchmark, "_LINUX_PROC_STATUS", status)

    assert benchmark._current_rss_bytes() == 1234 * 1024


def test_lifetime_maxrss_is_reported_separately(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(benchmark, "_current_rss_bytes", lambda: 1000)
    monkeypatch.setattr(benchmark, "_process_lifetime_maxrss_bytes", lambda: 5000)

    def run_once(_kind: str, _index: int, destination: Path) -> dict[str, object]:
        destination.mkdir()
        return {"delta_bundle_bytes": 1, "resource_audit": {}}

    result = benchmark_deployment_runs(
        run_once,
        tmp_path / "separate-rss",
        warmup_runs=0,
        measured_runs=1,
        clock_values_ms=[1.0],
        rss_samples_bytes=[1100],
        cuda_allocated_samples_bytes=[0],
        cuda_reserved_samples_bytes=[0],
    )

    assert result["process_lifetime_maxrss_bytes"] == {
        "median": 5000.0,
        "p90": 5000.0,
        "max": 5000.0,
    }
    assert result["samples"][0]["cpu_rss_current_start_bytes"] == 1000


def test_benchmark_excludes_warmups_and_summarizes_measured_runs(tmp_path: Path) -> None:
    calls: list[tuple[str, int]] = []

    def run_once(kind: str, index: int, destination: Path) -> dict[str, object]:
        calls.append((kind, index))
        destination.mkdir()
        return {
            "delta_bundle_bytes": 4500 + index,
            "resource_audit": {"prefix_cache_tensor_bytes": 2048},
        }

    result = benchmark_deployment_runs(
        run_once,
        tmp_path / "benchmark",
        warmup_runs=3,
        measured_runs=10,
        clock_values_ms=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        rss_samples_bytes=[1000] * 10,
        cuda_allocated_samples_bytes=[2000] * 10,
        cuda_reserved_samples_bytes=[3000] * 10,
        cuda_free_start_samples_bytes=[9000] * 10,
        cuda_free_min_samples_bytes=[7000] * 10,
        cuda_free_end_samples_bytes=[8000] * 10,
        execution_mode="resident_process",
    )

    assert calls[:3] == [("warmup", 0), ("warmup", 1), ("warmup", 2)]
    assert calls[3:] == [("measure", index) for index in range(10)]
    assert len(result["samples"]) == 10
    assert result["wall_clock_ms"] == {"median": 55.0, "p90": 90.0, "max": 100.0}
    assert result["cpu_rss_peak_bytes"]["max"] == 1000.0
    assert result["cuda_allocated_peak_bytes"]["max"] == 2000.0
    assert result["cuda_reserved_peak_bytes"]["max"] == 3000.0
    assert result["execution_mode"] == "resident_process"
    assert result["cuda_free_min_bytes"]["max"] == 7000.0
    assert result["cuda_free_consumed_peak_bytes"]["max"] == 2000.0
    assert result["cache_tensor_bytes"] == 2048
    assert result["delta_bundle_bytes"]["max"] == 4509.0


def test_benchmark_does_not_mislabel_same_process_as_cold_start(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="fresh subprocess"):
        benchmark_deployment_runs(
            lambda *_args: {},
            tmp_path / "cold-start",
            warmup_runs=0,
            measured_runs=1,
            execution_mode="cold_start",
        )
