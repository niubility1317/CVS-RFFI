from __future__ import annotations

from pathlib import Path

from cvsrffi.sf_tapft_deployment_benchmark import benchmark_deployment_runs


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
    )

    assert calls[:3] == [("warmup", 0), ("warmup", 1), ("warmup", 2)]
    assert calls[3:] == [("measure", index) for index in range(10)]
    assert len(result["samples"]) == 10
    assert result["wall_clock_ms"] == {"median": 55.0, "p90": 90.0, "max": 100.0}
    assert result["cpu_rss_peak_bytes"]["max"] == 1000.0
    assert result["cuda_allocated_peak_bytes"]["max"] == 2000.0
    assert result["cuda_reserved_peak_bytes"]["max"] == 3000.0
    assert result["cache_tensor_bytes"] == 2048
    assert result["delta_bundle_bytes"]["max"] == 4509.0
