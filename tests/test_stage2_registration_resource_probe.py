from __future__ import annotations

import threading
import time

import pytest

from cvsrffi.stage2_registration_resource_probe import (
    current_rss_bytes,
    measure_registration_call,
)


def test_measure_registration_call_subtracts_baseline_from_sampled_peak():
    rss_values = iter((100, 140, 180, 160))
    wall_clock = iter((1_000, 6_000))
    cpu_clock = iter((2_000, 4_000))

    def operation() -> str:
        time.sleep(0.01)
        return "ok"

    result, receipt = measure_registration_call(
        operation,
        rss_reader=lambda: next(rss_values, 160),
        perf_counter_ns=wall_clock.__next__,
        process_time_ns=cpu_clock.__next__,
        sample_interval_seconds=0.001,
    )

    assert result == "ok"
    assert receipt == {
        "schema": "cvs.phase2.registration_resource_receipt.v1",
        "registration_wall_time_ns": 5_000,
        "registration_process_cpu_time_ns": 2_000,
        "registration_baseline_rss_bytes": 100,
        "registration_peak_rss_bytes": 180,
        "registration_incremental_peak_working_set_bytes": 80,
        "rss_sampler": "current_process_working_set_1ms",
    }


def test_measure_registration_call_stops_sampler_when_operation_raises():
    def fail() -> None:
        raise RuntimeError("synthetic fit failure")

    with pytest.raises(RuntimeError, match="synthetic fit failure"):
        measure_registration_call(
            fail,
            rss_reader=lambda: 123,
            sample_interval_seconds=0.001,
        )

    assert all(
        thread.name != "d92-registration-rss-sampler"
        for thread in threading.enumerate()
    )


def test_current_rss_bytes_reports_the_live_process_working_set():
    assert current_rss_bytes() > 0


def test_measure_registration_call_rejects_nonpositive_sampling_interval():
    with pytest.raises(ValueError, match="sample interval must be positive"):
        measure_registration_call(
            lambda: None,
            rss_reader=lambda: 1,
            sample_interval_seconds=0.0,
        )
