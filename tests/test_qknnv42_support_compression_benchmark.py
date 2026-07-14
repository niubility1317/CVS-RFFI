from __future__ import annotations

from copy import deepcopy

from paper_reproduction.scripts.benchmark_qknnv42_support_compression import (
    METRICS,
    RESOURCE_KEYS,
    _aggregate,
)


def _row(value: float) -> dict[str, float | str]:
    row: dict[str, float | str] = {"run_key": "rx/seed/k"}
    row.update({metric: value for metric in METRICS})
    row.update({key: 1.0 for key in RESOURCE_KEYS})
    return row


def test_matrix_mean_gate_includes_exact_three_pp_boundary() -> None:
    dense = _row(0.50)
    baseline = {str(dense["run_key"]): dense}
    boundary = _aggregate([_row(0.47)], baseline=baseline)
    assert boundary["performance_gate_pass"] is True

    below_boundary = deepcopy(_row(0.47))
    below_boundary[METRICS[0]] = 0.469999
    failed = _aggregate([below_boundary], baseline=baseline)
    assert failed["performance_gate_pass"] is False
