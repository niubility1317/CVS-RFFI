from __future__ import annotations

import json
from pathlib import Path

import pytest

from paper_reproduction.scripts.summarize_adv3b02_ci_strict_matrix import (
    METRICS,
    _group_metrics,
    _group_resources,
    summarize,
)


def _metric_row(value: float, *, cell_id: str) -> dict[str, object]:
    row: dict[str, object] = {
        "method": "csil",
        "k_shot": 1,
        "new_class_count": 5,
        "cell_id": cell_id,
    }
    row.update({metric: value for metric in METRICS})
    return row


def test_metric_group_keeps_joint_rows_and_reports_mean_std() -> None:
    result = _group_metrics(
        [_metric_row(0.2, cell_id="a"), _metric_row(0.6, cell_id="b")],
        ("method", "k_shot", "new_class_count"),
    )
    assert len(result) == 1
    assert result[0]["cells"] == 2
    assert result[0]["H_old_new_mean"] == pytest.approx(0.4)
    assert result[0]["H_old_new_std"] == pytest.approx(0.2)


def test_resource_group_reports_mean_and_max() -> None:
    rows = []
    for value in (1.0, 3.0):
        rows.append(
            {
                "method": "csil",
                "k_shot": 1,
                "new_class_count": 5,
                "trainable_parameters": value,
                "persistent_state_bytes": value,
                "optimizer_steps_total": value,
                "adaptation_wall_seconds_total": value,
                "peak_cuda_memory_bytes": value,
                "support_backbone_forward_samples": value,
                "query_backbone_forward_samples": value,
                "query_view_count": value,
            }
        )
    result = _group_resources(rows, ("method", "k_shot", "new_class_count"))
    assert result[0]["trainable_parameters_mean"] == pytest.approx(2.0)
    assert result[0]["peak_cuda_memory_bytes_max"] == pytest.approx(3.0)


def test_summary_refuses_incomplete_or_unauthorized_plan(tmp_path: Path) -> None:
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "schema": "cvs.phase2.adv3b02_ci_strict_plan.v1",
                "launch_authority": False,
                "cells": [],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="smoke-authorized"):
        summarize(plan, tmp_path / "summary")
