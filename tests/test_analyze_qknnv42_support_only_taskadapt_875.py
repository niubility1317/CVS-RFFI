from __future__ import annotations

import pytest

from paper_reproduction.scripts.analyze_qknnv42_support_only_taskadapt_875 import (
    build_markdown,
    paired_deltas,
    summarize,
)


def _row(arm: str, old: float, new: float, h: float) -> dict:
    return {
        "arm": arm,
        "epochs": 0 if arm == "singlehead_fft96" else int(arm[1:]),
        "receiver": "20-1",
        "seed": 713101,
        "k_shot": 1,
        "old_before": old + 0.01,
        "old_acc": old,
        "seen_new_acc": new,
        "H_old_new": h,
        "average_forgetting": 0.01,
        "old_to_seen_new_rate": 0.02,
        "seen_new_to_old_rate": 0.03,
        "min_old_class_acc": old - 0.1,
        "min_seen_new_class_acc": new - 0.1,
        "adapter_parameters": 0 if arm == "singlehead_fft96" else 154,
        "adapter_state_bytes_fp16": 0 if arm == "singlehead_fft96" else 308,
        "adapter_macs_per_query": 0 if arm == "singlehead_fft96" else 34816,
        "adaptation_wall_seconds": 0.0 if arm == "singlehead_fft96" else 1.0,
        "peak_cuda_memory_bytes": 0 if arm == "singlehead_fft96" else 1024,
    }


def test_paired_delta_and_markdown_use_same_task_baseline() -> None:
    rows = [
        _row("singlehead_fft96", 0.70, 0.50, 0.58),
        _row("E2", 0.75, 0.60, 0.67),
    ]
    deltas = paired_deltas(rows)
    assert len(deltas) == 1
    assert deltas[0]["delta_old_acc_pp"] == pytest.approx(5.0)
    assert deltas[0]["delta_seen_new_acc_pp"] == pytest.approx(10.0)
    by_arm_k = summarize(rows, ("arm", "k_shot"))
    markdown = build_markdown(by_arm_k, deltas)
    assert "E2" in markdown
    assert "+5.00pp" in markdown
    assert "+10.00pp" in markdown
