from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.audit_d92_e0d_hard12v2_selection import (
    D92E0DSelectionAuditError,
    audit_selection,
)


OUTPUT = Path(
    r"E:\type10-7\automation_reports\CV-SincNet\d92_e0d_5arm_hard12v2_20260811_v1\selection_audit.json"
)


def test_selection_audit_recomputes_all_rows_and_solver_closure(tmp_path: Path) -> None:
    output = tmp_path / "selection_audit.json"
    receipt = audit_selection(output)
    assert receipt["status"] == "SELECTION_AUDIT_PASS"
    assert receipt["selection_sha256"] == (
        "2e3b3333a4a325bd0443a31065d3340d6a650a3e89620951a786637e6bce8d3a"
    )
    assert receipt["input_sha256"] == {
        "d92_retry2_row_metrics": "bc8070cd9235ab41eda5bafd2ec66e9afad48b6466d2066508d0bab46980fa62",
        "next_r5_r11_score": "fa2344ae037e4ab5dfec6fea9bb0f534c7d5c9cdeb3596797bdc403b3c9fcc23",
    }
    assert receipt["d92_outer_count"] == 125
    assert receipt["r5_outer_count"] == 125
    assert receipt["hard_score_max_abs_error"] < 1e-12
    assert receipt["historical_hard_sum"] == "7.076008064516129"
    assert receipt["selected_outer_count"] == 12
    assert receipt["v1_intersection_count"] == 0
    assert receipt["selected_outer_keys"] == [
        row["outer_key"] for row in receipt["selected_rows"]
    ]
    assert output.is_file()
    assert json.loads(output.read_text(encoding="utf-8")) == receipt


def test_selection_audit_receipt_is_non_overwriting(tmp_path: Path) -> None:
    output = tmp_path / "selection_audit.json"
    audit_selection(output)
    with pytest.raises(D92E0DSelectionAuditError, match="must not overwrite"):
        audit_selection(output)


def test_selection_audit_cli_accepts_only_output_path() -> None:
    from scripts.audit_d92_e0d_hard12v2_selection import parser

    args = parser().parse_args(["--output-path", str(OUTPUT)])
    assert args.output_path == str(OUTPUT)
    with pytest.raises(SystemExit):
        parser().parse_args(["--output-path", str(OUTPUT), "--truth", "x"])
