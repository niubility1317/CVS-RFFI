from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest


SCRIPT_ROOT = Path(__file__).resolve().parents[1] / "code" / "scripts"
sys.path.insert(0, str(SCRIPT_ROOT))
try:
    import score_d104_r1_held_predictions as scorer
finally:
    sys.path.remove(str(SCRIPT_ROOT))


def test_d104_scorer_fails_on_internal_seal_before_truth_read_or_event(
    monkeypatch,
    tmp_path,
) -> None:
    prediction_root = tmp_path / "predictions"
    rows_root = prediction_root / "rows"
    rows_root.mkdir(parents=True)
    artifact_path = rows_root / "first.json"
    artifact = {
        "held_receiver": "r0",
        "held_class": None,
        "K": 1,
        "arm_prediction_receipts": {
            arm: character * 64
            for arm, character in zip(
                scorer.ARMS,
                ("1", "2", "3", "4"),
                strict=True,
            )
        },
    }
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
    row = {
        "held_receiver": "r0",
        "held_class": None,
        "K": 1,
        "package_id": scorer.package_id("r0", 1),
        "path": "rows/first.json",
        "sha256": scorer.sha256_file(artifact_path),
        "arm_prediction_receipts": artifact["arm_prediction_receipts"],
        "prediction_receipt_sha256": "5" * 64,
        "scorer_input_seal_sha256": "6" * 64,
        "method_lock_sha256": "7" * 64,
        "registered_class_root_sha256": "8" * 64,
        "support_physical_id_root_sha256": "9" * 64,
        "query_physical_id_root_sha256": "a" * 64,
        "int8_gate_pass": True,
    }
    truth_seal_path = tmp_path / "truth_input_seal.json"
    truth_seal_path.write_text("{}", encoding="utf-8")
    manifest = {
        "schema": "cvs.d104_r1.rxid_angq.held_predictions.v1",
        "split_id": scorer.SPLIT_ID,
        "row_count": 63,
        "arm_row_prediction_unit_count": 252,
        "rows": [row] * 63,
        "day_stability_rows": [{}] * 49,
        "package_manifest_sha256": "b" * 64,
        "truth_input_seal_sha256": scorer.sha256_file(truth_seal_path),
        "method_lock_sha256": "7" * 64,
        "registered_classes": [f"c{i}" for i in range(6)],
        "registered_class_root_sha256": scorer.canonical_sha256(
            [f"c{i}" for i in range(6)]
        ),
        "scorer_input_root_sha256": "c" * 64,
        "all_arm_prediction_receipts_unique": True,
        "query_truth_access": False,
        "target_access": False,
        "formal_query_state_updates": 0,
    }
    (prediction_root / "prediction_manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    truth_path = tmp_path / "truth.json"
    truth_path.write_text('{"sentinel":"must-not-be-read"}', encoding="utf-8")
    event_path = tmp_path / "truth_open_event.json"
    output_path = tmp_path / "scores.json"
    truth_read = False
    original_read = scorer._read_json

    def guarded_read(path):
        nonlocal truth_read
        if path == truth_path.resolve():
            truth_read = True
        return original_read(path)

    def reject_internal_seal(_artifact):
        raise ValueError("internal prediction seal drift")

    monkeypatch.setattr(scorer, "_read_json", guarded_read)
    monkeypatch.setattr(
        scorer,
        "validate_d104_prediction_artifact_without_truth",
        reject_internal_seal,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "score-d104",
            "--prediction-root",
            str(prediction_root),
            "--truth-json",
            str(truth_path),
            "--truth-input-seal-json",
            str(truth_seal_path),
            "--output-json",
            str(output_path),
            "--truth-open-event-json",
            str(event_path),
        ],
    )
    with pytest.raises(ValueError, match="internal prediction seal drift"):
        scorer.main()
    assert truth_read is False
    assert event_path.exists() is False
    assert output_path.exists() is False
