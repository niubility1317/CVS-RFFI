from __future__ import annotations

import json
from pathlib import Path

from cvsrffi.stage2_ablation_release import SCORE_REQUEST_SCHEMA
from cvsrffi.stage2_ablation_row_executor import ROW_EXECUTION_SCHEMA
from cvsrffi.stage2_ablation_truth_scorer import SAME_ROW_SCORE_SCHEMA
from scripts.score_full_ablation_stage2_row import run_score_request


def test_alias_score_request_reuses_full_prediction(
    tmp_path: Path,
    monkeypatch,
) -> None:
    prediction_path = tmp_path / "predictions.cvspred"
    prediction_path.write_bytes(b"sealed")
    receipt_path = tmp_path / "row_execution_receipt.json"
    receipt = {
        "schema": ROW_EXECUTION_SCHEMA,
        "status": "PREDICTIONS_COMPLETE_TRUTH_UNOPENED",
        "ablation_id": "P2-FULL",
        "row_id": "physical-1",
        "query_truth_opened": False,
        "fit_query_rows_used": 0,
        "prediction": {
            "path": str(prediction_path),
            "artifact_sha256": "a" * 64,
            "seal_sha256": "b" * 64,
        },
        "behavior": {"receipt": "behavior"},
        "quantization": {"receipt": "quantization"},
        "resource": {"receipt": "resource"},
    }
    receipt_path.write_text(
        json.dumps(receipt), encoding="utf-8"
    )
    output_path = tmp_path / "score.json"
    request_path = tmp_path / "request.json"
    request = {
        "schema": SCORE_REQUEST_SCHEMA,
        "logical_row_key": "P2-F3__logical",
        "ablation_id": "P2-F3",
        "physical_execution_id": "physical-1",
        "effective_config_hash": "",
        "alias_of": "P2-FULL__logical",
        "row_execution_receipt": str(receipt_path),
        "scoring_manifest": str(tmp_path / "truth.manifest.json"),
        "scoring_manifest_sha256": "c" * 64,
        "output_path": str(output_path),
        "completion_receipt_path": str(
            tmp_path / "score.completion.json"
        ),
    }
    from cvsrffi.stage2_ablation_factory import (
        resolved_stage2_config_hash,
    )

    request["effective_config_hash"] = resolved_stage2_config_hash(
        "P2-F3"
    )
    request_path.write_text(
        json.dumps(request), encoding="utf-8"
    )
    captured = {}

    def fake_score(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return {
            "schema": SAME_ROW_SCORE_SCHEMA,
            "status": "PASS",
            "logical_row_key": "P2-F3__logical",
            "physical_execution_id": "physical-1",
            "scorer_receipt_sha256": "d" * 64,
        }

    def fake_write(path, payload):
        Path(path).write_text(
            json.dumps(payload), encoding="utf-8"
        )

    monkeypatch.setattr(
        "scripts.score_full_ablation_stage2_row."
        "score_full_ablation_row",
        fake_score,
    )
    monkeypatch.setattr(
        "scripts.score_full_ablation_stage2_row."
        "write_row_record_exclusive",
        fake_write,
    )
    result = run_score_request(request_path)
    assert result["status"] == "PASS"
    assert captured["kwargs"]["row_identity"]["alias_of"] == (
        "P2-FULL__logical"
    )
    assert captured["kwargs"]["behavior_receipt"] == {
        "receipt": "behavior"
    }
    assert output_path.is_file()
