#!/usr/bin/env python3
"""Score one sealed Phase2 row after immutable prediction publication."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

from cvsrffi.stage2_ablation_factory import (
    get_stage2_arm,
    resolved_stage2_config_hash,
)
from cvsrffi.stage2_ablation_release import (
    SCORE_COMPLETION_SCHEMA,
    SCORE_REQUEST_SCHEMA,
    canonical_json_bytes,
)
from cvsrffi.stage2_ablation_row_executor import ROW_EXECUTION_SCHEMA
from cvsrffi.stage2_ablation_truth_scorer import (
    SAME_ROW_SCORE_SCHEMA,
    score_full_ablation_row,
    write_row_record_exclusive,
)


_REQUEST_KEYS = {
    "schema",
    "logical_row_key",
    "ablation_id",
    "physical_execution_id",
    "effective_config_hash",
    "alias_of",
    "row_execution_receipt",
    "scoring_manifest",
    "scoring_manifest_sha256",
    "output_path",
    "completion_receipt_path",
}


class Stage2AblationScoreRequestError(ValueError):
    """Raised when a truth-side score request is not release-bound."""


def _load_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(value, Mapping):
        raise Stage2AblationScoreRequestError(
            f"JSON root must be an object: {path}"
        )
    return dict(value)


def _exclusive_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    data = canonical_json_bytes(payload) + b"\n"
    descriptor = os.open(
        destination,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL
        | getattr(os, "O_BINARY", 0),
        0o444,
    )
    try:
        view = memoryview(data)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short write while publishing score completion")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def run_score_request(path: str | Path) -> dict[str, Any]:
    request = _load_json(path)
    if (
        set(request) != _REQUEST_KEYS
        or request.get("schema") != SCORE_REQUEST_SCHEMA
    ):
        raise Stage2AblationScoreRequestError(
            "score request exact schema drift"
        )
    ablation_id = str(request["ablation_id"])
    spec = get_stage2_arm(ablation_id)
    expected_config_hash = resolved_stage2_config_hash(ablation_id)
    if request["effective_config_hash"] != expected_config_hash:
        raise Stage2AblationScoreRequestError(
            "score request effective config hash drift"
        )
    receipt = _load_json(request["row_execution_receipt"])
    if (
        receipt.get("schema") != ROW_EXECUTION_SCHEMA
        or receipt.get("status")
        != "PREDICTIONS_COMPLETE_TRUTH_UNOPENED"
        or receipt.get("row_id")
        != request["physical_execution_id"]
        or receipt.get("query_truth_opened") is not False
        or int(receipt.get("fit_query_rows_used", -1)) != 0
    ):
        raise Stage2AblationScoreRequestError(
            "row execution receipt is incomplete"
        )
    physical_ablation_id = str(receipt.get("ablation_id", ""))
    if request["alias_of"] is None:
        if physical_ablation_id != ablation_id:
            raise Stage2AblationScoreRequestError(
                "non-alias score request/receipt arm drift"
            )
    else:
        if spec.alias_of != physical_ablation_id:
            raise Stage2AblationScoreRequestError(
                "logical alias does not match physical arm"
            )
        if (
            resolved_stage2_config_hash(physical_ablation_id)
            != expected_config_hash
        ):
            raise Stage2AblationScoreRequestError(
                "logical alias physical config drift"
            )
    prediction = dict(receipt.get("prediction") or {})
    required_prediction = {
        "path",
        "artifact_sha256",
        "seal_sha256",
    }
    if not required_prediction <= set(prediction):
        raise Stage2AblationScoreRequestError(
            "prediction receipt is incomplete"
        )
    result = score_full_ablation_row(
        prediction["path"],
        request["scoring_manifest"],
        expected_prediction_artifact_sha256=prediction[
            "artifact_sha256"
        ],
        expected_prediction_seal_sha256=prediction["seal_sha256"],
        expected_scoring_manifest_sha256=request[
            "scoring_manifest_sha256"
        ],
        row_identity={
            "logical_row_key": request["logical_row_key"],
            "ablation_id": ablation_id,
            "physical_execution_id": request[
                "physical_execution_id"
            ],
            "effective_config_hash": expected_config_hash,
            "alias_of": request["alias_of"],
        },
        behavior_receipt=receipt["behavior"],
        quantization_receipt=receipt["quantization"],
        resource_receipt=receipt["resource"],
    )
    if (
        result.get("schema") != SAME_ROW_SCORE_SCHEMA
        or result.get("status") != "PASS"
    ):
        raise Stage2AblationScoreRequestError(
            "same-row scorer did not close"
        )
    write_row_record_exclusive(request["output_path"], result)
    output_bytes = Path(request["output_path"]).read_bytes()
    _exclusive_json(
        request["completion_receipt_path"],
        {
            "schema": SCORE_COMPLETION_SCHEMA,
            "status": "PASS",
            "logical_row_key": request["logical_row_key"],
            "ablation_id": ablation_id,
            "physical_execution_id": request[
                "physical_execution_id"
            ],
            "effective_config_hash": expected_config_hash,
            "alias_of": request["alias_of"],
            "score_output_path": request["output_path"],
            "score_output_sha256": hashlib.sha256(
                output_bytes
            ).hexdigest(),
            "formal_scenario_count": 3,
            "performance_values_present": False,
        },
    )
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True)
    return parser


def main() -> int:
    result = run_score_request(_parser().parse_args().request)
    print(
        json.dumps(
            {
                "status": result["status"],
                "logical_row_key": result["logical_row_key"],
                "physical_execution_id": result[
                    "physical_execution_id"
                ],
                "scorer_receipt_sha256": result[
                    "scorer_receipt_sha256"
                ],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
