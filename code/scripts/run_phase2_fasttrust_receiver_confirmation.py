#!/usr/bin/env python
"""Run the frozen seven-receiver ADV3B02 FastTrust Phase2 confirmation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import torch

from cvsrffi.leo_weak_cache import load_verified_leo_weak_cache_set
from cvsrffi.phase2_fasttrust_receiver_matrix import (
    CAPSULE_ID,
    FORMAL_SCENARIOS,
    FROZEN_CHECKPOINT_PATH,
    SPLIT_ID,
    TARGET_RECEIVERS,
    build_receiver_matrix,
)
from cvsrffi.phase2_fasttrust_staging import stage_receiver_arrays
from cvsrffi.stage2_structured_late_block_runner import run_stage2_row
from cvsrffi.stage2_structured_late_block_scorer import (
    StructuredLateBlockScoringError,
    _load_and_validate_predictions,
    score_stage2c_predictions,
)
from scripts.build_cvs_stage2_support_prototypes import build_support_prototypes


OLD_CLASS_NAMES = ("14-10", "14-7", "20-15", "20-19", "6-15", "8-20")
NEW_CLASS_NAMES = (
    "11-1", "7-11", "10-11", "10-7", "11-4", "11-7", "15-1",
    "16-16", "2-19", "20-12", "20-7", "3-13", "5-5", "6-1",
    "7-10", "8-18", "8-3", "13-3", "4-11", "3-18",
)
CLASS_NAMES = OLD_CLASS_NAMES + NEW_CLASS_NAMES


def _write_json_new(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _load_matrix(path: str | Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    rows = payload.get("rows") if isinstance(payload, Mapping) else None
    if not isinstance(rows, list) or len(rows) != 21:
        raise ValueError("matrix must contain exactly 21 rows")
    return [dict(row) for row in rows]


def stage_all(
    *, cache_set: str | Path, run_root: str | Path, checkpoint_path: str
) -> dict[str, Any]:
    if str(checkpoint_path) != FROZEN_CHECKPOINT_PATH:
        raise ValueError("stage must bind the frozen FastTrust checkpoint")
    root = Path(run_root)
    if root.exists():
        raise ValueError(f"run root already exists: {root}")
    arrays_by_scenario, manifest, audit = load_verified_leo_weak_cache_set(
        cache_set,
        expected_scope="stage2_canonical_registered",
        allowed_roles={"target_old", "target_new"},
    )
    required = {
        "protocol_schema": "p2_min_v1",
        "capsule_id": CAPSULE_ID,
        "split_id": SPLIT_ID,
    }
    mismatched = [key for key, value in required.items() if manifest.get(key) != value]
    if mismatched:
        raise ValueError(f"canonical cache binding mismatch: {mismatched}")
    if set(arrays_by_scenario) != set(FORMAL_SCENARIOS):
        raise ValueError("canonical cache scenario set mismatch")

    rows = build_receiver_matrix(
        run_root=str(root), checkpoint_path=checkpoint_path, seed=713104
    )
    receipts = []
    for receiver in TARGET_RECEIVERS:
        receiver_root = root / "receivers" / f"rx{receiver}" / "package"
        receipts.append(
            stage_receiver_arrays(
                arrays_by_scenario,
                receiver=receiver,
                output_root=receiver_root / "predictor",
                truth_root=receiver_root / "scorer",
                class_names=CLASS_NAMES,
                k_shot=20,
                token_salt=SPLIT_ID,
            )
        )
    matrix_payload = {
        "schema": "cvs.phase2.fasttrust.receiver_matrix.v1",
        "run_id": root.name,
        "protocol_schema": "p2_min_v1",
        "phase2_data_status": "VALIDATED_ONCE",
        "capsule_id": CAPSULE_ID,
        "split_id": SPLIT_ID,
        "seed": 713104,
        "k_shot": 20,
        "class_names": list(CLASS_NAMES),
        "old_class_names": list(OLD_CLASS_NAMES),
        "rows": rows,
    }
    _write_json_new(root / "matrix.json", matrix_payload)
    receipt = {
        "status": "STAGED",
        "rows": 21,
        "receivers": receipts,
        "canonical_cache_single_observation_compliant": audit.get(
            "phase2_single_observation_compliant"
        ) is True,
        "query_truth_in_predictor": False,
    }
    _write_json_new(root / "stage_receipt.json", receipt)
    return receipt


def predict_receiver(
    *, matrix_path: str | Path, receiver: str, device: str
) -> dict[str, Any]:
    rows = [row for row in _load_matrix(matrix_path) if row["receiver"] == receiver]
    if receiver not in TARGET_RECEIVERS or len(rows) != 3:
        raise ValueError("receiver must bind exactly three frozen rows")
    if {row["scenario"] for row in rows} != set(FORMAL_SCENARIOS):
        raise ValueError("receiver scenario set mismatch")

    row_receipts = []
    for row in rows:
        started = time.perf_counter()
        if torch.device(device).type == "cuda":
            torch.cuda.reset_peak_memory_stats(torch.device(device))
        prototype_config = {
            "protocol_schema": "p2_min_v1",
            "phase2_data_status": "VALIDATED_ONCE",
            "capsule_id": CAPSULE_ID,
            "split_id": SPLIT_ID,
            "checkpoint_path": row["checkpoint_path"],
            "support_path": row["support_path"],
            "prototype_path": row["prototype_path"],
            "candidate": "freq_f3_proj",
            "steps": 1,
            "learning_rate": 0.0005,
            "seed": 713104,
            "k_shot": 20,
        }
        prototype_receipt = build_support_prototypes(
            prototype_config,
            support_audit_path=row["support_audit_path"],
            scene=row["scenario"],
            receiver=row["receiver"],
            device=device,
        )
        runner_config = {
            "protocol_schema": "p2_min_v1",
            "phase2_data_status": "VALIDATED_ONCE",
            "capsule_id": CAPSULE_ID,
            "split_id": SPLIT_ID,
            "row_id": row["row_id"],
            "receiver": row["receiver"],
            "scenario": row["scenario"],
            "seed": 713104,
            "k_shot": 20,
            "checkpoint_path": row["checkpoint_path"],
            "support_path": row["support_path"],
            "query_path": row["query_path"],
            "prototype_path": row["prototype_path"],
            "candidate": "freq_f3_proj",
            "steps": 1,
            "learning_rate": 0.0005,
            "decision_rule": "frozen_prototype_cosine_v1",
            "min_trainable_fraction": 0.03,
            "max_trainable_fraction": 0.15,
        }
        prediction_receipt = run_stage2_row(
            runner_config,
            output_dir=row["prediction_output_dir"],
            device=device,
        )
        peak_mib = (
            float(torch.cuda.max_memory_allocated(torch.device(device)) / 2**20)
            if torch.device(device).type == "cuda"
            else 0.0
        )
        row_receipts.append(
            {
                **prediction_receipt,
                "prototype_feature_dim": prototype_receipt["feature_dim"],
                "elapsed_seconds": float(time.perf_counter() - started),
                "peak_cuda_memory_mib": peak_mib,
            }
        )
    receipt = {
        "status": "PREDICTIONS_COMPLETE",
        "receiver": receiver,
        "device": device,
        "rows": row_receipts,
        "truth_opened": False,
    }
    root = Path(matrix_path).parent
    _write_json_new(
        root / "receivers" / f"rx{receiver}" / "prediction_receipt.json",
        receipt,
    )
    return receipt


def assert_all_predictions_complete(rows: Sequence[Mapping[str, Any]]) -> None:
    if len(rows) != 21:
        raise ValueError(f"prediction-first gate requires 21/21 rows, observed {len(rows)}/21")
    for position, row in enumerate(rows):
        prediction = Path(str(row["prediction_output_dir"])) / "predictions.npz"
        if not prediction.is_file():
            raise ValueError(
                f"prediction-first gate requires 21/21 rows, missing {position + 1}/21"
            )
        try:
            query_ids, predicted, scores = _load_and_validate_predictions(prediction)
        except StructuredLateBlockScoringError as exc:
            raise ValueError(
                f"prediction-first validation failed for {row['row_id']}: {exc}"
            ) from exc
        expected_rows = int(row.get("expected_query_rows", 0))
        if (
            expected_rows != 1352
            or query_ids.shape[0] != expected_rows
            or scores.shape != (expected_rows, len(CLASS_NAMES))
            or (predicted >= len(CLASS_NAMES)).any()
        ):
            raise ValueError(
                f"prediction-first validation failed for {row['row_id']}: "
                "row count or class registry drift"
            )


def _aggregate_receiver(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    class_correct = {name: 0 for name in CLASS_NAMES}
    class_total = {name: 0 for name in CLASS_NAMES}
    for row in rows:
        for name in CLASS_NAMES:
            class_correct[name] += int(row["per_class_correct"][name])
            class_total[name] += int(row["per_class_total"][name])
    per_class = {
        name: float(class_correct[name] / class_total[name]) for name in CLASS_NAMES
    }
    old_correct = sum(class_correct[name] for name in OLD_CLASS_NAMES)
    old_total = sum(class_total[name] for name in OLD_CLASS_NAMES)
    new_correct = sum(class_correct[name] for name in NEW_CLASS_NAMES)
    new_total = sum(class_total[name] for name in NEW_CLASS_NAMES)
    return {
        "overall_accuracy": float((old_correct + new_correct) / (old_total + new_total)),
        "old_class_accuracy": float(old_correct / old_total),
        "new_class_accuracy": float(new_correct / new_total),
        "macro_accuracy": float(sum(per_class.values()) / len(per_class)),
        "floor_accuracy": float(min(per_class.values())),
        "per_class_accuracy": per_class,
        "query_rows": old_total + new_total,
    }


def score_all(*, matrix_path: str | Path) -> dict[str, Any]:
    rows = _load_matrix(matrix_path)
    assert_all_predictions_complete(rows)
    scored_rows = []
    for row in rows:
        score = score_stage2c_predictions(
            Path(row["prediction_output_dir"]) / "predictions.npz",
            row["truth_path"],
            output_path=row["score_path"],
            old_class_ids=list(range(len(OLD_CLASS_NAMES))),
            class_names=CLASS_NAMES,
        )
        scored_rows.append(
            {
                "row_id": row["row_id"],
                "receiver": row["receiver"],
                "scenario": row["scenario"],
                **score,
            }
        )
    by_receiver = {
        receiver: _aggregate_receiver(
            [row for row in scored_rows if row["receiver"] == receiver]
        )
        for receiver in TARGET_RECEIVERS
    }
    result = {
        "schema": "cvs.phase2.fasttrust.receiver_confirmation.summary.v1",
        "status": "ANALYZED",
        "state": "DA1_REG1",
        "prediction_first_rows": 21,
        "truth_joined_after_all_predictions": True,
        "rows": scored_rows,
        "by_receiver": by_receiver,
    }
    _write_json_new(Path(matrix_path).parent / "summary.json", result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    stage = sub.add_parser("stage")
    stage.add_argument("--cache-set", type=Path, required=True)
    stage.add_argument("--run-root", type=Path, required=True)
    stage.add_argument("--checkpoint", required=True)
    predict = sub.add_parser("predict-receiver")
    predict.add_argument("--matrix", type=Path, required=True)
    predict.add_argument("--receiver", choices=TARGET_RECEIVERS, required=True)
    predict.add_argument("--device", required=True)
    score = sub.add_parser("score-all")
    score.add_argument("--matrix", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "stage":
        result = stage_all(
            cache_set=args.cache_set,
            run_root=args.run_root,
            checkpoint_path=args.checkpoint,
        )
    elif args.command == "predict-receiver":
        result = predict_receiver(
            matrix_path=args.matrix, receiver=args.receiver, device=args.device
        )
    else:
        result = score_all(matrix_path=args.matrix)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CLASS_NAMES",
    "NEW_CLASS_NAMES",
    "OLD_CLASS_NAMES",
    "assert_all_predictions_complete",
    "main",
    "predict_receiver",
    "score_all",
    "stage_all",
]
