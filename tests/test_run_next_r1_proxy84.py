from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from cvsrffi import stage2_next_r1_matrix as matrix


SCRIPT = Path(__file__).parents[1] / "code" / "scripts" / "run_next_r1_proxy84.py"
SPEC = importlib.util.spec_from_file_location("run_next_r1_proxy84", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def test_predict_cli_has_no_truth_or_score_input() -> None:
    parser = runner._parser()
    predict = next(
        action for action in parser._actions if action.dest == "command"
    ).choices["predict"]
    destinations = {action.dest for action in predict._actions}
    assert "truth" not in destinations
    assert "score" not in destinations


def test_score_refuses_incomplete_manifest(tmp_path) -> None:
    with pytest.raises(runner.NextR1Proxy84Error, match="complete sealed"):
        runner.run_score(type("Args", (), {"run_root": tmp_path})())


def test_independent_score_uses_only_k5_rows_and_writes_once(tmp_path) -> None:
    predictions = np.zeros(
        (matrix.ROW_COUNT, len(matrix.ARM_IDS), matrix.QUERY_COUNT), dtype=np.int64
    )
    truth = np.zeros((matrix.ROW_COUNT, matrix.QUERY_COUNT), dtype=np.int64)
    for row in range(matrix.ROW_COUNT):
        truth[row] = np.repeat(np.arange(6, dtype=np.int64), 9)
        predictions[row, :, :] = truth[row]
    plan = matrix.build_next_r1_loco_plan(
        tuple(f"rx{index}" for index in range(7)),
        tuple(f"tx{index}" for index in range(6)),
    )
    row_ids = np.asarray(
        [value["row_id"] for value in plan["rows"]], dtype="<U96"
    )
    np.savez_compressed(
        tmp_path / "predictions.npz",
        row_ids=row_ids,
        arm_ids=np.asarray(matrix.ARM_IDS, dtype="<U4"),
        registered_classes=np.asarray(
            [tuple(value["registered_classes"]) for value in plan["rows"]],
            dtype="<U4",
        ),
        predictions=predictions,
    )
    np.savez_compressed(tmp_path / "truth_side.npz", row_ids=row_ids, truth=truth)
    (tmp_path / "rows").mkdir()
    runner._write_json_new(tmp_path / "plan.json", plan)
    manifest_rows = []
    for index, plan_row in enumerate(plan["rows"]):
        seal = runner.runtime.NextR1RowSeal(
            row_id=plan_row["row_id"], active_k=plan_row["active_k"],
            held_receiver=plan_row["held_receiver"], held_class=plan_row["held_class"],
            matrix_sha256=plan["matrix_sha256"], binding_sha256=f"{index + 1:064x}",
            prediction_receipt_sha256=f"{index + 101:064x}",
            resource_receipt_sha256=f"{index + 201:064x}",
            forward_receipt_sha256=f"{index + 301:064x}",
            smoke_receipt_sha256=f"{index + 401:064x}",
        )
        manifest_rows.append(dict(seal.wire_mapping()))
        stem = f"{index:03d}_{runner._sha(plan_row['row_id'].encode())[:16]}"
        row_npz = tmp_path / "rows" / f"{stem}.npz"
        np.savez_compressed(
            row_npz,
            arm_ids=np.asarray(matrix.ARM_IDS, dtype="<U4"),
            logits=np.zeros((6, 54, 6), dtype=np.float32),
            predictions=predictions[index],
        )
        runner._write_json_new(
            tmp_path / "rows" / f"{stem}.json",
            {
                "row": {
                    "row_id": plan_row["row_id"], "active_k": plan_row["active_k"],
                    "registered_classes": plan_row["registered_classes"],
                },
                "npz_sha256": runner._sha_file(row_npz),
                "row_seal": seal.wire_mapping(),
            },
        )
    manifest = {
        "schema": runner.runtime.SEALED_MANIFEST_SCHEMA,
        "matrix_sha256": plan["matrix_sha256"], "candidate_id": matrix.CANDIDATE_ID,
        "row_count": matrix.ROW_COUNT, "all_rows_sealed": True,
        "sealed_before_scoring": True, "rows": manifest_rows,
    }
    manifest["sealed_manifest_sha256"] = runner._sha(runner._canonical(manifest))
    runner._write_json_new(tmp_path / "manifest.json", manifest)
    (tmp_path / "completion.json").write_text(
        json.dumps(
            {
                "status": "ARTIFACTS_COMPLETE_NOT_SCORED",
                "row_count": matrix.ROW_COUNT,
                "plan_sha256": runner._sha_file(tmp_path / "plan.json"),
                "manifest_sha256": runner._sha_file(tmp_path / "manifest.json"),
                "predictions_sha256": runner._sha_file(tmp_path / "predictions.npz"),
                "truth_side_sha256": runner._sha_file(tmp_path / "truth_side.npz"),
            }
        ),
        encoding="utf-8",
    )
    args = type("Args", (), {"run_root": tmp_path})()
    runner.run_score(args)
    score = json.loads((tmp_path / "score" / "scores.json").read_text(encoding="utf-8"))
    assert score["k5_row_count"] == matrix.FOLD_COUNT
    assert score["candidate_promotable"] is False
    assert all(value["H_proxy"] == 1.0 for value in score["arm_metrics"].values())
    with pytest.raises(runner.NextR1Proxy84Error, match="overwrite"):
        runner.run_score(args)


def test_score_rejects_empty_or_mixed_manifest(tmp_path) -> None:
    for name in ("plan.json", "manifest.json", "completion.json", "predictions.npz", "truth_side.npz"):
        (tmp_path / name).write_bytes(b"{}" if name.endswith(".json") else b"broken")
    with pytest.raises(runner.NextR1Proxy84Error):
        runner.run_score(type("Args", (), {"run_root": tmp_path})())
