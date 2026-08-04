#!/usr/bin/env python3
"""Run and independently score the frozen NEXT-R1 84-row proxy matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np


SCRIPT_ROOT = Path(__file__).resolve().parent
CODE_ROOT = SCRIPT_ROOT.parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from cvsrffi import stage2_d129_joint6_da as d129_loco  # noqa: E402
from cvsrffi import stage2_d129_joint6_heads as d129_heads  # noqa: E402
from cvsrffi import stage2_next_r1_fabr as fabr  # noqa: E402
from cvsrffi import stage2_next_r1_matrix as matrix  # noqa: E402
from cvsrffi import stage2_next_r1_real as real  # noqa: E402
from cvsrffi import stage2_next_r1_runtime as runtime  # noqa: E402


SCHEMA = "cvs.stage2.next_r1.proxy84.runner.v1"
SCORE_SCHEMA = "cvs.stage2.next_r1.proxy84.score.v1"


class NextR1Proxy84Error(ValueError):
    """The immutable NEXT-R1 proxy84 execution closure did not hold."""


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _canonical(value: Any) -> bytes:
    return json.dumps(
        _plain(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_file(path: Path) -> str:
    return _sha(path.read_bytes())


def _write_json_new(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(_plain(value), handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")


def _new_root(path: Path) -> Path:
    if not path.is_absolute() or path.exists() or not path.parent.is_dir():
        raise NextR1Proxy84Error(
            "run root must be a new absolute child of an existing directory"
        )
    path.mkdir()
    (path / "rows").mkdir()
    return path


def _row_from_mapping(value: Mapping[str, Any]) -> matrix.NextR1LocoRow:
    return matrix.NextR1LocoRow(
        row_id=str(value["row_id"]),
        held_receiver=str(value["held_receiver"]),
        held_class=str(value["held_class"]),
        active_k=int(value["active_k"]),
        retained_classes=tuple(value["retained_classes"]),
        registered_classes=tuple(value["registered_classes"]),
    )


def _ordered_cell_indices(rows: real.NextR1RealRows) -> Mapping[tuple[str, str], tuple[int, ...]]:
    result: dict[tuple[str, str], tuple[int, ...]] = {}
    for receiver in rows.receiver_registry:
        for class_id in rows.class_registry:
            values = [
                index
                for index, (observed_receiver, observed_class) in enumerate(
                    zip(rows.receiver_ids, rows.tx_labels, strict=True)
                )
                if observed_receiver == receiver and observed_class == class_id
            ]
            if len(values) != matrix.PHYSICAL_PER_CELL:
                raise NextR1Proxy84Error("every receiver/class cell must contain 14 rows")
            result[(receiver, class_id)] = tuple(
                sorted(
                    values,
                    key=lambda index: _sha(
                        f"{d129_loco.LOCO_SALT}|{receiver}|{class_id}|{rows.physical_ids[index]}".encode()
                    ),
                )
            )
    return result


def _build_loco(rows: real.NextR1RealRows) -> d129_loco.D129LOCOPlan:
    return d129_loco.build_d129_loco_plan(
        d129_loco.D129LOCORecord(receiver, class_id, physical_id)
        for receiver, class_id, physical_id in zip(
            rows.receiver_ids, rows.tx_labels, rows.physical_ids, strict=True
        )
    )


def _fold_inputs(
    rows: real.NextR1RealRows,
    plan: Mapping[str, Any],
    loco: d129_loco.D129LOCOPlan,
    cells: Mapping[tuple[str, str], tuple[int, ...]],
    held_receiver: str,
    held_class: str,
) -> tuple[
    matrix.NextR1LocoRow,
    matrix.NextR1LocoRow,
    Mapping[str, Any],
    Mapping[str, tuple[int, ...]],
    Mapping[str, tuple[int, ...]],
    tuple[int, ...],
]:
    matched = [
        value
        for value in plan["rows"]
        if value["held_receiver"] == held_receiver and value["held_class"] == held_class
    ]
    if len(matched) != 2:
        raise NextR1Proxy84Error("frozen plan fold row pairing drift")
    row1 = _row_from_mapping(next(value for value in matched if value["active_k"] == 1))
    row5 = _row_from_mapping(next(value for value in matched if value["active_k"] == 5))
    support5 = {
        class_id: cells[(held_receiver, class_id)][:5]
        for class_id in row5.registered_classes
    }
    support1 = {class_id: value[:1] for class_id, value in support5.items()}
    query = {
        class_id: cells[(held_receiver, class_id)][5:]
        for class_id in row5.registered_classes
    }
    phase1_indices = tuple(
        index
        for receiver in rows.receiver_registry
        if receiver != held_receiver
        for class_id in rows.class_registry
        if class_id != held_class
        for index in cells[(receiver, class_id)]
    )
    fold = next(
        value
        for value in loco.folds
        if value.held_receiver == held_receiver and value.held_class == held_class
    )
    binding = matrix.bind_next_r1_physical_ids(
        row_k1=row1,
        row_k5=row5,
        loco_fold_receipt=fold.as_dict(),
        phase1_fit_ids=tuple(rows.physical_ids[index] for index in phase1_indices),
        k1_support_ids_by_class={
            key: tuple(rows.physical_ids[index] for index in value)
            for key, value in support1.items()
        },
        k5_support_ids_by_class={
            key: tuple(rows.physical_ids[index] for index in value)
            for key, value in support5.items()
        },
        query_ids_by_class={
            key: tuple(rows.physical_ids[index] for index in value)
            for key, value in query.items()
        },
    )
    return row1, row5, binding, support1, support5, tuple(
        index for class_id in row5.registered_classes for index in query[class_id]
    )


def _save_row(root: Path, index: int, result: runtime.NextR1RuntimeResult) -> None:
    stem = f"{index:03d}_{_sha(result.row.row_id.encode())[:16]}"
    path = root / "rows" / f"{stem}.npz"
    if path.exists():
        raise NextR1Proxy84Error("row output overwrite refused")
    logits = np.stack([result.arm_logits[arm] for arm in matrix.ARM_IDS])
    predictions = np.stack([result.arm_predictions[arm] for arm in matrix.ARM_IDS])
    np.savez_compressed(
        path,
        arm_ids=np.asarray(matrix.ARM_IDS, dtype="<U4"),
        logits=np.ascontiguousarray(logits, dtype=np.float32),
        predictions=np.ascontiguousarray(predictions, dtype=np.int64),
    )
    _write_json_new(
        root / "rows" / f"{stem}.json",
        {
            "schema": SCHEMA,
            "row": {
                "row_id": result.row.row_id,
                "active_k": result.row.active_k,
                "held_receiver": result.row.held_receiver,
                "held_class": result.row.held_class,
                "registered_classes": result.row.registered_classes,
            },
            "npz_sha256": _sha_file(path),
            "row_seal": result.row_seal.wire_mapping(),
            "prediction_receipt": result.prediction_receipt,
            "resource_receipt": result.resource_receipt,
            "forward_receipt": result.forward_receipt,
            "smoke_receipt": result.smoke_receipt,
        },
    )


def run_predict(args: argparse.Namespace) -> None:
    root = _new_root(args.run_root)
    rows = real.load_next_r1_real_rows(
        selected_iq_archive=args.selected_iq,
        selected_iq_archive_sha256=args.selected_iq_sha256,
        selected_iq_receipt=args.selected_receipt,
        selected_iq_receipt_sha256=args.selected_receipt_sha256,
        ls_label_join_archive=args.ls_join,
        ls_label_join_archive_sha256=args.ls_join_sha256,
    )
    bridge, model_receipt = real.load_next_r1_real_model(
        rows,
        checkpoint_path=args.checkpoint,
        checkpoint_sha256=args.checkpoint_sha256,
        device=args.device,
    )
    plan = matrix.build_next_r1_loco_plan(rows.receiver_registry, rows.class_registry)
    loco = _build_loco(rows)
    cells = _ordered_cell_indices(rows)
    plan_rows = [_row_from_mapping(value) for value in plan["rows"]]
    results: list[runtime.NextR1RuntimeResult] = []
    truth_rows: list[np.ndarray] = []
    truth_row_ids: list[str] = []
    registered_rows: list[tuple[str, ...]] = []
    f_archive_sha256 = _sha_file(Path(d129_heads.__file__).resolve())

    _write_json_new(
        root / "preregistration.json",
        {
            "schema": SCHEMA,
            "run_id": args.run_id,
            "matrix_sha256": plan["matrix_sha256"],
            "row_count": matrix.ROW_COUNT,
            "candidate_id": matrix.CANDIDATE_ID,
            "arm_ids": matrix.ARM_IDS,
            "checkpoint_sha256": args.checkpoint_sha256,
            "selected_iq_sha256": args.selected_iq_sha256,
            "selected_receipt_sha256": args.selected_receipt_sha256,
            "ls_join_sha256": args.ls_join_sha256,
            "frozen_f_archive_sha256": f_archive_sha256,
            "device": args.device,
            "model_receipt": model_receipt,
            "truth_opened_by_arm_callbacks": False,
        },
    )
    _write_json_new(root / "plan.json", plan)

    row_index = 0
    for held_receiver in rows.receiver_registry:
        for held_class in rows.class_registry:
            row1, row5, binding, support1, support5, query_indices = _fold_inputs(
                rows, plan, loco, cells, held_receiver, held_class
            )
            bundle = real.build_next_r1_real_asset(
                bridge,
                held_receiver=held_receiver,
                held_class=held_class,
                row_phase1_seal_sha256=binding["phase1_seal_sha256"],
                microbatch_size=args.microbatch_size,
            )
            basis = fabr.decode_fabr_basis(bundle.fabr_asset)
            smoke = real.verified_checkpoint_smoke(
                bridge,
                bundle,
                checkpoint_path=args.checkpoint,
                checkpoint_sha256=args.checkpoint_sha256,
                smoke_indices=query_indices[:2],
            )
            for row, support in ((row1, support1), (row5, support5)):
                expected_row = plan_rows[row_index]
                if row != expected_row:
                    raise NextR1Proxy84Error("execution order drifted from frozen matrix")
                support_indices = tuple(
                    index
                    for class_id in row.registered_classes
                    for index in support[class_id]
                )
                support_labels = tuple(
                    class_id
                    for class_id in row.registered_classes
                    for _ in support[class_id]
                )
                result = runtime.execute_next_r1_row(
                    plan=plan,
                    binding=binding,
                    row=row,
                    bundle=bundle,
                    verified_checkpoint_smoke=smoke,
                    support_token=(row.row_id, "support"),
                    query_token=(row.row_id, "query"),
                    support_labels=support_labels,
                    support_forward_with_coeff=real.make_forward_callback(
                        bridge, support_indices, bundle.fabr_asset.block_id, basis
                    ),
                    query_forward_with_coeff=real.make_forward_callback(
                        bridge, query_indices, bundle.fabr_asset.block_id, basis
                    ),
                    q_callback=real.frozen_qknn_head,
                    frozen_f_callback=real.frozen_d92_full160_head,
                    frozen_f_archive_sha256=f_archive_sha256,
                )
                _save_row(root, row_index, result)
                results.append(result)
                truth_rows.append(
                    np.asarray(
                        [
                            row.registered_classes.index(rows.tx_labels[index])
                            for index in query_indices
                        ],
                        dtype=np.int64,
                    )
                )
                truth_row_ids.append(row.row_id)
                registered_rows.append(row.registered_classes)
                row_index += 1
                print(
                    f"NEXT_R1_PROGRESS launched={row_index} completed={row_index} "
                    f"succeeded={row_index} failed=0 row={row.row_id}",
                    flush=True,
                )

    manifest = runtime.build_next_r1_sealed_manifest(plan, results)
    _write_json_new(root / "manifest.json", manifest)
    prediction_stack = np.stack(
        [
            np.stack([result.arm_predictions[arm] for arm in matrix.ARM_IDS])
            for result in results
        ]
    )
    np.savez_compressed(
        root / "predictions.npz",
        row_ids=np.asarray(truth_row_ids, dtype="<U96"),
        arm_ids=np.asarray(matrix.ARM_IDS, dtype="<U4"),
        registered_classes=np.asarray(registered_rows, dtype="<U96"),
        predictions=np.ascontiguousarray(prediction_stack, dtype=np.int64),
    )
    np.savez_compressed(
        root / "truth_side.npz",
        row_ids=np.asarray(truth_row_ids, dtype="<U96"),
        truth=np.ascontiguousarray(np.stack(truth_rows), dtype=np.int64),
    )
    completion = {
        "schema": SCHEMA,
        "status": "ARTIFACTS_COMPLETE_NOT_SCORED",
        "run_id": args.run_id,
        "row_count": len(results),
        "manifest_sha256": _sha_file(root / "manifest.json"),
        "plan_sha256": _sha_file(root / "plan.json"),
        "predictions_sha256": _sha_file(root / "predictions.npz"),
        "truth_side_sha256": _sha_file(root / "truth_side.npz"),
    }
    _write_json_new(root / "completion.json", completion)
    print(json.dumps(completion, sort_keys=True), flush=True)


def _metrics(
    predictions: np.ndarray,
    truth: np.ndarray,
    registered_classes: np.ndarray,
) -> Mapping[str, Any]:
    retained_correct = int(np.sum(predictions[:, :45] == truth[:, :45]))
    held_correct = int(np.sum(predictions[:, 45:] == truth[:, 45:]))
    retained_total = int(predictions.shape[0] * 45)
    held_total = int(predictions.shape[0] * 9)
    a_retained = retained_correct / retained_total
    a_held = held_correct / held_total
    harmonic = 0.0 if a_retained + a_held == 0.0 else 2.0 * a_retained * a_held / (a_retained + a_held)
    per_class: dict[str, list[int]] = {}
    for row in range(predictions.shape[0]):
        for class_index in range(5):
            label = str(registered_classes[row, class_index])
            per_class.setdefault(label, []).extend(
                (
                    predictions[row, class_index * 9 : (class_index + 1) * 9]
                    == truth[row, class_index * 9 : (class_index + 1) * 9]
                ).astype(np.int8).tolist()
            )
    if len(per_class) != matrix.CLASS_COUNT:
        raise NextR1Proxy84Error("retained-floor class coverage drift")
    floor = min(float(np.mean(values)) for values in per_class.values())
    return {
        "A_retained": a_retained,
        "A_held_proxy": a_held,
        "H_proxy": harmonic,
        "F_retained": floor,
        "total_correct": retained_correct + held_correct,
        "retained_correct": retained_correct,
        "held_correct": held_correct,
    }


def run_score(args: argparse.Namespace) -> None:
    root = args.run_root
    for name in (
        "plan.json",
        "manifest.json",
        "completion.json",
        "predictions.npz",
        "truth_side.npz",
    ):
        if not (root / name).is_file():
            raise NextR1Proxy84Error("score requires a complete sealed 84-row prediction set")
    completion = json.loads((root / "completion.json").read_text(encoding="utf-8"))
    expected_hashes = {
        "plan_sha256": _sha_file(root / "plan.json"),
        "manifest_sha256": _sha_file(root / "manifest.json"),
        "predictions_sha256": _sha_file(root / "predictions.npz"),
        "truth_side_sha256": _sha_file(root / "truth_side.npz"),
    }
    if (
        completion.get("status") != "ARTIFACTS_COMPLETE_NOT_SCORED"
        or completion.get("row_count") != matrix.ROW_COUNT
        or any(completion.get(key) != value for key, value in expected_hashes.items())
    ):
        raise NextR1Proxy84Error("score refused incomplete prediction closure")
    plan_value = json.loads((root / "plan.json").read_text(encoding="utf-8"))
    try:
        plan = matrix.validate_next_r1_plan(plan_value)
    except matrix.NextR1MatrixError as error:
        raise NextR1Proxy84Error("score refused invalid frozen plan") from error
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    manifest_payload = dict(manifest)
    observed_manifest_sha = manifest_payload.pop("sealed_manifest_sha256", None)
    if (
        observed_manifest_sha != _sha(_canonical(manifest_payload))
        or manifest_payload.get("schema") != runtime.SEALED_MANIFEST_SCHEMA
        or manifest_payload.get("matrix_sha256") != plan["matrix_sha256"]
        or manifest_payload.get("candidate_id") != matrix.CANDIDATE_ID
        or manifest_payload.get("row_count") != matrix.ROW_COUNT
        or manifest_payload.get("all_rows_sealed") is not True
        or manifest_payload.get("sealed_before_scoring") is not True
        or not isinstance(manifest_payload.get("rows"), list)
        or len(manifest_payload["rows"]) != matrix.ROW_COUNT
    ):
        raise NextR1Proxy84Error("score refused invalid sealed manifest")
    validated_seals: list[runtime.NextR1RowSeal] = []
    for plan_row, seal_value in zip(
        plan["rows"], manifest_payload["rows"], strict=True
    ):
        try:
            seal = runtime.NextR1RowSeal(**seal_value)
        except (TypeError, runtime.NextR1RuntimeError) as error:
            raise NextR1Proxy84Error("score refused invalid row seal") from error
        if (
            seal.row_id != plan_row["row_id"]
            or seal.active_k != plan_row["active_k"]
            or seal.held_receiver != plan_row["held_receiver"]
            or seal.held_class != plan_row["held_class"]
            or seal.matrix_sha256 != plan["matrix_sha256"]
        ):
            raise NextR1Proxy84Error("score row seal/frozen plan drift")
        validated_seals.append(seal)
    with np.load(root / "predictions.npz", allow_pickle=False) as archive:
        row_ids = np.asarray(archive["row_ids"])
        arm_ids = tuple(str(value) for value in archive["arm_ids"].tolist())
        registered_classes = np.asarray(archive["registered_classes"])
        predictions = np.asarray(archive["predictions"])
    with np.load(root / "truth_side.npz", allow_pickle=False) as archive:
        truth_row_ids = np.asarray(archive["row_ids"])
        truth = np.asarray(archive["truth"])
    if (
        arm_ids != matrix.ARM_IDS
        or predictions.shape != (matrix.ROW_COUNT, len(matrix.ARM_IDS), matrix.QUERY_COUNT)
        or registered_classes.shape != (matrix.ROW_COUNT, matrix.CLASS_COUNT)
        or truth.shape != (matrix.ROW_COUNT, matrix.QUERY_COUNT)
        or not np.array_equal(row_ids, truth_row_ids)
    ):
        raise NextR1Proxy84Error("prediction/truth score package drift")
    expected_row_ids = np.asarray(
        [value["row_id"] for value in plan["rows"]], dtype=row_ids.dtype
    )
    expected_classes = np.asarray(
        [value["registered_classes"] for value in plan["rows"]],
        dtype=registered_classes.dtype,
    )
    if not np.array_equal(row_ids, expected_row_ids) or not np.array_equal(
        registered_classes, expected_classes
    ):
        raise NextR1Proxy84Error("score package row/class order drift from frozen plan")
    for index, (plan_row, seal) in enumerate(
        zip(plan["rows"], validated_seals, strict=True)
    ):
        stem = f"{index:03d}_{_sha(seal.row_id.encode())[:16]}"
        row_json_path = root / "rows" / f"{stem}.json"
        row_npz_path = root / "rows" / f"{stem}.npz"
        if not row_json_path.is_file() or not row_npz_path.is_file():
            raise NextR1Proxy84Error("score requires every sealed row artifact")
        row_document = json.loads(row_json_path.read_text(encoding="utf-8"))
        if (
            row_document.get("npz_sha256") != _sha_file(row_npz_path)
            or row_document.get("row_seal") != dict(seal.wire_mapping())
            or row_document.get("row", {}).get("row_id") != plan_row["row_id"]
            or row_document.get("row", {}).get("active_k") != plan_row["active_k"]
            or tuple(row_document.get("row", {}).get("registered_classes", ()))
            != tuple(plan_row["registered_classes"])
        ):
            raise NextR1Proxy84Error("score row artifact/seal drift")
        with np.load(row_npz_path, allow_pickle=False) as row_archive:
            row_arm_ids = tuple(str(value) for value in row_archive["arm_ids"].tolist())
            row_predictions = np.asarray(row_archive["predictions"])
        if row_arm_ids != matrix.ARM_IDS or not np.array_equal(
            row_predictions, predictions[index]
        ):
            raise NextR1Proxy84Error("score combined/row prediction drift")
    k5 = np.asarray(
        [index for index, value in enumerate(plan["rows"]) if value["active_k"] == 5],
        dtype=np.int64,
    )
    if k5.shape != (matrix.FOLD_COUNT,):
        raise NextR1Proxy84Error("score K5 coverage drift")
    by_arm = {
        arm: _metrics(predictions[k5, index], truth[k5], registered_classes[k5])
        for index, arm in enumerate(matrix.ARM_IDS)
    }
    comparisons: dict[str, Any] = {}
    for name, left, right in (
        ("DA_R1Q_vs_R0Q", "R1Q", "R0Q"),
        ("TSL_R0L_vs_R0F", "R0L", "R0F"),
        ("JOINT_R1L_vs_R1F", "R1L", "R1F"),
    ):
        delta = {
            key: by_arm[left][key] - by_arm[right][key]
            for key in ("A_retained", "A_held_proxy", "H_proxy", "F_retained", "total_correct")
        }
        delta["passes_frozen_gate"] = bool(
            delta["H_proxy"] > 0.0
            and delta["total_correct"] > 0
            and delta["A_retained"] >= 0.0
            and delta["A_held_proxy"] >= 0.0
            and delta["F_retained"] >= 0.0
        )
        comparisons[name] = {"left": left, "right": right, "delta": delta}
    score = {
        "schema": SCORE_SCHEMA,
        "status": "ANALYZED",
        "row_count": matrix.ROW_COUNT,
        "k5_row_count": matrix.FOLD_COUNT,
        "arm_metrics": by_arm,
        "comparisons": comparisons,
        "candidate_promotable": all(
            value["delta"]["passes_frozen_gate"] for value in comparisons.values()
        ),
        "predictions_sha256": _sha_file(root / "predictions.npz"),
        "truth_side_sha256": _sha_file(root / "truth_side.npz"),
        "manifest_sha256": _sha_file(root / "manifest.json"),
    }
    score_dir = root / "score"
    if score_dir.exists():
        raise NextR1Proxy84Error("score output overwrite refused")
    score_dir.mkdir()
    _write_json_new(score_dir / "scores.json", score)
    print(json.dumps(score, sort_keys=True), flush=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    predict = sub.add_parser("predict")
    predict.add_argument("--run-id", required=True)
    predict.add_argument("--run-root", required=True, type=Path)
    predict.add_argument("--checkpoint", required=True, type=Path)
    predict.add_argument("--checkpoint-sha256", required=True)
    predict.add_argument("--selected-iq", required=True, type=Path)
    predict.add_argument("--selected-iq-sha256", required=True)
    predict.add_argument("--selected-receipt", required=True, type=Path)
    predict.add_argument("--selected-receipt-sha256", required=True)
    predict.add_argument("--ls-join", required=True, type=Path)
    predict.add_argument("--ls-join-sha256", required=True)
    predict.add_argument("--device", default="cuda:0")
    predict.add_argument("--microbatch-size", type=int, default=8)
    predict.set_defaults(func=run_predict)
    score = sub.add_parser("score")
    score.add_argument("--run-root", required=True, type=Path)
    score.set_defaults(func=run_score)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if getattr(args, "microbatch_size", 1) < 1:
        raise NextR1Proxy84Error("microbatch size must be positive")
    args.func(args)


if __name__ == "__main__":
    main()
