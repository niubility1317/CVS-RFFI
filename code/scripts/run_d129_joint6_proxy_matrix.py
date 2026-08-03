#!/usr/bin/env python3
"""Prepare, predict, and independently score the frozen D129 proxy matrix."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np


SCRIPT_ROOT = Path(__file__).resolve().parent
CODE_ROOT = SCRIPT_ROOT.parent
for candidate in (str(SCRIPT_ROOT), str(CODE_ROOT)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from run_d106_rcmr_g0_one_shot import (  # noqa: E402
    _predecessor_locks,
    _read_pinned_archive,
)
from run_d129_joint6_real_archive_smoke import (  # noqa: E402
    _load_d104_ls_rows,
    _ordered_cell_indices,
    _read_pinned_json,
    _require_sha256,
)
from cvsrffi import stage2_d129_joint6_da as da  # noqa: E402
from cvsrffi import stage2_d129_joint6_heads as heads  # noqa: E402
from cvsrffi import stage2_d129_joint6_matrix as matrix  # noqa: E402
from cvsrffi import stage2_d129_joint6_runtime as runtime  # noqa: E402
from cvsrffi import stage2_d129_joint6_scorer as scorer  # noqa: E402
from cvsrffi.stage2_zid_student_t_qknn import Phase1ZIDStudentTLock  # noqa: E402


PACKAGE_SCHEMA = "cvs.stage2.d129.joint6.proxy_package.v1"
PREPARE_SCHEMA = "cvs.stage2.d129.joint6.proxy_prepare.v1"
RESOURCE_SCHEMA = "cvs.stage2.d129.joint6.proxy_resources.v1"


class D129ProxyMatrixError(ValueError):
    """Raised when the immutable proxy-matrix closure drifts."""


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        _plain(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _new_dir(path: Path) -> Path:
    if not path.is_absolute() or path.exists() or not path.parent.is_dir():
        raise D129ProxyMatrixError(
            "output directory must be a new absolute child of an existing directory"
        )
    path.mkdir()
    return path


def _write_json_new(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(_plain(value), handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")


def _load_json(path: Path, expected_sha256: str, name: str) -> Mapping[str, Any]:
    if not path.is_absolute() or not path.is_file() or path.is_symlink():
        raise D129ProxyMatrixError(f"{name} must be an absolute regular file")
    if _sha256_file(path) != _require_sha256(expected_sha256, f"{name} SHA256"):
        raise D129ProxyMatrixError(f"{name} SHA256 mismatch")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise D129ProxyMatrixError(f"{name} must contain a JSON mapping")
    return value


def prepare_proxy_matrix(
    *,
    archive_path: Path,
    archive_sha256: str,
    fixture_path: Path,
    fixture_sha256: str,
    checkpoint_sha256: str,
    method_lock_path: Path,
    method_lock_sha256: str,
    capsule_id: str,
    split_id: str,
    run_id: str,
    output_dir: Path,
) -> Mapping[str, Any]:
    """Open source labels only to build predictor packages and separate truth."""

    archive_sha256 = _require_sha256(archive_sha256, "archive SHA256")
    checkpoint_sha256 = _require_sha256(checkpoint_sha256, "checkpoint SHA256")
    method_lock_sha256 = _require_sha256(method_lock_sha256, "method-lock SHA256")
    fixture = _read_pinned_json(fixture_path, fixture_sha256)
    method_lock = _load_json(method_lock_path, method_lock_sha256, "method lock")
    if (
        fixture.get("schema") != "cvs.d106.real_integration_fixture.v1"
        or fixture.get("protocol_schema") != "p2_min_v1"
        or fixture.get("ls_archive_sha256") != archive_sha256
        or fixture.get("checkpoint_sha256") != checkpoint_sha256
        or method_lock.get("schema") != "cvs.stage2.d129.joint6.method_lock.v1"
        or method_lock.get("protocol_schema") != "p2_min_v1"
        or not capsule_id
        or not split_id
        or not run_id
    ):
        raise D129ProxyMatrixError("prepare provenance/method-lock drift")
    rows = _load_d104_ls_rows(
        _read_pinned_archive(archive_path, expected_sha256=archive_sha256),
        archive_sha256=archive_sha256,
    )
    receivers_by_row = tuple(str(value) for value in rows.receiver_ids.tolist())
    classes_by_row = tuple(str(value) for value in rows.tx_labels.tolist())
    physical_ids = tuple(str(value) for value in rows.physical_ids.tolist())
    receivers = tuple(sorted(set(receivers_by_row)))
    classes = tuple(sorted(set(classes_by_row)))
    plan = matrix.build_joint6_loco_plan(receivers, classes)
    cell_indices = {
        (receiver, class_id): _ordered_cell_indices(
            receiver=receiver,
            class_id=class_id,
            receiver_ids=receivers_by_row,
            class_ids=classes_by_row,
            physical_ids=physical_ids,
        )
        for receiver in receivers
        for class_id in classes
    }
    loco = da.build_d129_loco_plan(
        da.D129LOCORecord(receiver, class_id, physical_ids[index])
        for (receiver, class_id), indices in cell_indices.items()
        for index in indices
    )
    phase1_values: list[np.ndarray] = []
    support5_values: list[np.ndarray] = []
    query_values: list[np.ndarray] = []
    support5_ids_values: list[tuple[str, ...]] = []
    query_ids_values: list[tuple[str, ...]] = []
    support_labels_values: list[tuple[str, ...]] = []
    registry_values: list[tuple[str, ...]] = []
    binding_values: list[str] = []
    held_receivers: list[str] = []
    held_classes: list[str] = []
    truth: dict[str, str] = {}
    plan_by_key = {
        (value["held_receiver"], value["held_class"], value["active_k"]): value
        for value in plan["rows"]
    }
    for held_receiver in receivers:
        for held_class in classes:
            retained = tuple(value for value in classes if value != held_class)
            registry = retained + (held_class,)
            row1_payload = plan_by_key[(held_receiver, held_class, 1)]
            row5_payload = plan_by_key[(held_receiver, held_class, 5)]
            row1 = matrix.Joint6LocoRow(
                row_id=row1_payload["row_id"],
                held_receiver=held_receiver,
                held_class=held_class,
                active_k=1,
                retained_classes=retained,
                registered_classes=registry,
            )
            row5 = matrix.Joint6LocoRow(
                row_id=row5_payload["row_id"],
                held_receiver=held_receiver,
                held_class=held_class,
                active_k=5,
                retained_classes=retained,
                registered_classes=registry,
            )
            fold = next(
                value
                for value in loco.folds
                if value.held_receiver == held_receiver
                and value.held_class == held_class
            )
            phase1_indices = tuple(
                index
                for receiver in receivers
                if receiver != held_receiver
                for class_id in classes
                if class_id != held_class
                for index in cell_indices[(receiver, class_id)]
            )
            support5_by_class = {
                class_id: cell_indices[(held_receiver, class_id)][:5]
                for class_id in registry
            }
            support1_by_class = {
                class_id: values[:1] for class_id, values in support5_by_class.items()
            }
            query_by_class = {
                class_id: cell_indices[(held_receiver, class_id)][5:]
                for class_id in registry
            }
            binding = matrix.bind_joint6_physical_ids(
                row_k1=row1,
                row_k5=row5,
                loco_fold_receipt=fold.as_dict(),
                phase1_fit_ids=tuple(physical_ids[index] for index in phase1_indices),
                k1_support_ids_by_class={
                    class_id: tuple(physical_ids[index] for index in values)
                    for class_id, values in support1_by_class.items()
                },
                k5_support_ids_by_class={
                    class_id: tuple(physical_ids[index] for index in values)
                    for class_id, values in support5_by_class.items()
                },
                query_ids_by_class={
                    class_id: tuple(physical_ids[index] for index in values)
                    for class_id, values in query_by_class.items()
                },
            )
            phase1_values.append(
                np.stack(
                    [
                        np.stack(
                            [
                                rows.z_id[list(cell_indices[(receiver, class_id)])]
                                for class_id in retained
                            ]
                        )
                        for receiver in receivers
                        if receiver != held_receiver
                    ]
                ).astype(np.float32, copy=False)
            )
            support5_indices = tuple(
                index
                for class_id in registry
                for index in support5_by_class[class_id]
            )
            fold_query_indices = tuple(
                index for class_id in registry for index in query_by_class[class_id]
            )
            support5_values.append(rows.z_id[list(support5_indices)])
            query_values.append(rows.z_id[list(fold_query_indices)])
            support5_ids_values.append(
                tuple(physical_ids[index] for index in support5_indices)
            )
            query_ids = tuple(physical_ids[index] for index in fold_query_indices)
            query_ids_values.append(query_ids)
            support_labels_values.append(
                tuple(class_id for class_id in registry for _ in range(5))
            )
            registry_values.append(registry)
            binding_values.append(json.dumps(dict(binding), sort_keys=True))
            held_receivers.append(held_receiver)
            held_classes.append(held_class)
            for class_id in registry:
                for index in query_by_class[class_id]:
                    query_id = physical_ids[index]
                    if query_id in truth and truth[query_id] != class_id:
                        raise D129ProxyMatrixError("query truth catalog conflict")
                    truth[query_id] = class_id
    locks = {
        str(lock.active_k): asdict(lock)
        for lock in _predecessor_locks(rows)
        if lock.active_k in matrix.K_VALUES
    }
    query_catalog_root = _sha256_bytes(
        "\n".join(sorted(truth)).encode("utf-8")
    )
    header = {
        "schema": PACKAGE_SCHEMA,
        "run_id": run_id,
        "protocol_schema": "p2_min_v1",
        "capsule_id": capsule_id,
        "split_id": split_id,
        "archive_sha256": archive_sha256,
        "fixture_sha256": fixture_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "method_lock_sha256": method_lock_sha256,
        "matrix_sha256": plan["matrix_sha256"],
        "query_catalog_root_sha256": query_catalog_root,
        "fold_count": 42,
        "truth_in_predictor_package": False,
    }
    target = _new_dir(output_dir)
    package_path = target / "predictor_package.npz"
    np.savez(
        package_path,
        header_json=np.asarray(json.dumps(header, sort_keys=True)),
        phase1=np.asarray(phase1_values, dtype=np.float32),
        support5=np.asarray(support5_values, dtype=np.float32),
        query=np.asarray(query_values, dtype=np.float32),
        support5_ids=np.asarray(support5_ids_values),
        query_ids=np.asarray(query_ids_values),
        support_labels=np.asarray(support_labels_values),
        registry=np.asarray(registry_values),
        bindings_json=np.asarray(binding_values),
        held_receivers=np.asarray(held_receivers),
        held_classes=np.asarray(held_classes),
        locks_json=np.asarray(json.dumps(locks, sort_keys=True)),
    )
    truth_path = target / "truth.json"
    plan_path = target / "plan.json"
    _write_json_new(truth_path, truth)
    _write_json_new(plan_path, dict(plan))
    receipt = {
        "schema": PREPARE_SCHEMA,
        **header,
        "package_sha256": _sha256_file(package_path),
        "truth_sha256": _sha256_file(truth_path),
        "plan_sha256": _sha256_file(plan_path),
        "package_members": [
            "header_json",
            "phase1",
            "support5",
            "query",
            "support5_ids",
            "query_ids",
            "support_labels",
            "registry",
            "bindings_json",
            "held_receivers",
            "held_classes",
            "locks_json",
        ],
        "query_truth_in_predictor_package": False,
        "data_revalidated": False,
    }
    _write_json_new(target / "prepare_receipt.json", receipt)
    return receipt


def _load_package(path: Path, expected_sha256: str) -> dict[str, np.ndarray]:
    if not path.is_absolute() or not path.is_file() or path.is_symlink():
        raise D129ProxyMatrixError("package must be an absolute regular file")
    if _sha256_file(path) != _require_sha256(expected_sha256, "package SHA256"):
        raise D129ProxyMatrixError("package SHA256 mismatch")
    with np.load(path, allow_pickle=False) as loaded:
        names = tuple(loaded.files)
        if any("truth" in name.lower() or "query_label" in name.lower() for name in names):
            raise D129ProxyMatrixError("predictor package contains a truth-like member")
        return {name: np.asarray(loaded[name]) for name in names}


def predict_proxy_matrix(
    *, package_path: Path, package_sha256: str, output_dir: Path
) -> Mapping[str, Any]:
    """Generate all 168 rows without importing or opening the truth artifact."""

    package = _load_package(package_path, package_sha256)
    header = json.loads(str(package["header_json"].item()))
    if (
        header.get("schema") != PACKAGE_SCHEMA
        or header.get("protocol_schema") != "p2_min_v1"
        or header.get("truth_in_predictor_package") is not False
        or package["phase1"].shape != (42, 6, 5, 14, 160)
        or package["support5"].shape != (42, 30, 160)
        or package["query"].shape != (42, 54, 160)
    ):
        raise D129ProxyMatrixError("predictor package shape/header drift")
    locks_payload = json.loads(str(package["locks_json"].item()))
    locks = {
        int(key): Phase1ZIDStudentTLock(**value)
        for key, value in locks_payload.items()
    }
    output_rows: list[Mapping[str, Any]] = []
    resources: list[Mapping[str, Any]] = []
    for fold_index in range(42):
        registry = tuple(str(value) for value in package["registry"][fold_index])
        held_receiver = str(package["held_receivers"][fold_index])
        held_class = str(package["held_classes"][fold_index])
        retained = registry[:-1]
        binding = json.loads(str(package["bindings_json"][fold_index]))
        assets = da.build_d129_phase1_assets(
            np.ascontiguousarray(package["phase1"][fold_index], dtype=np.float32),
            checkpoint_sha256=header["checkpoint_sha256"],
            phase1_seal_sha256=binding["phase1_seal_sha256"],
        )
        query = np.ascontiguousarray(package["query"][fold_index], dtype=np.float32)
        query_ids = tuple(str(value) for value in package["query_ids"][fold_index])
        support5 = np.ascontiguousarray(
            package["support5"][fold_index], dtype=np.float32
        )
        support5_ids = tuple(
            str(value) for value in package["support5_ids"][fold_index]
        )
        support5_labels = tuple(
            str(value) for value in package["support_labels"][fold_index]
        )
        for active_k in matrix.K_VALUES:
            if active_k == 1:
                indices = np.asarray(
                    [class_index * 5 for class_index in range(6)], dtype=np.int64
                )
                support = np.ascontiguousarray(support5[indices])
                support_ids = tuple(support5_ids[index] for index in indices)
                support_labels = tuple(support5_labels[index] for index in indices)
            else:
                support = support5
                support_ids = support5_ids
                support_labels = support5_labels
            row = matrix.Joint6LocoRow(
                row_id=f"rx={held_receiver}|held={held_class}|K={active_k}",
                held_receiver=held_receiver,
                held_class=held_class,
                active_k=active_k,
                retained_classes=retained,
                registered_classes=registry,
            )
            common = heads.build_d129_common_r0(
                base_support_zid=support,
                base_query_zid=query,
                support_labels=support_labels,
                registered_classes=registry,
                old_class_count=5,
                partition_semantics="phase1_seen_class_loco_directional_proxy",
                opaque_query_ids=query_ids,
                qknn_lock=locks[active_k],
            )
            for asset in assets:
                result = runtime.run_d129_candidate_joint6(
                    asset=asset,
                    base_support_zid160=support,
                    base_query_zid160=query,
                    support_labels=support_labels,
                    support_physical_ids=support_ids,
                    registered_classes=registry,
                    retained_class_count=5,
                    opaque_query_ids=query_ids,
                    qknn_lock=locks[active_k],
                    fold_binding=binding,
                    common_r0=common,
                )
                output_rows.append(runtime.build_joint6_prediction_row(row, result))
                resources.append(
                    {
                        "candidate_id": result.candidate_id,
                        "row_id": row.row_id,
                        "smoke_receipt": dict(result.smoke_receipt),
                        "runtime_receipt": dict(result.runtime_receipt),
                        "head_causal_resource_receipt": dict(
                            result.six_arm.head_causal_resource_receipt
                        ),
                        "system_formal_replacement_resource_receipt": dict(
                            result.six_arm.system_formal_replacement_resource_receipt
                        ),
                    }
                )
    prediction = {
        "schema": scorer.PREDICTION_SCHEMA,
        "protocol_schema": "p2_min_v1",
        "capsule_id": header["capsule_id"],
        "split_id": header["split_id"],
        "checkpoint_sha256": header["checkpoint_sha256"],
        "archive_sha256": header["archive_sha256"],
        "method_lock_sha256": header["method_lock_sha256"],
        "query_catalog_root_sha256": header["query_catalog_root_sha256"],
        "matrix_sha256": header["matrix_sha256"],
        "candidate_ids": list(matrix.CANDIDATE_IDS),
        "arm_ids": list(matrix.ARM_IDS),
        "rows_complete": len(output_rows) == 168,
        "truth_loaded": False,
        "query_rows_used_for_fit": 0,
        "query_state_updates": 0,
        "query_selection_count": 0,
        "rows": output_rows,
    }
    target = _new_dir(output_dir)
    prediction_path = target / "predictions.json"
    resources_path = target / "resources.json"
    _write_json_new(prediction_path, prediction)
    _write_json_new(
        resources_path,
        {
            "schema": RESOURCE_SCHEMA,
            "row_count": len(resources),
            "formal_efficiency_thresholds_evaluated": False,
            "rows": resources,
        },
    )
    return {
        "prediction_sha256": _sha256_file(prediction_path),
        "resources_sha256": _sha256_file(resources_path),
        "prediction_row_count": len(output_rows),
        "rows_complete": prediction["rows_complete"],
        "truth_loaded": False,
    }


def score_proxy_matrix(
    *,
    prediction_path: Path,
    prediction_sha256: str,
    plan_path: Path,
    plan_sha256: str,
    truth_path: Path,
    truth_sha256: str,
    output_path: Path,
) -> Mapping[str, Any]:
    """Open truth only after complete immutable prediction closure."""

    prediction = _load_json(prediction_path, prediction_sha256, "prediction")
    plan = _load_json(plan_path, plan_sha256, "plan")
    truth = _load_json(truth_path, truth_sha256, "truth")
    result = scorer.score_joint6_screen(
        prediction=prediction, plan=plan, truth_by_query_id=truth
    )
    if not output_path.is_absolute() or output_path.exists() or not output_path.parent.is_dir():
        raise D129ProxyMatrixError("score output must be a new absolute file")
    _write_json_new(output_path, result)
    return {
        "score_sha256": _sha256_file(output_path),
        "candidate_count": len(result["candidate_scores"]),
        "truth_opened_after_complete_prediction": result[
            "truth_opened_after_complete_prediction"
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--archive", type=Path, required=True)
    prepare.add_argument("--archive-sha256", required=True)
    prepare.add_argument("--fixture", type=Path, required=True)
    prepare.add_argument("--fixture-sha256", required=True)
    prepare.add_argument("--checkpoint-sha256", required=True)
    prepare.add_argument("--method-lock", type=Path, required=True)
    prepare.add_argument("--method-lock-sha256", required=True)
    prepare.add_argument("--capsule-id", required=True)
    prepare.add_argument("--split-id", required=True)
    prepare.add_argument("--run-id", required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)
    predict = commands.add_parser("predict")
    predict.add_argument("--package", type=Path, required=True)
    predict.add_argument("--package-sha256", required=True)
    predict.add_argument("--output-dir", type=Path, required=True)
    score = commands.add_parser("score")
    score.add_argument("--prediction", type=Path, required=True)
    score.add_argument("--prediction-sha256", required=True)
    score.add_argument("--plan", type=Path, required=True)
    score.add_argument("--plan-sha256", required=True)
    score.add_argument("--truth", type=Path, required=True)
    score.add_argument("--truth-sha256", required=True)
    score.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare_proxy_matrix(
            archive_path=args.archive,
            archive_sha256=args.archive_sha256,
            fixture_path=args.fixture,
            fixture_sha256=args.fixture_sha256,
            checkpoint_sha256=args.checkpoint_sha256,
            method_lock_path=args.method_lock,
            method_lock_sha256=args.method_lock_sha256,
            capsule_id=args.capsule_id,
            split_id=args.split_id,
            run_id=args.run_id,
            output_dir=args.output_dir,
        )
    elif args.command == "predict":
        result = predict_proxy_matrix(
            package_path=args.package,
            package_sha256=args.package_sha256,
            output_dir=args.output_dir,
        )
    else:
        result = score_proxy_matrix(
            prediction_path=args.prediction,
            prediction_sha256=args.prediction_sha256,
            plan_path=args.plan,
            plan_sha256=args.plan_sha256,
            truth_path=args.truth,
            truth_sha256=args.truth_sha256,
            output_path=args.output,
        )
    print(json.dumps(_plain(result), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
