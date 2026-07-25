#!/usr/bin/env python3
"""Run all 63 D104 rows and seal exactly 252 arm-row predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cvsrffi.rxid_metabias4_held_execution import (  # noqa: E402
    canonical_sha256,
    compile_teacher_bundle,
    package_id,
    sha256_file,
    validate_teacher_fit_manifest,
)
from cvsrffi.rxid_metabias4_held_falsifier import (  # noqa: E402
    build_complete_fit_plan,
)
from cvsrffi.stage2_d104_held_execution import (  # noqa: E402
    predict_d104_matched_row,
)
from cvsrffi.stage2_d104_rxid_angq import ARMS  # noqa: E402
from cvsrffi.stage2_d104_source_split import (  # noqa: E402
    CANDIDATE_ID,
    SPLIT_ID,
)


PACKAGE_KEYS = {
    "support_pre_relu",
    "support_zdom",
    "support_labels",
    "support_physical_ids",
    "query_pre_relu",
    "query_physical_ids",
    "registered_classes",
}
TEACHER_KEYS = {"U", "B", "bank_g", "bank_t", "bank_precision", "bank_sigma"}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write_json_new(path: Path, value: Any) -> None:
    payload = (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(payload)


def _load_bundle(
    fits_root: Path,
    fit_id: str,
    *,
    checkpoint_sha: str,
    runtime_sha: str,
    method_lock_sha: str,
    tx_receipt_sha: str,
    tx_mean: float,
    tx_max: float,
    expected_outer_spec: dict[str, str | None],
):
    root = fits_root / fit_id
    manifest_path = root / "fit_complete.json"
    teacher_path = root / "teacher_arrays_fp32_ground_only.npz"
    manifest = _read_json(manifest_path)
    teacher_archive_sha = sha256_file(teacher_path)
    if (
        manifest.get("completed_meta_steps") != 400
        or manifest.get("target_access") is not False
        or manifest.get("formal_query_access") is not False
        or manifest.get("source_val_rows_used_for_training") != 0
        or manifest.get("teacher_archive", {}).get("sha256")
        != teacher_archive_sha
    ):
        raise ValueError(f"D104 fit closure drift: {fit_id}")
    with np.load(teacher_path, allow_pickle=False) as archive:
        if set(archive.files) != TEACHER_KEYS:
            raise ValueError(f"D104 teacher closure drift: {fit_id}")
        teacher = {name: np.array(archive[name], copy=True) for name in archive.files}
    validate_teacher_fit_manifest(
        manifest,
        teacher,
        expected_outer_spec=expected_outer_spec,
        checkpoint_sha256=checkpoint_sha,
        runtime_sha256=runtime_sha,
        teacher_archive_sha256=teacher_archive_sha,
    )
    return compile_teacher_bundle(
        teacher,
        manifest,
        checkpoint_sha256=checkpoint_sha,
        runtime_sha256=runtime_sha,
        method_lock_sha256=method_lock_sha,
        training_receipt_sha256=sha256_file(manifest_path),
        tx_probe_receipt_sha256=tx_receipt_sha,
        tx_probe_mean=tx_mean,
        tx_probe_max=tx_max,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--fits-root", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--runtime-sha256", required=True)
    parser.add_argument("--method-lock-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    package_root = args.package_root.resolve(strict=True)
    fits_root = args.fits_root.resolve(strict=True)
    output = args.output_dir.resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"immutable D104 prediction root exists: {output}")
    package_manifest = _read_json(package_root / "package_manifest.json")
    package_manifest_path = package_root / "package_manifest.json"
    package_manifest_sha = sha256_file(package_manifest_path)
    tx_path = package_root / "tx_probe_receipt.json"
    tx = _read_json(tx_path)
    receivers = tuple(package_manifest["receiver_ids"])
    classes = tuple(package_manifest["class_ids"])
    days = tuple(package_manifest["day_ids"])
    if (
        set(package_manifest)
        != {
            "schema",
            "candidate_id",
            "split_id",
            "receiver_ids",
            "class_ids",
            "day_ids",
            "package_count",
            "packages",
            "registered_class_root_sha256",
            "source_val_scorer_manifest_sha256",
            "source_val_scorer_archive_sha256",
            "truth_input_seal_sha256",
            "query_truth_present",
            "target_access",
            "formal_query_state_updates",
        }
        or package_manifest.get("schema")
        != "cvs.d104_r1.rxid_angq.held_packages.v2"
        or package_manifest.get("candidate_id") != CANDIDATE_ID
        or package_manifest.get("split_id") != SPLIT_ID
        or package_manifest.get("package_count") != 21
        or not isinstance(package_manifest.get("packages"), list)
        or len(package_manifest["packages"]) != 21
        or package_manifest.get("query_truth_present") is not False
        or package_manifest.get("target_access") is not False
        or package_manifest.get("formal_query_state_updates") != 0
        or tx.get("fold_count") != 7
        or len(receivers) != 7
        or len(classes) != 6
        or len(days) != 4
        or canonical_sha256(list(classes))
        != package_manifest.get("registered_class_root_sha256")
    ):
        raise ValueError("D104 predictor input coverage drift")
    package_rows = package_manifest["packages"]
    if any(
        not isinstance(row, dict)
        or set(row)
        != {
            "package_id",
            "held_receiver",
            "K",
            "path",
            "sha256",
            "support_physical_id_root_sha256",
            "query_physical_id_root_sha256",
            "query_truth_present",
            "support_query_physical_disjoint",
        }
        for row in package_rows
    ):
        raise ValueError("D104 predictor package row closure drift")
    package_by_key = {
        (str(row["held_receiver"]), int(row["K"])): row
        for row in package_rows
    }
    expected_package_keys = {
        (receiver, k_shot)
        for receiver in receivers
        for k_shot in (1, 5, 10)
    }
    if (
        len(package_by_key) != 21
        or set(package_by_key) != expected_package_keys
        or len({str(row["package_id"]) for row in package_rows}) != 21
        or len({str(row["path"]) for row in package_rows}) != 21
    ):
        raise ValueError("D104 predictor package identity drift")
    plan = build_complete_fit_plan(receivers, classes, days)
    outer = {
        (spec.held_receiver, spec.held_class): spec
        for spec in plan
        if spec.fit_stage == "outer"
    }
    day_specs = {
        (receiver, held_class): [
            spec
            for spec in plan
            if spec.fit_stage == "leave_one_day"
            and spec.held_receiver == receiver
            and spec.held_class == held_class
        ]
        for receiver in receivers
        for held_class in (None, *classes)
    }
    output.mkdir(parents=True, exist_ok=False)
    row_root = output / "rows"
    row_root.mkdir()
    rows = []
    stability_rows = []
    row_specs = [
        (receiver, None, k_shot)
        for receiver in receivers
        for k_shot in (1, 5, 10)
    ] + [
        (receiver, class_id, 1)
        for receiver in receivers
        for class_id in classes
    ]
    tx_sha = sha256_file(tx_path)
    for receiver, held_class, k_shot in row_specs:
        package_row = package_by_key[(receiver, k_shot)]
        raw_package_path = Path(str(package_row["path"]))
        package_path = (package_root / raw_package_path).resolve(strict=True)
        if (
            raw_package_path.is_absolute()
            or not package_path.is_relative_to(package_root)
            or package_row["package_id"] != package_id(receiver, k_shot)
            or package_row["query_truth_present"] is not False
            or package_row["support_query_physical_disjoint"] is not True
            or sha256_file(package_path) != package_row["sha256"]
        ):
            raise ValueError("D104 package SHA drift")
        with np.load(package_path, allow_pickle=False) as archive:
            if set(archive.files) != PACKAGE_KEYS:
                raise ValueError("D104 package member closure drift")
            package = {name: np.array(archive[name], copy=True) for name in archive.files}
        support_ids = package["support_physical_ids"].astype(str).tolist()
        query_ids = package["query_physical_ids"].astype(str).tolist()
        if (
            package["registered_classes"].astype(str).tolist()
            != list(classes)
            or canonical_sha256(support_ids)
            != package_row["support_physical_id_root_sha256"]
            or canonical_sha256(query_ids)
            != package_row["query_physical_id_root_sha256"]
            or set(support_ids).intersection(query_ids)
        ):
            raise ValueError("D104 package registry/root/disjointness drift")
        outer_bundle = _load_bundle(
            fits_root,
            outer[(receiver, held_class)].fit_id,
            checkpoint_sha=args.checkpoint_sha256,
            runtime_sha=args.runtime_sha256,
            method_lock_sha=args.method_lock_sha256,
            tx_receipt_sha=tx_sha,
            tx_mean=float(tx["mean_fold_score"]),
            tx_max=float(tx["max_fold_score"]),
            expected_outer_spec={
                "held_receiver": receiver,
                "held_day": None,
                "held_class": held_class,
            },
        )
        day_bundles = [
            _load_bundle(
                fits_root,
                spec.fit_id,
                checkpoint_sha=args.checkpoint_sha256,
                runtime_sha=args.runtime_sha256,
                method_lock_sha=args.method_lock_sha256,
                tx_receipt_sha=tx_sha,
                tx_mean=float(tx["mean_fold_score"]),
                tx_max=float(tx["max_fold_score"]),
                expected_outer_spec={
                    "held_receiver": receiver,
                    "held_day": spec.held_day,
                    "held_class": held_class,
                },
            )
            for spec in sorted(
                day_specs[(receiver, held_class)],
                key=lambda value: value.held_day or "",
            )
        ]
        artifact, stability = predict_d104_matched_row(
            held_receiver=receiver,
            held_class=held_class,
            k_shot=k_shot,
            support_pre_relu=package["support_pre_relu"],
            support_zdom=package["support_zdom"],
            support_labels=package["support_labels"].astype(str),
            support_physical_ids=package["support_physical_ids"].astype(str),
            query_pre_relu=package["query_pre_relu"],
            query_physical_ids=package["query_physical_ids"].astype(str),
            registered_classes=package["registered_classes"].astype(str),
            d103_outer_bundle=outer_bundle,
            d103_day_bundles=day_bundles,
        )
        if (
            artifact["method_lock_sha256"] != args.method_lock_sha256
            or artifact["phase1_method_lock_sha256"]
            != args.method_lock_sha256
            or artifact["registered_classes"] != list(classes)
            or artifact["support_physical_id_root_sha256"]
            != package_row["support_physical_id_root_sha256"]
            or artifact["query_physical_id_root_sha256"]
            != package_row["query_physical_id_root_sha256"]
        ):
            raise ValueError("D104 row/package/method binding drift")
        row_id = package_id(f"{receiver}\0{held_class or ''}", k_shot)
        artifact_path = row_root / f"{row_id}.json"
        _write_json_new(artifact_path, artifact)
        rows.append(
            {
                "held_receiver": receiver,
                "held_class": held_class,
                "K": k_shot,
                "package_id": package_id(receiver, k_shot),
                "path": str(Path("rows") / artifact_path.name),
                "sha256": sha256_file(artifact_path),
                "arm_prediction_receipts": artifact["arm_prediction_receipts"],
                "prediction_receipt_sha256": artifact[
                    "prediction_receipt_sha256"
                ],
                "scorer_input_seal_sha256": artifact[
                    "scorer_input_seal_sha256"
                ],
                "method_lock_sha256": artifact["method_lock_sha256"],
                "registered_class_root_sha256": artifact[
                    "registered_class_root_sha256"
                ],
                "support_physical_id_root_sha256": artifact[
                    "support_physical_id_root_sha256"
                ],
                "query_physical_id_root_sha256": artifact[
                    "query_physical_id_root_sha256"
                ],
                "int8_gate_pass": bool(
                    artifact["int8_audit"]["passes_d104_int8_gate"]
                ),
            }
        )
        if stability is not None:
            stability_rows.append(
                {
                    "held_receiver": receiver,
                    "held_class": held_class,
                    **stability,
                }
            )
    arm_receipts = [
        receipt
        for row in rows
        for receipt in row["arm_prediction_receipts"].values()
    ]
    manifest = {
        "schema": "cvs.d104_r1.rxid_angq.held_predictions.v1",
        "split_id": SPLIT_ID,
        "row_count": len(rows),
        "arm_row_prediction_unit_count": len(arm_receipts),
        "rows": rows,
        "day_stability_rows": stability_rows,
        "package_manifest_sha256": package_manifest_sha,
        "truth_input_seal_sha256": package_manifest[
            "truth_input_seal_sha256"
        ],
        "method_lock_sha256": args.method_lock_sha256,
        "registered_classes": list(classes),
        "registered_class_root_sha256": canonical_sha256(list(classes)),
        "scorer_input_root_sha256": canonical_sha256(
            [row["scorer_input_seal_sha256"] for row in rows]
        ),
        "all_arm_prediction_receipts_unique": len(set(arm_receipts)) == 252,
        "query_truth_access": False,
        "target_access": False,
        "formal_query_state_updates": 0,
        "sealed_at_unix_ns": time.time_ns(),
    }
    if (
        len(rows) != 63
        or len(stability_rows) != 49
        or len(arm_receipts) != 252
        or len(set(arm_receipts)) != 252
        or any(set(row["arm_prediction_receipts"]) != set(ARMS) for row in rows)
    ):
        raise RuntimeError("D104 predictor coverage did not close at 63/252/49")
    _write_json_new(output / "prediction_manifest.json", manifest)
    print(output / "prediction_manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
