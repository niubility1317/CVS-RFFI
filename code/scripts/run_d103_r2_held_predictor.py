#!/usr/bin/env python3
"""Run all 63 matched M0/D102/D103 rows without any truth-side argument."""

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

from cvsrffi.phase1_rb_metabias4_bundle import (  # noqa: E402
    load_phase1_rb_metabias4_bundle,
)
from cvsrffi.rxid_metabias4_held_execution import (  # noqa: E402
    compile_teacher_bundle,
    package_id,
    predict_matched_row,
    sha256_file,
    validate_teacher_fit_manifest,
)
from cvsrffi.rxid_metabias4_held_falsifier import (  # noqa: E402
    build_complete_fit_plan,
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
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


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
        or manifest.get("teacher_archive", {}).get("sha256")
        != teacher_archive_sha
    ):
        raise ValueError(f"fit closure drift: {fit_id}")
    with np.load(teacher_path, allow_pickle=False) as archive:
        if set(archive.files) != TEACHER_KEYS:
            raise ValueError(f"teacher closure drift: {fit_id}")
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
    parser.add_argument("--d102-root", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--runtime-sha256", required=True)
    parser.add_argument("--method-lock-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    package_root = args.package_root.resolve(strict=True)
    fits_root = args.fits_root.resolve(strict=True)
    d102_root = args.d102_root.resolve(strict=True)
    output = args.output_dir.resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"immutable prediction root exists: {output}")
    package_manifest = _read_json(package_root / "package_manifest.json")
    tx_path = package_root / "tx_probe_receipt.json"
    tx = _read_json(tx_path)
    provenance = _read_json(d102_root / "d102_provenance.json")
    receivers = tuple(package_manifest["receiver_ids"])
    classes = tuple(package_manifest["class_ids"])
    days = tuple(package_manifest["day_ids"])
    if (
        package_manifest.get("package_count") != 21
        or package_manifest.get("query_truth_present") is not False
        or tx.get("fold_count") != 7
        or provenance.get("fold_count") != 49
    ):
        raise ValueError("held predictor input coverage drift")
    package_by_key = {
        (row["held_receiver"], int(row["K"])): row
        for row in package_manifest["packages"]
    }
    d102_by_key = {
        (row["held_receiver"], row.get("held_class")): row
        for row in provenance["folds"]
    }
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
    artifact_root = output / "rows"
    artifact_root.mkdir()
    artifact_rows = []
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
    tx_receipt_sha = sha256_file(tx_path)
    for receiver, held_class, k_shot in row_specs:
        package_row = package_by_key[(receiver, k_shot)]
        package_path = package_root / package_row["path"]
        if sha256_file(package_path) != package_row["sha256"]:
            raise ValueError("predictor package SHA drift")
        with np.load(package_path, allow_pickle=False) as archive:
            if set(archive.files) != PACKAGE_KEYS:
                raise ValueError("predictor package member closure drift")
            package = {name: np.array(archive[name], copy=True) for name in archive.files}
        d102_row = d102_by_key[(receiver, held_class)]
        d102_bundle = load_phase1_rb_metabias4_bundle(
            d102_root / d102_row["bundle_relative_path"],
            expected_checkpoint_sha256=args.checkpoint_sha256,
            expected_runtime_sha256=args.runtime_sha256,
        )
        if d102_bundle.content_root_sha256 != d102_row["bundle_content_root_sha256"]:
            raise ValueError("D102 fold content root drift")
        outer_bundle = _load_bundle(
            fits_root,
            outer[(receiver, held_class)].fit_id,
            checkpoint_sha=args.checkpoint_sha256,
            runtime_sha=args.runtime_sha256,
            method_lock_sha=args.method_lock_sha256,
            tx_receipt_sha=tx_receipt_sha,
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
                tx_receipt_sha=tx_receipt_sha,
                tx_mean=float(tx["mean_fold_score"]),
                tx_max=float(tx["max_fold_score"]),
                expected_outer_spec={
                    "held_receiver": receiver,
                    "held_day": spec.held_day,
                    "held_class": held_class,
                },
            )
            for spec in sorted(day_specs[(receiver, held_class)], key=lambda row: row.held_day or "")
        ]
        artifact, stability = predict_matched_row(
            held_receiver=receiver,
            held_class=held_class,
            k_shot=k_shot,
            support_pre_relu=package["support_pre_relu"],
            support_zdom=package["support_zdom"],
            support_labels=package["support_labels"].astype(str),
            query_pre_relu=package["query_pre_relu"],
            query_physical_ids=package["query_physical_ids"].astype(str),
            registered_classes=package["registered_classes"].astype(str),
            d102_bundle=d102_bundle,
            d103_outer_bundle=outer_bundle,
            d103_day_bundles=day_bundles,
        )
        row_id = package_id(f"{receiver}\0{held_class or ''}", k_shot)
        artifact_path = artifact_root / f"{row_id}.json"
        _write_json_new(artifact_path, artifact)
        artifact_rows.append(
            {
                "held_receiver": receiver,
                "held_class": held_class,
                "K": k_shot,
                "package_id": package_id(receiver, k_shot),
                "path": str(Path("rows") / artifact_path.name),
                "sha256": sha256_file(artifact_path),
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
    prediction_manifest = {
        "schema": "cvs.d103_r2.rxid_crossreceiver.held_predictions.v1",
        "row_count": len(artifact_rows),
        "rows": artifact_rows,
        "day_stability_rows": stability_rows,
        "query_truth_access": False,
        "target_access": False,
        "formal_query_access": False,
        "sealed_at_unix_ns": time.time_ns(),
    }
    if len(artifact_rows) != 63 or len(stability_rows) != 49:
        raise RuntimeError("held predictor coverage did not close at 63/49")
    _write_json_new(output / "prediction_manifest.json", prediction_manifest)
    print(output / "prediction_manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
