#!/usr/bin/env python3
"""Prepare D104 truth-separated packages and frozen TX-probe receipt."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cvsrffi.rxid_metabias4_held_execution import (  # noqa: E402
    build_receiver_package_indices,
    canonical_sha256,
    compile_teacher_bundle,
    encode_domain_rows,
    package_id,
    sha256_file,
    validate_teacher_fit_manifest,
)
from cvsrffi.rxid_metabias4_held_falsifier import (  # noqa: E402
    build_complete_fit_plan,
    run_fixed_tx_probe,
)
from cvsrffi.stage2_d104_source_split import (  # noqa: E402
    CANDIDATE_ID,
    SPLIT_ID,
)


SCORER_KEYS = {
    "z_id",
    "z_dom",
    "pre_relu",
    "labels",
    "receiver_ids",
    "day_ids",
    "physical_ids",
    "scenario_names",
    "observation_ids",
    "class_ids",
}
TEACHER_KEYS = {"U", "B", "bank_g", "bank_t", "bank_precision", "bank_sigma"}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write_new(path: Path, value: Any) -> None:
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


def _load_fit(
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
    fit_root = fits_root / fit_id
    manifest_path = fit_root / "fit_complete.json"
    teacher_path = fit_root / "teacher_arrays_fp32_ground_only.npz"
    manifest = _read_json(manifest_path)
    teacher_sha = sha256_file(teacher_path)
    if (
        manifest.get("completed_meta_steps") != 400
        or manifest.get("target_access") is not False
        or manifest.get("formal_query_access") is not False
        or manifest.get("source_val_rows_used_for_training") != 0
        or manifest.get("teacher_archive", {}).get("sha256") != teacher_sha
    ):
        raise ValueError(f"incomplete or drifted D104 fit: {fit_id}")
    with np.load(teacher_path, allow_pickle=False) as archive:
        if set(archive.files) != TEACHER_KEYS:
            raise ValueError(f"D104 teacher member closure drift: {fit_id}")
        teacher = {name: np.array(archive[name], copy=True) for name in archive.files}
    validate_teacher_fit_manifest(
        manifest,
        teacher,
        expected_outer_spec=expected_outer_spec,
        checkpoint_sha256=checkpoint_sha,
        runtime_sha256=runtime_sha,
        teacher_archive_sha256=teacher_sha,
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
    parser.add_argument("--source-val-archive", type=Path, required=True)
    parser.add_argument("--source-val-manifest", type=Path, required=True)
    parser.add_argument("--fits-root", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--runtime-sha256", required=True)
    parser.add_argument("--method-lock-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    archive_path = args.source_val_archive.resolve(strict=True)
    manifest_path = args.source_val_manifest.resolve(strict=True)
    fits_root = args.fits_root.resolve(strict=True)
    output = args.output_dir.resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"immutable D104 package root exists: {output}")
    manifest = _read_json(manifest_path)
    if (
        manifest.get("candidate_id") != CANDIDATE_ID
        or manifest.get("split_id") != SPLIT_ID
        or manifest.get("role") != "source_val_scorer_only"
        or manifest.get("archive", {}).get("sha256") != sha256_file(archive_path)
        or manifest.get("asset_access") is not False
        or manifest.get("gradient_access") is not False
        or manifest.get("target_access") is not False
        or manifest.get("formal_query_access") is not False
    ):
        raise ValueError("D104 source-val scorer manifest drift")
    with np.load(archive_path, allow_pickle=False) as archive:
        if set(archive.files) != SCORER_KEYS:
            raise ValueError("D104 source-val member closure drift")
        arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
    receivers = tuple(sorted(set(arrays["receiver_ids"].astype(str).tolist())))
    classes = tuple(sorted(set(arrays["labels"].astype(str).tolist())))
    days = tuple(sorted(set(arrays["day_ids"].astype(str).tolist())))
    if len(receivers) != 7 or len(classes) != 6 or len(days) != 4:
        raise ValueError("D104 source-val registry drift")
    plan = build_complete_fit_plan(receivers, classes, days)
    outer_by_receiver = {
        spec.held_receiver: spec
        for spec in plan
        if spec.fold_kind == "receiver_outer" and spec.fit_stage == "outer"
    }
    provisional_tx_sha = canonical_sha256(
        {"schema": "d104_r1_tx_probe_provisional", "persisted": False}
    )
    probe_rows = []
    for receiver in receivers:
        spec = outer_by_receiver[receiver]
        bundle = _load_fit(
            fits_root,
            spec.fit_id,
            checkpoint_sha=args.checkpoint_sha256,
            runtime_sha=args.runtime_sha256,
            method_lock_sha=args.method_lock_sha256,
            tx_receipt_sha=provisional_tx_sha,
            tx_mean=0.0,
            tx_max=0.0,
            expected_outer_spec={
                "held_receiver": receiver,
                "held_day": None,
                "held_class": None,
            },
        )
        local = arrays["receiver_ids"].astype(str) == receiver
        probe_rows.append(
            run_fixed_tx_probe(
                encode_domain_rows(
                    bundle,
                    np.asarray(arrays["z_dom"][local], dtype=np.float32),
                ),
                arrays["receiver_ids"][local],
                arrays["day_ids"][local],
                arrays["labels"][local],
                arrays["physical_ids"][local],
            )
        )
    tx_receipt = {
        "schema": "cvs.d104_r1.rxid_angq.tx_probe.v1",
        "split_id": SPLIT_ID,
        "fold_count": 7,
        "folds": probe_rows,
        "mean_fold_score": float(np.mean([row["fold_score"] for row in probe_rows])),
        "max_fold_score": float(np.max([row["fold_score"] for row in probe_rows])),
        "asset_frozen_before_probe": True,
        "probe_state_returned_to_asset": False,
        "target_access": False,
    }
    tx_receipt["receipt_sha256"] = canonical_sha256(tx_receipt)

    output.mkdir(parents=True, exist_ok=False)
    packages_root = output / "predictor_packages"
    packages_root.mkdir()
    package_rows = []
    truth_rows = []
    for receiver in receivers:
        for k_shot in (1, 5, 10):
            support, query = build_receiver_package_indices(
                arrays["receiver_ids"],
                arrays["labels"],
                arrays["physical_ids"],
                held_receiver=receiver,
                registered_classes=classes,
                k_shot=k_shot,
            )
            identity = package_id(receiver, k_shot)
            package_path = packages_root / f"{identity}.npz"
            with package_path.open("xb") as stream:
                np.savez(
                    stream,
                    support_pre_relu=np.asarray(
                        arrays["pre_relu"][support], dtype=np.float32
                    ),
                    support_zdom=np.asarray(
                        arrays["z_dom"][support], dtype=np.float32
                    ),
                    support_labels=arrays["labels"][support].astype(str),
                    support_physical_ids=arrays["physical_ids"][support].astype(str),
                    query_pre_relu=np.asarray(
                        arrays["pre_relu"][query], dtype=np.float32
                    ),
                    query_physical_ids=arrays["physical_ids"][query].astype(str),
                    registered_classes=np.asarray(classes, dtype=str),
                )
            package_rows.append(
                {
                    "package_id": identity,
                    "held_receiver": receiver,
                    "K": k_shot,
                    "path": str(Path("predictor_packages") / package_path.name),
                    "sha256": sha256_file(package_path),
                    "support_physical_id_root_sha256": canonical_sha256(
                        arrays["physical_ids"][support].astype(str).tolist()
                    ),
                    "query_physical_id_root_sha256": canonical_sha256(
                        arrays["physical_ids"][query].astype(str).tolist()
                    ),
                    "query_truth_present": False,
                    "support_query_physical_disjoint": True,
                }
            )
            truth_rows.append(
                {
                    "package_id": identity,
                    "query_physical_ids": (
                        arrays["physical_ids"][query].astype(str).tolist()
                    ),
                    "query_truth_labels": (
                        arrays["labels"][query].astype(str).tolist()
                    ),
                }
            )
    package_manifest = {
        "schema": "cvs.d104_r1.rxid_angq.held_packages.v2",
        "candidate_id": CANDIDATE_ID,
        "split_id": SPLIT_ID,
        "receiver_ids": list(receivers),
        "class_ids": list(classes),
        "day_ids": list(days),
        "package_count": len(package_rows),
        "packages": package_rows,
        "query_truth_present": False,
        "target_access": False,
        "formal_query_state_updates": 0,
    }
    truth = {
        "schema": "cvs.d104_r1.rxid_angq.held_truth.v2",
        "split_id": SPLIT_ID,
        "package_count": len(truth_rows),
        "packages": truth_rows,
        "predictor_access": False,
    }
    _write_new(output / "tx_probe_receipt.json", tx_receipt)
    _write_new(output / "package_manifest.json", package_manifest)
    scorer_root = output / "scorer_only"
    scorer_root.mkdir()
    _write_new(scorer_root / "truth.json", truth)
    print(output / "package_manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
