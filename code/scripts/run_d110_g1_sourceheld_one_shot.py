#!/usr/bin/env python3
"""Thin D110 adaptation of the existing D106 source-held one-shot path.

The common package writer, fixed 63-row matrix, independent truth scorer and
per-row metrics are reused from D106.  D110 adds only: the new unopened split
identity/1176-row closure, one pinned D106 588-tap geometry, and calls to the
frozen US-qKNN four-arm core.  Predict has no truth input; score delegates to
the established independent D106 scorer after the full prediction seal.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any, Iterator, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_d106_g1_sourceheld_one_shot as d106_g1  # noqa: E402
from scripts import run_d110_scpm_g0_one_shot as d110_g0  # noqa: E402
from cvsrffi import stage2_d110_usqknn as usqknn  # noqa: E402
from cvsrffi import stage2_zid_student_t_qknn as qknn  # noqa: E402
from cvsrffi.stage2_d110_sourceheld_split import (  # noqa: E402
    CANDIDATE_ID,
    EXPECTED_HELD_ROWS,
    GROUP_COUNT,
    HELD_PER_CELL,
    SCORER_MEMBERS,
    SCORER_SCHEMA,
    SPLIT_ID,
)


ARMS = usqknn.ARMS
K_VALUES = d106_g1.K_VALUES
PACKAGE_SCHEMA = "cvs.d110.scpm_usqknn.held_packages.v1"
PREDICTION_SCHEMA = "cvs.d110.scpm_usqknn.sourceheld.predictions.v1"
SCORE_SCHEMA = "cvs.d110.scpm_usqknn.sourceheld.scores.v1"
PACKAGE_KEYS = d106_g1.PACKAGE_KEYS
QUERY_PER_CLASS_BY_K = {1: 27, 5: 23, 10: 18}


class D110G1Error(ValueError):
    """Raised when the frozen D110 G1 adapter inputs or seal drift."""


def _sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _file_sha(path: Path) -> str:
    return d106_g1._file_sha(path)


def _new_run_id(value: str) -> str:
    if type(value) is not str or not value or len(value.encode("utf-8")) > 160:
        raise D110G1Error("run ID must be a short non-empty string")
    return value


@contextmanager
def _d110_d106_context() -> Iterator[None]:
    """Temporarily substitute only the identity/schema constants D106 shares."""

    replacement = {
        "D104_CANDIDATE_ID": CANDIDATE_ID,
        "SPLIT_ID": SPLIT_ID,
        "PACKAGE_SCHEMA": PACKAGE_SCHEMA,
        "PREDICTION_SCHEMA": PREDICTION_SCHEMA,
        "SCORE_SCHEMA": SCORE_SCHEMA,
    }
    previous = {name: getattr(d106_g1, name) for name in replacement}
    try:
        for name, value in replacement.items():
            setattr(d106_g1, name, value)
        yield
    finally:
        for name, value in previous.items():
            setattr(d106_g1, name, value)


def fixed_row_specs(
    receivers: Sequence[str], classes: Sequence[str]
) -> tuple[tuple[str, str | None, int], ...]:
    return d106_g1.fixed_row_specs(receivers, classes)


def _validate_new_source(args: argparse.Namespace) -> None:
    archive_path = args.source_val_archive.resolve(strict=True)
    manifest = d106_g1._read_json(args.source_val_manifest.resolve(strict=True))
    archive = manifest.get("archive")
    if (
        manifest.get("schema") != SCORER_SCHEMA
        or manifest.get("candidate_id") != CANDIDATE_ID
        or manifest.get("split_id") != SPLIT_ID
        or manifest.get("role") != "source_val_scorer_only"
        or not isinstance(archive, Mapping)
        or archive.get("sha256") != _file_sha(archive_path)
        or manifest.get("exact_member_allowlist") != list(SCORER_MEMBERS)
        or manifest.get("row_count") != EXPECTED_HELD_ROWS
        or any(
            manifest.get(name) is not False
            for name in (
                "asset_access",
                "gradient_access",
                "selection_access",
                "target_access",
                "formal_query_access",
                "performance_computed",
            )
        )
        or manifest.get("d106_prepare_member_compatible") is not True
    ):
        raise D110G1Error("D110 unopened source-held manifest drift")
    with np.load(archive_path, allow_pickle=False) as archive_data:
        if tuple(archive_data.files) != SCORER_MEMBERS:
            raise D110G1Error("D110 source-held archive member/order drift")
        z_id = np.asarray(archive_data["z_id"])
        pre_relu = np.asarray(archive_data["pre_relu"])
        labels = np.asarray(archive_data["labels"]).astype(str)
        receivers = np.asarray(archive_data["receiver_ids"]).astype(str)
        days = np.asarray(archive_data["day_ids"]).astype(str)
        physical_ids = np.asarray(archive_data["physical_ids"]).astype(str)
        class_ids = np.asarray(archive_data["class_ids"]).astype(str)
    if (
        z_id.dtype != np.float32
        or pre_relu.dtype != np.float32
        or z_id.shape != (EXPECTED_HELD_ROWS, 160)
        or pre_relu.shape != z_id.shape
        or not np.isfinite(z_id).all()
        or not np.isfinite(pre_relu).all()
        or not np.array_equal(np.maximum(pre_relu, np.float32(0.0)), z_id)
        or len(set(physical_ids.tolist())) != EXPECTED_HELD_ROWS
        or len(set(class_ids.tolist())) != 6
    ):
        raise D110G1Error("D110 source z_id/pre_relu/physical-ID closure drift")
    registry = tuple(sorted(class_ids.tolist()))
    receiver_set = tuple(sorted(set(receivers.tolist())))
    day_set = tuple(sorted(set(days.tolist())))
    group_counts = [
        int(np.sum((receivers == receiver) & (labels == class_id)))
        for receiver in receiver_set
        for class_id in registry
    ]
    cell_counts = [
        int(
            np.sum(
                (receivers == receiver) & (labels == class_id) & (days == day)
            )
        )
        for receiver in receiver_set
        for class_id in registry
        for day in day_set
    ]
    if (
        len(receiver_set) != 7
        or len(day_set) != 4
        or set(labels.tolist()) != set(registry)
        or group_counts != [28] * GROUP_COUNT
        or cell_counts != [HELD_PER_CELL] * (GROUP_COUNT * 4)
    ):
        raise D110G1Error("D110 source-held 1176/28/7 closure drift")


def prepare(args: argparse.Namespace) -> int:
    """Reuse D106 package/truth separation after checking the new D110 source."""

    _validate_new_source(args)
    with _d110_d106_context():
        return d106_g1.prepare(args)


def _geometry_from_pinned_tap(
    archive_path: Path, expected_sha256: str
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], str]:
    payload = d110_g0.base._read_pinned_archive(
        archive_path.resolve(strict=True), expected_sha256=expected_sha256
    )
    observed = d110_g0.base._sha256_bytes(payload)
    rows = d110_g0.base._load_rows(payload, archive_sha256=observed)
    closed_u, prior, receipt = d110_g0._geometry(rows)
    return closed_u, prior, receipt, observed


def _load_package(
    root: Path,
    package_row: Mapping[str, Any],
    *,
    classes: tuple[str, ...],
    k_shot: int,
) -> tuple[np.ndarray, tuple[str, ...], np.ndarray, list[str], str]:
    relative = Path(str(package_row["path"]))
    path = (root / relative).resolve(strict=True)
    if (
        relative.is_absolute()
        or not path.is_relative_to(root)
        or _file_sha(path) != package_row["sha256"]
    ):
        raise D110G1Error("D110 package path/SHA drift")
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != PACKAGE_KEYS or any("truth" in name.lower() for name in archive.files):
            raise D110G1Error("D110 predictor package member/truth closure drift")
        registry = tuple(archive["registered_classes"].astype(str).tolist())
        labels = tuple(archive["support_labels"].astype(str).tolist())
        support_ids = archive["support_physical_ids"].astype(str).tolist()
        query_ids = archive["query_physical_ids"].astype(str).tolist()
        support = np.maximum(
            np.asarray(archive["support_pre_relu"], dtype=np.float32), np.float32(0.0)
        )
        query = np.maximum(
            np.asarray(archive["query_pre_relu"], dtype=np.float32), np.float32(0.0)
        )
    expected_query_rows = len(classes) * QUERY_PER_CLASS_BY_K[k_shot]
    if (
        registry != classes
        or support.shape != (len(classes) * k_shot, 160)
        or query.shape != (expected_query_rows, 160)
        or len(labels) != len(classes) * k_shot
        or set(labels) != set(classes)
        or any(labels.count(class_id) != k_shot for class_id in classes)
        or len(query_ids) != expected_query_rows
        or set(support_ids).intersection(query_ids)
        or d106_g1.canonical_sha256(support_ids) != package_row["support_physical_id_root_sha256"]
        or d106_g1.canonical_sha256(query_ids) != package_row["query_physical_id_root_sha256"]
        or not np.isfinite(support).all()
        or not np.isfinite(query).all()
    ):
        raise D110G1Error("D110 package support/query/K closure drift")
    return support, labels, query, query_ids, str(package_row["sha256"])


def predict(args: argparse.Namespace) -> int:
    """Commit all 63 x four truth-free predictions using only the US-qKNN core."""

    root = args.package_root.resolve(strict=True)
    output = args.output_dir.resolve()
    run_id = _new_run_id(args.run_id)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"immutable D110 G1 output exists: {output}")
    manifest_path = root / "package_manifest.json"
    manifest = d106_g1._read_json(manifest_path)
    receivers = tuple(str(value) for value in manifest.get("receiver_ids", ()))
    classes = tuple(str(value) for value in manifest.get("class_ids", ()))
    packages = manifest.get("packages")
    truth_input_seal_sha256 = manifest.get("truth_input_seal_sha256")
    if (
        manifest.get("schema") != PACKAGE_SCHEMA
        or manifest.get("candidate_id") != CANDIDATE_ID
        or manifest.get("split_id") != SPLIT_ID
        or manifest.get("query_truth_present") is not False
        or len(receivers) != 7
        or len(classes) != 6
        or not isinstance(packages, list)
        or len(packages) != 21
        or not isinstance(truth_input_seal_sha256, str)
        or len(truth_input_seal_sha256) != 64
    ):
        raise D110G1Error("D110 truth-free package manifest closure drift")
    by_key = {(str(row["held_receiver"]), int(row["K"])): row for row in packages}
    if set(by_key) != {(receiver, k) for receiver in receivers for k in K_VALUES}:
        raise D110G1Error("D110 package matrix is not complete")
    closed_u, prior, geometry, tap_sha256 = _geometry_from_pinned_tap(
        args.d106_tap_archive, args.d106_tap_archive_sha256
    )
    output.mkdir(parents=True, exist_ok=False)
    row_root = output / "rows"
    row_root.mkdir()
    cache: dict[tuple[str, int], tuple[tuple[str, ...], np.ndarray, list[str], str, Mapping[str, usqknn.D110USQKNNState]]] = {}
    rows: list[dict[str, Any]] = []
    for receiver, held_class, k_shot in fixed_row_specs(receivers, classes):
        key = (receiver, k_shot)
        package_row = by_key[key]
        if key not in cache:
            support, labels, query, query_ids, package_sha = _load_package(
                root, package_row, classes=classes, k_shot=k_shot
            )
            states = usqknn.fit_d110_usqknn_four_arms(
                support,
                labels,
                classes,
                config=d106_g1._lock(k_shot, package_sha),
                closed_u=closed_u,
                prior_variances=prior,
            )
            cache[key] = (labels, query, query_ids, package_sha, states)
        _labels, query, query_ids, _package_sha, states = cache[key]
        predictions = {arm: list(usqknn.predict_d110_usqknn(states[arm], query)) for arm in ARMS}
        if k_shot == 1 and (
            predictions["M_HEAD"] != predictions["M0"]
            or predictions["M_JOINT"] != predictions["M_DA"]
        ):
            raise D110G1Error("D110 K1 head identity relation drift")
        audits = {arm: usqknn.audit_d110_usqknn_state(states[arm]) for arm in ARMS}
        if any(audit["query_rows_used_for_fit"] or audit["query_state_updates"] for audit in audits.values()):
            raise D110G1Error("D110 core query lifecycle drift")
        row = {
            "schema": PREDICTION_SCHEMA + ".row",
            "candidate_id": CANDIDATE_ID,
            "split_id": SPLIT_ID,
            "run_id": run_id,
            "held_receiver": receiver,
            "held_class": held_class,
            "K": k_shot,
            "package_id": str(package_row["package_id"]),
            "registered_classes": list(classes),
            "query_physical_ids": query_ids,
            "arm_predictions": predictions,
            "shared_component_receipts": {
                "d106_tap_archive_sha256": tap_sha256,
                "d110_geometry_root_sha256": geometry["geometry_root_sha256"],
                "student_t_lock_sha256": states["M0"].bank.config.lock_digest,
                "arm_state_audits": audits,
            },
            "query_truth_access": False,
            "target_access": False,
            "formal_p2_authority": False,
            "query_state_updates": 0,
        }
        row["prediction_receipt_sha256"] = _sha(row)
        path = row_root / f"{_sha({'receiver': receiver, 'held_class': held_class, 'K': k_shot})}.json"
        d106_g1._write_new(path, row)
        rows.append({
            "held_receiver": receiver, "held_class": held_class, "K": k_shot,
            "package_id": str(package_row["package_id"]),
            "path": str(Path("rows") / path.name), "sha256": _file_sha(path),
            "prediction_receipt_sha256": row["prediction_receipt_sha256"],
        })
    if len(rows) != 63 or len({row["prediction_receipt_sha256"] for row in rows}) != 63:
        raise D110G1Error("D110 G1 prediction coverage did not close at 63 rows")
    result = {
        "schema": PREDICTION_SCHEMA,
        "candidate_id": CANDIDATE_ID,
        "split_id": SPLIT_ID,
        "run_id": run_id,
        "row_count": 63,
        "arm_row_prediction_unit_count": 252,
        "rows": rows,
        "package_manifest_sha256": _file_sha(manifest_path),
        "truth_input_seal_sha256": truth_input_seal_sha256,
        "d106_tap_archive_sha256": tap_sha256,
        "d110_geometry_root_sha256": geometry["geometry_root_sha256"],
        "query_truth_access": False,
        "target_access": False,
        "query_state_updates": 0,
        "sourceheld_non_target": True,
        "formal_p2_authority": False,
        "sealed_at_unix_ns": time.time_ns(),
    }
    result["prediction_set_receipt_sha256"] = _sha(result)
    d106_g1._write_new(output / "prediction_manifest.json", result)
    print(output / "prediction_manifest.json")
    return 0


def _validate_truth_open_binding(args: argparse.Namespace) -> None:
    """Close the prediction -> package manifest -> truth-seal chain before open."""

    prediction_root = args.prediction_root.resolve(strict=True)
    prediction_manifest_path = prediction_root / "prediction_manifest.json"
    prediction_manifest = d106_g1._read_json(prediction_manifest_path)
    expected_package_sha = prediction_manifest.get("package_manifest_sha256")
    expected_seal_sha = prediction_manifest.get("truth_input_seal_sha256")
    expected_receipt = prediction_manifest.get("prediction_set_receipt_sha256")
    if (
        prediction_manifest.get("schema") != PREDICTION_SCHEMA
        or prediction_manifest.get("candidate_id") != CANDIDATE_ID
        or prediction_manifest.get("split_id") != SPLIT_ID
        or not isinstance(expected_package_sha, str)
        or not isinstance(expected_seal_sha, str)
        or _sha({
            key: value
            for key, value in prediction_manifest.items()
            if key != "prediction_set_receipt_sha256"
        }) != expected_receipt
    ):
        raise D110G1Error("D110 prediction/package-manifest binding drift")
    truth_seal_path = args.truth_input_seal_json.resolve(strict=True)
    actual_seal_sha = _file_sha(truth_seal_path)
    if actual_seal_sha != expected_seal_sha:
        raise D110G1Error("D110 truth-input seal SHA drift before truth open")
    package_manifest_path = truth_seal_path.parent.parent / "package_manifest.json"
    try:
        package_manifest_path = package_manifest_path.resolve(strict=True)
    except FileNotFoundError as exc:
        raise D110G1Error("D110 package manifest is unavailable for truth-open binding") from exc
    package_manifest = d106_g1._read_json(package_manifest_path)
    package_rows = package_manifest.get("packages")
    if not isinstance(package_rows, list) or len(package_rows) != 21 or any(
        not isinstance(row, Mapping) for row in package_rows
    ):
        raise D110G1Error("D110 package-manifest row chain drift before truth open")
    package_ids = [str(row.get("package_id")) for row in package_rows]
    query_roots = {
        str(row.get("package_id")): row.get("query_physical_id_root_sha256")
        for row in package_rows
    }
    truth_seal = d106_g1._read_json(truth_seal_path)
    if (
        _file_sha(package_manifest_path) != expected_package_sha
        or package_manifest.get("schema") != PACKAGE_SCHEMA
        or package_manifest.get("candidate_id") != CANDIDATE_ID
        or package_manifest.get("split_id") != SPLIT_ID
        or package_manifest.get("truth_input_seal_sha256") != actual_seal_sha
        or package_manifest.get("query_truth_present") is not False
        or package_manifest.get("target_access") is not False
        or package_manifest.get("formal_query_state_updates") != 0
        or len(set(package_ids)) != 21
        or len(query_roots) != 21
        or set(truth_seal.get("package_ids", ())) != set(package_ids)
        or truth_seal.get("query_physical_id_roots") != query_roots
        or truth_seal.get("source_val_scorer_manifest_sha256")
        != package_manifest.get("source_val_scorer_manifest_sha256")
        or truth_seal.get("source_val_scorer_archive_sha256")
        != package_manifest.get("source_val_scorer_archive_sha256")
    ):
        raise D110G1Error("D110 package-manifest/truth-seal SHA chain drift before truth open")


def score(args: argparse.Namespace) -> int:
    """Reuse D106's independent no-reprediction scorer under D110 schemas."""

    _validate_truth_open_binding(args)
    with _d110_d106_context():
        return d106_g1.score(args)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("--source-val-archive", type=Path, required=True)
    prepare_parser.add_argument("--source-val-manifest", type=Path, required=True)
    prepare_parser.add_argument("--output-dir", type=Path, required=True)
    predict_parser = commands.add_parser("predict")
    predict_parser.add_argument("--package-root", type=Path, required=True)
    predict_parser.add_argument("--d106-tap-archive", type=Path, required=True)
    predict_parser.add_argument("--d106-tap-archive-sha256", required=True)
    predict_parser.add_argument("--run-id", required=True)
    predict_parser.add_argument("--output-dir", type=Path, required=True)
    score_parser = commands.add_parser("score")
    score_parser.add_argument("--prediction-root", type=Path, required=True)
    score_parser.add_argument("--truth-json", type=Path, required=True)
    score_parser.add_argument("--truth-input-seal-json", type=Path, required=True)
    score_parser.add_argument("--truth-open-event-json", type=Path, required=True)
    score_parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "prepare":
        return prepare(args)
    return predict(args) if args.command == "predict" else score(args)


if __name__ == "__main__":
    raise SystemExit(main())
