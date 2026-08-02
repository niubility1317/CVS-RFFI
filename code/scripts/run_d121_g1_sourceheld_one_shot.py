#!/usr/bin/env python3
"""Run the frozen D121 four-arm source-held G1, then score separately.

``predict`` consumes only the already sealed D104 predictor packages.  It has
no truth argument and commits every fixed row before ``score`` may open the
independently held D104 truth package.  This is source-held evidence only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from scripts import run_d106_g1_sourceheld_one_shot as d106  # noqa: E402
from cvsrffi import stage2_zid_student_t_qknn as qknn  # noqa: E402
from cvsrffi.stage2_d121_lbr_qknn import (  # noqa: E402
    audit_lbr_qknn_state,
    build_lbr_qknn_state,
    score_lbr_qknn_trace,
    unique_lbr_argmax,
)
from cvsrffi.stage2_zid_student_t_qknn import (  # noqa: E402
    build_typed_zid_support_bank,
    identity_shared_psd_metric,
    score_zid_student_t_logits,
)


CANDIDATE_ID = "D121_RDCE_LBR_QKNN"
ARMS = ("M0", "M_DA", "M_HEAD", "M_JOINT")
EFFECT_PAIRS = {
    "HEAD_AT_ID": ("M_HEAD", "M0"),
    "HEAD_AT_DA": ("M_JOINT", "M_DA"),
}
SPLIT_ID = d106.SPLIT_ID
K_VALUES = d106.K_VALUES
PACKAGE_SCHEMA = d106.PACKAGE_SCHEMA
PACKAGE_KEYS = d106.PACKAGE_KEYS
PREDICTION_SCHEMA = "cvs.d121.rdce_lbr_qknn.sourceheld.predictions.v1"
SCORE_SCHEMA = "cvs.d121.rdce_lbr_qknn.sourceheld.scores.v1"
EFFECT_METRICS = (
    "old_balanced_accuracy",
    "seen_new_accuracy",
    "H_old_new",
    "old_floor",
    "all_class_floor",
    "balanced_accuracy",
    "old_correct_count",
    "seen_new_correct_count",
    "correct_count",
)


class D121G1Error(ValueError):
    """Raised when D121's fixed four-arm source-held contract drifts."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha(path: Path) -> str:
    return d106._file_sha(path)


def _read_json(path: Path) -> dict[str, Any]:
    return d106._read_json(path)


def _write_new(path: Path, value: Any) -> None:
    d106._write_new(path, value)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def fixed_row_specs(
    receivers: Sequence[str], classes: Sequence[str]
) -> tuple[tuple[str, str | None, int], ...]:
    """The frozen complete 63-row matrix; there is no slice selector."""

    return d106.fixed_row_specs(receivers, classes)


def _run_id(value: str) -> str:
    if type(value) is not str or not value or len(value.encode("utf-8")) > 160:
        raise D121G1Error("run ID must be a short non-empty string")
    return value


def prepare(args: argparse.Namespace) -> int:
    """Thin compatibility only; D121 releases reuse existing D104 packages."""

    return d106.prepare(args)


def _validate_package_manifest(
    root: Path,
) -> tuple[Path, dict[str, Any], tuple[str, ...], tuple[str, ...], dict[tuple[str, int], dict[str, Any]], str]:
    manifest_path = root / "package_manifest.json"
    manifest = _read_json(manifest_path)
    receivers = tuple(str(item) for item in manifest.get("receiver_ids", ()))
    classes = tuple(str(item) for item in manifest.get("class_ids", ()))
    packages = manifest.get("packages")
    truth_seal_sha = manifest.get("truth_input_seal_sha256")
    if (
        manifest.get("schema") != PACKAGE_SCHEMA
        or manifest.get("candidate_id") != d106.D104_CANDIDATE_ID
        or manifest.get("split_id") != SPLIT_ID
        or manifest.get("query_truth_present") is not False
        or manifest.get("target_access") is not False
        or len(receivers) != 7
        or len(set(receivers)) != 7
        or len(classes) != 6
        or len(set(classes)) != 6
        or not isinstance(packages, list)
        or len(packages) != 21
        or not isinstance(truth_seal_sha, str)
        or len(truth_seal_sha) != 64
    ):
        raise D121G1Error("D121 D104 predictor-package manifest closure drift")
    by_key = {
        (str(row.get("held_receiver")), int(row.get("K"))): row for row in packages
    }
    expected = {(receiver, k_shot) for receiver in receivers for k_shot in K_VALUES}
    if len(by_key) != 21 or set(by_key) != expected:
        raise D121G1Error("D121 fixed 21-package matrix drift")
    return manifest_path, manifest, receivers, classes, by_key, truth_seal_sha


def _load_package(
    root: Path,
    package_row: Mapping[str, Any],
    *,
    classes: tuple[str, ...],
    k_shot: int,
) -> tuple[np.ndarray, tuple[str, ...], tuple[str, ...], np.ndarray, tuple[str, ...], str]:
    relative = Path(str(package_row.get("path", "")))
    path = (root / relative).resolve(strict=True)
    if relative.is_absolute() or not path.is_relative_to(root):
        raise D121G1Error("D121 predictor package path escapes package root")
    if _file_sha(path) != package_row.get("sha256"):
        raise D121G1Error("D121 predictor package SHA drift")
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != PACKAGE_KEYS:
            raise D121G1Error("D121 predictor package member closure drift")
        support = np.asarray(archive["support_pre_relu"], dtype=np.float32).copy()
        labels = tuple(str(value) for value in archive["support_labels"].astype(str).tolist())
        support_ids = tuple(
            str(value) for value in archive["support_physical_ids"].astype(str).tolist()
        )
        query = np.asarray(archive["query_pre_relu"], dtype=np.float32).copy()
        query_ids = tuple(
            str(value) for value in archive["query_physical_ids"].astype(str).tolist()
        )
        registry = tuple(
            str(value) for value in archive["registered_classes"].astype(str).tolist()
        )
    if (
        registry != classes
        or support.shape != (len(classes) * k_shot, 160)
        or query.ndim != 2
        or query.shape[1] != 160
        or len(labels) != len(support)
        or len(support_ids) != len(support)
        or len(set(support_ids)) != len(support_ids)
        or not query_ids
        or len(query_ids) != len(query)
        or len(set(query_ids)) != len(query_ids)
        or set(support_ids).intersection(query_ids)
        or any(label not in classes for label in labels)
        or any(labels.count(class_id) != k_shot for class_id in classes)
        or not np.isfinite(support).all()
        or not np.isfinite(query).all()
    ):
        raise D121G1Error("D121 predictor package support/query closure drift")
    return support, labels, support_ids, query, query_ids, str(package_row["sha256"])


def _canonical_bank_physical_ids(
    support: np.ndarray,
    labels: Sequence[str],
    physical_ids: Sequence[str],
    registry: Sequence[str],
) -> tuple[str, ...]:
    """Reproduce qKNN's exact support-bank canonical order for physical IDs."""

    values = qknn.normalize_zid_rows(np.asarray(support, dtype=np.float32))
    typed_labels = tuple(str(value) for value in labels)
    typed_ids = tuple(str(value) for value in physical_ids)
    classes = tuple(qknn._registry(registry))
    if (
        len(values) != len(typed_labels)
        or len(values) != len(typed_ids)
        or len(set(typed_ids)) != len(typed_ids)
        or any(label not in classes for label in typed_labels)
    ):
        raise D121G1Error("D121 physical-ID canonical-order inputs drift")
    class_map = {class_id: index for index, class_id in enumerate(classes)}
    indices = np.asarray([class_map[label] for label in typed_labels], dtype=np.int16)
    codes, scales, _decoded = qknn._quantize_rows(values)
    order = qknn._canonical_order(codes, scales, indices)
    result = tuple(typed_ids[int(index)] for index in order)
    if len(result) != len(values) or len(set(result)) != len(result):
        raise D121G1Error("D121 physical-ID canonical order did not close")
    return result


def _baseline_predictions(
    bank: Any, query: np.ndarray, registry: Sequence[str], metric: Any
) -> list[str]:
    logits = score_zid_student_t_logits(bank, query, metric=metric)
    return d106._argmax(logits, registry)


def _assert_query_zero(audits: Mapping[str, Mapping[str, Any]]) -> None:
    for name, audit in audits.items():
        if any(
            int(audit.get(field, -1)) != 0
            for field in (
                "query_rows_used_for_fit",
                "query_state_updates",
                "query_selection_count",
            )
        ):
            raise D121G1Error(f"D121 query lifecycle drift in {name}")


def _build_four_arm_predictions(
    *,
    support_signed: np.ndarray,
    labels: tuple[str, ...],
    support_physical_ids: tuple[str, ...],
    query_signed: np.ndarray,
    registry: tuple[str, ...],
    k_shot: int,
    package_sha256: str,
    rdce_asset: Any,
) -> tuple[dict[str, list[str]], dict[str, Any]]:
    """Build exactly the frozen 2x2 arms from one support-only package."""

    support_plus = np.ascontiguousarray(np.maximum(support_signed, np.float32(0.0)))
    query_plus = np.ascontiguousarray(np.maximum(query_signed, np.float32(0.0)))
    lock = d106._lock(k_shot, package_sha256)

    identity_bank = build_typed_zid_support_bank(
        support_plus, labels, registry, config=lock
    )
    identity_metric = identity_shared_psd_metric(config=lock)
    identity_ids = _canonical_bank_physical_ids(
        support_plus, labels, support_physical_ids, registry
    )
    head_state = build_lbr_qknn_state(
        identity_bank, identity_ids, metric=identity_metric
    )
    head_trace = score_lbr_qknn_trace(
        head_state, identity_bank, query_plus, metric=identity_metric
    )

    # These are the exact D106 source-held RDCE functions, not a D121 variant.
    da_state = d106.fit_rdce_sourceheld_state(rdce_asset, support_plus, labels, k_shot)
    da_support = d106.apply_rdce_state(da_state, support_plus)
    da_query = d106.apply_rdce_state(da_state, query_plus)
    da_bank = build_typed_zid_support_bank(da_support, labels, registry, config=lock)
    da_metric = identity_shared_psd_metric(config=lock)
    da_ids = _canonical_bank_physical_ids(
        da_support, labels, support_physical_ids, registry
    )
    joint_state = build_lbr_qknn_state(da_bank, da_ids, metric=da_metric)
    joint_trace = score_lbr_qknn_trace(
        joint_state, da_bank, da_query, metric=da_metric
    )

    audits = {
        "M_HEAD": _jsonable(audit_lbr_qknn_state(head_state)),
        "M_JOINT": _jsonable(audit_lbr_qknn_state(joint_state)),
    }
    _assert_query_zero(audits)
    predictions = {
        "M0": _baseline_predictions(identity_bank, query_plus, registry, identity_metric),
        "M_DA": _baseline_predictions(da_bank, da_query, registry, da_metric),
        "M_HEAD": list(unique_lbr_argmax(head_trace.class_logits_fp32, registry)),
        "M_JOINT": list(unique_lbr_argmax(joint_trace.class_logits_fp32, registry)),
    }
    if set(predictions) != set(ARMS) or any(
        len(values) != len(query_plus) for values in predictions.values()
    ):
        raise D121G1Error("D121 four-arm prediction closure drift")
    receipts = {
        "M_DA_M_JOINT_rdce_state_sha256": str(da_state["receipt"]),
        "M0_M_HEAD_identity_view_sha256": _sha(
            {
                "support_plus_sha256": hashlib.sha256(support_plus.tobytes()).hexdigest(),
                "query_plus_sha256": hashlib.sha256(query_plus.tobytes()).hexdigest(),
                "support_physical_ids": list(support_physical_ids),
            }
        ),
        "student_t_lock_sha256": lock.lock_digest,
        "lbr_bank_rebuilds": {
            "M_HEAD": {
                "bank_receipt_sha256": identity_bank.bank_receipt_sha256,
                "canonical_support_physical_ids": list(identity_ids),
                "state_audit": audits["M_HEAD"],
            },
            "M_JOINT": {
                "bank_receipt_sha256": da_bank.bank_receipt_sha256,
                "canonical_support_physical_ids": list(da_ids),
                "state_audit": audits["M_JOINT"],
            },
        },
    }
    return predictions, receipts


def predict(args: argparse.Namespace) -> int:
    """Seal all 63 x four source-held predictions without opening truth."""

    root = args.package_root.resolve(strict=True)
    output = args.output_dir.resolve()
    run_id = _run_id(args.run_id)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"immutable D121 G1 prediction root exists: {output}")
    manifest_path, _manifest, receivers, classes, by_key, truth_seal_sha = (
        _validate_package_manifest(root)
    )
    rdce_asset = d106._parse_asset_wire(
        args.rdce_asset_wire.resolve(strict=True), args.rdce_wire_sha256
    )
    if rdce_asset.split_id != SPLIT_ID:
        raise D121G1Error("D121 RDCE asset/source-held split mismatch")

    output.mkdir(parents=True, exist_ok=False)
    row_root = output / "rows"
    row_root.mkdir()
    cache: dict[tuple[str, int], tuple[tuple[str, ...], str, dict[str, list[str]], dict[str, Any]]] = {}
    rows: list[dict[str, Any]] = []
    for receiver, held_class, k_shot in fixed_row_specs(receivers, classes):
        key = (receiver, k_shot)
        package_row = by_key[key]
        if key not in cache:
            support, labels, support_ids, query, query_ids, package_sha = _load_package(
                root, package_row, classes=classes, k_shot=k_shot
            )
            predictions, receipts = _build_four_arm_predictions(
                support_signed=support,
                labels=labels,
                support_physical_ids=support_ids,
                query_signed=query,
                registry=classes,
                k_shot=k_shot,
                package_sha256=package_sha,
                rdce_asset=rdce_asset,
            )
            cache[key] = (query_ids, package_sha, predictions, receipts)
        query_ids, package_sha, predictions, receipts = cache[key]
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
            "query_physical_ids": list(query_ids),
            "arm_predictions": {arm: list(predictions[arm]) for arm in ARMS},
            "shared_component_receipts": {
                "package_sha256": package_sha,
                "rdce_asset_wire_sha256": args.rdce_wire_sha256,
                **receipts,
            },
            "query_truth_access": False,
            "target_access": False,
            "formal_p2_authority": False,
            "query_rows_used_for_fit": 0,
            "query_state_updates": 0,
            "query_selection_count": 0,
        }
        row["prediction_receipt_sha256"] = _sha(row)
        path = row_root / (
            _sha({"receiver": receiver, "held_class": held_class, "K": k_shot})
            + ".json"
        )
        _write_new(path, row)
        rows.append(
            {
                "held_receiver": receiver,
                "held_class": held_class,
                "K": k_shot,
                "package_id": str(package_row["package_id"]),
                "path": str(Path("rows") / path.name),
                "sha256": _file_sha(path),
                "prediction_receipt_sha256": row["prediction_receipt_sha256"],
            }
        )
    if len(rows) != 63 or len({row["prediction_receipt_sha256"] for row in rows}) != 63:
        raise D121G1Error("D121 63-row prediction coverage did not close")
    result = {
        "schema": PREDICTION_SCHEMA,
        "candidate_id": CANDIDATE_ID,
        "split_id": SPLIT_ID,
        "run_id": run_id,
        "arms": list(ARMS),
        "row_count": 63,
        "arm_row_prediction_unit_count": 63 * len(ARMS),
        "rows": rows,
        "package_manifest_sha256": _file_sha(manifest_path),
        "truth_input_seal_sha256": truth_seal_sha,
        "rdce_asset_wire_sha256": args.rdce_wire_sha256,
        "query_truth_access": False,
        "target_access": False,
        "query_rows_used_for_fit": 0,
        "query_state_updates": 0,
        "query_selection_count": 0,
        "sourceheld_non_target": True,
        "formal_p2_authority": False,
        "sealed_at_unix_ns": time.time_ns(),
    }
    result["prediction_set_receipt_sha256"] = _sha(result)
    _write_new(output / "prediction_manifest.json", result)
    print(output / "prediction_manifest.json")
    return 0


def _validate_truth_open_binding(
    args: argparse.Namespace,
) -> tuple[Path, dict[str, Any], Path]:
    prediction_root = args.prediction_root.resolve(strict=True)
    manifest_path = prediction_root / "prediction_manifest.json"
    manifest = _read_json(manifest_path)
    if (
        manifest.get("schema") != PREDICTION_SCHEMA
        or manifest.get("candidate_id") != CANDIDATE_ID
        or manifest.get("split_id") != SPLIT_ID
        or manifest.get("arms") != list(ARMS)
        or manifest.get("row_count") != 63
        or manifest.get("arm_row_prediction_unit_count") != 63 * len(ARMS)
        or manifest.get("query_truth_access") is not False
        or manifest.get("target_access") is not False
        or manifest.get("query_rows_used_for_fit") != 0
        or manifest.get("query_state_updates") != 0
        or manifest.get("query_selection_count") != 0
        or _sha(
            {
                key: value
                for key, value in manifest.items()
                if key != "prediction_set_receipt_sha256"
            }
        )
        != manifest.get("prediction_set_receipt_sha256")
    ):
        raise D121G1Error("D121 prediction manifest binding drift")
    truth_seal_path = args.truth_input_seal_json.resolve(strict=True)
    if _file_sha(truth_seal_path) != manifest.get("truth_input_seal_sha256"):
        raise D121G1Error("D121 truth-input seal SHA drift")
    package_manifest_path = (truth_seal_path.parent.parent / "package_manifest.json").resolve(
        strict=True
    )
    package_manifest = _read_json(package_manifest_path)
    truth_seal = _read_json(truth_seal_path)
    packages = package_manifest.get("packages")
    if (
        _file_sha(package_manifest_path) != manifest.get("package_manifest_sha256")
        or package_manifest.get("schema") != PACKAGE_SCHEMA
        or package_manifest.get("candidate_id") != d106.D104_CANDIDATE_ID
        or package_manifest.get("split_id") != SPLIT_ID
        or package_manifest.get("query_truth_present") is not False
        or not isinstance(packages, list)
        or len(packages) != 21
        or package_manifest.get("truth_input_seal_sha256") != _file_sha(truth_seal_path)
        or truth_seal.get("split_id") != SPLIT_ID
        or truth_seal.get("package_count") != 21
        or truth_seal.get("predictor_truth_access") is not False
        or set(truth_seal.get("package_ids", ()))
        != {str(row.get("package_id")) for row in packages}
    ):
        raise D121G1Error("D121 D104 package/truth-seal chain drift")
    return manifest_path, manifest, truth_seal_path


def _metric(
    truth: np.ndarray,
    predicted: Sequence[str],
    classes: Sequence[str],
    held_class: str | None,
) -> dict[str, Any]:
    base = d106._metric(truth, predicted, classes, held_class)
    prediction = np.asarray(predicted, dtype=str)
    old_mask = np.ones(len(truth), dtype=bool) if held_class is None else truth != held_class
    old_correct = int(np.count_nonzero(prediction[old_mask] == truth[old_mask]))
    if held_class is None:
        seen_new_correct: int | None = None
        seen_new_count: int | None = None
    else:
        new_mask = truth == held_class
        seen_new_correct = int(np.count_nonzero(prediction[new_mask] == truth[new_mask]))
        seen_new_count = int(np.count_nonzero(new_mask))
    return {
        **base,
        "old_correct_count": old_correct,
        "old_query_count": int(np.count_nonzero(old_mask)),
        "seen_new_correct_count": seen_new_correct,
        "seen_new_query_count": seen_new_count,
    }


def _effect(
    left: Mapping[str, Any], right: Mapping[str, Any]
) -> dict[str, float | int | None]:
    values: dict[str, float | int | None] = {}
    for name in EFFECT_METRICS:
        left_value = left[name]
        right_value = right[name]
        values[name] = (
            None
            if left_value is None or right_value is None
            else left_value - right_value
        )
    return values


def _progression_summary(scored_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Apply only D121's frozen matrix-level promotion rule after scoring."""

    summaries: dict[str, Any] = {}
    for effect_name, (candidate, baseline) in EFFECT_PAIRS.items():
        old_net = 0
        new_net = 0
        k1_total_net = 0
        old_floor_candidate: list[float] = []
        old_floor_baseline: list[float] = []
        all_floor_candidate: list[float] = []
        all_floor_baseline: list[float] = []
        held_new_rows = 0
        for row in scored_rows:
            metrics = row["arm_metrics"]
            effects = row["same_row_effects"][effect_name]
            old_net += int(effects["old_correct_count"])
            if row["held_class"] is not None:
                held_new_rows += 1
                if effects["seen_new_correct_count"] is None:
                    raise D121G1Error("D121 held-class seen-new effect unexpectedly missing")
                new_net += int(effects["seen_new_correct_count"])
            elif effects["seen_new_correct_count"] is not None:
                raise D121G1Error("D121 held_class=None seen-new effect must remain None")
            old_floor_candidate.append(float(metrics[candidate]["old_floor"]))
            old_floor_baseline.append(float(metrics[baseline]["old_floor"]))
            all_floor_candidate.append(float(metrics[candidate]["all_class_floor"]))
            all_floor_baseline.append(float(metrics[baseline]["all_class_floor"]))
            if int(row["K"]) == 1:
                k1_total_net += int(effects["correct_count"])
        old_floor_delta = min(old_floor_candidate) - min(old_floor_baseline)
        all_floor_delta = min(all_floor_candidate) - min(all_floor_baseline)
        criteria = {
            "old_correct_count_net_nonnegative": old_net >= 0,
            "seen_new_correct_count_net_nonnegative": new_net >= 0,
            "old_floor_global_min_nonnegative": old_floor_delta >= 0.0,
            "all_class_floor_global_min_nonnegative": all_floor_delta >= 0.0,
            "k1_total_correct_count_net_strictly_positive": k1_total_net > 0,
        }
        summaries[effect_name] = {
            "candidate_arm": candidate,
            "baseline_arm": baseline,
            "held_class_row_count_used_for_seen_new": held_new_rows,
            "old_correct_count_net": old_net,
            "seen_new_correct_count_net": new_net,
            "old_floor_global_min_candidate": min(old_floor_candidate),
            "old_floor_global_min_baseline": min(old_floor_baseline),
            "old_floor_global_min_delta": old_floor_delta,
            "all_class_floor_global_min_candidate": min(all_floor_candidate),
            "all_class_floor_global_min_baseline": min(all_floor_baseline),
            "all_class_floor_global_min_delta": all_floor_delta,
            "k1_total_correct_count_net": k1_total_net,
            "criteria": criteria,
            "promotable": all(criteria.values()),
        }
    promoted = all(item["promotable"] for item in summaries.values())
    return {
        "rule": "matrix_aggregate_only_no_rowwise_nonnegative_requirement",
        "effects": summaries,
        "promotion_decision": (
            "PROMOTE_D121_NEXT_STAGE" if promoted else "REJECT_D121_REVISION_PERFORMANCE_WEAK"
        ),
        "promotion_allowed": promoted,
    }


def score(args: argparse.Namespace) -> int:
    """Open D104 truth only after the complete immutable prediction seal closes."""

    manifest_path, manifest, seal_path = _validate_truth_open_binding(args)
    output = args.output_json.resolve()
    event_path = args.truth_open_event_json.resolve()
    if output.exists() or event_path.exists():
        raise FileExistsError("immutable D121 G1 score/event output exists")
    root = args.prediction_root.resolve(strict=True)
    entries = manifest.get("rows")
    if not isinstance(entries, list) or len(entries) != 63:
        raise D121G1Error("D121 prediction row list drift")
    artifacts = []
    package_ids: set[str] = set()
    query_ids_by_package: dict[str, list[str]] = {}
    for entry in entries:
        relative = Path(str(entry.get("path", "")))
        path = (root / relative).resolve(strict=True)
        if relative.is_absolute() or not path.is_relative_to(root) or _file_sha(path) != entry.get("sha256"):
            raise D121G1Error("D121 prediction row seal drift")
        artifact = _read_json(path)
        if _sha(
            {
                key: value
                for key, value in artifact.items()
                if key != "prediction_receipt_sha256"
            }
        ) != artifact.get("prediction_receipt_sha256"):
            raise D121G1Error("D121 prediction row receipt drift")
        package_id = str(artifact.get("package_id"))
        query_ids = artifact.get("query_physical_ids")
        if (
            artifact.get("schema") != PREDICTION_SCHEMA + ".row"
            or artifact.get("candidate_id") != CANDIDATE_ID
            or artifact.get("split_id") != SPLIT_ID
            or artifact.get("query_truth_access") is not False
            or artifact.get("target_access") is not False
            or artifact.get("formal_p2_authority") is not False
            or artifact.get("query_rows_used_for_fit") != 0
            or artifact.get("query_state_updates") != 0
            or artifact.get("query_selection_count") != 0
            or set(artifact.get("arm_predictions", {})) != set(ARMS)
            or not isinstance(query_ids, list)
            or not query_ids
            or any(len(artifact["arm_predictions"][arm]) != len(query_ids) for arm in ARMS)
            or any(
                entry.get(name) != artifact.get(name)
                for name in (
                    "held_receiver",
                    "held_class",
                    "K",
                    "package_id",
                    "prediction_receipt_sha256",
                )
            )
        ):
            raise D121G1Error("D121 prediction row lifecycle drift")
        prior = query_ids_by_package.setdefault(package_id, query_ids)
        if prior != query_ids:
            raise D121G1Error("D121 package query physical IDs drift across matched rows")
        package_ids.add(package_id)
        artifacts.append(artifact)
    receivers = tuple(sorted({str(row["held_receiver"]) for row in artifacts}))
    classes = tuple(str(value) for value in artifacts[0]["registered_classes"])
    actual_rows = {
        (str(row["held_receiver"]), row["held_class"], int(row["K"]))
        for row in artifacts
    }
    if (
        len(package_ids) != 21
        or len(receivers) != 7
        or len(classes) != 6
        or any(tuple(row["registered_classes"]) != classes for row in artifacts)
        or actual_rows != set(fixed_row_specs(receivers, classes))
    ):
        raise D121G1Error("D121 fixed 63-row prediction coverage drift")
    event = {
        "schema": SCORE_SCHEMA + ".truth_open_event",
        "prediction_manifest_sha256": _file_sha(manifest_path),
        "truth_input_seal_sha256": _file_sha(seal_path),
        "prediction_manifest_mtime_ns": manifest_path.stat().st_mtime_ns,
        "truth_opened_after_all_predictions_committed": True,
        "opened_at_unix_ns": time.time_ns(),
    }
    if event["opened_at_unix_ns"] <= event["prediction_manifest_mtime_ns"]:
        raise D121G1Error("D121 truth-open timestamp is not after prediction seal")
    _write_new(event_path, event)
    truth = _read_json(args.truth_json.resolve(strict=True))
    truth_packages = truth.get("packages")
    truth_seal = _read_json(seal_path)
    if (
        truth.get("schema") != "cvs.d104_r1.rxid_angq.held_truth.v2"
        or truth.get("split_id") != SPLIT_ID
        or truth.get("package_count") != 21
        or truth.get("predictor_access") is not False
        or not isinstance(truth_packages, list)
        or len(truth_packages) != 21
        or d106.canonical_sha256(truth_packages)
        != truth_seal.get("truth_package_root_sha256")
    ):
        raise D121G1Error("D121 independent truth closure drift")
    truth_by_package = {str(row["package_id"]): row for row in truth_packages}
    if set(truth_by_package) != package_ids:
        raise D121G1Error("D121 truth/prediction package identity drift")
    negative = {
        effect: {metric: 0 for metric in EFFECT_METRICS} for effect in EFFECT_PAIRS
    }
    scored_rows = []
    for artifact in artifacts:
        matching = truth_by_package[str(artifact["package_id"])]
        if matching.get("query_physical_ids") != artifact["query_physical_ids"]:
            raise D121G1Error("D121 truth/prediction physical-ID alignment drift")
        labels = np.asarray(matching["query_truth_labels"], dtype=str)
        metrics = {
            arm: _metric(
                labels,
                artifact["arm_predictions"][arm],
                artifact["registered_classes"],
                artifact["held_class"],
            )
            for arm in ARMS
        }
        effects = {}
        for name, (left, right) in EFFECT_PAIRS.items():
            effect = _effect(metrics[left], metrics[right])
            effects[name] = effect
            for metric, delta in effect.items():
                if delta is not None and delta < 0:
                    negative[name][metric] += 1
        scored_rows.append(
            {
                "held_receiver": artifact["held_receiver"],
                "held_class": artifact["held_class"],
                "K": artifact["K"],
                "package_id": artifact["package_id"],
                "arm_metrics": metrics,
                "same_row_effects": effects,
                "prediction_receipt_sha256": artifact["prediction_receipt_sha256"],
            }
        )
    progression = _progression_summary(scored_rows)
    result = {
        "schema": SCORE_SCHEMA,
        "candidate_id": CANDIDATE_ID,
        "split_id": SPLIT_ID,
        "arms": list(ARMS),
        "performance_rows": scored_rows,
        "negative_tail_row_counts": negative,
        "progression_summary": progression,
        "prediction_manifest_sha256": _file_sha(manifest_path),
        "truth_input_seal_sha256": _file_sha(seal_path),
        "truth_sha256": _file_sha(args.truth_json.resolve(strict=True)),
        "truth_open_event_sha256": _file_sha(event_path),
        "prediction_artifact_committed_before_truth": True,
        "target_access": False,
    }
    result["score_set_receipt_sha256"] = _sha(result)
    _write_new(output, result)
    print(output)
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    preparer = commands.add_parser("prepare")
    preparer.add_argument("--source-val-archive", type=Path, required=True)
    preparer.add_argument("--source-val-manifest", type=Path, required=True)
    preparer.add_argument("--output-dir", type=Path, required=True)
    predictor = commands.add_parser("predict")
    predictor.add_argument("--package-root", type=Path, required=True)
    predictor.add_argument("--rdce-asset-wire", type=Path, required=True)
    predictor.add_argument("--rdce-wire-sha256", required=True)
    predictor.add_argument("--run-id", required=True)
    predictor.add_argument("--output-dir", type=Path, required=True)
    scorer = commands.add_parser("score")
    scorer.add_argument("--prediction-root", type=Path, required=True)
    scorer.add_argument("--truth-json", type=Path, required=True)
    scorer.add_argument("--truth-input-seal-json", type=Path, required=True)
    scorer.add_argument("--truth-open-event-json", type=Path, required=True)
    scorer.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "prepare":
        return prepare(args)
    return predict(args) if args.command == "predict" else score(args)


if __name__ == "__main__":
    raise SystemExit(main())
