"""Complete source-held coverage and fail-closed gate for D103-R1.

The module scores only complete, already-produced source-held receipts.  It
never dispatches training from interim performance and therefore cannot stop
or select folds based on BA, floor, H, or any other performance value.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

import numpy as np


CANDIDATE_ID = "D103-R2-RXID-CROSSRECEIVER-MB4"
SCHEMA = "cvs.d103_r2.rxid_dualsplit.phase1_held_falsifier.v1"
K_VALUES = (1, 5, 10)
TX_PROBE_LIMIT = 0.25
GPU_HOUR_LIMIT = 30.0
PEAK_MEMORY_LIMIT = 4 * 1024**3
DISK_LIMIT = 20 * 1024**3
STATE_LIMIT = 80 * 1024
MAC_LIMIT = 262_144
D102_METHOD_LOCK_SHA256 = (
    "9640267c2913e452a89be39e1b41e8b19d3371499afbed1efe8c9e3b7ad0e52f"
)
D102_ORIGINAL_REJECTED_RECEIPT_SHA256 = (
    "01a45e11fe519389071cf1eb279d293c958fc4fa48e0ed4c51bea9ff20c536b2"
)


class D103HeldFalsifierError(ValueError):
    """Raised when held coverage or receipt semantics are incomplete."""


@dataclass(frozen=True, slots=True)
class HeldFitSpec:
    fit_id: str
    fold_kind: str
    held_receiver: str | None
    held_class: str | None
    held_day: str | None
    fit_stage: str


def _token(value: str) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:12]


def build_complete_fit_plan(
    receiver_ids: Sequence[str],
    class_ids: Sequence[str],
    day_ids: Sequence[str],
) -> tuple[HeldFitSpec, ...]:
    receivers = tuple(sorted({str(value) for value in receiver_ids}))
    classes = tuple(sorted({str(value) for value in class_ids}))
    days = tuple(sorted({str(value) for value in day_ids}))
    if len(receivers) != 7 or len(classes) != 6 or len(days) != 4:
        raise D103HeldFalsifierError(
            f"frozen plan requires 7 receivers, 6 classes, 4 days; "
            f"got {len(receivers)}/{len(classes)}/{len(days)}"
        )
    specs: list[HeldFitSpec] = []
    for receiver in receivers:
        receiver_token = _token(receiver)
        for day in days:
            specs.append(
                HeldFitSpec(
                    fit_id=f"rx-{receiver_token}-day-{_token(day)}",
                    fold_kind="receiver_outer",
                    held_receiver=receiver,
                    held_class=None,
                    held_day=day,
                    fit_stage="leave_one_day",
                )
            )
        specs.append(
            HeldFitSpec(
                fit_id=f"rx-{receiver_token}-outer",
                fold_kind="receiver_outer",
                held_receiver=receiver,
                held_class=None,
                held_day=None,
                fit_stage="outer",
            )
        )
        for class_id in classes:
            class_token = _token(class_id)
            for day in days:
                specs.append(
                    HeldFitSpec(
                        fit_id=(
                            f"rxcls-{receiver_token}-{class_token}-day-{_token(day)}"
                        ),
                        fold_kind="receiver_class_outer",
                        held_receiver=receiver,
                        held_class=class_id,
                        held_day=day,
                        fit_stage="leave_one_day",
                    )
                )
            specs.append(
                HeldFitSpec(
                    fit_id=f"rxcls-{receiver_token}-{class_token}-outer",
                    fold_kind="receiver_class_outer",
                    held_receiver=receiver,
                    held_class=class_id,
                    held_day=None,
                    fit_stage="outer",
                )
            )
    specs.append(
        HeldFitSpec(
            fit_id="final-source-train",
            fold_kind="final_source_train",
            held_receiver=None,
            held_class=None,
            held_day=None,
            fit_stage="final",
        )
    )
    if len(specs) != 246 or len({spec.fit_id for spec in specs}) != 246:
        raise D103HeldFalsifierError("complete fit plan must contain 246 unique fits")
    return tuple(specs)


def probe_partition(
    receiver_ids: np.ndarray,
    day_ids: np.ndarray,
    labels: np.ndarray,
    physical_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    receiver_ids = np.asarray(receiver_ids).astype(str)
    day_ids = np.asarray(day_ids).astype(str)
    labels = np.asarray(labels).astype(str)
    physical_ids = np.asarray(physical_ids).astype(str)
    n = len(physical_ids)
    if any(values.shape != (n,) for values in (receiver_ids, day_ids, labels)):
        raise D103HeldFalsifierError("probe metadata arrays must all be [N]")
    if np.unique(physical_ids).size != n:
        raise D103HeldFalsifierError("probe physical IDs must be unique")

    train: list[int] = []
    test: list[int] = []
    cells = sorted(set(zip(receiver_ids.tolist(), day_ids.tolist(), labels.tolist())))
    for receiver, day, label in cells:
        local = np.flatnonzero(
            (receiver_ids == receiver) & (day_ids == day) & (labels == label)
        )
        if local.size < 5:
            raise D103HeldFalsifierError(
                f"probe cell requires at least 5 physical samples: {receiver}/{day}/{label}"
            )
        ranked = sorted(
            local.astype(int).tolist(),
            key=lambda index: hashlib.sha256(
                (
                    f"{CANDIDATE_ID}|{receiver}|{day}|{label}|"
                    f"{physical_ids[index]}|probe_v1"
                ).encode("utf-8")
            ).hexdigest(),
        )
        split = int(math.floor(0.60 * len(ranked)))
        if split < 1 or split >= len(ranked):
            raise D103HeldFalsifierError("probe 60/40 split is empty")
        train.extend(ranked[:split])
        test.extend(ranked[split:])
    train_rows = np.asarray(sorted(train), dtype=np.int64)
    test_rows = np.asarray(sorted(test), dtype=np.int64)
    if (
        np.intersect1d(physical_ids[train_rows], physical_ids[test_rows]).size
        or train_rows.size + test_rows.size != n
    ):
        raise D103HeldFalsifierError("probe train/test closure failed")
    return train_rows, test_rows


def aggregate_tx_probe_fold(
    capacity_receipts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if len(capacity_receipts) != 9:
        raise D103HeldFalsifierError(
            "TX probe requires 3 logistic and 6 RBF capacity receipts"
        )
    identities: set[str] = set()
    scores: list[float] = []
    normalized: list[dict[str, Any]] = []
    for receipt in capacity_receipts:
        identity = str(receipt.get("capacity_id", ""))
        pooled = float(receipt.get("pooled_ba", float("nan")))
        per_day = [float(value) for value in receipt.get("per_day_ba", ())]
        if (
            not identity
            or identity in identities
            or len(per_day) != 4
            or any(not np.isfinite(value) or value < 0.0 or value > 1.0 for value in [pooled, *per_day])
            or receipt.get("probe_train_test_physical_disjoint") is not True
        ):
            raise D103HeldFalsifierError("invalid TX probe capacity receipt")
        identities.add(identity)
        local = max([pooled, *per_day])
        scores.append(local)
        normalized.append(
            {
                "capacity_id": identity,
                "pooled_ba": pooled,
                "per_day_ba": per_day,
                "capacity_score": local,
            }
        )
    return {
        "capacity_count": len(normalized),
        "capacity_receipts": normalized,
        "fold_score": max(scores),
    }


def run_fixed_tx_probe(
    encoded_rows: np.ndarray,
    receiver_ids: np.ndarray,
    day_ids: np.ndarray,
    labels: np.ndarray,
    physical_ids: np.ndarray,
) -> dict[str, Any]:
    """Run the preregistered attacker capacities on one frozen held receiver."""

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    encoded = np.asarray(encoded_rows, dtype=np.float64)
    receiver_ids = np.asarray(receiver_ids).astype(str)
    day_ids = np.asarray(day_ids).astype(str)
    labels = np.asarray(labels).astype(str)
    physical_ids = np.asarray(physical_ids).astype(str)
    n = len(encoded)
    if (
        encoded.ndim != 2
        or encoded.shape != (n, 32)
        or not np.isfinite(encoded).all()
        or any(values.shape != (n,) for values in (receiver_ids, day_ids, labels, physical_ids))
        or np.unique(receiver_ids).size != 1
        or np.unique(day_ids).size != 4
        or np.unique(labels).size != 6
    ):
        raise D103HeldFalsifierError("TX probe requires one receiver, 4 days, 6 TX, 32D finite rows")
    train_rows, test_rows = probe_partition(
        receiver_ids, day_ids, labels, physical_ids
    )
    x_train = encoded[train_rows]
    x_test = encoded[test_rows]
    y_train = labels[train_rows]
    y_test = labels[test_rows]
    test_days = day_ids[test_rows]

    capacities: list[tuple[str, Any]] = []
    for c_value in (0.1, 1.0, 10.0):
        capacities.append(
            (
                f"logistic-C{c_value:g}",
                make_pipeline(
                    StandardScaler(),
                    LogisticRegression(
                        C=c_value,
                        class_weight="balanced",
                        max_iter=2000,
                        random_state=103713,
                        solver="lbfgs",
                    ),
                ),
            )
        )
    for c_value in (1.0, 10.0):
        for gamma in (0.5 / 32.0, 1.0 / 32.0, 2.0 / 32.0):
            capacities.append(
                (
                    f"rbf-C{c_value:g}-gamma{gamma:.8f}",
                    make_pipeline(
                        StandardScaler(),
                        SVC(
                            C=c_value,
                            gamma=gamma,
                            kernel="rbf",
                            class_weight="balanced",
                            max_iter=2000,
                            random_state=103713,
                        ),
                    ),
                )
            )
    receipts = []
    for capacity_id, model in capacities:
        model.fit(x_train, y_train)
        predicted = np.asarray(model.predict(x_test)).astype(str)
        pooled = float(balanced_accuracy_score(y_test, predicted))
        per_day = []
        for day in sorted(np.unique(day_ids).tolist()):
            local = test_days == day
            if set(y_test[local].tolist()) != set(np.unique(labels).tolist()):
                raise D103HeldFalsifierError("TX probe test day lacks a class")
            per_day.append(float(balanced_accuracy_score(y_test[local], predicted[local])))
        receipts.append(
            {
                "capacity_id": capacity_id,
                "pooled_ba": pooled,
                "per_day_ba": per_day,
                "probe_train_test_physical_disjoint": bool(
                    not set(physical_ids[train_rows].tolist())
                    & set(physical_ids[test_rows].tolist())
                ),
            }
        )
    result = aggregate_tx_probe_fold(receipts)
    result.update(
        {
            "held_receiver": str(receiver_ids[0]),
            "asset_frozen_before_probe": True,
            "probe_state_returned_to_asset": False,
            "train_physical_count": int(train_rows.size),
            "test_physical_count": int(test_rows.size),
        }
    )
    return result


def _performance_key(row: Mapping[str, Any]) -> tuple[str, str, int]:
    receiver = str(row.get("held_receiver", ""))
    held_class = str(row.get("held_class") or "")
    k_shot = int(row.get("K", -1))
    return receiver, held_class, k_shot


def _expected_performance_keys(
    receivers: Sequence[str],
    classes: Sequence[str],
) -> set[tuple[str, str, int]]:
    expected = {
        (receiver, "", k_shot)
        for receiver in receivers
        for k_shot in K_VALUES
    }
    expected.update(
        (receiver, class_id, 1)
        for receiver in receivers
        for class_id in classes
    )
    return expected


def evaluate_complete_gate(
    *,
    receiver_ids: Sequence[str],
    class_ids: Sequence[str],
    performance_rows: Sequence[Mapping[str, Any]],
    day_stability_rows: Sequence[Mapping[str, Any]],
    d102_provenance: Mapping[str, Any],
    tx_probe_rows: Sequence[Mapping[str, Any]],
    quantization_receipt: Mapping[str, Any],
    resource_receipt: Mapping[str, Any],
    access_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    receivers = tuple(sorted({str(value) for value in receiver_ids}))
    classes = tuple(sorted({str(value) for value in class_ids}))
    if len(receivers) != 7 or len(classes) != 6:
        raise D103HeldFalsifierError("gate requires exactly 7 receivers and 6 classes")
    expected_keys = _expected_performance_keys(receivers, classes)
    actual_keys = [_performance_key(row) for row in performance_rows]
    if len(actual_keys) != 63 or set(actual_keys) != expected_keys or len(set(actual_keys)) != 63:
        raise D103HeldFalsifierError("performance coverage must be exactly 63 unique rows")

    reasons: list[str] = []
    expected_outer_keys = {
        (receiver, "") for receiver in receivers
    } | {
        (receiver, class_id)
        for receiver in receivers
        for class_id in classes
    }
    stability_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for row in day_stability_rows:
        key = (str(row.get("held_receiver", "")), str(row.get("held_class") or ""))
        day_norms = [float(value) for value in row.get("day_shift_norms", ())]
        cosines = [float(value) for value in row.get("direction_cosines", ())]
        outer_norm = float(row.get("outer_shift_norm", float("nan")))
        median = float(row.get("direction_cosine_median", float("nan")))
        if (
            key not in expected_outer_keys
            or key in stability_by_key
            or len(day_norms) != 4
            or len(cosines) != 4
            or not np.isfinite([outer_norm, median, *day_norms, *cosines]).all()
            or outer_norm < 1.0e-4
            or any(value < 1.0e-4 for value in day_norms)
            or any(value < -1.0 or value > 1.0 for value in cosines)
            or abs(float(np.median(cosines)) - median) > 1.0e-12
            or row.get("actual_160d_shift_used") is not True
            or row.get("query_rows_used") != 0
        ):
            raise D103HeldFalsifierError("invalid leave-day stability receipt")
        stability_by_key[key] = {
            "outer_shift_norm": outer_norm,
            "day_shift_norms": day_norms,
            "direction_cosines": cosines,
            "direction_cosine_median": median,
        }
    if set(stability_by_key) != expected_outer_keys or len(stability_by_key) != 49:
        raise D103HeldFalsifierError(
            "leave-day stability coverage must be exactly 49 outer folds"
        )

    provenance_keys = {
        "status",
        "fold_count",
        "folds",
        "original_rejected_receipt_sha256",
        "method_lock_sha256",
        "code_sha256",
        "built_before_source_validation_open",
        "target_access",
        "formal_query_access",
    }
    if set(d102_provenance) != provenance_keys:
        raise D103HeldFalsifierError("D102 provenance key closure drift")
    provenance_folds = d102_provenance.get("folds")
    if (
        d102_provenance.get("status")
        != "DIAGNOSTIC_REJECTED_D102_COMPARATOR_NON_PROMOTABLE"
        or d102_provenance.get("fold_count") != 49
        or not isinstance(provenance_folds, list)
        or len(provenance_folds) != 49
        or d102_provenance.get("built_before_source_validation_open") is not True
        or d102_provenance.get("target_access") is not False
        or d102_provenance.get("formal_query_access") is not False
    ):
        raise D103HeldFalsifierError("invalid D102 diagnostic provenance")
    for field in (
        "original_rejected_receipt_sha256",
        "method_lock_sha256",
        "code_sha256",
    ):
        value = str(d102_provenance.get(field, ""))
        if len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
            raise D103HeldFalsifierError(f"invalid D102 provenance SHA: {field}")
    if (
        d102_provenance["method_lock_sha256"] != D102_METHOD_LOCK_SHA256
        or d102_provenance["original_rejected_receipt_sha256"]
        != D102_ORIGINAL_REJECTED_RECEIPT_SHA256
    ):
        raise D103HeldFalsifierError("D102 frozen parent evidence SHA drift")
    d102_by_key: dict[tuple[str, str], str] = {}
    for fold in provenance_folds:
        key = (
            str(fold.get("held_receiver", "")),
            str(fold.get("held_class") or ""),
        )
        root = str(fold.get("bundle_content_root_sha256", ""))
        physical_root = str(fold.get("l_s_physical_root_sha256", ""))
        if (
            key not in expected_outer_keys
            or key in d102_by_key
            or any(
                len(value) != 64
                or any(ch not in "0123456789abcdef" for ch in value)
                for value in (root, physical_root)
            )
            or fold.get("query_rows_used_for_fit") != 0
        ):
            raise D103HeldFalsifierError("invalid D102 fold provenance")
        d102_by_key[key] = root
    if set(d102_by_key) != expected_outer_keys:
        raise D103HeldFalsifierError("D102 provenance must cover 49 outer folds")

    normalized_rows: list[dict[str, Any]] = []
    joint_d103: list[float] = []
    joint_d102: list[float] = []
    truth_open_event_shas: set[str] = set()
    for row in performance_rows:
        key = _performance_key(row)
        values = {
            "base_ba": float(row.get("base_ba", float("nan"))),
            "adapted_ba": float(row.get("adapted_ba", float("nan"))),
            "base_floor": float(row.get("base_floor", float("nan"))),
            "adapted_floor": float(row.get("adapted_floor", float("nan"))),
            "wrong_to_correct": int(row.get("wrong_to_correct", -1)),
            "correct_to_wrong": int(row.get("correct_to_wrong", -1)),
            "joint_score_d102": float(row.get("joint_score_d102", float("nan"))),
            "joint_score_d103": float(row.get("joint_score_d103", float("nan"))),
        }
        if (
            any(not np.isfinite(value) for name, value in values.items() if "correct" not in name)
            or values["wrong_to_correct"] < 0
            or values["correct_to_wrong"] < 0
            or row.get("prediction_artifact_committed_before_truth") is not True
            or row.get("d102_prediction_committed_before_truth") is not True
            or len(str(row.get("truth_open_event_sha256", ""))) != 64
            or any(
                ch not in "0123456789abcdef"
                for ch in str(row.get("truth_open_event_sha256", ""))
            )
            or row.get("d102_comparator_status")
            != "DIAGNOSTIC_REJECTED_D102_COMPARATOR_NON_PROMOTABLE"
            or str(row.get("d102_bundle_content_root_sha256", ""))
            != d102_by_key[(key[0], key[1])]
        ):
            raise D103HeldFalsifierError(f"invalid performance receipt: {key}")
        truth_open_event_shas.add(str(row["truth_open_event_sha256"]))
        delta_ba = values["adapted_ba"] - values["base_ba"]
        delta_floor = values["adapted_floor"] - values["base_floor"]
        net_correct = values["wrong_to_correct"] - values["correct_to_wrong"]
        if delta_ba < -1.0e-12:
            reasons.append(f"BA_NEGATIVE:{key}")
        if delta_floor < -1.0e-12:
            reasons.append(f"FLOOR_NEGATIVE:{key}")
        if net_correct < 0:
            reasons.append(f"NET_CORRECT_NEGATIVE:{key}")

        if key[2] == 1:
            activity = {
                "active": bool(row.get("active", False)),
                "rank": int(row.get("information_rank", -1)),
                "min_sv": float(row.get("min_singular_value", float("nan"))),
                "condition": float(row.get("condition_number", float("nan"))),
                "prior_fraction": float(row.get("prior_fraction", float("nan"))),
                "coefficient_norm": float(row.get("coefficient_norm", float("nan"))),
                "view_agreement": float(row.get("view_top1_agreement", float("nan"))),
                "large_margin_flips": int(row.get("view_large_margin_flip_count", -1)),
                "direction_cosine": float(row.get("direction_cosine_median", float("nan"))),
                "evidence_scope": str(
                    row.get("k1_receipt_evidence_scope", "")
                ),
            }
            stability = stability_by_key[(key[0], key[1])]
            if (
                abs(
                    activity["direction_cosine"]
                    - float(stability["direction_cosine_median"])
                )
                > 1.0e-12
            ):
                raise D103HeldFalsifierError(
                    f"K1 direction receipt does not match leave-day evidence: {key}"
                )
            if not (
                activity["active"]
                and activity["rank"] == 4
                and activity["min_sv"] >= 0.05
                and activity["condition"] <= 10.0
                and activity["prior_fraction"] <= 0.80
                and activity["coefficient_norm"] >= 1.0e-4
                and activity["view_agreement"] >= 0.995
                and activity["large_margin_flips"] == 0
                and activity["direction_cosine"] >= 0.80
                and activity["evidence_scope"]
                == "support_only_no_held_query"
            ):
                reasons.append(f"K1_INACTIVE_OR_UNIDENTIFIED:{key}")

        joint_d102.append(values["joint_score_d102"])
        joint_d103.append(values["joint_score_d103"])
        normalized_rows.append(
            {
                "held_receiver": key[0],
                "held_class": key[1] or None,
                "K": key[2],
                **values,
                "delta_ba": delta_ba,
                "delta_floor": delta_floor,
                "net_correct": net_correct,
            }
        )
    if len(truth_open_event_shas) != 1:
        raise D103HeldFalsifierError(
            "held rows do not share one immutable truth-open event"
        )
    if float(np.mean(joint_d103)) <= float(np.mean(joint_d102)):
        reasons.append("JOINT_SCORE_NOT_STRICTLY_ABOVE_D102")

    probe_by_receiver: dict[str, float] = {}
    for row in tx_probe_rows:
        receiver = str(row.get("held_receiver", ""))
        score = float(row.get("fold_score", float("nan")))
        if (
            receiver not in receivers
            or receiver in probe_by_receiver
            or not np.isfinite(score)
            or score < 0.0
            or score > 1.0
            or row.get("asset_frozen_before_probe") is not True
            or row.get("probe_state_returned_to_asset") is not False
        ):
            raise D103HeldFalsifierError("invalid TX probe fold receipt")
        probe_by_receiver[receiver] = score
    if set(probe_by_receiver) != set(receivers):
        raise D103HeldFalsifierError("TX probe must cover all 7 held receivers")
    tx_mean = float(np.mean(list(probe_by_receiver.values())))
    tx_max = float(np.max(list(probe_by_receiver.values())))
    if tx_mean > TX_PROBE_LIMIT or tx_max > TX_PROBE_LIMIT:
        reasons.append("TX_PROBE_LIMIT_EXCEEDED")

    quant_ok = (
        float(quantization_receipt.get("top1_agreement", 0.0)) >= 0.995
        and int(quantization_receipt.get("large_margin_flip_count", -1)) == 0
        and quantization_receipt.get("persistent_fp_sidecar") is False
        and quantization_receipt.get("learning_arrays_int8_only") is True
    )
    if not quant_ok:
        reasons.append("QUANTIZATION_GATE_FAILED")

    resource_ok = (
        float(resource_receipt.get("total_gpu_hours", float("inf"))) <= GPU_HOUR_LIMIT
        and int(resource_receipt.get("peak_memory_bytes", PEAK_MEMORY_LIMIT + 1))
        <= PEAK_MEMORY_LIMIT
        and int(resource_receipt.get("run_root_bytes", DISK_LIMIT + 1)) <= DISK_LIMIT
        and int(resource_receipt.get("phase2_state_bytes", STATE_LIMIT + 1)) < STATE_LIMIT
        and int(resource_receipt.get("post_backbone_mac_per_query", MAC_LIMIT + 1))
        <= MAC_LIMIT
        and int(resource_receipt.get("completed_fit_count", -1)) == 246
        and int(resource_receipt.get("completed_meta_steps", -1)) == 98_400
    )
    if not resource_ok:
        reasons.append("RESOURCE_GATE_FAILED")

    access_ok = (
        access_receipt.get("protocol_schema") == "p2_min_v1"
        and access_receipt.get("labeled_ratio") == 0.07
        and access_receipt.get("unlabeled_ratio") == 0.63
        and access_receipt.get("source_validation_ratio") == 0.30
        and access_receipt.get("u_s_tx_label_access") is False
        and access_receipt.get("source_validation_gradient_access") is False
        and access_receipt.get("source_validation_asset_access") is False
        and access_receipt.get("target_access") is False
        and access_receipt.get("formal_query_access") is False
        and access_receipt.get("query_fit_rows") == 0
        and access_receipt.get("derived_from_fit_access_receipt_count") == 246
        and access_receipt.get("all_fit_manifests_identity_bound") is True
    )
    if not access_ok:
        reasons.append("ACCESS_GATE_FAILED")

    status = "PHASE1_HELD_ACCEPT" if not reasons else "PHASE1_HELD_REJECT"
    return {
        "schema": SCHEMA,
        "candidate_id": CANDIDATE_ID,
        "status": status,
        "target25_gate_eligible": status == "PHASE1_HELD_ACCEPT",
        "target25_authorized": False,
        "independent_release_review_required": True,
        "claim_semantics": "SOURCE_HELD_LIFECYCLE_PROXY_NON_PERFORMANCE_UNTIL_TARGET",
        "performance_row_count": len(normalized_rows),
        "performance_rows": normalized_rows,
        "leave_day_stability": {
            "outer_fold_count": len(stability_by_key),
            "minimum_direction_cosine_median": float(
                min(
                    value["direction_cosine_median"]
                    for value in stability_by_key.values()
                )
            ),
        },
        "d102_comparator": {
            "status": d102_provenance["status"],
            "fold_count": len(d102_by_key),
            "original_rejected_receipt_sha256": d102_provenance[
                "original_rejected_receipt_sha256"
            ],
            "method_lock_sha256": d102_provenance["method_lock_sha256"],
            "code_sha256": d102_provenance["code_sha256"],
        },
        "tx_probe": {
            "fold_count": len(probe_by_receiver),
            "mean_fold_score": tx_mean,
            "max_fold_score": tx_max,
            "limit": TX_PROBE_LIMIT,
        },
        "joint_score": {
            "mean_d102": float(np.mean(joint_d102)),
            "mean_d103": float(np.mean(joint_d103)),
        },
        "quantization_gate_pass": quant_ok,
        "resource_gate_pass": resource_ok,
        "access_gate_pass": access_ok,
        "rejection_reasons": sorted(set(reasons)),
    }


def canonical_json_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


__all__ = [
    "CANDIDATE_ID",
    "D103HeldFalsifierError",
    "HeldFitSpec",
    "SCHEMA",
    "aggregate_tx_probe_fold",
    "build_complete_fit_plan",
    "canonical_json_sha256",
    "evaluate_complete_gate",
    "probe_partition",
    "run_fixed_tx_probe",
]
