"""Source-only receiver-held/class-LOCO falsifier for D102 MetaBias4."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence

import numpy as np

from .phase1_rb_metabias4_bundle import (
    RBMetaBias4Config,
    RBMetaBias4BundleError,
    _tap_arrays,
    apply_metabias4,
    build_phase1_rb_metabias4_bundle,
    infer_metabias4_coefficient,
)
from .stage2_zid_student_t_qknn import (
    Phase1ZIDStudentTLock,
    build_typed_zid_support_bank,
    identity_shared_psd_metric,
    score_zid_student_t_logits,
)


SCHEMA = "cvs.phase1.rb_metabias4.held_falsifier.v1"
K_VALUES = (1, 5, 10)


class RBMetaBias4HeldError(ValueError):
    """Raised when held evidence cannot be constructed without leakage."""


def _canon(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _sha(value: Any) -> str:
    return hashlib.sha256(_canon(value)).hexdigest()


def _qknn_lock(k_shot: int) -> Phase1ZIDStudentTLock:
    receipt = {
        "schema": SCHEMA + ".qknn_lock",
        "K": k_shot,
        "phase1_receiver_held_only": True,
        "target_access": False,
        "query_rows_used_for_fit": 0,
    }
    return Phase1ZIDStudentTLock(
        k_shot,
        3.0,
        160,
        1.0,
        0.2,
        2.0,
        0.5,
        2.0,
        1.0,
        _sha({"kind": "held", **receipt}),
        _sha({"kind": "quantization", **receipt}),
    )


def _unit_relu(rows: np.ndarray) -> np.ndarray:
    value = np.maximum(np.asarray(rows, dtype=np.float64), 0.0)
    norms = np.linalg.norm(value, axis=1, keepdims=True)
    if np.any(norms <= 1.0e-12):
        raise RBMetaBias4HeldError("held pre_relu contains zero ReLU feature")
    return np.asarray(value / norms, dtype=np.float32)


def _balanced_accuracy(predicted: np.ndarray, truth: np.ndarray, count: int) -> tuple[float, float]:
    per_class = []
    for index in range(count):
        local = truth == index
        if not np.any(local):
            raise RBMetaBias4HeldError("held query lacks a class")
        per_class.append(float(np.mean(predicted[local] == index)))
    return float(np.mean(per_class)), float(np.min(per_class))


def _score(
    support: np.ndarray,
    support_labels: Sequence[str],
    query: np.ndarray,
    classes: tuple[str, ...],
    k_shot: int,
) -> tuple[np.ndarray, np.ndarray]:
    lock = _qknn_lock(k_shot)
    bank = build_typed_zid_support_bank(
        np.asarray(support, dtype=np.float32),
        tuple(support_labels),
        classes,
        config=lock,
    )
    metric = identity_shared_psd_metric(config=lock)
    logits = score_zid_student_t_logits(
        bank, np.asarray(query, dtype=np.float32), metric=metric
    )
    return logits, np.argmax(logits, axis=1)


def _select_fold(
    arrays: Mapping[str, np.ndarray],
    held_receiver: str,
    classes: tuple[str, ...],
    k_shot: int,
) -> tuple[np.ndarray, np.ndarray]:
    support: list[int] = []
    query: list[int] = []
    for class_id in classes:
        indices = np.flatnonzero(
            (arrays["receiver_ids"] == held_receiver) & (arrays["labels"] == class_id)
        )
        ordered = sorted(indices.tolist(), key=lambda i: str(arrays["physical_ids"][i]))
        if len(ordered) < k_shot + 1:
            raise RBMetaBias4HeldError(
                f"receiver {held_receiver} class lacks K{k_shot} support plus query"
            )
        support.extend(ordered[:k_shot])
        query.extend(ordered[k_shot:])
    support_array = np.asarray(support, dtype=np.int64)
    query_array = np.asarray(query, dtype=np.int64)
    if set(arrays["physical_ids"][support_array].tolist()) & set(
        arrays["physical_ids"][query_array].tolist()
    ):
        raise RBMetaBias4HeldError("held support/query physical overlap")
    return support_array, query_array


def _evaluate_receiver_fold(
    arrays: Mapping[str, np.ndarray],
    held_receiver: str,
    *,
    checkpoint_sha256: str,
    runtime_sha256: str,
    method_lock_sha256: str,
    config: RBMetaBias4Config,
    k_shot: int,
    excluded_classes: Sequence[str] = (),
) -> dict[str, Any]:
    bundle = build_phase1_rb_metabias4_bundle(
        arrays,
        checkpoint_sha256=checkpoint_sha256,
        runtime_sha256=runtime_sha256,
        method_lock_sha256=method_lock_sha256,
        config=config,
        excluded_receivers=(held_receiver,),
        excluded_classes=excluded_classes,
    )
    classes = tuple(sorted(set(arrays["labels"].tolist())))
    support, query = _select_fold(arrays, held_receiver, classes, k_shot)
    coefficient, solve = infer_metabias4_coefficient(
        bundle, arrays["z_dom"][support], arrays["labels"][support].tolist()
    )
    base_support = _unit_relu(arrays["pre_relu"][support])
    base_query = _unit_relu(arrays["pre_relu"][query])
    da_support = apply_metabias4(bundle, arrays["pre_relu"][support], coefficient)
    da_query = apply_metabias4(bundle, arrays["pre_relu"][query], coefficient)
    base_logits, base_pred = _score(
        base_support, arrays["labels"][support].tolist(), base_query, classes, k_shot
    )
    da_logits, da_pred = _score(
        da_support, arrays["labels"][support].tolist(), da_query, classes, k_shot
    )
    truth = np.asarray([classes.index(value) for value in arrays["labels"][query]])
    base_ba, base_floor = _balanced_accuracy(base_pred, truth, len(classes))
    da_ba, da_floor = _balanced_accuracy(da_pred, truth, len(classes))
    wrong_correct = int(np.sum((base_pred != truth) & (da_pred == truth)))
    correct_wrong = int(np.sum((base_pred == truth) & (da_pred != truth)))
    split = max(1, len(classes) // 2)
    group_a = truth < split
    group_b = ~group_a
    group_a_net = int(
        np.sum(
            (base_pred[group_a] != truth[group_a])
            & (da_pred[group_a] == truth[group_a])
        )
        - np.sum(
            (base_pred[group_a] == truth[group_a])
            & (da_pred[group_a] != truth[group_a])
        )
    )
    group_b_net = int(
        np.sum(
            (base_pred[group_b] != truth[group_b])
            & (da_pred[group_b] == truth[group_b])
        )
        - np.sum(
            (base_pred[group_b] == truth[group_b])
            & (da_pred[group_b] != truth[group_b])
        )
    )
    base_margin = np.partition(base_logits, -2, axis=1)[:, -1] - np.partition(
        base_logits, -2, axis=1
    )[:, -2]
    da_margin = np.partition(da_logits, -2, axis=1)[:, -1] - np.partition(
        da_logits, -2, axis=1
    )[:, -2]
    base_mask = arrays["pre_relu"][query] > 0.0
    shifted = arrays["pre_relu"][query] + (
        bundle.basis().astype(np.float64) @ coefficient.astype(np.float64)
    )[None, :]
    da_mask = shifted > 0.0
    return {
        "held_receiver_token_sha256": _sha({"receiver": held_receiver}),
        "K": k_shot,
        "support_rows": int(len(support)),
        "query_rows": int(len(query)),
        "support_query_physical_disjoint": True,
        "query_rows_used_for_fit": 0,
        "base_balanced_accuracy": base_ba,
        "adapted_balanced_accuracy": da_ba,
        "base_floor": base_floor,
        "adapted_floor": da_floor,
        "wrong_to_correct": wrong_correct,
        "correct_to_wrong": correct_wrong,
        "net_correct": wrong_correct - correct_wrong,
        "anonymous_group_a_net_correct": group_a_net,
        "anonymous_group_b_net_correct": group_b_net,
        "margin_changed_rows": int(np.count_nonzero(base_margin != da_margin)),
        "argmax_changed_rows": int(np.count_nonzero(base_pred != da_pred)),
        "relu_mask_flip_count": int(np.count_nonzero(base_mask != da_mask)),
        "coefficient_receipt": solve,
        "bundle_content_root_sha256": bundle.content_root_sha256,
        "excluded_class_count": len(tuple(excluded_classes)),
    }


def _ridge_probe_accuracy(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    class_count: int,
) -> float:
    x = np.asarray(train_x, dtype=np.float64)
    y = np.asarray(train_y, dtype=np.int64)
    one_hot = np.eye(class_count, dtype=np.float64)[y]
    ridge = 1.0e-2
    weight = np.linalg.solve(x.T @ x + ridge * np.eye(x.shape[1]), x.T @ one_hot)
    predicted = np.argmax(np.asarray(test_x, dtype=np.float64) @ weight, axis=1)
    values = [
        np.mean(predicted[test_y == index] == index)
        for index in range(class_count)
        if np.any(test_y == index)
    ]
    return float(np.mean(values))


def _tx_leakage_receipt(
    arrays: Mapping[str, np.ndarray],
    *,
    checkpoint_sha256: str,
    runtime_sha256: str,
    method_lock_sha256: str,
    config: RBMetaBias4Config,
) -> dict[str, Any]:
    classes = tuple(sorted(set(arrays["labels"].tolist())))
    receivers = tuple(sorted(set(arrays["receiver_ids"].tolist())))
    values = []
    for held in receivers:
        bundle = build_phase1_rb_metabias4_bundle(
            arrays,
            checkpoint_sha256=checkpoint_sha256,
            runtime_sha256=runtime_sha256,
            method_lock_sha256=method_lock_sha256,
            config=config,
            excluded_receivers=(held,),
        )
        u = bundle.domain_encoder().astype(np.float64)
        train = arrays["receiver_ids"] != held
        test = ~train
        train_r = arrays["z_dom"][train].astype(np.float64) @ u.T
        test_r = arrays["z_dom"][test].astype(np.float64) @ u.T
        train_r /= np.maximum(np.linalg.norm(train_r, axis=1, keepdims=True), 1.0e-12)
        test_r /= np.maximum(np.linalg.norm(test_r, axis=1, keepdims=True), 1.0e-12)
        train_y = np.asarray([classes.index(value) for value in arrays["labels"][train]])
        test_y = np.asarray([classes.index(value) for value in arrays["labels"][test]])
        values.append(
            {
                "held_receiver_token_sha256": _sha({"receiver": held}),
                "balanced_accuracy": _ridge_probe_accuracy(
                    train_r, train_y, test_r, test_y, len(classes)
                ),
            }
        )
    mean = float(np.mean([row["balanced_accuracy"] for row in values]))
    maximum = float(np.max([row["balanced_accuracy"] for row in values]))
    return {
        "schema": SCHEMA + ".tx_leakage",
        "receiver_held": values,
        "mean_balanced_accuracy": mean,
        "maximum_balanced_accuracy": maximum,
        "maximum_allowed": 0.25,
        "passed": maximum <= 0.25,
        "raw_zdom_used_for_bank_matching": False,
    }


def _label_permutation_receipt(
    arrays: Mapping[str, np.ndarray],
    *,
    checkpoint_sha256: str,
    runtime_sha256: str,
    method_lock_sha256: str,
    config: RBMetaBias4Config,
) -> dict[str, Any]:
    original = build_phase1_rb_metabias4_bundle(
        arrays,
        checkpoint_sha256=checkpoint_sha256,
        runtime_sha256=runtime_sha256,
        method_lock_sha256=method_lock_sha256,
        config=config,
    )
    classes = tuple(sorted(set(arrays["labels"].tolist())))
    mapping = dict(zip(classes, reversed(classes)))
    changed = {name: np.array(value, copy=True) for name, value in arrays.items()}
    changed["labels"] = np.asarray([mapping[value] for value in arrays["labels"]], dtype=np.str_)
    changed["class_ids"] = np.asarray([mapping[value] for value in arrays["class_ids"]], dtype=np.str_)
    permuted = build_phase1_rb_metabias4_bundle(
        changed,
        checkpoint_sha256=checkpoint_sha256,
        runtime_sha256=runtime_sha256,
        method_lock_sha256=method_lock_sha256,
        config=config,
    )
    fields = (
        "basis_codes_qint8",
        "basis_scales_fp16",
        "domain_encoder_codes_qint8",
        "domain_encoder_scales_fp16",
        "bank_g_codes_qint8",
        "bank_g_scales_fp16",
        "bank_t_codes_qint8",
        "bank_t_scales_fp16",
        "bank_precision_diag_fp16",
        "bank_sigma_fp16",
    )
    equivalent = all(
        np.array_equal(getattr(original, name), getattr(permuted, name))
        for name in fields
    )
    return {
        "schema": SCHEMA + ".label_permutation",
        "consistent_permutation_equivalent": equivalent,
        "compared_numeric_fields": list(fields),
        "class_handles_in_payload": False,
    }


def run_rb_metabias4_phase1_held_falsifier(
    tap_archive: Mapping[str, np.ndarray],
    *,
    checkpoint_sha256: str,
    runtime_sha256: str,
    method_lock_sha256: str,
    config: RBMetaBias4Config = RBMetaBias4Config(),
) -> dict[str, Any]:
    """Run deterministic Phase1-only gates; never opens target or formal query."""

    try:
        arrays = _tap_arrays(tap_archive)
    except RBMetaBias4BundleError as error:
        raise RBMetaBias4HeldError(str(error)) from error
    receivers = tuple(sorted(set(arrays["receiver_ids"].tolist())))
    classes = tuple(sorted(set(arrays["labels"].tolist())))
    if len(receivers) < 3 or len(classes) < 3:
        raise RBMetaBias4HeldError("held falsifier requires at least three receivers/classes")
    receiver_rows = [
        _evaluate_receiver_fold(
            arrays,
            held,
            checkpoint_sha256=checkpoint_sha256,
            runtime_sha256=runtime_sha256,
            method_lock_sha256=method_lock_sha256,
            config=config,
            k_shot=k,
        )
        for held in receivers
        for k in K_VALUES
    ]
    loco_rows = [
        _evaluate_receiver_fold(
            arrays,
            held_receiver,
            checkpoint_sha256=checkpoint_sha256,
            runtime_sha256=runtime_sha256,
            method_lock_sha256=method_lock_sha256,
            config=config,
            k_shot=1,
            excluded_classes=(class_id,),
        )
        for class_id in classes
        for held_receiver in receivers
    ]
    tx = _tx_leakage_receipt(
        arrays,
        checkpoint_sha256=checkpoint_sha256,
        runtime_sha256=runtime_sha256,
        method_lock_sha256=method_lock_sha256,
        config=config,
    )
    permutation = _label_permutation_receipt(
        arrays,
        checkpoint_sha256=checkpoint_sha256,
        runtime_sha256=runtime_sha256,
        method_lock_sha256=method_lock_sha256,
        config=config,
    )
    by_k = {}
    for k in K_VALUES:
        local = [row for row in receiver_rows if row["K"] == k]
        by_k[str(k)] = {
            "receiver_count": len(local),
            "mean_net_correct": float(np.mean([row["net_correct"] for row in local])),
            "anonymous_group_a_net_correct": int(
                sum(row["anonymous_group_a_net_correct"] for row in local)
            ),
            "anonymous_group_b_net_correct": int(
                sum(row["anonymous_group_b_net_correct"] for row in local)
            ),
            "wrong_to_correct": int(sum(row["wrong_to_correct"] for row in local)),
            "correct_to_wrong": int(sum(row["correct_to_wrong"] for row in local)),
            "mean_floor_delta": float(
                np.mean([row["adapted_floor"] - row["base_floor"] for row in local])
            ),
            "mask_flip_count": int(sum(row["relu_mask_flip_count"] for row in local)),
            "argmax_changed_rows": int(sum(row["argmax_changed_rows"] for row in local)),
        }
    k1 = by_k["1"]
    performance_pass = bool(
        k1["anonymous_group_a_net_correct"] >= 0
        and k1["anonymous_group_b_net_correct"] >= 0
        and (k1["mean_net_correct"] > 0 or k1["mean_floor_delta"] > 0.0)
        and all(
            by_k[str(k)]["anonymous_group_a_net_correct"] >= 0
            and by_k[str(k)]["anonymous_group_b_net_correct"] >= 0
            and by_k[str(k)]["wrong_to_correct"] > by_k[str(k)]["correct_to_wrong"]
            for k in (5, 10)
        )
    )
    loco_gate_passed = bool(
        all(
            row["adapted_balanced_accuracy"] >= row["base_balanced_accuracy"]
            for row in loco_rows
        )
    )
    all_gates_passed = bool(
        performance_pass
        and loco_gate_passed
        and tx["passed"]
        and permutation["consistent_permutation_equivalent"]
    )
    report = {
        "schema": SCHEMA,
        "status": (
            "PHASE1_HELD_FALSIFIER_PASS"
            if all_gates_passed
            else "PHASE1_HELD_FALSIFIER_REJECT"
        ),
        "target_access": False,
        "formal_target_query_access": False,
        "phase1_pseudoquery_truth_used_for_scoring_only": True,
        "anonymous_class_partition_semantics": (
            "deterministic class-order halves used only as a symmetry falsifier; "
            "not Phase2 old/new lifecycle evidence"
        ),
        "phase2_old_new_claimed": False,
        "deterministic_seed": config.deterministic_seed,
        "config_digest_sha256": config.digest,
        "receiver_held_complete_k1_k5_k10": len(receiver_rows) == len(receivers) * 3,
        "receiver_held_rows": receiver_rows,
        "class_loco_complete": len(loco_rows) == len(classes) * len(receivers),
        "class_loco_rows": loco_rows,
        "class_loco_gate_definition": (
            "every excluded-class x held-receiver K1 fold requires "
            "adapted balanced accuracy >= raw qKNN balanced accuracy"
        ),
        "class_loco_gate_passed": loco_gate_passed,
        "tx_leakage_receipt": tx,
        "label_permutation_receipt": permutation,
        "performance_summary_by_k": by_k,
        "performance_gate_passed": performance_pass,
        "all_gates_passed": all_gates_passed,
        "all_support_query_physical_disjoint": all(
            row["support_query_physical_disjoint"] for row in receiver_rows + loco_rows
        ),
        "all_query_rows_used_for_fit_zero": all(
            row["query_rows_used_for_fit"] == 0 for row in receiver_rows + loco_rows
        ),
        "claim_boundary": "PHASE1_SOURCE_ONLY_NOT_TARGET_PERFORMANCE",
    }
    report["receipt_sha256"] = _sha(report)
    return report


__all__ = [
    "K_VALUES",
    "RBMetaBias4HeldError",
    "SCHEMA",
    "run_rb_metabias4_phase1_held_falsifier",
]
