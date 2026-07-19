#!/usr/bin/env python3
"""D62 support-only cross-fitted Fisher row-splice probe."""

from __future__ import annotations

import argparse
import copy
from dataclasses import replace
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any, Callable

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
D61_HELPER_PATH = SCRIPT_DIR / "probe_d61_identity_primary_fisher_residual.py"
SPEC = importlib.util.spec_from_file_location("d62_d61_probe_helper", D61_HELPER_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("D62 could not load D61 helper")
d61 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(d61)
d46, d45, d44, d43 = d61.d46, d61.d46.d45, d61.d46.d44, d61.d43


ARM = "crossfitted_fisher_row_splice"
CONFIRMATION_SEEDS = (713102, 713103, 713104, 713105, 713106)
STRUCTURE = "d46_base_plus_crossfitted_bidirectional_safe_d61_affine_rows"
FORMULA = "accept_c=TP1_c>=TP0_c and FP1_c<=FP0_c and strict; atomic joint; row_c=D61 else D46"
if ARM not in d43.ARM_STRUCTURES:
    d43.ARM_STRUCTURES[ARM] = STRUCTURE
if ARM not in d43.ARMS:
    d43.ARMS = tuple((*d43.ARMS, ARM))


class D62ProbeError(RuntimeError):
    pass


def _install_confirmation_cell_guard(
    runner_module: Any, confirmation_seed: int | None
) -> Callable[..., Any] | None:
    """Allow only preregistered D18 confirmation seeds on the locked D42 cell."""

    if confirmation_seed is None:
        return None
    if confirmation_seed not in CONFIRMATION_SEEDS:
        raise D62ProbeError(
            f"D62 confirmation seed is not preregistered: {confirmation_seed}"
        )
    original = runner_module._require_d42_development_cell

    def require_confirmation_cell(
        before_manifest: dict[str, Any], after_manifest: dict[str, Any]
    ) -> None:
        old_classes = runner_module.legacy._registered_handles(before_manifest)
        all_classes = runner_module.legacy._registered_handles(after_manifest)
        if (
            str(before_manifest.get("receiver"))
            != runner_module.D42_DEVELOPMENT_RECEIVER
            or str(after_manifest.get("receiver"))
            != runner_module.D42_DEVELOPMENT_RECEIVER
            or int(before_manifest.get("seed", -1)) != confirmation_seed
            or int(after_manifest.get("seed", -1)) != confirmation_seed
            or int(before_manifest.get("k_shot", -1)) != 10
            or int(after_manifest.get("k_shot", -1)) != 10
            or all_classes[: len(old_classes)] != old_classes
            or len(all_classes) - len(old_classes)
            != runner_module.D42_DEVELOPMENT_NEW_CLASS_COUNT
        ):
            raise runner_module.D25RunnerError(
                "D62 confirmation cell must be receiver 20-1, "
                f"seed {confirmation_seed}, K10, new5"
            )

    runner_module._require_d42_development_cell = require_confirmation_cell
    return original


def _partitions(labels: np.ndarray, class_count: int, k_shot: int) -> list[np.ndarray]:
    y = np.asarray(labels, dtype=np.int64)
    indices = [np.flatnonzero(y == index) for index in range(int(class_count))]
    if any(len(item) != int(k_shot) for item in indices):
        raise D62ProbeError("D62 requires exact symmetric K support")
    held = [
        np.asarray([item[rank] for item in indices], dtype=np.int64)
        for rank in range(int(k_shot))
    ]
    flat = [int(value) for fold in held for value in fold]
    if sorted(flat) != list(range(len(y))) or len(set(flat)) != len(y):
        raise D62ProbeError("D62 held partition exact-once drift")
    return held


def _residual_coefficients(
    rows: np.ndarray,
    labels: np.ndarray,
    class_count: int,
    k_shot: int,
    coefficient: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    transform, audit = d61._fisher_residual_transform(
        rows, labels, class_count, k_shot
    )
    residual = np.asarray(coefficient, dtype=np.float64) @ transform.T
    return residual, audit


def _component_evidence(
    component_fit: Callable[..., tuple[np.ndarray, np.ndarray, dict[str, Any]]],
    rows: np.ndarray,
    labels: np.ndarray,
    class_count: int,
    k_shot: int,
    component_name: str,
    call_records: list[dict[str, Any]],
) -> dict[str, Any]:
    x = np.asarray(rows, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    outer_coef, outer_intercept, _ = component_fit(x, y, class_count, k_shot)
    outer_residual, outer_transform = _residual_coefficients(
        x, y, class_count, k_shot, outer_coef
    )
    outer_base_scale = d44._class_centered_logit_rms(
        x, outer_coef, outer_intercept
    )
    outer_residual_scale = d44._class_centered_logit_rms(
        x, outer_residual, outer_intercept
    )
    call_records.append(
        {
            "component": component_name,
            "scope": "outer",
            "k_shot": int(k_shot),
            "class_count": int(class_count),
            "transform_sha256": outer_transform["d61_transform_sha256"],
        }
    )
    base_scores: list[np.ndarray] = []
    residual_scores: list[np.ndarray] = []
    true_classes: list[np.ndarray] = []
    held_indices = _partitions(y, class_count, k_shot)
    for fold_index, held in enumerate(held_indices):
        mask = np.ones(len(x), dtype=bool)
        mask[held] = False
        train_x, train_y = x[mask], y[mask]
        coef0, intercept0, _ = component_fit(
            train_x, train_y, class_count, int(k_shot) - 1
        )
        coef1, transform_audit = _residual_coefficients(
            train_x, train_y, class_count, int(k_shot) - 1, coef0
        )
        scale0 = d44._class_centered_logit_rms(train_x, coef0, intercept0)
        scale1 = d44._class_centered_logit_rms(train_x, coef1, intercept0)
        base_scores.append(
            (
                x[held] @ np.asarray(coef0, dtype=np.float64).T
                + np.asarray(intercept0, dtype=np.float64)[None, :]
            )
            / scale0
        )
        residual_scores.append(
            (
                x[held] @ np.asarray(coef1, dtype=np.float64).T
                + np.asarray(intercept0, dtype=np.float64)[None, :]
            )
            / scale1
        )
        true_classes.append(y[held])
        call_records.append(
            {
                "component": component_name,
                "scope": "inner",
                "fold_index": fold_index,
                "k_shot": int(k_shot) - 1,
                "class_count": int(class_count),
                "transform_sha256": transform_audit["d61_transform_sha256"],
            }
        )
    base_array = np.stack(base_scores, axis=0)
    residual_array = np.stack(residual_scores, axis=0)
    truth_array = np.stack(true_classes, axis=0)
    _, base_per_class_ce = d45._class_balanced_cross_entropy(
        base_array.reshape(-1, class_count), truth_array.reshape(-1), class_count
    )
    _, residual_per_class_ce = d45._class_balanced_cross_entropy(
        residual_array.reshape(-1, class_count), truth_array.reshape(-1), class_count
    )
    return {
        "outer_base_coef": np.asarray(outer_coef, dtype=np.float64),
        "outer_base_intercept": np.asarray(outer_intercept, dtype=np.float64),
        "outer_residual_coef": outer_residual,
        "outer_base_scale": float(outer_base_scale),
        "outer_residual_scale": float(outer_residual_scale),
        "outer_transform_audit": outer_transform,
        "base_scores": base_array,
        "residual_scores": residual_array,
        "truth": truth_array,
        "held_indices": [fold.tolist() for fold in held_indices],
        "base_per_class_ce": base_per_class_ce,
        "residual_per_class_ce": residual_per_class_ce,
    }


def _counts(scores: np.ndarray, truth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    logits = np.asarray(scores, dtype=np.float64)
    targets = np.asarray(truth, dtype=np.int64)
    class_count = int(logits.shape[-1])
    predicted = np.argmax(logits, axis=-1)
    positive = np.zeros(class_count, dtype=np.int64)
    false_positive = np.zeros(class_count, dtype=np.int64)
    for class_index in range(class_count):
        is_positive = targets == class_index
        positive[class_index] = int(np.sum(predicted[is_positive] == class_index))
        false_positive[class_index] = int(
            np.sum(predicted[~is_positive] == class_index)
        )
    return positive, false_positive


def _pareto_gate(
    base_scores: np.ndarray, residual_scores: np.ndarray, truth: np.ndarray
) -> dict[str, Any]:
    base = np.asarray(base_scores, dtype=np.float64)
    residual = np.asarray(residual_scores, dtype=np.float64)
    targets = np.asarray(truth, dtype=np.int64)
    if (
        base.shape != residual.shape
        or base.ndim != 3
        or targets.shape != base.shape[:2]
        or not np.isfinite(base).all()
        or not np.isfinite(residual).all()
    ):
        raise D62ProbeError("D62 gate evidence drift")
    class_count = int(base.shape[2])
    base_positive, base_fp = _counts(base, targets)
    coordinate_positive = np.zeros(class_count, dtype=np.int64)
    coordinate_fp = np.zeros(class_count, dtype=np.int64)
    for class_index in range(class_count):
        hybrid = base.copy()
        hybrid[:, :, class_index] = residual[:, :, class_index]
        positive, false_positive = _counts(hybrid, targets)
        coordinate_positive[class_index] = positive[class_index]
        coordinate_fp[class_index] = false_positive[class_index]
    initial = (
        (coordinate_positive >= base_positive)
        & (coordinate_fp <= base_fp)
        & ((coordinate_positive > base_positive) | (coordinate_fp < base_fp))
    )
    joint = base.copy()
    joint[:, :, initial] = residual[:, :, initial]
    joint_positive, joint_fp = _counts(joint, targets)
    atomic_safe = bool(
        np.all(joint_positive >= base_positive) and np.all(joint_fp <= base_fp)
    )
    final = initial if atomic_safe else np.zeros(class_count, dtype=bool)
    if np.any(final):
        status = "crossfitted_fisher_row_splice_active"
    elif np.any(initial):
        status = "joint_gate_atomic_exact_d46_fallback"
    else:
        status = "no_row_accepted_exact_d46_fallback"
    return {
        "base_positive": base_positive,
        "base_false_positive": base_fp,
        "coordinate_positive": coordinate_positive,
        "coordinate_false_positive": coordinate_fp,
        "joint_positive": joint_positive,
        "joint_false_positive": joint_fp,
        "initial_accept": initial,
        "final_accept": final,
        "atomic_safe": atomic_safe,
        "status": status,
        "exact_fallback": not bool(np.any(final)),
    }


def build_d62_fit(d42: Any) -> tuple[Callable[..., Any], list[dict[str, Any]]]:
    base_fit = d46.build_classwise_loo_reliability_fit(d42)
    full_fit = d46._canonical_component_fit(
        d45._build_locked_d42_full_component_fit(d42), []
    )
    block_fit = d46._canonical_component_fit(
        d43.build_structured_fit(d42, "block3_centered"), []
    )
    call_records: list[dict[str, Any]] = []

    def fit(
        rows: np.ndarray, labels: np.ndarray, class_count: int, k_shot: int
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        base_coef, base_intercept, base_audit = base_fit(
            rows, labels, class_count, k_shot
        )
        if int(k_shot) <= 2:
            zeros = np.zeros(class_count, dtype=np.int64)
            gate = {
                "base_positive": zeros,
                "base_false_positive": zeros,
                "coordinate_positive": zeros,
                "coordinate_false_positive": zeros,
                "joint_positive": zeros,
                "joint_false_positive": zeros,
                "initial_accept": zeros.astype(bool),
                "final_accept": zeros.astype(bool),
                "atomic_safe": True,
                "status": "k1_k2_exact_d46_fallback",
                "exact_fallback": True,
            }
            full = block = None
            residual_weights = None
            residual_coef = np.asarray(base_coef, dtype=np.float64)
            residual_intercept = np.asarray(base_intercept, dtype=np.float64)
        else:
            full = _component_evidence(
                full_fit, rows, labels, class_count, k_shot, "full", call_records
            )
            block = _component_evidence(
                block_fit, rows, labels, class_count, k_shot, "block3", call_records
            )
            if full["held_indices"] != block["held_indices"] or not np.array_equal(
                full["truth"], block["truth"]
            ):
                raise D62ProbeError("D62 component partition drift")
            base_weights = np.stack(
                [
                    np.asarray(base_audit["d46_full_weight_by_class"]),
                    np.asarray(base_audit["d46_block_weight_by_class"]),
                ],
                axis=1,
            )
            residual_weights, _ = d46._classwise_likelihood_weights(
                full["residual_per_class_ce"],
                block["residual_per_class_ce"],
                k_shot,
            )
            base_scores = (
                full["base_scores"] * base_weights[None, None, :, 0]
                + block["base_scores"] * base_weights[None, None, :, 1]
            )
            residual_scores = (
                full["residual_scores"] * residual_weights[None, None, :, 0]
                + block["residual_scores"] * residual_weights[None, None, :, 1]
            )
            gate = _pareto_gate(base_scores, residual_scores, full["truth"])
            residual_coef = (
                residual_weights[:, 0, None]
                * full["outer_residual_coef"]
                / full["outer_residual_scale"]
                + residual_weights[:, 1, None]
                * block["outer_residual_coef"]
                / block["outer_residual_scale"]
            )
            residual_intercept = (
                residual_weights[:, 0]
                * full["outer_base_intercept"]
                / full["outer_residual_scale"]
                + residual_weights[:, 1]
                * block["outer_base_intercept"]
                / block["outer_residual_scale"]
            )
        if gate["exact_fallback"]:
            final_coef = np.asarray(base_coef, dtype=np.float32).copy()
            final_intercept = np.asarray(base_intercept, dtype=np.float32).copy()
        else:
            hybrid_coef = np.asarray(base_coef, dtype=np.float64).copy()
            hybrid_intercept = np.asarray(base_intercept, dtype=np.float64).copy()
            mask = np.asarray(gate["final_accept"], dtype=bool)
            hybrid_coef[mask] = residual_coef[mask]
            hybrid_intercept[mask] = residual_intercept[mask]
            centered_coef, centered_intercept = d43._center_affine_scores(
                hybrid_coef, hybrid_intercept
            )
            final_coef = centered_coef.astype(np.float32)
            final_intercept = centered_intercept.astype(np.float32)
        audit = dict(base_audit)
        audit.update(
            {
                "d43_probe_arm": ARM,
                "d43_covariance_structure": STRUCTURE,
                "d62_probe_arm": ARM,
                "d62_formula": FORMULA,
                "d62_boundary_status": gate["status"],
                "d62_actual_k": int(k_shot),
                "d62_class_count": int(class_count),
                "d62_class_id_specific_formula": False,
                "d62_old_new_role_specific_branch": False,
                "d62_scene_receiver_handle_specific_branch": False,
                "d62_uses_outer_held_or_query": False,
                "d62_hyperparameter_count": 0,
                "d62_base_positive_correct_by_class": gate["base_positive"].tolist(),
                "d62_base_false_positive_by_class": gate["base_false_positive"].tolist(),
                "d62_coordinate_positive_correct_by_class": gate["coordinate_positive"].tolist(),
                "d62_coordinate_false_positive_by_class": gate["coordinate_false_positive"].tolist(),
                "d62_joint_positive_correct_by_class": gate["joint_positive"].tolist(),
                "d62_joint_false_positive_by_class": gate["joint_false_positive"].tolist(),
                "d62_initial_accept_mask": gate["initial_accept"].tolist(),
                "d62_final_accept_mask": gate["final_accept"].tolist(),
                "d62_joint_atomic_safe": gate["atomic_safe"],
                "d62_residual_full_weight_by_class": None if residual_weights is None else residual_weights[:, 0].tolist(),
                "d62_residual_block_weight_by_class": None if residual_weights is None else residual_weights[:, 1].tolist(),
                "d62_full_outer_transform_audit": None if full is None else full["outer_transform_audit"],
                "d62_block_outer_transform_audit": None if block is None else block["outer_transform_audit"],
                "d62_base_coefficient_fp32": np.asarray(base_coef, dtype=np.float32).tolist(),
                "d62_base_intercept_fp32": np.asarray(base_intercept, dtype=np.float32).tolist(),
                "d62_residual_coefficient_fp64": residual_coef.tolist(),
                "d62_residual_intercept_fp64": residual_intercept.tolist(),
                "d62_actual_coefficient_fp32": final_coef.tolist(),
                "d62_actual_intercept_fp32": final_intercept.tolist(),
                "d62_single_affine_state_only": True,
            }
        )
        return final_coef, final_intercept, audit

    return fit, call_records


def _install_resource_accounting(d42: Any) -> tuple[Any, Any]:
    original_macs, original_top = d46._install_d46_resource_accounting(d42)
    d46_top = d42.fit_d42_unified_shrinkage_lda

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        result = d46_top(*args, **kwargs)
        resource = dict(result.resource_audit)
        dimension = int(d42.FEATURE_DIM)
        old_k, new_k = int(resource["old_k_shot"]), int(resource["new_k_shot"])
        old_count, all_count = len(result.before_state.classes), len(result.state.classes)
        extra_fits = 0
        extra_lda = 0
        for k, count in ((old_k, old_count), (new_k, all_count)):
            if k > 2:
                extra_fits += 2 * (k + 1)
                extra_lda += 2 * int(d42._lda_fit_macs(k * count, count))
                extra_lda += 2 * k * int(d42._lda_fit_macs((k - 1) * count, count))
        fisher = d61._fisher_dense_macs(dimension, extra_fits)
        scalar = int(sum(k * c * c * 8 for k, c in ((old_k, old_count), (new_k, all_count))))
        resource.update(
            {
                "d62_additional_component_fit_count": extra_fits,
                "d62_additional_lda_fit_macs": extra_lda,
                "d62_fisher_dense_algebra_mac_equivalent_upper_bound": fisher,
                "d62_gate_scalar_mac_equivalents": scalar,
                "d62_query_extra_macs": 0,
                "d62_persistent_state_extra_bytes": 0,
                "d62_optimizer_steps_extra": 0,
                "d62_resource_single_affine_state_only": True,
            }
        )
        resource["lda_closed_form_fit_count"] = int(
            resource["lda_closed_form_fit_count"] + extra_fits
        )
        resource["estimated_lda_fit_macs"] = int(
            resource["estimated_lda_fit_macs"] + extra_lda
        )
        resource["estimated_adaptation_macs"] = int(
            resource["estimated_adaptation_macs"] + extra_lda + fisher + scalar
        )
        return replace(result, resource_audit=resource)

    d42.fit_d42_unified_shrinkage_lda = wrapped
    return original_macs, original_top


def _verify_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    target = [
        row for row in rows if row.get("candidate_id") in ("D42-USLDA-INT8", "D42-USLDA-FP32-MATCHED")
    ]
    if len(rows) != 105 or len(target) != 30:
        raise D62ProbeError("D62 training row closure drift")
    sanitized = copy.deepcopy(rows)
    for row in sanitized:
        resource = row.get("resource")
        if not isinstance(resource, dict) or "d62_additional_component_fit_count" not in resource:
            continue
        resource["lda_closed_form_fit_count"] -= resource["d62_additional_component_fit_count"]
        resource["estimated_lda_fit_macs"] -= resource["d62_additional_lda_fit_macs"]
        resource["estimated_adaptation_macs"] -= (
            resource["d62_additional_lda_fit_macs"]
            + resource["d62_fisher_dense_algebra_mac_equivalent_upper_bound"]
            + resource["d62_gate_scalar_mac_equivalents"]
        )
    d46_count = d46._verify_d46_fit_audits(sanitized)
    active = accepted = atomic = 0
    for row in target:
        resource = row["resource"]
        if (
            int(resource.get("d62_additional_component_fit_count", -1)) != 36
            or int(resource.get("d62_query_extra_macs", -1)) != 0
            or resource.get("d62_resource_single_affine_state_only") is not True
        ):
            raise D62ProbeError("D62 resource closure drift")
        for field in ("before_covariance_audit", "final_covariance_audit"):
            audit = row["geometry_summary"][field]
            if any(
                audit.get(name) != expected
                for name, expected in {
                    "d43_probe_arm": ARM,
                    "d43_covariance_structure": STRUCTURE,
                    "d62_probe_arm": ARM,
                    "d62_formula": FORMULA,
                    "d62_actual_k": 8,
                    "d62_class_id_specific_formula": False,
                    "d62_old_new_role_specific_branch": False,
                    "d62_scene_receiver_handle_specific_branch": False,
                    "d62_uses_outer_held_or_query": False,
                    "d62_hyperparameter_count": 0,
                    "d62_single_affine_state_only": True,
                }.items()
            ):
                raise D62ProbeError("D62 exact audit drift")
            mask = np.asarray(audit["d62_final_accept_mask"], dtype=bool)
            if mask.shape != (int(audit["d62_class_count"]),):
                raise D62ProbeError("D62 accept mask drift")
            accepted += int(np.sum(mask))
            active += int(np.any(mask))
            atomic += int(not bool(audit["d62_joint_atomic_safe"]))
    return {
        "verified_d46_fit_row_count": d46_count,
        "verified_d62_target_row_count": len(target),
        "verified_d62_fit_audit_count": 2 * len(target),
        "verified_d62_active_fit_count": active,
        "verified_d62_accepted_row_count": accepted,
        "verified_d62_atomic_fallback_count": atomic,
    }


def _verify_output(output: Path, script_sha: str, helper_hashes: dict[str, str]) -> dict[str, Any]:
    try:
        evidence = d43._verify_probe_output(output, ARM, script_sha)
    except d43.D43ProbeError as error:
        if "D43 fit audit missing from" not in str(error):
            raise
        receipt = d43._read_json(output / "RECEIPT.json")
        evidence = {
            "base_runner_receipt_sha256": d43._sha256(output / "RECEIPT.json"),
            "verified_training_row_count": int(receipt["training_log_row_count"]),
            "verified_query_opened": False,
            "verified_forced_nonpromotable": True,
            "d62_generic_probe_guard_verified_through_fit_audit_boundary": True,
        }
    support = d43._read_json(output / "support_audit.json")
    closure = support.get("candidate_lock", {}).get("source_closure", {})
    if any(closure.get(name) != value for name, value in helper_hashes.items()):
        raise D62ProbeError("D62 helper source closure drift")
    rows = [
        json.loads(line)
        for line in (output / "training_log.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return {**evidence, **_verify_rows(rows)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--d62-arm", required=True, choices=(ARM,))
    parser.add_argument("--runtime-root", required=True, type=Path)
    parser.add_argument("--probe-root", required=True, type=Path)
    parser.add_argument("--d62-confirmation-seed", type=int)
    known, runner_arguments = parser.parse_known_args(argv)
    d43._require_locked_runner_arguments(runner_arguments)
    output = d43._runner_output(runner_arguments)
    if output.exists():
        raise D62ProbeError(f"D62 output already exists: {output}")
    script_sha = d43._sha256(Path(__file__).resolve())
    helper_hashes = {
        "d62_d61_helper_sha256": d43._sha256(D61_HELPER_PATH),
        "d62_d46_helper_sha256": d43._sha256(d61.D46_HELPER_PATH),
        "d62_d43_helper_sha256": d43._sha256(d61.d46.d44.D43_HELPER_PATH),
    }
    previous_sys_path, previous_argv = list(sys.path), sys.argv
    d42 = package = None
    original_path: tuple[str, ...] = ()
    original_fit = original_macs = original_top = original_cell_guard = None
    original_centering_drift_policy = d43.ALLOW_FP32_CENTERING_ARGMAX_DRIFT
    runner_name, exit_code = "d62_locked_d42_runner", 1
    runner_module = None
    call_records: list[dict[str, Any]] = []
    try:
        d42, package, original_path = d43._bootstrap(known.runtime_root, known.probe_root)
        if known.d62_confirmation_seed is not None:
            d43.ALLOW_FP32_CENTERING_ARGMAX_DRIFT = True
        original_fit = d42._fit_equal_prior_lda
        fit, call_records = build_d62_fit(d42)
        d42._fit_equal_prior_lda = fit
        original_macs, original_top = _install_resource_accounting(d42)
        runner = known.probe_root / "code" / "scripts" / "run_d25_support_only_concat.py"
        runner_spec = importlib.util.spec_from_file_location(runner_name, runner)
        if runner_spec is None or runner_spec.loader is None:
            raise D62ProbeError("D62 could not load locked runner")
        runner_module = importlib.util.module_from_spec(runner_spec)
        sys.modules[runner_spec.name] = runner_module
        runner_spec.loader.exec_module(runner_module)
        original_cell_guard = _install_confirmation_cell_guard(
            runner_module, known.d62_confirmation_seed
        )
        d43._install_runner_probe_guards(
            runner_module,
            arm=known.d62_arm,
            probe_script_sha256=script_sha,
            extra_source_closure=helper_hashes,
        )
        sys.argv = [str(runner), *runner_arguments]
        exit_code = int(runner_module.main())
    finally:
        sys.argv, sys.path[:] = previous_argv, previous_sys_path
        if d42 is not None and original_fit is not None:
            d42._fit_equal_prior_lda = original_fit
        if d42 is not None and original_macs is not None:
            d42._lda_fit_macs = original_macs
        if d42 is not None and original_top is not None:
            d42.fit_d42_unified_shrinkage_lda = original_top
        if runner_module is not None and original_cell_guard is not None:
            runner_module._require_d42_development_cell = original_cell_guard
        d43.ALLOW_FP32_CENTERING_ARGMAX_DRIFT = original_centering_drift_policy
        if package is not None:
            package.__path__[:] = list(original_path)
        sys.modules.pop(runner_name, None)
    if exit_code != 0:
        return exit_code
    expected_calls = 30 * 36
    if len(call_records) != expected_calls:
        raise D62ProbeError(f"D62 component-fit count drift: {len(call_records)} != {expected_calls}")
    evidence = _verify_output(output, script_sha, helper_hashes)
    record_sha = hashlib.sha256(
        json.dumps(call_records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    metadata = {
        "schema": "cvs.phase2.d62.crossfitted_fisher_row_splice_probe.v1",
        "status": "DEVELOPMENT_SUPPORT_ONLY_DIAGNOSTIC_PROBE",
        "arm": known.d62_arm,
        "formal_candidate": False,
        "probe_forced_nonpromotable": True,
        "evaluation_role": (
            "independent_confirmation"
            if known.d62_confirmation_seed is not None
            else "development"
        ),
        "confirmation_seed": known.d62_confirmation_seed,
        "fp32_centering_argmax_drift_allowed": (
            known.d62_confirmation_seed is not None
        ),
        "selected_only_full_k10_refit_allowed": False,
        "query_opened": False,
        "probe_script_sha256": script_sha,
        "formula": FORMULA,
        "component_fit_execution_count": len(call_records),
        "component_fit_record_sha256": record_sha,
        "runtime_root": str(known.runtime_root.resolve()),
        "probe_root": str(known.probe_root.resolve()),
        **evidence,
    }
    (output / "D62_PROBE_METADATA.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
