#!/usr/bin/env python3
"""D47 probe: positive-part anchored shrinkage of D46 classwise evidence."""

from __future__ import annotations

import argparse
import copy
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
D46_HELPER_PATH = SCRIPT_DIR / "probe_d46_classwise_loo_reliability_fusion.py"
D46_SPEC = importlib.util.spec_from_file_location("d47_d46_probe_helper", D46_HELPER_PATH)
if D46_SPEC is None or D46_SPEC.loader is None:
    raise RuntimeError("D47 could not load the D46 probe helper")
d46 = importlib.util.module_from_spec(D46_SPEC)
D46_SPEC.loader.exec_module(d46)
d45 = d46.d45
d44 = d46.d44
d43 = d46.d43


ARM = "anchored_reliability_shrinkage"
STRUCTURE = "full_block_support_inner_loo_positive_part_anchored_reliability_shrinkage"
WEIGHT_FORMULA = "z_c=K*dbar_c;z0=C*mu;z_post_c=(1-a_c)*z0+a_c*z_c;w_full_c=sigmoid(z_post_c)"
LOG_EVIDENCE_FORMULA = "anchored_log_evidence_pair_c=[z_post_c,0]"
HETEROGENEITY_FORMULA = "tau2=max(0,var_c(z_c)-mean_c(K^2*v_c))"
STATISTICAL_CLAIM = "eb_inspired_deterministic_shrinkage_not_calibrated_posterior"
SCALAR_OPERATION_UPPER_BOUND_DERIVATION = {
    "counting_unit": "conservative_scalar_MAC_equivalent",
    "fold_evidence_and_first_second_moments_per_state": "6*K*C",
    "cross_class_moments_and_positive_part_shrinkage_per_state": "16*C+8",
    "post_logit_sigmoid_and_endpoint_checks_per_state": "8*C+8",
    "two_state_total": "6*K*(C_old+C_all)+24*(C_old+C_all)+32",
    "valid_integer_k_domain": "K>=2;K1_executes_no_moment_algebra_and_is_zero",
}
d43.ARM_STRUCTURES[ARM] = STRUCTURE


class D47ProbeError(RuntimeError):
    pass


def _stable_sigmoid(values: np.ndarray) -> np.ndarray:
    logits = np.asarray(values, dtype=np.float64)
    if not np.isfinite(logits).all():
        raise D47ProbeError("D47 anchored log-odds became non-finite")
    result = np.empty_like(logits)
    positive = logits >= 0.0
    result[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
    exp_values = np.exp(logits[~positive])
    result[~positive] = exp_values / (1.0 + exp_values)
    if (
        not np.isfinite(result).all()
        or np.any(result <= 0.0)
        or np.any(result >= 1.0)
    ):
        raise D47ProbeError("D47 anchored sigmoid underflow/overflow drift")
    return result


def _fold_ce(partition: Any, k_shot: int, class_count: int) -> np.ndarray:
    if not isinstance(partition, dict):
        raise D47ProbeError("D47 partition evidence missing")
    values = np.asarray(partition.get("held_ce_by_fold_and_class"), dtype=np.float64)
    if (
        values.shape != (k_shot, class_count)
        or not np.isfinite(values).all()
        or np.any(values < 0.0)
    ):
        raise D47ProbeError("D47 fold CE evidence drift")
    return values


def _anchored_reliability_strategy(
    *,
    full_per_class_ce: Any,
    block_per_class_ce: Any,
    full_partition: Any,
    block_partition: Any,
    k_shot: int,
    class_count: int,
) -> tuple[np.ndarray, Any, dict[str, Any]]:
    k = int(k_shot)
    c = int(class_count)
    if c < 2 or k < 1:
        raise D47ProbeError("D47 requires C>=2 and K>=1")
    common = {
        "d43_probe_arm": ARM,
        "d43_covariance_structure": STRUCTURE,
        "d46_probe_arm": ARM,
        "d46_weight_formula": WEIGHT_FORMULA,
        "d46_log_evidence_formula": LOG_EVIDENCE_FORMULA,
        "d47_probe_arm": ARM,
        "d47_weight_formula": WEIGHT_FORMULA,
        "d47_log_evidence_formula": LOG_EVIDENCE_FORMULA,
        "d47_heterogeneity_formula": HETEROGENEITY_FORMULA,
        "d47_statistical_claim": STATISTICAL_CLAIM,
        "d47_no_temperature_clip_threshold_or_scan": True,
        "d47_reliability_uses_outer_held_or_query": False,
        "d47_class_id_specific_formula": False,
        "d47_old_new_role_specific_branch": False,
        "d47_scene_handle_specific_branch": False,
        "d47_actual_k": k,
        "d47_class_count": c,
    }
    if k == 1:
        weights = np.full((c, 2), 0.5, dtype=np.float64)
        return weights, None, {
            **common,
            "d47_boundary_status": "k1_equal_unit_fallback",
            "d47_fold_logprob_advantage_by_class": None,
            "d47_dbar_by_class": None,
            "d47_sample_variance_by_class": None,
            "d47_mean_variance_proxy_by_class": None,
            "d47_observation_log_odds_by_class": None,
            "d47_within_log_odds_variance_proxy_by_class": None,
            "d47_mu": None,
            "d47_zbar": None,
            "d47_d45_anchor_z0": None,
            "d47_between_observation_variance": None,
            "d47_mean_within_log_odds_variance_proxy": None,
            "d47_tau_raw_squared": None,
            "d47_tau_squared": None,
            "d47_shrink_factor_by_class": [0.0] * c,
            "d47_post_log_odds_by_class": [0.0] * c,
            "d47_full_weight_by_class": [0.5] * c,
            "d47_block_weight_by_class": [0.5] * c,
            "d47_complete_pooling_d45_formula_weight_error": 0.0,
            "d47_no_shrinkage_d46_max_abs_error": None,
        }

    full_fold = _fold_ce(full_partition, k, c)
    block_fold = _fold_ce(block_partition, k, c)
    full_ce = np.asarray(full_per_class_ce, dtype=np.float64)
    block_ce = np.asarray(block_per_class_ce, dtype=np.float64)
    if (
        full_ce.shape != (c,)
        or block_ce.shape != (c,)
        or not np.allclose(full_fold.mean(axis=0), full_ce, rtol=0.0, atol=1.0e-12)
        or not np.allclose(block_fold.mean(axis=0), block_ce, rtol=0.0, atol=1.0e-12)
    ):
        raise D47ProbeError("D47 fold/per-class CE closure drift")
    advantage = block_fold - full_fold
    if k == 2:
        if not np.allclose(advantage, 0.0, rtol=0.0, atol=1.0e-12):
            raise D47ProbeError("D47 K2 component-equivalence drift")
        weights = np.full((c, 2), 0.5, dtype=np.float64)
        zero = [0.0] * c
        log_evidence = np.zeros((c, 2), dtype=np.float64)
        return weights, log_evidence, {
            **common,
            "d47_boundary_status": "k2_equal_component_fallback",
            "d47_fold_logprob_advantage_by_class": advantage.tolist(),
            "d47_dbar_by_class": zero,
            "d47_sample_variance_by_class": zero,
            "d47_mean_variance_proxy_by_class": zero,
            "d47_observation_log_odds_by_class": zero,
            "d47_within_log_odds_variance_proxy_by_class": zero,
            "d47_mu": 0.0,
            "d47_zbar": 0.0,
            "d47_d45_anchor_z0": 0.0,
            "d47_between_observation_variance": 0.0,
            "d47_mean_within_log_odds_variance_proxy": 0.0,
            "d47_tau_raw_squared": 0.0,
            "d47_tau_squared": 0.0,
            "d47_shrink_factor_by_class": zero,
            "d47_post_log_odds_by_class": zero,
            "d47_full_weight_by_class": [0.5] * c,
            "d47_block_weight_by_class": [0.5] * c,
            "d47_complete_pooling_d45_formula_weight_error": 0.0,
            "d47_no_shrinkage_d46_max_abs_error": 0.0,
        }
    if k < 3:
        raise D47ProbeError("D47 unsupported K branch")

    dbar = advantage.mean(axis=0)
    sample_variance = advantage.var(axis=0, ddof=1)
    mean_variance = sample_variance / float(k)
    observation_log_odds = float(k) * dbar
    within_log_odds_variance = float(k * k) * mean_variance
    mu = float(np.mean(dbar))
    zbar = float(np.mean(observation_log_odds))
    z0 = float(c) * mu
    between_variance = float(np.var(observation_log_odds, ddof=1))
    mean_within = float(np.mean(within_log_odds_variance))
    tau_raw = between_variance - mean_within
    tau_squared = max(0.0, tau_raw)
    if not all(
        np.isfinite(value)
        for value in (
            mu,
            zbar,
            z0,
            between_variance,
            mean_within,
            tau_raw,
            tau_squared,
        )
    ) or np.any(sample_variance < 0.0) or np.any(within_log_odds_variance < 0.0):
        raise D47ProbeError("D47 anchored moment drift")
    if tau_squared == 0.0:
        shrink = np.zeros(c, dtype=np.float64)
        boundary = "positive_part_complete_pooling_d45"
    else:
        shrink = np.where(
            within_log_odds_variance == 0.0,
            1.0,
            tau_squared / (tau_squared + within_log_odds_variance),
        )
        boundary = "positive_tau_partial_or_no_shrinkage"
    post_log_odds = (1.0 - shrink) * z0 + shrink * observation_log_odds
    full_weight = _stable_sigmoid(post_log_odds)
    weights = np.stack([full_weight, 1.0 - full_weight], axis=1)
    if np.any(weights <= 0.0) or not np.allclose(
        weights.sum(axis=1), 1.0, rtol=0.0, atol=1.0e-15
    ):
        raise D47ProbeError("D47 anchored weight closure drift")
    d45_weight = float(_stable_sigmoid(np.asarray([z0]))[0])
    d46_weights, _ = d46._classwise_likelihood_weights(full_ce, block_ce, k)
    complete_pooling_error = float(np.max(np.abs(full_weight - d45_weight)))
    no_shrinkage_error = float(np.max(np.abs(full_weight - d46_weights[:, 0])))
    if tau_squared == 0.0 and complete_pooling_error > 1.0e-15:
        raise D47ProbeError("D47 complete-pooling D45 endpoint drift")
    log_evidence = np.stack([post_log_odds, np.zeros(c)], axis=1)
    return weights, log_evidence, {
        **common,
        "d47_boundary_status": boundary,
        "d47_fold_logprob_advantage_by_class": advantage.tolist(),
        "d47_dbar_by_class": dbar.tolist(),
        "d47_sample_variance_by_class": sample_variance.tolist(),
        "d47_mean_variance_proxy_by_class": mean_variance.tolist(),
        "d47_observation_log_odds_by_class": observation_log_odds.tolist(),
        "d47_within_log_odds_variance_proxy_by_class": within_log_odds_variance.tolist(),
        "d47_mu": mu,
        "d47_zbar": zbar,
        "d47_d45_anchor_z0": z0,
        "d47_between_observation_variance": between_variance,
        "d47_mean_within_log_odds_variance_proxy": mean_within,
        "d47_tau_raw_squared": tau_raw,
        "d47_tau_squared": tau_squared,
        "d47_shrink_factor_by_class": shrink.tolist(),
        "d47_post_log_odds_by_class": post_log_odds.tolist(),
        "d47_full_weight_by_class": full_weight.tolist(),
        "d47_block_weight_by_class": (1.0 - full_weight).tolist(),
        "d47_complete_pooling_d45_formula_weight_error": complete_pooling_error,
        "d47_no_shrinkage_d46_max_abs_error": no_shrinkage_error,
    }


def build_anchored_reliability_fit(d42: Any) -> Any:
    return d46.build_classwise_loo_reliability_fit(
        d42, reliability_strategy=_anchored_reliability_strategy
    )


def _scalar_operation_upper_bound(
    k_shot: int, old_class_count: int, all_class_count: int
) -> int:
    """Conservative MAC-equivalent upper bound for the D47 scalar moment algebra."""

    k = int(k_shot)
    class_sum = int(old_class_count) + int(all_class_count)
    if k < 1 or old_class_count < 2 or all_class_count < old_class_count:
        raise D47ProbeError("D47 scalar-operation inventory input drift")
    if k == 1:
        return 0
    return int(6 * k * class_sum + 24 * class_sum + 32)


def _install_d47_resource_accounting(d42: Any) -> tuple[Any, Any]:
    original_macs, original_top = d46._install_d46_resource_accounting(d42)
    d46_top = d42.fit_d42_unified_shrinkage_lda

    def fit_with_d47_resource_audit(*args: Any, **kwargs: Any) -> Any:
        result = d46_top(*args, **kwargs)
        resource = dict(result.resource_audit)
        old_class_count = len(result.before_state.classes)
        all_class_count = len(result.state.classes)
        scalar_upper_bound = _scalar_operation_upper_bound(
            int(resource["old_k_shot"]), old_class_count, all_class_count
        )
        resource.update(
            {
                "d47_additional_lda_fit_count": 0,
                "d47_additional_optimizer_steps": 0,
                "d47_additional_query_state_count": 0,
                "d47_query_sidecar_bytes": 0,
                "d47_scalar_statistics_complexity": "O(C*K)",
                "d47_scalar_operation_upper_bound_formula": (
                    "0_if_K1_else_6*K*(C_old+C_all)+"
                    "24*(C_old+C_all)+32"
                ),
                "d47_scalar_operation_upper_bound_derivation": dict(
                    SCALAR_OPERATION_UPPER_BOUND_DERIVATION
                ),
                "d47_estimated_scalar_operation_upper_bound": scalar_upper_bound,
                "d47_additional_adaptation_mac_equivalents": scalar_upper_bound,
                "d47_resource_reuses_d46_exact_inventory": True,
            }
        )
        resource["estimated_adaptation_macs"] = int(
            resource["estimated_adaptation_macs"] + scalar_upper_bound
        )
        return replace(result, resource_audit=resource)

    d42.fit_d42_unified_shrinkage_lda = fit_with_d47_resource_audit
    return original_macs, original_top


def _allclose(actual: Any, expected: Any, *, atol: float = 1.0e-12) -> bool:
    if actual is None or expected is None:
        return actual is None and expected is None
    try:
        left = np.asarray(actual, dtype=np.float64)
        right = np.asarray(expected, dtype=np.float64)
    except (TypeError, ValueError):
        return actual == expected
    return left.shape == right.shape and np.allclose(left, right, rtol=0.0, atol=atol)


def _verify_d47_fit_audits(training_rows: list[dict[str, Any]]) -> int:
    d47_rows = [
        row
        for row in training_rows
        if row.get("candidate_id") in ("D42-USLDA-INT8", "D42-USLDA-FP32-MATCHED")
    ]
    if len(d47_rows) != 30:
        raise D47ProbeError("D47 training-row closure drift")
    sanitized = copy.deepcopy(training_rows)
    for row in d47_rows:
        resource = row.get("resource", {})
        k = int(resource.get("old_k_shot", -1))
        class_counts: list[int] = []
        for field in ("before_covariance_audit", "final_covariance_audit"):
            audit = row.get("geometry_summary", {}).get(field)
            if not isinstance(audit, dict):
                raise D47ProbeError("D47 fit audit missing")
            c = len(audit.get("d47_full_weight_by_class", []))
            class_counts.append(c)
            expected_weights, expected_log, expected_audit = _anchored_reliability_strategy(
                full_per_class_ce=audit.get("d46_full_inner_loo_ce_by_class"),
                block_per_class_ce=audit.get("d46_block_inner_loo_ce_by_class"),
                full_partition=audit.get("d46_full_inner_partition_audit"),
                block_partition=audit.get("d46_block_inner_partition_audit"),
                k_shot=k,
                class_count=c,
            )
            scalar_fields = (
                "d47_actual_k",
                "d47_class_count",
                "d47_mu",
                "d47_zbar",
                "d47_d45_anchor_z0",
                "d47_between_observation_variance",
                "d47_mean_within_log_odds_variance_proxy",
                "d47_tau_raw_squared",
                "d47_tau_squared",
                "d47_complete_pooling_d45_formula_weight_error",
                "d47_no_shrinkage_d46_max_abs_error",
            )
            array_fields = (
                "d47_fold_logprob_advantage_by_class",
                "d47_dbar_by_class",
                "d47_sample_variance_by_class",
                "d47_mean_variance_proxy_by_class",
                "d47_observation_log_odds_by_class",
                "d47_within_log_odds_variance_proxy_by_class",
                "d47_shrink_factor_by_class",
                "d47_post_log_odds_by_class",
                "d47_full_weight_by_class",
                "d47_block_weight_by_class",
            )
            required_exact = {
                "d43_probe_arm": ARM,
                "d43_covariance_structure": STRUCTURE,
                "d46_probe_arm": ARM,
                "d46_weight_formula": WEIGHT_FORMULA,
                "d46_log_evidence_formula": LOG_EVIDENCE_FORMULA,
                "d47_probe_arm": ARM,
                "d47_weight_formula": WEIGHT_FORMULA,
                "d47_log_evidence_formula": LOG_EVIDENCE_FORMULA,
                "d47_heterogeneity_formula": HETEROGENEITY_FORMULA,
                "d47_statistical_claim": STATISTICAL_CLAIM,
                "d47_no_temperature_clip_threshold_or_scan": True,
                "d47_reliability_uses_outer_held_or_query": False,
                "d47_class_id_specific_formula": False,
                "d47_old_new_role_specific_branch": False,
                "d47_scene_handle_specific_branch": False,
                "d47_boundary_status": expected_audit["d47_boundary_status"],
            }
            if any(audit.get(name) != value for name, value in required_exact.items()):
                raise D47ProbeError("D47 exact audit drift")
            if any(not _allclose(audit.get(name), expected_audit.get(name)) for name in scalar_fields + array_fields):
                raise D47ProbeError("D47 anchored evidence closure drift")
            if not _allclose(audit.get("d46_full_weight_by_class"), expected_weights[:, 0]):
                raise D47ProbeError("D47 installed full-weight drift")
            if not _allclose(audit.get("d46_block_weight_by_class"), expected_weights[:, 1]):
                raise D47ProbeError("D47 installed block-weight drift")
            if not _allclose(audit.get("d46_log_evidence_by_class_and_component"), expected_log):
                raise D47ProbeError("D47 installed log-evidence drift")
        old_count, all_count = class_counts
        scalar_upper_bound = _scalar_operation_upper_bound(k, old_count, all_count)
        expected_resource = {
            "d47_additional_lda_fit_count": 0,
            "d47_additional_optimizer_steps": 0,
            "d47_additional_query_state_count": 0,
            "d47_query_sidecar_bytes": 0,
            "d47_scalar_statistics_complexity": "O(C*K)",
            "d47_scalar_operation_upper_bound_formula": (
                "0_if_K1_else_6*K*(C_old+C_all)+24*(C_old+C_all)+32"
            ),
            "d47_scalar_operation_upper_bound_derivation": (
                SCALAR_OPERATION_UPPER_BOUND_DERIVATION
            ),
            "d47_estimated_scalar_operation_upper_bound": scalar_upper_bound,
            "d47_additional_adaptation_mac_equivalents": scalar_upper_bound,
            "d47_resource_reuses_d46_exact_inventory": True,
        }
        if any(resource.get(name) != value for name, value in expected_resource.items()):
            raise D47ProbeError("D47 resource audit drift")
        expected_total = int(
            resource.get("estimated_metric_adaptation_macs", -1)
            + resource.get("estimated_lda_fit_macs", -1)
            + resource.get("d46_estimated_reliability_scoring_macs", -1)
            + resource.get("d46_estimated_classwise_affine_fusion_macs", -1)
            + scalar_upper_bound
        )
        if resource.get("estimated_adaptation_macs") != expected_total:
            raise D47ProbeError("D47 total adaptation MAC-equivalent drift")

    for row in sanitized:
        if row.get("candidate_id") not in ("D42-USLDA-INT8", "D42-USLDA-FP32-MATCHED"):
            continue
        k = int(row["resource"]["old_k_shot"])
        for field in ("before_covariance_audit", "final_covariance_audit"):
            audit = row["geometry_summary"][field]
            full_ce = audit["d46_full_inner_loo_ce_by_class"]
            block_ce = audit["d46_block_inner_loo_ce_by_class"]
            if k <= 1:
                weights = np.full((len(audit["d46_full_weight_by_class"]), 2), 0.5)
                log_evidence = None
            else:
                weights, log_evidence = d46._classwise_likelihood_weights(full_ce, block_ce, k)
                log_evidence = log_evidence.tolist()
            audit.update(
                {
                    "d43_probe_arm": d46.ARM,
                    "d43_covariance_structure": d46.STRUCTURE,
                    "d46_probe_arm": d46.ARM,
                    "d46_weight_formula": d46.WEIGHT_FORMULA,
                    "d46_log_evidence_formula": d46.LOG_EVIDENCE_FORMULA,
                    "d46_full_weight_by_class": weights[:, 0].tolist(),
                    "d46_block_weight_by_class": weights[:, 1].tolist(),
                    "d46_log_evidence_by_class_and_component": log_evidence,
                }
            )
        resource = row["resource"]
        resource["estimated_adaptation_macs"] = int(
            resource["estimated_adaptation_macs"]
            - resource["d47_additional_adaptation_mac_equivalents"]
        )
    d46._verify_d46_fit_audits(sanitized)
    return len(d47_rows)


def _verify_d47_output(
    output: Path,
    probe_script_sha256: str,
    d46_helper_sha256: str,
    d45_helper_sha256: str,
    d44_helper_sha256: str,
    d43_helper_sha256: str,
) -> dict[str, Any]:
    evidence = d43._verify_probe_output(output, ARM, probe_script_sha256)
    support = d43._read_json(output / "support_audit.json")
    closure = support.get("candidate_lock", {}).get("source_closure", {})
    expected = {
        "d47_d46_helper_sha256": d46_helper_sha256,
        "d47_d45_helper_sha256": d45_helper_sha256,
        "d47_d44_helper_sha256": d44_helper_sha256,
        "d47_d43_helper_sha256": d43_helper_sha256,
    }
    if any(closure.get(name) != value for name, value in expected.items()):
        raise D47ProbeError("D47 helper source closure drift")
    rows = [
        json.loads(line)
        for line in (output / "training_log.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    count = _verify_d47_fit_audits(rows)
    return {**evidence, "verified_d47_fit_row_count": count, **expected}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--d47-arm", required=True, choices=(ARM,))
    parser.add_argument("--runtime-root", required=True, type=Path)
    parser.add_argument("--probe-root", required=True, type=Path)
    known, runner_arguments = parser.parse_known_args(argv)
    d43._require_locked_runner_arguments(runner_arguments)
    output = d43._runner_output(runner_arguments)
    if output.exists():
        raise D47ProbeError(f"D47 output already exists: {output}")
    previous_sys_path = list(sys.path)
    previous_argv = sys.argv
    d42 = None
    package = None
    original_package_path: tuple[str, ...] = ()
    original_fit = None
    original_macs = None
    original_top = None
    runner_module_name = "d47_locked_d42_runner"
    probe_script_sha256 = d43._sha256(Path(__file__).resolve())
    d46_helper_sha256 = d43._sha256(D46_HELPER_PATH)
    d45_helper_sha256 = d43._sha256(d46.D45_HELPER_PATH)
    d44_helper_sha256 = d43._sha256(d45.D44_HELPER_PATH)
    d43_helper_sha256 = d43._sha256(d44.D43_HELPER_PATH)
    try:
        d42, package, original_package_path = d43._bootstrap(known.runtime_root, known.probe_root)
        original_fit = d42._fit_equal_prior_lda
        d42._fit_equal_prior_lda = build_anchored_reliability_fit(d42)
        original_macs, original_top = _install_d47_resource_accounting(d42)
        runner = known.probe_root / "code" / "scripts" / "run_d25_support_only_concat.py"
        spec = importlib.util.spec_from_file_location(runner_module_name, runner)
        if spec is None or spec.loader is None:
            raise D47ProbeError("D47 could not load the locked D42 runner")
        runner_module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = runner_module
        spec.loader.exec_module(runner_module)
        d43._install_runner_probe_guards(
            runner_module,
            arm=known.d47_arm,
            probe_script_sha256=probe_script_sha256,
            extra_source_closure={
                "d47_d46_helper_sha256": d46_helper_sha256,
                "d47_d45_helper_sha256": d45_helper_sha256,
                "d47_d44_helper_sha256": d44_helper_sha256,
                "d47_d43_helper_sha256": d43_helper_sha256,
            },
        )
        sys.argv = [str(runner), *runner_arguments]
        exit_code = int(runner_module.main())
    finally:
        sys.argv = previous_argv
        sys.path[:] = previous_sys_path
        if d42 is not None and original_fit is not None:
            d42._fit_equal_prior_lda = original_fit
        if d42 is not None and original_macs is not None:
            d42._lda_fit_macs = original_macs
        if d42 is not None and original_top is not None:
            d42.fit_d42_unified_shrinkage_lda = original_top
        if package is not None:
            package.__path__[:] = list(original_package_path)
        sys.modules.pop(runner_module_name, None)
    if exit_code != 0:
        return exit_code
    evidence = _verify_d47_output(
        output,
        probe_script_sha256,
        d46_helper_sha256,
        d45_helper_sha256,
        d44_helper_sha256,
        d43_helper_sha256,
    )
    metadata = {
        "schema": "cvs.phase2.d47.anchored_reliability_shrinkage_probe.v1",
        "status": "DEVELOPMENT_SUPPORT_ONLY_DIAGNOSTIC_PROBE",
        "arm": known.d47_arm,
        "formal_candidate": False,
        "probe_forced_nonpromotable": True,
        "selected_only_full_k10_refit_allowed": False,
        "query_opened": False,
        "probe_script_sha256": probe_script_sha256,
        "d46_helper_sha256": d46_helper_sha256,
        "d45_helper_sha256": d45_helper_sha256,
        "d44_helper_sha256": d44_helper_sha256,
        "d43_helper_sha256": d43_helper_sha256,
        "weight_formula": WEIGHT_FORMULA,
        "heterogeneity_formula": HETEROGENEITY_FORMULA,
        "statistical_claim": STATISTICAL_CLAIM,
        "runtime_root": str(known.runtime_root.resolve()),
        "probe_root": str(known.probe_root.resolve()),
        **evidence,
    }
    (output / "D47_PROBE_METADATA.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
