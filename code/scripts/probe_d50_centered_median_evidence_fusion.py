#!/usr/bin/env python3
"""D50 probe: D45-anchored centered-median classwise LOO fusion."""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
D47_HELPER_PATH = SCRIPT_DIR / "probe_d47_anchored_reliability_shrinkage.py"
D47_SPEC = importlib.util.spec_from_file_location("d50_d47_probe_helper", D47_HELPER_PATH)
if D47_SPEC is None or D47_SPEC.loader is None:
    raise RuntimeError("D50 could not load the D47 probe helper")
d47 = importlib.util.module_from_spec(D47_SPEC)
D47_SPEC.loader.exec_module(d47)
d46 = d47.d46
d45 = d46.d45
d44 = d46.d44
d43 = d46.d43


ARM = "centered_median_evidence_fusion"
STRUCTURE = "full_block_d45_anchor_centered_classwise_median_rank_evidence"
WEIGHT_FORMULA = (
    "d_rc=ce_block_rc-ce_full_rc;z0=C*mean_rc(d_rc);"
    "m_c=median_r(d_rc);delta_c=K*(m_c-mean_c(m_c));"
    "w_full_c=sigmoid(z0+delta_c)"
)
LOG_EVIDENCE_FORMULA = "anchored_centered_median_log_evidence_pair_c=[z0+delta_c,0]"
STATISTICAL_CLAIM = "deterministic_robust_support_statistic_not_calibrated_posterior"
d43.ARM_STRUCTURES[ARM] = STRUCTURE


class D50ProbeError(RuntimeError):
    pass


def _stable_sigmoid(values: np.ndarray) -> np.ndarray:
    logits = np.asarray(values, dtype=np.float64)
    if not np.isfinite(logits).all():
        raise D50ProbeError("D50 log-odds became non-finite")
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
        raise D50ProbeError("D50 sigmoid endpoint drift")
    return result


def _fold_ce(partition: Any, k_shot: int, class_count: int) -> np.ndarray:
    if not isinstance(partition, dict):
        raise D50ProbeError("D50 partition evidence missing")
    values = np.asarray(partition.get("held_ce_by_fold_and_class"), dtype=np.float64)
    if (
        values.shape != (k_shot, class_count)
        or not np.isfinite(values).all()
        or np.any(values < 0.0)
    ):
        raise D50ProbeError("D50 fold CE evidence drift")
    return values


def _centered_median_strategy(
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
        raise D50ProbeError("D50 requires C>=2 and K>=1")
    common = {
        "d43_probe_arm": ARM,
        "d43_covariance_structure": STRUCTURE,
        "d46_probe_arm": ARM,
        "d46_weight_formula": WEIGHT_FORMULA,
        "d46_log_evidence_formula": LOG_EVIDENCE_FORMULA,
        "d50_probe_arm": ARM,
        "d50_weight_formula": WEIGHT_FORMULA,
        "d50_log_evidence_formula": LOG_EVIDENCE_FORMULA,
        "d50_statistical_claim": STATISTICAL_CLAIM,
        "d50_even_k_median_policy": "mean_of_two_middle_order_statistics",
        "d50_no_temperature_clip_threshold_sign_gate_or_scan": True,
        "d50_reliability_uses_outer_held_or_query": False,
        "d50_class_id_specific_formula": False,
        "d50_old_new_role_specific_branch": False,
        "d50_scene_receiver_handle_specific_branch": False,
        "d50_actual_k": k,
        "d50_class_count": c,
    }
    if k == 1:
        weights = np.full((c, 2), 0.5, dtype=np.float64)
        zeros = [0.0] * c
        return weights, None, {
            **common,
            "d50_boundary_status": "k1_d45_equal_unit_fallback",
            "d50_fold_logprob_advantage_by_class": None,
            "d50_mean_advantage_by_class": None,
            "d50_median_advantage_by_class": None,
            "d50_median_center": None,
            "d50_d45_anchor_z0": 0.0,
            "d50_centered_median_delta_by_class": zeros,
            "d50_post_log_odds_by_class": zeros,
            "d50_full_weight_by_class": [0.5] * c,
            "d50_block_weight_by_class": [0.5] * c,
            "d50_post_log_odds_mean_anchor_error": 0.0,
            "d50_median_mean_abs_difference": 0.0,
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
        raise D50ProbeError("D50 fold/per-class CE closure drift")
    advantage = block_fold - full_fold
    if k == 2 and not np.allclose(advantage, 0.0, rtol=0.0, atol=1.0e-12):
        raise D50ProbeError("D50 K2 component-equivalence drift")

    mean_advantage = advantage.mean(axis=0)
    median_advantage = np.median(advantage, axis=0)
    median_center = float(np.mean(median_advantage))
    z0 = float(c) * float(np.mean(mean_advantage))
    delta = float(k) * (median_advantage - median_center)
    post_log_odds = z0 + delta
    anchor_error = float(abs(np.mean(post_log_odds) - z0))
    median_mean_abs_difference = float(np.mean(np.abs(median_advantage - mean_advantage)))
    if (
        not np.isfinite(mean_advantage).all()
        or not np.isfinite(median_advantage).all()
        or not np.isfinite(delta).all()
        or not np.isfinite(post_log_odds).all()
        or anchor_error > 1.0e-12
    ):
        raise D50ProbeError("D50 centered-median closure drift")
    full_weight = _stable_sigmoid(post_log_odds)
    weights = np.stack([full_weight, 1.0 - full_weight], axis=1)
    if np.any(weights <= 0.0) or not np.allclose(
        weights.sum(axis=1), 1.0, rtol=0.0, atol=1.0e-15
    ):
        raise D50ProbeError("D50 weight closure drift")
    if k == 2 and not np.allclose(weights, 0.5, rtol=0.0, atol=1.0e-12):
        raise D50ProbeError("D50 K2 equal-weight drift")
    log_evidence = np.stack([post_log_odds, np.zeros(c)], axis=1)
    return weights, log_evidence, {
        **common,
        "d50_boundary_status": (
            "k2_equal_component_fallback" if k == 2 else "centered_median_active"
        ),
        "d50_fold_logprob_advantage_by_class": advantage.tolist(),
        "d50_mean_advantage_by_class": mean_advantage.tolist(),
        "d50_median_advantage_by_class": median_advantage.tolist(),
        "d50_median_center": median_center,
        "d50_d45_anchor_z0": z0,
        "d50_centered_median_delta_by_class": delta.tolist(),
        "d50_post_log_odds_by_class": post_log_odds.tolist(),
        "d50_full_weight_by_class": full_weight.tolist(),
        "d50_block_weight_by_class": (1.0 - full_weight).tolist(),
        "d50_post_log_odds_mean_anchor_error": anchor_error,
        "d50_median_mean_abs_difference": median_mean_abs_difference,
    }


def build_centered_median_fit(d42: Any) -> Any:
    return d46.build_classwise_loo_reliability_fit(
        d42, reliability_strategy=_centered_median_strategy
    )


def _allclose(actual: Any, expected: Any, *, atol: float = 1.0e-12) -> bool:
    if actual is None or expected is None:
        return actual is None and expected is None
    try:
        left = np.asarray(actual, dtype=np.float64)
        right = np.asarray(expected, dtype=np.float64)
    except (TypeError, ValueError):
        return actual == expected
    return left.shape == right.shape and np.allclose(left, right, rtol=0.0, atol=atol)


def _verify_d50_fit_audits(training_rows: list[dict[str, Any]]) -> int:
    d50_rows = [
        row
        for row in training_rows
        if row.get("candidate_id") in ("D42-USLDA-INT8", "D42-USLDA-FP32-MATCHED")
    ]
    if len(d50_rows) != 30:
        raise D50ProbeError("D50 training-row closure drift")
    sanitized = copy.deepcopy(training_rows)
    for row in d50_rows:
        resource = row.get("resource", {})
        k = int(resource.get("old_k_shot", -1))
        class_counts: list[int] = []
        for field in ("before_covariance_audit", "final_covariance_audit"):
            audit = row.get("geometry_summary", {}).get(field)
            if not isinstance(audit, dict):
                raise D50ProbeError("D50 fit audit missing")
            c = len(audit.get("d50_full_weight_by_class", []))
            class_counts.append(c)
            weights, log_evidence, expected = _centered_median_strategy(
                full_per_class_ce=audit.get("d46_full_inner_loo_ce_by_class"),
                block_per_class_ce=audit.get("d46_block_inner_loo_ce_by_class"),
                full_partition=audit.get("d46_full_inner_partition_audit"),
                block_partition=audit.get("d46_block_inner_partition_audit"),
                k_shot=k,
                class_count=c,
            )
            exact_fields = (
                "d43_probe_arm",
                "d43_covariance_structure",
                "d46_probe_arm",
                "d46_weight_formula",
                "d46_log_evidence_formula",
                "d50_probe_arm",
                "d50_weight_formula",
                "d50_log_evidence_formula",
                "d50_statistical_claim",
                "d50_even_k_median_policy",
                "d50_no_temperature_clip_threshold_sign_gate_or_scan",
                "d50_reliability_uses_outer_held_or_query",
                "d50_class_id_specific_formula",
                "d50_old_new_role_specific_branch",
                "d50_scene_receiver_handle_specific_branch",
                "d50_boundary_status",
                "d50_actual_k",
                "d50_class_count",
            )
            numeric_fields = (
                "d50_fold_logprob_advantage_by_class",
                "d50_mean_advantage_by_class",
                "d50_median_advantage_by_class",
                "d50_median_center",
                "d50_d45_anchor_z0",
                "d50_centered_median_delta_by_class",
                "d50_post_log_odds_by_class",
                "d50_full_weight_by_class",
                "d50_block_weight_by_class",
                "d50_post_log_odds_mean_anchor_error",
                "d50_median_mean_abs_difference",
            )
            if any(audit.get(name) != expected.get(name) for name in exact_fields):
                raise D50ProbeError("D50 exact audit drift")
            if any(not _allclose(audit.get(name), expected.get(name)) for name in numeric_fields):
                raise D50ProbeError("D50 numeric evidence closure drift")
            if not _allclose(audit.get("d46_full_weight_by_class"), weights[:, 0]):
                raise D50ProbeError("D50 installed full-weight drift")
            if not _allclose(audit.get("d46_block_weight_by_class"), weights[:, 1]):
                raise D50ProbeError("D50 installed block-weight drift")
            if not _allclose(
                audit.get("d46_log_evidence_by_class_and_component"), log_evidence
            ):
                raise D50ProbeError("D50 installed log-evidence drift")
        old_count, all_count = class_counts
        bound = d47._scalar_operation_upper_bound(k, old_count, all_count)
        if (
            resource.get("d47_estimated_scalar_operation_upper_bound") != bound
            or resource.get("d47_additional_adaptation_mac_equivalents") != bound
            or resource.get("d47_resource_reuses_d46_exact_inventory") is not True
        ):
            raise D50ProbeError("D50 inherited scalar resource audit drift")

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
                weights, log_evidence = d46._classwise_likelihood_weights(
                    full_ce, block_ce, k
                )
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
        row["resource"]["estimated_adaptation_macs"] = int(
            row["resource"]["estimated_adaptation_macs"]
            - row["resource"]["d47_additional_adaptation_mac_equivalents"]
        )
    d46._verify_d46_fit_audits(sanitized)
    return len(d50_rows)


def _verify_d50_output(
    output: Path,
    probe_script_sha256: str,
    helper_hashes: dict[str, str],
) -> dict[str, Any]:
    evidence = d43._verify_probe_output(output, ARM, probe_script_sha256)
    support = d43._read_json(output / "support_audit.json")
    closure = support.get("candidate_lock", {}).get("source_closure", {})
    if any(closure.get(name) != value for name, value in helper_hashes.items()):
        raise D50ProbeError("D50 helper source closure drift")
    rows = [
        json.loads(line)
        for line in (output / "training_log.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    count = _verify_d50_fit_audits(rows)
    return {**evidence, "verified_d50_fit_row_count": count, **helper_hashes}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--d50-arm", required=True, choices=(ARM,))
    parser.add_argument("--runtime-root", required=True, type=Path)
    parser.add_argument("--probe-root", required=True, type=Path)
    known, runner_arguments = parser.parse_known_args(argv)
    d43._require_locked_runner_arguments(runner_arguments)
    output = d43._runner_output(runner_arguments)
    if output.exists():
        raise D50ProbeError(f"D50 output already exists: {output}")
    probe_script_sha256 = d43._sha256(Path(__file__).resolve())
    helper_hashes = {
        "d50_d47_helper_sha256": d43._sha256(D47_HELPER_PATH),
        "d50_d46_helper_sha256": d43._sha256(d47.D46_HELPER_PATH),
        "d50_d45_helper_sha256": d43._sha256(d46.D45_HELPER_PATH),
        "d50_d44_helper_sha256": d43._sha256(d45.D44_HELPER_PATH),
        "d50_d43_helper_sha256": d43._sha256(d44.D43_HELPER_PATH),
    }
    previous_sys_path = list(sys.path)
    previous_argv = sys.argv
    d42 = None
    package = None
    original_package_path: tuple[str, ...] = ()
    original_fit = None
    original_macs = None
    original_top = None
    runner_module_name = "d50_locked_d42_runner"
    try:
        d42, package, original_package_path = d43._bootstrap(
            known.runtime_root, known.probe_root
        )
        original_fit = d42._fit_equal_prior_lda
        d42._fit_equal_prior_lda = build_centered_median_fit(d42)
        original_macs, original_top = d47._install_d47_resource_accounting(d42)
        runner = known.probe_root / "code" / "scripts" / "run_d25_support_only_concat.py"
        spec = importlib.util.spec_from_file_location(runner_module_name, runner)
        if spec is None or spec.loader is None:
            raise D50ProbeError("D50 could not load the locked D42 runner")
        runner_module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = runner_module
        spec.loader.exec_module(runner_module)
        d43._install_runner_probe_guards(
            runner_module,
            arm=known.d50_arm,
            probe_script_sha256=probe_script_sha256,
            extra_source_closure=helper_hashes,
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
    evidence = _verify_d50_output(
        output, probe_script_sha256, helper_hashes
    )
    metadata = {
        "schema": "cvs.phase2.d50.centered_median_evidence_fusion_probe.v1",
        "status": "DEVELOPMENT_SUPPORT_ONLY_DIAGNOSTIC_PROBE",
        "arm": known.d50_arm,
        "formal_candidate": False,
        "probe_forced_nonpromotable": True,
        "selected_only_full_k10_refit_allowed": False,
        "query_opened": False,
        "probe_script_sha256": probe_script_sha256,
        "weight_formula": WEIGHT_FORMULA,
        "statistical_claim": STATISTICAL_CLAIM,
        "runtime_root": str(known.runtime_root.resolve()),
        "probe_root": str(known.probe_root.resolve()),
        **evidence,
    }
    (output / "D50_PROBE_METADATA.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
