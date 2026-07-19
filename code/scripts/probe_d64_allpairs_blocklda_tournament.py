#!/usr/bin/env python3
"""D64 support-only all-pairs block-LDA tournament probe."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
from typing import Any, Callable

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
D43_HELPER_PATH = SCRIPT_DIR / "probe_d43_structured_covariance.py"
SPEC = importlib.util.spec_from_file_location("d64_d43_probe_helper", D43_HELPER_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("D64 could not load D43 helper")
d43 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(d43)


ARM = "allpairs_blocklda_tournament"
STRUCTURE = "all_pairs_three_block_auto_shrinkage_lda_rms_tournament"
STATE_COVARIANCE_POLICY = "sklearn_lsqr_auto_shrinkage_equal_prior"
FORMULA = (
    "for every unordered class pair fit block3 LDA; normalize oriented margin by "
    "pair-support RMS; average all incident margins into one affine row"
)
if ARM not in d43.ARM_STRUCTURES:
    d43.ARM_STRUCTURES[ARM] = STRUCTURE
if ARM not in d43.ARMS:
    d43.ARMS = tuple((*d43.ARMS, ARM))


class D64ProbeError(RuntimeError):
    pass


def _validate_symmetric_support(
    rows: np.ndarray, labels: np.ndarray, class_count: int, k_shot: int
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(rows, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    if (
        x.ndim != 2
        or y.shape != (len(x),)
        or int(class_count) < 2
        or int(k_shot) < 1
        or len(x) != int(class_count) * int(k_shot)
        or not np.isfinite(x).all()
        or not np.array_equal(np.unique(y), np.arange(int(class_count)))
        or any(int(np.sum(y == index)) != int(k_shot) for index in range(int(class_count)))
    ):
        raise D64ProbeError("D64 requires finite exact symmetric support")
    return x, y


def _normalized_pair_margin(
    pair_fit: Callable[..., tuple[np.ndarray, np.ndarray, dict[str, Any]]],
    pair_rows: np.ndarray,
    pair_labels: np.ndarray,
    k_shot: int,
) -> tuple[np.ndarray, float, float, dict[str, Any]]:
    coefficient, intercept, audit = pair_fit(
        pair_rows, pair_labels, 2, int(k_shot)
    )
    coef = np.asarray(coefficient, dtype=np.float64)
    bias = np.asarray(intercept, dtype=np.float64)
    if coef.shape != (2, pair_rows.shape[1]) or bias.shape != (2,):
        raise D64ProbeError("D64 pair affine shape drift")
    margin_coef = coef[0] - coef[1]
    margin_intercept = float(bias[0] - bias[1])
    raw_margin = pair_rows @ margin_coef + margin_intercept
    scale = float(np.sqrt(np.mean(raw_margin**2)))
    if not np.isfinite(scale) or scale <= np.finfo(np.float64).eps:
        raise D64ProbeError("D64 pair margin RMS is not positive finite")
    normalized_coef = margin_coef / scale
    normalized_intercept = margin_intercept / scale
    normalized_margin = pair_rows @ normalized_coef + normalized_intercept
    predicted = (normalized_margin < 0.0).astype(np.int64)
    pair_accuracy = float(np.mean(predicted == pair_labels))
    return normalized_coef, normalized_intercept, pair_accuracy, {
        **dict(audit),
        "d64_pair_margin_rms": scale,
        "d64_pair_support_accuracy": pair_accuracy,
        "d64_pair_normalized_margin_abs_min": float(np.min(np.abs(normalized_margin))),
        "d64_pair_normalized_margin_abs_mean": float(np.mean(np.abs(normalized_margin))),
        "d64_pair_normalized_margin_abs_max": float(np.max(np.abs(normalized_margin))),
    }


def build_d64_fit(d42: Any) -> tuple[Callable[..., Any], list[dict[str, Any]]]:
    pair_fit = d43.build_structured_fit(d42, "block3_centered")
    call_records: list[dict[str, Any]] = []

    def fit(
        rows: np.ndarray, labels: np.ndarray, class_count: int, k_shot: int
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        x, y = _validate_symmetric_support(rows, labels, class_count, k_shot)
        dimension = int(x.shape[1])
        coefficients = np.zeros((class_count, dimension), dtype=np.float64)
        intercepts = np.zeros(class_count, dtype=np.float64)
        pair_audits: list[dict[str, Any]] = []
        for first in range(int(class_count)):
            for second in range(first + 1, int(class_count)):
                mask = (y == first) | (y == second)
                pair_rows = x[mask]
                pair_labels = (y[mask] == second).astype(np.int64)
                if len(pair_rows) != 2 * int(k_shot):
                    raise D64ProbeError("D64 pair support count drift")
                margin_coef, margin_intercept, pair_accuracy, pair_audit = (
                    _normalized_pair_margin(
                        pair_fit, pair_rows, pair_labels, int(k_shot)
                    )
                )
                coefficients[first] += margin_coef
                intercepts[first] += margin_intercept
                coefficients[second] -= margin_coef
                intercepts[second] -= margin_intercept
                record = {
                    "class_pair": [int(first), int(second)],
                    "class_count": int(class_count),
                    "k_shot": int(k_shot),
                    "pair_margin_rms": pair_audit["d64_pair_margin_rms"],
                    "pair_support_accuracy": pair_accuracy,
                    "pair_covariance_condition_number": pair_audit.get(
                        "d43_covariance_condition_number"
                    ),
                    "pair_unit_covariance_fallback": bool(
                        pair_audit.get("unit_covariance_fallback", False)
                    ),
                }
                call_records.append(record)
                pair_audits.append({**record, **pair_audit})
        divisor = float(int(class_count) - 1)
        coefficients /= divisor
        intercepts /= divisor
        centered_coef, centered_intercept = d43._center_affine_scores(
            coefficients, intercepts
        )
        final_coef = centered_coef.astype(np.float32)
        final_intercept = centered_intercept.astype(np.float32)
        support_scores = x.astype(np.float32) @ final_coef.T + final_intercept[None, :]
        support_accuracy = float(np.mean(np.argmax(support_scores, axis=1) == y))
        expected_pairs = math.comb(int(class_count), 2)
        if len(pair_audits) != expected_pairs:
            raise D64ProbeError("D64 pair closure drift")
        scales = np.asarray(
            [audit["d64_pair_margin_rms"] for audit in pair_audits], dtype=np.float64
        )
        conditions = np.asarray(
            [
                audit.get("d43_covariance_condition_number", 1.0)
                for audit in pair_audits
            ],
            dtype=np.float64,
        )
        pair_accuracies = np.asarray(
            [audit["d64_pair_support_accuracy"] for audit in pair_audits],
            dtype=np.float64,
        )
        audit = {
            "solver": "allpairs_block3_lsqr_auto_shrinkage",
            "shrinkage": "auto_per_pair",
            "prior_policy": "equal_1_over_2_with_equal_pair_tournament_average",
            # The D42 state schema records the supported solver family here;
            # D64's pair-local structure remains explicit in the D43/D64 audit.
            "covariance_policy": STATE_COVARIANCE_POLICY,
            "unit_covariance_fallback": bool(
                any(audit.get("unit_covariance_fallback", False) for audit in pair_audits)
            ),
            "support_rows": int(len(x)),
            "class_count": int(class_count),
            "k_shot": int(k_shot),
            "coefficient_source": "rms_normalized_allpairs_block3_lda_tournament",
            "covariance_equation_residual_max": float(
                max(audit.get("covariance_equation_residual_max", 0.0) for audit in pair_audits)
            ),
            "d43_probe_arm": ARM,
            "d43_covariance_structure": STRUCTURE,
            "d43_class_common_affine_omitted": True,
            "d64_probe_arm": ARM,
            "d64_formula": FORMULA,
            "d64_pair_count": expected_pairs,
            "d64_pair_margin_rms_min": float(np.min(scales)),
            "d64_pair_margin_rms_mean": float(np.mean(scales)),
            "d64_pair_margin_rms_max": float(np.max(scales)),
            "d64_pair_covariance_condition_min": float(np.min(conditions)),
            "d64_pair_covariance_condition_mean": float(np.mean(conditions)),
            "d64_pair_covariance_condition_max": float(np.max(conditions)),
            "d64_pair_support_accuracy_min": float(np.min(pair_accuracies)),
            "d64_pair_support_accuracy_mean": float(np.mean(pair_accuracies)),
            "d64_pair_support_accuracy_max": float(np.max(pair_accuracies)),
            "d64_compiled_support_accuracy": support_accuracy,
            "d64_pair_audits": pair_audits,
            "d64_actual_k": int(k_shot),
            "d64_class_id_specific_formula": False,
            "d64_old_new_role_specific_branch": False,
            "d64_scene_receiver_handle_specific_branch": False,
            "d64_uses_outer_held_or_query": False,
            "d64_hyperparameter_count": 0,
            "d64_pair_graph_persisted_for_query": False,
            "d64_query_joint_optimization": False,
            "d64_single_affine_state_only": True,
            "d64_actual_coefficient_fp32": final_coef.tolist(),
            "d64_actual_intercept_fp32": final_intercept.tolist(),
        }
        return final_coef, final_intercept, audit

    return fit, call_records


def _install_resource_accounting(d42: Any) -> tuple[Any, Any]:
    original_macs = d42._lda_fit_macs
    original_top = d42.fit_d42_unified_shrinkage_lda

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        result = original_top(*args, **kwargs)
        resource = dict(result.resource_audit)
        dimension = int(d42.FEATURE_DIM)
        old_k, new_k = int(resource["old_k_shot"]), int(resource["new_k_shot"])
        old_count, all_count = len(result.before_state.classes), len(result.state.classes)
        pair_specs = ((old_k, old_count), (new_k, all_count))
        pair_fit_count = int(sum(math.comb(count, 2) for _, count in pair_specs))
        pair_lda_macs = int(
            sum(
                math.comb(count, 2) * original_macs(2 * k_shot, 2)
                for k_shot, count in pair_specs
            )
        )
        normalization_macs = int(
            sum(
                math.comb(count, 2) * 2 * k_shot * (2 * dimension + 3)
                for k_shot, count in pair_specs
            )
        )
        compilation_macs = int(
            sum(math.comb(count, 2) * 2 * (dimension + 1) for _, count in pair_specs)
        )
        resource.update(
            {
                "lda_closed_form_fit_count": pair_fit_count,
                "estimated_lda_fit_macs": pair_lda_macs,
                "d64_pair_fit_count": pair_fit_count,
                "d64_pair_margin_normalization_macs": normalization_macs,
                "d64_pair_affine_compilation_macs": compilation_macs,
                "d64_query_extra_macs": 0,
                "d64_persistent_state_extra_bytes": 0,
                "d64_optimizer_steps_extra": 0,
                "d64_pair_graph_persisted_for_query": False,
                "d64_resource_single_affine_state_only": True,
            }
        )
        resource["estimated_adaptation_macs"] = int(
            resource["estimated_metric_adaptation_macs"]
            + pair_lda_macs
            + normalization_macs
            + compilation_macs
        )
        return replace(result, resource_audit=resource)

    d42.fit_d42_unified_shrinkage_lda = wrapped
    return original_macs, original_top


def _verify_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    target = [
        row
        for row in rows
        if row.get("candidate_id") in ("D42-USLDA-INT8", "D42-USLDA-FP32-MATCHED")
    ]
    if len(rows) != 105 or len(target) != 30:
        raise D64ProbeError("D64 training row closure drift")
    pair_fit_audits = pair_count = 0
    for row in target:
        resource = row["resource"]
        if (
            int(resource.get("d64_pair_fit_count", -1)) != 70
            or int(resource.get("d64_query_extra_macs", -1)) != 0
            or resource.get("d64_pair_graph_persisted_for_query") is not False
            or resource.get("d64_resource_single_affine_state_only") is not True
        ):
            raise D64ProbeError("D64 resource closure drift")
        for field, expected_pairs in (
            ("before_covariance_audit", 15),
            ("final_covariance_audit", 55),
        ):
            audit = row["geometry_summary"][field]
            expected = {
                "d43_probe_arm": ARM,
                "d43_covariance_structure": STRUCTURE,
                "d43_class_common_affine_omitted": True,
                "d64_probe_arm": ARM,
                "d64_formula": FORMULA,
                "d64_actual_k": 8,
                "d64_class_id_specific_formula": False,
                "d64_old_new_role_specific_branch": False,
                "d64_scene_receiver_handle_specific_branch": False,
                "d64_uses_outer_held_or_query": False,
                "d64_hyperparameter_count": 0,
                "d64_pair_graph_persisted_for_query": False,
                "d64_query_joint_optimization": False,
                "d64_single_affine_state_only": True,
                "d64_pair_count": expected_pairs,
            }
            if any(audit.get(name) != value for name, value in expected.items()):
                raise D64ProbeError("D64 exact audit drift")
            if len(audit.get("d64_pair_audits", [])) != expected_pairs:
                raise D64ProbeError("D64 pair audit closure drift")
            pair_fit_audits += 1
            pair_count += expected_pairs
    return {
        "verified_d64_target_row_count": len(target),
        "verified_d64_fit_audit_count": pair_fit_audits,
        "verified_d64_pair_fit_count": pair_count,
    }


def _verify_output(
    output: Path, script_sha: str, helper_hashes: dict[str, str]
) -> dict[str, Any]:
    evidence = d43._verify_probe_output(output, ARM, script_sha)
    support = d43._read_json(output / "support_audit.json")
    closure = support.get("candidate_lock", {}).get("source_closure", {})
    if any(closure.get(name) != value for name, value in helper_hashes.items()):
        raise D64ProbeError("D64 helper source closure drift")
    rows = [
        json.loads(line)
        for line in (output / "training_log.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return {**evidence, **_verify_rows(rows)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--d64-arm", required=True, choices=(ARM,))
    parser.add_argument("--runtime-root", required=True, type=Path)
    parser.add_argument("--probe-root", required=True, type=Path)
    known, runner_arguments = parser.parse_known_args(argv)
    d43._require_locked_runner_arguments(runner_arguments)
    output = d43._runner_output(runner_arguments)
    if output.exists():
        raise D64ProbeError(f"D64 output already exists: {output}")
    script_sha = d43._sha256(Path(__file__).resolve())
    helper_hashes = {"d64_d43_helper_sha256": d43._sha256(D43_HELPER_PATH)}
    previous_sys_path, previous_argv = list(sys.path), sys.argv
    d42 = package = None
    original_path: tuple[str, ...] = ()
    original_fit = original_macs = original_top = None
    runner_name, exit_code = "d64_locked_d42_runner", 1
    call_records: list[dict[str, Any]] = []
    try:
        d42, package, original_path = d43._bootstrap(known.runtime_root, known.probe_root)
        original_fit = d42._fit_equal_prior_lda
        fit, call_records = build_d64_fit(d42)
        d42._fit_equal_prior_lda = fit
        original_macs, original_top = _install_resource_accounting(d42)
        runner = known.probe_root / "code" / "scripts" / "run_d25_support_only_concat.py"
        runner_spec = importlib.util.spec_from_file_location(runner_name, runner)
        if runner_spec is None or runner_spec.loader is None:
            raise D64ProbeError("D64 could not load locked runner")
        runner_module = importlib.util.module_from_spec(runner_spec)
        sys.modules[runner_spec.name] = runner_module
        runner_spec.loader.exec_module(runner_module)
        d43._install_runner_probe_guards(
            runner_module,
            arm=known.d64_arm,
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
        if package is not None:
            package.__path__[:] = list(original_path)
        sys.modules.pop(runner_name, None)
    if exit_code != 0:
        return exit_code
    expected_calls = 30 * (math.comb(6, 2) + math.comb(11, 2))
    if len(call_records) != expected_calls:
        raise D64ProbeError(
            f"D64 pair-fit count drift: {len(call_records)} != {expected_calls}"
        )
    evidence = _verify_output(output, script_sha, helper_hashes)
    record_sha = hashlib.sha256(
        json.dumps(call_records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    metadata = {
        "schema": "cvs.phase2.d64.allpairs_blocklda_tournament_probe.v1",
        "status": "DEVELOPMENT_SUPPORT_ONLY_DIAGNOSTIC_PROBE",
        "arm": known.d64_arm,
        "formal_candidate": False,
        "probe_forced_nonpromotable": True,
        "selected_only_full_k10_refit_allowed": False,
        "query_opened": False,
        "probe_script_sha256": script_sha,
        "formula": FORMULA,
        "pair_fit_execution_count": len(call_records),
        "pair_fit_record_sha256": record_sha,
        "runtime_root": str(known.runtime_root.resolve()),
        "probe_root": str(known.probe_root.resolve()),
        **evidence,
    }
    (output / "D64_PROBE_METADATA.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
