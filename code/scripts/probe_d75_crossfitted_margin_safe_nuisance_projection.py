#!/usr/bin/env python3
"""D75 nested support-held margin-safe nuisance-projection probe."""

from __future__ import annotations

import argparse
import copy
from dataclasses import replace
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
D74_HELPER_PATH = SCRIPT_DIR / "probe_d74_orthogonal_nuisance_direction_removal.py"
CORE_PATH = SCRIPT_DIR.parent / "cvsrffi" / "stage2_d75_crossfitted_margin_safe_projection.py"


def _load(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"D75 could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


d74 = _load("d75_d74_probe_helper", D74_HELPER_PATH)
core = _load("d75_margin_safe_projection_core", CORE_PATH)
d62, d43 = d74.d62, d74.d43

ARM = "crossfitted_margin_safe_nuisance_projection"
STRUCTURE = d62.STRUCTURE
FORMULA = (
    "freeze D62 before/final state; obtain the unique D74 rank-one proposal; "
    "accept it only when leave-one-physical-rank equal-prior LDA true-vs-best-other "
    "margin does not decrease for any registered class, does not decrease globally, "
    "and does not reduce held correctness; otherwise exact D62 identity fallback"
)
d43.ARM_STRUCTURES[ARM] = STRUCTURE
if ARM not in d43.ARMS:
    d43.ARMS = tuple((*d43.ARMS, ARM))
d74.ARM = ARM
d74.FORMULA = FORMULA


class D75ProbeError(RuntimeError):
    pass


def build_d75_fit(d42: Any) -> tuple[Any, list[dict[str, Any]]]:
    base_fit, component_records = d62.build_d62_fit(d42)

    def fit(
        rows: np.ndarray, labels: np.ndarray, class_count: int, k_shot: int
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        coefficient, intercept, audit = base_fit(rows, labels, class_count, k_shot)
        audit.update(
            {
                "d43_probe_arm": ARM,
                "d43_covariance_structure": STRUCTURE,
                "d75_probe_arm": ARM,
                "d75_formula": FORMULA,
            }
        )
        return coefficient, intercept, audit

    return fit, component_records


class MarginSafeRegistry(d74.ProjectionRegistry):
    def __init__(self, d42: Any, base_fit: Any, native_lda_fit: Any, native_lda_macs: Any) -> None:
        super().__init__(d42, base_fit)
        self.native_lda_fit = native_lda_fit
        self.native_lda_macs = native_lda_macs
        self.base_direction_fit = d74.core.fit_orthogonal_nuisance_direction
        self.accepted_count = 0
        self.rejected_count = 0

    def _gated_fit(
        self,
        rows: np.ndarray,
        labels: np.ndarray,
        class_count: int,
        k_shot: int,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        return core.fit_crossfitted_margin_safe_projection(
            rows,
            labels,
            class_count,
            k_shot,
            direction_fit=self.base_direction_fit,
            lda_fit=self.native_lda_fit,
        )

    def wrap_top(self, base_top: Any) -> Any:
        parent = super().wrap_top(base_top)

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            original = d74.core.fit_orthogonal_nuisance_direction
            d74.core.fit_orthogonal_nuisance_direction = self._gated_fit
            try:
                result = parent(*args, **kwargs)
            finally:
                d74.core.fit_orthogonal_nuisance_direction = original
            geometry = dict(result.geometry_audit)
            audit = dict(geometry["d74_projection_audit"])
            gate_pass = bool(audit["crossfit_gate_pass"])
            self.accepted_count += int(gate_pass)
            self.rejected_count += int(not gate_pass)
            geometry.update(
                {
                    "d75_probe_arm": ARM,
                    "d75_formula": FORMULA,
                    "d75_projection_audit": audit,
                    "d75_before_state_exact_d62_unchanged": True,
                    "d75_class_id_specific_formula": False,
                    "d75_old_new_role_specific_branch": False,
                    "d75_query_role_specific_branch": False,
                    "d75_scene_receiver_handle_specific_branch": False,
                    "d75_uses_outer_held_or_query_for_fit": False,
                    "d75_query_joint_optimization": False,
                    "d75_ground_component_input_count": 0,
                    "d75_projection_direction_persisted": False,
                    "d75_projection_compiled_into_affine": gate_pass,
                    "d75_single_affine_state_only": True,
                    "d75_dense_query_graph_bytes": 0,
                    "stage2c_classifier": (
                        "d75_margin_safe_projected_frozen_d62_compiled_affine"
                        if gate_pass
                        else "d75_margin_rejected_exact_d62_fallback"
                    ),
                }
            )
            resource = dict(result.resource_audit)
            k = int(audit["crossfit_fold_count"])
            classes = int(audit["class_count"])
            dimension = int(audit["dimension"])
            lda_macs = int(k * self.native_lda_macs((k - 1) * classes, classes))
            loo_projection_macs = int(
                k * d74._projection_mac_upper_bound(dimension, classes, k - 1)
            )
            crossfit_total = int(lda_macs + loo_projection_macs)
            resource.update(
                {
                    "d75_crossfit_fold_count": k,
                    "d75_crossfit_held_row_count": int(
                        audit["crossfit_held_row_count"]
                    ),
                    "d75_crossfit_lda_fit_count": k,
                    "d75_crossfit_lda_fit_macs": lda_macs,
                    "d75_crossfit_projection_mac_upper_bound": loo_projection_macs,
                    "d75_crossfit_total_added_adaptation_macs": crossfit_total,
                    "d75_projection_accepted": gate_pass,
                    "d75_projection_removed_rank": int(
                        audit["projection_removed_rank"]
                    ),
                    "d75_query_extra_mac_equivalents": 0,
                    "d75_persistent_state_extra_bytes": 0,
                    "d75_ground_component_input_count": 0,
                    "d75_dense_query_graph_bytes": 0,
                    "d75_single_affine_state_only": True,
                }
            )
            resource["lda_closed_form_fit_count"] = int(
                resource["lda_closed_form_fit_count"] + k
            )
            resource["estimated_lda_fit_macs"] = int(
                resource["estimated_lda_fit_macs"] + lda_macs
            )
            resource["estimated_adaptation_macs"] = int(
                resource["estimated_adaptation_macs"] + crossfit_total
            )
            resource["estimated_metric_adaptation_macs"] = int(
                resource["estimated_metric_adaptation_macs"]
                + loo_projection_macs
            )
            return replace(result, geometry_audit=geometry, resource_audit=resource)

        return wrapped


def _sanitize_for_d62(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sanitized = d74._sanitize_for_d62(rows)
    for row in sanitized:
        if row.get("candidate_id") not in (
            "D42-USLDA-INT8",
            "D42-USLDA-FP32-MATCHED",
        ):
            continue
        resource = row["resource"]
        resource["lda_closed_form_fit_count"] -= resource[
            "d75_crossfit_lda_fit_count"
        ]
        resource["estimated_lda_fit_macs"] -= resource[
            "d75_crossfit_lda_fit_macs"
        ]
        resource["estimated_adaptation_macs"] -= resource[
            "d75_crossfit_total_added_adaptation_macs"
        ]
        resource["estimated_metric_adaptation_macs"] -= resource[
            "d75_crossfit_projection_mac_upper_bound"
        ]
    return sanitized


def _verify_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    target = [
        row
        for row in rows
        if row.get("candidate_id")
        in ("D42-USLDA-INT8", "D42-USLDA-FP32-MATCHED")
    ]
    if len(rows) != 105 or len(target) != 30:
        raise D75ProbeError("D75 training row closure drift")
    d62_evidence = d62._verify_rows(_sanitize_for_d62(rows))
    accepted = rejected = 0
    unique_proposals: set[str] = set()
    min_class_deltas: list[float] = []
    for row in target:
        geometry, resource = row["geometry_summary"], row["resource"]
        if any(
            geometry.get(name) != value
            for name, value in {
                "d75_probe_arm": ARM,
                "d75_formula": FORMULA,
                "d75_before_state_exact_d62_unchanged": True,
                "d75_class_id_specific_formula": False,
                "d75_old_new_role_specific_branch": False,
                "d75_query_role_specific_branch": False,
                "d75_scene_receiver_handle_specific_branch": False,
                "d75_uses_outer_held_or_query_for_fit": False,
                "d75_query_joint_optimization": False,
                "d75_ground_component_input_count": 0,
                "d75_projection_direction_persisted": False,
                "d75_single_affine_state_only": True,
                "d75_dense_query_graph_bytes": 0,
            }.items()
        ):
            raise D75ProbeError("D75 geometry closure drift")
        audit = geometry["d75_projection_audit"]
        gate = bool(audit["crossfit_gate_pass"])
        accepted += int(gate)
        rejected += int(not gate)
        if (
            int(audit["k_shot"]) != 8
            or int(audit["dimension"]) != 288
            or int(audit["crossfit_fold_count"]) != 8
            or int(audit["crossfit_lda_fit_count"]) != 8
            or int(audit["crossfit_held_row_count"]) != 88
            or len(audit["crossfit_per_class_margin_delta_mean"]) != 11
            or int(audit["query_rows_used"]) != 0
            or int(audit["ground_component_input_count"]) != 0
        ):
            raise D75ProbeError("D75 crossfit audit closure drift")
        tolerance = float(audit["crossfit_numeric_tolerance"])
        if gate:
            if (
                audit["status"] != "crossfitted_margin_safe_projection_active"
                or int(audit["projection_removed_rank"]) != 1
                or float(audit["crossfit_margin_delta_min_class_mean"])
                < -tolerance
                or float(audit["crossfit_margin_delta_mean"]) < -tolerance
                or int(audit["crossfit_correct_delta"]) < 0
            ):
                raise D75ProbeError("D75 accepted gate drift")
        elif (
            audit["status"]
            != "crossfitted_margin_rejected_exact_d62_fallback"
            or int(audit["projection_removed_rank"]) != 0
        ):
            raise D75ProbeError("D75 rejected gate drift")
        if any(
            resource.get(name) != value
            for name, value in {
                "d75_crossfit_fold_count": 8,
                "d75_crossfit_held_row_count": 88,
                "d75_crossfit_lda_fit_count": 8,
                "d75_projection_accepted": gate,
                "d75_projection_removed_rank": int(gate),
                "d75_query_extra_mac_equivalents": 0,
                "d75_persistent_state_extra_bytes": 0,
                "d75_ground_component_input_count": 0,
                "d75_dense_query_graph_bytes": 0,
                "d75_single_affine_state_only": True,
            }.items()
        ):
            raise D75ProbeError("D75 resource closure drift")
        if (
            int(resource["optimizer_steps"]) != 20
            or int(resource["adaptation_epochs"]) != 20
            or len(resource["complete_loss_trace"]) != 20
            or int(resource["persistent_state_bytes"]) > 256 * 1024
        ):
            raise D75ProbeError("D75 cap/resource drift")
        unique_proposals.add(str(audit["proposed_direction_sha256"]))
        min_class_deltas.append(
            float(audit["crossfit_margin_delta_min_class_mean"])
        )
    return {
        **d62_evidence,
        "verified_d75_target_row_count": len(target),
        "verified_d75_crossfit_audit_count": len(target),
        "verified_d75_accepted_count": accepted,
        "verified_d75_rejected_count": rejected,
        "verified_d75_unique_proposal_count": len(unique_proposals),
        "verified_d75_min_class_margin_delta_min": min(min_class_deltas),
        "verified_d75_min_class_margin_delta_max": max(min_class_deltas),
        "verified_d75_ground_component_input_count": 0,
    }


def _verify_output(output: Path, script_sha: str, helper_hashes: dict[str, str]) -> dict[str, Any]:
    evidence = d43._verify_probe_output(output, ARM, script_sha)
    support = d43._read_json(output / "support_audit.json")
    closure = support.get("candidate_lock", {}).get("source_closure", {})
    if any(closure.get(name) != value for name, value in helper_hashes.items()):
        raise D75ProbeError("D75 helper source closure drift")
    rows = [
        json.loads(line)
        for line in (output / "training_log.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return {**evidence, **_verify_rows(rows)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--d75-arm", required=True, choices=(ARM,))
    parser.add_argument("--runtime-root", required=True, type=Path)
    parser.add_argument("--probe-root", required=True, type=Path)
    known, runner_arguments = parser.parse_known_args(argv)
    d43._require_locked_runner_arguments(runner_arguments)
    output = d43._runner_output(runner_arguments)
    if output.exists():
        raise D75ProbeError(f"D75 output already exists: {output}")
    script_sha = d43._sha256(Path(__file__).resolve())
    helper_hashes = {
        "d75_core_sha256": d43._sha256(CORE_PATH),
        "d75_d74_helper_sha256": d43._sha256(D74_HELPER_PATH),
        "d75_d62_helper_sha256": d43._sha256(d74.D62_HELPER_PATH),
        "d75_d61_helper_sha256": d43._sha256(d62.D61_HELPER_PATH),
        "d75_d46_helper_sha256": d43._sha256(d62.d61.D46_HELPER_PATH),
        "d75_d43_helper_sha256": d43._sha256(d62.d61.d46.d44.D43_HELPER_PATH),
    }
    previous_sys_path, previous_argv = list(sys.path), sys.argv
    d42 = package = registry = None
    original_path: tuple[str, ...] = ()
    original_fit = original_macs = original_top = None
    runner_name, exit_code = "d75_locked_d42_runner", 1
    component_records: list[dict[str, Any]] = []
    try:
        d42, package, original_path = d43._bootstrap(known.runtime_root, known.probe_root)
        original_fit, original_macs = d42._fit_equal_prior_lda, d42._lda_fit_macs
        fit, component_records = build_d75_fit(d42)
        d42._fit_equal_prior_lda = fit
        _, original_top = d62._install_resource_accounting(d42)
        d62_top = d42.fit_d42_unified_shrinkage_lda
        registry = MarginSafeRegistry(d42, fit, original_fit, original_macs)
        d42.fit_d42_unified_shrinkage_lda = registry.wrap_top(d62_top)
        runner = known.probe_root / "code" / "scripts" / "run_d25_support_only_concat.py"
        runner_spec = importlib.util.spec_from_file_location(runner_name, runner)
        if runner_spec is None or runner_spec.loader is None:
            raise D75ProbeError("D75 could not load locked runner")
        runner_module = importlib.util.module_from_spec(runner_spec)
        sys.modules[runner_spec.name] = runner_module
        runner_spec.loader.exec_module(runner_module)
        d43._install_runner_probe_guards(
            runner_module,
            arm=known.d75_arm,
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
    if (
        registry is None
        or registry.top_fit_count != 30
        or registry.extra_d62_fit_count != 0
        or len(registry.records) != 30
        or registry.accepted_count + registry.rejected_count != 30
        or len(component_records) != 1080
    ):
        raise D75ProbeError("D75 fit/component call closure drift")
    evidence = _verify_output(output, script_sha, helper_hashes)
    record_sha = hashlib.sha256(
        json.dumps(registry.records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    metadata = {
        "schema": "cvs.phase2.d75.crossfitted_margin_safe_projection_probe.v1",
        "status": "DEVELOPMENT_SUPPORT_ONLY_DIAGNOSTIC_PROBE",
        "arm": known.d75_arm,
        "formal_candidate": False,
        "probe_forced_nonpromotable": True,
        "selected_only_full_k10_refit_allowed": False,
        "query_opened": False,
        "probe_script_sha256": script_sha,
        "formula": FORMULA,
        "top_fit_count": registry.top_fit_count,
        "extra_d62_fit_count": registry.extra_d62_fit_count,
        "accepted_count": registry.accepted_count,
        "rejected_count": registry.rejected_count,
        "component_fit_execution_count": len(component_records),
        "fit_record_sha256": record_sha,
        "runtime_root": str(known.runtime_root.resolve()),
        "probe_root": str(known.probe_root.resolve()),
        **evidence,
    }
    (output / "D75_PROBE_METADATA.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
