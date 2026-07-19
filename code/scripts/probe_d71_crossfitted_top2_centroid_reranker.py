#!/usr/bin/env python3
"""D71 support-only cross-fitted top-2 centroid-reranker probe."""

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
D62_HELPER_PATH = SCRIPT_DIR / "probe_d62_crossfitted_fisher_row_splice.py"
CORE_PATH = SCRIPT_DIR.parent / "cvsrffi" / "stage2_d71_top2_centroid_reranker.py"


def _load(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"D71 could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


d62 = _load("d71_d62_probe_helper", D62_HELPER_PATH)
core = _load("d71_top2_centroid_core", CORE_PATH)
d43 = d62.d43

ARM = "crossfitted_top2_centroid_reranker"
STRUCTURE = d62.STRUCTURE
FORMULA = (
    "retain D62 all-class joint scores; only accepted cross-fitted centroid "
    "pairs may reorder the current top two classes"
)
d43.ARM_STRUCTURES[ARM] = STRUCTURE
if ARM not in d43.ARMS:
    d43.ARMS = tuple((*d43.ARMS, ARM))


class D71ProbeError(RuntimeError):
    pass


class RerankerRegistry:
    def __init__(self, d42: Any, base_fit: Any, original_score: Any) -> None:
        self.d42 = d42
        self.base_fit = base_fit
        self.original_score = original_score
        self.states: dict[int, tuple[Any, Any]] = {}
        self.records: list[dict[str, Any]] = []
        self.top_fit_count = 0
        self.inner_base_fit_count = 0
        self.score_call_count = 0
        self.reranked_prediction_count = 0

    def _register(self, base_state: Any, pair_state: Any) -> None:
        self.states[id(base_state)] = (base_state, pair_state)

    def wrap_top(self, base_top: Any) -> Any:
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            result = base_top(*args, **kwargs)
            old_features = np.asarray(args[0], dtype=np.float32)
            old_labels = tuple(str(value) for value in args[1])
            old_classes = tuple(str(value) for value in args[2])
            new_features = np.asarray(args[3], dtype=np.float32)
            new_labels = tuple(str(value) for value in args[4])
            new_classes = tuple(str(value) for value in args[5])
            old_targets = np.asarray(
                [old_classes.index(value) for value in old_labels], dtype=np.int64
            )
            all_classes = old_classes + new_classes
            all_features = np.concatenate([old_features, new_features], axis=0)
            all_labels = old_labels + new_labels
            all_targets = np.asarray(
                [all_classes.index(value) for value in all_labels], dtype=np.int64
            )
            old_k = int(len(old_features) // len(old_classes))
            all_k = int(len(all_features) // len(all_classes))
            old_transformed = self.d42._transform(
                old_features, result.before_state.log_diag_fp32
            )
            all_transformed = self.d42._transform(
                all_features, result.state.log_diag_fp32
            )
            before_int8, before_fp32, before_audit = (
                core.fit_crossfitted_pair_reranker(
                    old_transformed,
                    old_targets,
                    len(old_classes),
                    old_k,
                    self.base_fit,
                )
            )
            final_int8, final_fp32, final_audit = (
                core.fit_crossfitted_pair_reranker(
                    all_transformed,
                    all_targets,
                    len(all_classes),
                    all_k,
                    self.base_fit,
                )
            )
            self._register(result.before_state, before_int8)
            self._register(result.matched_fp32_before_state, before_fp32)
            self._register(result.state, final_int8)
            self._register(result.matched_fp32_state, final_fp32)
            self.top_fit_count += 1
            self.inner_base_fit_count += int(
                before_audit["inner_base_fit_count"]
                + final_audit["inner_base_fit_count"]
            )
            geometry = dict(result.geometry_audit)
            geometry.update(
                {
                    "d71_probe_arm": ARM,
                    "d71_formula": FORMULA,
                    "d71_before_reranker_audit": before_audit,
                    "d71_final_reranker_audit": final_audit,
                    "d71_class_id_specific_formula": False,
                    "d71_old_new_role_specific_branch": False,
                    "d71_scene_receiver_handle_specific_branch": False,
                    "d71_uses_outer_held_or_query_for_fit": False,
                    "d71_query_joint_optimization": False,
                    "d71_ground_component_input_count": 0,
                    "d71_top2_only_no_third_class_introduction": True,
                    "d71_dense_query_graph_bytes": 0,
                }
            )
            dimension = int(self.d42.FEATURE_DIM)
            inner_specs = ((4, 6), (4, 11), (4, 6), (4, 11))
            extra_component_fits = 0
            extra_lda = 0
            for k_shot, class_count in inner_specs:
                fits = 2 * (k_shot + 1)
                extra_component_fits += fits
                extra_lda += 2 * int(
                    self.d42._lda_fit_macs(k_shot * class_count, class_count)
                )
                extra_lda += 2 * k_shot * int(
                    self.d42._lda_fit_macs(
                        (k_shot - 1) * class_count, class_count
                    )
                )
            fisher = d62.d61._fisher_dense_macs(dimension, extra_component_fits)
            pair_count_before = len(old_classes) * (len(old_classes) - 1) // 2
            pair_count_final = len(all_classes) * (len(all_classes) - 1) // 2
            pair_scalar = int(
                dimension
                * (
                    3 * (len(old_features) + len(all_features))
                    + 3 * (pair_count_before + pair_count_final)
                    + 2 * old_k * pair_count_before
                    + 2 * all_k * pair_count_final
                )
            )
            held_base_score = int(
                dimension
                * (
                    len(old_features) * len(old_classes)
                    + len(all_features) * len(all_classes)
                )
            )
            gate_scalar = int(
                8
                * (
                    len(old_features) * len(old_classes)
                    + len(all_features) * len(all_classes)
                )
            )
            added = extra_lda + fisher + pair_scalar + held_base_score + gate_scalar
            query_extra = int(5 * dimension)
            int8_pair_bytes = int(final_int8.persistent_state_bytes)
            resource = dict(result.resource_audit)
            resource.update(
                {
                    "d71_inner_d62_fit_count": 4,
                    "d71_inner_component_fit_count": extra_component_fits,
                    "d71_inner_lda_fit_macs": extra_lda,
                    "d71_inner_fisher_dense_mac_upper_bound": fisher,
                    "d71_pair_fit_and_score_scalar_mac_equivalents": pair_scalar,
                    "d71_held_base_score_macs": held_base_score,
                    "d71_gate_scalar_mac_equivalents": gate_scalar,
                    "d71_total_added_adaptation_macs": added,
                    "d71_query_extra_mac_equivalents": query_extra,
                    "d71_int8_pair_state_extra_bytes": int8_pair_bytes,
                    "d71_combined_int8_persistent_state_bytes": int(
                        result.state.persistent_state_bytes + int8_pair_bytes
                    ),
                    "d71_before_active_pair_count": int(
                        before_audit["active_pair_count"]
                    ),
                    "d71_final_active_pair_count": int(
                        final_audit["active_pair_count"]
                    ),
                    "d71_ground_component_input_count": 0,
                    "d71_dense_query_graph_bytes": 0,
                    "d71_top2_only": True,
                    "d71_single_affine_state_only": False,
                }
            )
            resource["lda_closed_form_fit_count"] = int(
                resource["lda_closed_form_fit_count"] + extra_component_fits
            )
            resource["estimated_lda_fit_macs"] = int(
                resource["estimated_lda_fit_macs"] + extra_lda
            )
            resource["estimated_adaptation_macs"] = int(
                resource["estimated_adaptation_macs"] + added
            )
            resource["estimated_macs_per_query"] = int(
                resource["estimated_macs_per_query"] + query_extra
            )
            self.records.append(
                {
                    "before_gate_status": before_audit["gate_status"],
                    "final_gate_status": final_audit["gate_status"],
                    "before_active_pair_count": before_audit["active_pair_count"],
                    "final_active_pair_count": final_audit["active_pair_count"],
                    "before_pair_state_bytes": before_int8.persistent_state_bytes,
                    "final_pair_state_bytes": final_int8.persistent_state_bytes,
                }
            )
            return replace(
                result, geometry_audit=geometry, resource_audit=resource
            )

        return wrapped

    def score(self, state: Any, features: np.ndarray) -> np.ndarray:
        base_scores = self.original_score(state, features)
        record = self.states.get(id(state))
        if record is None:
            return base_scores
        registered_state, pair_state = record
        if registered_state is not state:
            raise D71ProbeError("D71 state identity drift")
        transformed = self.d42._transform(
            np.asarray(features, dtype=np.float32), state.log_diag_fp32
        )
        scores, changed = core.score_with_pair_state(
            base_scores, transformed, pair_state
        )
        self.score_call_count += 1
        self.reranked_prediction_count += int(changed)
        result = np.ascontiguousarray(scores, dtype=np.float32)
        result.setflags(write=False)
        return result


def build_d71_fit(d42: Any) -> tuple[Any, list[dict[str, Any]]]:
    base_fit, component_records = d62.build_d62_fit(d42)

    def fit(
        rows: np.ndarray, labels: np.ndarray, class_count: int, k_shot: int
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        coefficient, intercept, audit = base_fit(
            rows, labels, class_count, k_shot
        )
        audit.update(
            {
                "d43_probe_arm": ARM,
                "d43_covariance_structure": STRUCTURE,
                "d71_probe_arm": ARM,
                "d71_formula": FORMULA,
            }
        )
        return coefficient, intercept, audit

    return fit, component_records


def _verify_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    target = [
        row
        for row in rows
        if row.get("candidate_id")
        in ("D42-USLDA-INT8", "D42-USLDA-FP32-MATCHED")
    ]
    if len(rows) != 105 or len(target) != 30:
        raise D71ProbeError("D71 training row closure drift")
    sanitized = copy.deepcopy(rows)
    for row in sanitized:
        if row.get("candidate_id") not in (
            "D42-USLDA-INT8",
            "D42-USLDA-FP32-MATCHED",
        ):
            continue
        resource = row["resource"]
        resource["lda_closed_form_fit_count"] -= resource[
            "d71_inner_component_fit_count"
        ]
        resource["estimated_lda_fit_macs"] -= resource[
            "d71_inner_lda_fit_macs"
        ]
        resource["estimated_adaptation_macs"] -= resource[
            "d71_total_added_adaptation_macs"
        ]
        resource["estimated_macs_per_query"] -= resource[
            "d71_query_extra_mac_equivalents"
        ]
        for field in ("before_covariance_audit", "final_covariance_audit"):
            row["geometry_summary"][field]["d43_probe_arm"] = d62.ARM
    d62_evidence = d62._verify_rows(sanitized)
    active_before = active_final = accepted_before = accepted_final = 0
    atomic_fallback = 0
    for row in target:
        geometry = row["geometry_summary"]
        resource = row["resource"]
        if any(
            geometry.get(name) != value
            for name, value in {
                "d71_probe_arm": ARM,
                "d71_formula": FORMULA,
                "d71_class_id_specific_formula": False,
                "d71_old_new_role_specific_branch": False,
                "d71_scene_receiver_handle_specific_branch": False,
                "d71_uses_outer_held_or_query_for_fit": False,
                "d71_query_joint_optimization": False,
                "d71_ground_component_input_count": 0,
                "d71_top2_only_no_third_class_introduction": True,
                "d71_dense_query_graph_bytes": 0,
            }.items()
        ):
            raise D71ProbeError("D71 geometry closure drift")
        before = geometry["d71_before_reranker_audit"]
        final = geometry["d71_final_reranker_audit"]
        for audit, class_count, pair_count in ((before, 6, 15), (final, 11, 55)):
            if (
                len(audit["final_accept_mask"]) != pair_count
                or audit["partition_exact_once"] is not True
                or len(audit["partition_audit"]) != 2
                or any(
                    part["train_held_overlap_count"] != 0
                    for part in audit["partition_audit"]
                )
                or len(audit["base_positive"]) != class_count
                or len(audit["joint_positive"]) != class_count
            ):
                raise D71ProbeError("D71 gate/partition closure drift")
            mask = np.asarray(audit["final_accept_mask"], dtype=bool)
            if mask.any() and (
                not np.all(
                    np.asarray(audit["joint_positive"])
                    >= np.asarray(audit["base_positive"])
                )
                or not np.all(
                    np.asarray(audit["joint_false_positive"])
                    <= np.asarray(audit["base_false_positive"])
                )
            ):
                raise D71ProbeError("D71 accepted pairs are not atomic safe")
        before_count = int(before["active_pair_count"])
        final_count = int(final["active_pair_count"])
        active_before += int(before_count > 0)
        active_final += int(final_count > 0)
        accepted_before += before_count
        accepted_final += final_count
        atomic_fallback += int(
            final["gate_status"] == "joint_atomic_failure_exact_d62_fallback"
        )
        if any(
            resource.get(name) != value
            for name, value in {
                "d71_inner_d62_fit_count": 4,
                "d71_inner_component_fit_count": 40,
                "d71_ground_component_input_count": 0,
                "d71_dense_query_graph_bytes": 0,
                "d71_top2_only": True,
                "d71_single_affine_state_only": False,
            }.items()
        ):
            raise D71ProbeError("D71 resource closure drift")
        if int(resource["d71_combined_int8_persistent_state_bytes"]) > 256 * 1024:
            raise D71ProbeError("D71 state cap drift")
    return {
        **d62_evidence,
        "verified_d71_target_row_count": len(target),
        "verified_d71_fit_audit_count": 2 * len(target),
        "verified_d71_active_before_fit_count": active_before,
        "verified_d71_active_final_fit_count": active_final,
        "verified_d71_accepted_before_pair_count": accepted_before,
        "verified_d71_accepted_final_pair_count": accepted_final,
        "verified_d71_final_atomic_fallback_count": atomic_fallback,
        "verified_d71_ground_component_input_count": 0,
    }


def _verify_output(
    output: Path, script_sha: str, helper_hashes: dict[str, str]
) -> dict[str, Any]:
    evidence = d43._verify_probe_output(output, ARM, script_sha)
    support = d43._read_json(output / "support_audit.json")
    closure = support.get("candidate_lock", {}).get("source_closure", {})
    if any(closure.get(name) != value for name, value in helper_hashes.items()):
        raise D71ProbeError("D71 helper source closure drift")
    rows = [
        json.loads(line)
        for line in (output / "training_log.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    return {**evidence, **_verify_rows(rows)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--d71-arm", required=True, choices=(ARM,))
    parser.add_argument("--runtime-root", required=True, type=Path)
    parser.add_argument("--probe-root", required=True, type=Path)
    known, runner_arguments = parser.parse_known_args(argv)
    d43._require_locked_runner_arguments(runner_arguments)
    output = d43._runner_output(runner_arguments)
    if output.exists():
        raise D71ProbeError(f"D71 output already exists: {output}")
    script_sha = d43._sha256(Path(__file__).resolve())
    helper_hashes = {
        "d71_core_sha256": d43._sha256(CORE_PATH),
        "d71_d62_helper_sha256": d43._sha256(D62_HELPER_PATH),
        "d71_d61_helper_sha256": d43._sha256(d62.D61_HELPER_PATH),
        "d71_d46_helper_sha256": d43._sha256(d62.d61.D46_HELPER_PATH),
        "d71_d43_helper_sha256": d43._sha256(
            d62.d61.d46.d44.D43_HELPER_PATH
        ),
    }
    previous_sys_path, previous_argv = list(sys.path), sys.argv
    d42 = package = None
    original_path: tuple[str, ...] = ()
    original_fit = original_macs = original_top = original_score = None
    runner_name, exit_code = "d71_locked_d42_runner", 1
    registry = None
    component_records: list[dict[str, Any]] = []
    try:
        d42, package, original_path = d43._bootstrap(
            known.runtime_root, known.probe_root
        )
        original_fit = d42._fit_equal_prior_lda
        fit, component_records = build_d71_fit(d42)
        d42._fit_equal_prior_lda = fit
        original_macs, original_top = d62._install_resource_accounting(d42)
        d62_top = d42.fit_d42_unified_shrinkage_lda
        original_score = d42.score_d42_unified_shrinkage_lda
        registry = RerankerRegistry(d42, fit, original_score)
        d42.fit_d42_unified_shrinkage_lda = registry.wrap_top(d62_top)
        d42.score_d42_unified_shrinkage_lda = registry.score
        runner = known.probe_root / "code" / "scripts" / "run_d25_support_only_concat.py"
        runner_spec = importlib.util.spec_from_file_location(runner_name, runner)
        if runner_spec is None or runner_spec.loader is None:
            raise D71ProbeError("D71 could not load locked runner")
        runner_module = importlib.util.module_from_spec(runner_spec)
        sys.modules[runner_spec.name] = runner_module
        runner_spec.loader.exec_module(runner_module)
        d43._install_runner_probe_guards(
            runner_module,
            arm=known.d71_arm,
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
        if d42 is not None and original_score is not None:
            d42.score_d42_unified_shrinkage_lda = original_score
        if package is not None:
            package.__path__[:] = list(original_path)
        sys.modules.pop(runner_name, None)
    if exit_code != 0:
        return exit_code
    if (
        registry is None
        or registry.top_fit_count != 30
        or registry.inner_base_fit_count != 120
        or len(registry.records) != 30
        or len(component_records) != 2280
        or registry.score_call_count <= 0
    ):
        raise D71ProbeError("D71 fit/score/component call closure drift")
    evidence = _verify_output(output, script_sha, helper_hashes)
    record_sha = hashlib.sha256(
        json.dumps(
            registry.records, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    metadata = {
        "schema": "cvs.phase2.d71.crossfitted_top2_centroid_reranker_probe.v1",
        "status": "DEVELOPMENT_SUPPORT_ONLY_DIAGNOSTIC_PROBE",
        "arm": known.d71_arm,
        "formal_candidate": False,
        "probe_forced_nonpromotable": True,
        "selected_only_full_k10_refit_allowed": False,
        "query_opened": False,
        "probe_script_sha256": script_sha,
        "formula": FORMULA,
        "top_fit_count": registry.top_fit_count,
        "inner_d62_fit_count": registry.inner_base_fit_count,
        "component_fit_execution_count": len(component_records),
        "score_call_count": registry.score_call_count,
        "reranked_prediction_count": registry.reranked_prediction_count,
        "fit_record_sha256": record_sha,
        "runtime_root": str(known.runtime_root.resolve()),
        "probe_root": str(known.probe_root.resolve()),
        **evidence,
    }
    (output / "D71_PROBE_METADATA.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
