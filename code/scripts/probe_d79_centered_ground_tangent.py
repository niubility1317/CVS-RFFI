#!/usr/bin/env python3
"""D79 support-centered ground-tangent diagnostic probe."""

from __future__ import annotations

import copy
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import sys
import types
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
D78_HELPER_PATH = SCRIPT_DIR / "probe_d78_ground_tangent_worstclass_margin.py"
CORE_PATH = SCRIPT_DIR.parent / "cvsrffi" / "stage2_d79_centered_ground_tangent.py"


def _load(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"D79 could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


d78 = _load("d79_d78_probe_scaffold", D78_HELPER_PATH)
core = _load("d79_centered_ground_tangent_core", CORE_PATH)
d43, d62 = d78.d43, d78.d62

ARM = "centered_ground_tangent_worstclass_top2_margin"
STRUCTURE = d62.STRUCTURE
FORMULA = (
    "freeze the D78 immutable-ground rank-13 tangent and smooth-worst top-2 "
    "optimizer; subtract the all-registered-class equal-K target-support mean "
    "before tangent projection; learn the same 20-step class-symmetric residual; "
    "compile delta_W into D62 final rows and delta_b=-delta_W*support_mean into "
    "the intercept so residual logits are exactly zero at the target-domain center"
)
d43.ARM_STRUCTURES[ARM] = STRUCTURE
if ARM not in d43.ARMS:
    d43.ARMS = tuple((*d43.ARMS, ARM))


class D79ProbeError(RuntimeError):
    """Raised when the D79 integration closure drifts."""


facade = types.ModuleType("d79_centered_core_facade")
facade.FW_ITERATIONS = core.FW_ITERATIONS
facade.OPTIMIZER_STEPS = core.OPTIMIZER_STEPS
facade.ground_domain_tangent_basis = core.ground_domain_tangent_basis
facade._bias_residual = None


def _set_bias_residual(value: np.ndarray) -> None:
    array = np.ascontiguousarray(value, dtype=np.float32)
    if array.ndim != 1 or not np.isfinite(array).all():
        raise D79ProbeError("D79 bias residual drift")
    facade._bias_residual = array


def _consume_bias_residual(class_count: int) -> np.ndarray:
    value = facade._bias_residual
    facade._bias_residual = None
    if value is None or value.shape != (int(class_count),):
        raise D79ProbeError("D79 bias residual compile handoff drift")
    return value


facade.set_bias_residual = _set_bias_residual
facade.fit_ground_preconditioned_common_descent = (
    core.fit_ground_preconditioned_common_descent
)
sys.modules[facade.__name__] = facade


class CenteredGroundTangentRegistry(d78.GroundTangentRegistry):
    def wrap_top(self, base_top: Any) -> Any:
        inherited = super().wrap_top(base_top)

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            original_compile = d78.d77._compile_pair

            def compile_with_centered_bias(
                d42: Any,
                template: Any,
                coefficient: np.ndarray,
                intercept: np.ndarray,
            ) -> tuple[Any, Any, dict[str, float]]:
                bias = _consume_bias_residual(len(intercept))
                return original_compile(
                    d42,
                    template,
                    coefficient,
                    np.asarray(intercept, dtype=np.float32) + bias,
                )

            if facade._bias_residual is not None:
                raise D79ProbeError("D79 stale bias residual before fit")
            d78.d77._compile_pair = compile_with_centered_bias
            try:
                result = inherited(*args, **kwargs)
            finally:
                d78.d77._compile_pair = original_compile
            if facade._bias_residual is not None:
                raise D79ProbeError("D79 bias residual was not compiled")
            geometry = dict(result.geometry_audit)
            for key in list(geometry):
                if key.startswith("d78_"):
                    geometry["d79_" + key[4:]] = geometry.pop(key)
            geometry["stage2c_classifier"] = (
                "d79_centered_ground_tangent_compiled_affine"
                if geometry["d79_worstclass_margin_audit"]["residual_active"]
                else "d79_centered_exact_d62_fallback"
            )
            trace = [dict(item) for item in result.training_trace]
            for item in trace:
                if item.get("phase") == "stage2c_ground_tangent_worstclass_top2_margin":
                    item["phase"] = "stage2c_centered_ground_tangent_worstclass_top2_margin"
            resource = dict(result.resource_audit)
            for key in list(resource):
                if key.startswith("d78_"):
                    resource["d79_" + key[4:]] = resource.pop(key)
            classes = int(geometry["d79_worstclass_margin_audit"]["class_count"])
            dimension = int(geometry["d79_worstclass_margin_audit"]["dimension"])
            bias_compile_macs = classes * dimension
            resource["d79_bias_compile_mac_equivalents"] = bias_compile_macs
            resource["d79_non_lda_added_adaptation_macs"] += bias_compile_macs
            resource["d79_total_added_adaptation_macs"] += bias_compile_macs
            resource["estimated_adaptation_macs"] += bias_compile_macs
            resource["estimated_metric_adaptation_macs"] += bias_compile_macs
            resource["complete_loss_trace"] = trace
            return replace(
                result,
                training_trace=tuple(trace),
                geometry_audit=geometry,
                resource_audit=resource,
            )

        return wrapped


def _install_runner_resource_accounting(runner: Any) -> None:
    original_evaluate = runner._evaluate_d42_fold

    def evaluate(*args: Any, **kwargs: Any) -> dict[str, Any]:
        row = original_evaluate(*args, **kwargs)
        resource = dict(row["resource"])
        if "d79_ground_component_logical_state_bytes" not in resource:
            return row
        head_bytes = int(resource["persistent_state_bytes"])
        component_bytes = int(resource["d79_ground_component_logical_state_bytes"])
        total_bytes = head_bytes + component_bytes
        resource.update(
            {
                "d79_compiled_affine_state_bytes": head_bytes,
                "d79_component_inclusive_persistent_state_bytes": total_bytes,
                "persistent_state_bytes": total_bytes,
                "persistent_state_cap_pass": total_bytes
                <= int(resource["persistent_state_cap_bytes"]),
            }
        )
        return {**row, "resource": resource}

    runner._evaluate_d42_fold = evaluate


def _translate_rows_to_d78(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    translated = copy.deepcopy(rows)
    for row in translated:
        if row.get("candidate_id") not in (
            "D42-USLDA-INT8",
            "D42-USLDA-FP32-MATCHED",
        ):
            continue
        geometry = row["geometry_summary"]
        for key in list(geometry):
            if key.startswith("d79_"):
                geometry["d78_" + key[4:]] = geometry.pop(key)
        audit = geometry["d78_worstclass_margin_audit"]
        if audit["status"] == "centered_ground_tangent_worstclass_top2_margin_active":
            audit["status"] = "ground_tangent_worstclass_top2_margin_active"
        elif audit["status"] == "centered_zero_projected_gradient_exact_d62_fallback":
            audit["status"] = "zero_projected_gradient_exact_d62_fallback"
        resource = row["resource"]
        for key in list(resource):
            if key.startswith("d79_"):
                resource["d78_" + key[4:]] = resource.pop(key)
        bias_macs = int(resource.pop("d78_bias_compile_mac_equivalents"))
        resource["d78_non_lda_added_adaptation_macs"] -= bias_macs
        resource["d78_total_added_adaptation_macs"] -= bias_macs
        resource["estimated_adaptation_macs"] -= bias_macs
        resource["estimated_metric_adaptation_macs"] -= bias_macs
        for item in row["training_trace"]:
            if item.get("phase") == (
                "stage2c_centered_ground_tangent_worstclass_top2_margin"
            ):
                item["phase"] = "stage2c_ground_tangent_worstclass_top2_margin"
    return translated


_original_d78_verify_rows = d78._verify_rows


def _verify_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    evidence = _original_d78_verify_rows(_translate_rows_to_d78(rows))
    target = [
        row
        for row in rows
        if row.get("candidate_id")
        in ("D42-USLDA-INT8", "D42-USLDA-FP32-MATCHED")
    ]
    center_errors: list[float] = []
    bias_norms: list[float] = []
    for row in target:
        geometry, resource = row["geometry_summary"], row["resource"]
        audit = geometry["d79_worstclass_margin_audit"]
        if (
            geometry["d79_probe_arm"] != ARM
            or geometry["d79_formula"] != FORMULA
            or audit["support_centering_enabled"] is not True
            or audit["centered_affine_compile"] is not True
            or float(audit["residual_logit_at_support_center_max_abs"]) > 1e-10
            or int(resource["d79_bias_compile_mac_equivalents"]) != 3168
            or resource["d79_single_affine_state_only"] is not True
            or resource["d79_ground_class_score_access"] is not False
            or resource["d79_ground_component_update_access"] is not False
        ):
            raise D79ProbeError("D79 centered compile closure drift")
        center_errors.append(float(audit["centered_support_mean_max_abs"]))
        bias_norms.append(float(audit["bias_residual_frobenius"]))
    renamed = {
        key.replace("verified_d78_", "verified_d79_"): value
        for key, value in evidence.items()
    }
    return {
        **renamed,
        "verified_d79_centered_compile_count": len(target),
        "verified_d79_center_error_max": max(center_errors),
        "verified_d79_bias_residual_norm_min": min(bias_norms),
        "verified_d79_bias_residual_norm_max": max(bias_norms),
    }


def _verify_output(
    output: Path, script_sha: str, helper_hashes: dict[str, str]
) -> dict[str, Any]:
    evidence = d43._verify_probe_output(output, ARM, script_sha)
    support = d43._read_json(output / "support_audit.json")
    closure = support.get("candidate_lock", {}).get("source_closure", {})
    if any(closure.get(name) != value for name, value in helper_hashes.items()):
        raise D79ProbeError("D79 helper source closure drift")
    rows = [
        json.loads(line)
        for line in (output / "training_log.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    return {**evidence, **_verify_rows(rows)}


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    try:
        arm_index = arguments.index("--d79-arm")
        output_index = arguments.index("--output")
    except ValueError as exc:
        raise D79ProbeError("D79 locked arm/output argument missing") from exc
    if arguments[arm_index + 1] != ARM:
        raise D79ProbeError("D79 arm drift")
    output = Path(arguments[output_index + 1]).resolve()
    arguments[arm_index] = "--d78-arm"

    d78.ARM = ARM
    d78.STRUCTURE = STRUCTURE
    d78.FORMULA = FORMULA
    d78.core = facade
    d78.d77.ARM = ARM
    d78.d77.FORMULA = FORMULA
    d78.d77.core = facade
    d78.GroundTangentRegistry = CenteredGroundTangentRegistry
    d78._install_runner_resource_accounting = _install_runner_resource_accounting
    d78._verify_output = _verify_output
    d78.CORE_PATH = CORE_PATH
    d78.D77_HELPER_PATH = D78_HELPER_PATH
    d78.__file__ = str(Path(__file__).resolve())
    exit_code = int(d78.main(arguments))
    if exit_code != 0:
        return exit_code
    old_metadata = output / "D78_PROBE_METADATA.json"
    new_metadata = output / "D79_PROBE_METADATA.json"
    metadata = json.loads(old_metadata.read_text(encoding="utf-8"))
    metadata["schema"] = "cvs.phase2.d79.centered_ground_tangent_probe.v1"
    metadata["d78_scaffold_compatibility_only"] = True
    metadata["centered_ground_tangent"] = True
    new_metadata.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    old_metadata.unlink()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
