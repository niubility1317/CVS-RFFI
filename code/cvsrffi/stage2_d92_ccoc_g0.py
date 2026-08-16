"""Mechanical G0 receipt validation for the frozen D92 CCOC candidate.

This module only joins support-side receipts emitted by the existing D92-E0D
runner.  It never loads query labels, truth, or scorer output and it does not
choose a method, arm, receiver, seed, scene, or resource limit.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping, Sequence


G0_SCHEMA = "cvs.phase2.d92_ccoc.truth_free_g0_validation.v1"
G0_MARKER = "D92_CCOC_G0_ACTIVE_QUANTUM_RESOURCE_PASS"
G0_OUTER_KEY = "rx_7_7__seed_713106__k_10__new_5"
G0_SCENES = (
    "leo_clear_weak",
    "leo_low_elev_weak",
    "leo_rain_weak",
)
REFERENCE_ARM = "E0_FULL_ONLY"
CANDIDATE_ARM = "E0_FULL_CROSS_CLASS_OFFBLOCK_CONSENSUS"
ALLOWED_ARMS = (REFERENCE_ARM, CANDIDATE_ARM)
BLOCK_NAMES = ("z160", "fft96", "rf32")
WALL_LIMIT_NS = 150_000_000
WALL_RATIO_LIMIT = 1.50
PEAK_DELTA_LIMIT_BYTES = 1024 * 1024
QUERY_DISABLE_FIELDS = (
    "query_access",
    "truth_access",
    "query_fit_access",
    "query_update_access",
    "query_selection_access",
    "query_role_oracle_access",
    "query_class_quota_access",
    "query_global_reassignment",
)


class D92CCOCG0Error(ValueError):
    """Raised when a G0 receipt cannot be joined without guessing."""


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise D92CCOCG0Error(f"{name} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise D92CCOCG0Error(f"{name} must be a finite number") from error
    if not math.isfinite(result):
        raise D92CCOCG0Error(f"{name} must be a finite number")
    return result


def _integer(value: Any, name: str) -> int:
    result = _finite(value, name)
    if result != float(int(result)):
        raise D92CCOCG0Error(f"{name} must be an integer")
    return int(result)


def _first(row: Mapping[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in row:
            return row[name]
    return default


def _scene_rows(value: Any) -> dict[str, Mapping[str, Any]]:
    """Normalize a top-level scene mapping without touching hidden data."""

    if isinstance(value, Mapping):
        scenes = value.get("scenes", value)
        if isinstance(scenes, Mapping):
            rows = {str(key): row for key, row in scenes.items()}
        elif isinstance(scenes, Sequence) and not isinstance(scenes, (str, bytes)):
            rows = {}
            for row in scenes:
                if not isinstance(row, Mapping) or "scene" not in row:
                    raise D92CCOCG0Error("scene receipt row is malformed")
                scene = str(row["scene"])
                if scene in rows:
                    raise D92CCOCG0Error(f"duplicate scene receipt: {scene}")
                rows[scene] = row
        else:
            raise D92CCOCG0Error("scene receipt collection is malformed")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        rows = {}
        for row in value:
            if not isinstance(row, Mapping) or "scene" not in row:
                raise D92CCOCG0Error("scene receipt row is malformed")
            scene = str(row["scene"])
            if scene in rows:
                raise D92CCOCG0Error(f"duplicate scene receipt: {scene}")
            rows[scene] = row
    else:
        raise D92CCOCG0Error("scene receipt collection is malformed")
    if not rows:
        raise D92CCOCG0Error("scene receipt collection is empty")
    for scene, row in rows.items():
        if not isinstance(row, Mapping):
            raise D92CCOCG0Error(f"scene receipt is not a mapping: {scene}")
        if str(row.get("scene", scene)) != scene:
            raise D92CCOCG0Error(f"scene receipt handle drift: {scene}")
    return rows


def _canonical_support_identity(row: Mapping[str, Any]) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
    identity = _first(
        row,
        "canonical_support_identity_sha256",
        "support_identity_sha256",
    )
    if not isinstance(identity, str) or not identity:
        raise D92CCOCG0Error("canonical support identity is missing")
    class_handles = _first(row, "canonical_class_handles", "class_handles")
    support_handles = _first(row, "canonical_support_handles", "support_handles")
    if not isinstance(class_handles, Sequence) or isinstance(class_handles, (str, bytes)):
        raise D92CCOCG0Error("canonical class handles are missing")
    if not isinstance(support_handles, Sequence) or isinstance(support_handles, (str, bytes)):
        raise D92CCOCG0Error("canonical support handles are missing")
    classes = tuple(str(handle) for handle in class_handles)
    supports = tuple(str(handle) for handle in support_handles)
    if not classes or len(set(classes)) != len(classes):
        raise D92CCOCG0Error("canonical class handles are not unique")
    if not supports or len(set(supports)) != len(supports):
        raise D92CCOCG0Error("canonical support handles are not unique")
    return identity, classes, supports


def _margin_map(row: Mapping[str, Any]) -> dict[str, float]:
    values = _first(
        row,
        "cross_group_margin_by_support_handle",
        "margins_by_support_handle",
        "margins",
    )
    result: dict[str, float] = {}
    if isinstance(values, Mapping):
        iterator = ((key, value) for key, value in values.items())
        for key, value in iterator:
            handle = str(key)
            margin = value.get("cross_group_margin") if isinstance(value, Mapping) else value
            if handle in result:
                raise D92CCOCG0Error("duplicate canonical margin handle")
            result[handle] = _finite(margin, f"margin[{handle}]")
    elif isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
        for item in values:
            if not isinstance(item, Mapping):
                raise D92CCOCG0Error("canonical margin row is malformed")
            handle_value = _first(item, "canonical_row_handle", "support_handle")
            if not isinstance(handle_value, str) or not handle_value:
                raise D92CCOCG0Error("canonical margin handle is missing")
            if handle_value in result:
                raise D92CCOCG0Error("duplicate canonical margin handle")
            result[handle_value] = _finite(
                item.get("cross_group_margin"),
                f"margin[{handle_value}]",
            )
    else:
        raise D92CCOCG0Error("canonical margin collection is missing")
    if not result:
        raise D92CCOCG0Error("canonical margin collection is empty")
    return result


def _block_values(row: Mapping[str, Any], *fields: str) -> dict[str, float]:
    values = _first(row, *fields)
    field = fields[0]
    if isinstance(values, Mapping):
        result = {}
        for name in BLOCK_NAMES:
            if name not in values:
                raise D92CCOCG0Error(f"{field} missing block {name}")
            result[name] = abs(_finite(values[name], f"{field}[{name}]"))
        return result
    if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
        if len(values) != len(BLOCK_NAMES):
            raise D92CCOCG0Error(f"{field} must have three blocks")
        return {
            name: abs(_finite(value, f"{field}[{name}]"))
            for name, value in zip(BLOCK_NAMES, values)
        }
    raise D92CCOCG0Error(f"{field} is missing")


def _maximum_cross_group_margin_quantum(
    reference: Mapping[str, Any], candidate: Mapping[str, Any]
) -> float:
    """Return the largest executable D42 scale quantum over the three blocks."""

    ref_amplitude = _block_values(reference, "support_block_absmax", "A_b")
    cand_amplitude = _block_values(candidate, "support_block_absmax", "A_b")
    ref_scale1 = _block_values(reference, "scale1_block_max_abs", "e0_scale1")
    ref_scale2 = _block_values(reference, "scale2_block_max_abs", "e0_scale2")
    cand_scale1 = _block_values(candidate, "scale1_block_max_abs", "ccoc_scale1")
    cand_scale2 = _block_values(candidate, "scale2_block_max_abs", "ccoc_scale2")
    values = []
    for name in BLOCK_NAMES:
        amplitude = max(ref_amplitude[name], cand_amplitude[name])
        scale = max(
            ref_scale1[name],
            ref_scale2[name],
            cand_scale1[name],
            cand_scale2[name],
        )
        values.append(amplitude * scale)
    quantum = max(values)
    if not math.isfinite(quantum) or quantum <= 0.0:
        return 0.0
    return float(quantum)


def _paired_margin_delta_by_handle(
    reference: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, float]:
    ref_identity, ref_classes, ref_supports = _canonical_support_identity(reference)
    cand_identity, cand_classes, cand_supports = _canonical_support_identity(candidate)
    if (
        ref_identity != cand_identity
        or ref_classes != cand_classes
        or ref_supports != cand_supports
    ):
        raise D92CCOCG0Error("canonical support identity mismatch")
    ref_margins = _margin_map(reference)
    cand_margins = _margin_map(candidate)
    if set(ref_margins) != set(cand_margins) or set(ref_margins) != set(ref_supports):
        raise D92CCOCG0Error("canonical support margin handle mismatch")
    return {
        handle: float(cand_margins[handle] - ref_margins[handle])
        for handle in sorted(ref_margins)
    }


def _state_sha(row: Mapping[str, Any]) -> str | None:
    value = _first(
        row,
        "state_fingerprint_sha256",
        "state_sha256",
        "state_sha",
        "after_state_fingerprint_sha256",
        "final_d42_coefficient_bias_state_sha256",
        "final_state_sha256",
        "d92_e0d_ccoc_final_state_sha256",
    )
    return str(value) if isinstance(value, str) and value else None


def _resource_value(row: Mapping[str, Any], *names: str) -> float | None:
    resource = row.get("after_registration_resource")
    if not isinstance(resource, Mapping):
        resource = row.get("resource_audit")
    for name in names:
        if name in row:
            return _finite(row[name], name)
        if isinstance(resource, Mapping) and name in resource:
            return _finite(resource[name], name)
    return None


def _merge_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Expose only receipt/audit scalars needed by the frozen gates."""

    result = dict(row)
    for nested_name in ("receipt", "audit", "fit_audit"):
        nested = row.get(nested_name)
        if isinstance(nested, Mapping):
            for key, value in nested.items():
                result.setdefault(str(key), value)
    resource = row.get("after_registration_resource")
    if isinstance(resource, Mapping):
        for key, value in resource.items():
            result.setdefault(str(key), value)
    inventory = result.get("after_actual_component_inventory")
    if isinstance(inventory, Mapping):
        result.setdefault(
            "actual_full_fit_count",
            inventory.get("actual_component_fit_count"),
        )
    result.setdefault(
        "actual_full_fit_count",
        _first(
            result,
            "d92_e0d_ccoc_actual_full_fit_count",
            "actual_candidate_full_fit_count",
            "actual_candidate_full_fit",
            "d92_e0d_actual_component_fit_count",
        ),
    )
    result.setdefault(
        "persistent_state_bytes",
        _first(result, "after_state_bytes", "state_bytes"),
    )
    result.setdefault("state_fingerprint_sha256", _state_sha(result))
    if "query_macs" not in result:
        result["query_macs"] = _first(result, "d92_e0d_query_macs")
    if "active" not in result:
        result["active"] = _first(
            result,
            "d92_e0d_ccoc_active",
            "d92_ccoc_active",
            default=True,
        )
    if "fallback_active" not in result:
        result["fallback_active"] = _first(
            result,
            "d92_e0d_ccoc_fallback_active",
            "d92_ccoc_fallback_active",
            default=False,
        )
    return result


def _flag(row: Mapping[str, Any], *names: str) -> bool:
    value = _first(row, *names, default=False)
    return value is True


def _query_disabled(row: Mapping[str, Any], field: str) -> bool:
    return not _flag(
        row,
        field,
        f"d92_e0d_{field}",
        f"d92_ccoc_{field}",
    )


def _ccoc_g0_gates(
    reference: Mapping[str, Any],
    candidate: Mapping[str, Any],
    margins: Mapping[str, float],
    quantum: float,
) -> dict[str, bool]:
    reference = _merge_row(reference)
    candidate = _merge_row(candidate)
    reference_state = _state_sha(reference)
    candidate_state = _state_sha(candidate)
    ref_wall = _resource_value(reference, "registration_wall_time_ns")
    cand_wall = _resource_value(candidate, "registration_wall_time_ns")
    cand_peak = _resource_value(
        candidate,
        "registration_incremental_peak_working_set_bytes",
        "peak_delta_bytes",
    )
    rho_values = (
        _first(candidate, "old_rho", "d92_e0d_ccoc_old_rho"),
        _first(candidate, "new_rho", "d92_e0d_ccoc_new_rho"),
    )
    rho_finite = []
    for value in rho_values:
        try:
            rho_finite.append(_finite(value, "rho"))
        except D92CCOCG0Error:
            rho_finite.append(float("nan"))
    reference_query_gates = {
        f"reference_{field}": _query_disabled(reference, field)
        for field in QUERY_DISABLE_FIELDS
    }
    candidate_query_gates = {
        f"candidate_{field}": _query_disabled(candidate, field)
        for field in QUERY_DISABLE_FIELDS
    }
    query_gates = {**reference_query_gates, **candidate_query_gates}
    legacy_candidate_query_gates = {
        field: candidate_query_gates[f"candidate_{field}"]
        for field in QUERY_DISABLE_FIELDS
    }
    margin_delta = max((abs(float(value)) for value in margins.values()), default=0.0)
    candidate_fit = candidate.get("actual_full_fit_count")
    candidate_fit_pass = False
    if candidate_fit is not None:
        try:
            candidate_fit_pass = _integer(candidate_fit, "actual_full_fit_count") == 1
        except D92CCOCG0Error:
            candidate_fit_pass = False
    wall_pass = (
        cand_wall is not None
        and cand_wall <= float(WALL_LIMIT_NS)
    )
    ratio_pass = (
        ref_wall is not None
        and cand_wall is not None
        and ref_wall > 0.0
        and cand_wall / ref_wall <= WALL_RATIO_LIMIT
    )
    peak_pass = (
        cand_peak is not None
        and cand_peak <= float(PEAK_DELTA_LIMIT_BYTES)
    )
    state_bytes_pass = (
        reference.get("persistent_state_bytes") is not None
        and candidate.get("persistent_state_bytes") is not None
        and reference.get("persistent_state_bytes")
        == candidate.get("persistent_state_bytes")
    )
    query_macs_pass = (
        reference.get("query_macs") is not None
        and candidate.get("query_macs") is not None
        and reference.get("query_macs") == candidate.get("query_macs")
    )
    gates: dict[str, bool] = {
        "active": candidate.get("active") is True,
        "fallback_active": candidate.get("fallback_active") is False,
        "old_rho": math.isfinite(rho_finite[0]) and 0.0 <= rho_finite[0] <= 1.0,
        "new_rho": math.isfinite(rho_finite[1]) and 0.0 <= rho_finite[1] <= 1.0,
        "rho_interior": any(0.0 < value < 1.0 for value in rho_finite),
        "state": (
            reference_state is not None
            and candidate_state is not None
            and reference_state != candidate_state
        ),
        "actual_full_fit_count": candidate_fit_pass,
        "query": all(query_gates.values()),
        "wall": wall_pass,
        "registration_wall_time_ns": wall_pass,
        "ratio": ratio_pass,
        "peak": peak_pass,
        "registration_incremental_peak_working_set_bytes": peak_pass,
        "state_bytes": state_bytes_pass,
        "query_macs": query_macs_pass,
        "quantum": quantum > 0.0 and margin_delta >= quantum,
    }
    gates.update(query_gates)
    gates.update(legacy_candidate_query_gates)
    return gates


def _nearest_rank_p90(values: Sequence[float | None]) -> float | None:
    if not values or any(value is None for value in values):
        return None
    ordered = sorted(float(value) for value in values if value is not None)
    rank = max(0, math.ceil(0.90 * len(ordered)) - 1)
    return float(ordered[rank])


def validate_ccoc_g0(
    reference_rows: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    candidate_rows: Mapping[str, Any] | Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate paired support receipts without reading query-side evidence."""

    reference = _scene_rows(reference_rows)
    candidate = _scene_rows(candidate_rows)
    expected_scenes = set(G0_SCENES)
    if set(reference) != expected_scenes or set(candidate) != expected_scenes:
        raise D92CCOCG0Error("G0 scene set mismatch")
    scene_results: dict[str, Any] = {}
    all_pass = True
    all_deltas: list[float] = []
    all_quanta: list[float] = []
    candidate_walls: list[float | None] = []
    candidate_reference_ratios: list[float | None] = []
    candidate_wall_by_scene: dict[str, float | None] = {}
    candidate_reference_ratio_by_scene: dict[str, float | None] = {}
    for scene in sorted(reference):
        ref_row = reference[scene]
        cand_row = candidate[scene]
        margins = _paired_margin_delta_by_handle(ref_row, cand_row)
        quantum = _maximum_cross_group_margin_quantum(ref_row, cand_row)
        gates = _ccoc_g0_gates(ref_row, cand_row, margins, quantum)
        delta = max((abs(value) for value in margins.values()), default=0.0)
        scene_pass = all(gates.values())
        scene_results[scene] = {
            "max_cross_group_margin_change_abs": float(delta),
            "cross_group_margin_quantum": float(quantum),
            "gates": gates,
            "pass": scene_pass,
        }
        merged_reference = _merge_row(ref_row)
        merged_candidate = _merge_row(cand_row)
        candidate_wall = _resource_value(
            merged_candidate,
            "registration_wall_time_ns",
        )
        reference_wall = _resource_value(
            merged_reference,
            "registration_wall_time_ns",
        )
        candidate_ratio = (
            candidate_wall / reference_wall
            if candidate_wall is not None
            and reference_wall is not None
            and reference_wall > 0.0
            else None
        )
        candidate_walls.append(candidate_wall)
        candidate_reference_ratios.append(candidate_ratio)
        candidate_wall_by_scene[scene] = candidate_wall
        candidate_reference_ratio_by_scene[scene] = candidate_ratio
        all_pass = all_pass and scene_pass
        all_deltas.append(float(delta))
        all_quanta.append(float(quantum))
    aggregate_gates: dict[str, bool] = {}
    for details in scene_results.values():
        for name, state in details["gates"].items():
            aggregate_gates[name] = aggregate_gates.get(name, True) and bool(state)
    result: dict[str, Any] = {
        "schema": G0_SCHEMA,
        "scenes": scene_results,
        "max_cross_group_margin_change_abs": max(all_deltas),
        "cross_group_margin_quantum": max(all_quanta),
        "candidate_wall_p90_ns": _nearest_rank_p90(candidate_walls),
        "candidate_reference_ratio_p90": _nearest_rank_p90(
            candidate_reference_ratios
        ),
        "candidate_reference_wall_ratio_p90": _nearest_rank_p90(
            candidate_reference_ratios
        ),
        "candidate_wall_time_ns_by_scene": candidate_wall_by_scene,
        "candidate_reference_ratio_by_scene": candidate_reference_ratio_by_scene,
        "gates": aggregate_gates,
        "scene_gates": {
            scene: bool(details["pass"])
            for scene, details in scene_results.items()
        },
        "pass": bool(all_pass),
    }
    if all_pass:
        result["marker"] = G0_MARKER
    return result


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def receipt_sha256(value: Mapping[str, Any]) -> str:
    """Return the deterministic digest used by the immutable G0 receipt."""

    return hashlib.sha256(_canonical_json(value)).hexdigest()


__all__ = [
    "ALLOWED_ARMS",
    "BLOCK_NAMES",
    "CANDIDATE_ARM",
    "D92CCOCG0Error",
    "G0_MARKER",
    "G0_OUTER_KEY",
    "G0_SCHEMA",
    "G0_SCENES",
    "REFERENCE_ARM",
    "_maximum_cross_group_margin_quantum",
    "_paired_margin_delta_by_handle",
    "receipt_sha256",
    "validate_ccoc_g0",
]
