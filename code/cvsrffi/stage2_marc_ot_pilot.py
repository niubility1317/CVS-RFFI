"""Frozen R-matrix and support-before-query lifecycle for the MARC-OT pilot."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import math
from typing import Any

from .stage2_wiser_pilot import (
    WISERQueryPackage,
    WISERSupportPackage,
    load_query_package,
    load_support_package,
)


SCENARIOS = ("leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak")
FORMAL_ARMS = ("R0", "R1", "R2", "R4", "R6", "R8")
MRIOR_CONTROLS = ("MRIOR-H", "MRIOR-B", "MRIOR-HB")
FORMAL_ARM_SPECS: Mapping[str, Mapping[str, bool]] = {
    "R0": {
        "support_residual": False,
        "cross_fit": False,
        "bank_initialization": False,
        "support_bank_ot": False,
        "blockwise_projection": False,
    },
    "R1": {
        "support_residual": True,
        "cross_fit": False,
        "bank_initialization": False,
        "support_bank_ot": False,
        "blockwise_projection": False,
    },
    "R2": {
        "support_residual": True,
        "cross_fit": True,
        "bank_initialization": False,
        "support_bank_ot": False,
        "blockwise_projection": False,
    },
    "R4": {
        "support_residual": True,
        "cross_fit": True,
        "bank_initialization": True,
        "support_bank_ot": False,
        "blockwise_projection": False,
    },
    "R6": {
        "support_residual": True,
        "cross_fit": True,
        "bank_initialization": True,
        "support_bank_ot": True,
        "blockwise_projection": False,
    },
    "R8": {
        "support_residual": True,
        "cross_fit": True,
        "bank_initialization": True,
        "support_bank_ot": True,
        "blockwise_projection": True,
    },
}


_CONFIG_SCHEMA = "cvs.phase2.marc_ot.pilot_config.v1"
_CONFIG_KEYS = frozenset(
    {
        "schema",
        "protocol_schema",
        "phase2_data_status",
        "capsule_id",
        "split_id",
        "pilot_outer_key",
        "checkpoint_id",
        "receiver",
        "seed",
        "k_shot",
        "scenarios",
        "arms",
        "fold_count",
        "stage_steps",
        "learning_rate_bounds",
        "ot",
        "ratio_cap",
        "interpolation_grid",
        "promotion_gates",
        "mrior_controls",
    }
)
_PROMOTION_GATE_KEYS = frozenset(
    {
        "median_p3_ba_delta_pp",
        "worst_scene_p3_ba_delta_pp",
        "median_p3_floor_delta_pp",
        "low_elev_p3_floor_delta_pp",
        "max_p1_p2_scene_drop_pp",
        "minimum_help_gt_harm_scenes",
    }
)


def normalize_formal_arms(values: Sequence[str]) -> tuple[str, ...]:
    selected = tuple(str(value).upper() for value in values)
    if selected != FORMAL_ARMS:
        raise ValueError("formal MARC-OT arms must be exactly R0/R1/R2/R4/R6/R8")
    return selected


def _walk_members(value: Any, prefix: str = ""):
    if isinstance(value, Mapping):
        for key, item in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            yield name, item
            yield from _walk_members(item, name)
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from _walk_members(item, f"{prefix}[{index}]")


def validate_mrior_controls(value: Any) -> Mapping[str, Mapping[str, str]]:
    """Validate mechanism controls without importing historical MRIOR numbers."""

    if not isinstance(value, Mapping) or set(value) != set(MRIOR_CONTROLS):
        raise ValueError("MRIOR control registry must contain exactly MRIOR-H/B/HB")
    for path, item in _walk_members(value):
        lowered = path.lower().replace("-", "_")
        if "mrior_sda" in lowered:
            raise ValueError("MRIOR-SDA historical results cannot backfill H/B/HB controls")
        if "histor" in lowered and isinstance(item, (int, float)) and not isinstance(item, bool):
            raise ValueError("MRIOR controls cannot contain historical numerical fields")
    result: dict[str, Mapping[str, str]] = {}
    for name in MRIOR_CONTROLS:
        row = value[name]
        if not isinstance(row, Mapping):
            raise ValueError(f"{name} control must be an object")
        if "permission_scope" not in row:
            raise ValueError(f"{name} permission_scope is required")
        if set(row) != {"permission_scope", "claim_scope"}:
            if any(
                isinstance(item, (int, float)) and not isinstance(item, bool)
                for item in row.values()
            ):
                raise ValueError("MRIOR controls cannot contain historical numerical fields")
            raise ValueError(f"{name} control fields must be permission_scope and claim_scope")
        permission = row["permission_scope"]
        claim = row["claim_scope"]
        if not isinstance(permission, str) or not permission.strip():
            raise ValueError(f"{name} permission_scope must be a nonempty string")
        if not isinstance(claim, str) or not claim.strip():
            raise ValueError(f"{name} claim_scope must be a nonempty string")
        result[name] = {
            "permission_scope": permission,
            "claim_scope": claim,
        }
    return result


def _strict_int(value: Any, *, field: str, minimum: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"{field} must be an integer >= {minimum}")
    return value


def _finite_number(value: Any, *, field: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def validate_pilot_config(value: Any) -> Mapping[str, Any]:
    """Validate only frozen method/data handles; never revalidate Phase2 data."""

    if not isinstance(value, Mapping) or set(value) != _CONFIG_KEYS:
        raise ValueError("MARC-OT pilot config field set drift")
    if value.get("schema") != _CONFIG_SCHEMA:
        raise ValueError("MARC-OT pilot config schema drift")
    if value.get("protocol_schema") != "p2_min_v1":
        raise ValueError("MARC-OT pilot requires p2_min_v1")
    if value.get("phase2_data_status") != "VALIDATED_ONCE":
        raise ValueError("MARC-OT pilot requires VALIDATED_ONCE data")
    for field in ("capsule_id", "split_id", "pilot_outer_key", "checkpoint_id", "receiver"):
        if not isinstance(value.get(field), str) or not str(value[field]).strip():
            raise ValueError(f"MARC-OT {field} binding is missing")
    _strict_int(value.get("seed"), field="seed", minimum=0)
    if _strict_int(value.get("k_shot"), field="k_shot", minimum=1) != 10:
        raise ValueError("first MARC-OT pilot is frozen at K10")
    if tuple(value.get("scenarios", ())) != SCENARIOS:
        raise ValueError("MARC-OT scenario registry must be the exact three LEO weak scenes")
    normalize_formal_arms(value.get("arms", ()))
    fold_count = _strict_int(value.get("fold_count"), field="fold_count", minimum=2)
    if fold_count > 10:
        raise ValueError("fold_count cannot exceed frozen K10")
    steps = value.get("stage_steps")
    if not isinstance(steps, list) or len(steps) != 4:
        raise ValueError("stage_steps must freeze four progressive stages")
    for index, step in enumerate(steps):
        _strict_int(step, field=f"stage_steps[{index}]", minimum=0)
    bounds = value.get("learning_rate_bounds")
    if not isinstance(bounds, Mapping) or set(bounds) != {"min", "max"}:
        raise ValueError("learning_rate_bounds field set drift")
    lr_min = _finite_number(bounds["min"], field="learning_rate_bounds.min")
    lr_max = _finite_number(bounds["max"], field="learning_rate_bounds.max")
    if lr_min <= 0.0 or lr_min >= lr_max:
        raise ValueError("learning_rate_bounds must be positive and ordered")
    ot = value.get("ot")
    if not isinstance(ot, Mapping) or set(ot) != {"epsilon", "iterations"}:
        raise ValueError("OT config field set drift")
    if _finite_number(ot["epsilon"], field="ot.epsilon") <= 0.0:
        raise ValueError("ot.epsilon must be positive")
    _strict_int(ot["iterations"], field="ot.iterations", minimum=1)
    if _finite_number(value.get("ratio_cap"), field="ratio_cap") < 0.0:
        raise ValueError("ratio_cap must be nonnegative")
    grid = value.get("interpolation_grid")
    if not isinstance(grid, list) or not grid:
        raise ValueError("interpolation_grid must be a nonempty list")
    alphas = tuple(_finite_number(alpha, field="interpolation_grid") for alpha in grid)
    if any(not 0.0 <= alpha <= 1.0 for alpha in alphas) or 0.0 not in alphas:
        raise ValueError("interpolation_grid must contain alpha=0 and stay in [0,1]")
    gates = value.get("promotion_gates")
    if not isinstance(gates, Mapping) or set(gates) != _PROMOTION_GATE_KEYS:
        raise ValueError("promotion_gates field set drift")
    for field, gate in gates.items():
        if field == "minimum_help_gt_harm_scenes":
            count = _strict_int(gate, field=field, minimum=0)
            if count > len(SCENARIOS):
                raise ValueError("minimum_help_gt_harm_scenes exceeds scene count")
        else:
            _finite_number(gate, field=field)
    validate_mrior_controls(value.get("mrior_controls"))
    return dict(value)


def validate_manifest_job(
    manifest: Mapping[str, Any], *, outer_key: str, config: Mapping[str, Any]
) -> Mapping[str, Any]:
    """Read back matching one-time-validation handles without rebuilding data."""

    if manifest.get("protocol_schema") != "p2_min_v1":
        raise ValueError("MARC-OT manifest requires p2_min_v1")
    rows = [row for row in manifest.get("jobs", ()) if row.get("outer_key") == outer_key]
    if len(rows) != 1:
        raise ValueError("MARC-OT manifest outer-key coverage drift")
    row = rows[0]
    expected = {
        "protocol_schema": "p2_min_v1",
        "phase2_data_status": "VALIDATED_ONCE",
        "capsule_id": config["capsule_id"],
        "split_id": config["split_id"],
    }
    for field, expected_value in expected.items():
        observed = row.get(field, manifest.get(field))
        if observed != expected_value:
            raise ValueError(f"MARC-OT manifest/config {field} binding drift")
    return row


def run_support_then_query(
    *,
    scenarios: Sequence[str],
    arms: Sequence[str],
    support_loader: Callable[[str], Any],
    adapt_and_freeze: Callable[[str, str, Any], Any],
    support_state_writer: Callable[[str, str, Any], None],
    query_loader: Callable[[str], Any],
    predict_and_write: Callable[[str, str, Any, Any, Any], None],
) -> Mapping[str, Any]:
    """Freeze and record every support state before the first query load."""

    if tuple(scenarios) != SCENARIOS:
        raise ValueError("pilot scenarios must be the exact frozen registry")
    selected_arms = normalize_formal_arms(arms)
    support_by_scene: dict[str, Any] = {}
    states: dict[tuple[str, str], Any] = {}
    support_units: list[dict[str, Any]] = []
    for scenario in SCENARIOS:
        support = support_loader(scenario)
        support_by_scene[scenario] = support
        for arm in selected_arms:
            state = adapt_and_freeze(scenario, arm, support)
            support_state_writer(scenario, arm, state)
            states[(scenario, arm)] = state
            support_units.append(
                {
                    "scenario": scenario,
                    "arm": arm,
                    "status": "SUPPORT_STATE_FROZEN",
                    "query_opened": False,
                }
            )
    prediction_units: list[dict[str, Any]] = []
    for scenario in SCENARIOS:
        query = query_loader(scenario)
        for arm in selected_arms:
            predict_and_write(
                scenario,
                arm,
                support_by_scene[scenario],
                query,
                states[(scenario, arm)],
            )
            prediction_units.append(
                {"scenario": scenario, "arm": arm, "status": "PREDICTIONS_COMPLETE"}
            )
    return {
        "schema": "cvs.phase2.marc_ot.pilot_lifecycle.v1",
        "status": "ARTIFACTS_COMPLETE",
        "arms": list(selected_arms),
        "scenarios": list(SCENARIOS),
        "support_frozen_unit_count": len(support_units),
        "prediction_unit_count": len(prediction_units),
        "support_units": support_units,
        "prediction_units": prediction_units,
        "truth_opened": False,
        "query_policy": "per_sample_all_registered_classes_after_all_support_freeze",
    }


__all__ = [
    "FORMAL_ARMS",
    "FORMAL_ARM_SPECS",
    "MRIOR_CONTROLS",
    "SCENARIOS",
    "WISERQueryPackage",
    "WISERSupportPackage",
    "load_query_package",
    "load_support_package",
    "normalize_formal_arms",
    "run_support_then_query",
    "validate_manifest_job",
    "validate_mrior_controls",
    "validate_pilot_config",
]
