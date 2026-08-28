"""Strict plan resolver for D3 plus ERBT-IDR Stage2-C runs."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Mapping

from cvsrffi.target_only_progressive_adapt import SFTAPFTConfig


FORMAL_SCENES = (
    "leo_clear_weak",
    "leo_low_elev_weak",
    "leo_rain_weak",
)
_TOP_KEYS = frozenset(
    {
        "schema",
        "run_id",
        "capsule_id",
        "base_checkpoint_path",
        "phase1_bundle",
        "d3_config",
        "scenes",
    }
)
_SCENE_KEYS = frozenset(
    {
        "gpu",
        "split_id",
        "old_support_input",
        "registered_support_input",
        "query_input",
        "old_support_output",
        "registered_support_output",
        "query_output",
    }
)


def _checked(plan: Mapping[str, Any], scenario: str) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(plan, Mapping) or frozenset(plan) != _TOP_KEYS:
        raise ValueError("D3 ERBT plan top-level allowlist mismatch")
    value = dict(plan)
    if value["schema"] != "cvs.sf_d3_erbt.plan.v1":
        raise ValueError("D3 ERBT plan schema mismatch")
    if scenario not in FORMAL_SCENES or scenario not in value["scenes"]:
        raise ValueError("D3 ERBT scenario is not registered")
    scene = value["scenes"][scenario]
    if not isinstance(scene, Mapping) or frozenset(scene) != _SCENE_KEYS:
        raise ValueError("D3 ERBT scene allowlist mismatch")
    scene_value = dict(scene)
    if any(
        not isinstance(value.get(name), str) or not value[name].strip()
        for name in ("run_id", "capsule_id", "base_checkpoint_path")
    ):
        raise ValueError("D3 ERBT plan identity is empty")
    if not isinstance(value["phase1_bundle"], Mapping) or not isinstance(
        value["d3_config"], Mapping
    ):
        raise ValueError("D3 ERBT plan config mapping drift")
    if isinstance(scene_value["gpu"], bool) or scene_value["gpu"] not in range(8):
        raise ValueError("D3 ERBT scene GPU drift")
    if any(
        not isinstance(scene_value[name], str) or not scene_value[name].strip()
        for name in _SCENE_KEYS - {"gpu"}
    ):
        raise ValueError("D3 ERBT scene path or split is empty")
    return value, scene_value


def build_d3_config(plan: Mapping[str, Any], scenario: str) -> dict[str, Any]:
    value, scene = _checked(plan, scenario)
    raw = dict(value["d3_config"])
    # These fields define the selected D3/R1-T working point.  They are locked
    # here rather than inherited from launcher defaults so a compact plan cannot
    # silently fall back to the slower three-stage or cached variants.
    raw.update(
        phase_steps=(327, 0, 0),
        norm_rules=(),
        inference_temperature=1.0,
        oof_temperature_calibration=True,
        cache_storage_dtype="off",
        suffix_compute_dtype="off",
        fast_tail_start_step=0,
        fast_tail_steps=0,
        head_polish_steps=0,
    )
    config = SFTAPFTConfig(**raw)
    if (
        config.phase_steps != (327, 0, 0)
        or not config.oof_temperature_calibration
        or config.inference_temperature != 1.0
        or config.cache_storage_dtype != "off"
        or config.suffix_compute_dtype != "off"
    ):
        raise ValueError("D3 R1-T method lock drift")
    sf_tapft = asdict(config)
    sf_tapft["phase_steps"] = tuple(config.phase_steps)
    return {
        "method": "sf_tapft_v1",
        "permission": "DIAGNOSTIC_NON_FORMAL",
        "protocol_schema": "p2_min_v1",
        "phase2_data_status": "VALIDATED_ONCE",
        "capsule_id": value["capsule_id"],
        "split_id": scene["split_id"],
        "checkpoint_path": value["base_checkpoint_path"],
        "support_path": scene["old_support_output"],
        "phase1_bundle": dict(value["phase1_bundle"]),
        "candidate_id": f"D3_R1_T_ERBT_{scenario.upper()}",
        "sf_tapft": sf_tapft,
    }


def build_data_handle(
    plan: Mapping[str, Any],
    scenario: str,
    *,
    new_class_count: int = 5,
    query_rows: int,
    split_id: str | None = None,
) -> dict[str, Any]:
    value, scene = _checked(plan, scenario)
    new_count = int(new_class_count)
    if new_count not in {1, 2, 3, 5, 10, 15, 20}:
        raise ValueError("D3 ERBT new-class matrix drift")
    if int(query_rows) <= 0:
        raise ValueError("D3 ERBT query row count must be positive")
    return {
        "schema": "cvs.sf_erbt_four_state.handle.v1",
        "protocol_schema": "p2_min_v1",
        "phase2_data_status": "VALIDATED_ONCE",
        "capsule_id": value["capsule_id"],
        "split_id": split_id or f"{scene['split_id']}-new{new_count}",
        "da_split_id": scene["split_id"],
        "scenario": scenario,
        "k_shot": 10,
        "old_class_count": 6,
        "new_class_count": new_count,
        "old_support_rows": 60,
        "registered_support_rows": (6 + new_count) * 10,
        "query_rows": int(query_rows),
        "base_checkpoint_path": value["base_checkpoint_path"],
    }


__all__ = ["FORMAL_SCENES", "build_d3_config", "build_data_handle"]
