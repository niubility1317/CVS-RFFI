"""Frozen plan expansion for the three t3.norm plus D92 combinations."""

from __future__ import annotations

from types import MappingProxyType
from typing import Any, Mapping

from cvsrffi.target_only_progressive_adapt import SFTAPFTConfig


FORMAL_SCENES = (
    "leo_clear_weak",
    "leo_low_elev_weak",
    "leo_rain_weak",
)
NEW_CLASS_COUNTS = (2, 10, 20)
CANDIDATES = (
    "D0_T3_D92",
    "S02_T3_D92",
    "R3_DUALDELTA_T3_D92_INLOOP",
)
FOUR_STATES = ("DA0_REG0", "DA1_REG0", "DA0_REG1", "DA1_REG1")
D92_METHOD_LOCK = "D92-E0-NORF32"

_TOP_KEYS = frozenset(
    {
        "schema",
        "run_id",
        "protocol_schema",
        "phase2_data_status",
        "capsule_id",
        "base_checkpoint_path",
        "phase1_bundle",
        "old_class_count",
        "k_shot",
        "d92_method_lock",
        "rf32_used",
        "new_class_counts",
        "candidates",
        "scenes",
    }
)
_SCENE_KEYS = frozenset(
    {
        "gpu",
        "split_id",
        "old_support",
        "registered_support_pattern",
        "query_pattern",
        "data_handle_pattern",
    }
)


def validate_combo_plan(plan: Mapping[str, Any]) -> Mapping[str, Any]:
    """Validate only the scientific and execution fields consumed by the runner."""

    if not isinstance(plan, Mapping) or frozenset(plan) != _TOP_KEYS:
        raise ValueError("combo plan top-level allowlist mismatch")
    value = dict(plan)
    if value["schema"] != "cvs.sf_t3_d92.combo_plan.v1":
        raise ValueError("combo plan schema mismatch")
    if value["protocol_schema"] != "p2_min_v1":
        raise ValueError("combo plan protocol mismatch")
    if value["phase2_data_status"] != "VALIDATED_ONCE":
        raise ValueError("combo plan data status mismatch")
    if value["old_class_count"] != 6 or value["k_shot"] != 10:
        raise ValueError("combo plan support geometry mismatch")
    if (
        value["d92_method_lock"] != D92_METHOD_LOCK
        or value["rf32_used"] is not False
    ):
        raise ValueError("combo plan D92 method lock mismatch")
    if tuple(value["new_class_counts"]) != NEW_CLASS_COUNTS:
        raise ValueError("combo plan new-class matrix drift")
    if tuple(value["candidates"]) != CANDIDATES:
        raise ValueError("combo plan candidate matrix drift")
    if not isinstance(value["phase1_bundle"], Mapping):
        raise ValueError("combo plan Phase1 bundle mapping drift")
    for name in ("run_id", "capsule_id", "base_checkpoint_path"):
        if not isinstance(value[name], str) or not value[name].strip():
            raise ValueError(f"combo plan {name} is empty")
    scenes = value["scenes"]
    if not isinstance(scenes, Mapping) or tuple(scenes) != FORMAL_SCENES:
        raise ValueError("combo plan scene matrix drift")
    normalized_scenes: dict[str, Mapping[str, Any]] = {}
    for scene in FORMAL_SCENES:
        raw = scenes[scene]
        if not isinstance(raw, Mapping) or frozenset(raw) != _SCENE_KEYS:
            raise ValueError("combo plan scene allowlist mismatch")
        row = dict(raw)
        if isinstance(row["gpu"], bool) or int(row["gpu"]) not in range(8):
            raise ValueError("combo plan GPU drift")
        if any(
            not isinstance(row[name], str) or not row[name].strip()
            for name in _SCENE_KEYS - {"gpu"}
        ):
            raise ValueError("combo plan scene path is empty")
        for name in (
            "registered_support_pattern",
            "query_pattern",
            "data_handle_pattern",
        ):
            if "{new_count}" not in row[name]:
                raise ValueError("combo plan nested path pattern drift")
        normalized_scenes[scene] = MappingProxyType(row)
    value["scenes"] = MappingProxyType(normalized_scenes)
    value["phase1_bundle"] = MappingProxyType(dict(value["phase1_bundle"]))
    return MappingProxyType(value)


def build_experiment_rows(plan: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    """Expand the locked 27-cell first-falsification matrix."""

    value = validate_combo_plan(plan)
    rows = []
    for candidate in CANDIDATES:
        for scenario in FORMAL_SCENES:
            scene = value["scenes"][scenario]
            for new_count in NEW_CLASS_COUNTS:
                rows.append(
                    {
                        "row_id": f"{candidate}__{scenario}__new{new_count}",
                        "candidate_id": candidate,
                        "scenario": scenario,
                        "new_class_count": new_count,
                        "gpu": int(scene["gpu"]),
                        "split_id": f"{scene['split_id']}-new{new_count}",
                        "old_support": scene["old_support"],
                        "registered_support": scene[
                            "registered_support_pattern"
                        ].format(new_count=new_count),
                        "query": scene["query_pattern"].format(
                            new_count=new_count
                        ),
                        "data_handle": scene["data_handle_pattern"].format(
                            new_count=new_count
                        ),
                        "four_states": list(FOUR_STATES),
                        "d92_method_lock": D92_METHOD_LOCK,
                        "rf32_used": False,
                        "query_fit_access": False,
                    }
                )
    return tuple(rows)


def build_candidate_config(candidate_id: str) -> SFTAPFTConfig:
    """Return the historical same-method training lock for one combo candidate."""

    if candidate_id not in CANDIDATES:
        raise ValueError("unknown t3 plus D92 candidate")
    common = dict(
        adapter_rank=16,
        trainability_profile="p1_head_norm",
        norm_rules=(("t3", "weight_bias"),),
        classifier_source_target_interpolation=0.5,
        prototype_scale=8.0,
        label_smoothing=0.05,
        lambda_proto=0.5,
        lambda_l2sp=1.0e-4,
        selective_kd_weight=0.0,
        selective_kd_temperature=2.0,
        selective_kd_gamma=2.0,
        lr_head_initial=1.0e-3,
        lr_norm=1.0e-4,
        lr_head_middle=3.0e-4,
        lr_head_late=1.0e-4,
        weight_decay=1.0e-4,
        warmup_ratio=0.05,
        gradient_clip_norm=1.0,
        mixed_precision=True,
        seed=392002,
    )
    if candidate_id == "S02_T3_D92":
        return SFTAPFTConfig(
            **common,
            phase_steps=(4500, 0, 0),
            scheduler_reference_steps=0,
            validation_steps=(),
            checkpoint_average_top_k=3,
        )
    return SFTAPFTConfig(
        **common,
        inference_temperature=0.9443055457851892,
        phase_steps=(300, 150, 70),
        scheduler_reference_steps=4500,
        fast_tail_start_step=300,
        fast_tail_steps=150,
        fast_tail_lr_head_start=2.0e-4,
        fast_tail_lr_head_end=2.0e-5,
        fast_tail_lr_norm_start=3.0e-5,
        fast_tail_lr_norm_end=3.0e-6,
        head_polish_steps=70,
        head_polish_lr=5.0e-5,
        cache_storage_dtype="float32",
        suffix_compute_dtype="float32",
        cache_device="model",
        checkpoint_average_top_k=1,
    )


__all__ = [
    "CANDIDATES",
    "D92_METHOD_LOCK",
    "FORMAL_SCENES",
    "FOUR_STATES",
    "NEW_CLASS_COUNTS",
    "build_candidate_config",
    "build_experiment_rows",
    "validate_combo_plan",
]
