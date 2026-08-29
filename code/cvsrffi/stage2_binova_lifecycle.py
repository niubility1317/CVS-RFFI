"""BiNOVA support freeze, conditional fallback, and read-only four-state query."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from cvsrffi.stage2_binova_d92 import exact_d92_fit
from cvsrffi.stage2_binova_da import NOVA_DA_State, apply_nova_da
from cvsrffi.stage2_binova_features import BiNOVAQuery, BiNOVASupport
from cvsrffi.stage2_binova_reg import NOVA_REG_State, apply_nova_reg


class BiNOVALifecycleError(ValueError):
    """Raised when support freeze or query-read-only ordering is violated."""


@dataclass(frozen=True)
class StageAContinuationGate:
    passed: bool
    pseudo_h_gain: float
    forgetting_increase: float
    old_floor_change: float
    non_affine_fraction: float
    query_rows_used: int = 0


@dataclass(frozen=True)
class FrozenBiNOVAStates:
    states: Mapping[str, Any]
    stage_a: NOVA_DA_State
    stage_b: NOVA_REG_State | None
    selected_mode: str
    old_class_count: int
    support_physical_ids: frozenset[str]
    context_binding: tuple[str, str, str, str]

    def __post_init__(self) -> None:
        expected = ("DA0_REG0", "DA1_REG0", "DA0_REG1", "DA1_REG1")
        if tuple(self.states) != expected:
            raise BiNOVALifecycleError("four support states must be frozen in canonical order")
        object.__setattr__(self, "states", MappingProxyType(dict(self.states)))


def evaluate_stage_a_continuation_gate(
    control: Mapping[str, float],
    candidate: Mapping[str, float],
    *,
    non_affine_fraction: float,
) -> StageAContinuationGate:
    required = ("pseudo_h", "pseudo_forgetting", "pseudo_old_floor")
    if any(key not in control or key not in candidate for key in required):
        raise BiNOVALifecycleError("Stage A gate requires support cross-fit metrics")
    h_gain = float(candidate["pseudo_h"]) - float(control["pseudo_h"])
    forgetting_increase = float(candidate["pseudo_forgetting"]) - float(control["pseudo_forgetting"])
    floor_change = float(candidate["pseudo_old_floor"]) - float(control["pseudo_old_floor"])
    nonlinear = float(non_affine_fraction)
    passed = h_gain >= 0.005 and forgetting_increase <= 0.0 and floor_change >= 0.0 and nonlinear >= 0.20
    return StageAContinuationGate(
        passed=passed,
        pseudo_h_gain=h_gain,
        forgetting_increase=forgetting_increase,
        old_floor_change=floor_change,
        non_affine_fraction=nonlinear,
    )


def select_binova_mode(*, stage_a_gate_passed: bool, stage_b_gate_passed: bool) -> str:
    if not stage_a_gate_passed:
        return "S0"
    return "S2" if stage_b_gate_passed else "S1"


def _binding(support: BiNOVASupport | BiNOVAQuery) -> tuple[str, str, str, str]:
    return tuple(
        str(support.context[key])
        for key in ("protocol_schema", "phase2_data_status", "capsule_id", "split_id")
    )


def freeze_binova_support_states(
    old_support: BiNOVASupport,
    registered_support: BiNOVASupport,
    *,
    stage_a: NOVA_DA_State,
    stage_b: NOVA_REG_State | None,
    selected_mode: str,
    seed: int,
    device: Any,
) -> FrozenBiNOVAStates:
    if not isinstance(old_support, BiNOVASupport) or not isinstance(registered_support, BiNOVASupport):
        raise TypeError("support states require old and registered BiNOVASupport")
    if not isinstance(stage_a, NOVA_DA_State):
        raise TypeError("support states require a frozen Stage A state")
    if selected_mode not in {"S0", "S1", "S2"} or (selected_mode == "S2" and stage_b is None):
        raise BiNOVALifecycleError("selected fallback mode is inconsistent with frozen states")
    if _binding(old_support) != _binding(registered_support):
        raise BiNOVALifecycleError("old/registered support capsule or split binding mismatch")
    old_ids = tuple(int(value) for value in np.unique(old_support.labels))
    registered_ids = tuple(int(value) for value in np.unique(registered_support.labels))
    if len(old_ids) != 6 or registered_ids[:6] != old_ids or len(registered_ids) <= 6:
        raise BiNOVALifecycleError("BiNOVA lifecycle requires six old plus registered-new classes")
    da0_old_identity = old_support.features.identity160
    da1_old_identity = apply_nova_da(stage_a, old_support.features)
    da0_registered_identity = registered_support.features.identity160
    da1_registered_identity = apply_nova_da(stage_a, registered_support.features)
    da1_registered_fft = registered_support.features.fft96
    if stage_b is not None:
        da1_registered_identity, da1_registered_fft = apply_nova_reg(stage_b, registered_support.features)
    states = {
        "DA0_REG0": exact_d92_fit(
            da0_old_identity, old_support.features.fft96, old_support.labels,
            class_ids=old_ids, old_class_count=6, seed=seed, device=device,
        ),
        "DA1_REG0": exact_d92_fit(
            da1_old_identity, old_support.features.fft96, old_support.labels,
            class_ids=old_ids, old_class_count=6, seed=seed, device=device,
        ),
        "DA0_REG1": exact_d92_fit(
            da0_registered_identity, registered_support.features.fft96, registered_support.labels,
            class_ids=registered_ids, old_class_count=6, seed=seed, device=device,
        ),
        "DA1_REG1": exact_d92_fit(
            da1_registered_identity, da1_registered_fft, registered_support.labels,
            class_ids=registered_ids, old_class_count=6, seed=seed, device=device,
        ),
    }
    support_ids = frozenset(old_support.features.physical_ids) | frozenset(
        registered_support.features.physical_ids
    )
    return FrozenBiNOVAStates(
        states=states,
        stage_a=stage_a,
        stage_b=stage_b,
        selected_mode=selected_mode,
        old_class_count=6,
        support_physical_ids=support_ids,
        context_binding=_binding(old_support),
    )


def predict_binova_query_read_only(
    frozen: FrozenBiNOVAStates,
    query: BiNOVAQuery,
) -> Mapping[str, Mapping[str, Any]]:
    if not isinstance(frozen, FrozenBiNOVAStates):
        raise TypeError("query cannot open before frozen support states exist")
    if not isinstance(query, BiNOVAQuery):
        raise TypeError("read-only prediction requires a label-free query object")
    if _binding(query) != frozen.context_binding:
        raise BiNOVALifecycleError("query capsule or split binding mismatch")
    if frozen.support_physical_ids.intersection(query.features.physical_ids):
        raise BiNOVALifecycleError("support/query physical IDs must be disjoint")
    da0_identity = query.features.identity160
    da1_identity = apply_nova_da(frozen.stage_a, query.features)
    da1_fft = query.features.fft96
    if frozen.stage_b is not None:
        da1_identity, da1_fft = apply_nova_reg(frozen.stage_b, query.features)
    feature_by_state = {
        "DA0_REG0": (da0_identity, query.features.fft96),
        "DA1_REG0": (da1_identity, query.features.fft96),
        "DA0_REG1": (da0_identity, query.features.fft96),
        "DA1_REG1": (da1_identity, da1_fft),
    }
    output: dict[str, Mapping[str, Any]] = {}
    for state_name, state in frozen.states.items():
        identity, fft = feature_by_state[state_name]
        logits = np.asarray(state.score(identity, fft), dtype=np.float32)
        class_ids = np.asarray(state.class_ids, dtype=np.int64)
        predictions = class_ids[np.argmax(logits, axis=1)]
        output[state_name] = MappingProxyType(
            {
                "query_ids": np.asarray(query.features.physical_ids),
                "class_ids": class_ids,
                "logits": logits,
                "predictions": predictions,
                "new_accuracy": "N/A" if state_name.endswith("REG0") else "PENDING_TRUTH",
            }
        )
    return MappingProxyType(output)


__all__ = [
    "BiNOVALifecycleError",
    "FrozenBiNOVAStates",
    "StageAContinuationGate",
    "evaluate_stage_a_continuation_gate",
    "freeze_binova_support_states",
    "predict_binova_query_read_only",
    "select_binova_mode",
]
