"""BiSAGE support freeze, automatic stage gate, and truth-free query lifecycle."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np
import torch

from cvsrffi.stage2_binova_d92 import d92_geometry_features, exact_d92_fit
from cvsrffi.stage2_binova_features import BiNOVAQuery, BiNOVASupport
from cvsrffi.stage2_bisage_d92 import fit_bisage_d92
from cvsrffi.stage2_bisage_da import (
    SAGEDState,
    apply_sage_d,
    fit_sage_d,
    support_crossfit_masks,
)
from cvsrffi.stage2_bisage_reg import SAGERState, apply_sage_r, fit_sage_r


class BiSAGELifecycleError(ValueError):
    """Raised when stage ordering, state selection, or query isolation drifts."""


_FOUR_STATES = ("DA0_REG0", "DA1_REG0", "DA0_REG1", "DA1_REG1")


@dataclass(frozen=True)
class FrozenBiSAGEStates:
    states: Mapping[str, Any]
    stage_a: SAGEDState
    stage_b: SAGERState | None
    selected_mode: str
    old_class_count: int
    support_physical_ids: frozenset[str]
    context_binding: tuple[str, str, str, str]

    def __post_init__(self) -> None:
        if tuple(self.states) != _FOUR_STATES:
            raise BiSAGELifecycleError("four support states must use canonical ordering")
        if self.selected_mode not in {"S0", "S1", "S2"}:
            raise BiSAGELifecycleError("selected support mode is invalid")
        if self.selected_mode == "S2" and self.stage_b is None:
            raise BiSAGELifecycleError("S2 requires a frozen Stage B state")
        object.__setattr__(self, "states", MappingProxyType(dict(self.states)))


def next_stage(stage_a_gate: Mapping[str, Any]) -> str:
    if "stage_a_gate_passed" not in stage_a_gate:
        raise BiSAGELifecycleError("Stage A gate result is incomplete")
    return "STAGE_B" if bool(stage_a_gate["stage_a_gate_passed"]) else "STOPPED_SCIENTIFIC_GATE"


def _safe_against_s0(
    candidate: Mapping[str, Any], baseline: Mapping[str, Any], *, epsilon_new: float
) -> bool:
    required = (
        "old_accuracy",
        "new_accuracy",
        "h",
        "old_floor",
        "new_floor",
        "old_to_new",
        "positive_definite",
    )
    if any(key not in candidate or key not in baseline for key in required):
        raise BiSAGELifecycleError("registered mode metrics are incomplete")
    return (
        bool(candidate["positive_definite"])
        and float(candidate["old_accuracy"]) >= float(baseline["old_accuracy"])
        and float(candidate["h"]) >= float(baseline["h"])
        and float(candidate["old_floor"]) >= float(baseline["old_floor"])
        and float(candidate["new_floor"])
        >= float(baseline["new_floor"]) - float(epsilon_new)
        and float(candidate["old_to_new"]) <= float(baseline["old_to_new"])
    )


def select_mode_from_metrics(
    s0: Mapping[str, Any],
    s1: Mapping[str, Any],
    s2: Mapping[str, Any] | None,
    *,
    epsilon_new: float = 0.0,
) -> str:
    if s2 is not None and _safe_against_s0(s2, s0, epsilon_new=epsilon_new):
        return "S2"
    if _safe_against_s0(s1, s0, epsilon_new=epsilon_new):
        return "S1"
    return "S0"


def _mode_metrics(
    identity: np.ndarray,
    fft: np.ndarray,
    labels_np: np.ndarray,
    fit_np: np.ndarray,
    held_np: np.ndarray,
    old_class_count: int,
) -> Mapping[str, Any]:
    rows = torch.as_tensor(identity, dtype=torch.float64)
    spectra = torch.as_tensor(fft, dtype=torch.float64)
    labels = torch.as_tensor(labels_np, dtype=torch.long)
    fit = torch.as_tensor(fit_np, dtype=torch.bool)
    held = torch.as_tensor(held_np, dtype=torch.bool)
    geometry = d92_geometry_features(rows, spectra)
    state = fit_bisage_d92(
        geometry[fit], labels[fit], old_class_count=old_class_count
    )
    truth = labels[held]
    prediction = state.score(geometry[held]).argmax(1)
    old = truth < old_class_count
    new = ~old
    old_accuracy = float((prediction[old] == truth[old]).double().mean())
    new_accuracy = float((prediction[new] == truth[new]).double().mean())
    harmonic = 2.0 * old_accuracy * new_accuracy / max(
        old_accuracy + new_accuracy, 1.0e-12
    )
    per_class = {
        int(class_id): float(
            (prediction[truth == class_id] == truth[truth == class_id]).double().mean()
        )
        for class_id in torch.unique(truth, sorted=True).tolist()
    }
    old_to_new = float((prediction[old] >= old_class_count).double().mean())
    return MappingProxyType(
        {
            "old_accuracy": old_accuracy,
            "new_accuracy": new_accuracy,
            "h": harmonic,
            "old_floor": min(per_class[index] for index in range(old_class_count)),
            "new_floor": min(
                value for index, value in per_class.items() if index >= old_class_count
            ),
            "old_to_new": old_to_new,
            "positive_definite": state.audit["covariance_eigenvalue_min"] > 0.0,
            "query_rows_used": 0,
        }
    )


def evaluate_registered_modes(
    stage_a: SAGEDState,
    stage_b: SAGERState | None,
    support: BiNOVASupport,
    *,
    old_class_count: int,
    epsilon_new: float = 0.0,
) -> Mapping[str, Any]:
    classes, counts = np.unique(support.labels, return_counts=True)
    if int(counts.min()) < 5 or len(set(counts.tolist())) != 1:
        return MappingProxyType(
            {"selected_mode": "S0", "reason": "LOW_K_FALLBACK", "query_rows_used": 0}
        )
    if not np.array_equal(classes, np.arange(len(classes))):
        raise BiSAGELifecycleError("registered support labels must be contiguous")
    fit, held = support_crossfit_masks(support.labels, support.ranks)
    s0 = _mode_metrics(
        support.features.identity160,
        support.features.fft96,
        support.labels,
        fit,
        held,
        old_class_count,
    )
    stage_a_identity = apply_sage_d(stage_a, support.features)
    s1 = _mode_metrics(
        stage_a_identity,
        support.features.fft96,
        support.labels,
        fit,
        held,
        old_class_count,
    )
    s2 = None
    if stage_b is not None:
        stage_b_identity, stage_b_fft = apply_sage_r(stage_b, support.features)
        s2 = _mode_metrics(
            stage_b_identity,
            stage_b_fft,
            support.labels,
            fit,
            held,
            old_class_count,
        )
    selected = select_mode_from_metrics(s0, s1, s2, epsilon_new=epsilon_new)
    return MappingProxyType(
        {
            "selected_mode": selected,
            "S0": s0,
            "S1": s1,
            "S2": s2,
            "query_rows_used": 0,
        }
    )


def _binding(value: BiNOVASupport | BiNOVAQuery) -> tuple[str, str, str, str]:
    return tuple(
        str(value.context[key])
        for key in ("protocol_schema", "phase2_data_status", "capsule_id", "split_id")
    )


def freeze_bisage_support_states(
    old_support: BiNOVASupport,
    registered_support: BiNOVASupport,
    *,
    stage_a: SAGEDState,
    stage_b: SAGERState | None,
    selected_mode: str,
    seed: int,
    device: Any,
) -> FrozenBiSAGEStates:
    if not isinstance(old_support, BiNOVASupport) or not isinstance(
        registered_support, BiNOVASupport
    ):
        raise TypeError("support freeze requires old and registered support objects")
    if not isinstance(stage_a, SAGEDState):
        raise TypeError("support freeze requires Stage A state")
    if _binding(old_support) != _binding(registered_support):
        raise BiSAGELifecycleError("old/registered capsule or split binding mismatch")
    if selected_mode not in {"S0", "S1", "S2"} or (
        selected_mode == "S2" and stage_b is None
    ):
        raise BiSAGELifecycleError("selected mode is inconsistent with frozen adapters")
    old_ids = tuple(int(value) for value in np.unique(old_support.labels))
    registered_ids = tuple(int(value) for value in np.unique(registered_support.labels))
    if len(old_ids) != 6 or registered_ids[:6] != old_ids or len(registered_ids) <= 6:
        raise BiSAGELifecycleError("lifecycle requires six old plus registered new classes")

    da0_old_identity = old_support.features.identity160
    da1_old_identity = apply_sage_d(stage_a, old_support.features)
    da0_registered_identity = registered_support.features.identity160
    s1_registered_identity = apply_sage_d(stage_a, registered_support.features)
    selected_identity = da0_registered_identity
    selected_fft = registered_support.features.fft96
    if selected_mode == "S1":
        selected_identity = s1_registered_identity
    elif selected_mode == "S2":
        selected_identity, selected_fft = apply_sage_r(stage_b, registered_support.features)
    states = {
        "DA0_REG0": exact_d92_fit(
            da0_old_identity,
            old_support.features.fft96,
            old_support.labels,
            class_ids=old_ids,
            old_class_count=6,
            seed=int(seed),
            device=device,
        ),
        "DA1_REG0": exact_d92_fit(
            da1_old_identity,
            old_support.features.fft96,
            old_support.labels,
            class_ids=old_ids,
            old_class_count=6,
            seed=int(seed),
            device=device,
        ),
        "DA0_REG1": exact_d92_fit(
            da0_registered_identity,
            registered_support.features.fft96,
            registered_support.labels,
            class_ids=registered_ids,
            old_class_count=6,
            seed=int(seed),
            device=device,
        ),
        "DA1_REG1": exact_d92_fit(
            selected_identity,
            selected_fft,
            registered_support.labels,
            class_ids=registered_ids,
            old_class_count=6,
            seed=int(seed),
            device=device,
        ),
    }
    return FrozenBiSAGEStates(
        states=states,
        stage_a=stage_a,
        stage_b=stage_b,
        selected_mode=selected_mode,
        old_class_count=6,
        support_physical_ids=frozenset(old_support.features.physical_ids)
        | frozenset(registered_support.features.physical_ids),
        context_binding=_binding(old_support),
    )


def predict_bisage_query_read_only(
    frozen: FrozenBiSAGEStates, query: BiNOVAQuery
) -> Mapping[str, Mapping[str, Any]]:
    if not isinstance(frozen, FrozenBiSAGEStates):
        raise TypeError("query cannot open before support states are frozen")
    if not isinstance(query, BiNOVAQuery):
        raise TypeError("read-only prediction requires a label-free query object")
    if _binding(query) != frozen.context_binding:
        raise BiSAGELifecycleError("query capsule or split binding mismatch")
    if frozen.support_physical_ids.intersection(query.features.physical_ids):
        raise BiSAGELifecycleError("support/query physical IDs must be disjoint")
    da0_identity = query.features.identity160
    da1_identity = apply_sage_d(frozen.stage_a, query.features)
    selected_identity = da0_identity
    selected_fft = query.features.fft96
    if frozen.selected_mode == "S1":
        selected_identity = da1_identity
    elif frozen.selected_mode == "S2":
        selected_identity, selected_fft = apply_sage_r(frozen.stage_b, query.features)
    features = {
        "DA0_REG0": (da0_identity, query.features.fft96),
        "DA1_REG0": (da1_identity, query.features.fft96),
        "DA0_REG1": (da0_identity, query.features.fft96),
        "DA1_REG1": (selected_identity, selected_fft),
    }
    output = {}
    for state_name, state in frozen.states.items():
        identity, fft = features[state_name]
        logits = np.asarray(state.score(identity, fft), dtype=np.float32)
        class_ids = np.asarray(state.class_ids, dtype=np.int64)
        output[state_name] = MappingProxyType(
            {
                "query_ids": np.asarray(query.features.physical_ids),
                "class_ids": class_ids,
                "logits": logits,
                "predictions": class_ids[np.argmax(logits, axis=1)],
                "new_accuracy": "N/A"
                if state_name.endswith("REG0")
                else "PENDING_TRUTH",
            }
        )
    return MappingProxyType(output)


__all__ = [
    "BiSAGELifecycleError",
    "FrozenBiSAGEStates",
    "evaluate_registered_modes",
    "fit_sage_d",
    "fit_sage_r",
    "freeze_bisage_support_states",
    "next_stage",
    "predict_bisage_query_read_only",
    "select_mode_from_metrics",
]
