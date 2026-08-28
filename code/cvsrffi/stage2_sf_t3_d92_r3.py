"""R3 dual-delta t3-norm adaptation with support-only D92 cross-fit risk."""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor

from cvsrffi.stage2_sf_erbt_four_state import fit_registered_erbt


CANDIDATE_ID = "R3_T3NORM_D92_INLOOP"
_T3_NORM_KEYS = (
    "model.t3.norm.bias",
    "model.t3.norm.weight",
)
_RISK_WEIGHTS = MappingProxyType(
    {
        "macro_nll": 1.0,
        "class_tail_nll": 0.30,
        "class_floor_error": 0.20,
        "old_new_balance": 0.10,
    }
)


@dataclass(frozen=True)
class D92RiskFold:
    fold: int
    train_indices: tuple[int, ...]
    heldout_indices: tuple[int, ...]
    fit_support_rows: int
    heldout_support_rows: int
    macro_nll: float
    class_tail_nll: float
    class_floor_error: float
    old_new_balance: float
    total: float


@dataclass(frozen=True)
class D92SupportRisk:
    total: float
    macro_nll: float
    class_tail_nll: float
    class_floor_error: float
    old_new_balance: float
    folds: tuple[D92RiskFold, ...]
    audit: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "audit", MappingProxyType(dict(self.audit)))


def aggregate_r3_t3_norm_delta(
    anchor: Mapping[str, Tensor],
    fitted_states: Sequence[Mapping[str, Tensor]],
) -> tuple[Mapping[str, Tensor], Mapping[str, Any]]:
    """Average exactly two aligned t3-norm deltas and discard temporary heads."""

    states = tuple(fitted_states)
    if len(states) != 2:
        raise ValueError("R3 requires exactly two fitted delta states")
    output: dict[str, Tensor] = {}
    for name in _T3_NORM_KEYS:
        if name not in anchor or any(name not in state for state in states):
            raise ValueError(f"R3 state is missing required parameter: {name}")
        base = anchor[name]
        values = tuple(state[name] for state in states)
        if not torch.is_tensor(base) or any(not torch.is_tensor(value) for value in values):
            raise TypeError(f"R3 parameter must be a tensor: {name}")
        if any(value.shape != base.shape for value in values):
            raise ValueError(f"R3 parameter shape mismatch: {name}")
        if not bool(torch.isfinite(base).all()) or any(
            not bool(torch.isfinite(value).all()) for value in values
        ):
            raise ValueError(f"R3 parameter must be finite: {name}")
        deltas = torch.stack(
            [value.detach().cpu() - base.detach().cpu() for value in values]
        )
        output[name] = deltas.mean(dim=0)
    audit = MappingProxyType(
        {
            "ensemble": "R3_DUAL_DELTA",
            "dual_delta_count": 2,
            "deployment_delta_parameter_names": list(_T3_NORM_KEYS),
            "temporary_target_head_persisted": False,
        }
    )
    return MappingProxyType(output), audit


def _validate_support(
    identity160: Any,
    fft96: Any,
    labels: Any,
    *,
    class_ids: Sequence[int],
    old_class_ids: Sequence[int],
    folds: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, ...], tuple[int, ...], int]:
    identity = np.asarray(identity160, dtype=np.float32)
    fft = np.asarray(fft96, dtype=np.float32)
    target = np.asarray(labels, dtype=np.int64)
    registry = tuple(int(value) for value in class_ids)
    old_registry = tuple(int(value) for value in old_class_ids)
    if len(registry) <= 6 or len(set(registry)) != len(registry):
        raise ValueError("D92 registration requires six old classes and at least one new class")
    if old_registry != registry[:6] or len(old_registry) != 6:
        raise ValueError("old_class_ids must be the six-class registry prefix")
    if identity.ndim != 2 or identity.shape[1] != 160:
        raise ValueError("identity160 must be an N x 160 matrix")
    if fft.shape != (len(identity), 96) or target.shape != (len(identity),):
        raise ValueError("support identity, FFT96 and labels must be row aligned")
    if not np.isfinite(identity).all() or not np.isfinite(fft).all():
        raise ValueError("support features must be finite")
    if set(target.tolist()) != set(registry):
        raise ValueError("support labels must exactly cover the registered classes")
    counts = tuple(int(np.sum(target == class_id)) for class_id in registry)
    if not counts or len(set(counts)) != 1:
        raise ValueError("D92 cross-fit support must be balanced K-shot")
    k_shot = counts[0]
    if isinstance(folds, bool) or int(folds) < 2 or int(folds) > k_shot:
        raise ValueError("folds must be in [2, K]")
    return identity, fft, target, registry, old_registry, int(folds)


def _class_balanced_folds(
    labels: np.ndarray,
    registry: tuple[int, ...],
    *,
    folds: int,
    seed: int,
) -> tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]:
    generator = np.random.default_rng(int(seed))
    heldout_by_fold: list[list[int]] = [[] for _ in range(folds)]
    for class_id in registry:
        indices = np.flatnonzero(labels == class_id).astype(np.int64)
        shuffled = generator.permutation(indices)
        for fold, rows in enumerate(np.array_split(shuffled, folds)):
            heldout_by_fold[fold].extend(int(value) for value in rows.tolist())
    universe = set(range(len(labels)))
    result = []
    for rows in heldout_by_fold:
        heldout = tuple(sorted(rows))
        train = tuple(sorted(universe - set(heldout)))
        result.append((train, heldout))
    return tuple(result)


def _log_softmax(values: np.ndarray) -> np.ndarray:
    logits = np.asarray(values, dtype=np.float64)
    maximum = np.max(logits, axis=1, keepdims=True)
    shifted = logits - maximum
    return shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))


def _fold_risk(
    logits: np.ndarray,
    labels: np.ndarray,
    *,
    registry: tuple[int, ...],
    old_registry: tuple[int, ...],
) -> tuple[float, float, float, float, float]:
    if logits.shape != (len(labels), len(registry)) or not np.isfinite(logits).all():
        raise ValueError("heldout D92 logits have invalid geometry")
    lookup = {class_id: column for column, class_id in enumerate(registry)}
    columns = np.asarray([lookup[int(value)] for value in labels], dtype=np.int64)
    row_nll = -_log_softmax(logits)[np.arange(len(labels)), columns]
    predictions = np.asarray(registry, dtype=np.int64)[np.argmax(logits, axis=1)]
    class_nll = np.asarray(
        [row_nll[labels == class_id].mean() for class_id in registry], dtype=np.float64
    )
    class_accuracy = np.asarray(
        [np.mean(predictions[labels == class_id] == class_id) for class_id in registry],
        dtype=np.float64,
    )
    macro_nll = float(class_nll.mean())
    class_tail_nll = float(class_nll.max())
    class_floor_error = float(1.0 - class_accuracy.min())
    old_columns = [registry.index(class_id) for class_id in old_registry]
    new_columns = [index for index in range(len(registry)) if index not in old_columns]
    old_new_balance = float(
        abs(class_nll[old_columns].mean() - class_nll[new_columns].mean())
    )
    total = float(
        _RISK_WEIGHTS["macro_nll"] * macro_nll
        + _RISK_WEIGHTS["class_tail_nll"] * class_tail_nll
        + _RISK_WEIGHTS["class_floor_error"] * class_floor_error
        + _RISK_WEIGHTS["old_new_balance"] * old_new_balance
    )
    return total, macro_nll, class_tail_nll, class_floor_error, old_new_balance


def crossfit_d92_support_risk(
    identity160: Any,
    fft96: Any,
    labels: Any,
    *,
    class_ids: Sequence[int],
    old_class_ids: Sequence[int],
    folds: int = 2,
    seed: int = 713101,
    device: Any = "cpu",
) -> D92SupportRisk:
    """Fit D92-E0 on train support and score only heldout support in each fold."""

    identity, fft, target, registry, old_registry, fold_count = _validate_support(
        identity160,
        fft96,
        labels,
        class_ids=class_ids,
        old_class_ids=old_class_ids,
        folds=folds,
    )
    rows = []
    for fold, (train_indices, heldout_indices) in enumerate(
        _class_balanced_folds(
            target, registry, folds=fold_count, seed=int(seed)
        )
    ):
        train = np.asarray(train_indices, dtype=np.int64)
        heldout = np.asarray(heldout_indices, dtype=np.int64)
        state = fit_registered_erbt(
            identity[train],
            fft[train],
            target[train],
            class_ids=registry,
            old_class_count=6,
            seed=int(seed) + fold,
            device=device,
        )
        if (
            state.audit.get("method_lock") != "D92-E0-NORF32"
            or state.audit.get("rf32_used") is not False
            or state.audit.get("support_only") is not True
            or int(state.audit.get("query_rows_used", -1)) != 0
        ):
            raise RuntimeError("D92-E0 support-only fit audit drift")
        metrics = _fold_risk(
            state.score(identity[heldout], fft[heldout]),
            target[heldout],
            registry=registry,
            old_registry=old_registry,
        )
        rows.append(
            D92RiskFold(
                fold=fold,
                train_indices=train_indices,
                heldout_indices=heldout_indices,
                fit_support_rows=len(train_indices),
                heldout_support_rows=len(heldout_indices),
                total=metrics[0],
                macro_nll=metrics[1],
                class_tail_nll=metrics[2],
                class_floor_error=metrics[3],
                old_new_balance=metrics[4],
            )
        )
    values = np.asarray(
        [
            [
                row.total,
                row.macro_nll,
                row.class_tail_nll,
                row.class_floor_error,
                row.old_new_balance,
            ]
            for row in rows
        ],
        dtype=np.float64,
    ).mean(axis=0)
    if not np.isfinite(values).all() or not all(math.isfinite(row.total) for row in rows):
        raise RuntimeError("D92 support risk is not finite")
    audit = {
        "candidate_id": CANDIDATE_ID,
        "support_only": True,
        "crossfit_folds": fold_count,
        "d92_fit_count": fold_count,
        "d92_method_lock": "D92-E0-NORF32",
        "rf32_used": False,
        "label_permutation_invariant": True,
        "risk_weights": dict(_RISK_WEIGHTS),
        "query_rows_used": 0,
        "query_truth_opened": False,
        "query_role_opened": False,
    }
    return D92SupportRisk(
        total=float(values[0]),
        macro_nll=float(values[1]),
        class_tail_nll=float(values[2]),
        class_floor_error=float(values[3]),
        old_new_balance=float(values[4]),
        folds=tuple(rows),
        audit=audit,
    )


def build_candidate_spec() -> dict[str, Any]:
    """Return the serializable contract consumed by the unified Stage2 runner."""

    return {
        "schema": "cvs.stage2.sf_t3_d92_r3.candidate.v1",
        "candidate_id": CANDIDATE_ID,
        "runner_entrypoints": {
            "aggregate_delta": "aggregate_r3_t3_norm_delta",
            "support_risk": "crossfit_d92_support_risk",
        },
        "adaptation": {
            "ensemble": "R3_DUAL_DELTA",
            "delta_count": 2,
            "persistent_parameter_names": list(_T3_NORM_KEYS),
            "temporary_target_head_persisted": False,
        },
        "registration": {
            "method_lock": "D92-E0-NORF32",
            "rf32_used": False,
            "registration_head_persistent": False,
        },
        "risk": {
            "crossfit": True,
            "support_only": True,
            "label_permutation_invariant": True,
            "components": dict(_RISK_WEIGHTS),
        },
        "protocol_audit": {
            "protocol_schema": "p2_min_v1",
            "query_rows_used": 0,
            "query_truth_opened": False,
            "query_role_opened": False,
        },
    }


__all__ = [
    "CANDIDATE_ID",
    "D92RiskFold",
    "D92SupportRisk",
    "aggregate_r3_t3_norm_delta",
    "build_candidate_spec",
    "crossfit_d92_support_risk",
]
