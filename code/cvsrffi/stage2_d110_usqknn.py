"""D110 US-qKNN: a four-arm, support-only Student-t qKNN core.

The module deliberately separates the two proposed effects without changing the
already locked qKNN likelihood:

* ``M0`` keeps the current identity qKNN exactly;
* ``M_DA`` replaces only its pair distance by D110's safe-relative SCPM
  Mahalanobis distance;
* ``M_HEAD`` keeps identity distance but shares the support-pair bandwidth;
* ``M_JOINT`` combines those two replacements.

All states are fit from target support only.  Queries enter only the pure score
functions and cannot update the support bank, SCPM state, bandwidths, or class
registry.  The SCPM distance uses ``safe_relative_variances`` intentionally:
the predictive ``(1 + 1/K)`` factor belongs to the center classifier and is not
part of the qKNN support-pair likelihood.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

from . import stage2_d110_scpm_runtime as scpm
from . import stage2_zid_student_t_qknn as qknn


ARMS = ("M0", "M_DA", "M_HEAD", "M_JOINT")
_DA_ARMS = frozenset(("M_DA", "M_JOINT"))
_SHARED_HEAD_ARMS = frozenset(("M_HEAD", "M_JOINT"))
SCHEMA = "cvs.phase2.d110.usqknn.v1"


class D110USQKNNError(ValueError):
    """Raised when the frozen four-arm qKNN contract drifts."""


def _readonly(value: np.ndarray, dtype: np.dtype | type) -> np.ndarray:
    copied = np.array(value, dtype=dtype, copy=True, order="C")
    copied.setflags(write=False)
    return copied


def _require_arm(arm: str) -> str:
    if type(arm) is not str or arm not in ARMS:
        raise D110USQKNNError(f"arm must be one of {ARMS}")
    return arm


def _identity_distances(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Use the exact identity-qKNN distance expression, not a new L2 path."""

    return np.maximum(2.0 * (1.0 - left @ right.T), 0.0)


def _safe_relative_scpm_distances(
    left: np.ndarray,
    right: np.ndarray,
    state: scpm.D110SCPMRuntimeState,
) -> np.ndarray:
    """Return D110's support-pair metric without a predictive K multiplier."""

    relative = np.asarray(state.safe_relative_variances, dtype=np.float64)
    basis = np.asarray(state.closed_u, dtype=np.float64)
    if (
        relative.shape != (basis.shape[0] + 1,)
        or not np.isfinite(relative).all()
        or np.any(relative <= 0.0)
    ):
        raise D110USQKNNError("D110 SCPM safe-relative variance state drift")
    left64 = np.asarray(left, dtype=np.float64)
    right64 = np.asarray(right, dtype=np.float64)
    left_projected = left64 @ basis.T
    right_projected = right64 @ basis.T
    projected_delta = left_projected[:, None, :] - right_projected[None, :, :]
    parallel_energy = np.sum(np.square(projected_delta), axis=2)
    total_energy = (
        np.sum(np.square(left64), axis=1)[:, None]
        + np.sum(np.square(right64), axis=1)[None, :]
        - 2.0 * (left64 @ right64.T)
    )
    perpendicular_energy = np.maximum(total_energy - parallel_energy, 0.0)
    distance = (
        np.sum(np.square(projected_delta) / relative[None, None, :-1], axis=2)
        + perpendicular_energy / relative[-1]
    )
    if not np.isfinite(distance).all() or np.any(distance < 0.0):
        raise D110USQKNNError("D110 SCPM qKNN pair distance became invalid")
    return np.ascontiguousarray(distance, dtype=np.float64)


def _per_class_pair_energies(
    pair_distances: np.ndarray,
    bank: qknn.TypedINT8ZIDSupportBank,
) -> np.ndarray:
    """One unordered-pair mean per class, in the bank's registry order."""

    distances = np.asarray(pair_distances, dtype=np.float64)
    rows = bank.support_row_count
    if (
        distances.shape != (rows, rows)
        or not np.isfinite(distances).all()
        or np.any(distances < 0.0)
    ):
        raise D110USQKNNError("support-pair distance matrix closure drift")
    values: list[float] = []
    for class_index, expected in enumerate(bank.support_counts):
        local = distances[
            np.ix_(
                bank.class_indices_int16 == class_index,
                bank.class_indices_int16 == class_index,
            )
        ]
        if local.shape != (expected, expected):
            raise D110USQKNNError("class support count drift in pair energy")
        if expected == 1:
            values.append(0.0)
            continue
        upper = local[np.triu_indices(expected, 1)]
        if len(upper) != expected * (expected - 1) // 2:
            raise D110USQKNNError("unordered support-pair closure drift")
        values.append(float(np.mean(upper)))
    return np.asarray(values, dtype=np.float64)


def _shrink_and_clip_scales(
    empirical: np.ndarray,
    bank: qknn.TypedINT8ZIDSupportBank,
) -> np.ndarray:
    """The existing locked qKNN shrink/clip equation, then its FP16 closure."""

    values = np.asarray(empirical, dtype=np.float64)
    config = bank.config
    if (
        values.shape != (len(bank.classes),)
        or not np.isfinite(values).all()
        or np.any(values < 0.0)
    ):
        raise D110USQKNNError("support pair-energy vector drift")
    shrunk = (
        values + config.scale_prior_strength * config.shared_h0**2
    ) / (1.0 + config.scale_prior_strength)
    scales = np.clip(
        np.sqrt(np.maximum(shrunk, qknn.EPSILON)),
        config.shared_h0 * config.scale_min_ratio,
        config.shared_h0 * config.scale_max_ratio,
    )
    closed = np.asarray(scales, dtype=np.float16)
    if not np.isfinite(closed).all() or np.any(closed <= 0.0):
        raise D110USQKNNError("FP16 class-scale closure drift")
    return _readonly(closed, np.float16)


def _shared_scales(
    pair_distances: np.ndarray,
    bank: qknn.TypedINT8ZIDSupportBank,
) -> np.ndarray:
    """Equal-class mean energy followed by the same locked shrink/clip rule."""

    if bank.active_k == 1:
        # Reuse exactly the existing qKNN K=1 FP16 closure.  This makes the
        # head main effect an identity at K=1, as frozen in the theory.
        return _readonly(bank.class_scales_fp16, np.float16)
    per_class = _per_class_pair_energies(pair_distances, bank)
    shared_energy = np.full(
        len(bank.classes), float(np.mean(per_class, dtype=np.float64)), dtype=np.float64
    )
    return _shrink_and_clip_scales(shared_energy, bank)


def _da_class_specific_scales(
    pair_distances: np.ndarray,
    bank: qknn.TypedINT8ZIDSupportBank,
) -> np.ndarray:
    if bank.active_k == 1:
        return _readonly(bank.class_scales_fp16, np.float16)
    return _shrink_and_clip_scales(_per_class_pair_energies(pair_distances, bank), bank)


def _score_from_distances(
    distances: np.ndarray,
    bank: qknn.TypedINT8ZIDSupportBank,
    class_scales: np.ndarray,
) -> np.ndarray:
    """The unchanged Student-t LSE-minus-log-K likelihood core."""

    values = np.asarray(distances, dtype=np.float64)
    scales = np.asarray(class_scales, dtype=np.float16)
    if (
        values.ndim != 2
        or values.shape[1] != bank.support_row_count
        or not np.isfinite(values).all()
        or np.any(values < 0.0)
        or scales.shape != (len(bank.classes),)
        or not np.isfinite(scales).all()
        or np.any(scales <= 0.0)
    ):
        raise D110USQKNNError("Student-t qKNN scoring input closure drift")
    config = bank.config
    columns: list[np.ndarray] = []
    for class_index, expected in enumerate(bank.support_counts):
        local = values[:, bank.class_indices_int16 == class_index]
        if local.shape[1] != expected:
            raise D110USQKNNError("class support count drift during qKNN score")
        h = float(scales[class_index])
        kernel = (
            -config.kernel_volume_gamma
            * config.kernel_effective_dim
            * math.log(h)
            - 0.5
            * (config.student_nu + config.kernel_effective_dim)
            * np.log1p(local / (config.student_nu * h * h))
        )
        maximum = np.max(kernel, axis=1, keepdims=True)
        columns.append(
            maximum[:, 0]
            + np.log(np.sum(np.exp(kernel - maximum), axis=1))
            - math.log(expected)
        )
    logits = np.stack(columns, axis=1)
    if not np.isfinite(logits).all():
        raise D110USQKNNError("Student-t qKNN logits became non-finite")
    return _readonly(logits, np.float32)


@dataclass(frozen=True, slots=True)
class D110USQKNNState:
    """One immutable arm state; no query values, labels, roles, or counts exist."""

    arm: str
    bank: qknn.TypedINT8ZIDSupportBank
    class_scales_fp16: np.ndarray
    scpm_state: scpm.D110SCPMRuntimeState | None
    query_rows_used_for_fit: int = 0
    query_state_updates: int = 0
    schema: str = SCHEMA

    def __post_init__(self) -> None:
        arm = _require_arm(self.arm)
        if type(self.bank) is not qknn.TypedINT8ZIDSupportBank:
            raise D110USQKNNError("US-qKNN requires an exact qKNN support bank")
        if (
            self.schema != SCHEMA
            or type(self.query_rows_used_for_fit) is not int
            or type(self.query_state_updates) is not int
            or self.query_rows_used_for_fit != 0
            or self.query_state_updates != 0
        ):
            raise D110USQKNNError("US-qKNN query lifecycle drift")
        scales = np.asarray(self.class_scales_fp16)
        if (
            scales.dtype != np.float16
            or scales.shape != (len(self.bank.classes),)
            or not np.isfinite(scales).all()
            or np.any(scales <= 0.0)
        ):
            raise D110USQKNNError("US-qKNN class-scale closure drift")
        if arm in _DA_ARMS:
            if type(self.scpm_state) is not scpm.D110SCPMRuntimeState:
                raise D110USQKNNError("D110 adaptation arms require an exact SCPM state")
            if (
                self.scpm_state.active_k != self.bank.active_k
                or self.scpm_state.query_rows_used_for_fit != 0
                or self.scpm_state.query_state_updates != 0
                or set(self.scpm_state.class_labels.astype(str).tolist())
                != set(self.bank.classes)
            ):
                raise D110USQKNNError("D110 SCPM/qKNN support state drift")
        elif self.scpm_state is not None:
            raise D110USQKNNError("identity arms must not retain a D110 metric state")
        object.__setattr__(self, "arm", arm)
        object.__setattr__(self, "class_scales_fp16", _readonly(scales, np.float16))


def fit_d110_usqknn_four_arms(
    support_zid: np.ndarray,
    support_labels: Sequence[str],
    registered_classes: Sequence[str],
    *,
    config: qknn.Phase1ZIDStudentTLock,
    closed_u: np.ndarray,
    prior_variances: np.ndarray,
) -> Mapping[str, D110USQKNNState]:
    """Fit the four frozen arms from the same INT8-decoded support bank.

    There is deliberately no query argument.  ``closed_u`` and
    ``prior_variances`` are externally sealed Phase1 inputs; this function
    only applies support-side SCPM closure and the already locked qKNN rules.
    """

    if type(config) is not qknn.Phase1ZIDStudentTLock:
        raise D110USQKNNError("US-qKNN requires an exact Phase1 qKNN lock")
    bank = qknn.build_typed_zid_support_bank(
        support_zid, support_labels, registered_classes, config=config
    )
    decoded_support = qknn.decode_zid_support_bank(bank)
    canonical_labels = np.asarray(
        [bank.classes[int(index)] for index in bank.class_indices_int16], dtype=str
    )
    scpm_state = scpm.fit_d110_scpm_runtime(
        decoded_support, canonical_labels, closed_u, prior_variances
    )
    if scpm_state.active_k != bank.active_k:
        raise D110USQKNNError("D110 SCPM active-K drift")
    support = decoded_support.astype(np.float64)
    identity_pairs = _identity_distances(support, support)
    scpm_pairs = _safe_relative_scpm_distances(support, support, scpm_state)

    states = {
        "M0": D110USQKNNState(
            arm="M0",
            bank=bank,
            class_scales_fp16=bank.class_scales_fp16,
            scpm_state=None,
        ),
        "M_DA": D110USQKNNState(
            arm="M_DA",
            bank=bank,
            class_scales_fp16=_da_class_specific_scales(scpm_pairs, bank),
            scpm_state=scpm_state,
        ),
        "M_HEAD": D110USQKNNState(
            arm="M_HEAD",
            bank=bank,
            class_scales_fp16=_shared_scales(identity_pairs, bank),
            scpm_state=None,
        ),
        "M_JOINT": D110USQKNNState(
            arm="M_JOINT",
            bank=bank,
            class_scales_fp16=_shared_scales(scpm_pairs, bank),
            scpm_state=scpm_state,
        ),
    }
    return MappingProxyType(states)


def fit_d110_usqknn_state(
    support_zid: np.ndarray,
    support_labels: Sequence[str],
    registered_classes: Sequence[str],
    *,
    arm: str,
    config: qknn.Phase1ZIDStudentTLock,
    closed_u: np.ndarray,
    prior_variances: np.ndarray,
) -> D110USQKNNState:
    """Fit one requested frozen arm; all four rules remain mechanically shared."""

    return fit_d110_usqknn_four_arms(
        support_zid,
        support_labels,
        registered_classes,
        config=config,
        closed_u=closed_u,
        prior_variances=prior_variances,
    )[_require_arm(arm)]


def _checked_state(state: object) -> D110USQKNNState:
    if type(state) is not D110USQKNNState:
        raise D110USQKNNError("US-qKNN scoring requires an exact D110USQKNNState")
    # Re-enter the dataclass invariant checker without replacing the state.
    D110USQKNNState(
        arm=state.arm,
        bank=state.bank,
        class_scales_fp16=state.class_scales_fp16,
        scpm_state=state.scpm_state,
        query_rows_used_for_fit=state.query_rows_used_for_fit,
        query_state_updates=state.query_state_updates,
        schema=state.schema,
    )
    return state


def score_d110_usqknn_logits(
    state: D110USQKNNState, query_zid: np.ndarray
) -> np.ndarray:
    """Score each query independently over all registered classes.

    The M0 branch calls the established qKNN scorer directly, preserving its
    bit-level identity baseline.  The other branches alter only distance and/or
    closed bandwidths while retaining the identical Student-t likelihood.
    """

    checked = _checked_state(state)
    if checked.arm == "M0":
        return qknn.score_zid_student_t_logits(
            checked.bank,
            query_zid,
            metric=qknn.identity_shared_psd_metric(config=checked.bank.config),
        )
    query = qknn.normalize_zid_rows(query_zid).astype(np.float64)
    support = qknn.decode_zid_support_bank(checked.bank).astype(np.float64)
    if checked.arm in _DA_ARMS:
        assert checked.scpm_state is not None
        distances = _safe_relative_scpm_distances(query, support, checked.scpm_state)
    else:
        distances = _identity_distances(query, support)
    return _score_from_distances(distances, checked.bank, checked.class_scales_fp16)


def predict_d110_usqknn(
    state: D110USQKNNState, query_zid: np.ndarray
) -> tuple[str, ...]:
    """Return a fail-closed all-class prediction for each independent query."""

    logits = score_d110_usqknn_logits(state, query_zid)
    labels: list[str] = []
    for row in logits.astype(np.float64):
        maximum = float(np.max(row))
        winners = np.flatnonzero(row == maximum)
        if len(winners) != 1:
            raise D110USQKNNError("cross-class qKNN tie is fail-closed")
        labels.append(checked_label := state.bank.classes[int(winners[0])])
        if not checked_label:
            raise D110USQKNNError("empty registered class label")
    return tuple(labels)


def audit_d110_usqknn_state(state: D110USQKNNState) -> dict[str, int | bool | str]:
    """Return a small state-only resource/lifecycle receipt without scoring data."""

    checked = _checked_state(state)
    bank_arrays = (
        checked.bank.codes_qint8,
        checked.bank.scales_fp16,
        checked.bank.class_indices_int16,
        checked.bank.class_scales_fp16,
        checked.class_scales_fp16,
    )
    metric_arrays: tuple[np.ndarray, ...] = ()
    if checked.scpm_state is not None:
        metric_arrays = (
            checked.scpm_state.centers,
            checked.scpm_state.closed_u,
            checked.scpm_state.prior_variances,
            checked.scpm_state.variances,
            checked.scpm_state.safe_relative_variances,
        )
        if checked.scpm_state.target_variances is not None:
            metric_arrays += (checked.scpm_state.target_variances,)
    return {
        "schema": SCHEMA + ".state_audit.v1",
        "arm": checked.arm,
        "active_k": checked.bank.active_k,
        "support_bank_numeric_bytes": sum(int(value.nbytes) for value in bank_arrays),
        "metric_numeric_bytes": sum(int(value.nbytes) for value in metric_arrays),
        "query_rows_used_for_fit": checked.query_rows_used_for_fit,
        "query_state_updates": checked.query_state_updates,
        "parameter_scan_count": 0,
        "predictive_variance_used": False,
        "all_class_scoring": True,
    }


__all__ = [
    "ARMS",
    "D110USQKNNError",
    "D110USQKNNState",
    "SCHEMA",
    "audit_d110_usqknn_state",
    "fit_d110_usqknn_four_arms",
    "fit_d110_usqknn_state",
    "predict_d110_usqknn",
    "score_d110_usqknn_logits",
]
