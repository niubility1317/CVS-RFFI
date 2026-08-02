"""Query-independent D112 SEAM-qKNN over the frozen M0 Student-t head.

The module deliberately has only two mutable-time inputs: a Phase1-sealed
``D112Bundle`` and the labelled support bank for one Phase2 row.  Queries are
never used while constructing :class:`D112SEAMState`; scoring is per-row,
read-only and always begins from the frozen M0 qKNN logit matrix.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from cvsrffi.stage2_d112_seam_bundle import D112Bundle, FEATURE_DIM
from cvsrffi.stage2_zid_student_t_qknn import (
    TypedINT8ZIDSupportBank,
    decode_zid_support_bank,
    identity_shared_psd_metric,
    normalize_zid_rows,
    score_zid_student_t_logits,
)


SCHEMA = "cvs.stage2.d112.seam_qknn.score_state.v1"
EXPECTED_OLD_CLASS_COUNT = 6
SHARED_RANK = 3
EPSILON_GEO = 64.0 * float(np.finfo(np.float32).eps)
EPSILON_VARIANCE_R = EPSILON_GEO**2 / SHARED_RANK
EPSILON_VARIANCE_AMB = EPSILON_GEO**2 / FEATURE_DIM
G0_COMPONENT_STATE = "NONFORMAL_G0_FUNCTIONAL_ONLY"


class D112SEAMError(ValueError):
    """Raised when D112 cannot retain its frozen support-only semantics."""


def _readonly(value: np.ndarray, dtype: Any) -> np.ndarray:
    array = np.ascontiguousarray(value, dtype=dtype)
    frozen = np.frombuffer(array.tobytes(order="C"), dtype=array.dtype).reshape(array.shape)
    frozen.setflags(write=False)
    return frozen


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _array_receipt(value: np.ndarray) -> dict[str, Any]:
    array = np.ascontiguousarray(value)
    return {
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
    }


def _unit_vector(value: np.ndarray, field: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (FEATURE_DIM,) or not np.isfinite(vector).all():
        raise D112SEAMError(f"{field} must be a finite [{FEATURE_DIM}] vector")
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm <= EPSILON_GEO:
        raise D112SEAMError(f"{field} is geometrically degenerate")
    return vector / norm


def _tangent(value: np.ndarray, base: np.ndarray) -> np.ndarray:
    """Project a numerically near-tangent vector onto ``T_base S``."""

    vector = np.asarray(value, dtype=np.float64)
    return vector - float(np.dot(vector, base)) * base


def sphere_log(base: np.ndarray, point: np.ndarray) -> np.ndarray:
    """Stable spherical Log using atan2, with only the fixed chart boundary."""

    x = _unit_vector(base, "sphere Log base")
    y = _unit_vector(point, "sphere Log point")
    cosine = float(np.clip(np.dot(x, y), -1.0, 1.0))
    if 1.0 + cosine <= EPSILON_GEO:
        raise D112SEAMError("sphere Log antipodal chart is undefined")
    residual = y - cosine * x
    residual_norm = float(np.linalg.norm(residual))
    theta = math.atan2(residual_norm, cosine)
    if residual_norm <= EPSILON_GEO:
        # This is the continuous theta/sin(theta) limit at the identity.
        return np.zeros(FEATURE_DIM, dtype=np.float64)
    result = (theta / residual_norm) * residual
    if not np.isfinite(result).all():
        raise D112SEAMError("sphere Log produced non-finite coordinates")
    return _tangent(result, x)


def sphere_parallel_transport(
    base: np.ndarray, destination: np.ndarray, tangent: np.ndarray
) -> np.ndarray:
    """Parallel transport along the unique shortest great-circle segment."""

    x = _unit_vector(base, "parallel-transport base")
    y = _unit_vector(destination, "parallel-transport destination")
    vector = _tangent(np.asarray(tangent, dtype=np.float64), x)
    if vector.shape != (FEATURE_DIM,) or not np.isfinite(vector).all():
        raise D112SEAMError("parallel-transport tangent must be finite [160]")
    denominator = 1.0 + float(np.clip(np.dot(x, y), -1.0, 1.0))
    if denominator <= EPSILON_GEO:
        raise D112SEAMError("parallel-transport antipodal chart is undefined")
    result = vector - (float(np.dot(vector, y)) / denominator) * (x + y)
    result = _tangent(result, y)
    if not np.isfinite(result).all():
        raise D112SEAMError("parallel transport produced non-finite coordinates")
    return result


def sphere_exp(base: np.ndarray, tangent: np.ndarray) -> np.ndarray:
    """Stable spherical Exp with its zero-vector continuous limit."""

    x = _unit_vector(base, "sphere Exp base")
    vector = _tangent(np.asarray(tangent, dtype=np.float64), x)
    if vector.shape != (FEATURE_DIM,) or not np.isfinite(vector).all():
        raise D112SEAMError("sphere Exp tangent must be finite [160]")
    length = float(np.linalg.norm(vector))
    if not math.isfinite(length):
        raise D112SEAMError("sphere Exp tangent norm is non-finite")
    if length <= EPSILON_GEO:
        return x.copy()
    result = math.cos(length) * x + (math.sin(length) / length) * vector
    return _unit_vector(result, "sphere Exp result")


def radial_pi_compress(tangent: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Return ``R_pi(u)``, ``||u||`` and its radial scale without a hard angle gate."""

    vector = np.asarray(tangent, dtype=np.float64)
    if vector.shape != (FEATURE_DIM,) or not np.isfinite(vector).all():
        raise D112SEAMError("R_pi tangent must be finite [160]")
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm):
        raise D112SEAMError("R_pi tangent norm is non-finite")
    scale = math.pi / math.sqrt(math.pi * math.pi + norm * norm)
    result = scale * vector
    if not np.isfinite(result).all() or float(np.linalg.norm(result)) >= math.pi:
        raise D112SEAMError("R_pi failed its open-ball invariant")
    return result, norm, scale


def seam_jacobian_trace(*, alpha: float, uncompressed_norm: float) -> float:
    """Closed-form ``tr(J J^T)`` for ``Exp_g o R_pi o alpha PT U``."""

    local_alpha = float(alpha)
    norm = float(uncompressed_norm)
    if (
        not math.isfinite(local_alpha)
        or not math.isfinite(norm)
        or local_alpha < 0.0
        or norm < 0.0
    ):
        raise D112SEAMError("SEAM Jacobian arguments are invalid")
    denominator = math.pi * math.pi + norm * norm
    beta = math.pi / math.sqrt(denominator)
    compressed_norm = beta * norm
    kappa_radial = math.pi**3 / denominator ** 1.5
    if compressed_norm <= EPSILON_GEO:
        sine_ratio = 1.0
    else:
        sine_ratio = math.sin(compressed_norm) / compressed_norm
    kappa_tangent = beta * sine_ratio
    trace = local_alpha * local_alpha * (
        kappa_radial * kappa_radial
        + (SHARED_RANK - 1) * kappa_tangent * kappa_tangent
    )
    if not math.isfinite(trace) or trace < 0.0:
        raise D112SEAMError("SEAM Jacobian trace is invalid")
    return trace


def _global_geometry_valid(bundle: D112Bundle) -> bool:
    """Check only fixed bundle geometry; any failure is the specified all-M0 fallback."""

    try:
        q0 = _unit_vector(bundle.q0, "D112 q0")
        g = np.asarray(bundle.g, dtype=np.float64)
        basis = np.asarray(bundle.U, dtype=np.float64)
        if (
            g.shape != (EXPECTED_OLD_CLASS_COUNT, FEATURE_DIM)
            or basis.shape != (SHARED_RANK, FEATURE_DIM)
            or not np.isfinite(g).all()
            or not np.isfinite(basis).all()
            or not np.allclose(basis @ basis.T, np.eye(SHARED_RANK), atol=1.0e-4, rtol=0.0)
            or not np.all(np.abs(basis @ q0) <= 1.0e-4)
        ):
            return False
        for row in g:
            anchor = _unit_vector(row, "D112 ground anchor")
            if 1.0 + float(np.dot(q0, anchor)) <= EPSILON_GEO:
                return False
        for name in ("sigma0_r", "sigma0_amb", "v_g_r", "v_g_amb"):
            value = np.asarray(getattr(bundle, name), dtype=np.float64)
            if value.shape != (EXPECTED_OLD_CLASS_COUNT,) or not np.isfinite(value).all() or np.any(value <= 0.0):
                return False
        if not math.isfinite(float(bundle.tau_h_r)) or float(bundle.tau_h_r) <= 0.0:
            return False
    except (AttributeError, D112SEAMError, TypeError, ValueError):
        return False
    return True


@dataclass(frozen=True, slots=True)
class D112SEAMState:
    """Immutable support-only state for one independent Phase2 row."""

    classes: tuple[str, ...]
    old_class_indices: tuple[int, ...]
    anchors: np.ndarray
    rho: np.ndarray
    information_valid: np.ndarray
    donor_valid: np.ndarray
    alpha: np.ndarray
    v_s_r: np.ndarray
    v_s_amb: np.ndarray
    loo_variance_r: np.ndarray
    loo_disagreement_r: np.ndarray
    jacobian_trace: np.ndarray
    v_h_amb: np.ndarray
    discrepancy_amb: np.ndarray
    anchor_shift_l2: np.ndarray
    global_bundle_valid: bool
    bank_receipt_sha256: str
    config_lock_digest: str
    bundle_content_root_sha256: str
    resource_receipt: Mapping[str, int]
    state_receipt_sha256: str
    schema: str = SCHEMA

    def __post_init__(self) -> None:
        count = len(self.classes)
        expected = (count,)
        if (
            self.schema != SCHEMA
            or len(self.old_class_indices) != EXPECTED_OLD_CLASS_COUNT
            or self.anchors.shape != (count, FEATURE_DIM)
            or any(
                array.shape != expected
                for array in (
                    self.rho,
                    self.information_valid,
                    self.donor_valid,
                    self.alpha,
                    self.v_s_r,
                    self.v_s_amb,
                    self.loo_variance_r,
                    self.loo_disagreement_r,
                    self.jacobian_trace,
                    self.v_h_amb,
                    self.discrepancy_amb,
                    self.anchor_shift_l2,
                )
            )
        ):
            raise D112SEAMError("D112 SEAM state shape/schema drift")
        if any(
            array.flags.writeable
            for array in (
                self.anchors,
                self.rho,
                self.information_valid,
                self.donor_valid,
                self.alpha,
                self.v_s_r,
                self.v_s_amb,
                self.loo_variance_r,
                self.loo_disagreement_r,
                self.jacobian_trace,
                self.v_h_amb,
                self.discrepancy_amb,
                self.anchor_shift_l2,
            )
        ):
            raise D112SEAMError("D112 SEAM state arrays must be deeply readonly")


def _state_payload(state: D112SEAMState) -> dict[str, Any]:
    arrays = (
        "anchors",
        "rho",
        "information_valid",
        "donor_valid",
        "alpha",
        "v_s_r",
        "v_s_amb",
        "loo_variance_r",
        "loo_disagreement_r",
        "jacobian_trace",
        "v_h_amb",
        "discrepancy_amb",
        "anchor_shift_l2",
    )
    return {
        "schema": state.schema,
        "classes": list(state.classes),
        "old_class_indices": list(state.old_class_indices),
        "global_bundle_valid": bool(state.global_bundle_valid),
        "bank_receipt_sha256": state.bank_receipt_sha256,
        "config_lock_digest": state.config_lock_digest,
        "bundle_content_root_sha256": state.bundle_content_root_sha256,
        "arrays": {name: _array_receipt(getattr(state, name)) for name in arrays},
        "resource_receipt": {str(key): int(value) for key, value in state.resource_receipt.items()},
        "query_rows_used_for_fit": 0,
        "truth_role_quota_inputs": 0,
    }


def _verify_state(state: D112SEAMState) -> None:
    if _canonical_sha256(_state_payload(state)) != state.state_receipt_sha256:
        raise D112SEAMError("D112 SEAM state receipt verification failed")
    old = np.asarray(state.old_class_indices, dtype=np.int64)
    if (
        len(set(state.classes)) != len(state.classes)
        or len(set(state.old_class_indices)) != EXPECTED_OLD_CLASS_COUNT
        or np.any(old < 0)
        or np.any(old >= len(state.classes))
        or not np.isfinite(state.anchors).all()
        or not np.isfinite(state.rho).all()
        or np.any(state.rho < 0.0)
        or np.any(state.rho >= 1.0)
        or not np.isfinite(state.alpha).all()
        or np.any(state.alpha < 0.0)
        or np.any(state.alpha >= 1.0)
        or any(
            not np.isfinite(array).all() or np.any(array < 0.0)
            for array in (
                state.v_s_r,
                state.v_s_amb,
                state.loo_variance_r,
                state.loo_disagreement_r,
                state.jacobian_trace,
                state.v_h_amb,
                state.discrepancy_amb,
                state.anchor_shift_l2,
            )
        )
    ):
        raise D112SEAMError("D112 SEAM state numeric invariant drift")
    old_mask = np.zeros(len(state.classes), dtype=bool)
    old_mask[old] = True
    if np.any(state.information_valid & ~old_mask) or np.any(state.donor_valid & ~old_mask):
        raise D112SEAMError("D112 SEAM non-old class state drift")
    if np.any((state.rho > 0.0) != state.information_valid):
        raise D112SEAMError("D112 SEAM unit-mass activation drift")
    if np.any(~state.information_valid & (state.alpha != 0.0)):
        raise D112SEAMError("D112 SEAM inactive class shrinkage drift")
    if not state.global_bundle_valid and (
        np.any(state.information_valid) or np.any(state.donor_valid) or np.any(state.rho > 0.0)
    ):
        raise D112SEAMError("global-invalid D112 bundle must be exact M0")
    active = np.flatnonzero(state.information_valid)
    if len(active) and not np.allclose(
        np.linalg.norm(state.anchors[active].astype(np.float64), axis=1),
        1.0,
        atol=1.0e-5,
        rtol=0.0,
    ):
        raise D112SEAMError("active D112 anchors must remain on the unit sphere")


@dataclass(frozen=True, slots=True)
class _SupportObservation:
    prototype: np.ndarray
    coordinate: np.ndarray | None
    v_s_r: float
    v_s_amb: float


def _support_observation(
    *,
    support: np.ndarray,
    ground: np.ndarray,
    q0: np.ndarray,
    basis: np.ndarray,
    sigma0_r: float,
    sigma0_amb: float,
    active_k: int,
) -> _SupportObservation | None:
    """One class's legal support-only observation, or a per-class M0 fallback."""

    try:
        local = np.asarray(support, dtype=np.float64)
        if local.shape != (active_k, FEATURE_DIM) or not np.isfinite(local).all():
            return None
        total = np.sum(local, axis=0)
        total_norm = float(np.linalg.norm(total))
        if not math.isfinite(total_norm) or total_norm <= EPSILON_GEO:
            return None
        prototype = total / total_norm
        if active_k == 1:
            hat_sigma_amb = 0.0
        else:
            hat_sigma_amb = float(
                np.sum(np.square(local - prototype[None, :]))
                / ((active_k - 1) * FEATURE_DIM)
            )
        v_s_amb = (float(sigma0_amb) + hat_sigma_amb) / active_k
        if not math.isfinite(v_s_amb) or v_s_amb <= 0.0:
            return None
        try:
            coordinate = basis @ sphere_parallel_transport(
                ground, q0, sphere_log(ground, prototype)
            )
        except D112SEAMError:
            # The support prototype and ambient reliability remain valid for
            # this class as a recipient, but it cannot donate a Log coordinate.
            return _SupportObservation(prototype, None, 0.0, v_s_amb)
        if active_k == 1:
            hat_sigma_r = 0.0
        else:
            try:
                coordinate_rows = np.asarray(
                    [
                        basis
                        @ sphere_parallel_transport(
                            prototype, q0, sphere_log(prototype, row)
                        )
                        for row in local
                    ],
                    dtype=np.float64,
                )
                # Fixed, per-coordinate dispersion proxy around the normalized
                # support prototype; it is not claimed to be unbiased.
                hat_sigma_r = float(
                    np.sum(np.square(coordinate_rows))
                    / ((active_k - 1) * SHARED_RANK)
                )
            except D112SEAMError:
                return _SupportObservation(prototype, None, 0.0, v_s_amb)
        v_s_r = (float(sigma0_r) + hat_sigma_r) / active_k
        if (
            not np.isfinite(coordinate).all()
            or not math.isfinite(v_s_r)
            or v_s_r <= 0.0
        ):
            return _SupportObservation(prototype, None, 0.0, v_s_amb)
        return _SupportObservation(prototype, coordinate, v_s_r, v_s_amb)
    except D112SEAMError:
        return None


def _require_fit_surface(bundle: D112Bundle, *, formal: bool) -> bool:
    """Return global asset validity after checking the selected immutable surface."""

    manifest = bundle.manifest
    if not isinstance(manifest, Mapping):
        raise D112SEAMError("D112 manifest must be a mapping")
    if formal:
        raise D112SEAMError(
            "formal D112 surface is not implemented; G0 bundles cannot enter it"
        )
    if (
        manifest.get("component_state") != G0_COMPONENT_STATE
        or manifest.get("formal_phase2_eligible") is not False
        or manifest.get("performance_claim_allowed") is not False
        or manifest.get("performance_metrics_allowed") is not False
        or manifest.get("target_access_allowed") is not False
        or manifest.get("query_rows_used_for_fit") != 0
    ):
        raise D112SEAMError("D112 G0 fit requires exact functional-only markers")
    global_valid = manifest.get("global_bundle_valid") is True and _global_geometry_valid(bundle)
    # A sealed bundle that reports fixed-asset invalidity is specified to resolve
    # to M0, not to invent a replacement asset from target support.
    if not global_valid:
        return False
    return True


def _fit_d112_seam_state(
    bundle: D112Bundle,
    bank: TypedINT8ZIDSupportBank,
    *,
    formal: bool,
) -> D112SEAMState:
    """Build one immutable state from Phase1 assets and current labelled support only."""

    if type(bundle) is not D112Bundle or type(bank) is not TypedINT8ZIDSupportBank:
        raise D112SEAMError("D112 fit requires exact bundle and qKNN bank types")
    if float(bank.config.kernel_volume_gamma) != 1.0:
        raise D112SEAMError("D112 unit-mass density requires kernel_volume_gamma=1")
    global_valid = _require_fit_surface(bundle, formal=formal)
    old_classes = tuple(bundle.class_registry)
    if len(old_classes) != EXPECTED_OLD_CLASS_COUNT:
        raise D112SEAMError("D112 is frozen for exactly six Phase1 old classes")
    if any(name not in bank.classes for name in old_classes):
        raise D112SEAMError("D112 old registry is not contained in the support bank")
    old_indices = tuple(bank.classes.index(name) for name in old_classes)
    class_count = len(bank.classes)
    ground = np.asarray(bundle.g, dtype=np.float64)
    q0 = _unit_vector(bundle.q0, "D112 q0")
    basis = np.asarray(bundle.U, dtype=np.float64)
    sigma0_r = np.asarray(bundle.sigma0_r, dtype=np.float64)
    sigma0_amb = np.asarray(bundle.sigma0_amb, dtype=np.float64)
    v_g_r = np.asarray(bundle.v_g_r, dtype=np.float64)
    v_g_amb = np.asarray(bundle.v_g_amb, dtype=np.float64)

    anchors = np.zeros((class_count, FEATURE_DIM), dtype=np.float64)
    for old_position, bank_index in enumerate(old_indices):
        anchors[bank_index] = _unit_vector(ground[old_position], "D112 ground anchor")
    rho = np.zeros(class_count, dtype=np.float64)
    information_valid = np.zeros(class_count, dtype=np.bool_)
    donor_valid = np.zeros(class_count, dtype=np.bool_)
    alpha = np.zeros(class_count, dtype=np.float64)
    v_s_r = np.zeros(class_count, dtype=np.float64)
    v_s_amb = np.zeros(class_count, dtype=np.float64)
    loo_variance_r = np.zeros(class_count, dtype=np.float64)
    loo_disagreement_r = np.zeros(class_count, dtype=np.float64)
    jacobian_trace = np.zeros(class_count, dtype=np.float64)
    v_h_amb = np.zeros(class_count, dtype=np.float64)
    discrepancy_amb = np.zeros(class_count, dtype=np.float64)
    anchor_shift_l2 = np.zeros(class_count, dtype=np.float64)

    observations: list[_SupportObservation | None] = [None] * EXPECTED_OLD_CLASS_COUNT
    if global_valid:
        decoded = normalize_zid_rows(decode_zid_support_bank(bank).astype(np.float32)).astype(
            np.float64
        )
        for old_position, bank_index in enumerate(old_indices):
            local = decoded[bank.class_indices_int16 == bank_index]
            observation = _support_observation(
                support=local,
                ground=anchors[bank_index],
                q0=q0,
                basis=basis,
                sigma0_r=float(sigma0_r[old_position]),
                sigma0_amb=float(sigma0_amb[old_position]),
                active_k=bank.active_k,
            )
            observations[old_position] = observation
            if observation is not None:
                v_s_amb[bank_index] = observation.v_s_amb
                if observation.coordinate is not None:
                    donor_valid[bank_index] = True
                    v_s_r[bank_index] = observation.v_s_r

        tau_h_r = float(bundle.tau_h_r)
        for old_position, bank_index in enumerate(old_indices):
            observation = observations[old_position]
            if observation is None:
                continue
            donor_positions = [
                donor
                for donor in range(EXPECTED_OLD_CLASS_COUNT)
                if donor != old_position
                and observations[donor] is not None
                and observations[donor].coordinate is not None
            ]
            if len(donor_positions) < 2:
                continue
            weights = np.asarray(
                [
                    1.0
                    / (
                        observations[donor].v_s_r  # type: ignore[union-attr]
                        + float(v_g_r[donor])
                    )
                    for donor in donor_positions
                ],
                dtype=np.float64,
            )
            weight_sum = float(np.sum(weights))
            if not np.isfinite(weights).all() or weight_sum <= EPSILON_GEO:
                continue
            normalized_weights = weights / weight_sum
            effective_denominator = 1.0 - float(np.sum(np.square(normalized_weights)))
            if not math.isfinite(effective_denominator) or effective_denominator <= EPSILON_GEO:
                continue
            coordinates = np.asarray(
                [observations[donor].coordinate for donor in donor_positions],
                dtype=np.float64,
            )
            h_loo = np.sum(normalized_weights[:, None] * coordinates, axis=0)
            local_variance = 1.0 / weight_sum
            local_disagreement = float(
                np.sum(
                    normalized_weights
                    * np.sum(np.square(coordinates - h_loo[None, :]), axis=1)
                )
                / (SHARED_RANK * effective_denominator)
            )
            shrinkage_denominator = tau_h_r + local_variance + local_disagreement
            if (
                not np.isfinite(h_loo).all()
                or not all(
                    math.isfinite(value) and value >= 0.0
                    for value in (local_variance, local_disagreement)
                )
                or not math.isfinite(shrinkage_denominator)
                or shrinkage_denominator <= 0.0
            ):
                continue
            local_alpha = tau_h_r / shrinkage_denominator
            try:
                transported = sphere_parallel_transport(
                    q0, anchors[bank_index], basis.T @ h_loo
                )
                uncompressed = local_alpha * transported
                compressed, uncompressed_norm, _ = radial_pi_compress(uncompressed)
                anchor = sphere_exp(anchors[bank_index], compressed)
                local_trace = seam_jacobian_trace(
                    alpha=local_alpha, uncompressed_norm=uncompressed_norm
                )
            except D112SEAMError:
                # The row/class has no well-defined chart, so exactly that M0
                # column remains untouched.  No alternate target-derived chart
                # or analytic M0 replacement is introduced.
                continue
            local_v_h = (local_variance + local_disagreement) * local_trace / FEATURE_DIM
            local_discrepancy = float(
                np.sum(np.square(observation.prototype - anchor)) / FEATURE_DIM
            )
            mixture_denominator = (
                observation.v_s_amb
                + float(v_g_amb[old_position])
                + local_v_h
                + local_discrepancy
            )
            if (
                not all(
                    math.isfinite(value) and value >= 0.0
                    for value in (local_alpha, local_v_h, local_discrepancy)
                )
                or not math.isfinite(mixture_denominator)
                or mixture_denominator <= 0.0
            ):
                continue
            local_rho = observation.v_s_amb / mixture_denominator
            if not math.isfinite(local_rho) or not 0.0 < local_rho < 1.0:
                continue
            anchors[bank_index] = anchor
            rho[bank_index] = local_rho
            information_valid[bank_index] = True
            alpha[bank_index] = local_alpha
            loo_variance_r[bank_index] = local_variance
            loo_disagreement_r[bank_index] = local_disagreement
            jacobian_trace[bank_index] = local_trace
            v_h_amb[bank_index] = local_v_h
            discrepancy_amb[bank_index] = local_discrepancy
            anchor_shift_l2[bank_index] = float(
                np.linalg.norm(anchor - _unit_vector(ground[old_position], "D112 ground anchor"))
            )

    arrays = {
        "anchors": _readonly(anchors, np.float32),
        "rho": _readonly(rho, np.float32),
        "information_valid": _readonly(information_valid, np.bool_),
        "donor_valid": _readonly(donor_valid, np.bool_),
        "alpha": _readonly(alpha, np.float64),
        "v_s_r": _readonly(v_s_r, np.float64),
        "v_s_amb": _readonly(v_s_amb, np.float64),
        "loo_variance_r": _readonly(loo_variance_r, np.float64),
        "loo_disagreement_r": _readonly(loo_disagreement_r, np.float64),
        "jacobian_trace": _readonly(jacobian_trace, np.float64),
        "v_h_amb": _readonly(v_h_amb, np.float64),
        "discrepancy_amb": _readonly(discrepancy_amb, np.float64),
        "anchor_shift_l2": _readonly(anchor_shift_l2, np.float64),
    }
    resource = {
        "persistent_numeric_bytes": int(sum(array.nbytes for array in arrays.values())),
        "enrollment_projection_macs": EXPECTED_OLD_CLASS_COUNT * SHARED_RANK * FEATURE_DIM,
        "enrollment_pairwise_loo_scalar_terms": EXPECTED_OLD_CLASS_COUNT
        * (EXPECTED_OLD_CLASS_COUNT - 1)
        * SHARED_RANK,
        "extra_query_macs_per_row_upper_bound": EXPECTED_OLD_CLASS_COUNT * FEATURE_DIM,
        "query_dependent_state_bytes": 0,
    }
    state = D112SEAMState(
        classes=tuple(bank.classes),
        old_class_indices=old_indices,
        global_bundle_valid=bool(global_valid),
        bank_receipt_sha256=bank.bank_receipt_sha256,
        config_lock_digest=bank.config_lock_digest,
        bundle_content_root_sha256=str(bundle.manifest.get("content_root_sha256", "")),
        resource_receipt=MappingProxyType(resource),
        state_receipt_sha256="0" * 64,
        **arrays,
    )
    object.__setattr__(state, "state_receipt_sha256", _canonical_sha256(_state_payload(state)))
    _verify_state(state)
    return state


def fit_d112_seam_state(
    bundle: D112Bundle, bank: TypedINT8ZIDSupportBank
) -> D112SEAMState:
    """Fit the formal support-only D112 state; no query or truth API exists."""

    return _fit_d112_seam_state(bundle, bank, formal=True)


def fit_d112_seam_g0_state(
    bundle: D112Bundle, bank: TypedINT8ZIDSupportBank
) -> D112SEAMState:
    """Fit the nonformal source-only G0 functional state without a target surface."""

    return _fit_d112_seam_state(bundle, bank, formal=False)


def score_d112_seam_logits(
    state: D112SEAMState,
    bank: TypedINT8ZIDSupportBank,
    query_zid: np.ndarray,
) -> np.ndarray:
    """Score independent query rows over all registered classes with unit mass."""

    if type(state) is not D112SEAMState or type(bank) is not TypedINT8ZIDSupportBank:
        raise D112SEAMError("D112 score requires exact state and bank types")
    _verify_state(state)
    if (
        state.classes != tuple(bank.classes)
        or state.bank_receipt_sha256 != bank.bank_receipt_sha256
        or state.config_lock_digest != bank.config_lock_digest
    ):
        raise D112SEAMError("D112 state/support-bank binding drift")
    metric = identity_shared_psd_metric(config=bank.config)
    baseline = score_zid_student_t_logits(bank, query_zid, metric=metric)
    active = np.flatnonzero(state.information_valid)
    if len(active) == 0:
        return baseline
    query = normalize_zid_rows(query_zid).astype(np.float64)
    output = np.asarray(baseline, dtype=np.float64).copy()
    dimension = bank.config.kernel_effective_dim
    nu = float(bank.config.student_nu)
    for class_index in active:
        local_rho = float(state.rho[class_index])
        anchor = np.asarray(state.anchors[class_index], dtype=np.float64)
        cosine = np.clip(query @ anchor, -1.0, 1.0)
        distance = np.maximum(2.0 * (1.0 - cosine), 0.0)
        h = float(bank.class_scales_fp16[class_index])
        # kernel_volume_gamma=1 was checked at fit: this is therefore the
        # identical M0 Student-t kernel and global logit origin, not an old bias.
        anchor_kernel = (
            -dimension * math.log(h)
            - 0.5 * (nu + dimension) * np.log1p(distance / (nu * h * h))
        )
        output[:, class_index] = np.logaddexp(
            math.log1p(-local_rho) + output[:, class_index],
            math.log(local_rho) + anchor_kernel,
        )
    if not np.isfinite(output).all():
        raise D112SEAMError("D112 SEAM logits became non-finite")
    return _readonly(output, np.float32)


def predict_d112_seam(
    state: D112SEAMState,
    bank: TypedINT8ZIDSupportBank,
    query_zid: np.ndarray,
) -> tuple[str, ...]:
    logits = score_d112_seam_logits(state, bank, query_zid)
    return tuple(state.classes[index] for index in np.argmax(logits, axis=1))


def audit_d112_seam_state(state: D112SEAMState) -> Mapping[str, Any]:
    """Return function/resource receipts only; no labels or performance metrics."""

    if type(state) is not D112SEAMState:
        raise D112SEAMError("D112 audit requires an exact state")
    _verify_state(state)
    old = np.asarray(state.old_class_indices, dtype=np.int64)
    return MappingProxyType(
        {
            "schema": state.schema,
            "state_receipt_sha256": state.state_receipt_sha256,
            "global_bundle_valid": bool(state.global_bundle_valid),
            "old_class_count": len(state.old_class_indices),
            "donor_valid_old_count": int(np.sum(state.donor_valid[old])),
            "information_valid_old_count": int(np.sum(state.information_valid[old])),
            "positive_rho_count": int(np.sum(state.rho > 0.0)),
            "max_rho": float(np.max(state.rho)),
            "max_alpha": float(np.max(state.alpha)),
            "max_anchor_shift_l2": float(np.max(state.anchor_shift_l2)),
            "max_jacobian_trace": float(np.max(state.jacobian_trace)),
            "resource_receipt": state.resource_receipt,
            "query_rows_used_for_fit": 0,
            "truth_role_quota_inputs": 0,
        }
    )


__all__ = [
    "D112SEAMError",
    "D112SEAMState",
    "EPSILON_GEO",
    "EPSILON_VARIANCE_AMB",
    "EPSILON_VARIANCE_R",
    "EXPECTED_OLD_CLASS_COUNT",
    "SCHEMA",
    "SHARED_RANK",
    "audit_d112_seam_state",
    "fit_d112_seam_g0_state",
    "fit_d112_seam_state",
    "predict_d112_seam",
    "radial_pi_compress",
    "seam_jacobian_trace",
    "score_d112_seam_logits",
    "sphere_exp",
    "sphere_log",
    "sphere_parallel_transport",
]
