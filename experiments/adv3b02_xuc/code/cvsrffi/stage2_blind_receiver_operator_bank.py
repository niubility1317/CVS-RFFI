"""D10 blind receiver-side operator bank over one sealed received IQ.

The three registered views are deterministic post-reception computations
from the same physical LEO_weak observation.  D10 reuses the D9 sparse
selection/registration engine after validating and mapping the real D10
operator provenance into the engine's fixed three slots.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, replace
from typing import Any, Callable, Mapping, Sequence

import numpy as np

import cvsrffi.stage2_floor_sparse_operator_fusion as d9
from cvsrffi.stage2_class_conditional_iq_head import OperatorCalibration


EPS = 1.0e-8
BASE = "base"
WL_IQ_CIRCULARIZE = "wl_iq_circularize"
FFT_ENVELOPE_EQ = "fft_envelope_eq"
OPERATORS = (BASE, WL_IQ_CIRCULARIZE, FFT_ENVELOPE_EQ)
_SLOT_OPERATORS = tuple(d9.OPERATORS)
_TO_SLOT = dict(zip(OPERATORS, _SLOT_OPERATORS))
_FROM_SLOT = dict(zip(_SLOT_OPERATORS, OPERATORS))

WL_RHO_MAGNITUDE_CAP = 0.35
WL_OUTPUT_GAIN_CAP = 1.25
FFT_ENVELOPE_WINDOW = 9
FFT_ENVELOPE_SHRINK = 0.12
FFT_GAIN_MIN = 0.88
FFT_GAIN_MAX = 1.14


class BlindReceiverOperatorBankError(ValueError):
    """Raised when D10 operator, support, or deployment invariants drift."""


@dataclass(frozen=True)
class BlindReceiverOperatorBankState:
    """D10 public state wrapping the unchanged D9 sparse engine state."""

    inner: d9.FloorSparseOperatorFusionState
    received_iq_length: int
    strict_accuracy_gate: Mapping[str, Any]

    @property
    def schema(self) -> str:
        return "cvs.phase2.d10_blind_receiver_operator_bank.v1"

    @property
    def classes(self) -> tuple[str, ...]:
        return self.inner.classes

    @property
    def class_count(self) -> int:
        return self.inner.class_count

    @property
    def operator_indices(self) -> np.ndarray:
        return self.inner.operator_indices

    @property
    def weights(self) -> np.ndarray:
        return self.inner.weights

    @property
    def prototypes(self) -> np.ndarray:
        return self.inner.prototypes

    @property
    def calibrations(self) -> tuple[OperatorCalibration, ...]:
        return tuple(
            OperatorCalibration(
                operator_id=_FROM_SLOT[value.operator_id],
                center=value.center,
                scale=value.scale,
            )
            for value in self.inner.calibrations
        )

    @property
    def feature_dim(self) -> int:
        return self.inner.feature_dim

    @property
    def used_operators(self) -> tuple[str, ...]:
        return tuple(_FROM_SLOT[value] for value in self.inner.used_operators)

    @property
    def old_class_count(self) -> int:
        return self.inner.old_class_count

    @property
    def registration_generation(self) -> int:
        return self.inner.registration_generation

    @property
    def current_k(self) -> int:
        return self.inner.current_k

    @property
    def selection_lock_k(self) -> int:
        return self.inner.selection_lock_k

    @property
    def selection_lock_sha256(self) -> str:
        return self.inner.selection_lock_sha256

    @property
    def support_lineage(self) -> tuple[tuple[str, str, str], ...]:
        return self.inner.support_lineage

    @property
    def persistent_state_bytes(self) -> int:
        return self.inner.persistent_state_bytes

    @property
    def support_audit(self) -> Mapping[str, Any]:
        inner = self.inner.support_audit
        audit = {
            key: _translate_trace(value)
            for key, value in inner.items()
        }
        audit.update(
            {
                "schema": (
                    "cvs.phase2.d10_nested_k_rebuild_audit.v1"
                    if "selection" not in inner
                    else "cvs.phase2.d10_support_audit.v1"
                ),
                "d9_sparse_selection_engine_reused": True,
                "fixed_received_iq_operator_set": list(OPERATORS),
                "strict_accuracy_gate": dict(self.strict_accuracy_gate),
            }
        )
        return audit

    def resource_audit(self) -> dict[str, Any]:
        base = dict(self.inner.resource_audit())
        operator_macs = sum(
            _operator_macs(operator, self.received_iq_length)
            for operator in self.used_operators
        )
        base.update(
            {
                "schema": "cvs.phase2.d10_resource.v1",
                "candidate": "d10_blind_receiver_operator_bank",
                "d9_sparse_selection_engine_reused": True,
                "used_operators": list(self.used_operators),
                "used_operator_count": len(self.used_operators),
                "maximum_fixed_received_iq_views": 3,
                "backbone_forwards_per_query": len(self.used_operators),
                "fft_representation_branches": int(
                    FFT_ENVELOPE_EQ in self.used_operators
                ),
                "received_iq_length": self.received_iq_length,
                "estimated_post_reception_operator_macs_per_query": (
                    operator_macs
                ),
                "combined_estimated_head_and_operator_macs_per_query": (
                    int(base["combined_head_macs_per_query"])
                    + operator_macs
                ),
                "operator_cost_excludes_backbone_macs": True,
                "wl_rho_magnitude_cap": WL_RHO_MAGNITUDE_CAP,
                "wl_output_gain_cap": WL_OUTPUT_GAIN_CAP,
                "fft_envelope_window": FFT_ENVELOPE_WINDOW,
                "fft_envelope_shrink": FFT_ENVELOPE_SHRINK,
                "fft_gain_clip": [FFT_GAIN_MIN, FFT_GAIN_MAX],
                "cfo_estimation": False,
                "cfo_frequency_shift": False,
                "cfo_derotation": False,
                "fft_phase_preserved_binwise": True,
            }
        )
        return base


@dataclass(frozen=True)
class BlindReceiverOperatorPrediction:
    labels: tuple[str, ...]
    scores: np.ndarray
    operators_computed: tuple[str, ...]


SamplewiseSealedFeatureExtractor = d9.SamplewiseSealedFeatureExtractor
seal_samplewise_feature_extractor = d9.seal_samplewise_feature_extractor


def _validate_iq(received_iq: np.ndarray) -> np.ndarray:
    iq = np.asarray(received_iq)
    if (
        iq.dtype != np.float32
        or iq.ndim != 3
        or iq.shape[1] != 2
        or iq.shape[2] < 3
        or not np.isfinite(iq).all()
    ):
        raise BlindReceiverOperatorBankError(
            "D10 received IQ must be finite float32 [N,2,L], L>=3"
        )
    return iq


def _complex_rows(iq: np.ndarray) -> np.ndarray:
    return (
        iq[:, 0].astype(np.float64)
        + 1j * iq[:, 1].astype(np.float64)
    )


def _widely_linear_circularize(iq: np.ndarray) -> np.ndarray:
    rows = _complex_rows(iq)
    mean = np.mean(rows, axis=1, keepdims=True)
    centered = rows - mean
    power = np.mean(np.abs(centered) ** 2, axis=1, keepdims=True)
    pseudo = np.mean(centered**2, axis=1, keepdims=True)
    valid = power > EPS
    rho = np.zeros_like(pseudo)
    np.divide(pseudo, power, out=rho, where=valid)
    rho_magnitude = np.abs(rho)
    cap_scale = np.minimum(
        1.0, WL_RHO_MAGNITUDE_CAP / np.maximum(rho_magnitude, EPS)
    )
    capped_rho = rho * cap_scale
    discriminant = np.sqrt(
        np.maximum(1.0 - np.abs(capped_rho) ** 2, EPS)
    )
    beta = -capped_rho / (1.0 + discriminant)
    corrected = centered + beta * np.conjugate(centered)
    corrected_power = np.mean(
        np.abs(corrected) ** 2, axis=1, keepdims=True
    )
    gain = np.sqrt(
        np.divide(
            power,
            np.maximum(corrected_power, EPS),
            out=np.ones_like(power),
            where=valid,
        )
    )
    gain = np.clip(
        gain, 1.0 / WL_OUTPUT_GAIN_CAP, WL_OUTPUT_GAIN_CAP
    )
    output = mean + corrected * gain
    output = np.where(valid, output, rows)
    return np.ascontiguousarray(
        np.stack([output.real, output.imag], axis=1),
        dtype=np.float32,
    )


def _circular_moving_average(
    rows: np.ndarray, window: int
) -> np.ndarray:
    half = window // 2
    return sum(
        np.roll(rows, shift, axis=1)
        for shift in range(-half, half + 1)
    ) / float(window)


def _fft_envelope_equalize(iq: np.ndarray) -> np.ndarray:
    rows = _complex_rows(iq)
    length = rows.shape[1]
    window = min(FFT_ENVELOPE_WINDOW, length)
    if window % 2 == 0:
        window -= 1
    if window < 3:
        return np.ascontiguousarray(iq)
    spectrum = np.fft.fft(rows, axis=1)
    magnitude = np.abs(spectrum)
    log_magnitude = np.log(np.maximum(magnitude, EPS))
    envelope = _circular_moving_average(log_magnitude, window)
    target = np.mean(envelope, axis=1, keepdims=True)
    log_gain = FFT_ENVELOPE_SHRINK * (target - envelope)
    gain = np.clip(np.exp(log_gain), FFT_GAIN_MIN, FFT_GAIN_MAX)
    adjusted = spectrum * gain
    output = np.fft.ifft(adjusted, axis=1)
    input_power = np.mean(np.abs(rows) ** 2, axis=1, keepdims=True)
    output_power = np.mean(np.abs(output) ** 2, axis=1, keepdims=True)
    rms_gain = np.sqrt(
        np.divide(
            input_power,
            np.maximum(output_power, EPS),
            out=np.ones_like(input_power),
            where=input_power > EPS,
        )
    )
    output *= np.clip(
        rms_gain, 1.0 / WL_OUTPUT_GAIN_CAP, WL_OUTPUT_GAIN_CAP
    )
    output = np.where(input_power > EPS, output, rows)
    return np.ascontiguousarray(
        np.stack([output.real, output.imag], axis=1),
        dtype=np.float32,
    )


def apply_received_iq_operator(
    received_iq: np.ndarray, operator_id: str
) -> np.ndarray:
    """Apply one preregistered blind operator independently per IQ row."""

    iq = _validate_iq(received_iq)
    operator = str(operator_id)
    if operator not in OPERATORS:
        raise BlindReceiverOperatorBankError(
            "unsupported D10 received-IQ operator"
        )
    if operator == BASE:
        return np.ascontiguousarray(iq)
    if operator == WL_IQ_CIRCULARIZE:
        return _widely_linear_circularize(iq)
    return _fft_envelope_equalize(iq)


def extract_operator_features(
    received_iq: np.ndarray,
    *,
    feature_extractor: Callable[[np.ndarray], np.ndarray],
    operator_ids: Sequence[str] = OPERATORS,
) -> dict[str, np.ndarray]:
    iq = _validate_iq(received_iq)
    operators = tuple(dict.fromkeys(str(value) for value in operator_ids))
    if not operators or any(value not in OPERATORS for value in operators):
        raise BlindReceiverOperatorBankError(
            "D10 operator feature set drift"
        )
    result: dict[str, np.ndarray] = {}
    dimension: int | None = None
    for operator in operators:
        view = apply_received_iq_operator(iq, operator)
        features = np.asarray(feature_extractor(view), dtype=np.float32)
        if (
            features.ndim != 2
            or len(features) != len(iq)
            or not np.isfinite(features).all()
        ):
            raise BlindReceiverOperatorBankError(
                "D10 feature extractor output drift"
            )
        if dimension is None:
            dimension = int(features.shape[1])
        elif features.shape[1] != dimension:
            raise BlindReceiverOperatorBankError(
                "D10 operator feature dimension drift"
            )
        result[operator] = np.ascontiguousarray(features)
    return result


def build_operator_feature_provenance(
    parent_received_iq_sha256: Sequence[str],
    *,
    view_seed: int,
) -> dict[str, tuple[dict[str, Any], ...]]:
    hashes = tuple(
        str(value).lower() for value in parent_received_iq_sha256
    )
    return {
        operator: tuple(
            {
                "parent_received_iq_sha256": digest,
                "operator_id": operator,
                "view_seed": int(view_seed),
            }
            for digest in hashes
        )
        for operator in OPERATORS
    }


def _map_to_d9(
    features_by_operator: Mapping[str, np.ndarray],
    provenance_by_operator: Mapping[
        str, Sequence[Mapping[str, Any]]
    ],
) -> tuple[
    dict[str, np.ndarray],
    dict[str, tuple[dict[str, Any], ...]],
]:
    if (
        set(features_by_operator) != set(OPERATORS)
        or set(provenance_by_operator) != set(OPERATORS)
    ):
        raise BlindReceiverOperatorBankError(
            "D10 exact operator/provenance set drift"
        )
    mapped_features: dict[str, np.ndarray] = {}
    mapped_provenance: dict[str, tuple[dict[str, Any], ...]] = {}
    for operator in OPERATORS:
        slot = _TO_SLOT[operator]
        rows = tuple(provenance_by_operator[operator])
        translated = []
        for record in rows:
            if (
                set(record)
                != {
                    "parent_received_iq_sha256",
                    "operator_id",
                    "view_seed",
                }
                or str(record["operator_id"]) != operator
            ):
                raise BlindReceiverOperatorBankError(
                    "D10 real operator provenance drift"
                )
            translated.append(
                {
                    "parent_received_iq_sha256": record[
                        "parent_received_iq_sha256"
                    ],
                    "operator_id": slot,
                    "view_seed": record["view_seed"],
                }
            )
        mapped_features[slot] = np.asarray(
            features_by_operator[operator], dtype=np.float32
        )
        mapped_provenance[slot] = tuple(translated)
    return mapped_features, mapped_provenance


def _translate_string(value: str) -> str:
    result = value
    for slot in sorted(_SLOT_OPERATORS, key=len, reverse=True):
        result = result.replace(slot, _FROM_SLOT[slot])
    return result.replace("d9_", "d10_").replace(".d9_", ".d10_")


def _translate_trace(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _translate_trace(item)
            for key, item in value.items()
        }
    if isinstance(value, tuple):
        return tuple(_translate_trace(item) for item in value)
    if isinstance(value, list):
        return [_translate_trace(item) for item in value]
    if isinstance(value, str):
        return _translate_string(value)
    return value


def _strict_gate(
    baseline: Mapping[str, Any],
    final: Mapping[str, Any],
    *,
    classes: Sequence[str],
    scope: str,
) -> dict[str, Any]:
    per_class_nondegradation = all(
        float(final["per_class_accuracy"][label])
        >= float(baseline["per_class_accuracy"][label])
        for label in classes
    )
    nondegradation = bool(
        float(final["overall_accuracy"])
        >= float(baseline["overall_accuracy"])
        and float(final["min_class_accuracy"])
        >= float(baseline["min_class_accuracy"])
        and per_class_nondegradation
    )
    strict_improvement = bool(
        float(final["overall_accuracy"])
        > float(baseline["overall_accuracy"])
        or float(final["min_class_accuracy"])
        > float(baseline["min_class_accuracy"])
        or any(
            float(final["per_class_accuracy"][label])
            > float(baseline["per_class_accuracy"][label])
            for label in classes
        )
    )
    return {
        "scope": scope,
        "overall_non_degradation_pass": (
            float(final["overall_accuracy"])
            >= float(baseline["overall_accuracy"])
        ),
        "floor_non_degradation_pass": (
            float(final["min_class_accuracy"])
            >= float(baseline["min_class_accuracy"])
        ),
        "every_class_non_degradation_pass": per_class_nondegradation,
        "any_accuracy_strict_improvement": strict_improvement,
        "pass": nondegradation and strict_improvement,
        "fallback_reason": (
            None
            if nondegradation and strict_improvement
            else (
                "accuracy_non_degradation_failure"
                if not nondegradation
                else "no_accuracy_improvement_over_all_base"
            )
        ),
    }


def _force_before_base(
    inner: d9.FloorSparseOperatorFusionState,
    mapped_features: Mapping[str, np.ndarray],
    labels: Sequence[str],
    gate: Mapping[str, Any],
) -> d9.FloorSparseOperatorFusionState:
    label_array = np.asarray(tuple(str(value) for value in labels))
    prototypes = d9._operator_prototypes(
        mapped_features[d9.BASE], label_array, inner.classes
    )[:, None, :]
    prototypes = np.concatenate(
        [prototypes, np.zeros_like(prototypes)], axis=1
    )
    indices = np.full((inner.class_count, 2), -1, dtype=np.int8)
    indices[:, 0] = 0
    weights = np.zeros((inner.class_count, 2), dtype=np.float32)
    weights[:, 0] = 1.0
    selection = dict(inner.support_audit["selection"])
    selection["d10_assignment_before_strict_fallback"] = [
        _translate_string(value)
        for value in selection["assignment_before_fallback"]
    ]
    selection["d10_strict_accuracy_gate"] = dict(gate)
    selection["d10_fallback_to_all_base"] = True
    selection["fallback_to_all_base"] = True
    selection["combined_final"] = selection["baseline"]
    audit = dict(inner.support_audit)
    audit["selection"] = selection
    readonly_indices = d9._readonly(indices, np.int8)
    readonly_weights = d9._readonly(weights, np.float32)
    readonly_prototypes = d9._readonly(prototypes, np.float32)
    return replace(
        inner,
        operator_indices=readonly_indices,
        weights=readonly_weights,
        prototypes=readonly_prototypes,
        used_operators=(d9.BASE,),
        selection_lock_sha256=d9._selection_lock_sha256(
            inner.classes,
            readonly_indices,
            readonly_weights,
            inner.calibrations,
        ),
        support_audit=audit,
    )


def _force_new_classes_base(
    inner: d9.FloorSparseOperatorFusionState,
    mapped_features: Mapping[str, np.ndarray],
    labels: Sequence[str],
    gate: Mapping[str, Any],
) -> d9.FloorSparseOperatorFusionState:
    label_array = np.asarray(tuple(str(value) for value in labels))
    indices = np.array(inner.operator_indices, copy=True)
    weights = np.array(inner.weights, copy=True)
    prototypes = np.array(inner.prototypes, copy=True)
    for class_index in range(inner.old_class_count, inner.class_count):
        label = inner.classes[class_index]
        indices[class_index] = (0, -1)
        weights[class_index] = (1.0, 0.0)
        prototypes[class_index, 0] = d9._prototype(
            mapped_features[d9.BASE][label_array == label]
        )
        prototypes[class_index, 1] = 0.0
    selection = dict(inner.support_audit["selection"])
    selection["d10_assignment_before_strict_fallback"] = [
        _translate_string(value)
        for value in selection["assignment_before_fallback"]
    ]
    selection["d10_strict_accuracy_gate"] = dict(gate)
    selection["d10_fallback_new_classes_to_base"] = True
    selection["fallback_new_classes_to_base"] = True
    selection["combined_final_new"] = selection["baseline_new"]
    selection["combined_final_old"] = selection["baseline_old"]
    audit = dict(inner.support_audit)
    audit["selection"] = selection
    readonly_indices = d9._readonly(indices, np.int8)
    readonly_weights = d9._readonly(weights, np.float32)
    readonly_prototypes = d9._readonly(prototypes, np.float32)
    used = d9._used_operators(readonly_indices, readonly_weights)
    return replace(
        inner,
        operator_indices=readonly_indices,
        weights=readonly_weights,
        prototypes=readonly_prototypes,
        used_operators=used,
        selection_lock_sha256=d9._selection_lock_sha256(
            inner.classes,
            readonly_indices,
            readonly_weights,
            inner.calibrations,
        ),
        support_audit=audit,
    )


def fit_blind_receiver_operator_bank(
    features_by_operator: Mapping[str, np.ndarray],
    operator_feature_provenance: Mapping[
        str, Sequence[Mapping[str, Any]]
    ],
    support_labels: Sequence[str],
    *,
    physical_sample_ids: Sequence[str],
    parent_received_iq_sha256: Sequence[str],
    base_resource_audit: Mapping[str, Any],
    received_iq_length: int,
    floor_priority_classes: Sequence[str] = (),
) -> BlindReceiverOperatorBankState:
    mapped_features, mapped_provenance = _map_to_d9(
        features_by_operator, operator_feature_provenance
    )
    if int(received_iq_length) < 3:
        raise BlindReceiverOperatorBankError(
            "D10 received IQ length audit drift"
        )
    inner = d9.fit_floor_sparse_operator_fusion(
        mapped_features,
        mapped_provenance,
        support_labels,
        physical_sample_ids=physical_sample_ids,
        parent_received_iq_sha256=parent_received_iq_sha256,
        base_resource_audit=base_resource_audit,
        floor_priority_classes=floor_priority_classes,
    )
    selection = inner.support_audit["selection"]
    gate = _strict_gate(
        selection["baseline"],
        selection["combined_final"],
        classes=inner.classes,
        scope="before_all_registered_classes",
    )
    if not gate["pass"]:
        inner = _force_before_base(
            inner, mapped_features, support_labels, gate
        )
    return BlindReceiverOperatorBankState(
        inner=inner,
        received_iq_length=int(received_iq_length),
        strict_accuracy_gate=gate,
    )


def extend_blind_receiver_operator_bank(
    parent: BlindReceiverOperatorBankState,
    features_by_operator: Mapping[str, np.ndarray],
    operator_feature_provenance: Mapping[
        str, Sequence[Mapping[str, Any]]
    ],
    support_labels: Sequence[str],
    *,
    physical_sample_ids: Sequence[str],
    parent_received_iq_sha256: Sequence[str],
    floor_priority_classes: Sequence[str] = (),
) -> BlindReceiverOperatorBankState:
    if not isinstance(parent, BlindReceiverOperatorBankState):
        raise BlindReceiverOperatorBankError(
            "D10 parent state is required"
        )
    mapped_features, mapped_provenance = _map_to_d9(
        features_by_operator, operator_feature_provenance
    )
    inner = d9.extend_floor_sparse_operator_fusion(
        parent.inner,
        mapped_features,
        mapped_provenance,
        support_labels,
        physical_sample_ids=physical_sample_ids,
        parent_received_iq_sha256=parent_received_iq_sha256,
        floor_priority_classes=floor_priority_classes,
    )
    selection = inner.support_audit["selection"]
    new_classes = inner.classes[inner.old_class_count :]
    gate = _strict_gate(
        selection["baseline_new"],
        selection["combined_final_new"],
        classes=new_classes,
        scope="after_new_registered_classes",
    )
    old_final = selection["combined_final_old"]
    old_base = selection["baseline_old"]
    old_pass = bool(
        float(old_final["overall_accuracy"])
        >= float(old_base["overall_accuracy"])
        and all(
            float(old_final["per_class_accuracy"][label])
            >= float(old_base["per_class_accuracy"][label])
            for label in parent.classes
        )
    )
    gate = {**gate, "old_support_intrusion_non_degradation_pass": old_pass}
    gate["pass"] = bool(gate["pass"] and old_pass)
    if not gate["pass"]:
        inner = _force_new_classes_base(
            inner, mapped_features, support_labels, gate
        )
    old_count = parent.class_count
    if (
        not np.array_equal(
            inner.operator_indices[:old_count],
            parent.operator_indices,
        )
        or not np.array_equal(
            inner.weights[:old_count], parent.weights
        )
        or not np.array_equal(
            inner.prototypes[:old_count], parent.prototypes
        )
        or inner.calibrations != parent.inner.calibrations
    ):
        raise BlindReceiverOperatorBankError(
            "D10 old state changed during registration"
        )
    return BlindReceiverOperatorBankState(
        inner=inner,
        received_iq_length=parent.received_iq_length,
        strict_accuracy_gate=gate,
    )


def rebuild_locked_blind_receiver_prototypes(
    locked_k10_state: BlindReceiverOperatorBankState,
    features_by_operator: Mapping[str, np.ndarray],
    operator_feature_provenance: Mapping[
        str, Sequence[Mapping[str, Any]]
    ],
    support_labels: Sequence[str],
    *,
    physical_sample_ids: Sequence[str],
    parent_received_iq_sha256: Sequence[str],
) -> BlindReceiverOperatorBankState:
    if not isinstance(
        locked_k10_state, BlindReceiverOperatorBankState
    ):
        raise BlindReceiverOperatorBankError(
            "D10 K10 state is required"
        )
    mapped_features, mapped_provenance = _map_to_d9(
        features_by_operator, operator_feature_provenance
    )
    inner = d9.rebuild_locked_floor_sparse_prototypes(
        locked_k10_state.inner,
        mapped_features,
        mapped_provenance,
        support_labels,
        physical_sample_ids=physical_sample_ids,
        parent_received_iq_sha256=parent_received_iq_sha256,
    )
    return BlindReceiverOperatorBankState(
        inner=inner,
        received_iq_length=locked_k10_state.received_iq_length,
        strict_accuracy_gate=locked_k10_state.strict_accuracy_gate,
    )


def predict_blind_receiver_operator_bank(
    state: BlindReceiverOperatorBankState,
    received_iq: np.ndarray,
    *,
    feature_extractor: SamplewiseSealedFeatureExtractor,
) -> BlindReceiverOperatorPrediction:
    if not isinstance(state, BlindReceiverOperatorBankState):
        raise BlindReceiverOperatorBankError("D10 state is required")
    if not isinstance(
        feature_extractor, SamplewiseSealedFeatureExtractor
    ):
        raise BlindReceiverOperatorBankError(
            "D10 query extractor must be samplewise sealed"
        )
    expected = d9._samplewise_seal_digest(
        feature_extractor.extractor_id,
        feature_extractor.validation_rows_sha256,
    )
    if (
        feature_extractor.samplewise_contract_sha256 != expected
        or feature_extractor.validation_max_abs_error > 1.0e-6
        or not feature_extractor.batch_independent
        or feature_extractor.query_updates != 0
    ):
        raise BlindReceiverOperatorBankError(
            "D10 samplewise extractor contract drift"
        )
    actual = extract_operator_features(
        received_iq,
        feature_extractor=feature_extractor,
        operator_ids=state.used_operators,
    )
    mapped = {
        _TO_SLOT[operator]: rows for operator, rows in actual.items()
    }
    scores = d9._readonly(
        d9._score_from_features(state.inner, mapped), np.float32
    )
    prediction = np.argmax(scores, axis=1)
    return BlindReceiverOperatorPrediction(
        labels=tuple(
            state.classes[int(index)] for index in prediction
        ),
        scores=scores,
        operators_computed=state.used_operators,
    )


def _operator_macs(operator: str, length: int) -> int:
    if operator == BASE:
        return 0
    if operator == WL_IQ_CIRCULARIZE:
        return 40 * int(length)
    log2_length = int(np.ceil(np.log2(max(int(length), 2))))
    return 10 * int(length) * log2_length + 24 * int(length)


def public_query_interface_is_oracle_free() -> bool:
    forbidden = {"label", "truth", "role", "quota", "assignment", "graph"}
    return not any(
        token in parameter.lower()
        for parameter in inspect.signature(
            predict_blind_receiver_operator_bank
        ).parameters
        for token in forbidden
    )


__all__ = [
    "BASE",
    "FFT_ENVELOPE_EQ",
    "FFT_ENVELOPE_SHRINK",
    "FFT_GAIN_MAX",
    "FFT_GAIN_MIN",
    "OPERATORS",
    "WL_IQ_CIRCULARIZE",
    "WL_RHO_MAGNITUDE_CAP",
    "BlindReceiverOperatorBankError",
    "BlindReceiverOperatorBankState",
    "BlindReceiverOperatorPrediction",
    "SamplewiseSealedFeatureExtractor",
    "apply_received_iq_operator",
    "build_operator_feature_provenance",
    "extend_blind_receiver_operator_bank",
    "extract_operator_features",
    "fit_blind_receiver_operator_bank",
    "predict_blind_receiver_operator_bank",
    "public_query_interface_is_oracle_free",
    "rebuild_locked_blind_receiver_prototypes",
    "seal_samplewise_feature_extractor",
]
