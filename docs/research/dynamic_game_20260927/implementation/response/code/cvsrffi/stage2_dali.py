"""D20 fixed-medoid, max-old-preserving identity reranking.

The registered target-support prototypes define the old/new group decision.
The immutable Phase1 int8 component and same-received-IQ ADV3B02 logits may
only rerank identities inside the old-class group.  New-class scores are never
modified, and the maximum old-class score is restored exactly for every
sample.  No query-side fit, role, quota, label, or batch operation is exposed.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from cvsrffi.stage2_ciaf import (
    CiafError,
    FEATURE_DIM,
    Int8DomainClassComponent,
    MAX_PERSISTENT_STATE_BYTES,
    _normalize_rows,
    _normalize_vector,
    _readonly,
    _support_matrix,
    _support_prototypes,
)


SCHEMA = "cvs.phase2.d20_dali_maxold.v2"
EPS = 1.0e-8


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _deep_freeze(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze(item) for item in value)
    return value


@dataclass(frozen=True)
class DaliConfig:
    """Pre-registered DALI hyperparameters.

    The superseded support-selected ``top_m`` surface is deliberately absent.
    Pair fields remain constructor-compatible with the draft runner, but
    strict max-old mode only accepts ``pair_weight=0`` so a legacy candidate
    cannot silently modify new-class scores.
    """

    ground_weight: float = 0.05
    direct_weight: float = 0.0
    evidence_clip: float = 0.05
    k_shrink_offset: float = 2.0
    scale_floor: float = 1.0e-4
    medoid_domain_index: int | None = None

    # Deprecated pair compatibility fields.  Only zero pair weight passes.
    pair_weight: float = 0.0
    pair_threshold: float = 0.60
    pair_margin: float = 0.02

    def validate(self) -> None:
        values = (
            self.ground_weight,
            self.direct_weight,
            self.evidence_clip,
            self.k_shrink_offset,
            self.scale_floor,
            self.pair_weight,
            self.pair_threshold,
            self.pair_margin,
        )
        medoid = self.medoid_domain_index
        if not all(math.isfinite(float(value)) for value in values):
            raise CiafError("DALI configuration must be finite")
        if (
            not 0.0 <= self.ground_weight <= 0.25
            or not 0.0 <= self.direct_weight <= 1.0
            or not 0.0 <= self.evidence_clip <= 0.25
            or not 0.0 <= self.k_shrink_offset <= 100.0
            or not 0.0 < self.scale_floor <= 0.10
            or not math.isclose(float(self.pair_weight), 0.0, abs_tol=1.0e-12)
            or not -1.0 < self.pair_threshold < 1.0
            or self.pair_margin < 0.0
            or (
                medoid is not None
                and (
                    isinstance(medoid, (bool, np.bool_))
                    or not isinstance(medoid, (int, np.integer))
                    or int(medoid) < 0
                )
            )
        ):
            raise CiafError(
                "DALI configuration is out of range for fixed-medoid max-old mode"
            )


def _transient_domain_anchors(
    component: Int8DomainClassComponent, domain_index: int
) -> np.ndarray:
    """Dequantize one domain transiently; callers must not persist the result."""

    index = int(domain_index)
    if index < 0 or index >= component.domain_class_q.shape[0]:
        raise CiafError("DALI medoid domain index is out of range")
    if not np.all(component.domain_class_mask[index] == 1):
        raise CiafError("DALI medoid domain must cover every old class")
    q = component.domain_class_q[index].astype(np.float32)
    scale = component.domain_class_scale[index, :, None].astype(np.float32)
    return _normalize_rows(q * scale)


def _component_maximin_medoid(component: Int8DomainClassComponent) -> int:
    """Return a deterministic component-only global max-min medoid.

    Eligible domains must cover every old class.  Each candidate is scored by
    its worst same-class cosine against every active domain/class anchor.
    Stable ``argmax`` makes the smallest domain index the deterministic tie
    break.  Target support and query data are intentionally absent.
    """

    mask = component.domain_class_mask.astype(bool)
    eligible = np.flatnonzero(np.all(mask, axis=1))
    if not len(eligible):
        raise CiafError("DALI component has no all-old-class medoid domain")
    candidate_scores: list[float] = []
    for domain_index in eligible:
        candidate = _transient_domain_anchors(component, int(domain_index))
        worst = 1.0
        for class_index in range(len(component.class_registry)):
            active = component.dequantized_class_anchors(class_index)
            worst = min(worst, float(np.min(active @ candidate[class_index])))
        candidate_scores.append(worst)
    return int(eligible[int(np.argmax(np.asarray(candidate_scores, dtype=np.float64)))])


def _centered_support_scale(value: np.ndarray, floor: float) -> float:
    matrix = np.asarray(value, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[1] < 2 or not np.isfinite(matrix).all():
        raise CiafError("DALI support evidence matrix drift")
    centered = matrix - np.mean(matrix, axis=1, keepdims=True, dtype=np.float32)
    rms = float(np.sqrt(np.mean(np.square(centered), dtype=np.float64)))
    return max(rms, float(floor))


def _direct_evidence(
    logits: np.ndarray | None, old_count: int, support_scale: float
) -> np.ndarray:
    if logits is None:
        raise CiafError("sealed ADV3B02 logits are required by direct evidence")
    value = np.asarray(logits, dtype=np.float32)
    if value.ndim != 1:
        raise CiafError("sealed ADV3B02 logits must describe exactly one sample")
    if value.shape[0] < old_count or not np.isfinite(value).all():
        raise CiafError("sealed ADV3B02 logits do not cover old classes")
    old = value[:old_count]
    centered = old - np.mean(old, dtype=np.float32)
    return np.tanh(centered / np.float32(support_scale)).astype(np.float32)


@dataclass(frozen=True)
class DaliState:
    schema: str
    classes: tuple[str, ...]
    target_prototypes: np.ndarray
    old_support_prototypes: np.ndarray
    fixed_medoid_domain_index: int
    ground_weight_by_old_class: np.ndarray
    support_margin_q25_by_old_class: np.ndarray
    support_count_by_class: np.ndarray
    support_ground_scale: float
    support_direct_scale: float
    evidence_k_shrink: float
    old_class_count: int
    k_shot: int
    component: Int8DomainClassComponent
    config: DaliConfig
    support_audit: Mapping[str, Any]

    def __post_init__(self) -> None:
        self.config.validate()
        classes = tuple(str(value) for value in self.classes)
        prototypes = np.asarray(self.target_prototypes)
        old_support = np.asarray(self.old_support_prototypes)
        ground = np.asarray(self.ground_weight_by_old_class)
        margins = np.asarray(self.support_margin_q25_by_old_class)
        counts = np.asarray(self.support_count_by_class)
        old_count = int(self.old_class_count)
        medoid = int(self.fixed_medoid_domain_index)
        expected_medoid = (
            int(self.config.medoid_domain_index)
            if self.config.medoid_domain_index is not None
            else _component_maximin_medoid(self.component)
        )
        active = (
            self.component.domain_class_mask[medoid]
            if 0 <= medoid < self.component.domain_class_q.shape[0]
            else np.empty(0, dtype=np.uint8)
        )
        expected_shrink = float(self.k_shot) / (
            float(self.k_shot) + float(self.config.k_shrink_offset)
        )
        if (
            self.schema != SCHEMA
            or not isinstance(self.component, Int8DomainClassComponent)
            or old_count != len(self.component.class_registry)
            or old_count < 2
            or len(classes) < old_count
            or len(set(classes)) != len(classes)
            or any(not value for value in classes)
            or classes[:old_count] != self.component.class_registry
            or prototypes.shape != (len(classes), FEATURE_DIM)
            or old_support.shape != (old_count, FEATURE_DIM)
            or active.shape != (old_count,)
            or not np.all(active == 1)
            or medoid != expected_medoid
            or ground.shape != (old_count,)
            or not np.allclose(
                ground,
                self.config.ground_weight * expected_shrink,
                atol=1.0e-7,
            )
            or margins.shape != (old_count,)
            or counts.shape != (len(classes),)
            or int(self.k_shot) < 1
            or bool(np.any(counts != int(self.k_shot)))
            or not np.array_equal(prototypes[:old_count], old_support)
            or not np.allclose(np.linalg.norm(prototypes, axis=1), 1.0, atol=1.0e-5)
            or not math.isfinite(float(self.support_ground_scale))
            or not math.isfinite(float(self.support_direct_scale))
            or float(self.support_ground_scale) < self.config.scale_floor
            or float(self.support_direct_scale) < self.config.scale_floor
            or not math.isclose(
                float(self.evidence_k_shrink), expected_shrink, abs_tol=1.0e-7
            )
            or not isinstance(self.support_audit, Mapping)
            or not all(
                np.isfinite(value).all()
                for value in (prototypes, old_support, ground, margins)
            )
        ):
            raise CiafError("DALI state drift")
        object.__setattr__(self, "classes", classes)
        object.__setattr__(self, "target_prototypes", _readonly(prototypes, np.float32))
        object.__setattr__(self, "old_support_prototypes", _readonly(old_support, np.float32))
        object.__setattr__(self, "ground_weight_by_old_class", _readonly(ground, np.float32))
        object.__setattr__(
            self,
            "support_margin_q25_by_old_class",
            _readonly(margins, np.float32),
        )
        object.__setattr__(self, "support_count_by_class", _readonly(counts, np.int16))
        object.__setattr__(self, "support_audit", _deep_freeze(self.support_audit))
        if self.persistent_state_bytes > MAX_PERSISTENT_STATE_BYTES:
            raise CiafError("DALI persistent state exceeds 256KiB")

    @property
    def medoid_domain_index(self) -> int:
        return int(self.fixed_medoid_domain_index)

    @property
    def persistent_state_bytes(self) -> int:
        arrays = (
            self.target_prototypes,
            self.old_support_prototypes,
            self.ground_weight_by_old_class,
            self.support_margin_q25_by_old_class,
            self.support_count_by_class,
        )
        scalar_bytes = 3 * np.dtype(np.float32).itemsize
        metadata = len(self.schema.encode("utf-8")) + sum(
            len(value.encode("utf-8")) for value in self.classes
        )
        return int(
            self.component.state_bytes
            + sum(value.nbytes for value in arrays)
            + scalar_bytes
            + metadata
        )

    def resource_audit(self) -> dict[str, Any]:
        prototype_macs = len(self.classes) * FEATURE_DIM
        ground_macs = (
            self.old_class_count * FEATURE_DIM if self.config.ground_weight > 0.0 else 0
        )
        scalar_ops = 12 * self.old_class_count
        return {
            "schema": "cvs.phase2.d20_dali_maxold.resource.v2",
            "adaptation_mode": "EVAL_ONLY_CLOSED_FORM_ADAPTATION",
            "trainable_parameters": 0,
            "adaptation_epochs": 0,
            "optimizer_steps": 0,
            "persistent_state_bytes": self.persistent_state_bytes,
            "persistent_state_limit_bytes": MAX_PERSISTENT_STATE_BYTES,
            "persistent_state_limit_pass": self.persistent_state_bytes
            <= MAX_PERSISTENT_STATE_BYTES,
            "int8_component_state_bytes": self.component.state_bytes,
            "estimated_head_macs_per_query": int(
                prototype_macs + ground_macs + scalar_ops
            ),
            "prototype_score_macs_per_query": int(prototype_macs),
            "fixed_medoid_ground_macs_per_query": int(ground_macs),
            "fixed_medoid_domain_index": self.medoid_domain_index,
            "selected_int8_domain_anchors_per_old_class": 1,
            "support_domain_selection_access": False,
            "query_domain_selection_access": False,
            "max_old_preservation": "strict_exact_per_sample",
            "new_score_policy": "bitwise_unchanged",
            "persistent_full_precision_ground_anchor_count": 0,
            "ground_anchor_dequantization_policy": "transient_score_only_not_persisted",
            "query_rows_used_for_fit": 0,
            "query_updates": 0,
            "dense_query_graph_bytes": 0,
            "per_sample_all_registered_classes": True,
            "source_sample_access": False,
            "sample_level_source_feature_access": False,
            "int8_component_update_access": False,
        }


def fit_old_dali(
    component: Int8DomainClassComponent,
    support_z_id: np.ndarray,
    support_labels: Sequence[str],
    support_direct_logits: np.ndarray | None = None,
    *,
    config: DaliConfig,
) -> DaliState:
    if not isinstance(component, Int8DomainClassComponent):
        raise CiafError("immutable int8 domain-class component required")
    config.validate()
    features, labels, _, k_shot = _support_matrix(
        support_z_id, support_labels, expected_classes=component.class_registry
    )
    direct: np.ndarray | None
    if support_direct_logits is None:
        direct = None
        if config.direct_weight > 0.0:
            raise CiafError("support direct logits are required by direct evidence")
    else:
        direct = np.asarray(support_direct_logits, dtype=np.float32)
        if (
            direct.ndim != 2
            or direct.shape[0] != len(features)
            or direct.shape[1] < len(component.class_registry)
            or not np.isfinite(direct).all()
        ):
            raise CiafError("support direct-logit matrix drift")

    if config.medoid_domain_index is None:
        medoid = _component_maximin_medoid(component)
        medoid_policy = "offline_component_global_maximin_medoid_v1"
    else:
        medoid = int(config.medoid_domain_index)
        _transient_domain_anchors(component, medoid)
        medoid_policy = "explicit_preregistered_component_domain_index"

    classes = component.class_registry
    prototypes = _support_prototypes(features, labels, classes)
    anchors = _transient_domain_anchors(component, medoid)
    support_ground_scale = _centered_support_scale(
        features @ anchors.T, config.scale_floor
    )
    support_direct_scale = (
        _centered_support_scale(direct[:, : len(classes)], config.scale_floor)
        if direct is not None
        else 1.0
    )
    margin_q25: list[float] = []
    for class_index, label in enumerate(classes):
        own_rows = features[labels == label]
        rivals = np.delete(prototypes, class_index, axis=0)
        margins = own_rows @ prototypes[class_index] - np.max(
            own_rows @ rivals.T, axis=1
        )
        margin_q25.append(float(np.quantile(margins, 0.25)))

    shrink = float(k_shot) / (float(k_shot) + float(config.k_shrink_offset))
    effective_ground = np.full(
        len(classes), config.ground_weight * shrink, dtype=np.float32
    )
    audit = {
        "selection_data": "immutable_int8_component_only_no_target_support",
        "medoid_policy": medoid_policy,
        "fixed_medoid_domain_index": medoid,
        "query_rows_used_for_fit": 0,
        "target_prototype_policy": "unchanged_support_centroid",
        "ground_policy": "fixed_medoid_transient_int8_old_internal_rerank",
        "direct_logit_policy": "same_received_iq_sealed_ADV3B02_old_internal_rerank",
        "support_scale_policy": "row_centered_rms_support_only",
        "support_ground_scale": support_ground_scale,
        "support_direct_scale": support_direct_scale,
        "k_shrink": shrink,
        "max_old_preservation": "strict_exact_per_sample",
        "new_score_policy": "bitwise_unchanged",
        "ground_weight_by_old_class": dict(
            zip(classes, map(float, effective_ground))
        ),
        "support_margin_q25_by_old_class": dict(zip(classes, margin_q25)),
    }
    return DaliState(
        schema=SCHEMA,
        classes=classes,
        target_prototypes=prototypes,
        old_support_prototypes=prototypes,
        fixed_medoid_domain_index=medoid,
        ground_weight_by_old_class=effective_ground,
        support_margin_q25_by_old_class=np.asarray(margin_q25, dtype=np.float32),
        support_count_by_class=np.full(len(classes), k_shot, dtype=np.int16),
        support_ground_scale=support_ground_scale,
        support_direct_scale=support_direct_scale,
        evidence_k_shrink=shrink,
        old_class_count=len(classes),
        k_shot=k_shot,
        component=component,
        config=config,
        support_audit=audit,
    )


def register_new_dali(
    state: DaliState,
    support_z_id: np.ndarray,
    support_labels: Sequence[str],
    *,
    registered_classes: Sequence[str],
) -> DaliState:
    if not isinstance(state, DaliState):
        raise CiafError("valid DALI state required")
    new_classes = tuple(str(value) for value in registered_classes)
    if (
        not new_classes
        or len(set(new_classes)) != len(new_classes)
        or any(not value for value in new_classes)
    ):
        raise CiafError("DALI new registry drift")
    features, labels, discovered, _ = _support_matrix(
        support_z_id,
        support_labels,
        expected_classes=registered_classes,
        expected_k=state.k_shot,
    )
    if set(new_classes) != set(discovered) or set(new_classes) & set(state.classes):
        raise CiafError("DALI new registry drift")
    new_proto = _support_prototypes(features, labels, new_classes)
    audit = dict(state.support_audit)
    audit.update(
        {
            "new_registration_rule": "unchanged_target_support_centroid",
            "old_score_state_bitwise_unchanged": True,
            "new_score_modification_access": False,
        }
    )
    return DaliState(
        schema=SCHEMA,
        classes=state.classes + new_classes,
        target_prototypes=np.concatenate([state.target_prototypes, new_proto], axis=0),
        old_support_prototypes=state.old_support_prototypes,
        fixed_medoid_domain_index=state.fixed_medoid_domain_index,
        ground_weight_by_old_class=state.ground_weight_by_old_class,
        support_margin_q25_by_old_class=state.support_margin_q25_by_old_class,
        support_count_by_class=np.concatenate(
            [
                state.support_count_by_class,
                np.full(len(new_classes), state.k_shot, dtype=np.int16),
            ]
        ),
        support_ground_scale=state.support_ground_scale,
        support_direct_scale=state.support_direct_scale,
        evidence_k_shrink=state.evidence_k_shrink,
        old_class_count=state.old_class_count,
        k_shot=state.k_shot,
        component=state.component,
        config=state.config,
        support_audit=audit,
    )


def _base_scores(state: DaliState, value: np.ndarray) -> np.ndarray:
    # Score old and appended prototypes separately so registering new classes
    # cannot change the floating-point reduction used for any old score column.
    old = value @ state.old_support_prototypes.T
    if len(state.classes) == state.old_class_count:
        return np.ascontiguousarray(old, dtype=np.float32)
    new = value @ state.target_prototypes[state.old_class_count :].T
    return np.ascontiguousarray(np.concatenate([old, new]), dtype=np.float32)


def _validated_feature(z_id: np.ndarray) -> np.ndarray:
    value = np.asarray(z_id, dtype=np.float32)
    if value.shape == (1, FEATURE_DIM):
        value = value[0]
    if value.shape != (FEATURE_DIM,):
        raise CiafError(f"DALI scoring requires exactly one {FEATURE_DIM}-D z_id")
    return _normalize_vector(value)


def rerank_old_scores_dali(
    state: DaliState,
    base_scores: np.ndarray,
    z_id: np.ndarray,
    direct_logits: np.ndarray | None = None,
) -> np.ndarray:
    """Rerank old identities without changing either old/new group maximum.

    ``base_scores`` may come from the identity-only prototype head or from the
    fixed-received-IQ ``z_id+FFT/RF`` head.  Its class order must exactly match
    ``state.classes``.  The returned new-class slice is bitwise identical to
    the canonical float32 input slice and ``max(old)`` is exactly preserved.
    """

    if not isinstance(state, DaliState):
        raise CiafError("valid DALI state required")
    value = _validated_feature(z_id)
    base = np.asarray(base_scores, dtype=np.float32)
    if base.ndim != 1 or base.shape != (len(state.classes),) or not np.isfinite(base).all():
        raise CiafError("DALI base scores must cover all registered classes")
    base = np.ascontiguousarray(base, dtype=np.float32)

    if state.config.ground_weight == 0.0 and state.config.direct_weight == 0.0:
        return _readonly(base, np.float32)

    old_count = state.old_class_count
    auxiliary = np.zeros(old_count, dtype=np.float32)
    if state.config.ground_weight > 0.0:
        anchors = _transient_domain_anchors(state.component, state.medoid_domain_index)
        ground = anchors @ value
        ground -= np.mean(ground, dtype=np.float32)
        auxiliary += np.float32(
            state.config.ground_weight * state.evidence_k_shrink
        ) * np.tanh(ground / np.float32(state.support_ground_scale)).astype(np.float32)
    if state.config.direct_weight > 0.0:
        auxiliary += np.float32(
            state.config.direct_weight * state.evidence_k_shrink
        ) * _direct_evidence(direct_logits, old_count, state.support_direct_scale)

    auxiliary -= np.mean(auxiliary, dtype=np.float32)
    delta = np.clip(
        auxiliary,
        -np.float32(state.config.evidence_clip),
        np.float32(state.config.evidence_clip),
    ).astype(np.float32)
    base_old = base[:old_count]
    base_max = np.max(base_old)
    uncalibrated = np.ascontiguousarray(base_old + delta, dtype=np.float32)
    winner = int(np.argmax(uncalibrated))
    shifted = np.ascontiguousarray(
        uncalibrated + np.float32(base_max - uncalibrated[winner]),
        dtype=np.float32,
    )
    # Force exact representational equality, not merely an allclose claim.
    shifted = np.minimum(shifted, base_max).astype(np.float32)
    shifted[winner] = base_max
    scores = base.copy()
    scores[:old_count] = shifted

    if np.max(scores[:old_count]) != base_max:
        raise CiafError("DALI max-old preservation drift")
    if not np.array_equal(scores[old_count:], base[old_count:]):
        raise CiafError("DALI new-score immutability drift")
    return _readonly(scores, np.float32)


def score_one_dali(
    state: DaliState,
    z_id: np.ndarray,
    direct_logits: np.ndarray | None = None,
) -> np.ndarray:
    if not isinstance(state, DaliState):
        raise CiafError("valid DALI state required")
    value = _validated_feature(z_id)
    return rerank_old_scores_dali(
        state,
        _base_scores(state, value),
        value,
        direct_logits,
    )


def predict_one_dali(
    state: DaliState,
    z_id: np.ndarray,
    direct_logits: np.ndarray | None = None,
) -> tuple[str, np.ndarray]:
    scores = score_one_dali(state, z_id, direct_logits)
    return state.classes[int(np.argmax(scores))], scores


__all__ = [
    "DaliConfig",
    "DaliState",
    "fit_old_dali",
    "predict_one_dali",
    "rerank_old_scores_dali",
    "register_new_dali",
    "score_one_dali",
]
