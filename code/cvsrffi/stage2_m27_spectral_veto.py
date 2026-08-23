"""Support-only spectral consensus veto for ERBT-IDR M2.7.

The frozen no-RF32 D92 E0 head remains the primary decision and the frozen
M2.5 B3 state remains the only performance branch.  Spectral representations
can only accept or veto a B3 row; they never add an independent class logit.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from cvsrffi.stage2_m24_compiler import M24InferenceState
from cvsrffi.stage2_m24_features import physical_if256
from cvsrffi.stage2_m25_anchored_residual import (
    B3,
    M25AnchoredResidualState,
    fit_m25_anchored_residual,
)
from cvsrffi.stage2_m26_spectral_anchor import fft_magnitude_geometry


V1 = "M27-V1-B3-MGD-CONSENSUS-VETO"
V2 = "M27-V2-B3-PHASE32-CONSENSUS-VETO"
M27_SPECTRAL_VETO_ARMS = (V1, V2)

RELIABILITY_MIN_ACCURACY = 0.50
RELIABILITY_MIN_GROUP_ACCURACY = 0.40
MARGIN_QUANTILE = 10.0
_EPS = 1.0e-12


class M27SpectralVetoError(ValueError):
    """Raised when an M2.7 support state or query view fails closed."""


def _readonly(value: Any, dtype: Any) -> np.ndarray:
    array = np.ascontiguousarray(value, dtype=dtype)
    result = np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)
    result.setflags(write=False)
    return result


def _rows(value: Any, *, name: str, width: int | None = None) -> np.ndarray:
    rows = np.asarray(value, dtype=np.float64)
    if (
        rows.ndim != 2
        or rows.shape[0] <= 0
        or (width is not None and rows.shape[1] != int(width))
        or not np.isfinite(rows).all()
    ):
        expected = "N x D" if width is None else f"N x {int(width)}"
        raise M27SpectralVetoError(f"{name} must be finite {expected}")
    return rows


def _unit_rows(value: Any, *, name: str) -> np.ndarray:
    rows = _rows(value, name=name)
    norm = np.linalg.norm(rows, axis=1, keepdims=True)
    if np.any(norm <= _EPS):
        raise M27SpectralVetoError(f"{name} contains a degenerate row")
    return rows / norm


def _canonical_digest(payload: Mapping[str, Any], arrays: Sequence[np.ndarray]) -> str:
    digest = hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    )
    for array in arrays:
        value = np.ascontiguousarray(array)
        digest.update(value.dtype.str.encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest()


def m27_arm_config_hash(arm: str) -> str:
    if arm not in M27_SPECTRAL_VETO_ARMS:
        raise M27SpectralVetoError("unknown M2.7 spectral veto arm")
    payload = {
        "schema": "cvs.erbt_idr.m27.spectral_veto_config.v1",
        "arm": arm,
        "protocol_schema": "p2_min_v1",
        "base": "P2-A1_NO_RF32_R1",
        "performance_branch": B3,
        "representation_policy": "B3_FLIP_CONSENSUS_VETO_ONLY",
        "target_shift_source": "CLASS_BALANCED_OLD_SUPPORT",
        "reliability_min_accuracy": RELIABILITY_MIN_ACCURACY,
        "reliability_min_group_accuracy": RELIABILITY_MIN_GROUP_ACCURACY,
        "margin_quantile": MARGIN_QUANTILE,
        "query_state_update": False,
        "query_truth_access": False,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class TargetCenteredCompetitionModel:
    """Class-balanced target-support geometry used only for consensus."""

    classes: tuple[str, ...]
    old_classes: tuple[str, ...]
    shared_target_center: np.ndarray
    centered_old_class_centres: np.ndarray
    class_prototypes: np.ndarray
    reliability_accepted: bool
    loo_accuracy: float
    old_loo_accuracy: float
    new_loo_accuracy: float
    margin_threshold: float
    state_digest: str

    def __post_init__(self) -> None:
        shared = np.asarray(self.shared_target_center, dtype=np.float32)
        centered = np.asarray(self.centered_old_class_centres, dtype=np.float32)
        prototypes = np.asarray(self.class_prototypes, dtype=np.float32)
        if (
            shared.ndim != 1
            or centered.shape != (len(self.old_classes), len(shared))
            or prototypes.shape != (len(self.classes), len(shared))
            or not np.isfinite(shared).all()
            or not np.isfinite(centered).all()
            or not np.isfinite(prototypes).all()
            or len(set(self.classes)) != len(self.classes)
            or not set(self.old_classes).issubset(self.classes)
            or len(self.state_digest) != 64
            or float(self.margin_threshold) < 0.0
        ):
            raise M27SpectralVetoError("target competition state drift")
        object.__setattr__(self, "shared_target_center", _readonly(shared, np.float32))
        object.__setattr__(
            self,
            "centered_old_class_centres",
            _readonly(centered, np.float32),
        )
        object.__setattr__(self, "class_prototypes", _readonly(prototypes, np.float32))

    @property
    def feature_dim(self) -> int:
        return int(len(self.shared_target_center))

    @property
    def state_bytes(self) -> int:
        return int(
            self.shared_target_center.nbytes
            + self.centered_old_class_centres.nbytes
            + self.class_prototypes.nbytes
        )

    def transform(self, value: Any) -> np.ndarray:
        rows = _rows(value, name="representation query", width=self.feature_dim)
        return _unit_rows(
            rows - self.shared_target_center.astype(np.float64)[None, :],
            name="target-centred representation query",
        ).astype(np.float32)

    def score(self, value: Any) -> np.ndarray:
        transformed = self.transform(value).astype(np.float64)
        return (transformed @ self.class_prototypes.astype(np.float64).T).astype(
            np.float32
        )


def _validate_support(
    support: Any,
    support_labels: Any,
    *,
    classes: Sequence[str],
    old_class_count: int,
    k_shot: int,
    old_classes: Sequence[str] | None,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...], tuple[str, ...]]:
    rows = _rows(support, name="representation support")
    labels = np.asarray(support_labels).astype(str)
    registry = tuple(str(item) for item in classes)
    old_registry = (
        tuple(str(item) for item in old_classes)
        if old_classes is not None
        else registry[: int(old_class_count)]
    )
    if (
        labels.shape != (len(rows),)
        or len(registry) < 2
        or len(set(registry)) != len(registry)
        or len(old_registry) != int(old_class_count)
        or len(set(old_registry)) != len(old_registry)
        or not set(old_registry).issubset(registry)
        or int(k_shot) < 1
        or set(labels.tolist()) != set(registry)
        or len(rows) != len(registry) * int(k_shot)
        or any(int(np.sum(labels == name)) != int(k_shot) for name in registry)
    ):
        raise M27SpectralVetoError(
            "representation support must be exact class-symmetric K-shot"
        )
    return rows, labels, registry, old_registry


def fit_target_centered_competition(
    support: Any,
    support_labels: Any,
    *,
    classes: Sequence[str],
    old_class_count: int,
    k_shot: int,
    old_classes: Sequence[str] | None = None,
) -> tuple[TargetCenteredCompetitionModel, Mapping[str, Any]]:
    """Fit a class-shared target shift and truth-free support consensus head."""

    rows, labels, registry, old_registry = _validate_support(
        support,
        support_labels,
        classes=classes,
        old_class_count=int(old_class_count),
        k_shot=int(k_shot),
        old_classes=old_classes,
    )
    robust_centres = np.stack(
        [np.median(rows[labels == name], axis=0) for name in registry]
    )
    old_indices = [registry.index(name) for name in old_registry]
    shared = np.mean(robust_centres[old_indices], axis=0)
    centered_old = robust_centres[old_indices] - shared[None, :]
    centered_support = rows - shared[None, :]
    fallback_policy: str | None = None
    if int(k_shot) == 1:
        prototype_rows = np.stack(
            [np.mean(centered_support[labels == name], axis=0) for name in registry]
        )
        prototype_norm = np.linalg.norm(prototype_rows, axis=1, keepdims=True)
        prototypes = np.divide(
            prototype_rows,
            prototype_norm,
            out=np.zeros_like(prototype_rows),
            where=prototype_norm > _EPS,
        )
        loo_accuracy = 0.0
        old_accuracy = 0.0
        new_accuracy = 0.0
        margin_threshold = 0.0
        reliability = False
        fallback_policy = "K1_EXACT_B0"
    else:
        transformed = _unit_rows(
            centered_support, name="target-centred representation support"
        )
        prototypes = _unit_rows(
            np.stack(
                [np.mean(transformed[labels == name], axis=0) for name in registry]
            ),
            name="target-centred class prototypes",
        )

        loo_scores = np.empty((len(rows), len(registry)), dtype=np.float64)
        target_indices = np.asarray(
            [registry.index(name) for name in labels.tolist()]
        )
        for held in range(len(rows)):
            folded = []
            for name in registry:
                mask = labels == name
                mask[held] = False
                folded.append(np.mean(transformed[mask], axis=0))
            folded_prototypes = _unit_rows(
                np.stack(folded), name="leave-one-out representation prototypes"
            )
            loo_scores[held] = transformed[held] @ folded_prototypes.T
        predictions = np.argmax(loo_scores, axis=1)
        correct = predictions == target_indices
        old_mask = np.isin(labels, np.asarray(old_registry))
        new_mask = ~old_mask
        loo_accuracy = float(np.mean(correct))
        old_accuracy = float(np.mean(correct[old_mask]))
        new_accuracy = (
            float(np.mean(correct[new_mask])) if np.any(new_mask) else old_accuracy
        )
        masked = loo_scores.copy()
        masked[np.arange(len(rows)), target_indices] = -np.inf
        true_margin = (
            loo_scores[np.arange(len(rows)), target_indices]
            - np.max(masked, axis=1)
        )
        correct_margin = true_margin[correct]
        margin_threshold = (
            max(0.0, float(np.percentile(correct_margin, MARGIN_QUANTILE)))
            if len(correct_margin)
            else 0.0
        )
        reliability = bool(
            loo_accuracy >= RELIABILITY_MIN_ACCURACY
            and old_accuracy >= RELIABILITY_MIN_GROUP_ACCURACY
            and new_accuracy >= RELIABILITY_MIN_GROUP_ACCURACY
            and len(correct_margin) > 0
        )
    state_digest = _canonical_digest(
        {
            "schema": "cvs.erbt_idr.m27.target_competition_state.v1",
            "classes": list(registry),
            "old_classes": list(old_registry),
            "reliability_accepted": reliability,
            "loo_accuracy": loo_accuracy,
            "old_loo_accuracy": old_accuracy,
            "new_loo_accuracy": new_accuracy,
            "margin_threshold": margin_threshold,
            "fallback_policy": fallback_policy,
        },
        [
            shared.astype(np.float32),
            centered_old.astype(np.float32),
            prototypes.astype(np.float32),
        ],
    )
    audit = {
        "schema": "cvs.erbt_idr.m27.target_competition_fit_audit.v1",
        "support_only": True,
        "query_rows_used": 0,
        "query_state_update": False,
        "target_shift_source": "CLASS_BALANCED_OLD_SUPPORT",
        "class_center_estimator": "COMPONENTWISE_MEDIAN",
        "old_class_count": int(len(old_registry)),
        "class_count": int(len(registry)),
        "k_shot": int(k_shot),
        "feature_dim": int(rows.shape[1]),
        "loo_accuracy": loo_accuracy,
        "old_loo_accuracy": old_accuracy,
        "new_loo_accuracy": new_accuracy,
        "margin_threshold": margin_threshold,
        "reliability_accepted": reliability,
        "fallback_policy": fallback_policy,
        "state_digest": state_digest,
    }
    model = TargetCenteredCompetitionModel(
        classes=registry,
        old_classes=old_registry,
        shared_target_center=shared.astype(np.float32),
        centered_old_class_centres=centered_old.astype(np.float32),
        class_prototypes=prototypes.astype(np.float32),
        reliability_accepted=reliability,
        loo_accuracy=loo_accuracy,
        old_loo_accuracy=old_accuracy,
        new_loo_accuracy=new_accuracy,
        margin_threshold=margin_threshold,
        state_digest=state_digest,
    )
    return model, MappingProxyType(audit)


def apply_consensus_veto(
    base_scores: Any,
    b3_scores: Any,
    representation_scores: Any,
    *,
    reliability_accepted: bool,
    margin_threshold: float,
) -> tuple[np.ndarray, Mapping[str, Any]]:
    """Select each complete score row from B0 or B3 without score fusion."""

    base = np.asarray(base_scores)
    branch = np.asarray(b3_scores)
    representation = np.asarray(representation_scores, dtype=np.float64)
    if (
        base.ndim != 2
        or base.shape[1] < 2
        or branch.shape != base.shape
        or representation.shape != base.shape
        or not np.isfinite(base).all()
        or not np.isfinite(branch).all()
        or not np.isfinite(representation).all()
        or float(margin_threshold) < 0.0
    ):
        raise M27SpectralVetoError("consensus score geometry drift")
    base_prediction = np.argmax(base, axis=1)
    branch_prediction = np.argmax(branch, axis=1)
    branch_flip = branch_prediction != base_prediction
    if not bool(reliability_accepted):
        selected = np.array(base, copy=True)
        return selected, MappingProxyType(
            {
                "fallback_reason": "SUPPORT_REPRESENTATION_UNRELIABLE",
                "query_count": int(len(base)),
                "b3_flip_count": int(np.sum(branch_flip)),
                "selected_b3_count": 0,
                "vetoed_b3_flip_count": int(np.sum(branch_flip)),
                "query_state_update": False,
                "row_source_allowlist": ["B0", "B3"],
            }
        )
    representation_prediction = np.argmax(representation, axis=1)
    ordered = np.partition(representation, -2, axis=1)
    representation_margin = ordered[:, -1] - ordered[:, -2]
    consensus = (
        (representation_prediction == branch_prediction)
        & (representation_margin >= float(margin_threshold))
    )
    select_branch = (~branch_flip) | consensus
    selected = np.where(select_branch[:, None], branch, base)
    vetoed = branch_flip & ~select_branch
    return selected, MappingProxyType(
        {
            "fallback_reason": None,
            "query_count": int(len(base)),
            "b3_flip_count": int(np.sum(branch_flip)),
            "selected_b3_count": int(np.sum(select_branch)),
            "accepted_b3_flip_count": int(np.sum(branch_flip & select_branch)),
            "vetoed_b3_flip_count": int(np.sum(vetoed)),
            "margin_threshold": float(margin_threshold),
            "query_state_update": False,
            "row_source_allowlist": ["B0", "B3"],
        }
    )


@dataclass(frozen=True)
class M27SpectralVetoState:
    classes: tuple[str, ...]
    arm: str
    base_state: M24InferenceState
    b3_state: M25AnchoredResidualState
    representation_model: TargetCenteredCompetitionModel
    domain_digest: str
    config_hash: str
    audit: Mapping[str, Any]

    def __post_init__(self) -> None:
        if (
            self.arm not in M27_SPECTRAL_VETO_ARMS
            or tuple(self.base_state.classes) != tuple(self.classes)
            or tuple(self.b3_state.classes) != tuple(self.classes)
            or tuple(self.representation_model.classes) != tuple(self.classes)
        ):
            raise M27SpectralVetoError("M2.7 inference state drift")
        object.__setattr__(self, "audit", MappingProxyType(dict(self.audit)))

    @property
    def feature_dim(self) -> int:
        return int(self.base_state.compiled_affine_state.feature_dim)

    @property
    def state_bytes(self) -> int:
        return int(self.b3_state.state_bytes + self.representation_model.state_bytes)

    def _representation(self, blocks: Any, phase32: Any | None) -> np.ndarray:
        if self.arm == V1:
            raw = _rows(blocks, name="M2.7 IF blocks")
            if raw.shape[1] < 256:
                raise M27SpectralVetoError("M2.7 IF blocks require FFT96")
            return fft_magnitude_geometry(raw[:, 160:256])
        if phase32 is None:
            raise M27SpectralVetoError("phase32 query view is required for V2")
        return _rows(phase32, name="phase32 query", width=32).astype(np.float32)

    def representation_features(
        self, blocks: Any, *, phase32: Any | None = None
    ) -> np.ndarray:
        return self._representation(blocks, phase32)

    def score_with_audit(
        self, blocks: Any, *, phase32: Any | None = None
    ) -> tuple[np.ndarray, Mapping[str, Any]]:
        physical = physical_if256(blocks)
        base_scores = self.base_state.score(physical)
        b3_scores = self.b3_state.score(blocks)
        if self.representation_model.reliability_accepted:
            representation_scores = self.representation_model.score(
                self._representation(blocks, phase32)
            )
        else:
            representation_scores = np.zeros_like(base_scores)
        return apply_consensus_veto(
            base_scores,
            b3_scores,
            representation_scores,
            reliability_accepted=self.representation_model.reliability_accepted,
            margin_threshold=self.representation_model.margin_threshold,
        )

    def score(self, blocks: Any, *, phase32: Any | None = None) -> np.ndarray:
        scores, _audit = self.score_with_audit(blocks, phase32=phase32)
        return scores

    def predict(self, blocks: Any, *, phase32: Any | None = None) -> np.ndarray:
        return np.asarray(self.classes)[np.argmax(self.score(blocks, phase32=phase32), axis=1)]


def fit_m27_spectral_veto(
    *,
    arm: str,
    base_state: M24InferenceState,
    support_blocks: Any,
    support_labels: Any,
    classes: Sequence[str],
    k_shot: int,
    old_class_count: int,
    domain_digest: str,
    support_phase32: Any | None = None,
) -> tuple[M27SpectralVetoState, Mapping[str, Any]]:
    if arm not in M27_SPECTRAL_VETO_ARMS:
        raise M27SpectralVetoError("unknown M2.7 spectral veto arm")
    blocks = _rows(support_blocks, name="M2.7 support blocks")
    labels = np.asarray(support_labels).astype(str)
    registry = tuple(str(item) for item in classes)
    b3_state, b3_audit = fit_m25_anchored_residual(
        arm=B3,
        base_state=base_state,
        support_blocks=blocks,
        support_labels=labels,
        classes=registry,
        k_shot=int(k_shot),
        old_class_count=int(old_class_count),
        domain_digest=str(domain_digest),
    )
    if arm == V1:
        if blocks.shape[1] < 256:
            raise M27SpectralVetoError("M2.7 MGD support requires FFT96")
        representation = fft_magnitude_geometry(blocks[:, 160:256])
        representation_name = "MGD96"
    else:
        if support_phase32 is None:
            raise M27SpectralVetoError("phase32 support view is required for V2")
        representation = _rows(
            support_phase32, name="phase32 support", width=32
        ).astype(np.float32)
        if len(representation) != len(blocks):
            raise M27SpectralVetoError("phase32 support row count drift")
        representation_name = "PHASE_CEPSTRAL32"
    model, representation_audit = fit_target_centered_competition(
        representation,
        labels,
        classes=registry,
        old_class_count=int(old_class_count),
        k_shot=int(k_shot),
    )
    resource = {
        "compiled_inference_state_bytes": int(
            b3_state.state_bytes + model.state_bytes
        ),
        "persistent_update_state_bytes": 0,
        "transient_registration_workspace_peak_bytes": int(
            representation.nbytes
        ),
    }
    audit = {
        "schema": "cvs.erbt_idr.m27.spectral_veto_fit_audit.v1",
        "arm": arm,
        "base_method": "P2-A1_NO_RF32_R1",
        "performance_branch": B3,
        "representation": representation_name,
        "selection_policy": "B3_FLIP_CONSENSUS_VETO_ONLY",
        "support_only": True,
        "query_rows_used": 0,
        "query_state_update": False,
        "k_shot": int(k_shot),
        "b3": dict(b3_audit),
        "representation_fit": dict(representation_audit),
        "quantization": dict(b3_audit["quantization"]),
        "resource": resource,
    }
    state = M27SpectralVetoState(
        classes=registry,
        arm=arm,
        base_state=base_state,
        b3_state=b3_state,
        representation_model=model,
        domain_digest=str(domain_digest),
        config_hash=m27_arm_config_hash(arm),
        audit=audit,
    )
    return state, MappingProxyType(audit)


__all__ = [
    "M27_SPECTRAL_VETO_ARMS",
    "M27SpectralVetoError",
    "M27SpectralVetoState",
    "TargetCenteredCompetitionModel",
    "V1",
    "V2",
    "apply_consensus_veto",
    "fit_m27_spectral_veto",
    "fit_target_centered_competition",
    "m27_arm_config_hash",
]
