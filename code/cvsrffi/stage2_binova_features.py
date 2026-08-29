"""Frozen feature and protocol objects for BiNOVA-D92.

Every row originates from one fixed received IQ. Support owns labels and ranks;
query deliberately has no label, role, quota, or scorer surface.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from cvsrffi.stage2_sf_erbt_oldonly import make_fft96


_REQUIRED_CONTEXT = (
    "protocol_schema",
    "phase2_data_status",
    "capsule_id",
    "split_id",
)
_FORBIDDEN_CONTEXT = (
    "query",
    "truth",
    "role",
    "quota",
    "scorer",
    "source",
    "clean",
)


class BiNOVAFeatureError(ValueError):
    """Raised when the BiNOVA feature or Phase2 contract drifts."""


def _readonly_float32(value: Any, *, rows: int, columns: int, name: str) -> np.ndarray:
    array = np.ascontiguousarray(value, dtype=np.float32)
    if array.shape != (rows, columns) or not np.isfinite(array).all():
        raise BiNOVAFeatureError(
            f"{name} must be a finite [{rows},{columns}] matrix"
        )
    immutable = np.frombuffer(array.tobytes(), dtype=np.float32).reshape(array.shape)
    return immutable


def _validate_context(context: Mapping[str, Any]) -> Mapping[str, Any]:
    if not isinstance(context, Mapping):
        raise BiNOVAFeatureError("Phase2 context must be a mapping")
    missing = [key for key in _REQUIRED_CONTEXT if key not in context]
    if missing:
        raise BiNOVAFeatureError(f"missing Phase2 context fields: {missing}")
    for raw_key in context:
        key = str(raw_key).strip().lower()
        if any(token in key for token in _FORBIDDEN_CONTEXT):
            raise BiNOVAFeatureError(f"forbidden Phase2 context field: {raw_key}")
    if str(context["protocol_schema"]) != "p2_min_v1":
        raise BiNOVAFeatureError("protocol_schema must be p2_min_v1")
    if str(context["phase2_data_status"]) != "VALIDATED_ONCE":
        raise BiNOVAFeatureError("phase2_data_status must be VALIDATED_ONCE")
    if not str(context["capsule_id"]).strip() or not str(context["split_id"]).strip():
        raise BiNOVAFeatureError("capsule_id and split_id must be nonempty")
    return MappingProxyType(dict(context))


@dataclass(frozen=True)
class BiNOVAFeatures:
    identity160: np.ndarray
    late_time160: np.ndarray
    domain160: np.ndarray
    fft96: np.ndarray
    physical6: np.ndarray
    physical_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        ids = tuple(str(value) for value in self.physical_ids)
        rows = len(ids)
        if rows < 1 or any(not value for value in ids):
            raise BiNOVAFeatureError("physical IDs must be nonempty")
        if len(set(ids)) != rows:
            raise BiNOVAFeatureError("physical IDs must be unique")
        object.__setattr__(self, "physical_ids", ids)
        for name, columns in (
            ("identity160", 160),
            ("late_time160", 160),
            ("domain160", 160),
            ("fft96", 96),
            ("physical6", 6),
        ):
            object.__setattr__(
                self,
                name,
                _readonly_float32(
                    getattr(self, name), rows=rows, columns=columns, name=name
                ),
            )

    @property
    def row_count(self) -> int:
        return len(self.physical_ids)


@dataclass(frozen=True)
class BiNOVASupport:
    features: BiNOVAFeatures
    labels: np.ndarray
    ranks: np.ndarray
    context: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.features, BiNOVAFeatures):
            raise BiNOVAFeatureError("support requires BiNOVAFeatures")
        rows = self.features.row_count
        labels = np.ascontiguousarray(self.labels, dtype=np.int64)
        ranks = np.ascontiguousarray(self.ranks, dtype=np.int64)
        if labels.shape != (rows,) or ranks.shape != (rows,):
            raise BiNOVAFeatureError("support labels/ranks must align with features")
        if np.any(labels < 0) or np.any(ranks < 0):
            raise BiNOVAFeatureError("support labels/ranks must be nonnegative")
        for class_id in np.unique(labels):
            class_ranks = ranks[labels == class_id]
            if len(set(class_ranks.tolist())) != len(class_ranks):
                raise BiNOVAFeatureError("support ranks must be unique within class")
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "ranks", ranks)
        object.__setattr__(self, "context", _validate_context(self.context))


@dataclass(frozen=True)
class BiNOVAQuery:
    features: BiNOVAFeatures
    context: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.features, BiNOVAFeatures):
            raise BiNOVAFeatureError("query requires BiNOVAFeatures")
        object.__setattr__(self, "context", _validate_context(self.context))


def class_balanced_domain_context(
    rows: Any,
    labels: Any,
    *,
    iterations: int = 32,
    epsilon: float = 1.0e-6,
) -> np.ndarray:
    """Return a geometric median over equal-weight class means."""

    values = np.asarray(rows, dtype=np.float64)
    targets = np.asarray(labels, dtype=np.int64)
    if (
        values.ndim != 2
        or len(values) < 1
        or targets.shape != (len(values),)
        or not np.isfinite(values).all()
        or np.any(targets < 0)
    ):
        raise BiNOVAFeatureError("domain context rows/labels are invalid")
    classes = np.unique(targets)
    means = np.stack([values[targets == class_id].mean(axis=0) for class_id in classes])
    estimate = np.median(means, axis=0)
    for _ in range(int(iterations)):
        distances = np.linalg.norm(means - estimate[None, :], axis=1)
        exact = np.flatnonzero(distances <= epsilon)
        if len(exact):
            estimate = means[int(exact[0])]
            break
        weights = 1.0 / np.maximum(distances, epsilon)
        updated = np.sum(means * weights[:, None], axis=0) / np.sum(weights)
        if np.linalg.norm(updated - estimate) <= epsilon:
            estimate = updated
            break
        estimate = updated
    return np.asarray(estimate, dtype=np.float32)


def _physical_statistics(received_iq: np.ndarray) -> np.ndarray:
    rows = np.asarray(received_iq, dtype=np.float64)
    complex_rows = rows[:, 0, :] + 1j * rows[:, 1, :]
    power = np.mean(np.abs(complex_rows) ** 2, axis=1)
    spectrum = np.abs(np.fft.fft(complex_rows, axis=1)) ** 2 + 1.0e-12
    probability = spectrum / np.sum(spectrum, axis=1, keepdims=True)
    entropy = -np.sum(probability * np.log(probability), axis=1) / np.log(
        spectrum.shape[1]
    )
    x_axis = np.linspace(-1.0, 1.0, spectrum.shape[1], dtype=np.float64)
    centered_x = x_axis - x_axis.mean()
    log_spectrum = np.log(spectrum)
    slope = np.sum(
        (log_spectrum - log_spectrum.mean(axis=1, keepdims=True))
        * centered_x[None, :],
        axis=1,
    ) / np.sum(centered_x**2)
    phase_delta = np.angle(complex_rows[:, 1:] * np.conj(complex_rows[:, :-1]))
    phase_mean = np.mean(phase_delta, axis=1)
    phase_variance = np.var(phase_delta, axis=1)
    i_abs = np.mean(np.abs(rows[:, 0, :]), axis=1)
    q_abs = np.mean(np.abs(rows[:, 1, :]), axis=1)
    imbalance = (i_abs - q_abs) / np.maximum(i_abs + q_abs, 1.0e-12)
    result = np.stack(
        [np.log1p(power), entropy, slope, phase_mean, phase_variance, imbalance],
        axis=1,
    )
    if not np.isfinite(result).all():
        raise BiNOVAFeatureError("physical statistics became non-finite")
    return np.asarray(result, dtype=np.float32)


def _to_numpy_2d(value: Any, rows: int, columns: int, name: str) -> np.ndarray:
    if not torch.is_tensor(value):
        raise BiNOVAFeatureError(f"model output lacks tensor {name}")
    result = np.asarray(value.detach().cpu().tolist(), dtype=np.float32)
    if result.shape != (rows, columns) or not np.isfinite(result).all():
        raise BiNOVAFeatureError(f"model output {name} geometry drift")
    return result


def extract_binova_features(
    model: torch.nn.Module,
    received_iq: torch.Tensor | np.ndarray,
    *,
    physical_ids: Sequence[str],
    device: str | torch.device,
) -> BiNOVAFeatures:
    """Extract BiNOVA features once from fixed received IQ."""

    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise BiNOVAFeatureError("feature extraction requires a frozen eval model")
    rows = np.ascontiguousarray(received_iq, dtype=np.float32)
    if rows.ndim != 3 or rows.shape[1:] != (2, 256) or not np.isfinite(rows).all():
        raise BiNOVAFeatureError("received IQ must be finite [N,2,256]")
    ids = tuple(str(value) for value in physical_ids)
    if len(ids) != len(rows):
        raise BiNOVAFeatureError("received IQ and physical IDs must align")
    target_device = torch.device(device)
    tensor = torch.tensor(rows.tolist(), dtype=torch.float32, device=target_device)
    from cvsrffi.target_only_progressive_adapt import _forward_aux

    with torch.inference_mode():
        outputs = _forward_aux(model, tensor)
    auxiliary = outputs.get("aux_id")
    if not isinstance(auxiliary, Mapping):
        raise BiNOVAFeatureError("model output lacks aux_id mapping")
    return BiNOVAFeatures(
        identity160=_to_numpy_2d(outputs.get("z_id"), len(rows), 160, "z_id"),
        late_time160=_to_numpy_2d(
            auxiliary.get("t_emb"), len(rows), 160, "aux_id.t_emb"
        ),
        domain160=_to_numpy_2d(outputs.get("z_dom"), len(rows), 160, "z_dom"),
        fft96=make_fft96(rows),
        physical6=_physical_statistics(rows),
        physical_ids=ids,
    )


__all__ = [
    "BiNOVAFeatureError",
    "BiNOVAFeatures",
    "BiNOVAQuery",
    "BiNOVASupport",
    "class_balanced_domain_context",
    "extract_binova_features",
]
