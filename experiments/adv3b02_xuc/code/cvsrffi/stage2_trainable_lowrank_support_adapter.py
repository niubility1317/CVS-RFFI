"""Trainable low-rank residual adapter for sealed Phase2 support only.

The module has no dataset, source, clean, query, truth, role, quota, or scorer
interface.  Every selectable quantity is derived from registered support
labels and physical support ranks.  Multiple post-reception views, when
provided, remain grouped by the same physical sample rank and never increase
K.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F


EPS = 1.0e-8
SCHEMA = "cvs.phase2.trainable_lowrank_support_adapter.v1"
MAX_PARAMETERS = 12_000
MAX_EPOCHS = 20
MAX_VIEWS = 3
MAX_STATE_BYTES = 256 * 1024


class TrainableLowRankAdapterError(ValueError):
    """Raised when the support-only or resource contract fails closed."""


_ARTIFACT_TOKEN = object()


class ValidatedFeatureArtifact:
    """Runtime/code/checkpoint-bound per-sample feature artifact."""

    def __init__(
        self,
        *,
        features_by_view: Mapping[str, np.ndarray],
        physical_sample_ids: Sequence[str],
        parent_received_iq_sha256: Sequence[str],
        sealed_runtime_sha256: str,
        feature_code_sha256: str,
        sealed_phase1_checkpoint_sha256: str,
        view_seed_by_id: Mapping[str, int],
        _token: object,
    ) -> None:
        if _token is not _ARTIFACT_TOKEN:
            raise TrainableLowRankAdapterError(
                "feature artifact must come from validated internal extraction"
            )
        view_ids = tuple(features_by_view)
        arrays = {
            name: np.ascontiguousarray(features_by_view[name], dtype=np.float32)
            for name in view_ids
        }
        ids = tuple(str(value) for value in physical_sample_ids)
        parents = tuple(str(value) for value in parent_received_iq_sha256)
        if (
            not view_ids
            or len(view_ids) > MAX_VIEWS
            or set(view_seed_by_id) != set(view_ids)
            or any(array.ndim != 2 or not np.isfinite(array).all() for array in arrays.values())
            or any(array.shape != next(iter(arrays.values())).shape for array in arrays.values())
            or len(ids) != len(parents)
            or len(ids) != len(next(iter(arrays.values())))
            or len(set(ids)) != len(ids)
            or len(set(parents)) != len(parents)
            or any(len(value) != 64 for value in parents)
            or any(
                not value or len(value) != 64
                for value in (
                    sealed_runtime_sha256,
                    feature_code_sha256,
                    sealed_phase1_checkpoint_sha256,
                )
            )
        ):
            raise TrainableLowRankAdapterError("validated feature artifact drift")
        for array in arrays.values():
            array.setflags(write=False)
        per_view_hashes = {
            name: [
                hashlib.sha256(np.ascontiguousarray(row).tobytes()).hexdigest()
                for row in array
            ]
            for name, array in arrays.items()
        }
        canonical = {
            "physical_sample_ids": ids,
            "parent_received_iq_sha256": parents,
            "sealed_runtime_sha256": sealed_runtime_sha256,
            "feature_code_sha256": feature_code_sha256,
            "sealed_phase1_checkpoint_sha256": sealed_phase1_checkpoint_sha256,
            "view_seed_by_id": dict(view_seed_by_id),
            "per_view_feature_sha256": per_view_hashes,
        }
        self.features_by_view = arrays
        self.physical_sample_ids = ids
        self.parent_received_iq_sha256 = parents
        self.sealed_runtime_sha256 = sealed_runtime_sha256
        self.feature_code_sha256 = feature_code_sha256
        self.sealed_phase1_checkpoint_sha256 = sealed_phase1_checkpoint_sha256
        self.view_seed_by_id = dict(view_seed_by_id)
        self.per_view_feature_sha256 = per_view_hashes
        self.artifact_sha256 = hashlib.sha256(
            json.dumps(
                canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=True
            ).encode("utf-8")
        ).hexdigest()


def _build_validated_feature_artifact_internal(
    received_iq: np.ndarray,
    *,
    physical_sample_ids: Sequence[str],
    parent_received_iq_sha256: Sequence[str],
    sealed_runtime_sha256: str,
    feature_code_sha256: str,
    sealed_phase1_checkpoint_sha256: str,
    view_seed_by_id: Mapping[str, int],
    extract_single_received_iq_view: Callable[[np.ndarray, str], np.ndarray],
) -> ValidatedFeatureArtifact:
    """Runner-internal factory; formal callers must not supply an extractor.

    The formal support runner owns the fixed sealed-runtime extractor and calls
    this private function.  There is intentionally no public callback-bearing
    artifact factory or public raw-feature wrapping API.
    """

    iq = np.asarray(received_iq, dtype=np.float32)
    parents = tuple(str(value) for value in parent_received_iq_sha256)
    if (
        iq.ndim != 3
        or iq.shape[1] != 2
        or not np.isfinite(iq).all()
        or len(iq) != len(parents)
    ):
        raise TrainableLowRankAdapterError("received-IQ artifact input drift")
    computed = tuple(
        hashlib.sha256(np.ascontiguousarray(row).tobytes()).hexdigest()
        for row in iq
    )
    if computed != parents:
        raise TrainableLowRankAdapterError("actual received-IQ SHA binding mismatch")
    features: dict[str, np.ndarray] = {}
    for view_id in view_seed_by_id:
        rows = []
        for row in iq:
            value = np.asarray(
                extract_single_received_iq_view(row[None, ...], view_id),
                dtype=np.float32,
            )
            if value.ndim != 2 or len(value) != 1 or not np.isfinite(value).all():
                raise TrainableLowRankAdapterError(
                    "authorized extractor must return one finite feature row"
                )
            rows.append(value[0])
        features[view_id] = np.stack(rows).astype(np.float32)
    return ValidatedFeatureArtifact(
        features_by_view=features,
        physical_sample_ids=physical_sample_ids,
        parent_received_iq_sha256=parent_received_iq_sha256,
        sealed_runtime_sha256=sealed_runtime_sha256,
        feature_code_sha256=feature_code_sha256,
        sealed_phase1_checkpoint_sha256=sealed_phase1_checkpoint_sha256,
        view_seed_by_id=view_seed_by_id,
        _token=_ARTIFACT_TOKEN,
    )


@dataclass(frozen=True)
class AdapterHyperparameters:
    candidate_id: str
    rank: int = 8
    epochs: int = 12
    learning_rate: float = 0.02
    temperature: float = 0.10
    prototype_weight: float = 1.0
    supervised_contrastive_weight: float = 0.25
    identity_weight: float = 5.0
    factor_weight: float = 0.02
    seed: int = 20260717


@dataclass(frozen=True)
class TrainableLowRankAdapterState:
    schema: str
    candidate_id: str
    classes: tuple[str, ...]
    prototypes: np.ndarray
    low_rank_u: np.ndarray
    low_rank_v: np.ndarray
    gate: np.ndarray
    hyperparameters: AdapterHyperparameters
    feature_dim: int
    k_shot: int
    view_ids: tuple[str, ...]
    old_class_count: int
    registration_generation: int
    resource: Mapping[str, Any]
    support_feature_artifact_sha256: str
    sealed_runtime_sha256: str
    feature_code_sha256: str
    sealed_phase1_checkpoint_sha256: str
    state_content_sha256: str = ""

    def __post_init__(self) -> None:
        for name in ("prototypes", "low_rank_u", "low_rank_v", "gate"):
            source = np.ascontiguousarray(getattr(self, name), dtype=np.float32)
            immutable = np.frombuffer(source.tobytes(), dtype=np.float32).reshape(
                source.shape
            )
            object.__setattr__(self, name, immutable)
        actual_state_bytes = int(
            self.prototypes.nbytes
            + self.low_rank_u.nbytes
            + self.low_rank_v.nbytes
            + self.gate.nbytes
        )
        if (
            actual_state_bytes > MAX_STATE_BYTES
            or int(self.resource.get("persistent_state_bytes", -1))
            != actual_state_bytes
            or _parameter_count(self.feature_dim, self.hyperparameters.rank)
            > MAX_PARAMETERS
        ):
            raise TrainableLowRankAdapterError("adapter state resource audit drift")
        computed = _compute_state_content_sha256(self)
        if self.state_content_sha256 and self.state_content_sha256 != computed:
            raise TrainableLowRankAdapterError("adapter state content SHA mismatch")
        object.__setattr__(self, "state_content_sha256", computed)


@dataclass(frozen=True)
class AdapterFitResult:
    state: TrainableLowRankAdapterState
    trace: tuple[dict[str, Any], ...]
    validation: Mapping[str, Any]


class _Adapter(torch.nn.Module):
    def __init__(self, feature_dim: int, rank: int) -> None:
        super().__init__()
        scale = 1.0 / max(float(feature_dim) ** 0.5, 1.0)
        self.u = torch.nn.Parameter(torch.randn(feature_dim, rank) * scale)
        self.v = torch.nn.Parameter(torch.randn(feature_dim, rank) * scale)
        self.gate = torch.nn.Parameter(torch.zeros(rank))

    def forward(self, rows: torch.Tensor) -> torch.Tensor:
        base = F.normalize(rows, dim=-1)
        residual = ((base @ self.v) * torch.tanh(self.gate)) @ self.u.T
        return F.normalize(base + residual, dim=-1)


def _parameter_count(feature_dim: int, rank: int) -> int:
    return int(2 * feature_dim * rank + rank)


def _artifact_features(value: ValidatedFeatureArtifact) -> Mapping[str, np.ndarray]:
    if not isinstance(value, ValidatedFeatureArtifact):
        raise TrainableLowRankAdapterError(
            "ordinary feature mappings are forbidden; validated artifact required"
        )
    return value.features_by_view


def _compute_state_content_sha256(
    state: TrainableLowRankAdapterState,
) -> str:
    digest = hashlib.sha256()
    digest.update(state.schema.encode("utf-8"))
    digest.update(state.candidate_id.encode("utf-8"))
    digest.update(json.dumps(state.classes, separators=(",", ":")).encode("utf-8"))
    for value in (
        state.prototypes,
        state.low_rank_u,
        state.low_rank_v,
        state.gate,
    ):
        digest.update(str(value.shape).encode("ascii"))
        digest.update(np.ascontiguousarray(value).tobytes())
    digest.update(
        json.dumps(
            {
                "feature_dim": state.feature_dim,
                "k_shot": state.k_shot,
                "view_ids": state.view_ids,
                "old_class_count": state.old_class_count,
                "registration_generation": state.registration_generation,
                "support_feature_artifact_sha256": (
                    state.support_feature_artifact_sha256
                ),
                "sealed_runtime_sha256": state.sealed_runtime_sha256,
                "feature_code_sha256": state.feature_code_sha256,
                "sealed_phase1_checkpoint_sha256": (
                    state.sealed_phase1_checkpoint_sha256
                ),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    return digest.hexdigest()


def _validate_state(state: TrainableLowRankAdapterState) -> None:
    actual_state_bytes = int(
        state.prototypes.nbytes
        + state.low_rank_u.nbytes
        + state.low_rank_v.nbytes
        + state.gate.nbytes
    )
    if (
        actual_state_bytes > MAX_STATE_BYTES
        or int(state.resource.get("persistent_state_bytes", -1))
        != actual_state_bytes
        or state.prototypes.shape != (len(state.classes), state.feature_dim)
        or state.low_rank_u.shape
        != (state.feature_dim, state.hyperparameters.rank)
        or state.low_rank_v.shape
        != (state.feature_dim, state.hyperparameters.rank)
        or state.gate.shape != (state.hyperparameters.rank,)
        or any(
            len(value) != 64
            for value in (
                state.support_feature_artifact_sha256,
                state.sealed_runtime_sha256,
                state.feature_code_sha256,
                state.sealed_phase1_checkpoint_sha256,
            )
        )
        or state.state_content_sha256 != _compute_state_content_sha256(state)
    ):
        raise TrainableLowRankAdapterError("adapter state resource/binding drift")


def _validate_hyperparameters(
    value: AdapterHyperparameters, *, feature_dim: int
) -> None:
    if not value.candidate_id:
        raise TrainableLowRankAdapterError("candidate_id must be non-empty")
    if value.rank not in (8, 16):
        raise TrainableLowRankAdapterError("rank must be preregistered as 8 or 16")
    if value.epochs < 1 or value.epochs > MAX_EPOCHS:
        raise TrainableLowRankAdapterError("adaptation epochs exceed locked bound")
    if _parameter_count(feature_dim, value.rank) > MAX_PARAMETERS:
        raise TrainableLowRankAdapterError("adapter parameter budget exceeded")
    scalars = (
        value.learning_rate,
        value.temperature,
        value.prototype_weight,
        value.supervised_contrastive_weight,
        value.identity_weight,
        value.factor_weight,
    )
    if not all(np.isfinite(item) and item >= 0.0 for item in scalars):
        raise TrainableLowRankAdapterError("hyperparameters must be finite/nonnegative")
    if value.learning_rate <= 0.0 or value.temperature <= 0.0:
        raise TrainableLowRankAdapterError("learning rate/temperature must be positive")


def _validate_support(
    features_by_view: Mapping[str, np.ndarray],
    labels: Sequence[str] | np.ndarray,
    ranks: Sequence[int] | np.ndarray,
    *,
    k_shot: int,
) -> tuple[tuple[str, ...], np.ndarray, np.ndarray, np.ndarray]:
    view_ids = tuple(features_by_view)
    if not view_ids or len(view_ids) > MAX_VIEWS or len(set(view_ids)) != len(view_ids):
        raise TrainableLowRankAdapterError("support requires one to three fixed-IQ views")
    arrays = [np.asarray(features_by_view[name], dtype=np.float32) for name in view_ids]
    if any(array.ndim != 2 or not np.isfinite(array).all() for array in arrays):
        raise TrainableLowRankAdapterError("support features must be finite matrices")
    if any(array.shape != arrays[0].shape for array in arrays[1:]):
        raise TrainableLowRankAdapterError("fixed-IQ view feature alignment drift")
    label_rows = np.asarray(labels).astype(str)
    rank_rows = np.asarray(ranks, dtype=np.int64)
    if len(label_rows) != len(arrays[0]) or len(rank_rows) != len(label_rows):
        raise TrainableLowRankAdapterError("support label/rank alignment drift")
    classes, counts = np.unique(label_rows, return_counts=True)
    if (
        int(k_shot) < 1
        or len(classes) < 2
        or set(counts.tolist()) != {int(k_shot)}
        or any(
            set(rank_rows[label_rows == label].tolist()) != set(range(int(k_shot)))
            for label in classes
        )
    ):
        raise TrainableLowRankAdapterError("strict physical K-shot support drift")
    stacked = np.stack(arrays, axis=0)
    return view_ids, stacked, label_rows, rank_rows


def _class_indices(labels: np.ndarray) -> tuple[tuple[str, ...], np.ndarray]:
    classes = tuple(sorted(set(labels.tolist())))
    lookup = {label: index for index, label in enumerate(classes)}
    return classes, np.asarray([lookup[label] for label in labels], dtype=np.int64)


def _adapt_numpy(
    rows: np.ndarray, u: np.ndarray, v: np.ndarray, gate: np.ndarray
) -> np.ndarray:
    base = rows / np.maximum(np.linalg.norm(rows, axis=-1, keepdims=True), EPS)
    residual = ((base @ v) * np.tanh(gate)) @ u.T
    values = base + residual
    return values / np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), EPS)


def _prototypes(
    adapted_views: np.ndarray, labels: np.ndarray, classes: Sequence[str]
) -> np.ndarray:
    mean_views = np.mean(adapted_views, axis=0)
    rows = np.stack([np.mean(mean_views[labels == label], axis=0) for label in classes])
    return rows / np.maximum(np.linalg.norm(rows, axis=1, keepdims=True), EPS)


def _metrics(
    adapted_views: np.ndarray,
    labels: np.ndarray,
    classes: Sequence[str],
    prototypes: np.ndarray,
) -> dict[str, Any]:
    query = np.mean(adapted_views, axis=0)
    query /= np.maximum(np.linalg.norm(query, axis=1, keepdims=True), EPS)
    scores = query @ prototypes.T
    predictions = np.asarray(classes)[np.argmax(scores, axis=1)]
    per_class = {
        label: float(np.mean(predictions[labels == label] == label))
        for label in classes
    }
    return {
        "overall_accuracy": float(np.mean(predictions == labels)),
        "min_class_accuracy": float(min(per_class.values())),
        "per_class_accuracy": per_class,
    }


def _supcon_loss(features: torch.Tensor, targets: torch.Tensor, temperature: float) -> torch.Tensor:
    logits = features @ features.T / temperature
    mask_self = torch.eye(len(features), dtype=torch.bool, device=features.device)
    positive = targets[:, None].eq(targets[None, :]) & ~mask_self
    logits = logits.masked_fill(mask_self, -1.0e9)
    log_probability = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    positive_count = positive.sum(dim=1).clamp_min(1)
    return -((log_probability * positive).sum(dim=1) / positive_count).mean()


def _tensor_from_numpy_compatible(
    value: np.ndarray,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Bridge NumPy 2.x arrays into PyTorch 2.1 without its legacy C-API."""

    if dtype == torch.float32:
        rows = np.ascontiguousarray(value, dtype=np.float32)
    elif dtype == torch.long:
        rows = np.ascontiguousarray(value, dtype=np.int64)
    else:
        raise TrainableLowRankAdapterError("unsupported tensor bridge dtype")
    return (
        torch.frombuffer(rows, dtype=dtype)
        .reshape(rows.shape)
        .clone()
        .to(device)
    )


def _numpy_float32_compatible(value: torch.Tensor) -> np.ndarray:
    """Return a detached float32 array without Tensor.numpy()."""

    return np.asarray(value.detach().cpu().tolist(), dtype=np.float32)


def _train_once(
    features_by_view: Mapping[str, np.ndarray],
    labels: np.ndarray,
    ranks: np.ndarray,
    *,
    k_shot: int,
    hyperparameters: AdapterHyperparameters,
    device: torch.device,
    trace_context: Mapping[str, Any],
    validation_features_by_view: Mapping[str, np.ndarray] | None = None,
    validation_labels: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[dict[str, Any], ...]]:
    _, stacked, labels, _ = _validate_support(
        features_by_view, labels, ranks, k_shot=k_shot
    )
    feature_dim = int(stacked.shape[-1])
    _validate_hyperparameters(hyperparameters, feature_dim=feature_dim)
    classes, targets = _class_indices(labels)
    torch.manual_seed(int(hyperparameters.seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(hyperparameters.seed))
    adapter = _Adapter(feature_dim, hyperparameters.rank).to(device)
    optimizer = torch.optim.Adam(
        adapter.parameters(), lr=hyperparameters.learning_rate
    )
    tensor = _tensor_from_numpy_compatible(
        stacked,
        dtype=torch.float32,
        device=device,
    )
    target_tensor = _tensor_from_numpy_compatible(
        targets,
        dtype=torch.long,
        device=device,
    )
    trace: list[dict[str, Any]] = []
    for epoch in range(1, hyperparameters.epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        adapted = adapter(tensor.reshape(-1, feature_dim)).reshape(tensor.shape)
        mean_sample = F.normalize(adapted.mean(dim=0), dim=-1)
        prototype_rows = torch.stack(
            [mean_sample[target_tensor == index].mean(dim=0) for index in range(len(classes))]
        )
        prototype_rows = F.normalize(prototype_rows, dim=-1)
        proto_loss = F.cross_entropy(
            mean_sample @ prototype_rows.T / hyperparameters.temperature,
            target_tensor,
        )
        all_targets = target_tensor.repeat(len(stacked))
        contrastive_loss = _supcon_loss(
            adapted.reshape(-1, feature_dim),
            all_targets,
            hyperparameters.temperature,
        )
        base = F.normalize(tensor, dim=-1)
        identity_loss = (adapted - base).square().mean()
        factor_loss = (
            adapter.u.square().mean()
            + adapter.v.square().mean()
            + adapter.gate.square().mean()
        )
        loss = (
            hyperparameters.prototype_weight * proto_loss
            + hyperparameters.supervised_contrastive_weight * contrastive_loss
            + hyperparameters.identity_weight * identity_loss
            + hyperparameters.factor_weight * factor_loss
        )
        loss.backward()
        optimizer.step()
        u_now = _numpy_float32_compatible(adapter.u)
        v_now = _numpy_float32_compatible(adapter.v)
        gate_now = _numpy_float32_compatible(adapter.gate)
        train_adapted = np.stack(
            [
                _adapt_numpy(np.asarray(features_by_view[name]), u_now, v_now, gate_now)
                for name in features_by_view
            ]
        )
        train_prototypes = _prototypes(train_adapted, labels, classes)
        train_metric = _metrics(train_adapted, labels, classes, train_prototypes)
        loo_metric = None
        if validation_features_by_view is not None and validation_labels is not None:
            validation_adapted = np.stack(
                [
                    _adapt_numpy(
                        np.asarray(validation_features_by_view[name]),
                        u_now,
                        v_now,
                        gate_now,
                    )
                    for name in features_by_view
                ]
            )
            loo_metric = _metrics(
                validation_adapted,
                np.asarray(validation_labels).astype(str),
                classes,
                train_prototypes,
            )
        trace.append(
            {
                **dict(trace_context),
                "candidate_id": hyperparameters.candidate_id,
                "rank": hyperparameters.rank,
                "seed": hyperparameters.seed,
                "learning_rate": hyperparameters.learning_rate,
                "temperature": hyperparameters.temperature,
                "prototype_weight": hyperparameters.prototype_weight,
                "supervised_contrastive_weight": (
                    hyperparameters.supervised_contrastive_weight
                ),
                "identity_weight": hyperparameters.identity_weight,
                "factor_weight": hyperparameters.factor_weight,
                "epoch": epoch,
                "total_loss": float(loss.detach().cpu()),
                "prototype_loss": float(proto_loss.detach().cpu()),
                "supervised_contrastive_loss": float(contrastive_loss.detach().cpu()),
                "identity_loss": float(identity_loss.detach().cpu()),
                "factor_loss": float(factor_loss.detach().cpu()),
                "gate_linf": float(torch.tanh(adapter.gate).abs().max().detach().cpu()),
                "support_overall_accuracy": train_metric["overall_accuracy"],
                "support_floor_accuracy": train_metric["min_class_accuracy"],
                "support_per_class_accuracy": train_metric["per_class_accuracy"],
                "loo_overall_accuracy": (
                    None if loo_metric is None else loo_metric["overall_accuracy"]
                ),
                "loo_floor_accuracy": (
                    None if loo_metric is None else loo_metric["min_class_accuracy"]
                ),
                "loo_per_class_accuracy": (
                    None if loo_metric is None else loo_metric["per_class_accuracy"]
                ),
            }
        )
    return (
        _numpy_float32_compatible(adapter.u),
        _numpy_float32_compatible(adapter.v),
        _numpy_float32_compatible(adapter.gate),
        tuple(trace),
    )


def _leave_two_out_masks(labels: np.ndarray, ranks: np.ndarray) -> tuple[np.ndarray, ...]:
    masks = []
    for first in range(0, 10, 2):
        held = np.isin(ranks, (first, first + 1))
        if any(np.sum(held & (labels == label)) != 2 for label in np.unique(labels)):
            raise TrainableLowRankAdapterError("leave-two-out physical fold drift")
        masks.append(held)
    return tuple(masks)


def _identity_validation(
    stacked: np.ndarray, labels: np.ndarray, ranks: np.ndarray
) -> dict[str, Any]:
    classes = tuple(sorted(set(labels.tolist())))
    folds = []
    for fold_index, held in enumerate(_leave_two_out_masks(labels, ranks)):
        prototypes = _prototypes(stacked[:, ~held], labels[~held], classes)
        metric = _metrics(stacked[:, held], labels[held], classes, prototypes)
        folds.append({"fold": fold_index, **metric})
    return _aggregate_fold_metrics(folds, classes)


def _aggregate_fold_metrics(
    folds: Sequence[Mapping[str, Any]], classes: Sequence[str]
) -> dict[str, Any]:
    per_class = {
        label: float(np.mean([row["per_class_accuracy"][label] for row in folds]))
        for label in classes
    }
    return {
        "overall_accuracy": float(np.mean([row["overall_accuracy"] for row in folds])),
        "min_class_accuracy": float(min(per_class.values())),
        "per_class_accuracy": per_class,
        "folds": [dict(row) for row in folds],
    }


def select_and_fit_k10(
    feature_artifact: ValidatedFeatureArtifact,
    labels: Sequence[str] | np.ndarray,
    ranks: Sequence[int] | np.ndarray,
    *,
    candidates: Sequence[AdapterHyperparameters],
    device: torch.device | str = "cpu",
    require_gate: bool = True,
) -> AdapterFitResult:
    """Select on physical K10 leave-two-out folds, then fit all K10 support."""

    features_by_view = _artifact_features(feature_artifact)
    view_ids, stacked, label_rows, rank_rows = _validate_support(
        features_by_view, labels, ranks, k_shot=10
    )
    if not candidates:
        raise TrainableLowRankAdapterError("at least one preregistered candidate required")
    classes = tuple(sorted(set(label_rows.tolist())))
    identity = _identity_validation(stacked, label_rows, rank_rows)
    candidate_rows: list[dict[str, Any]] = []
    all_trace: list[dict[str, Any]] = []
    target_device = torch.device(device)
    for candidate in candidates:
        folds: list[dict[str, Any]] = []
        for fold_index, held in enumerate(_leave_two_out_masks(label_rows, rank_rows)):
            train_views = {
                name: np.asarray(features_by_view[name])[~held] for name in view_ids
            }
            train_labels = label_rows[~held]
            train_ranks = np.tile(
                np.arange(8, dtype=np.int64), len(classes)
            )
            u, v, gate, trace = _train_once(
                train_views,
                train_labels,
                train_ranks,
                k_shot=8,
                hyperparameters=candidate,
                device=target_device,
                trace_context={"phase": "selection", "fold": fold_index},
                validation_features_by_view={
                    name: np.asarray(features_by_view[name])[held] for name in view_ids
                },
                validation_labels=label_rows[held],
            )
            all_trace.extend(trace)
            held_views = np.stack(
                [_adapt_numpy(np.asarray(features_by_view[name])[held], u, v, gate) for name in view_ids]
            )
            train_adapted = np.stack(
                [_adapt_numpy(np.asarray(train_views[name]), u, v, gate) for name in view_ids]
            )
            prototypes = _prototypes(train_adapted, train_labels, classes)
            metric = _metrics(held_views, label_rows[held], classes, prototypes)
            folds.append({"fold": fold_index, **metric})
        aggregate = _aggregate_fold_metrics(folds, classes)
        per_class_gate = all(
            aggregate["per_class_accuracy"][label]
            + 1.0e-12
            >= identity["per_class_accuracy"][label]
            for label in classes
        )
        gate_pass = (
            aggregate["overall_accuracy"] + 1.0e-12 >= identity["overall_accuracy"]
            and aggregate["min_class_accuracy"] + 1.0e-12
            >= identity["min_class_accuracy"]
            and per_class_gate
        )
        candidate_rows.append(
            {
                "candidate_id": candidate.candidate_id,
                "gate_pass": gate_pass,
                **aggregate,
            }
        )
    passing = [row for row in candidate_rows if row["gate_pass"]]
    if not passing and require_gate:
        raise TrainableLowRankAdapterError(
            "no trainable candidate passes overall/per-class/floor identity gates"
        )
    selectable = passing if passing else candidate_rows
    selectable.sort(
        key=lambda row: (
            row["min_class_accuracy"],
            row["overall_accuracy"],
            row["candidate_id"],
        ),
        reverse=True,
    )
    selected_id = str(selectable[0]["candidate_id"])
    selected = next(item for item in candidates if item.candidate_id == selected_id)
    u, v, gate, fit_trace = _train_once(
        features_by_view,
        label_rows,
        rank_rows,
        k_shot=10,
        hyperparameters=selected,
        device=target_device,
        trace_context={"phase": "full_fit", "fold": None},
    )
    all_trace.extend(fit_trace)
    adapted = np.stack(
        [_adapt_numpy(np.asarray(features_by_view[name]), u, v, gate) for name in view_ids]
    )
    prototypes = _prototypes(adapted, label_rows, classes)
    parameters = _parameter_count(int(stacked.shape[-1]), selected.rank)
    state_bytes = int(u.nbytes + v.nbytes + gate.nbytes + prototypes.nbytes)
    if state_bytes > MAX_STATE_BYTES:
        raise TrainableLowRankAdapterError("persistent state budget exceeded")
    resource = {
        "trainable_parameters": parameters,
        "adapt_epochs": selected.epochs,
        "persistent_state_bytes": state_bytes,
        "adapter_mac_per_view": parameters,
        "backbone_forwards_per_physical_sample": len(view_ids),
        "max_views": MAX_VIEWS,
    }
    state = TrainableLowRankAdapterState(
        schema=SCHEMA,
        candidate_id=selected_id,
        classes=classes,
        prototypes=prototypes.astype(np.float32),
        low_rank_u=u,
        low_rank_v=v,
        gate=gate,
        hyperparameters=selected,
        feature_dim=int(stacked.shape[-1]),
        k_shot=10,
        view_ids=view_ids,
        old_class_count=len(classes),
        registration_generation=0,
        resource=resource,
        support_feature_artifact_sha256=feature_artifact.artifact_sha256,
        sealed_runtime_sha256=feature_artifact.sealed_runtime_sha256,
        feature_code_sha256=feature_artifact.feature_code_sha256,
        sealed_phase1_checkpoint_sha256=(
            feature_artifact.sealed_phase1_checkpoint_sha256
        ),
    )
    return AdapterFitResult(
        state=state,
        trace=tuple(all_trace),
        validation={
            "selection_policy": "physical_sample_leave_two_out_k10",
            "identity_baseline": identity,
            "candidates": candidate_rows,
            "selected_candidate_id": selected_id,
            "selection_gate_pass": bool(passing),
            "full_support": _metrics(adapted, label_rows, classes, prototypes),
        },
    )


def fit_locked(
    feature_artifact: ValidatedFeatureArtifact,
    labels: Sequence[str] | np.ndarray,
    ranks: Sequence[int] | np.ndarray,
    *,
    k_shot: int,
    hyperparameters: AdapterHyperparameters,
    device: torch.device | str = "cpu",
) -> AdapterFitResult:
    """Fit locked K10-selected settings on one independently sealed K package."""

    features_by_view = _artifact_features(feature_artifact)
    view_ids, stacked, label_rows, rank_rows = _validate_support(
        features_by_view, labels, ranks, k_shot=k_shot
    )
    classes = tuple(sorted(set(label_rows.tolist())))
    u, v, gate, trace = _train_once(
        features_by_view,
        label_rows,
        rank_rows,
        k_shot=k_shot,
        hyperparameters=hyperparameters,
        device=torch.device(device),
        trace_context={"phase": "locked_fit", "fold": None},
    )
    adapted = np.stack(
        [_adapt_numpy(np.asarray(features_by_view[name]), u, v, gate) for name in view_ids]
    )
    prototypes = _prototypes(adapted, label_rows, classes)
    parameters = _parameter_count(int(stacked.shape[-1]), hyperparameters.rank)
    state_bytes = int(u.nbytes + v.nbytes + gate.nbytes + prototypes.nbytes)
    if state_bytes > MAX_STATE_BYTES:
        raise TrainableLowRankAdapterError("persistent state budget exceeded")
    state = TrainableLowRankAdapterState(
        schema=SCHEMA,
        candidate_id=hyperparameters.candidate_id,
        classes=classes,
        prototypes=prototypes.astype(np.float32),
        low_rank_u=u,
        low_rank_v=v,
        gate=gate,
        hyperparameters=hyperparameters,
        feature_dim=int(stacked.shape[-1]),
        k_shot=int(k_shot),
        view_ids=view_ids,
        old_class_count=len(classes),
        registration_generation=0,
        resource={
            "trainable_parameters": parameters,
            "adapt_epochs": hyperparameters.epochs,
            "persistent_state_bytes": state_bytes,
            "adapter_mac_per_view": parameters,
            "backbone_forwards_per_physical_sample": len(view_ids),
            "max_views": MAX_VIEWS,
        },
        support_feature_artifact_sha256=feature_artifact.artifact_sha256,
        sealed_runtime_sha256=feature_artifact.sealed_runtime_sha256,
        feature_code_sha256=feature_artifact.feature_code_sha256,
        sealed_phase1_checkpoint_sha256=(
            feature_artifact.sealed_phase1_checkpoint_sha256
        ),
    )
    return AdapterFitResult(
        state=state,
        trace=trace,
        validation={
            "selection_policy": "locked_from_k10_no_local_selection",
            "full_support": _metrics(adapted, label_rows, classes, prototypes),
        },
    )


def register_new_classes(
    before: TrainableLowRankAdapterState,
    feature_artifact: ValidatedFeatureArtifact,
    labels: Sequence[str] | np.ndarray,
    ranks: Sequence[int] | np.ndarray,
    *,
    k_shot: int,
    expected_old_support_feature_artifact_sha256: str,
) -> TrainableLowRankAdapterState:
    """Freeze the before adapter/prototypes and append new class prototypes."""

    _validate_state(before)
    features_by_view = _artifact_features(feature_artifact)
    if (
        expected_old_support_feature_artifact_sha256
        != before.support_feature_artifact_sha256
    ):
        raise TrainableLowRankAdapterError("old support feature fingerprint mismatch")
    if (
        feature_artifact.sealed_runtime_sha256 != before.sealed_runtime_sha256
        or feature_artifact.feature_code_sha256 != before.feature_code_sha256
        or feature_artifact.sealed_phase1_checkpoint_sha256
        != before.sealed_phase1_checkpoint_sha256
    ):
        raise TrainableLowRankAdapterError("registration artifact binding mismatch")
    view_ids, _, label_rows, _ = _validate_support(
        features_by_view, labels, ranks, k_shot=k_shot
    )
    if view_ids != before.view_ids:
        raise TrainableLowRankAdapterError("registration view schema drift")
    new_classes = tuple(sorted(set(label_rows.tolist()) - set(before.classes)))
    if not new_classes:
        raise TrainableLowRankAdapterError("registration contains no absent class")
    if any(label in before.classes for label in label_rows):
        raise TrainableLowRankAdapterError(
            "after registration input must contain absent classes only"
        )
    adapted = np.stack(
        [
            _adapt_numpy(
                np.asarray(features_by_view[name]),
                before.low_rank_u,
                before.low_rank_v,
                before.gate,
            )
            for name in view_ids
        ]
    )
    appended = _prototypes(adapted, label_rows, new_classes)
    prototypes = np.concatenate([before.prototypes, appended], axis=0).astype(np.float32)
    resource = dict(before.resource)
    resource["persistent_state_bytes"] = int(
        before.low_rank_u.nbytes
        + before.low_rank_v.nbytes
        + before.gate.nbytes
        + prototypes.nbytes
    )
    if resource["persistent_state_bytes"] > MAX_STATE_BYTES:
        raise TrainableLowRankAdapterError("registered persistent state budget exceeded")
    return TrainableLowRankAdapterState(
        schema=before.schema,
        candidate_id=before.candidate_id,
        classes=before.classes + new_classes,
        prototypes=prototypes,
        low_rank_u=before.low_rank_u.copy(),
        low_rank_v=before.low_rank_v.copy(),
        gate=before.gate.copy(),
        hyperparameters=before.hyperparameters,
        feature_dim=before.feature_dim,
        k_shot=int(k_shot),
        view_ids=before.view_ids,
        old_class_count=len(before.classes),
        registration_generation=before.registration_generation + 1,
        resource=resource,
        support_feature_artifact_sha256=before.support_feature_artifact_sha256,
        sealed_runtime_sha256=before.sealed_runtime_sha256,
        feature_code_sha256=before.feature_code_sha256,
        sealed_phase1_checkpoint_sha256=before.sealed_phase1_checkpoint_sha256,
    )


def evaluate_registration_leave_two_out(
    before: TrainableLowRankAdapterState,
    feature_artifact: ValidatedFeatureArtifact,
    labels: Sequence[str] | np.ndarray,
    ranks: Sequence[int] | np.ndarray,
    *,
    k_shot: int = 10,
) -> dict[str, Any]:
    """Evaluate absent-class registration using frozen old state and L2O support."""

    _validate_state(before)
    features_by_view = _artifact_features(feature_artifact)
    if (
        feature_artifact.sealed_runtime_sha256 != before.sealed_runtime_sha256
        or feature_artifact.feature_code_sha256 != before.feature_code_sha256
        or feature_artifact.sealed_phase1_checkpoint_sha256
        != before.sealed_phase1_checkpoint_sha256
    ):
        raise TrainableLowRankAdapterError("registration audit binding mismatch")
    view_ids, _, label_rows, rank_rows = _validate_support(
        features_by_view, labels, ranks, k_shot=k_shot
    )
    if int(k_shot) != 10:
        raise TrainableLowRankAdapterError("registration selection audit is K10-only")
    if view_ids != before.view_ids or set(label_rows.tolist()).intersection(before.classes):
        raise TrainableLowRankAdapterError("registration audit requires absent classes only")
    new_classes = tuple(sorted(set(label_rows.tolist())))
    all_classes = before.classes + new_classes
    adapted = np.stack(
        [
            _adapt_numpy(
                np.asarray(features_by_view[name]),
                before.low_rank_u,
                before.low_rank_v,
                before.gate,
            )
            for name in view_ids
        ]
    )
    folds: list[dict[str, Any]] = []
    for fold_index, held in enumerate(_leave_two_out_masks(label_rows, rank_rows)):
        new_prototypes = _prototypes(
            adapted[:, ~held], label_rows[~held], new_classes
        )
        prototypes = np.concatenate([before.prototypes, new_prototypes], axis=0)
        mean_held = np.mean(adapted[:, held], axis=0)
        mean_held /= np.maximum(
            np.linalg.norm(mean_held, axis=1, keepdims=True), EPS
        )
        predicted = np.asarray(all_classes)[
            np.argmax(mean_held @ prototypes.T, axis=1)
        ]
        held_labels = label_rows[held]
        per_class = {
            label: float(np.mean(predicted[held_labels == label] == label))
            for label in new_classes
        }
        folds.append(
            {
                "fold": fold_index,
                "overall_accuracy": float(np.mean(predicted == held_labels)),
                "min_class_accuracy": float(min(per_class.values())),
                "per_class_accuracy": per_class,
            }
        )
    return _aggregate_fold_metrics(folds, new_classes)


def _prediction_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    classes: Sequence[str],
    evaluated_classes: Sequence[str],
) -> dict[str, Any]:
    predictions = np.asarray(classes)[np.argmax(scores, axis=1)]
    per_class = {
        label: float(np.mean(predictions[labels == label] == label))
        for label in evaluated_classes
    }
    return {
        "overall_accuracy": float(np.mean(predictions == labels)),
        "min_class_accuracy": float(min(per_class.values())),
        "per_class_accuracy": per_class,
    }


def evaluate_joint_registration_leave_two_out(
    old_feature_artifact: ValidatedFeatureArtifact,
    old_labels: Sequence[str] | np.ndarray,
    old_ranks: Sequence[int] | np.ndarray,
    new_feature_artifact: ValidatedFeatureArtifact,
    new_labels: Sequence[str] | np.ndarray,
    new_ranks: Sequence[int] | np.ndarray,
    *,
    hyperparameters: AdapterHyperparameters,
    device: torch.device | str = "cpu",
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    """Joint old/new K10 L2O with no held row entering adapter or prototype fit."""

    old_features_by_view = _artifact_features(old_feature_artifact)
    new_features_by_view = _artifact_features(new_feature_artifact)
    if (
        old_feature_artifact.sealed_runtime_sha256
        != new_feature_artifact.sealed_runtime_sha256
        or old_feature_artifact.feature_code_sha256
        != new_feature_artifact.feature_code_sha256
        or old_feature_artifact.sealed_phase1_checkpoint_sha256
        != new_feature_artifact.sealed_phase1_checkpoint_sha256
    ):
        raise TrainableLowRankAdapterError("joint feature artifact binding mismatch")
    old_view_ids, old_stacked, old_label_rows, old_rank_rows = _validate_support(
        old_features_by_view, old_labels, old_ranks, k_shot=10
    )
    new_view_ids, new_stacked, new_label_rows, new_rank_rows = _validate_support(
        new_features_by_view, new_labels, new_ranks, k_shot=10
    )
    if old_view_ids != new_view_ids:
        raise TrainableLowRankAdapterError("joint registration view schema drift")
    old_classes = tuple(sorted(set(old_label_rows.tolist())))
    new_classes = tuple(sorted(set(new_label_rows.tolist())))
    if set(old_classes).intersection(new_classes):
        raise TrainableLowRankAdapterError("joint registration class sets overlap")
    all_classes = old_classes + new_classes
    fold_rows: list[dict[str, Any]] = []
    trace: list[dict[str, Any]] = []
    old_masks = _leave_two_out_masks(old_label_rows, old_rank_rows)
    new_masks = _leave_two_out_masks(new_label_rows, new_rank_rows)
    target_device = torch.device(device)
    for fold_index, (old_held, new_held) in enumerate(zip(old_masks, new_masks)):
        old_train = {
            name: np.asarray(old_features_by_view[name])[~old_held]
            for name in old_view_ids
        }
        old_validation = {
            name: np.asarray(old_features_by_view[name])[old_held]
            for name in old_view_ids
        }
        old_train_labels = old_label_rows[~old_held]
        remapped_ranks = np.tile(np.arange(8, dtype=np.int64), len(old_classes))
        u, v, gate, fold_trace = _train_once(
            old_train,
            old_train_labels,
            remapped_ranks,
            k_shot=8,
            hyperparameters=hyperparameters,
            device=target_device,
            trace_context={"phase": "joint_registration_l2o", "fold": fold_index},
            validation_features_by_view=old_validation,
            validation_labels=old_label_rows[old_held],
        )
        trace.extend(fold_trace)
        adapted_old_train = np.stack(
            [
                _adapt_numpy(np.asarray(old_train[name]), u, v, gate)
                for name in old_view_ids
            ]
        )
        adapted_old_held = np.stack(
            [
                _adapt_numpy(
                    np.asarray(old_features_by_view[name])[old_held], u, v, gate
                )
                for name in old_view_ids
            ]
        )
        adapted_new_train = np.stack(
            [
                _adapt_numpy(
                    np.asarray(new_features_by_view[name])[~new_held], u, v, gate
                )
                for name in old_view_ids
            ]
        )
        adapted_new_held = np.stack(
            [
                _adapt_numpy(
                    np.asarray(new_features_by_view[name])[new_held], u, v, gate
                )
                for name in old_view_ids
            ]
        )
        old_prototypes = _prototypes(
            adapted_old_train, old_train_labels, old_classes
        )
        new_prototypes = _prototypes(
            adapted_new_train, new_label_rows[~new_held], new_classes
        )
        all_prototypes = np.concatenate([old_prototypes, new_prototypes], axis=0)
        old_held_mean = np.mean(adapted_old_held, axis=0)
        old_held_mean /= np.maximum(
            np.linalg.norm(old_held_mean, axis=1, keepdims=True), EPS
        )
        new_held_mean = np.mean(adapted_new_held, axis=0)
        new_held_mean /= np.maximum(
            np.linalg.norm(new_held_mean, axis=1, keepdims=True), EPS
        )
        before_adapter = _prediction_metrics(
            old_held_mean @ old_prototypes.T,
            old_label_rows[old_held],
            old_classes,
            old_classes,
        )
        after_old = _prediction_metrics(
            old_held_mean @ all_prototypes.T,
            old_label_rows[old_held],
            all_classes,
            old_classes,
        )
        after_new = _prediction_metrics(
            new_held_mean @ all_prototypes.T,
            new_label_rows[new_held],
            all_classes,
            new_classes,
        )
        identity_old_train = old_stacked[:, ~old_held]
        identity_old_held = np.mean(old_stacked[:, old_held], axis=0)
        identity_old_held /= np.maximum(
            np.linalg.norm(identity_old_held, axis=1, keepdims=True), EPS
        )
        identity_prototypes = _prototypes(
            identity_old_train, old_train_labels, old_classes
        )
        before_identity = _prediction_metrics(
            identity_old_held @ identity_prototypes.T,
            old_label_rows[old_held],
            old_classes,
            old_classes,
        )
        harmonic = (
            0.0
            if after_old["overall_accuracy"] + after_new["overall_accuracy"] <= 0.0
            else 2.0
            * after_old["overall_accuracy"]
            * after_new["overall_accuracy"]
            / (after_old["overall_accuracy"] + after_new["overall_accuracy"])
        )
        fold_rows.append(
            {
                "fold": fold_index,
                "before_identity_old": before_identity,
                "before_adapter_old": before_adapter,
                "after_old": after_old,
                "after_new": after_new,
                "joint_accuracy": float(
                    (
                        after_old["overall_accuracy"] * int(np.sum(old_held))
                        + after_new["overall_accuracy"] * int(np.sum(new_held))
                    )
                    / (int(np.sum(old_held)) + int(np.sum(new_held)))
                ),
                "h_old_new": float(harmonic),
                "old_forgetting_vs_before_adapter": float(
                    before_adapter["overall_accuracy"]
                    - after_old["overall_accuracy"]
                ),
                "old_forgetting_vs_before_identity": float(
                    before_identity["overall_accuracy"]
                    - after_old["overall_accuracy"]
                ),
            }
        )
        trace.append(
            {
                "phase": "joint_registration_fold_summary",
                "fold": fold_index,
                "epoch": hyperparameters.epochs,
                "candidate_id": hyperparameters.candidate_id,
                "rank": hyperparameters.rank,
                "seed": hyperparameters.seed,
                "learning_rate": hyperparameters.learning_rate,
                "identity_weight": hyperparameters.identity_weight,
                "factor_weight": hyperparameters.factor_weight,
                "after_old_overall_accuracy": after_old["overall_accuracy"],
                "after_old_floor_accuracy": after_old["min_class_accuracy"],
                "after_old_per_class_accuracy": after_old["per_class_accuracy"],
                "after_new_overall_accuracy": after_new["overall_accuracy"],
                "after_new_floor_accuracy": after_new["min_class_accuracy"],
                "after_new_per_class_accuracy": after_new["per_class_accuracy"],
                "joint_accuracy": fold_rows[-1]["joint_accuracy"],
                "h_old_new": fold_rows[-1]["h_old_new"],
                "old_forgetting_vs_before_adapter": fold_rows[-1][
                    "old_forgetting_vs_before_adapter"
                ],
                "old_forgetting_vs_before_identity": fold_rows[-1][
                    "old_forgetting_vs_before_identity"
                ],
            }
        )

    def aggregate_metric(key: str, classes: Sequence[str]) -> dict[str, Any]:
        rows = [fold[key] for fold in fold_rows]
        per_class = {
            label: float(np.mean([row["per_class_accuracy"][label] for row in rows]))
            for label in classes
        }
        return {
            "overall_accuracy": float(np.mean([row["overall_accuracy"] for row in rows])),
            "min_class_accuracy": float(min(per_class.values())),
            "per_class_accuracy": per_class,
        }

    before_identity_old = aggregate_metric("before_identity_old", old_classes)
    before_adapter_old = aggregate_metric("before_adapter_old", old_classes)
    after_old = aggregate_metric("after_old", old_classes)
    after_new = aggregate_metric("after_new", new_classes)
    old_per_class_non_degraded = all(
        after_old["per_class_accuracy"][label] + 1.0e-12
        >= before_adapter_old["per_class_accuracy"][label]
        for label in old_classes
    )
    return (
        {
            "policy": "joint_physical_leave_two_out_old_k8_fit_new_k8_register",
            "before_identity_old": before_identity_old,
            "before_adapter_old": before_adapter_old,
            "after_old": after_old,
            "after_new": after_new,
            "joint_accuracy": float(np.mean([row["joint_accuracy"] for row in fold_rows])),
            "h_old_new": float(np.mean([row["h_old_new"] for row in fold_rows])),
            "old_forgetting_vs_before_adapter": float(
                before_adapter_old["overall_accuracy"] - after_old["overall_accuracy"]
            ),
            "old_forgetting_vs_before_identity": float(
                before_identity_old["overall_accuracy"] - after_old["overall_accuracy"]
            ),
            "old_per_class_non_degraded": old_per_class_non_degraded,
            "folds": fold_rows,
        },
        tuple(trace),
    )


def predict_all_registered(
    state: TrainableLowRankAdapterState,
    feature_artifact: ValidatedFeatureArtifact,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-sample prediction over every registered class; no role input."""

    _validate_state(state)
    features_by_view = _artifact_features(feature_artifact)
    if (
        feature_artifact.sealed_runtime_sha256 != state.sealed_runtime_sha256
        or feature_artifact.feature_code_sha256 != state.feature_code_sha256
        or feature_artifact.sealed_phase1_checkpoint_sha256
        != state.sealed_phase1_checkpoint_sha256
    ):
        raise TrainableLowRankAdapterError("query artifact/state binding mismatch")
    if tuple(features_by_view) != state.view_ids:
        raise TrainableLowRankAdapterError("prediction view schema drift")
    arrays = [np.asarray(features_by_view[name], dtype=np.float32) for name in state.view_ids]
    if (
        not arrays
        or len(arrays[0]) != 1
        or any(array.ndim != 2 or array.shape[1] != state.feature_dim for array in arrays)
        or any(array.shape != arrays[0].shape for array in arrays[1:])
        or any(not np.isfinite(array).all() for array in arrays)
    ):
        raise TrainableLowRankAdapterError(
            "prediction callback requires exactly one per-sample row"
        )
    adapted = np.stack(
        [
            _adapt_numpy(array, state.low_rank_u, state.low_rank_v, state.gate)
            for array in arrays
        ]
    )
    rows = np.mean(adapted, axis=0)
    rows /= np.maximum(np.linalg.norm(rows, axis=1, keepdims=True), EPS)
    scores = rows @ state.prototypes.T
    indices = np.argmax(scores, axis=1)
    return np.asarray(state.classes)[indices], scores.astype(np.float32)
