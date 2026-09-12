"""D12 support-only joint residual logit head for sealed Phase2 features.

The formal lifecycle accepts only runtime-authorized per-sample feature
artifacts produced from fixed received LEO_weak IQ.  There is no public raw
feature wrapping API and no query-label, role, quota, batch assignment, source,
clean, dataset, or scorer interface.
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
SCHEMA = "cvs.phase2.joint_residual_logit_head.v1"
MAX_PARAMETERS = 12_000
MAX_EPOCHS = 15
MAX_STATE_BYTES = 256 * 1024
_ARTIFACT_TOKEN = object()


class JointResidualLogitHeadError(ValueError):
    """Raised when the D12 support-only contract fails closed."""


class RuntimeAuthorizedFeatureArtifact:
    """Immutable feature rows bound to actual received IQ and sealed runtime."""

    def __init__(
        self,
        *,
        features: np.ndarray,
        physical_sample_ids: Sequence[str],
        parent_received_iq_sha256: Sequence[str],
        sealed_runtime_sha256: str,
        feature_code_sha256: str,
        sealed_phase1_checkpoint_sha256: str,
        operator_id: str,
        view_seed: int,
        _token: object,
    ) -> None:
        if _token is not _ARTIFACT_TOKEN:
            raise JointResidualLogitHeadError(
                "feature artifact must come from internal runtime extraction"
            )
        rows = np.ascontiguousarray(features, dtype=np.float32)
        ids = tuple(str(value) for value in physical_sample_ids)
        parents = tuple(str(value) for value in parent_received_iq_sha256)
        bindings = (
            sealed_runtime_sha256,
            feature_code_sha256,
            sealed_phase1_checkpoint_sha256,
        )
        if (
            rows.ndim != 2
            or not len(rows)
            or not np.isfinite(rows).all()
            or len(ids) != len(rows)
            or len(parents) != len(rows)
            or len(set(ids)) != len(ids)
            or len(set(parents)) != len(parents)
            or any(len(value) != 64 for value in parents)
            or any(len(value) != 64 for value in bindings)
            or not operator_id
        ):
            raise JointResidualLogitHeadError("runtime-authorized artifact drift")
        immutable = np.frombuffer(rows.tobytes(), dtype=np.float32).reshape(rows.shape)
        per_row = tuple(
            hashlib.sha256(np.ascontiguousarray(row).tobytes()).hexdigest()
            for row in immutable
        )
        canonical = {
            "physical_sample_ids": ids,
            "parent_received_iq_sha256": parents,
            "per_row_feature_sha256": per_row,
            "sealed_runtime_sha256": sealed_runtime_sha256,
            "feature_code_sha256": feature_code_sha256,
            "sealed_phase1_checkpoint_sha256": sealed_phase1_checkpoint_sha256,
            "operator_id": operator_id,
            "view_seed": int(view_seed),
        }
        self.features = immutable
        self.physical_sample_ids = ids
        self.parent_received_iq_sha256 = parents
        self.per_row_feature_sha256 = per_row
        self.sealed_runtime_sha256 = sealed_runtime_sha256
        self.feature_code_sha256 = feature_code_sha256
        self.sealed_phase1_checkpoint_sha256 = sealed_phase1_checkpoint_sha256
        self.operator_id = operator_id
        self.view_seed = int(view_seed)
        self.artifact_sha256 = hashlib.sha256(
            json.dumps(
                canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=True
            ).encode("utf-8")
        ).hexdigest()


def _build_runtime_authorized_feature_artifact_internal(
    received_iq: np.ndarray,
    *,
    physical_sample_ids: Sequence[str],
    parent_received_iq_sha256: Sequence[str],
    sealed_runtime_sha256: str,
    feature_code_sha256: str,
    sealed_phase1_checkpoint_sha256: str,
    extract_single_received_iq: Callable[[np.ndarray], np.ndarray],
    operator_id: str = "base",
    view_seed: int = 0,
) -> RuntimeAuthorizedFeatureArtifact:
    """Runner-internal factory with actual-IQ SHA and physical batch=1 checks."""

    iq = np.asarray(received_iq, dtype=np.float32)
    parents = tuple(str(value) for value in parent_received_iq_sha256)
    if (
        iq.ndim != 3
        or iq.shape[1] != 2
        or not np.isfinite(iq).all()
        or len(iq) != len(parents)
    ):
        raise JointResidualLogitHeadError("received-IQ input drift")
    computed = tuple(
        hashlib.sha256(np.ascontiguousarray(row).tobytes()).hexdigest()
        for row in iq
    )
    if computed != parents:
        raise JointResidualLogitHeadError("actual received-IQ SHA binding mismatch")
    features = []
    for row in iq:
        value = np.asarray(
            extract_single_received_iq(row[None, ...]), dtype=np.float32
        )
        if value.ndim != 2 or value.shape[0] != 1 or not np.isfinite(value).all():
            raise JointResidualLogitHeadError(
                "authorized extractor must return exactly one finite feature row"
            )
        features.append(value[0])
    return RuntimeAuthorizedFeatureArtifact(
        features=np.stack(features),
        physical_sample_ids=physical_sample_ids,
        parent_received_iq_sha256=parents,
        sealed_runtime_sha256=sealed_runtime_sha256,
        feature_code_sha256=feature_code_sha256,
        sealed_phase1_checkpoint_sha256=sealed_phase1_checkpoint_sha256,
        operator_id=operator_id,
        view_seed=view_seed,
        _token=_ARTIFACT_TOKEN,
    )


@dataclass(frozen=True)
class ResidualHeadHyperparameters:
    candidate_id: str
    rank: int = 8
    epochs: int = 12
    learning_rate: float = 0.02
    alpha: float = 0.10
    temperature: float = 0.10
    old_logit_distillation_weight: float = 4.0
    residual_identity_weight: float = 2.0
    factor_weight: float = 0.01
    seed: int = 20260717


@dataclass(frozen=True)
class JointResidualLogitHeadState:
    schema: str
    candidate_id: str
    classes: tuple[str, ...]
    prototypes: np.ndarray
    w1: np.ndarray
    w2: np.ndarray
    hyperparameters: ResidualHeadHyperparameters
    feature_dim: int
    k_shot: int
    old_class_count: int
    registration_generation: int
    resource: Mapping[str, Any]
    support_feature_artifact_sha256: str
    support_selection_sha256: str
    sealed_runtime_sha256: str
    feature_code_sha256: str
    sealed_phase1_checkpoint_sha256: str
    operator_id: str
    view_seed: int
    state_content_sha256: str = ""

    def __post_init__(self) -> None:
        for name in ("prototypes", "w1", "w2"):
            source = np.ascontiguousarray(getattr(self, name), dtype=np.float32)
            immutable = np.frombuffer(source.tobytes(), dtype=np.float32).reshape(
                source.shape
            )
            object.__setattr__(self, name, immutable)
        actual_bytes = int(self.prototypes.nbytes + self.w1.nbytes + self.w2.nbytes)
        if (
            actual_bytes > MAX_STATE_BYTES
            or int(self.resource.get("persistent_state_bytes", -1)) != actual_bytes
            or int(self.resource.get("trainable_parameters", -1))
            != (
                0
                if self.hyperparameters.alpha == 0.0
                else _parameter_count(
                    self.feature_dim,
                    self.hyperparameters.rank,
                    len(self.classes),
                )
            )
        ):
            raise JointResidualLogitHeadError("state resource audit drift")
        computed = _state_content_sha256(self)
        if self.state_content_sha256 and self.state_content_sha256 != computed:
            raise JointResidualLogitHeadError("state content SHA mismatch")
        object.__setattr__(self, "state_content_sha256", computed)


@dataclass(frozen=True)
class BeforeAfterFitResult:
    before_state: JointResidualLogitHeadState
    after_state: JointResidualLogitHeadState
    trace: tuple[dict[str, Any], ...]


class _ResidualHead(torch.nn.Module):
    def __init__(self, feature_dim: int, rank: int, class_count: int) -> None:
        super().__init__()
        scale = 1.0 / max(float(feature_dim) ** 0.5, 1.0)
        self.w1 = torch.nn.Parameter(torch.randn(feature_dim, rank) * scale)
        self.w2 = torch.nn.Parameter(torch.zeros(rank, class_count))

    def residual(self, rows: torch.Tensor) -> torch.Tensor:
        return torch.tanh(rows @ self.w1) @ self.w2


def _parameter_count(feature_dim: int, rank: int, class_count: int) -> int:
    return int(feature_dim * rank + rank * class_count)


def _validate_hyperparameters(
    value: ResidualHeadHyperparameters, *, feature_dim: int, class_count: int
) -> None:
    if (
        not value.candidate_id
        or value.rank != 8
        or (
            (value.alpha == 0.0 and value.epochs != 0)
            or (value.alpha > 0.0 and not (1 <= value.epochs <= MAX_EPOCHS))
        )
        or _parameter_count(feature_dim, value.rank, class_count) > MAX_PARAMETERS
        or not (0.0 <= value.alpha <= 0.25)
        or (value.alpha > 0.0 and value.learning_rate <= 0.0)
        or value.temperature <= 0.0
        or any(
            not np.isfinite(item) or item < 0.0
            for item in (
                value.old_logit_distillation_weight,
                value.residual_identity_weight,
                value.factor_weight,
            )
        )
    ):
        raise JointResidualLogitHeadError("hyperparameter/resource drift")


def _artifact_rows(value: RuntimeAuthorizedFeatureArtifact) -> np.ndarray:
    if not isinstance(value, RuntimeAuthorizedFeatureArtifact):
        raise JointResidualLogitHeadError(
            "ordinary feature mapping/array forbidden; authorized artifact required"
        )
    return value.features


def _validate_support(
    artifact: RuntimeAuthorizedFeatureArtifact,
    labels: Sequence[str] | np.ndarray,
    ranks: Sequence[int] | np.ndarray,
    *,
    k_shot: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[str, ...]]:
    rows = _artifact_rows(artifact)
    label_rows = np.asarray(labels).astype(str)
    rank_rows = np.asarray(ranks, dtype=np.int64)
    if len(rows) != len(label_rows) or len(rows) != len(rank_rows):
        raise JointResidualLogitHeadError("support alignment drift")
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
        raise JointResidualLogitHeadError("strict physical K-shot support drift")
    return rows, label_rows, rank_rows, tuple(sorted(classes.tolist()))


def _validate_old_lineage_exact_reuse(
    before_artifact: RuntimeAuthorizedFeatureArtifact,
    before_labels: np.ndarray,
    before_ranks: np.ndarray,
    after_artifact: RuntimeAuthorizedFeatureArtifact,
    after_labels: np.ndarray,
    after_ranks: np.ndarray,
    old_classes: Sequence[str],
) -> None:
    def keyed(
        artifact: RuntimeAuthorizedFeatureArtifact,
        labels: np.ndarray,
        ranks: np.ndarray,
        allowed: set[str],
    ) -> dict[tuple[str, int], tuple[str, str, str]]:
        return {
            (str(labels[index]), int(ranks[index])): (
                artifact.physical_sample_ids[index],
                artifact.parent_received_iq_sha256[index],
                artifact.per_row_feature_sha256[index],
            )
            for index in range(len(labels))
            if str(labels[index]) in allowed
        }

    allowed = set(old_classes)
    before = keyed(
        before_artifact, before_labels, before_ranks, allowed
    )
    after = keyed(after_artifact, after_labels, after_ranks, allowed)
    if before != after:
        raise JointResidualLogitHeadError(
            "after old support lineage exact-reuse lock failed"
        )


def _normalize(rows: np.ndarray) -> np.ndarray:
    values = np.asarray(rows, dtype=np.float32)
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), EPS)


def _prototypes(
    rows: np.ndarray, labels: np.ndarray, classes: Sequence[str]
) -> np.ndarray:
    normalized = _normalize(rows)
    values = np.stack(
        [np.mean(normalized[labels == label], axis=0) for label in classes]
    )
    return _normalize(values)


def _score_numpy(
    rows: np.ndarray,
    prototypes: np.ndarray,
    w1: np.ndarray,
    w2: np.ndarray,
    alpha: float,
) -> np.ndarray:
    normalized = _normalize(rows)
    base = normalized @ prototypes.T
    residual = np.tanh(normalized @ w1) @ w2
    return base + float(alpha) * residual


def _state_content_sha256(state: JointResidualLogitHeadState) -> str:
    digest = hashlib.sha256()
    digest.update(state.schema.encode("utf-8"))
    digest.update(state.candidate_id.encode("utf-8"))
    digest.update(json.dumps(state.classes, separators=(",", ":")).encode("utf-8"))
    for value in (state.prototypes, state.w1, state.w2):
        digest.update(str(value.shape).encode("ascii"))
        digest.update(np.ascontiguousarray(value).tobytes())
    digest.update(
        json.dumps(
            {
                "feature_dim": state.feature_dim,
                "k_shot": state.k_shot,
                "old_class_count": state.old_class_count,
                "registration_generation": state.registration_generation,
                "support_feature_artifact_sha256": state.support_feature_artifact_sha256,
                "support_selection_sha256": state.support_selection_sha256,
                "sealed_runtime_sha256": state.sealed_runtime_sha256,
                "feature_code_sha256": state.feature_code_sha256,
                "sealed_phase1_checkpoint_sha256": (
                    state.sealed_phase1_checkpoint_sha256
                ),
                "operator_id": state.operator_id,
                "view_seed": state.view_seed,
                "hyperparameters": {
                    "candidate_id": state.hyperparameters.candidate_id,
                    "rank": state.hyperparameters.rank,
                    "epochs": state.hyperparameters.epochs,
                    "learning_rate": state.hyperparameters.learning_rate,
                    "alpha": state.hyperparameters.alpha,
                    "temperature": state.hyperparameters.temperature,
                    "old_logit_distillation_weight": (
                        state.hyperparameters.old_logit_distillation_weight
                    ),
                    "residual_identity_weight": (
                        state.hyperparameters.residual_identity_weight
                    ),
                    "factor_weight": state.hyperparameters.factor_weight,
                    "seed": state.hyperparameters.seed,
                },
                "resource": dict(state.resource),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    return digest.hexdigest()


def _validate_state(state: JointResidualLogitHeadState) -> None:
    actual = int(state.prototypes.nbytes + state.w1.nbytes + state.w2.nbytes)
    effective_rank = 0 if state.hyperparameters.alpha == 0.0 else state.hyperparameters.rank
    if (
        state.prototypes.shape != (len(state.classes), state.feature_dim)
        or state.w1.shape != (state.feature_dim, effective_rank)
        or state.w2.shape != (effective_rank, len(state.classes))
        or actual > MAX_STATE_BYTES
        or int(state.resource.get("persistent_state_bytes", -1)) != actual
        or state.state_content_sha256 != _state_content_sha256(state)
        or any(
            len(value) != 64
            for value in (
                state.support_feature_artifact_sha256,
                state.support_selection_sha256,
                state.sealed_runtime_sha256,
                state.feature_code_sha256,
                state.sealed_phase1_checkpoint_sha256,
            )
        )
        or not state.operator_id
    ):
        raise JointResidualLogitHeadError("state content/resource/binding drift")


def _support_selection_sha256(
    artifact: RuntimeAuthorizedFeatureArtifact,
    labels: np.ndarray,
    ranks: np.ndarray,
    selection: np.ndarray | None = None,
) -> str:
    if selection is None:
        selected = np.ones(len(labels), dtype=bool)
    else:
        selected = np.asarray(selection, dtype=bool)
    if len(selected) != len(labels) or len(labels) != len(artifact.features):
        raise JointResidualLogitHeadError("support selection binding alignment drift")
    rows = [
        {
            "label": str(labels[index]),
            "rank": int(ranks[index]),
            "physical_sample_id": artifact.physical_sample_ids[index],
            "parent_received_iq_sha256": artifact.parent_received_iq_sha256[index],
            "feature_sha256": artifact.per_row_feature_sha256[index],
        }
        for index in np.flatnonzero(selected)
    ]
    return hashlib.sha256(
        json.dumps(
            rows, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
    ).hexdigest()


def _labels_to_targets(labels: np.ndarray, classes: Sequence[str]) -> np.ndarray:
    lookup = {label: index for index, label in enumerate(classes)}
    return np.asarray([lookup[label] for label in labels], dtype=np.int64)


def _fit_head(
    rows: np.ndarray,
    labels: np.ndarray,
    classes: tuple[str, ...],
    *,
    hyperparameters: ResidualHeadHyperparameters,
    device: torch.device,
    trace_context: Mapping[str, Any],
    teacher_state: JointResidualLogitHeadState | None = None,
    old_class_count: int,
) -> tuple[np.ndarray, np.ndarray, tuple[dict[str, Any], ...]]:
    feature_dim = int(rows.shape[1])
    _validate_hyperparameters(
        hyperparameters, feature_dim=feature_dim, class_count=len(classes)
    )
    if teacher_state is not None:
        _validate_state(teacher_state)
        if (
            teacher_state.classes != classes[:old_class_count]
            or teacher_state.feature_dim != feature_dim
        ):
            raise JointResidualLogitHeadError("teacher class/feature binding drift")
    normalized = _normalize(rows)
    prototypes = _prototypes(rows, labels, classes)
    targets = _labels_to_targets(labels, classes)
    if hyperparameters.alpha == 0.0:
        base_logits = normalized @ prototypes.T
        predictions = np.argmax(base_logits, axis=1)
        per_class = {
            label: float(np.mean(predictions[targets == index] == index))
            for index, label in enumerate(classes)
        }
        ce = F.cross_entropy(
            torch.from_numpy(base_logits / hyperparameters.temperature),
            torch.from_numpy(targets),
        )
        return (
            np.zeros((feature_dim, 0), dtype=np.float32),
            np.zeros((0, len(classes)), dtype=np.float32),
            (
                {
                    **dict(trace_context),
                    "candidate_id": hyperparameters.candidate_id,
                    "epoch": 0,
                    "rank": 0,
                    "alpha": 0.0,
                    "learning_rate": 0.0,
                    "temperature": hyperparameters.temperature,
                    "old_logit_distillation_weight": 0.0,
                    "residual_identity_weight": 0.0,
                    "factor_weight": 0.0,
                    "total_loss": float(ce),
                    "cross_entropy_loss": float(ce),
                    "old_logit_distillation_loss": 0.0,
                    "residual_identity_loss": 0.0,
                    "factor_loss": 0.0,
                    "support_overall_accuracy": float(np.mean(predictions == targets)),
                    "support_floor_accuracy": min(per_class.values()),
                    "support_per_class_accuracy": per_class,
                    "base_logit_l2_mean": float(
                        np.mean(np.linalg.norm(base_logits, axis=1))
                    ),
                    "residual_logit_l2_mean": 0.0,
                    "max_abs_logit_correction": 0.0,
                    "residual_logit_linf": 0.0,
                    "residual_training_skipped": True,
                },
            ),
        )
    torch.manual_seed(int(hyperparameters.seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(hyperparameters.seed))
    model = _ResidualHead(feature_dim, hyperparameters.rank, len(classes)).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=hyperparameters.learning_rate
    )
    row_tensor = torch.from_numpy(np.ascontiguousarray(normalized)).to(device)
    proto_tensor = torch.from_numpy(np.ascontiguousarray(prototypes)).to(device)
    target_tensor = torch.from_numpy(targets).to(device)
    old_mask_np = targets < old_class_count
    old_mask = torch.from_numpy(old_mask_np).to(device)
    teacher_logits = None
    if teacher_state is not None and old_mask_np.any():
        teacher_logits = torch.from_numpy(
            _score_numpy(
                normalized[old_mask_np],
                teacher_state.prototypes,
                teacher_state.w1,
                teacher_state.w2,
                teacher_state.hyperparameters.alpha,
            ).astype(np.float32)
        ).to(device)
    trace: list[dict[str, Any]] = []
    for epoch in range(1, hyperparameters.epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        base = row_tensor @ proto_tensor.T
        residual = model.residual(row_tensor)
        logits = base + hyperparameters.alpha * residual
        ce_loss = F.cross_entropy(logits / hyperparameters.temperature, target_tensor)
        if teacher_logits is None:
            distill_loss = logits.new_zeros(())
        else:
            distill_loss = F.mse_loss(
                logits[old_mask, :old_class_count], teacher_logits
            )
        identity_loss = residual.square().mean()
        factor_loss = model.w1.square().mean() + model.w2.square().mean()
        loss = (
            ce_loss
            + hyperparameters.old_logit_distillation_weight * distill_loss
            + hyperparameters.residual_identity_weight * identity_loss
            + hyperparameters.factor_weight * factor_loss
        )
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            predictions = torch.argmax(logits, dim=1)
            overall = float((predictions == target_tensor).float().mean().cpu())
            per_class = {
                label: float(
                    (
                        predictions[target_tensor == index]
                        == target_tensor[target_tensor == index]
                    )
                    .float()
                    .mean()
                    .cpu()
                )
                for index, label in enumerate(classes)
            }
            correction = hyperparameters.alpha * residual
        trace.append(
            {
                **dict(trace_context),
                "candidate_id": hyperparameters.candidate_id,
                "epoch": epoch,
                "rank": hyperparameters.rank,
                "alpha": hyperparameters.alpha,
                "learning_rate": hyperparameters.learning_rate,
                "temperature": hyperparameters.temperature,
                "old_logit_distillation_weight": (
                    hyperparameters.old_logit_distillation_weight
                ),
                "residual_identity_weight": (
                    hyperparameters.residual_identity_weight
                ),
                "factor_weight": hyperparameters.factor_weight,
                "total_loss": float(loss.detach().cpu()),
                "cross_entropy_loss": float(ce_loss.detach().cpu()),
                "old_logit_distillation_loss": float(distill_loss.detach().cpu()),
                "residual_identity_loss": float(identity_loss.detach().cpu()),
                "factor_loss": float(factor_loss.detach().cpu()),
                "support_overall_accuracy": overall,
                "support_floor_accuracy": min(per_class.values()),
                "support_per_class_accuracy": per_class,
                "residual_logit_linf": float(
                    correction.abs().max().detach().cpu()
                ),
                "base_logit_l2_mean": float(
                    torch.linalg.vector_norm(base, dim=1).mean().detach().cpu()
                ),
                "residual_logit_l2_mean": float(
                    torch.linalg.vector_norm(correction, dim=1)
                    .mean()
                    .detach()
                    .cpu()
                ),
                "max_abs_logit_correction": float(
                    correction.abs().max().detach().cpu()
                ),
                "residual_training_skipped": False,
            }
        )
    return (
        model.w1.detach().cpu().numpy().astype(np.float32),
        model.w2.detach().cpu().numpy().astype(np.float32),
        tuple(trace),
    )


def _make_state(
    artifact: RuntimeAuthorizedFeatureArtifact,
    rows: np.ndarray,
    labels: np.ndarray,
    classes: tuple[str, ...],
    w1: np.ndarray,
    w2: np.ndarray,
    *,
    k_shot: int,
    old_class_count: int,
    registration_generation: int,
    hyperparameters: ResidualHeadHyperparameters,
    support_selection_sha256: str,
) -> JointResidualLogitHeadState:
    prototypes = _prototypes(rows, labels, classes).astype(np.float32)
    parameters = (
        0
        if hyperparameters.alpha == 0.0
        else _parameter_count(rows.shape[1], hyperparameters.rank, len(classes))
    )
    state_bytes = int(prototypes.nbytes + w1.nbytes + w2.nbytes)
    if parameters > MAX_PARAMETERS or state_bytes > MAX_STATE_BYTES:
        raise JointResidualLogitHeadError("D12 resource budget exceeded")
    return JointResidualLogitHeadState(
        schema=SCHEMA,
        candidate_id=hyperparameters.candidate_id,
        classes=classes,
        prototypes=prototypes,
        w1=w1,
        w2=w2,
        hyperparameters=hyperparameters,
        feature_dim=int(rows.shape[1]),
        k_shot=int(k_shot),
        old_class_count=int(old_class_count),
        registration_generation=int(registration_generation),
        resource={
            "trainable_parameters": parameters,
            "adapt_epochs": hyperparameters.epochs,
            "persistent_state_bytes": state_bytes,
            "residual_head_mac_per_sample": int(parameters),
            "prototype_cosine_mac_per_sample": int(len(classes) * rows.shape[1]),
            "backbone_forwards_per_physical_sample": 1,
            "activation": "tanh",
        },
        support_feature_artifact_sha256=artifact.artifact_sha256,
        support_selection_sha256=support_selection_sha256,
        sealed_runtime_sha256=artifact.sealed_runtime_sha256,
        feature_code_sha256=artifact.feature_code_sha256,
        sealed_phase1_checkpoint_sha256=(
            artifact.sealed_phase1_checkpoint_sha256
        ),
        operator_id=artifact.operator_id,
        view_seed=artifact.view_seed,
    )


def fit_before_after_locked(
    before_artifact: RuntimeAuthorizedFeatureArtifact,
    before_labels: Sequence[str] | np.ndarray,
    before_ranks: Sequence[int] | np.ndarray,
    after_artifact: RuntimeAuthorizedFeatureArtifact,
    after_labels: Sequence[str] | np.ndarray,
    after_ranks: Sequence[int] | np.ndarray,
    *,
    k_shot: int,
    hyperparameters: ResidualHeadHyperparameters,
    device: torch.device | str = "cpu",
) -> BeforeAfterFitResult:
    """Fit old-only Before and jointly fit old+new After with old distillation."""

    before_rows, old_labels, old_rank_rows, old_classes = _validate_support(
        before_artifact, before_labels, before_ranks, k_shot=k_shot
    )
    after_rows, joint_labels, joint_rank_rows, joint_classes = _validate_support(
        after_artifact, after_labels, after_ranks, k_shot=k_shot
    )
    if (
        not set(old_classes) < set(joint_classes)
        or before_artifact.sealed_runtime_sha256
        != after_artifact.sealed_runtime_sha256
        or before_artifact.feature_code_sha256 != after_artifact.feature_code_sha256
        or before_artifact.sealed_phase1_checkpoint_sha256
        != after_artifact.sealed_phase1_checkpoint_sha256
    ):
        raise JointResidualLogitHeadError("before/after class or runtime binding drift")
    _validate_old_lineage_exact_reuse(
        before_artifact,
        old_labels,
        old_rank_rows,
        after_artifact,
        joint_labels,
        joint_rank_rows,
        old_classes,
    )
    joint_classes = old_classes + tuple(
        sorted(set(joint_classes) - set(old_classes))
    )
    target_device = torch.device(device)
    before_w1, before_w2, before_trace = _fit_head(
        before_rows,
        old_labels,
        old_classes,
        hyperparameters=hyperparameters,
        device=target_device,
        trace_context={"phase": "before_full_fit", "fold": None},
        teacher_state=None,
        old_class_count=len(old_classes),
    )
    before_state = _make_state(
        before_artifact,
        before_rows,
        old_labels,
        old_classes,
        before_w1,
        before_w2,
        k_shot=k_shot,
        old_class_count=len(old_classes),
        registration_generation=0,
        hyperparameters=hyperparameters,
        support_selection_sha256=_support_selection_sha256(
            before_artifact, old_labels, old_rank_rows
        ),
    )
    after_w1, after_w2, after_trace = _fit_head(
        after_rows,
        joint_labels,
        joint_classes,
        hyperparameters=hyperparameters,
        device=target_device,
        trace_context={"phase": "after_joint_full_fit", "fold": None},
        teacher_state=before_state,
        old_class_count=len(old_classes),
    )
    after_state = _make_state(
        after_artifact,
        after_rows,
        joint_labels,
        joint_classes,
        after_w1,
        after_w2,
        k_shot=k_shot,
        old_class_count=len(old_classes),
        registration_generation=1,
        hyperparameters=hyperparameters,
        support_selection_sha256=_support_selection_sha256(
            after_artifact, joint_labels, joint_rank_rows
        ),
    )
    return BeforeAfterFitResult(
        before_state=before_state,
        after_state=after_state,
        trace=before_trace + after_trace,
    )


def _leave_two_out_masks(labels: np.ndarray, ranks: np.ndarray) -> tuple[np.ndarray, ...]:
    if set(np.unique(ranks).tolist()) != set(range(10)):
        raise JointResidualLogitHeadError("joint L2O requires strict K10 ranks")
    masks = []
    for first in range(0, 10, 2):
        held = np.isin(ranks, (first, first + 1))
        if any(np.sum(held & (labels == label)) != 2 for label in np.unique(labels)):
            raise JointResidualLogitHeadError("leave-two-out physical fold drift")
        masks.append(held)
    return tuple(masks)


def _prediction_metrics(
    truth: np.ndarray, predictions: np.ndarray, classes: Sequence[str]
) -> dict[str, Any]:
    per_class = {
        label: float(np.mean(predictions[truth == label] == label))
        for label in classes
    }
    return {
        "overall_accuracy": float(np.mean(predictions == truth)),
        "min_class_accuracy": float(min(per_class.values())),
        "per_class_accuracy": per_class,
    }


def _aggregate_metrics(
    folds: Sequence[Mapping[str, Any]], classes: Sequence[str], key: str
) -> dict[str, Any]:
    per_class = {
        label: float(np.mean([row[key]["per_class_accuracy"][label] for row in folds]))
        for label in classes
    }
    return {
        "overall_accuracy": float(np.mean([row[key]["overall_accuracy"] for row in folds])),
        "min_class_accuracy": float(min(per_class.values())),
        "per_class_accuracy": per_class,
    }


def evaluate_joint_leave_two_out(
    before_artifact: RuntimeAuthorizedFeatureArtifact,
    before_labels: Sequence[str] | np.ndarray,
    before_ranks: Sequence[int] | np.ndarray,
    after_artifact: RuntimeAuthorizedFeatureArtifact,
    after_labels: Sequence[str] | np.ndarray,
    after_ranks: Sequence[int] | np.ndarray,
    *,
    hyperparameters: ResidualHeadHyperparameters,
    device: torch.device | str = "cpu",
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    """Joint K10 L2O: old/new each held2 and classified over all classes."""

    old_rows, old_labels, old_ranks, old_classes = _validate_support(
        before_artifact, before_labels, before_ranks, k_shot=10
    )
    joint_rows, joint_labels, joint_ranks, joint_classes = _validate_support(
        after_artifact, after_labels, after_ranks, k_shot=10
    )
    if (
        not set(old_classes) < set(joint_classes)
        or before_artifact.sealed_runtime_sha256
        != after_artifact.sealed_runtime_sha256
        or before_artifact.feature_code_sha256 != after_artifact.feature_code_sha256
        or before_artifact.sealed_phase1_checkpoint_sha256
        != after_artifact.sealed_phase1_checkpoint_sha256
    ):
        raise JointResidualLogitHeadError("joint L2O before/after binding drift")
    _validate_old_lineage_exact_reuse(
        before_artifact,
        old_labels,
        old_ranks,
        after_artifact,
        joint_labels,
        joint_ranks,
        old_classes,
    )
    joint_classes = old_classes + tuple(
        sorted(set(joint_classes) - set(old_classes))
    )
    old_masks = _leave_two_out_masks(old_labels, old_ranks)
    joint_masks = _leave_two_out_masks(joint_labels, joint_ranks)
    old_indices_in_joint = np.isin(joint_labels, old_classes)
    folds: list[dict[str, Any]] = []
    trace: list[dict[str, Any]] = []
    target_device = torch.device(device)
    for fold, (held_old, held_joint) in enumerate(zip(old_masks, joint_masks)):
        train_old = ~held_old
        train_joint = ~held_joint
        before_w1, before_w2, before_trace = _fit_head(
            old_rows[train_old],
            old_labels[train_old],
            old_classes,
            hyperparameters=hyperparameters,
            device=target_device,
            trace_context={"phase": "joint_l2o_before", "fold": fold},
            teacher_state=None,
            old_class_count=len(old_classes),
        )
        before_state = _make_state(
            before_artifact,
            old_rows[train_old],
            old_labels[train_old],
            old_classes,
            before_w1,
            before_w2,
            k_shot=8,
            old_class_count=len(old_classes),
            registration_generation=0,
            hyperparameters=hyperparameters,
            support_selection_sha256=_support_selection_sha256(
                before_artifact, old_labels, old_ranks, train_old
            ),
        )
        after_w1, after_w2, after_trace = _fit_head(
            joint_rows[train_joint],
            joint_labels[train_joint],
            joint_classes,
            hyperparameters=hyperparameters,
            device=target_device,
            trace_context={"phase": "joint_l2o_after", "fold": fold},
            teacher_state=before_state,
            old_class_count=len(old_classes),
        )
        after_state = _make_state(
            after_artifact,
            joint_rows[train_joint],
            joint_labels[train_joint],
            joint_classes,
            after_w1,
            after_w2,
            k_shot=8,
            old_class_count=len(old_classes),
            registration_generation=1,
            hyperparameters=hyperparameters,
            support_selection_sha256=_support_selection_sha256(
                after_artifact, joint_labels, joint_ranks, train_joint
            ),
        )
        held_old_joint = held_joint & old_indices_in_joint
        held_new_joint = held_joint & ~old_indices_in_joint
        before_scores = _score_numpy(
            old_rows[held_old],
            before_state.prototypes,
            before_state.w1,
            before_state.w2,
            hyperparameters.alpha,
        )
        after_old_scores = _score_numpy(
            joint_rows[held_old_joint],
            after_state.prototypes,
            after_state.w1,
            after_state.w2,
            hyperparameters.alpha,
        )
        after_new_scores = _score_numpy(
            joint_rows[held_new_joint],
            after_state.prototypes,
            after_state.w1,
            after_state.w2,
            hyperparameters.alpha,
        )
        base_after_old_scores = _score_numpy(
            joint_rows[held_old_joint],
            after_state.prototypes,
            np.zeros_like(after_state.w1),
            np.zeros_like(after_state.w2),
            0.0,
        )
        base_after_new_scores = _score_numpy(
            joint_rows[held_new_joint],
            after_state.prototypes,
            np.zeros_like(after_state.w1),
            np.zeros_like(after_state.w2),
            0.0,
        )
        before_predictions = np.asarray(old_classes)[np.argmax(before_scores, axis=1)]
        after_old_predictions = np.asarray(joint_classes)[
            np.argmax(after_old_scores, axis=1)
        ]
        after_new_predictions = np.asarray(joint_classes)[
            np.argmax(after_new_scores, axis=1)
        ]
        base_after_old_predictions = np.asarray(joint_classes)[
            np.argmax(base_after_old_scores, axis=1)
        ]
        base_after_new_predictions = np.asarray(joint_classes)[
            np.argmax(base_after_new_scores, axis=1)
        ]
        before_metric = _prediction_metrics(
            old_labels[held_old], before_predictions, old_classes
        )
        old_metric = _prediction_metrics(
            joint_labels[held_old_joint], after_old_predictions, old_classes
        )
        new_classes = joint_classes[len(old_classes) :]
        new_metric = _prediction_metrics(
            joint_labels[held_new_joint], after_new_predictions, new_classes
        )
        base_old_metric = _prediction_metrics(
            joint_labels[held_old_joint],
            base_after_old_predictions,
            old_classes,
        )
        base_new_metric = _prediction_metrics(
            joint_labels[held_new_joint],
            base_after_new_predictions,
            new_classes,
        )
        joint_truth = np.concatenate(
            [joint_labels[held_old_joint], joint_labels[held_new_joint]]
        )
        joint_predictions = np.concatenate(
            [after_old_predictions, after_new_predictions]
        )
        fold_row = {
            "fold": fold,
            "before_old": before_metric,
            "after_old": old_metric,
            "after_new": new_metric,
            "base_after_old": base_old_metric,
            "base_after_new": base_new_metric,
            "joint_accuracy": float(np.mean(joint_predictions == joint_truth)),
            "old_forgetting": (
                before_metric["overall_accuracy"] - old_metric["overall_accuracy"]
            ),
            "delta_vs_base_old_overall": (
                old_metric["overall_accuracy"] - base_old_metric["overall_accuracy"]
            ),
            "delta_vs_base_old_floor": (
                old_metric["min_class_accuracy"] - base_old_metric["min_class_accuracy"]
            ),
            "delta_vs_base_new_overall": (
                new_metric["overall_accuracy"] - base_new_metric["overall_accuracy"]
            ),
            "delta_vs_base_new_floor": (
                new_metric["min_class_accuracy"] - base_new_metric["min_class_accuracy"]
            ),
            "old_train_rows_per_class": 8,
            "new_train_rows_per_class": 8,
            "old_held_rows_per_class": 2,
            "new_held_rows_per_class": 2,
        }
        folds.append(fold_row)
        trace.extend(before_trace)
        trace.extend(after_trace)
        trace.append({"phase": "joint_l2o_fold_summary", **fold_row})
    before_old = _aggregate_metrics(folds, old_classes, "before_old")
    after_old = _aggregate_metrics(folds, old_classes, "after_old")
    new_classes = joint_classes[len(old_classes) :]
    after_new = _aggregate_metrics(folds, new_classes, "after_new")
    base_after_old = _aggregate_metrics(folds, old_classes, "base_after_old")
    base_after_new = _aggregate_metrics(folds, new_classes, "base_after_new")
    old_non_degraded_vs_before = all(
        after_old["per_class_accuracy"][label] + 1.0e-12
        >= before_old["per_class_accuracy"][label]
        for label in old_classes
    )
    old_non_degraded_vs_base = all(
        after_old["per_class_accuracy"][label] + 1.0e-12
        >= base_after_old["per_class_accuracy"][label]
        for label in old_classes
    )
    denominator = after_old["overall_accuracy"] + after_new["overall_accuracy"]
    h_value = (
        0.0
        if denominator <= 0.0
        else 2.0
        * after_old["overall_accuracy"]
        * after_new["overall_accuracy"]
        / denominator
    )
    base_denominator = (
        base_after_old["overall_accuracy"] + base_after_new["overall_accuracy"]
    )
    base_h = (
        0.0
        if base_denominator <= 0.0
        else 2.0
        * base_after_old["overall_accuracy"]
        * base_after_new["overall_accuracy"]
        / base_denominator
    )
    return (
        {
            "selection_policy": (
                "joint_physical_leave_two_out_old_new_each_held2_all_registered"
            ),
            "before_old": before_old,
            "after_old": after_old,
            "after_new": after_new,
            "base_after_old": base_after_old,
            "base_after_new": base_after_new,
            "joint_accuracy": float(np.mean([row["joint_accuracy"] for row in folds])),
            "h_old_new": float(h_value),
            "base_h_old_new": float(base_h),
            "delta_vs_base_h_old_new": float(h_value - base_h),
            "delta_vs_base_joint_accuracy": float(
                np.mean([row["joint_accuracy"] for row in folds])
                - (
                    len(old_classes) * base_after_old["overall_accuracy"]
                    + len(new_classes) * base_after_new["overall_accuracy"]
                )
                / len(joint_classes)
            ),
            "old_forgetting": (
                before_old["overall_accuracy"] - after_old["overall_accuracy"]
            ),
            "old_per_class_non_degraded_vs_before": bool(
                old_non_degraded_vs_before
            ),
            "old_per_class_non_degraded_vs_base_cosine": bool(
                old_non_degraded_vs_base
            ),
            "folds": folds,
        },
        tuple(trace),
    )


def predict_all_registered(
    state: JointResidualLogitHeadState,
    query_artifact: RuntimeAuthorizedFeatureArtifact,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict exactly one physical query over every registered class."""

    _validate_state(state)
    rows = _artifact_rows(query_artifact)
    if len(rows) != 1:
        raise JointResidualLogitHeadError(
            "formal prediction requires exactly one physical query"
        )
    if (
        query_artifact.sealed_runtime_sha256 != state.sealed_runtime_sha256
        or query_artifact.feature_code_sha256 != state.feature_code_sha256
        or query_artifact.sealed_phase1_checkpoint_sha256
        != state.sealed_phase1_checkpoint_sha256
        or query_artifact.operator_id != state.operator_id
        or query_artifact.view_seed != state.view_seed
    ):
        raise JointResidualLogitHeadError(
            "query runtime/code/checkpoint/operator binding mismatch"
        )
    scores = _score_numpy(
        rows,
        state.prototypes,
        state.w1,
        state.w2,
        state.hyperparameters.alpha,
    )
    predictions = np.asarray(state.classes)[np.argmax(scores, axis=1)]
    return predictions, scores
