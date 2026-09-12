"""D32 in-loop safety-capped Stage2-C new-class registration.

Only registered LEO_weak support enters this module.  The D26 diagonal and
old-class weight prefix are immutable.  New weights are optimized in chunks
of at most seven classes, while a differentiable old-support safety cap is
recomputed at step zero and inside every loss.  The exact same cap is stored
and used by the deployable per-sample all-registered-class scorer.

The safety cap for new class ``j`` is

    b_j(U) = min(0, min_i(a_i,true - n_i,j(U) - delta)),

where ``i`` ranges only over old-support rows that the pre-registration D26
head classifies correctly.  It therefore cannot use query labels, query role,
batch quotas, clean IQ, source IQ, or a dense query graph.  K=1 performs a
centroid-plus-cap registration with zero optimizer updates.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from cvsrffi.stage2_multimodal_compact_diag import (
    D26CompactDiagState,
    FEATURE_DIM,
    MAX_PERSISTENT_STATE_BYTES,
    TEMPERATURE,
)


SCHEMA = "cvs.phase2.d32_inloop_safe_cap_suffix.v1"
RESOURCE_SCHEMA = "cvs.phase2.d32_inloop_safe_cap_suffix_resource.v1"
GATE_SCHEMA = "cvs.phase2.d32_support_only_checkpoint_gate.v1"
MAX_STAGE2C_STEPS = 15
MAX_ACTIVE_NEW_CLASSES = 7
MAX_PEAK_TRAINABLE_PARAMETERS = 2_016
SAFETY_DELTA = 1.0e-4
BIAS_RECOVERY_TARGET = -4.0

D32_GROUP_BALANCED_CAP = "D32-A-INLOOP-CAP-GROUP-BALANCED"
D32_NEW_CVAR_CAP = "D32-B-INLOOP-CAP-NEW-CVAR"
D32_BIAS_RECOVERY_CAP = "D32-C-INLOOP-CAP-CVAR-BIAS-RECOVERY"
D32_METHODS = (
    D32_GROUP_BALANCED_CAP,
    D32_NEW_CVAR_CAP,
    D32_BIAS_RECOVERY_CAP,
)


class D32InLoopSafeCapError(ValueError):
    """Raised when the immutable state, support, or D32 method lock drifts."""


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _readonly(value: np.ndarray, dtype: Any) -> np.ndarray:
    contiguous = np.ascontiguousarray(value, dtype=dtype)
    result = np.frombuffer(contiguous.tobytes(), dtype=contiguous.dtype).reshape(
        contiguous.shape
    )
    result.setflags(write=False)
    return result


def _normalize_np(rows: np.ndarray) -> np.ndarray:
    value = np.asarray(rows, dtype=np.float32)
    norms = np.linalg.norm(value, axis=1, keepdims=True).astype(np.float32)
    if bool(np.any(norms <= np.float32(1.0e-12))):
        raise D32InLoopSafeCapError("D32 encountered a zero-norm row")
    return np.asarray(value / norms, dtype=np.float32)


def _validate_rows(value: np.ndarray, *, name: str) -> np.ndarray:
    rows = np.asarray(value, dtype=np.float32)
    if (
        rows.ndim != 2
        or rows.shape[1] != FEATURE_DIM
        or len(rows) < 1
        or not np.isfinite(rows).all()
        or bool(np.any(np.linalg.norm(rows, axis=1) <= 1.0e-12))
    ):
        raise D32InLoopSafeCapError(f"{name} must be finite nonzero [N,{FEATURE_DIM}]")
    return np.ascontiguousarray(rows, dtype=np.float32)


def _validate_support(
    features: np.ndarray,
    labels: Sequence[str],
    classes: Sequence[str],
    *,
    name: str,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...], int]:
    rows = _validate_rows(features, name=f"{name} features")
    label_values = np.asarray(tuple(str(value) for value in labels))
    registry = tuple(str(value) for value in classes)
    if (
        label_values.ndim != 1
        or len(label_values) != len(rows)
        or not registry
        or len(set(registry)) != len(registry)
        or any(not value for value in registry)
        or set(label_values.tolist()) != set(registry)
    ):
        raise D32InLoopSafeCapError(f"{name} registry or labels drift")
    counts = [int(np.sum(label_values == value)) for value in registry]
    if min(counts) < 1 or len(set(counts)) != 1:
        raise D32InLoopSafeCapError(f"{name} must be class-symmetric K-shot")
    return rows, label_values, registry, counts[0]


def _canonicalize_support(
    rows: np.ndarray, labels: np.ndarray, classes: Sequence[str]
) -> tuple[np.ndarray, np.ndarray]:
    class_index = {value: index for index, value in enumerate(classes)}
    order = sorted(
        range(len(rows)),
        key=lambda index: (
            class_index[str(labels[index])],
            np.ascontiguousarray(rows[index], dtype=np.float32).tobytes(),
        ),
    )
    selected = np.asarray(order, dtype=np.int64)
    return (
        np.ascontiguousarray(rows[selected], dtype=np.float32),
        np.asarray(labels[selected]).astype(str),
    )


def _class_indices(labels: np.ndarray, classes: Sequence[str]) -> np.ndarray:
    lookup = {value: index for index, value in enumerate(classes)}
    return np.asarray([lookup[str(value)] for value in labels], dtype=np.int64)


def _scaled_normalized(rows: np.ndarray, log_diag: np.ndarray) -> np.ndarray:
    multiplier = np.exp(np.asarray(log_diag, dtype=np.float32)).astype(np.float32)
    return _normalize_np(np.multiply(rows, multiplier[None, :], dtype=np.float32))


def _old_prefix_sha256(
    classes: Sequence[str], log_diag: np.ndarray, old_weights: np.ndarray
) -> str:
    digest = hashlib.sha256(b"cvs.phase2.d32.old.prefix.v1\0")
    digest.update(_canonical_json(tuple(classes)).encode("utf-8"))
    digest.update(np.ascontiguousarray(log_diag, dtype=np.float32).tobytes())
    digest.update(np.ascontiguousarray(old_weights, dtype=np.float32).tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class D32Stage2CConfig:
    """One of three complete D32 locks; no free tuning surface is exposed."""

    method_id: str = D32_GROUP_BALANCED_CAP

    def __post_init__(self) -> None:
        if str(self.method_id) not in D32_METHODS:
            raise D32InLoopSafeCapError("unknown D32 method lock")
        object.__setattr__(self, "method_id", str(self.method_id))

    @property
    def optimizer_steps(self) -> int:
        return 15 if self.method_id == D32_BIAS_RECOVERY_CAP else 10

    @property
    def learning_rate(self) -> float:
        return 0.025 if self.method_id == D32_BIAS_RECOVERY_CAP else 0.03

    @property
    def new_cvar_weight(self) -> float:
        if self.method_id == D32_NEW_CVAR_CAP:
            return 0.35
        if self.method_id == D32_BIAS_RECOVERY_CAP:
            return 0.35
        return 0.0

    @property
    def new_cvar_tail_fraction(self) -> float:
        return 0.20

    @property
    def bias_recovery_weight(self) -> float:
        return 0.15 if self.method_id == D32_BIAS_RECOVERY_CAP else 0.0

    @property
    def centroid_anchor_weight(self) -> float:
        return 0.03 if self.method_id == D32_BIAS_RECOVERY_CAP else 0.02

    @property
    def safety_delta(self) -> float:
        return 1.0e-4 if self.method_id == D32_GROUP_BALANCED_CAP else 0.10

    def audit(self) -> dict[str, Any]:
        return {
            "method_id": self.method_id,
            "optimizer": "full_batch_sgd_no_momentum_chunked_new_suffix",
            "optimizer_steps": self.optimizer_steps,
            "learning_rate": self.learning_rate,
            "old_new_group_balanced_ce": True,
            "new_class_cvar_weight": self.new_cvar_weight,
            "new_class_cvar_tail_fraction": self.new_cvar_tail_fraction,
            "bias_recovery_weight": self.bias_recovery_weight,
            "bias_recovery_target": BIAS_RECOVERY_TARGET,
            "bias_recovery_normalization": 4.0,
            "centroid_anchor_weight": self.centroid_anchor_weight,
            "inloop_safety_delta": self.safety_delta,
            "max_active_new_classes_per_step": MAX_ACTIVE_NEW_CLASSES,
        }


@dataclass(frozen=True)
class D32InLoopSafeCapState:
    schema: str
    classes: tuple[str, ...]
    old_class_count: int
    log_diag: np.ndarray
    weights: np.ndarray
    new_class_biases: np.ndarray
    support_count_by_class: np.ndarray
    stage2b_optimizer_steps: int
    stage2c_optimizer_steps: int
    selected_checkpoint_step: int
    rollback_count: int
    base_old_lock_sha256: str
    old_prefix_sha256: str
    support_gate_sha256: str
    support_gate_json: str
    config: D32Stage2CConfig

    def __post_init__(self) -> None:
        classes = tuple(str(value) for value in self.classes)
        old_count = int(self.old_class_count)
        log_diag = np.asarray(self.log_diag)
        weights = np.asarray(self.weights)
        biases = np.asarray(self.new_class_biases)
        counts = np.asarray(self.support_count_by_class)
        if (
            self.schema != SCHEMA
            or len(set(classes)) != len(classes)
            or not 1 <= old_count < len(classes)
            or log_diag.dtype != np.float32
            or log_diag.shape != (FEATURE_DIM,)
            or weights.dtype != np.float32
            or weights.shape != (len(classes), FEATURE_DIM)
            or biases.dtype != np.float32
            or biases.shape != (len(classes) - old_count,)
            or bool(np.any(biases > np.float32(0.0)))
            or counts.dtype != np.uint16
            or counts.shape != (len(classes),)
            or bool(np.any(counts < 1))
            or not np.isfinite(log_diag).all()
            or not np.isfinite(weights).all()
            or not np.isfinite(biases).all()
            or not 0 <= int(self.stage2c_optimizer_steps) <= MAX_STAGE2C_STEPS
            or not 0 <= int(self.selected_checkpoint_step) <= int(self.stage2c_optimizer_steps)
            or not 0 <= int(self.rollback_count) <= int(self.stage2c_optimizer_steps)
        ):
            raise D32InLoopSafeCapError("D32 state drift")
        expected = _old_prefix_sha256(classes[:old_count], log_diag, weights[:old_count])
        if expected != str(self.old_prefix_sha256):
            raise D32InLoopSafeCapError("D32 immutable old-prefix hash drift")
        try:
            gate = json.loads(str(self.support_gate_json))
        except json.JSONDecodeError as exc:
            raise D32InLoopSafeCapError("D32 support gate is invalid JSON") from exc
        canonical_gate = _canonical_json(gate)
        if (
            not isinstance(gate, dict)
            or not bool(gate.get("support_only_checkpoint_gate_pass"))
            or hashlib.sha256(canonical_gate.encode("utf-8")).hexdigest()
            != str(self.support_gate_sha256)
        ):
            raise D32InLoopSafeCapError("D32 external support gate drift")
        object.__setattr__(self, "classes", classes)
        object.__setattr__(self, "log_diag", _readonly(log_diag, np.float32))
        object.__setattr__(self, "weights", _readonly(weights, np.float32))
        object.__setattr__(self, "new_class_biases", _readonly(biases, np.float32))
        object.__setattr__(self, "support_count_by_class", _readonly(counts, np.uint16))
        object.__setattr__(self, "support_gate_json", canonical_gate)
        audit = self.resource_audit()
        if audit["peak_trainable_parameters"] > MAX_PEAK_TRAINABLE_PARAMETERS:
            raise D32InLoopSafeCapError("D32 peak trainable parameter cap exceeded")
        if self.persistent_state_bytes > MAX_PERSISTENT_STATE_BYTES:
            raise D32InLoopSafeCapError("D32 persistent state cap exceeded")

    @property
    def persistent_state_bytes(self) -> int:
        """Deployable predictor state; support-gate JSON stays external."""

        metadata = (
            len(self.schema.encode("utf-8"))
            + sum(len(value.encode("utf-8")) for value in self.classes)
            + len(self.base_old_lock_sha256)
            + len(self.old_prefix_sha256)
            + len(self.support_gate_sha256)
            + len(self.config.method_id.encode("utf-8"))
            + 48
        )
        return int(
            self.log_diag.nbytes
            + self.weights.nbytes
            + self.new_class_biases.nbytes
            + self.support_count_by_class.nbytes
            + metadata
        )

    def resource_audit(self) -> dict[str, Any]:
        old_count = self.old_class_count
        new_count = len(self.classes) - old_count
        stage2b_trainable = FEATURE_DIM + old_count * FEATURE_DIM
        stage2c_peak = min(MAX_ACTIVE_NEW_CLASSES, new_count) * FEATURE_DIM
        peak = max(stage2b_trainable, stage2c_peak)
        old_rows = int(np.sum(self.support_count_by_class[:old_count], dtype=np.int64))
        all_rows = int(np.sum(self.support_count_by_class, dtype=np.int64))
        stage2b_macs = (
            3
            * int(self.stage2b_optimizer_steps)
            * old_rows
            * (FEATURE_DIM + old_count * FEATURE_DIM)
        )
        # Old/new score matrices used by the in-loop cap are the same matrices
        # consumed by the CE loss; no second dense product is needed.
        stage2c_macs = (
            3
            * int(self.stage2c_optimizer_steps)
            * all_rows
            * len(self.classes)
            * FEATURE_DIM
        )
        total_steps = int(self.stage2b_optimizer_steps) + int(self.stage2c_optimizer_steps)
        return {
            "schema": RESOURCE_SCHEMA,
            "feature_dim": FEATURE_DIM,
            "class_count": len(self.classes),
            "old_class_count": old_count,
            "new_class_count": new_count,
            "stage2b_trainable_parameters": int(stage2b_trainable),
            "stage2c_total_new_weight_state_scalars": int(new_count * FEATURE_DIM),
            "stage2c_active_trainable_parameters_per_step_max": int(stage2c_peak),
            "peak_trainable_parameters": int(peak),
            "trainable_parameters": int(peak),
            "peak_trainable_parameter_cap": MAX_PEAK_TRAINABLE_PARAMETERS,
            "peak_trainable_parameter_cap_pass": peak <= MAX_PEAK_TRAINABLE_PARAMETERS,
            "stage2c_optimizer_steps": int(self.stage2c_optimizer_steps),
            "stage2c_optimizer_step_cap": MAX_STAGE2C_STEPS,
            "stage2c_optimizer_step_cap_pass": int(self.stage2c_optimizer_steps) <= MAX_STAGE2C_STEPS,
            "selected_checkpoint_step": int(self.selected_checkpoint_step),
            "rollback_count": int(self.rollback_count),
            "k1_safe_bypass": bool(self.stage2c_optimizer_steps == 0),
            "persistent_state_bytes": self.persistent_state_bytes,
            "deployable_predictor_state_bytes": self.persistent_state_bytes,
            "persistent_state_cap_bytes": MAX_PERSISTENT_STATE_BYTES,
            "persistent_state_cap_pass": self.persistent_state_bytes <= MAX_PERSISTENT_STATE_BYTES,
            "support_gate_external_evidence_bytes": len(self.support_gate_json.encode("utf-8")),
            "support_gate_external_evidence_sha256": self.support_gate_sha256,
            "support_gate_external_evidence_excluded_from_deployment_state": True,
            "stage2b_optimizer_steps": int(self.stage2b_optimizer_steps),
            "total_optimizer_steps": total_steps,
            "total_adaptation_epochs": total_steps,
            "formal_total_optimizer_step_cap": 30,
            "formal_total_optimizer_step_cap_pass": total_steps <= 30,
            "estimated_macs_per_query": int(FEATURE_DIM + len(self.classes) * FEATURE_DIM),
            "estimated_scalar_bias_adds_per_query": int(new_count),
            "estimated_adaptation_macs": int(stage2b_macs + stage2c_macs),
            "estimated_stage2b_adaptation_macs": int(stage2b_macs),
            "estimated_stage2c_adaptation_macs": int(stage2c_macs),
            "inloop_cap_reuses_registered_score_matrix": True,
            "complete_loss_trace_required": True,
            "method_lock": self.config.audit(),
            "old_log_diag_bitwise_frozen": True,
            "old_weight_prefix_bitwise_frozen": True,
            "only_new_weights_receive_gradient": True,
            "all_registered_old_and_new_support_in_every_loss": True,
            "new_class_bias_trainable_parameters": 0,
            "safety_bias_trainable_parameters": 0,
            "in_loop_bias_applied_at_step0": True,
            "safety_cap_recomputed_each_optimizer_step": True,
            "training_and_deployment_score_surface_identical": True,
            "dense_query_graph_bytes": 0,
            "query_rows_used_for_fit": 0,
            "query_labels_used_for_fit": False,
            "query_features_used_for_fit": False,
            "query_role_oracle_access": False,
            "query_true_batch_class_count_access": False,
            "query_class_quota_access": False,
            "query_batch_global_assignment": False,
            "clean_sample_access": False,
            "clean_derived_signal_access": False,
            "source_sample_access": False,
            "source_derived_signal_access": False,
            "phase2_sample_view_policy": "leo_weak_only_no_clean_access",
            "phase2_query_decision_policy": "per_sample_all_registered_classes",
            "single_received_iq_row_per_support_sample": True,
        }


@dataclass(frozen=True)
class D32InLoopSafeCapFitResult:
    state: D32InLoopSafeCapState
    loss_trace: tuple[dict[str, Any], ...]


def _cap_from_pre_correct_old(
    old_support: torch.Tensor,
    old_targets: torch.Tensor,
    old_weights: torch.Tensor,
    new_weights: torch.Tensor,
    pre_correct_mask: torch.Tensor,
    old_classes: Sequence[str],
    safety_delta: float,
) -> tuple[torch.Tensor, dict[str, int]]:
    normalized_new = F.normalize(new_weights, dim=1)
    old_scores = TEMPERATURE * (old_support @ old_weights.T)
    new_scores = TEMPERATURE * (old_support @ normalized_new.T)
    positions = torch.nonzero(pre_correct_mask, as_tuple=False).flatten()
    if len(positions) == 0:
        return torch.zeros(len(new_weights), dtype=new_weights.dtype), {}
    winning = old_scores[positions, old_targets[positions]]
    gaps = winning[:, None] - new_scores[positions] - float(safety_delta)
    raw_cap, active_local = torch.min(gaps, dim=0)
    biases = torch.minimum(raw_cap, torch.zeros_like(raw_cap))
    active_targets = old_targets[positions][active_local]
    histogram: dict[str, int] = {}
    for class_index, active in zip(active_targets.detach().tolist(), (biases < 0).detach().tolist()):
        if active:
            label = str(old_classes[int(class_index)])
            histogram[label] = histogram.get(label, 0) + 1
    return biases, histogram


def _surface(
    support: torch.Tensor,
    targets: torch.Tensor,
    old_support: torch.Tensor,
    old_targets: torch.Tensor,
    old_weights: torch.Tensor,
    new_weights: torch.Tensor,
    new_initial: torch.Tensor,
    pre_correct_mask: torch.Tensor,
    old_classes: Sequence[str],
    config: D32Stage2CConfig,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor, torch.Tensor, dict[str, int]]:
    normalized_new = F.normalize(new_weights, dim=1)
    registered = torch.cat((old_weights, normalized_new), dim=0)
    precap_logits = TEMPERATURE * (support @ registered.T)
    biases, histogram = _cap_from_pre_correct_old(
        old_support,
        old_targets,
        old_weights,
        new_weights,
        pre_correct_mask,
        old_classes,
        config.safety_delta,
    )
    postcap_logits = torch.cat(
        (
            precap_logits[:, : len(old_weights)],
            precap_logits[:, len(old_weights) :] + biases[None, :],
        ),
        dim=1,
    )
    sample_loss = F.cross_entropy(postcap_logits, targets, reduction="none")
    class_loss = torch.stack(
        [sample_loss[targets == index].mean() for index in range(len(registered))]
    )
    old_group_ce = class_loss[: len(old_weights)].mean()
    new_group_ce = class_loss[len(old_weights) :].mean()
    group_balanced_ce = 0.5 * (old_group_ce + new_group_ce)
    new_losses = class_loss[len(old_weights) :]
    tail_count = max(1, int(math.ceil(config.new_cvar_tail_fraction * len(new_losses))))
    new_cvar = torch.topk(new_losses, k=tail_count, largest=True).values.mean()
    recovery = torch.mean((F.relu(-biases - 4.0) / 4.0) ** 2)
    centroid_anchor = torch.mean((normalized_new - new_initial) ** 2)
    loss = (
        group_balanced_ce
        + config.new_cvar_weight * new_cvar
        + config.bias_recovery_weight * recovery
        + config.centroid_anchor_weight * centroid_anchor
    )
    return (
        loss,
        {
            "old_group_ce": old_group_ce,
            "new_group_ce": new_group_ce,
            "group_balanced_ce": group_balanced_ce,
            "new_cvar": new_cvar,
            "bias_recovery": recovery,
            "centroid_anchor": centroid_anchor,
            "worst_new_class_loss": new_losses.max(),
        },
        precap_logits,
        postcap_logits,
        histogram,
    )


def _accuracy_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    old_count: int,
) -> dict[str, Any]:
    predictions = logits.argmax(dim=1)
    per_class = [
        float((predictions[targets == index] == index).float().mean())
        for index in range(int(logits.shape[1]))
    ]
    old_rows = targets < old_count
    new_rows = ~old_rows
    return {
        "old_accuracy": float((predictions[old_rows] == targets[old_rows]).float().mean()),
        "new_accuracy": float((predictions[new_rows] == targets[new_rows]).float().mean()),
        "old_floor": float(min(per_class[:old_count])),
        "new_floor": float(min(per_class[old_count:])),
        "per_class": per_class,
        "predictions": predictions,
    }


def _checkpoint_key(metrics: dict[str, Any], biases: torch.Tensor, step: int) -> tuple[float, float, float, int]:
    # Biases are non-positive, so a larger mean is closer to zero.
    return (
        float(metrics["new_floor"]),
        float(metrics["new_accuracy"]),
        float(biases.detach().mean()),
        -int(step),
    )


def append_stage2c_inloop_safe_cap_suffix(
    base_state: D26CompactDiagState,
    new_support_features: np.ndarray,
    new_support_labels: Sequence[str],
    new_registered_classes: Sequence[str],
    old_support_features: np.ndarray,
    old_support_labels: Sequence[str],
    *,
    config: D32Stage2CConfig | None = None,
) -> D32InLoopSafeCapFitResult:
    """Append a safety-capped new suffix using registered support only."""

    locked = config or D32Stage2CConfig()
    if not isinstance(base_state, D26CompactDiagState):
        raise D32InLoopSafeCapError("D32 requires a validated D26 Stage2-B state")
    if base_state.old_class_count != len(base_state.classes):
        raise D32InLoopSafeCapError("D32 permits one atomic append only")
    if FEATURE_DIM + len(base_state.classes) * FEATURE_DIM > MAX_PEAK_TRAINABLE_PARAMETERS:
        raise D32InLoopSafeCapError("D32 base Stage2-B exceeds the 2016 cap")
    old_rows, old_labels, old_classes, old_k = _validate_support(
        old_support_features,
        old_support_labels,
        base_state.classes,
        name="D32 old registered support",
    )
    new_rows, new_labels, new_classes, new_k = _validate_support(
        new_support_features,
        new_support_labels,
        new_registered_classes,
        name="D32 new registered support",
    )
    if old_classes != base_state.classes or set(old_classes) & set(new_classes):
        raise D32InLoopSafeCapError("D32 registry overlap or old-order drift")
    old_rows, old_labels = _canonicalize_support(old_rows, old_labels, old_classes)
    new_rows, new_labels = _canonicalize_support(new_rows, new_labels, new_classes)
    old_x_np = _scaled_normalized(old_rows, base_state.log_diag)
    new_x_np = _scaled_normalized(new_rows, base_state.log_diag)
    old_weights_np = _normalize_np(base_state.weights)
    new_targets_local = _class_indices(new_labels, new_classes)
    new_initial_np = np.stack(
        [
            _normalize_np(
                new_x_np[new_targets_local == index]
                .mean(axis=0, keepdims=True, dtype=np.float64)
                .astype(np.float32)
            )[0]
            for index in range(len(new_classes))
        ]
    ).astype(np.float32)
    registry = old_classes + new_classes
    support_np = np.concatenate((old_x_np, new_x_np), axis=0)
    targets_np = np.concatenate(
        (
            _class_indices(old_labels, old_classes),
            new_targets_local + len(old_classes),
        )
    )
    support = torch.tensor(support_np.tolist(), dtype=torch.float32)
    targets = torch.tensor(targets_np.tolist(), dtype=torch.long)
    old_support = support[: len(old_rows)]
    old_targets = targets[: len(old_rows)]
    old_weights = torch.tensor(old_weights_np.tolist(), dtype=torch.float32)
    new_initial = torch.tensor(new_initial_np.tolist(), dtype=torch.float32)
    current_new = new_initial.detach().clone()
    with torch.no_grad():
        pre_old_scores = TEMPERATURE * (old_support @ old_weights.T)
        pre_correct_mask = pre_old_scores.argmax(dim=1) == old_targets
    groups = [
        tuple(range(start, min(start + MAX_ACTIVE_NEW_CLASSES, len(new_classes))))
        for start in range(0, len(new_classes), MAX_ACTIVE_NEW_CLASSES)
    ]
    optimizer_steps = 0 if new_k == 1 else locked.optimizer_steps
    trace: list[dict[str, Any]] = []
    rollback_count = 0
    best_key: tuple[float, float, float, int] | None = None
    best_step = 0
    best_new = current_new.detach().clone()
    best_biases = torch.zeros(len(new_classes), dtype=torch.float32)

    def evaluate(
        candidate: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor, torch.Tensor, dict[str, int], dict[str, Any], dict[str, Any], torch.Tensor, bool]:
        loss, pieces, precap, postcap, histogram = _surface(
            support,
            targets,
            old_support,
            old_targets,
            old_weights,
            candidate,
            new_initial,
            pre_correct_mask,
            old_classes,
            locked,
        )
        biases, _ = _cap_from_pre_correct_old(
            old_support,
            old_targets,
            old_weights,
            candidate,
            pre_correct_mask,
            old_classes,
            locked.safety_delta,
        )
        pre_metrics = _accuracy_metrics(precap, targets, len(old_classes))
        post_metrics = _accuracy_metrics(postcap, targets, len(old_classes))
        post_old_predictions = post_metrics["predictions"][: len(old_rows)]
        preserved = bool(
            torch.all((~pre_correct_mask) | (post_old_predictions == old_targets))
        )
        before_by_class = []
        after_by_class = []
        for index in range(len(old_classes)):
            rows = old_targets == index
            before_by_class.append(float((pre_correct_mask[rows]).float().mean()))
            after_by_class.append(float((post_old_predictions[rows] == old_targets[rows]).float().mean()))
        class_safe = all(after + 1e-12 >= before for before, after in zip(before_by_class, after_by_class))
        safe = bool(preserved and class_safe and torch.isfinite(postcap).all() and torch.isfinite(biases).all())
        return loss, pieces, precap, postcap, histogram, pre_metrics, post_metrics, biases, safe

    def record(
        *,
        step: int,
        candidate: torch.Tensor,
        gradient_norm: float,
        active: tuple[int, ...],
        rollback_applied: bool,
    ) -> tuple[tuple[float, float, float, int], torch.Tensor, bool]:
        with torch.no_grad():
            loss, pieces, _, _, histogram, pre, post, biases, safe = evaluate(candidate)
        row = {
            "phase": "stage2c_inloop_safe_cap_new_suffix_only",
            "method_id": locked.method_id,
            "step": int(step),
            "optimizer_step": int(step),
            "total_optimizer_steps": int(optimizer_steps),
            "loss": float(loss),
            "old_group_ce": float(pieces["old_group_ce"]),
            "new_group_ce": float(pieces["new_group_ce"]),
            "group_balanced_old_new_ce": float(pieces["group_balanced_ce"]),
            "new_class_cvar_loss": float(pieces["new_cvar"]),
            "new_class_cvar_weight": locked.new_cvar_weight,
            "bias_recovery_loss": float(pieces["bias_recovery"]),
            "bias_recovery_weight": locked.bias_recovery_weight,
            "bias_recovery_target": BIAS_RECOVERY_TARGET,
            "bias_recovery_normalization": 4.0,
            "centroid_anchor_loss": float(pieces["centroid_anchor"]),
            "centroid_anchor_weight": locked.centroid_anchor_weight,
            "inloop_safety_delta": locked.safety_delta,
            "worst_new_class_loss": float(pieces["worst_new_class_loss"]),
            "precap_old_support_accuracy": pre["old_accuracy"],
            "precap_new_support_accuracy": pre["new_accuracy"],
            "precap_old_support_class_floor": pre["old_floor"],
            "precap_new_support_class_floor": pre["new_floor"],
            "postcap_old_support_accuracy": post["old_accuracy"],
            "postcap_new_support_accuracy": post["new_accuracy"],
            "postcap_old_support_class_floor": post["old_floor"],
            "postcap_new_support_class_floor": post["new_floor"],
            "postcap_per_class_support_accuracy": {
                label: post["per_class"][index] for index, label in enumerate(registry)
            },
            "bias_min": float(biases.min()),
            "bias_mean": float(biases.mean()),
            "bias_max": float(biases.max()),
            "cap_active_old_class_histogram": histogram,
            "candidate_safety_pass": bool(safe),
            "rollback_applied": bool(rollback_applied),
            "selected_checkpoint": False,
            "gradient_norm": float(gradient_norm),
            "active_new_class_indices": list(active),
            "active_trainable_parameters": len(active) * FEATURE_DIM,
            "all_registered_support_rows_in_loss": len(support_np),
            "old_registered_support_rows_in_loss": len(old_rows),
            "new_registered_support_rows_in_loss": len(new_rows),
            "old_weight_update_count": 0,
            "shared_diagonal_update_count": 0,
            "runtime_dtype": "float32",
        }
        scalar_values = [
            value
            for key, value in row.items()
            if key not in {
                "phase",
                "method_id",
                "postcap_per_class_support_accuracy",
                "cap_active_old_class_histogram",
                "candidate_safety_pass",
                "rollback_applied",
                "selected_checkpoint",
                "active_new_class_indices",
                "runtime_dtype",
            }
        ]
        if not all(math.isfinite(float(value)) for value in scalar_values):
            raise D32InLoopSafeCapError("non-finite D32 loss trace")
        trace.append(row)
        return _checkpoint_key(post, biases, step), biases.detach().clone(), bool(safe)

    key, biases, safe = record(
        step=0,
        candidate=current_new,
        gradient_norm=0.0,
        active=tuple(),
        rollback_applied=False,
    )
    if not safe:
        raise D32InLoopSafeCapError("D32 centroid checkpoint failed safety cap")
    best_key = key
    best_biases = biases

    for step in range(1, optimizer_steps + 1):
        active = groups[(step - 1) % len(groups)]
        active_index = torch.tensor(active, dtype=torch.long)
        trainable = torch.nn.Parameter(current_new[active_index].detach().clone())
        optimizer = torch.optim.SGD([trainable], lr=locked.learning_rate, momentum=0.0)
        optimizer.zero_grad(set_to_none=True)
        candidate = current_new.index_copy(0, active_index, F.normalize(trainable, dim=1))
        loss, _, _, _, _, _, _, _, _ = evaluate(candidate)
        loss.backward()
        gradient_norm = float(torch.nn.utils.clip_grad_norm_([trainable], max_norm=5.0).detach())
        optimizer.step()
        with torch.no_grad():
            candidate = current_new.index_copy(0, active_index, F.normalize(trainable, dim=1))
            _, _, _, _, _, _, _, _, candidate_safe = evaluate(candidate)
        rollback = not candidate_safe
        if rollback:
            rollback_count += 1
        else:
            current_new = candidate.detach().clone()
        key, biases, safe = record(
            step=step,
            candidate=candidate,
            gradient_norm=gradient_norm,
            active=active,
            rollback_applied=rollback,
        )
        if safe and (best_key is None or key > best_key):
            best_key = key
            best_step = step
            best_new = candidate.detach().clone()
            best_biases = biases

    trace[best_step]["selected_checkpoint"] = True
    learned_new_np = np.asarray(best_new.tolist(), dtype=np.float32)
    biases_np = np.asarray(best_biases.tolist(), dtype=np.float32)
    weights = np.concatenate((base_state.weights, learned_new_np), axis=0).astype(np.float32)
    counts = np.concatenate(
        (
            np.full(len(old_classes), old_k, dtype=np.uint16),
            np.full(len(new_classes), new_k, dtype=np.uint16),
        )
    )
    selected = trace[best_step]
    gate = {
        "schema": GATE_SCHEMA,
        "method_id": locked.method_id,
        "support_only_checkpoint_gate_pass": True,
        "selected_checkpoint_step": best_step,
        "checkpoint_order": ["new_class_floor", "new_overall", "bias_closer_zero", "earlier_step"],
        "selected_postcap_old_support_accuracy": selected["postcap_old_support_accuracy"],
        "selected_postcap_new_support_accuracy": selected["postcap_new_support_accuracy"],
        "selected_postcap_old_support_class_floor": selected["postcap_old_support_class_floor"],
        "selected_postcap_new_support_class_floor": selected["postcap_new_support_class_floor"],
        "selected_bias_min": selected["bias_min"],
        "selected_bias_mean": selected["bias_mean"],
        "selected_bias_max": selected["bias_max"],
        "selected_cap_active_old_class_histogram": selected["cap_active_old_class_histogram"],
        "pre_registration_correct_old_rows_preserved": True,
        "per_old_class_non_degradation": True,
        "new_class_biases_non_positive": bool(np.all(biases_np <= 0.0)),
        "old_first_exact_tie_policy_preserved": True,
        "inloop_and_deployment_biases_identical": True,
        "new_class_biases": [float(value) for value in biases_np],
        "rollback_count": rollback_count,
        "k_shot_new": new_k,
        "k1_centroid_cap_zero_update_bypass": bool(new_k == 1),
        "stage2c_optimizer_steps": optimizer_steps,
        "support_rows_used": len(support_np),
        "old_support_rows_used_as_negative_evidence": len(old_rows),
        "registered_support_labels_only": True,
        "query_rows_used": 0,
    }
    gate_json = _canonical_json(gate)
    state = D32InLoopSafeCapState(
        schema=SCHEMA,
        classes=registry,
        old_class_count=len(old_classes),
        log_diag=np.asarray(base_state.log_diag, dtype=np.float32),
        weights=weights,
        new_class_biases=biases_np,
        support_count_by_class=counts,
        stage2b_optimizer_steps=int(base_state.stage2b_optimizer_steps),
        stage2c_optimizer_steps=optimizer_steps,
        selected_checkpoint_step=best_step,
        rollback_count=rollback_count,
        base_old_lock_sha256=base_state.old_lock_sha256,
        old_prefix_sha256=_old_prefix_sha256(old_classes, base_state.log_diag, base_state.weights),
        support_gate_sha256=hashlib.sha256(gate_json.encode("utf-8")).hexdigest(),
        support_gate_json=gate_json,
        config=locked,
    )
    if (
        state.log_diag.tobytes() != base_state.log_diag.tobytes()
        or state.weights[: len(old_classes)].tobytes() != base_state.weights.tobytes()
        or state.base_old_lock_sha256 != base_state.old_lock_sha256
    ):
        raise D32InLoopSafeCapError("D32 mutated the immutable D26 old head")
    return D32InLoopSafeCapFitResult(state=state, loss_trace=tuple(trace))


def score_all_registered(state: D32InLoopSafeCapState, features: np.ndarray) -> np.ndarray:
    rows = _validate_rows(features, name="D32 scoring features")
    transformed = _scaled_normalized(rows, state.log_diag)
    old_scores = np.float32(TEMPERATURE) * (
        transformed @ _normalize_np(state.weights[: state.old_class_count]).T
    )
    new_scores = np.float32(TEMPERATURE) * (
        transformed @ _normalize_np(state.weights[state.old_class_count :]).T
    )
    new_scores += state.new_class_biases[None, :]
    return _readonly(np.concatenate((old_scores, new_scores), axis=1), np.float32)


def predict_all_registered(state: D32InLoopSafeCapState, features: np.ndarray) -> np.ndarray:
    scores = score_all_registered(state, features)
    return np.asarray(state.classes)[np.argmax(scores, axis=1)]


__all__ = [
    "BIAS_RECOVERY_TARGET",
    "D32_BIAS_RECOVERY_CAP",
    "D32_GROUP_BALANCED_CAP",
    "D32_METHODS",
    "D32_NEW_CVAR_CAP",
    "D32InLoopSafeCapError",
    "D32InLoopSafeCapFitResult",
    "D32InLoopSafeCapState",
    "D32Stage2CConfig",
    "MAX_PEAK_TRAINABLE_PARAMETERS",
    "MAX_STAGE2C_STEPS",
    "SAFETY_DELTA",
    "append_stage2c_inloop_safe_cap_suffix",
    "predict_all_registered",
    "score_all_registered",
]
