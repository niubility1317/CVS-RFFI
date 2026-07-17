"""D31 all-registered-support Stage2-C with an immutable D26 old head.

The append API consumes only registered LEO_weak support.  It freezes the
shared D26 diagonal and every old-class weight byte, initializes one weight
per new class, and optimizes only a bounded new-class suffix.  Every optimizer
step evaluates class-balanced cross entropy over *all* registered old and new
support, so old support contributes genuine negative evidence for new-class
weights without exposing a query role, query label, class quota, clean sample,
or source sample.

At most seven 288-D new weights are active in an optimizer step.  Consequently
the Stage2-C peak is at most 2,016 trainable scalars even when twenty new
classes are registered.  A deterministic support-only safety cap is applied to
the new score columns after fitting; the terminal full-refit gate requires
every pre-registration-correct old support row and every old-class support
accuracy to be preserved.  K=1 takes a centroid-only, zero-update bypass.
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


SCHEMA = "cvs.phase2.d31_all_registered_new_suffix.v1"
RESOURCE_SCHEMA = "cvs.phase2.d31_all_registered_new_suffix_resource.v1"
MAX_STAGE2C_STEPS = 15
MAX_ACTIVE_NEW_CLASSES = 7
MAX_PEAK_TRAINABLE_PARAMETERS = 2_016
SAFETY_EPS = np.float32(1.0e-4)

D31_PLAIN_BALANCED_CE = "D31-A-ALLCLASS-BALANCED"
D31_NEW_CVAR_FLOOR = "D31-B-ALLCLASS-NEW-CVAR"
D31_OLD_MARGIN_PROTECTION = "D31-C-ALLCLASS-NEW-CVAR-OLD-MARGIN"
D31_METHODS = (
    D31_PLAIN_BALANCED_CE,
    D31_NEW_CVAR_FLOOR,
    D31_OLD_MARGIN_PROTECTION,
)


class D31AllRegisteredSuffixError(ValueError):
    """Raised when the immutable state, support, or method lock drifts."""


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
        raise D31AllRegisteredSuffixError("D31 encountered a zero-norm row")
    return np.asarray(value / norms, dtype=np.float32)


def _validate_rows(value: np.ndarray, *, name: str) -> np.ndarray:
    rows = np.asarray(value, dtype=np.float32)
    if (
        rows.ndim != 2
        or rows.shape[1] != FEATURE_DIM
        or len(rows) < 1
        or not np.isfinite(rows).all()
    ):
        raise D31AllRegisteredSuffixError(
            f"{name} must be finite [N,{FEATURE_DIM}]"
        )
    if bool(np.any(np.linalg.norm(rows, axis=1) <= 1.0e-12)):
        raise D31AllRegisteredSuffixError(f"{name} contains a zero-norm row")
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
        raise D31AllRegisteredSuffixError(f"{name} registry or labels drift")
    counts = [int(np.sum(label_values == value)) for value in registry]
    if min(counts) < 1 or len(set(counts)) != 1:
        raise D31AllRegisteredSuffixError(f"{name} must be class-symmetric K-shot")
    return rows, label_values, registry, counts[0]


def _canonicalize_support(
    rows: np.ndarray, labels: np.ndarray, classes: Sequence[str]
) -> tuple[np.ndarray, np.ndarray]:
    """Make full-batch FP32 reduction order independent of caller row order."""

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
    digest = hashlib.sha256(b"cvs.phase2.d31.old.prefix.v1\0")
    digest.update(_canonical_json(tuple(classes)).encode("utf-8"))
    digest.update(np.ascontiguousarray(log_diag, dtype=np.float32).tobytes())
    digest.update(np.ascontiguousarray(old_weights, dtype=np.float32).tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class D31Stage2CConfig:
    """One of three complete method locks; no free hyperparameter surface."""

    method_id: str = D31_PLAIN_BALANCED_CE

    def __post_init__(self) -> None:
        if str(self.method_id) not in D31_METHODS:
            raise D31AllRegisteredSuffixError("unknown D31 method lock")
        object.__setattr__(self, "method_id", str(self.method_id))

    @property
    def optimizer_steps(self) -> int:
        return 15 if self.method_id == D31_OLD_MARGIN_PROTECTION else 10

    @property
    def learning_rate(self) -> float:
        return 0.05

    @property
    def centroid_anchor_weight(self) -> float:
        if self.method_id == D31_NEW_CVAR_FLOOR:
            return 0.02
        if self.method_id == D31_OLD_MARGIN_PROTECTION:
            return 0.05
        return 0.01

    @property
    def new_cvar_weight(self) -> float:
        if self.method_id == D31_NEW_CVAR_FLOOR:
            return 0.35
        if self.method_id == D31_OLD_MARGIN_PROTECTION:
            return 0.25
        return 0.0

    @property
    def new_cvar_tail_fraction(self) -> float:
        # With five new classes this is the single weakest/floor class.
        return 0.20

    @property
    def old_margin_weight(self) -> float:
        return 0.75 if self.method_id == D31_OLD_MARGIN_PROTECTION else 0.0

    @property
    def old_margin(self) -> float:
        # Cosine margin 0.05 expressed on the TEMPERATURE=18 logit scale.
        return 0.90

    def audit(self) -> dict[str, Any]:
        return {
            "method_id": self.method_id,
            "optimizer": "full_batch_sgd_no_momentum_chunked_new_suffix",
            "optimizer_steps": self.optimizer_steps,
            "learning_rate": self.learning_rate,
            "centroid_anchor_weight": self.centroid_anchor_weight,
            "class_balanced_all_registered_ce": True,
            "new_class_cvar_weight": self.new_cvar_weight,
            "new_class_cvar_tail_fraction": self.new_cvar_tail_fraction,
            "old_margin_protection_weight": self.old_margin_weight,
            "old_margin": self.old_margin,
            "max_active_new_classes_per_step": MAX_ACTIVE_NEW_CLASSES,
        }


@dataclass(frozen=True)
class D31AllRegisteredSuffixState:
    schema: str
    classes: tuple[str, ...]
    old_class_count: int
    log_diag: np.ndarray
    weights: np.ndarray
    new_class_biases: np.ndarray
    support_count_by_class: np.ndarray
    stage2b_optimizer_steps: int
    stage2c_optimizer_steps: int
    base_old_lock_sha256: str
    old_prefix_sha256: str
    support_gate_sha256: str
    support_gate_json: str
    config: D31Stage2CConfig

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
            or counts.dtype != np.uint16
            or counts.shape != (len(classes),)
            or bool(np.any(counts < 1))
            or not np.isfinite(log_diag).all()
            or not np.isfinite(weights).all()
            or not np.isfinite(biases).all()
            or not 0 <= int(self.stage2c_optimizer_steps) <= MAX_STAGE2C_STEPS
        ):
            raise D31AllRegisteredSuffixError("D31 state drift")
        expected = _old_prefix_sha256(
            classes[:old_count], log_diag, weights[:old_count]
        )
        if expected != str(self.old_prefix_sha256):
            raise D31AllRegisteredSuffixError("D31 immutable old-prefix hash drift")
        try:
            gate = json.loads(str(self.support_gate_json))
        except json.JSONDecodeError as exc:
            raise D31AllRegisteredSuffixError("D31 support gate is invalid JSON") from exc
        if not isinstance(gate, dict) or not bool(gate.get("old_support_gate_pass")):
            raise D31AllRegisteredSuffixError("D31 old-support gate did not pass")
        canonical_gate = _canonical_json(gate)
        expected_gate_sha256 = hashlib.sha256(canonical_gate.encode("utf-8")).hexdigest()
        if expected_gate_sha256 != str(self.support_gate_sha256):
            raise D31AllRegisteredSuffixError("D31 external support-gate hash drift")
        object.__setattr__(self, "classes", classes)
        object.__setattr__(self, "log_diag", _readonly(log_diag, np.float32))
        object.__setattr__(self, "weights", _readonly(weights, np.float32))
        object.__setattr__(self, "new_class_biases", _readonly(biases, np.float32))
        object.__setattr__(
            self, "support_count_by_class", _readonly(counts, np.uint16)
        )
        object.__setattr__(self, "support_gate_json", canonical_gate)
        if self.resource_audit()["peak_trainable_parameters"] > MAX_PEAK_TRAINABLE_PARAMETERS:
            raise D31AllRegisteredSuffixError("D31 peak trainable parameter cap exceeded")
        if self.persistent_state_bytes > MAX_PERSISTENT_STATE_BYTES:
            raise D31AllRegisteredSuffixError("D31 persistent state cap exceeded")

    @property
    def persistent_state_bytes(self) -> int:
        """Deployable predictor state; support-gate evidence is external."""

        metadata = (
            len(self.schema.encode("utf-8"))
            + sum(len(value.encode("utf-8")) for value in self.classes)
            + len(self.base_old_lock_sha256)
            + len(self.old_prefix_sha256)
            + len(self.support_gate_sha256)
            + len(self.config.method_id.encode("utf-8"))
            + 32
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
        total_optimizer_steps = (
            int(self.stage2b_optimizer_steps) + int(self.stage2c_optimizer_steps)
        )
        old_support_rows = int(
            np.sum(
                self.support_count_by_class[:old_count], dtype=np.int64
            )
        )
        support_rows = int(np.sum(self.support_count_by_class, dtype=np.int64))
        stage2b_macs = (
            3
            * int(self.stage2b_optimizer_steps)
            * old_support_rows
            * (FEATURE_DIM + old_count * FEATURE_DIM)
        )
        stage2c_macs = (
            3
            * int(self.stage2c_optimizer_steps)
            * support_rows
            * len(self.classes)
            * FEATURE_DIM
        )
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
            "stage2c_optimizer_step_cap_pass": (
                int(self.stage2c_optimizer_steps) <= MAX_STAGE2C_STEPS
            ),
            "k1_safe_bypass": bool(self.stage2c_optimizer_steps == 0),
            "persistent_state_bytes": self.persistent_state_bytes,
            "deployable_predictor_state_bytes": self.persistent_state_bytes,
            "persistent_state_cap_bytes": MAX_PERSISTENT_STATE_BYTES,
            "persistent_state_cap_pass": (
                self.persistent_state_bytes <= MAX_PERSISTENT_STATE_BYTES
            ),
            "support_gate_external_evidence_bytes": len(
                self.support_gate_json.encode("utf-8")
            ),
            "support_gate_external_evidence_sha256": self.support_gate_sha256,
            "support_gate_external_evidence_excluded_from_deployment_state": True,
            "stage2b_optimizer_steps": int(self.stage2b_optimizer_steps),
            "total_optimizer_steps": total_optimizer_steps,
            "total_adaptation_epochs": total_optimizer_steps,
            "formal_total_optimizer_step_cap": 30,
            "formal_total_optimizer_step_cap_pass": total_optimizer_steps <= 30,
            "estimated_macs_per_query": int(
                FEATURE_DIM + len(self.classes) * FEATURE_DIM
            ),
            "estimated_adaptation_macs": int(stage2b_macs + stage2c_macs),
            "estimated_stage2b_adaptation_macs": int(stage2b_macs),
            "estimated_stage2c_adaptation_macs": int(stage2c_macs),
            "adaptation_mac_scope": (
                "all_registered_support_forward_backward_per_chunked_new_suffix_step"
            ),
            "complete_loss_trace_required": True,
            "method_lock": self.config.audit(),
            "old_log_diag_bitwise_frozen": True,
            "old_weight_prefix_bitwise_frozen": True,
            "only_new_weights_receive_gradient": True,
            "all_registered_old_and_new_support_in_every_loss": True,
            "registered_support_labels_define_roles": True,
            "new_class_bias_trainable_parameters": 0,
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
class D31AllRegisteredSuffixFitResult:
    state: D31AllRegisteredSuffixState
    loss_trace: tuple[dict[str, Any], ...]


def _loss_components(
    support: torch.Tensor,
    targets: torch.Tensor,
    old_weights: torch.Tensor,
    new_weights: torch.Tensor,
    new_initial: torch.Tensor,
    *,
    old_class_count: int,
    config: D31Stage2CConfig,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
    registered_weights = torch.cat(
        (old_weights, F.normalize(new_weights, dim=1)), dim=0
    )
    logits = TEMPERATURE * (support @ registered_weights.T)
    sample_loss = F.cross_entropy(logits, targets, reduction="none")
    class_loss = torch.stack(
        [sample_loss[targets == index].mean() for index in range(len(registered_weights))]
    )
    balanced_ce = class_loss.mean()
    new_class_loss = class_loss[old_class_count:]
    tail_count = max(
        1, int(math.ceil(config.new_cvar_tail_fraction * len(new_class_loss)))
    )
    new_cvar = torch.topk(new_class_loss, k=tail_count, largest=True).values.mean()
    old_rows = targets < old_class_count
    true_old = logits[old_rows, targets[old_rows]]
    max_new = logits[old_rows, old_class_count:].max(dim=1).values
    old_margin = F.relu(max_new - true_old + config.old_margin).mean()
    anchor = torch.mean((F.normalize(new_weights, dim=1) - new_initial) ** 2)
    loss = (
        balanced_ce
        + config.new_cvar_weight * new_cvar
        + config.old_margin_weight * old_margin
        + config.centroid_anchor_weight * anchor
    )
    return (
        loss,
        {
            "balanced_ce": balanced_ce,
            "new_cvar": new_cvar,
            "old_margin": old_margin,
            "centroid_anchor": anchor,
            "worst_class_loss": class_loss.max(),
            "worst_new_class_loss": new_class_loss.max(),
        },
        logits,
    )


def _support_gate(
    *,
    old_features: np.ndarray,
    old_labels: np.ndarray,
    old_classes: tuple[str, ...],
    log_diag: np.ndarray,
    old_weights: np.ndarray,
    new_weights: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    transformed = _scaled_normalized(old_features, log_diag)
    old_scores = np.float32(TEMPERATURE) * (transformed @ _normalize_np(old_weights).T)
    new_scores = np.float32(TEMPERATURE) * (transformed @ _normalize_np(new_weights).T)
    truth = _class_indices(old_labels, old_classes)
    before_predictions = np.argmax(old_scores, axis=1)
    before_correct = before_predictions == truth
    correct_positions = np.flatnonzero(before_correct)
    if len(correct_positions):
        winning = old_scores[correct_positions, truth[correct_positions]]
        caps = np.min(winning[:, None] - new_scores[correct_positions], axis=0)
        # A positive registration bias is unnecessary for old protection and
        # could overturn an otherwise old-first exact tie.  Clamp at zero so
        # the registered suffix can never receive a positive safety boost.
        biases = np.minimum(
            np.asarray(caps - SAFETY_EPS, dtype=np.float32), np.float32(0.0)
        ).astype(np.float32, copy=False)
    else:
        # No previously correct row can be forgotten; keep a conservative cap.
        biases = np.full(new_scores.shape[1], -32.0, dtype=np.float32)
    combined = np.concatenate((old_scores, new_scores + biases[None, :]), axis=1)
    after_predictions = np.argmax(combined, axis=1)
    after_correct = after_predictions == truth
    before_by_class = {
        label: float(np.mean(before_correct[old_labels == label]))
        for label in old_classes
    }
    after_by_class = {
        label: float(np.mean(after_correct[old_labels == label]))
        for label in old_classes
    }
    row_guard = bool(np.all(~before_correct | after_correct))
    class_guard = all(
        after_by_class[label] + 1.0e-12 >= before_by_class[label]
        for label in old_classes
    )
    gate = {
        "schema": "cvs.phase2.d31_old_support_full_refit_gate.v1",
        "old_support_gate_pass": bool(row_guard and class_guard),
        "full_refit_gate_pass": bool(row_guard and class_guard),
        "pre_registration_correct_rows_preserved": row_guard,
        "per_old_class_non_degradation": class_guard,
        "before_old_support_accuracy": float(np.mean(before_correct)),
        "after_old_support_accuracy": float(np.mean(after_correct)),
        "before_per_old_class_accuracy": before_by_class,
        "after_per_old_class_accuracy": after_by_class,
        "new_score_safety_epsilon": float(SAFETY_EPS),
        "new_class_biases_non_positive": bool(np.all(biases <= 0.0)),
        "old_first_exact_tie_policy_preserved": True,
        "new_class_biases": [float(value) for value in biases],
        "query_rows_used": 0,
    }
    return biases, gate


def append_stage2c_all_registered_new_suffix(
    base_state: D26CompactDiagState,
    new_support_features: np.ndarray,
    new_support_labels: Sequence[str],
    new_registered_classes: Sequence[str],
    old_support_features: np.ndarray,
    old_support_labels: Sequence[str],
    *,
    config: D31Stage2CConfig | None = None,
) -> D31AllRegisteredSuffixFitResult:
    """Append a new suffix using registered support only and no query API."""

    locked = config or D31Stage2CConfig()
    if not isinstance(base_state, D26CompactDiagState):
        raise D31AllRegisteredSuffixError("D31 requires a validated D26 Stage2-B state")
    if base_state.old_class_count != len(base_state.classes):
        raise D31AllRegisteredSuffixError("D31 permits one atomic append only")
    stage2b_peak = FEATURE_DIM + len(base_state.classes) * FEATURE_DIM
    if stage2b_peak > MAX_PEAK_TRAINABLE_PARAMETERS:
        raise D31AllRegisteredSuffixError("D31 base Stage2-B exceeds the 2016 cap")
    old_rows, old_labels, old_classes, old_k = _validate_support(
        old_support_features,
        old_support_labels,
        base_state.classes,
        name="D31 old registered support",
    )
    new_rows, new_labels, new_classes, new_k = _validate_support(
        new_support_features,
        new_support_labels,
        new_registered_classes,
        name="D31 new registered support",
    )
    if old_classes != base_state.classes or set(old_classes) & set(new_classes):
        raise D31AllRegisteredSuffixError("D31 registry overlap or old-order drift")
    old_rows, old_labels = _canonicalize_support(old_rows, old_labels, old_classes)
    new_rows, new_labels = _canonicalize_support(new_rows, new_labels, new_classes)
    old_transformed = _scaled_normalized(old_rows, base_state.log_diag)
    new_transformed = _scaled_normalized(new_rows, base_state.log_diag)
    old_weights_np = _normalize_np(base_state.weights)
    new_targets_local = _class_indices(new_labels, new_classes)
    new_initial_np = np.stack(
        [
            _normalize_np(
                new_transformed[new_targets_local == index].mean(
                    axis=0, keepdims=True, dtype=np.float64
                ).astype(np.float32)
            )[0]
            for index in range(len(new_classes))
        ]
    ).astype(np.float32)
    registry = old_classes + new_classes
    support_np = np.concatenate((old_transformed, new_transformed), axis=0)
    targets_np = np.concatenate(
        (
            _class_indices(old_labels, old_classes),
            new_targets_local + len(old_classes),
        )
    )
    support = torch.tensor(support_np.tolist(), dtype=torch.float32)
    targets = torch.tensor(targets_np.tolist(), dtype=torch.long)
    old_weights = torch.tensor(old_weights_np.tolist(), dtype=torch.float32)
    new_initial = torch.tensor(new_initial_np.tolist(), dtype=torch.float32)
    current_new = new_initial.detach().clone()
    group_indices = [
        tuple(range(start, min(start + MAX_ACTIVE_NEW_CLASSES, len(new_classes))))
        for start in range(0, len(new_classes), MAX_ACTIVE_NEW_CLASSES)
    ]
    optimizer_steps = 0 if new_k == 1 else locked.optimizer_steps
    trace: list[dict[str, Any]] = []

    def record(step: int, gradient_norm: float, active: tuple[int, ...]) -> None:
        with torch.no_grad():
            loss, pieces, logits = _loss_components(
                support,
                targets,
                old_weights,
                current_new,
                new_initial,
                old_class_count=len(old_classes),
                config=locked,
            )
            predictions = logits.argmax(dim=1)
            class_accuracy = [
                float((predictions[targets == index] == index).float().mean())
                for index in range(len(registry))
            ]
        row = {
            "phase": "stage2c_all_registered_support_new_suffix_only",
            "method_id": locked.method_id,
            "step": int(step),
            "optimizer_step": int(step),
            "total_optimizer_steps": int(optimizer_steps),
            "loss": float(loss),
            "balanced_all_registered_ce": float(pieces["balanced_ce"]),
            "new_class_cvar_loss": float(pieces["new_cvar"]),
            "old_margin_protection_loss": float(pieces["old_margin"]),
            "new_centroid_anchor_loss": float(pieces["centroid_anchor"]),
            "worst_registered_class_loss": float(pieces["worst_class_loss"]),
            "worst_new_class_loss": float(pieces["worst_new_class_loss"]),
            "new_class_cvar_weight": locked.new_cvar_weight,
            "old_margin_protection_weight": locked.old_margin_weight,
            "gradient_norm": float(gradient_norm),
            "support_accuracy": float((predictions == targets).float().mean()),
            "old_support_accuracy": float(
                (predictions[: len(old_rows)] == targets[: len(old_rows)]).float().mean()
            ),
            "new_support_accuracy": float(
                (predictions[len(old_rows) :] == targets[len(old_rows) :]).float().mean()
            ),
            "old_support_class_floor": float(min(class_accuracy[: len(old_classes)])),
            "new_support_class_floor": float(min(class_accuracy[len(old_classes) :])),
            "per_class_support_accuracy": {
                label: class_accuracy[index] for index, label in enumerate(registry)
            },
            "active_new_class_indices": list(active),
            "active_trainable_parameters": len(active) * FEATURE_DIM,
            "all_registered_support_rows_in_loss": len(support_np),
            "old_registered_support_rows_in_loss": len(old_rows),
            "new_registered_support_rows_in_loss": len(new_rows),
            "old_weight_update_count": 0,
            "shared_diagonal_update_count": 0,
            "runtime_dtype": "float32",
        }
        numeric = [
            value
            for key, value in row.items()
            if key
            not in {
                "phase",
                "method_id",
                "runtime_dtype",
                "active_new_class_indices",
                "per_class_support_accuracy",
            }
        ]
        if not all(math.isfinite(float(value)) for value in numeric):
            raise D31AllRegisteredSuffixError("non-finite D31 loss trace")
        trace.append(row)

    record(0, 0.0, tuple())
    for step in range(1, optimizer_steps + 1):
        active = group_indices[(step - 1) % len(group_indices)]
        active_index = torch.tensor(active, dtype=torch.long)
        trainable = torch.nn.Parameter(current_new[active_index].detach().clone())
        optimizer = torch.optim.SGD([trainable], lr=locked.learning_rate, momentum=0.0)
        optimizer.zero_grad(set_to_none=True)
        candidate_new = current_new.index_copy(
            0, active_index, F.normalize(trainable, dim=1)
        )
        loss, _, _ = _loss_components(
            support,
            targets,
            old_weights,
            candidate_new,
            new_initial,
            old_class_count=len(old_classes),
            config=locked,
        )
        loss.backward()
        gradient_norm = float(
            torch.nn.utils.clip_grad_norm_([trainable], max_norm=5.0).detach()
        )
        optimizer.step()
        with torch.no_grad():
            current_new[active_index] = F.normalize(trainable, dim=1)
        record(step, gradient_norm, active)

    learned_new_np = np.asarray(current_new.tolist(), dtype=np.float32)
    biases, gate = _support_gate(
        old_features=old_rows,
        old_labels=old_labels,
        old_classes=old_classes,
        log_diag=base_state.log_diag,
        old_weights=base_state.weights,
        new_weights=learned_new_np,
    )
    gate.update(
        {
            "method_id": locked.method_id,
            "k_shot_new": new_k,
            "k1_centroid_only_safe_bypass": bool(new_k == 1),
            "full_refit_attempted": bool(new_k > 1),
            "full_refit_selected": bool(new_k > 1),
            "stage2c_optimizer_steps": optimizer_steps,
            "support_rows_used": len(support_np),
            "old_support_rows_used_as_negative_evidence": len(old_rows),
            "registered_support_labels_only": True,
        }
    )
    if not bool(gate["old_support_gate_pass"]):
        raise D31AllRegisteredSuffixError("D31 full-refit old-support gate failed")
    weights = np.concatenate((base_state.weights, learned_new_np), axis=0).astype(
        np.float32
    )
    counts = np.concatenate(
        (
            np.full(len(old_classes), old_k, dtype=np.uint16),
            np.full(len(new_classes), new_k, dtype=np.uint16),
        )
    )
    old_prefix = _old_prefix_sha256(
        old_classes, base_state.log_diag, base_state.weights
    )
    gate_json = _canonical_json(gate)
    state = D31AllRegisteredSuffixState(
        schema=SCHEMA,
        classes=registry,
        old_class_count=len(old_classes),
        log_diag=np.asarray(base_state.log_diag, dtype=np.float32),
        weights=weights,
        new_class_biases=biases,
        support_count_by_class=counts,
        stage2b_optimizer_steps=int(base_state.stage2b_optimizer_steps),
        stage2c_optimizer_steps=optimizer_steps,
        base_old_lock_sha256=base_state.old_lock_sha256,
        old_prefix_sha256=old_prefix,
        support_gate_sha256=hashlib.sha256(gate_json.encode("utf-8")).hexdigest(),
        support_gate_json=gate_json,
        config=locked,
    )
    if (
        state.log_diag.tobytes() != base_state.log_diag.tobytes()
        or state.weights[: len(old_classes)].tobytes() != base_state.weights.tobytes()
        or state.base_old_lock_sha256 != base_state.old_lock_sha256
    ):
        raise D31AllRegisteredSuffixError("D31 mutated the immutable D26 old head")
    return D31AllRegisteredSuffixFitResult(state=state, loss_trace=tuple(trace))


def score_all_registered(
    state: D31AllRegisteredSuffixState, features: np.ndarray
) -> np.ndarray:
    rows = _validate_rows(features, name="D31 scoring features")
    transformed = _scaled_normalized(rows, state.log_diag)
    # Keep the registered-old GEMM shape identical to the pre-registration
    # D26 scorer.  A single wider old+new GEMM can select a different BLAS
    # kernel and perturb the immutable old prefix by a few FP32 ulps even when
    # its weight bytes are unchanged.
    old_scores = np.float32(TEMPERATURE) * (
        transformed @ _normalize_np(state.weights[: state.old_class_count]).T
    )
    new_scores = np.float32(TEMPERATURE) * (
        transformed @ _normalize_np(state.weights[state.old_class_count :]).T
    )
    new_scores += state.new_class_biases[None, :]
    scores = np.concatenate((old_scores, new_scores), axis=1)
    return _readonly(scores, np.float32)


def predict_all_registered(
    state: D31AllRegisteredSuffixState, features: np.ndarray
) -> np.ndarray:
    scores = score_all_registered(state, features)
    return np.asarray(state.classes)[np.argmax(scores, axis=1)]


__all__ = [
    "D31AllRegisteredSuffixError",
    "D31AllRegisteredSuffixFitResult",
    "D31AllRegisteredSuffixState",
    "D31Stage2CConfig",
    "D31_METHODS",
    "D31_NEW_CVAR_FLOOR",
    "D31_OLD_MARGIN_PROTECTION",
    "D31_PLAIN_BALANCED_CE",
    "MAX_PEAK_TRAINABLE_PARAMETERS",
    "MAX_STAGE2C_STEPS",
    "append_stage2c_all_registered_new_suffix",
    "predict_all_registered",
    "score_all_registered",
]
