"""D26 compact support-only adaptation in the 288-D concat identity space.

Every input row describes one already received LEO_weak IQ observation.  The
module has no clean/source loader and no query-fitting API.  Stage2-B learns a
shared diagonal and one cosine weight per old class with a tiny full-batch
optimizer.  Stage2-C freezes that complete old head and may optimize only an
atomically appended new-class suffix.

Prediction is one independent argmax over all registered classes.  The only
registration calibration parameter is one scalar new-group score bias, chosen
from a method-locked support-only grid.  The v2 default hard guard preserves
Stage2-B old-only per-class accuracy and every old-only-correct support row;
the historical joint-head bias-zero guard remains an explicit config option.
D27 additionally offers an explicit per-new-class mode whose closed-form caps
preserve those same old-only support decisions before a bounded support-LOO
coordinate search.  Query scoring remains one all-registered-class argmax.
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


FEATURE_DIM = 288
BLOCK_DIMS = (160, 96, 32)
TEMPERATURE = 18.0
LOG_DIAG_LIMIT = math.log(1.5)
MAX_STAGE2B_STEPS = 15
ALLOWED_STAGE2C_STEPS = (0, 10, 15)
MAX_TOTAL_STEPS = 30
MAX_TRAINABLE_PARAMETERS = 80_000
MAX_PERSISTENT_STATE_BYTES = 256 * 1024
NEW_GROUP_BIAS_GRID = (-12.0, -8.0, -6.0, -4.0, -3.0, -2.0, -1.0, 0.0)
NEW_CLASS_BIAS_OFFSETS = (0.0, -0.5, -1.0, -2.0, -4.0)
NEW_CLASS_BIAS_SAFETY_EPS = 1.0e-4
BIAS_GUARD_MODES = (
    "joint_bias0",
    "pre_registration_old_only",
    "per_new_class_pre_registration_old_only",
)
SCHEMA = "cvs.phase2.d26_multimodal_compact_diag.v1"


class D26CompactDiagError(ValueError):
    """Raised when support, configuration, or immutable state drifts."""


def _readonly(value: np.ndarray, dtype: Any) -> np.ndarray:
    contiguous = np.ascontiguousarray(value, dtype=dtype)
    result = np.frombuffer(contiguous.tobytes(), dtype=contiguous.dtype).reshape(
        contiguous.shape
    )
    result.setflags(write=False)
    return result


def _tensor_support_bridge(value: np.ndarray, *, dtype: torch.dtype) -> torch.Tensor:
    """Use a bounded list bridge, avoiding the NumPy2/Torch2 ABI boundary."""

    return torch.tensor(np.asarray(value).tolist(), dtype=dtype)


def _numpy_support_bridge(value: torch.Tensor, *, dtype: Any) -> np.ndarray:
    return np.asarray(value.detach().cpu().tolist(), dtype=dtype)


def _block_slices() -> tuple[slice, ...]:
    start = 0
    result: list[slice] = []
    for dimension in BLOCK_DIMS:
        result.append(slice(start, start + dimension))
        start += dimension
    return tuple(result)


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _normalize_np(rows: np.ndarray) -> np.ndarray:
    values = np.asarray(rows, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True).astype(np.float32)
    if bool(np.any(norms <= np.float32(1.0e-12))):
        raise D26CompactDiagError("D26 encountered a zero-norm feature or weight")
    return np.asarray(values / norms, dtype=np.float32)


def _validate_rows(value: np.ndarray, *, name: str) -> np.ndarray:
    rows = np.asarray(value, dtype=np.float32)
    if (
        rows.ndim != 2
        or rows.shape[1] != FEATURE_DIM
        or len(rows) < 1
        or not np.isfinite(rows).all()
    ):
        raise D26CompactDiagError(f"{name} must be finite [N,{FEATURE_DIM}]")
    if bool(np.any(np.linalg.norm(rows, axis=1) <= 1.0e-12)):
        raise D26CompactDiagError(f"{name} contains a zero-norm row")
    return np.ascontiguousarray(rows, dtype=np.float32)


def _validate_support(
    features: np.ndarray,
    labels: Sequence[str],
    classes: Sequence[str] | None,
    *,
    name: str,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...], int]:
    rows = _validate_rows(features, name=f"{name} features")
    label_values = np.asarray(tuple(str(value) for value in labels))
    if label_values.ndim != 1 or len(label_values) != len(rows):
        raise D26CompactDiagError(f"{name} labels do not match support rows")
    registry = (
        tuple(str(value) for value in classes)
        if classes is not None
        else tuple(dict.fromkeys(label_values.tolist()))
    )
    if (
        not registry
        or len(set(registry)) != len(registry)
        or any(not value for value in registry)
        or set(label_values.tolist()) != set(registry)
    ):
        raise D26CompactDiagError(f"{name} class registry drift")
    counts = [int(np.sum(label_values == value)) for value in registry]
    if min(counts) < 1 or len(set(counts)) != 1:
        raise D26CompactDiagError(f"{name} must be class-symmetric K-shot")
    return rows, label_values, registry, counts[0]


def _class_indices(labels: np.ndarray, classes: Sequence[str]) -> np.ndarray:
    mapping = {value: index for index, value in enumerate(classes)}
    return np.asarray([mapping[str(value)] for value in labels.tolist()], dtype=np.int64)


def _project_log_diag_tensor(log_diag: torch.Tensor) -> None:
    """Remove unidentifiable block means and enforce the small diagonal cap."""

    with torch.no_grad():
        for block in _block_slices():
            selected = log_diag[block]
            selected.sub_(selected.mean())
            selected.clamp_(-LOG_DIAG_LIMIT, LOG_DIAG_LIMIT)
            # A second centering makes the invariant stable after clipping.
            selected.sub_(selected.mean())
            selected.clamp_(-LOG_DIAG_LIMIT, LOG_DIAG_LIMIT)


def _scores_np(
    features: np.ndarray,
    log_diag: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    rows = _validate_rows(features, name="D26 scoring features")
    multiplier = np.exp(np.asarray(log_diag, dtype=np.float32)).astype(np.float32)
    scaled = np.multiply(rows, multiplier[None, :], dtype=np.float32)
    scores = np.float32(TEMPERATURE) * (
        _normalize_np(scaled) @ _normalize_np(weights).T
    )
    return np.asarray(scores, dtype=np.float32)


def _state_sha256(
    classes: Sequence[str],
    log_diag: np.ndarray,
    weights: np.ndarray,
    old_class_count: int,
) -> str:
    digest = hashlib.sha256(b"cvs.phase2.d26.compact.oldlock.v1\0")
    digest.update(_canonical_json_bytes(tuple(classes[:old_class_count])))
    digest.update(np.ascontiguousarray(log_diag, dtype=np.float32).tobytes())
    digest.update(
        np.ascontiguousarray(weights[:old_class_count], dtype=np.float32).tobytes()
    )
    return digest.hexdigest()


@dataclass(frozen=True)
class D26CompactDiagConfig:
    stage2b_steps: int = 15
    stage2c_steps: int = 0
    learning_rate: float = 0.01
    weight_decay: float = 0.002
    prototype_anchor_weight: float = 0.05
    diagonal_proximity_weight: float = 0.01
    new_group_bias_grid: tuple[float, ...] = NEW_GROUP_BIAS_GRID
    bias_guard_mode: str = "pre_registration_old_only"
    new_class_bias_offsets: tuple[float, ...] = NEW_CLASS_BIAS_OFFSETS

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "new_group_bias_grid", tuple(float(v) for v in self.new_group_bias_grid)
        )
        object.__setattr__(self, "bias_guard_mode", str(self.bias_guard_mode))
        object.__setattr__(
            self,
            "new_class_bias_offsets",
            tuple(float(v) for v in self.new_class_bias_offsets),
        )
        self.validate()

    def validate(self) -> None:
        if not 0 <= int(self.stage2b_steps) <= MAX_STAGE2B_STEPS:
            raise D26CompactDiagError("Stage2-B must use at most 15 full-batch steps")
        if int(self.stage2c_steps) not in ALLOWED_STAGE2C_STEPS:
            raise D26CompactDiagError("Stage2-C steps must be one of 0/10/15")
        if int(self.stage2b_steps) + int(self.stage2c_steps) > MAX_TOTAL_STEPS:
            raise D26CompactDiagError("D26 exceeds 30 total optimizer steps")
        scalars = (
            self.learning_rate,
            self.weight_decay,
            self.prototype_anchor_weight,
            self.diagonal_proximity_weight,
        )
        if not all(math.isfinite(float(v)) and float(v) >= 0.0 for v in scalars):
            raise D26CompactDiagError("D26 optimizer configuration is invalid")
        if float(self.learning_rate) <= 0.0:
            raise D26CompactDiagError("D26 learning rate must be positive")
        grid = self.new_group_bias_grid
        if (
            not grid
            or 0.0 not in grid
            or len(set(grid)) != len(grid)
            or not all(math.isfinite(v) for v in grid)
        ):
            raise D26CompactDiagError("D26 bias grid must be finite, unique, and contain 0")
        if self.bias_guard_mode not in BIAS_GUARD_MODES:
            raise D26CompactDiagError(
                "D26 bias guard mode must be joint_bias0, "
                "pre_registration_old_only, or "
                "per_new_class_pre_registration_old_only"
            )
        offsets = self.new_class_bias_offsets
        if (
            not offsets
            or 0.0 not in offsets
            or len(set(offsets)) != len(offsets)
            or not all(math.isfinite(v) and v <= 0.0 for v in offsets)
        ):
            raise D26CompactDiagError(
                "D26 per-new-class bias offsets must be finite, unique, "
                "non-positive, and contain 0"
            )


@dataclass(frozen=True)
class D26CompactDiagState:
    schema: str
    classes: tuple[str, ...]
    log_diag: np.ndarray
    weights: np.ndarray
    support_count_by_class: np.ndarray
    old_class_count: int
    stage2b_optimizer_steps: int
    stage2c_optimizer_steps: int
    new_group_bias: float
    old_lock_sha256: str
    bias_audit_json: str
    config: D26CompactDiagConfig
    new_class_biases: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.config.validate()
        classes = tuple(str(value) for value in self.classes)
        log_diag = np.asarray(self.log_diag)
        weights = np.asarray(self.weights)
        counts = np.asarray(self.support_count_by_class)
        old_count = int(self.old_class_count)
        new_class_count = len(classes) - old_count
        new_class_biases = (
            np.empty(0, dtype=np.float32)
            if self.new_class_biases is None
            else np.asarray(self.new_class_biases)
        )
        if (
            self.schema != SCHEMA
            or not classes
            or len(set(classes)) != len(classes)
            or not 1 <= old_count <= len(classes)
            or log_diag.dtype != np.float32
            or log_diag.shape != (FEATURE_DIM,)
            or weights.dtype != np.float32
            or weights.shape != (len(classes), FEATURE_DIM)
            or counts.dtype != np.uint16
            or counts.shape != (len(classes),)
            or bool(np.any(counts < 1))
            or not np.isfinite(log_diag).all()
            or not np.isfinite(weights).all()
            or not math.isfinite(float(self.new_group_bias))
            or not 0 <= int(self.stage2b_optimizer_steps) <= MAX_STAGE2B_STEPS
            or int(self.stage2c_optimizer_steps) not in ALLOWED_STAGE2C_STEPS
            or int(self.stage2b_optimizer_steps) + int(self.stage2c_optimizer_steps)
            > MAX_TOTAL_STEPS
        ):
            raise D26CompactDiagError("D26 state drift")
        if old_count == len(classes) and float(self.new_group_bias) != 0.0:
            raise D26CompactDiagError("pre-registration state cannot carry new bias")
        if (
            new_class_biases.dtype != np.float32
            or new_class_biases.ndim != 1
            or not np.isfinite(new_class_biases).all()
        ):
            raise D26CompactDiagError("D26 per-new-class bias state drift")
        if (
            self.config.bias_guard_mode
            == "per_new_class_pre_registration_old_only"
        ):
            if (
                new_class_biases.shape != (new_class_count,)
                or float(self.new_group_bias) != 0.0
            ):
                raise D26CompactDiagError(
                    "D26 safety-cap mode requires one bias per registered new class"
                )
        elif new_class_biases.shape != (0,):
            raise D26CompactDiagError(
                "D26 historical group-bias modes cannot carry a per-class bias vector"
            )
        if float(np.max(np.abs(log_diag))) > LOG_DIAG_LIMIT + 1.0e-6:
            raise D26CompactDiagError("D26 log diagonal exceeds its cap")
        if _state_sha256(classes, log_diag, weights, old_count) != self.old_lock_sha256:
            raise D26CompactDiagError("D26 old-head immutable hash drift")
        try:
            audit = json.loads(str(self.bias_audit_json))
        except json.JSONDecodeError as exc:
            raise D26CompactDiagError("D26 bias audit is invalid JSON") from exc
        if not isinstance(audit, dict):
            raise D26CompactDiagError("D26 bias audit must be one JSON object")
        object.__setattr__(self, "classes", classes)
        object.__setattr__(self, "log_diag", _readonly(log_diag, np.float32))
        object.__setattr__(self, "weights", _readonly(weights, np.float32))
        object.__setattr__(self, "support_count_by_class", _readonly(counts, np.uint16))
        object.__setattr__(
            self, "new_class_biases", _readonly(new_class_biases, np.float32)
        )
        object.__setattr__(self, "bias_audit_json", _canonical_json_bytes(audit).decode("utf-8"))
        if self.trainable_parameters > MAX_TRAINABLE_PARAMETERS:
            raise D26CompactDiagError("D26 trainable parameter cap exceeded")
        if self.persistent_state_bytes > MAX_PERSISTENT_STATE_BYTES:
            raise D26CompactDiagError("D26 persistent state cap exceeded")

    @property
    def trainable_parameters(self) -> int:
        stage2b = FEATURE_DIM + self.old_class_count * FEATURE_DIM
        stage2c = (len(self.classes) - self.old_class_count) * FEATURE_DIM
        return int(max(stage2b, stage2c))

    @property
    def persistent_state_bytes(self) -> int:
        metadata = (
            len(self.schema.encode("utf-8"))
            + sum(len(value.encode("utf-8")) for value in self.classes)
            + len(self.old_lock_sha256)
            + len(self.bias_audit_json.encode("utf-8"))
            + 32
        )
        return int(
            self.log_diag.nbytes
            + self.weights.nbytes
            + self.support_count_by_class.nbytes
            + self.new_class_biases.nbytes
            + np.dtype(np.float32).itemsize
            + metadata
        )

    def resource_audit(self) -> dict[str, Any]:
        class_count = len(self.classes)
        score_macs = FEATURE_DIM + class_count * FEATURE_DIM
        stage2b_trainable = FEATURE_DIM + self.old_class_count * FEATURE_DIM
        stage2c_trainable = (
            class_count - self.old_class_count
        ) * FEATURE_DIM
        old_support_rows = int(
            np.sum(self.support_count_by_class[: self.old_class_count], dtype=np.int64)
        )
        new_support_rows = int(
            np.sum(self.support_count_by_class[self.old_class_count :], dtype=np.int64)
        )
        stage2b_macs = (
            3
            * self.stage2b_optimizer_steps
            * old_support_rows
            * (FEATURE_DIM + self.old_class_count * FEATURE_DIM)
        )
        stage2c_macs = (
            3
            * self.stage2c_optimizer_steps
            * new_support_rows
            * score_macs
        )
        bias_audit = json.loads(self.bias_audit_json)
        bias_selection_macs = int(bias_audit.get("estimated_bias_selection_macs", 0))
        bias_candidate_evaluations = int(
            bias_audit.get("bias_candidate_evaluation_count", 0)
        )
        return {
            "schema": "cvs.phase2.d26_compact_diag_resource.v1",
            "feature_dim": FEATURE_DIM,
            "class_count": class_count,
            "old_class_count": self.old_class_count,
            "trainable_parameters": self.trainable_parameters,
            "peak_trainable_parameters": self.trainable_parameters,
            "stage2b_trainable_parameters": int(stage2b_trainable),
            "stage2c_trainable_parameters": int(stage2c_trainable),
            "trainable_parameter_cap": MAX_TRAINABLE_PARAMETERS,
            "trainable_parameter_cap_pass": (
                self.trainable_parameters <= MAX_TRAINABLE_PARAMETERS
            ),
            "persistent_state_bytes": self.persistent_state_bytes,
            "persistent_state_cap_bytes": MAX_PERSISTENT_STATE_BYTES,
            "persistent_state_cap_pass": (
                self.persistent_state_bytes <= MAX_PERSISTENT_STATE_BYTES
            ),
            "stage2b_optimizer_steps": self.stage2b_optimizer_steps,
            "stage2c_optimizer_steps": self.stage2c_optimizer_steps,
            "total_optimizer_steps": (
                self.stage2b_optimizer_steps + self.stage2c_optimizer_steps
            ),
            "optimizer_step_cap": MAX_TOTAL_STEPS,
            "optimizer_step_cap_pass": (
                self.stage2b_optimizer_steps + self.stage2c_optimizer_steps
                <= MAX_TOTAL_STEPS
            ),
            "stage2b_adaptation_epochs": self.stage2b_optimizer_steps,
            "stage2c_adaptation_epochs": self.stage2c_optimizer_steps,
            "total_adaptation_epochs": (
                self.stage2b_optimizer_steps + self.stage2c_optimizer_steps
            ),
            "formal_adaptation_epoch_cap": MAX_TOTAL_STEPS,
            "formal_adaptation_epoch_cap_pass": (
                self.stage2b_optimizer_steps + self.stage2c_optimizer_steps
                <= MAX_TOTAL_STEPS
            ),
            "estimated_macs_per_query": int(score_macs),
            "bias_trainable_parameters": 0,
            "new_group_bias_scalar_count": int(
                class_count > self.old_class_count
                and self.config.bias_guard_mode
                != "per_new_class_pre_registration_old_only"
            ),
            "new_class_bias_scalar_count": int(self.new_class_biases.size),
            "new_class_bias_vector_bytes": int(self.new_class_biases.nbytes),
            "registered_bias_additions_per_query": int(
                class_count - self.old_class_count
            ),
            "bias_additions_counted_as_macs": False,
            "estimated_adaptation_macs": int(
                stage2b_macs + stage2c_macs + bias_selection_macs
            ),
            "estimated_stage2b_adaptation_macs": int(stage2b_macs),
            "estimated_stage2c_adaptation_macs": int(stage2c_macs),
            "estimated_bias_selection_macs": bias_selection_macs,
            "bias_candidate_evaluation_count": bias_candidate_evaluations,
            "dense_query_graph_bytes": 0,
            # Conservative batch-1 bound for multiplier, scaled/normalized row,
            # normalized registered weights, norms, scores, and bias workspace.
            "estimated_query_temporary_bytes": int(
                (
                    3 * FEATURE_DIM
                    + class_count * FEATURE_DIM
                    + 3 * class_count
                    + 1
                )
                * np.dtype(np.float32).itemsize
            ),
            "estimated_peak_adam_state_bytes": int(
                2 * self.trainable_parameters * np.dtype(np.float32).itemsize
            ),
            "query_score_dtype": "float32",
            "persistent_state_dtype": "float32",
            "old_support_rows": old_support_rows,
            "new_support_rows": new_support_rows,
            "support_only": True,
            "query_rows_used_for_fit": 0,
            "query_labels_used_for_fit": False,
            "query_features_used_for_fit": False,
            "query_role_oracle_access": False,
            "query_true_batch_class_count_access": False,
            "query_class_quota_access": False,
            "query_batch_global_assignment": False,
            "query_query_graph_used": False,
            "clean_sample_access": False,
            "clean_derived_signal_access": False,
            "source_sample_access": False,
            "source_derived_signal_access": False,
            "phase2_sample_view_policy": "leo_weak_only_no_clean_access",
            "phase2_query_decision_policy": "per_sample_all_registered_classes",
            "single_received_iq_row_per_support_sample": True,
        }


@dataclass(frozen=True)
class D26CompactDiagFitResult:
    state: D26CompactDiagState
    loss_trace: tuple[dict[str, Any], ...]


def _make_state(
    *,
    classes: Sequence[str],
    log_diag: np.ndarray,
    weights: np.ndarray,
    counts: np.ndarray,
    old_class_count: int,
    stage2b_steps: int,
    stage2c_steps: int,
    new_group_bias: float,
    bias_audit: dict[str, Any],
    config: D26CompactDiagConfig,
    new_class_biases: np.ndarray | None = None,
) -> D26CompactDiagState:
    return D26CompactDiagState(
        schema=SCHEMA,
        classes=tuple(classes),
        log_diag=np.asarray(log_diag, dtype=np.float32),
        weights=np.asarray(weights, dtype=np.float32),
        support_count_by_class=np.asarray(counts, dtype=np.uint16),
        old_class_count=int(old_class_count),
        stage2b_optimizer_steps=int(stage2b_steps),
        stage2c_optimizer_steps=int(stage2c_steps),
        new_group_bias=float(new_group_bias),
        old_lock_sha256=_state_sha256(classes, log_diag, weights, old_class_count),
        bias_audit_json=_canonical_json_bytes(bias_audit).decode("utf-8"),
        config=config,
        new_class_biases=(
            None
            if new_class_biases is None
            else np.asarray(new_class_biases, dtype=np.float32)
        ),
    )


def fit_stage2b_compact_diag(
    support_features: np.ndarray,
    support_labels: Sequence[str],
    registered_classes: Sequence[str] | None = None,
    *,
    config: D26CompactDiagConfig | None = None,
) -> D26CompactDiagFitResult:
    """Fit old class weights plus one shared diagonal from old support only."""

    locked = config or D26CompactDiagConfig()
    locked.validate()
    rows, labels, classes, k_shot = _validate_support(
        support_features,
        support_labels,
        registered_classes,
        name="D26 Stage2-B",
    )
    class_count = len(classes)
    trainable = FEATURE_DIM + class_count * FEATURE_DIM
    if trainable > MAX_TRAINABLE_PARAMETERS:
        raise D26CompactDiagError("D26 Stage2-B trainable parameter cap exceeded")
    targets_np = _class_indices(labels, classes)
    x = _tensor_support_bridge(rows, dtype=torch.float32)
    targets = _tensor_support_bridge(targets_np, dtype=torch.long)
    prototypes = torch.stack(
        [F.normalize(x[targets == index].mean(dim=0), dim=0) for index in range(class_count)]
    )
    log_diag = torch.nn.Parameter(torch.zeros(FEATURE_DIM, dtype=torch.float32))
    weights = torch.nn.Parameter(prototypes.detach().clone())
    optimizer = torch.optim.AdamW(
        [log_diag, weights],
        lr=float(locked.learning_rate),
        weight_decay=float(locked.weight_decay),
    )
    trace: list[dict[str, Any]] = []
    for step in range(0, int(locked.stage2b_steps) + 1):
        if step:
            optimizer.zero_grad(set_to_none=True)
        scaled = x * torch.exp(log_diag).unsqueeze(0)
        logits = TEMPERATURE * (
            F.normalize(scaled, dim=1) @ F.normalize(weights, dim=1).T
        )
        ce = F.cross_entropy(logits, targets)
        anchor = torch.mean((F.normalize(weights, dim=1) - prototypes) ** 2)
        diag_proximity = torch.mean(log_diag**2)
        loss = (
            ce
            + float(locked.prototype_anchor_weight) * anchor
            + float(locked.diagonal_proximity_weight) * diag_proximity
        )
        gradient_norm = 0.0
        if step:
            loss.backward()
            gradient_norm = float(
                torch.nn.utils.clip_grad_norm_([log_diag, weights], max_norm=5.0).detach()
            )
            optimizer.step()
            _project_log_diag_tensor(log_diag)
            with torch.no_grad():
                scaled = x * torch.exp(log_diag).unsqueeze(0)
                logits = TEMPERATURE * (
                    F.normalize(scaled, dim=1) @ F.normalize(weights, dim=1).T
                )
                ce = F.cross_entropy(logits, targets)
                anchor = torch.mean((F.normalize(weights, dim=1) - prototypes) ** 2)
                diag_proximity = torch.mean(log_diag**2)
                loss = (
                    ce
                    + float(locked.prototype_anchor_weight) * anchor
                    + float(locked.diagonal_proximity_weight) * diag_proximity
                )
        predictions = logits.argmax(dim=1)
        per_class = [
            float((predictions[targets == index] == index).float().mean())
            for index in range(class_count)
        ]
        row = {
            "phase": "stage2b_old_support_full_batch",
            "step": step,
            "optimizer_step": step,
            "total_optimizer_steps": int(locked.stage2b_steps),
            "loss": float(loss.detach()),
            "ce_loss": float(ce.detach()),
            "prototype_anchor_loss": float(anchor.detach()),
            "diagonal_proximity_loss": float(diag_proximity.detach()),
            "gradient_norm": gradient_norm,
            "support_accuracy": float((predictions == targets).float().mean()),
            "support_class_floor": float(min(per_class)),
            "per_class_support_accuracy": {
                class_name: per_class[index]
                for index, class_name in enumerate(classes)
            },
            "max_abs_log_diag": float(torch.max(torch.abs(log_diag)).detach()),
            "learning_rate": float(locked.learning_rate),
            "prototype_anchor_weight": float(locked.prototype_anchor_weight),
            "diagonal_proximity_weight": float(locked.diagonal_proximity_weight),
            "runtime_dtype": "float32",
        }
        if not all(
            math.isfinite(float(value))
            for key, value in row.items()
            if key not in {"phase", "runtime_dtype", "per_class_support_accuracy"}
        ):
            raise D26CompactDiagError("non-finite D26 Stage2-B loss trace")
        trace.append(row)
    log_diag_np = _numpy_support_bridge(log_diag, dtype=np.float32)
    weights_np = _numpy_support_bridge(F.normalize(weights, dim=1), dtype=np.float32)
    state = _make_state(
        classes=classes,
        log_diag=log_diag_np,
        weights=weights_np,
        counts=np.full(class_count, k_shot, dtype=np.uint16),
        old_class_count=class_count,
        stage2b_steps=int(locked.stage2b_steps),
        stage2c_steps=0,
        new_group_bias=0.0,
        bias_audit={
            "schema": "cvs.phase2.d26_new_group_bias_audit.v1",
            "registration_state": "before",
            "selected_bias": 0.0,
            "query_rows_used": 0,
        },
        config=locked,
    )
    return D26CompactDiagFitResult(state=state, loss_trace=tuple(trace))


def _select_new_group_bias(
    *,
    state: D26CompactDiagState,
    new_weights: np.ndarray,
    new_rows: np.ndarray,
    new_labels: np.ndarray,
    new_classes: tuple[str, ...],
    old_rows: np.ndarray,
    old_labels: np.ndarray,
) -> tuple[float, dict[str, Any]]:
    old_scores = _scores_np(old_rows, state.log_diag, state.weights)
    new_scores_on_old = _scores_np(old_rows, state.log_diag, new_weights)
    old_truth = _class_indices(old_labels, state.classes)
    old_only_predictions = np.argmax(old_scores, axis=1)
    old_only_correct = old_only_predictions == old_truth
    per_class_old_only = {
        value: float(np.mean(old_only_correct[old_labels == value]))
        for value in state.classes
    }
    base_combined = np.concatenate((old_scores, new_scores_on_old), axis=1)
    bias0_predictions = np.argmax(base_combined, axis=1)
    bias0_correct = bias0_predictions == old_truth
    per_class_bias0 = {
        value: float(np.mean(bias0_correct[old_labels == value]))
        for value in state.classes
    }

    guard_mode = state.config.bias_guard_mode
    if guard_mode == "joint_bias0":
        guard_correct = bias0_correct
        per_class_guard = per_class_bias0
        guard_baseline_semantics = (
            "post_registration_combined_old_plus_new_head_with_new_group_bias_zero"
        )
    elif guard_mode == "pre_registration_old_only":
        guard_correct = old_only_correct
        per_class_guard = per_class_old_only
        guard_baseline_semantics = "stage2b_pre_registration_old_only_head"
    else:  # Config validation should make this unreachable; keep selection fail closed.
        raise D26CompactDiagError("D26 bias guard mode drift")

    k_shot = int(np.sum(new_labels == new_classes[0]))
    transformed_new: np.ndarray | None = None
    transformed_old_weights: np.ndarray | None = None
    current_new_weights: np.ndarray | None = None
    new_to_index: dict[str, int] | None = None
    if k_shot > 1:
        transformed_new = _normalize_np(
            new_rows * np.exp(state.log_diag.astype(np.float32))[None, :]
        )
        transformed_old_weights = _normalize_np(state.weights)
        current_new_weights = _normalize_np(new_weights)
        new_to_index = {value: index for index, value in enumerate(new_classes)}
    candidates: list[tuple[tuple[float, ...], float, dict[str, Any]]] = []
    for bias in state.config.new_group_bias_grid:
        combined_old = np.concatenate(
            (old_scores, new_scores_on_old + np.float32(bias)), axis=1
        )
        after_predictions = np.argmax(combined_old, axis=1)
        after_correct = after_predictions == old_truth
        per_class_after = {
            value: float(np.mean(after_correct[old_labels == value]))
            for value in state.classes
        }
        guard = all(
            per_class_after[value] + 1.0e-12 >= per_class_guard[value]
            for value in state.classes
        ) and bool(np.all(~guard_correct | after_correct))

        evidence: dict[str, Any] = {
            "bias": float(bias),
            "bias_guard_mode": guard_mode,
            "guard_baseline_semantics": guard_baseline_semantics,
            "old_guard_pass": bool(guard),
            "old_correct_rows_preserved": bool(
                np.all(~guard_correct | after_correct)
            ),
            "guard_baseline_correct_row_count": int(np.sum(guard_correct)),
            "selected_correct_row_count": int(np.sum(after_correct)),
            "per_old_class_guard_baseline_accuracy": per_class_guard,
            "per_old_class_accuracy": per_class_after,
            "new_support_loo_evaluated": bool(k_shot > 1),
        }

        if k_shot == 1:
            # One physical support row cannot be held out while still forming its
            # class prototype.  Only the old-support guard may select the bias.
            ranking = (1.0 if guard else 0.0, -abs(float(bias)))
            candidates.append((ranking, float(bias), evidence))
            continue

        loo_correct: list[bool] = []
        loo_margin: list[float] = []
        per_new: dict[str, dict[str, float | int]] = {}
        if (
            transformed_new is None
            or transformed_old_weights is None
            or current_new_weights is None
            or new_to_index is None
        ):
            raise D26CompactDiagError("D26 K>1 LOO state was not initialized")
        for class_name in new_classes:
            records: list[tuple[bool, float]] = []
            class_positions = np.flatnonzero(new_labels == class_name)
            for position in class_positions.tolist():
                remaining = class_positions[class_positions != position]
                loo_weight = _normalize_np(
                    transformed_new[remaining].mean(axis=0, keepdims=True)
                )[0]
                loo_weights = current_new_weights.copy()
                loo_weights[new_to_index[class_name]] = loo_weight
                old_part = np.float32(TEMPERATURE) * (
                    transformed_new[position] @ transformed_old_weights.T
                )
                new_part = np.float32(TEMPERATURE) * (
                    transformed_new[position] @ loo_weights.T
                ) + np.float32(bias)
                scores = np.concatenate((old_part, new_part))
                truth = len(state.classes) + new_to_index[class_name]
                other = np.delete(scores, truth)
                record = (
                    int(np.argmax(scores)) == truth,
                    float(scores[truth] - np.max(other)),
                )
                records.append(record)
                loo_correct.append(record[0])
                loo_margin.append(record[1])
            per_new[class_name] = {
                "loo_rows": len(records),
                "loo_accuracy": float(np.mean([record[0] for record in records])),
                "worst_margin": float(min(record[1] for record in records)),
            }
        min_new_accuracy = min(float(v["loo_accuracy"]) for v in per_new.values())
        overall_new_accuracy = float(np.mean(loo_correct))
        worst_new_margin = float(min(loo_margin))
        evidence.update({
            "per_new_class": per_new,
            "min_new_class_loo_accuracy": min_new_accuracy,
            "overall_new_loo_accuracy": overall_new_accuracy,
            "worst_new_loo_margin": worst_new_margin,
        })
        ranking = (
            1.0 if guard else 0.0,
            min_new_accuracy,
            overall_new_accuracy,
            worst_new_margin,
            -abs(float(bias)),
        )
        candidates.append((ranking, float(bias), evidence))
    feasible = [item for item in candidates if item[2]["old_guard_pass"]]

    if not feasible:
        if guard_mode == "pre_registration_old_only":
            raise D26CompactDiagError(
                "D26 strict bias guard found no pre-registration-old-only-safe bias"
            )
        # In the historical mode bias zero is its own baseline and must be feasible.
        selected = next((item for item in candidates if item[1] == 0.0), None)
        if selected is None or not selected[2]["old_guard_pass"]:
            raise D26CompactDiagError("D26 historical bias-zero guard drift")
        fallback = True
    elif k_shot == 1 and guard_mode == "pre_registration_old_only":
        selected = min(feasible, key=lambda item: (abs(item[1]), item[1]))
        fallback = False
    elif k_shot == 1:
        # Preserve the v1 K=1 decision exactly: zero bias and no pseudo-LOO.
        selected = next(item for item in feasible if item[1] == 0.0)
        fallback = False
    else:
        selected = max(feasible, key=lambda item: item[0])
        fallback = False

    selected_bias = selected[1]
    selected_evidence = selected[2]
    selection_policy = (
        "k1_closest_to_zero_with_pre_registration_old_only_guard_no_loo"
        if k_shot == 1 and guard_mode == "pre_registration_old_only"
        else "k1_safe_zero_no_pseudo_loo"
        if k_shot == 1
        else "new_support_leave_one_out_with_old_support_floor_guard"
    )
    audit = {
        "schema": "cvs.phase2.d26_new_group_bias_audit.v1",
        "selection_policy": selection_policy,
        "bias_guard_mode": guard_mode,
        "guard_baseline_semantics": guard_baseline_semantics,
        "guard_baseline_correct_row_count": int(np.sum(guard_correct)),
        "guard_baseline_support_accuracy": float(np.mean(guard_correct)),
        "per_old_class_guard_baseline_accuracy": per_class_guard,
        "per_old_class_old_only_accuracy": per_class_old_only,
        "bias0_baseline_semantics": (
            "post_registration_combined_old_plus_new_head_with_new_group_bias_zero"
        ),
        "bias0_is_not_stage2b_old_only_baseline": True,
        "selected_bias": selected_bias,
        "bias_grid": list(state.config.new_group_bias_grid),
        "fallback_to_zero": fallback,
        "old_guard_pass": bool(selected_evidence["old_guard_pass"]),
        "old_correct_rows_preserved": bool(
            selected_evidence["old_correct_rows_preserved"]
        ),
        "per_old_class_bias0_accuracy": per_class_bias0,
        "per_old_class_selected_accuracy": selected_evidence[
            "per_old_class_accuracy"
        ],
        "new_support_loo_evaluated": bool(k_shot > 1),
        "new_support_selection_rows": len(new_rows) if k_shot > 1 else 0,
        "query_rows_used": 0,
        "old_guard_support_non_degradation_guaranteed": bool(
            guard_mode == "pre_registration_old_only"
            and selected_evidence["old_guard_pass"]
        ),
        "registration_non_forgetting_guaranteed": False,
        "terminal_old_support_non_degradation_gate_required": True,
        "candidate_evidence": [item[2] for item in candidates],
    }
    if k_shot > 1:
        audit.update(
            {
                "per_new_class": selected_evidence["per_new_class"],
                "min_new_class_loo_accuracy": selected_evidence[
                    "min_new_class_loo_accuracy"
                ],
                "overall_new_loo_accuracy": selected_evidence[
                    "overall_new_loo_accuracy"
                ],
                "worst_new_loo_margin": selected_evidence[
                    "worst_new_loo_margin"
                ],
            }
        )
    return float(selected_bias), audit


def _select_new_class_safety_cap_bias(
    *,
    state: D26CompactDiagState,
    new_weights: np.ndarray,
    new_rows: np.ndarray,
    new_labels: np.ndarray,
    new_classes: tuple[str, ...],
    old_rows: np.ndarray,
    old_labels: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Select one support-only bias per new class under exact old-only caps."""

    old_scores = _scores_np(old_rows, state.log_diag, state.weights)
    new_scores_on_old = _scores_np(old_rows, state.log_diag, new_weights)
    old_truth = _class_indices(old_labels, state.classes)
    old_only_predictions = np.argmax(old_scores, axis=1)
    old_only_correct = old_only_predictions == old_truth
    correct_positions = np.flatnonzero(old_only_correct)
    if len(correct_positions) == 0:
        raise D26CompactDiagError(
            "D27 safety-cap bias requires at least one old-only-correct guard row"
        )
    per_class_old_only = {
        value: float(np.mean(old_only_correct[old_labels == value]))
        for value in state.classes
    }
    winning_old_scores = old_scores[correct_positions, old_truth[correct_positions]]
    safety_gaps = winning_old_scores[:, None] - new_scores_on_old[correct_positions]
    limiting_local_positions = np.argmin(safety_gaps, axis=0)
    raw_caps = np.min(safety_gaps, axis=0)
    caps = np.asarray(raw_caps - np.float32(NEW_CLASS_BIAS_SAFETY_EPS), dtype=np.float32)
    if not np.isfinite(caps).all():
        raise D26CompactDiagError("D27 produced a non-finite new-class safety cap")

    k_shot = int(np.sum(new_labels == new_classes[0]))
    loo_raw_scores: np.ndarray | None = None
    loo_truth: np.ndarray | None = None
    if k_shot > 1:
        transformed_new = _normalize_np(
            new_rows * np.exp(state.log_diag.astype(np.float32))[None, :]
        )
        transformed_old_weights = _normalize_np(state.weights)
        current_new_weights = _normalize_np(new_weights)
        new_to_index = {value: index for index, value in enumerate(new_classes)}
        raw_rows: list[np.ndarray] = []
        truth_rows: list[int] = []
        for position, class_name_value in enumerate(new_labels.tolist()):
            class_name = str(class_name_value)
            class_index = new_to_index[class_name]
            class_positions = np.flatnonzero(new_labels == class_name)
            remaining = class_positions[class_positions != position]
            if len(remaining) < 1:
                raise D26CompactDiagError("D27 K>1 LOO has no remaining prototype row")
            loo_weight = _normalize_np(
                transformed_new[remaining].mean(axis=0, keepdims=True)
            )[0]
            loo_weights = current_new_weights.copy()
            loo_weights[class_index] = loo_weight
            old_part = np.float32(TEMPERATURE) * (
                transformed_new[position] @ transformed_old_weights.T
            )
            new_part = np.float32(TEMPERATURE) * (
                transformed_new[position] @ loo_weights.T
            )
            raw_rows.append(
                np.concatenate((old_part, new_part)).astype(np.float32, copy=False)
            )
            truth_rows.append(len(state.classes) + class_index)
        loo_raw_scores = np.stack(raw_rows).astype(np.float32, copy=False)
        loo_truth = np.asarray(truth_rows, dtype=np.int64)

    def evaluate(candidate_biases: np.ndarray) -> dict[str, Any]:
        biases = np.asarray(candidate_biases, dtype=np.float32)
        cap_pass = bool(np.all(biases <= caps + np.float32(1.0e-7)))
        combined_old = np.concatenate(
            (old_scores, new_scores_on_old + biases[None, :]), axis=1
        )
        after_predictions = np.argmax(combined_old, axis=1)
        after_correct = after_predictions == old_truth
        per_class_after = {
            value: float(np.mean(after_correct[old_labels == value]))
            for value in state.classes
        }
        correct_rows_preserved = bool(np.all(~old_only_correct | after_correct))
        per_class_guard = all(
            per_class_after[value] + 1.0e-12 >= per_class_old_only[value]
            for value in state.classes
        )
        old_guard_pass = bool(cap_pass and correct_rows_preserved and per_class_guard)
        evidence: dict[str, Any] = {
            "biases": [float(value) for value in biases.tolist()],
            "cap_pass": cap_pass,
            "old_guard_pass": old_guard_pass,
            "old_correct_rows_preserved": correct_rows_preserved,
            "per_old_class_accuracy": per_class_after,
            "new_support_loo_evaluated": bool(k_shot > 1),
        }
        if k_shot == 1:
            return evidence
        if loo_raw_scores is None or loo_truth is None:
            raise D26CompactDiagError("D27 LOO score cache was not initialized")
        loo_scores = loo_raw_scores.copy()
        loo_scores[:, len(state.classes) :] += biases[None, :]
        loo_predictions = np.argmax(loo_scores, axis=1)
        loo_correct = loo_predictions == loo_truth
        truth_scores = loo_scores[np.arange(len(loo_scores)), loo_truth]
        masked_scores = loo_scores.copy()
        masked_scores[np.arange(len(masked_scores)), loo_truth] = -np.inf
        margins = truth_scores - np.max(masked_scores, axis=1)
        per_new = {
            class_name: {
                "loo_rows": int(np.sum(new_labels == class_name)),
                "loo_accuracy": float(
                    np.mean(loo_correct[new_labels == class_name])
                ),
                "worst_margin": float(np.min(margins[new_labels == class_name])),
            }
            for class_name in new_classes
        }
        evidence.update(
            {
                "per_new_class": per_new,
                "min_new_class_loo_accuracy": min(
                    float(value["loo_accuracy"]) for value in per_new.values()
                ),
                "overall_new_loo_accuracy": float(np.mean(loo_correct)),
                "worst_new_loo_margin": float(np.min(margins)),
            }
        )
        return evidence

    coordinate_evidence: list[dict[str, Any]] = []
    selected_biases = caps.copy()
    if k_shot > 1:
        for class_index, class_name in enumerate(new_classes):
            coordinate_candidates: list[tuple[dict[str, Any], np.ndarray]] = []
            for offset in state.config.new_class_bias_offsets:
                candidate_biases = selected_biases.copy()
                candidate_biases[class_index] = np.float32(
                    caps[class_index] + np.float32(offset)
                )
                evidence = evaluate(candidate_biases)
                evidence.update(
                    {
                        "coordinate_class": class_name,
                        "coordinate_class_index": class_index,
                        "offset_from_cap": float(offset),
                    }
                )
                if not evidence["old_guard_pass"]:
                    raise D26CompactDiagError(
                        "D27 cap-bounded coordinate candidate violated old-only guard"
                    )
                coordinate_candidates.append((evidence, candidate_biases))
                coordinate_evidence.append(
                    {
                        "coordinate_class": class_name,
                        "coordinate_class_index": class_index,
                        "offset_from_cap": float(offset),
                        "biases": evidence["biases"],
                        "cap_pass": evidence["cap_pass"],
                        "old_guard_pass": evidence["old_guard_pass"],
                        "old_correct_rows_preserved": evidence[
                            "old_correct_rows_preserved"
                        ],
                        "min_new_class_loo_accuracy": evidence[
                            "min_new_class_loo_accuracy"
                        ],
                        "overall_new_loo_accuracy": evidence[
                            "overall_new_loo_accuracy"
                        ],
                        "worst_new_loo_margin": evidence[
                            "worst_new_loo_margin"
                        ],
                    }
                )
            selected_evidence, selected_biases = max(
                coordinate_candidates,
                key=lambda item: (
                    item[0]["min_new_class_loo_accuracy"],
                    item[0]["overall_new_loo_accuracy"],
                    item[0]["worst_new_loo_margin"],
                ),
            )
            if selected_evidence["coordinate_class"] != class_name:
                raise D26CompactDiagError("D27 coordinate selection drift")

    final_evidence = evaluate(selected_biases)
    if not final_evidence["old_guard_pass"]:
        raise D26CompactDiagError("D27 selected bias vector violated old-only guard")
    cap_evidence = []
    for class_index, class_name in enumerate(new_classes):
        limiting_position = int(
            correct_positions[int(limiting_local_positions[class_index])]
        )
        cap_evidence.append(
            {
                "new_class": class_name,
                "raw_min_winning_old_minus_new_score": float(raw_caps[class_index]),
                "safety_epsilon": float(NEW_CLASS_BIAS_SAFETY_EPS),
                "safety_cap": float(caps[class_index]),
                "limiting_old_support_row_index": limiting_position,
                "limiting_old_class": str(old_labels[limiting_position]),
            }
        )
    new_class_count = len(new_classes)
    cap_score_macs = len(old_rows) * (FEATURE_DIM + new_class_count * FEATURE_DIM)
    loo_score_macs = (
        len(new_rows)
        * (len(state.classes) + new_class_count)
        * FEATURE_DIM
        if k_shot > 1
        else 0
    )
    audit: dict[str, Any] = {
        "schema": "cvs.phase2.d27_new_class_safety_cap_bias_audit.v1",
        "selection_policy": (
            "k1_direct_per_new_class_safety_cap_no_pseudo_loo"
            if k_shot == 1
            else "deterministic_per_new_class_cap_bounded_loo_coordinate_search"
        ),
        "bias_guard_mode": "per_new_class_pre_registration_old_only",
        "guard_baseline_semantics": "stage2b_pre_registration_old_only_head",
        "per_old_class_old_only_accuracy": per_class_old_only,
        "old_only_correct_row_count": int(np.sum(old_only_correct)),
        "safety_epsilon": float(NEW_CLASS_BIAS_SAFETY_EPS),
        "bias_caps": [float(value) for value in caps.tolist()],
        "bias_cap_by_new_class": {
            class_name: float(caps[index])
            for index, class_name in enumerate(new_classes)
        },
        "selected_biases": [float(value) for value in selected_biases.tolist()],
        "selected_bias_by_new_class": {
            class_name: float(selected_biases[index])
            for index, class_name in enumerate(new_classes)
        },
        "bias_offsets": list(state.config.new_class_bias_offsets),
        "cap_evidence": cap_evidence,
        "old_guard_pass": True,
        "old_correct_rows_preserved": bool(
            final_evidence["old_correct_rows_preserved"]
        ),
        "per_old_class_selected_accuracy": final_evidence[
            "per_old_class_accuracy"
        ],
        "old_guard_support_non_degradation_guaranteed": True,
        "new_support_loo_evaluated": bool(k_shot > 1),
        "new_support_selection_rows": len(new_rows) if k_shot > 1 else 0,
        "coordinate_pass_count": 1 if k_shot > 1 else 0,
        "bias_candidate_evaluation_count": len(coordinate_evidence),
        "candidate_evidence": coordinate_evidence,
        "estimated_bias_selection_macs": int(cap_score_macs + loo_score_macs),
        "query_rows_used": 0,
        "registration_non_forgetting_guaranteed": False,
        "terminal_old_support_non_degradation_gate_required": True,
    }
    if k_shot > 1:
        audit.update(
            {
                "per_new_class": final_evidence["per_new_class"],
                "min_new_class_loo_accuracy": final_evidence[
                    "min_new_class_loo_accuracy"
                ],
                "overall_new_loo_accuracy": final_evidence[
                    "overall_new_loo_accuracy"
                ],
                "worst_new_loo_margin": final_evidence[
                    "worst_new_loo_margin"
                ],
            }
        )
    return _readonly(selected_biases, np.float32), audit


def append_stage2c_new_suffix(
    state: D26CompactDiagState,
    new_support_features: np.ndarray,
    new_support_labels: Sequence[str],
    new_registered_classes: Sequence[str],
    old_guard_support_features: np.ndarray,
    old_guard_support_labels: Sequence[str],
) -> D26CompactDiagFitResult:
    """Atomically register all new classes while freezing the complete old head."""

    if state.old_class_count != len(state.classes):
        raise D26CompactDiagError("D26 permits one atomic new-class append only")
    new_rows, new_labels, new_classes, new_k = _validate_support(
        new_support_features,
        new_support_labels,
        new_registered_classes,
        name="D26 Stage2-C new",
    )
    old_rows, old_labels, old_classes, _ = _validate_support(
        old_guard_support_features,
        old_guard_support_labels,
        state.classes,
        name="D26 Stage2-C old guard",
    )
    if old_classes != state.classes or set(new_classes) & set(state.classes):
        raise D26CompactDiagError("D26 Stage2-C class registry overlaps or drifts")
    total_steps = state.stage2b_optimizer_steps + state.config.stage2c_steps
    if total_steps > MAX_TOTAL_STEPS:
        raise D26CompactDiagError("D26 Stage2-C exceeds 30 total optimizer steps")
    x_np = _normalize_np(
        new_rows * np.exp(state.log_diag.astype(np.float32))[None, :]
    )
    targets_np = _class_indices(new_labels, new_classes)
    x = _tensor_support_bridge(x_np, dtype=torch.float32)
    targets_local = _tensor_support_bridge(targets_np, dtype=torch.long)
    old_weights = _tensor_support_bridge(state.weights, dtype=torch.float32)
    prototypes = torch.stack(
        [
            F.normalize(x[targets_local == index].mean(dim=0), dim=0)
            for index in range(len(new_classes))
        ]
    )
    new_weights = torch.nn.Parameter(prototypes.detach().clone())
    optimizer = torch.optim.AdamW(
        [new_weights],
        lr=float(state.config.learning_rate),
        weight_decay=float(state.config.weight_decay),
    )
    targets_global = targets_local + len(state.classes)
    trace: list[dict[str, Any]] = []
    for step in range(0, int(state.config.stage2c_steps) + 1):
        if step:
            optimizer.zero_grad(set_to_none=True)
        logits = TEMPERATURE * torch.cat(
            (
                F.normalize(x, dim=1) @ F.normalize(old_weights, dim=1).T,
                F.normalize(x, dim=1) @ F.normalize(new_weights, dim=1).T,
            ),
            dim=1,
        )
        ce = F.cross_entropy(logits, targets_global)
        anchor = torch.mean((F.normalize(new_weights, dim=1) - prototypes) ** 2)
        loss = ce + float(state.config.prototype_anchor_weight) * anchor
        gradient_norm = 0.0
        if step:
            loss.backward()
            gradient_norm = float(
                torch.nn.utils.clip_grad_norm_([new_weights], max_norm=5.0).detach()
            )
            optimizer.step()
            with torch.no_grad():
                logits = TEMPERATURE * torch.cat(
                    (
                        F.normalize(x, dim=1) @ F.normalize(old_weights, dim=1).T,
                        F.normalize(x, dim=1) @ F.normalize(new_weights, dim=1).T,
                    ),
                    dim=1,
                )
                ce = F.cross_entropy(logits, targets_global)
                anchor = torch.mean(
                    (F.normalize(new_weights, dim=1) - prototypes) ** 2
                )
                loss = ce + float(state.config.prototype_anchor_weight) * anchor
        predictions = logits.argmax(dim=1)
        per_class = [
            float(
                (predictions[targets_local == index] == len(state.classes) + index)
                .float()
                .mean()
            )
            for index in range(len(new_classes))
        ]
        row = {
            "phase": "stage2c_new_suffix_support_full_batch",
            "step": step,
            "optimizer_step": step,
            "total_optimizer_steps": int(state.config.stage2c_steps),
            "loss": float(loss.detach()),
            "ce_loss": float(ce.detach()),
            "prototype_anchor_loss": float(anchor.detach()),
            "gradient_norm": gradient_norm,
            "new_support_accuracy": float((predictions == targets_global).float().mean()),
            "new_support_class_floor": float(min(per_class)),
            "per_class_new_support_accuracy": {
                class_name: per_class[index]
                for index, class_name in enumerate(new_classes)
            },
            "old_weight_update_count": 0,
            "shared_diagonal_update_count": 0,
            "learning_rate": float(state.config.learning_rate),
            "prototype_anchor_weight": float(state.config.prototype_anchor_weight),
            "runtime_dtype": "float32",
        }
        if not all(
            math.isfinite(float(value))
            for key, value in row.items()
            if key
            not in {"phase", "runtime_dtype", "per_class_new_support_accuracy"}
        ):
            raise D26CompactDiagError("non-finite D26 Stage2-C loss trace")
        trace.append(row)
    new_weights_np = _numpy_support_bridge(
        F.normalize(new_weights, dim=1), dtype=np.float32
    )
    selected_biases: np.ndarray | None = None
    if (
        state.config.bias_guard_mode
        == "per_new_class_pre_registration_old_only"
    ):
        selected_bias = 0.0
        selected_biases, bias_audit = _select_new_class_safety_cap_bias(
            state=state,
            new_weights=new_weights_np,
            new_rows=new_rows,
            new_labels=new_labels,
            new_classes=new_classes,
            old_rows=old_rows,
            old_labels=old_labels,
        )
    else:
        selected_bias, bias_audit = _select_new_group_bias(
            state=state,
            new_weights=new_weights_np,
            new_rows=new_rows,
            new_labels=new_labels,
            new_classes=new_classes,
            old_rows=old_rows,
            old_labels=old_labels,
        )
    classes = state.classes + new_classes
    weights = np.concatenate((state.weights, new_weights_np), axis=0).astype(np.float32)
    counts = np.concatenate(
        (state.support_count_by_class, np.full(len(new_classes), new_k, dtype=np.uint16))
    )
    appended = _make_state(
        classes=classes,
        log_diag=state.log_diag,
        weights=weights,
        counts=counts,
        old_class_count=state.old_class_count,
        stage2b_steps=state.stage2b_optimizer_steps,
        stage2c_steps=int(state.config.stage2c_steps),
        new_group_bias=selected_bias,
        bias_audit=bias_audit,
        config=state.config,
        new_class_biases=selected_biases,
    )
    before_raw = _scores_np(old_rows, state.log_diag, state.weights)
    after_raw = _scores_np(
        old_rows,
        appended.log_diag,
        appended.weights[: appended.old_class_count],
    )
    if (
        appended.old_lock_sha256 != state.old_lock_sha256
        or appended.log_diag.tobytes() != state.log_diag.tobytes()
        or appended.weights[: state.old_class_count].tobytes()
        != state.weights.tobytes()
        or not np.array_equal(before_raw, after_raw)
    ):
        raise D26CompactDiagError("D26 Stage2-C mutated old raw score prefix")
    return D26CompactDiagFitResult(state=appended, loss_trace=tuple(trace))


def score_all_registered(
    state: D26CompactDiagState,
    features: np.ndarray,
) -> np.ndarray:
    """Return independent FP32 scores against every registered class."""

    scores = _scores_np(features, state.log_diag, state.weights)
    if len(state.classes) > state.old_class_count:
        if (
            state.config.bias_guard_mode
            == "per_new_class_pre_registration_old_only"
        ):
            scores[:, state.old_class_count :] += state.new_class_biases[None, :]
        else:
            scores[:, state.old_class_count :] += np.float32(state.new_group_bias)
    return _readonly(scores, np.float32)


def predict_all_registered(
    state: D26CompactDiagState,
    features: np.ndarray,
) -> np.ndarray:
    """Perform one per-row argmax without roles, quotas, or batch assignment."""

    scores = score_all_registered(state, features)
    classes = np.asarray(state.classes)
    return classes[np.argmax(scores, axis=1)]


__all__ = [
    "ALLOWED_STAGE2C_STEPS",
    "BIAS_GUARD_MODES",
    "D26CompactDiagConfig",
    "D26CompactDiagError",
    "D26CompactDiagFitResult",
    "D26CompactDiagState",
    "FEATURE_DIM",
    "MAX_PERSISTENT_STATE_BYTES",
    "MAX_TRAINABLE_PARAMETERS",
    "NEW_CLASS_BIAS_OFFSETS",
    "NEW_CLASS_BIAS_SAFETY_EPS",
    "append_stage2c_new_suffix",
    "fit_stage2b_compact_diag",
    "predict_all_registered",
    "score_all_registered",
]
