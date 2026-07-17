"""D25-C3 support-only diagonal adaptation for the 288-D concat space.

The input is exactly one feature row per already received LEO_weak IQ.  The
160-D identity, FFT96, and RF32 descriptions remain three blocks of that one
row; this module never creates another physical sample or another LEO view.

Stage2-B may update only the 288 shared log-diagonal values.  Stage2-C freezes
that shared adapter and the complete old-class prefix byte-for-byte.  Its
default is a zero-step normalized-mean append; an explicitly locked optional
path may optimize only the new-class prototype suffix.

No fitting API accepts query data, query truth, query roles, class quotas, or
batch-global assignment information.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import struct
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F


Z_DIM = 160
FFT_DIM = 96
RF_DIM = 32
FEATURE_DIM = Z_DIM + FFT_DIM + RF_DIM
BLOCK_DIMS = (Z_DIM, FFT_DIM, RF_DIM)
DEFAULT_BLOCK_ENERGY = (5.0 / 9.0, 1.0 / 3.0, 1.0 / 9.0)
MAX_GAMMA_ABS = 0.35
MAX_STAGE2B_STEPS = 20
MAX_STAGE2C_STEPS = 30
MAX_TOTAL_STEPS = 50
FORMAL_MAX_ADAPTATION_EPOCHS = 30
EXPLORATION_MAX_ADAPTATION_EPOCHS = 45
MAX_TRAINABLE_PARAMETERS = 80_000
MAX_PERSISTENT_STATE_BYTES = 256 * 1024
SCHEMA = "cvs.phase2.d25_c3_multimodal_diag_floor.v1"


class D25C3Error(ValueError):
    """Raised when C3 support, configuration, or immutable state drifts."""


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _readonly(value: np.ndarray, dtype: Any) -> np.ndarray:
    contiguous = np.ascontiguousarray(value, dtype=dtype)
    result = np.frombuffer(contiguous.tobytes(), dtype=contiguous.dtype).reshape(
        contiguous.shape
    )
    result.setflags(write=False)
    return result


def _torch_tensor_abi_safe(value: np.ndarray, *, dtype: torch.dtype) -> torch.Tensor:
    """Bridge tiny support state without the NumPy C-API used by from_numpy.

    N607 currently pairs NumPy 2.x with a Torch build whose from_numpy bridge
    rejects even genuine numpy.ndarray objects.  C3 tensors are at most a few
    tens of thousands of support scalars, so the Python-sequence bridge is a
    bounded adaptation-time cost and never touches the per-query path.
    """

    return torch.tensor(np.asarray(value).tolist(), dtype=dtype)


def _numpy_array_abi_safe(value: torch.Tensor, *, dtype: Any) -> np.ndarray:
    """Return a detached CPU tensor through the ABI-independent list bridge."""

    return np.asarray(value.detach().cpu().tolist(), dtype=dtype)


def _block_slices() -> tuple[slice, slice, slice]:
    return (
        slice(0, Z_DIM),
        slice(Z_DIM, Z_DIM + FFT_DIM),
        slice(Z_DIM + FFT_DIM, FEATURE_DIM),
    )


@dataclass(frozen=True)
class D25C3LossWeights:
    """Semantic C3 loss weights, deliberately independent of Phase1 splits.

    These values must be supplied explicitly.  In particular, the Phase1
    ``0.07/0.63/0.30`` labeled/unlabeled/validation proportions are not C3
    loss defaults and are never inferred here.
    """

    equal_class_ce: float
    tail_cvar: float
    hard_negative_margin: float
    proximity: float

    def validate(self) -> None:
        values = (
            float(self.equal_class_ce),
            float(self.tail_cvar),
            float(self.hard_negative_margin),
            float(self.proximity),
        )
        if not all(math.isfinite(value) and value >= 0.0 for value in values):
            raise D25C3Error("C3 loss weights must be finite and non-negative")
        if sum(values) <= 0.0:
            raise D25C3Error("at least one C3 loss weight must be positive")


@dataclass(frozen=True)
class D25C3Config:
    """Method-locked support-only optimizer configuration."""

    loss_weights: D25C3LossWeights
    block_energy: tuple[float, float, float] = DEFAULT_BLOCK_ENERGY
    gamma_clip: float = MAX_GAMMA_ABS
    temperature: float = 0.10
    learning_rate: float = 0.03
    tail_fraction: float = 0.25
    support_margin: float = 0.05
    stage2b_steps: int = MAX_STAGE2B_STEPS
    stage2c_steps: int = 0
    suffix_intrusion_margin: float = 0.02

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "block_energy", tuple(float(value) for value in self.block_energy)
        )
        self.validate()

    def validate(self) -> None:
        if not isinstance(self.loss_weights, D25C3LossWeights):
            raise D25C3Error("C3 loss_weights must be explicitly configured")
        self.loss_weights.validate()
        energy = tuple(float(value) for value in self.block_energy)
        if (
            len(energy) != 3
            or not all(math.isfinite(value) and value > 0.0 for value in energy)
            or not math.isclose(sum(energy), 1.0, rel_tol=0.0, abs_tol=1.0e-7)
        ):
            raise D25C3Error("C3 block energy must be positive and sum to one")
        scalars = (
            float(self.gamma_clip),
            float(self.temperature),
            float(self.learning_rate),
            float(self.tail_fraction),
            float(self.support_margin),
            float(self.suffix_intrusion_margin),
        )
        if not all(math.isfinite(value) for value in scalars):
            raise D25C3Error("C3 configuration contains non-finite values")
        if not 0.0 < float(self.gamma_clip) <= MAX_GAMMA_ABS:
            raise D25C3Error("C3 gamma clip exceeds the +/-0.35 lock")
        if float(self.temperature) <= 0.0 or float(self.learning_rate) <= 0.0:
            raise D25C3Error("C3 temperature and learning rate must be positive")
        if not 0.0 < float(self.tail_fraction) <= 1.0:
            raise D25C3Error("C3 tail fraction must be in (0,1]")
        if not 0.0 <= float(self.support_margin) < 2.0:
            raise D25C3Error("C3 support margin is out of range")
        if not 0.0 <= float(self.suffix_intrusion_margin) < 1.0:
            raise D25C3Error("C3 suffix intrusion margin is out of range")
        before = int(self.stage2b_steps)
        after = int(self.stage2c_steps)
        if not 0 <= before <= MAX_STAGE2B_STEPS:
            raise D25C3Error("Stage2-B exceeds 20 full-batch optimizer steps")
        if not 0 <= after <= MAX_STAGE2C_STEPS:
            raise D25C3Error("Stage2-C exceeds 30 new-suffix optimizer steps")
        if before + after > MAX_TOTAL_STEPS:
            raise D25C3Error("C3 exceeds the 50 optimizer-step total")
        # Every C3 update is one full-batch pass and therefore one adaptation
        # epoch.  The 150% exploration tier stops at 45 epochs even though the
        # independent sparse-key-layer optimizer-step ceiling is 50.
        if before + after > EXPLORATION_MAX_ADAPTATION_EPOCHS:
            raise D25C3Error("C3 exceeds the 45 adaptation-epoch exploration limit")

    def lock_payload(self) -> dict[str, Any]:
        return {
            "loss_weights": {
                "equal_class_ce": float(self.loss_weights.equal_class_ce),
                "tail_cvar": float(self.loss_weights.tail_cvar),
                "hard_negative_margin": float(
                    self.loss_weights.hard_negative_margin
                ),
                "proximity": float(self.loss_weights.proximity),
                "phase1_split_semantics": False,
            },
            "block_energy": [float(value) for value in self.block_energy],
            "gamma_clip": float(self.gamma_clip),
            "temperature": float(self.temperature),
            "learning_rate": float(self.learning_rate),
            "tail_fraction": float(self.tail_fraction),
            "support_margin": float(self.support_margin),
            "stage2b_steps": int(self.stage2b_steps),
            "stage2c_steps": int(self.stage2c_steps),
            "suffix_intrusion_margin": float(self.suffix_intrusion_margin),
        }


def _validate_concat_rows(
    value: np.ndarray,
    *,
    block_energy: Sequence[float],
    name: str,
) -> np.ndarray:
    rows = np.asarray(value, dtype=np.float32)
    if rows.ndim != 2 or rows.shape[1] != FEATURE_DIM:
        raise D25C3Error(f"{name} must have shape [N,{FEATURE_DIM}]")
    if len(rows) < 1 or not np.isfinite(rows).all():
        raise D25C3Error(f"{name} is empty or contains non-finite values")
    for block, expected in zip(_block_slices(), block_energy):
        selected = rows[:, block]
        squared_norm = np.sum(
            np.multiply(selected, selected, dtype=np.float32),
            axis=1,
            dtype=np.float32,
        )
        if not np.allclose(squared_norm, float(expected), atol=2.0e-5, rtol=0.0):
            raise D25C3Error(f"{name} does not preserve locked block energy")
    total_squared_norm = np.sum(
        np.multiply(rows, rows, dtype=np.float32),
        axis=1,
        dtype=np.float32,
    )
    if not np.allclose(total_squared_norm, 1.0, atol=2.0e-5, rtol=0.0):
        raise D25C3Error(f"{name} rows are not unit-normalized")
    return np.ascontiguousarray(rows, dtype=np.float32)


def _validate_support(
    features: np.ndarray,
    labels: Sequence[str],
    registered_classes: Sequence[str],
    *,
    config: D25C3Config,
    expected_k: int | None = None,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...], int]:
    rows = _validate_concat_rows(
        features, block_energy=config.block_energy, name="C3 support features"
    )
    label_array = np.asarray(tuple(str(value) for value in labels))
    classes = tuple(str(value) for value in registered_classes)
    if (
        label_array.ndim != 1
        or len(label_array) != len(rows)
        or not classes
        or len(set(classes)) != len(classes)
        or any(not value for value in classes)
    ):
        raise D25C3Error("C3 support labels or class registry are invalid")
    if set(label_array.tolist()) != set(classes):
        raise D25C3Error("C3 support class registry drift")
    counts = [int(np.sum(label_array == label)) for label in classes]
    if min(counts) < 1 or len(set(counts)) != 1:
        raise D25C3Error("C3 support must be class-symmetric K-shot")
    k_shot = counts[0]
    if expected_k is not None and k_shot != int(expected_k):
        raise D25C3Error("C3 support K-shot drift")
    return rows, label_array, classes, k_shot


def project_block_centered_gamma(
    gamma: np.ndarray,
    *,
    clip: float = MAX_GAMMA_ABS,
) -> np.ndarray:
    """Project log diagonal onto per-block zero-mean and box constraints."""

    values = np.asarray(gamma, dtype=np.float64)
    if values.shape != (FEATURE_DIM,) or not np.isfinite(values).all():
        raise D25C3Error("C3 gamma must be one finite 288-D vector")
    if not 0.0 < float(clip) <= MAX_GAMMA_ABS:
        raise D25C3Error("C3 gamma projection clip exceeds +/-0.35")
    projected = np.empty(FEATURE_DIM, dtype=np.float64)
    for block in _block_slices():
        source = values[block]
        low = float(np.min(source) - clip)
        high = float(np.max(source) + clip)
        for _ in range(80):
            midpoint = 0.5 * (low + high)
            total = float(np.sum(np.clip(source - midpoint, -clip, clip)))
            if total > 0.0:
                low = midpoint
            else:
                high = midpoint
        target = np.clip(source - 0.5 * (low + high), -clip, clip)
        projected[block] = target
    result = projected.astype(np.float32)
    # Repair float32 summation residual without leaving the box.
    for block in _block_slices():
        target = result[block]
        for _ in range(4):
            residual = float(np.sum(target.astype(np.float64)))
            if abs(residual) <= 1.0e-7:
                break
            if residual > 0.0:
                candidates = np.flatnonzero(target > -float(clip) + 1.0e-7)
                index = int(candidates[0])
                target[index] = np.float32(max(-float(clip), float(target[index]) - residual))
            else:
                candidates = np.flatnonzero(target < float(clip) - 1.0e-7)
                index = int(candidates[0])
                target[index] = np.float32(min(float(clip), float(target[index]) - residual))
    if (
        float(np.max(np.abs(result))) > float(clip) + 1.0e-7
        or any(abs(float(np.sum(result[block], dtype=np.float64))) > 2.0e-6 for block in _block_slices())
    ):
        raise D25C3Error("C3 gamma projection invariant failure")
    return _readonly(result, np.float32)


def _torch_transform(
    rows: torch.Tensor,
    gamma: torch.Tensor,
    block_energy: Sequence[float],
) -> torch.Tensor:
    blocks: list[torch.Tensor] = []
    for block, energy in zip(_block_slices(), block_energy):
        weighted = rows[:, block] * torch.exp(gamma[block]).unsqueeze(0)
        blocks.append(F.normalize(weighted, dim=1) * math.sqrt(float(energy)))
    return torch.cat(blocks, dim=1)


def transform_concat288(
    features: np.ndarray,
    gamma: np.ndarray,
    *,
    block_energy: Sequence[float] = DEFAULT_BLOCK_ENERGY,
) -> np.ndarray:
    """Apply one shared diagonal while preserving every block's energy."""

    rows = _validate_concat_rows(
        features, block_energy=block_energy, name="C3 transform input"
    )
    projected = project_block_centered_gamma(gamma)
    multiplier = np.exp(projected.astype(np.float32)).astype(np.float32)
    weighted = np.multiply(rows, multiplier[None, :], dtype=np.float32)
    output = np.empty_like(weighted, dtype=np.float32)
    for block, energy in zip(_block_slices(), block_energy):
        selected = weighted[:, block]
        norm = np.linalg.norm(selected, axis=1, keepdims=True)
        if bool(np.any(norm <= 1.0e-12)):
            raise D25C3Error("C3 transformed block has zero norm")
        output[:, block] = (
            selected / norm * np.float32(math.sqrt(float(energy)))
        ).astype(np.float32)
    return _readonly(output, np.float32)


def _project_prototype_rows(
    rows: torch.Tensor,
    block_energy: Sequence[float],
) -> torch.Tensor:
    blocks = [
        F.normalize(rows[:, block], dim=1) * math.sqrt(float(energy))
        for block, energy in zip(_block_slices(), block_energy)
    ]
    return torch.cat(blocks, dim=1)


def _class_prototypes_torch(
    rows: torch.Tensor,
    labels: torch.Tensor,
    class_count: int,
    block_energy: Sequence[float],
) -> torch.Tensor:
    means = torch.stack([rows[labels == index].mean(dim=0) for index in range(class_count)])
    return _project_prototype_rows(means, block_energy)


def _class_loss_terms(
    logits: torch.Tensor,
    labels: torch.Tensor,
    class_count: int,
    *,
    tail_fraction: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    per_row = F.cross_entropy(logits, labels, reduction="none")
    per_class = torch.stack(
        [per_row[labels == index].mean() for index in range(class_count)]
    )
    equal = per_class.mean()
    count = max(1, int(math.ceil(float(tail_fraction) * class_count)))
    tail = torch.topk(per_class, k=count, largest=True).values.mean()
    return per_row, per_class, equal, tail


def _shared_sha256(gamma: np.ndarray, config: D25C3Config) -> str:
    digest = hashlib.sha256(b"cvs.phase2.d25_c3.shared.v1\0")
    digest.update(_canonical_json_bytes(config.lock_payload()))
    digest.update(np.ascontiguousarray(gamma, dtype=np.float32).tobytes())
    return digest.hexdigest()


def _old_prefix_sha256(
    classes: Sequence[str],
    prototypes: np.ndarray,
    counts: np.ndarray,
    old_count: int,
) -> str:
    digest = hashlib.sha256(b"cvs.phase2.d25_c3.old_prefix.v1\0")
    digest.update(_canonical_json_bytes(tuple(classes[:old_count])))
    for value in (prototypes[:old_count], counts[:old_count]):
        array = np.ascontiguousarray(value)
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(struct.pack("<B", array.ndim))
        for dimension in array.shape:
            digest.update(struct.pack("<I", int(dimension)))
        digest.update(array.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class D25C3State:
    """Immutable deployed C3 state with an append-only class suffix."""

    schema: str
    classes: tuple[str, ...]
    gamma: np.ndarray
    prototypes: np.ndarray
    support_count_by_class: np.ndarray
    old_class_count: int
    stage2b_optimizer_steps: int
    stage2c_optimizer_steps: int
    shared_sha256: str
    old_prefix_sha256: str
    config: D25C3Config

    def __post_init__(self) -> None:
        self.config.validate()
        classes = tuple(str(value) for value in self.classes)
        gamma = np.asarray(self.gamma)
        prototypes = np.asarray(self.prototypes)
        counts = np.asarray(self.support_count_by_class)
        old_count = int(self.old_class_count)
        if (
            self.schema != SCHEMA
            or not classes
            or len(set(classes)) != len(classes)
            or old_count < 1
            or old_count > len(classes)
            or gamma.dtype != np.float32
            or gamma.shape != (FEATURE_DIM,)
            or prototypes.dtype != np.float32
            or prototypes.shape != (len(classes), FEATURE_DIM)
            or counts.dtype != np.uint16
            or counts.shape != (len(classes),)
            or bool(np.any(counts < 1))
            or len(set(int(value) for value in counts.tolist())) != 1
            or not np.isfinite(gamma).all()
            or not np.isfinite(prototypes).all()
            or not 0 <= int(self.stage2b_optimizer_steps) <= MAX_STAGE2B_STEPS
            or not 0 <= int(self.stage2c_optimizer_steps) <= MAX_STAGE2C_STEPS
            or int(self.stage2b_optimizer_steps) + int(self.stage2c_optimizer_steps)
            > MAX_TOTAL_STEPS
        ):
            raise D25C3Error("C3 state drift")
        if (
            float(np.max(np.abs(gamma))) > float(self.config.gamma_clip) + 1.0e-7
            or any(
                abs(float(np.sum(gamma[block], dtype=np.float64))) > 2.0e-6
                for block in _block_slices()
            )
        ):
            raise D25C3Error("C3 persisted gamma violates projection invariants")
        _validate_concat_rows(
            prototypes,
            block_energy=self.config.block_energy,
            name="C3 prototypes",
        )
        shared = _shared_sha256(gamma, self.config)
        prefix = _old_prefix_sha256(classes, prototypes, counts, old_count)
        if shared != str(self.shared_sha256) or prefix != str(self.old_prefix_sha256):
            raise D25C3Error("C3 immutable hash drift")
        object.__setattr__(self, "classes", classes)
        object.__setattr__(self, "gamma", _readonly(gamma, np.float32))
        object.__setattr__(self, "prototypes", _readonly(prototypes, np.float32))
        object.__setattr__(
            self, "support_count_by_class", _readonly(counts, np.uint16)
        )
        object.__setattr__(self, "old_class_count", old_count)
        object.__setattr__(self, "shared_sha256", shared)
        object.__setattr__(self, "old_prefix_sha256", prefix)
        if self.persistent_state_bytes > MAX_PERSISTENT_STATE_BYTES:
            raise D25C3Error("C3 persistent state exceeds 256KiB")

    @property
    def class_count(self) -> int:
        return len(self.classes)

    @property
    def k_shot(self) -> int:
        return int(self.support_count_by_class[0])

    @property
    def persistent_state_bytes(self) -> int:
        metadata = (
            len(self.schema.encode("utf-8"))
            + sum(len(value.encode("utf-8")) for value in self.classes)
            + 2 * 64
            + 8 * 4
        )
        return int(
            self.gamma.nbytes
            + self.prototypes.nbytes
            + self.support_count_by_class.nbytes
            + metadata
        )

    def resource_audit(self) -> dict[str, Any]:
        suffix_parameters = (
            (self.class_count - self.old_class_count) * FEATURE_DIM
            if int(self.stage2c_optimizer_steps) > 0
            else 0
        )
        peak_trainable = max(FEATURE_DIM, suffix_parameters)
        before_epochs = int(self.stage2b_optimizer_steps)
        after_epochs = int(self.stage2c_optimizer_steps)
        total_epochs = before_epochs + after_epochs
        if total_epochs <= FORMAL_MAX_ADAPTATION_EPOCHS:
            resource_tier = "FORMAL_DEPLOYMENT"
        elif total_epochs <= EXPLORATION_MAX_ADAPTATION_EPOCHS:
            resource_tier = "PERFORMANCE_EXPLORATION_150PCT"
        else:  # State validation/configuration should make this unreachable.
            resource_tier = "RESOURCE_LIMIT_INVALID"
        return {
            "schema": "cvs.phase2.d25_c3.resource_audit.v1",
            "feature_dimension": FEATURE_DIM,
            "block_dimensions": list(BLOCK_DIMS),
            "block_energy": [float(value) for value in self.config.block_energy],
            "shared_adapter_trainable_parameters": FEATURE_DIM,
            "stage2b_trainable_parameters": FEATURE_DIM,
            "stage2c_optional_new_suffix_parameters": int(suffix_parameters),
            "peak_trainable_parameters": int(peak_trainable),
            "trainable_parameter_limit": MAX_TRAINABLE_PARAMETERS,
            "trainable_parameter_limit_pass": peak_trainable
            <= MAX_TRAINABLE_PARAMETERS,
            "stage2b_full_batch_optimizer_steps": int(
                self.stage2b_optimizer_steps
            ),
            "stage2c_new_suffix_optimizer_steps": int(
                self.stage2c_optimizer_steps
            ),
            "total_optimizer_steps": int(
                self.stage2b_optimizer_steps + self.stage2c_optimizer_steps
            ),
            "optimizer_step_limit": MAX_TOTAL_STEPS,
            "optimizer_step_limit_pass": (
                self.stage2b_optimizer_steps + self.stage2c_optimizer_steps
            )
            <= MAX_TOTAL_STEPS,
            "stage2b_adaptation_epochs": before_epochs,
            "stage2c_adaptation_epochs": after_epochs,
            "total_adaptation_epochs": total_epochs,
            "formal_adaptation_epoch_limit": FORMAL_MAX_ADAPTATION_EPOCHS,
            "formal_adaptation_epoch_limit_pass": total_epochs
            <= FORMAL_MAX_ADAPTATION_EPOCHS,
            "exploration_150pct_adaptation_epoch_limit": EXPLORATION_MAX_ADAPTATION_EPOCHS,
            "exploration_150pct_adaptation_epoch_limit_pass": total_epochs
            <= EXPLORATION_MAX_ADAPTATION_EPOCHS,
            "resource_tier": resource_tier,
            "persistent_state_bytes": self.persistent_state_bytes,
            "persistent_state_limit_bytes": MAX_PERSISTENT_STATE_BYTES,
            "persistent_state_limit_pass": self.persistent_state_bytes
            <= MAX_PERSISTENT_STATE_BYTES,
            "diag_transform_macs_per_query": FEATURE_DIM,
            "registered_prototype_dot_macs_per_query": self.class_count
            * FEATURE_DIM,
            "estimated_head_macs_per_query": (self.class_count + 1)
            * FEATURE_DIM,
            "compute_dtype": "fp32",
            "temporary_bytes_upper_bound": int(
                3 * FEATURE_DIM * np.dtype(np.float32).itemsize
                + self.class_count * np.dtype(np.float32).itemsize
            ),
            "dense_query_graph_bytes": 0,
            "query_rows_used_for_fit": 0,
            "query_fit": False,
            "query_truth_opened": False,
            "query_role_oracle_access": False,
            "query_true_batch_class_count_access": False,
            "query_class_quota_access": False,
            "query_batch_global_assignment": False,
            "support_row_multiplicity": 1,
            "additional_leo_overlay_count": 0,
            "old_raw_score_prefix_frozen": True,
            "old_prediction_non_forgetting_guaranteed": False,
            "requires_runner_old_support_non_degradation_gate": True,
        }


@dataclass(frozen=True)
class D25C3FitResult:
    state: D25C3State
    training_trace: tuple[dict[str, Any], ...]


def _make_state(
    *,
    classes: tuple[str, ...],
    gamma: np.ndarray,
    prototypes: np.ndarray,
    counts: np.ndarray,
    old_count: int,
    before_steps: int,
    after_steps: int,
    config: D25C3Config,
) -> D25C3State:
    gamma32 = np.ascontiguousarray(gamma, dtype=np.float32)
    prototypes32 = np.ascontiguousarray(prototypes, dtype=np.float32)
    counts16 = np.ascontiguousarray(counts, dtype=np.uint16)
    return D25C3State(
        schema=SCHEMA,
        classes=classes,
        gamma=gamma32,
        prototypes=prototypes32,
        support_count_by_class=counts16,
        old_class_count=int(old_count),
        stage2b_optimizer_steps=int(before_steps),
        stage2c_optimizer_steps=int(after_steps),
        shared_sha256=_shared_sha256(gamma32, config),
        old_prefix_sha256=_old_prefix_sha256(
            classes, prototypes32, counts16, int(old_count)
        ),
        config=config,
    )


def fit_stage2b_diag_floor(
    support_features: np.ndarray,
    support_labels: Sequence[str],
    registered_classes: Sequence[str],
    *,
    config: D25C3Config,
) -> D25C3FitResult:
    """Fit the 288 shared gamma using one support-only full batch per step."""

    config.validate()
    rows, labels, classes, k_shot = _validate_support(
        support_features,
        support_labels,
        registered_classes,
        config=config,
    )
    class_index = {label: index for index, label in enumerate(classes)}
    label_ids = np.asarray([class_index[str(value)] for value in labels], dtype=np.int64)
    gamma_np = np.zeros(FEATURE_DIM, dtype=np.float32)
    trace: list[dict[str, Any]] = []
    actual_steps = int(config.stage2b_steps)
    if k_shot == 1 or actual_steps == 0:
        actual_steps = 0
        trace.append(
            {
                "phase": "stage2b",
                "optimizer_step": 0,
                "status": "K1_IDENTITY_FALLBACK" if k_shot == 1 else "ZERO_STEP_LOCK",
                "support_rows": int(len(rows)),
                "query_rows_used": 0,
            }
        )
    else:
        device = torch.device("cpu")
        x = _torch_tensor_abi_safe(rows, dtype=torch.float32).to(device=device)
        y = _torch_tensor_abi_safe(label_ids, dtype=torch.long).to(device=device)
        gamma = torch.nn.Parameter(torch.zeros(FEATURE_DIM, device=device))
        optimizer = torch.optim.Adam([gamma], lr=float(config.learning_rate))
        weights = config.loss_weights
        class_count = len(classes)
        for step in range(1, actual_steps + 1):
            optimizer.zero_grad(set_to_none=True)
            adapted = _torch_transform(x, gamma, config.block_energy)
            prototypes = _class_prototypes_torch(
                adapted, y, class_count, config.block_energy
            )
            logits = adapted @ prototypes.T
            # The true-class training logit must exclude the row itself.
            for index in range(len(rows)):
                selected = y == y[index]
                loo_mean = adapted[selected].sum(dim=0) - adapted[index]
                loo_mean = loo_mean / float(int(selected.sum().item()) - 1)
                loo_proto = _project_prototype_rows(
                    loo_mean.unsqueeze(0), config.block_energy
                )[0]
                logits[index, y[index]] = adapted[index] @ loo_proto
            logits = logits / float(config.temperature)
            _, per_class, equal, tail = _class_loss_terms(
                logits,
                y,
                class_count,
                tail_fraction=config.tail_fraction,
            )
            true_score = logits.gather(1, y.unsqueeze(1)).squeeze(1)
            competitor = logits.masked_fill(
                F.one_hot(y, num_classes=class_count).bool(), float("-inf")
            ).max(dim=1).values
            margin_rows = F.relu(
                float(config.support_margin) / float(config.temperature)
                - (true_score - competitor)
            )
            margin_by_class = torch.stack(
                [margin_rows[y == index].mean() for index in range(class_count)]
            )
            margin_loss = margin_by_class.mean()
            proximity = gamma.square().mean()
            total = (
                float(weights.equal_class_ce) * equal
                + float(weights.tail_cvar) * tail
                + float(weights.hard_negative_margin) * margin_loss
                + float(weights.proximity) * proximity
            )
            if not bool(torch.isfinite(total)):
                raise D25C3Error("non-finite C3 Stage2-B loss")
            total.backward()
            if gamma.grad is None or not bool(torch.isfinite(gamma.grad).all()):
                raise D25C3Error("non-finite C3 Stage2-B gradient")
            gradient_norm = float(torch.linalg.vector_norm(gamma.grad).item())
            optimizer.step()
            projected = project_block_centered_gamma(
                _numpy_array_abi_safe(gamma, dtype=np.float32), clip=config.gamma_clip
            )
            with torch.no_grad():
                gamma.copy_(_torch_tensor_abi_safe(projected, dtype=torch.float32))
            trace.append(
                {
                    "phase": "stage2b",
                    "optimizer_step": step,
                    "full_batch": True,
                    "support_rows": int(len(rows)),
                    "query_rows_used": 0,
                    "total_loss": float(total.detach().item()),
                    "equal_class_ce": float(equal.detach().item()),
                    "tail_cvar": float(tail.detach().item()),
                    "hard_negative_margin": float(margin_loss.detach().item()),
                    "proximity": float(proximity.detach().item()),
                    "gradient_l2": gradient_norm,
                    "gamma_abs_max": float(np.max(np.abs(projected))),
                    "shared_gamma_sha256": hashlib.sha256(
                        np.ascontiguousarray(projected, dtype=np.float32).tobytes()
                    ).hexdigest(),
                    "gamma_block_sums": [
                        float(np.sum(projected[block], dtype=np.float64))
                        for block in _block_slices()
                    ],
                    "per_class_ce": {
                        label: float(per_class[index].detach().item())
                        for index, label in enumerate(classes)
                    },
                    "loss_weights": {
                        "equal_class_ce": float(weights.equal_class_ce),
                        "tail_cvar": float(weights.tail_cvar),
                        "hard_negative_margin": float(
                            weights.hard_negative_margin
                        ),
                        "proximity": float(weights.proximity),
                        "phase1_split_semantics": False,
                    },
                }
            )
        gamma_np = np.asarray(
            project_block_centered_gamma(
                _numpy_array_abi_safe(gamma, dtype=np.float32), clip=config.gamma_clip
            ),
            dtype=np.float32,
        ).copy()
    adapted_np = transform_concat288(
        rows, gamma_np, block_energy=config.block_energy
    )
    x_final = _torch_tensor_abi_safe(adapted_np, dtype=torch.float32)
    y_final = _torch_tensor_abi_safe(label_ids, dtype=torch.long)
    with torch.no_grad():
        prototypes_np = (
            _class_prototypes_torch(
                x_final, y_final, len(classes), config.block_energy
            )
            .cpu()
        )
        prototypes_np = _numpy_array_abi_safe(prototypes_np, dtype=np.float32)
    counts = np.full(len(classes), k_shot, dtype=np.uint16)
    state = _make_state(
        classes=classes,
        gamma=gamma_np,
        prototypes=prototypes_np,
        counts=counts,
        old_count=len(classes),
        before_steps=actual_steps,
        after_steps=0,
        config=config,
    )
    return D25C3FitResult(state=state, training_trace=tuple(trace))


def append_stage2c_new_suffix(
    state: D25C3State,
    support_features: np.ndarray,
    support_labels: Sequence[str],
    registered_classes: Sequence[str],
) -> D25C3FitResult:
    """Append target-only classes while freezing old/shared bytes exactly."""

    config = state.config
    if state.class_count != state.old_class_count:
        raise D25C3Error(
            "C3 Stage2-C is one atomic batch registration; repeated append is forbidden"
        )
    rows, labels, new_classes, k_shot = _validate_support(
        support_features,
        support_labels,
        registered_classes,
        config=config,
        expected_k=state.k_shot,
    )
    if set(new_classes).intersection(state.classes):
        raise D25C3Error("C3 new-class registry overlaps the frozen old prefix")
    transformed = transform_concat288(
        rows, state.gamma, block_energy=config.block_energy
    )
    class_index = {label: index for index, label in enumerate(new_classes)}
    label_ids = np.asarray([class_index[str(value)] for value in labels], dtype=np.int64)
    x = _torch_tensor_abi_safe(transformed, dtype=torch.float32)
    y = _torch_tensor_abi_safe(label_ids, dtype=torch.long)
    with torch.no_grad():
        initial_new = _class_prototypes_torch(
            x, y, len(new_classes), config.block_energy
        )
    trace: list[dict[str, Any]] = []
    steps = int(config.stage2c_steps)
    new_final = initial_new.detach().clone()
    if steps == 0:
        trace.append(
            {
                "phase": "stage2c",
                "optimizer_step": 0,
                "status": "ZERO_STEP_TARGET_ONLY_APPEND",
                "support_rows": int(len(rows)),
                "query_rows_used": 0,
            }
        )
    else:
        raw_new = torch.nn.Parameter(initial_new.detach().clone())
        optimizer = torch.optim.Adam([raw_new], lr=float(config.learning_rate))
        old_tensor = _torch_tensor_abi_safe(
            state.prototypes[: state.old_class_count], dtype=torch.float32
        )
        weights = config.loss_weights
        for step in range(1, steps + 1):
            optimizer.zero_grad(set_to_none=True)
            new_projected = _project_prototype_rows(raw_new, config.block_energy)
            logits = torch.cat([x @ old_tensor.T, x @ new_projected.T], dim=1)
            shifted_y = y + int(state.old_class_count)
            per_row = F.cross_entropy(
                logits / float(config.temperature), shifted_y, reduction="none"
            )
            per_class = torch.stack(
                [per_row[y == index].mean() for index in range(len(new_classes))]
            )
            equal = per_class.mean()
            tail_count = max(
                1, int(math.ceil(float(config.tail_fraction) * len(new_classes)))
            )
            tail = torch.topk(per_class, k=tail_count, largest=True).values.mean()
            true_score = logits.gather(1, shifted_y.unsqueeze(1)).squeeze(1)
            competitor = logits.masked_fill(
                F.one_hot(shifted_y, num_classes=logits.shape[1]).bool(),
                float("-inf"),
            ).max(dim=1).values
            margin_rows = F.relu(
                float(config.support_margin) - (true_score - competitor)
            )
            margin_by_class = torch.stack(
                [margin_rows[y == index].mean() for index in range(len(new_classes))]
            )
            margin_loss = margin_by_class.mean()
            # CE is class-balanced over registered new support while retaining
            # all frozen old classes in the denominator.  The intrusion term
            # separately protects the frozen old prototype shell.
            proximity = (1.0 - torch.sum(new_projected * initial_new, dim=1)).mean()
            strongest_new_at_old = torch.max(old_tensor @ new_projected.T, dim=1).values
            intrusion = F.relu(
                strongest_new_at_old
                - (1.0 - float(config.suffix_intrusion_margin))
            ).mean()
            preserve = 0.5 * (proximity + intrusion)
            total = (
                float(weights.equal_class_ce) * equal
                + float(weights.tail_cvar) * tail
                + float(weights.hard_negative_margin) * margin_loss
                + float(weights.proximity) * preserve
            )
            if not bool(torch.isfinite(total)):
                raise D25C3Error("non-finite C3 Stage2-C loss")
            total.backward()
            if raw_new.grad is None or not bool(torch.isfinite(raw_new.grad).all()):
                raise D25C3Error("non-finite C3 Stage2-C gradient")
            gradient_norm = float(torch.linalg.vector_norm(raw_new.grad).item())
            optimizer.step()
            with torch.no_grad():
                raw_new.copy_(_project_prototype_rows(raw_new, config.block_energy))
            suffix_current = _numpy_array_abi_safe(raw_new, dtype=np.float32)
            trace.append(
                {
                    "phase": "stage2c",
                    "optimizer_step": step,
                    "updated_state": "new_prototype_suffix_only",
                    "support_rows": int(len(rows)),
                    "query_rows_used": 0,
                    "total_loss": float(total.detach().item()),
                    "equal_class_ce": float(equal.detach().item()),
                    "tail_cvar": float(tail.detach().item()),
                    "hard_negative_margin": float(margin_loss.detach().item()),
                    "new_suffix_proximity": float(proximity.detach().item()),
                    "old_prototype_intrusion": float(intrusion.detach().item()),
                    "gradient_l2": gradient_norm,
                    "per_class_ce": {
                        label: float(per_class[index].detach().item())
                        for index, label in enumerate(new_classes)
                    },
                    "shared_gamma_frozen_sha256": state.shared_sha256,
                    "old_prefix_frozen_sha256": state.old_prefix_sha256,
                    "new_suffix_sha256": hashlib.sha256(
                        np.ascontiguousarray(suffix_current).tobytes()
                    ).hexdigest(),
                }
            )
        new_final = _project_prototype_rows(raw_new.detach(), config.block_energy)
    combined_classes = state.classes + new_classes
    combined_prototypes = np.concatenate(
        [state.prototypes, _numpy_array_abi_safe(new_final, dtype=np.float32)], axis=0
    )
    combined_counts = np.concatenate(
        [state.support_count_by_class, np.full(len(new_classes), k_shot, dtype=np.uint16)]
    )
    appended = _make_state(
        classes=combined_classes,
        gamma=state.gamma,
        prototypes=combined_prototypes,
        counts=combined_counts,
        old_count=state.old_class_count,
        before_steps=state.stage2b_optimizer_steps,
        after_steps=steps,
        config=config,
    )
    if (
        appended.shared_sha256 != state.shared_sha256
        or appended.old_prefix_sha256 != state.old_prefix_sha256
        or appended.gamma.tobytes() != state.gamma.tobytes()
        or appended.prototypes[: state.old_class_count].tobytes()
        != state.prototypes[: state.old_class_count].tobytes()
    ):
        raise D25C3Error("C3 Stage2-C mutated shared or old-prefix bytes")
    return D25C3FitResult(state=appended, training_trace=tuple(trace))


def score_one(state: D25C3State, feature: np.ndarray) -> np.ndarray:
    """Score one sample over all registered classes without batch context."""

    value = np.asarray(feature, dtype=np.float32)
    if value.ndim != 1 or value.shape != (FEATURE_DIM,):
        raise D25C3Error("C3 scoring accepts exactly one 288-D feature")
    transformed = transform_concat288(
        value.reshape(1, -1), state.gamma, block_energy=state.config.block_energy
    )[0]
    old = np.matmul(
        state.prototypes[: state.old_class_count], transformed
    ).astype(np.float32, copy=False)
    if state.class_count == state.old_class_count:
        return _readonly(old, np.float32)
    new = np.matmul(
        state.prototypes[state.old_class_count :], transformed
    ).astype(np.float32, copy=False)
    return _readonly(np.concatenate([old, new]), np.float32)


def predict_one(state: D25C3State, feature: np.ndarray) -> tuple[str, np.ndarray]:
    scores = score_one(state, feature)
    return state.classes[int(np.argmax(scores))], scores
