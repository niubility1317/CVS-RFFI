"""D38 full-batch B3-geometry adaptation with residual-int8 identities.

The module consumes admitted 288-D support features only.  Stage2-B learns a
shared positive diagonal operator and target-old directions in twenty
full-batch steps.  The old directions are compiled before Stage2-C.  Stage2-C
then freezes the operator and decoded int8 old head while either registering
new centroids (arm A) or optimizing only new directions for ten steps (arm B).

No query, source sample, clean sample, role, quota, or batch-assignment surface
is present in this API.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np
import torch
import torch.nn.functional as F


SCHEMA_INT8 = "cvs.phase2.d38.fullbatch_b3_residual_int8.v1"
SCHEMA_FP32 = "cvs.phase2.d38.fullbatch_b3_fp32_ablation.v1"
FEATURE_DIM = 288
TEMPERATURE = 18.0
D38_SCORE_TEMPERATURE = TEMPERATURE
BLOCK_SLICES = (slice(0, 160), slice(160, 256), slice(256, 288))
ALLOWED_NEW_CLASS_COUNTS = (2, 5, 10, 20)
STAGE2B_STEPS = 20
STAGE2C_STEPS = 10
STAGE2B_LR = 0.01
STAGE2C_LR = 0.05
WEIGHT_DECAY = 0.002
FEATURE_NOISE_STD = 0.01
PROTOTYPE_ANCHOR_WEIGHT = 0.05
WORST_CLASS_WEIGHT = 0.20
WORST_CLASS_TAU = 0.25
NEW_ANCHOR_WEIGHT = 0.01
GRAD_CLIP = 5.0
LOG_SCALE_LIMIT = 1.5
FFT_LOG_SCALE_LIMIT = math.log(1.5)


class D38StrongB3QuantizedError(ValueError):
    pass


def _readonly(value: np.ndarray, dtype: Any) -> np.ndarray:
    contiguous = np.ascontiguousarray(value, dtype=dtype)
    result = np.frombuffer(contiguous.tobytes(), dtype=contiguous.dtype).reshape(
        contiguous.shape
    )
    result.setflags(write=False)
    return result


def _rows(value: np.ndarray, name: str) -> np.ndarray:
    rows = np.asarray(value)
    if (
        rows.dtype != np.float32
        or rows.ndim != 2
        or rows.shape[1] != FEATURE_DIM
        or len(rows) < 1
        or not np.isfinite(rows).all()
    ):
        raise D38StrongB3QuantizedError(
            f"{name} must be finite float32 [N,{FEATURE_DIM}]"
        )
    return np.ascontiguousarray(rows)


def _normalize_numpy(value: np.ndarray) -> np.ndarray:
    rows = np.asarray(value, dtype=np.float32)
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    if np.any(norms <= 1.0e-12) or not np.isfinite(norms).all():
        raise D38StrongB3QuantizedError("zero or non-finite row norm")
    return np.asarray(rows / norms, dtype=np.float32)


def _support(
    features: np.ndarray,
    labels: Sequence[str],
    classes: Sequence[str],
    name: str,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...], int]:
    rows = _rows(features, f"{name} features")
    y = np.asarray(tuple(str(value) for value in labels))
    registry = tuple(str(value) for value in classes)
    if (
        len(y) != len(rows)
        or not registry
        or len(set(registry)) != len(registry)
        or any(not value for value in registry)
        or set(y.tolist()) != set(registry)
    ):
        raise D38StrongB3QuantizedError(f"{name} registry drift")
    counts = [int(np.sum(y == label)) for label in registry]
    if min(counts) < 1 or len(set(counts)) != 1:
        raise D38StrongB3QuantizedError(f"{name} must be symmetric K-shot")
    targets = np.asarray([registry.index(value) for value in y], dtype=np.int64)
    return rows, targets, registry, counts[0]


def _bounds() -> tuple[np.ndarray, np.ndarray]:
    lower = np.full(FEATURE_DIM, -LOG_SCALE_LIMIT, dtype=np.float32)
    upper = np.full(FEATURE_DIM, LOG_SCALE_LIMIT, dtype=np.float32)
    lower[160:256] = -FFT_LOG_SCALE_LIMIT
    upper[160:256] = FFT_LOG_SCALE_LIMIT
    return lower, upper


def _transform(rows: np.ndarray, log_diag: np.ndarray) -> np.ndarray:
    scale = np.exp(np.asarray(log_diag, dtype=np.float32)).astype(np.float32)
    return _normalize_numpy(np.asarray(rows, dtype=np.float32) * scale[None, :])


def _tensor(value: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(value)).clone().to(device)


def _positive_fp16(value: float) -> np.float16:
    smallest = np.nextafter(np.float16(0), np.float16(1))
    return np.float16(max(float(value), float(smallest)))


def _decode(
    code1: np.ndarray,
    code2: np.ndarray,
    scale1: np.ndarray,
    scale2: np.ndarray,
) -> np.ndarray:
    decoded = np.empty((len(code1), FEATURE_DIM), dtype=np.float32)
    for block_index, block in enumerate(BLOCK_SLICES):
        decoded[:, block] = (
            code1[:, block].astype(np.float32)
            * scale1[:, block_index].astype(np.float32)[:, None]
            + code2[:, block].astype(np.float32)
            * scale2[:, block_index].astype(np.float32)[:, None]
        )
    return decoded


def _residual_quantize(
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows = _normalize_numpy(weights)
    code1 = np.zeros(rows.shape, dtype=np.int8)
    code2 = np.zeros(rows.shape, dtype=np.int8)
    scale1 = np.empty((len(rows), len(BLOCK_SLICES)), dtype=np.float16)
    scale2 = np.empty((len(rows), len(BLOCK_SLICES)), dtype=np.float16)
    for row_index, row in enumerate(rows):
        for block_index, block in enumerate(BLOCK_SLICES):
            values = row[block]
            s1 = _positive_fp16(float(np.max(np.abs(values))) / 127.0)
            q1 = np.clip(np.rint(values / np.float32(s1)), -127, 127).astype(
                np.int8
            )
            residual = values - np.float32(s1) * q1.astype(np.float32)
            s2 = _positive_fp16(float(np.max(np.abs(residual))) / 127.0)
            q2 = np.clip(np.rint(residual / np.float32(s2)), -127, 127).astype(
                np.int8
            )
            code1[row_index, block] = q1
            code2[row_index, block] = q2
            scale1[row_index, block_index] = s1
            scale2[row_index, block_index] = s2
    decoded = _decode(code1, code2, scale1, scale2)
    norms = np.linalg.norm(decoded, axis=1)
    if np.any(norms <= 1.0e-12) or not np.isfinite(norms).all():
        raise D38StrongB3QuantizedError("residual-int8 decode norm drift")
    inverse_norm = np.asarray(1.0 / norms, dtype=np.float16)
    deployed = decoded * inverse_norm.astype(np.float32)[:, None]
    return code1, code2, scale1, scale2, inverse_norm, deployed


@dataclass(frozen=True)
class D38StrongB3Config:
    arm: str = "A"

    def __post_init__(self) -> None:
        arm = str(self.arm).upper()
        if arm.startswith("D38-"):
            arm = arm[4:]
        if arm not in {"A", "B"}:
            raise D38StrongB3QuantizedError("D38 arm lock drift")
        object.__setattr__(self, "arm", arm)

    @property
    def stage2c_steps(self) -> int:
        return 0 if self.arm == "A" else STAGE2C_STEPS


@dataclass(frozen=True)
class D38StrongB3State:
    schema: str
    classes: tuple[str, ...]
    old_class_count: int
    log_diag_fp32: np.ndarray
    code1_qint8: np.ndarray
    code2_qint8: np.ndarray
    scale1_fp16: np.ndarray
    scale2_fp16: np.ndarray
    inverse_norm_fp16: np.ndarray
    fp32_weights: np.ndarray
    arm: str

    def __post_init__(self) -> None:
        count = len(self.classes)
        new_count = count - int(self.old_class_count)
        is_int8 = self.schema == SCHEMA_INT8
        is_fp32 = self.schema == SCHEMA_FP32
        if (
            not (is_int8 or is_fp32)
            or not 2 <= int(self.old_class_count) <= 20
            or new_count not in (0,) + ALLOWED_NEW_CLASS_COUNTS
            or len(set(self.classes)) != count
            or self.arm not in {"A", "B"}
            or self.log_diag_fp32.dtype != np.float32
            or self.log_diag_fp32.shape != (FEATURE_DIM,)
            or not np.isfinite(self.log_diag_fp32).all()
        ):
            raise D38StrongB3QuantizedError("D38 state drift")
        if is_int8:
            valid = (
                self.code1_qint8.dtype == np.int8
                and self.code1_qint8.shape == (count, FEATURE_DIM)
                and self.code2_qint8.dtype == np.int8
                and self.code2_qint8.shape == (count, FEATURE_DIM)
                and self.scale1_fp16.dtype == np.float16
                and self.scale1_fp16.shape == (count, len(BLOCK_SLICES))
                and self.scale2_fp16.dtype == np.float16
                and self.scale2_fp16.shape == (count, len(BLOCK_SLICES))
                and self.inverse_norm_fp16.dtype == np.float16
                and self.inverse_norm_fp16.shape == (count,)
                and self.fp32_weights.shape == (0, FEATURE_DIM)
                and np.isfinite(self.scale1_fp16).all()
                and np.isfinite(self.scale2_fp16).all()
                and np.isfinite(self.inverse_norm_fp16).all()
                and bool(np.all(self.scale1_fp16 > 0))
                and bool(np.all(self.scale2_fp16 > 0))
                and bool(np.all(self.inverse_norm_fp16 > 0))
            )
        else:
            valid = (
                self.code1_qint8.shape == (0, FEATURE_DIM)
                and self.code2_qint8.shape == (0, FEATURE_DIM)
                and self.scale1_fp16.shape == (0, len(BLOCK_SLICES))
                and self.scale2_fp16.shape == (0, len(BLOCK_SLICES))
                and self.inverse_norm_fp16.shape == (0,)
                and self.fp32_weights.dtype == np.float32
                and self.fp32_weights.shape == (count, FEATURE_DIM)
                and np.isfinite(self.fp32_weights).all()
            )
        if not valid:
            raise D38StrongB3QuantizedError("D38 state storage drift")
        for name, dtype in (
            ("log_diag_fp32", np.float32),
            ("code1_qint8", np.int8),
            ("code2_qint8", np.int8),
            ("scale1_fp16", np.float16),
            ("scale2_fp16", np.float16),
            ("inverse_norm_fp16", np.float16),
            ("fp32_weights", np.float32),
        ):
            object.__setattr__(self, name, _readonly(getattr(self, name), dtype))

    @property
    def is_int8(self) -> bool:
        return self.schema == SCHEMA_INT8

    @property
    def registry_state_bytes(self) -> int:
        metadata = {
            "arm": self.arm,
            "classes": list(self.classes),
            "old_class_count": int(self.old_class_count),
            "schema": self.schema,
        }
        return len(
            json.dumps(
                metadata,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        )

    @property
    def persistent_state_bytes(self) -> int:
        arrays = (
            self.log_diag_fp32,
            self.code1_qint8,
            self.code2_qint8,
            self.scale1_fp16,
            self.scale2_fp16,
            self.inverse_norm_fp16,
            self.fp32_weights,
        )
        return int(sum(value.nbytes for value in arrays) + self.registry_state_bytes)


@dataclass(frozen=True)
class D38StrongB3Result:
    before_state: D38StrongB3State
    state: D38StrongB3State
    matched_fp32_state: D38StrongB3State
    training_trace: tuple[dict[str, Any], ...]
    geometry_audit: dict[str, Any]
    resource_audit: dict[str, Any]


def _int8_state(
    classes: tuple[str, ...],
    old_count: int,
    log_diag: np.ndarray,
    weights: np.ndarray,
    arm: str,
) -> tuple[D38StrongB3State, np.ndarray, dict[str, float]]:
    reference = _normalize_numpy(weights)
    code1, code2, scale1, scale2, inv_norm, deployed = _residual_quantize(
        reference
    )
    empty = np.zeros((0, FEATURE_DIM), dtype=np.float32)
    state = D38StrongB3State(
        schema=SCHEMA_INT8,
        classes=classes,
        old_class_count=old_count,
        log_diag_fp32=np.asarray(log_diag, dtype=np.float32),
        code1_qint8=code1,
        code2_qint8=code2,
        scale1_fp16=scale1,
        scale2_fp16=scale2,
        inverse_norm_fp16=inv_norm,
        fp32_weights=empty,
        arm=arm,
    )
    error = np.abs(deployed - reference)
    return state, deployed, {
        "quantization_error_mean": float(np.mean(error)),
        "quantization_error_max": float(np.max(error)),
    }


def _fp32_state(
    classes: tuple[str, ...],
    old_count: int,
    log_diag: np.ndarray,
    weights: np.ndarray,
    arm: str,
) -> D38StrongB3State:
    empty_code = np.zeros((0, FEATURE_DIM), dtype=np.int8)
    empty_scale = np.zeros((0, len(BLOCK_SLICES)), dtype=np.float16)
    return D38StrongB3State(
        schema=SCHEMA_FP32,
        classes=classes,
        old_class_count=old_count,
        log_diag_fp32=np.asarray(log_diag, dtype=np.float32),
        code1_qint8=empty_code,
        code2_qint8=empty_code,
        scale1_fp16=empty_scale,
        scale2_fp16=empty_scale,
        inverse_norm_fp16=np.zeros(0, dtype=np.float16),
        fp32_weights=_normalize_numpy(weights),
        arm=arm,
    )


def _state_weights(state: D38StrongB3State) -> np.ndarray:
    if not state.is_int8:
        return _normalize_numpy(state.fp32_weights)
    decoded = _decode(
        state.code1_qint8,
        state.code2_qint8,
        state.scale1_fp16,
        state.scale2_fp16,
    )
    return np.asarray(
        decoded * state.inverse_norm_fp16.astype(np.float32)[:, None],
        dtype=np.float32,
    )


def fit_d38_strong_b3_quantized(
    old_support_features: np.ndarray,
    old_support_labels: Sequence[str],
    old_classes: Sequence[str],
    new_support_features: np.ndarray,
    new_support_labels: Sequence[str],
    new_classes: Sequence[str],
    *,
    seed: int,
    device: torch.device | str = "cpu",
    config: D38StrongB3Config | None = None,
    before_stage2c_hook: (
        Callable[[D38StrongB3State, tuple[dict[str, Any], ...]], None] | None
    ) = None,
) -> D38StrongB3Result:
    locked = config or D38StrongB3Config()
    old_rows, old_targets, old_registry, old_k = _support(
        old_support_features, old_support_labels, old_classes, "old support"
    )
    new_rows, new_targets, new_registry, new_k = _support(
        new_support_features, new_support_labels, new_classes, "new support"
    )
    if (
        len(new_registry) not in ALLOWED_NEW_CLASS_COUNTS
        or set(old_registry) & set(new_registry)
        or old_k != new_k
    ):
        raise D38StrongB3QuantizedError("D38 class/K closure drift")
    runtime_device = torch.device(device)
    torch.manual_seed(int(seed))
    if runtime_device.type == "cuda":
        torch.cuda.set_device(runtime_device)
        torch.cuda.manual_seed_all(int(seed))
        torch.cuda.reset_peak_memory_stats(runtime_device)
    x_old = _tensor(old_rows, runtime_device)
    y_old = torch.as_tensor(old_targets, dtype=torch.long, device=runtime_device)
    prototypes = torch.stack(
        [F.normalize(x_old[y_old == index].mean(dim=0), dim=0) for index in range(len(old_registry))]
    )
    log_diag = torch.nn.Parameter(torch.zeros(FEATURE_DIM, device=runtime_device))
    old_weights = torch.nn.Parameter(prototypes.detach().clone())
    lower_np, upper_np = _bounds()
    lower = _tensor(lower_np, runtime_device)
    upper = _tensor(upper_np, runtime_device)
    optimizer_b = torch.optim.AdamW(
        [log_diag, old_weights], lr=STAGE2B_LR, weight_decay=WEIGHT_DECAY
    )
    generator = torch.Generator(device=runtime_device).manual_seed(int(seed))
    trace: list[dict[str, Any]] = []
    for step in range(1, STAGE2B_STEPS + 1):
        optimizer_b.zero_grad(set_to_none=True)
        noisy = x_old + FEATURE_NOISE_STD * torch.randn(
            x_old.shape,
            generator=generator,
            device=runtime_device,
            dtype=x_old.dtype,
        )
        effective = torch.minimum(torch.maximum(log_diag, lower), upper)
        transformed = F.normalize(noisy * torch.exp(effective), dim=1)
        logits = TEMPERATURE * (transformed @ F.normalize(old_weights, dim=1).T)
        ce_loss = F.cross_entropy(logits, y_old)
        anchor_loss = torch.mean(
            (F.normalize(old_weights, dim=1) - prototypes) ** 2
        )
        loss = ce_loss + PROTOTYPE_ANCHOR_WEIGHT * anchor_loss
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [log_diag, old_weights], max_norm=GRAD_CLIP
        )
        optimizer_b.step()
        with torch.no_grad():
            log_diag.copy_(torch.minimum(torch.maximum(log_diag, lower), upper))
        row = {
            "phase": "stage2b_fullbatch_old_adaptation",
            "epoch": step,
            "optimizer_step": step,
            "loss": float(loss.detach().cpu()),
            "ce_loss": float(ce_loss.detach().cpu()),
            "prototype_anchor_loss": float(anchor_loss.detach().cpu()),
            "gradient_norm": float(grad_norm.detach().cpu()),
            "support_accuracy": float(
                (logits.argmax(dim=1) == y_old).float().mean().detach().cpu()
            ),
            "query_rows_used": 0,
        }
        if not all(math.isfinite(float(value)) for key, value in row.items() if key not in {"phase"}):
            raise D38StrongB3QuantizedError("non-finite Stage2-B trace")
        trace.append(row)
    log_diag_np = np.asarray(log_diag.detach().cpu().tolist(), dtype=np.float32)
    old_fp32 = np.asarray(old_weights.detach().cpu().tolist(), dtype=np.float32)
    before_state, old_deployed, old_quant = _int8_state(
        old_registry, len(old_registry), log_diag_np, old_fp32, locked.arm
    )
    if before_stage2c_hook is not None:
        before_stage2c_hook(
            before_state,
            tuple(dict(row) for row in trace),
        )

    transformed_new = _transform(new_rows, log_diag_np)
    new_init = np.stack(
        [
            _normalize_numpy(
                np.mean(transformed_new[new_targets == index], axis=0, keepdims=True)
            )[0]
            for index in range(len(new_registry))
        ]
    ).astype(np.float32)
    new_final = np.array(new_init, copy=True)
    if locked.stage2c_steps:
        all_rows = np.concatenate([old_rows, new_rows], axis=0)
        all_targets_np = np.concatenate(
            [old_targets, new_targets + len(old_registry)], axis=0
        )
        transformed_all = _tensor(_transform(all_rows, log_diag_np), runtime_device)
        all_targets = torch.as_tensor(
            all_targets_np, dtype=torch.long, device=runtime_device
        )
        frozen_old = _tensor(old_deployed, runtime_device).detach()
        new_initial_tensor = _tensor(new_init, runtime_device).detach()
        new_parameter = torch.nn.Parameter(new_initial_tensor.clone())
        optimizer_c = torch.optim.SGD([new_parameter], lr=STAGE2C_LR, momentum=0.0)
        for step in range(1, STAGE2C_STEPS + 1):
            optimizer_c.zero_grad(set_to_none=True)
            weights = torch.cat(
                [frozen_old, F.normalize(new_parameter, dim=1)], dim=0
            )
            logits = TEMPERATURE * (transformed_all @ weights.T)
            sample_loss = F.cross_entropy(logits, all_targets, reduction="none")
            class_loss = torch.stack(
                [sample_loss[all_targets == index].mean() for index in range(len(weights))]
            )
            worst = WORST_CLASS_TAU * (
                torch.logsumexp(class_loss / WORST_CLASS_TAU, dim=0)
                - math.log(len(weights))
            )
            anchor = torch.mean(
                (F.normalize(new_parameter, dim=1) - new_initial_tensor) ** 2
            )
            loss = class_loss.mean() + WORST_CLASS_WEIGHT * worst + NEW_ANCHOR_WEIGHT * anchor
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_([new_parameter], GRAD_CLIP)
            optimizer_c.step()
            row = {
                "phase": "stage2c_all_support_new_weight_only",
                "epoch": STAGE2B_STEPS + step,
                "optimizer_step": STAGE2B_STEPS + step,
                "loss": float(loss.detach().cpu()),
                "mean_class_loss": float(class_loss.mean().detach().cpu()),
                "worst_class_loss": float(class_loss.max().detach().cpu()),
                "worst_class_surrogate": float(worst.detach().cpu()),
                "new_centroid_anchor_loss": float(anchor.detach().cpu()),
                "gradient_norm": float(grad_norm.detach().cpu()),
                "support_accuracy": float(
                    (logits.argmax(dim=1) == all_targets).float().mean().detach().cpu()
                ),
                "old_support_rows_used": int(len(old_rows)),
                "new_support_rows_used": int(len(new_rows)),
                "query_rows_used": 0,
            }
            if not all(math.isfinite(float(value)) for key, value in row.items() if key != "phase"):
                raise D38StrongB3QuantizedError("non-finite Stage2-C trace")
            trace.append(row)
        new_final = np.asarray(
            F.normalize(new_parameter.detach(), dim=1).cpu().tolist(),
            dtype=np.float32,
        )

    all_classes = old_registry + new_registry
    new_reference = _normalize_numpy(new_final)
    (
        new_code1,
        new_code2,
        new_scale1,
        new_scale2,
        new_inverse_norm,
        new_deployed,
    ) = _residual_quantize(new_reference)
    int8_state = D38StrongB3State(
        schema=SCHEMA_INT8,
        classes=all_classes,
        old_class_count=len(old_registry),
        log_diag_fp32=log_diag_np,
        code1_qint8=np.concatenate(
            [before_state.code1_qint8, new_code1], axis=0
        ),
        code2_qint8=np.concatenate(
            [before_state.code2_qint8, new_code2], axis=0
        ),
        scale1_fp16=np.concatenate(
            [before_state.scale1_fp16, new_scale1], axis=0
        ),
        scale2_fp16=np.concatenate(
            [before_state.scale2_fp16, new_scale2], axis=0
        ),
        inverse_norm_fp16=np.concatenate(
            [before_state.inverse_norm_fp16, new_inverse_norm], axis=0
        ),
        fp32_weights=np.zeros((0, FEATURE_DIM), dtype=np.float32),
        arm=locked.arm,
    )
    deployed_all = np.concatenate([old_deployed, new_deployed], axis=0)
    quant_reference = np.concatenate(
        [_normalize_numpy(old_fp32), new_reference], axis=0
    )
    all_quant_error = np.abs(deployed_all - quant_reference)
    all_quant = {
        "quantization_error_mean": float(np.mean(all_quant_error)),
        "quantization_error_max": float(np.max(all_quant_error)),
    }
    matched_fp32 = _fp32_state(
        all_classes,
        len(old_registry),
        log_diag_np,
        np.concatenate([old_fp32, new_final], axis=0),
        locked.arm,
    )
    if not old_prefix_bitwise_unchanged_d38(before_state, int8_state):
        raise D38StrongB3QuantizedError("D38 old int8 prefix changed during append")
    int8_vs_fp32_support_changes = int(
        np.sum(
            np.argmax(
                TEMPERATURE * (_transform(np.concatenate([old_rows, new_rows]), log_diag_np) @ deployed_all.T),
                axis=1,
            )
            != np.argmax(
                TEMPERATURE * (_transform(np.concatenate([old_rows, new_rows]), log_diag_np) @ quant_reference.T),
                axis=1,
            )
        )
    )
    geometry = {
        "schema": "cvs.phase2.d38.geometry.v1",
        "feature_geometry": "normalized_z160_plus_4x_joint_normalized_fft96_rf32",
        "stage2b_solver": "fullbatch_adamw20_no_bias",
        "stage2c_solver": (
            "centroid_only" if locked.arm == "A" else "all_support_new_weight_only_sgd10"
        ),
        "exact_legacy_strong_b3_claimed": False,
        "old_compiled_before_stage2c": True,
        "stage2c_uses_decoded_int8_old_head": True,
        "old_prefix_bitwise_unchanged": True,
        "target_old_int8_used_for_prediction": True,
        "target_new_int8_used_for_prediction": True,
        "fp32_target_prototype_stored_in_formal_state": False,
        "fixed_quantization_blocks": [160, 96, 32],
        "decode_weight_renormalization": True,
        "class_id_specific_branch": False,
        "label_permutation_equivariant": True,
        "old_quantization_error_mean": old_quant["quantization_error_mean"],
        "old_quantization_error_max": old_quant["quantization_error_max"],
        **all_quant,
        "int8_vs_fp32_support_argmax_change_count": int8_vs_fp32_support_changes,
    }
    peak_parameters = max(
        FEATURE_DIM + len(old_registry) * FEATURE_DIM,
        len(new_registry) * FEATURE_DIM if locked.stage2c_steps else 0,
    )
    total_steps = STAGE2B_STEPS + locked.stage2c_steps
    class_count = len(all_classes)
    row_count = len(old_rows) + len(new_rows)
    resource = {
        "schema": "cvs.phase2.d38.resource.v1",
        "support_only": True,
        "trainable_parameters": int(peak_parameters),
        "trainable_parameter_cap": 80_000,
        "trainable_parameter_cap_pass": peak_parameters <= 80_000,
        "adaptation_epochs": total_steps,
        "optimizer_steps": total_steps,
        "adaptation_epoch_cap": 30,
        "optimizer_step_cap": 50,
        "adaptation_epoch_cap_pass": total_steps <= 30,
        "optimizer_step_cap_pass": total_steps <= 50,
        "persistent_state_bytes": int8_state.persistent_state_bytes,
        "registry_state_bytes": int8_state.registry_state_bytes,
        "persistent_state_cap_bytes": 256 * 1024,
        "persistent_state_cap_pass": int8_state.persistent_state_bytes <= 256 * 1024,
        "estimated_adaptation_macs": int(
            3
            * FEATURE_DIM
            * (
                STAGE2B_STEPS * len(old_rows) * len(old_registry)
                + locked.stage2c_steps * row_count * class_count
            )
        ),
        "estimated_macs_per_query": int(FEATURE_DIM + 2 * FEATURE_DIM * class_count),
        "dense_query_graph_bytes": 0,
        "query_dependent_batch_optimization": False,
        "query_rows_used_for_fit": 0,
        "query_labels_used_for_fit": False,
        "query_role_oracle_access": False,
        "query_true_batch_class_count_access": False,
        "query_class_quota_access": False,
        "query_batch_global_assignment": False,
        "clean_sample_access": False,
        "source_sample_access": False,
        "resident_fp32_target_prototype_count": 0,
        "old_k_shot": old_k,
        "new_k_shot": new_k,
        "registered_class_count": class_count,
        "runtime_device": str(runtime_device),
        "peak_cuda_memory_bytes": (
            int(torch.cuda.max_memory_allocated(runtime_device))
            if runtime_device.type == "cuda"
            else 0
        ),
    }
    return D38StrongB3Result(
        before_state=before_state,
        state=int8_state,
        matched_fp32_state=matched_fp32,
        training_trace=tuple(trace),
        geometry_audit=geometry,
        resource_audit=resource,
    )


def score_d38_strong_b3(
    state: D38StrongB3State, features: np.ndarray
) -> np.ndarray:
    rows = _rows(features, "D38 scoring features")
    transformed = _transform(rows, state.log_diag_fp32)
    weights = _state_weights(state)
    scores = np.stack(
        [
            np.asarray(
                np.float32(TEMPERATURE) * (row @ weights.T), dtype=np.float32
            )
            for row in transformed
        ]
    )
    if not np.isfinite(scores).all():
        raise D38StrongB3QuantizedError("non-finite D38 score")
    return _readonly(scores, np.float32)


def predict_d38_strong_b3(
    state: D38StrongB3State, features: np.ndarray
) -> np.ndarray:
    return np.asarray(state.classes)[np.argmax(score_d38_strong_b3(state, features), axis=1)]


def old_prefix_bitwise_unchanged_d38(
    before: D38StrongB3State, after: D38StrongB3State
) -> bool:
    count = before.old_class_count
    return bool(
        before.is_int8
        and after.is_int8
        and before.classes == after.classes[:count]
        and np.array_equal(before.log_diag_fp32, after.log_diag_fp32)
        and np.array_equal(before.code1_qint8, after.code1_qint8[:count])
        and np.array_equal(before.code2_qint8, after.code2_qint8[:count])
        and np.array_equal(before.scale1_fp16, after.scale1_fp16[:count])
        and np.array_equal(before.scale2_fp16, after.scale2_fp16[:count])
        and np.array_equal(before.inverse_norm_fp16, after.inverse_norm_fp16[:count])
    )


def pairwise_support_diagnostics_d38(
    state: D38StrongB3State,
    held_new_features: np.ndarray,
    held_new_labels: Sequence[str],
    physical_ids: Sequence[str],
    *,
    scenario: str,
    outer_fold: int,
    physical_ranks: Sequence[int],
) -> list[dict[str, Any]]:
    labels = tuple(str(value) for value in held_new_labels)
    ids = tuple(str(value) for value in physical_ids)
    ranks = tuple(int(value) for value in physical_ranks)
    scores = score_d38_strong_b3(state, held_new_features)
    old_count = state.old_class_count
    if (
        len(state.classes) == old_count
        or len(labels) != len(scores)
        or len(ids) != len(scores)
        or len(ranks) != len(scores)
        or len(set(ids)) != len(ids)
        or any(label not in state.classes[old_count:] for label in labels)
        or not str(scenario)
    ):
        raise D38StrongB3QuantizedError("D38 pairwise diagnostic closure drift")
    output: list[dict[str, Any]] = []
    for row, truth, physical_id, rank in zip(scores, labels, ids, ranks, strict=True):
        truth_index = state.classes.index(truth)
        competing_new = np.array(row[old_count:], copy=True)
        competing_new[truth_index - old_count] = -np.inf
        competitor_local = int(np.argmax(competing_new))
        competitor_index = old_count + competitor_local
        top_old_index = int(np.argmax(row[:old_count]))
        output.append(
            {
                "scenario": str(scenario),
                "outer_fold": int(outer_fold),
                "physical_rank": rank,
                "physical_sample_id": physical_id,
                "true_new_handle": truth,
                "top_competing_new_handle": state.classes[competitor_index],
                "true_new_score": float(row[truth_index]),
                "top_competing_new_score": float(row[competitor_index]),
                "new_new_margin": float(row[truth_index] - row[competitor_index]),
                "top_old_handle": state.classes[top_old_index],
                "top_old_score": float(row[top_old_index]),
                "new_old_margin": float(row[truth_index] - row[top_old_index]),
                "query_rows_used": 0,
            }
        )
    return output


__all__ = [
    "D38_SCORE_TEMPERATURE",
    "D38StrongB3Config",
    "D38StrongB3QuantizedError",
    "D38StrongB3Result",
    "D38StrongB3State",
    "fit_d38_strong_b3_quantized",
    "old_prefix_bitwise_unchanged_d38",
    "pairwise_support_diagnostics_d38",
    "predict_d38_strong_b3",
    "score_d38_strong_b3",
]
