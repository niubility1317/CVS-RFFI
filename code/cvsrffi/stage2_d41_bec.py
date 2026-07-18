"""D41 deterministic block-erasure consistency adaptation and registration.

The core consumes admitted 288-D D40/D38 features only.  Four deterministic
mathematical views of each fixed received-IQ feature are used during support
adaptation; query scoring always uses the unchanged full feature row.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from cvsrffi.stage2_d38_strong_b3_quantized import (
    D38_SCORE_TEMPERATURE,
    D38StrongB3State,
    compile_d38_state,
    decode_d38_state_weights,
    score_d38_strong_b3,
    transform_d38_features,
)


FEATURE_DIM = 288
TEMPERATURE = D38_SCORE_TEMPERATURE
NORM_EPSILON = 1.0e-12
VIEW_NAMES = ("full", "minus_z", "minus_fft", "minus_rf")
BLOCK_SLICES = {
    "minus_z": slice(0, 160),
    "minus_fft": slice(160, 256),
    "minus_rf": slice(256, 288),
}
ALLOWED_NEW_CLASS_COUNTS = (2, 5, 10, 20)
STAGE2B_STEPS = 20
STAGE2C_STEPS = 10
STAGE2B_LR = 0.01
STAGE2C_LR = 0.05
WEIGHT_DECAY = 0.002
GRAD_CLIP = 5.0
LOG_SCALE_LIMIT = 1.5
FFT_LOG_SCALE_LIMIT = math.log(1.5)
SCHEMA_INT8 = "cvs.phase2.d41.bec_residual_int8.v1"
SCHEMA_FP32 = "cvs.phase2.d41.bec_fp32_ablation.v1"


class D41BECError(ValueError):
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
        raise D41BECError(f"{name} must be finite float32 [N,{FEATURE_DIM}]")
    return np.ascontiguousarray(rows)


def _normalize_rows(value: np.ndarray, name: str) -> np.ndarray:
    rows = _rows(value, name)
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    if not np.isfinite(norms).all() or bool(np.any(norms <= NORM_EPSILON)):
        raise D41BECError(f"{name} contains a near-zero row")
    return np.asarray(rows / norms, dtype=np.float32)


def _support(
    features: np.ndarray,
    labels: Sequence[str],
    classes: Sequence[str],
    name: str,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...], int]:
    rows = _rows(features, f"{name} features")
    registry = tuple(str(value) for value in classes)
    y = np.asarray(tuple(str(value) for value in labels))
    if (
        len(y) != len(rows)
        or not registry
        or len(set(registry)) != len(registry)
        or any(not value for value in registry)
        or set(y.tolist()) != set(registry)
    ):
        raise D41BECError(f"{name} registry drift")
    counts = [int(np.sum(y == label)) for label in registry]
    if min(counts) < 1 or len(set(counts)) != 1:
        raise D41BECError(f"{name} must be symmetric K-shot")
    targets = np.asarray([registry.index(value) for value in y], dtype=np.int64)
    return rows, targets, registry, counts[0]


def make_d41_views(features: np.ndarray) -> dict[str, np.ndarray]:
    """Create the exact full view and three deterministic block erasures."""

    rows = _rows(features, "D41 full view")
    norms = np.linalg.norm(rows, axis=1)
    if (
        not np.isfinite(norms).all()
        or bool(np.any(norms <= NORM_EPSILON))
        or not np.allclose(norms, 1.0, rtol=0.0, atol=5.0e-4)
    ):
        raise D41BECError("D41 full view must preserve unit D40 geometry")
    output = {"full": _readonly(rows, np.float32)}
    for name, block in BLOCK_SLICES.items():
        masked = np.array(rows, copy=True)
        masked[:, block] = 0.0
        output[name] = _readonly(
            _normalize_rows(masked, f"D41 {name} view"), np.float32
        )
    return output


def _bounds(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    lower = np.full(FEATURE_DIM, -LOG_SCALE_LIMIT, dtype=np.float32)
    upper = np.full(FEATURE_DIM, LOG_SCALE_LIMIT, dtype=np.float32)
    lower[160:256] = -FFT_LOG_SCALE_LIMIT
    upper[160:256] = FFT_LOG_SCALE_LIMIT
    return _tensor(lower, device), _tensor(upper, device)


def _tensor(value: np.ndarray, device: torch.device) -> torch.Tensor:
    writable = np.array(value, dtype=np.float32, copy=True, order="C")
    return torch.from_numpy(writable).to(device)


def _view_tensors(
    views: Mapping[str, np.ndarray], device: torch.device
) -> tuple[torch.Tensor, ...]:
    if tuple(views) != VIEW_NAMES:
        raise D41BECError("D41 view order drift")
    return tuple(_tensor(views[name], device) for name in VIEW_NAMES)


def _macro_ce(
    logits: torch.Tensor, targets: torch.Tensor, class_count: int
) -> torch.Tensor:
    sample_loss = F.cross_entropy(logits, targets, reduction="none")
    class_losses: list[torch.Tensor] = []
    for class_index in range(class_count):
        selected = sample_loss[targets == class_index]
        if selected.numel() < 1:
            raise D41BECError("D41 macro CE class support missing")
        class_losses.append(selected.mean())
    return torch.stack(class_losses).mean()


def d41_bec_loss(
    logits_by_view: Sequence[torch.Tensor],
    targets: torch.Tensor,
    class_count: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return total BEC, four-view macro CE mean, and three-view JS mean."""

    logits = tuple(logits_by_view)
    if (
        len(logits) != len(VIEW_NAMES)
        or targets.ndim != 1
        or int(class_count) < 2
        or any(
            value.ndim != 2
            or value.shape != logits[0].shape
            or value.shape[0] != targets.shape[0]
            or value.shape[1] != int(class_count)
            or not bool(torch.isfinite(value).all().item())
            for value in logits
        )
    ):
        raise D41BECError("D41 BEC loss contract drift")
    macro_ce = torch.stack(
        [_macro_ce(value, targets, int(class_count)) for value in logits]
    ).mean()
    full_log_prob = F.log_softmax(logits[0], dim=1)
    js_terms: list[torch.Tensor] = []
    for masked_logits in logits[1:]:
        masked_log_prob = F.log_softmax(masked_logits, dim=1)
        log_mixture = (
            torch.logaddexp(full_log_prob, masked_log_prob) - math.log(2.0)
        )
        full_kl = torch.sum(
            torch.exp(full_log_prob) * (full_log_prob - log_mixture), dim=1
        )
        masked_kl = torch.sum(
            torch.exp(masked_log_prob) * (masked_log_prob - log_mixture), dim=1
        )
        js_terms.append((0.5 * (full_kl + masked_kl)).mean())
    mean_js = torch.stack(js_terms).mean()
    total = macro_ce + mean_js
    if not bool(torch.isfinite(total).item()):
        raise D41BECError("D41 BEC loss became non-finite")
    return total, macro_ce, mean_js


def _logits_for_views(
    view_tensors: Sequence[torch.Tensor],
    log_diag: torch.Tensor,
    weights: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    effective = torch.minimum(torch.maximum(log_diag, lower), upper)
    scale = torch.exp(effective)
    normalized_weights = F.normalize(weights, dim=1)
    return tuple(
        TEMPERATURE
        * (F.normalize(view * scale, dim=1) @ normalized_weights.T)
        for view in view_tensors
    )


def _state_array_snapshot(state: D38StrongB3State) -> tuple[bytes, ...]:
    return tuple(
        np.ascontiguousarray(value).tobytes()
        for value in (
            state.log_diag_fp32,
            state.code1_qint8,
            state.code2_qint8,
            state.scale1_fp16,
            state.scale2_fp16,
            state.inverse_norm_fp16,
            state.fp32_weights,
        )
    )


@dataclass(frozen=True)
class D41BECConfig:
    stage2b_steps: int = STAGE2B_STEPS
    stage2c_steps: int = STAGE2C_STEPS
    stage2b_lr: float = STAGE2B_LR
    stage2c_lr: float = STAGE2C_LR
    weight_decay: float = WEIGHT_DECAY
    gradient_clip: float = GRAD_CLIP

    def __post_init__(self) -> None:
        if (
            int(self.stage2b_steps) != STAGE2B_STEPS
            or int(self.stage2c_steps) != STAGE2C_STEPS
            or float(self.stage2b_lr) != STAGE2B_LR
            or float(self.stage2c_lr) != STAGE2C_LR
            or float(self.weight_decay) != WEIGHT_DECAY
            or float(self.gradient_clip) != GRAD_CLIP
        ):
            raise D41BECError("D41 BEC mechanism lock drift")


@dataclass(frozen=True)
class D41BECState:
    schema: str
    base_state: D38StrongB3State

    def __post_init__(self) -> None:
        is_int8 = self.schema == SCHEMA_INT8
        is_fp32 = self.schema == SCHEMA_FP32
        if (
            not (is_int8 or is_fp32)
            or is_int8 != self.base_state.is_int8
            or self.base_state.arm != "A"
        ):
            raise D41BECError("D41 state drift")
        if is_int8 and self.base_state.fp32_weights.shape != (0, FEATURE_DIM):
            raise D41BECError("formal D41 state contains FP32 target direction")

    @property
    def is_int8(self) -> bool:
        return self.schema == SCHEMA_INT8

    @property
    def classes(self) -> tuple[str, ...]:
        return self.base_state.classes

    @property
    def old_class_count(self) -> int:
        return int(self.base_state.old_class_count)

    @property
    def wrapper_metadata_bytes(self) -> int:
        metadata = {
            "schema": self.schema,
            "support_views": list(VIEW_NAMES),
            "temperature": TEMPERATURE,
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
        return int(
            self.base_state.persistent_state_bytes + self.wrapper_metadata_bytes
        )


@dataclass(frozen=True)
class D41BECResult:
    before_state: D41BECState
    state: D41BECState
    matched_fp32_before_state: D41BECState
    matched_fp32_state: D41BECState
    training_trace: tuple[dict[str, Any], ...]
    geometry_audit: dict[str, Any]
    resource_audit: dict[str, Any]


def fit_d41_bec(
    old_support_features: np.ndarray,
    old_support_labels: Sequence[str],
    old_classes: Sequence[str],
    new_support_features: np.ndarray,
    new_support_labels: Sequence[str],
    new_classes: Sequence[str],
    *,
    seed: int,
    device: torch.device | str = "cpu",
    config: D41BECConfig | None = None,
) -> D41BECResult:
    locked = config or D41BECConfig()
    old_rows, old_targets_np, old_registry, old_k = _support(
        old_support_features, old_support_labels, old_classes, "old"
    )
    new_rows, new_targets_np, new_registry, new_k = _support(
        new_support_features, new_support_labels, new_classes, "new"
    )
    if (
        not 2 <= len(old_registry) <= 20
        or len(new_registry) not in ALLOWED_NEW_CLASS_COUNTS
        or set(old_registry) & set(new_registry)
        or old_k != new_k
    ):
        raise D41BECError("D41 class/K closure drift")
    old_views_np = make_d41_views(old_rows)
    new_views_np = make_d41_views(new_rows)
    runtime_device = torch.device(device)
    torch.manual_seed(int(seed))
    if runtime_device.type == "cuda":
        torch.cuda.set_device(runtime_device)
        torch.cuda.manual_seed_all(int(seed))
        torch.cuda.reset_peak_memory_stats(runtime_device)
    old_views = _view_tensors(old_views_np, runtime_device)
    old_targets = torch.as_tensor(
        old_targets_np, dtype=torch.long, device=runtime_device
    )
    old_count = len(old_registry)
    old_initial = torch.stack(
        [
            F.normalize(old_views[0][old_targets == index].mean(dim=0), dim=0)
            for index in range(old_count)
        ]
    )
    log_diag_b = torch.nn.Parameter(
        torch.zeros(FEATURE_DIM, dtype=torch.float32, device=runtime_device)
    )
    old_weights_b = torch.nn.Parameter(old_initial.detach().clone())
    lower, upper = _bounds(runtime_device)
    optimizer_b = torch.optim.AdamW(
        [log_diag_b, old_weights_b],
        lr=locked.stage2b_lr,
        weight_decay=locked.weight_decay,
    )
    trace: list[dict[str, Any]] = []
    for step in range(1, locked.stage2b_steps + 1):
        optimizer_b.zero_grad(set_to_none=True)
        logits = _logits_for_views(
            old_views, log_diag_b, old_weights_b, lower, upper
        )
        loss, macro_ce, mean_js = d41_bec_loss(logits, old_targets, old_count)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [log_diag_b, old_weights_b], locked.gradient_clip
        )
        optimizer_b.step()
        with torch.no_grad():
            log_diag_b.copy_(torch.minimum(torch.maximum(log_diag_b, lower), upper))
        row = {
            "phase": "stage2b_four_view_bec_old",
            "epoch": step,
            "optimizer_step": step,
            "loss": float(loss.detach().cpu()),
            "macro_ce": float(macro_ce.detach().cpu()),
            "mean_js": float(mean_js.detach().cpu()),
            "gradient_norm": float(grad_norm.detach().cpu()),
            "support_accuracy_full": float(
                (logits[0].argmax(dim=1) == old_targets)
                .float()
                .mean()
                .detach()
                .cpu()
            ),
            "support_view_count": 4,
            "query_rows_used": 0,
        }
        if not all(
            math.isfinite(float(value))
            for key, value in row.items()
            if key != "phase"
        ):
            raise D41BECError("non-finite D41 Stage2-B trace")
        trace.append(row)
    log_diag_before = np.asarray(
        log_diag_b.detach().cpu().tolist(), dtype=np.float32
    )
    old_weights_before = np.asarray(
        F.normalize(old_weights_b.detach(), dim=1).cpu().tolist(),
        dtype=np.float32,
    )
    before_int8_base = compile_d38_state(
        old_registry,
        old_count,
        log_diag_before,
        old_weights_before,
        arm="A",
        precision="int8",
    )
    before_fp32_base = compile_d38_state(
        old_registry,
        old_count,
        log_diag_before,
        old_weights_before,
        arm="A",
        precision="fp32",
    )
    before_int8_snapshot = _state_array_snapshot(before_int8_base)
    before_fp32_snapshot = _state_array_snapshot(before_fp32_base)

    transformed_new_full = transform_d38_features(before_fp32_base, new_rows)
    new_initial_np = np.stack(
        [
            _normalize_rows(
                np.asarray(
                    transformed_new_full[new_targets_np == index].mean(axis=0),
                    dtype=np.float32,
                )[None, :],
                "D41 new centroid",
            )[0]
            for index in range(len(new_registry))
        ]
    ).astype(np.float32)
    all_rows = np.concatenate([old_rows, new_rows], axis=0).astype(np.float32)
    all_views = _view_tensors(make_d41_views(all_rows), runtime_device)
    all_targets_np = np.concatenate(
        [old_targets_np, new_targets_np + old_count], axis=0
    )
    all_targets = torch.as_tensor(
        all_targets_np, dtype=torch.long, device=runtime_device
    )
    log_diag_c = torch.nn.Parameter(
        _tensor(log_diag_before, runtime_device).detach().clone()
    )
    all_weights_initial = np.concatenate(
        [old_weights_before, new_initial_np], axis=0
    ).astype(np.float32)
    all_weights_c = torch.nn.Parameter(
        _tensor(all_weights_initial, runtime_device).detach().clone()
    )
    optimizer_c = torch.optim.SGD(
        [log_diag_c, all_weights_c],
        lr=locked.stage2c_lr,
        momentum=0.0,
    )
    class_count = old_count + len(new_registry)
    for step in range(1, locked.stage2c_steps + 1):
        optimizer_c.zero_grad(set_to_none=True)
        logits = _logits_for_views(
            all_views, log_diag_c, all_weights_c, lower, upper
        )
        loss, macro_ce, mean_js = d41_bec_loss(
            logits, all_targets, class_count
        )
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [log_diag_c, all_weights_c], locked.gradient_clip
        )
        optimizer_c.step()
        with torch.no_grad():
            log_diag_c.copy_(torch.minimum(torch.maximum(log_diag_c, lower), upper))
        row = {
            "phase": "stage2c_four_view_bec_all_registry",
            "epoch": STAGE2B_STEPS + step,
            "optimizer_step": STAGE2B_STEPS + step,
            "loss": float(loss.detach().cpu()),
            "macro_ce": float(macro_ce.detach().cpu()),
            "mean_js": float(mean_js.detach().cpu()),
            "gradient_norm": float(grad_norm.detach().cpu()),
            "support_accuracy_full": float(
                (logits[0].argmax(dim=1) == all_targets)
                .float()
                .mean()
                .detach()
                .cpu()
            ),
            "old_support_rows_used": int(len(old_rows)),
            "new_support_rows_used": int(len(new_rows)),
            "support_view_count": 4,
            "query_rows_used": 0,
        }
        if not all(
            math.isfinite(float(value))
            for key, value in row.items()
            if key != "phase"
        ):
            raise D41BECError("non-finite D41 Stage2-C trace")
        trace.append(row)
    log_diag_final = np.asarray(
        log_diag_c.detach().cpu().tolist(), dtype=np.float32
    )
    all_weights_final = np.asarray(
        F.normalize(all_weights_c.detach(), dim=1).cpu().tolist(),
        dtype=np.float32,
    )
    all_classes = old_registry + new_registry
    final_int8_base = compile_d38_state(
        all_classes,
        old_count,
        log_diag_final,
        all_weights_final,
        arm="A",
        precision="int8",
    )
    final_fp32_base = compile_d38_state(
        all_classes,
        old_count,
        log_diag_final,
        all_weights_final,
        arm="A",
        precision="fp32",
    )
    if (
        before_int8_snapshot != _state_array_snapshot(before_int8_base)
        or before_fp32_snapshot != _state_array_snapshot(before_fp32_base)
    ):
        raise D41BECError("D41 before state mutated during Stage2-C")

    before_state = D41BECState(SCHEMA_INT8, before_int8_base)
    state = D41BECState(SCHEMA_INT8, final_int8_base)
    matched_before = D41BECState(SCHEMA_FP32, before_fp32_base)
    matched_state = D41BECState(SCHEMA_FP32, final_fp32_base)
    before_int8_scores = score_d41_bec(before_state, old_rows)
    before_fp32_scores = score_d41_bec(matched_before, old_rows)
    final_int8_scores = score_d41_bec(state, all_rows)
    final_fp32_scores = score_d41_bec(matched_state, all_rows)
    before_argmax_changes = int(
        np.sum(
            np.argmax(before_int8_scores, axis=1)
            != np.argmax(before_fp32_scores, axis=1)
        )
    )
    final_argmax_changes = int(
        np.sum(
            np.argmax(final_int8_scores, axis=1)
            != np.argmax(final_fp32_scores, axis=1)
        )
    )
    formal_directions = decode_d38_state_weights(final_int8_base)
    reference_directions = decode_d38_state_weights(final_fp32_base)
    quant_error = np.abs(formal_directions - reference_directions)
    b_logdiag_delta = float(np.linalg.norm(log_diag_before))
    old_initial_np = np.asarray(
        old_initial.detach().cpu().tolist(), dtype=np.float32
    )
    b_old_delta = float(np.linalg.norm(old_weights_before - old_initial_np))
    c_logdiag_delta = float(np.linalg.norm(log_diag_final - log_diag_before))
    c_old_delta = float(
        np.linalg.norm(all_weights_final[:old_count] - old_weights_before)
    )
    c_new_delta = float(
        np.linalg.norm(all_weights_final[old_count:] - new_initial_np)
    )

    b_single_macs = int(
        STAGE2B_STEPS
        * len(old_rows)
        * FEATURE_DIM
        * (3 + 2 * old_count)
    )
    c_single_macs = int(
        STAGE2C_STEPS
        * len(all_rows)
        * FEATURE_DIM
        * (3 + 2 * class_count)
    )
    single_view_lower = b_single_macs + c_single_macs
    four_view_macs = 4 * single_view_lower
    js_scalar_ops = int(
        3
        * 8
        * (
            STAGE2B_STEPS * len(old_rows) * old_count
            + STAGE2C_STEPS * len(all_rows) * class_count
        )
    )
    adaptation_macs = four_view_macs + js_scalar_ops
    peak_parameters_b = FEATURE_DIM * (1 + old_count)
    peak_parameters_c = FEATURE_DIM * (1 + class_count)
    peak_parameters = max(peak_parameters_b, peak_parameters_c)

    geometry = {
        "schema": "cvs.phase2.d41.bec_geometry.v1",
        "feature_geometry": "D40_D38_full_288d_feature_unchanged",
        "support_view_names": list(VIEW_NAMES),
        "support_view_count": 4,
        "physical_support_observation_multiplicity": 1,
        "query_view": "full_only",
        "query_view_count": 1,
        "masked_view_policy": "zero_one_fixed_block_then_l2_renormalize",
        "bec_loss": "mean_four_class_macro_ce_plus_mean_three_js",
        "loss_formula": "mean_four_class_macro_ce_plus_mean_three_js",
        "macro_ce_coefficient": 1.0,
        "js_coefficient": 1.0,
        "js_implementation": "log_softmax_logaddexp_minus_log2_no_epsilon",
        "temperature": TEMPERATURE,
        "feature_noise_used": False,
        "prototype_anchor_used": False,
        "worst_class_surrogate_used": False,
        "new_anchor_used": False,
        "hnbr_used": False,
        "bias_used": False,
        "radius_used": False,
        "stage2b_solver": "adamw20_joint_logdiag_old_weights",
        "stage2c_solver": "sgd10_joint_logdiag_all_old_new_weights",
        "stage2b_trainable_state": ["log_diag", "all_old_weights"],
        "stage2c_trainable_state": [
            "log_diag",
            "all_old_weights",
            "all_new_weights",
        ],
        "new_weight_initialization": "stage2b_metric_full_view_transformed_centroid",
        "stage2b_before_artifact_immutable": True,
        "stage2c_continues_stage2b_fp32_state": True,
        "stage2c_old_weights_trainable": True,
        "stage2c_new_weights_trainable": True,
        "stage2c_log_diag_trainable": True,
        "before_state_immutable_during_stage2c": True,
        "final_registry_recompiled_atomically": True,
        "final_all_registry_residual_int8": True,
        "matched_fp32_shared_reference_directions": True,
        "target_old_int8_used_for_formal_prediction": True,
        "target_new_int8_used_for_formal_prediction": True,
        "fp32_target_direction_stored_in_formal_state": False,
        "ground_int8_component_input_count": 0,
        "ground_int8_update_access": False,
        "class_id_specific_branch": False,
        "label_permutation_equivariant": True,
        "b_logdiag_update_norm": b_logdiag_delta,
        "b_old_weight_update_norm": b_old_delta,
        "c_logdiag_update_norm": c_logdiag_delta,
        "c_old_weight_update_norm": c_old_delta,
        "c_new_weight_update_norm": c_new_delta,
        "quantization_error_mean": float(np.mean(quant_error)),
        "quantization_error_max": float(np.max(quant_error)),
        "int8_vs_fp32_before_support_argmax_change_count": before_argmax_changes,
        "int8_vs_fp32_final_support_argmax_change_count": final_argmax_changes,
        "query_rows_used": 0,
    }
    resource = {
        "schema": "cvs.phase2.d41.bec_resource.v1",
        "support_only": True,
        "stage2b_trainable_parameters": peak_parameters_b,
        "stage2c_trainable_parameters": peak_parameters_c,
        "trainable_parameters": peak_parameters,
        "trainable_parameter_cap": 80_000,
        "trainable_parameter_cap_pass": peak_parameters <= 80_000,
        "adaptation_epochs": 30,
        "optimizer_steps": 30,
        "stage2b_optimizer_steps": 20,
        "stage2c_optimizer_steps": 10,
        "adaptation_epoch_cap": 30,
        "optimizer_step_cap": 50,
        "adaptation_epoch_cap_pass": True,
        "optimizer_step_cap_pass": True,
        "stage2c_step_lock_pass": True,
        "persistent_state_bytes": state.persistent_state_bytes,
        "base_persistent_state_bytes": state.base_state.persistent_state_bytes,
        "wrapper_metadata_bytes": state.wrapper_metadata_bytes,
        "persistent_state_cap_bytes": 256 * 1024,
        "persistent_state_cap_pass": state.persistent_state_bytes <= 256 * 1024,
        "estimated_single_view_adaptation_macs_lower_bound": single_view_lower,
        "estimated_single_view_support_macs": single_view_lower,
        "estimated_four_view_transform_classification_macs": four_view_macs,
        "estimated_js_scalar_operations": js_scalar_ops,
        "estimated_adaptation_macs": adaptation_macs,
        "estimated_bec_support_macs": adaptation_macs,
        "four_view_plus_js_exceeds_single_view_lower_bound": (
            adaptation_macs > single_view_lower
        ),
        "estimated_macs_per_query": int(
            FEATURE_DIM + 2 * FEATURE_DIM * class_count
        ),
        "support_view_names": list(VIEW_NAMES),
        "support_view_count": 4,
        "physical_support_observation_multiplicity": 1,
        "query_view": "full_only",
        "query_view_count": 1,
        "additional_physical_sample_count": 0,
        "additional_leo_overlay_count": 0,
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
        "ground_int8_component_input_count": 0,
        "ground_int8_update_access": False,
        "resident_fp32_target_prototype_count": 0,
        "formal_state_int8_only": True,
        "old_k_shot": old_k,
        "new_k_shot": new_k,
        "registered_class_count": class_count,
        "runtime_device": str(runtime_device),
        "peak_cuda_memory_bytes": (
            int(torch.cuda.max_memory_allocated(runtime_device))
            if runtime_device.type == "cuda"
            else 0
        ),
        "int8_vs_fp32_before_support_argmax_change_count": before_argmax_changes,
        "int8_vs_fp32_final_support_argmax_change_count": final_argmax_changes,
    }
    return D41BECResult(
        before_state=before_state,
        state=state,
        matched_fp32_before_state=matched_before,
        matched_fp32_state=matched_state,
        training_trace=tuple(dict(row) for row in trace),
        geometry_audit=geometry,
        resource_audit=resource,
    )


def score_d41_bec(state: D41BECState, features: np.ndarray) -> np.ndarray:
    if not isinstance(state, D41BECState):
        raise D41BECError("D41 score state drift")
    return score_d38_strong_b3(state.base_state, features)


def predict_d41_bec(state: D41BECState, features: np.ndarray) -> np.ndarray:
    return np.asarray(state.classes)[
        np.argmax(score_d41_bec(state, features), axis=1)
    ]


def pairwise_support_diagnostics_d41(
    state: D41BECState,
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
    scores = score_d41_bec(state, held_new_features)
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
        raise D41BECError("D41 pairwise diagnostic closure drift")
    output: list[dict[str, Any]] = []
    for row, truth, physical_id, rank in zip(scores, labels, ids, ranks, strict=True):
        truth_index = state.classes.index(truth)
        competing_new = np.array(row[old_count:], copy=True)
        competing_new[truth_index - old_count] = -np.inf
        competitor_index = old_count + int(np.argmax(competing_new))
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
    "D41BECConfig",
    "D41BECError",
    "D41BECResult",
    "D41BECState",
    "VIEW_NAMES",
    "d41_bec_loss",
    "fit_d41_bec",
    "make_d41_views",
    "pairwise_support_diagnostics_d41",
    "predict_d41_bec",
    "score_d41_bec",
]
