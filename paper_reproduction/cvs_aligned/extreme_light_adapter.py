"""Support-only extreme-light classifier for frozen ADV3B02 features.

The adapter learns a diagonal metric and a cosine classifier from registered
target-old/target-new support only. Query rows are scored independently after
the state is frozen; they never participate in optimization or calibration.
"""

from __future__ import annotations

import math
import time
from typing import Any

import numpy as np


EPS = 1.0e-8


def _numpy_norm(rows: np.ndarray) -> np.ndarray:
    values = np.asarray(rows, dtype=np.float32)
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), EPS)


def concatenate_registered_features(
    primary: np.ndarray,
    auxiliary: np.ndarray | None,
    *,
    auxiliary_weight: float,
    source_logits: np.ndarray | None = None,
    source_logit_weight: float = 0.0,
) -> np.ndarray:
    """Build one deployable sample feature from frozen ADV3B02 outputs.

    ``source_logits`` are the frozen source-classifier outputs for the same
    physical row. They are consumed only as per-sample inference features and
    never expose the row's old/new role or any query-batch statistic.
    """
    primary_norm = _numpy_norm(primary)
    weight = float(auxiliary_weight)
    logit_weight = float(source_logit_weight)
    if not math.isfinite(weight) or weight < 0.0:
        raise ValueError("auxiliary_weight must be finite and nonnegative")
    if not math.isfinite(logit_weight) or logit_weight < 0.0:
        raise ValueError("source_logit_weight must be finite and nonnegative")
    blocks = [primary_norm]
    if weight > 0.0:
        if auxiliary is None:
            raise ValueError("positive auxiliary_weight requires auxiliary features")
        if len(primary_norm) != len(auxiliary):
            raise ValueError("primary and auxiliary rows must align")
        blocks.append(weight * _numpy_norm(auxiliary))
    if logit_weight > 0.0:
        if source_logits is None:
            raise ValueError("positive source_logit_weight requires source logits")
        if len(primary_norm) != len(source_logits):
            raise ValueError("primary and source-logit rows must align")
        logits = np.asarray(source_logits, dtype=np.float32)
        if logits.ndim != 2 or not np.all(np.isfinite(logits)):
            raise ValueError("source logits must be a finite rank-2 matrix")
        blocks.append(logit_weight * _numpy_norm(logits))
    if len(blocks) == 1:
        return primary_norm
    return _numpy_norm(np.concatenate(blocks, axis=1))


def fit_predict_extreme_light_diag_cosine(
    support_x: np.ndarray,
    support_y: np.ndarray,
    query_x: np.ndarray,
    *,
    seed: int,
    epochs: int = 20,
    learning_rate: float = 0.01,
    batch_size: int = 32,
    temperature: float = 18.0,
    prototype_anchor_weight: float = 0.05,
    feature_noise_std: float = 0.01,
    weight_decay: float = 0.002,
    max_trainable_parameters: int = 50_000,
    max_persistent_state_bytes: int = 128 * 1024,
    source_anchor_x: np.ndarray | None = None,
    source_anchor_y: np.ndarray | None = None,
    source_anchor_strength: float = 0.0,
    source_anchor_blend: float = 0.25,
    device: str = "cpu",
) -> tuple[np.ndarray, dict[str, Any], list[dict[str, Any]]]:
    """Fit the support-only diagonal metric and return independent query predictions."""
    import torch
    import torch.nn.functional as F

    if int(epochs) <= 0 or int(epochs) > 20:
        raise ValueError("extreme-light adaptation epochs must be in [1,20]")
    if int(batch_size) <= 0:
        raise ValueError("batch_size must be positive")
    if not math.isfinite(float(learning_rate)) or float(learning_rate) <= 0.0:
        raise ValueError("learning_rate must be finite and positive")
    if not math.isfinite(float(temperature)) or float(temperature) <= 0.0:
        raise ValueError("temperature must be finite and positive")
    if not math.isfinite(float(source_anchor_strength)) or float(source_anchor_strength) < 0.0:
        raise ValueError("source_anchor_strength must be finite and nonnegative")
    if not math.isfinite(float(source_anchor_blend)) or not 0.0 <= float(source_anchor_blend) <= 1.0:
        raise ValueError("source_anchor_blend must be in [0,1]")
    support = np.asarray(support_x, dtype=np.float32)
    query = np.asarray(query_x, dtype=np.float32)
    labels = np.asarray(support_y).astype(str)
    if support.ndim != 2 or query.ndim != 2 or support.shape[1] != query.shape[1]:
        raise ValueError("support/query features must be aligned rank-2 matrices")
    if len(support) != len(labels) or len(support) == 0:
        raise ValueError("support features and labels must be non-empty and aligned")
    classes = sorted(set(labels.tolist()))
    if len(classes) < 2:
        raise ValueError("extreme-light classifier requires at least two registered classes")
    class_to_index = {label: index for index, label in enumerate(classes)}
    targets_np = np.asarray([class_to_index[label] for label in labels], dtype=np.int64)
    feature_dim = int(support.shape[1])
    class_count = int(len(classes))
    trainable_parameters = int(feature_dim + class_count * feature_dim + class_count)
    persistent_state_bytes = int(trainable_parameters * np.dtype(np.float32).itemsize)
    if trainable_parameters > int(max_trainable_parameters):
        raise ValueError(
            f"extreme-light parameter cap exceeded: {trainable_parameters}>{max_trainable_parameters}"
        )
    if persistent_state_bytes > int(max_persistent_state_bytes):
        raise ValueError(
            "extreme-light persistent-state cap exceeded: "
            f"{persistent_state_bytes}>{max_persistent_state_bytes}"
        )

    requested_device = str(device)
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError(f"CUDA device requested but unavailable: {requested_device}")
    runtime_device = torch.device(requested_device)
    torch.manual_seed(int(seed))
    if runtime_device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))
        # ``reset_peak_memory_stats`` requires an initialized CUDA context.
        # A zero-sized allocation initializes the selected visible device
        # without contributing persistent model state.
        torch.empty(0, device=runtime_device)
        torch.cuda.reset_peak_memory_stats(runtime_device)
    x = torch.as_tensor(support, dtype=torch.float32, device=runtime_device)
    q = torch.as_tensor(query, dtype=torch.float32, device=runtime_device)
    targets = torch.as_tensor(targets_np, dtype=torch.long, device=runtime_device)
    prototypes = torch.stack(
        [F.normalize(x[targets == index].mean(dim=0), dim=0) for index in range(class_count)]
    )
    source_anchor_rows = 0
    source_anchor_class_indices: list[int] = []
    source_anchor_prototypes: list[torch.Tensor] = []
    if float(source_anchor_strength) > 0.0:
        if source_anchor_x is None or source_anchor_y is None:
            raise ValueError("positive source_anchor_strength requires a frozen source bank")
        source_values = np.asarray(source_anchor_x, dtype=np.float32)
        source_labels = np.asarray(source_anchor_y).astype(str)
        if source_values.ndim != 2 or source_values.shape[1] != feature_dim:
            raise ValueError("source anchor features must align with support feature dimension")
        if len(source_values) != len(source_labels) or len(source_values) == 0:
            raise ValueError("source anchor features and labels must be non-empty and aligned")
        if not np.all(np.isfinite(source_values)):
            raise ValueError("source anchor features must be finite")
        unknown_source_labels = sorted(set(source_labels.tolist()) - set(classes))
        if unknown_source_labels:
            raise ValueError(f"source anchor labels are not registered: {unknown_source_labels}")
        source_tensor = torch.as_tensor(source_values, dtype=torch.float32, device=runtime_device)
        for label in sorted(set(source_labels.tolist())):
            index = class_to_index[label]
            mask = torch.as_tensor(source_labels == label, dtype=torch.bool, device=runtime_device)
            source_prototype = F.normalize(source_tensor[mask].mean(dim=0), dim=0)
            blended = F.normalize(
                (1.0 - float(source_anchor_blend)) * prototypes[index]
                + float(source_anchor_blend) * source_prototype,
                dim=0,
            )
            source_anchor_class_indices.append(index)
            source_anchor_prototypes.append(blended)
        source_anchor_rows = int(len(source_values))
    log_scale = torch.nn.Parameter(torch.zeros(feature_dim, device=runtime_device))
    weights = torch.nn.Parameter(prototypes.detach().clone())
    bias = torch.nn.Parameter(torch.zeros(class_count, device=runtime_device))
    parameters = [log_scale, weights, bias]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )
    generator = torch.Generator(device=runtime_device).manual_seed(int(seed))
    trace: list[dict[str, Any]] = []
    started = time.perf_counter()
    for epoch in range(1, int(epochs) + 1):
        permutation = torch.randperm(len(x), generator=generator, device=runtime_device)
        epoch_ce = 0.0
        epoch_anchor = 0.0
        epoch_frozen_source_anchor = 0.0
        epoch_total = 0.0
        epoch_correct = 0
        epoch_seen = 0
        epoch_grad_norm = 0.0
        batch_count = 0
        for positions in permutation.split(min(int(batch_size), len(x))):
            optimizer.zero_grad(set_to_none=True)
            rows = x[positions]
            if float(feature_noise_std) > 0.0:
                rows = rows + float(feature_noise_std) * torch.randn(
                    rows.shape,
                    generator=generator,
                    device=runtime_device,
                    dtype=rows.dtype,
                )
            scaled = rows * torch.exp(torch.clamp(log_scale, -1.5, 1.5))
            logits = float(temperature) * (
                F.normalize(scaled, dim=1) @ F.normalize(weights, dim=1).T
            ) + bias
            ce_loss = F.cross_entropy(logits, targets[positions])
            anchor_loss = torch.mean((F.normalize(weights, dim=1) - prototypes) ** 2)
            frozen_source_anchor_loss = torch.zeros((), dtype=weights.dtype, device=runtime_device)
            if source_anchor_prototypes:
                selected_weights = F.normalize(weights[source_anchor_class_indices], dim=1)
                selected_anchors = torch.stack(source_anchor_prototypes)
                frozen_source_anchor_loss = torch.mean(
                    1.0 - torch.sum(selected_weights * selected_anchors, dim=1)
                )
            total_loss = (
                ce_loss
                + float(prototype_anchor_weight) * anchor_loss
                + float(source_anchor_strength) * frozen_source_anchor_loss
            )
            total_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm=5.0)
            optimizer.step()
            count = int(len(positions))
            epoch_ce += float(ce_loss.detach()) * count
            epoch_anchor += float(anchor_loss.detach()) * count
            epoch_frozen_source_anchor += float(frozen_source_anchor_loss.detach()) * count
            epoch_total += float(total_loss.detach()) * count
            epoch_correct += int((logits.argmax(dim=1) == targets[positions]).sum().item())
            epoch_seen += count
            epoch_grad_norm += float(grad_norm.detach())
            batch_count += 1
        row = {
            "phase": "support_only_extreme_light_fit",
            "epoch": int(epoch),
            "step": int(epoch),
            "total_steps": int(epochs),
            "loss": epoch_total / max(1, epoch_seen),
            "total_loss": epoch_total / max(1, epoch_seen),
            "ce_loss": epoch_ce / max(1, epoch_seen),
            "source_anchor_loss": epoch_anchor / max(1, epoch_seen),
            "frozen_source_bank_anchor_loss": epoch_frozen_source_anchor / max(1, epoch_seen),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "gradient_norm": epoch_grad_norm / max(1, batch_count),
            "support_accuracy": epoch_correct / max(1, epoch_seen),
        }
        if not all(math.isfinite(float(value)) for key, value in row.items() if key != "phase"):
            raise FloatingPointError(f"non-finite extreme-light loss trace: {row}")
        trace.append(row)
    adaptation_elapsed = time.perf_counter() - started

    scoring_started = time.perf_counter()
    with torch.no_grad():
        scaled_query = q * torch.exp(torch.clamp(log_scale, -1.5, 1.5))
        query_logits = float(temperature) * (
            F.normalize(scaled_query, dim=1) @ F.normalize(weights, dim=1).T
        ) + bias
        predicted_indices = query_logits.argmax(dim=1).detach().cpu().numpy()
    scoring_elapsed = time.perf_counter() - scoring_started
    peak_device_memory = (
        int(torch.cuda.max_memory_allocated(runtime_device)) if runtime_device.type == "cuda" else 0
    )
    macs_per_query = int(feature_dim + class_count * feature_dim)
    forward_macs_per_support = int(feature_dim + class_count * feature_dim)
    estimated_adaptation_macs = int(
        int(epochs) * len(support) * forward_macs_per_support * 3
        + source_anchor_rows * feature_dim
    )
    info = {
        "adaptation_objective": "support_only_extreme_light_diag_metric_cosine_ce",
        "head_mode": "extreme_light_diag_cosine",
        "support_only": True,
        "frozen_source_bank_used": bool(source_anchor_prototypes),
        "frozen_source_bank_rows": source_anchor_rows,
        "frozen_source_bank_class_count": int(len(source_anchor_prototypes)),
        "frozen_source_bank_anchor_strength": float(source_anchor_strength),
        "frozen_source_bank_anchor_blend": float(source_anchor_blend),
        "frozen_source_bank_updated": False,
        "query_labels_used_for_adaptation": False,
        "query_features_used_for_adaptation": False,
        "query_query_graph_used": False,
        "query_batch_state_required": False,
        "decision_batch_state_required": False,
        "role_oracle_used": False,
        "equal_class_quota_used": False,
        "feature_adapter_updates_adv3b02": False,
        "feature_adapter_gradient_updates": int(epochs),
        "feature_adapter_mode": "support_diag_metric_cosine",
        "trainable_parameters": trainable_parameters,
        "persistent_state_bytes": persistent_state_bytes,
        "persistent_state_bytes_with_post_adapter": persistent_state_bytes,
        "stored_raw_support_count": 0,
        "stored_quantized_support_code_count": 0,
        "stored_class_prototype_count": class_count,
        "feature_dim": feature_dim,
        "class_count": class_count,
        "adaptation_epochs": int(epochs),
        "adaptation_batch_size": int(batch_size),
        "estimated_adaptation_macs": estimated_adaptation_macs,
        "estimated_head_macs": int(len(query) * macs_per_query),
        "estimated_head_macs_with_post_adapter": int(len(query) * macs_per_query),
        "estimated_macs_per_query": macs_per_query,
        "dense_graph_bytes_lower_bound": 0,
        "dense_graph_peak_bytes_lower_bound": 0,
        "dense_graph_cumulative_bytes": 0,
        "decision_workspace_bytes_lower_bound": 0,
        "adaptation_latency_sec": float(adaptation_elapsed),
        "score_matrix_latency_sec": float(scoring_elapsed),
        "onboard_scoring_latency_sec": float(scoring_elapsed),
        "onboard_scoring_latency_per_query_ms": float(
            scoring_elapsed * 1000.0 / max(1, len(query))
        ),
        "peak_device_memory_bytes": peak_device_memory,
        "loss": float(trace[-1]["loss"]),
        "loss_initial": float(trace[0]["loss"]),
        "loss_final": float(trace[-1]["loss"]),
        "support_accuracy_final": float(trace[-1]["support_accuracy"]),
        "runtime_device": str(runtime_device),
    }
    predicted = np.asarray(classes, dtype=object)[predicted_indices]
    return predicted, info, trace


def predict_support_prototype_cosine(
    support_x: np.ndarray,
    support_y: np.ndarray,
    query_x: np.ndarray,
    *,
    max_persistent_state_bytes: int = 128 * 1024,
) -> tuple[np.ndarray, dict[str, Any], list[dict[str, Any]]]:
    """Closed-form support prototypes with independent per-query cosine scoring."""
    support = _numpy_norm(np.asarray(support_x, dtype=np.float32))
    query = _numpy_norm(np.asarray(query_x, dtype=np.float32))
    labels = np.asarray(support_y).astype(str)
    if support.ndim != 2 or query.ndim != 2 or support.shape[1] != query.shape[1]:
        raise ValueError("support/query features must be aligned rank-2 matrices")
    if len(support) != len(labels) or len(support) == 0:
        raise ValueError("support features and labels must be non-empty and aligned")
    classes = sorted(set(labels.tolist()))
    if len(classes) < 2:
        raise ValueError("prototype classifier requires at least two registered classes")
    started = time.perf_counter()
    prototypes = np.stack(
        [_numpy_norm(support[labels == label].mean(axis=0, keepdims=True))[0] for label in classes]
    ).astype(np.float32)
    enrollment_elapsed = time.perf_counter() - started
    persistent_state_bytes = int(prototypes.nbytes)
    if persistent_state_bytes > int(max_persistent_state_bytes):
        raise ValueError(
            "extreme-light persistent-state cap exceeded: "
            f"{persistent_state_bytes}>{max_persistent_state_bytes}"
        )
    scoring_started = time.perf_counter()
    scores = query @ prototypes.T
    predicted_indices = np.argmax(scores, axis=1)
    scoring_elapsed = time.perf_counter() - scoring_started
    assigned = np.asarray([classes.index(label) for label in labels], dtype=np.int64)
    compactness_loss = float(np.mean(1.0 - np.sum(support * prototypes[assigned], axis=1)))
    feature_dim = int(support.shape[1])
    class_count = int(len(classes))
    macs_per_query = int(feature_dim * class_count)
    info = {
        "adaptation_objective": "closed_form_support_prototype_cosine",
        "head_mode": "extreme_light_prototype_cosine",
        "support_only": True,
        "query_labels_used_for_adaptation": False,
        "query_features_used_for_adaptation": False,
        "query_query_graph_used": False,
        "query_batch_state_required": False,
        "decision_batch_state_required": False,
        "role_oracle_used": False,
        "equal_class_quota_used": False,
        "feature_adapter_updates_adv3b02": False,
        "feature_adapter_gradient_updates": 0,
        "feature_adapter_mode": "closed_form_support_prototype_cosine",
        "trainable_parameters": 0,
        "persistent_state_bytes": persistent_state_bytes,
        "persistent_state_bytes_with_post_adapter": persistent_state_bytes,
        "stored_raw_support_count": 0,
        "stored_quantized_support_code_count": 0,
        "stored_class_prototype_count": class_count,
        "feature_dim": feature_dim,
        "class_count": class_count,
        "adaptation_epochs": 0,
        "adaptation_batch_size": 0,
        "estimated_adaptation_macs": int(len(support) * feature_dim),
        "estimated_head_macs": int(len(query) * macs_per_query),
        "estimated_head_macs_with_post_adapter": int(len(query) * macs_per_query),
        "estimated_macs_per_query": macs_per_query,
        "dense_graph_bytes_lower_bound": 0,
        "dense_graph_peak_bytes_lower_bound": 0,
        "dense_graph_cumulative_bytes": 0,
        "decision_workspace_bytes_lower_bound": 0,
        "adaptation_latency_sec": float(enrollment_elapsed),
        "score_matrix_latency_sec": float(scoring_elapsed),
        "onboard_scoring_latency_sec": float(scoring_elapsed),
        "onboard_scoring_latency_per_query_ms": float(
            scoring_elapsed * 1000.0 / max(1, len(query))
        ),
        "peak_device_memory_bytes": 0,
        "loss": compactness_loss,
        "loss_initial": compactness_loss,
        "loss_final": compactness_loss,
        "support_accuracy_final": float(
            np.mean(np.argmax(support @ prototypes.T, axis=1) == assigned)
        ),
        "runtime_device": "cpu_closed_form",
    }
    trace = [
        {
            "phase": "closed_form_support_prototype_fit",
            "epoch": 0,
            "step": 1,
            "total_steps": 1,
            "loss": compactness_loss,
            "total_loss": compactness_loss,
            "ce_loss": 0.0,
            "source_anchor_loss": compactness_loss,
            "learning_rate": 0.0,
            "gradient_norm": 0.0,
            "support_accuracy": info["support_accuracy_final"],
        }
    ]
    predicted = np.asarray(classes, dtype=object)[predicted_indices]
    return predicted, info, trace


def predict_support_diag_gaussian(
    support_x: np.ndarray,
    support_y: np.ndarray,
    query_x: np.ndarray,
    *,
    variance_shrinkage: float = 0.9,
    logdet_weight: float = 0.25,
    variance_floor_ratio: float = 0.01,
    max_persistent_state_bytes: int = 128 * 1024,
) -> tuple[np.ndarray, dict[str, Any], list[dict[str, Any]]]:
    """Closed-form class-diagonal Gaussian fitted from registered support only.

    Each class variance is shrunk toward the pooled within-class diagonal
    variance. The frozen means, inverse variances, and log-determinant offsets
    are sufficient to score every query row independently.
    """
    shrinkage = float(variance_shrinkage)
    determinant_weight = float(logdet_weight)
    floor_ratio = float(variance_floor_ratio)
    if not math.isfinite(shrinkage) or not 0.0 <= shrinkage <= 1.0:
        raise ValueError("variance_shrinkage must be finite and in [0,1]")
    if not math.isfinite(determinant_weight) or determinant_weight < 0.0:
        raise ValueError("logdet_weight must be finite and nonnegative")
    if not math.isfinite(floor_ratio) or floor_ratio <= 0.0:
        raise ValueError("variance_floor_ratio must be finite and positive")

    support = np.asarray(support_x, dtype=np.float32)
    query = np.asarray(query_x, dtype=np.float32)
    labels = np.asarray(support_y).astype(str)
    if support.ndim != 2 or query.ndim != 2 or support.shape[1] != query.shape[1]:
        raise ValueError("support/query features must be aligned rank-2 matrices")
    if len(support) != len(labels) or len(support) == 0:
        raise ValueError("support features and labels must be non-empty and aligned")
    if not np.all(np.isfinite(support)) or not np.all(np.isfinite(query)):
        raise ValueError("support/query features must be finite")
    classes = sorted(set(labels.tolist()))
    if len(classes) < 2:
        raise ValueError("diagonal Gaussian classifier requires at least two registered classes")

    started = time.perf_counter()
    means = np.stack([support[labels == label].mean(axis=0) for label in classes]).astype(
        np.float32
    )
    assigned = np.asarray([classes.index(label) for label in labels], dtype=np.int64)
    residuals = support - means[assigned]
    class_variances = np.stack(
        [np.mean(np.square(residuals[labels == label]), axis=0) for label in classes]
    ).astype(np.float32)
    pooled_variance = np.mean(np.square(residuals), axis=0).astype(np.float32)
    variance_floor = np.maximum(
        pooled_variance * np.float32(floor_ratio), np.float32(1.0e-6)
    )
    variances = (
        (1.0 - shrinkage) * class_variances + shrinkage * pooled_variance[None, :]
    ).astype(np.float32)
    variances = np.maximum(variances, variance_floor[None, :]).astype(np.float32)
    inverse_variances = np.reciprocal(variances).astype(np.float32)
    logdet_offsets = (
        determinant_weight * np.sum(np.log(variances), axis=1)
    ).astype(np.float32)
    enrollment_elapsed = time.perf_counter() - started

    persistent_state_bytes = int(
        means.nbytes + inverse_variances.nbytes + logdet_offsets.nbytes
    )
    if persistent_state_bytes > int(max_persistent_state_bytes):
        raise ValueError(
            "extreme-light persistent-state cap exceeded: "
            f"{persistent_state_bytes}>{max_persistent_state_bytes}"
        )

    scoring_started = time.perf_counter()
    weighted_means = means * inverse_variances
    quadratic_offsets = np.sum(means * weighted_means, axis=1)
    quadratic = (
        np.square(query) @ inverse_variances.T
        - 2.0 * (query @ weighted_means.T)
        + quadratic_offsets[None, :]
    )
    scores = -0.5 * (quadratic + logdet_offsets[None, :])
    predicted_indices = np.argmax(scores, axis=1)
    scoring_elapsed = time.perf_counter() - scoring_started

    support_quadratic = np.sum(
        np.square(support - means[assigned]) * inverse_variances[assigned], axis=1
    )
    compactness_loss = float(np.mean(support_quadratic))
    feature_dim = int(support.shape[1])
    class_count = int(len(classes))
    macs_per_query = int(4 * feature_dim * class_count + feature_dim + class_count)
    support_scores = -0.5 * (
        np.square(support) @ inverse_variances.T
        - 2.0 * (support @ weighted_means.T)
        + quadratic_offsets[None, :]
        + logdet_offsets[None, :]
    )
    info = {
        "adaptation_objective": "closed_form_support_diag_gaussian",
        "head_mode": "extreme_light_diag_gaussian",
        "support_only": True,
        "query_labels_used_for_adaptation": False,
        "query_features_used_for_adaptation": False,
        "query_query_graph_used": False,
        "query_batch_state_required": False,
        "decision_batch_state_required": False,
        "role_oracle_used": False,
        "equal_class_quota_used": False,
        "feature_adapter_updates_adv3b02": False,
        "feature_adapter_gradient_updates": 0,
        "feature_adapter_mode": "closed_form_support_diag_gaussian",
        "trainable_parameters": 0,
        "persistent_state_bytes": persistent_state_bytes,
        "persistent_state_bytes_with_post_adapter": persistent_state_bytes,
        "stored_raw_support_count": 0,
        "stored_quantized_support_code_count": 0,
        "stored_class_prototype_count": class_count,
        "stored_class_variance_count": class_count,
        "feature_dim": feature_dim,
        "class_count": class_count,
        "adaptation_epochs": 0,
        "adaptation_batch_size": 0,
        "variance_shrinkage": shrinkage,
        "logdet_weight": determinant_weight,
        "variance_floor_ratio": floor_ratio,
        "estimated_adaptation_macs": int(5 * len(support) * feature_dim),
        "estimated_head_macs": int(len(query) * macs_per_query),
        "estimated_head_macs_with_post_adapter": int(len(query) * macs_per_query),
        "estimated_macs_per_query": macs_per_query,
        "dense_graph_bytes_lower_bound": 0,
        "dense_graph_peak_bytes_lower_bound": 0,
        "dense_graph_cumulative_bytes": 0,
        "decision_workspace_bytes_lower_bound": int(class_count * np.dtype(np.float32).itemsize),
        "adaptation_latency_sec": float(enrollment_elapsed),
        "score_matrix_latency_sec": float(scoring_elapsed),
        "onboard_scoring_latency_sec": float(scoring_elapsed),
        "onboard_scoring_latency_per_query_ms": float(
            scoring_elapsed * 1000.0 / max(1, len(query))
        ),
        "peak_device_memory_bytes": 0,
        "loss": compactness_loss,
        "loss_initial": compactness_loss,
        "loss_final": compactness_loss,
        "support_accuracy_final": float(
            np.mean(np.argmax(support_scores, axis=1) == assigned)
        ),
        "runtime_device": "cpu_closed_form",
    }
    trace = [
        {
            "phase": "closed_form_support_diag_gaussian_fit",
            "epoch": 0,
            "step": 1,
            "total_steps": 1,
            "loss": compactness_loss,
            "total_loss": compactness_loss,
            "ce_loss": 0.0,
            "source_anchor_loss": compactness_loss,
            "learning_rate": 0.0,
            "gradient_norm": 0.0,
            "support_accuracy": info["support_accuracy_final"],
        }
    ]
    predicted = np.asarray(classes, dtype=object)[predicted_indices]
    return predicted, info, trace


def predict_support_multiprototype_cosine(
    support_x: np.ndarray,
    support_y: np.ndarray,
    query_x: np.ndarray,
    *,
    prototypes_per_class: int,
    kmeans_steps: int = 5,
    max_persistent_state_bytes: int = 128 * 1024,
) -> tuple[np.ndarray, dict[str, Any], list[dict[str, Any]]]:
    """Closed-form deterministic spherical multi-prototypes from support only."""
    support = _numpy_norm(np.asarray(support_x, dtype=np.float32))
    query = _numpy_norm(np.asarray(query_x, dtype=np.float32))
    labels = np.asarray(support_y).astype(str)
    proto_count = int(prototypes_per_class)
    steps = int(kmeans_steps)
    if support.ndim != 2 or query.ndim != 2 or support.shape[1] != query.shape[1]:
        raise ValueError("support/query features must be aligned rank-2 matrices")
    if len(support) != len(labels) or len(support) == 0:
        raise ValueError("support features and labels must be non-empty and aligned")
    if proto_count < 2 or proto_count > 4:
        raise ValueError("prototypes_per_class must be in [2,4]")
    if steps < 1 or steps > 10:
        raise ValueError("kmeans_steps must be in [1,10]")
    classes = sorted(set(labels.tolist()))
    if len(classes) < 2:
        raise ValueError("multi-prototype classifier requires at least two registered classes")

    started = time.perf_counter()
    class_prototypes: list[np.ndarray] = []
    for label in classes:
        points = support[labels == label]
        class_k = min(proto_count, len(points))
        mean_center = _numpy_norm(points.mean(axis=0, keepdims=True))[0]
        centers = [mean_center]
        while len(centers) < class_k:
            current = np.stack(centers, axis=0)
            nearest_similarity = np.max(points @ current.T, axis=1)
            centers.append(points[int(np.argmin(nearest_similarity))].copy())
        centers_array = np.stack(centers, axis=0).astype(np.float32)
        for _ in range(steps):
            assigned = np.argmax(points @ centers_array.T, axis=1)
            updated = centers_array.copy()
            for cluster in range(class_k):
                members = points[assigned == cluster]
                if len(members):
                    updated[cluster] = _numpy_norm(members.mean(axis=0, keepdims=True))[0]
            if np.allclose(updated, centers_array, atol=1.0e-6):
                centers_array = updated
                break
            centers_array = updated
        if class_k < proto_count:
            centers_array = np.concatenate(
                [centers_array, np.repeat(centers_array[-1:], proto_count - class_k, axis=0)], axis=0
            )
        class_prototypes.append(centers_array.astype(np.float32))
    prototypes = np.stack(class_prototypes, axis=0).astype(np.float32)
    enrollment_elapsed = time.perf_counter() - started
    persistent_state_bytes = int(prototypes.nbytes)
    if persistent_state_bytes > int(max_persistent_state_bytes):
        raise ValueError(
            "extreme-light persistent-state cap exceeded: "
            f"{persistent_state_bytes}>{max_persistent_state_bytes}"
        )

    scoring_started = time.perf_counter()
    flat_scores = query @ prototypes.reshape(-1, support.shape[1]).T
    scores = flat_scores.reshape(len(query), len(classes), proto_count).max(axis=2)
    predicted_indices = np.argmax(scores, axis=1)
    scoring_elapsed = time.perf_counter() - scoring_started
    support_scores = (support @ prototypes.reshape(-1, support.shape[1]).T).reshape(
        len(support), len(classes), proto_count
    ).max(axis=2)
    support_pred = np.argmax(support_scores, axis=1)
    class_to_index = {label: index for index, label in enumerate(classes)}
    targets = np.asarray([class_to_index[label] for label in labels], dtype=np.int64)
    compactness_loss = float(np.mean(1.0 - support_scores[np.arange(len(support)), targets]))
    feature_dim = int(support.shape[1])
    class_count = int(len(classes))
    macs_per_query = int(feature_dim * class_count * proto_count)
    info = {
        "adaptation_objective": "closed_form_support_spherical_multiprototype_cosine",
        "head_mode": "extreme_light_multiprototype_cosine",
        "support_only": True,
        "query_labels_used_for_adaptation": False,
        "query_features_used_for_adaptation": False,
        "query_query_graph_used": False,
        "query_batch_state_required": False,
        "decision_batch_state_required": False,
        "role_oracle_used": False,
        "equal_class_quota_used": False,
        "feature_adapter_updates_adv3b02": False,
        "feature_adapter_gradient_updates": 0,
        "feature_adapter_mode": "closed_form_support_spherical_multiprototype_cosine",
        "trainable_parameters": 0,
        "persistent_state_bytes": persistent_state_bytes,
        "persistent_state_bytes_with_post_adapter": persistent_state_bytes,
        "stored_raw_support_count": 0,
        "stored_quantized_support_code_count": 0,
        "stored_class_prototype_count": class_count * proto_count,
        "prototypes_per_class": proto_count,
        "kmeans_steps": steps,
        "feature_dim": feature_dim,
        "class_count": class_count,
        "adaptation_epochs": 0,
        "adaptation_batch_size": 0,
        "estimated_adaptation_macs": int(steps * len(support) * feature_dim * proto_count),
        "estimated_head_macs": int(len(query) * macs_per_query),
        "estimated_head_macs_with_post_adapter": int(len(query) * macs_per_query),
        "estimated_macs_per_query": macs_per_query,
        "dense_graph_bytes_lower_bound": 0,
        "dense_graph_peak_bytes_lower_bound": 0,
        "dense_graph_cumulative_bytes": 0,
        "decision_workspace_bytes_lower_bound": 0,
        "adaptation_latency_sec": float(enrollment_elapsed),
        "score_matrix_latency_sec": float(scoring_elapsed),
        "onboard_scoring_latency_sec": float(scoring_elapsed),
        "onboard_scoring_latency_per_query_ms": float(
            scoring_elapsed * 1000.0 / max(1, len(query))
        ),
        "peak_device_memory_bytes": 0,
        "loss": compactness_loss,
        "loss_initial": compactness_loss,
        "loss_final": compactness_loss,
        "support_accuracy_final": float(np.mean(support_pred == targets)),
        "runtime_device": "cpu",
    }
    trace = [
        {
            "phase": "closed_form_support_fit",
            "step": 1,
            "total_steps": 1,
            "loss": compactness_loss,
            "total_loss": compactness_loss,
            "ce_loss": 0.0,
            "source_anchor_loss": compactness_loss,
            "learning_rate": 0.0,
            "gradient_norm": 0.0,
            "support_accuracy": info["support_accuracy_final"],
        }
    ]
    predicted = np.asarray(classes, dtype=object)[predicted_indices]
    return predicted, info, trace


def fit_predict_support_ridge(
    support_x: np.ndarray,
    support_y: np.ndarray,
    query_x: np.ndarray,
    *,
    ridge_lambda: float,
    max_persistent_state_bytes: int = 128 * 1024,
) -> tuple[np.ndarray, dict[str, Any], list[dict[str, Any]]]:
    """Closed-form support-only multiclass ridge head with an explicit bias.

    Hyperparameters are preregistered from development runs. The fit consumes
    only labeled support rows and stores the resulting linear head; query rows
    are scored independently after enrollment.
    """
    support = np.asarray(support_x, dtype=np.float32)
    query = np.asarray(query_x, dtype=np.float32)
    labels = np.asarray(support_y).astype(str)
    lam = float(ridge_lambda)
    if support.ndim != 2 or query.ndim != 2 or support.shape[1] != query.shape[1]:
        raise ValueError("support/query features must be aligned rank-2 matrices")
    if len(support) != len(labels) or len(support) == 0:
        raise ValueError("support features and labels must be non-empty and aligned")
    if not math.isfinite(lam) or lam <= 0.0:
        raise ValueError("ridge_lambda must be finite and positive")
    classes = sorted(set(labels.tolist()))
    if len(classes) < 2:
        raise ValueError("ridge classifier requires at least two registered classes")
    class_to_index = {label: index for index, label in enumerate(classes)}
    targets = np.asarray([class_to_index[label] for label in labels], dtype=np.int64)
    class_count = len(classes)
    feature_dim = int(support.shape[1])
    design_dim = feature_dim + 1
    persistent_state_bytes = int(design_dim * class_count * np.dtype(np.float32).itemsize)
    if persistent_state_bytes > int(max_persistent_state_bytes):
        raise ValueError(
            "extreme-light persistent-state cap exceeded: "
            f"{persistent_state_bytes}>{max_persistent_state_bytes}"
        )

    started = time.perf_counter()
    design = np.concatenate(
        [support.astype(np.float64), np.ones((len(support), 1), dtype=np.float64)], axis=1
    )
    one_hot = np.eye(class_count, dtype=np.float64)[targets]
    gram = design.T @ design
    penalty = np.eye(design_dim, dtype=np.float64) * lam
    penalty[-1, -1] = 0.0
    weights = np.linalg.solve(gram + penalty, design.T @ one_hot).astype(np.float32)
    adaptation_elapsed = time.perf_counter() - started

    scoring_started = time.perf_counter()
    query_design = np.concatenate(
        [query, np.ones((len(query), 1), dtype=np.float32)], axis=1
    )
    query_scores = query_design @ weights
    predicted_indices = np.argmax(query_scores, axis=1)
    scoring_elapsed = time.perf_counter() - scoring_started
    support_scores = design.astype(np.float32) @ weights
    support_pred = np.argmax(support_scores, axis=1)
    mse_loss = float(np.mean((support_scores - one_hot.astype(np.float32)) ** 2))
    estimated_adaptation_macs = int(
        len(support) * design_dim * design_dim
        + (design_dim**3) / 3
        + len(support) * design_dim * class_count
    )
    macs_per_query = int(design_dim * class_count)
    info = {
        "adaptation_objective": "closed_form_support_multiclass_ridge",
        "head_mode": "extreme_light_support_ridge",
        "support_only": True,
        "query_labels_used_for_adaptation": False,
        "query_features_used_for_adaptation": False,
        "query_query_graph_used": False,
        "query_batch_state_required": False,
        "decision_batch_state_required": False,
        "role_oracle_used": False,
        "equal_class_quota_used": False,
        "feature_adapter_updates_adv3b02": False,
        "feature_adapter_gradient_updates": 0,
        "feature_adapter_mode": "closed_form_support_multiclass_ridge",
        "trainable_parameters": 0,
        "persistent_state_bytes": persistent_state_bytes,
        "persistent_state_bytes_with_post_adapter": persistent_state_bytes,
        "stored_raw_support_count": 0,
        "stored_quantized_support_code_count": 0,
        "stored_class_prototype_count": 0,
        "feature_dim": feature_dim,
        "class_count": class_count,
        "adaptation_epochs": 0,
        "adaptation_batch_size": 0,
        "ridge_lambda": lam,
        "estimated_adaptation_macs": estimated_adaptation_macs,
        "estimated_head_macs": int(len(query) * macs_per_query),
        "estimated_head_macs_with_post_adapter": int(len(query) * macs_per_query),
        "estimated_macs_per_query": macs_per_query,
        "dense_graph_bytes_lower_bound": 0,
        "dense_graph_peak_bytes_lower_bound": 0,
        "dense_graph_cumulative_bytes": 0,
        "decision_workspace_bytes_lower_bound": int(
            gram.nbytes + penalty.nbytes + design.nbytes + one_hot.nbytes
        ),
        "adaptation_latency_sec": float(adaptation_elapsed),
        "score_matrix_latency_sec": float(scoring_elapsed),
        "onboard_scoring_latency_sec": float(scoring_elapsed),
        "onboard_scoring_latency_per_query_ms": float(
            scoring_elapsed * 1000.0 / max(1, len(query))
        ),
        "peak_device_memory_bytes": 0,
        "loss": mse_loss,
        "loss_initial": mse_loss,
        "loss_final": mse_loss,
        "support_accuracy_final": float(np.mean(support_pred == targets)),
        "runtime_device": "cpu_closed_form",
    }
    trace = [
        {
            "phase": "closed_form_support_ridge_fit",
            "epoch": 0,
            "step": 1,
            "total_steps": 1,
            "loss": mse_loss,
            "total_loss": mse_loss,
            "ce_loss": mse_loss,
            "source_anchor_loss": 0.0,
            "learning_rate": 0.0,
            "gradient_norm": 0.0,
            "support_accuracy": info["support_accuracy_final"],
        }
    ]
    predicted = np.asarray(classes, dtype=object)[predicted_indices]
    return predicted, info, trace


def fit_predict_extreme_light_low_rank_cosine(
    support_x: np.ndarray,
    support_y: np.ndarray,
    query_x: np.ndarray,
    *,
    seed: int,
    rank: int,
    cosine_margin: float,
    residual_alpha: float = 0.5,
    epochs: int = 20,
    learning_rate: float = 0.01,
    batch_size: int = 32,
    temperature: float = 18.0,
    feature_noise_std: float = 0.01,
    weight_decay: float = 0.002,
    max_trainable_parameters: int = 50_000,
    max_persistent_state_bytes: int = 128 * 1024,
    device: str = "cpu",
) -> tuple[np.ndarray, dict[str, Any], list[dict[str, Any]]]:
    """Support-only low-rank residual metric with class-symmetric CosFace loss."""
    import torch
    import torch.nn.functional as F

    support = np.asarray(support_x, dtype=np.float32)
    query = np.asarray(query_x, dtype=np.float32)
    labels = np.asarray(support_y).astype(str)
    if support.ndim != 2 or query.ndim != 2 or support.shape[1] != query.shape[1]:
        raise ValueError("support/query features must be aligned rank-2 matrices")
    if len(support) != len(labels) or len(support) == 0:
        raise ValueError("support features and labels must be non-empty and aligned")
    if int(epochs) <= 0 or int(epochs) > 20:
        raise ValueError("extreme-light adaptation epochs must be in [1,20]")
    if int(rank) <= 0 or int(rank) > 32:
        raise ValueError("extreme-light low-rank width must be in [1,32]")
    if not math.isfinite(float(cosine_margin)) or not 0.0 <= float(cosine_margin) <= 0.5:
        raise ValueError("cosine_margin must be finite and in [0,0.5]")
    if not math.isfinite(float(residual_alpha)) or not 0.0 < float(residual_alpha) <= 1.0:
        raise ValueError("residual_alpha must be finite and in (0,1]")
    classes = sorted(set(labels.tolist()))
    class_to_index = {label: index for index, label in enumerate(classes)}
    targets_np = np.asarray([class_to_index[label] for label in labels], dtype=np.int64)
    feature_dim = int(support.shape[1])
    class_count = len(classes)
    trainable_parameters = int(
        feature_dim + 2 * feature_dim * int(rank) + class_count * feature_dim + class_count
    )
    persistent_state_bytes = int(trainable_parameters * np.dtype(np.float32).itemsize)
    if trainable_parameters > int(max_trainable_parameters):
        raise ValueError(
            f"extreme-light parameter cap exceeded: {trainable_parameters}>{max_trainable_parameters}"
        )
    if persistent_state_bytes > int(max_persistent_state_bytes):
        raise ValueError(
            "extreme-light persistent-state cap exceeded: "
            f"{persistent_state_bytes}>{max_persistent_state_bytes}"
        )
    requested_device = str(device)
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError(f"CUDA device requested but unavailable: {requested_device}")
    runtime_device = torch.device(requested_device)
    torch.manual_seed(int(seed))
    if runtime_device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))
        torch.empty(0, device=runtime_device)
        torch.cuda.reset_peak_memory_stats(runtime_device)
    x = torch.as_tensor(support, dtype=torch.float32, device=runtime_device)
    q = torch.as_tensor(query, dtype=torch.float32, device=runtime_device)
    targets = torch.as_tensor(targets_np, dtype=torch.long, device=runtime_device)
    prototypes = torch.stack(
        [F.normalize(x[targets == index].mean(dim=0), dim=0) for index in range(class_count)]
    )
    log_scale = torch.nn.Parameter(torch.zeros(feature_dim, device=runtime_device))
    down = torch.nn.Parameter(
        torch.randn(feature_dim, int(rank), device=runtime_device) / math.sqrt(feature_dim)
    )
    up = torch.nn.Parameter(torch.zeros(int(rank), feature_dim, device=runtime_device))
    weights = torch.nn.Parameter(prototypes.detach().clone())
    bias = torch.nn.Parameter(torch.zeros(class_count, device=runtime_device))
    parameters = [log_scale, down, up, weights, bias]
    optimizer = torch.optim.AdamW(parameters, lr=float(learning_rate), weight_decay=float(weight_decay))
    generator = torch.Generator(device=runtime_device).manual_seed(int(seed))

    def transform(rows: Any) -> Any:
        scaled = rows * torch.exp(torch.clamp(log_scale, -1.5, 1.5))
        residual = F.gelu(rows @ down) @ up
        return F.normalize(scaled + float(residual_alpha) * residual, dim=1)

    trace: list[dict[str, Any]] = []
    started = time.perf_counter()
    for epoch in range(1, int(epochs) + 1):
        permutation = torch.randperm(len(x), generator=generator, device=runtime_device)
        total_loss_sum = 0.0
        correct = 0
        seen = 0
        grad_norm_sum = 0.0
        batches = 0
        for positions in permutation.split(min(int(batch_size), len(x))):
            optimizer.zero_grad(set_to_none=True)
            rows = x[positions]
            if float(feature_noise_std) > 0.0:
                rows = rows + float(feature_noise_std) * torch.randn(
                    rows.shape, generator=generator, device=runtime_device, dtype=rows.dtype
                )
            logits = float(temperature) * (transform(rows) @ F.normalize(weights, dim=1).T) + bias
            margin_logits = logits.clone()
            margin_logits[
                torch.arange(len(positions), device=runtime_device), targets[positions]
            ] -= float(temperature) * float(cosine_margin)
            loss = F.cross_entropy(margin_logits, targets[positions])
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm=5.0)
            optimizer.step()
            count = len(positions)
            total_loss_sum += float(loss.detach()) * count
            correct += int((logits.argmax(dim=1) == targets[positions]).sum().item())
            seen += count
            grad_norm_sum += float(grad_norm.detach())
            batches += 1
        row = {
            "phase": "support_only_extreme_light_low_rank_fit",
            "epoch": epoch,
            "step": epoch,
            "total_steps": int(epochs),
            "loss": total_loss_sum / max(1, seen),
            "total_loss": total_loss_sum / max(1, seen),
            "ce_loss": total_loss_sum / max(1, seen),
            "source_anchor_loss": 0.0,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "gradient_norm": grad_norm_sum / max(1, batches),
            "support_accuracy": correct / max(1, seen),
        }
        if not all(math.isfinite(float(value)) for key, value in row.items() if key != "phase"):
            raise FloatingPointError(f"non-finite extreme-light loss trace: {row}")
        trace.append(row)
    adaptation_elapsed = time.perf_counter() - started
    scoring_started = time.perf_counter()
    with torch.no_grad():
        logits = float(temperature) * (transform(q) @ F.normalize(weights, dim=1).T) + bias
        predicted_indices = logits.argmax(dim=1).cpu().numpy()
    scoring_elapsed = time.perf_counter() - scoring_started
    peak_device_memory = (
        int(torch.cuda.max_memory_allocated(runtime_device)) if runtime_device.type == "cuda" else 0
    )
    transform_macs = int(feature_dim + 2 * feature_dim * int(rank))
    macs_per_query = int(transform_macs + class_count * feature_dim)
    info = {
        "adaptation_objective": "support_only_low_rank_residual_cosface_ce",
        "head_mode": "extreme_light_low_rank_cosine",
        "support_only": True,
        "query_labels_used_for_adaptation": False,
        "query_features_used_for_adaptation": False,
        "query_query_graph_used": False,
        "query_batch_state_required": False,
        "decision_batch_state_required": False,
        "role_oracle_used": False,
        "equal_class_quota_used": False,
        "feature_adapter_updates_adv3b02": False,
        "feature_adapter_gradient_updates": int(epochs),
        "feature_adapter_mode": "support_low_rank_residual_cosine",
        "trainable_parameters": trainable_parameters,
        "persistent_state_bytes": persistent_state_bytes,
        "persistent_state_bytes_with_post_adapter": persistent_state_bytes,
        "stored_raw_support_count": 0,
        "stored_quantized_support_code_count": 0,
        "stored_class_prototype_count": class_count,
        "feature_dim": feature_dim,
        "class_count": class_count,
        "adaptation_epochs": int(epochs),
        "adaptation_batch_size": int(batch_size),
        "low_rank_width": int(rank),
        "cosine_margin": float(cosine_margin),
        "residual_alpha": float(residual_alpha),
        "estimated_adaptation_macs": int(int(epochs) * len(support) * macs_per_query * 3),
        "estimated_head_macs": int(len(query) * macs_per_query),
        "estimated_head_macs_with_post_adapter": int(len(query) * macs_per_query),
        "estimated_macs_per_query": macs_per_query,
        "dense_graph_bytes_lower_bound": 0,
        "dense_graph_peak_bytes_lower_bound": 0,
        "dense_graph_cumulative_bytes": 0,
        "decision_workspace_bytes_lower_bound": 0,
        "adaptation_latency_sec": float(adaptation_elapsed),
        "score_matrix_latency_sec": float(scoring_elapsed),
        "onboard_scoring_latency_sec": float(scoring_elapsed),
        "onboard_scoring_latency_per_query_ms": float(scoring_elapsed * 1000.0 / max(1, len(query))),
        "peak_device_memory_bytes": peak_device_memory,
        "loss": float(trace[-1]["loss"]),
        "loss_initial": float(trace[0]["loss"]),
        "loss_final": float(trace[-1]["loss"]),
        "support_accuracy_final": float(trace[-1]["support_accuracy"]),
        "runtime_device": str(runtime_device),
    }
    predicted = np.asarray(classes, dtype=object)[predicted_indices]
    return predicted, info, trace
