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
            total_loss = ce_loss + float(prototype_anchor_weight) * anchor_loss
            total_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm=5.0)
            optimizer.step()
            count = int(len(positions))
            epoch_ce += float(ce_loss.detach()) * count
            epoch_anchor += float(anchor_loss.detach()) * count
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
    )
    info = {
        "adaptation_objective": "support_only_extreme_light_diag_metric_cosine_ce",
        "head_mode": "extreme_light_diag_cosine",
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
