"""Support-only D21 prototype adaptation on an existing Phase2 capsule.

The ``predict`` command never opens the truth sidecar. It fits every candidate
from registered support only, writes immutable-style truth-free predictions,
and records resource measurements. The separate ``score`` command joins truth
after prediction and cannot alter candidate state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "code"))
from cvsrffi.stage2_diag_cosine_exploration import spectral_logmag_sketch


SCENARIOS = ("leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak")
METHODS = (
    "L0_identity",
    "L1_radius",
    "L2_old_guard",
    "L3_sparse_rival",
    "L4_two_proto",
    "L5_fixed_iq_fft96_top1",
    "L5q_fixed_iq_fft96_top1_int8",
    "L6_diag_metric_fft96_top1",
    "L6q_diag_metric_fft96_top1_int8",
    "L7q_oldlock_cvar_diag_metric_int8",
    "L8q_diag_metric_blend_int8",
    "M2q_lowrank_metric_int8",
)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _json_dump(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _normalise(rows: np.ndarray) -> np.ndarray:
    rows = np.asarray(rows, dtype=np.float32)
    return rows / np.maximum(np.linalg.norm(rows, axis=1, keepdims=True), 1e-8)


def _prototypes(features: np.ndarray, labels: np.ndarray, class_count: int) -> np.ndarray:
    out = np.empty((class_count, features.shape[1]), dtype=np.float32)
    for class_index in range(class_count):
        members = features[labels == class_index]
        if members.shape[0] == 0:
            raise RuntimeError(f"empty support class {class_index}")
        out[class_index] = _normalise(members.mean(axis=0, keepdims=True))[0]
    return out


def _radii(features: np.ndarray, labels: np.ndarray, prototypes: np.ndarray) -> np.ndarray:
    values = []
    for class_index in range(prototypes.shape[0]):
        distances = 1.0 - features[labels == class_index] @ prototypes[class_index]
        values.append(float(np.quantile(distances, 0.9, method="higher")))
    return np.maximum(np.asarray(values, dtype=np.float32), 1e-4)


def _radius_bias(radii: np.ndarray, gamma: float) -> np.ndarray:
    median = float(np.median(radii))
    return gamma * np.clip(np.log(median / np.maximum(radii, 1e-6)), -1.0, 1.0)


def _loo_base(
    features: np.ndarray, labels: np.ndarray, class_count: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    count, dim = features.shape
    sums = np.zeros((class_count, dim), dtype=np.float32)
    class_counts = np.zeros(class_count, dtype=np.int64)
    for class_index in range(class_count):
        mask = labels == class_index
        sums[class_index] = features[mask].sum(axis=0)
        class_counts[class_index] = int(mask.sum())
    scores = np.empty((count, class_count), dtype=np.float32)
    for row in range(count):
        protos = sums.copy()
        own = int(labels[row])
        protos[own] -= features[row]
        denominators = class_counts.astype(np.float32)
        denominators[own] -= 1.0
        if denominators[own] < 1:
            raise RuntimeError("LOO requires at least two support rows per class")
        protos /= denominators[:, None]
        protos = _normalise(protos)
        scores[row] = features[row] @ protos.T
    full_proto = _prototypes(features, labels, class_count)
    radii = _radii(features, labels, full_proto)
    return scores, full_proto, radii


def _two_prototypes(rows: np.ndarray) -> np.ndarray:
    """Deterministic two-centre spherical clustering for one registered class."""
    rows = _normalise(rows)
    if rows.shape[0] < 2:
        return np.repeat(rows[:1], 2, axis=0)
    distance = 1.0 - rows @ rows.T
    first, second = np.unravel_index(int(np.argmax(distance)), distance.shape)
    centres = rows[[first, second]].copy()
    for _ in range(5):
        assignment = np.argmax(rows @ centres.T, axis=1)
        if np.all(assignment == 0) or np.all(assignment == 1):
            assignment[np.argmax(1.0 - rows @ centres[0])] = 1
            assignment[np.argmax(1.0 - rows @ centres[1])] = 0
        centres = np.stack(
            [_normalise(rows[assignment == slot].mean(axis=0, keepdims=True))[0] for slot in (0, 1)]
        )
    return centres.astype(np.float32)


def _multi_proto_state(
    features: np.ndarray, labels: np.ndarray, class_count: int
) -> tuple[np.ndarray, np.ndarray]:
    prototypes = np.stack(
        [_two_prototypes(features[labels == class_index]) for class_index in range(class_count)]
    )
    loo = np.empty((features.shape[0], class_count), dtype=np.float32)
    for row in range(features.shape[0]):
        for class_index in range(class_count):
            members = features[labels == class_index]
            if int(labels[row]) == class_index:
                members = features[(labels == class_index) & (np.arange(features.shape[0]) != row)]
            centres = _two_prototypes(members)
            loo[row, class_index] = float(np.max(features[row] @ centres.T))
    return loo, prototypes


def _top1_class_scores(
    query: np.ndarray, support: np.ndarray, labels: np.ndarray, class_count: int
) -> np.ndarray:
    similarities = query @ support.T
    return np.stack(
        [np.max(similarities[:, labels == class_index], axis=1) for class_index in range(class_count)],
        axis=1,
    ).astype(np.float32)


def _top1_loo_scores(features: np.ndarray, labels: np.ndarray, class_count: int) -> np.ndarray:
    similarities = features @ features.T
    np.fill_diagonal(similarities, -np.inf)
    return np.stack(
        [np.max(similarities[:, labels == class_index], axis=1) for class_index in range(class_count)],
        axis=1,
    ).astype(np.float32)


def _quantize_support_codes(features: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(features, dtype=np.float32)
    scale = np.maximum(np.max(np.abs(values), axis=1) / 127.0, np.finfo(np.float16).tiny)
    scale_fp16 = scale.astype(np.float16)
    quantized = np.clip(np.rint(values / scale_fp16.astype(np.float32)[:, None]), -127, 127).astype(np.int8)
    dequantized = _normalise(quantized.astype(np.float32) * scale_fp16.astype(np.float32)[:, None])
    return quantized, scale_fp16, dequantized


def _top1_cross_loo_scores(
    query: np.ndarray, support: np.ndarray, labels: np.ndarray, class_count: int
) -> np.ndarray:
    similarities = query @ support.T
    np.fill_diagonal(similarities, -np.inf)
    return np.stack(
        [np.max(similarities[:, labels == class_index], axis=1) for class_index in range(class_count)],
        axis=1,
    ).astype(np.float32)


def _blend_class_scores(
    query: np.ndarray,
    support: np.ndarray,
    labels: np.ndarray,
    class_count: int,
    alpha: float,
    *,
    aligned_loo: bool = False,
) -> np.ndarray:
    similarity = query @ support.T
    top_similarity = similarity.copy()
    if aligned_loo:
        np.fill_diagonal(top_similarity, -np.inf)
    rows = []
    for class_index in range(class_count):
        mask = labels == class_index
        top1 = np.max(top_similarity[:, mask], axis=1)
        total = np.sum(similarity[:, mask], axis=1)
        denominator = np.full(query.shape[0], int(mask.sum()), dtype=np.float32)
        if aligned_loo:
            own = labels == class_index
            total[own] -= np.diag(similarity)[own]
            denominator[own] -= 1.0
        mean_score = total / np.maximum(denominator, 1.0)
        rows.append(alpha * top1 + (1.0 - alpha) * mean_score)
    return np.stack(rows, axis=1).astype(np.float32)


def _select_blend_alpha(scene_states: list[dict[str, Any]], old_count: int) -> tuple[float, dict[str, Any]]:
    evaluations = []
    for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
        predictions = []
        truths = []
        scene_metrics = []
        for state in scene_states:
            scores = _blend_class_scores(
                state["support_diag_after"],
                state["support_diag_after_int8_dequant"],
                state["support_labels"],
                int(np.max(state["support_labels"])) + 1,
                alpha,
                aligned_loo=True,
            )
            pred = scores.argmax(axis=1)
            metrics = _metrics(pred, state["support_labels"], old_count)
            predictions.append(pred)
            truths.append(state["support_labels"])
            scene_metrics.append(metrics)
        pooled = _metrics(np.concatenate(predictions), np.concatenate(truths), old_count)
        evaluations.append(
            {
                "alpha": alpha,
                "pooled": pooled,
                "worst_scene_floor": min(row["joint_floor"] for row in scene_metrics),
                "worst_scene_H": min(row["H_old_new"] for row in scene_metrics),
            }
        )
    selected = max(
        evaluations,
        key=lambda row: (
            row["worst_scene_floor"],
            row["pooled"]["joint_floor"],
            row["worst_scene_H"],
            row["pooled"]["H_old_new"],
            -row["alpha"],
        ),
    )
    return float(selected["alpha"]), selected


def _fit_diag_metric(
    features: np.ndarray,
    labels: np.ndarray,
    class_count: int,
    *,
    epochs: int = 20,
    learning_rate: float = 0.05,
    regularization: float = 0.02,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, float]]]:
    """Fit a 256-parameter diagonal metric from registered support LOO only."""
    x = torch.from_numpy(np.asarray(features, dtype=np.float32)).cuda()
    y = torch.from_numpy(np.asarray(labels, dtype=np.int64)).cuda()
    theta = torch.zeros(x.shape[1], dtype=torch.float32, device="cuda", requires_grad=True)
    optimizer = torch.optim.Adam([theta], lr=learning_rate)
    masks = [y == class_index for class_index in range(class_count)]
    trace: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        scale = torch.exp(torch.clamp(theta, -1.0, 1.0))
        transformed = torch.nn.functional.normalize(x * scale[None, :], dim=1)
        similarity = transformed @ transformed.T
        similarity = similarity - torch.eye(similarity.shape[0], device="cuda") * 1e4
        class_scores = torch.stack(
            [torch.max(similarity[:, mask], dim=1).values for mask in masks], dim=1
        )
        loo_ce = torch.nn.functional.cross_entropy(class_scores / 0.07, y)
        prototypes = torch.stack(
            [torch.nn.functional.normalize(transformed[mask].mean(dim=0), dim=0) for mask in masks],
            dim=0,
        )
        prototype_scores = transformed @ prototypes.T
        prototype_ce = torch.nn.functional.cross_entropy(prototype_scores / 0.07, y)
        ce = 0.5 * loo_ce + 0.5 * prototype_ce
        reg = regularization * torch.mean(theta.square())
        loss = ce + reg
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            prediction = class_scores.argmax(dim=1)
            per_class = [
                torch.mean((prediction[mask] == y[mask]).float()).item() for mask in masks
            ]
        trace.append(
            {
                "epoch": float(epoch),
                "loss": float(loss.detach().item()),
                "cross_entropy": float(ce.detach().item()),
                "loo_cross_entropy": float(loo_ce.detach().item()),
                "prototype_cross_entropy": float(prototype_ce.detach().item()),
                "identity_regularization": float(reg.detach().item()),
                "support_loo_accuracy": float((prediction == y).float().mean().item()),
                "support_loo_floor": float(min(per_class)),
                "scale_min": float(torch.exp(torch.clamp(theta, -1.0, 1.0)).min().item()),
                "scale_max": float(torch.exp(torch.clamp(theta, -1.0, 1.0)).max().item()),
            }
        )
    with torch.no_grad():
        scale = torch.exp(torch.clamp(theta, -1.0, 1.0))
        transformed = torch.nn.functional.normalize(x * scale[None, :], dim=1)
    transformed_np = transformed.detach().cpu().numpy().astype(np.float32)
    scale_np = scale.detach().cpu().numpy().astype(np.float32)
    return scale_np, _top1_loo_scores(transformed_np, labels, class_count), trace


def _fit_oldlock_metric(
    features: np.ndarray,
    labels: np.ndarray,
    old_count: int,
    initial_scale: np.ndarray,
    config: dict[str, float],
    *,
    epochs: int = 20,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, float]]]:
    """Stage2-C diagonal metric initialised from the Stage2-B metric."""
    x = torch.from_numpy(np.asarray(features, dtype=np.float32)).cuda()
    y = torch.from_numpy(np.asarray(labels, dtype=np.int64)).cuda()
    class_count = int(np.max(labels)) + 1
    theta = torch.log(
        torch.from_numpy(np.asarray(initial_scale, dtype=np.float32)).cuda().clamp_min(1e-6)
    ).detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([theta], lr=0.03)
    masks = [y == class_index for class_index in range(class_count)]
    old_rows = y < old_count
    with torch.no_grad():
        reference_old = torch.nn.functional.normalize(
            x[old_rows]
            * torch.from_numpy(np.asarray(initial_scale, dtype=np.float32)).cuda()[None, :],
            dim=1,
        )
        reference_pair = reference_old @ reference_old.T
    trace: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        scale = torch.exp(torch.clamp(theta, -1.0, 1.0))
        transformed = torch.nn.functional.normalize(x * scale[None, :], dim=1)
        similarity = transformed @ transformed.T
        similarity = similarity - torch.eye(similarity.shape[0], device="cuda") * 1e4
        class_scores = torch.stack(
            [torch.max(similarity[:, mask], dim=1).values for mask in masks], dim=1
        )
        sample_ce = torch.nn.functional.cross_entropy(class_scores / 0.07, y, reduction="none")
        loo_ce = sample_ce.mean()
        prototypes = torch.stack(
            [torch.nn.functional.normalize(transformed[mask].mean(dim=0), dim=0) for mask in masks],
            dim=0,
        )
        prototype_ce = torch.nn.functional.cross_entropy((transformed @ prototypes.T) / 0.07, y)
        base_ce = 0.5 * loo_ce + 0.5 * prototype_ce
        class_ce = torch.stack([sample_ce[mask].mean() for mask in masks])
        cvar = torch.topk(class_ce, k=min(2, class_count), largest=True).values.mean()
        current_old_pair = transformed[old_rows] @ transformed[old_rows].T
        pair_loss = torch.mean((current_old_pair - reference_pair).square())
        old_indices = torch.nonzero(old_rows, as_tuple=False).squeeze(1)
        old_true = class_scores[old_indices, y[old_rows]]
        max_new = class_scores[old_rows, old_count:].max(dim=1).values
        invasion = torch.relu(max_new - old_true + config["margin"]).mean()
        reg = 0.01 * torch.mean((theta - torch.log(torch.from_numpy(initial_scale).cuda())).square())
        loss = (
            base_ce
            + config["beta"] * cvar
            + config["lambda_pair"] * pair_loss
            + config["lambda_inv"] * invasion
            + reg
        )
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            prediction = class_scores.argmax(dim=1)
            per_class = [
                torch.mean((prediction[mask] == y[mask]).float()).item() for mask in masks
            ]
            old_accuracy = torch.mean((prediction[old_rows] == y[old_rows]).float()).item()
            new_accuracy = torch.mean((prediction[~old_rows] == y[~old_rows]).float()).item()
        trace.append(
            {
                "epoch": float(epoch),
                "loss": float(loss.detach().item()),
                "base_cross_entropy": float(base_ce.detach().item()),
                "loo_cross_entropy": float(loo_ce.detach().item()),
                "prototype_cross_entropy": float(prototype_ce.detach().item()),
                "cvar_top2_class_ce": float(cvar.detach().item()),
                "old_pair_preservation_mse": float(pair_loss.detach().item()),
                "old_invasion_loss": float(invasion.detach().item()),
                "identity_regularization": float(reg.detach().item()),
                "support_loo_accuracy": float((prediction == y).float().mean().item()),
                "support_loo_old_accuracy": float(old_accuracy),
                "support_loo_new_accuracy": float(new_accuracy),
                "support_loo_floor": float(min(per_class)),
            }
        )
    with torch.no_grad():
        scale = torch.exp(torch.clamp(theta, -1.0, 1.0))
        transformed = torch.nn.functional.normalize(x * scale[None, :], dim=1)
    scale_np = scale.detach().cpu().numpy().astype(np.float32)
    transformed_np = transformed.detach().cpu().numpy().astype(np.float32)
    return scale_np, _top1_loo_scores(transformed_np, labels, class_count), trace


def _select_oldlock_config(
    scene_states: list[dict[str, Any]], old_count: int
) -> tuple[dict[str, float], list[dict[str, Any]], dict[str, Any]]:
    grid = [
        {"beta": beta, "lambda_pair": pair, "lambda_inv": inv, "margin": 0.01}
        for beta in (0.2, 0.5)
        for pair in (0.1, 0.5)
        for inv in (0.2, 0.5)
    ]
    evaluations = []
    for config in grid:
        fitted = []
        pooled_pred = []
        pooled_truth = []
        scene_metrics = []
        for state in scene_states:
            scale, loo_scores, trace = _fit_oldlock_metric(
                state["support_fused"],
                state["support_labels"],
                old_count,
                state["diag_before"],
                config,
            )
            pred = loo_scores.argmax(axis=1)
            metrics = _metrics(pred, state["support_labels"], old_count)
            fitted.append({"scale": scale, "loo_scores": loo_scores, "trace": trace})
            pooled_pred.append(pred)
            pooled_truth.append(state["support_labels"])
            scene_metrics.append(metrics)
        pooled = _metrics(np.concatenate(pooled_pred), np.concatenate(pooled_truth), old_count)
        evaluations.append(
            {
                "config": config,
                "fitted": fitted,
                "pooled": pooled,
                "worst_scene_floor": min(row["joint_floor"] for row in scene_metrics),
                "worst_scene_H": min(row["H_old_new"] for row in scene_metrics),
            }
        )
    selected = max(
        evaluations,
        key=lambda row: (
            row["worst_scene_floor"],
            row["pooled"]["joint_floor"],
            row["worst_scene_H"],
            row["pooled"]["H_old_new"],
            -row["config"]["lambda_inv"],
            -row["config"]["lambda_pair"],
            -row["config"]["beta"],
        ),
    )
    summary = {key: value for key, value in selected.items() if key != "fitted"}
    return dict(selected["config"]), selected["fitted"], summary


def _lowrank_transform_np(features: np.ndarray, state: dict[str, np.ndarray]) -> np.ndarray:
    scale = np.exp(np.clip(state["theta"], -1.0, 1.0)).astype(np.float32)
    residual = (features @ state["u"]) @ state["v"].T
    return _normalise(features * scale[None, :] + residual)


def _fit_lowrank_metric(
    features: np.ndarray,
    labels: np.ndarray,
    class_count: int,
    old_count: int,
    config: dict[str, float],
    *,
    initial_state: dict[str, np.ndarray] | None = None,
    epochs: int = 20,
) -> tuple[dict[str, np.ndarray], np.ndarray, list[dict[str, float]]]:
    x = torch.from_numpy(np.asarray(features, dtype=np.float32)).cuda()
    y = torch.from_numpy(np.asarray(labels, dtype=np.int64)).cuda()
    rank = int(config["rank"])
    if initial_state is None:
        rng = np.random.default_rng(20260717 + rank)
        theta_init = np.zeros(features.shape[1], dtype=np.float32)
        u_init = rng.normal(0.0, 0.005, size=(features.shape[1], rank)).astype(np.float32)
        v_init = rng.normal(0.0, 0.005, size=(features.shape[1], rank)).astype(np.float32)
    else:
        theta_init = np.asarray(initial_state["theta"], dtype=np.float32)
        u_init = np.asarray(initial_state["u"], dtype=np.float32)
        v_init = np.asarray(initial_state["v"], dtype=np.float32)
    theta = torch.from_numpy(theta_init).cuda().detach().clone().requires_grad_(True)
    u = torch.from_numpy(u_init).cuda().detach().clone().requires_grad_(True)
    v = torch.from_numpy(v_init).cuda().detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([theta, u, v], lr=0.02)
    masks = [y == class_index for class_index in range(class_count)]
    old_rows = y < old_count
    theta_ref = torch.from_numpy(theta_init).cuda()
    u_ref = torch.from_numpy(u_init).cuda()
    v_ref = torch.from_numpy(v_init).cuda()
    with torch.no_grad():
        ref_scale = torch.exp(torch.clamp(theta_ref, -1.0, 1.0))
        ref_all = torch.nn.functional.normalize(
            x * ref_scale[None, :] + (x @ u_ref) @ v_ref.T, dim=1
        )
        reference_pair = ref_all[old_rows] @ ref_all[old_rows].T
    trace: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        scale = torch.exp(torch.clamp(theta, -1.0, 1.0))
        transformed = torch.nn.functional.normalize(
            x * scale[None, :] + (x @ u) @ v.T, dim=1
        )
        similarity = transformed @ transformed.T
        similarity = similarity - torch.eye(similarity.shape[0], device="cuda") * 1e4
        class_scores = torch.stack(
            [torch.max(similarity[:, mask], dim=1).values for mask in masks], dim=1
        )
        sample_ce = torch.nn.functional.cross_entropy(class_scores / 0.07, y, reduction="none")
        loo_ce = sample_ce.mean()
        prototypes = torch.stack(
            [torch.nn.functional.normalize(transformed[mask].mean(dim=0), dim=0) for mask in masks],
            dim=0,
        )
        prototype_ce = torch.nn.functional.cross_entropy((transformed @ prototypes.T) / 0.07, y)
        base_ce = 0.5 * loo_ce + 0.5 * prototype_ce
        class_ce = torch.stack([sample_ce[mask].mean() for mask in masks])
        cvar = torch.topk(class_ce, k=min(2, class_count), largest=True).values.mean()
        pair_loss = torch.mean(
            ((transformed[old_rows] @ transformed[old_rows].T) - reference_pair).square()
        )
        if class_count > old_count:
            old_indices = torch.nonzero(old_rows, as_tuple=False).squeeze(1)
            old_true = class_scores[old_indices, y[old_rows]]
            max_new = class_scores[old_rows, old_count:].max(dim=1).values
            invasion = torch.relu(max_new - old_true + 0.01).mean()
        else:
            invasion = torch.zeros((), device="cuda")
        residual_reg = config["residual_reg"] * (u.square().mean() + v.square().mean())
        init_reg = 0.01 * (
            (theta - theta_ref).square().mean()
            + (u - u_ref).square().mean()
            + (v - v_ref).square().mean()
        )
        loss = base_ce + 0.2 * cvar + 0.1 * pair_loss + 0.2 * invasion + residual_reg + init_reg
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            prediction = class_scores.argmax(dim=1)
            per_class = [
                torch.mean((prediction[mask] == y[mask]).float()).item() for mask in masks
            ]
        trace.append(
            {
                "epoch": float(epoch),
                "loss": float(loss.detach().item()),
                "base_cross_entropy": float(base_ce.detach().item()),
                "loo_cross_entropy": float(loo_ce.detach().item()),
                "prototype_cross_entropy": float(prototype_ce.detach().item()),
                "cvar_top2_class_ce": float(cvar.detach().item()),
                "old_pair_preservation_mse": float(pair_loss.detach().item()),
                "old_invasion_loss": float(invasion.detach().item()),
                "residual_regularization": float(residual_reg.detach().item()),
                "initial_state_regularization": float(init_reg.detach().item()),
                "support_loo_accuracy": float((prediction == y).float().mean().item()),
                "support_loo_floor": float(min(per_class)),
            }
        )
    state = {
        "theta": theta.detach().cpu().numpy().astype(np.float32),
        "u": u.detach().cpu().numpy().astype(np.float32),
        "v": v.detach().cpu().numpy().astype(np.float32),
    }
    transformed_np = _lowrank_transform_np(features, state)
    return state, _top1_loo_scores(transformed_np, labels, class_count), trace


def _select_lowrank_config(
    scene_states: list[dict[str, Any]], old_count: int
) -> tuple[dict[str, float], list[dict[str, Any]], dict[str, Any]]:
    grid = [
        {"rank": float(rank), "residual_reg": residual_reg}
        for rank in (4, 8)
        for residual_reg in (0.01, 0.05)
    ]
    evaluations = []
    for config in grid:
        fitted = []
        predictions = []
        truths = []
        scene_metrics = []
        for state in scene_states:
            old_mask = state["support_labels"] < old_count
            before_state, _, before_trace = _fit_lowrank_metric(
                state["support_fused"][old_mask],
                state["support_labels"][old_mask],
                old_count,
                old_count,
                config,
            )
            after_state, after_loo, after_trace = _fit_lowrank_metric(
                state["support_fused"],
                state["support_labels"],
                int(np.max(state["support_labels"])) + 1,
                old_count,
                config,
                initial_state=before_state,
            )
            pred = after_loo.argmax(axis=1)
            metrics = _metrics(pred, state["support_labels"], old_count)
            fitted.append(
                {
                    "before_state": before_state,
                    "after_state": after_state,
                    "after_loo": after_loo,
                    "before_trace": before_trace,
                    "after_trace": after_trace,
                }
            )
            predictions.append(pred)
            truths.append(state["support_labels"])
            scene_metrics.append(metrics)
        pooled = _metrics(np.concatenate(predictions), np.concatenate(truths), old_count)
        evaluations.append(
            {
                "config": config,
                "fitted": fitted,
                "pooled": pooled,
                "worst_scene_floor": min(row["joint_floor"] for row in scene_metrics),
                "worst_scene_H": min(row["H_old_new"] for row in scene_metrics),
            }
        )
    selected = max(
        evaluations,
        key=lambda row: (
            row["worst_scene_floor"],
            row["pooled"]["joint_floor"],
            row["worst_scene_H"],
            row["pooled"]["H_old_new"],
            -row["config"]["rank"],
            -row["config"]["residual_reg"],
        ),
    )
    summary = {key: value for key, value in selected.items() if key != "fitted"}
    return dict(selected["config"]), selected["fitted"], summary


def _guard_penalties(
    base_scores: np.ndarray,
    labels: np.ndarray,
    old_count: int,
    class_count: int,
    quantile: float = 0.9,
) -> np.ndarray:
    old_rows = labels < old_count
    old_true = base_scores[old_rows, labels[old_rows]]
    penalties = np.zeros(class_count, dtype=np.float32)
    for class_index in range(old_count, class_count):
        invasion = base_scores[old_rows, class_index] - old_true
        penalties[class_index] = max(0.0, float(np.quantile(invasion, quantile)) + 0.005)
    return penalties


def _rivals(prototypes: np.ndarray, old_count: int) -> np.ndarray:
    out = np.full(prototypes.shape[0], -1, dtype=np.int64)
    for new_index in range(old_count, prototypes.shape[0]):
        out[new_index] = int(np.argmax(prototypes[new_index] @ prototypes[:old_count].T))
    return out


def _new_margin_thresholds(
    base_scores: np.ndarray,
    labels: np.ndarray,
    rivals: np.ndarray,
    old_count: int,
) -> np.ndarray:
    out = np.zeros(base_scores.shape[1], dtype=np.float32)
    for new_index in range(old_count, base_scores.shape[1]):
        rows = labels == new_index
        margins = base_scores[rows, new_index] - base_scores[rows, rivals[new_index]]
        out[new_index] = float(np.quantile(margins, 0.1))
    return np.clip(out, -0.1, 0.1)


def _apply_scores(
    base_scores: np.ndarray,
    *,
    radius_bias: np.ndarray,
    old_count: int,
    new_bias: float,
    guard_penalties: np.ndarray,
    guard_scale: float,
    rivals: np.ndarray,
    rival_thresholds: np.ndarray,
    rival_beta: float,
) -> np.ndarray:
    scores = np.asarray(base_scores, dtype=np.float32).copy()
    scores += radius_bias[None, :]
    if scores.shape[1] > old_count:
        new_slice = slice(old_count, scores.shape[1])
        scores[:, new_slice] -= new_bias
        scores[:, new_slice] -= guard_scale * guard_penalties[new_slice][None, :]
        if rival_beta > 0:
            for new_index in range(old_count, scores.shape[1]):
                rival = int(rivals[new_index])
                margin = scores[:, new_index] - scores[:, rival]
                shortfall = np.maximum(0.0, rival_thresholds[new_index] - margin)
                scores[:, new_index] -= rival_beta * shortfall
    return scores


def _metrics(pred: np.ndarray, truth: np.ndarray, old_count: int) -> dict[str, float]:
    per_class = []
    for class_index in sorted(set(int(v) for v in truth.tolist())):
        mask = truth == class_index
        per_class.append((class_index, float(np.mean(pred[mask] == truth[mask]))))
    old = [value for key, value in per_class if key < old_count]
    new = [value for key, value in per_class if key >= old_count]
    old_acc = float(np.mean(pred[truth < old_count] == truth[truth < old_count]))
    new_acc = (
        float(np.mean(pred[truth >= old_count] == truth[truth >= old_count]))
        if new
        else 0.0
    )
    harmonic = 2 * old_acc * new_acc / max(old_acc + new_acc, 1e-12) if new else old_acc
    return {
        "accuracy": float(np.mean(pred == truth)),
        "old_acc": old_acc,
        "new_acc": new_acc,
        "old_floor": min(old),
        "new_floor": min(new) if new else 0.0,
        "joint_floor": min(old + new),
        "H_old_new": harmonic,
    }


def _candidate_grid(method: str) -> list[dict[str, float]]:
    if method == "L0_identity":
        return [{"gamma": 0.0, "new_bias": 0.0, "guard_scale": 0.0, "rival_beta": 0.0}]
    if method == "L1_radius":
        return [
            {"gamma": g, "new_bias": 0.0, "guard_scale": 0.0, "rival_beta": 0.0}
            for g in (-0.04, -0.02, 0.0, 0.02, 0.04)
        ]
    if method == "L2_old_guard":
        return [
            {"gamma": g, "new_bias": b, "guard_scale": q, "rival_beta": 0.0}
            for g in (-0.02, 0.0, 0.02)
            for b in (0.0, 0.01, 0.02, 0.04, 0.06)
            for q in (0.0, 0.5, 1.0)
        ]
    if method == "L3_sparse_rival":
        return [
            {"gamma": g, "new_bias": b, "guard_scale": q, "rival_beta": beta}
            for g in (-0.02, 0.0, 0.02)
            for b in (0.0, 0.01, 0.02, 0.04)
            for q in (0.0, 0.5, 1.0)
            for beta in (0.25, 0.5, 1.0)
        ]
    if method == "L4_two_proto":
        return [
            {"gamma": g, "new_bias": b, "guard_scale": q, "rival_beta": beta}
            for g in (-0.02, 0.0, 0.02)
            for b in (0.0, 0.01, 0.02, 0.04)
            for q in (0.0, 0.5)
            for beta in (0.0, 0.5)
        ]
    if method == "L5_fixed_iq_fft96_top1":
        return [{"gamma": 0.0, "new_bias": 0.0, "guard_scale": 0.0, "rival_beta": 0.0}]
    if method == "L5q_fixed_iq_fft96_top1_int8":
        return [{"gamma": 0.0, "new_bias": 0.0, "guard_scale": 0.0, "rival_beta": 0.0}]
    if method == "L6_diag_metric_fft96_top1":
        return [{"gamma": 0.0, "new_bias": 0.0, "guard_scale": 0.0, "rival_beta": 0.0}]
    if method == "L6q_diag_metric_fft96_top1_int8":
        return [{"gamma": 0.0, "new_bias": 0.0, "guard_scale": 0.0, "rival_beta": 0.0}]
    if method == "L7q_oldlock_cvar_diag_metric_int8":
        return [{"gamma": 0.0, "new_bias": 0.0, "guard_scale": 0.0, "rival_beta": 0.0}]
    if method == "L8q_diag_metric_blend_int8":
        return [{"gamma": 0.0, "new_bias": 0.0, "guard_scale": 0.0, "rival_beta": 0.0}]
    if method == "M2q_lowrank_metric_int8":
        return [{"gamma": 0.0, "new_bias": 0.0, "guard_scale": 0.0, "rival_beta": 0.0}]
    raise KeyError(method)


def _select_config(method: str, scene_states: list[dict[str, Any]], old_count: int) -> tuple[dict[str, float], dict[str, Any]]:
    evaluations = []
    for config in _candidate_grid(method):
        all_pred = []
        all_truth = []
        scene_metrics = []
        for state in scene_states:
            selection_base = (
                state["multi_loo_scores"]
                if method == "L4_two_proto"
                else state["diag_after_loo_scores"]
                if method in {
                    "L6_diag_metric_fft96_top1",
                    "L6q_diag_metric_fft96_top1_int8",
                }
                else state["fft_int8_top1_loo_scores"]
                if method == "L5q_fixed_iq_fft96_top1_int8"
                else state["fft_top1_loo_scores"]
                if method == "L5_fixed_iq_fft96_top1"
                else state["loo_scores"]
            )
            scores = _apply_scores(
                selection_base,
                radius_bias=_radius_bias(state["radii"], config["gamma"]),
                old_count=old_count,
                new_bias=config["new_bias"],
                guard_penalties=state["guard_penalties"],
                guard_scale=config["guard_scale"],
                rivals=state["rivals"],
                rival_thresholds=state["rival_thresholds"],
                rival_beta=config["rival_beta"],
            )
            pred = scores.argmax(axis=1)
            truth = state["support_labels"]
            all_pred.append(pred)
            all_truth.append(truth)
            scene_metrics.append(_metrics(pred, truth, old_count))
        pooled = _metrics(np.concatenate(all_pred), np.concatenate(all_truth), old_count)
        worst_scene_floor = min(row["joint_floor"] for row in scene_metrics)
        evaluations.append({"config": config, "pooled": pooled, "worst_scene_floor": worst_scene_floor})
    selected = max(
        evaluations,
        key=lambda row: (
            row["worst_scene_floor"],
            row["pooled"]["joint_floor"],
            row["pooled"]["H_old_new"],
            row["pooled"]["accuracy"],
            -row["config"]["rival_beta"],
            -row["config"]["guard_scale"],
            -row["config"]["new_bias"],
            -abs(row["config"]["gamma"]),
        ),
    )
    return dict(selected["config"]), selected


def _extract(runtime: torch.jit.ScriptModule, iq: np.ndarray) -> tuple[np.ndarray, float]:
    rows = torch.from_numpy(np.asarray(iq, dtype=np.float32)).cuda()
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        features, _ = runtime(rows)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return _normalise(features.detach().cpu().numpy()), elapsed


def _classifier_latency(features: np.ndarray, prototypes: np.ndarray, repeats: int = 300) -> tuple[float, float]:
    durations = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        np.argmax(features @ prototypes.T, axis=1)
        durations.append((time.perf_counter_ns() - start) / 1e6 / features.shape[0])
    return float(np.mean(durations)), float(np.quantile(durations, 0.95))


def _state_bytes(method: str, class_count: int, dim: int, new_count: int) -> int:
    if method == "L5_fixed_iq_fft96_top1":
        return class_count * 10 * 256 * 4
    if method == "L5q_fixed_iq_fft96_top1_int8":
        return class_count * 10 * (256 + 2)
    if method == "L6_diag_metric_fft96_top1":
        return class_count * 10 * 256 * 4 + 256 * 4
    if method == "L6q_diag_metric_fft96_top1_int8":
        return class_count * 10 * (256 + 2) + 256 * 2
    if method == "L7q_oldlock_cvar_diag_metric_int8":
        return class_count * 10 * (256 + 2) + 256 * 2
    if method == "L8q_diag_metric_blend_int8":
        return class_count * 10 * (256 + 2) + 256 * 2
    if method == "M2q_lowrank_metric_int8":
        raise RuntimeError("M2 state size depends on selected rank and is computed inline")
    total = class_count * dim * 4 * (2 if method == "L4_two_proto" else 1)
    if method != "L0_identity":
        total += class_count * 4 * 2  # radius and radius bias
    if method in {"L2_old_guard", "L3_sparse_rival", "L4_two_proto"}:
        total += new_count * 4  # classwise guard penalty
    if method in {"L3_sparse_rival", "L4_two_proto"}:
        total += new_count * (8 + 4)  # rival int64 and threshold fp32
    return total


def predict(capsule: Path, output: Path) -> None:
    after_root = capsule / "predictor" / "after"
    enrollment = after_root / "enrollment_only"
    apply_root = after_root / "apply_only_staging"
    manifest = json.loads((enrollment / "package_manifest.json").read_text(encoding="utf-8"))
    old_manifest = json.loads(
        (capsule / "predictor" / "before" / "enrollment_only" / "package_manifest.json").read_text(encoding="utf-8")
    )
    class_count = int(manifest["registered_class_count"])
    old_count = int(old_manifest["registered_class_count"])
    if class_count <= old_count:
        raise RuntimeError("after registry has no new classes")
    runtime_path = enrollment / "sealed_feature_runtime.pt"
    runtime = torch.jit.load(str(runtime_path)).eval()
    # Exclude one-time TorchScript/CUDA/cuDNN materialisation from steady-state latency.
    with np.load(enrollment / f"support_{SCENARIOS[0]}.npz", allow_pickle=False) as warm_file:
        warm_rows = torch.from_numpy(warm_file["support_leo_weak_iq"]).cuda()
    with torch.inference_mode():
        runtime(warm_rows)
        runtime(warm_rows)
    torch.cuda.synchronize()
    single_query_runtime_ms = []
    for _ in range(20):
        torch.cuda.synchronize()
        single_start = time.perf_counter()
        with torch.inference_mode():
            runtime(warm_rows[:1])
        torch.cuda.synchronize()
        single_query_runtime_ms.append((time.perf_counter() - single_start) * 1000)
    scene_states: list[dict[str, Any]] = []
    prediction_rows: dict[str, list[np.ndarray]] = {}
    timings: dict[str, Any] = {}
    loss_traces: dict[str, Any] = {}
    torch.cuda.reset_peak_memory_stats()
    for scenario in SCENARIOS:
        with np.load(enrollment / f"support_{scenario}.npz", allow_pickle=False) as support:
            support_iq = support["support_leo_weak_iq"]
            support_labels = support["support_class_indices"].astype(np.int64)
        with np.load(apply_root / f"query_{scenario}.npz", allow_pickle=False) as query:
            query_iq = query["query_leo_weak_iq"]
            query_tokens = query["query_tokens"].astype(str)
        support_features, support_seconds = _extract(runtime, support_iq)
        query_features, query_seconds = _extract(runtime, query_iq)
        fft_start = time.perf_counter()
        support_fft = spectral_logmag_sketch(support_iq, dim=96)
        support_fused = _normalise(np.concatenate([support_features, 8.0 * support_fft], axis=1))
        support_fft_seconds = time.perf_counter() - fft_start
        fft_start = time.perf_counter()
        query_fft = spectral_logmag_sketch(query_iq, dim=96)
        query_fused = _normalise(np.concatenate([query_features, 8.0 * query_fft], axis=1))
        query_fft_seconds = time.perf_counter() - fft_start
        _, _, support_fused_int8_dequant = _quantize_support_codes(support_fused)
        diag_after, diag_after_loo_scores, diag_after_trace = _fit_diag_metric(
            support_fused, support_labels, class_count
        )
        old_support_mask = support_labels < old_count
        diag_before, _, diag_before_trace = _fit_diag_metric(
            support_fused[old_support_mask],
            support_labels[old_support_mask],
            old_count,
        )
        support_diag_after = _normalise(support_fused * diag_after[None, :])
        support_diag_before = _normalise(support_fused[old_support_mask] * diag_before[None, :])
        query_diag_after = _normalise(query_fused * diag_after[None, :])
        query_diag_before = _normalise(query_fused * diag_before[None, :])
        _, _, support_diag_after_int8_dequant = _quantize_support_codes(support_diag_after)
        _, _, support_diag_before_int8_dequant = _quantize_support_codes(support_diag_before)
        loss_traces[scenario] = {
            "before_old_support_only": diag_before_trace,
            "after_all_registered_support_only": diag_after_trace,
        }
        loo_scores, prototypes, radii = _loo_base(support_features, support_labels, class_count)
        multi_loo_scores, multi_prototypes = _multi_proto_state(
            support_features, support_labels, class_count
        )
        guards = _guard_penalties(loo_scores, support_labels, old_count, class_count)
        rivals = _rivals(prototypes, old_count)
        thresholds = _new_margin_thresholds(loo_scores, support_labels, rivals, old_count)
        scene_states.append(
            {
                "scenario": scenario,
                "support_labels": support_labels,
                "support_features": support_features,
                "query_features": query_features,
                "support_fused": support_fused,
                "support_fused_int8_dequant": support_fused_int8_dequant,
                "query_fused": query_fused,
                "support_diag_after": support_diag_after,
                "support_diag_before": support_diag_before,
                "support_diag_after_int8_dequant": support_diag_after_int8_dequant,
                "support_diag_before_int8_dequant": support_diag_before_int8_dequant,
                "query_diag_after": query_diag_after,
                "query_diag_before": query_diag_before,
                "diag_before": diag_before,
                "query_tokens": query_tokens,
                "loo_scores": loo_scores,
                "multi_loo_scores": multi_loo_scores,
                "prototypes": prototypes,
                "multi_prototypes": multi_prototypes,
                "fft_top1_loo_scores": _top1_loo_scores(
                    support_fused, support_labels, class_count
                ),
                "fft_int8_top1_loo_scores": _top1_cross_loo_scores(
                    support_fused, support_fused_int8_dequant, support_labels, class_count
                ),
                "diag_after_loo_scores": diag_after_loo_scores,
                "radii": radii,
                "guard_penalties": guards,
                "rivals": rivals,
                "rival_thresholds": thresholds,
            }
        )
        timings[scenario] = {
            "support_forward_ms_per_sample": support_seconds * 1000 / support_iq.shape[0],
            "query_forward_ms_per_sample": query_seconds * 1000 / query_iq.shape[0],
            "support_fft96_ms_per_sample": support_fft_seconds * 1000 / support_iq.shape[0],
            "query_fft96_ms_per_sample": query_fft_seconds * 1000 / query_iq.shape[0],
        }
    l7_config, l7_fitted, l7_support_summary = _select_oldlock_config(
        scene_states, old_count
    )
    l8_alpha, l8_support_summary = _select_blend_alpha(scene_states, old_count)
    m2_config, m2_fitted, m2_support_summary = _select_lowrank_config(
        scene_states, old_count
    )
    for state, fitted in zip(scene_states, l7_fitted):
        l7_scale = fitted["scale"]
        support_l7 = _normalise(state["support_fused"] * l7_scale[None, :])
        query_l7 = _normalise(state["query_fused"] * l7_scale[None, :])
        _, _, support_l7_int8 = _quantize_support_codes(support_l7)
        state["l7_after_loo_scores"] = fitted["loo_scores"]
        state["support_l7_int8_dequant"] = support_l7_int8
        state["query_l7"] = query_l7
        loss_traces[state["scenario"]]["L7_after_oldlock_support_only"] = fitted["trace"]
    for state, fitted in zip(scene_states, m2_fitted):
        old_mask = state["support_labels"] < old_count
        support_m2_before = _lowrank_transform_np(
            state["support_fused"][old_mask], fitted["before_state"]
        )
        support_m2_after = _lowrank_transform_np(
            state["support_fused"], fitted["after_state"]
        )
        _, _, state["support_m2_before_int8_dequant"] = _quantize_support_codes(
            support_m2_before
        )
        _, _, state["support_m2_after_int8_dequant"] = _quantize_support_codes(
            support_m2_after
        )
        state["query_m2_before"] = _lowrank_transform_np(
            state["query_fused"], fitted["before_state"]
        )
        state["query_m2_after"] = _lowrank_transform_np(
            state["query_fused"], fitted["after_state"]
        )
        loss_traces[state["scenario"]]["M2_before_old_support_only"] = fitted[
            "before_trace"
        ]
        loss_traces[state["scenario"]]["M2_after_all_registered_support_only"] = fitted[
            "after_trace"
        ]
    selected: dict[str, Any] = {}
    arrays: dict[str, np.ndarray] = {}
    resources: dict[str, Any] = {}
    before_token_sets: dict[str, np.ndarray] = {}
    for scenario in SCENARIOS:
        before_path = capsule / "predictor" / "before" / "apply_only_staging" / f"query_{scenario}.npz"
        with np.load(before_path, allow_pickle=False) as before_query:
            before_token_sets[scenario] = before_query["query_tokens"].astype(str)
    for method in METHODS:
        if method == "L7q_oldlock_cvar_diag_metric_int8":
            config = {
                "gamma": 0.0,
                "new_bias": 0.0,
                "guard_scale": 0.0,
                "rival_beta": 0.0,
                **l7_config,
            }
            support_eval = l7_support_summary
        elif method == "L8q_diag_metric_blend_int8":
            config = {
                "gamma": 0.0,
                "new_bias": 0.0,
                "guard_scale": 0.0,
                "rival_beta": 0.0,
                "alpha": l8_alpha,
            }
            support_eval = l8_support_summary
        elif method == "M2q_lowrank_metric_int8":
            config = {
                "gamma": 0.0,
                "new_bias": 0.0,
                "guard_scale": 0.0,
                "rival_beta": 0.0,
                **m2_config,
            }
            support_eval = m2_support_summary
        else:
            config, support_eval = _select_config(method, scene_states, old_count)
        selected[method] = {"config": config, "support_loo_selection": support_eval}
        after_predictions = []
        after_tokens = []
        after_scenarios = []
        before_predictions = []
        before_tokens = []
        before_scenarios = []
        classifier_latencies = []
        for state in scene_states:
            base_query_scores = (
                _top1_class_scores(
                    state["query_m2_after"],
                    state["support_m2_after_int8_dequant"],
                    state["support_labels"],
                    class_count,
                )
                if method == "M2q_lowrank_metric_int8"
                else
                _blend_class_scores(
                    state["query_diag_after"],
                    state["support_diag_after_int8_dequant"],
                    state["support_labels"],
                    class_count,
                    config["alpha"],
                )
                if method == "L8q_diag_metric_blend_int8"
                else
                _top1_class_scores(
                    state["query_l7"],
                    state["support_l7_int8_dequant"],
                    state["support_labels"],
                    class_count,
                )
                if method == "L7q_oldlock_cvar_diag_metric_int8"
                else
                _top1_class_scores(
                    state["query_diag_after"],
                    state["support_diag_after"],
                    state["support_labels"],
                    class_count,
                )
                if method == "L6_diag_metric_fft96_top1"
                else
                _top1_class_scores(
                    state["query_diag_after"],
                    state["support_diag_after_int8_dequant"],
                    state["support_labels"],
                    class_count,
                )
                if method == "L6q_diag_metric_fft96_top1_int8"
                else
                _top1_class_scores(
                    state["query_fused"],
                    state["support_fused_int8_dequant"],
                    state["support_labels"],
                    class_count,
                )
                if method == "L5q_fixed_iq_fft96_top1_int8"
                else
                _top1_class_scores(
                    state["query_fused"],
                    state["support_fused"],
                    state["support_labels"],
                    class_count,
                )
                if method == "L5_fixed_iq_fft96_top1"
                else
                np.max(
                    np.einsum("nd,ckd->nck", state["query_features"], state["multi_prototypes"]),
                    axis=2,
                )
                if method == "L4_two_proto"
                else state["query_features"] @ state["prototypes"].T
            )
            scores = _apply_scores(
                base_query_scores,
                radius_bias=_radius_bias(state["radii"], config["gamma"]),
                old_count=old_count,
                new_bias=config["new_bias"],
                guard_penalties=state["guard_penalties"],
                guard_scale=config["guard_scale"],
                rivals=state["rivals"],
                rival_thresholds=state["rival_thresholds"],
                rival_beta=config["rival_beta"],
            )
            pred_after = scores.argmax(axis=1).astype(np.int64)
            token_to_row = {token: row for row, token in enumerate(state["query_tokens"].tolist())}
            indices = np.asarray([token_to_row[token] for token in before_token_sets[state["scenario"]]], dtype=np.int64)
            old_proto = state["prototypes"][:old_count]
            old_radii = state["radii"][:old_count]
            before_scores = (
                _top1_class_scores(
                    state["query_m2_before"][indices],
                    state["support_m2_before_int8_dequant"],
                    state["support_labels"][state["support_labels"] < old_count],
                    old_count,
                )
                if method == "M2q_lowrank_metric_int8"
                else
                _blend_class_scores(
                    state["query_diag_before"][indices],
                    state["support_diag_before_int8_dequant"],
                    state["support_labels"][state["support_labels"] < old_count],
                    old_count,
                    config["alpha"],
                )
                if method == "L8q_diag_metric_blend_int8"
                else
                _top1_class_scores(
                    state["query_diag_before"][indices],
                    state["support_diag_before_int8_dequant"],
                    state["support_labels"][state["support_labels"] < old_count],
                    old_count,
                )
                if method == "L7q_oldlock_cvar_diag_metric_int8"
                else
                _top1_class_scores(
                    state["query_diag_before"][indices],
                    state["support_diag_before"],
                    state["support_labels"][state["support_labels"] < old_count],
                    old_count,
                )
                if method == "L6_diag_metric_fft96_top1"
                else
                _top1_class_scores(
                    state["query_diag_before"][indices],
                    state["support_diag_before_int8_dequant"],
                    state["support_labels"][state["support_labels"] < old_count],
                    old_count,
                )
                if method == "L6q_diag_metric_fft96_top1_int8"
                else
                _top1_class_scores(
                    state["query_fused"][indices],
                    state["support_fused_int8_dequant"][state["support_labels"] < old_count],
                    state["support_labels"][state["support_labels"] < old_count],
                    old_count,
                )
                if method == "L5q_fixed_iq_fft96_top1_int8"
                else
                _top1_class_scores(
                    state["query_fused"][indices],
                    state["support_fused"][state["support_labels"] < old_count],
                    state["support_labels"][state["support_labels"] < old_count],
                    old_count,
                )
                if method == "L5_fixed_iq_fft96_top1"
                else
                np.max(
                    np.einsum(
                        "nd,ckd->nck",
                        state["query_features"][indices],
                        state["multi_prototypes"][:old_count],
                    ),
                    axis=2,
                )
                if method == "L4_two_proto"
                else state["query_features"][indices] @ old_proto.T
            )
            before_scores += _radius_bias(old_radii, config["gamma"])[None, :]
            pred_before = before_scores.argmax(axis=1).astype(np.int64)
            after_predictions.append(pred_after)
            after_tokens.append(state["query_tokens"])
            after_scenarios.append(np.full(pred_after.shape[0], state["scenario"]))
            before_predictions.append(pred_before)
            before_tokens.append(before_token_sets[state["scenario"]])
            before_scenarios.append(np.full(pred_before.shape[0], state["scenario"]))
            latency_prototypes = (
                state["support_m2_after_int8_dequant"]
                if method == "M2q_lowrank_metric_int8"
                else
                state["support_diag_after_int8_dequant"]
                if method == "L8q_diag_metric_blend_int8"
                else
                state["support_l7_int8_dequant"]
                if method == "L7q_oldlock_cvar_diag_metric_int8"
                else
                state["support_diag_after"]
                if method == "L6_diag_metric_fft96_top1"
                else
                state["support_diag_after_int8_dequant"]
                if method == "L6q_diag_metric_fft96_top1_int8"
                else
                state["support_fused_int8_dequant"]
                if method == "L5q_fixed_iq_fft96_top1_int8"
                else
                state["support_fused"]
                if method == "L5_fixed_iq_fft96_top1"
                else
                state["multi_prototypes"].reshape(-1, state["multi_prototypes"].shape[-1])
                if method == "L4_two_proto"
                else state["prototypes"]
            )
            latency_features = (
                state["query_m2_after"]
                if method == "M2q_lowrank_metric_int8"
                else
                state["query_diag_after"]
                if method == "L8q_diag_metric_blend_int8"
                else
                state["query_l7"]
                if method == "L7q_oldlock_cvar_diag_metric_int8"
                else
                state["query_diag_after"]
                if method in {
                    "L6_diag_metric_fft96_top1",
                    "L6q_diag_metric_fft96_top1_int8",
                }
                else
                state["query_fused"]
                if method in {
                    "L5_fixed_iq_fft96_top1",
                    "L5q_fixed_iq_fft96_top1_int8",
                }
                else state["query_features"]
            )
            classifier_latencies.append(_classifier_latency(latency_features, latency_prototypes))
        arrays[f"{method}__after_predictions"] = np.concatenate(after_predictions)
        arrays[f"{method}__after_tokens"] = np.concatenate(after_tokens)
        arrays[f"{method}__after_scenarios"] = np.concatenate(after_scenarios)
        arrays[f"{method}__before_predictions"] = np.concatenate(before_predictions)
        arrays[f"{method}__before_tokens"] = np.concatenate(before_tokens)
        arrays[f"{method}__before_scenarios"] = np.concatenate(before_scenarios)
        selected_rank = int(config.get("rank", 0))
        m2_parameter_count = 256 + 512 * selected_rank
        resources[method] = {
            "trainable_parameters": m2_parameter_count
            if method == "M2q_lowrank_metric_int8"
            else 256
            if method in {
                "L6_diag_metric_fft96_top1",
                "L6q_diag_metric_fft96_top1_int8",
                "L7q_oldlock_cvar_diag_metric_int8",
                "L8q_diag_metric_blend_int8",
            }
            else 0,
            "adaptation_epochs": 20
            if method in {"L6_diag_metric_fft96_top1", "L6q_diag_metric_fft96_top1_int8"}
            or method == "L7q_oldlock_cvar_diag_metric_int8"
            or method == "L8q_diag_metric_blend_int8"
            or method == "M2q_lowrank_metric_int8"
            else 0,
            "adaptation_type": (
                "SUPPORT_ONLY_LOWRANK_METRIC_ADAPTATION"
                if method == "M2q_lowrank_metric_int8"
                else "SUPPORT_ONLY_DIAGONAL_METRIC_ADAPTATION"
                if method in {"L6_diag_metric_fft96_top1", "L6q_diag_metric_fft96_top1_int8"}
                or method == "L7q_oldlock_cvar_diag_metric_int8"
                or method == "L8q_diag_metric_blend_int8"
                else "EVAL_ONLY_CLOSED_FORM_ADAPTATION"
            ),
            "persistent_state_bytes_before": old_count * 10 * (256 + 2) + m2_parameter_count * 2
            if method == "M2q_lowrank_metric_int8"
            else _state_bytes(method, old_count, 160, 0),
            "persistent_state_bytes_after": class_count * 10 * (256 + 2) + m2_parameter_count * 2
            if method == "M2q_lowrank_metric_int8"
            else _state_bytes(method, class_count, 160, class_count - old_count),
            "enrollment_prototype_accumulation_MAC": (
                0
                if method in {
                    "L5_fixed_iq_fft96_top1",
                    "L5q_fixed_iq_fft96_top1_int8",
                    "L6_diag_metric_fft96_top1",
                    "L6q_diag_metric_fft96_top1_int8",
                    "L7q_oldlock_cvar_diag_metric_int8",
                    "L8q_diag_metric_blend_int8",
                    "M2q_lowrank_metric_int8",
                }
                else
                class_count * (10 * 10 * 160 + 5 * 10 * 2 * 160)
                if method == "L4_two_proto"
                else class_count * 10 * 160
            ),
            "query_classifier_MAC_before": (
                old_count * 10 * 256
                if method in {
                    "L5_fixed_iq_fft96_top1",
                    "L5q_fixed_iq_fft96_top1_int8",
                    "L6_diag_metric_fft96_top1",
                    "L6q_diag_metric_fft96_top1_int8",
                    "L7q_oldlock_cvar_diag_metric_int8",
                    "L8q_diag_metric_blend_int8",
                    "M2q_lowrank_metric_int8",
                }
                else old_count * 160 * (2 if method == "L4_two_proto" else 1)
            ),
            "query_classifier_MAC_after": (
                class_count * 10 * 256
                if method in {
                    "L5_fixed_iq_fft96_top1",
                    "L5q_fixed_iq_fft96_top1_int8",
                    "L6_diag_metric_fft96_top1",
                    "L6q_diag_metric_fft96_top1_int8",
                    "L7q_oldlock_cvar_diag_metric_int8",
                    "L8q_diag_metric_blend_int8",
                    "M2q_lowrank_metric_int8",
                }
                else class_count * 160 * (2 if method == "L4_two_proto" else 1)
            )
            + ((class_count - old_count) if method in {"L3_sparse_rival", "L4_two_proto"} else 0),
            "fixed_received_iq_fft96_operator": method in {
                "L5_fixed_iq_fft96_top1",
                "L5q_fixed_iq_fft96_top1_int8",
                "L6_diag_metric_fft96_top1",
                "L6q_diag_metric_fft96_top1_int8",
                "L7q_oldlock_cvar_diag_metric_int8",
                "L8q_diag_metric_blend_int8",
                "M2q_lowrank_metric_int8",
            },
            "fft96_state_int8_plan_bytes_after": (
                class_count * 10 * 256
                if method in {
                    "L5_fixed_iq_fft96_top1",
                    "L5q_fixed_iq_fft96_top1_int8",
                    "L6_diag_metric_fft96_top1",
                    "L6q_diag_metric_fft96_top1_int8",
                    "L7q_oldlock_cvar_diag_metric_int8",
                    "L8q_diag_metric_blend_int8",
                    "M2q_lowrank_metric_int8",
                }
                else 0
            ),
            "metric_transform_MAC_per_query": m2_parameter_count
            if method == "M2q_lowrank_metric_int8"
            else 256
            if method in {"L6_diag_metric_fft96_top1", "L6q_diag_metric_fft96_top1_int8"}
            or method == "L7q_oldlock_cvar_diag_metric_int8"
            or method == "L8q_diag_metric_blend_int8"
            else 0,
            "adaptation_similarity_MAC": (
                20 * class_count * 10 * class_count * 10 * 256
                if method in {"L6_diag_metric_fft96_top1", "L6q_diag_metric_fft96_top1_int8"}
                or method == "L7q_oldlock_cvar_diag_metric_int8"
                or method == "L8q_diag_metric_blend_int8"
                else 20
                * (
                    class_count * 10 * m2_parameter_count
                    + class_count * 10 * class_count * 10 * 256
                )
                if method == "M2q_lowrank_metric_int8"
                else 0
            ),
            "classifier_ms_per_sample_mean": float(np.mean([v[0] for v in classifier_latencies])),
            "classifier_ms_per_sample_p95": float(np.max([v[1] for v in classifier_latencies])),
        }
    arrays["schema_json"] = np.asarray(
        json.dumps(
            {
                "schema": "cvs.phase2.d21_capsule_fast_adapt_predictions.v1",
                "truth_or_role_in_predictor_input": False,
                "query_fit": False,
                "query_quota_or_global_assignment": False,
                "all_registered_classes_per_sample": True,
                "methods": METHODS,
            },
            sort_keys=True,
        )
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    np.savez_compressed(output, **arrays)
    loss_trace_path = output.with_suffix(".loss_trace.json")
    _json_dump(
        loss_trace_path,
        {
            "schema": "cvs.phase2.d21_diag_metric_loss_trace.v1",
            "support_only": True,
            "query_access": False,
            "parameter_counts": {
                "diagonal_metric": 256,
                "M2_selected_lowrank_metric": 256 + 512 * int(m2_config["rank"]),
            },
            "epochs": 20,
            "scenarios": loss_traces,
        },
    )
    receipt = {
        "schema": "cvs.phase2.d21_capsule_fast_adapt_predict_receipt.v1",
        "prediction_sha256": _sha256(output),
        "capsule_offline_receipt_sha256": _sha256(capsule / "offline_build_receipt.json"),
        "sealed_runtime_sha256": _sha256(runtime_path),
        "selected_from_support_only": selected,
        "resources": resources,
        "feature_runtime_timings": timings,
        "single_query_runtime_ms_mean": float(np.mean(single_query_runtime_ms)),
        "single_query_runtime_ms_p95": float(np.quantile(single_query_runtime_ms, 0.95)),
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "loss_trace_relative_path": loss_trace_path.name,
        "loss_trace_sha256": _sha256(loss_trace_path),
        "query_truth_opened": False,
        "query_role_oracle_access": False,
        "query_true_batch_class_count_access": False,
        "query_class_quota_access": False,
        "query_batch_global_assignment": False,
    }
    _json_dump(output.with_suffix(".receipt.json"), receipt)


def _class_metrics(pred: np.ndarray, truth_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    truth = np.asarray([int(row["true_class_index"]) for row in truth_rows])
    out: dict[str, dict[str, Any]] = {}
    for class_index in sorted(set(truth.tolist())):
        mask = truth == class_index
        first = truth_rows[int(np.flatnonzero(mask)[0])]
        out[first["true_class_handle"]] = {
            "class_index": class_index,
            "transmitter_label": first["transmitter_label"],
            "count": int(mask.sum()),
            "accuracy": float(np.mean(pred[mask] == truth[mask])),
        }
    return out


def score(prediction: Path, truth_path: Path, output: Path) -> None:
    # This is the first and only point where query truth is opened.
    truth_doc = json.loads(truth_path.read_text(encoding="utf-8"))
    truth_by_token = {row["query_token"]: row for row in truth_doc["rows"]}
    results: dict[str, Any] = {
        "schema": "cvs.phase2.d21_capsule_fast_adapt_score.v1",
        "prediction_sha256": _sha256(prediction),
        "truth_sidecar_sha256": _sha256(truth_path),
        "query_truth_joined_only_after_immutable_predictions": True,
        "scorer_feedback_to_predictor": False,
        "methods": {},
    }
    with np.load(prediction, allow_pickle=False) as data:
        for method in METHODS:
            method_rows = []
            pooled_before_pred = []
            pooled_before_truth = []
            pooled_after_pred = []
            pooled_after_truth = []
            for scenario in SCENARIOS:
                before_mask = data[f"{method}__before_scenarios"] == scenario
                after_mask = data[f"{method}__after_scenarios"] == scenario
                before_tokens = data[f"{method}__before_tokens"][before_mask].astype(str)
                after_tokens = data[f"{method}__after_tokens"][after_mask].astype(str)
                before_truth_rows = [truth_by_token[token] for token in before_tokens]
                after_truth_rows = [truth_by_token[token] for token in after_tokens]
                before_pred = data[f"{method}__before_predictions"][before_mask]
                after_pred = data[f"{method}__after_predictions"][after_mask]
                before_truth = np.asarray([row["true_class_index"] for row in before_truth_rows], dtype=np.int64)
                after_truth = np.asarray([row["true_class_index"] for row in after_truth_rows], dtype=np.int64)
                old_count = len(set(before_truth.tolist()))
                before_old = float(np.mean(before_pred == before_truth))
                after_metrics = _metrics(after_pred, after_truth, old_count)
                old_after_mask = after_truth < old_count
                after_old_class = _class_metrics(after_pred[old_after_mask], [row for row in after_truth_rows if int(row["true_class_index"]) < old_count])
                new_after_class = _class_metrics(after_pred[~old_after_mask], [row for row in after_truth_rows if int(row["true_class_index"]) >= old_count])
                method_rows.append(
                    {
                        "scenario": scenario,
                        "old_acc_before_increment": before_old,
                        "old_acc": after_metrics["old_acc"],
                        "min_old_class_acc": min(row["accuracy"] for row in after_old_class.values()),
                        "seen_new_acc": after_metrics["new_acc"],
                        "min_seen_new_class_acc": min(row["accuracy"] for row in new_after_class.values()),
                        "H_old_new": after_metrics["H_old_new"],
                        "average_forgetting": before_old - after_metrics["old_acc"],
                        "old_per_class": after_old_class,
                        "seen_new_per_class": new_after_class,
                    }
                )
                pooled_before_pred.append(before_pred)
                pooled_before_truth.append(before_truth)
                pooled_after_pred.append(after_pred)
                pooled_after_truth.append(after_truth)
            bpred = np.concatenate(pooled_before_pred)
            btruth = np.concatenate(pooled_before_truth)
            apred = np.concatenate(pooled_after_pred)
            atruth = np.concatenate(pooled_after_truth)
            old_count = len(set(btruth.tolist()))
            am = _metrics(apred, atruth, old_count)
            old_mask = atruth < old_count
            old_rows = [truth_by_token[token] for token in data[f"{method}__after_tokens"].astype(str) if int(truth_by_token[token]["true_class_index"]) < old_count]
            new_rows = [truth_by_token[token] for token in data[f"{method}__after_tokens"].astype(str) if int(truth_by_token[token]["true_class_index"]) >= old_count]
            old_class = _class_metrics(apred[old_mask], old_rows)
            new_class = _class_metrics(apred[~old_mask], new_rows)
            before_old = float(np.mean(bpred == btruth))
            results["methods"][method] = {
                "scenario_rows": method_rows,
                "aggregate": {
                    "old_acc_before_increment": before_old,
                    "old_acc": am["old_acc"],
                    "min_old_class_acc": min(row["accuracy"] for row in old_class.values()),
                    "seen_new_acc": am["new_acc"],
                    "min_seen_new_class_acc": min(row["accuracy"] for row in new_class.values()),
                    "H_old_new": am["H_old_new"],
                    "average_forgetting": before_old - am["old_acc"],
                    "old_per_class": old_class,
                    "seen_new_per_class": new_class,
                },
            }
    _json_dump(output, results)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    pred = sub.add_parser("predict")
    pred.add_argument("--capsule", type=Path, required=True)
    pred.add_argument("--output", type=Path, required=True)
    scorer = sub.add_parser("score")
    scorer.add_argument("--prediction", type=Path, required=True)
    scorer.add_argument("--truth", type=Path, required=True)
    scorer.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "predict":
        predict(args.capsule.resolve(), args.output.resolve())
    else:
        score(args.prediction.resolve(), args.truth.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
