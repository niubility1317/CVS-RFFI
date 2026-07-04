#!/usr/bin/env python3
"""Probe support-only metric transforms before compressed qKNN.

The transform is fitted from target support only. Query labels are used only
for audit metrics. The deployed state stores transform scalars plus quantized
support codes, not raw IQ or raw support samples.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from itertools import combinations, product
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import phase2_confusion_aware_qknn_probe as base
import phase2_metric_adapter_probe as metric
import phase2_qknn_active_support_select as active
import phase2_source_guarded_qknn_sweep as qknn


Split = tuple[np.ndarray, np.ndarray]


def _collect_support(
    old_splits: dict[str, Split],
    new_splits: dict[str, Split],
    old_labels: list[str],
    new_labels: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    support_indices: list[int] = []
    support_labels: list[str] = []
    for label in old_labels:
        support, _query = old_splits[label]
        support_indices.extend(support.tolist())
        support_labels.extend([label] * int(support.size))
    for label in new_labels:
        support, _query = new_splits[label]
        support_indices.extend(support.tolist())
        support_labels.extend([label] * int(support.size))
    return np.asarray(support_indices, dtype=int), np.asarray(support_labels, dtype=object).astype(str)


def _topm_mean(scores: np.ndarray, topm: int) -> np.ndarray:
    k = max(1, min(int(topm), int(scores.shape[1])))
    part = np.partition(scores, kth=scores.shape[1] - k, axis=1)[:, -k:]
    return np.mean(part, axis=1)


def _repel_prototypes(
    prototypes: np.ndarray,
    *,
    repel_lambda: float,
    repel_margin: float,
    repel_steps: int,
    anchor_lambda: float,
) -> np.ndarray:
    anchors = qknn._normalize_rows(prototypes)
    proto = anchors.copy()
    if proto.shape[0] < 2 or float(repel_lambda) <= 0.0 or int(repel_steps) <= 0:
        return proto
    for _step in range(int(repel_steps)):
        sim = proto @ proto.T
        np.fill_diagonal(sim, -np.inf)
        grad = float(anchor_lambda) * (anchors - proto)
        active_pairs = sim > float(repel_margin)
        for index in range(proto.shape[0]):
            close = np.where(active_pairs[index])[0]
            if close.size == 0:
                continue
            excess = (sim[index, close] - float(repel_margin))[:, None]
            grad[index] -= float(repel_lambda) * np.sum(excess * proto[close], axis=0)
        proto = qknn._normalize_rows(proto + grad)
    return proto


def _repelled_class_scores(
    *,
    features: np.ndarray,
    support_indices: np.ndarray,
    support_labels: np.ndarray,
    query_indices: np.ndarray,
    scenarios: np.ndarray,
    class_labels: list[str],
    old_labels: set[str],
    topm: int,
    proto_mix: float,
    radius_norm: float,
    old_bias: float,
    neg_lambda: float,
    neg_threshold: float,
    neg_margin: float,
    mutual_only: bool,
    scenario_aware: bool,
    proto_repel_lambda: float,
    proto_repel_margin: float,
    proto_repel_steps: int,
    proto_repel_anchor: float,
) -> tuple[np.ndarray, dict[str, float], np.ndarray]:
    if scenario_aware:
        query_scenarios = np.asarray(scenarios[query_indices], dtype=object).astype(str)
        support_scenarios = np.asarray(scenarios[support_indices], dtype=object).astype(str)
        out = np.full((query_indices.size, len(class_labels)), -1e9, dtype=np.float64)
        radii: dict[str, float] = {}
        proto_sim = np.zeros((len(class_labels), len(class_labels)), dtype=np.float64)
        for scenario in sorted({str(value) for value in query_scenarios.tolist()}):
            query_mask = query_scenarios == scenario
            support_mask = support_scenarios == scenario
            if int(np.sum(support_mask)) < max(1, int(topm)) or len(set(support_labels[support_mask].tolist())) < 2:
                support_mask = np.ones_like(support_mask, dtype=bool)
            sub_scores, sub_radii, sub_proto_sim = _repelled_class_scores(
                features=features,
                support_indices=support_indices[support_mask],
                support_labels=support_labels[support_mask],
                query_indices=query_indices[query_mask],
                scenarios=scenarios,
                class_labels=class_labels,
                old_labels=old_labels,
                topm=topm,
                proto_mix=proto_mix,
                radius_norm=radius_norm,
                old_bias=old_bias,
                neg_lambda=neg_lambda,
                neg_threshold=neg_threshold,
                neg_margin=neg_margin,
                mutual_only=mutual_only,
                scenario_aware=False,
                proto_repel_lambda=proto_repel_lambda,
                proto_repel_margin=proto_repel_margin,
                proto_repel_steps=proto_repel_steps,
                proto_repel_anchor=proto_repel_anchor,
            )
            out[query_mask] = sub_scores
            radii.update(sub_radii)
            proto_sim = sub_proto_sim
        return out, radii, proto_sim

    query = qknn._normalize_rows(features[query_indices])
    support = qknn._normalize_rows(features[support_indices])
    prototypes: list[np.ndarray] = []
    radii: list[float] = []
    local_scores: list[np.ndarray] = []
    for label in class_labels:
        class_support = support[support_labels == label]
        if class_support.size == 0:
            prototypes.append(np.zeros(features.shape[1], dtype=np.float64))
            radii.append(1.0)
            local_scores.append(np.full(query.shape[0], -1e9, dtype=np.float64))
            continue
        prototype = qknn._normalize_rows(class_support.mean(axis=0, keepdims=True))[0]
        prototypes.append(prototype)
        radius = float(np.mean(1.0 - class_support @ prototype))
        radii.append(radius)
        local = _topm_mean(query @ class_support.T, int(topm))
        if float(radius_norm) != 0.0:
            local = 1.0 - ((1.0 - local) / (max(radius, 1e-4) ** float(radius_norm)))
        local_scores.append(local)

    proto_matrix = qknn._normalize_rows(np.stack(prototypes, axis=0))
    repelled_proto = _repel_prototypes(
        proto_matrix,
        repel_lambda=float(proto_repel_lambda),
        repel_margin=float(proto_repel_margin),
        repel_steps=int(proto_repel_steps),
        anchor_lambda=float(proto_repel_anchor),
    )
    proto_scores = query @ repelled_proto.T
    if float(radius_norm) != 0.0:
        denom = np.power(np.maximum(np.asarray(radii, dtype=np.float64), 1e-4), float(radius_norm))[None, :]
        proto_scores = 1.0 - ((1.0 - proto_scores) / denom)
    score_matrix = (1.0 - float(proto_mix)) * np.stack(local_scores, axis=1) + float(proto_mix) * proto_scores
    for class_index, label in enumerate(class_labels):
        if label in old_labels:
            score_matrix[:, class_index] += float(old_bias)
    proto_sim = repelled_proto @ repelled_proto.T
    if float(neg_lambda) > 0.0:
        penalties = np.zeros_like(score_matrix)
        for class_i in range(len(class_labels)):
            close_mask = proto_sim[class_i] >= float(neg_threshold)
            close_mask[class_i] = False
            if mutual_only:
                close_mask = close_mask & (proto_sim[:, class_i] >= float(neg_threshold))
            if not bool(np.any(close_mask)):
                continue
            other = score_matrix[:, close_mask]
            penalties[:, class_i] = np.maximum(0.0, np.max(other, axis=1) - score_matrix[:, class_i] + float(neg_margin))
        score_matrix = score_matrix - float(neg_lambda) * penalties
    radius_by_label = {label: radii[i] for i, label in enumerate(class_labels)}
    return score_matrix, radius_by_label, proto_sim


def _pairwise_quota_refine(
    pred: np.ndarray,
    scores: np.ndarray,
    *,
    class_labels: list[str],
    proto_sim: np.ndarray,
    similarity_threshold: float,
) -> tuple[np.ndarray, int]:
    if float(similarity_threshold) > 1.0:
        return pred, 0
    refined = np.asarray(pred, dtype=object).copy()
    labels = np.asarray(class_labels, dtype=object)
    changed = 0
    pairs: list[tuple[float, int, int]] = []
    for left in range(len(class_labels)):
        for right in range(left + 1, len(class_labels)):
            sim = float(proto_sim[left, right])
            if sim >= float(similarity_threshold):
                pairs.append((sim, left, right))
    for _sim, left, right in sorted(pairs, reverse=True):
        left_label = labels[left]
        right_label = labels[right]
        pair_mask = (refined == left_label) | (refined == right_label)
        pair_count = int(np.sum(pair_mask))
        if pair_count <= 1:
            continue
        left_quota = int(np.sum(refined[pair_mask] == left_label))
        if left_quota <= 0 or left_quota >= pair_count:
            continue
        pair_indices = np.where(pair_mask)[0]
        delta = scores[pair_indices, left] - scores[pair_indices, right]
        order = np.argsort(-delta)
        new_pair = np.full(pair_count, right_label, dtype=object)
        new_pair[order[:left_quota]] = left_label
        changed += int(np.sum(refined[pair_indices] != new_pair))
        refined[pair_indices] = new_pair
    return refined.astype(str), changed


def _pair_axis_adjust_scores(
    scores: np.ndarray,
    *,
    features: np.ndarray,
    support_indices: np.ndarray,
    support_labels: np.ndarray,
    query_indices: np.ndarray,
    class_labels: list[str],
    proto_sim: np.ndarray,
    similarity_threshold: float,
    weight: float,
    clip: float,
) -> tuple[np.ndarray, int]:
    if float(weight) == 0.0 or float(similarity_threshold) > 1.0:
        return scores, 0
    support = qknn._normalize_rows(features[support_indices])
    query = qknn._normalize_rows(features[query_indices])
    labels = np.asarray(support_labels, dtype=object).astype(str)
    prototypes: list[np.ndarray] = []
    for label in class_labels:
        class_support = support[labels == label]
        if class_support.size == 0:
            prototypes.append(np.zeros(features.shape[1], dtype=np.float64))
        else:
            prototypes.append(qknn._normalize_rows(class_support.mean(axis=0, keepdims=True))[0])
    proto = qknn._normalize_rows(np.stack(prototypes, axis=0))
    adjusted = np.asarray(scores, dtype=np.float64).copy()
    used = 0
    for left in range(len(class_labels)):
        for right in range(left + 1, len(class_labels)):
            if float(proto_sim[left, right]) < float(similarity_threshold):
                continue
            axis = proto[left] - proto[right]
            norm = float(np.linalg.norm(axis))
            if norm < 1e-8:
                continue
            axis = axis / norm
            left_center = float(proto[left] @ axis)
            right_center = float(proto[right] @ axis)
            sep = abs(left_center - right_center)
            if sep < 1e-6:
                continue
            midpoint = 0.5 * (left_center + right_center)
            margin = (query @ axis - midpoint) / max(sep, 1e-6)
            if right_center > left_center:
                margin = -margin
            margin = np.clip(margin, -float(clip), float(clip))
            adjusted[:, left] += float(weight) * margin
            adjusted[:, right] -= float(weight) * margin
            used += 1
    return adjusted, used


def _pair_gaussian_adjust_scores(
    scores: np.ndarray,
    *,
    features: np.ndarray,
    support_indices: np.ndarray,
    support_labels: np.ndarray,
    query_indices: np.ndarray,
    class_labels: list[str],
    proto_sim: np.ndarray,
    similarity_threshold: float,
    weight: float,
    clip: float,
) -> tuple[np.ndarray, int]:
    if float(weight) == 0.0 or float(similarity_threshold) > 1.0:
        return scores, 0
    support = qknn._normalize_rows(features[support_indices])
    query = qknn._normalize_rows(features[query_indices])
    labels = np.asarray(support_labels, dtype=object).astype(str)
    prototypes: list[np.ndarray] = []
    for label in class_labels:
        class_support = support[labels == label]
        if class_support.size == 0:
            prototypes.append(np.zeros(features.shape[1], dtype=np.float64))
        else:
            prototypes.append(qknn._normalize_rows(class_support.mean(axis=0, keepdims=True))[0])
    proto = qknn._normalize_rows(np.stack(prototypes, axis=0))
    adjusted = np.asarray(scores, dtype=np.float64).copy()
    used = 0
    for left in range(len(class_labels)):
        for right in range(left + 1, len(class_labels)):
            if float(proto_sim[left, right]) < float(similarity_threshold):
                continue
            left_support = support[labels == class_labels[left]]
            right_support = support[labels == class_labels[right]]
            if left_support.size == 0 or right_support.size == 0:
                continue
            axis = proto[left] - proto[right]
            norm = float(np.linalg.norm(axis))
            if norm < 1e-8:
                continue
            axis = axis / norm
            left_proj = left_support @ axis
            right_proj = right_support @ axis
            left_mean = float(np.mean(left_proj))
            right_mean = float(np.mean(right_proj))
            left_var = float(np.var(left_proj) + 1e-4)
            right_var = float(np.var(right_proj) + 1e-4)
            query_proj = query @ axis
            left_ll = -0.5 * (((query_proj - left_mean) ** 2) / left_var + np.log(left_var))
            right_ll = -0.5 * (((query_proj - right_mean) ** 2) / right_var + np.log(right_var))
            margin = np.clip(left_ll - right_ll, -float(clip), float(clip))
            adjusted[:, left] += float(weight) * margin
            adjusted[:, right] -= float(weight) * margin
            used += 1
    return adjusted, used


def _pair_fisher_adjust_scores(
    scores: np.ndarray,
    *,
    features: np.ndarray,
    support_indices: np.ndarray,
    support_labels: np.ndarray,
    query_indices: np.ndarray,
    class_labels: list[str],
    proto_sim: np.ndarray,
    similarity_threshold: float,
    weight: float,
    alpha: float,
    clip: float,
) -> tuple[np.ndarray, int]:
    if float(weight) == 0.0 or float(similarity_threshold) > 1.0:
        return scores, 0
    support = qknn._normalize_rows(features[support_indices])
    query = qknn._normalize_rows(features[query_indices])
    labels = np.asarray(support_labels, dtype=object).astype(str)
    adjusted = np.asarray(scores, dtype=np.float64).copy()
    used = 0
    for left in range(len(class_labels)):
        for right in range(left + 1, len(class_labels)):
            if float(proto_sim[left, right]) < float(similarity_threshold):
                continue
            left_support = support[labels == class_labels[left]]
            right_support = support[labels == class_labels[right]]
            if left_support.size == 0 or right_support.size == 0:
                continue
            left_mean_vec = np.mean(left_support, axis=0)
            right_mean_vec = np.mean(right_support, axis=0)
            centered = np.concatenate(
                [left_support - left_mean_vec[None, :], right_support - right_mean_vec[None, :]],
                axis=0,
            )
            cov = (centered.T @ centered) / max(1, centered.shape[0] - 2)
            trace_scale = float(np.trace(cov) / max(1, cov.shape[0]))
            cov = cov + float(alpha) * max(trace_scale, 1e-6) * np.eye(cov.shape[0], dtype=np.float64)
            axis = np.linalg.solve(cov, left_mean_vec - right_mean_vec)
            norm = float(np.linalg.norm(axis))
            if norm < 1e-8:
                continue
            axis = axis / norm
            left_proj = left_support @ axis
            right_proj = right_support @ axis
            left_mean = float(np.mean(left_proj))
            right_mean = float(np.mean(right_proj))
            if left_mean < right_mean:
                axis = -axis
                left_proj = -left_proj
                right_proj = -right_proj
                left_mean = float(np.mean(left_proj))
                right_mean = float(np.mean(right_proj))
            left_var = float(np.var(left_proj) + 1e-4)
            right_var = float(np.var(right_proj) + 1e-4)
            query_proj = query @ axis
            left_ll = -0.5 * (((query_proj - left_mean) ** 2) / left_var + np.log(left_var))
            right_ll = -0.5 * (((query_proj - right_mean) ** 2) / right_var + np.log(right_var))
            margin = np.clip(left_ll - right_ll, -float(clip), float(clip))
            adjusted[:, left] += float(weight) * margin
            adjusted[:, right] -= float(weight) * margin
            used += 1
    return adjusted, used


def _pair_logreg_adjust_scores(
    scores: np.ndarray,
    *,
    features: np.ndarray,
    support_indices: np.ndarray,
    support_labels: np.ndarray,
    query_indices: np.ndarray,
    class_labels: list[str],
    old_labels: set[str],
    proto_sim: np.ndarray,
    similarity_threshold: float,
    weight: float,
    alpha: float,
    clip: float,
    scope: str,
) -> tuple[np.ndarray, int, int]:
    if float(weight) == 0.0 or float(similarity_threshold) > 1.0:
        return scores, 0, 0
    scope_norm = str(scope).strip().lower()
    if scope_norm not in {"all", "old_new"}:
        raise ValueError(f"unsupported pair_logreg_scope: {scope}")
    support = qknn._normalize_rows(features[support_indices])
    query = qknn._normalize_rows(features[query_indices])
    labels = np.asarray(support_labels, dtype=object).astype(str)
    prototypes: list[np.ndarray] = []
    for label in class_labels:
        class_support = support[labels == label]
        if class_support.size == 0:
            prototypes.append(np.zeros(features.shape[1], dtype=np.float64))
        else:
            prototypes.append(qknn._normalize_rows(class_support.mean(axis=0, keepdims=True))[0])
    proto = qknn._normalize_rows(np.stack(prototypes, axis=0))
    adjusted = np.asarray(scores, dtype=np.float64).copy()
    used = 0
    stored_scalars = 0
    for left in range(len(class_labels)):
        for right in range(left + 1, len(class_labels)):
            left_is_old = class_labels[left] in old_labels
            right_is_old = class_labels[right] in old_labels
            if scope_norm == "old_new" and left_is_old == right_is_old:
                continue
            if float(proto_sim[left, right]) < float(similarity_threshold):
                continue
            pair_mask = (labels == class_labels[left]) | (labels == class_labels[right])
            pair_support = support[pair_mask]
            pair_labels = labels[pair_mask]
            if pair_support.shape[0] < 4 or len(set(pair_labels.tolist())) < 2:
                continue
            left_sim = pair_support @ proto[left]
            right_sim = pair_support @ proto[right]
            x = np.stack(
                [
                    left_sim - right_sim,
                    left_sim,
                    right_sim,
                    np.ones_like(left_sim),
                ],
                axis=1,
            )
            y = np.where(pair_labels == class_labels[left], 1.0, -1.0)
            reg = float(alpha) * np.eye(x.shape[1], dtype=np.float64)
            reg[-1, -1] = float(alpha) * 0.01
            try:
                coeff = np.linalg.solve(x.T @ x + reg, x.T @ y)
            except np.linalg.LinAlgError:
                coeff = np.linalg.pinv(x.T @ x + reg) @ x.T @ y
            q_left_sim = query @ proto[left]
            q_right_sim = query @ proto[right]
            qx = np.stack(
                [
                    q_left_sim - q_right_sim,
                    q_left_sim,
                    q_right_sim,
                    np.ones_like(q_left_sim),
                ],
                axis=1,
            )
            margin = np.clip(qx @ coeff, -float(clip), float(clip))
            adjusted[:, left] += float(weight) * margin
            adjusted[:, right] -= float(weight) * margin
            used += 1
            stored_scalars += int(coeff.size)
    return adjusted, int(used), int(stored_scalars)


def _new_old_conflict_bias_scores(
    scores: np.ndarray,
    *,
    features: np.ndarray,
    support_indices: np.ndarray,
    support_labels: np.ndarray,
    class_labels: list[str],
    old_labels: set[str],
    threshold: float,
    weight: float,
) -> tuple[np.ndarray, dict[str, float]]:
    if float(weight) == 0.0 or float(threshold) > 1.0:
        return scores, {}
    support = qknn._normalize_rows(features[support_indices])
    labels = np.asarray(support_labels, dtype=object).astype(str)
    prototypes: list[np.ndarray] = []
    for label in class_labels:
        cls = support[labels == label]
        if cls.size == 0:
            prototypes.append(np.zeros(support.shape[1], dtype=np.float64))
        else:
            prototypes.append(qknn._normalize_rows(cls.mean(axis=0, keepdims=True))[0])
    proto = qknn._normalize_rows(np.stack(prototypes, axis=0))
    old_indices = [index for index, label in enumerate(class_labels) if label in old_labels]
    adjusted = np.asarray(scores, dtype=np.float64).copy()
    bias_by_label: dict[str, float] = {}
    if not old_indices:
        return adjusted, bias_by_label
    for class_index, label in enumerate(class_labels):
        if label in old_labels:
            continue
        max_old_sim = float(np.max(proto[class_index] @ proto[old_indices].T))
        bias = float(weight) * max(0.0, max_old_sim - float(threshold))
        if bias == 0.0:
            continue
        adjusted[:, class_index] += bias
        bias_by_label[label] = bias
    return adjusted, bias_by_label


def _calibrate_score_columns(scores: np.ndarray, mode: str) -> np.ndarray:
    mode = str(mode).strip().lower()
    if mode in {"", "none", "identity"}:
        return scores
    adjusted = np.asarray(scores, dtype=np.float64).copy()
    if mode == "column_center":
        return adjusted - np.mean(adjusted, axis=0, keepdims=True)
    if mode == "column_zscore":
        centered = adjusted - np.mean(adjusted, axis=0, keepdims=True)
        return centered / (np.std(centered, axis=0, keepdims=True) + 1e-6)
    if mode == "column_robust":
        median = np.median(adjusted, axis=0, keepdims=True)
        q75 = np.percentile(adjusted, 75, axis=0, keepdims=True)
        q25 = np.percentile(adjusted, 25, axis=0, keepdims=True)
        return (adjusted - median) / (q75 - q25 + 1e-6)
    if mode == "column_rank":
        ranked = np.empty_like(adjusted, dtype=np.float64)
        if adjusted.shape[0] <= 1:
            return np.zeros_like(adjusted, dtype=np.float64)
        scale = 2.0 / float(adjusted.shape[0] - 1)
        for col in range(adjusted.shape[1]):
            order = np.argsort(adjusted[:, col], kind="mergesort")
            ranks = np.empty(adjusted.shape[0], dtype=np.float64)
            ranks[order] = np.arange(adjusted.shape[0], dtype=np.float64)
            ranked[:, col] = ranks * scale - 1.0
        return ranked
    raise ValueError(f"unsupported score_calibration mode: {mode}")


def _assignment_margin_adjust_scores(scores: np.ndarray, weight: float, clip: float) -> np.ndarray:
    if float(weight) == 0.0:
        return scores
    score_matrix = np.asarray(scores, dtype=np.float64)
    if score_matrix.shape[1] <= 1:
        return score_matrix
    sorted_scores = np.sort(score_matrix, axis=1)
    top1 = sorted_scores[:, -1:]
    top2 = sorted_scores[:, -2:-1]
    max_other = np.where(score_matrix == top1, top2, top1)
    margin = score_matrix - max_other
    if float(clip) > 0.0:
        margin = np.clip(margin, -float(clip), float(clip))
    return score_matrix + float(weight) * margin


def _source_old_guard_adjust_scores(
    scores: np.ndarray,
    *,
    logits: np.ndarray,
    query_indices: np.ndarray,
    old_labels: list[str],
    source_guard_mode: str,
    source_guard_weight: float,
    source_guard_conf_min: float,
    source_guard_margin_min: float,
) -> tuple[np.ndarray, int]:
    mode = str(source_guard_mode).strip().lower()
    if mode in {"", "none"} or float(source_guard_weight) == 0.0:
        return scores, 0
    query_logits = np.asarray(logits[query_indices], dtype=np.float64)
    if query_logits.ndim != 2 or query_logits.shape[1] < len(old_labels):
        raise ValueError("source old guard requires tx_logits columns for old labels")
    query_logits = query_logits[:, : len(old_labels)]
    exp = np.exp(query_logits - np.max(query_logits, axis=1, keepdims=True))
    probs = exp / np.maximum(np.sum(exp, axis=1, keepdims=True), 1e-12)
    old_argmax = np.argmax(probs, axis=1)
    if query_logits.shape[1] > 1:
        top2 = np.partition(query_logits, kth=-2, axis=1)[:, -2:]
        margin = top2[:, 1] - top2[:, 0]
    else:
        margin = np.full(query_logits.shape[0], np.inf, dtype=np.float64)
    confidence = np.max(probs, axis=1)
    guard = (confidence >= float(source_guard_conf_min)) & (margin >= float(source_guard_margin_min))
    adjusted = np.asarray(scores, dtype=np.float64).copy()
    old_count = len(old_labels)
    if mode == "add_old":
        rows = np.where(guard)[0]
        adjusted[rows, old_argmax[rows]] += float(source_guard_weight)
    elif mode == "penalize_new":
        adjusted[guard, old_count:] -= float(source_guard_weight)
    elif mode == "add_old_penalize_new":
        rows = np.where(guard)[0]
        adjusted[rows, old_argmax[rows]] += float(source_guard_weight)
        adjusted[guard, old_count:] -= float(source_guard_weight)
    else:
        raise ValueError(f"unsupported source_guard_mode: {source_guard_mode}")
    return adjusted, int(np.sum(guard))


def _source_proto_anchor_adjust_scores(
    scores: np.ndarray,
    *,
    adapted_features: np.ndarray,
    tx_ids: np.ndarray,
    roles: np.ndarray,
    query_indices: np.ndarray,
    old_labels: list[str],
    source_proto_anchor_mode: str,
    source_proto_anchor_weight: float,
    source_proto_anchor_center: float,
) -> tuple[np.ndarray, int, int]:
    mode = str(source_proto_anchor_mode).strip().lower()
    weight = float(source_proto_anchor_weight)
    if mode in {"", "none"} or weight == 0.0:
        return scores, 0, 0
    prototypes: list[np.ndarray] = []
    for label in old_labels:
        source_idx = np.where((tx_ids == label) & (roles == "source"))[0].astype(int)
        if source_idx.size == 0:
            return scores, 0, 0
        proto = qknn._normalize_rows(adapted_features[source_idx].mean(axis=0, keepdims=True))[0]
        prototypes.append(proto)
    proto_matrix = np.vstack(prototypes)
    query = qknn._normalize_rows(adapted_features[query_indices])
    similarity = query @ proto_matrix.T
    if mode == "add":
        adjustment = similarity
    elif mode == "centered":
        adjustment = similarity - float(source_proto_anchor_center)
    elif mode == "penalize_low":
        adjustment = np.minimum(similarity - float(source_proto_anchor_center), 0.0)
    else:
        raise ValueError(f"unsupported source_proto_anchor_mode: {source_proto_anchor_mode}")
    adjusted = np.asarray(scores, dtype=np.float64).copy()
    adjusted[:, : len(old_labels)] += weight * adjustment
    return adjusted, int(query_indices.size), int(proto_matrix.size)


def _old_new_runnerup_rescue_scores(
    scores: np.ndarray,
    *,
    proto_sim: np.ndarray,
    class_labels: list[str],
    old_labels: set[str],
    similarity_threshold: float,
    margin: float,
    weight: float,
) -> tuple[np.ndarray, int, int]:
    if float(weight) == 0.0 or float(similarity_threshold) > 1.0 or float(margin) <= 0.0:
        return scores, 0, 0
    old_indices = [index for index, label in enumerate(class_labels) if label in old_labels]
    new_indices = [index for index, label in enumerate(class_labels) if label not in old_labels]
    if not old_indices or not new_indices:
        return scores, 0, 0
    adjusted = np.asarray(scores, dtype=np.float64).copy()
    pair_count = 0
    rescue_count = 0
    for old_index in old_indices:
        for new_index in new_indices:
            if float(proto_sim[old_index, new_index]) < float(similarity_threshold):
                continue
            pair_count += 1
            old_score = scores[:, old_index]
            new_score = scores[:, new_index]
            gap = old_score - new_score
            mask = (gap > 0.0) & (gap <= float(margin))
            if not np.any(mask):
                continue
            adjusted[mask, new_index] += float(weight) * (float(margin) - gap[mask])
            rescue_count += int(np.sum(mask))
    return adjusted, int(pair_count), int(rescue_count)


def _bootstrap_proto_scores(
    *,
    features: np.ndarray,
    support_indices: np.ndarray,
    support_labels: np.ndarray,
    query_indices: np.ndarray,
    class_labels: list[str],
    old_labels: set[str],
    topm: int,
    drop: int,
    radius_norm: float,
    old_bias: float,
) -> tuple[np.ndarray, int]:
    """Score queries against support-only leave-subset prototypes.

    This expands each class into virtual prototypes computed from the K-shot
    support set. Deployment stores the derived prototypes, not raw support.
    """
    support = qknn._normalize_rows(features[support_indices])
    query = qknn._normalize_rows(features[query_indices])
    labels = np.asarray(support_labels, dtype=object).astype(str)
    score_columns: list[np.ndarray] = []
    prototype_count = 0
    for label in class_labels:
        class_support = support[labels == label]
        n_support = int(class_support.shape[0])
        if n_support == 0:
            score_columns.append(np.full(query.shape[0], -1e9, dtype=np.float64))
            continue
        drop_count = max(1, min(int(drop), n_support - 1))
        if n_support <= 1:
            protos = qknn._normalize_rows(class_support.mean(axis=0, keepdims=True))
        else:
            proto_rows: list[np.ndarray] = []
            support_sum = np.sum(class_support, axis=0)
            for excluded in combinations(range(n_support), drop_count):
                keep_count = n_support - len(excluded)
                if keep_count <= 0:
                    continue
                excluded_sum = np.sum(class_support[list(excluded)], axis=0)
                proto_rows.append((support_sum - excluded_sum) / float(keep_count))
            protos = qknn._normalize_rows(np.stack(proto_rows, axis=0)) if proto_rows else qknn._normalize_rows(class_support.mean(axis=0, keepdims=True))
        prototype_count += int(protos.shape[0])
        class_prototype = qknn._normalize_rows(class_support.mean(axis=0, keepdims=True))[0]
        radius = float(np.mean(1.0 - class_support @ class_prototype))
        boot = _topm_mean(query @ protos.T, int(topm))
        if float(radius_norm) != 0.0:
            boot = 1.0 - ((1.0 - boot) / (max(radius, 1e-4) ** float(radius_norm)))
        if label in old_labels:
            boot = boot + float(old_bias)
        score_columns.append(boot)
    return np.stack(score_columns, axis=1), prototype_count


def _ridge_head_scores(
    *,
    features: np.ndarray,
    support_indices: np.ndarray,
    support_labels: np.ndarray,
    query_indices: np.ndarray,
    class_labels: list[str],
    alpha: float,
    clip: float,
) -> tuple[np.ndarray, int]:
    support = qknn._normalize_rows(features[support_indices])
    query = qknn._normalize_rows(features[query_indices])
    label_to_index = {label: index for index, label in enumerate(class_labels)}
    y = np.full((support.shape[0], len(class_labels)), -1.0 / max(1, len(class_labels)), dtype=np.float64)
    for row_index, label in enumerate(np.asarray(support_labels, dtype=object).astype(str)):
        if label in label_to_index:
            y[row_index, label_to_index[label]] = 1.0
    support_aug = np.concatenate([support, np.ones((support.shape[0], 1), dtype=np.float64)], axis=1)
    query_aug = np.concatenate([query, np.ones((query.shape[0], 1), dtype=np.float64)], axis=1)
    reg = float(alpha) * np.eye(support_aug.shape[1], dtype=np.float64)
    reg[-1, -1] = float(alpha) * 0.01
    weights = np.linalg.solve(support_aug.T @ support_aug + reg, support_aug.T @ y)
    logits = query_aug @ weights
    logits = logits - np.mean(logits, axis=1, keepdims=True)
    logits = logits / (np.std(logits, axis=1, keepdims=True) + 1e-6)
    logits = np.clip(logits, -float(clip), float(clip))
    return logits, int(weights.size)


def _support_subspace_proto_scores(
    *,
    features: np.ndarray,
    support_indices: np.ndarray,
    support_labels: np.ndarray,
    query_indices: np.ndarray,
    class_labels: list[str],
    rank: int,
    power: float,
    clip: float,
) -> tuple[np.ndarray, int, int]:
    support = qknn._normalize_rows(features[support_indices])
    query = qknn._normalize_rows(features[query_indices])
    labels = np.asarray(support_labels, dtype=object).astype(str)
    prototypes: list[np.ndarray] = []
    valid_mask: list[bool] = []
    for label in class_labels:
        cls = support[labels == label]
        if cls.size == 0:
            prototypes.append(np.zeros(support.shape[1], dtype=np.float64))
            valid_mask.append(False)
            continue
        prototypes.append(cls.mean(axis=0))
        valid_mask.append(True)
    proto = np.stack(prototypes, axis=0)
    centered_proto = proto - proto.mean(axis=0, keepdims=True)
    max_rank = min(int(rank), max(1, centered_proto.shape[0] - 1), centered_proto.shape[1])
    if max_rank <= 0:
        return np.zeros((query.shape[0], len(class_labels)), dtype=np.float64), 0, 0
    _u, singular_values, vh = np.linalg.svd(centered_proto, full_matrices=False)
    basis = vh[:max_rank].T
    query_proj = query @ basis
    proto_proj = proto @ basis
    if float(power) != 0.0 and singular_values.size:
        scale = np.power(np.maximum(singular_values[:max_rank], 1e-6), float(power))
        query_proj = query_proj * scale[None, :]
        proto_proj = proto_proj * scale[None, :]
    query_proj = qknn._normalize_rows(query_proj)
    proto_proj = qknn._normalize_rows(proto_proj)
    scores = query_proj @ proto_proj.T
    for class_index, is_valid in enumerate(valid_mask):
        if not is_valid:
            scores[:, class_index] = -1e9
    scores = scores - np.mean(scores, axis=1, keepdims=True)
    scores = scores / (np.std(scores, axis=1, keepdims=True) + 1e-6)
    scores = np.clip(scores, -float(clip), float(clip))
    stored_scalars = int(basis.size + proto_proj.size)
    return scores, int(max_rank), stored_scalars


def _support_loo_base_scores(
    *,
    features: np.ndarray,
    support_indices: np.ndarray,
    support_labels: np.ndarray,
    scenarios: np.ndarray,
    class_labels: list[str],
    old_labels: set[str],
    topm: int,
    proto_mix: float,
    radius_norm: float,
    old_bias: float,
    neg_lambda: float,
    neg_threshold: float,
    neg_margin: float,
    mutual_only: bool,
    scenario_aware: bool,
) -> np.ndarray:
    labels = np.asarray(support_labels, dtype=object).astype(str)
    rows: list[np.ndarray] = []
    for row_index, query_index in enumerate(np.asarray(support_indices, dtype=int).tolist()):
        keep = np.ones(int(support_indices.size), dtype=bool)
        keep[row_index] = False
        loo_scores, _radii, _proto_sim = base._class_scores(
            features=features,
            support_indices=support_indices[keep],
            support_labels=labels[keep],
            query_indices=np.asarray([query_index], dtype=int),
            scenarios=scenarios,
            class_labels=class_labels,
            old_labels=old_labels,
            topm=int(topm),
            proto_mix=float(proto_mix),
            radius_norm=float(radius_norm),
            old_bias=float(old_bias),
            neg_lambda=float(neg_lambda),
            neg_threshold=float(neg_threshold),
            neg_margin=float(neg_margin),
            mutual_only=bool(mutual_only),
            scenario_aware=bool(scenario_aware),
        )
        rows.append(loo_scores[0])
    return np.stack(rows, axis=0)


def _support_bias_vector(
    *,
    support_scores: np.ndarray,
    support_labels: np.ndarray,
    class_labels: list[str],
    old_labels: list[str],
    new_labels: list[str],
    step: float,
    rounds: int,
) -> tuple[np.ndarray, float, float]:
    labels = np.asarray(support_labels, dtype=object).astype(str)
    label_to_index = {label: index for index, label in enumerate(class_labels)}
    truth_idx = np.asarray([label_to_index[label] for label in labels], dtype=int)
    old_count = int(sum(1 for label in labels.tolist() if label in set(old_labels)))

    def objective(bias: np.ndarray) -> tuple[float, float, float]:
        pred = base._balanced_predict(
            support_scores + bias[None, :],
            old_count=old_count,
            old_labels=old_labels,
            new_labels=new_labels,
        )
        pred_idx = np.asarray([label_to_index[label] for label in pred.tolist()], dtype=int)
        per_class: list[float] = []
        for class_index, label in enumerate(class_labels):
            mask = labels == label
            per_class.append(float(np.mean(pred_idx[mask] == truth_idx[mask])) if bool(np.any(mask)) else 0.0)
        return float(min(per_class)), float(np.mean(per_class)), float(np.mean(pred_idx == truth_idx))

    bias = np.zeros(len(class_labels), dtype=np.float64)
    best = objective(bias)
    for _round in range(max(0, int(rounds))):
        improved = False
        for class_index in range(len(class_labels)):
            local_best = best
            local_bias = bias.copy()
            for delta in (-float(step), 0.0, float(step)):
                candidate = bias.copy()
                candidate[class_index] += delta
                candidate -= np.mean(candidate)
                score = objective(candidate)
                if score > local_best:
                    local_best = score
                    local_bias = candidate
            if local_best > best:
                best = local_best
                bias = local_bias
                improved = True
        if not improved:
            break
    return bias, best[0], best[1]


def _mahalanobis_proto_scores(
    *,
    features: np.ndarray,
    support_indices: np.ndarray,
    support_labels: np.ndarray,
    query_indices: np.ndarray,
    class_labels: list[str],
    alpha: float,
    diag_mix: float,
    clip: float,
) -> tuple[np.ndarray, int]:
    support = qknn._normalize_rows(features[support_indices])
    query = qknn._normalize_rows(features[query_indices])
    labels = np.asarray(support_labels, dtype=object).astype(str)
    prototypes: list[np.ndarray] = []
    residuals: list[np.ndarray] = []
    for label in class_labels:
        cls = support[labels == label]
        if cls.size == 0:
            prototypes.append(np.zeros(support.shape[1], dtype=np.float64))
            continue
        proto = cls.mean(axis=0)
        prototypes.append(proto)
        residuals.append(cls - proto[None, :])
    proto = qknn._normalize_rows(np.stack(prototypes, axis=0))
    if residuals:
        centered = np.concatenate(residuals, axis=0)
    else:
        centered = support - support.mean(axis=0, keepdims=True)
    cov = (centered.T @ centered) / max(1, centered.shape[0] - 1)
    diag_cov = np.diag(np.diag(cov))
    mixed_cov = (1.0 - float(diag_mix)) * cov + float(diag_mix) * diag_cov
    trace_scale = float(np.trace(mixed_cov) / max(1, mixed_cov.shape[0]))
    reg = float(alpha) * max(trace_scale, 1e-6)
    inv_cov = np.linalg.pinv(mixed_cov + reg * np.eye(mixed_cov.shape[0], dtype=np.float64))
    columns: list[np.ndarray] = []
    for class_index in range(len(class_labels)):
        diff = query - proto[class_index][None, :]
        dist = np.sum((diff @ inv_cov) * diff, axis=1)
        columns.append(-dist)
    logits = np.stack(columns, axis=1)
    logits = logits - np.mean(logits, axis=1, keepdims=True)
    logits = logits / (np.std(logits, axis=1, keepdims=True) + 1e-6)
    logits = np.clip(logits, -float(clip), float(clip))
    return logits, int(inv_cov.size)


def _evaluate_metric_qknn(
    *,
    features: np.ndarray,
    aux_features: np.ndarray | None,
    logits: np.ndarray,
    tx_ids: np.ndarray,
    roles: np.ndarray,
    scenarios: np.ndarray,
    old_splits: dict[str, Split],
    new_splits: dict[str, Split],
    old_labels: list[str],
    new_labels: list[str],
    transform_mode: str,
    transform_strength: float,
    topm: int,
    proto_mix: float,
    aux_score_weight: float,
    radius_norm: float,
    old_bias: float,
    neg_lambda: float,
    neg_threshold: float,
    neg_margin: float,
    mutual_only: bool,
    scenario_aware: bool,
    balanced_assignment: bool,
    scenario_balanced_assignment: bool,
    proto_repel_lambda: float,
    proto_repel_margin: float,
    proto_repel_steps: int,
    proto_repel_anchor: float,
    pair_refine_similarity: float,
    pair_axis_similarity: float,
    pair_axis_weight: float,
    pair_axis_clip: float,
    pair_gaussian_similarity: float,
    pair_gaussian_weight: float,
    pair_gaussian_clip: float,
    pair_fisher_similarity: float,
    pair_fisher_weight: float,
    pair_fisher_alpha: float,
    pair_fisher_clip: float,
    pair_logreg_similarity: float,
    pair_logreg_weight: float,
    pair_logreg_alpha: float,
    pair_logreg_clip: float,
    pair_logreg_scope: str,
    new_old_conflict_bias_threshold: float,
    new_old_conflict_bias_weight: float,
    bootstrap_proto_mix: float,
    bootstrap_proto_drop: int,
    bootstrap_proto_topm: int,
    ridge_head_weight: float,
    ridge_head_alpha: float,
    ridge_head_clip: float,
    subspace_proto_weight: float,
    subspace_proto_rank: int,
    subspace_proto_power: float,
    subspace_proto_clip: float,
    support_bias_weight: float,
    support_bias_step: float,
    support_bias_rounds: int,
    mahal_proto_weight: float,
    mahal_proto_alpha: float,
    mahal_proto_diag_mix: float,
    mahal_proto_clip: float,
    score_calibration: str,
    assignment_margin_weight: float,
    assignment_margin_clip: float,
    source_guard_mode: str,
    source_guard_weight: float,
    source_guard_conf_min: float,
    source_guard_margin_min: float,
    source_proto_anchor_mode: str,
    source_proto_anchor_weight: float,
    source_proto_anchor_center: float,
    old_new_runnerup_rescue_similarity: float,
    old_new_runnerup_rescue_margin: float,
    old_new_runnerup_rescue_weight: float,
    old_target: float,
    old_floor: float,
    new_target: float,
    new_floor: float,
    collect_predictions: bool,
) -> dict[str, Any]:
    support_indices, support_labels = _collect_support(old_splits, new_splits, old_labels, new_labels)
    old_query: list[int] = []
    new_query: list[int] = []
    for label in old_labels:
        _support, query = old_splits[label]
        old_query.extend(query.tolist())
    for label in new_labels:
        _support, query = new_splits[label]
        new_query.extend(query.tolist())
    query_indices = np.asarray(old_query + new_query, dtype=int)
    old_count = len(old_query)

    transform = metric._fit_transform(
        features[support_indices],
        support_labels,
        str(transform_mode),
        float(transform_strength),
    )
    adapted = metric._apply_transform(features, transform)
    if float(proto_repel_lambda) > 0.0 and int(proto_repel_steps) > 0:
        scores, radii, proto_sim = _repelled_class_scores(
            features=adapted,
            support_indices=support_indices,
            support_labels=support_labels,
            query_indices=query_indices,
            scenarios=scenarios,
            class_labels=old_labels + new_labels,
            old_labels=set(old_labels),
            topm=int(topm),
            proto_mix=float(proto_mix),
            radius_norm=float(radius_norm),
            old_bias=float(old_bias),
            neg_lambda=float(neg_lambda),
            neg_threshold=float(neg_threshold),
            neg_margin=float(neg_margin),
            mutual_only=bool(mutual_only),
            scenario_aware=bool(scenario_aware),
            proto_repel_lambda=float(proto_repel_lambda),
            proto_repel_margin=float(proto_repel_margin),
            proto_repel_steps=int(proto_repel_steps),
            proto_repel_anchor=float(proto_repel_anchor),
        )
    else:
        scores, radii, proto_sim = base._class_scores(
            features=adapted,
            support_indices=support_indices,
            support_labels=support_labels,
            query_indices=query_indices,
            scenarios=scenarios,
            class_labels=old_labels + new_labels,
            old_labels=set(old_labels),
            topm=int(topm),
            proto_mix=float(proto_mix),
            radius_norm=float(radius_norm),
            old_bias=float(old_bias),
            neg_lambda=float(neg_lambda),
            neg_threshold=float(neg_threshold),
            neg_margin=float(neg_margin),
            mutual_only=bool(mutual_only),
            scenario_aware=bool(scenario_aware),
        )
    if aux_features is not None and float(aux_score_weight) > 0.0:
        aux_transform = metric._fit_transform(
            aux_features[support_indices],
            support_labels,
            str(transform_mode),
            float(transform_strength),
        )
        aux_adapted = metric._apply_transform(aux_features, aux_transform)
        if float(proto_repel_lambda) > 0.0 and int(proto_repel_steps) > 0:
            aux_scores, _aux_radii, _aux_proto_sim = _repelled_class_scores(
                features=aux_adapted,
                support_indices=support_indices,
                support_labels=support_labels,
                query_indices=query_indices,
                scenarios=scenarios,
                class_labels=old_labels + new_labels,
                old_labels=set(old_labels),
                topm=int(topm),
                proto_mix=float(proto_mix),
                radius_norm=float(radius_norm),
                old_bias=float(old_bias),
                neg_lambda=float(neg_lambda),
                neg_threshold=float(neg_threshold),
                neg_margin=float(neg_margin),
                mutual_only=bool(mutual_only),
                scenario_aware=bool(scenario_aware),
                proto_repel_lambda=float(proto_repel_lambda),
                proto_repel_margin=float(proto_repel_margin),
                proto_repel_steps=int(proto_repel_steps),
                proto_repel_anchor=float(proto_repel_anchor),
            )
        else:
            aux_scores, _aux_radii, _aux_proto_sim = base._class_scores(
                features=aux_adapted,
                support_indices=support_indices,
                support_labels=support_labels,
                query_indices=query_indices,
                scenarios=scenarios,
                class_labels=old_labels + new_labels,
                old_labels=set(old_labels),
                topm=int(topm),
                proto_mix=float(proto_mix),
                radius_norm=float(radius_norm),
                old_bias=float(old_bias),
                neg_lambda=float(neg_lambda),
                neg_threshold=float(neg_threshold),
                neg_margin=float(neg_margin),
                mutual_only=bool(mutual_only),
                scenario_aware=bool(scenario_aware),
            )
        scores = (1.0 - float(aux_score_weight)) * scores + float(aux_score_weight) * aux_scores
    scores, pair_axis_count = _pair_axis_adjust_scores(
        scores,
        features=adapted,
        support_indices=support_indices,
        support_labels=support_labels,
        query_indices=query_indices,
        class_labels=old_labels + new_labels,
        proto_sim=proto_sim,
        similarity_threshold=float(pair_axis_similarity),
        weight=float(pair_axis_weight),
        clip=float(pair_axis_clip),
    )
    scores, pair_gaussian_count = _pair_gaussian_adjust_scores(
        scores,
        features=adapted,
        support_indices=support_indices,
        support_labels=support_labels,
        query_indices=query_indices,
        class_labels=old_labels + new_labels,
        proto_sim=proto_sim,
        similarity_threshold=float(pair_gaussian_similarity),
        weight=float(pair_gaussian_weight),
        clip=float(pair_gaussian_clip),
    )
    scores, pair_fisher_count = _pair_fisher_adjust_scores(
        scores,
        features=adapted,
        support_indices=support_indices,
        support_labels=support_labels,
        query_indices=query_indices,
        class_labels=old_labels + new_labels,
        proto_sim=proto_sim,
        similarity_threshold=float(pair_fisher_similarity),
        weight=float(pair_fisher_weight),
        alpha=float(pair_fisher_alpha),
        clip=float(pair_fisher_clip),
    )
    scores, pair_logreg_count, stored_pair_logreg_scalars = _pair_logreg_adjust_scores(
        scores,
        features=adapted,
        support_indices=support_indices,
        support_labels=support_labels,
        query_indices=query_indices,
        class_labels=old_labels + new_labels,
        old_labels=set(old_labels),
        proto_sim=proto_sim,
        similarity_threshold=float(pair_logreg_similarity),
        weight=float(pair_logreg_weight),
        alpha=float(pair_logreg_alpha),
        clip=float(pair_logreg_clip),
        scope=str(pair_logreg_scope),
    )
    scores, new_old_conflict_bias = _new_old_conflict_bias_scores(
        scores,
        features=adapted,
        support_indices=support_indices,
        support_labels=support_labels,
        class_labels=old_labels + new_labels,
        old_labels=set(old_labels),
        threshold=float(new_old_conflict_bias_threshold),
        weight=float(new_old_conflict_bias_weight),
    )
    scores, old_new_runnerup_rescue_pairs, old_new_runnerup_rescue_count = _old_new_runnerup_rescue_scores(
        scores,
        proto_sim=proto_sim,
        class_labels=old_labels + new_labels,
        old_labels=set(old_labels),
        similarity_threshold=float(old_new_runnerup_rescue_similarity),
        margin=float(old_new_runnerup_rescue_margin),
        weight=float(old_new_runnerup_rescue_weight),
    )
    bootstrap_proto_count = 0
    if float(bootstrap_proto_mix) > 0.0:
        bootstrap_scores, bootstrap_proto_count = _bootstrap_proto_scores(
            features=adapted,
            support_indices=support_indices,
            support_labels=support_labels,
            query_indices=query_indices,
            class_labels=old_labels + new_labels,
            old_labels=set(old_labels),
            topm=int(bootstrap_proto_topm),
            drop=int(bootstrap_proto_drop),
            radius_norm=float(radius_norm),
            old_bias=float(old_bias),
        )
        scores = (1.0 - float(bootstrap_proto_mix)) * scores + float(bootstrap_proto_mix) * bootstrap_scores
    stored_ridge_head_scalars = 0
    if float(ridge_head_weight) > 0.0:
        ridge_scores, stored_ridge_head_scalars = _ridge_head_scores(
            features=adapted,
            support_indices=support_indices,
            support_labels=support_labels,
            query_indices=query_indices,
            class_labels=old_labels + new_labels,
            alpha=float(ridge_head_alpha),
            clip=float(ridge_head_clip),
        )
        scores = scores + float(ridge_head_weight) * ridge_scores
    stored_subspace_proto_scalars = 0
    subspace_proto_rank_used = 0
    if float(subspace_proto_weight) > 0.0:
        subspace_scores, subspace_proto_rank_used, stored_subspace_proto_scalars = _support_subspace_proto_scores(
            features=adapted,
            support_indices=support_indices,
            support_labels=support_labels,
            query_indices=query_indices,
            class_labels=old_labels + new_labels,
            rank=int(subspace_proto_rank),
            power=float(subspace_proto_power),
            clip=float(subspace_proto_clip),
        )
        scores = scores + float(subspace_proto_weight) * subspace_scores
    stored_support_bias_scalars = 0
    support_bias_loo_min_acc = 0.0
    support_bias_loo_mean_acc = 0.0
    if float(support_bias_weight) > 0.0 and float(support_bias_step) > 0.0 and int(support_bias_rounds) > 0:
        support_scores = _support_loo_base_scores(
            features=adapted,
            support_indices=support_indices,
            support_labels=support_labels,
            scenarios=scenarios,
            class_labels=old_labels + new_labels,
            old_labels=set(old_labels),
            topm=int(topm),
            proto_mix=float(proto_mix),
            radius_norm=float(radius_norm),
            old_bias=float(old_bias),
            neg_lambda=float(neg_lambda),
            neg_threshold=float(neg_threshold),
            neg_margin=float(neg_margin),
            mutual_only=bool(mutual_only),
            scenario_aware=bool(scenario_aware),
        )
        support_bias, support_bias_loo_min_acc, support_bias_loo_mean_acc = _support_bias_vector(
            support_scores=support_scores,
            support_labels=support_labels,
            class_labels=old_labels + new_labels,
            old_labels=old_labels,
            new_labels=new_labels,
            step=float(support_bias_step),
            rounds=int(support_bias_rounds),
        )
        scores = scores + float(support_bias_weight) * support_bias[None, :]
        stored_support_bias_scalars = int(support_bias.size)
    stored_mahal_proto_scalars = 0
    if float(mahal_proto_weight) > 0.0:
        mahal_scores, stored_mahal_proto_scalars = _mahalanobis_proto_scores(
            features=adapted,
            support_indices=support_indices,
            support_labels=support_labels,
            query_indices=query_indices,
            class_labels=old_labels + new_labels,
            alpha=float(mahal_proto_alpha),
            diag_mix=float(mahal_proto_diag_mix),
            clip=float(mahal_proto_clip),
        )
        scores = scores + float(mahal_proto_weight) * mahal_scores
    scores = _calibrate_score_columns(scores, str(score_calibration))
    scores = _assignment_margin_adjust_scores(
        scores,
        weight=float(assignment_margin_weight),
        clip=float(assignment_margin_clip),
    )
    scores, source_proto_anchor_count, stored_source_proto_anchor_scalars = _source_proto_anchor_adjust_scores(
        scores,
        adapted_features=adapted,
        tx_ids=tx_ids,
        roles=roles,
        query_indices=query_indices,
        old_labels=old_labels,
        source_proto_anchor_mode=str(source_proto_anchor_mode),
        source_proto_anchor_weight=float(source_proto_anchor_weight),
        source_proto_anchor_center=float(source_proto_anchor_center),
    )
    scores, source_guard_count = _source_old_guard_adjust_scores(
        scores,
        logits=logits,
        query_indices=query_indices,
        old_labels=old_labels,
        source_guard_mode=str(source_guard_mode),
        source_guard_weight=float(source_guard_weight),
        source_guard_conf_min=float(source_guard_conf_min),
        source_guard_margin_min=float(source_guard_margin_min),
    )
    if scenario_balanced_assignment:
        pred = base._scenario_balanced_predict(
            scores,
            query_scenarios=scenarios[query_indices],
            query_old_mask=np.arange(query_indices.size) < old_count,
            old_labels=old_labels,
            new_labels=new_labels,
        )
    elif balanced_assignment:
        pred = base._balanced_predict(scores, old_count=old_count, old_labels=old_labels, new_labels=new_labels)
    else:
        pred = base._predict(scores, old_labels + new_labels)
    pred, pair_refine_changed = _pairwise_quota_refine(
        pred,
        scores,
        class_labels=old_labels + new_labels,
        proto_sim=proto_sim,
        similarity_threshold=float(pair_refine_similarity),
    )
    truth = tx_ids[query_indices]
    metrics = base._metrics(
        pred,
        truth,
        old_count=old_count,
        old_labels=old_labels,
        new_labels=new_labels,
        old_target=old_target,
        old_floor=old_floor,
        new_target=new_target,
        new_floor=new_floor,
    )
    row: dict[str, Any] = {
        "transform_mode": str(transform_mode),
        "transform_strength": float(transform_strength),
        "topm": int(topm),
        "proto_mix": float(proto_mix),
        "aux_score_weight": float(aux_score_weight),
        "radius_norm": float(radius_norm),
        "old_bias": float(old_bias),
        "neg_lambda": float(neg_lambda),
        "neg_threshold": float(neg_threshold),
        "neg_margin": float(neg_margin),
        "mutual_only": bool(mutual_only),
        "scenario_aware": bool(scenario_aware),
        "balanced_assignment": bool(balanced_assignment),
        "scenario_balanced_assignment": bool(scenario_balanced_assignment),
        "proto_repel_lambda": float(proto_repel_lambda),
        "proto_repel_margin": float(proto_repel_margin),
        "proto_repel_steps": int(proto_repel_steps),
        "proto_repel_anchor": float(proto_repel_anchor),
        "pair_refine_similarity": float(pair_refine_similarity),
        "pair_refine_changed_predictions": int(pair_refine_changed),
        "pair_axis_similarity": float(pair_axis_similarity),
        "pair_axis_weight": float(pair_axis_weight),
        "pair_axis_clip": float(pair_axis_clip),
        "pair_axis_count": int(pair_axis_count),
        "pair_gaussian_similarity": float(pair_gaussian_similarity),
        "pair_gaussian_weight": float(pair_gaussian_weight),
        "pair_gaussian_clip": float(pair_gaussian_clip),
        "pair_gaussian_count": int(pair_gaussian_count),
        "pair_fisher_similarity": float(pair_fisher_similarity),
        "pair_fisher_weight": float(pair_fisher_weight),
        "pair_fisher_alpha": float(pair_fisher_alpha),
        "pair_fisher_clip": float(pair_fisher_clip),
        "pair_fisher_count": int(pair_fisher_count),
        "pair_logreg_similarity": float(pair_logreg_similarity),
        "pair_logreg_weight": float(pair_logreg_weight),
        "pair_logreg_alpha": float(pair_logreg_alpha),
        "pair_logreg_clip": float(pair_logreg_clip),
        "pair_logreg_scope": str(pair_logreg_scope),
        "pair_logreg_count": int(pair_logreg_count),
        "stored_pair_logreg_scalars": int(stored_pair_logreg_scalars),
        "new_old_conflict_bias_threshold": float(new_old_conflict_bias_threshold),
        "new_old_conflict_bias_weight": float(new_old_conflict_bias_weight),
        "new_old_conflict_bias": new_old_conflict_bias,
        "old_new_runnerup_rescue_similarity": float(old_new_runnerup_rescue_similarity),
        "old_new_runnerup_rescue_margin": float(old_new_runnerup_rescue_margin),
        "old_new_runnerup_rescue_weight": float(old_new_runnerup_rescue_weight),
        "old_new_runnerup_rescue_pairs": int(old_new_runnerup_rescue_pairs),
        "old_new_runnerup_rescue_count": int(old_new_runnerup_rescue_count),
        "bootstrap_proto_mix": float(bootstrap_proto_mix),
        "bootstrap_proto_drop": int(bootstrap_proto_drop),
        "bootstrap_proto_topm": int(bootstrap_proto_topm),
        "stored_bootstrap_prototype_count": int(bootstrap_proto_count),
        "ridge_head_weight": float(ridge_head_weight),
        "ridge_head_alpha": float(ridge_head_alpha),
        "ridge_head_clip": float(ridge_head_clip),
        "stored_ridge_head_scalars": int(stored_ridge_head_scalars),
        "subspace_proto_weight": float(subspace_proto_weight),
        "subspace_proto_rank": int(subspace_proto_rank),
        "subspace_proto_rank_used": int(subspace_proto_rank_used),
        "subspace_proto_power": float(subspace_proto_power),
        "subspace_proto_clip": float(subspace_proto_clip),
        "stored_subspace_proto_scalars": int(stored_subspace_proto_scalars),
        "support_bias_weight": float(support_bias_weight),
        "support_bias_step": float(support_bias_step),
        "support_bias_rounds": int(support_bias_rounds),
        "support_bias_loo_min_acc": float(support_bias_loo_min_acc),
        "support_bias_loo_mean_acc": float(support_bias_loo_mean_acc),
        "stored_support_bias_scalars": int(stored_support_bias_scalars),
        "mahal_proto_weight": float(mahal_proto_weight),
        "mahal_proto_alpha": float(mahal_proto_alpha),
        "mahal_proto_diag_mix": float(mahal_proto_diag_mix),
        "mahal_proto_clip": float(mahal_proto_clip),
        "score_calibration": str(score_calibration),
        "assignment_margin_weight": float(assignment_margin_weight),
        "assignment_margin_clip": float(assignment_margin_clip),
        "source_guard_mode": str(source_guard_mode),
        "source_guard_weight": float(source_guard_weight),
        "source_guard_conf_min": float(source_guard_conf_min),
        "source_guard_margin_min": float(source_guard_margin_min),
        "source_guard_count": int(source_guard_count),
        "source_proto_anchor_mode": str(source_proto_anchor_mode),
        "source_proto_anchor_weight": float(source_proto_anchor_weight),
        "source_proto_anchor_center": float(source_proto_anchor_center),
        "source_proto_anchor_count": int(source_proto_anchor_count),
        "stored_source_proto_anchor_scalars": int(stored_source_proto_anchor_scalars),
        "stored_mahal_proto_scalars": int(stored_mahal_proto_scalars),
        "stored_quantized_support_code_count": int(support_indices.size),
        "stored_raw_support_count": 0,
        "stored_class_prototype_count": int(len(old_labels) + len(new_labels)),
        "stored_transform_scalars": int(2 * features.shape[1]),
        "stored_aux_transform_scalars": int(2 * aux_features.shape[1]) if aux_features is not None and float(aux_score_weight) > 0.0 else 0,
        "transform_scale_min": float(np.min(transform["scale"])),
        "transform_scale_max": float(np.max(transform["scale"])),
        "transform_scale_mean": float(np.mean(transform["scale"])),
        "class_radii": radii,
        "max_offdiag_proto_sim": float(np.max(proto_sim - np.eye(proto_sim.shape[0]) * 2.0)) if proto_sim.size else 0.0,
    }
    row.update({f"query_{key}": value for key, value in metrics.items()})
    row["query_rank_score"] = base._rank(metrics)
    if bool(collect_predictions):
        class_labels = old_labels + new_labels
        class_to_index = {label: index for index, label in enumerate(class_labels)}
        score_order = np.argsort(scores, axis=1)[:, ::-1]
        debug_rows: list[dict[str, Any]] = []
        for local_index, query_index in enumerate(query_indices.tolist()):
            truth_label = str(truth[local_index])
            pred_label = str(pred[local_index])
            top_indices = score_order[local_index, : min(3, score_order.shape[1])]
            true_score = float(scores[local_index, class_to_index[truth_label]])
            pred_score = float(scores[local_index, class_to_index[pred_label]])
            top_score = float(scores[local_index, top_indices[0]])
            debug_rows.append(
                {
                    "local_query_index": int(local_index),
                    "source_index": int(query_index),
                    "role": "old" if local_index < old_count else "new",
                    "truth": truth_label,
                    "pred": pred_label,
                    "correct": bool(pred_label == truth_label),
                    "scenario": str(scenarios[query_index]),
                    "truth_score": true_score,
                    "assigned_pred_score": pred_score,
                    "top_score": top_score,
                    "truth_minus_assigned_pred": float(true_score - pred_score),
                    "truth_minus_raw_top": float(true_score - top_score),
                    "raw_top1": str(class_labels[int(top_indices[0])]),
                    "raw_top1_score": float(scores[local_index, top_indices[0]]),
                    "raw_top2": str(class_labels[int(top_indices[1])]) if len(top_indices) > 1 else "",
                    "raw_top2_score": float(scores[local_index, top_indices[1]]) if len(top_indices) > 1 else "",
                    "raw_top3": str(class_labels[int(top_indices[2])]) if len(top_indices) > 2 else "",
                    "raw_top3_score": float(scores[local_index, top_indices[2]]) if len(top_indices) > 2 else "",
                }
            )
        row["_debug_predictions"] = debug_rows
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature_npz", required=True)
    parser.add_argument("--aux_feature_npz", default="")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--output_predictions_csv", default="")
    parser.add_argument("--old_tx_ids", default="14-10,14-7,20-15,20-19,6-15,8-20")
    parser.add_argument("--new_tx_ids", required=True)
    parser.add_argument("--old_role", default="target_old")
    parser.add_argument("--new_role", default="target_unknown")
    parser.add_argument("--policies", default="stable_first")
    parser.add_argument("--seed_start", type=int, default=421000)
    parser.add_argument("--seed_count", type=int, default=120)
    parser.add_argument("--k_old", type=int, default=10)
    parser.add_argument("--k_new", type=int, default=10)
    parser.add_argument("--query_per_old", type=int, default=70)
    parser.add_argument("--query_per_new", type=int, default=70)
    parser.add_argument("--pool_per_old", type=int, default=10)
    parser.add_argument("--pool_per_new", type=int, default=10)
    parser.add_argument("--transform_modes", default="identity,diag_fisher,diag_whiten_fisher")
    parser.add_argument("--transform_strengths", default="0,0.1,0.25,0.5,0.75,1.0")
    parser.add_argument("--topm_grid", default="4")
    parser.add_argument("--proto_mix_grid", default="0.25")
    parser.add_argument("--aux_score_weight_grid", default="0")
    parser.add_argument("--radius_norm_grid", default="0")
    parser.add_argument("--old_bias_grid", default="0.001")
    parser.add_argument("--neg_lambda_grid", default="0.7")
    parser.add_argument("--neg_threshold_grid", default="0.75")
    parser.add_argument("--neg_margin_grid", default="0.01")
    parser.add_argument("--mutual_only_grid", default="true")
    parser.add_argument("--proto_repel_lambda_grid", default="0")
    parser.add_argument("--proto_repel_margin_grid", default="0.85")
    parser.add_argument("--proto_repel_steps_grid", default="0")
    parser.add_argument("--proto_repel_anchor_grid", default="0.5")
    parser.add_argument("--pair_refine_similarity_grid", default="1.1")
    parser.add_argument("--pair_axis_similarity_grid", default="1.1")
    parser.add_argument("--pair_axis_weight_grid", default="0")
    parser.add_argument("--pair_axis_clip_grid", default="1.0")
    parser.add_argument("--pair_gaussian_similarity_grid", default="1.1")
    parser.add_argument("--pair_gaussian_weight_grid", default="0")
    parser.add_argument("--pair_gaussian_clip_grid", default="5.0")
    parser.add_argument("--pair_fisher_similarity_grid", default="1.1")
    parser.add_argument("--pair_fisher_weight_grid", default="0")
    parser.add_argument("--pair_fisher_alpha_grid", default="1.0")
    parser.add_argument("--pair_fisher_clip_grid", default="5.0")
    parser.add_argument("--pair_logreg_similarity_grid", default="1.1")
    parser.add_argument("--pair_logreg_weight_grid", default="0")
    parser.add_argument("--pair_logreg_alpha_grid", default="1.0")
    parser.add_argument("--pair_logreg_clip_grid", default="2.0")
    parser.add_argument("--pair_logreg_scope_grid", default="all")
    parser.add_argument("--new_old_conflict_bias_threshold_grid", default="1.1")
    parser.add_argument("--new_old_conflict_bias_weight_grid", default="0")
    parser.add_argument("--old_new_runnerup_rescue_similarity_grid", default="1.1")
    parser.add_argument("--old_new_runnerup_rescue_margin_grid", default="0")
    parser.add_argument("--old_new_runnerup_rescue_weight_grid", default="0")
    parser.add_argument("--bootstrap_proto_mix_grid", default="0")
    parser.add_argument("--bootstrap_proto_drop_grid", default="1")
    parser.add_argument("--bootstrap_proto_topm_grid", default="1")
    parser.add_argument("--ridge_head_weight_grid", default="0")
    parser.add_argument("--ridge_head_alpha_grid", default="1.0")
    parser.add_argument("--ridge_head_clip_grid", default="3.0")
    parser.add_argument("--subspace_proto_weight_grid", default="0")
    parser.add_argument("--subspace_proto_rank_grid", default="8")
    parser.add_argument("--subspace_proto_power_grid", default="0")
    parser.add_argument("--subspace_proto_clip_grid", default="3.0")
    parser.add_argument("--support_bias_weight_grid", default="0")
    parser.add_argument("--support_bias_step_grid", default="0.01")
    parser.add_argument("--support_bias_rounds_grid", default="4")
    parser.add_argument("--mahal_proto_weight_grid", default="0")
    parser.add_argument("--mahal_proto_alpha_grid", default="1.0")
    parser.add_argument("--mahal_proto_diag_mix_grid", default="0.5")
    parser.add_argument("--mahal_proto_clip_grid", default="3.0")
    parser.add_argument("--score_calibration_grid", default="none")
    parser.add_argument("--assignment_margin_weight_grid", default="0")
    parser.add_argument("--assignment_margin_clip_grid", default="1.0")
    parser.add_argument("--source_guard_mode_grid", default="none")
    parser.add_argument("--source_guard_weight_grid", default="0")
    parser.add_argument("--source_guard_conf_min_grid", default="0")
    parser.add_argument("--source_guard_margin_min_grid", default="0")
    parser.add_argument("--source_proto_anchor_mode_grid", default="none")
    parser.add_argument("--source_proto_anchor_weight_grid", default="0")
    parser.add_argument("--source_proto_anchor_center_grid", default="0")
    parser.add_argument("--scenario_aware", action="store_true")
    parser.add_argument("--balanced_assignment", action="store_true")
    parser.add_argument("--scenario_balanced_assignment", action="store_true")
    parser.add_argument("--exclude_pool_from_query", action="store_true")
    parser.add_argument("--old_target", type=float, default=0.80)
    parser.add_argument("--old_floor", type=float, default=0.75)
    parser.add_argument("--seen_new_target", type=float, default=0.75)
    parser.add_argument("--seen_new_floor", type=float, default=0.75)
    args = parser.parse_args()

    data = np.load(Path(args.feature_npz), allow_pickle=True)
    features = qknn._normalize_rows(data["features"])
    tx_ids = np.asarray(data["tx_ids"], dtype=object).astype(str)
    roles = np.asarray(data["dataset_role"], dtype=object).astype(str)
    logits = np.asarray(data["tx_logits"], dtype=np.float64)
    scenarios = np.asarray(data["sat_scenarios"], dtype=object).astype(str)
    aux_features = None
    if str(args.aux_feature_npz).strip():
        aux_data = np.load(Path(args.aux_feature_npz), allow_pickle=True)
        aux_features = qknn._normalize_rows(aux_data["features"])
        for key, primary in (("tx_ids", tx_ids), ("dataset_role", roles), ("sat_scenarios", scenarios)):
            aux_values = np.asarray(aux_data[key], dtype=object).astype(str)
            if aux_values.shape != primary.shape or not bool(np.all(aux_values == primary)):
                raise ValueError(f"aux_feature_npz metadata mismatch for {key}: {args.aux_feature_npz}")
    old_labels = qknn._parse_csv(args.old_tx_ids)
    new_labels = qknn._parse_csv(args.new_tx_ids)
    source_probs = active._softmax(logits)
    source_label_to_idx = {label: idx for idx, label in enumerate(old_labels)}
    source_prototypes: dict[str, np.ndarray] = {}
    for label in old_labels:
        source_idx = np.where((tx_ids == label) & (roles == "source"))[0].astype(int)
        if source_idx.size:
            source_prototypes[label] = qknn._normalize_rows(features[source_idx].mean(axis=0, keepdims=True))[0]

    search_grid = list(
        product(
            qknn._parse_csv(args.transform_modes),
            qknn._parse_float_csv(args.transform_strengths),
            qknn._parse_int_csv(args.topm_grid),
            qknn._parse_float_csv(args.proto_mix_grid),
            qknn._parse_float_csv(args.aux_score_weight_grid),
            qknn._parse_float_csv(args.radius_norm_grid),
            qknn._parse_float_csv(args.old_bias_grid),
            qknn._parse_float_csv(args.neg_lambda_grid),
            qknn._parse_float_csv(args.neg_threshold_grid),
            qknn._parse_float_csv(args.neg_margin_grid),
            qknn._parse_csv(args.mutual_only_grid),
            qknn._parse_float_csv(args.proto_repel_lambda_grid),
            qknn._parse_float_csv(args.proto_repel_margin_grid),
            qknn._parse_int_csv(args.proto_repel_steps_grid),
            qknn._parse_float_csv(args.proto_repel_anchor_grid),
            qknn._parse_float_csv(args.pair_refine_similarity_grid),
            qknn._parse_float_csv(args.pair_axis_similarity_grid),
            qknn._parse_float_csv(args.pair_axis_weight_grid),
            qknn._parse_float_csv(args.pair_axis_clip_grid),
            qknn._parse_float_csv(args.pair_gaussian_similarity_grid),
            qknn._parse_float_csv(args.pair_gaussian_weight_grid),
            qknn._parse_float_csv(args.pair_gaussian_clip_grid),
            qknn._parse_float_csv(args.pair_fisher_similarity_grid),
            qknn._parse_float_csv(args.pair_fisher_weight_grid),
            qknn._parse_float_csv(args.pair_fisher_alpha_grid),
            qknn._parse_float_csv(args.pair_fisher_clip_grid),
            qknn._parse_float_csv(args.pair_logreg_similarity_grid),
            qknn._parse_float_csv(args.pair_logreg_weight_grid),
            qknn._parse_float_csv(args.pair_logreg_alpha_grid),
            qknn._parse_float_csv(args.pair_logreg_clip_grid),
            qknn._parse_csv(args.pair_logreg_scope_grid),
            qknn._parse_float_csv(args.new_old_conflict_bias_threshold_grid),
            qknn._parse_float_csv(args.new_old_conflict_bias_weight_grid),
            qknn._parse_float_csv(args.old_new_runnerup_rescue_similarity_grid),
            qknn._parse_float_csv(args.old_new_runnerup_rescue_margin_grid),
            qknn._parse_float_csv(args.old_new_runnerup_rescue_weight_grid),
            qknn._parse_float_csv(args.bootstrap_proto_mix_grid),
            qknn._parse_int_csv(args.bootstrap_proto_drop_grid),
            qknn._parse_int_csv(args.bootstrap_proto_topm_grid),
            qknn._parse_float_csv(args.ridge_head_weight_grid),
            qknn._parse_float_csv(args.ridge_head_alpha_grid),
            qknn._parse_float_csv(args.ridge_head_clip_grid),
            qknn._parse_float_csv(args.subspace_proto_weight_grid),
            qknn._parse_int_csv(args.subspace_proto_rank_grid),
            qknn._parse_float_csv(args.subspace_proto_power_grid),
            qknn._parse_float_csv(args.subspace_proto_clip_grid),
            qknn._parse_float_csv(args.support_bias_weight_grid),
            qknn._parse_float_csv(args.support_bias_step_grid),
            qknn._parse_int_csv(args.support_bias_rounds_grid),
            qknn._parse_float_csv(args.mahal_proto_weight_grid),
            qknn._parse_float_csv(args.mahal_proto_alpha_grid),
            qknn._parse_float_csv(args.mahal_proto_diag_mix_grid),
            qknn._parse_float_csv(args.mahal_proto_clip_grid),
            qknn._parse_csv(args.score_calibration_grid),
            qknn._parse_float_csv(args.assignment_margin_weight_grid),
            qknn._parse_float_csv(args.assignment_margin_clip_grid),
            qknn._parse_csv(args.source_guard_mode_grid),
            qknn._parse_float_csv(args.source_guard_weight_grid),
            qknn._parse_float_csv(args.source_guard_conf_min_grid),
            qknn._parse_float_csv(args.source_guard_margin_min_grid),
            qknn._parse_csv(args.source_proto_anchor_mode_grid),
            qknn._parse_float_csv(args.source_proto_anchor_weight_grid),
            qknn._parse_float_csv(args.source_proto_anchor_center_grid),
        )
    )

    rows: list[dict[str, Any]] = []
    for seed in range(int(args.seed_start), int(args.seed_start) + int(args.seed_count)):
        for policy in qknn._parse_csv(args.policies):
            common = {
                "tx_ids": tx_ids,
                "roles": roles,
                "features": features,
                "scenarios": scenarios,
                "source_probs": source_probs,
                "source_label_to_idx": source_label_to_idx,
                "source_prototypes": source_prototypes,
                "policy": policy,
                "seed": seed,
                "exclude_pool_from_query": bool(args.exclude_pool_from_query),
            }
            old_raw = active._build_active_splits(
                labels=old_labels,
                role=str(args.old_role),
                k=int(args.k_old),
                query_per_class=int(args.query_per_old),
                pool_per_class=int(args.pool_per_old),
                **common,
            )
            new_raw = active._build_active_splits(
                labels=new_labels,
                role=str(args.new_role),
                k=int(args.k_new),
                query_per_class=int(args.query_per_new),
                pool_per_class=int(args.pool_per_new),
                **common,
            )
            if set(old_raw) != set(old_labels) or set(new_raw) != set(new_labels):
                continue
            old_splits = active._as_eval_splits(old_raw)
            new_splits = active._as_eval_splits(new_raw)
            for (
                mode,
                strength,
                topm,
                proto_mix,
                aux_score_weight,
                radius_norm,
                old_bias,
                neg_lambda,
                neg_threshold,
                neg_margin,
                mutual_raw,
                proto_repel_lambda,
                proto_repel_margin,
                proto_repel_steps,
                proto_repel_anchor,
                pair_refine_similarity,
                pair_axis_similarity,
                pair_axis_weight,
                pair_axis_clip,
                pair_gaussian_similarity,
                pair_gaussian_weight,
                pair_gaussian_clip,
                pair_fisher_similarity,
                pair_fisher_weight,
                pair_fisher_alpha,
                pair_fisher_clip,
                pair_logreg_similarity,
                pair_logreg_weight,
                pair_logreg_alpha,
                pair_logreg_clip,
                pair_logreg_scope,
                new_old_conflict_bias_threshold,
                new_old_conflict_bias_weight,
                old_new_runnerup_rescue_similarity,
                old_new_runnerup_rescue_margin,
                old_new_runnerup_rescue_weight,
                bootstrap_proto_mix,
                bootstrap_proto_drop,
                bootstrap_proto_topm,
                ridge_head_weight,
                ridge_head_alpha,
                ridge_head_clip,
                subspace_proto_weight,
                subspace_proto_rank,
                subspace_proto_power,
                subspace_proto_clip,
                support_bias_weight,
                support_bias_step,
                support_bias_rounds,
                mahal_proto_weight,
                mahal_proto_alpha,
                mahal_proto_diag_mix,
                mahal_proto_clip,
                score_calibration,
                assignment_margin_weight,
                assignment_margin_clip,
                source_guard_mode,
                source_guard_weight,
                source_guard_conf_min,
                source_guard_margin_min,
                source_proto_anchor_mode,
                source_proto_anchor_weight,
                source_proto_anchor_center,
            ) in search_grid:
                row = _evaluate_metric_qknn(
                    features=features,
                    aux_features=aux_features,
                    logits=logits,
                    tx_ids=tx_ids,
                    roles=roles,
                    scenarios=scenarios,
                    old_splits=old_splits,
                    new_splits=new_splits,
                    old_labels=old_labels,
                    new_labels=new_labels,
                    transform_mode=mode,
                    transform_strength=float(strength),
                    topm=int(topm),
                    proto_mix=float(proto_mix),
                    aux_score_weight=float(aux_score_weight),
                    radius_norm=float(radius_norm),
                    old_bias=float(old_bias),
                    neg_lambda=float(neg_lambda),
                    neg_threshold=float(neg_threshold),
                    neg_margin=float(neg_margin),
                    mutual_only=str(mutual_raw).lower() == "true",
                    scenario_aware=bool(args.scenario_aware),
                    balanced_assignment=bool(args.balanced_assignment),
                    scenario_balanced_assignment=bool(args.scenario_balanced_assignment),
                    proto_repel_lambda=float(proto_repel_lambda),
                    proto_repel_margin=float(proto_repel_margin),
                    proto_repel_steps=int(proto_repel_steps),
                    proto_repel_anchor=float(proto_repel_anchor),
                    pair_refine_similarity=float(pair_refine_similarity),
                    pair_axis_similarity=float(pair_axis_similarity),
                    pair_axis_weight=float(pair_axis_weight),
                    pair_axis_clip=float(pair_axis_clip),
                    pair_gaussian_similarity=float(pair_gaussian_similarity),
                    pair_gaussian_weight=float(pair_gaussian_weight),
                    pair_gaussian_clip=float(pair_gaussian_clip),
                    pair_fisher_similarity=float(pair_fisher_similarity),
                    pair_fisher_weight=float(pair_fisher_weight),
                    pair_fisher_alpha=float(pair_fisher_alpha),
                    pair_fisher_clip=float(pair_fisher_clip),
                    pair_logreg_similarity=float(pair_logreg_similarity),
                    pair_logreg_weight=float(pair_logreg_weight),
                    pair_logreg_alpha=float(pair_logreg_alpha),
                    pair_logreg_clip=float(pair_logreg_clip),
                    pair_logreg_scope=str(pair_logreg_scope),
                    new_old_conflict_bias_threshold=float(new_old_conflict_bias_threshold),
                    new_old_conflict_bias_weight=float(new_old_conflict_bias_weight),
                    old_new_runnerup_rescue_similarity=float(old_new_runnerup_rescue_similarity),
                    old_new_runnerup_rescue_margin=float(old_new_runnerup_rescue_margin),
                    old_new_runnerup_rescue_weight=float(old_new_runnerup_rescue_weight),
                    bootstrap_proto_mix=float(bootstrap_proto_mix),
                    bootstrap_proto_drop=int(bootstrap_proto_drop),
                    bootstrap_proto_topm=int(bootstrap_proto_topm),
                    ridge_head_weight=float(ridge_head_weight),
                    ridge_head_alpha=float(ridge_head_alpha),
                    ridge_head_clip=float(ridge_head_clip),
                    subspace_proto_weight=float(subspace_proto_weight),
                    subspace_proto_rank=int(subspace_proto_rank),
                    subspace_proto_power=float(subspace_proto_power),
                    subspace_proto_clip=float(subspace_proto_clip),
                    support_bias_weight=float(support_bias_weight),
                    support_bias_step=float(support_bias_step),
                    support_bias_rounds=int(support_bias_rounds),
                    mahal_proto_weight=float(mahal_proto_weight),
                    mahal_proto_alpha=float(mahal_proto_alpha),
                    mahal_proto_diag_mix=float(mahal_proto_diag_mix),
                    mahal_proto_clip=float(mahal_proto_clip),
                    score_calibration=str(score_calibration),
                    assignment_margin_weight=float(assignment_margin_weight),
                    assignment_margin_clip=float(assignment_margin_clip),
                    source_guard_mode=str(source_guard_mode),
                    source_guard_weight=float(source_guard_weight),
                    source_guard_conf_min=float(source_guard_conf_min),
                    source_guard_margin_min=float(source_guard_margin_min),
                    source_proto_anchor_mode=str(source_proto_anchor_mode),
                    source_proto_anchor_weight=float(source_proto_anchor_weight),
                    source_proto_anchor_center=float(source_proto_anchor_center),
                    old_target=float(args.old_target),
                    old_floor=float(args.old_floor),
                    new_target=float(args.seen_new_target),
                    new_floor=float(args.seen_new_floor),
                    collect_predictions=bool(str(args.output_predictions_csv).strip()),
                )
                row["seed"] = int(seed)
                row["support_selection_policy"] = policy
                row["k_old"] = int(args.k_old)
                row["k_new"] = int(args.k_new)
                row["pool_per_old"] = int(args.pool_per_old)
                row["pool_per_new"] = int(args.pool_per_new)
                rows.append(row)

    rows.sort(
        key=lambda row: (
            float(row["query_passes_new_floor75"]),
            float(row["query_min_seen_new_class_acc"]),
            float(row["query_seen_new_acc"]),
            float(row["query_old_acc"]),
            float(row["query_min_old_class_acc"]),
        ),
        reverse=True,
    )
    summary = {
        "diagnostic_scope": "SUPPORT_ONLY_METRIC_QKNN_COMPRESSED_NO_RAW_SUPPORT",
        "feature_npz": str(args.feature_npz),
        "aux_feature_npz": str(args.aux_feature_npz),
        "old_tx_ids": old_labels,
        "new_tx_ids": new_labels,
        "rows_count": len(rows),
        "best": rows[:20],
    }
    output_json = Path(args.output_json)
    output_csv = Path(args.output_csv)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    if str(args.output_predictions_csv).strip() and rows:
        prediction_fields = [
            "local_query_index",
            "source_index",
            "role",
            "truth",
            "pred",
            "correct",
            "scenario",
            "truth_score",
            "assigned_pred_score",
            "top_score",
            "truth_minus_assigned_pred",
            "truth_minus_raw_top",
            "raw_top1",
            "raw_top1_score",
            "raw_top2",
            "raw_top2_score",
            "raw_top3",
            "raw_top3_score",
        ]
        predictions_path = Path(args.output_predictions_csv)
        predictions_path.parent.mkdir(parents=True, exist_ok=True)
        with predictions_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=prediction_fields)
            writer.writeheader()
            for debug_row in rows[0].get("_debug_predictions", []):
                writer.writerow({key: debug_row.get(key) for key in prediction_fields})
    for row in rows:
        row.pop("_debug_predictions", None)
    output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    fields = [
        "seed",
        "support_selection_policy",
        "transform_mode",
        "transform_strength",
        "topm",
        "proto_mix",
        "aux_score_weight",
        "radius_norm",
        "old_bias",
        "neg_lambda",
        "neg_threshold",
        "neg_margin",
        "mutual_only",
        "scenario_aware",
        "balanced_assignment",
        "scenario_balanced_assignment",
        "proto_repel_lambda",
        "proto_repel_margin",
        "proto_repel_steps",
        "proto_repel_anchor",
        "pair_refine_similarity",
        "pair_refine_changed_predictions",
        "pair_axis_similarity",
        "pair_axis_weight",
        "pair_axis_clip",
        "pair_axis_count",
        "pair_gaussian_similarity",
        "pair_gaussian_weight",
        "pair_gaussian_clip",
        "pair_gaussian_count",
        "pair_fisher_similarity",
        "pair_fisher_weight",
        "pair_fisher_alpha",
        "pair_fisher_clip",
        "pair_fisher_count",
        "pair_logreg_similarity",
        "pair_logreg_weight",
        "pair_logreg_alpha",
        "pair_logreg_clip",
        "pair_logreg_scope",
        "pair_logreg_count",
        "stored_pair_logreg_scalars",
        "new_old_conflict_bias_threshold",
        "new_old_conflict_bias_weight",
        "new_old_conflict_bias",
        "old_new_runnerup_rescue_similarity",
        "old_new_runnerup_rescue_margin",
        "old_new_runnerup_rescue_weight",
        "old_new_runnerup_rescue_pairs",
        "old_new_runnerup_rescue_count",
        "bootstrap_proto_mix",
        "bootstrap_proto_drop",
        "bootstrap_proto_topm",
        "stored_bootstrap_prototype_count",
        "ridge_head_weight",
        "ridge_head_alpha",
        "ridge_head_clip",
        "stored_ridge_head_scalars",
        "subspace_proto_weight",
        "subspace_proto_rank",
        "subspace_proto_rank_used",
        "subspace_proto_power",
        "subspace_proto_clip",
        "stored_subspace_proto_scalars",
        "support_bias_weight",
        "support_bias_step",
        "support_bias_rounds",
        "support_bias_loo_min_acc",
        "support_bias_loo_mean_acc",
        "stored_support_bias_scalars",
        "mahal_proto_weight",
        "mahal_proto_alpha",
        "mahal_proto_diag_mix",
        "mahal_proto_clip",
        "score_calibration",
        "assignment_margin_weight",
        "assignment_margin_clip",
        "source_guard_mode",
        "source_guard_weight",
        "source_guard_conf_min",
        "source_guard_margin_min",
        "source_guard_count",
        "source_proto_anchor_mode",
        "source_proto_anchor_weight",
        "source_proto_anchor_center",
        "source_proto_anchor_count",
        "stored_source_proto_anchor_scalars",
        "stored_mahal_proto_scalars",
        "query_old_acc",
        "query_min_old_class_acc",
        "query_seen_new_acc",
        "query_min_seen_new_class_acc",
        "query_passes_new_floor75",
        "query_passes_joint_target",
        "query_per_old_acc",
        "query_per_new_acc",
        "stored_quantized_support_code_count",
        "stored_raw_support_count",
        "stored_class_prototype_count",
        "stored_transform_scalars",
        "stored_aux_transform_scalars",
        "transform_scale_min",
        "transform_scale_max",
        "transform_scale_mean",
        "max_offdiag_proto_sim",
    ]
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            out = {key: row.get(key) for key in fields}
            out["query_per_old_acc"] = json.dumps(row["query_per_old_acc"], ensure_ascii=False, sort_keys=True)
            out["query_per_new_acc"] = json.dumps(row["query_per_new_acc"], ensure_ascii=False, sort_keys=True)
            out["new_old_conflict_bias"] = json.dumps(row.get("new_old_conflict_bias", {}), ensure_ascii=False, sort_keys=True)
            writer.writerow(out)
    print(json.dumps({"best": rows[:5], "output_json": str(output_json)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
