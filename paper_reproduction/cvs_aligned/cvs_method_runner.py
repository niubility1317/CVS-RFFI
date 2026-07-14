"""Publication runner for the proposed CVS Stage2 heads on frozen CVS features.

Stage2-B uses support-only prototype-Gaussian calibration (OPGAC). Stage2-C
uses the documented qKNNV42 int8 support-memory head. Target query labels are
used only after prediction to compute metrics and detailed result tables.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from paper_reproduction.common.config import load_json_config


METHOD_STAGE = {"cvs_opgac": "Stage2-B", "cvs_qknnv42": "Stage2-C"}
SCENARIOS = ("leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak")
EPS = 1.0e-8


def _norm(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), EPS)


def _stable_rank(seed: int, *parts: object) -> int:
    raw = ":".join([str(seed), *(str(value) for value in parts)])
    return int(hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16], 16)


def _load_npz(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    with np.load(path, allow_pickle=False) as data:
        required = {
            "features", "tx_ids", "rx_ids", "day_ids", "eq_ids", "sig_ids",
            "dataset_role", "sat_scenarios",
        }
        missing = sorted(required - set(data.files))
        if missing:
            raise ValueError(f"feature NPZ is missing keys: {missing}")
        arrays = {key: np.asarray(data[key]) for key in data.files if key != "manifest_json"}
        manifest = json.loads(str(data["manifest_json"].item())) if "manifest_json" in data.files else {}
        return arrays, manifest


def _sample_id(arrays: dict[str, np.ndarray], index: int) -> str:
    return "|".join(
        str(arrays[key][index])
        for key in ("dataset_role", "tx_ids", "rx_ids", "day_ids", "eq_ids", "sig_ids")
    )


def _feature_path(config: dict[str, Any], receiver: str, scenario: str) -> Path:
    nested = config.get("feature_npz_by_receiver_scenario", {})
    if nested:
        if receiver not in nested or scenario not in nested[receiver]:
            raise ValueError(f"missing feature cache for receiver={receiver}, scenario={scenario}")
        return Path(nested[receiver][scenario])
    mapping = config.get("feature_npz_by_scenario", {})
    if scenario not in mapping:
        raise ValueError(f"missing feature cache for scenario={scenario}")
    return Path(mapping[scenario])


def _select_split(
    arrays: dict[str, np.ndarray], *, role: str, tx_labels: list[str], receiver: str,
    seed: int, k_shot: int, support_pool_max_k: int, query_per_tx: int,
    scenario: str | None = None,
) -> tuple[list[int], list[int]]:
    roles = arrays["dataset_role"].astype(str)
    tx = arrays["tx_ids"].astype(str)
    rx = arrays["rx_ids"].astype(str)
    support: list[int] = []
    query: list[int] = []
    for label in tx_labels:
        mask = (roles == role) & (tx == label) & (rx == receiver)
        if scenario is not None:
            mask &= arrays["sat_scenarios"].astype(str) == str(scenario)
        candidates = np.where(mask)[0].tolist()
        ordered = sorted(
            (int(i) for i in candidates),
            key=lambda i: _stable_rank(
                seed, role, label, receiver, arrays["day_ids"][i], arrays["eq_ids"][i], arrays["sig_ids"][i]
            ),
        )
        needed = int(support_pool_max_k) + int(query_per_tx)
        if len(ordered) < needed:
            raise ValueError(f"insufficient {role}/{label}/{receiver}: {len(ordered)} < {needed}")
        support.extend(ordered[: int(k_shot)])
        query.extend(ordered[int(support_pool_max_k) : needed])
    return support, query


def _class_scores(features: np.ndarray, labels: np.ndarray, query: np.ndarray) -> tuple[list[str], np.ndarray]:
    classes = sorted(set(labels.astype(str).tolist()))
    prototypes = np.vstack([_norm(features[labels.astype(str) == label].mean(axis=0, keepdims=True))[0] for label in classes])
    return classes, _norm(query) @ prototypes.T


def _opgac_predict(
    source_x: np.ndarray, source_y: np.ndarray, support_x: np.ndarray, support_y: np.ndarray,
    query_x: np.ndarray, *, shrinkage_kappa: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    source_x = _norm(source_x)
    support_x = _norm(support_x)
    query_x = _norm(query_x)
    classes, before_scores = _class_scores(source_x, source_y, query_x)
    before = np.asarray(classes, dtype=object)[np.argmax(before_scores, axis=1)]
    global_var = np.maximum(np.var(source_x, axis=0), 1.0e-4)
    score_columns: list[np.ndarray] = []
    compactness: list[float] = []
    for label in classes:
        src = source_x[source_y.astype(str) == label]
        sup = support_x[support_y.astype(str) == label]
        if sup.size == 0:
            raise ValueError(f"OPGAC missing labeled target support for {label}")
        ground_mean = _norm(src.mean(axis=0, keepdims=True))[0]
        target_mean = _norm(sup.mean(axis=0, keepdims=True))[0]
        alpha = len(sup) / (len(sup) + float(shrinkage_kappa))
        mean = _norm(((1.0 - alpha) * ground_mean + alpha * target_mean)[None, :])[0]
        local_var = np.var(sup, axis=0) if len(sup) > 1 else global_var
        diag = np.maximum((1.0 - alpha) * global_var + alpha * local_var, 1.0e-4)
        diff = query_x - mean[None, :]
        score_columns.append(-0.5 * (np.sum(diff * diff / diag[None, :], axis=1) + np.log(diag).sum()))
        compactness.append(float(np.mean(np.sum((sup - mean[None, :]) ** 2 / diag[None, :], axis=1))))
    scores = np.stack(score_columns, axis=1)
    predicted = np.asarray(classes, dtype=object)[np.argmax(scores, axis=1)]
    return predicted, before, {
        "adaptation_objective": "opgac_support_only_prototype_gaussian_calibration",
        "support_only": True,
        "query_update_forbidden": True,
        "loss": float(np.mean(compactness)),
        "class_count": len(classes),
    }


def _labelprop(
    support: np.ndarray, support_y: np.ndarray, query: np.ndarray, classes: list[str],
    *, neighbors: int = 10, alpha: float = 0.76, temperature: float = 0.05, rounds: int = 8,
) -> np.ndarray:
    x = _norm(np.vstack([support, query]))
    n_support = len(support)
    similarity = x @ x.T
    np.fill_diagonal(similarity, -np.inf)
    k = min(int(neighbors), max(1, len(x) - 1))
    positions = np.argpartition(-similarity, kth=k - 1, axis=1)[:, :k]
    w = np.zeros_like(similarity)
    rows = np.arange(len(x))[:, None]
    logits = similarity[rows, positions] / float(temperature)
    logits -= np.max(logits, axis=1, keepdims=True)
    weights = np.exp(logits)
    weights /= np.maximum(weights.sum(axis=1, keepdims=True), EPS)
    w[rows, positions] = weights
    class_to_i = {label: i for i, label in enumerate(classes)}
    y = np.zeros((len(x), len(classes)), dtype=np.float64)
    for row, label in enumerate(support_y.astype(str).tolist()):
        y[row, class_to_i[label]] = 1.0
    f = y.copy()
    for _ in range(int(rounds)):
        f = float(alpha) * (w @ f) + (1.0 - float(alpha)) * y
        f[:n_support] = y[:n_support]
    q = f[n_support:]
    return np.clip((q - q.mean(axis=1, keepdims=True)) / np.maximum(q.std(axis=1, keepdims=True), 1.0e-6), -2.0, 2.0)


def _fit_diag_whiten_fisher(
    support: np.ndarray, labels: np.ndarray, *, strength: float = 0.1,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    support = _norm(support)
    labels = labels.astype(str)
    center = support.mean(axis=0)
    centered = support - center
    classes = sorted(set(labels.tolist()))
    means = np.vstack([centered[labels == label].mean(axis=0) for label in classes])
    counts = np.asarray([np.sum(labels == label) for label in classes], dtype=np.float64)
    global_mean = np.average(means, axis=0, weights=counts)
    between = np.average((means - global_mean) ** 2, axis=0, weights=counts)
    within = np.concatenate([
        (centered[labels == label] - centered[labels == label].mean(axis=0, keepdims=True)) ** 2
        for label in classes
    ]).mean(axis=0)
    fisher = between / np.maximum(within, 1.0e-6)
    fisher /= max(float(np.median(fisher)), 1.0e-6)
    whiten = 1.0 / np.sqrt(np.maximum(centered.var(axis=0), 1.0e-5))
    scale = np.power(np.clip(fisher, 0.05, 20.0), float(strength))
    scale *= np.power(whiten / max(float(np.median(whiten)), 1.0e-6), 0.5)
    scale = np.clip(scale, 0.05, 20.0)
    return center, scale, {
        "transform_scale_min": float(scale.min()),
        "transform_scale_max": float(scale.max()),
        "transform_scale_mean": float(scale.mean()),
    }


def _apply_diag_whiten_fisher(rows: np.ndarray, center: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return _norm((_norm(rows) - center[None, :]) * scale[None, :])


def _fit_qknnv42_state(
    support_x: np.ndarray, support_y: np.ndarray, *, support_representation: str = "all_support",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Enroll support once into the exact persistent state needed by onboard scoring."""
    labels = support_y.astype(str)
    classes = sorted(set(labels.tolist()))
    class_to_i = {label: index for index, label in enumerate(classes)}
    class_indices = np.asarray([class_to_i[label] for label in labels.tolist()], dtype=np.int32)
    center, scale, transform_info = _fit_diag_whiten_fisher(support_x, labels, strength=0.1)
    support = _apply_diag_whiten_fisher(support_x, center, scale)
    quantized_all = np.clip(np.rint(127.0 * support), -127, 127).astype(np.int8)
    restored_all = _norm(quantized_all.astype(np.float64) / 127.0)
    prototypes = np.vstack([
        _norm(restored_all[class_indices == index].mean(axis=0, keepdims=True))[0]
        for index in range(len(classes))
    ])
    representation = str(support_representation).strip().lower()
    if representation == "all_support":
        quantized = quantized_all
        stored_class_indices = class_indices
    elif representation in {"class_medoid", "class_diverse2", "class_diverse4"}:
        budget = {"class_medoid": 1, "class_diverse2": 2, "class_diverse4": 4}[representation]
        selected: list[int] = []
        for index in range(len(classes)):
            positions = np.flatnonzero(class_indices == index)
            class_rows = restored_all[positions]
            local_selected = [int(np.argmax(class_rows @ prototypes[index]))]
            while len(local_selected) < min(int(budget), len(positions)):
                similarity_to_selected = class_rows @ class_rows[local_selected].T
                min_distance = 1.0 - np.max(similarity_to_selected, axis=1)
                min_distance[local_selected] = -np.inf
                local_selected.append(int(np.argmax(min_distance)))
            selected.extend(int(positions[local]) for local in local_selected)
        quantized = quantized_all[selected]
        stored_class_indices = class_indices[selected].astype(np.int32, copy=False)
    elif representation == "prototype_only":
        quantized = np.empty((0, quantized_all.shape[1]), dtype=np.int8)
        stored_class_indices = np.empty((0,), dtype=np.int32)
    else:
        raise ValueError(f"unsupported qKNNV42 support representation: {support_representation}")
    compactness = 1.0 - np.max(restored_all @ prototypes.T, axis=1)
    class_label_table_bytes = sum(len(label.encode("utf-8")) + 1 for label in classes)
    state = {
        "quantized_support": quantized,
        "class_indices": stored_class_indices,
        "classes": classes,
        "prototypes": prototypes,
        "center": center,
        "scale": scale,
    }
    state_info = {
        **transform_info,
        "support_representation": representation,
        "enrollment_support_count": int(len(quantized_all)),
        "stored_quantized_support_code_count": int(len(quantized)),
        "stored_raw_support_count": 0,
        "stored_class_prototype_count": len(classes),
        "feature_dim": int(quantized.shape[1]),
        "support_code_bytes": int(quantized.nbytes),
        "class_index_bytes": int(stored_class_indices.nbytes),
        "class_label_table_bytes": int(class_label_table_bytes),
        "prototype_bytes_float64": int(prototypes.nbytes),
        "transform_state_bytes_float64": int(center.nbytes + scale.nbytes),
        "enrollment_compactness_loss": float(np.mean(compactness)),
    }
    state_info["persistent_state_bytes"] = int(
        state_info["support_code_bytes"]
        + state_info["class_index_bytes"]
        + state_info["class_label_table_bytes"]
        + state_info["prototype_bytes_float64"]
        + state_info["transform_state_bytes_float64"]
    )
    return state, state_info


def _qknnv42_score_matrix(
    support_x: np.ndarray, support_y: np.ndarray, query_x: np.ndarray, *, old_labels: set[str],
    labelprop_mode: str = "dense_transductive",
    old_anchor_bias: float = 0.001,
    support_representation: str = "all_support",
) -> tuple[list[str], np.ndarray, dict[str, Any]]:
    enrollment_started = time.perf_counter()
    state, state_info = _fit_qknnv42_state(
        support_x, support_y, support_representation=support_representation
    )
    enrollment_elapsed = time.perf_counter() - enrollment_started
    scoring_started = time.perf_counter()
    query = _apply_diag_whiten_fisher(query_x, state["center"], state["scale"])
    restored = _norm(state["quantized_support"].astype(np.float64) / 127.0)
    class_indices = state["class_indices"]
    classes = state["classes"]
    prototype_matrix = state["prototypes"]
    representation = str(state_info["support_representation"])
    score_columns: list[np.ndarray] = []
    for class_index, label in enumerate(classes):
        prototype = prototype_matrix[class_index]
        prototype_score = query @ prototype
        if representation == "prototype_only":
            score = prototype_score
        else:
            class_support = restored[class_indices == class_index]
            similarity = query @ class_support.T
            knn = np.max(similarity, axis=1)
            score = 0.55 * knn + 0.45 * prototype_score
        if label in old_labels:
            score = score + float(old_anchor_bias)
        score_columns.append(score)
    scores = np.stack(score_columns, axis=1)
    lp_mode = str(labelprop_mode).strip().lower()
    graph_nodes = int(len(restored) + len(query))
    dense_graph_bytes = 0
    labelprop_macs = 0
    if lp_mode == "dense_transductive":
        if representation != "all_support":
            raise ValueError("dense_transductive label propagation requires all_support representation")
        labels = np.asarray(classes, dtype=object)[class_indices]
        scores += 0.025 * _labelprop(restored, labels, query, classes)
        # _labelprop materializes both similarity and transition matrices as float64.
        dense_graph_bytes = int(2 * graph_nodes * graph_nodes * np.dtype(np.float64).itemsize)
        labelprop_macs = int(
            graph_nodes * graph_nodes * restored.shape[1]
            + 8 * graph_nodes * graph_nodes * len(classes)
        )
    elif lp_mode == "support_prototype":
        # Streaming substitute: retain the small V42 residual correction, but
        # derive it from registered support prototypes only. No query-query
        # affinity matrix or query-batch state is created.
        residual = query @ prototype_matrix.T
        residual = (residual - residual.mean(axis=1, keepdims=True)) / np.maximum(
            residual.std(axis=1, keepdims=True), 1.0e-6
        )
        scores += 0.025 * np.clip(residual, -2.0, 2.0)
        labelprop_macs = int(len(query) * len(classes) * restored.shape[1])
    elif lp_mode == "disabled":
        pass
    else:
        raise ValueError(f"unsupported qKNNV42 labelprop mode: {labelprop_mode}")
    feature_dim = int(prototype_matrix.shape[1])
    support_score_macs = int(len(query) * len(restored) * feature_dim)
    prototype_score_macs = int(len(query) * len(classes) * feature_dim)
    objective = (
        "qknnv42_int8_top1_proto45_old_anchor_labelprop"
        if lp_mode == "dense_transductive" and representation == "all_support"
        else f"qknnv42_int8_top1_proto45_old_anchor_{lp_mode}"
    )
    if representation != "all_support":
        objective = f"{objective}_{representation}"
    score_matrix_elapsed = time.perf_counter() - scoring_started
    return classes, scores, {
        "adaptation_objective": objective,
        "transform_mode": "diag_whiten_fisher",
        "transform_strength": 0.1,
        **state_info,
        "query_labels_used_for_adaptation": False,
        "labelprop_mode": lp_mode,
        "old_anchor_bias": float(old_anchor_bias),
        "query_query_graph_used": lp_mode == "dense_transductive",
        "query_batch_state_required": lp_mode == "dense_transductive",
        "dense_graph_node_count": graph_nodes if lp_mode == "dense_transductive" else 0,
        "dense_graph_bytes_lower_bound": dense_graph_bytes,
        "dense_graph_peak_bytes_lower_bound": dense_graph_bytes,
        "dense_graph_cumulative_bytes": dense_graph_bytes,
        "estimated_support_score_macs": support_score_macs,
        "estimated_prototype_score_macs": prototype_score_macs,
        "estimated_labelprop_macs": labelprop_macs,
        "estimated_head_macs": support_score_macs + prototype_score_macs + labelprop_macs,
        "enrollment_latency_sec": enrollment_elapsed,
        "score_matrix_latency_sec": score_matrix_elapsed,
        "scenario_residual_weight": 0.5,
        "scenario_residual_applied": False,
        "scenario_residual_note": "zero_by_full_same_scenario_support_for_every_registered_class",
        "loss": float(state_info["enrollment_compactness_loss"]),
    }


def _qknnv42_predict(
    support_x: np.ndarray,
    support_y: np.ndarray,
    query_x: np.ndarray,
    *,
    old_labels: set[str],
    aux_support_x: np.ndarray | None = None,
    aux_query_x: np.ndarray | None = None,
    aux_score_weight: float = 0.0,
    decision_mode: str = "per_sample_argmax",
    old_query_count: int | None = None,
    query_per_class: int | None = None,
    labelprop_mode: str = "dense_transductive",
    old_anchor_bias: float = 0.001,
    support_representation: str = "all_support",
) -> tuple[np.ndarray, dict[str, Any]]:
    decision = str(decision_mode).strip().lower()
    lp_mode = str(labelprop_mode).strip().lower()
    if decision == "legacy_role_quota_oracle" and lp_mode != "dense_transductive":
        raise ValueError("lightweight qKNNV42 modes require per_sample_argmax deployment inference")
    classes, scores, info = _qknnv42_score_matrix(
        support_x,
        support_y,
        query_x,
        old_labels=old_labels,
        labelprop_mode=labelprop_mode,
        old_anchor_bias=old_anchor_bias,
        support_representation=support_representation,
    )
    weight = float(aux_score_weight)
    if not 0.0 <= weight <= 1.0:
        raise ValueError("qKNNV42 auxiliary score weight must be in [0,1]")
    if (aux_support_x is None) != (aux_query_x is None):
        raise ValueError("qKNNV42 auxiliary support/query features must be provided together")
    if weight > 0.0:
        if aux_support_x is None or aux_query_x is None:
            raise ValueError("qKNNV42 positive auxiliary score weight requires auxiliary features")
        aux_classes, aux_scores, aux_info = _qknnv42_score_matrix(
            aux_support_x,
            support_y,
            aux_query_x,
            old_labels=old_labels,
            labelprop_mode=labelprop_mode,
            old_anchor_bias=old_anchor_bias,
            support_representation=support_representation,
        )
        if aux_classes != classes:
            raise ValueError("qKNNV42 primary and auxiliary class orders differ")
        scores = (1.0 - weight) * scores + weight * aux_scores
        info.update(
            {
                "aux_feature_enabled": True,
                "aux_feature_dim": int(aux_support_x.shape[1]),
                "aux_score_weight": weight,
                "aux_transform_mode": str(aux_info["transform_mode"]),
                "aux_transform_scale_min": float(aux_info["transform_scale_min"]),
                "aux_transform_scale_max": float(aux_info["transform_scale_max"]),
                "aux_transform_scale_mean": float(aux_info["transform_scale_mean"]),
                "aux_loss": float(aux_info["loss"]),
                "aux_dense_graph_bytes_lower_bound": int(aux_info["dense_graph_bytes_lower_bound"]),
                "aux_estimated_head_macs": int(aux_info["estimated_head_macs"]),
                "aux_support_code_bytes": int(aux_info["support_code_bytes"]),
                "aux_class_index_bytes": int(aux_info["class_index_bytes"]),
                "aux_class_label_table_bytes": int(aux_info["class_label_table_bytes"]),
                "aux_prototype_bytes_float64": int(aux_info["prototype_bytes_float64"]),
                "aux_transform_state_bytes_float64": int(aux_info["transform_state_bytes_float64"]),
                "aux_persistent_state_bytes": int(aux_info["persistent_state_bytes"]),
                "aux_enrollment_support_count": int(aux_info["enrollment_support_count"]),
                "aux_stored_quantized_support_code_count": int(
                    aux_info["stored_quantized_support_code_count"]
                ),
                "aux_estimated_support_score_macs": int(aux_info["estimated_support_score_macs"]),
                "aux_estimated_prototype_score_macs": int(aux_info["estimated_prototype_score_macs"]),
                "aux_enrollment_latency_sec": float(aux_info["enrollment_latency_sec"]),
                "aux_score_matrix_latency_sec": float(aux_info["score_matrix_latency_sec"]),
            }
        )
        primary_graph_bytes = int(info["dense_graph_bytes_lower_bound"])
        aux_graph_bytes = int(aux_info["dense_graph_bytes_lower_bound"])
        info["dense_graph_bytes_lower_bound"] = max(primary_graph_bytes, aux_graph_bytes)
        info["dense_graph_peak_bytes_lower_bound"] = max(primary_graph_bytes, aux_graph_bytes)
        info["dense_graph_cumulative_bytes"] = primary_graph_bytes + aux_graph_bytes
        for key in (
            "support_code_bytes",
            "class_index_bytes",
            "class_label_table_bytes",
            "prototype_bytes_float64",
            "transform_state_bytes_float64",
            "persistent_state_bytes",
        ):
            info[key] = int(info[key] + aux_info[key])
        info["stored_quantized_support_code_count_total"] = int(
            info["stored_quantized_support_code_count"]
            + aux_info["stored_quantized_support_code_count"]
        )
        info["estimated_support_score_macs"] = int(
            info["estimated_support_score_macs"] + aux_info["estimated_support_score_macs"]
        )
        info["estimated_prototype_score_macs"] = int(
            info["estimated_prototype_score_macs"] + aux_info["estimated_prototype_score_macs"]
        )
        info["estimated_labelprop_macs"] = int(
            info["estimated_labelprop_macs"] + aux_info["estimated_labelprop_macs"]
        )
        info["enrollment_latency_sec"] = float(
            info["enrollment_latency_sec"] + aux_info["enrollment_latency_sec"]
        )
        info["score_matrix_latency_sec"] = float(
            info["score_matrix_latency_sec"] + aux_info["score_matrix_latency_sec"]
        )
        info["estimated_head_macs"] = int(info["estimated_head_macs"] + aux_info["estimated_head_macs"])
    else:
        info.update(
            {
                "aux_feature_enabled": False,
                "aux_feature_dim": 0,
                "aux_score_weight": 0.0,
                "stored_quantized_support_code_count_total": int(
                    info["stored_quantized_support_code_count"]
                ),
            }
        )
    decision_started = time.perf_counter()
    if decision == "per_sample_argmax":
        predicted = np.asarray(classes, dtype=object)[np.argmax(scores, axis=1)]
    elif decision == "legacy_role_quota_oracle":
        from scipy.optimize import linear_sum_assignment

        if old_query_count is None or query_per_class is None:
            raise ValueError("legacy role/quota oracle requires old_query_count and query_per_class")
        old_positions = [i for i, label in enumerate(classes) if label in old_labels]
        new_positions = [i for i, label in enumerate(classes) if label not in old_labels]

        def assign(block: np.ndarray, positions: list[int]) -> np.ndarray:
            if block.shape[0] == 0:
                return np.asarray([], dtype=object)
            expected = len(positions) * int(query_per_class)
            if block.shape[0] != expected:
                raise ValueError(f"role/quota block has {block.shape[0]} rows; expected {expected}")
            slot_scores = np.repeat(block[:, positions], int(query_per_class), axis=1)
            row_ind, col_ind = linear_sum_assignment(-slot_scores)
            if not np.array_equal(row_ind, np.arange(block.shape[0])):
                raise ValueError("Hungarian assignment did not cover every query row")
            class_pos = np.asarray(positions, dtype=np.int64)[col_ind // int(query_per_class)]
            return np.asarray(classes, dtype=object)[class_pos]

        old_count = int(old_query_count)
        predicted = np.concatenate(
            [assign(scores[:old_count], old_positions), assign(scores[old_count:], new_positions)]
        )
    else:
        raise ValueError(f"unsupported qKNNV42 decision mode: {decision_mode}")
    decision_elapsed = time.perf_counter() - decision_started
    info.update(
        {
            "decision_mode": decision,
            "role_oracle_used": decision == "legacy_role_quota_oracle",
            "equal_class_quota_used": decision == "legacy_role_quota_oracle",
            "scenario_hard_filter_effective": False,
            "scenario_hard_filter_note": "all registered classes have support in every formal scenario",
            "decision_latency_sec": decision_elapsed,
            "onboard_scoring_latency_sec": float(info["score_matrix_latency_sec"] + decision_elapsed),
            "onboard_scoring_latency_per_query_ms": float(
                (info["score_matrix_latency_sec"] + decision_elapsed) * 1000.0 / max(1, len(query_x))
            ),
        }
    )
    return predicted, info


def _accuracy(pred: np.ndarray, truth: np.ndarray) -> float:
    return float(np.mean(pred.astype(str) == truth.astype(str)))


def _detail_rows(pred: np.ndarray, truth: np.ndarray, meta: list[dict[str, str]], scenario: str) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str, str], list[int]] = {}
    for i, row in enumerate(meta):
        rx, tx, day, role = row["receiver_label"], row["transmitter_label"], row["day_i"], row["role"]
        for key in (("per_receiver", rx, "ALL", "ALL", role), ("per_transmitter", "ALL", tx, "ALL", role),
                    ("per_receiver_transmitter", rx, tx, "ALL", role),
                    ("per_receiver_transmitter_day", rx, tx, day, role)):
            groups.setdefault(key, []).append(i)
    rows: list[dict[str, Any]] = []
    for (kind, rx, tx, day, role), indices in sorted(groups.items()):
        confusion: dict[str, int] = {}
        for i in indices:
            key = f"{truth[i]}->{pred[i]}"
            confusion[key] = confusion.get(key, 0) + 1
        correct = sum(str(pred[i]) == str(truth[i]) for i in indices)
        rows.append({"scenario": scenario, "group_type": kind, "receiver_label": rx,
                     "transmitter_label": tx, "day": day, "role": role, "sample_count": len(indices),
                     "correct_count": correct, "accuracy": correct / len(indices),
                     "confusion_json": json.dumps(confusion, ensure_ascii=False, sort_keys=True)})
    return rows


def _meta(arrays: dict[str, np.ndarray], indices: list[int]) -> list[dict[str, str]]:
    return [{"sample_id": _sample_id(arrays, i), "receiver_label": str(arrays["rx_ids"][i]),
             "transmitter_label": str(arrays["tx_ids"][i]), "day_i": str(arrays["day_ids"][i]),
             "eq_i": str(arrays["eq_ids"][i]), "sig_i": str(arrays["sig_ids"][i]),
             "role": str(arrays["dataset_role"][i])} for i in indices]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def validate_config(config: dict[str, Any]) -> None:
    method = str(config.get("method", "")).lower()
    if method not in METHOD_STAGE:
        raise ValueError(f"method must be one of {sorted(METHOD_STAGE)}")
    if str(config.get("stage")) != METHOD_STAGE[method]:
        raise ValueError(f"{method} requires stage={METHOD_STAGE[method]}")
    if len(config.get("target_receiver_labels", [])) != 1:
        raise ValueError("each run must contain exactly one target receiver")
    if list(config.get("target_channel_scenarios", [])) != list(SCENARIOS):
        raise ValueError(f"formal tests must be exactly {list(SCENARIOS)}")
    mapping = config.get("feature_npz_by_scenario", {})
    nested_mapping = config.get("feature_npz_by_receiver_scenario", {})
    if not nested_mapping and set(mapping) != set(SCENARIOS):
        raise ValueError("feature_npz_by_scenario must contain all formal LEO scenarios")
    if nested_mapping:
        receivers = [str(value) for value in config.get("publication_target_receiver_grid", [])]
        if not receivers:
            receivers = [str(value) for value in config.get("target_receiver_labels", [])]
        missing = [rx for rx in receivers if rx not in nested_mapping or set(nested_mapping[rx]) != set(SCENARIOS)]
        if missing:
            raise ValueError(f"feature_npz_by_receiver_scenario is incomplete for receivers={missing}")
    if int(config.get("k_shot", 0)) <= 0 or int(config.get("support_pool_max_k", 0)) < int(config["k_shot"]):
        raise ValueError("invalid nested K-shot settings")
    if bool(config.get("unknown_rejection_enabled", False)) or config.get("target_unknown_tx_labels"):
        raise ValueError("Phase2 publication mainline excludes unknown rejection")
    aux_key = str(config.get("qknnv42_aux_feature_key", "")).strip()
    aux_weight = float(config.get("qknnv42_aux_score_weight", 0.0))
    if not 0.0 <= aux_weight <= 1.0:
        raise ValueError("qknnv42_aux_score_weight must be in [0,1]")
    if aux_weight > 0.0 and method != "cvs_qknnv42":
        raise ValueError("FFT auxiliary fusion is defined only for cvs_qknnv42")
    if aux_weight > 0.0 and not aux_key:
        raise ValueError("positive qknnv42_aux_score_weight requires qknnv42_aux_feature_key")
    decision_mode = str(config.get("qknnv42_decision_mode", "per_sample_argmax")).strip().lower()
    if decision_mode not in {"per_sample_argmax", "legacy_role_quota_oracle"}:
        raise ValueError(f"unsupported qknnv42_decision_mode: {decision_mode}")
    if decision_mode == "legacy_role_quota_oracle" and method != "cvs_qknnv42":
        raise ValueError("legacy role/quota oracle is defined only for cvs_qknnv42")
    labelprop_mode = str(config.get("qknnv42_labelprop_mode", "dense_transductive")).strip().lower()
    if labelprop_mode not in {"dense_transductive", "support_prototype", "disabled"}:
        raise ValueError(f"unsupported qknnv42_labelprop_mode: {labelprop_mode}")
    support_representation = str(
        config.get("qknnv42_support_representation", "all_support")
    ).strip().lower()
    if support_representation not in {
        "all_support", "class_medoid", "class_diverse2", "class_diverse4", "prototype_only"
    }:
        raise ValueError(f"unsupported qknnv42_support_representation: {support_representation}")
    if labelprop_mode == "dense_transductive" and support_representation != "all_support":
        raise ValueError("dense_transductive label propagation requires all_support representation")
    if labelprop_mode != "dense_transductive" and decision_mode != "per_sample_argmax":
        raise ValueError("lightweight qKNNV42 modes require per_sample_argmax deployment inference")
    old_anchor_bias = float(config.get("qknnv42_old_anchor_bias", 0.001))
    if not math.isfinite(old_anchor_bias) or abs(old_anchor_bias) > 0.1:
        raise ValueError("qknnv42_old_anchor_bias must be finite and within [-0.1,0.1]")


def run(config: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    validate_config(config)
    method = str(config["method"]).lower()
    stage = METHOD_STAGE[method]
    receiver = str(config["target_receiver_labels"][0])
    seed = int(config["split_seed"])
    old_labels = [str(v) for v in config["target_old_tx_labels"]]
    new_labels = [str(v) for v in config.get("target_new_tx_labels", [])] if stage == "Stage2-C" else []
    if stage == "Stage2-C" and not new_labels:
        raise ValueError("Stage2-C requires target-new labels")
    metrics_by_scenario: dict[str, dict[str, Any]] = {}
    score_rows: list[dict[str, Any]] = []
    detailed: list[dict[str, Any]] = []
    trace: list[dict[str, Any]] = []
    manifest_splits: dict[str, Any] = {}
    aux_key = str(config.get("qknnv42_aux_feature_key", "")).strip()
    aux_weight = float(config.get("qknnv42_aux_score_weight", 0.0))
    aux_dim = int(config.get("qknnv42_aux_feature_dim", 0))
    decision_mode = str(config.get("qknnv42_decision_mode", "per_sample_argmax")).strip().lower()
    labelprop_mode = str(config.get("qknnv42_labelprop_mode", "dense_transductive")).strip().lower()
    support_representation = str(
        config.get("qknnv42_support_representation", "all_support")
    ).strip().lower()
    old_anchor_bias = float(config.get("qknnv42_old_anchor_bias", 0.001))
    expected_tta_views = int(config.get("qknnv42_expected_tta_view_count", 1))
    for scenario in SCENARIOS:
        arrays, cache_manifest = _load_npz(_feature_path(config, receiver, scenario))
        if aux_weight > 0.0:
            if aux_key not in arrays:
                raise ValueError(f"feature NPZ for {scenario} is missing auxiliary key {aux_key!r}")
            if arrays[aux_key].ndim != 2 or int(arrays[aux_key].shape[0]) != int(arrays["features"].shape[0]):
                raise ValueError(f"misaligned auxiliary feature matrix for {scenario}: {arrays[aux_key].shape}")
            if aux_dim > 0 and int(arrays[aux_key].shape[1]) != aux_dim:
                raise ValueError(
                    f"auxiliary feature dimension mismatch for {scenario}: {arrays[aux_key].shape[1]} != {aux_dim}"
                )
            if int(cache_manifest.get("satellite_tta_view_count", 0)) != expected_tta_views:
                raise ValueError(
                    f"FFT experiment requires satellite_tta_view_count={expected_tta_views} for {scenario}"
                )
            if str(cache_manifest.get("aux_fft_view_alignment", "")) != "same_post_channel_view_as_backbone":
                raise ValueError(f"FFT feature is not certified as same-view aligned for {scenario}")
        roles = arrays["dataset_role"].astype(str)
        scenario_values = arrays["sat_scenarios"].astype(str)
        target_mask = np.isin(roles, ["target_old", "target_new"])
        if str(scenario) not in set(scenario_values[target_mask].tolist()):
            raise ValueError(f"feature cache has no target rows for required scenario {scenario}")
        old_support, old_query = _select_split(
            arrays, role="target_old", tx_labels=old_labels, receiver=receiver, seed=seed,
            k_shot=int(config["k_shot"]), support_pool_max_k=int(config["support_pool_max_k"]),
            query_per_tx=int(config["query_per_tx"]),
            scenario=scenario,
        )
        new_support: list[int] = []
        new_query: list[int] = []
        if stage == "Stage2-C":
            new_support, new_query = _select_split(
                arrays, role="target_new", tx_labels=new_labels, receiver=receiver, seed=seed,
                k_shot=int(config["k_shot"]), support_pool_max_k=int(config["support_pool_max_k"]),
                query_per_tx=int(config["query_per_tx"]),
                scenario=scenario,
            )
        support_idx = old_support + new_support
        query_idx = old_query + new_query
        support_x = arrays["features"][support_idx]
        support_y = arrays["tx_ids"][support_idx].astype(str)
        query_x = arrays["features"][query_idx]
        aux_support_x = arrays[aux_key][support_idx] if aux_weight > 0.0 else None
        aux_query_x = arrays[aux_key][query_idx] if aux_weight > 0.0 else None
        truth = arrays["tx_ids"][query_idx].astype(str)
        if method == "cvs_opgac":
            started = time.perf_counter()
            source_mask = (roles == "source") & np.isin(arrays["tx_ids"].astype(str), old_labels)
            predicted, before, info = _opgac_predict(
                arrays["features"][source_mask], arrays["tx_ids"][source_mask].astype(str),
                support_x, support_y, query_x, shrinkage_kappa=float(config.get("opgac_old_shrinkage_kappa", 3.0)),
            )
            metrics = {"target_old_accuracy": _accuracy(predicted, truth),
                       "target_old_accuracy_before_adaptation": _accuracy(before, truth)}
            metrics["target_old_accuracy_delta"] = metrics["target_old_accuracy"] - metrics["target_old_accuracy_before_adaptation"]
            elapsed = time.perf_counter() - started
        else:
            diagnostic_started = time.perf_counter()
            old_pred, _ = _qknnv42_predict(
                arrays["features"][old_support],
                arrays["tx_ids"][old_support],
                arrays["features"][old_query],
                old_labels=set(old_labels),
                aux_support_x=arrays[aux_key][old_support] if aux_weight > 0.0 else None,
                aux_query_x=arrays[aux_key][old_query] if aux_weight > 0.0 else None,
                aux_score_weight=aux_weight,
                decision_mode=decision_mode,
                old_query_count=len(old_query),
                query_per_class=int(config["query_per_tx"]),
                labelprop_mode=labelprop_mode,
                old_anchor_bias=old_anchor_bias,
                support_representation=support_representation,
            )
            diagnostic_elapsed = time.perf_counter() - diagnostic_started
            deploy_started = time.perf_counter()
            predicted, info = _qknnv42_predict(
                support_x,
                support_y,
                query_x,
                old_labels=set(old_labels),
                aux_support_x=aux_support_x,
                aux_query_x=aux_query_x,
                aux_score_weight=aux_weight,
                decision_mode=decision_mode,
                old_query_count=len(old_query),
                query_per_class=int(config["query_per_tx"]),
                labelprop_mode=labelprop_mode,
                old_anchor_bias=old_anchor_bias,
                support_representation=support_representation,
            )
            elapsed = time.perf_counter() - deploy_started
            old_count = len(old_query)
            old_acc = _accuracy(predicted[:old_count], truth[:old_count])
            new_acc = _accuracy(predicted[old_count:], truth[old_count:])
            harmonic = 0.0 if old_acc + new_acc <= 0 else 2.0 * old_acc * new_acc / (old_acc + new_acc)
            old_before = _accuracy(old_pred, arrays["tx_ids"][old_query].astype(str))
            metrics = {"old_acc": old_acc, "seen_new_acc": new_acc, "H_old_new": harmonic,
                       "old_acc_before_increment": old_before, "average_forgetting": old_before - old_acc,
                       "old_before_increment_diagnostic_latency_sec": diagnostic_elapsed,
                       "old_to_seen_new_rate": float(np.mean(np.isin(predicted[:old_count], new_labels))),
                       "seen_new_to_old_rate": float(np.mean(np.isin(predicted[old_count:], old_labels)))}
        metrics.update({"adaptation_latency_sec": elapsed,
                        "latency_per_query_ms": elapsed * 1000.0 / len(query_idx), **info})
        metrics_by_scenario[scenario] = metrics
        trace.append({"method": method, "scenario": scenario, "phase": "support_only_fit", "step": 1,
                      "total_steps": 1, "loss": float(info["loss"]), "gradient_updates": 0})
        meta = _meta(arrays, query_idx)
        detailed.extend(_detail_rows(predicted, truth, meta, scenario))
        for row, true, pred in zip(meta, truth.tolist(), predicted.tolist()):
            score_rows.append({**row, "true_label": true, "predicted_label": pred,
                               "correct": int(str(true) == str(pred)), "scenario": scenario})
        support_ids = [_sample_id(arrays, i) for i in support_idx]
        query_ids = [_sample_id(arrays, i) for i in query_idx]
        if set(support_ids) & set(query_ids):
            raise ValueError("support/query overlap")
        manifest_splits[scenario] = {"support_sample_ids": support_ids, "query_sample_ids": query_ids,
                                     "support_count": len(support_ids), "query_count": len(query_ids)}
    aggregate_keys = ["target_old_accuracy", "target_old_accuracy_before_adaptation", "target_old_accuracy_delta"] \
        if stage == "Stage2-B" else ["old_acc", "seen_new_acc", "H_old_new", "average_forgetting"]
    aggregate = {key + "_mean": float(np.mean([row[key] for row in metrics_by_scenario.values()])) for key in aggregate_keys}
    manifest = {"stage": stage, "method": method, "cvs_proposed_method": True,
                "backbone": str(config.get("backbone_id", "ADV3B02_CORE90_SOFT_E200")),
                "target_receiver_labels": [receiver], "target_old_tx_labels": old_labels,
                "target_new_tx_labels": new_labels, "target_labels_scope": "registered_support_only",
                "target_query_used_for_training": False, "target_query_used_for_model_selection": False,
                "query_used_for_transductive_inference": method == "cvs_qknnv42"
                and labelprop_mode == "dense_transductive",
                "support_query_overlap": False, "all_tests_satellite_augmented": True,
                "seed": int(config["seed"]), "split_seed": seed, "k_shot": int(config["k_shot"]),
                "support_pool_max_k": int(config["support_pool_max_k"]), "target_sample_strategy": "seeded_nested",
                "splits_by_scenario": manifest_splits, "unknown_rejection_enabled": False,
                "satellite_tta_view_count": expected_tta_views if aux_weight > 0.0 else None,
                "qknnv42_aux_feature_key": aux_key,
                "qknnv42_aux_feature_dim": aux_dim if aux_weight > 0.0 else 0,
                "qknnv42_aux_score_weight": aux_weight,
                "qknnv42_decision_mode": decision_mode,
                "qknnv42_labelprop_mode": labelprop_mode,
                "qknnv42_support_representation": support_representation,
                "qknnv42_old_anchor_bias": old_anchor_bias,
                "non_deployment_oracle_diagnostic": decision_mode == "legacy_role_quota_oracle"}
    result = {"experiment_id": config.get("experiment_id", f"{method}_{seed}"), "method": method,
              "stage": stage, "seed": int(config["seed"]), "target_receiver_label": receiver,
              "metrics": aggregate, "metrics_by_scenario": metrics_by_scenario,
              "detailed_result_rows": detailed, "split_manifest": manifest}
    run_dir.mkdir(parents=True, exist_ok=True)
    for name, payload in (("metrics.json", result), ("split_manifest.json", manifest),
                          ("resolved_config.json", config), ("detailed_metrics.json", detailed),
                          ("loss_trace.json", trace)):
        (run_dir / name).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_csv(run_dir / "score_table.csv", score_rows)
    _write_csv(run_dir / "detailed_metrics.csv", detailed)
    _write_csv(run_dir / "loss_trace.csv", trace)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--method", choices=sorted(METHOD_STAGE), default=None)
    parser.add_argument("--target-receiver", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--split-seed", type=int, default=None)
    parser.add_argument("--k-shot", type=int, default=None)
    parser.add_argument("--experiment-id", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = load_json_config(args.config)
    for key, value in (("method", args.method), ("seed", args.seed), ("split_seed", args.split_seed),
                       ("k_shot", args.k_shot), ("experiment_id", args.experiment_id)):
        if value is not None:
            config[key] = value
    if args.target_receiver is not None:
        config["target_receiver_labels"] = [args.target_receiver]
    if args.method is not None:
        config["stage"] = METHOD_STAGE[args.method]
    validate_config(config)
    if args.dry_run:
        print(json.dumps(config, ensure_ascii=False, sort_keys=True))
        return 0
    result = run(config, args.run_dir)
    print(json.dumps({"experiment_id": result["experiment_id"], "method": result["method"],
                      "metrics": result["metrics"], "run_dir": str(args.run_dir)}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
