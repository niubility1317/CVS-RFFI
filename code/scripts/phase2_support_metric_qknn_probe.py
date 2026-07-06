#!/usr/bin/env python3
"""Probe support-only metric transforms before compressed qKNN.

The transform is fitted from target support only. Query labels are used only
for audit metrics. The deployed state stores transform scalars plus quantized
support codes, not raw IQ or raw support samples.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _indices_digest(indices: np.ndarray) -> str:
    arr = np.asarray(indices, dtype=np.int64)
    return hashlib.sha256(arr.tobytes()).hexdigest()[:16]


def _split_fingerprint(
    old_splits: dict[str, Split],
    new_splits: dict[str, Split],
    old_labels: list[str],
    new_labels: list[str],
) -> dict[str, Any]:
    support_all: list[int] = []
    old_query_all: list[int] = []
    new_query_all: list[int] = []
    per_label: dict[str, dict[str, Any]] = {}
    for role, labels, splits, query_all in (
        ("old", old_labels, old_splits, old_query_all),
        ("new", new_labels, new_splits, new_query_all),
    ):
        for label in labels:
            support, query = splits[label]
            support_all.extend(support.astype(int).tolist())
            query_all.extend(query.astype(int).tolist())
            per_label[str(label)] = {
                "role": role,
                "support_count": int(support.size),
                "query_count": int(query.size),
                "support_sha16": _indices_digest(support),
                "query_sha16": _indices_digest(query),
            }
    query_all_combined = old_query_all + new_query_all
    return {
        "support_index_sha16": _indices_digest(np.asarray(support_all, dtype=np.int64)),
        "old_query_index_sha16": _indices_digest(np.asarray(old_query_all, dtype=np.int64)),
        "new_query_index_sha16": _indices_digest(np.asarray(new_query_all, dtype=np.int64)),
        "query_index_sha16": _indices_digest(np.asarray(query_all_combined, dtype=np.int64)),
        "split_fingerprint": per_label,
    }


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


def _role_balanced_predict(
    scores: np.ndarray,
    *,
    old_count: int,
    old_labels: list[str],
    new_labels: list[str],
) -> np.ndarray:
    """Balanced assignment inside the known old/new query partitions.

    This preserves the closed-set equal-quota protocol while preventing global
    Hungarian assignment from swapping old-query quota with new-query quota.
    """
    from scipy.optimize import linear_sum_assignment

    class_labels = old_labels + new_labels
    labels = np.asarray(class_labels, dtype=object)
    out = np.empty(scores.shape[0], dtype=object)
    old_rows = np.arange(int(old_count), dtype=int)
    new_rows = np.arange(int(old_count), int(scores.shape[0]), dtype=int)

    def assign(row_indices: np.ndarray, slot_indices: list[int]) -> bool:
        if row_indices.size == 0:
            return True
        if len(slot_indices) != int(row_indices.size):
            return False
        slot_array = np.asarray(slot_indices, dtype=int)
        slot_scores = scores[row_indices][:, slot_array]
        row_ind, col_ind = linear_sum_assignment(-slot_scores)
        out[row_indices[row_ind]] = labels[slot_array[col_ind]]
        return True

    old_slots = base._quota_slots(int(old_rows.size), old_labels, offset=0)
    new_slots = base._quota_slots(int(new_rows.size), new_labels, offset=len(old_labels))
    if not assign(old_rows, old_slots) or not assign(new_rows, new_slots):
        return base._balanced_predict(scores, old_count=old_count, old_labels=old_labels, new_labels=new_labels)
    return out.astype(str)


def _quota_predict_fast(
    scores: np.ndarray,
    *,
    labels: np.ndarray,
    class_indices: list[int],
    quotas: list[int],
) -> np.ndarray:
    """Fast quota assignment for few-class closed-set episodes.

    This avoids materializing one slot per query sample. It starts from row-wise
    argmax and repairs class quotas by moving the lowest-loss rows from
    overfull classes to underfull classes. It is deterministic and preserves
    exact quotas, but it is a greedy transport approximation rather than the
    exact Hungarian solution.
    """
    local_scores = scores[:, np.asarray(class_indices, dtype=int)]
    target = np.asarray(quotas, dtype=int)
    if local_scores.shape[0] == 0:
        return np.asarray([], dtype=str)
    if int(np.sum(target)) != int(local_scores.shape[0]) or np.any(target < 0):
        return labels[np.asarray(class_indices, dtype=int)[np.argmax(local_scores, axis=1)]].astype(str)
    pred = np.argmax(local_scores, axis=1).astype(int)
    counts = np.bincount(pred, minlength=len(class_indices)).astype(int)
    max_steps = int(local_scores.shape[0]) * max(1, len(class_indices))
    steps = 0
    while bool(np.any(counts != target)) and steps < max_steps:
        under = np.where(counts < target)[0]
        over_mask = counts > target
        if under.size == 0 or not bool(np.any(over_mask)):
            break
        best_row = -1
        best_class = -1
        best_loss = np.inf
        movable = over_mask[pred]
        if not bool(np.any(movable)):
            break
        movable_rows = np.where(movable)[0]
        current_scores = local_scores[movable_rows, pred[movable_rows]]
        for candidate in under:
            losses = current_scores - local_scores[movable_rows, candidate]
            local_best = int(np.argmin(losses))
            loss = float(losses[local_best])
            if loss < best_loss:
                best_loss = loss
                best_row = int(movable_rows[local_best])
                best_class = int(candidate)
        if best_row < 0 or best_class < 0:
            break
        old_class = int(pred[best_row])
        pred[best_row] = best_class
        counts[old_class] -= 1
        counts[best_class] += 1
        steps += 1
    if bool(np.any(counts != target)):
        return labels[np.asarray(class_indices, dtype=int)[np.argmax(local_scores, axis=1)]].astype(str)
    return labels[np.asarray(class_indices, dtype=int)[pred]].astype(str)


def _role_balanced_predict_fast(
    scores: np.ndarray,
    *,
    old_count: int,
    old_labels: list[str],
    new_labels: list[str],
) -> np.ndarray:
    class_labels = old_labels + new_labels
    labels = np.asarray(class_labels, dtype=object)
    out = np.empty(scores.shape[0], dtype=object)
    old_rows = np.arange(int(old_count), dtype=int)
    new_rows = np.arange(int(old_count), int(scores.shape[0]), dtype=int)

    old_quotas = [base._quota_slots(int(old_rows.size), old_labels, offset=0).count(i) for i in range(len(old_labels))]
    new_slots = base._quota_slots(int(new_rows.size), new_labels, offset=len(old_labels))
    new_quotas = [new_slots.count(len(old_labels) + i) for i in range(len(new_labels))]
    if old_rows.size:
        out[old_rows] = _quota_predict_fast(
            scores[old_rows],
            labels=labels,
            class_indices=list(range(len(old_labels))),
            quotas=old_quotas,
        )
    if new_rows.size:
        out[new_rows] = _quota_predict_fast(
            scores[new_rows],
            labels=labels,
            class_indices=list(range(len(old_labels), len(old_labels) + len(new_labels))),
            quotas=new_quotas,
        )
    return out.astype(str)


def _local_competition_adjust_scores(
    scores: np.ndarray,
    *,
    proto_sim: np.ndarray,
    old_labels: list[str],
    new_labels: list[str],
    neighbor_k: int,
    weight: float,
    clip: float,
    scope: str,
) -> tuple[np.ndarray, int]:
    """Sharpen class scores against support-prototype neighbors only.

    The adjustment stores no raw support samples. It uses only the compressed
    prototype similarity graph, so adding new classes only adds prototype nodes
    and a small local-neighbor list.
    """
    if float(weight) == 0.0 or int(neighbor_k) <= 1 or scores.shape[1] < 2:
        return scores, 0
    mode = str(scope).strip().lower()
    if mode not in {"all", "role"}:
        raise ValueError(f"unsupported local_competition_scope: {scope}")
    class_count = int(scores.shape[1])
    old_count = len(old_labels)
    sim = np.asarray(proto_sim, dtype=np.float64).copy()
    if sim.shape != (class_count, class_count):
        return scores, 0
    adjusted = scores.copy()
    changes = np.zeros_like(scores, dtype=np.float64)
    neighbor_edges = 0
    for class_index in range(class_count):
        allowed = np.ones(class_count, dtype=bool)
        if mode == "role":
            if class_index < old_count:
                allowed[old_count:] = False
            else:
                allowed[:old_count] = False
        allowed[class_index] = False
        candidates = np.where(allowed)[0]
        if candidates.size == 0:
            continue
        k = max(1, min(int(neighbor_k) - 1, int(candidates.size)))
        order = candidates[np.argsort(sim[class_index, candidates])[::-1][:k]]
        competitor = np.max(scores[:, order], axis=1)
        margin = scores[:, class_index] - competitor
        changes[:, class_index] = np.clip(margin, -float(clip), float(clip))
        neighbor_edges += int(order.size)
    adjusted = adjusted + float(weight) * changes
    return adjusted, neighbor_edges


def _assignment_predict(
    scores: np.ndarray,
    *,
    old_count: int,
    old_labels: list[str],
    new_labels: list[str],
    query_scenarios: np.ndarray,
    scenario_balanced_assignment: bool,
    role_balanced_assignment: bool,
    fast_role_balanced_assignment: bool = False,
    balanced_assignment: bool,
) -> np.ndarray:
    if scenario_balanced_assignment:
        return base._scenario_balanced_predict(
            scores,
            query_scenarios=query_scenarios,
            query_old_mask=np.arange(scores.shape[0]) < int(old_count),
            old_labels=old_labels,
            new_labels=new_labels,
        )
    if role_balanced_assignment:
        if bool(fast_role_balanced_assignment):
            return _role_balanced_predict_fast(scores, old_count=old_count, old_labels=old_labels, new_labels=new_labels)
        return _role_balanced_predict(scores, old_count=old_count, old_labels=old_labels, new_labels=new_labels)
    if balanced_assignment:
        return base._balanced_predict(scores, old_count=old_count, old_labels=old_labels, new_labels=new_labels)
    return base._predict(scores, old_labels + new_labels)


def _query_proto_refine_scores(
    scores: np.ndarray,
    *,
    features: np.ndarray,
    query_indices: np.ndarray,
    provisional_pred: np.ndarray,
    class_labels: list[str],
    topm: int,
    weight: float,
    clip: float,
) -> tuple[np.ndarray, int]:
    """Refine scores using temporary query-batch prototypes from pseudo labels."""
    if float(weight) == 0.0:
        return scores, 0
    query = qknn._normalize_rows(features[query_indices])
    pred = np.asarray(provisional_pred, dtype=object).astype(str)
    base_scores = np.asarray(scores, dtype=np.float64)
    proto_rows: list[np.ndarray] = []
    stored_count = 0
    for class_index, label in enumerate(class_labels):
        rows = np.where(pred == str(label))[0]
        if rows.size == 0:
            proto_rows.append(np.zeros(query.shape[1], dtype=np.float64))
            continue
        if int(topm) > 0 and int(rows.size) > int(topm):
            row_scores = base_scores[rows, class_index]
            rows = rows[np.argsort(row_scores)[::-1][: int(topm)]]
        proto_rows.append(query[rows].mean(axis=0))
        stored_count += 1
    proto = qknn._normalize_rows(np.stack(proto_rows, axis=0))
    refine = query @ proto.T
    refine = refine - np.mean(refine, axis=1, keepdims=True)
    refine = refine / (np.std(refine, axis=1, keepdims=True) + 1e-6)
    if float(clip) > 0.0:
        refine = np.clip(refine, -float(clip), float(clip))
    return base_scores + float(weight) * refine, int(stored_count)


def _support_anchored_transductive_proto_scores(
    scores: np.ndarray,
    *,
    features: np.ndarray,
    support_indices: np.ndarray,
    support_labels: np.ndarray,
    query_indices: np.ndarray,
    class_labels: list[str],
    old_count: int,
    old_labels: list[str],
    new_labels: list[str],
    query_scenarios: np.ndarray,
    scenario_balanced_assignment: bool,
    role_balanced_assignment: bool,
    balanced_assignment: bool,
    rounds: int,
    weight: float,
    support_weight: float,
    query_weight: float,
    query_topm: int,
    clip: float,
) -> tuple[np.ndarray, int]:
    """Iteratively refine with support-anchored temporary query prototypes.

    This is a transductive compressed-KNN step: persistent state remains class
    support prototypes, while query prototypes are temporary per batch.
    """
    if float(weight) == 0.0 or int(rounds) <= 0:
        return scores, 0
    support = qknn._normalize_rows(features[support_indices])
    query = qknn._normalize_rows(features[query_indices])
    labels = np.asarray(support_labels, dtype=object).astype(str)
    base_scores = np.asarray(scores, dtype=np.float64)
    support_proto_rows: list[np.ndarray] = []
    for label in class_labels:
        cls = support[labels == str(label)]
        if cls.size == 0:
            support_proto_rows.append(np.zeros(support.shape[1], dtype=np.float64))
        else:
            support_proto_rows.append(qknn._normalize_rows(cls.mean(axis=0, keepdims=True))[0])
    support_proto = qknn._normalize_rows(np.stack(support_proto_rows, axis=0))
    work_scores = base_scores.copy()
    stored_count = 0
    for _round in range(int(rounds)):
        provisional = _assignment_predict(
            work_scores,
            old_count=old_count,
            old_labels=old_labels,
            new_labels=new_labels,
            query_scenarios=query_scenarios,
            scenario_balanced_assignment=bool(scenario_balanced_assignment),
            role_balanced_assignment=bool(role_balanced_assignment),
            balanced_assignment=bool(balanced_assignment),
        )
        proto_rows: list[np.ndarray] = []
        for class_index, label in enumerate(class_labels):
            rows = np.where(provisional == str(label))[0]
            if rows.size and int(query_topm) > 0 and rows.size > int(query_topm):
                row_scores = work_scores[rows, class_index]
                rows = rows[np.argsort(row_scores)[::-1][: int(query_topm)]]
            if rows.size:
                query_proto = qknn._normalize_rows(query[rows].mean(axis=0, keepdims=True))[0]
                proto_rows.append(float(support_weight) * support_proto[class_index] + float(query_weight) * query_proto)
                stored_count += 1
            else:
                proto_rows.append(support_proto[class_index])
        proto = qknn._normalize_rows(np.stack(proto_rows, axis=0))
        refine = query @ proto.T
        refine = refine - np.mean(refine, axis=1, keepdims=True)
        refine = refine / (np.std(refine, axis=1, keepdims=True) + 1e-6)
        if float(clip) > 0.0:
            refine = np.clip(refine, -float(clip), float(clip))
        work_scores = base_scores + float(weight) * refine
    return work_scores, int(stored_count)


def _dense_cluster_query_refine_scores(
    scores: np.ndarray,
    *,
    features: np.ndarray,
    support_indices: np.ndarray,
    support_labels: np.ndarray,
    query_indices: np.ndarray,
    class_labels: list[str],
    old_count: int,
    old_labels: list[str],
    new_labels: list[str],
    proto_sim: np.ndarray,
    query_scenarios: np.ndarray,
    scenario_balanced_assignment: bool,
    role_balanced_assignment: bool,
    balanced_assignment: bool,
    rounds: int,
    weight: float,
    similarity_threshold: float,
    neighbor_k: int,
    candidate_topn: int,
    query_topm: int,
    clip: float,
    scope: str,
) -> tuple[np.ndarray, int, int, int]:
    """Refine dense support-prototype clusters with temporary query prototypes.

    Persistent state remains compressed: the cluster graph is derived from class
    prototypes, while query prototypes exist only for the current inference batch.
    """
    if float(weight) == 0.0 or int(rounds) <= 0:
        return scores, 0, 0, 0
    mode = str(scope).strip().lower()
    if mode not in {"all", "role", "new", "old"}:
        raise ValueError(f"unsupported dense_cluster_scope: {scope}")
    class_count = len(class_labels)
    old_class_count = len(old_labels)
    sim = np.asarray(proto_sim, dtype=np.float64)
    if sim.shape != (class_count, class_count) or class_count < 2:
        return scores, 0, 0, 0

    support = qknn._normalize_rows(features[support_indices])
    query = qknn._normalize_rows(features[query_indices])
    labels = np.asarray(support_labels, dtype=object).astype(str)
    support_proto_rows: list[np.ndarray] = []
    for label in class_labels:
        class_support = support[labels == str(label)]
        if class_support.size == 0:
            support_proto_rows.append(np.zeros(support.shape[1], dtype=np.float64))
        else:
            support_proto_rows.append(qknn._normalize_rows(class_support.mean(axis=0, keepdims=True))[0])
    support_proto = qknn._normalize_rows(np.stack(support_proto_rows, axis=0))

    def components(indices: list[int]) -> list[list[int]]:
        index_set = set(indices)
        graph = {idx: set() for idx in indices}
        for idx in indices:
            candidates = [other for other in indices if other != idx]
            if not candidates:
                continue
            connected: set[int] = set()
            if float(similarity_threshold) <= 1.0:
                connected.update(other for other in candidates if float(sim[idx, other]) >= float(similarity_threshold))
            if int(neighbor_k) > 0:
                ordered = sorted(candidates, key=lambda other: float(sim[idx, other]), reverse=True)
                connected.update(ordered[: int(neighbor_k)])
            for other in connected:
                if other in index_set:
                    graph[idx].add(other)
                    graph[other].add(idx)
        seen: set[int] = set()
        out: list[list[int]] = []
        for idx in indices:
            if idx in seen:
                continue
            stack = [idx]
            comp: list[int] = []
            seen.add(idx)
            while stack:
                node = stack.pop()
                comp.append(node)
                for nxt in graph[node]:
                    if nxt not in seen:
                        seen.add(nxt)
                        stack.append(nxt)
            if len(comp) >= 2:
                out.append(sorted(comp))
        return out

    scopes: list[list[int]] = []
    if mode in {"all", "role"}:
        if mode == "all":
            scopes.append(list(range(class_count)))
        else:
            scopes.append(list(range(old_class_count)))
            scopes.append(list(range(old_class_count, class_count)))
    elif mode == "old":
        scopes.append(list(range(old_class_count)))
    else:
        scopes.append(list(range(old_class_count, class_count)))
    cluster_components = [comp for scope_indices in scopes for comp in components(scope_indices)]
    if not cluster_components:
        return scores, 0, 0, 0

    work_scores = np.asarray(scores, dtype=np.float64).copy()
    used_clusters = 0
    temp_proto_count = 0
    adjusted_rows_total = 0
    for _round in range(int(rounds)):
        provisional = _assignment_predict(
            work_scores,
            old_count=old_count,
            old_labels=old_labels,
            new_labels=new_labels,
            query_scenarios=query_scenarios,
            scenario_balanced_assignment=bool(scenario_balanced_assignment),
            role_balanced_assignment=bool(role_balanced_assignment),
            balanced_assignment=bool(balanced_assignment),
        )
        pred_indices = np.asarray([class_labels.index(str(label)) for label in provisional], dtype=int)
        topn = max(1, min(int(candidate_topn), class_count))
        top_order = np.argsort(work_scores, axis=1)[:, ::-1][:, :topn]
        for comp in cluster_components:
            comp_array = np.asarray(comp, dtype=int)
            if comp_array.size < 2:
                continue
            if mode != "all" and comp_array[0] < old_class_count:
                role_rows = np.arange(0, int(old_count), dtype=int)
            elif mode != "all":
                role_rows = np.arange(int(old_count), work_scores.shape[0], dtype=int)
            else:
                role_rows = np.arange(work_scores.shape[0], dtype=int)
            if role_rows.size == 0:
                continue
            in_pred = np.isin(pred_indices[role_rows], comp_array)
            in_top = np.any(np.isin(top_order[role_rows], comp_array), axis=1)
            candidate_rows = role_rows[in_pred | in_top]
            if candidate_rows.size < comp_array.size:
                continue
            local_proto_rows: list[np.ndarray] = []
            local_temp = 0
            for class_index in comp:
                rows = candidate_rows[pred_indices[candidate_rows] == class_index]
                if rows.size == 0:
                    local_proto_rows.append(support_proto[class_index])
                    continue
                if int(query_topm) > 0 and rows.size > int(query_topm):
                    rows = rows[np.argsort(work_scores[rows, class_index])[::-1][: int(query_topm)]]
                query_proto = qknn._normalize_rows(query[rows].mean(axis=0, keepdims=True))[0]
                local_proto_rows.append(qknn._normalize_rows((support_proto[class_index] + query_proto)[None, :])[0])
                local_temp += 1
            local_proto = qknn._normalize_rows(np.stack(local_proto_rows, axis=0))
            refine = query[candidate_rows] @ local_proto.T
            refine = refine - np.mean(refine, axis=1, keepdims=True)
            refine = refine / (np.std(refine, axis=1, keepdims=True) + 1e-6)
            if float(clip) > 0.0:
                refine = np.clip(refine, -float(clip), float(clip))
            work_scores[np.ix_(candidate_rows, comp_array)] += float(weight) * refine
            used_clusters += 1
            temp_proto_count += int(local_temp)
            adjusted_rows_total += int(candidate_rows.size)
    return work_scores, int(used_clusters), int(temp_proto_count), int(adjusted_rows_total)


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


def _weighted_topm_mean(scores: np.ndarray, weights: np.ndarray, topm: int) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if scores.ndim != 2 or scores.shape[1] == 0:
        return np.full(scores.shape[0], -1e9, dtype=np.float64)
    k = max(1, min(int(topm), int(scores.shape[1])))
    top_idx = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
    top_scores = np.take_along_axis(scores, top_idx, axis=1)
    top_weights = np.maximum(weights[top_idx], 1e-6)
    return np.sum(top_scores * top_weights, axis=1) / np.maximum(np.sum(top_weights, axis=1), 1e-6)


def _support_quality_weights(
    *,
    support_scores: np.ndarray,
    support_labels: np.ndarray,
    class_labels: list[str],
    floor: float,
    margin_scale: float,
) -> tuple[np.ndarray, float, float]:
    labels = np.asarray(support_labels, dtype=object).astype(str)
    scores = np.asarray(support_scores, dtype=np.float64)
    label_to_index = {label: index for index, label in enumerate(class_labels)}
    truth_index = np.asarray([label_to_index[str(label)] for label in labels.tolist()], dtype=int)
    truth_scores = scores[np.arange(scores.shape[0]), truth_index]
    masked = scores.copy()
    masked[np.arange(scores.shape[0]), truth_index] = -np.inf
    runner_up = np.max(masked, axis=1)
    margin = truth_scores - runner_up
    scale = max(float(margin_scale), 1e-4)
    reliability = 1.0 / (1.0 + np.exp(-np.clip(margin / scale, -12.0, 12.0)))
    floor_value = float(np.clip(float(floor), 0.0, 1.0))
    weights = floor_value + (1.0 - floor_value) * reliability
    per_class_acc: list[float] = []
    pred = np.asarray(class_labels, dtype=object)[np.argmax(scores, axis=1)].astype(str)
    for label in class_labels:
        mask = labels == label
        if bool(np.any(mask)):
            per_class_acc.append(float(np.mean(pred[mask] == label)))
    return weights.astype(np.float64), float(min(per_class_acc, default=0.0)), float(np.mean(per_class_acc) if per_class_acc else 0.0)


def _support_quality_weighted_scores(
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
    support_scores: np.ndarray,
    quality_floor: float,
    margin_scale: float,
) -> tuple[np.ndarray, dict[str, float], np.ndarray, int, float, float]:
    labels = np.asarray(support_labels, dtype=object).astype(str)
    support_weights, loo_min_acc, loo_mean_acc = _support_quality_weights(
        support_scores=support_scores,
        support_labels=labels,
        class_labels=class_labels,
        floor=float(quality_floor),
        margin_scale=float(margin_scale),
    )

    if scenario_aware:
        query_scenarios = np.asarray(scenarios[query_indices], dtype=object).astype(str)
        support_scenarios = np.asarray(scenarios[support_indices], dtype=object).astype(str)
        out = np.full((query_indices.size, len(class_labels)), -1e9, dtype=np.float64)
        radii: dict[str, float] = {}
        proto_sim = np.zeros((len(class_labels), len(class_labels)), dtype=np.float64)
        for scenario in sorted({str(value) for value in query_scenarios.tolist()}):
            query_mask = query_scenarios == scenario
            support_mask = support_scenarios == scenario
            if int(np.sum(support_mask)) < max(1, int(topm)) or len(set(labels[support_mask].tolist())) < 2:
                support_mask = np.ones_like(support_mask, dtype=bool)
            sub_scores, sub_radii, sub_proto_sim, _sub_stored, _min_acc, _mean_acc = _support_quality_weighted_scores(
                features=features,
                support_indices=support_indices[support_mask],
                support_labels=labels[support_mask],
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
                support_scores=support_scores[support_mask],
                quality_floor=quality_floor,
                margin_scale=margin_scale,
            )
            out[query_mask] = sub_scores
            radii.update(sub_radii)
            proto_sim = sub_proto_sim
        return out, radii, proto_sim, int(support_weights.size), loo_min_acc, loo_mean_acc

    query = qknn._normalize_rows(features[query_indices])
    support = qknn._normalize_rows(features[support_indices])
    prototypes: list[np.ndarray] = []
    radii: list[float] = []
    local_scores: list[np.ndarray] = []
    for label in class_labels:
        class_mask = labels == label
        class_support = support[class_mask]
        class_weights = support_weights[class_mask]
        if class_support.size == 0:
            prototypes.append(np.zeros(features.shape[1], dtype=np.float64))
            radii.append(1.0)
            local_scores.append(np.full(query.shape[0], -1e9, dtype=np.float64))
            continue
        weighted_mean = np.sum(class_support * class_weights[:, None], axis=0) / max(float(np.sum(class_weights)), 1e-6)
        prototype = qknn._normalize_rows(weighted_mean[None, :])[0]
        prototypes.append(prototype)
        radius = float(np.sum(class_weights * (1.0 - class_support @ prototype)) / max(float(np.sum(class_weights)), 1e-6))
        radii.append(radius)
        local = _weighted_topm_mean(query @ class_support.T, class_weights, int(topm))
        if float(radius_norm) != 0.0:
            local = 1.0 - ((1.0 - local) / (max(radius, 1e-4) ** float(radius_norm)))
        local_scores.append(local)

    proto_matrix = qknn._normalize_rows(np.stack(prototypes, axis=0))
    proto_scores = query @ proto_matrix.T
    if float(radius_norm) != 0.0:
        denom = np.power(np.maximum(np.asarray(radii, dtype=np.float64), 1e-4), float(radius_norm))[None, :]
        proto_scores = 1.0 - ((1.0 - proto_scores) / denom)
    score_matrix = (1.0 - float(proto_mix)) * np.stack(local_scores, axis=1) + float(proto_mix) * proto_scores
    for class_index, label in enumerate(class_labels):
        if label in old_labels:
            score_matrix[:, class_index] += float(old_bias)
    proto_sim = proto_matrix @ proto_matrix.T
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
    return score_matrix, radius_by_label, proto_sim, int(support_weights.size), loo_min_acc, loo_mean_acc


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


def _query_pair_cluster_quota_refine(
    pred: np.ndarray,
    scores: np.ndarray,
    *,
    features: np.ndarray,
    support_indices: np.ndarray,
    support_labels: np.ndarray,
    query_indices: np.ndarray,
    class_labels: list[str],
    old_count: int,
    old_labels: list[str],
    new_labels: list[str],
    proto_sim: np.ndarray,
    top_pairs: int,
    similarity_threshold: float,
    score_weight: float,
    query_weight: float,
    clip: float,
    scope: str,
) -> tuple[np.ndarray, int, int, str]:
    """Pair-local quota refinement with temporary query clusters.

    The persistent state is still compressed support prototypes plus scalar
    gates. Query centers are batch-local and discarded after inference.
    """
    if int(top_pairs) <= 0 or float(query_weight) == 0.0:
        return pred, 0, 0, ""
    scope_norm = str(scope).strip().lower()
    if scope_norm not in {"new", "old", "all", "role"}:
        raise ValueError(f"unsupported query_pair_cluster_scope: {scope}")
    refined = np.asarray(pred, dtype=object).astype(str).copy()
    labels = np.asarray(class_labels, dtype=object).astype(str)
    label_to_index = {label: index for index, label in enumerate(labels.tolist())}
    old_set = set(old_labels)
    new_set = set(new_labels)
    support = qknn._normalize_rows(features[support_indices])
    query = qknn._normalize_rows(features[query_indices])
    support_label_arr = np.asarray(support_labels, dtype=object).astype(str)
    support_proto_rows: list[np.ndarray] = []
    for label in labels.tolist():
        cls = support[support_label_arr == str(label)]
        if cls.size == 0:
            support_proto_rows.append(np.zeros(support.shape[1], dtype=np.float64))
        else:
            support_proto_rows.append(qknn._normalize_rows(cls.mean(axis=0, keepdims=True))[0])
    support_proto = qknn._normalize_rows(np.stack(support_proto_rows, axis=0))

    class_indices: list[int] = []
    for label in labels.tolist():
        is_old = label in old_set
        is_new = label in new_set
        if scope_norm == "new" and not is_new:
            continue
        if scope_norm == "old" and not is_old:
            continue
        class_indices.append(label_to_index[label])
    if scope_norm == "role":
        class_indices = list(range(len(class_labels)))
    if len(class_indices) < 2:
        return refined, 0, 0, ""

    top2_counts: dict[tuple[int, int], int] = {}
    if scope_norm == "old":
        row_pool = np.arange(0, int(old_count), dtype=int)
    elif scope_norm == "new":
        row_pool = np.arange(int(old_count), int(scores.shape[0]), dtype=int)
    else:
        row_pool = np.arange(int(scores.shape[0]), dtype=int)
    local_cols = np.asarray(class_indices, dtype=int)
    if row_pool.size and local_cols.size >= 2:
        local_scores = scores[row_pool][:, local_cols]
        order = np.argsort(-local_scores, axis=1)[:, :2]
        for pair in order.tolist():
            left = int(local_cols[pair[0]])
            right = int(local_cols[pair[1]])
            key = tuple(sorted((left, right)))
            top2_counts[key] = top2_counts.get(key, 0) + 1

    pairs: list[tuple[float, int, int]] = []
    for pos, left in enumerate(class_indices):
        for right in class_indices[pos + 1 :]:
            sim = float(proto_sim[left, right])
            count = float(top2_counts.get(tuple(sorted((left, right))), 0))
            if sim < float(similarity_threshold) and count <= 0.0:
                continue
            risk = sim + 0.002 * count
            pairs.append((risk, int(left), int(right)))
    if not pairs:
        return refined, 0, 0, ""

    changed = 0
    applied = 0
    pair_names: list[str] = []
    for _risk, left, right in sorted(pairs, reverse=True)[: int(top_pairs)]:
        left_label = str(labels[left])
        right_label = str(labels[right])
        pair_mask = (refined == left_label) | (refined == right_label)
        if scope_norm == "new":
            pair_mask[: int(old_count)] = False
        elif scope_norm == "old":
            pair_mask[int(old_count) :] = False
        pair_indices = np.where(pair_mask)[0]
        pair_count = int(pair_indices.size)
        if pair_count <= 2:
            continue
        left_quota = int(np.sum(refined[pair_indices] == left_label))
        if left_quota <= 0 or left_quota >= pair_count:
            continue
        left_rows = pair_indices[refined[pair_indices] == left_label]
        right_rows = pair_indices[refined[pair_indices] == right_label]
        if left_rows.size == 0 or right_rows.size == 0:
            continue
        left_center = qknn._normalize_rows(query[left_rows].mean(axis=0, keepdims=True))[0]
        right_center = qknn._normalize_rows(query[right_rows].mean(axis=0, keepdims=True))[0]
        support_axis = support_proto[left] - support_proto[right]
        support_axis = qknn._normalize_rows(support_axis[None, :])[0]
        query_axis = left_center - right_center
        query_axis = qknn._normalize_rows(query_axis[None, :])[0]
        if float(np.dot(query_axis, support_axis)) < 0.0:
            query_axis = -query_axis
        axis = qknn._normalize_rows((support_axis + float(query_weight) * query_axis)[None, :])[0]
        cluster_margin = query[pair_indices] @ axis
        cluster_margin = cluster_margin - float(np.mean(cluster_margin))
        cluster_margin = cluster_margin / (float(np.std(cluster_margin)) + 1e-6)
        if float(clip) > 0.0:
            cluster_margin = np.clip(cluster_margin, -float(clip), float(clip))
        score_margin = scores[pair_indices, left] - scores[pair_indices, right]
        combined = float(score_weight) * score_margin + float(query_weight) * cluster_margin
        order = np.argsort(-combined)
        new_pair = np.full(pair_count, right_label, dtype=object)
        new_pair[order[:left_quota]] = left_label
        local_changed = int(np.sum(refined[pair_indices] != new_pair))
        if local_changed <= 0:
            continue
        refined[pair_indices] = new_pair
        changed += local_changed
        applied += 1
        pair_names.append(f"{left_label}<->{right_label}")
    return refined.astype(str), int(changed), int(applied), ";".join(pair_names[:16])


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


def _proxy_prototypes(features: np.ndarray, tx_ids: np.ndarray, roles: np.ndarray) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for label in sorted({str(value) for value in tx_ids[roles == "proxy_unknown"].tolist()}):
        idx = np.where((tx_ids == label) & (roles == "proxy_unknown"))[0].astype(int)
        if idx.size:
            out[label] = qknn._normalize_rows(features[idx].mean(axis=0, keepdims=True))[0]
    return out


def _support_guided_proxy_adjust_scores(
    scores: np.ndarray,
    *,
    features: np.ndarray,
    tx_ids: np.ndarray,
    roles: np.ndarray,
    support_indices: np.ndarray,
    support_labels: np.ndarray,
    query_indices: np.ndarray,
    class_labels: list[str],
    candidate_rows: list[dict[str, Any]],
    weight: float,
    top_pairs: int,
    clip: float,
) -> tuple[np.ndarray, int, int]:
    if float(weight) == 0.0 or int(top_pairs) <= 0 or not candidate_rows:
        return scores, 0, 0
    label_to_index = {label: idx for idx, label in enumerate(class_labels)}
    support = qknn._normalize_rows(features[support_indices])
    query = qknn._normalize_rows(features[query_indices])
    support_label_arr = np.asarray(support_labels, dtype=object).astype(str)
    proxy_proto = _proxy_prototypes(qknn._normalize_rows(features), tx_ids, roles)
    adjusted = np.asarray(scores, dtype=np.float64).copy()
    pair_accum: dict[tuple[str, str], list[np.ndarray]] = {}
    rows = sorted(
        candidate_rows,
        key=lambda row: float(row.get("analogy_score", 0.0)),
        reverse=True,
    )[: int(top_pairs)]
    used = 0
    for row in rows:
        new_label = str(row.get("target_new", ""))
        old_label = str(row.get("hard_old", ""))
        left_proxy = str(row.get("left_proxy", ""))
        right_proxy = str(row.get("right_proxy", ""))
        if new_label not in label_to_index or old_label not in label_to_index:
            continue
        if left_proxy not in proxy_proto or right_proxy not in proxy_proto:
            continue
        new_support = support[support_label_arr == new_label]
        old_support = support[support_label_arr == old_label]
        if new_support.size == 0 or old_support.size == 0:
            continue
        axis = proxy_proto[left_proxy] - proxy_proto[right_proxy]
        norm = float(np.linalg.norm(axis))
        if norm < 1e-8:
            continue
        axis = axis / norm
        new_mean_vec = qknn._normalize_rows(new_support.mean(axis=0, keepdims=True))[0]
        old_mean_vec = qknn._normalize_rows(old_support.mean(axis=0, keepdims=True))[0]
        if float(axis @ (new_mean_vec - old_mean_vec)) < 0.0:
            axis = -axis
        new_proj = new_support @ axis
        old_proj = old_support @ axis
        midpoint = 0.5 * (float(np.mean(new_proj)) + float(np.mean(old_proj)))
        scale = float(np.std(np.concatenate([new_proj, old_proj], axis=0)) + 1e-3)
        margin = np.clip((query @ axis - midpoint) / scale, -float(clip), float(clip))
        pair_accum.setdefault((new_label, old_label), []).append(margin)
        used += 1
    for (new_label, old_label), margins in pair_accum.items():
        mean_margin = np.mean(np.stack(margins, axis=0), axis=0)
        new_idx = label_to_index[new_label]
        old_idx = label_to_index[old_label]
        adjusted[:, new_idx] += float(weight) * mean_margin
        adjusted[:, old_idx] -= float(weight) * mean_margin
    stored_scalars = int(used * 3)
    return adjusted, int(used), stored_scalars


def _support_loo_proxy_candidate_rows(
    *,
    features: np.ndarray,
    tx_ids: np.ndarray,
    roles: np.ndarray,
    support_indices: np.ndarray,
    support_labels: np.ndarray,
    support_scores: np.ndarray,
    class_labels: list[str],
    old_labels: list[str],
    new_labels: list[str],
    top_rows: int,
    min_errors: int,
    scope: str,
    balance_classes: bool,
    bundle_rows: int,
    analogy_mode: bool,
) -> list[dict[str, Any]]:
    if int(top_rows) <= 0:
        return []
    scope_norm = str(scope).strip().lower()
    if scope_norm not in {"all", "role", "new"}:
        raise ValueError(f"unsupported support_guided_proxy_scope: {scope}")
    labels = np.asarray(support_labels, dtype=object).astype(str)
    label_to_index = {label: index for index, label in enumerate(class_labels)}
    old_set = set(old_labels)
    new_set = set(new_labels)
    old_count = int(sum(1 for label in labels.tolist() if label in old_set))
    loo_pred = _assignment_predict(
        np.asarray(support_scores, dtype=np.float64),
        old_count=old_count,
        old_labels=old_labels,
        new_labels=new_labels,
        query_scenarios=np.asarray(["support"] * int(labels.size), dtype=object),
        scenario_balanced_assignment=False,
        role_balanced_assignment=True,
        balanced_assignment=True,
    )
    pair_errors: dict[tuple[str, str], int] = {}
    pair_softness: dict[tuple[str, str], float] = {}
    for row_idx, (truth_label, pred_label) in enumerate(zip(labels.tolist(), loo_pred.tolist())):
        truth = str(truth_label)
        pred = str(pred_label)
        if truth not in label_to_index:
            continue
        score_row = np.asarray(support_scores[row_idx], dtype=np.float64)
        truth_idx = label_to_index[truth]
        ranked = np.argsort(score_row)[::-1]
        runner_idx = next((int(idx) for idx in ranked.tolist() if int(idx) != truth_idx), -1)
        if runner_idx < 0:
            continue
        runner = class_labels[runner_idx]
        margin = float(score_row[truth_idx] - score_row[runner_idx])
        hard = pred if pred != truth else runner
        if hard not in label_to_index or hard == truth:
            continue
        if scope_norm == "new" and (truth not in new_set or hard not in new_set):
            continue
        if scope_norm == "role" and ((truth in old_set) != (hard in old_set)):
            continue
        key = (truth, hard)
        if pred != truth:
            pair_errors[key] = pair_errors.get(key, 0) + 1
        pair_softness[key] = pair_softness.get(key, 0.0) + max(0.0, 0.10 - margin)

    support = qknn._normalize_rows(features[support_indices])
    support_proto: dict[str, np.ndarray] = {}
    for label in class_labels:
        cls = support[labels == label]
        if cls.size:
            support_proto[label] = qknn._normalize_rows(cls.mean(axis=0, keepdims=True))[0]
    proto_labels = [label for label in class_labels if label in support_proto]
    for truth in proto_labels:
        for hard in proto_labels:
            if hard == truth:
                continue
            if scope_norm == "new" and (truth not in new_set or hard not in new_set):
                continue
            if scope_norm == "role" and ((truth in old_set) != (hard in old_set)):
                continue
            sim = float(support_proto[truth] @ support_proto[hard])
            geometry_bonus = max(0.0, sim - 0.72) * 5.0
            if geometry_bonus > 0.0:
                key = (truth, hard)
                pair_softness[key] = pair_softness.get(key, 0.0) + geometry_bonus

    candidates: list[tuple[str, str, int, float]] = []
    for key in sorted(set(pair_errors) | set(pair_softness)):
        err = int(pair_errors.get(key, 0))
        soft = float(pair_softness.get(key, 0.0))
        if err < max(1, int(min_errors)) and soft <= 0.0:
            continue
        candidates.append((key[0], key[1], err, soft))
    candidates.sort(key=lambda item: (-(2.0 * float(item[2]) + float(item[3])), item[0], item[1]))
    if not candidates:
        return []

    proxy_proto = _proxy_prototypes(qknn._normalize_rows(features), tx_ids, roles)
    proxy_items = sorted(proxy_proto.items())
    if len(proxy_items) < 2:
        return []
    proxy_labels = [label for label, _proto in proxy_items]
    proxy_matrix = np.stack([proto for _label, proto in proxy_items], axis=0)

    rows: list[dict[str, Any]] = []
    seen_proxy_pairs: set[tuple[str, str, str, str]] = set()
    for truth, hard, err, soft in candidates[: max(4, int(top_rows) * 4)]:
        if truth not in support_proto or hard not in support_proto:
            continue
        local_rows: list[dict[str, Any]] = []
        risk_score = 2.0 * float(err) + float(soft)
        if bool(analogy_mode):
            truth_sim = proxy_matrix @ support_proto[truth]
            hard_sim = proxy_matrix @ support_proto[hard]
            topn = max(4, min(8, int(top_rows)))
            near_truth = np.argsort(-truth_sim)[:topn]
            near_hard = np.argsort(-hard_sim)[:topn]
            for left_i in near_truth.tolist():
                for right_i in near_hard.tolist():
                    if left_i == right_i:
                        continue
                    left_label = proxy_labels[left_i]
                    right_label = proxy_labels[right_i]
                    proxy_pair_sim = float(proxy_matrix[left_i] @ proxy_matrix[right_i])
                    score = (
                        risk_score
                        + float(truth_sim[left_i])
                        + float(hard_sim[right_i])
                        + 0.5 * proxy_pair_sim
                        - 0.25 * abs(float(truth_sim[right_i]) - float(hard_sim[left_i]))
                    )
                    local_rows.append(
                        {
                            "target_new": truth,
                            "hard_old": hard,
                            "hard_label": hard,
                            "left_proxy": left_label,
                            "right_proxy": right_label,
                            "analogy_score": float(score),
                            "support_loo_errors": int(err),
                            "support_loo_softness": float(soft),
                        }
                    )
        else:
            axis = support_proto[truth] - support_proto[hard]
            axis_norm = float(np.linalg.norm(axis))
            if axis_norm < 1e-8:
                continue
            axis = axis / axis_norm
            for left_label, left_proto in proxy_items:
                for right_label, right_proto in proxy_items:
                    if left_label == right_label:
                        continue
                    proxy_axis = left_proto - right_proto
                    proxy_norm = float(np.linalg.norm(proxy_axis))
                    if proxy_norm < 1e-8:
                        continue
                    sim = float((proxy_axis / proxy_norm) @ axis)
                    if sim <= 0.0:
                        continue
                    local_rows.append(
                        {
                            "target_new": truth,
                            "hard_old": hard,
                            "hard_label": hard,
                            "left_proxy": left_label,
                            "right_proxy": right_label,
                            "analogy_score": float(risk_score + sim),
                            "support_loo_errors": int(err),
                            "support_loo_softness": float(soft),
                        }
                    )
        local_rows.sort(key=lambda row: float(row.get("analogy_score", 0.0)), reverse=True)
        for row in local_rows[: max(1, int(top_rows))]:
            row_key = (
                str(row["target_new"]),
                str(row["hard_label"]),
                str(row["left_proxy"]),
                str(row["right_proxy"]),
            )
            if row_key in seen_proxy_pairs:
                continue
            seen_proxy_pairs.add(row_key)
            rows.append(row)
    rows.sort(key=lambda row: float(row.get("analogy_score", 0.0)), reverse=True)
    bundle_size = max(1, int(bundle_rows))
    if not bool(balance_classes) and bundle_size <= 1:
        return rows[: int(top_rows)]

    by_truth: dict[str, list[dict[str, Any]]] = {}
    truth_risk: dict[str, float] = {}
    for row in rows:
        truth = str(row.get("target_new", ""))
        by_truth.setdefault(truth, []).append(row)
        truth_risk[truth] = max(truth_risk.get(truth, 0.0), float(row.get("analogy_score", 0.0)))
    truth_order = sorted(by_truth, key=lambda label: (-truth_risk.get(label, 0.0), label))
    if bundle_size > 1:
        bundled: list[dict[str, Any]] = []
        for truth in truth_order:
            for row in by_truth[truth][:bundle_size]:
                bundled.append(row)
                if len(bundled) >= int(top_rows):
                    return bundled
        return bundled[: int(top_rows)]

    balanced: list[dict[str, Any]] = []
    offset = 0
    while len(balanced) < int(top_rows):
        added = False
        for truth in truth_order:
            bucket = by_truth[truth]
            if offset < len(bucket):
                balanced.append(bucket[offset])
                added = True
                if len(balanced) >= int(top_rows):
                    break
        if not added:
            break
        offset += 1
    return balanced[: int(top_rows)]


def _gate_support_guided_proxy_rows(
    *,
    scores: np.ndarray,
    features: np.ndarray,
    tx_ids: np.ndarray,
    roles: np.ndarray,
    support_indices: np.ndarray,
    support_labels: np.ndarray,
    class_labels: list[str],
    old_labels: list[str],
    new_labels: list[str],
    candidate_rows: list[dict[str, Any]],
    weight: float,
    max_rows: int,
    clip: float,
    floor_tol: float,
    mean_tol: float,
) -> tuple[list[dict[str, Any]], float, float, float, float]:
    if not candidate_rows or int(max_rows) <= 0 or float(weight) == 0.0:
        before_min, before_mean = _support_loo_accuracy_summary(
            scores,
            support_labels=support_labels,
            old_labels=old_labels,
            new_labels=new_labels,
        )
        return [], before_min, before_mean, before_min, before_mean
    gate_scores = np.asarray(scores, dtype=np.float64).copy()
    before_min, before_mean = _support_loo_accuracy_summary(
        gate_scores,
        support_labels=support_labels,
        old_labels=old_labels,
        new_labels=new_labels,
    )
    current_min = before_min
    current_mean = before_mean
    accepted: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for row in candidate_rows:
        key = (
            str(row.get("target_new", "")),
            str(row.get("hard_label", row.get("hard_old", ""))),
            str(row.get("left_proxy", "")),
            str(row.get("right_proxy", "")),
        )
        if key in seen:
            continue
        seen.add(key)
        trial_scores, used, _stored = _support_guided_proxy_adjust_scores(
            gate_scores,
            features=features,
            tx_ids=tx_ids,
            roles=roles,
            support_indices=support_indices,
            support_labels=support_labels,
            query_indices=support_indices,
            class_labels=class_labels,
            candidate_rows=[row],
            weight=float(weight),
            top_pairs=1,
            clip=float(clip),
        )
        if int(used) <= 0:
            continue
        trial_min, trial_mean = _support_loo_accuracy_summary(
            trial_scores,
            support_labels=support_labels,
            old_labels=old_labels,
            new_labels=new_labels,
        )
        if trial_min + float(floor_tol) < current_min:
            continue
        if trial_mean + float(mean_tol) < current_mean:
            continue
        accepted.append(row)
        gate_scores = trial_scores
        current_min = trial_min
        current_mean = trial_mean
        if len(accepted) >= int(max_rows):
            break
    return accepted, before_min, before_mean, current_min, current_mean


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
    if scope_norm not in {"all", "old_new", "new"}:
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
            if scope_norm == "new" and (left_is_old or right_is_old):
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


def _query_graph_smooth_scores(
    scores: np.ndarray,
    *,
    features: np.ndarray,
    query_indices: np.ndarray,
    scenarios: np.ndarray,
    neighbor_k: int,
    weight: float,
    temperature: float,
    rounds: int,
    scope: str,
) -> tuple[np.ndarray, int]:
    if float(weight) == 0.0 or int(neighbor_k) <= 0 or int(rounds) <= 0:
        return scores, 0
    scope_norm = str(scope).strip().lower()
    if scope_norm not in {"all", "scenario"}:
        raise ValueError(f"unsupported query_graph_scope: {scope}")
    query = qknn._normalize_rows(features[query_indices])
    query_scenarios = np.asarray(scenarios[query_indices], dtype=object).astype(str)
    smoothed = np.asarray(scores, dtype=np.float64).copy()
    total_edges = 0
    groups: list[np.ndarray] = []
    if scope_norm == "scenario":
        for scenario in sorted({str(value) for value in query_scenarios.tolist()}):
            groups.append(np.where(query_scenarios == scenario)[0])
    else:
        groups.append(np.arange(query.shape[0], dtype=int))
    for group in groups:
        if int(group.size) <= 1:
            continue
        k = max(1, min(int(neighbor_k), int(group.size) - 1))
        group_query = query[group]
        similarity = group_query @ group_query.T
        np.fill_diagonal(similarity, -np.inf)
        neighbor_idx = np.argpartition(similarity, kth=similarity.shape[1] - k, axis=1)[:, -k:]
        neighbor_sim = np.take_along_axis(similarity, neighbor_idx, axis=1)
        temp = max(float(temperature), 1e-6)
        exp = np.exp((neighbor_sim - np.max(neighbor_sim, axis=1, keepdims=True)) / temp)
        weights = exp / np.maximum(np.sum(exp, axis=1, keepdims=True), 1e-12)
        total_edges += int(group.size * k)
        for _round in range(int(rounds)):
            source = smoothed[group]
            neighbor_scores = source[neighbor_idx]
            averaged = np.sum(neighbor_scores * weights[:, :, None], axis=1)
            smoothed[group] = (1.0 - float(weight)) * source + float(weight) * averaged
    return smoothed, int(total_edges)


def _query_cluster_align_scores(
    scores: np.ndarray,
    *,
    features: np.ndarray,
    support_indices: np.ndarray,
    support_labels: np.ndarray,
    query_indices: np.ndarray,
    old_count: int,
    class_labels: list[str],
    old_labels: list[str],
    new_labels: list[str],
    weight: float,
    rounds: int,
    support_weight: float,
    temperature: float,
    clip: float,
    scope: str,
    agreement_min: float,
    margin_min: float,
) -> tuple[np.ndarray, int, int]:
    """Align temporary query clusters to support prototypes under known quotas.

    The persistent state remains the compressed support prototype bank. Query
    clusters are batch-local transductive state and are discarded after scoring.
    """
    if float(weight) == 0.0 or int(rounds) <= 0:
        return scores, 0, 0
    scope_norm = str(scope).strip().lower()
    if scope_norm not in {"all", "role", "new", "old"}:
        raise ValueError(f"unsupported query_cluster_scope: {scope}")

    support = qknn._normalize_rows(features[support_indices])
    query = qknn._normalize_rows(features[query_indices])
    labels = np.asarray(support_labels, dtype=object).astype(str)
    label_to_index = {label: index for index, label in enumerate(class_labels)}
    support_proto: dict[str, np.ndarray] = {}
    for label in class_labels:
        cls = support[labels == label]
        if cls.size:
            support_proto[label] = qknn._normalize_rows(cls.mean(axis=0, keepdims=True))[0]

    cluster_scores = np.zeros_like(scores, dtype=np.float64)
    temp_proto_count = 0
    assigned_rows = 0
    support_anchor = float(np.clip(float(support_weight), 0.0, 1.0))
    temp = max(float(temperature), 1e-6)

    groups: list[tuple[np.ndarray, list[str]]] = []
    if scope_norm == "all":
        groups.append((np.arange(query_indices.size, dtype=int), list(class_labels)))
    else:
        if scope_norm in {"role", "old"}:
            groups.append((np.arange(int(old_count), dtype=int), list(old_labels)))
        if scope_norm in {"role", "new"}:
            groups.append((np.arange(int(old_count), query_indices.size, dtype=int), list(new_labels)))

    for rows, labels_subset in groups:
        labels_subset = [label for label in labels_subset if label in support_proto and label in label_to_index]
        if rows.size == 0 or not labels_subset:
            continue
        local_query = query[rows]
        local_class_indices = [label_to_index[label] for label in labels_subset]
        centers = qknn._normalize_rows(np.stack([support_proto[label] for label in labels_subset], axis=0))
        local_label_array = np.asarray(labels_subset, dtype=object)
        slot_indices = base._quota_slots(int(rows.size), labels_subset, offset=0)
        quotas = [slot_indices.count(i) for i in range(len(labels_subset))]
        if sum(quotas) != int(rows.size):
            quotas = [int(rows.size) // len(labels_subset)] * len(labels_subset)
            for idx in range(int(rows.size) - sum(quotas)):
                quotas[idx % len(quotas)] += 1
        assigned_local = np.zeros(int(rows.size), dtype=int)
        for _round in range(int(rounds)):
            sim = local_query @ centers.T
            assigned_labels = _quota_predict_fast(
                sim,
                labels=local_label_array,
                class_indices=list(range(len(labels_subset))),
                quotas=quotas,
            )
            assigned_local = np.asarray(
                [labels_subset.index(str(label)) for label in assigned_labels.tolist()],
                dtype=int,
            )
            updated: list[np.ndarray] = []
            for local_idx, label in enumerate(labels_subset):
                member = local_query[assigned_local == local_idx]
                if member.size:
                    query_center = qknn._normalize_rows(member.mean(axis=0, keepdims=True))[0]
                    center = support_anchor * support_proto[label] + (1.0 - support_anchor) * query_center
                    updated.append(center)
                else:
                    updated.append(support_proto[label])
            centers = qknn._normalize_rows(np.stack(updated, axis=0))

        local_scores = (local_query @ centers.T) / temp
        local_base_scores = np.asarray(scores[rows][:, local_class_indices], dtype=np.float64)
        if local_base_scores.shape[1] > 1:
            base_order = np.argsort(local_base_scores, axis=1)
            base_top = base_order[:, -1]
            base_second = base_order[:, -2]
            base_margin = local_base_scores[np.arange(local_base_scores.shape[0]), base_top] - local_base_scores[
                np.arange(local_base_scores.shape[0]), base_second
            ]
            agreement = float(np.mean(base_top == assigned_local))
            if float(margin_min) < 0.0:
                margin_floor = 0.02 + 0.01 * np.log1p(len(labels_subset))
            else:
                margin_floor = max(float(margin_min), 0.0)
            if agreement < float(agreement_min) or float(np.mean(base_margin)) < margin_floor:
                continue
        local_scores = local_scores - np.mean(local_scores, axis=1, keepdims=True)
        local_scores = local_scores / (np.std(local_scores, axis=1, keepdims=True) + 1e-6)
        local_scores = np.clip(local_scores, -float(clip), float(clip))
        for local_idx, label in enumerate(labels_subset):
            cluster_scores[rows, label_to_index[label]] = local_scores[:, local_idx]
        temp_proto_count += len(labels_subset)
        assigned_rows += int(rows.size)

    adjusted = np.asarray(scores, dtype=np.float64) + float(weight) * cluster_scores
    return adjusted, int(temp_proto_count), int(assigned_rows)


def _support_query_labelprop_scores(
    *,
    features: np.ndarray,
    support_indices: np.ndarray,
    support_labels: np.ndarray,
    query_indices: np.ndarray,
    scenarios: np.ndarray,
    class_labels: list[str],
    neighbor_k: int,
    alpha: float,
    temperature: float,
    rounds: int,
    clip: float,
    scope: str,
) -> tuple[np.ndarray, int]:
    if int(neighbor_k) <= 0 or int(rounds) <= 0:
        return np.zeros((query_indices.size, len(class_labels)), dtype=np.float64), 0
    scope_norm = str(scope).strip().lower()
    if scope_norm not in {"all", "scenario"}:
        raise ValueError(f"unsupported labelprop_scope: {scope}")
    labels = np.asarray(support_labels, dtype=object).astype(str)
    label_to_index = {label: index for index, label in enumerate(class_labels)}
    support_scenarios = np.asarray(scenarios[support_indices], dtype=object).astype(str)
    query_scenarios = np.asarray(scenarios[query_indices], dtype=object).astype(str)
    out = np.zeros((query_indices.size, len(class_labels)), dtype=np.float64)
    total_edges = 0
    groups: list[tuple[np.ndarray, np.ndarray]] = []
    if scope_norm == "scenario":
        for scenario in sorted({str(value) for value in query_scenarios.tolist()}):
            q_local = np.where(query_scenarios == scenario)[0]
            s_local = np.where(support_scenarios == scenario)[0]
            if s_local.size < max(2, len(class_labels)) or len(set(labels[s_local].tolist())) < 2:
                s_local = np.arange(support_indices.size, dtype=int)
            groups.append((s_local, q_local))
    else:
        groups.append((np.arange(support_indices.size, dtype=int), np.arange(query_indices.size, dtype=int)))
    for support_local, query_local in groups:
        if query_local.size == 0 or support_local.size == 0:
            continue
        node_indices = np.concatenate([support_indices[support_local], query_indices[query_local]])
        node_features = qknn._normalize_rows(features[node_indices])
        support_count = int(support_local.size)
        node_count = int(node_indices.size)
        if node_count <= 1:
            continue
        k = max(1, min(int(neighbor_k), node_count - 1))
        similarity = node_features @ node_features.T
        np.fill_diagonal(similarity, -np.inf)
        neighbor_idx = np.argpartition(similarity, kth=similarity.shape[1] - k, axis=1)[:, -k:]
        neighbor_sim = np.take_along_axis(similarity, neighbor_idx, axis=1)
        temp = max(float(temperature), 1e-6)
        exp = np.exp((neighbor_sim - np.max(neighbor_sim, axis=1, keepdims=True)) / temp)
        weights = exp / np.maximum(np.sum(exp, axis=1, keepdims=True), 1e-12)
        y = np.zeros((node_count, len(class_labels)), dtype=np.float64)
        for local_row, label in enumerate(labels[support_local].tolist()):
            if label in label_to_index:
                y[local_row, label_to_index[label]] = 1.0
        f = y.copy()
        alpha_clamped = min(max(float(alpha), 0.0), 0.999)
        for _round in range(int(rounds)):
            propagated = np.sum(f[neighbor_idx] * weights[:, :, None], axis=1)
            f = alpha_clamped * propagated + (1.0 - alpha_clamped) * y
            f[:support_count] = y[:support_count]
        query_scores = f[support_count:]
        query_scores = query_scores - np.mean(query_scores, axis=1, keepdims=True)
        query_scores = query_scores / (np.std(query_scores, axis=1, keepdims=True) + 1e-6)
        query_scores = np.clip(query_scores, -float(clip), float(clip))
        out[query_local] = query_scores
        total_edges += int(node_count * k)
    return out, int(total_edges)


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


def _core_proto_scores(
    *,
    features: np.ndarray,
    support_indices: np.ndarray,
    support_labels: np.ndarray,
    query_indices: np.ndarray,
    class_labels: list[str],
    old_labels: set[str],
    core_count: int,
    topm: int,
    radius_norm: float,
    old_bias: float,
    mode: str,
) -> tuple[np.ndarray, int]:
    """Score against compressed support-derived cores instead of raw support.

    The stored state is a small set of normalized class cores. In centroid mode
    each core is an average of a support subset, so deployment does not need to
    retain individual support embeddings.
    """
    support = qknn._normalize_rows(features[support_indices])
    query = qknn._normalize_rows(features[query_indices])
    labels = np.asarray(support_labels, dtype=object).astype(str)
    mode_norm = str(mode).strip().lower()
    if mode_norm not in {"centroid", "axis"}:
        raise ValueError(f"unsupported core_proto_mode: {mode}")
    score_columns: list[np.ndarray] = []
    stored_count = 0
    for label_index, label in enumerate(class_labels):
        cls = support[labels == label]
        n_support = int(cls.shape[0])
        if n_support == 0:
            score_columns.append(np.full(query.shape[0], -1e9, dtype=np.float64))
            continue
        class_mean = qknn._normalize_rows(cls.mean(axis=0, keepdims=True))[0]
        n_core = max(1, min(int(core_count), n_support))
        cores: list[np.ndarray] = []
        if mode_norm == "axis" and n_core > 1 and len(class_labels) > 1:
            all_means: list[np.ndarray] = []
            for other in class_labels:
                other_cls = support[labels == other]
                if other_cls.size == 0:
                    all_means.append(np.zeros(support.shape[1], dtype=np.float64))
                else:
                    all_means.append(qknn._normalize_rows(other_cls.mean(axis=0, keepdims=True))[0])
            mean_matrix = np.stack(all_means, axis=0)
            sim = mean_matrix[label_index] @ mean_matrix.T
            sim[label_index] = -np.inf
            competitor = mean_matrix[int(np.argmax(sim))]
            axis = class_mean - competitor
            axis_norm = float(np.linalg.norm(axis))
            if axis_norm > 1e-8:
                proj = cls @ (axis / axis_norm)
                bins = np.array_split(np.argsort(proj), n_core)
                for bin_indices in bins:
                    if int(bin_indices.size):
                        cores.append(cls[bin_indices].mean(axis=0))
        if not cores:
            chosen = [int(np.argmax(cls @ class_mean))]
            while len(chosen) < n_core:
                selected = cls[np.asarray(chosen, dtype=int)]
                min_dist = np.min(1.0 - cls @ selected.T, axis=1)
                min_dist[np.asarray(chosen, dtype=int)] = -np.inf
                next_index = int(np.argmax(min_dist))
                if next_index in chosen:
                    break
                chosen.append(next_index)
            centers = cls[np.asarray(chosen, dtype=int)]
            assign = np.argmax(cls @ centers.T, axis=1)
            for core_index in range(len(chosen)):
                member = cls[assign == core_index]
                if int(member.shape[0]):
                    cores.append(member.mean(axis=0))
        core_matrix = qknn._normalize_rows(np.stack(cores, axis=0))
        stored_count += int(core_matrix.shape[0])
        radius = float(np.mean(1.0 - cls @ class_mean))
        score = _topm_mean(query @ core_matrix.T, int(topm))
        if float(radius_norm) != 0.0:
            score = 1.0 - ((1.0 - score) / (max(radius, 1e-4) ** float(radius_norm)))
        if label in old_labels:
            score = score + float(old_bias)
        score_columns.append(score)
    scores = np.stack(score_columns, axis=1)
    scores = scores - np.mean(scores, axis=1, keepdims=True)
    scores = scores / (np.std(scores, axis=1, keepdims=True) + 1e-6)
    return scores, int(stored_count)


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


def _old_residual_new_scores(
    *,
    features: np.ndarray,
    support_indices: np.ndarray,
    support_labels: np.ndarray,
    query_indices: np.ndarray,
    class_labels: list[str],
    old_labels: list[str],
    topm: int,
    proto_mix: float,
    rank: int,
    clip: float,
) -> tuple[np.ndarray, int, int]:
    """Score new classes after removing the compressed old-prototype subspace."""
    support = qknn._normalize_rows(features[support_indices])
    query = qknn._normalize_rows(features[query_indices])
    labels = np.asarray(support_labels, dtype=object).astype(str)
    old_set = {str(label) for label in old_labels}
    old_proto_rows: list[np.ndarray] = []
    for label in old_labels:
        cls = support[labels == str(label)]
        if cls.size:
            old_proto_rows.append(qknn._normalize_rows(cls.mean(axis=0, keepdims=True))[0])
    if len(old_proto_rows) < 2:
        return np.zeros((query.shape[0], len(class_labels)), dtype=np.float64), 0, 0
    old_proto = np.stack(old_proto_rows, axis=0)
    old_center = old_proto.mean(axis=0, keepdims=True)
    centered = old_proto - old_center
    max_rank = min(int(rank), centered.shape[0] - 1, centered.shape[1])
    if max_rank <= 0:
        return np.zeros((query.shape[0], len(class_labels)), dtype=np.float64), 0, 0
    _u, _s, vh = np.linalg.svd(centered, full_matrices=False)
    basis = vh[:max_rank].T

    def residualize(matrix: np.ndarray) -> np.ndarray:
        centered_matrix = matrix - old_center
        residual = centered_matrix - (centered_matrix @ basis) @ basis.T
        return qknn._normalize_rows(residual)

    support_res = residualize(support)
    query_res = residualize(query)
    out = np.zeros((query.shape[0], len(class_labels)), dtype=np.float64)
    used_new = 0
    for class_index, label in enumerate(class_labels):
        if str(label) in old_set:
            continue
        cls = support_res[labels == str(label)]
        if cls.size == 0:
            continue
        prototype = qknn._normalize_rows(cls.mean(axis=0, keepdims=True))[0]
        sims = query_res @ cls.T
        local = _topm_mean(sims, int(topm))
        proto = query_res @ prototype
        out[:, class_index] = (1.0 - float(proto_mix)) * local + float(proto_mix) * proto
        used_new += 1
    if used_new:
        new_indices = [idx for idx, label in enumerate(class_labels) if str(label) not in old_set]
        new_scores = out[:, new_indices]
        new_scores = new_scores - np.mean(new_scores, axis=1, keepdims=True)
        new_scores = new_scores / (np.std(new_scores, axis=1, keepdims=True) + 1e-6)
        new_scores = np.clip(new_scores, -float(clip), float(clip))
        out[:, new_indices] = new_scores
    stored_scalars = int(basis.size + old_center.size + used_new * features.shape[1])
    return out, int(max_rank), int(stored_scalars)


def _class_diag_metric_scores(
    *,
    features: np.ndarray,
    support_indices: np.ndarray,
    support_labels: np.ndarray,
    query_indices: np.ndarray,
    class_labels: list[str],
    topm: int,
    proto_mix: float,
    similarity_threshold: float,
    alpha: float,
    power: float,
    clip: float,
) -> tuple[np.ndarray, int, int]:
    support = qknn._normalize_rows(features[support_indices])
    query = qknn._normalize_rows(features[query_indices])
    labels = np.asarray(support_labels, dtype=object).astype(str)
    means: list[np.ndarray] = []
    variances: list[np.ndarray] = []
    valid: list[bool] = []
    for label in class_labels:
        cls = support[labels == label]
        if cls.size == 0:
            means.append(np.zeros(support.shape[1], dtype=np.float64))
            variances.append(np.ones(support.shape[1], dtype=np.float64))
            valid.append(False)
            continue
        means.append(np.mean(cls, axis=0))
        variances.append(np.var(cls, axis=0) + 1e-6)
        valid.append(True)
    mean_matrix = qknn._normalize_rows(np.stack(means, axis=0))
    var_matrix = np.stack(variances, axis=0)
    proto_sim = mean_matrix @ mean_matrix.T
    score_columns: list[np.ndarray] = []
    metric_count = 0
    for class_index, label in enumerate(class_labels):
        cls = support[labels == label]
        if cls.size == 0:
            score_columns.append(np.full(query.shape[0], -1e9, dtype=np.float64))
            continue
        competitors = np.where(proto_sim[class_index] >= float(similarity_threshold))[0]
        competitors = competitors[competitors != class_index]
        competitors = np.asarray([idx for idx in competitors.tolist() if valid[idx]], dtype=int)
        if competitors.size == 0:
            order = np.argsort(-proto_sim[class_index])
            competitors = np.asarray([idx for idx in order.tolist() if idx != class_index and valid[idx]][:1], dtype=int)
        if competitors.size == 0:
            weights = np.ones(support.shape[1], dtype=np.float64)
        else:
            diff2 = (mean_matrix[class_index][None, :] - mean_matrix[competitors]) ** 2
            denom = var_matrix[class_index][None, :] + var_matrix[competitors]
            fisher = np.mean(diff2 / (denom + float(alpha)), axis=0)
            fisher = fisher / (float(np.mean(fisher)) + 1e-12)
            weights = np.power(np.maximum(fisher, 1e-6), float(power))
            weights = np.clip(weights, 0.05, 20.0)
            weights = weights / (float(np.mean(weights)) + 1e-12)
        sqrt_w = np.sqrt(weights)
        cls_w = qknn._normalize_rows(cls * sqrt_w[None, :])
        query_w = qknn._normalize_rows(query * sqrt_w[None, :])
        proto = qknn._normalize_rows(cls_w.mean(axis=0, keepdims=True))[0]
        local = _topm_mean(query_w @ cls_w.T, int(topm))
        proto_score = query_w @ proto
        score_columns.append((1.0 - float(proto_mix)) * local + float(proto_mix) * proto_score)
        metric_count += 1
    scores = np.stack(score_columns, axis=1)
    scores = scores - np.mean(scores, axis=1, keepdims=True)
    scores = scores / (np.std(scores, axis=1, keepdims=True) + 1e-6)
    scores = np.clip(scores, -float(clip), float(clip))
    stored_scalars = int(metric_count * support.shape[1])
    return scores, int(metric_count), int(stored_scalars)


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


def _support_loo_accuracy_summary(
    support_scores: np.ndarray,
    *,
    support_labels: np.ndarray,
    old_labels: list[str],
    new_labels: list[str],
) -> tuple[float, float]:
    labels = np.asarray(support_labels, dtype=object).astype(str)
    class_labels = list(old_labels) + list(new_labels)
    old_set = set(old_labels)
    old_count = int(sum(1 for label in labels.tolist() if label in old_set))
    pred = _assignment_predict(
        np.asarray(support_scores, dtype=np.float64),
        old_count=old_count,
        old_labels=old_labels,
        new_labels=new_labels,
        query_scenarios=np.asarray(["support"] * int(labels.size), dtype=object),
        scenario_balanced_assignment=False,
        role_balanced_assignment=True,
        balanced_assignment=True,
    )
    per_class: list[float] = []
    for label in class_labels:
        mask = labels == label
        if bool(np.any(mask)):
            per_class.append(float(np.mean(pred[mask] == label)))
    if not per_class:
        return 0.0, 0.0
    return float(np.mean(pred == labels)), float(min(per_class))


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


def _support_loo_pair_rescue_scores(
    scores: np.ndarray,
    *,
    features: np.ndarray,
    support_indices: np.ndarray,
    support_labels: np.ndarray,
    query_indices: np.ndarray,
    support_scores: np.ndarray,
    class_labels: list[str],
    old_labels: list[str],
    new_labels: list[str],
    top_pairs: int,
    min_errors: int,
    weight: float,
    alpha: float,
    clip: float,
    scope: str,
    proto_neighbors: int,
    proto_min_sim: float,
) -> tuple[np.ndarray, int, int, float, float, str]:
    if float(weight) == 0.0 or int(top_pairs) <= 0:
        return scores, 0, 0, 0.0, 0.0, ""
    scope_norm = str(scope).strip().lower()
    if scope_norm not in {"all", "role", "new"}:
        raise ValueError(f"unsupported support_loo_pair_rescue_scope: {scope}")
    labels = np.asarray(support_labels, dtype=object).astype(str)
    label_to_index = {label: index for index, label in enumerate(class_labels)}
    old_set = set(old_labels)
    new_set = set(new_labels)
    old_count = int(sum(1 for label in labels.tolist() if label in old_set))
    loo_pred = _assignment_predict(
        np.asarray(support_scores, dtype=np.float64),
        old_count=old_count,
        old_labels=old_labels,
        new_labels=new_labels,
        query_scenarios=np.asarray(["support"] * int(labels.size), dtype=object),
        scenario_balanced_assignment=False,
        role_balanced_assignment=True,
        balanced_assignment=True,
    )
    before_per_class: list[float] = []
    before_acc_by_label: dict[str, float] = {}
    for label in class_labels:
        mask = labels == label
        acc = float(np.mean(loo_pred[mask] == label)) if bool(np.any(mask)) else 0.0
        before_per_class.append(acc)
        before_acc_by_label[label] = acc

    pair_counts: dict[tuple[str, str], int] = {}
    runner_counts: dict[tuple[str, str], int] = {}
    for truth_label, pred_label in zip(labels.tolist(), loo_pred.tolist()):
        truth = str(truth_label)
        pred = str(pred_label)
        if truth not in label_to_index or pred not in label_to_index:
            continue
        if scope_norm == "new" and (truth not in new_set or pred not in new_set):
            continue
        if scope_norm == "role" and ((truth in old_set) != (pred in old_set)):
            continue
        if truth != pred:
            pair_counts[(truth, pred)] = pair_counts.get((truth, pred), 0) + 1
    use_legacy_candidates = int(proto_neighbors) <= 0 and float(proto_min_sim) > 1.0
    if use_legacy_candidates:
        candidates = [
            (truth, pred, count, float(count))
            for (truth, pred), count in pair_counts.items()
            if int(count) >= max(1, int(min_errors))
        ]
        candidates.sort(key=lambda item: (-int(item[2]), item[0], item[1]))
        candidates = candidates[: max(0, int(top_pairs))]
        if not candidates:
            return (
                scores,
                0,
                0,
                float(min(before_per_class, default=0.0)),
                float(np.mean(before_per_class) if before_per_class else 0.0),
                "",
            )
        support = qknn._normalize_rows(features[support_indices])
        query = qknn._normalize_rows(features[query_indices])
        prototypes: list[np.ndarray] = []
        for label in class_labels:
            class_support = support[labels == label]
            if class_support.size == 0:
                prototypes.append(np.zeros(features.shape[1], dtype=np.float64))
            else:
                prototypes.append(qknn._normalize_rows(class_support.mean(axis=0, keepdims=True))[0])
        proto = qknn._normalize_rows(np.stack(prototypes, axis=0))
        adjusted = np.asarray(scores, dtype=np.float64).copy()
        rescue_scores = np.asarray(support_scores, dtype=np.float64).copy()
        used = 0
        stored_scalars = 0
        used_pairs: list[str] = []
        for truth, pred, _count, _score in candidates:
            truth_idx = label_to_index[truth]
            pred_idx = label_to_index[pred]
            pair_mask = (labels == truth) | (labels == pred)
            pair_support = support[pair_mask]
            pair_labels = labels[pair_mask]
            if pair_support.shape[0] < 4 or len(set(pair_labels.tolist())) < 2:
                continue
            truth_sim = pair_support @ proto[truth_idx]
            pred_sim = pair_support @ proto[pred_idx]
            x = np.stack(
                [
                    truth_sim - pred_sim,
                    truth_sim,
                    pred_sim,
                    np.ones_like(truth_sim),
                ],
                axis=1,
            )
            y = np.where(pair_labels == truth, 1.0, -1.0)
            reg = float(alpha) * np.eye(x.shape[1], dtype=np.float64)
            reg[-1, -1] = float(alpha) * 0.01
            try:
                coeff = np.linalg.solve(x.T @ x + reg, x.T @ y)
            except np.linalg.LinAlgError:
                coeff = np.linalg.pinv(x.T @ x + reg) @ x.T @ y
            q_truth = query @ proto[truth_idx]
            q_pred = query @ proto[pred_idx]
            qx = np.stack([q_truth - q_pred, q_truth, q_pred, np.ones_like(q_truth)], axis=1)
            margin = np.clip(qx @ coeff, -float(clip), float(clip))
            adjusted[:, truth_idx] += float(weight) * margin
            adjusted[:, pred_idx] -= float(weight) * margin

            s_truth = support @ proto[truth_idx]
            s_pred = support @ proto[pred_idx]
            sx = np.stack([s_truth - s_pred, s_truth, s_pred, np.ones_like(s_truth)], axis=1)
            s_margin = np.clip(sx @ coeff, -float(clip), float(clip))
            rescue_scores[:, truth_idx] += float(weight) * s_margin
            rescue_scores[:, pred_idx] -= float(weight) * s_margin
            used += 1
            used_pairs.append(f"{truth}->{pred}")
            stored_scalars += int(coeff.size)
        after_pred = _assignment_predict(
            rescue_scores,
            old_count=old_count,
            old_labels=old_labels,
            new_labels=new_labels,
            query_scenarios=np.asarray(["support"] * int(labels.size), dtype=object),
            scenario_balanced_assignment=False,
            role_balanced_assignment=True,
            balanced_assignment=True,
        )
        after_per_class: list[float] = []
        for label in class_labels:
            mask = labels == label
            after_per_class.append(float(np.mean(after_pred[mask] == label)) if bool(np.any(mask)) else 0.0)
        return (
            adjusted,
            int(used),
            int(stored_scalars),
            float(min(after_per_class, default=0.0)),
            float(np.mean(after_per_class) if after_per_class else 0.0),
            ";".join(used_pairs[:32]),
        )
    for row_idx, truth_label in enumerate(labels.tolist()):
        truth = str(truth_label)
        if truth not in label_to_index:
            continue
        truth_idx = label_to_index[truth]
        score_row = np.asarray(support_scores[row_idx], dtype=np.float64)
        ranked = np.argsort(score_row)[::-1]
        for idx in ranked.tolist():
            competitor = class_labels[int(idx)]
            if int(idx) == truth_idx:
                continue
            if scope_norm == "new" and (truth not in new_set or competitor not in new_set):
                continue
            if scope_norm == "role" and ((truth in old_set) != (competitor in old_set)):
                continue
            runner_counts[(truth, competitor)] = runner_counts.get((truth, competitor), 0) + 1
            break

    support = qknn._normalize_rows(features[support_indices])
    query = qknn._normalize_rows(features[query_indices])
    prototypes: list[np.ndarray] = []
    valid_proto: list[bool] = []
    for label in class_labels:
        cls = support[labels == label]
        if cls.size == 0:
            prototypes.append(np.zeros(support.shape[1], dtype=np.float64))
            valid_proto.append(False)
        else:
            prototypes.append(qknn._normalize_rows(cls.mean(axis=0, keepdims=True))[0])
            valid_proto.append(True)
    proto = qknn._normalize_rows(np.stack(prototypes, axis=0))
    proto_sim = proto @ proto.T

    class_risk: dict[str, float] = {}
    for label in class_labels:
        if scope_norm == "new" and label not in new_set:
            continue
        class_idx = label_to_index[label]
        class_mask = labels == label
        if not bool(np.any(class_mask)):
            continue
        allowed = [
            idx
            for idx, other in enumerate(class_labels)
            if idx != class_idx
            and not (scope_norm == "new" and other not in new_set)
            and not (scope_norm == "role" and ((label in old_set) != (other in old_set)))
        ]
        if not allowed:
            continue
        cls_scores = np.asarray(support_scores[class_mask], dtype=np.float64)
        margin = cls_scores[:, class_idx] - np.max(cls_scores[:, allowed], axis=1)
        max_sim = float(np.max(proto_sim[class_idx, allowed])) if valid_proto[class_idx] else 0.0
        risk = (
            2.0 * max(0.0, 1.0 - before_acc_by_label.get(label, 0.0))
            + 3.0 * max(0.0, 0.08 - float(np.mean(margin)))
            + 1.5 * max(0.0, max_sim - float(proto_min_sim))
        )
        class_risk[label] = float(risk)

    pair_scores: dict[tuple[str, str], float] = {}
    for key, count in pair_counts.items():
        if class_risk.get(key[0], 0.0) + 0.05 < class_risk.get(key[1], 0.0):
            continue
        pair_scores[key] = max(pair_scores.get(key, 0.0), 4.0 * float(count) + class_risk.get(key[0], 0.0))
    for key, count in runner_counts.items():
        if class_risk.get(key[0], 0.0) + 0.05 < class_risk.get(key[1], 0.0):
            continue
        pair_scores[key] = max(pair_scores.get(key, 0.0), 1.0 * float(count) + class_risk.get(key[0], 0.0))
    if int(proto_neighbors) > 0:
        for truth in class_labels:
            if truth not in label_to_index:
                continue
            if scope_norm == "new" and truth not in new_set:
                continue
            truth_idx = label_to_index[truth]
            if not valid_proto[truth_idx]:
                continue
            allowed = [
                idx
                for idx, other in enumerate(class_labels)
                if idx != truth_idx
                and valid_proto[idx]
                and not (scope_norm == "new" and other not in new_set)
                and not (scope_norm == "role" and ((truth in old_set) != (other in old_set)))
            ]
            ordered = sorted(allowed, key=lambda idx: float(proto_sim[truth_idx, idx]), reverse=True)
            used_neighbors = 0
            for pred_idx in ordered:
                sim = float(proto_sim[truth_idx, pred_idx])
                if sim < float(proto_min_sim):
                    continue
                pred = class_labels[pred_idx]
                key = (truth, pred)
                if class_risk.get(truth, 0.0) + 0.05 < class_risk.get(pred, 0.0):
                    continue
                score = class_risk.get(truth, 0.0) + 2.0 * sim
                pair_scores[key] = max(pair_scores.get(key, 0.0), score)
                used_neighbors += 1
                if used_neighbors >= int(proto_neighbors):
                    break

    candidates = [
        (truth, pred, int(pair_counts.get((truth, pred), 0)), float(score))
        for (truth, pred), score in pair_scores.items()
        if int(pair_counts.get((truth, pred), 0)) >= max(1, int(min_errors)) or float(score) > 0.0
    ]
    candidates.sort(key=lambda item: (-float(item[3]), item[0], item[1]))
    candidates = candidates[: max(0, int(top_pairs))]
    if not candidates:
        return scores, 0, 0, float(min(before_per_class, default=0.0)), float(np.mean(before_per_class) if before_per_class else 0.0), ""
    adjusted = np.asarray(scores, dtype=np.float64).copy()
    rescue_scores = np.asarray(support_scores, dtype=np.float64).copy()
    used = 0
    stored_scalars = 0
    used_pairs: list[str] = []
    for truth, pred, _count, _score in candidates:
        truth_idx = label_to_index[truth]
        pred_idx = label_to_index[pred]
        pair_mask = (labels == truth) | (labels == pred)
        pair_support = support[pair_mask]
        pair_labels = labels[pair_mask]
        if pair_support.shape[0] < 4 or len(set(pair_labels.tolist())) < 2:
            continue
        truth_sim = pair_support @ proto[truth_idx]
        pred_sim = pair_support @ proto[pred_idx]
        x = np.stack(
            [
                truth_sim - pred_sim,
                truth_sim,
                pred_sim,
                np.ones_like(truth_sim),
            ],
            axis=1,
        )
        y = np.where(pair_labels == truth, 1.0, -1.0)
        reg = float(alpha) * np.eye(x.shape[1], dtype=np.float64)
        reg[-1, -1] = float(alpha) * 0.01
        try:
            coeff = np.linalg.solve(x.T @ x + reg, x.T @ y)
        except np.linalg.LinAlgError:
            coeff = np.linalg.pinv(x.T @ x + reg) @ x.T @ y
        q_truth = query @ proto[truth_idx]
        q_pred = query @ proto[pred_idx]
        qx = np.stack([q_truth - q_pred, q_truth, q_pred, np.ones_like(q_truth)], axis=1)
        margin = np.clip(qx @ coeff, -float(clip), float(clip))
        adjusted[:, truth_idx] += float(weight) * margin
        adjusted[:, pred_idx] -= float(weight) * margin

        s_truth = support @ proto[truth_idx]
        s_pred = support @ proto[pred_idx]
        sx = np.stack([s_truth - s_pred, s_truth, s_pred, np.ones_like(s_truth)], axis=1)
        s_margin = np.clip(sx @ coeff, -float(clip), float(clip))
        rescue_scores[:, truth_idx] += float(weight) * s_margin
        rescue_scores[:, pred_idx] -= float(weight) * s_margin
        used += 1
        used_pairs.append(f"{truth}->{pred}")
        stored_scalars += int(coeff.size)

    after_pred = _assignment_predict(
        rescue_scores,
        old_count=old_count,
        old_labels=old_labels,
        new_labels=new_labels,
        query_scenarios=np.asarray(["support"] * int(labels.size), dtype=object),
        scenario_balanced_assignment=False,
        role_balanced_assignment=True,
        balanced_assignment=True,
    )
    after_per_class: list[float] = []
    for label in class_labels:
        mask = labels == label
        after_per_class.append(float(np.mean(after_pred[mask] == label)) if bool(np.any(mask)) else 0.0)
    return (
        adjusted,
        int(used),
        int(stored_scalars),
        float(min(after_per_class, default=0.0)),
        float(np.mean(after_per_class) if after_per_class else 0.0),
        ";".join(used_pairs[:32]),
    )


def _support_loo_pair_linear_scores(
    scores: np.ndarray,
    *,
    features: np.ndarray,
    support_indices: np.ndarray,
    support_labels: np.ndarray,
    query_indices: np.ndarray,
    support_scores: np.ndarray,
    class_labels: list[str],
    old_labels: list[str],
    new_labels: list[str],
    top_pairs: int,
    min_errors: int,
    weight: float,
    alpha: float,
    clip: float,
    scope: str,
) -> tuple[np.ndarray, int, int, float, float]:
    if float(weight) == 0.0 or int(top_pairs) <= 0:
        return scores, 0, 0, 0.0, 0.0
    scope_norm = str(scope).strip().lower()
    if scope_norm not in {"all", "role", "new"}:
        raise ValueError(f"unsupported support_loo_pair_linear_scope: {scope}")
    labels = np.asarray(support_labels, dtype=object).astype(str)
    label_to_index = {label: index for index, label in enumerate(class_labels)}
    old_set = set(old_labels)
    new_set = set(new_labels)
    old_count = int(sum(1 for label in labels.tolist() if label in old_set))
    loo_pred = _assignment_predict(
        np.asarray(support_scores, dtype=np.float64),
        old_count=old_count,
        old_labels=old_labels,
        new_labels=new_labels,
        query_scenarios=np.asarray(["support"] * int(labels.size), dtype=object),
        scenario_balanced_assignment=False,
        role_balanced_assignment=True,
        balanced_assignment=True,
    )
    before_per_class: list[float] = []
    for label in class_labels:
        mask = labels == label
        before_per_class.append(float(np.mean(loo_pred[mask] == label)) if bool(np.any(mask)) else 0.0)

    pair_counts: dict[tuple[str, str], int] = {}
    for truth_label, pred_label in zip(labels.tolist(), loo_pred.tolist()):
        truth = str(truth_label)
        pred = str(pred_label)
        if truth == pred:
            continue
        if truth not in label_to_index or pred not in label_to_index:
            continue
        if scope_norm == "new" and (truth not in new_set or pred not in new_set):
            continue
        if scope_norm == "role" and ((truth in old_set) != (pred in old_set)):
            continue
        left, right = sorted((truth, pred))
        pair_counts[(left, right)] = pair_counts.get((left, right), 0) + 1
    candidates = [
        (left, right, count)
        for (left, right), count in pair_counts.items()
        if int(count) >= max(1, int(min_errors))
    ]
    candidates.sort(key=lambda item: (-int(item[2]), item[0], item[1]))
    candidates = candidates[: max(0, int(top_pairs))]
    if not candidates:
        return scores, 0, 0, float(min(before_per_class, default=0.0)), float(np.mean(before_per_class) if before_per_class else 0.0)

    support = qknn._normalize_rows(features[support_indices])
    query = qknn._normalize_rows(features[query_indices])
    adjusted = np.asarray(scores, dtype=np.float64).copy()
    linear_support_scores = np.asarray(support_scores, dtype=np.float64).copy()
    used = 0
    stored_scalars = 0
    alpha_value = max(float(alpha), 1e-8)
    for left, right, _count in candidates:
        left_idx = label_to_index[left]
        right_idx = label_to_index[right]
        pair_mask = (labels == left) | (labels == right)
        pair_support = support[pair_mask]
        pair_labels = labels[pair_mask]
        if pair_support.shape[0] < 4 or len(set(pair_labels.tolist())) < 2:
            continue
        x = np.concatenate(
            [pair_support, np.ones((pair_support.shape[0], 1), dtype=np.float64)],
            axis=1,
        )
        y = np.where(pair_labels == left, 1.0, -1.0).astype(np.float64)
        # The learned state is only the compact pair boundary coeff, not raw support samples.
        gram = x @ x.T + alpha_value * np.eye(x.shape[0], dtype=np.float64)
        try:
            dual = np.linalg.solve(gram, y)
        except np.linalg.LinAlgError:
            dual = np.linalg.pinv(gram) @ y
        coeff = x.T @ dual
        qx = np.concatenate([query, np.ones((query.shape[0], 1), dtype=np.float64)], axis=1)
        margin = np.clip(qx @ coeff, -float(clip), float(clip))
        adjusted[:, left_idx] += float(weight) * margin
        adjusted[:, right_idx] -= float(weight) * margin

        sx = np.concatenate([support, np.ones((support.shape[0], 1), dtype=np.float64)], axis=1)
        support_margin = np.clip(sx @ coeff, -float(clip), float(clip))
        linear_support_scores[:, left_idx] += float(weight) * support_margin
        linear_support_scores[:, right_idx] -= float(weight) * support_margin
        used += 1
        stored_scalars += int(coeff.size)

    after_pred = _assignment_predict(
        linear_support_scores,
        old_count=old_count,
        old_labels=old_labels,
        new_labels=new_labels,
        query_scenarios=np.asarray(["support"] * int(labels.size), dtype=object),
        scenario_balanced_assignment=False,
        role_balanced_assignment=True,
        balanced_assignment=True,
    )
    after_per_class: list[float] = []
    for label in class_labels:
        mask = labels == label
        after_per_class.append(float(np.mean(after_pred[mask] == label)) if bool(np.any(mask)) else 0.0)
    return (
        adjusted,
        int(used),
        int(stored_scalars),
        float(min(after_per_class, default=0.0)),
        float(np.mean(after_per_class) if after_per_class else 0.0),
    )


def _support_loo_top2_pair_gate_scores(
    scores: np.ndarray,
    *,
    features: np.ndarray,
    support_indices: np.ndarray,
    support_labels: np.ndarray,
    query_indices: np.ndarray,
    support_scores: np.ndarray,
    class_labels: list[str],
    old_labels: list[str],
    new_labels: list[str],
    old_query_count: int,
    top_pairs: int,
    min_errors: int,
    weight: float,
    alpha: float,
    clip: float,
    gate_margin: float,
    query_pair_weight: float,
) -> tuple[np.ndarray, int, int, float, float, str]:
    if float(weight) == 0.0 or int(top_pairs) <= 0:
        return scores, 0, 0, 0.0, 0.0, ""
    labels = np.asarray(support_labels, dtype=object).astype(str)
    label_to_index = {label: index for index, label in enumerate(class_labels)}
    old_set = set(old_labels)
    new_set = set(new_labels)
    old_count = int(sum(1 for label in labels.tolist() if label in old_set))
    loo_pred = _assignment_predict(
        np.asarray(support_scores, dtype=np.float64),
        old_count=old_count,
        old_labels=old_labels,
        new_labels=new_labels,
        query_scenarios=np.asarray(["support"] * int(labels.size), dtype=object),
        scenario_balanced_assignment=False,
        role_balanced_assignment=True,
        balanced_assignment=True,
    )
    before_per_class: list[float] = []
    for label in class_labels:
        mask = labels == label
        before_per_class.append(float(np.mean(loo_pred[mask] == label)) if bool(np.any(mask)) else 0.0)

    new_indices = np.asarray([label_to_index[label] for label in new_labels if label in label_to_index], dtype=int)
    if new_indices.size < 2:
        return scores, 0, 0, float(min(before_per_class, default=0.0)), float(np.mean(before_per_class) if before_per_class else 0.0), ""

    pair_counts: dict[tuple[str, str], float] = {}
    for truth_label, pred_label in zip(labels.tolist(), loo_pred.tolist()):
        truth = str(truth_label)
        pred = str(pred_label)
        if truth == pred or truth not in new_set or pred not in new_set:
            continue
        left, right = sorted((truth, pred))
        pair_counts[(left, right)] = pair_counts.get((left, right), 0.0) + 1.0
    query_new_rows = np.arange(int(old_query_count), int(scores.shape[0]), dtype=int)
    if float(query_pair_weight) > 0.0 and query_new_rows.size:
        local = np.asarray(scores, dtype=np.float64)[query_new_rows][:, new_indices]
        order = np.argsort(local, axis=1)[:, -2:]
        top_pair = np.sort(new_indices[order], axis=1)
        pair_gap = np.abs(local[np.arange(local.shape[0]), order[:, 1]] - local[np.arange(local.shape[0]), order[:, 0]])
        gate = max(float(gate_margin), 1e-8)
        keep = pair_gap <= (1.5 * gate)
        index_to_label = {index: label for label, index in label_to_index.items()}
        for left_idx, right_idx in top_pair[keep].tolist():
            left = index_to_label.get(int(left_idx))
            right = index_to_label.get(int(right_idx))
            if left is None or right is None:
                continue
            if left == right or left not in new_set or right not in new_set:
                continue
            pair = tuple(sorted((left, right)))
            pair_counts[pair] = pair_counts.get(pair, 0.0) + float(query_pair_weight)
    candidates = [
        (left, right, count)
        for (left, right), count in pair_counts.items()
        if float(count) >= float(max(1, int(min_errors)))
    ]
    candidates.sort(key=lambda item: (-float(item[2]), item[0], item[1]))
    candidates = candidates[: max(0, int(top_pairs))]
    if not candidates:
        return scores, 0, 0, float(min(before_per_class, default=0.0)), float(np.mean(before_per_class) if before_per_class else 0.0), ""

    support = qknn._normalize_rows(features[support_indices])
    query = qknn._normalize_rows(features[query_indices])
    prototypes: list[np.ndarray] = []
    for label in class_labels:
        class_support = support[labels == label]
        if class_support.size == 0:
            prototypes.append(np.zeros(features.shape[1], dtype=np.float64))
        else:
            prototypes.append(qknn._normalize_rows(class_support.mean(axis=0, keepdims=True))[0])
    proto = qknn._normalize_rows(np.stack(prototypes, axis=0))

    adjusted = np.asarray(scores, dtype=np.float64).copy()
    gate_support_scores = np.asarray(support_scores, dtype=np.float64).copy()
    support_new_rows = np.where(np.asarray([label in new_set for label in labels.tolist()], dtype=bool))[0]
    alpha_value = max(float(alpha), 1e-8)
    gate = max(float(gate_margin), 1e-8)
    used = 0
    stored_scalars = 0
    used_pairs: list[str] = []

    def gated_rows(score_matrix: np.ndarray, rows: np.ndarray, left_idx: int, right_idx: int) -> tuple[np.ndarray, np.ndarray]:
        if rows.size == 0:
            return rows, np.asarray([], dtype=np.float64)
        local = score_matrix[rows][:, new_indices]
        order = np.argsort(local, axis=1)[:, -2:]
        top_pair = np.sort(new_indices[order], axis=1)
        pair = np.asarray(sorted((left_idx, right_idx)), dtype=int)
        pair_mask = np.all(top_pair == pair[None, :], axis=1)
        pair_gap = np.abs(score_matrix[rows, left_idx] - score_matrix[rows, right_idx])
        margin_mask = pair_gap <= gate
        keep = pair_mask & margin_mask
        kept_rows = rows[keep]
        if kept_rows.size == 0:
            return kept_rows, np.asarray([], dtype=np.float64)
        factors = 1.0 - np.clip(pair_gap[keep] / gate, 0.0, 1.0)
        return kept_rows, factors.astype(np.float64)

    for left, right, _count in candidates:
        left_idx = label_to_index[left]
        right_idx = label_to_index[right]
        pair_mask = (labels == left) | (labels == right)
        pair_support = support[pair_mask]
        pair_labels = labels[pair_mask]
        if pair_support.shape[0] < 4 or len(set(pair_labels.tolist())) < 2:
            continue
        left_sim = pair_support @ proto[left_idx]
        right_sim = pair_support @ proto[right_idx]
        x = np.stack([left_sim - right_sim, left_sim, right_sim, np.ones_like(left_sim)], axis=1)
        y = np.where(pair_labels == left, 1.0, -1.0).astype(np.float64)
        reg = alpha_value * np.eye(x.shape[1], dtype=np.float64)
        reg[-1, -1] = alpha_value * 0.01
        try:
            coeff = np.linalg.solve(x.T @ x + reg, x.T @ y)
        except np.linalg.LinAlgError:
            coeff = np.linalg.pinv(x.T @ x + reg) @ x.T @ y

        rows, factors = gated_rows(adjusted, query_new_rows, left_idx, right_idx)
        if rows.size:
            q_left = query[rows] @ proto[left_idx]
            q_right = query[rows] @ proto[right_idx]
            qx = np.stack([q_left - q_right, q_left, q_right, np.ones_like(q_left)], axis=1)
            margin = np.clip(qx @ coeff, -float(clip), float(clip)) * factors
            adjusted[rows, left_idx] += float(weight) * margin
            adjusted[rows, right_idx] -= float(weight) * margin

        s_rows, s_factors = gated_rows(gate_support_scores, support_new_rows, left_idx, right_idx)
        if s_rows.size:
            s_left = support[s_rows] @ proto[left_idx]
            s_right = support[s_rows] @ proto[right_idx]
            sx = np.stack([s_left - s_right, s_left, s_right, np.ones_like(s_left)], axis=1)
            s_margin = np.clip(sx @ coeff, -float(clip), float(clip)) * s_factors
            gate_support_scores[s_rows, left_idx] += float(weight) * s_margin
            gate_support_scores[s_rows, right_idx] -= float(weight) * s_margin
        used += 1
        used_pairs.append(f"{left}<->{right}")
        stored_scalars += int(coeff.size + 2)

    after_pred = _assignment_predict(
        gate_support_scores,
        old_count=old_count,
        old_labels=old_labels,
        new_labels=new_labels,
        query_scenarios=np.asarray(["support"] * int(labels.size), dtype=object),
        scenario_balanced_assignment=False,
        role_balanced_assignment=True,
        balanced_assignment=True,
    )
    after_per_class: list[float] = []
    for label in class_labels:
        mask = labels == label
        after_per_class.append(float(np.mean(after_pred[mask] == label)) if bool(np.any(mask)) else 0.0)
    return (
        adjusted,
        int(used),
        int(stored_scalars),
        float(min(after_per_class, default=0.0)),
        float(np.mean(after_per_class) if after_per_class else 0.0),
        ";".join(used_pairs[:32]),
    )


def _support_query_neighborhood_gate_scores(
    scores: np.ndarray,
    *,
    features: np.ndarray,
    support_indices: np.ndarray,
    support_labels: np.ndarray,
    query_indices: np.ndarray,
    support_scores: np.ndarray,
    class_labels: list[str],
    old_labels: list[str],
    new_labels: list[str],
    old_query_count: int,
    top_classes: int,
    neighbor_count: int,
    query_topm: int,
    weight: float,
    alpha: float,
    clip: float,
    gate_margin: float,
    query_neighbor_weight: float,
) -> tuple[np.ndarray, int, int, float, float, str]:
    """Compressed one-vs-neighborhood gate for hard new classes.

    Candidate classes come from support LOO errors plus unlabeled query top-M
    ambiguity. The fitted state is a small ridge boundary over prototype
    similarity features; no raw support or query samples are persisted.
    """
    if float(weight) == 0.0 or int(top_classes) <= 0 or int(neighbor_count) <= 0:
        return scores, 0, 0, 0.0, 0.0, ""
    labels = np.asarray(support_labels, dtype=object).astype(str)
    label_to_index = {label: index for index, label in enumerate(class_labels)}
    new_set = set(new_labels)
    old_set = set(old_labels)
    old_support_count = int(sum(1 for label in labels.tolist() if label in old_set))
    loo_pred = _assignment_predict(
        np.asarray(support_scores, dtype=np.float64),
        old_count=old_support_count,
        old_labels=old_labels,
        new_labels=new_labels,
        query_scenarios=np.asarray(["support"] * int(labels.size), dtype=object),
        scenario_balanced_assignment=False,
        role_balanced_assignment=True,
        balanced_assignment=True,
    )
    before_per_class: list[float] = []
    for label in class_labels:
        mask = labels == label
        before_per_class.append(float(np.mean(loo_pred[mask] == label)) if bool(np.any(mask)) else 0.0)

    new_indices = np.asarray([label_to_index[label] for label in new_labels if label in label_to_index], dtype=int)
    if new_indices.size < 2:
        return scores, 0, 0, float(min(before_per_class, default=0.0)), float(np.mean(before_per_class) if before_per_class else 0.0), ""

    risk: dict[str, float] = {label: 0.0 for label in new_labels}
    neighbor_votes: dict[str, dict[str, float]] = {label: {} for label in new_labels}
    for truth_label, pred_label in zip(labels.tolist(), loo_pred.tolist()):
        truth = str(truth_label)
        pred = str(pred_label)
        if truth == pred or truth not in new_set or pred not in new_set:
            continue
        risk[truth] = risk.get(truth, 0.0) + 1.0
        votes = neighbor_votes.setdefault(truth, {})
        votes[pred] = votes.get(pred, 0.0) + 1.0

    query_new_rows = np.arange(int(old_query_count), int(scores.shape[0]), dtype=int)
    gate = max(float(gate_margin), 1e-8)
    local_topm = max(2, min(int(query_topm), int(new_indices.size)))
    index_to_label = {index: label for label, index in label_to_index.items()}
    if float(query_neighbor_weight) > 0.0 and query_new_rows.size:
        local = np.asarray(scores, dtype=np.float64)[query_new_rows][:, new_indices]
        order = np.argsort(local, axis=1)[:, -local_topm:][:, ::-1]
        top_global = new_indices[order]
        top_scores = np.take_along_axis(local, order, axis=1)
        ambiguity = 1.0 - np.clip((top_scores[:, 0] - top_scores[:, -1]) / (2.0 * gate), 0.0, 1.0)
        for row_labels, row_ambiguity in zip(top_global.tolist(), ambiguity.tolist()):
            if float(row_ambiguity) <= 0.0:
                continue
            labels_top = [index_to_label.get(int(idx), "") for idx in row_labels]
            labels_top = [label for label in labels_top if label in new_set]
            for position, target in enumerate(labels_top):
                position_weight = 1.0 / float(position + 1)
                risk[target] = risk.get(target, 0.0) + float(query_neighbor_weight) * float(row_ambiguity) * position_weight
                votes = neighbor_votes.setdefault(target, {})
                for neighbor in labels_top:
                    if neighbor == target:
                        continue
                    votes[neighbor] = votes.get(neighbor, 0.0) + (
                        float(query_neighbor_weight) * float(row_ambiguity) / max(1.0, float(len(labels_top) - 1))
                    )

    support = qknn._normalize_rows(features[support_indices])
    query = qknn._normalize_rows(features[query_indices])
    prototypes: list[np.ndarray] = []
    for label in class_labels:
        class_support = support[labels == label]
        if class_support.size == 0:
            prototypes.append(np.zeros(features.shape[1], dtype=np.float64))
        else:
            prototypes.append(qknn._normalize_rows(class_support.mean(axis=0, keepdims=True))[0])
    proto = qknn._normalize_rows(np.stack(prototypes, axis=0))
    proto_sim = proto @ proto.T

    candidates = [(label, score) for label, score in risk.items() if float(score) > 0.0]
    candidates.sort(key=lambda item: (-float(item[1]), item[0]))
    candidates = candidates[: max(0, int(top_classes))]
    if not candidates:
        return scores, 0, 0, float(min(before_per_class, default=0.0)), float(np.mean(before_per_class) if before_per_class else 0.0), ""

    adjusted = np.asarray(scores, dtype=np.float64).copy()
    gate_support_scores = np.asarray(support_scores, dtype=np.float64).copy()
    support_new_rows = np.where(np.asarray([label in new_set for label in labels.tolist()], dtype=bool))[0]
    alpha_value = max(float(alpha), 1e-8)
    used = 0
    stored_scalars = 0
    used_neighborhoods: list[str] = []

    def support_audit(score_matrix: np.ndarray) -> tuple[dict[str, float], float, float]:
        pred = _assignment_predict(
            score_matrix,
            old_count=old_support_count,
            old_labels=old_labels,
            new_labels=new_labels,
            query_scenarios=np.asarray(["support"] * int(labels.size), dtype=object),
            scenario_balanced_assignment=False,
            role_balanced_assignment=True,
            balanced_assignment=True,
        )
        per_label: dict[str, float] = {}
        per_values: list[float] = []
        for class_label in class_labels:
            mask = labels == class_label
            value = float(np.mean(pred[mask] == class_label)) if bool(np.any(mask)) else 0.0
            per_label[class_label] = value
            per_values.append(value)
        return per_label, float(min(per_values, default=0.0)), float(np.mean(per_values) if per_values else 0.0)

    def choose_neighbors(target: str) -> list[str]:
        votes = neighbor_votes.get(target, {})
        ranked = [(neighbor, value) for neighbor, value in votes.items() if neighbor in new_set and neighbor != target]
        ranked.sort(key=lambda item: (-float(item[1]), item[0]))
        selected = [neighbor for neighbor, _value in ranked[: max(0, int(neighbor_count))]]
        if len(selected) >= int(neighbor_count):
            return selected
        target_idx = label_to_index[target]
        sim_rank = []
        for label in new_labels:
            if label == target or label in selected:
                continue
            sim_rank.append((label, float(proto_sim[target_idx, label_to_index[label]])))
        sim_rank.sort(key=lambda item: (-float(item[1]), item[0]))
        for label, _sim in sim_rank:
            selected.append(label)
            if len(selected) >= int(neighbor_count):
                break
        return selected

    def gated_rows(score_matrix: np.ndarray, rows: np.ndarray, target_idx: int, neighbor_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if rows.size == 0 or neighbor_indices.size == 0:
            return rows[:0], np.asarray([], dtype=int), np.asarray([], dtype=np.float64)
        local_new = score_matrix[rows][:, new_indices]
        order = np.argsort(local_new, axis=1)[:, -local_topm:]
        top_set = new_indices[order]
        in_top = np.any(top_set == int(target_idx), axis=1)
        for neighbor_idx in neighbor_indices.tolist():
            in_top = in_top | np.any(top_set == int(neighbor_idx), axis=1)
        neighbor_scores = score_matrix[rows][:, neighbor_indices]
        best_neighbor_local = np.argmax(neighbor_scores, axis=1)
        best_neighbor_idx = neighbor_indices[best_neighbor_local]
        gap = np.abs(score_matrix[rows, target_idx] - neighbor_scores[np.arange(neighbor_scores.shape[0]), best_neighbor_local])
        keep = in_top & (gap <= gate)
        kept_rows = rows[keep]
        if kept_rows.size == 0:
            return kept_rows, np.asarray([], dtype=int), np.asarray([], dtype=np.float64)
        factors = 1.0 - np.clip(gap[keep] / gate, 0.0, 1.0)
        return kept_rows, best_neighbor_idx[keep].astype(int), factors.astype(np.float64)

    for target, _risk_value in candidates:
        if target not in label_to_index:
            continue
        neighbors = choose_neighbors(target)
        if not neighbors:
            continue
        target_idx = label_to_index[target]
        neighbor_indices = np.asarray([label_to_index[label] for label in neighbors if label in label_to_index], dtype=int)
        if neighbor_indices.size == 0:
            continue
        train_mask = (labels == target) | np.isin(labels, np.asarray(neighbors, dtype=object))
        train_support = support[train_mask]
        train_labels = labels[train_mask]
        if train_support.shape[0] < 4 or len(set(train_labels.tolist())) < 2:
            continue
        target_sim = train_support @ proto[target_idx]
        neighbor_sim = train_support @ proto[neighbor_indices].T
        max_neighbor_sim = np.max(neighbor_sim, axis=1)
        mean_neighbor_sim = np.mean(neighbor_sim, axis=1)
        x = np.stack(
            [
                target_sim - max_neighbor_sim,
                target_sim,
                max_neighbor_sim,
                mean_neighbor_sim,
                np.ones_like(target_sim),
            ],
            axis=1,
        )
        y = np.where(train_labels == target, 1.0, -1.0).astype(np.float64)
        reg = alpha_value * np.eye(x.shape[1], dtype=np.float64)
        reg[-1, -1] = alpha_value * 0.01
        try:
            coeff = np.linalg.solve(x.T @ x + reg, x.T @ y)
        except np.linalg.LinAlgError:
            coeff = np.linalg.pinv(x.T @ x + reg) @ x.T @ y

        before_label_acc, before_floor, before_mean = support_audit(gate_support_scores)
        proposal_support_scores = gate_support_scores.copy()
        s_rows, s_best_neighbor_idx, s_factors = gated_rows(
            proposal_support_scores,
            support_new_rows,
            target_idx,
            neighbor_indices,
        )
        if s_rows.size:
            s_target = support[s_rows] @ proto[target_idx]
            s_neighbor = support[s_rows] @ proto[neighbor_indices].T
            s_max_neighbor = np.max(s_neighbor, axis=1)
            s_mean_neighbor = np.mean(s_neighbor, axis=1)
            sx = np.stack(
                [s_target - s_max_neighbor, s_target, s_max_neighbor, s_mean_neighbor, np.ones_like(s_target)],
                axis=1,
            )
            s_margin = np.maximum(np.clip(sx @ coeff, -float(clip), float(clip)), 0.0) * s_factors
            s_delta = float(weight) * s_margin
            proposal_support_scores[s_rows, target_idx] += s_delta
            proposal_support_scores[s_rows, s_best_neighbor_idx] -= s_delta
        after_label_acc, after_floor, after_mean = support_audit(proposal_support_scores)
        target_before = float(before_label_acc.get(target, 0.0))
        target_after = float(after_label_acc.get(target, 0.0))
        if (
            target_after <= target_before + 1e-9
            or after_mean + 1e-9 < before_mean
            or after_floor + 1e-9 < before_floor
        ):
            continue
        rows, best_neighbor_idx, factors = gated_rows(adjusted, query_new_rows, target_idx, neighbor_indices)
        if rows.size:
            q_target = query[rows] @ proto[target_idx]
            q_neighbor = query[rows] @ proto[neighbor_indices].T
            q_max_neighbor = np.max(q_neighbor, axis=1)
            q_mean_neighbor = np.mean(q_neighbor, axis=1)
            qx = np.stack(
                [q_target - q_max_neighbor, q_target, q_max_neighbor, q_mean_neighbor, np.ones_like(q_target)],
                axis=1,
            )
            margin = np.maximum(np.clip(qx @ coeff, -float(clip), float(clip)), 0.0) * factors
            delta = float(weight) * margin
            adjusted[rows, target_idx] += delta
            adjusted[rows, best_neighbor_idx] -= delta
        gate_support_scores = proposal_support_scores
        used += 1
        stored_scalars += int(coeff.size + neighbor_indices.size + 2)
        used_neighborhoods.append(f"{target}->" + ",".join(neighbors[:8]))

    after_pred = _assignment_predict(
        gate_support_scores,
        old_count=old_support_count,
        old_labels=old_labels,
        new_labels=new_labels,
        query_scenarios=np.asarray(["support"] * int(labels.size), dtype=object),
        scenario_balanced_assignment=False,
        role_balanced_assignment=True,
        balanced_assignment=True,
    )
    after_per_class: list[float] = []
    for label in class_labels:
        mask = labels == label
        after_per_class.append(float(np.mean(after_pred[mask] == label)) if bool(np.any(mask)) else 0.0)
    return (
        adjusted,
        int(used),
        int(stored_scalars),
        float(min(after_per_class, default=0.0)),
        float(np.mean(after_per_class) if after_per_class else 0.0),
        ";".join(used_neighborhoods[:32]),
    )


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


def _metadata_domain_values(
    *,
    key: str,
    scenarios: np.ndarray,
    rx_ids: np.ndarray,
    channel_views: np.ndarray,
    day_ids: np.ndarray,
) -> np.ndarray:
    mode = str(key).strip().lower()
    if mode in {"", "none"}:
        return np.asarray([""] * int(scenarios.size), dtype=object)
    if mode == "scenario":
        return np.asarray(scenarios, dtype=object).astype(str)
    if mode == "day":
        return np.asarray(day_ids, dtype=object).astype(str)
    if mode == "rx":
        return np.asarray(rx_ids, dtype=object).astype(str)
    if mode == "channel":
        return np.asarray(channel_views, dtype=object).astype(str)
    if mode == "day_scenario":
        day = np.asarray(day_ids, dtype=object).astype(str)
        sc = np.asarray(scenarios, dtype=object).astype(str)
        return np.asarray([f"{left}|{right}" for left, right in zip(day.tolist(), sc.tolist())], dtype=object)
    if mode == "rx_scenario":
        rx = np.asarray(rx_ids, dtype=object).astype(str)
        sc = np.asarray(scenarios, dtype=object).astype(str)
        return np.asarray([f"{left}|{right}" for left, right in zip(rx.tolist(), sc.tolist())], dtype=object)
    if mode == "rx_channel":
        rx = np.asarray(rx_ids, dtype=object).astype(str)
        view = np.asarray(channel_views, dtype=object).astype(str)
        return np.asarray([f"{left}|{right}" for left, right in zip(rx.tolist(), view.tolist())], dtype=object)
    raise ValueError(f"unsupported domain_refine_key: {key}")


def _evaluate_metric_qknn(
    *,
    features: np.ndarray,
    aux_features: np.ndarray | None,
    logits: np.ndarray,
    tx_ids: np.ndarray,
    roles: np.ndarray,
    scenarios: np.ndarray,
    domain_values: np.ndarray,
    old_splits: dict[str, Split],
    new_splits: dict[str, Split],
    old_labels: list[str],
    new_labels: list[str],
    transform_mode: str,
    transform_strength: float,
    topm: int,
    proto_mix: float,
    aux_score_weight: float,
    adaptive_qknn_policy: str,
    radius_norm: float,
    old_bias: float,
    neg_lambda: float,
    neg_threshold: float,
    neg_margin: float,
    mutual_only: bool,
    scenario_aware: bool,
    balanced_assignment: bool,
    role_balanced_assignment: bool,
    fast_role_balanced_assignment: bool,
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
    support_guided_proxy_rows: list[dict[str, Any]],
    support_guided_proxy_weight: float,
    support_guided_proxy_top_pairs: int,
    support_guided_proxy_clip: float,
    support_guided_proxy_min_errors: int,
    support_guided_proxy_scope: str,
    support_guided_proxy_balance: bool,
    support_guided_proxy_bundle_rows: int,
    support_guided_proxy_analogy: bool,
    support_guided_proxy_gate: bool,
    support_guided_proxy_gate_floor_tol: float,
    support_guided_proxy_gate_mean_tol: float,
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
    core_proto_weight: float,
    core_proto_count: int,
    core_proto_topm: int,
    core_proto_mode: str,
    ridge_head_weight: float,
    ridge_head_alpha: float,
    ridge_head_clip: float,
    subspace_proto_weight: float,
    subspace_proto_rank: int,
    subspace_proto_power: float,
    subspace_proto_clip: float,
    old_residual_new_weight: float,
    old_residual_new_rank: int,
    old_residual_new_proto_mix: float,
    old_residual_new_clip: float,
    domain_refine_key: str,
    domain_refine_weight: float,
    domain_refine_scope: str,
    class_diag_metric_weight: float,
    class_diag_metric_similarity: float,
    class_diag_metric_alpha: float,
    class_diag_metric_power: float,
    class_diag_metric_clip: float,
    support_bias_weight: float,
    support_bias_step: float,
    support_bias_rounds: int,
    support_quality_weight: float,
    support_quality_floor: float,
    support_quality_margin_scale: float,
    support_loo_pair_rescue_weight: float,
    support_loo_pair_rescue_top_pairs: int,
    support_loo_pair_rescue_min_errors: int,
    support_loo_pair_rescue_alpha: float,
    support_loo_pair_rescue_clip: float,
    support_loo_pair_rescue_scope: str,
    support_loo_pair_rescue_proto_neighbors: int,
    support_loo_pair_rescue_proto_min_sim: float,
    support_loo_pair_linear_weight: float,
    support_loo_pair_linear_top_pairs: int,
    support_loo_pair_linear_min_errors: int,
    support_loo_pair_linear_alpha: float,
    support_loo_pair_linear_clip: float,
    support_loo_pair_linear_scope: str,
    mahal_proto_weight: float,
    mahal_proto_alpha: float,
    mahal_proto_diag_mix: float,
    mahal_proto_clip: float,
    score_calibration: str,
    assignment_margin_weight: float,
    assignment_margin_clip: float,
    labelprop_weight: float,
    labelprop_k: int,
    labelprop_alpha: float,
    labelprop_temperature: float,
    labelprop_rounds: int,
    labelprop_clip: float,
    labelprop_scope: str,
    query_graph_weight: float,
    query_graph_k: int,
    query_graph_temperature: float,
    query_graph_rounds: int,
    query_graph_scope: str,
    query_cluster_weight: float,
    query_cluster_rounds: int,
    query_cluster_support_weight: float,
    query_cluster_temperature: float,
    query_cluster_clip: float,
    query_cluster_scope: str,
    query_cluster_agreement_min: float,
    query_cluster_margin_min: float,
    local_competition_weight: float,
    local_competition_k: int,
    local_competition_clip: float,
    local_competition_scope: str,
    query_proto_refine_weight: float,
    query_proto_refine_topm: int,
    query_proto_refine_clip: float,
    transductive_proto_weight: float,
    transductive_proto_rounds: int,
    transductive_proto_query_topm: int,
    transductive_proto_support_weight: float,
    transductive_proto_query_weight: float,
    transductive_proto_clip: float,
    dense_cluster_weight: float,
    dense_cluster_similarity: float,
    dense_cluster_neighbor_k: int,
    dense_cluster_rounds: int,
    dense_cluster_candidate_topn: int,
    dense_cluster_query_topm: int,
    dense_cluster_clip: float,
    dense_cluster_scope: str,
    query_pair_cluster_top_pairs: int,
    query_pair_cluster_similarity: float,
    query_pair_cluster_score_weight: float,
    query_pair_cluster_query_weight: float,
    query_pair_cluster_clip: float,
    query_pair_cluster_scope: str,
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
    effective_aux_score_weight = float(aux_score_weight)
    aux_support_gate_factor = 1.0
    aux_support_primary_loo_acc = 0.0
    aux_support_primary_loo_min_acc = 0.0
    aux_support_aux_loo_acc = 0.0
    aux_support_aux_loo_min_acc = 0.0
    aux_support_loo_delta = 0.0
    aux_support_min_delta = 0.0
    aux_support_absolute_floor_gate = 1.0
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
        policy_name = str(adaptive_qknn_policy).strip().lower()
        if policy_name in {
            "dualview_support_v10",
            "stable_dualview_v10",
            "dualview_support_v11",
            "stable_dualview_v11",
            "dualview_support_v12",
            "stable_dualview_v12",
            "dualview_support_v23",
            "stable_dualview_v23",
            "dualview_support_v24",
            "stable_dualview_v24",
            "dualview_support_v25",
            "stable_dualview_v25",
            "dualview_support_v30",
            "stable_dualview_v30",
        }:
            primary_loo_scores = _support_loo_base_scores(
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
            aux_loo_scores = _support_loo_base_scores(
                features=aux_adapted,
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
            (
                aux_support_primary_loo_acc,
                aux_support_primary_loo_min_acc,
            ) = _support_loo_accuracy_summary(
                primary_loo_scores,
                support_labels=support_labels,
                old_labels=old_labels,
                new_labels=new_labels,
            )
            (
                aux_support_aux_loo_acc,
                aux_support_aux_loo_min_acc,
            ) = _support_loo_accuracy_summary(
                aux_loo_scores,
                support_labels=support_labels,
                old_labels=old_labels,
                new_labels=new_labels,
            )
            aux_support_loo_delta = float(aux_support_aux_loo_acc - aux_support_primary_loo_acc)
            aux_support_min_delta = float(aux_support_aux_loo_min_acc - aux_support_primary_loo_min_acc)
            if policy_name in {
                "dualview_support_v23",
                "stable_dualview_v23",
                "dualview_support_v24",
                "stable_dualview_v24",
                "dualview_support_v30",
                "stable_dualview_v30",
            }:
                mean_gate = float(np.clip((aux_support_loo_delta + 0.01) / 0.05, 0.0, 1.0))
                floor_gate = float(np.clip((aux_support_min_delta + 0.02) / 0.06, 0.0, 1.0))
                aux_support_absolute_floor_gate = float(
                    np.clip((aux_support_aux_loo_min_acc - 0.40) / 0.25, 0.0, 1.0)
                )
            else:
                mean_gate = float(np.clip((aux_support_loo_delta + 0.02) / 0.08, 0.0, 1.0))
                floor_gate = float(np.clip((aux_support_min_delta + 0.05) / 0.12, 0.0, 1.0))
                aux_support_absolute_floor_gate = float(
                    np.clip((aux_support_aux_loo_min_acc - 0.20) / 0.30, 0.0, 1.0)
                )
            aux_support_gate_factor = float(min(mean_gate, floor_gate, aux_support_absolute_floor_gate))
            effective_aux_score_weight = float(aux_score_weight) * aux_support_gate_factor
        scores = (1.0 - effective_aux_score_weight) * scores + effective_aux_score_weight * aux_scores
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
    effective_support_guided_proxy_rows = support_guided_proxy_rows
    support_guided_proxy_auto_rows = 0
    support_guided_proxy_auto_pairs = ""
    support_guided_proxy_gate_before_min_acc = 0.0
    support_guided_proxy_gate_before_mean_acc = 0.0
    support_guided_proxy_gate_after_min_acc = 0.0
    support_guided_proxy_gate_after_mean_acc = 0.0
    if (
        not effective_support_guided_proxy_rows
        and float(support_guided_proxy_weight) > 0.0
        and int(support_guided_proxy_top_pairs) > 0
    ):
        proxy_support_loo_scores = _support_loo_base_scores(
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
        candidate_budget = int(support_guided_proxy_top_pairs)
        if bool(support_guided_proxy_gate):
            candidate_budget = max(candidate_budget, int(support_guided_proxy_top_pairs) * 4)
        effective_support_guided_proxy_rows = _support_loo_proxy_candidate_rows(
            features=adapted,
            tx_ids=tx_ids,
            roles=roles,
            support_indices=support_indices,
            support_labels=support_labels,
            support_scores=proxy_support_loo_scores,
            class_labels=old_labels + new_labels,
            old_labels=old_labels,
            new_labels=new_labels,
            top_rows=int(candidate_budget),
            min_errors=int(support_guided_proxy_min_errors),
            scope=str(support_guided_proxy_scope),
            balance_classes=bool(support_guided_proxy_balance),
            bundle_rows=int(support_guided_proxy_bundle_rows),
            analogy_mode=bool(support_guided_proxy_analogy),
        )
        if bool(support_guided_proxy_gate):
            (
                effective_support_guided_proxy_rows,
                support_guided_proxy_gate_before_min_acc,
                support_guided_proxy_gate_before_mean_acc,
                support_guided_proxy_gate_after_min_acc,
                support_guided_proxy_gate_after_mean_acc,
            ) = _gate_support_guided_proxy_rows(
                scores=proxy_support_loo_scores,
                features=adapted,
                tx_ids=tx_ids,
                roles=roles,
                support_indices=support_indices,
                support_labels=support_labels,
                class_labels=old_labels + new_labels,
                old_labels=old_labels,
                new_labels=new_labels,
                candidate_rows=effective_support_guided_proxy_rows,
                weight=float(support_guided_proxy_weight),
                max_rows=int(support_guided_proxy_top_pairs),
                clip=float(support_guided_proxy_clip),
                floor_tol=float(support_guided_proxy_gate_floor_tol),
                mean_tol=float(support_guided_proxy_gate_mean_tol),
            )
        support_guided_proxy_auto_rows = len(effective_support_guided_proxy_rows)
        support_guided_proxy_auto_pairs = ";".join(
            f"{row.get('target_new')}->{row.get('hard_label')}" for row in effective_support_guided_proxy_rows[:16]
        )
    scores, support_guided_proxy_count, stored_support_guided_proxy_scalars = _support_guided_proxy_adjust_scores(
        scores,
        features=adapted,
        tx_ids=tx_ids,
        roles=roles,
        support_indices=support_indices,
        support_labels=support_labels,
        query_indices=query_indices,
        class_labels=old_labels + new_labels,
        candidate_rows=effective_support_guided_proxy_rows,
        weight=float(support_guided_proxy_weight),
        top_pairs=int(support_guided_proxy_top_pairs),
        clip=float(support_guided_proxy_clip),
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
    core_proto_count_stored = 0
    if float(core_proto_weight) > 0.0:
        core_scores, core_proto_count_stored = _core_proto_scores(
            features=adapted,
            support_indices=support_indices,
            support_labels=support_labels,
            query_indices=query_indices,
            class_labels=old_labels + new_labels,
            old_labels=set(old_labels),
            core_count=int(core_proto_count),
            topm=int(core_proto_topm),
            radius_norm=float(radius_norm),
            old_bias=float(old_bias),
            mode=str(core_proto_mode),
        )
        scores = scores + float(core_proto_weight) * core_scores
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
    old_residual_new_rank_used = 0
    stored_old_residual_new_scalars = 0
    if float(old_residual_new_weight) > 0.0:
        old_residual_scores, old_residual_new_rank_used, stored_old_residual_new_scalars = _old_residual_new_scores(
            features=adapted,
            support_indices=support_indices,
            support_labels=support_labels,
            query_indices=query_indices,
            class_labels=old_labels + new_labels,
            old_labels=old_labels,
            topm=int(topm),
            proto_mix=float(old_residual_new_proto_mix),
            rank=int(old_residual_new_rank),
            clip=float(old_residual_new_clip),
        )
        scores = scores + float(old_residual_new_weight) * old_residual_scores
    domain_refine_domain_count = 0
    stored_domain_refine_prototype_count = 0
    if float(domain_refine_weight) > 0.0 and str(domain_refine_key).strip().lower() not in {"", "none"}:
        refine_scope = str(domain_refine_scope).strip().lower()
        if refine_scope not in {"all", "new", "old"}:
            raise ValueError(f"unsupported domain_refine_scope: {domain_refine_scope}")
        if refine_scope == "new":
            refine_labels = list(new_labels)
            if bool(role_balanced_assignment):
                refine_query_indices = query_indices[old_count:]
                refine_rows = np.arange(old_count, query_indices.size, dtype=int)
            else:
                refine_query_indices = query_indices
                refine_rows = np.arange(query_indices.size, dtype=int)
        elif refine_scope == "old":
            refine_labels = list(old_labels)
            if bool(role_balanced_assignment):
                refine_query_indices = query_indices[:old_count]
                refine_rows = np.arange(old_count, dtype=int)
            else:
                refine_query_indices = query_indices
                refine_rows = np.arange(query_indices.size, dtype=int)
        else:
            refine_labels = list(old_labels) + list(new_labels)
            refine_query_indices = query_indices
            refine_rows = np.arange(query_indices.size, dtype=int)
        domain_scores, _domain_radii, _domain_proto_sim = base._class_scores(
            features=adapted,
            support_indices=support_indices,
            support_labels=support_labels,
            query_indices=refine_query_indices,
            scenarios=domain_values,
            class_labels=refine_labels,
            old_labels=set(old_labels),
            topm=int(topm),
            proto_mix=float(proto_mix),
            radius_norm=float(radius_norm),
            old_bias=float(old_bias),
            neg_lambda=float(neg_lambda),
            neg_threshold=float(neg_threshold),
            neg_margin=float(neg_margin),
            mutual_only=bool(mutual_only),
            scenario_aware=True,
        )
        if refine_scope == "all":
            blended_scores = (1.0 - float(domain_refine_weight)) * scores + float(domain_refine_weight) * domain_scores
            scores = blended_scores
        elif refine_scope == "new":
            scores[refine_rows, len(old_labels) :] = (
                (1.0 - float(domain_refine_weight)) * scores[refine_rows, len(old_labels) :]
                + float(domain_refine_weight) * domain_scores
            )
        elif refine_scope == "old":
            scores[refine_rows, : len(old_labels)] = (
                (1.0 - float(domain_refine_weight)) * scores[refine_rows, : len(old_labels)]
                + float(domain_refine_weight) * domain_scores
            )
        query_domains = set(np.asarray(domain_values[query_indices], dtype=object).astype(str).tolist())
        support_domains = set(np.asarray(domain_values[support_indices], dtype=object).astype(str).tolist())
        domain_refine_domain_count = len(query_domains & support_domains)
        stored_domain_refine_prototype_count = int(domain_refine_domain_count * len(refine_labels))
    class_diag_metric_count = 0
    stored_class_diag_metric_scalars = 0
    if float(class_diag_metric_weight) > 0.0:
        class_diag_scores, class_diag_metric_count, stored_class_diag_metric_scalars = _class_diag_metric_scores(
            features=adapted,
            support_indices=support_indices,
            support_labels=support_labels,
            query_indices=query_indices,
            class_labels=old_labels + new_labels,
            topm=int(topm),
            proto_mix=float(proto_mix),
            similarity_threshold=float(class_diag_metric_similarity),
            alpha=float(class_diag_metric_alpha),
            power=float(class_diag_metric_power),
            clip=float(class_diag_metric_clip),
        )
        scores = scores + float(class_diag_metric_weight) * class_diag_scores
    stored_support_bias_scalars = 0
    support_bias_loo_min_acc = 0.0
    support_bias_loo_mean_acc = 0.0
    support_loo_scores: np.ndarray | None = None
    support_quality_stored_scalars = 0
    support_quality_loo_min_acc = 0.0
    support_quality_loo_mean_acc = 0.0
    if float(support_quality_weight) > 0.0:
        support_loo_scores = _support_loo_base_scores(
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
        (
            quality_scores,
            _quality_radii,
            _quality_proto_sim,
            support_quality_stored_scalars,
            support_quality_loo_min_acc,
            support_quality_loo_mean_acc,
        ) = _support_quality_weighted_scores(
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
            support_scores=support_loo_scores,
            quality_floor=float(support_quality_floor),
            margin_scale=float(support_quality_margin_scale),
        )
        quality_weight = float(np.clip(float(support_quality_weight), 0.0, 1.0))
        scores = (1.0 - quality_weight) * scores + quality_weight * quality_scores
    if float(support_bias_weight) > 0.0 and float(support_bias_step) > 0.0 and int(support_bias_rounds) > 0:
        if support_loo_scores is None:
            support_loo_scores = _support_loo_base_scores(
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
            support_scores=support_loo_scores,
            support_labels=support_labels,
            class_labels=old_labels + new_labels,
            old_labels=old_labels,
            new_labels=new_labels,
            step=float(support_bias_step),
            rounds=int(support_bias_rounds),
        )
        scores = scores + float(support_bias_weight) * support_bias[None, :]
        stored_support_bias_scalars = int(support_bias.size)
    support_loo_pair_rescue_count = 0
    stored_support_loo_pair_rescue_scalars = 0
    support_loo_pair_rescue_min_acc = 0.0
    support_loo_pair_rescue_mean_acc = 0.0
    support_loo_pair_rescue_pairs = ""
    if float(support_loo_pair_rescue_weight) > 0.0 and int(support_loo_pair_rescue_top_pairs) > 0:
        if support_loo_scores is None:
            support_loo_scores = _support_loo_base_scores(
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
        (
            scores,
            support_loo_pair_rescue_count,
            stored_support_loo_pair_rescue_scalars,
            support_loo_pair_rescue_min_acc,
            support_loo_pair_rescue_mean_acc,
            support_loo_pair_rescue_pairs,
        ) = _support_loo_pair_rescue_scores(
            scores,
            features=adapted,
            support_indices=support_indices,
            support_labels=support_labels,
            query_indices=query_indices,
            support_scores=support_loo_scores,
            class_labels=old_labels + new_labels,
            old_labels=old_labels,
            new_labels=new_labels,
            top_pairs=int(support_loo_pair_rescue_top_pairs),
            min_errors=int(support_loo_pair_rescue_min_errors),
            weight=float(support_loo_pair_rescue_weight),
            alpha=float(support_loo_pair_rescue_alpha),
            clip=float(support_loo_pair_rescue_clip),
            scope=str(support_loo_pair_rescue_scope),
            proto_neighbors=int(support_loo_pair_rescue_proto_neighbors),
            proto_min_sim=float(support_loo_pair_rescue_proto_min_sim),
        )
    support_loo_pair_linear_count = 0
    stored_support_loo_pair_linear_scalars = 0
    support_loo_pair_linear_min_acc = 0.0
    support_loo_pair_linear_mean_acc = 0.0
    if float(support_loo_pair_linear_weight) > 0.0 and int(support_loo_pair_linear_top_pairs) > 0:
        if support_loo_scores is None:
            support_loo_scores = _support_loo_base_scores(
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
        (
            scores,
            support_loo_pair_linear_count,
            stored_support_loo_pair_linear_scalars,
            support_loo_pair_linear_min_acc,
            support_loo_pair_linear_mean_acc,
        ) = _support_loo_pair_linear_scores(
            scores,
            features=adapted,
            support_indices=support_indices,
            support_labels=support_labels,
            query_indices=query_indices,
            support_scores=support_loo_scores,
            class_labels=old_labels + new_labels,
            old_labels=old_labels,
            new_labels=new_labels,
            top_pairs=int(support_loo_pair_linear_top_pairs),
            min_errors=int(support_loo_pair_linear_min_errors),
            weight=float(support_loo_pair_linear_weight),
            alpha=float(support_loo_pair_linear_alpha),
            clip=float(support_loo_pair_linear_clip),
            scope=str(support_loo_pair_linear_scope),
        )
    top2_pair_gate_count = 0
    stored_top2_pair_gate_scalars = 0
    top2_pair_gate_loo_min_acc = 0.0
    top2_pair_gate_loo_mean_acc = 0.0
    top2_pair_gate_pairs = ""
    top2_pair_gate_weight = 0.0
    top2_pair_gate_top_pairs = 0
    top2_pair_gate_margin = 0.0
    top2_pair_gate_query_weight = 0.0
    neighborhood_gate_count = 0
    stored_neighborhood_gate_scalars = 0
    neighborhood_gate_loo_min_acc = 0.0
    neighborhood_gate_loo_mean_acc = 0.0
    neighborhood_gate_neighborhoods = ""
    neighborhood_gate_weight = 0.0
    neighborhood_gate_top_classes = 0
    neighborhood_gate_neighbor_count = 0
    neighborhood_gate_margin = 0.0
    neighborhood_gate_query_weight = 0.0
    policy_norm = str(adaptive_qknn_policy).strip().lower()
    if policy_norm in {
        "dualview_support_v29",
        "stable_dualview_v29",
        "dualview_support_v30",
        "stable_dualview_v30",
    }:
        if support_loo_scores is None:
            support_loo_scores = _support_loo_base_scores(
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
        label_counts = [
            int(np.sum(np.asarray(support_labels, dtype=object).astype(str) == str(label)))
            for label in old_labels + new_labels
        ]
        min_support = float(min(label_counts) if label_counts else 0)
        class_load_gate = _clip01((float(len(new_labels)) - 2.0) / 18.0)
        low_k_gate = _clip01(1.0 - ((min_support - 5.0) / 15.0))
        neighborhood_gate_weight = float(np.clip(0.022 + 0.020 * class_load_gate + 0.014 * low_k_gate, 0.018, 0.060))
        neighborhood_gate_top_classes = int(max(2, min(8, round(0.28 * max(float(len(new_labels)), 1.0)))))
        neighborhood_gate_neighbor_count = int(max(2, min(4, round(2.0 + 2.0 * class_load_gate))))
        neighborhood_gate_margin = float(np.clip(0.22 + 0.08 * class_load_gate + 0.10 * low_k_gate, 0.20, 0.42))
        neighborhood_gate_query_weight = float(np.clip(0.018 + 0.034 * class_load_gate, 0.0, 0.055))
        (
            scores,
            neighborhood_gate_count,
            stored_neighborhood_gate_scalars,
            neighborhood_gate_loo_min_acc,
            neighborhood_gate_loo_mean_acc,
            neighborhood_gate_neighborhoods,
        ) = _support_query_neighborhood_gate_scores(
            scores,
            features=adapted,
            support_indices=support_indices,
            support_labels=support_labels,
            query_indices=query_indices,
            support_scores=support_loo_scores,
            class_labels=old_labels + new_labels,
            old_labels=old_labels,
            new_labels=new_labels,
            old_query_count=old_count,
            top_classes=neighborhood_gate_top_classes,
            neighbor_count=neighborhood_gate_neighbor_count,
            query_topm=4,
            weight=neighborhood_gate_weight,
            alpha=0.1,
            clip=1.5,
            gate_margin=neighborhood_gate_margin,
            query_neighbor_weight=neighborhood_gate_query_weight,
        )
    if policy_norm in {
        "dualview_support_v25",
        "stable_dualview_v25",
        "dualview_support_v28",
        "stable_dualview_v28",
    }:
        if support_loo_scores is None:
            support_loo_scores = _support_loo_base_scores(
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
        label_counts = [
            int(np.sum(np.asarray(support_labels, dtype=object).astype(str) == str(label)))
            for label in old_labels + new_labels
        ]
        min_support = float(min(label_counts) if label_counts else 0)
        class_load_gate = _clip01((float(len(new_labels)) - 2.0) / 18.0)
        k_gate = _clip01((min_support - 5.0) / 15.0)
        if policy_norm in {"dualview_support_v28", "stable_dualview_v28"}:
            top2_pair_gate_weight = float(np.clip(0.020 + 0.018 * class_load_gate + 0.012 * k_gate, 0.018, 0.050))
            top2_pair_gate_top_pairs = int(max(2, min(6, round(0.22 * max(float(len(new_labels)), 1.0)))))
            top2_pair_gate_margin = float(np.clip(0.18 + 0.12 * k_gate + 0.06 * class_load_gate, 0.16, 0.36))
            top2_pair_gate_query_weight = float(np.clip(0.020 + 0.020 * class_load_gate, 0.0, 0.040))
        else:
            top2_pair_gate_weight = float(np.clip(0.035 + 0.030 * class_load_gate + 0.020 * k_gate, 0.03, 0.08))
            top2_pair_gate_top_pairs = int(max(2, min(8, round(0.30 * max(float(len(new_labels)), 1.0)))))
            top2_pair_gate_margin = float(np.clip(0.28 + 0.18 * k_gate + 0.08 * class_load_gate, 0.25, 0.55))
        (
            scores,
            top2_pair_gate_count,
            stored_top2_pair_gate_scalars,
            top2_pair_gate_loo_min_acc,
            top2_pair_gate_loo_mean_acc,
            top2_pair_gate_pairs,
        ) = _support_loo_top2_pair_gate_scores(
            scores,
            features=adapted,
            support_indices=support_indices,
            support_labels=support_labels,
            query_indices=query_indices,
            support_scores=support_loo_scores,
            class_labels=old_labels + new_labels,
            old_labels=old_labels,
            new_labels=new_labels,
            old_query_count=old_count,
            top_pairs=top2_pair_gate_top_pairs,
            min_errors=1,
            weight=top2_pair_gate_weight,
            alpha=0.1,
            clip=1.5,
            gate_margin=top2_pair_gate_margin,
            query_pair_weight=top2_pair_gate_query_weight,
        )
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
    scores, local_competition_edges = _local_competition_adjust_scores(
        scores,
        proto_sim=proto_sim,
        old_labels=old_labels,
        new_labels=new_labels,
        neighbor_k=int(local_competition_k),
        weight=float(local_competition_weight),
        clip=float(local_competition_clip),
        scope=str(local_competition_scope),
    )
    labelprop_edges = 0
    if float(labelprop_weight) != 0.0:
        labelprop_scores, labelprop_edges = _support_query_labelprop_scores(
            features=adapted,
            support_indices=support_indices,
            support_labels=support_labels,
            query_indices=query_indices,
            scenarios=scenarios,
            class_labels=old_labels + new_labels,
            neighbor_k=int(labelprop_k),
            alpha=float(labelprop_alpha),
            temperature=float(labelprop_temperature),
            rounds=int(labelprop_rounds),
            clip=float(labelprop_clip),
            scope=str(labelprop_scope),
        )
        scores = scores + float(labelprop_weight) * labelprop_scores
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
    scores, query_graph_edges = _query_graph_smooth_scores(
        scores,
        features=adapted,
        query_indices=query_indices,
        scenarios=scenarios,
        neighbor_k=int(query_graph_k),
        weight=float(query_graph_weight),
        temperature=float(query_graph_temperature),
        rounds=int(query_graph_rounds),
        scope=str(query_graph_scope),
    )
    query_cluster_temp_proto_count = 0
    query_cluster_assigned_rows = 0
    if abs(float(query_cluster_weight)) > 1e-12 and int(query_cluster_rounds) > 0:
        scores, query_cluster_temp_proto_count, query_cluster_assigned_rows = _query_cluster_align_scores(
            scores,
            features=adapted,
            support_indices=support_indices,
            support_labels=support_labels,
            query_indices=query_indices,
            old_count=old_count,
            class_labels=old_labels + new_labels,
            old_labels=old_labels,
            new_labels=new_labels,
            weight=float(query_cluster_weight),
            rounds=int(query_cluster_rounds),
            support_weight=float(query_cluster_support_weight),
            temperature=float(query_cluster_temperature),
            clip=float(query_cluster_clip),
            scope=str(query_cluster_scope),
            agreement_min=float(query_cluster_agreement_min),
            margin_min=float(query_cluster_margin_min),
        )
    transductive_proto_count = 0
    if float(transductive_proto_weight) != 0.0 and int(transductive_proto_rounds) > 0:
        scores, transductive_proto_count = _support_anchored_transductive_proto_scores(
            scores,
            features=adapted,
            support_indices=support_indices,
            support_labels=support_labels,
            query_indices=query_indices,
            class_labels=old_labels + new_labels,
            old_count=old_count,
            old_labels=old_labels,
            new_labels=new_labels,
            query_scenarios=scenarios[query_indices],
            scenario_balanced_assignment=bool(scenario_balanced_assignment),
            role_balanced_assignment=bool(role_balanced_assignment),
            balanced_assignment=bool(balanced_assignment),
            rounds=int(transductive_proto_rounds),
            weight=float(transductive_proto_weight),
            support_weight=float(transductive_proto_support_weight),
            query_weight=float(transductive_proto_query_weight),
            query_topm=int(transductive_proto_query_topm),
            clip=float(transductive_proto_clip),
        )
    query_proto_refine_count = 0
    if float(query_proto_refine_weight) != 0.0:
        provisional_pred = _assignment_predict(
            scores,
            old_count=old_count,
            old_labels=old_labels,
            new_labels=new_labels,
            query_scenarios=scenarios[query_indices],
            scenario_balanced_assignment=bool(scenario_balanced_assignment),
            role_balanced_assignment=bool(role_balanced_assignment),
            balanced_assignment=bool(balanced_assignment),
        )
        scores, query_proto_refine_count = _query_proto_refine_scores(
            scores,
            features=adapted,
            query_indices=query_indices,
            provisional_pred=provisional_pred,
            class_labels=old_labels + new_labels,
            topm=int(query_proto_refine_topm),
            weight=float(query_proto_refine_weight),
            clip=float(query_proto_refine_clip),
        )
    dense_cluster_count = 0
    dense_cluster_temp_proto_count = 0
    dense_cluster_adjusted_rows = 0
    if float(dense_cluster_weight) != 0.0 and int(dense_cluster_rounds) > 0:
        (
            scores,
            dense_cluster_count,
            dense_cluster_temp_proto_count,
            dense_cluster_adjusted_rows,
        ) = _dense_cluster_query_refine_scores(
            scores,
            features=adapted,
            support_indices=support_indices,
            support_labels=support_labels,
            query_indices=query_indices,
            class_labels=old_labels + new_labels,
            old_count=old_count,
            old_labels=old_labels,
            new_labels=new_labels,
            proto_sim=proto_sim,
            query_scenarios=scenarios[query_indices],
            scenario_balanced_assignment=bool(scenario_balanced_assignment),
            role_balanced_assignment=bool(role_balanced_assignment),
            balanced_assignment=bool(balanced_assignment),
            rounds=int(dense_cluster_rounds),
            weight=float(dense_cluster_weight),
            similarity_threshold=float(dense_cluster_similarity),
            neighbor_k=int(dense_cluster_neighbor_k),
            candidate_topn=int(dense_cluster_candidate_topn),
            query_topm=int(dense_cluster_query_topm),
            clip=float(dense_cluster_clip),
            scope=str(dense_cluster_scope),
        )
    pred = _assignment_predict(
        scores,
        old_count=old_count,
        old_labels=old_labels,
        new_labels=new_labels,
        query_scenarios=scenarios[query_indices],
        scenario_balanced_assignment=bool(scenario_balanced_assignment),
        role_balanced_assignment=bool(role_balanced_assignment),
        fast_role_balanced_assignment=bool(fast_role_balanced_assignment),
        balanced_assignment=bool(balanced_assignment),
    )
    pred, query_pair_cluster_changed, query_pair_cluster_count, query_pair_cluster_pairs = (
        _query_pair_cluster_quota_refine(
            pred,
            scores,
            features=adapted,
            support_indices=support_indices,
            support_labels=support_labels,
            query_indices=query_indices,
            class_labels=old_labels + new_labels,
            old_count=old_count,
            old_labels=old_labels,
            new_labels=new_labels,
            proto_sim=proto_sim,
            top_pairs=int(query_pair_cluster_top_pairs),
            similarity_threshold=float(query_pair_cluster_similarity),
            score_weight=float(query_pair_cluster_score_weight),
            query_weight=float(query_pair_cluster_query_weight),
            clip=float(query_pair_cluster_clip),
            scope=str(query_pair_cluster_scope),
        )
    )
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
        "effective_aux_score_weight": float(effective_aux_score_weight),
        "aux_support_gate_factor": float(aux_support_gate_factor),
        "aux_support_primary_loo_acc": float(aux_support_primary_loo_acc),
        "aux_support_primary_loo_min_acc": float(aux_support_primary_loo_min_acc),
        "aux_support_aux_loo_acc": float(aux_support_aux_loo_acc),
        "aux_support_aux_loo_min_acc": float(aux_support_aux_loo_min_acc),
        "aux_support_loo_delta": float(aux_support_loo_delta),
        "aux_support_min_delta": float(aux_support_min_delta),
        "aux_support_absolute_floor_gate": float(aux_support_absolute_floor_gate),
        "radius_norm": float(radius_norm),
        "old_bias": float(old_bias),
        "neg_lambda": float(neg_lambda),
        "neg_threshold": float(neg_threshold),
        "neg_margin": float(neg_margin),
        "mutual_only": bool(mutual_only),
        "scenario_aware": bool(scenario_aware),
        "balanced_assignment": bool(balanced_assignment),
        "role_balanced_assignment": bool(role_balanced_assignment),
        "fast_role_balanced_assignment": bool(fast_role_balanced_assignment),
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
        "support_guided_proxy_weight": float(support_guided_proxy_weight),
        "support_guided_proxy_top_pairs": int(support_guided_proxy_top_pairs),
        "support_guided_proxy_clip": float(support_guided_proxy_clip),
        "support_guided_proxy_min_errors": int(support_guided_proxy_min_errors),
        "support_guided_proxy_scope": str(support_guided_proxy_scope),
        "support_guided_proxy_balance": bool(support_guided_proxy_balance),
        "support_guided_proxy_bundle_rows": int(support_guided_proxy_bundle_rows),
        "support_guided_proxy_analogy": bool(support_guided_proxy_analogy),
        "support_guided_proxy_gate": bool(support_guided_proxy_gate),
        "support_guided_proxy_gate_floor_tol": float(support_guided_proxy_gate_floor_tol),
        "support_guided_proxy_gate_mean_tol": float(support_guided_proxy_gate_mean_tol),
        "support_guided_proxy_gate_before_min_acc": float(support_guided_proxy_gate_before_min_acc),
        "support_guided_proxy_gate_before_mean_acc": float(support_guided_proxy_gate_before_mean_acc),
        "support_guided_proxy_gate_after_min_acc": float(support_guided_proxy_gate_after_min_acc),
        "support_guided_proxy_gate_after_mean_acc": float(support_guided_proxy_gate_after_mean_acc),
        "support_guided_proxy_auto_rows": int(support_guided_proxy_auto_rows),
        "support_guided_proxy_auto_pairs": support_guided_proxy_auto_pairs,
        "support_guided_proxy_count": int(support_guided_proxy_count),
        "stored_support_guided_proxy_scalars": int(stored_support_guided_proxy_scalars),
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
        "core_proto_weight": float(core_proto_weight),
        "core_proto_count": int(core_proto_count),
        "core_proto_topm": int(core_proto_topm),
        "core_proto_mode": str(core_proto_mode),
        "stored_core_prototype_count": int(core_proto_count_stored),
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
        "old_residual_new_weight": float(old_residual_new_weight),
        "old_residual_new_rank": int(old_residual_new_rank),
        "old_residual_new_rank_used": int(old_residual_new_rank_used),
        "old_residual_new_proto_mix": float(old_residual_new_proto_mix),
        "old_residual_new_clip": float(old_residual_new_clip),
        "stored_old_residual_new_scalars": int(stored_old_residual_new_scalars),
        "domain_refine_key": str(domain_refine_key),
        "domain_refine_weight": float(domain_refine_weight),
        "domain_refine_scope": str(domain_refine_scope),
        "domain_refine_domain_count": int(domain_refine_domain_count),
        "stored_domain_refine_prototype_count": int(stored_domain_refine_prototype_count),
        "class_diag_metric_weight": float(class_diag_metric_weight),
        "class_diag_metric_similarity": float(class_diag_metric_similarity),
        "class_diag_metric_alpha": float(class_diag_metric_alpha),
        "class_diag_metric_power": float(class_diag_metric_power),
        "class_diag_metric_clip": float(class_diag_metric_clip),
        "class_diag_metric_count": int(class_diag_metric_count),
        "stored_class_diag_metric_scalars": int(stored_class_diag_metric_scalars),
        "support_bias_weight": float(support_bias_weight),
        "support_bias_step": float(support_bias_step),
        "support_bias_rounds": int(support_bias_rounds),
        "support_bias_loo_min_acc": float(support_bias_loo_min_acc),
        "support_bias_loo_mean_acc": float(support_bias_loo_mean_acc),
        "stored_support_bias_scalars": int(stored_support_bias_scalars),
        "support_quality_weight": float(support_quality_weight),
        "support_quality_floor": float(support_quality_floor),
        "support_quality_margin_scale": float(support_quality_margin_scale),
        "support_quality_loo_min_acc": float(support_quality_loo_min_acc),
        "support_quality_loo_mean_acc": float(support_quality_loo_mean_acc),
        "stored_support_quality_scalars": int(support_quality_stored_scalars),
        "support_loo_pair_rescue_weight": float(support_loo_pair_rescue_weight),
        "support_loo_pair_rescue_top_pairs": int(support_loo_pair_rescue_top_pairs),
        "support_loo_pair_rescue_min_errors": int(support_loo_pair_rescue_min_errors),
        "support_loo_pair_rescue_alpha": float(support_loo_pair_rescue_alpha),
        "support_loo_pair_rescue_clip": float(support_loo_pair_rescue_clip),
        "support_loo_pair_rescue_scope": str(support_loo_pair_rescue_scope),
        "support_loo_pair_rescue_proto_neighbors": int(support_loo_pair_rescue_proto_neighbors),
        "support_loo_pair_rescue_proto_min_sim": float(support_loo_pair_rescue_proto_min_sim),
        "support_loo_pair_rescue_count": int(support_loo_pair_rescue_count),
        "support_loo_pair_rescue_pairs": support_loo_pair_rescue_pairs,
        "support_loo_pair_rescue_loo_min_acc": float(support_loo_pair_rescue_min_acc),
        "support_loo_pair_rescue_loo_mean_acc": float(support_loo_pair_rescue_mean_acc),
        "stored_support_loo_pair_rescue_scalars": int(stored_support_loo_pair_rescue_scalars),
        "support_loo_pair_linear_weight": float(support_loo_pair_linear_weight),
        "support_loo_pair_linear_top_pairs": int(support_loo_pair_linear_top_pairs),
        "support_loo_pair_linear_min_errors": int(support_loo_pair_linear_min_errors),
        "support_loo_pair_linear_alpha": float(support_loo_pair_linear_alpha),
        "support_loo_pair_linear_clip": float(support_loo_pair_linear_clip),
        "support_loo_pair_linear_scope": str(support_loo_pair_linear_scope),
        "support_loo_pair_linear_count": int(support_loo_pair_linear_count),
        "support_loo_pair_linear_loo_min_acc": float(support_loo_pair_linear_min_acc),
        "support_loo_pair_linear_loo_mean_acc": float(support_loo_pair_linear_mean_acc),
        "stored_support_loo_pair_linear_scalars": int(stored_support_loo_pair_linear_scalars),
        "top2_pair_gate_weight": float(top2_pair_gate_weight),
        "top2_pair_gate_top_pairs": int(top2_pair_gate_top_pairs),
        "top2_pair_gate_margin": float(top2_pair_gate_margin),
        "top2_pair_gate_query_weight": float(top2_pair_gate_query_weight),
        "top2_pair_gate_count": int(top2_pair_gate_count),
        "top2_pair_gate_pairs": top2_pair_gate_pairs,
        "top2_pair_gate_loo_min_acc": float(top2_pair_gate_loo_min_acc),
        "top2_pair_gate_loo_mean_acc": float(top2_pair_gate_loo_mean_acc),
        "stored_top2_pair_gate_scalars": int(stored_top2_pair_gate_scalars),
        "neighborhood_gate_weight": float(neighborhood_gate_weight),
        "neighborhood_gate_top_classes": int(neighborhood_gate_top_classes),
        "neighborhood_gate_neighbor_count": int(neighborhood_gate_neighbor_count),
        "neighborhood_gate_margin": float(neighborhood_gate_margin),
        "neighborhood_gate_query_weight": float(neighborhood_gate_query_weight),
        "neighborhood_gate_count": int(neighborhood_gate_count),
        "neighborhood_gate_neighborhoods": neighborhood_gate_neighborhoods,
        "neighborhood_gate_loo_min_acc": float(neighborhood_gate_loo_min_acc),
        "neighborhood_gate_loo_mean_acc": float(neighborhood_gate_loo_mean_acc),
        "stored_neighborhood_gate_scalars": int(stored_neighborhood_gate_scalars),
        "mahal_proto_weight": float(mahal_proto_weight),
        "mahal_proto_alpha": float(mahal_proto_alpha),
        "mahal_proto_diag_mix": float(mahal_proto_diag_mix),
        "mahal_proto_clip": float(mahal_proto_clip),
        "score_calibration": str(score_calibration),
        "assignment_margin_weight": float(assignment_margin_weight),
        "assignment_margin_clip": float(assignment_margin_clip),
        "labelprop_weight": float(labelprop_weight),
        "labelprop_k": int(labelprop_k),
        "labelprop_alpha": float(labelprop_alpha),
        "labelprop_temperature": float(labelprop_temperature),
        "labelprop_rounds": int(labelprop_rounds),
        "labelprop_clip": float(labelprop_clip),
        "labelprop_scope": str(labelprop_scope),
        "labelprop_edges": int(labelprop_edges),
        "query_graph_weight": float(query_graph_weight),
        "query_graph_k": int(query_graph_k),
        "query_graph_temperature": float(query_graph_temperature),
        "query_graph_rounds": int(query_graph_rounds),
        "query_graph_scope": str(query_graph_scope),
        "query_graph_edges": int(query_graph_edges),
        "query_cluster_weight": float(query_cluster_weight),
        "query_cluster_rounds": int(query_cluster_rounds),
        "query_cluster_support_weight": float(query_cluster_support_weight),
        "query_cluster_temperature": float(query_cluster_temperature),
        "query_cluster_clip": float(query_cluster_clip),
        "query_cluster_scope": str(query_cluster_scope),
        "query_cluster_agreement_min": float(query_cluster_agreement_min),
        "query_cluster_margin_min": float(query_cluster_margin_min),
        "query_cluster_temp_proto_count": int(query_cluster_temp_proto_count),
        "query_cluster_assigned_rows": int(query_cluster_assigned_rows),
        "local_competition_weight": float(local_competition_weight),
        "local_competition_k": int(local_competition_k),
        "local_competition_clip": float(local_competition_clip),
        "local_competition_scope": str(local_competition_scope),
        "local_competition_edges": int(local_competition_edges),
        "query_proto_refine_weight": float(query_proto_refine_weight),
        "query_proto_refine_topm": int(query_proto_refine_topm),
        "query_proto_refine_clip": float(query_proto_refine_clip),
        "query_proto_refine_count": int(query_proto_refine_count),
        "transductive_proto_weight": float(transductive_proto_weight),
        "transductive_proto_rounds": int(transductive_proto_rounds),
        "transductive_proto_query_topm": int(transductive_proto_query_topm),
        "transductive_proto_support_weight": float(transductive_proto_support_weight),
        "transductive_proto_query_weight": float(transductive_proto_query_weight),
        "transductive_proto_clip": float(transductive_proto_clip),
        "transductive_proto_count": int(transductive_proto_count),
        "dense_cluster_weight": float(dense_cluster_weight),
        "dense_cluster_similarity": float(dense_cluster_similarity),
        "dense_cluster_neighbor_k": int(dense_cluster_neighbor_k),
        "dense_cluster_rounds": int(dense_cluster_rounds),
        "dense_cluster_candidate_topn": int(dense_cluster_candidate_topn),
        "dense_cluster_query_topm": int(dense_cluster_query_topm),
        "dense_cluster_clip": float(dense_cluster_clip),
        "dense_cluster_scope": str(dense_cluster_scope),
        "dense_cluster_count": int(dense_cluster_count),
        "dense_cluster_temp_proto_count": int(dense_cluster_temp_proto_count),
        "dense_cluster_adjusted_rows": int(dense_cluster_adjusted_rows),
        "query_pair_cluster_top_pairs": int(query_pair_cluster_top_pairs),
        "query_pair_cluster_similarity": float(query_pair_cluster_similarity),
        "query_pair_cluster_score_weight": float(query_pair_cluster_score_weight),
        "query_pair_cluster_query_weight": float(query_pair_cluster_query_weight),
        "query_pair_cluster_clip": float(query_pair_cluster_clip),
        "query_pair_cluster_scope": str(query_pair_cluster_scope),
        "query_pair_cluster_count": int(query_pair_cluster_count),
        "query_pair_cluster_changed": int(query_pair_cluster_changed),
        "query_pair_cluster_pairs": query_pair_cluster_pairs,
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
        "stored_aux_transform_scalars": int(2 * aux_features.shape[1])
        if aux_features is not None and float(effective_aux_score_weight) > 0.0
        else 0,
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


def _support_geometry_summary(
    *,
    features: np.ndarray,
    support_indices: np.ndarray,
    support_labels: np.ndarray,
    old_labels: list[str],
    new_labels: list[str],
    k_old: int,
    k_new: int,
) -> dict[str, float]:
    class_labels = old_labels + new_labels
    prototypes: list[np.ndarray] = []
    radii: list[float] = []
    counts: list[int] = []
    for label in class_labels:
        idx = support_indices[support_labels == label]
        counts.append(int(idx.size))
        if idx.size == 0:
            continue
        vectors = qknn._normalize_rows(features[idx])
        proto = qknn._normalize_rows(vectors.mean(axis=0, keepdims=True))[0]
        prototypes.append(proto)
        radii.append(float(np.mean(1.0 - np.clip(vectors @ proto, -1.0, 1.0))))
    if len(prototypes) >= 2:
        proto_matrix = np.stack(prototypes, axis=0)
        sim = proto_matrix @ proto_matrix.T
        offdiag = sim[~np.eye(sim.shape[0], dtype=bool)]
        max_offdiag = float(np.max(offdiag))
        p90_offdiag = float(np.quantile(offdiag, 0.90))
        mean_offdiag = float(np.mean(offdiag))
    else:
        max_offdiag = 0.0
        p90_offdiag = 0.0
        mean_offdiag = 0.0
    min_k = float(min([count for count in counts if count > 0], default=min(int(k_old), int(k_new))))
    mean_radius = float(np.mean(radii)) if radii else 0.0
    return {
        "adaptive_support_min_k": min_k,
        "adaptive_old_class_count": float(len(old_labels)),
        "adaptive_new_class_count": float(len(new_labels)),
        "adaptive_total_class_count": float(len(class_labels)),
        "adaptive_support_max_offdiag_proto_sim": max_offdiag,
        "adaptive_support_p90_offdiag_proto_sim": p90_offdiag,
        "adaptive_support_mean_offdiag_proto_sim": mean_offdiag,
        "adaptive_support_mean_radius": mean_radius,
    }


def _clip01(value: float) -> float:
    return float(np.clip(float(value), 0.0, 1.0))


def _adaptive_qknn_overrides(
    *,
    policy: str,
    geometry: dict[str, float],
    aux_available: bool,
) -> dict[str, Any]:
    name = str(policy).strip().lower()
    if name in {"", "none"}:
        return {
            "adaptive_qknn_policy": "none",
            "adaptive_support_hardness": 0.0,
            "adaptive_class_load": 0.0,
            "adaptive_k_reliability": 0.0,
        }
    if name not in {
        "dualview_support_v1",
        "stable_dualview_v1",
        "dualview_support_v2",
        "stable_dualview_v2",
        "dualview_support_v3",
        "stable_dualview_v3",
        "dualview_support_v4",
        "stable_dualview_v4",
        "dualview_support_v5",
        "stable_dualview_v5",
        "dualview_support_v6",
        "stable_dualview_v6",
        "dualview_support_v7",
        "stable_dualview_v7",
        "dualview_support_v8",
        "stable_dualview_v8",
        "dualview_support_v9",
        "stable_dualview_v9",
        "dualview_support_v10",
        "stable_dualview_v10",
        "dualview_support_v11",
        "stable_dualview_v11",
        "dualview_support_v12",
        "stable_dualview_v12",
        "dualview_support_v13",
        "stable_dualview_v13",
        "dualview_support_v14",
        "stable_dualview_v14",
        "dualview_support_v15",
        "stable_dualview_v15",
        "dualview_support_v16",
        "stable_dualview_v16",
        "dualview_support_v17",
        "stable_dualview_v17",
        "dualview_support_v18",
        "stable_dualview_v18",
        "dualview_support_v19",
        "stable_dualview_v19",
        "dualview_support_v20",
        "stable_dualview_v20",
        "dualview_support_v21",
        "stable_dualview_v21",
        "dualview_support_v22",
        "stable_dualview_v22",
        "dualview_support_v23",
        "stable_dualview_v23",
        "dualview_support_v24",
        "stable_dualview_v24",
        "dualview_support_v25",
        "stable_dualview_v25",
        "dualview_support_v27",
        "stable_dualview_v27",
        "dualview_support_v28",
        "stable_dualview_v28",
        "dualview_support_v29",
        "stable_dualview_v29",
        "dualview_support_v30",
        "stable_dualview_v30",
    }:
        raise ValueError(f"unsupported adaptive_qknn_policy: {policy}")
    use_v2 = name in {"dualview_support_v2", "stable_dualview_v2"}
    use_v3 = name in {"dualview_support_v3", "stable_dualview_v3"}
    use_v4 = name in {"dualview_support_v4", "stable_dualview_v4"}
    use_v5 = name in {"dualview_support_v5", "stable_dualview_v5"}
    use_v6 = name in {"dualview_support_v6", "stable_dualview_v6"}
    use_v7 = name in {"dualview_support_v7", "stable_dualview_v7"}
    use_v8 = name in {"dualview_support_v8", "stable_dualview_v8"}
    use_v9 = name in {"dualview_support_v9", "stable_dualview_v9"}
    use_v10 = name in {"dualview_support_v10", "stable_dualview_v10"}
    use_v11 = name in {"dualview_support_v11", "stable_dualview_v11"}
    use_v12 = name in {"dualview_support_v12", "stable_dualview_v12"}
    use_v13 = name in {"dualview_support_v13", "stable_dualview_v13"}
    use_v14 = name in {"dualview_support_v14", "stable_dualview_v14"}
    use_v15 = name in {"dualview_support_v15", "stable_dualview_v15"}
    use_v16 = name in {"dualview_support_v16", "stable_dualview_v16"}
    use_v17 = name in {"dualview_support_v17", "stable_dualview_v17"}
    use_v18 = name in {"dualview_support_v18", "stable_dualview_v18"}
    use_v19 = name in {"dualview_support_v19", "stable_dualview_v19"}
    use_v20 = name in {"dualview_support_v20", "stable_dualview_v20"}
    use_v21 = name in {"dualview_support_v21", "stable_dualview_v21"}
    use_v22 = name in {"dualview_support_v22", "stable_dualview_v22"}
    use_v23 = name in {"dualview_support_v23", "stable_dualview_v23"}
    use_v24 = name in {"dualview_support_v24", "stable_dualview_v24"}
    use_v25 = name in {"dualview_support_v25", "stable_dualview_v25"}
    use_v27 = name in {"dualview_support_v27", "stable_dualview_v27"}
    use_v28 = name in {"dualview_support_v28", "stable_dualview_v28"}
    use_v29 = name in {"dualview_support_v29", "stable_dualview_v29"}
    use_v30 = name in {"dualview_support_v30", "stable_dualview_v30"}
    use_v9 = use_v9 or use_v27 or use_v28 or use_v29 or use_v30

    min_k = float(geometry["adaptive_support_min_k"])
    new_count = float(geometry["adaptive_new_class_count"])
    max_sim = float(geometry["adaptive_support_max_offdiag_proto_sim"])
    p90_sim = float(geometry["adaptive_support_p90_offdiag_proto_sim"])
    radius = float(geometry["adaptive_support_mean_radius"])
    hardness = _clip01(max((max_sim - 0.82) / 0.16, (p90_sim - 0.68) / 0.22, (radius - 0.08) / 0.20))
    if use_v3 or use_v4 or use_v5 or use_v7 or use_v8 or use_v9 or use_v10 or use_v11 or use_v12 or use_v13 or use_v14 or use_v15 or use_v16 or use_v17 or use_v18 or use_v19 or use_v20 or use_v21 or use_v22 or use_v23 or use_v24 or use_v25:
        class_load = _clip01((new_count - 2.0) / 18.0)
    else:
        class_load = _clip01((new_count - 10.0) / 20.0)
    k_reliability = _clip01((min_k - 5.0) / 15.0)
    stable_gate = _clip01(max(hardness, 0.6 * class_load))
    enhancement_gate = _clip01((1.0 - stable_gate) * k_reliability)

    aux_weight = 0.0
    if bool(aux_available):
        aux_weight = float(np.clip(0.16 + 0.04 * stable_gate + 0.02 * class_load, 0.12, 0.24))
    source_guard_weight = float(np.clip(0.05 * stable_gate + 0.02 * class_load, 0.0, 0.07))
    core_count = int(max(1, min(int(min_k), round(np.sqrt(max(min_k, 1.0)) / 1.6))))
    core_topm = int(max(1, min(core_count, 2 + int(class_load >= 0.5))))

    overrides: dict[str, Any] = {
        "adaptive_qknn_policy": name,
        "adaptive_support_hardness": hardness,
        "adaptive_class_load": class_load,
        "adaptive_k_reliability": k_reliability,
        "adaptive_stable_gate": stable_gate,
        "adaptive_enhancement_gate": enhancement_gate,
        "transform_mode": "diag_whiten_fisher" if stable_gate >= 0.50 else "diag_fisher",
        "transform_strength": float(np.clip(0.10 + 0.40 * enhancement_gate, 0.10, 0.50)),
        "proto_mix": float(np.clip(0.40 - 0.15 * enhancement_gate, 0.25, 0.40)),
        "aux_score_weight": aux_weight,
        "pair_gaussian_similarity": float(np.clip(0.85 + 0.10 * enhancement_gate, 0.85, 0.95)),
        "pair_gaussian_weight": float(np.clip(0.02 - 0.015 * enhancement_gate, 0.005, 0.02)),
        "pair_gaussian_clip": 2.0,
        "pair_fisher_similarity": 0.90,
        "pair_fisher_weight": 0.01,
        "pair_fisher_alpha": 1.0,
        "pair_fisher_clip": 2.0,
        "ridge_head_weight": float(np.clip(0.015 * enhancement_gate, 0.0, 0.015)),
        "ridge_head_alpha": 0.01 if enhancement_gate > 0.0 else 1.0,
        "ridge_head_clip": 2.0,
        "core_proto_weight": float(np.clip(0.10 * enhancement_gate, 0.0, 0.10)),
        "core_proto_count": core_count,
        "core_proto_topm": core_topm,
        "core_proto_mode": "axis" if enhancement_gate >= 0.25 else "centroid",
        "source_guard_mode": "add_old" if source_guard_weight > 0.0 else "none",
        "source_guard_weight": source_guard_weight,
        "source_guard_conf_min": 0.0,
        "source_guard_margin_min": 0.0,
    }
    if use_v2 or use_v3 or use_v4 or use_v5 or use_v6 or use_v7 or use_v8 or use_v9 or use_v10 or use_v11 or use_v12 or use_v13 or use_v14 or use_v15 or use_v16 or use_v17 or use_v18 or use_v19 or use_v20 or use_v21 or use_v22 or use_v23 or use_v24 or use_v25:
        competition_load = class_load
        if (
            use_v3
            or use_v4
            or use_v5
            or use_v6
            or use_v7
            or use_v8
            or use_v9
            or use_v10
            or use_v11
            or use_v12
            or use_v13
            or use_v14
            or use_v15
            or use_v16
            or use_v17
            or use_v18
            or use_v19
            or use_v20
            or use_v21
            or use_v22
            or use_v23
            or use_v24
            or use_v25
        ) and new_count >= 2.0:
            competition_load = max(competition_load, 0.25)
        overrides.update(
            {
                "role_balanced_assignment": bool(
                    class_load > 0.0
                    or (
                        (
                            use_v3
                            or use_v4
                            or use_v5
                            or use_v6
                            or use_v7
                            or use_v8
                            or use_v9
                            or use_v10
                            or use_v11
                            or use_v12
                            or use_v13
                            or use_v14
                            or use_v15
                            or use_v16
                            or use_v17
                            or use_v18
                            or use_v19
                            or use_v20
                            or use_v21
                            or use_v22
                            or use_v23
                            or use_v24
                            or use_v25
                        )
                        and stable_gate >= 0.50
                    )
                ),
                "local_competition_weight": float(0.02 * competition_load * stable_gate),
                "local_competition_k": int(3 + round(4.0 * competition_load)),
                "local_competition_clip": 1.0,
                "local_competition_scope": "role",
            }
        )
    if use_v4:
        labelprop_gate = _clip01(k_reliability * stable_gate)
        labelprop_weight = float(np.clip(0.60 * labelprop_gate, 0.0, 0.20))
        overrides.update(
            {
                "labelprop_weight": labelprop_weight,
                "labelprop_k": 8,
                "labelprop_alpha": 0.8,
                "labelprop_temperature": 0.05,
                "labelprop_rounds": 10,
                "labelprop_clip": 2.0,
                "labelprop_scope": "scenario" if labelprop_weight > 0.0 else "all",
                "query_graph_weight": 0.0,
            }
        )
    if use_v5:
        labelprop_gate = _clip01(k_reliability * stable_gate)
        labelprop_weight = float(np.clip(0.90 * labelprop_gate, 0.0, 0.30))
        overrides.update(
            {
                "labelprop_weight": labelprop_weight,
                "labelprop_k": 8,
                "labelprop_alpha": 0.8,
                "labelprop_temperature": 0.10,
                "labelprop_rounds": 10,
                "labelprop_clip": 2.0,
                "labelprop_scope": "scenario" if labelprop_weight > 0.0 else "all",
                "query_graph_weight": 0.0,
            }
        )
    if use_v6:
        rescue_gate = _clip01(((new_count - 10.0) / 10.0) * stable_gate)
        labelprop_gate = _clip01(k_reliability * stable_gate * (1.0 - rescue_gate))
        labelprop_weight = float(np.clip(0.60 * labelprop_gate, 0.0, 0.20))
        support_loo_weight = float(np.clip(0.02 * rescue_gate, 0.0, 0.02))
        support_loo_top_pairs = int(max(4, min(12, round(0.40 * new_count))))
        support_loo_min_errors = int(max(3, np.ceil(0.20 * max(min_k, 1.0))))
        overrides.update(
            {
                "labelprop_weight": labelprop_weight,
                "labelprop_k": 8,
                "labelprop_alpha": 0.8,
                "labelprop_temperature": 0.08,
                "labelprop_rounds": 10,
                "labelprop_clip": 2.0,
                "labelprop_scope": "scenario" if labelprop_weight > 0.0 else "all",
                "query_graph_weight": 0.0,
                "support_loo_pair_rescue_weight": support_loo_weight,
                "support_loo_pair_rescue_top_pairs": support_loo_top_pairs,
                "support_loo_pair_rescue_min_errors": support_loo_min_errors,
                "support_loo_pair_rescue_alpha": 0.1,
                "support_loo_pair_rescue_clip": 2.0,
                "support_loo_pair_rescue_scope": "new",
            }
        )
    if use_v7 or use_v8 or use_v9 or use_v11 or use_v12 or use_v13 or use_v14 or use_v15 or use_v16 or use_v17 or use_v18 or use_v19 or use_v20 or use_v21 or use_v22 or use_v23 or use_v24 or use_v25:
        pair_gate = _clip01(max(stable_gate, class_load) * (0.35 + 0.65 * k_reliability))
        labelprop_gate = _clip01(k_reliability * stable_gate * (1.0 - 0.5 * class_load))
        labelprop_weight = float(np.clip(0.50 * labelprop_gate, 0.0, 0.18))
        pair_weight = float(np.clip(0.04 * pair_gate, 0.0, 0.04))
        pair_similarity = float(np.clip(0.94 - 0.10 * class_load - 0.05 * stable_gate, 0.80, 0.94))
        overrides.update(
            {
                "labelprop_weight": labelprop_weight,
                "labelprop_k": 8,
                "labelprop_alpha": 0.8,
                "labelprop_temperature": 0.08,
                "labelprop_rounds": 10,
                "labelprop_clip": 2.0,
                "labelprop_scope": "scenario" if labelprop_weight > 0.0 else "all",
                "query_graph_weight": 0.0,
                "pair_logreg_similarity": pair_similarity,
                "pair_logreg_weight": pair_weight,
                "pair_logreg_alpha": float(np.clip(1.0 - 0.9 * k_reliability, 0.1, 1.0)),
                "pair_logreg_clip": 2.0,
                "pair_logreg_scope": "new",
            }
        )
    if use_v8 or use_v9 or use_v11 or use_v12 or use_v13 or use_v14 or use_v15 or use_v16 or use_v17 or use_v18 or use_v19 or use_v20 or use_v21 or use_v22 or use_v23 or use_v24 or use_v25:
        low_load_residual = float(np.clip(0.20 - 0.30 * k_reliability, 0.05, 0.20))
        high_load_residual = float(np.clip(0.10 + 0.60 * k_reliability, 0.10, 0.30))
        load_blend = _clip01((class_load - 0.50) / 0.25)
        residual_weight = (1.0 - load_blend) * low_load_residual + load_blend * high_load_residual
        residual_weight = float(np.clip(stable_gate * residual_weight, 0.0, 0.30))
        residual_proto_mix = float(np.clip(0.25 + 0.15 * class_load * (1.0 - 0.5 * k_reliability), 0.25, 0.40))
        overrides.update(
            {
                "old_residual_new_weight": residual_weight,
                "old_residual_new_rank": 5,
                "old_residual_new_proto_mix": residual_proto_mix,
                "old_residual_new_clip": 2.0,
            }
        )
    if use_v9 or use_v10 or use_v11 or use_v12 or use_v13 or use_v14 or use_v15 or use_v16 or use_v17 or use_v18 or use_v19 or use_v20 or use_v21 or use_v22 or use_v23 or use_v24 or use_v25:
        rescue_gate = _clip01(max(stable_gate, class_load))
        if use_v11 or use_v12 or use_v13 or use_v14 or use_v15 or use_v16 or use_v17 or use_v18 or use_v19 or use_v20 or use_v21 or use_v22 or use_v23 or use_v24 or use_v25:
            rescue_weight = float(np.clip((0.10 + 0.30 * k_reliability) * rescue_gate, 0.05, 0.20))
        else:
            rescue_weight = float(np.clip((0.10 - 0.15 * k_reliability) * rescue_gate, 0.02, 0.10))
        rescue_top_pairs = int(max(4, min(12, round(0.40 * max(new_count, 1.0)))))
        rescue_proto_neighbors = 0
        rescue_proto_min_sim = 1.1
        if use_v20:
            rescue_weight = float(np.clip((0.04 + 0.10 * k_reliability + 0.02 * class_load) * rescue_gate, 0.04, 0.10))
            rescue_top_pairs = int(max(10, min(18, round(0.65 * max(new_count, 1.0)))))
            rescue_proto_neighbors = 2
            rescue_proto_min_sim = float(np.clip(0.72 + 0.10 * k_reliability - 0.04 * class_load, 0.68, 0.82))
        overrides.update(
            {
                "support_loo_pair_rescue_weight": rescue_weight,
                "support_loo_pair_rescue_top_pairs": rescue_top_pairs,
                "support_loo_pair_rescue_min_errors": 1,
                "support_loo_pair_rescue_alpha": 0.1,
                "support_loo_pair_rescue_clip": 2.0,
                "support_loo_pair_rescue_scope": "new",
                "support_loo_pair_rescue_proto_neighbors": rescue_proto_neighbors,
                "support_loo_pair_rescue_proto_min_sim": rescue_proto_min_sim,
            }
        )
    if use_v12:
        linear_gate = _clip01(max(stable_gate, class_load))
        linear_weight = float(np.clip((0.004 + 0.006 * k_reliability) * linear_gate, 0.0, 0.008))
        linear_top_pairs = int(max(1, min(3, round(0.15 * max(new_count, 1.0)))))
        overrides.update(
            {
                "support_loo_pair_linear_weight": linear_weight,
                "support_loo_pair_linear_top_pairs": linear_top_pairs,
                "support_loo_pair_linear_min_errors": 1,
                "support_loo_pair_linear_alpha": 10.0,
                "support_loo_pair_linear_clip": 0.5,
                "support_loo_pair_linear_scope": "new",
            }
        )
    if use_v13 or use_v14 or use_v15 or use_v16 or use_v17 or use_v18 or use_v19 or use_v20 or use_v21 or use_v22 or use_v23 or use_v24 or use_v25:
        proxy_gate = _clip01(max(stable_gate, class_load))
        proxy_weight = float(np.clip((0.10 + 0.90 * k_reliability) * proxy_gate, 0.0, 0.40))
        proxy_top_pairs = int(max(8, min(16, round(0.40 * max(new_count, 1.0)))))
        if use_v16:
            proxy_top_pairs = int(max(16, min(32, round(0.80 * max(new_count, 1.0)))))
        proxy_balance = bool(use_v14 or use_v18 or use_v20 or ((use_v15 or use_v22 or use_v23 or use_v24 or use_v25) and k_reliability < 0.25))
        proxy_bundle_rows = 4 if use_v16 else 1
        proxy_analogy = bool(use_v17 or use_v18)
        proxy_gate_enabled = bool(use_v19)
        if use_v20:
            proxy_weight = 0.0
            proxy_top_pairs = int(max(6, min(12, round(0.35 * max(new_count, 1.0)))))
        overrides.update(
            {
                "support_guided_proxy_weight": proxy_weight,
                "support_guided_proxy_top_pairs": proxy_top_pairs,
                "support_guided_proxy_clip": 2.0,
                "support_guided_proxy_min_errors": 1,
                "support_guided_proxy_scope": "new",
                "support_guided_proxy_balance": proxy_balance,
                "support_guided_proxy_bundle_rows": proxy_bundle_rows,
                "support_guided_proxy_analogy": proxy_analogy,
                "support_guided_proxy_gate": proxy_gate_enabled,
                "support_guided_proxy_gate_floor_tol": 0.0,
                "support_guided_proxy_gate_mean_tol": 0.0,
            }
        )
    if use_v21 or use_v22 or use_v23 or use_v24 or use_v25:
        cluster_gate = _clip01(max(stable_gate, class_load) * (0.55 + 0.45 * k_reliability))
        if use_v22 or use_v23 or use_v24 or use_v25:
            v22_base_weight = 0.04 * (1.0 - 0.35 * k_reliability) + 0.02 * class_load * k_reliability
            v22_overload_gate = 1.0 - _clip01((class_load - 0.55) / 0.45) * (1.0 - k_reliability)
            cluster_weight = float(
                np.clip(v22_base_weight * (0.50 + 0.50 * stable_gate) * v22_overload_gate, 0.0, 0.06)
            )
        else:
            cluster_weight = float(
                np.clip((0.05 + 0.11 * class_load + 0.07 * (1.0 - k_reliability)) * cluster_gate, 0.03, 0.16)
            )
        cluster_support_weight = float(np.clip(0.55 + 0.25 * k_reliability - 0.10 * class_load, 0.40, 0.75))
        cluster_temperature = float(np.clip(0.10 - 0.04 * k_reliability + 0.02 * class_load, 0.05, 0.12))
        overrides.update(
            {
                "query_cluster_weight": cluster_weight,
                "query_cluster_rounds": 3,
                "query_cluster_support_weight": cluster_support_weight,
                "query_cluster_temperature": cluster_temperature,
                "query_cluster_clip": 2.0,
                "query_cluster_scope": "new",
            }
        )
        if use_v22 or use_v23 or use_v24 or use_v25:
            overrides.update(
                {
                    "query_cluster_agreement_min": float(np.clip(0.35 + 0.35 * k_reliability, 0.35, 0.70)),
                    "query_cluster_margin_min": float(np.clip(0.005 + 0.025 * k_reliability, 0.005, 0.03)),
                }
            )
        if use_v21 and k_reliability < 0.15:
            overrides.update(
                {
                    "support_loo_pair_rescue_weight": 0.0,
                    "support_loo_pair_rescue_top_pairs": 0,
                }
            )
        if use_v24:
            overrides.update(
                {
                    "query_cluster_weight": float(
                        np.clip((0.08 + 0.08 * class_load + 0.02 * k_reliability) * stable_gate, 0.04, 0.12)
                    ),
                    "query_cluster_support_weight": float(
                        np.clip(0.34 + 0.24 * k_reliability - 0.08 * class_load, 0.28, 0.58)
                    ),
                    "query_cluster_temperature": float(
                        np.clip(0.08 - 0.02 * k_reliability + 0.02 * class_load, 0.06, 0.10)
                    ),
                    "query_cluster_agreement_min": float(np.clip(0.20 + 0.20 * k_reliability, 0.20, 0.40)),
                    "query_cluster_margin_min": 0.0,
                }
            )
    if use_v27 or use_v28 or use_v29 or use_v30:
        quality_gate = _clip01(max(stable_gate, class_load))
        low_k_gate = _clip01(1.0 - k_reliability)
        quality_weight = float(np.clip(0.10 * quality_gate * (0.55 + 0.45 * low_k_gate), 0.0, 0.10))
        overrides.update(
            {
                "support_quality_weight": quality_weight,
                "support_quality_floor": 0.25,
                "support_quality_margin_scale": 0.1,
                "support_bias_weight": 0.0,
            }
        )
    if use_v11 or use_v12 or use_v13 or use_v14 or use_v15 or use_v16 or use_v17 or use_v18 or use_v19 or use_v20 or use_v21 or use_v22 or use_v23 or use_v24 or use_v25:
        # ASLR: Adaptive Support-LOO Rescue. v12 adds compressed pairwise
        # linear boundaries; v13 adds compressed support-proxy direction rescue.
        # These variants do not persist raw support or query state.
        overrides.update(
            {
                "score_calibration": "none",
                "transductive_proto_weight": 0.0,
                "transductive_proto_rounds": 0,
                "transductive_proto_query_topm": 0,
                "transductive_proto_support_weight": 1.0,
                "transductive_proto_query_weight": 1.0,
                "transductive_proto_clip": 1.5,
                "dense_cluster_weight": 0.0,
                "dense_cluster_similarity": 0.9,
                "dense_cluster_neighbor_k": 0,
                "dense_cluster_rounds": 1,
                "dense_cluster_candidate_topn": 3,
                "dense_cluster_query_topm": 0,
                "dense_cluster_clip": 1.5,
                "dense_cluster_scope": "new",
            }
        )
    return overrides


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
    parser.add_argument("--adaptive_qknn_policy_grid", default="none")
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
    parser.add_argument("--support_guided_proxy_json", default="")
    parser.add_argument("--support_guided_proxy_weight_grid", default="0")
    parser.add_argument("--support_guided_proxy_top_pairs_grid", default="0")
    parser.add_argument("--support_guided_proxy_clip_grid", default="2.0")
    parser.add_argument("--support_guided_proxy_min_errors_grid", default="1")
    parser.add_argument("--support_guided_proxy_scope_grid", default="new")
    parser.add_argument("--support_guided_proxy_balance_grid", default="false")
    parser.add_argument("--support_guided_proxy_bundle_rows_grid", default="1")
    parser.add_argument("--support_guided_proxy_analogy_grid", default="false")
    parser.add_argument("--support_guided_proxy_gate_grid", default="false")
    parser.add_argument("--support_guided_proxy_gate_floor_tol_grid", default="0")
    parser.add_argument("--support_guided_proxy_gate_mean_tol_grid", default="0")
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
    parser.add_argument("--core_proto_weight_grid", default="0")
    parser.add_argument("--core_proto_count_grid", default="3")
    parser.add_argument("--core_proto_topm_grid", default="1")
    parser.add_argument("--core_proto_mode_grid", default="centroid")
    parser.add_argument("--ridge_head_weight_grid", default="0")
    parser.add_argument("--ridge_head_alpha_grid", default="1.0")
    parser.add_argument("--ridge_head_clip_grid", default="3.0")
    parser.add_argument("--subspace_proto_weight_grid", default="0")
    parser.add_argument("--subspace_proto_rank_grid", default="8")
    parser.add_argument("--subspace_proto_power_grid", default="0")
    parser.add_argument("--subspace_proto_clip_grid", default="3.0")
    parser.add_argument("--old_residual_new_weight_grid", default="0")
    parser.add_argument("--old_residual_new_rank_grid", default="2")
    parser.add_argument("--old_residual_new_proto_mix_grid", default="0.4")
    parser.add_argument("--old_residual_new_clip_grid", default="2.0")
    parser.add_argument("--domain_refine_key_grid", default="none")
    parser.add_argument("--domain_refine_weight_grid", default="0")
    parser.add_argument("--domain_refine_scope_grid", default="all")
    parser.add_argument("--class_diag_metric_weight_grid", default="0")
    parser.add_argument("--class_diag_metric_similarity_grid", default="0.9")
    parser.add_argument("--class_diag_metric_alpha_grid", default="0.01")
    parser.add_argument("--class_diag_metric_power_grid", default="0.5")
    parser.add_argument("--class_diag_metric_clip_grid", default="2.0")
    parser.add_argument("--support_bias_weight_grid", default="0")
    parser.add_argument("--support_bias_step_grid", default="0.01")
    parser.add_argument("--support_bias_rounds_grid", default="4")
    parser.add_argument("--support_quality_weight_grid", default="0")
    parser.add_argument("--support_quality_floor_grid", default="0.25")
    parser.add_argument("--support_quality_margin_scale_grid", default="0.1")
    parser.add_argument("--support_loo_pair_rescue_weight_grid", default="0")
    parser.add_argument("--support_loo_pair_rescue_top_pairs_grid", default="0")
    parser.add_argument("--support_loo_pair_rescue_min_errors_grid", default="1")
    parser.add_argument("--support_loo_pair_rescue_alpha_grid", default="0.1")
    parser.add_argument("--support_loo_pair_rescue_clip_grid", default="2.0")
    parser.add_argument("--support_loo_pair_rescue_scope_grid", default="new")
    parser.add_argument("--support_loo_pair_rescue_proto_neighbors_grid", default="0")
    parser.add_argument("--support_loo_pair_rescue_proto_min_sim_grid", default="1.1")
    parser.add_argument("--support_loo_pair_linear_weight_grid", default="0")
    parser.add_argument("--support_loo_pair_linear_top_pairs_grid", default="0")
    parser.add_argument("--support_loo_pair_linear_min_errors_grid", default="1")
    parser.add_argument("--support_loo_pair_linear_alpha_grid", default="0.1")
    parser.add_argument("--support_loo_pair_linear_clip_grid", default="1.5")
    parser.add_argument("--support_loo_pair_linear_scope_grid", default="new")
    parser.add_argument("--mahal_proto_weight_grid", default="0")
    parser.add_argument("--mahal_proto_alpha_grid", default="1.0")
    parser.add_argument("--mahal_proto_diag_mix_grid", default="0.5")
    parser.add_argument("--mahal_proto_clip_grid", default="3.0")
    parser.add_argument("--score_calibration_grid", default="none")
    parser.add_argument("--assignment_margin_weight_grid", default="0")
    parser.add_argument("--assignment_margin_clip_grid", default="1.0")
    parser.add_argument("--labelprop_weight_grid", default="0")
    parser.add_argument("--labelprop_k_grid", default="8")
    parser.add_argument("--labelprop_alpha_grid", default="0.8")
    parser.add_argument("--labelprop_temperature_grid", default="0.05")
    parser.add_argument("--labelprop_rounds_grid", default="10")
    parser.add_argument("--labelprop_clip_grid", default="2.0")
    parser.add_argument("--labelprop_scope_grid", default="all")
    parser.add_argument("--query_graph_weight_grid", default="0")
    parser.add_argument("--query_graph_k_grid", default="8")
    parser.add_argument("--query_graph_temperature_grid", default="0.05")
    parser.add_argument("--query_graph_rounds_grid", default="1")
    parser.add_argument("--query_graph_scope_grid", default="all")
    parser.add_argument("--query_cluster_weight_grid", default="0")
    parser.add_argument("--query_cluster_rounds_grid", default="3")
    parser.add_argument("--query_cluster_support_weight_grid", default="0.6")
    parser.add_argument("--query_cluster_temperature_grid", default="0.08")
    parser.add_argument("--query_cluster_clip_grid", default="2.0")
    parser.add_argument("--query_cluster_scope_grid", default="new")
    parser.add_argument("--query_cluster_agreement_min_grid", default="0.70")
    parser.add_argument("--query_cluster_margin_min_grid", default="-1")
    parser.add_argument("--local_competition_weight_grid", default="0")
    parser.add_argument("--local_competition_k_grid", default="4")
    parser.add_argument("--local_competition_clip_grid", default="1.0")
    parser.add_argument("--local_competition_scope_grid", default="role")
    parser.add_argument("--query_proto_refine_weight_grid", default="0")
    parser.add_argument("--query_proto_refine_topm_grid", default="0")
    parser.add_argument("--query_proto_refine_clip_grid", default="2.0")
    parser.add_argument("--transductive_proto_weight_grid", default="0")
    parser.add_argument("--transductive_proto_rounds_grid", default="0")
    parser.add_argument("--transductive_proto_query_topm_grid", default="50")
    parser.add_argument("--transductive_proto_support_weight_grid", default="1.0")
    parser.add_argument("--transductive_proto_query_weight_grid", default="1.0")
    parser.add_argument("--transductive_proto_clip_grid", default="1.5")
    parser.add_argument("--dense_cluster_weight_grid", default="0")
    parser.add_argument("--dense_cluster_similarity_grid", default="0.9")
    parser.add_argument("--dense_cluster_neighbor_k_grid", default="0")
    parser.add_argument("--dense_cluster_rounds_grid", default="1")
    parser.add_argument("--dense_cluster_candidate_topn_grid", default="3")
    parser.add_argument("--dense_cluster_query_topm_grid", default="50")
    parser.add_argument("--dense_cluster_clip_grid", default="1.5")
    parser.add_argument("--dense_cluster_scope_grid", default="new")
    parser.add_argument("--query_pair_cluster_top_pairs_grid", default="0")
    parser.add_argument("--query_pair_cluster_similarity_grid", default="0.78")
    parser.add_argument("--query_pair_cluster_score_weight_grid", default="1.0")
    parser.add_argument("--query_pair_cluster_query_weight_grid", default="0")
    parser.add_argument("--query_pair_cluster_clip_grid", default="2.0")
    parser.add_argument("--query_pair_cluster_scope_grid", default="new")
    parser.add_argument("--source_guard_mode_grid", default="none")
    parser.add_argument("--source_guard_weight_grid", default="0")
    parser.add_argument("--source_guard_conf_min_grid", default="0")
    parser.add_argument("--source_guard_margin_min_grid", default="0")
    parser.add_argument("--source_proto_anchor_mode_grid", default="none")
    parser.add_argument("--source_proto_anchor_weight_grid", default="0")
    parser.add_argument("--source_proto_anchor_center_grid", default="0")
    parser.add_argument("--scenario_aware", action="store_true")
    parser.add_argument("--balanced_assignment", action="store_true")
    parser.add_argument("--role_balanced_assignment", action="store_true")
    parser.add_argument("--fast_role_balanced_assignment", action="store_true")
    parser.add_argument("--scenario_balanced_assignment", action="store_true")
    parser.add_argument("--exclude_pool_from_query", action="store_true")
    parser.add_argument("--old_target", type=float, default=0.80)
    parser.add_argument("--old_floor", type=float, default=0.75)
    parser.add_argument("--seen_new_target", type=float, default=0.75)
    parser.add_argument("--seen_new_floor", type=float, default=0.75)
    args = parser.parse_args()

    feature_path = Path(args.feature_npz)
    aux_feature_path = Path(args.aux_feature_npz) if str(args.aux_feature_npz).strip() else None
    data = np.load(feature_path, allow_pickle=True)
    features = qknn._normalize_rows(data["features"])
    tx_ids = np.asarray(data["tx_ids"], dtype=object).astype(str)
    roles = np.asarray(data["dataset_role"], dtype=object).astype(str)
    logits = np.asarray(data["tx_logits"], dtype=np.float64)
    scenarios = np.asarray(data["sat_scenarios"], dtype=object).astype(str)
    rx_ids = np.asarray(data["rx_ids"], dtype=object).astype(str) if "rx_ids" in data.files else np.asarray([""] * int(tx_ids.size), dtype=object)
    day_ids = np.asarray(data["day_ids"], dtype=object).astype(str) if "day_ids" in data.files else np.asarray([""] * int(tx_ids.size), dtype=object)
    channel_views = (
        np.asarray(data["channel_views"], dtype=object).astype(str)
        if "channel_views" in data.files
        else np.asarray([""] * int(tx_ids.size), dtype=object)
    )
    aux_features = None
    if aux_feature_path is not None:
        aux_data = np.load(aux_feature_path, allow_pickle=True)
        aux_features = qknn._normalize_rows(aux_data["features"])
        required_aux_keys = {
            "tx_ids": tx_ids,
            "dataset_role": roles,
            "sat_scenarios": scenarios,
            "rx_ids": rx_ids,
            "channel_views": channel_views,
        }
        for optional_key in ("day_ids", "eq_ids", "sig_ids"):
            if optional_key in data.files or optional_key in aux_data.files:
                if optional_key not in data.files or optional_key not in aux_data.files:
                    raise ValueError(f"aux_feature_npz missing alignment key {optional_key}: {args.aux_feature_npz}")
                required_aux_keys[optional_key] = np.asarray(data[optional_key], dtype=object).astype(str)
        for key, primary in required_aux_keys.items():
            if key not in aux_data.files:
                raise ValueError(f"aux_feature_npz missing alignment key {key}: {args.aux_feature_npz}")
            aux_values = np.asarray(aux_data[key], dtype=object).astype(str)
            if aux_values.shape != primary.shape or not bool(np.all(aux_values == primary)):
                raise ValueError(f"aux_feature_npz metadata mismatch for {key}: {args.aux_feature_npz}")
    old_labels = qknn._parse_csv(args.old_tx_ids)
    new_labels = qknn._parse_csv(args.new_tx_ids)
    support_guided_proxy_rows: list[dict[str, Any]] = []
    if str(args.support_guided_proxy_json).strip():
        proxy_manifest = json.loads(Path(args.support_guided_proxy_json).read_text(encoding="utf-8"))
        support_guided_proxy_rows = list(proxy_manifest.get("candidate_rows", []))
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
            qknn._parse_float_csv(args.support_guided_proxy_weight_grid),
            qknn._parse_int_csv(args.support_guided_proxy_top_pairs_grid),
            qknn._parse_float_csv(args.support_guided_proxy_clip_grid),
            qknn._parse_int_csv(args.support_guided_proxy_min_errors_grid),
            qknn._parse_csv(args.support_guided_proxy_scope_grid),
            qknn._parse_csv(args.support_guided_proxy_balance_grid),
            qknn._parse_int_csv(args.support_guided_proxy_bundle_rows_grid),
            qknn._parse_csv(args.support_guided_proxy_analogy_grid),
            qknn._parse_csv(args.support_guided_proxy_gate_grid),
            qknn._parse_float_csv(args.support_guided_proxy_gate_floor_tol_grid),
            qknn._parse_float_csv(args.support_guided_proxy_gate_mean_tol_grid),
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
            qknn._parse_float_csv(args.core_proto_weight_grid),
            qknn._parse_int_csv(args.core_proto_count_grid),
            qknn._parse_int_csv(args.core_proto_topm_grid),
            qknn._parse_csv(args.core_proto_mode_grid),
            qknn._parse_float_csv(args.ridge_head_weight_grid),
            qknn._parse_float_csv(args.ridge_head_alpha_grid),
            qknn._parse_float_csv(args.ridge_head_clip_grid),
            qknn._parse_float_csv(args.subspace_proto_weight_grid),
            qknn._parse_int_csv(args.subspace_proto_rank_grid),
            qknn._parse_float_csv(args.subspace_proto_power_grid),
            qknn._parse_float_csv(args.subspace_proto_clip_grid),
            qknn._parse_float_csv(args.old_residual_new_weight_grid),
            qknn._parse_int_csv(args.old_residual_new_rank_grid),
            qknn._parse_float_csv(args.old_residual_new_proto_mix_grid),
            qknn._parse_float_csv(args.old_residual_new_clip_grid),
            qknn._parse_csv(args.domain_refine_key_grid),
            qknn._parse_float_csv(args.domain_refine_weight_grid),
            qknn._parse_csv(args.domain_refine_scope_grid),
            qknn._parse_float_csv(args.class_diag_metric_weight_grid),
            qknn._parse_float_csv(args.class_diag_metric_similarity_grid),
            qknn._parse_float_csv(args.class_diag_metric_alpha_grid),
            qknn._parse_float_csv(args.class_diag_metric_power_grid),
            qknn._parse_float_csv(args.class_diag_metric_clip_grid),
            qknn._parse_float_csv(args.support_bias_weight_grid),
            qknn._parse_float_csv(args.support_bias_step_grid),
            qknn._parse_int_csv(args.support_bias_rounds_grid),
            qknn._parse_float_csv(args.support_quality_weight_grid),
            qknn._parse_float_csv(args.support_quality_floor_grid),
            qknn._parse_float_csv(args.support_quality_margin_scale_grid),
            qknn._parse_float_csv(args.support_loo_pair_rescue_weight_grid),
            qknn._parse_int_csv(args.support_loo_pair_rescue_top_pairs_grid),
            qknn._parse_int_csv(args.support_loo_pair_rescue_min_errors_grid),
            qknn._parse_float_csv(args.support_loo_pair_rescue_alpha_grid),
            qknn._parse_float_csv(args.support_loo_pair_rescue_clip_grid),
            qknn._parse_csv(args.support_loo_pair_rescue_scope_grid),
            qknn._parse_int_csv(args.support_loo_pair_rescue_proto_neighbors_grid),
            qknn._parse_float_csv(args.support_loo_pair_rescue_proto_min_sim_grid),
            qknn._parse_float_csv(args.support_loo_pair_linear_weight_grid),
            qknn._parse_int_csv(args.support_loo_pair_linear_top_pairs_grid),
            qknn._parse_int_csv(args.support_loo_pair_linear_min_errors_grid),
            qknn._parse_float_csv(args.support_loo_pair_linear_alpha_grid),
            qknn._parse_float_csv(args.support_loo_pair_linear_clip_grid),
            qknn._parse_csv(args.support_loo_pair_linear_scope_grid),
            qknn._parse_float_csv(args.mahal_proto_weight_grid),
            qknn._parse_float_csv(args.mahal_proto_alpha_grid),
            qknn._parse_float_csv(args.mahal_proto_diag_mix_grid),
            qknn._parse_float_csv(args.mahal_proto_clip_grid),
            qknn._parse_csv(args.score_calibration_grid),
            qknn._parse_float_csv(args.assignment_margin_weight_grid),
            qknn._parse_float_csv(args.assignment_margin_clip_grid),
            qknn._parse_float_csv(args.labelprop_weight_grid),
            qknn._parse_int_csv(args.labelprop_k_grid),
            qknn._parse_float_csv(args.labelprop_alpha_grid),
            qknn._parse_float_csv(args.labelprop_temperature_grid),
            qknn._parse_int_csv(args.labelprop_rounds_grid),
            qknn._parse_float_csv(args.labelprop_clip_grid),
            qknn._parse_csv(args.labelprop_scope_grid),
            qknn._parse_float_csv(args.query_graph_weight_grid),
            qknn._parse_int_csv(args.query_graph_k_grid),
            qknn._parse_float_csv(args.query_graph_temperature_grid),
            qknn._parse_int_csv(args.query_graph_rounds_grid),
            qknn._parse_csv(args.query_graph_scope_grid),
            qknn._parse_float_csv(args.query_cluster_weight_grid),
            qknn._parse_int_csv(args.query_cluster_rounds_grid),
            qknn._parse_float_csv(args.query_cluster_support_weight_grid),
            qknn._parse_float_csv(args.query_cluster_temperature_grid),
            qknn._parse_float_csv(args.query_cluster_clip_grid),
            qknn._parse_csv(args.query_cluster_scope_grid),
            qknn._parse_float_csv(args.query_cluster_agreement_min_grid),
            qknn._parse_float_csv(args.query_cluster_margin_min_grid),
            qknn._parse_float_csv(args.local_competition_weight_grid),
            qknn._parse_int_csv(args.local_competition_k_grid),
            qknn._parse_float_csv(args.local_competition_clip_grid),
            qknn._parse_csv(args.local_competition_scope_grid),
            qknn._parse_float_csv(args.query_proto_refine_weight_grid),
            qknn._parse_int_csv(args.query_proto_refine_topm_grid),
            qknn._parse_float_csv(args.query_proto_refine_clip_grid),
            qknn._parse_float_csv(args.transductive_proto_weight_grid),
            qknn._parse_int_csv(args.transductive_proto_rounds_grid),
            qknn._parse_int_csv(args.transductive_proto_query_topm_grid),
            qknn._parse_float_csv(args.transductive_proto_support_weight_grid),
            qknn._parse_float_csv(args.transductive_proto_query_weight_grid),
            qknn._parse_float_csv(args.transductive_proto_clip_grid),
            qknn._parse_float_csv(args.dense_cluster_weight_grid),
            qknn._parse_float_csv(args.dense_cluster_similarity_grid),
            qknn._parse_int_csv(args.dense_cluster_neighbor_k_grid),
            qknn._parse_int_csv(args.dense_cluster_rounds_grid),
            qknn._parse_int_csv(args.dense_cluster_candidate_topn_grid),
            qknn._parse_int_csv(args.dense_cluster_query_topm_grid),
            qknn._parse_float_csv(args.dense_cluster_clip_grid),
            qknn._parse_csv(args.dense_cluster_scope_grid),
            qknn._parse_int_csv(args.query_pair_cluster_top_pairs_grid),
            qknn._parse_float_csv(args.query_pair_cluster_similarity_grid),
            qknn._parse_float_csv(args.query_pair_cluster_score_weight_grid),
            qknn._parse_float_csv(args.query_pair_cluster_query_weight_grid),
            qknn._parse_float_csv(args.query_pair_cluster_clip_grid),
            qknn._parse_csv(args.query_pair_cluster_scope_grid),
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
            split_fingerprint = _split_fingerprint(old_splits, new_splits, old_labels, new_labels)
            support_indices, support_labels = _collect_support(old_splits, new_splits, old_labels, new_labels)
            support_geometry = _support_geometry_summary(
                features=features,
                support_indices=support_indices,
                support_labels=support_labels,
                old_labels=old_labels,
                new_labels=new_labels,
                k_old=int(args.k_old),
                k_new=int(args.k_new),
            )
            for adaptive_policy in qknn._parse_csv(args.adaptive_qknn_policy_grid):
                adaptive_overrides = _adaptive_qknn_overrides(
                    policy=str(adaptive_policy),
                    geometry=support_geometry,
                    aux_available=aux_features is not None,
                )
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
                    support_guided_proxy_weight,
                    support_guided_proxy_top_pairs,
                    support_guided_proxy_clip,
                    support_guided_proxy_min_errors,
                    support_guided_proxy_scope,
                    support_guided_proxy_balance,
                    support_guided_proxy_bundle_rows,
                    support_guided_proxy_analogy,
                    support_guided_proxy_gate,
                    support_guided_proxy_gate_floor_tol,
                    support_guided_proxy_gate_mean_tol,
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
                    core_proto_weight,
                    core_proto_count,
                    core_proto_topm,
                    core_proto_mode,
                    ridge_head_weight,
                    ridge_head_alpha,
                    ridge_head_clip,
                    subspace_proto_weight,
                    subspace_proto_rank,
                    subspace_proto_power,
                    subspace_proto_clip,
                    old_residual_new_weight,
                    old_residual_new_rank,
                    old_residual_new_proto_mix,
                    old_residual_new_clip,
                    domain_refine_key,
                    domain_refine_weight,
                    domain_refine_scope,
                    class_diag_metric_weight,
                    class_diag_metric_similarity,
                    class_diag_metric_alpha,
                    class_diag_metric_power,
                    class_diag_metric_clip,
                    support_bias_weight,
                    support_bias_step,
                    support_bias_rounds,
                    support_quality_weight,
                    support_quality_floor,
                    support_quality_margin_scale,
                    support_loo_pair_rescue_weight,
                    support_loo_pair_rescue_top_pairs,
                    support_loo_pair_rescue_min_errors,
                    support_loo_pair_rescue_alpha,
                    support_loo_pair_rescue_clip,
                    support_loo_pair_rescue_scope,
                    support_loo_pair_rescue_proto_neighbors,
                    support_loo_pair_rescue_proto_min_sim,
                    support_loo_pair_linear_weight,
                    support_loo_pair_linear_top_pairs,
                    support_loo_pair_linear_min_errors,
                    support_loo_pair_linear_alpha,
                    support_loo_pair_linear_clip,
                    support_loo_pair_linear_scope,
                    mahal_proto_weight,
                    mahal_proto_alpha,
                    mahal_proto_diag_mix,
                    mahal_proto_clip,
                    score_calibration,
                    assignment_margin_weight,
                    assignment_margin_clip,
                    labelprop_weight,
                    labelprop_k,
                    labelprop_alpha,
                    labelprop_temperature,
                    labelprop_rounds,
                    labelprop_clip,
                    labelprop_scope,
                    query_graph_weight,
                    query_graph_k,
                    query_graph_temperature,
                    query_graph_rounds,
                    query_graph_scope,
                    query_cluster_weight,
                    query_cluster_rounds,
                    query_cluster_support_weight,
                    query_cluster_temperature,
                    query_cluster_clip,
                    query_cluster_scope,
                    query_cluster_agreement_min,
                    query_cluster_margin_min,
                    local_competition_weight,
                    local_competition_k,
                    local_competition_clip,
                    local_competition_scope,
                    query_proto_refine_weight,
                    query_proto_refine_topm,
                    query_proto_refine_clip,
                    transductive_proto_weight,
                    transductive_proto_rounds,
                    transductive_proto_query_topm,
                    transductive_proto_support_weight,
                    transductive_proto_query_weight,
                    transductive_proto_clip,
                    dense_cluster_weight,
                    dense_cluster_similarity,
                    dense_cluster_neighbor_k,
                    dense_cluster_rounds,
                    dense_cluster_candidate_topn,
                    dense_cluster_query_topm,
                    dense_cluster_clip,
                    dense_cluster_scope,
                    query_pair_cluster_top_pairs,
                    query_pair_cluster_similarity,
                    query_pair_cluster_score_weight,
                    query_pair_cluster_query_weight,
                    query_pair_cluster_clip,
                    query_pair_cluster_scope,
                    source_guard_mode,
                    source_guard_weight,
                    source_guard_conf_min,
                    source_guard_margin_min,
                    source_proto_anchor_mode,
                    source_proto_anchor_weight,
                    source_proto_anchor_center,
                ) in search_grid:
                    params: dict[str, Any] = {
                        "mode": mode,
                        "strength": float(strength),
                        "proto_mix": float(proto_mix),
                        "aux_score_weight": float(aux_score_weight),
                        "pair_gaussian_similarity": float(pair_gaussian_similarity),
                        "pair_gaussian_weight": float(pair_gaussian_weight),
                        "pair_gaussian_clip": float(pair_gaussian_clip),
                        "pair_fisher_similarity": float(pair_fisher_similarity),
                        "pair_fisher_weight": float(pair_fisher_weight),
                        "pair_fisher_alpha": float(pair_fisher_alpha),
                        "pair_fisher_clip": float(pair_fisher_clip),
                        "pair_logreg_similarity": float(pair_logreg_similarity),
                        "pair_logreg_weight": float(pair_logreg_weight),
                        "pair_logreg_alpha": float(pair_logreg_alpha),
                        "pair_logreg_clip": float(pair_logreg_clip),
                        "pair_logreg_scope": str(pair_logreg_scope),
                        "core_proto_weight": float(core_proto_weight),
                        "core_proto_count": int(core_proto_count),
                        "core_proto_topm": int(core_proto_topm),
                        "core_proto_mode": str(core_proto_mode),
                        "ridge_head_weight": float(ridge_head_weight),
                        "ridge_head_alpha": float(ridge_head_alpha),
                        "ridge_head_clip": float(ridge_head_clip),
                        "old_residual_new_weight": float(old_residual_new_weight),
                        "old_residual_new_rank": int(old_residual_new_rank),
                        "old_residual_new_proto_mix": float(old_residual_new_proto_mix),
                        "old_residual_new_clip": float(old_residual_new_clip),
                        "source_guard_mode": str(source_guard_mode),
                        "source_guard_weight": float(source_guard_weight),
                        "source_guard_conf_min": float(source_guard_conf_min),
                        "source_guard_margin_min": float(source_guard_margin_min),
                        "support_guided_proxy_weight": float(support_guided_proxy_weight),
                        "support_guided_proxy_top_pairs": int(support_guided_proxy_top_pairs),
                        "support_guided_proxy_clip": float(support_guided_proxy_clip),
                        "support_guided_proxy_min_errors": int(support_guided_proxy_min_errors),
                        "support_guided_proxy_scope": str(support_guided_proxy_scope),
                        "support_guided_proxy_balance": str(support_guided_proxy_balance).lower() == "true",
                        "support_guided_proxy_bundle_rows": int(support_guided_proxy_bundle_rows),
                        "support_guided_proxy_analogy": str(support_guided_proxy_analogy).lower() == "true",
                        "support_guided_proxy_gate": str(support_guided_proxy_gate).lower() == "true",
                        "support_guided_proxy_gate_floor_tol": float(support_guided_proxy_gate_floor_tol),
                        "support_guided_proxy_gate_mean_tol": float(support_guided_proxy_gate_mean_tol),
                        "role_balanced_assignment": bool(args.role_balanced_assignment),
                        "local_competition_weight": float(local_competition_weight),
                        "local_competition_k": int(local_competition_k),
                        "local_competition_clip": float(local_competition_clip),
                        "local_competition_scope": str(local_competition_scope),
                        "support_bias_weight": float(support_bias_weight),
                        "support_bias_step": float(support_bias_step),
                        "support_bias_rounds": int(support_bias_rounds),
                        "support_quality_weight": float(support_quality_weight),
                        "support_quality_floor": float(support_quality_floor),
                        "support_quality_margin_scale": float(support_quality_margin_scale),
                        "support_loo_pair_rescue_weight": float(support_loo_pair_rescue_weight),
                        "support_loo_pair_rescue_top_pairs": int(support_loo_pair_rescue_top_pairs),
                        "support_loo_pair_rescue_min_errors": int(support_loo_pair_rescue_min_errors),
                        "support_loo_pair_rescue_alpha": float(support_loo_pair_rescue_alpha),
                        "support_loo_pair_rescue_clip": float(support_loo_pair_rescue_clip),
                        "support_loo_pair_rescue_scope": str(support_loo_pair_rescue_scope),
                        "support_loo_pair_rescue_proto_neighbors": int(support_loo_pair_rescue_proto_neighbors),
                        "support_loo_pair_rescue_proto_min_sim": float(support_loo_pair_rescue_proto_min_sim),
                        "support_loo_pair_linear_weight": float(support_loo_pair_linear_weight),
                        "support_loo_pair_linear_top_pairs": int(support_loo_pair_linear_top_pairs),
                        "support_loo_pair_linear_min_errors": int(support_loo_pair_linear_min_errors),
                        "support_loo_pair_linear_alpha": float(support_loo_pair_linear_alpha),
                        "support_loo_pair_linear_clip": float(support_loo_pair_linear_clip),
                        "support_loo_pair_linear_scope": str(support_loo_pair_linear_scope),
                        "labelprop_weight": float(labelprop_weight),
                        "labelprop_k": int(labelprop_k),
                        "labelprop_alpha": float(labelprop_alpha),
                        "labelprop_temperature": float(labelprop_temperature),
                        "labelprop_rounds": int(labelprop_rounds),
                        "labelprop_clip": float(labelprop_clip),
                        "labelprop_scope": str(labelprop_scope),
                        "query_graph_weight": float(query_graph_weight),
                        "query_cluster_weight": float(query_cluster_weight),
                        "query_cluster_rounds": int(query_cluster_rounds),
                        "query_cluster_support_weight": float(query_cluster_support_weight),
                        "query_cluster_temperature": float(query_cluster_temperature),
                        "query_cluster_clip": float(query_cluster_clip),
                        "query_cluster_scope": str(query_cluster_scope),
                        "query_cluster_agreement_min": float(query_cluster_agreement_min),
                        "query_cluster_margin_min": float(query_cluster_margin_min),
                        "query_proto_refine_weight": float(query_proto_refine_weight),
                        "query_proto_refine_topm": int(query_proto_refine_topm),
                        "query_proto_refine_clip": float(query_proto_refine_clip),
                        "transductive_proto_weight": float(transductive_proto_weight),
                        "transductive_proto_rounds": int(transductive_proto_rounds),
                        "transductive_proto_query_topm": int(transductive_proto_query_topm),
                        "transductive_proto_support_weight": float(transductive_proto_support_weight),
                        "transductive_proto_query_weight": float(transductive_proto_query_weight),
                        "transductive_proto_clip": float(transductive_proto_clip),
                        "dense_cluster_weight": float(dense_cluster_weight),
                        "dense_cluster_similarity": float(dense_cluster_similarity),
                        "dense_cluster_neighbor_k": int(dense_cluster_neighbor_k),
                        "dense_cluster_rounds": int(dense_cluster_rounds),
                        "dense_cluster_candidate_topn": int(dense_cluster_candidate_topn),
                        "dense_cluster_query_topm": int(dense_cluster_query_topm),
                        "dense_cluster_clip": float(dense_cluster_clip),
                        "dense_cluster_scope": str(dense_cluster_scope),
                        "query_pair_cluster_top_pairs": int(query_pair_cluster_top_pairs),
                        "query_pair_cluster_similarity": float(query_pair_cluster_similarity),
                        "query_pair_cluster_score_weight": float(query_pair_cluster_score_weight),
                        "query_pair_cluster_query_weight": float(query_pair_cluster_query_weight),
                        "query_pair_cluster_clip": float(query_pair_cluster_clip),
                        "query_pair_cluster_scope": str(query_pair_cluster_scope),
                    }
                    params.update(
                        {
                            "mode": adaptive_overrides.get("transform_mode", params["mode"]),
                            "strength": adaptive_overrides.get("transform_strength", params["strength"]),
                            "proto_mix": adaptive_overrides.get("proto_mix", params["proto_mix"]),
                            "aux_score_weight": adaptive_overrides.get("aux_score_weight", params["aux_score_weight"]),
                            "pair_gaussian_similarity": adaptive_overrides.get("pair_gaussian_similarity", params["pair_gaussian_similarity"]),
                            "pair_gaussian_weight": adaptive_overrides.get("pair_gaussian_weight", params["pair_gaussian_weight"]),
                            "pair_gaussian_clip": adaptive_overrides.get("pair_gaussian_clip", params["pair_gaussian_clip"]),
                            "pair_fisher_similarity": adaptive_overrides.get("pair_fisher_similarity", params["pair_fisher_similarity"]),
                            "pair_fisher_weight": adaptive_overrides.get("pair_fisher_weight", params["pair_fisher_weight"]),
                            "pair_fisher_alpha": adaptive_overrides.get("pair_fisher_alpha", params["pair_fisher_alpha"]),
                            "pair_fisher_clip": adaptive_overrides.get("pair_fisher_clip", params["pair_fisher_clip"]),
                            "pair_logreg_similarity": adaptive_overrides.get(
                                "pair_logreg_similarity", params["pair_logreg_similarity"]
                            ),
                            "pair_logreg_weight": adaptive_overrides.get("pair_logreg_weight", params["pair_logreg_weight"]),
                            "pair_logreg_alpha": adaptive_overrides.get("pair_logreg_alpha", params["pair_logreg_alpha"]),
                            "pair_logreg_clip": adaptive_overrides.get("pair_logreg_clip", params["pair_logreg_clip"]),
                            "pair_logreg_scope": adaptive_overrides.get("pair_logreg_scope", params["pair_logreg_scope"]),
                            "core_proto_weight": adaptive_overrides.get("core_proto_weight", params["core_proto_weight"]),
                            "core_proto_count": adaptive_overrides.get("core_proto_count", params["core_proto_count"]),
                            "core_proto_topm": adaptive_overrides.get("core_proto_topm", params["core_proto_topm"]),
                            "core_proto_mode": adaptive_overrides.get("core_proto_mode", params["core_proto_mode"]),
                            "ridge_head_weight": adaptive_overrides.get("ridge_head_weight", params["ridge_head_weight"]),
                            "ridge_head_alpha": adaptive_overrides.get("ridge_head_alpha", params["ridge_head_alpha"]),
                            "ridge_head_clip": adaptive_overrides.get("ridge_head_clip", params["ridge_head_clip"]),
                            "old_residual_new_weight": adaptive_overrides.get(
                                "old_residual_new_weight", params["old_residual_new_weight"]
                            ),
                            "old_residual_new_rank": adaptive_overrides.get(
                                "old_residual_new_rank", params["old_residual_new_rank"]
                            ),
                            "old_residual_new_proto_mix": adaptive_overrides.get(
                                "old_residual_new_proto_mix", params["old_residual_new_proto_mix"]
                            ),
                            "old_residual_new_clip": adaptive_overrides.get(
                                "old_residual_new_clip", params["old_residual_new_clip"]
                            ),
                            "source_guard_mode": adaptive_overrides.get("source_guard_mode", params["source_guard_mode"]),
                            "source_guard_weight": adaptive_overrides.get("source_guard_weight", params["source_guard_weight"]),
                            "source_guard_conf_min": adaptive_overrides.get("source_guard_conf_min", params["source_guard_conf_min"]),
                            "source_guard_margin_min": adaptive_overrides.get("source_guard_margin_min", params["source_guard_margin_min"]),
                            "support_guided_proxy_weight": adaptive_overrides.get(
                                "support_guided_proxy_weight", params["support_guided_proxy_weight"]
                            ),
                            "support_guided_proxy_top_pairs": adaptive_overrides.get(
                                "support_guided_proxy_top_pairs", params["support_guided_proxy_top_pairs"]
                            ),
                            "support_guided_proxy_clip": adaptive_overrides.get(
                                "support_guided_proxy_clip", params["support_guided_proxy_clip"]
                            ),
                            "support_guided_proxy_min_errors": adaptive_overrides.get(
                                "support_guided_proxy_min_errors", params["support_guided_proxy_min_errors"]
                            ),
                            "support_guided_proxy_scope": adaptive_overrides.get(
                                "support_guided_proxy_scope", params["support_guided_proxy_scope"]
                            ),
                            "support_guided_proxy_balance": adaptive_overrides.get(
                                "support_guided_proxy_balance", params["support_guided_proxy_balance"]
                            ),
                            "support_guided_proxy_bundle_rows": adaptive_overrides.get(
                                "support_guided_proxy_bundle_rows", params["support_guided_proxy_bundle_rows"]
                            ),
                            "support_guided_proxy_analogy": adaptive_overrides.get(
                                "support_guided_proxy_analogy", params["support_guided_proxy_analogy"]
                            ),
                            "support_guided_proxy_gate": adaptive_overrides.get(
                                "support_guided_proxy_gate", params["support_guided_proxy_gate"]
                            ),
                            "support_guided_proxy_gate_floor_tol": adaptive_overrides.get(
                                "support_guided_proxy_gate_floor_tol", params["support_guided_proxy_gate_floor_tol"]
                            ),
                            "support_guided_proxy_gate_mean_tol": adaptive_overrides.get(
                                "support_guided_proxy_gate_mean_tol", params["support_guided_proxy_gate_mean_tol"]
                            ),
                            "role_balanced_assignment": adaptive_overrides.get(
                                "role_balanced_assignment", params["role_balanced_assignment"]
                            ),
                            "local_competition_weight": adaptive_overrides.get(
                                "local_competition_weight", params["local_competition_weight"]
                            ),
                            "local_competition_k": adaptive_overrides.get("local_competition_k", params["local_competition_k"]),
                            "local_competition_clip": adaptive_overrides.get(
                                "local_competition_clip", params["local_competition_clip"]
                            ),
                            "local_competition_scope": adaptive_overrides.get(
                                "local_competition_scope", params["local_competition_scope"]
                            ),
                            "support_bias_weight": adaptive_overrides.get(
                                "support_bias_weight", params["support_bias_weight"]
                            ),
                            "support_bias_step": adaptive_overrides.get(
                                "support_bias_step", params["support_bias_step"]
                            ),
                            "support_bias_rounds": adaptive_overrides.get(
                                "support_bias_rounds", params["support_bias_rounds"]
                            ),
                            "support_quality_weight": adaptive_overrides.get(
                                "support_quality_weight", params["support_quality_weight"]
                            ),
                            "support_quality_floor": adaptive_overrides.get(
                                "support_quality_floor", params["support_quality_floor"]
                            ),
                            "support_quality_margin_scale": adaptive_overrides.get(
                                "support_quality_margin_scale", params["support_quality_margin_scale"]
                            ),
                            "support_loo_pair_rescue_weight": adaptive_overrides.get(
                                "support_loo_pair_rescue_weight", params["support_loo_pair_rescue_weight"]
                            ),
                            "support_loo_pair_rescue_top_pairs": adaptive_overrides.get(
                                "support_loo_pair_rescue_top_pairs", params["support_loo_pair_rescue_top_pairs"]
                            ),
                            "support_loo_pair_rescue_min_errors": adaptive_overrides.get(
                                "support_loo_pair_rescue_min_errors", params["support_loo_pair_rescue_min_errors"]
                            ),
                            "support_loo_pair_rescue_alpha": adaptive_overrides.get(
                                "support_loo_pair_rescue_alpha", params["support_loo_pair_rescue_alpha"]
                            ),
                            "support_loo_pair_rescue_clip": adaptive_overrides.get(
                                "support_loo_pair_rescue_clip", params["support_loo_pair_rescue_clip"]
                            ),
                            "support_loo_pair_rescue_scope": adaptive_overrides.get(
                                "support_loo_pair_rescue_scope", params["support_loo_pair_rescue_scope"]
                            ),
                            "support_loo_pair_rescue_proto_neighbors": adaptive_overrides.get(
                                "support_loo_pair_rescue_proto_neighbors",
                                params["support_loo_pair_rescue_proto_neighbors"],
                            ),
                            "support_loo_pair_rescue_proto_min_sim": adaptive_overrides.get(
                                "support_loo_pair_rescue_proto_min_sim",
                                params["support_loo_pair_rescue_proto_min_sim"],
                            ),
                            "support_loo_pair_linear_weight": adaptive_overrides.get(
                                "support_loo_pair_linear_weight", params["support_loo_pair_linear_weight"]
                            ),
                            "support_loo_pair_linear_top_pairs": adaptive_overrides.get(
                                "support_loo_pair_linear_top_pairs", params["support_loo_pair_linear_top_pairs"]
                            ),
                            "support_loo_pair_linear_min_errors": adaptive_overrides.get(
                                "support_loo_pair_linear_min_errors", params["support_loo_pair_linear_min_errors"]
                            ),
                            "support_loo_pair_linear_alpha": adaptive_overrides.get(
                                "support_loo_pair_linear_alpha", params["support_loo_pair_linear_alpha"]
                            ),
                            "support_loo_pair_linear_clip": adaptive_overrides.get(
                                "support_loo_pair_linear_clip", params["support_loo_pair_linear_clip"]
                            ),
                            "support_loo_pair_linear_scope": adaptive_overrides.get(
                                "support_loo_pair_linear_scope", params["support_loo_pair_linear_scope"]
                            ),
                            "labelprop_weight": adaptive_overrides.get("labelprop_weight", params["labelprop_weight"]),
                            "labelprop_k": adaptive_overrides.get("labelprop_k", params["labelprop_k"]),
                            "labelprop_alpha": adaptive_overrides.get("labelprop_alpha", params["labelprop_alpha"]),
                            "labelprop_temperature": adaptive_overrides.get(
                                "labelprop_temperature", params["labelprop_temperature"]
                            ),
                            "labelprop_rounds": adaptive_overrides.get("labelprop_rounds", params["labelprop_rounds"]),
                            "labelprop_clip": adaptive_overrides.get("labelprop_clip", params["labelprop_clip"]),
                            "labelprop_scope": adaptive_overrides.get("labelprop_scope", params["labelprop_scope"]),
                            "query_graph_weight": adaptive_overrides.get(
                                "query_graph_weight", params["query_graph_weight"]
                            ),
                            "query_cluster_weight": adaptive_overrides.get(
                                "query_cluster_weight", params["query_cluster_weight"]
                            ),
                            "query_cluster_rounds": adaptive_overrides.get(
                                "query_cluster_rounds", params["query_cluster_rounds"]
                            ),
                            "query_cluster_support_weight": adaptive_overrides.get(
                                "query_cluster_support_weight", params["query_cluster_support_weight"]
                            ),
                            "query_cluster_temperature": adaptive_overrides.get(
                                "query_cluster_temperature", params["query_cluster_temperature"]
                            ),
                            "query_cluster_clip": adaptive_overrides.get(
                                "query_cluster_clip", params["query_cluster_clip"]
                            ),
                            "query_cluster_scope": adaptive_overrides.get(
                                "query_cluster_scope", params["query_cluster_scope"]
                            ),
                            "query_cluster_agreement_min": adaptive_overrides.get(
                                "query_cluster_agreement_min", params["query_cluster_agreement_min"]
                            ),
                            "query_cluster_margin_min": adaptive_overrides.get(
                                "query_cluster_margin_min", params["query_cluster_margin_min"]
                            ),
                            "query_proto_refine_weight": adaptive_overrides.get(
                                "query_proto_refine_weight", params["query_proto_refine_weight"]
                            ),
                            "query_proto_refine_topm": adaptive_overrides.get(
                                "query_proto_refine_topm", params["query_proto_refine_topm"]
                            ),
                            "query_proto_refine_clip": adaptive_overrides.get(
                                "query_proto_refine_clip", params["query_proto_refine_clip"]
                            ),
                            "transductive_proto_weight": adaptive_overrides.get(
                                "transductive_proto_weight", params["transductive_proto_weight"]
                            ),
                            "transductive_proto_rounds": adaptive_overrides.get(
                                "transductive_proto_rounds", params["transductive_proto_rounds"]
                            ),
                            "transductive_proto_query_topm": adaptive_overrides.get(
                                "transductive_proto_query_topm", params["transductive_proto_query_topm"]
                            ),
                            "transductive_proto_support_weight": adaptive_overrides.get(
                                "transductive_proto_support_weight", params["transductive_proto_support_weight"]
                            ),
                            "transductive_proto_query_weight": adaptive_overrides.get(
                                "transductive_proto_query_weight", params["transductive_proto_query_weight"]
                            ),
                            "transductive_proto_clip": adaptive_overrides.get(
                                "transductive_proto_clip", params["transductive_proto_clip"]
                            ),
                            "dense_cluster_weight": adaptive_overrides.get(
                                "dense_cluster_weight", params["dense_cluster_weight"]
                            ),
                            "dense_cluster_similarity": adaptive_overrides.get(
                                "dense_cluster_similarity", params["dense_cluster_similarity"]
                            ),
                            "dense_cluster_neighbor_k": adaptive_overrides.get(
                                "dense_cluster_neighbor_k", params["dense_cluster_neighbor_k"]
                            ),
                            "dense_cluster_rounds": adaptive_overrides.get(
                                "dense_cluster_rounds", params["dense_cluster_rounds"]
                            ),
                            "dense_cluster_candidate_topn": adaptive_overrides.get(
                                "dense_cluster_candidate_topn", params["dense_cluster_candidate_topn"]
                            ),
                            "dense_cluster_query_topm": adaptive_overrides.get(
                                "dense_cluster_query_topm", params["dense_cluster_query_topm"]
                            ),
                            "dense_cluster_clip": adaptive_overrides.get(
                                "dense_cluster_clip", params["dense_cluster_clip"]
                            ),
                            "dense_cluster_scope": adaptive_overrides.get(
                                "dense_cluster_scope", params["dense_cluster_scope"]
                            ),
                            "query_pair_cluster_top_pairs": adaptive_overrides.get(
                                "query_pair_cluster_top_pairs", params["query_pair_cluster_top_pairs"]
                            ),
                            "query_pair_cluster_similarity": adaptive_overrides.get(
                                "query_pair_cluster_similarity", params["query_pair_cluster_similarity"]
                            ),
                            "query_pair_cluster_score_weight": adaptive_overrides.get(
                                "query_pair_cluster_score_weight", params["query_pair_cluster_score_weight"]
                            ),
                            "query_pair_cluster_query_weight": adaptive_overrides.get(
                                "query_pair_cluster_query_weight", params["query_pair_cluster_query_weight"]
                            ),
                            "query_pair_cluster_clip": adaptive_overrides.get(
                                "query_pair_cluster_clip", params["query_pair_cluster_clip"]
                            ),
                            "query_pair_cluster_scope": adaptive_overrides.get(
                                "query_pair_cluster_scope", params["query_pair_cluster_scope"]
                            ),
                        }
                    )
                    row = _evaluate_metric_qknn(
                        features=features,
                        aux_features=aux_features,
                        logits=logits,
                        tx_ids=tx_ids,
                        roles=roles,
                        scenarios=scenarios,
                        domain_values=_metadata_domain_values(
                            key=str(domain_refine_key),
                            scenarios=scenarios,
                            rx_ids=rx_ids,
                            channel_views=channel_views,
                            day_ids=day_ids,
                        ),
                        old_splits=old_splits,
                        new_splits=new_splits,
                        old_labels=old_labels,
                        new_labels=new_labels,
                        transform_mode=params["mode"],
                        transform_strength=float(params["strength"]),
                        topm=int(topm),
                        proto_mix=float(params["proto_mix"]),
                        aux_score_weight=float(params["aux_score_weight"]),
                        adaptive_qknn_policy=str(
                            params.get("adaptive_qknn_policy", adaptive_overrides.get("adaptive_qknn_policy", ""))
                        ),
                        radius_norm=float(radius_norm),
                        old_bias=float(old_bias),
                        neg_lambda=float(neg_lambda),
                        neg_threshold=float(neg_threshold),
                        neg_margin=float(neg_margin),
                        mutual_only=str(mutual_raw).lower() == "true",
                        scenario_aware=bool(args.scenario_aware),
                        balanced_assignment=bool(args.balanced_assignment),
                        role_balanced_assignment=bool(params["role_balanced_assignment"]),
                        fast_role_balanced_assignment=bool(args.fast_role_balanced_assignment),
                        scenario_balanced_assignment=bool(args.scenario_balanced_assignment),
                        proto_repel_lambda=float(proto_repel_lambda),
                        proto_repel_margin=float(proto_repel_margin),
                        proto_repel_steps=int(proto_repel_steps),
                        proto_repel_anchor=float(proto_repel_anchor),
                        pair_refine_similarity=float(pair_refine_similarity),
                        pair_axis_similarity=float(pair_axis_similarity),
                        pair_axis_weight=float(pair_axis_weight),
                        pair_axis_clip=float(pair_axis_clip),
                        pair_gaussian_similarity=float(params["pair_gaussian_similarity"]),
                        pair_gaussian_weight=float(params["pair_gaussian_weight"]),
                        pair_gaussian_clip=float(params["pair_gaussian_clip"]),
                        pair_fisher_similarity=float(params["pair_fisher_similarity"]),
                        pair_fisher_weight=float(params["pair_fisher_weight"]),
                        pair_fisher_alpha=float(params["pair_fisher_alpha"]),
                        pair_fisher_clip=float(params["pair_fisher_clip"]),
                        support_guided_proxy_rows=support_guided_proxy_rows,
                        support_guided_proxy_weight=float(params["support_guided_proxy_weight"]),
                        support_guided_proxy_top_pairs=int(params["support_guided_proxy_top_pairs"]),
                        support_guided_proxy_clip=float(params["support_guided_proxy_clip"]),
                        support_guided_proxy_min_errors=int(params["support_guided_proxy_min_errors"]),
                        support_guided_proxy_scope=str(params["support_guided_proxy_scope"]),
                        support_guided_proxy_balance=bool(params["support_guided_proxy_balance"]),
                        support_guided_proxy_bundle_rows=int(params["support_guided_proxy_bundle_rows"]),
                        support_guided_proxy_analogy=bool(params["support_guided_proxy_analogy"]),
                        support_guided_proxy_gate=bool(params["support_guided_proxy_gate"]),
                        support_guided_proxy_gate_floor_tol=float(params["support_guided_proxy_gate_floor_tol"]),
                        support_guided_proxy_gate_mean_tol=float(params["support_guided_proxy_gate_mean_tol"]),
                        pair_logreg_similarity=float(params["pair_logreg_similarity"]),
                        pair_logreg_weight=float(params["pair_logreg_weight"]),
                        pair_logreg_alpha=float(params["pair_logreg_alpha"]),
                        pair_logreg_clip=float(params["pair_logreg_clip"]),
                        pair_logreg_scope=str(params["pair_logreg_scope"]),
                        new_old_conflict_bias_threshold=float(new_old_conflict_bias_threshold),
                        new_old_conflict_bias_weight=float(new_old_conflict_bias_weight),
                        old_new_runnerup_rescue_similarity=float(old_new_runnerup_rescue_similarity),
                        old_new_runnerup_rescue_margin=float(old_new_runnerup_rescue_margin),
                        old_new_runnerup_rescue_weight=float(old_new_runnerup_rescue_weight),
                        bootstrap_proto_mix=float(bootstrap_proto_mix),
                        bootstrap_proto_drop=int(bootstrap_proto_drop),
                        bootstrap_proto_topm=int(bootstrap_proto_topm),
                        core_proto_weight=float(params["core_proto_weight"]),
                        core_proto_count=int(params["core_proto_count"]),
                        core_proto_topm=int(params["core_proto_topm"]),
                        core_proto_mode=str(params["core_proto_mode"]),
                        ridge_head_weight=float(params["ridge_head_weight"]),
                        ridge_head_alpha=float(params["ridge_head_alpha"]),
                        ridge_head_clip=float(params["ridge_head_clip"]),
                        subspace_proto_weight=float(subspace_proto_weight),
                        subspace_proto_rank=int(subspace_proto_rank),
                        subspace_proto_power=float(subspace_proto_power),
                        subspace_proto_clip=float(subspace_proto_clip),
                        old_residual_new_weight=float(params["old_residual_new_weight"]),
                        old_residual_new_rank=int(params["old_residual_new_rank"]),
                        old_residual_new_proto_mix=float(params["old_residual_new_proto_mix"]),
                        old_residual_new_clip=float(params["old_residual_new_clip"]),
                        domain_refine_key=str(domain_refine_key),
                        domain_refine_weight=float(domain_refine_weight),
                        domain_refine_scope=str(domain_refine_scope),
                        class_diag_metric_weight=float(class_diag_metric_weight),
                        class_diag_metric_similarity=float(class_diag_metric_similarity),
                        class_diag_metric_alpha=float(class_diag_metric_alpha),
                        class_diag_metric_power=float(class_diag_metric_power),
                        class_diag_metric_clip=float(class_diag_metric_clip),
                        support_bias_weight=float(params["support_bias_weight"]),
                        support_bias_step=float(params["support_bias_step"]),
                        support_bias_rounds=int(params["support_bias_rounds"]),
                        support_quality_weight=float(params["support_quality_weight"]),
                        support_quality_floor=float(params["support_quality_floor"]),
                        support_quality_margin_scale=float(params["support_quality_margin_scale"]),
                        support_loo_pair_rescue_weight=float(params["support_loo_pair_rescue_weight"]),
                        support_loo_pair_rescue_top_pairs=int(params["support_loo_pair_rescue_top_pairs"]),
                        support_loo_pair_rescue_min_errors=int(params["support_loo_pair_rescue_min_errors"]),
                        support_loo_pair_rescue_alpha=float(params["support_loo_pair_rescue_alpha"]),
                        support_loo_pair_rescue_clip=float(params["support_loo_pair_rescue_clip"]),
                        support_loo_pair_rescue_scope=str(params["support_loo_pair_rescue_scope"]),
                        support_loo_pair_rescue_proto_neighbors=int(params["support_loo_pair_rescue_proto_neighbors"]),
                        support_loo_pair_rescue_proto_min_sim=float(params["support_loo_pair_rescue_proto_min_sim"]),
                        support_loo_pair_linear_weight=float(params["support_loo_pair_linear_weight"]),
                        support_loo_pair_linear_top_pairs=int(params["support_loo_pair_linear_top_pairs"]),
                        support_loo_pair_linear_min_errors=int(params["support_loo_pair_linear_min_errors"]),
                        support_loo_pair_linear_alpha=float(params["support_loo_pair_linear_alpha"]),
                        support_loo_pair_linear_clip=float(params["support_loo_pair_linear_clip"]),
                        support_loo_pair_linear_scope=str(params["support_loo_pair_linear_scope"]),
                        mahal_proto_weight=float(mahal_proto_weight),
                        mahal_proto_alpha=float(mahal_proto_alpha),
                        mahal_proto_diag_mix=float(mahal_proto_diag_mix),
                        mahal_proto_clip=float(mahal_proto_clip),
                        score_calibration=str(score_calibration),
                        assignment_margin_weight=float(assignment_margin_weight),
                        assignment_margin_clip=float(assignment_margin_clip),
                        labelprop_weight=float(params["labelprop_weight"]),
                        labelprop_k=int(params["labelprop_k"]),
                        labelprop_alpha=float(params["labelprop_alpha"]),
                        labelprop_temperature=float(params["labelprop_temperature"]),
                        labelprop_rounds=int(params["labelprop_rounds"]),
                        labelprop_clip=float(params["labelprop_clip"]),
                        labelprop_scope=str(params["labelprop_scope"]),
                        query_graph_weight=float(params["query_graph_weight"]),
                        query_graph_k=int(query_graph_k),
                        query_graph_temperature=float(query_graph_temperature),
                        query_graph_rounds=int(query_graph_rounds),
                        query_graph_scope=str(query_graph_scope),
                        query_cluster_weight=float(params["query_cluster_weight"]),
                        query_cluster_rounds=int(params["query_cluster_rounds"]),
                        query_cluster_support_weight=float(params["query_cluster_support_weight"]),
                        query_cluster_temperature=float(params["query_cluster_temperature"]),
                        query_cluster_clip=float(params["query_cluster_clip"]),
                        query_cluster_scope=str(params["query_cluster_scope"]),
                        query_cluster_agreement_min=float(params["query_cluster_agreement_min"]),
                        query_cluster_margin_min=float(params["query_cluster_margin_min"]),
                        local_competition_weight=float(params["local_competition_weight"]),
                        local_competition_k=int(params["local_competition_k"]),
                        local_competition_clip=float(params["local_competition_clip"]),
                        local_competition_scope=str(params["local_competition_scope"]),
                        query_proto_refine_weight=float(params["query_proto_refine_weight"]),
                        query_proto_refine_topm=int(params["query_proto_refine_topm"]),
                        query_proto_refine_clip=float(params["query_proto_refine_clip"]),
                        transductive_proto_weight=float(params["transductive_proto_weight"]),
                        transductive_proto_rounds=int(params["transductive_proto_rounds"]),
                        transductive_proto_query_topm=int(params["transductive_proto_query_topm"]),
                        transductive_proto_support_weight=float(params["transductive_proto_support_weight"]),
                        transductive_proto_query_weight=float(params["transductive_proto_query_weight"]),
                        transductive_proto_clip=float(params["transductive_proto_clip"]),
                        dense_cluster_weight=float(params["dense_cluster_weight"]),
                        dense_cluster_similarity=float(params["dense_cluster_similarity"]),
                        dense_cluster_neighbor_k=int(params["dense_cluster_neighbor_k"]),
                        dense_cluster_rounds=int(params["dense_cluster_rounds"]),
                        dense_cluster_candidate_topn=int(params["dense_cluster_candidate_topn"]),
                        dense_cluster_query_topm=int(params["dense_cluster_query_topm"]),
                        dense_cluster_clip=float(params["dense_cluster_clip"]),
                        dense_cluster_scope=str(params["dense_cluster_scope"]),
                        query_pair_cluster_top_pairs=int(params["query_pair_cluster_top_pairs"]),
                        query_pair_cluster_similarity=float(params["query_pair_cluster_similarity"]),
                        query_pair_cluster_score_weight=float(params["query_pair_cluster_score_weight"]),
                        query_pair_cluster_query_weight=float(params["query_pair_cluster_query_weight"]),
                        query_pair_cluster_clip=float(params["query_pair_cluster_clip"]),
                        query_pair_cluster_scope=str(params["query_pair_cluster_scope"]),
                        source_guard_mode=str(params["source_guard_mode"]),
                        source_guard_weight=float(params["source_guard_weight"]),
                        source_guard_conf_min=float(params["source_guard_conf_min"]),
                        source_guard_margin_min=float(params["source_guard_margin_min"]),
                        source_proto_anchor_mode=str(source_proto_anchor_mode),
                        source_proto_anchor_weight=float(source_proto_anchor_weight),
                        source_proto_anchor_center=float(source_proto_anchor_center),
                        old_target=float(args.old_target),
                        old_floor=float(args.old_floor),
                        new_target=float(args.seen_new_target),
                        new_floor=float(args.seen_new_floor),
                        collect_predictions=bool(str(args.output_predictions_csv).strip()),
                    )
                    row.update(support_geometry)
                    for key, value in adaptive_overrides.items():
                        if key not in {"transform_mode", "transform_strength"}:
                            row[key] = value
                    row["seed"] = int(seed)
                    row["support_selection_policy"] = policy
                    row["k_old"] = int(args.k_old)
                    row["k_new"] = int(args.k_new)
                    row["old_role"] = str(args.old_role)
                    row["new_role"] = str(args.new_role)
                    row["query_per_old"] = int(args.query_per_old)
                    row["query_per_new"] = int(args.query_per_new)
                    row["pool_per_old"] = int(args.pool_per_old)
                    row["pool_per_new"] = int(args.pool_per_new)
                    row["exclude_pool_from_query"] = bool(args.exclude_pool_from_query)
                    row.update(split_fingerprint)
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
        "feature_npz_sha256": _sha256_file(feature_path),
        "aux_feature_npz_sha256": _sha256_file(aux_feature_path) if aux_feature_path is not None else "",
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
        "old_role",
        "new_role",
        "transform_mode",
        "transform_strength",
        "topm",
        "proto_mix",
        "aux_score_weight",
        "effective_aux_score_weight",
        "aux_support_gate_factor",
        "aux_support_primary_loo_acc",
        "aux_support_primary_loo_min_acc",
        "aux_support_aux_loo_acc",
        "aux_support_aux_loo_min_acc",
        "aux_support_loo_delta",
        "aux_support_min_delta",
        "aux_support_absolute_floor_gate",
        "radius_norm",
        "old_bias",
        "neg_lambda",
        "neg_threshold",
        "neg_margin",
        "mutual_only",
        "scenario_aware",
        "balanced_assignment",
        "role_balanced_assignment",
        "fast_role_balanced_assignment",
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
        "support_guided_proxy_weight",
        "support_guided_proxy_top_pairs",
        "support_guided_proxy_clip",
        "support_guided_proxy_min_errors",
        "support_guided_proxy_scope",
        "support_guided_proxy_balance",
        "support_guided_proxy_bundle_rows",
        "support_guided_proxy_analogy",
        "support_guided_proxy_gate",
        "support_guided_proxy_gate_floor_tol",
        "support_guided_proxy_gate_mean_tol",
        "support_guided_proxy_gate_before_min_acc",
        "support_guided_proxy_gate_before_mean_acc",
        "support_guided_proxy_gate_after_min_acc",
        "support_guided_proxy_gate_after_mean_acc",
        "support_guided_proxy_auto_rows",
        "support_guided_proxy_auto_pairs",
        "support_guided_proxy_count",
        "stored_support_guided_proxy_scalars",
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
        "core_proto_weight",
        "core_proto_count",
        "core_proto_topm",
        "core_proto_mode",
        "stored_core_prototype_count",
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
        "old_residual_new_weight",
        "old_residual_new_rank",
        "old_residual_new_rank_used",
        "old_residual_new_proto_mix",
        "old_residual_new_clip",
        "stored_old_residual_new_scalars",
        "domain_refine_key",
        "domain_refine_weight",
        "domain_refine_scope",
        "domain_refine_domain_count",
        "stored_domain_refine_prototype_count",
        "class_diag_metric_weight",
        "class_diag_metric_similarity",
        "class_diag_metric_alpha",
        "class_diag_metric_power",
        "class_diag_metric_clip",
        "class_diag_metric_count",
        "stored_class_diag_metric_scalars",
        "support_bias_weight",
        "support_bias_step",
        "support_bias_rounds",
        "support_bias_loo_min_acc",
        "support_bias_loo_mean_acc",
        "stored_support_bias_scalars",
        "support_quality_weight",
        "support_quality_floor",
        "support_quality_margin_scale",
        "support_quality_loo_min_acc",
        "support_quality_loo_mean_acc",
        "stored_support_quality_scalars",
        "support_loo_pair_rescue_weight",
        "support_loo_pair_rescue_top_pairs",
        "support_loo_pair_rescue_min_errors",
        "support_loo_pair_rescue_alpha",
        "support_loo_pair_rescue_clip",
        "support_loo_pair_rescue_scope",
        "support_loo_pair_rescue_proto_neighbors",
        "support_loo_pair_rescue_proto_min_sim",
        "support_loo_pair_rescue_count",
        "support_loo_pair_rescue_pairs",
        "support_loo_pair_rescue_loo_min_acc",
        "support_loo_pair_rescue_loo_mean_acc",
        "stored_support_loo_pair_rescue_scalars",
        "support_loo_pair_linear_weight",
        "support_loo_pair_linear_top_pairs",
        "support_loo_pair_linear_min_errors",
        "support_loo_pair_linear_alpha",
        "support_loo_pair_linear_clip",
        "support_loo_pair_linear_scope",
        "support_loo_pair_linear_count",
        "support_loo_pair_linear_loo_min_acc",
        "support_loo_pair_linear_loo_mean_acc",
        "stored_support_loo_pair_linear_scalars",
        "top2_pair_gate_weight",
        "top2_pair_gate_top_pairs",
        "top2_pair_gate_margin",
        "top2_pair_gate_query_weight",
        "top2_pair_gate_count",
        "top2_pair_gate_pairs",
        "top2_pair_gate_loo_min_acc",
        "top2_pair_gate_loo_mean_acc",
        "stored_top2_pair_gate_scalars",
        "neighborhood_gate_weight",
        "neighborhood_gate_top_classes",
        "neighborhood_gate_neighbor_count",
        "neighborhood_gate_margin",
        "neighborhood_gate_query_weight",
        "neighborhood_gate_count",
        "neighborhood_gate_neighborhoods",
        "neighborhood_gate_loo_min_acc",
        "neighborhood_gate_loo_mean_acc",
        "stored_neighborhood_gate_scalars",
        "mahal_proto_weight",
        "mahal_proto_alpha",
        "mahal_proto_diag_mix",
        "mahal_proto_clip",
        "score_calibration",
        "assignment_margin_weight",
        "assignment_margin_clip",
        "labelprop_weight",
        "labelprop_k",
        "labelprop_alpha",
        "labelprop_temperature",
        "labelprop_rounds",
        "labelprop_clip",
        "labelprop_scope",
        "labelprop_edges",
        "query_graph_weight",
        "query_graph_k",
        "query_graph_temperature",
        "query_graph_rounds",
        "query_graph_scope",
        "query_graph_edges",
        "query_cluster_weight",
        "query_cluster_rounds",
        "query_cluster_support_weight",
        "query_cluster_temperature",
        "query_cluster_clip",
        "query_cluster_scope",
        "query_cluster_agreement_min",
        "query_cluster_margin_min",
        "query_cluster_temp_proto_count",
        "query_cluster_assigned_rows",
        "local_competition_weight",
        "local_competition_k",
        "local_competition_clip",
        "local_competition_scope",
        "local_competition_edges",
        "query_proto_refine_weight",
        "query_proto_refine_topm",
        "query_proto_refine_clip",
        "query_proto_refine_count",
        "transductive_proto_weight",
        "transductive_proto_rounds",
        "transductive_proto_query_topm",
        "transductive_proto_support_weight",
        "transductive_proto_query_weight",
        "transductive_proto_clip",
        "transductive_proto_count",
        "dense_cluster_weight",
        "dense_cluster_similarity",
        "dense_cluster_neighbor_k",
        "dense_cluster_rounds",
        "dense_cluster_candidate_topn",
        "dense_cluster_query_topm",
        "dense_cluster_clip",
        "dense_cluster_scope",
        "dense_cluster_count",
        "dense_cluster_temp_proto_count",
        "dense_cluster_adjusted_rows",
        "query_pair_cluster_top_pairs",
        "query_pair_cluster_similarity",
        "query_pair_cluster_score_weight",
        "query_pair_cluster_query_weight",
        "query_pair_cluster_clip",
        "query_pair_cluster_scope",
        "query_pair_cluster_count",
        "query_pair_cluster_changed",
        "query_pair_cluster_pairs",
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
        "k_old",
        "k_new",
        "query_per_old",
        "query_per_new",
        "pool_per_old",
        "pool_per_new",
        "exclude_pool_from_query",
        "support_index_sha16",
        "old_query_index_sha16",
        "new_query_index_sha16",
        "query_index_sha16",
        "split_fingerprint",
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
            out["split_fingerprint"] = json.dumps(row.get("split_fingerprint", {}), ensure_ascii=False, sort_keys=True)
            writer.writerow(out)
    print(json.dumps({"best": rows[:5], "output_json": str(output_json)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
