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
) -> tuple[np.ndarray, int, int, float, float]:
    if float(weight) == 0.0 or int(top_pairs) <= 0:
        return scores, 0, 0, 0.0, 0.0
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
        pair_counts[(truth, pred)] = pair_counts.get((truth, pred), 0) + 1
    candidates = [
        (truth, pred, count)
        for (truth, pred), count in pair_counts.items()
        if int(count) >= max(1, int(min_errors))
    ]
    candidates.sort(key=lambda item: (-int(item[2]), item[0], item[1]))
    candidates = candidates[: max(0, int(top_pairs))]
    if not candidates:
        return scores, 0, 0, float(min(before_per_class, default=0.0)), float(np.mean(before_per_class) if before_per_class else 0.0)

    support = qknn._normalize_rows(features[support_indices])
    query = qknn._normalize_rows(features[query_indices])
    prototypes: list[np.ndarray] = []
    for label in class_labels:
        cls = support[labels == label]
        if cls.size == 0:
            prototypes.append(np.zeros(support.shape[1], dtype=np.float64))
        else:
            prototypes.append(qknn._normalize_rows(cls.mean(axis=0, keepdims=True))[0])
    proto = qknn._normalize_rows(np.stack(prototypes, axis=0))
    adjusted = np.asarray(scores, dtype=np.float64).copy()
    rescue_scores = np.asarray(support_scores, dtype=np.float64).copy()
    used = 0
    stored_scalars = 0
    for truth, pred, _count in candidates:
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
) -> np.ndarray:
    mode = str(key).strip().lower()
    if mode in {"", "none"}:
        return np.asarray([""] * int(scenarios.size), dtype=object)
    if mode == "scenario":
        return np.asarray(scenarios, dtype=object).astype(str)
    if mode == "rx":
        return np.asarray(rx_ids, dtype=object).astype(str)
    if mode == "channel":
        return np.asarray(channel_views, dtype=object).astype(str)
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
    support_loo_pair_rescue_weight: float,
    support_loo_pair_rescue_top_pairs: int,
    support_loo_pair_rescue_min_errors: int,
    support_loo_pair_rescue_alpha: float,
    support_loo_pair_rescue_clip: float,
    support_loo_pair_rescue_scope: str,
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
    scores, support_guided_proxy_count, stored_support_guided_proxy_scalars = _support_guided_proxy_adjust_scores(
        scores,
        features=adapted,
        tx_ids=tx_ids,
        roles=roles,
        support_indices=support_indices,
        support_labels=support_labels,
        query_indices=query_indices,
        class_labels=old_labels + new_labels,
        candidate_rows=support_guided_proxy_rows,
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
    if float(support_bias_weight) > 0.0 and float(support_bias_step) > 0.0 and int(support_bias_rounds) > 0:
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
        "support_loo_pair_rescue_weight": float(support_loo_pair_rescue_weight),
        "support_loo_pair_rescue_top_pairs": int(support_loo_pair_rescue_top_pairs),
        "support_loo_pair_rescue_min_errors": int(support_loo_pair_rescue_min_errors),
        "support_loo_pair_rescue_alpha": float(support_loo_pair_rescue_alpha),
        "support_loo_pair_rescue_clip": float(support_loo_pair_rescue_clip),
        "support_loo_pair_rescue_scope": str(support_loo_pair_rescue_scope),
        "support_loo_pair_rescue_count": int(support_loo_pair_rescue_count),
        "support_loo_pair_rescue_loo_min_acc": float(support_loo_pair_rescue_min_acc),
        "support_loo_pair_rescue_loo_mean_acc": float(support_loo_pair_rescue_mean_acc),
        "stored_support_loo_pair_rescue_scalars": int(stored_support_loo_pair_rescue_scalars),
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
    }:
        raise ValueError(f"unsupported adaptive_qknn_policy: {policy}")
    use_v2 = name in {"dualview_support_v2", "stable_dualview_v2"}
    use_v3 = name in {"dualview_support_v3", "stable_dualview_v3"}
    use_v4 = name in {"dualview_support_v4", "stable_dualview_v4"}
    use_v5 = name in {"dualview_support_v5", "stable_dualview_v5"}
    use_v6 = name in {"dualview_support_v6", "stable_dualview_v6"}
    use_v7 = name in {"dualview_support_v7", "stable_dualview_v7"}

    min_k = float(geometry["adaptive_support_min_k"])
    new_count = float(geometry["adaptive_new_class_count"])
    max_sim = float(geometry["adaptive_support_max_offdiag_proto_sim"])
    p90_sim = float(geometry["adaptive_support_p90_offdiag_proto_sim"])
    radius = float(geometry["adaptive_support_mean_radius"])
    hardness = _clip01(max((max_sim - 0.82) / 0.16, (p90_sim - 0.68) / 0.22, (radius - 0.08) / 0.20))
    if use_v3 or use_v4 or use_v5 or use_v7:
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
    if use_v2 or use_v3 or use_v4 or use_v5 or use_v6 or use_v7:
        competition_load = class_load
        if (use_v3 or use_v4 or use_v5 or use_v6 or use_v7) and new_count >= 2.0:
            competition_load = max(competition_load, 0.25)
        overrides.update(
            {
                "role_balanced_assignment": bool(
                    class_load > 0.0 or ((use_v3 or use_v4 or use_v5 or use_v6 or use_v7) and stable_gate >= 0.50)
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
    if use_v7:
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
    parser.add_argument("--support_loo_pair_rescue_weight_grid", default="0")
    parser.add_argument("--support_loo_pair_rescue_top_pairs_grid", default="0")
    parser.add_argument("--support_loo_pair_rescue_min_errors_grid", default="1")
    parser.add_argument("--support_loo_pair_rescue_alpha_grid", default="0.1")
    parser.add_argument("--support_loo_pair_rescue_clip_grid", default="2.0")
    parser.add_argument("--support_loo_pair_rescue_scope_grid", default="new")
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
    channel_views = (
        np.asarray(data["channel_views"], dtype=object).astype(str)
        if "channel_views" in data.files
        else np.asarray([""] * int(tx_ids.size), dtype=object)
    )
    aux_features = None
    if aux_feature_path is not None:
        aux_data = np.load(aux_feature_path, allow_pickle=True)
        aux_features = qknn._normalize_rows(aux_data["features"])
        for key, primary in (("tx_ids", tx_ids), ("dataset_role", roles), ("sat_scenarios", scenarios)):
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
            qknn._parse_float_csv(args.support_loo_pair_rescue_weight_grid),
            qknn._parse_int_csv(args.support_loo_pair_rescue_top_pairs_grid),
            qknn._parse_int_csv(args.support_loo_pair_rescue_min_errors_grid),
            qknn._parse_float_csv(args.support_loo_pair_rescue_alpha_grid),
            qknn._parse_float_csv(args.support_loo_pair_rescue_clip_grid),
            qknn._parse_csv(args.support_loo_pair_rescue_scope_grid),
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
                    support_loo_pair_rescue_weight,
                    support_loo_pair_rescue_top_pairs,
                    support_loo_pair_rescue_min_errors,
                    support_loo_pair_rescue_alpha,
                    support_loo_pair_rescue_clip,
                    support_loo_pair_rescue_scope,
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
                        "source_guard_mode": str(source_guard_mode),
                        "source_guard_weight": float(source_guard_weight),
                        "source_guard_conf_min": float(source_guard_conf_min),
                        "source_guard_margin_min": float(source_guard_margin_min),
                        "role_balanced_assignment": bool(args.role_balanced_assignment),
                        "local_competition_weight": float(local_competition_weight),
                        "local_competition_k": int(local_competition_k),
                        "local_competition_clip": float(local_competition_clip),
                        "local_competition_scope": str(local_competition_scope),
                        "labelprop_weight": float(labelprop_weight),
                        "labelprop_k": int(labelprop_k),
                        "labelprop_alpha": float(labelprop_alpha),
                        "labelprop_temperature": float(labelprop_temperature),
                        "labelprop_rounds": int(labelprop_rounds),
                        "labelprop_clip": float(labelprop_clip),
                        "labelprop_scope": str(labelprop_scope),
                        "query_graph_weight": float(query_graph_weight),
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
                            "source_guard_mode": adaptive_overrides.get("source_guard_mode", params["source_guard_mode"]),
                            "source_guard_weight": adaptive_overrides.get("source_guard_weight", params["source_guard_weight"]),
                            "source_guard_conf_min": adaptive_overrides.get("source_guard_conf_min", params["source_guard_conf_min"]),
                            "source_guard_margin_min": adaptive_overrides.get("source_guard_margin_min", params["source_guard_margin_min"]),
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
                        support_guided_proxy_weight=float(support_guided_proxy_weight),
                        support_guided_proxy_top_pairs=int(support_guided_proxy_top_pairs),
                        support_guided_proxy_clip=float(support_guided_proxy_clip),
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
                        old_residual_new_weight=float(old_residual_new_weight),
                        old_residual_new_rank=int(old_residual_new_rank),
                        old_residual_new_proto_mix=float(old_residual_new_proto_mix),
                        old_residual_new_clip=float(old_residual_new_clip),
                        domain_refine_key=str(domain_refine_key),
                        domain_refine_weight=float(domain_refine_weight),
                        domain_refine_scope=str(domain_refine_scope),
                        class_diag_metric_weight=float(class_diag_metric_weight),
                        class_diag_metric_similarity=float(class_diag_metric_similarity),
                        class_diag_metric_alpha=float(class_diag_metric_alpha),
                        class_diag_metric_power=float(class_diag_metric_power),
                        class_diag_metric_clip=float(class_diag_metric_clip),
                        support_bias_weight=float(support_bias_weight),
                        support_bias_step=float(support_bias_step),
                        support_bias_rounds=int(support_bias_rounds),
                        support_loo_pair_rescue_weight=float(support_loo_pair_rescue_weight),
                        support_loo_pair_rescue_top_pairs=int(support_loo_pair_rescue_top_pairs),
                        support_loo_pair_rescue_min_errors=int(support_loo_pair_rescue_min_errors),
                        support_loo_pair_rescue_alpha=float(support_loo_pair_rescue_alpha),
                        support_loo_pair_rescue_clip=float(support_loo_pair_rescue_clip),
                        support_loo_pair_rescue_scope=str(support_loo_pair_rescue_scope),
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
        "support_loo_pair_rescue_weight",
        "support_loo_pair_rescue_top_pairs",
        "support_loo_pair_rescue_min_errors",
        "support_loo_pair_rescue_alpha",
        "support_loo_pair_rescue_clip",
        "support_loo_pair_rescue_scope",
        "support_loo_pair_rescue_count",
        "support_loo_pair_rescue_loo_min_acc",
        "support_loo_pair_rescue_loo_mean_acc",
        "stored_support_loo_pair_rescue_scalars",
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
