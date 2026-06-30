#!/usr/bin/env python
"""Evaluate Stage2 OPGAC-Net on exported z_id feature NPZ files."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = ROOT / "code"
for _path in (str(ROOT), str(CODE_ROOT)):
    while _path in sys.path:
        sys.path.remove(_path)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(CODE_ROOT))

from cvsrffi.opgac_net import (  # noqa: E402
    DECISION_AMBIGUOUS,
    DECISION_NEW,
    DECISION_OLD,
    DECISION_UNKNOWN,
    GaussianClassState,
    OPGACConfig,
    OPGACMemory,
    OPGACNet,
    OPGACPrediction,
)
from cvsrffi.wisig_fewshot_payload import (  # noqa: E402
    UNKNOWN_LABEL,
    build_sfe_payload_from_feature_arrays,
    parse_tx_id_list,
)


EPS = 1.0e-8


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (Path, os.PathLike)):
        return str(value)
    return value


def _load_manifest(data: np.lib.npyio.NpzFile) -> dict[str, Any]:
    if "manifest_json" not in data:
        return {}
    raw = data["manifest_json"]
    try:
        item = raw.item() if getattr(raw, "shape", ()) == () else raw
        if isinstance(item, bytes):
            item = item.decode("utf-8")
        if isinstance(item, str):
            return json.loads(item)
        if isinstance(item, dict):
            return dict(item)
    except Exception:
        return {}
    return {}


def _norm_np(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), EPS)


def _quantile(values: np.ndarray | torch.Tensor, q: float, default: float) -> float:
    if isinstance(values, torch.Tensor):
        flat_values = values.detach().cpu().float().reshape(-1).tolist()
    else:
        flat_values = np.asarray(values, dtype=np.float64).reshape(-1).tolist()
    arr = sorted(float(v) for v in flat_values if math.isfinite(float(v)))
    if not arr:
        return float(default)
    qq = min(max(float(q), 0.0), 1.0)
    pos = qq * float(len(arr) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return arr[lo]
    weight = pos - float(lo)
    return float((1.0 - weight) * arr[lo] + weight * arr[hi])


def _iqr(values: torch.Tensor, eps: float) -> float:
    values = values.detach().float().reshape(-1)
    values = values[torch.isfinite(values)]
    if values.numel() == 0:
        return float(eps)
    if values.numel() < 4:
        return float(values.std(unbiased=False).clamp_min(eps).item())
    q75 = _quantile(values, 0.75, default=0.0)
    q25 = _quantile(values, 0.25, default=0.0)
    return float(max(float(q75) - float(q25), float(eps)))


def _component_nll(features: torch.Tensor, state: GaussianClassState, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.as_tensor(features, dtype=torch.float32, device=state.means.device)
    means = state.means.to(x.device, x.dtype)
    diag = state.diag_vars.to(x.device, x.dtype).clamp_min(eps)
    weights = state.weights.to(x.device, x.dtype).clamp_min(eps)
    weights = weights / weights.sum().clamp_min(eps)
    diff = x[:, None, :] - means[None, :, :]
    d2 = (diff.pow(2) / diag[None, :, :]).sum(dim=-1)
    logdet = torch.log(diag).sum(dim=-1)
    log_scores = torch.log(weights).view(1, -1) - 0.5 * (d2 + logdet.view(1, -1))
    return -torch.logsumexp(log_scores, dim=1), d2


def _component_groups(
    labels: np.ndarray,
    component_ids: np.ndarray | None,
    label: int,
    *,
    min_count: int,
    max_components: int,
) -> list[np.ndarray]:
    class_idx = np.flatnonzero(labels == int(label)).astype(np.int64)
    if component_ids is None:
        return [class_idx]
    values = component_ids[class_idx].astype(str)
    groups = []
    for value in sorted(set(values.tolist())):
        idx = class_idx[values == value]
        if idx.size >= int(min_count):
            groups.append(idx.astype(np.int64))
    if not groups:
        return [class_idx]
    if int(max_components) > 0 and len(groups) > int(max_components):
        groups = sorted(groups, key=lambda item: (-int(item.size), str(component_ids[item[0]])))
        keep = groups[: max(1, int(max_components) - 1)]
        rest = groups[max(1, int(max_components) - 1) :]
        if rest:
            keep.append(np.concatenate(rest).astype(np.int64))
        groups = keep
    return groups


def _make_state(
    *,
    class_id: int,
    group: str,
    class_features: torch.Tensor,
    component_members: list[torch.Tensor],
    global_var: torch.Tensor,
    config: OPGACConfig,
    radius_quantile: float,
    radius_slack: float,
    energy_quantile: float,
    energy_slack: float,
    var_shrinkage: float,
    metadata: Mapping[str, Any],
) -> GaussianClassState:
    means = []
    diag_vars = []
    thresholds = []
    weights = []
    for members in component_members:
        mean = members.mean(dim=0, keepdim=True)
        if config.normalize_features:
            mean = mean / mean.norm(dim=1, keepdim=True).clamp_min(float(config.eps))
        if members.size(0) > 1:
            local_var = (members - mean).pow(2).mean(dim=0)
        else:
            local_var = global_var
        diag_var = ((1.0 - float(var_shrinkage)) * local_var + float(var_shrinkage) * global_var).clamp_min(
            float(config.min_variance)
        )
        d2 = ((members - mean).pow(2) / diag_var.view(1, -1)).sum(dim=1)
        threshold = _quantile(d2, radius_quantile, default=float(d2.max().item() if d2.numel() else 1.0))
        means.append(mean.squeeze(0))
        diag_vars.append(diag_var)
        thresholds.append(max(float(config.min_threshold), float(threshold) + float(radius_slack)))
        weights.append(float(members.size(0)))
    state = GaussianClassState(
        class_id=int(class_id),
        group=str(group),
        means=torch.stack(means, dim=0).float(),
        diag_vars=torch.stack(diag_vars, dim=0).float(),
        weights=torch.as_tensor(weights, dtype=torch.float32),
        component_thresholds=torch.as_tensor(thresholds, dtype=torch.float32),
        class_threshold=float(config.default_class_threshold),
        energy_median=0.0,
        energy_iqr=1.0,
        support_count=int(class_features.size(0)),
        metadata=dict(metadata),
    )
    _calibrate_state(
        state,
        class_features,
        config=config,
        radius_quantile=radius_quantile,
        radius_slack=radius_slack,
        energy_quantile=energy_quantile,
        energy_slack=energy_slack,
    )
    state.validate(config.feature_dim)
    return state


def _calibrate_state(
    state: GaussianClassState,
    features: torch.Tensor,
    *,
    config: OPGACConfig,
    radius_quantile: float,
    radius_slack: float,
    energy_quantile: float,
    energy_slack: float,
) -> None:
    if features.numel() == 0:
        return
    x = torch.as_tensor(features, dtype=torch.float32, device=state.means.device)
    if config.normalize_features:
        x = x / x.norm(dim=1, keepdim=True).clamp_min(float(config.eps))
    nll, d2 = _component_nll(x, state, float(config.eps))
    nearest = torch.argmin(d2, dim=1)
    for comp_idx in range(state.means.size(0)):
        values = d2[nearest == int(comp_idx), comp_idx]
        if values.numel() == 0:
            continue
        tau = _quantile(values, radius_quantile, default=float(values.max().item()))
        state.component_thresholds[comp_idx] = max(float(state.component_thresholds[comp_idx].item()), float(tau) + float(radius_slack))
    median = float(torch.median(nll).item())
    iqr = _iqr(nll, float(config.eps))
    norm_energy = (nll - median) / max(iqr, float(config.eps))
    state.energy_median = median
    state.energy_iqr = iqr
    state.class_threshold = float(
        max(
            float(config.min_threshold),
            _quantile(norm_energy, energy_quantile, default=float(config.default_class_threshold)) + float(energy_slack),
        )
    )


def _build_old_memory(
    arrays: Mapping[str, np.ndarray],
    *,
    config: OPGACConfig,
    component_mode: str,
    min_component_count: int,
    max_components_per_class: int,
    radius_quantile: float,
    radius_slack: float,
    energy_quantile: float,
    energy_slack: float,
    var_shrinkage: float,
) -> OPGACMemory:
    source_x = torch.as_tensor(_norm_np(np.asarray(arrays["source_features"], dtype=np.float32)), dtype=torch.float32)
    source_y = np.asarray(arrays["source_labels"], dtype=np.int64).reshape(-1)
    if source_x.numel() == 0:
        raise ValueError("source_features is empty")
    global_var = source_x.var(dim=0, unbiased=False).clamp_min(float(config.min_variance))
    component_ids = None
    if component_mode == "rx":
        component_ids = np.asarray(arrays.get("source_rx_ids", []), dtype=str).reshape(-1)
    elif component_mode == "rx_day":
        rx = np.asarray(arrays.get("source_rx_ids", []), dtype=str).reshape(-1)
        day = np.asarray(arrays.get("source_day_ids", []), dtype=str).reshape(-1)
        if rx.size == source_y.size and day.size == source_y.size:
            component_ids = np.asarray([f"{r}|{d}" for r, d in zip(rx.tolist(), day.tolist())], dtype=str)
    if component_ids is not None and component_ids.size != source_y.size:
        component_ids = None
    states: dict[int, GaussianClassState] = {}
    for label in sorted(int(v) for v in np.unique(source_y).tolist()):
        class_idx = np.flatnonzero(source_y == int(label)).astype(np.int64)
        class_features = source_x[torch.as_tensor(class_idx, dtype=torch.long)]
        groups = _component_groups(
            source_y,
            component_ids,
            int(label),
            min_count=int(min_component_count),
            max_components=int(max_components_per_class),
        )
        component_members = [source_x[torch.as_tensor(idx, dtype=torch.long)] for idx in groups]
        states[int(label)] = _make_state(
            class_id=int(label),
            group="old",
            class_features=class_features,
            component_members=component_members,
            global_var=global_var,
            config=config,
            radius_quantile=radius_quantile,
            radius_slack=radius_slack,
            energy_quantile=energy_quantile,
            energy_slack=energy_slack,
            var_shrinkage=var_shrinkage,
            metadata={
                "source": "source_features",
                "component_mode": component_mode,
                "component_count": len(component_members),
                "source_count": int(class_features.size(0)),
            },
        )
    return OPGACMemory(old_states=states)


def _calibrate_memory_after_registration(
    memory: OPGACMemory,
    arrays: Mapping[str, np.ndarray],
    *,
    config: OPGACConfig,
    old_label_max: int,
    radius_quantile: float,
    radius_slack: float,
    energy_quantile: float,
    energy_slack: float,
) -> OPGACMemory:
    out = memory.clone()
    source_x = torch.as_tensor(_norm_np(np.asarray(arrays["source_features"], dtype=np.float32)), dtype=torch.float32)
    source_y = np.asarray(arrays["source_labels"], dtype=np.int64).reshape(-1)
    support_x = torch.as_tensor(_norm_np(np.asarray(arrays["support_features"], dtype=np.float32)), dtype=torch.float32)
    support_y = np.asarray(arrays["support_labels"], dtype=np.int64).reshape(-1)
    for label, state in out.old_states.items():
        parts = [source_x[torch.as_tensor(np.flatnonzero(source_y == int(label)), dtype=torch.long)]]
        old_support_idx = np.flatnonzero(support_y == int(label)).astype(np.int64)
        if old_support_idx.size:
            parts.append(support_x[torch.as_tensor(old_support_idx, dtype=torch.long)])
        calib = torch.cat([part for part in parts if part.numel() > 0], dim=0)
        _calibrate_state(
            state,
            calib,
            config=config,
            radius_quantile=radius_quantile,
            radius_slack=radius_slack,
            energy_quantile=energy_quantile,
            energy_slack=energy_slack,
        )
        state.metadata = {**state.metadata, "post_registration_calibration": "source_plus_target_old_support"}
    for label, state in out.new_states.items():
        if int(label) <= int(old_label_max):
            continue
        new_support_idx = np.flatnonzero(support_y == int(label)).astype(np.int64)
        if new_support_idx.size:
            calib = support_x[torch.as_tensor(new_support_idx, dtype=torch.long)]
            _calibrate_state(
                state,
                calib,
                config=config,
                radius_quantile=radius_quantile,
                radius_slack=radius_slack,
                energy_quantile=energy_quantile,
                energy_slack=energy_slack,
            )
            state.metadata = {**state.metadata, "post_registration_calibration": "target_new_support_only"}
    return out


def _closed_set_prediction(pred: OPGACPrediction) -> np.ndarray:
    old_scores = pred.old_scores.detach().cpu().float().reshape(-1).tolist()
    new_scores = pred.new_scores.detach().cpu().float().reshape(-1).tolist()
    old_labels = pred.best_old_labels.detach().cpu().long().reshape(-1).tolist()
    new_labels = pred.best_new_labels.detach().cpu().long().reshape(-1).tolist()
    closed = []
    for old_score, new_score, old_label, new_label in zip(old_scores, new_scores, old_labels, new_labels):
        old_value = float(old_score) if math.isfinite(float(old_score)) else float("inf")
        new_value = float(new_score) if math.isfinite(float(new_score)) else float("inf")
        closed.append(int(old_label) if old_value <= new_value else int(new_label))
    return np.asarray(closed, dtype=np.int64)


def _metrics(
    *,
    labels: np.ndarray,
    tx_ids: np.ndarray,
    query_roles: np.ndarray,
    prediction: OPGACPrediction,
    old_labels: set[int],
    new_labels: set[int],
) -> dict[str, Any]:
    label_list = [int(v) for v in np.asarray(labels, dtype=np.int64).reshape(-1).tolist()]
    pred_list = [int(v) for v in prediction.predicted_labels.detach().cpu().long().reshape(-1).tolist()]
    accepted_list = [bool(v) for v in prediction.accepted.detach().cpu().bool().reshape(-1).tolist()]
    closed_list = [int(v) for v in _closed_set_prediction(prediction).reshape(-1).tolist()]
    tx_list = [str(v) for v in np.asarray(tx_ids).reshape(-1).tolist()]
    role_list = [str(v) for v in np.asarray(query_roles).reshape(-1).tolist()]

    def rate(num: int | np.integer, den: int | np.integer) -> float:
        return float(num) / float(den) if int(den) > 0 else math.nan

    old_n = seen_new_n = unknown_n = known_n = 0
    accepted_n = accepted_old_n = accepted_seen_new_n = accepted_unknown_n = 0
    rejected_unknown_n = 0
    old_correct_n = seen_new_correct_n = known_correct_n = 0
    closed_old_correct_n = closed_seen_new_correct_n = closed_known_correct_n = 0
    for label, pred_label, accepted, closed_label in zip(label_list, pred_list, accepted_list, closed_list):
        is_old = int(label) in old_labels
        is_new = int(label) in new_labels
        is_unknown = int(label) == UNKNOWN_LABEL
        is_known = is_old or is_new
        if is_old:
            old_n += 1
        if is_new:
            seen_new_n += 1
        if is_unknown:
            unknown_n += 1
        if is_known:
            known_n += 1
        if accepted:
            accepted_n += 1
            if is_old:
                accepted_old_n += 1
            if is_new:
                accepted_seen_new_n += 1
            if is_unknown:
                accepted_unknown_n += 1
        if is_unknown and not accepted:
            rejected_unknown_n += 1
        if accepted and is_old and int(pred_label) == int(label):
            old_correct_n += 1
            known_correct_n += 1
        if accepted and is_new and int(pred_label) == int(label):
            seen_new_correct_n += 1
            known_correct_n += 1
        if is_old and int(closed_label) == int(label):
            closed_old_correct_n += 1
        if is_new and int(closed_label) == int(label):
            closed_seen_new_correct_n += 1
        if is_known and int(closed_label) == int(label):
            closed_known_correct_n += 1
    decision_counts: dict[str, int] = {}
    for decision in prediction.decisions:
        decision_counts[str(decision)] = decision_counts.get(str(decision), 0) + 1
    reason_counts: dict[str, int] = {}
    for row in prediction.reject_reasons:
        for reason in row:
            reason_counts[str(reason)] = reason_counts.get(str(reason), 0) + 1
    per_tx: dict[str, dict[str, Any]] = {}
    for tx in sorted(set(tx_list)):
        idx = [i for i, value in enumerate(tx_list) if value == tx]
        tx_accepted_n = sum(1 for i in idx if accepted_list[i])
        tx_correct_n = sum(
            1
            for i in idx
            if accepted_list[i] and label_list[i] != UNKNOWN_LABEL and int(pred_list[i]) == int(label_list[i])
        )
        per_tx[tx] = {
            "query_n": int(len(idx)),
            "accepted_n": int(tx_accepted_n),
            "correct_accepted_n": int(tx_correct_n),
            "accepted_rate": rate(tx_accepted_n, len(idx)),
        }
    unknown_nearest_old: dict[str, int] = {}
    best_old_list = [int(v) for v in prediction.best_old_labels.detach().cpu().long().reshape(-1).tolist()]
    for label, value in zip(label_list, best_old_list):
        if int(label) == UNKNOWN_LABEL:
            unknown_nearest_old[str(int(value))] = unknown_nearest_old.get(str(int(value)), 0) + 1
    query_role_counts: dict[str, int] = {}
    for role in role_list:
        query_role_counts[str(role)] = query_role_counts.get(str(role), 0) + 1
    return {
        "old_acc": rate(old_correct_n, old_n),
        "old_coverage": rate(accepted_old_n, old_n),
        "seen_new_acc": rate(seen_new_correct_n, seen_new_n),
        "seen_new_coverage": rate(accepted_seen_new_n, seen_new_n),
        "known_acc": rate(known_correct_n, known_n),
        "known_coverage": rate(accepted_old_n + accepted_seen_new_n, known_n),
        "coverage": rate(accepted_n, len(label_list)),
        "unknown_far": rate(accepted_unknown_n, unknown_n),
        "unknown_reject": rate(rejected_unknown_n, unknown_n),
        "full_acc": rate(known_correct_n + rejected_unknown_n, len(label_list)),
        "accepted_known_acc": rate(known_correct_n, accepted_old_n + accepted_seen_new_n),
        "no_reject_old_acc": rate(closed_old_correct_n, old_n),
        "no_reject_seen_new_acc": rate(closed_seen_new_correct_n, seen_new_n),
        "no_reject_known_acc": rate(closed_known_correct_n, known_n),
        "no_reject_full_acc": rate(closed_known_correct_n, len(label_list)),
        "old_n": int(old_n),
        "seen_new_n": int(seen_new_n),
        "unknown_n": int(unknown_n),
        "known_n": int(known_n),
        "total_n": int(len(label_list)),
        "accepted_n": int(accepted_n),
        "accepted_old_n": int(accepted_old_n),
        "accepted_seen_new_n": int(accepted_seen_new_n),
        "accepted_unknown_n": int(accepted_unknown_n),
        "rejected_unknown_n": int(rejected_unknown_n),
        "old_correct_n": int(old_correct_n),
        "seen_new_correct_n": int(seen_new_correct_n),
        "known_correct_n": int(known_correct_n),
        "decision_counts": decision_counts,
        "reject_reason_counts": reason_counts,
        "query_role_counts": query_role_counts,
        "per_tx": per_tx,
        "unknown_nearest_old_label_counts": unknown_nearest_old,
    }


def _load_payload(path: Path, args: argparse.Namespace) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    with np.load(path, allow_pickle=True) as data:
        manifest = _load_manifest(data)
        source_tx_ids = parse_tx_id_list(args.source_tx_ids or manifest.get("source_tx_ids", []))
        target_old_tx_ids = parse_tx_id_list(args.target_old_tx_ids or manifest.get("target_old_tx_ids", source_tx_ids))
        new_tx_ids = parse_tx_id_list(args.new_tx_ids or manifest.get("new_tx_ids", []))
        unknown_tx_ids = parse_tx_id_list(args.unknown_tx_ids or manifest.get("unknown_tx_ids", []))
        if not source_tx_ids or not target_old_tx_ids or not new_tx_ids or not unknown_tx_ids:
            raise ValueError(
                f"{path}: source/target_old/new/unknown tx ids are required for Stage2-C OPGAC evaluation"
            )
        sample_metadata = {
            key: data[key]
            for key in ("rx_ids", "day_ids", "eq_ids", "sig_ids", "channel_views", "sat_scenarios")
            if key in data
        }
        payload = build_sfe_payload_from_feature_arrays(
            features=np.asarray(data[str(args.features_key)], dtype=np.float32),
            tx_ids=data[str(args.tx_ids_key)],
            dataset_roles=data[str(args.role_key)] if str(args.role_key) in data else None,
            sample_metadata=sample_metadata,
            source_tx_ids=source_tx_ids,
            target_old_tx_ids=target_old_tx_ids,
            new_tx_ids=new_tx_ids,
            unknown_tx_ids=unknown_tx_ids,
            shots=int(args.shots),
            source_proto_per_tx=int(args.source_proto_per_tx),
            source_query_per_tx=int(args.source_query_per_tx),
            target_old_support_per_tx=int(args.target_old_support_per_tx),
            target_old_query_per_tx=int(args.target_old_query_per_tx),
            query_per_tx=int(args.query_per_tx),
            seed=int(args.seed),
            extra_metadata={
                "payload_source": str(path),
                "analysis_tool": "evaluate_opgac_stage2",
                "support_query_permission": "support_only_registration_query_eval_only",
            },
        )
        merged_manifest = dict(manifest)
        merged_manifest.update(payload.manifest)
        return {key: np.asarray(value) for key, value in payload.arrays.items()}, merged_manifest


def evaluate_file(path: Path, args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    arrays, manifest = _load_payload(path, args)
    feature_dim = int(np.asarray(arrays["source_features"]).shape[1])
    config = OPGACConfig(
        feature_dim=feature_dim,
        context_dim=int(args.context_dim),
        hidden_dim=int(args.hidden_dim),
        low_rank=int(args.low_rank),
        old_new_margin=float(args.old_new_margin),
        top2_margin=float(args.top2_margin),
        overlap_margin=float(args.overlap_margin),
        ambiguous_on_overlap=bool(args.ambiguous_on_overlap),
        default_component_threshold_quantile=float(args.radius_quantile),
        default_class_threshold=float(args.default_class_threshold),
        old_shrinkage_kappa=float(args.old_shrinkage_kappa),
        new_shrinkage_nu=float(args.new_shrinkage_nu),
        cov_shrinkage_nu=float(args.cov_shrinkage_nu),
        threshold_shrinkage_nu=float(args.threshold_shrinkage_nu),
    )
    ground_memory = _build_old_memory(
        arrays,
        config=config,
        component_mode=str(args.component_mode),
        min_component_count=int(args.min_component_count),
        max_components_per_class=int(args.max_components_per_class),
        radius_quantile=float(args.radius_quantile),
        radius_slack=float(args.radius_slack),
        energy_quantile=float(args.energy_quantile),
        energy_slack=float(args.energy_slack),
        var_shrinkage=float(args.var_shrinkage),
    )
    torch.manual_seed(int(args.seed))
    model = OPGACNet(config)
    support_x = torch.as_tensor(_norm_np(np.asarray(arrays["support_features"], dtype=np.float32)), dtype=torch.float32)
    support_y = torch.as_tensor(np.asarray(arrays["support_labels"], dtype=np.int64).reshape(-1), dtype=torch.long)
    target_old_support_mask = support_y < len(manifest["source_tx_ids"])
    target_new_support_mask = support_y >= len(manifest["source_tx_ids"])
    memory = model.initialize_memory(
        ground_memory,
        stage="Stage2-C",
        target_old_support=support_x[target_old_support_mask],
        target_old_labels=support_y[target_old_support_mask],
        target_new_support=support_x[target_new_support_mask],
        target_new_labels=support_y[target_new_support_mask],
    )
    memory = _calibrate_memory_after_registration(
        memory,
        arrays,
        config=config,
        old_label_max=len(manifest["source_tx_ids"]) - 1,
        radius_quantile=float(args.radius_quantile),
        radius_slack=float(args.radius_slack),
        energy_quantile=float(args.energy_quantile),
        energy_slack=float(args.energy_slack),
    )
    old_labels = set(int(v) for v in range(len(manifest["source_tx_ids"])))
    new_labels = set(int(v) for v in range(len(manifest["source_tx_ids"]), len(manifest["source_tx_ids"]) + len(manifest["new_tx_ids"])))
    query_x = torch.as_tensor(_norm_np(np.asarray(arrays["query_features"], dtype=np.float32)), dtype=torch.float32)
    query_y = np.asarray(arrays["query_labels"], dtype=np.int64).reshape(-1)
    query_tx = np.asarray(arrays["query_tx_ids"]).astype(str).reshape(-1)
    query_roles = np.asarray(arrays["query_roles"]).astype(str).reshape(-1)

    rows = []
    variants = ["opgac_strict"]
    if bool(args.include_confirm_new_variant):
        variants.append("opgac_confirm_new")
    detail: dict[str, Any] = {
        "feature_npz": str(path),
        "manifest": manifest,
        "memory_summary": {
            "old_component_counts": {str(k): int(v.means.size(0)) for k, v in memory.old_states.items()},
            "new_lifecycle": {str(k): str(v.lifecycle) for k, v in memory.new_states.items()},
            "new_overlap": {str(k): v.metadata for k, v in memory.new_states.items()},
            "version": int(memory.version),
            "uncertainty": float(memory.uncertainty),
        },
    }
    for variant in variants:
        eval_memory = memory.clone()
        if variant == "opgac_confirm_new":
            for state in eval_memory.new_states.values():
                state.lifecycle = "confirmed"
                state.metadata = {**state.metadata, "forced_confirmed_for_eval_variant": True}
        pred = model.predict(query_x, eval_memory)
        row = _metrics(
            labels=query_y,
            tx_ids=query_tx,
            query_roles=query_roles,
            prediction=pred,
            old_labels=old_labels,
            new_labels=new_labels,
        )
        row.update(
            {
                "feature_npz": str(path),
                "candidate": path.parent.name,
                "domain": str(manifest.get("target_old", {}).get("rxs") or manifest.get("target_receiver_label") or path.parent.name),
                "variant": variant,
                "shots": int(args.shots),
                "target_old_support_per_tx": int(args.target_old_support_per_tx),
                "target_old_query_per_tx": int(args.target_old_query_per_tx),
                "query_per_tx": int(args.query_per_tx),
                "component_mode": str(args.component_mode),
                "max_components_per_class": int(args.max_components_per_class),
                "radius_quantile": float(args.radius_quantile),
                "energy_quantile": float(args.energy_quantile),
                "energy_slack": float(args.energy_slack),
                "old_new_margin": float(args.old_new_margin),
                "top2_margin": float(args.top2_margin),
            }
        )
        rows.append(row)
        detail[variant] = {
            "metrics": row,
            "prediction_sample": {
                "labels": query_y[: int(args.prediction_sample_rows)].tolist(),
                "tx_ids": query_tx[: int(args.prediction_sample_rows)].tolist(),
                "query_roles": query_roles[: int(args.prediction_sample_rows)].tolist(),
                "predicted_labels": pred.predicted_labels[: int(args.prediction_sample_rows)].tolist(),
                "accepted": pred.accepted[: int(args.prediction_sample_rows)].tolist(),
                "decisions": pred.decisions[: int(args.prediction_sample_rows)],
                "reject_reasons": pred.reject_reasons[: int(args.prediction_sample_rows)],
            },
        }
    return rows, detail


def _collect_feature_paths(args: argparse.Namespace) -> list[Path]:
    paths: list[Path] = []
    for pattern in args.feature_npz:
        if any(ch in str(pattern) for ch in "*?["):
            paths.extend(sorted(Path().glob(str(pattern))))
        else:
            paths.append(Path(pattern))
    return sorted({p.resolve() for p in paths if p.exists()})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-npz", action="append", default=[], required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--features-key", default="features")
    parser.add_argument("--tx-ids-key", default="tx_ids")
    parser.add_argument("--role-key", default="dataset_role")
    parser.add_argument("--source-tx-ids", default=None)
    parser.add_argument("--target-old-tx-ids", default=None)
    parser.add_argument("--new-tx-ids", default=None)
    parser.add_argument("--unknown-tx-ids", default=None)
    parser.add_argument("--shots", type=int, default=10)
    parser.add_argument("--source-proto-per-tx", type=int, default=240)
    parser.add_argument("--source-query-per-tx", type=int, default=0)
    parser.add_argument("--target-old-support-per-tx", type=int, default=10)
    parser.add_argument("--target-old-query-per-tx", type=int, default=50)
    parser.add_argument("--query-per-tx", type=int, default=50)
    parser.add_argument("--seed", type=int, default=362017)
    parser.add_argument("--component-mode", choices=["class", "rx", "rx_day"], default="rx")
    parser.add_argument("--min-component-count", type=int, default=8)
    parser.add_argument("--max-components-per-class", type=int, default=4)
    parser.add_argument("--var-shrinkage", type=float, default=0.20)
    parser.add_argument("--radius-quantile", type=float, default=0.99)
    parser.add_argument("--radius-slack", type=float, default=0.0)
    parser.add_argument("--energy-quantile", type=float, default=0.99)
    parser.add_argument("--energy-slack", type=float, default=0.25)
    parser.add_argument("--context-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--low-rank", type=int, default=4)
    parser.add_argument("--old-new-margin", type=float, default=0.10)
    parser.add_argument("--top2-margin", type=float, default=0.02)
    parser.add_argument("--overlap-margin", type=float, default=0.20)
    parser.add_argument("--default-class-threshold", type=float, default=3.0)
    parser.add_argument("--old-shrinkage-kappa", type=float, default=3.0)
    parser.add_argument("--new-shrinkage-nu", type=float, default=10.0)
    parser.add_argument("--cov-shrinkage-nu", type=float, default=20.0)
    parser.add_argument("--threshold-shrinkage-nu", type=float, default=10.0)
    parser.add_argument("--ambiguous-on-overlap", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-confirm-new-variant", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prediction-sample-rows", type=int, default=12)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    feature_paths = _collect_feature_paths(args)
    if not feature_paths:
        raise FileNotFoundError("no feature NPZ files found")
    all_rows: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    for path in feature_paths:
        rows, detail = evaluate_file(path, args)
        all_rows.extend(rows)
        details.append(detail)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "stage2_opgac_eval_v1",
        "protocol_boundary": {
            "stage": "Stage2-C",
            "target_old_support": "labeled_support_only",
            "target_new_support": "labeled_support_only",
            "unknown_query": "eval_only_no_threshold_fit",
            "query_samples_used_for_registration": False,
        },
        "config": _jsonable(vars(args)),
        "rows": _jsonable(all_rows),
        "details": _jsonable(details),
    }
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    fieldnames = [
        "feature_npz",
        "candidate",
        "domain",
        "variant",
        "shots",
        "target_old_support_per_tx",
        "target_old_query_per_tx",
        "query_per_tx",
        "old_acc",
        "old_coverage",
        "seen_new_acc",
        "seen_new_coverage",
        "known_acc",
        "known_coverage",
        "coverage",
        "unknown_far",
        "unknown_reject",
        "full_acc",
        "accepted_known_acc",
        "no_reject_old_acc",
        "no_reject_seen_new_acc",
        "no_reject_known_acc",
        "no_reject_full_acc",
        "old_n",
        "seen_new_n",
        "unknown_n",
        "known_n",
        "total_n",
        "accepted_n",
        "accepted_old_n",
        "accepted_seen_new_n",
        "accepted_unknown_n",
        "old_correct_n",
        "seen_new_correct_n",
        "known_correct_n",
        "component_mode",
        "max_components_per_class",
        "radius_quantile",
        "energy_quantile",
        "energy_slack",
        "old_new_margin",
        "top2_margin",
    ]
    with args.summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)
    print(
        json.dumps(
            {
                "feature_files": len(feature_paths),
                "rows": len(all_rows),
                "summary_csv": str(args.summary_csv),
                "output_json": str(args.output_json),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
