from __future__ import annotations

import csv
import json
import math
import os
import random
import re
import time
from collections import OrderedDict
from datetime import datetime
from types import SimpleNamespace
from typing import Any, Callable, Dict, Mapping, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from cvsrffi.runtime_threads import snapshot_thread_runtime
from cvsrffi.losses import (
    SmoothGroupDROState,
    domain_aware_supcon_loss,
    groupdro_or_hard_domain_ce_loss,
)

try:
    from training_controls import compute_mixstyle_epoch_state
except ImportError:  # pragma: no cover - package import fallback
    from ..training_controls import compute_mixstyle_epoch_state

from .client_split import build_client_loaders, build_client_splits, get_sample_metadata, summarize_client_splits
from .conditioned_receiver_dg import StyleConditionedReceiverDG
from .activation_tokens import ActivationTokenCodec
from .distill_anchors import (
    LogitAnchorBank,
    build_logit_anchor_stats,
    logit_anchor_kd_loss,
    logit_anchor_stats_payload_size_bytes,
    merge_logit_anchor_stats,
)
from .fed_aggregate import aggregate_state_dicts, resolve_exclude_keys
from .fed_fishr import (
    FedFishrBank,
    build_fed_fishr_stats,
    fed_fishr_log_summary,
    fed_fishr_reweight,
    fed_fishr_target_loss,
    merge_fed_fishr_stats,
)
from .fed_grl_controller import FedCGRLController
from .fedcvs_vmb import (
    FedCVSCoralStatsBank,
    FedCVSVMBPrototypeBank,
    adversarial_warmup_weight,
    aggregate_gradients,
    apply_server_gradient_step,
    build_class_conditional_coral_stats,
    build_prototype_stats,
    class_conditional_coral_loss,
    client_domain_from_id,
    coral_stats_payload_size_bytes,
    domain_balanced_weights,
    gradient_cosine_summary,
    gradient_norm,
    gradient_payload_size_bytes,
    merge_coral_stats,
    merge_prototype_stats,
    prototype_stats_payload_size_bytes,
    prototype_contrastive_loss,
    select_domain_balanced_clients,
    select_transmitter_balanced_indices,
    vmb_stage_for_round,
)
from .fedprox import compute_fedprox_loss
from .gradient_stats import conflict_aware_aggregate_gradients
from .proto_evidence_bank import ProtoEvidence, ProtoEvidenceBank
from .reliability_fusion import (
    collaborative_probability_fusion,
    conservative_probability_fusion,
    harm_rescue_report,
    normalize_probabilities,
)
from .rf_style_extractor import RFStyleExtractor
from .style_bank import FederatedStyleBank
from .style_packet import StyleDomainBatch, StylePacket, style_code_from_stats
from .virtual_domain_sampler import VirtualDomainSampler, VirtualStyleView


_DG_DIAG_FIELD_MAP = OrderedDict(
    [
        ("diag_domain_count", "client_diag_domain_count_avg"),
        ("diag_fishr_domain_count", "client_diag_fishr_domain_count_avg"),
        ("diag_domain_loss_active", "client_diag_domain_loss_active_rate"),
        ("diag_adv_active", "client_diag_adv_active_rate"),
        ("diag_cons_active", "client_diag_cons_active_rate"),
        ("diag_group_ce_active", "client_diag_group_ce_active_rate"),
        ("diag_fishr_active", "client_diag_fishr_active_rate"),
        ("diag_rx_adv_active", "client_diag_rx_adv_active_rate"),
        ("diag_sat_aug_active", "client_diag_sat_aug_active_rate"),
        ("diag_baseline_sat_view_active", "client_diag_baseline_sat_view_active_rate"),
        ("diag_sat_cls_active", "client_diag_sat_cls_active_rate"),
        ("diag_sat_cons_active", "client_diag_sat_cons_active_rate"),
        ("diag_style_batch_active", "client_diag_style_batch_active_rate"),
        ("diag_style_domain_count", "client_diag_style_domain_count_avg"),
        ("diag_stage1_aux_active", "client_diag_stage1_aux_active_rate"),
        ("diag_coral_global_active", "client_diag_coral_global_active_rate"),
        ("diag_coral_virtual_active", "client_diag_coral_virtual_active_rate"),
        ("diag_coral_zdom_active", "client_diag_coral_zdom_active_rate"),
    ]
)

_CORAL_METRIC_KEYS = {
    "loss_coral_zid_global",
    "loss_coral_zid_virtual",
    "loss_coral_zdom_global",
    "coral_zid_global_active_classes",
    "coral_zid_global_mean_dist",
    "coral_zid_global_cov_dist",
    "coral_zid_global_skip_rate",
    "coral_zid_virtual_active_classes",
    "coral_zid_virtual_mean_dist",
    "coral_zid_virtual_cov_dist",
    "coral_zid_virtual_skip_rate",
    "coral_zdom_global_active_classes",
    "coral_zdom_global_mean_dist",
    "coral_zdom_global_cov_dist",
    "coral_zdom_global_skip_rate",
    "coral_payload_bytes",
}

_SCALAR_COMPONENT_METRIC_KEYS = {
    "cons_cos",
    "sat_cos",
    "fed_proto_cos",
    "vmb_tx_proto_cos",
    "vmb_rx_proto_cos",
    "vmb_tx_proto_active",
    "vmb_rx_proto_active",
    "tx_adv_r_acc",
    "dom_acc",
    "zdom_target_acc",
    "grl_target_acc",
    "style_num_domains",
    "style_batch_views",
    "style_domain_entropy",
    "style_dg_ready",
    "style_target_domain_count",
    "style_requested_remote_views",
    "style_appended_remote_views",
    "style_real_mix_active",
    "stage1_domain_pretrain_active",
    "domain_unsup_active",
    "domain_unsup_view_count",
    "domain_unsup_zdom_cos",
    "domain_unsup_client_compact",
    "domain_unsup_client_radius",
    "domain_unsup_dom_entropy",
    "domain_unsup_dom_acc",
    "domain_unsup_zdom_var",
    "style_gate_value",
    "fishr_gate_value",
    "kd_active",
    "logit_anchor_count",
    "logit_anchor_payload_bytes",
    "activation_token_payload_bytes",
    "activation_token_compression_ratio",
    "activation_token_quant_error",
    "feature_probe_samples",
    "fed_fishr_active",
    "fed_fishr_active_classes",
    "fed_fishr_var_dist",
    "fed_fishr_skip_rate",
    "fed_fishr_payload_bytes",
    "fed_fishr_target_ready",
}


def _is_train_component_metric(name: str) -> bool:
    return (
        name.startswith("loss_")
        or name.startswith("diag_")
        or name.startswith("fed_cgrl_")
        or name in _CORAL_METRIC_KEYS
        or name in _SCALAR_COMPONENT_METRIC_KEYS
    )


def _batch_to_xyd(batch, device):
    if isinstance(batch, Mapping):
        x = batch.get("iq", batch.get("x"))
        y = batch.get("label", batch.get("y"))
        d = batch.get("domain", batch.get("d", batch.get("receiver", None)))
    else:
        x = batch[0]
        y = batch[1]
        d = batch[2] if len(batch) > 2 else None
    x = x.to(device, non_blocking=True)
    y = y.to(device, non_blocking=True).long()
    if d is not None:
        try:
            d = torch.as_tensor(d, device=device).view(-1).long()
        except Exception:
            d = None
    return x, y, d


def _client_metric_avg(client_results: Mapping[str, Mapping[str, Any]], name: str) -> float:
    vals = []
    for result in client_results.values():
        if name not in result:
            continue
        try:
            value = float(result[name])
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            vals.append(value)
    return float(sum(vals) / max(1, len(vals))) if vals else float("nan")


def _client_seen_weighted_avg(client_results: Mapping[str, Mapping[str, Any]], name: str) -> float:
    weighted = 0.0
    total = 0
    for result in client_results.values():
        if name not in result:
            continue
        try:
            value = float(result[name])
            seen = int(result.get("seen", 0))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value) and seen > 0:
            weighted += value * seen
            total += seen
    return float(weighted / max(1, total)) if total > 0 else float("nan")


def _domain_metric_variance(client_results: Mapping[str, Mapping[str, Any]], name: str, domain_name: str) -> float:
    by_domain: Dict[str, list[float]] = {}
    for result in client_results.values():
        if name not in result:
            continue
        domain = str(result.get(domain_name, "unknown"))
        try:
            value = float(result[name])
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            by_domain.setdefault(domain, []).append(value)
    domain_means = [sum(vals) / max(1, len(vals)) for vals in by_domain.values() if vals]
    if len(domain_means) < 2:
        return float("nan")
    mean = sum(domain_means) / float(len(domain_means))
    return float(sum((value - mean) ** 2 for value in domain_means) / float(len(domain_means)))


def _sum_client_hist(client_results: Mapping[str, Mapping[str, Any]], name: str) -> list[int]:
    hist: list[int] = []
    for result in client_results.values():
        raw = result.get(name)
        if not isinstance(raw, list):
            continue
        if len(hist) < len(raw):
            hist.extend([0] * (len(raw) - len(hist)))
        for idx, value in enumerate(raw):
            try:
                hist[idx] += int(value)
            except (TypeError, ValueError):
                continue
    return hist


_STYLE_ZDOM_BUCKETS = ("clean", "virtual", "all_style", "real")


def _empty_style_zdom_bucket() -> Dict[str, Any]:
    return {
        "correct": 0,
        "total": 0,
        "acc": float("nan"),
        "target_hist": {},
        "pred_hist": {},
        "examples": [],
    }


def _histogram_dict(values: Optional[torch.Tensor]) -> Dict[str, int]:
    if values is None:
        return {}
    vals = values.detach().view(-1).cpu().long()
    vals = vals[vals >= 0]
    if vals.numel() == 0:
        return {}
    counts = torch.bincount(vals, minlength=int(vals.max().item()) + 1)
    return {str(i): int(v) for i, v in enumerate(counts.tolist()) if int(v) > 0}


def _finalize_style_zdom_bucket(bucket: Mapping[str, Any]) -> Dict[str, Any]:
    out = dict(bucket)
    total = int(out.get("total", 0) or 0)
    correct = int(out.get("correct", 0) or 0)
    out["acc"] = 100.0 * float(correct) / float(total) if total > 0 else float("nan")
    out["correct"] = correct
    out["total"] = total
    out["target_hist"] = dict(out.get("target_hist", {}) or {})
    out["pred_hist"] = dict(out.get("pred_hist", {}) or {})
    out["examples"] = list(out.get("examples", []) or [])
    return out


def _domain_probe_stats(
    domain_logits: Optional[torch.Tensor],
    targets: Optional[torch.Tensor],
    *,
    mask: Optional[torch.Tensor] = None,
    max_examples: int = 4,
    sources: Optional[list[str]] = None,
) -> Dict[str, Any]:
    bucket = _empty_style_zdom_bucket()
    if not torch.is_tensor(domain_logits) or targets is None or not torch.is_tensor(targets):
        return bucket
    if domain_logits.dim() != 2:
        return bucket
    targets = targets.to(device=domain_logits.device).view(-1).long()
    if int(domain_logits.size(0)) != int(targets.numel()):
        return bucket
    valid = targets >= 0
    if mask is not None:
        mask = mask.to(device=domain_logits.device).view(-1).bool()
        if int(mask.numel()) != int(targets.numel()):
            return bucket
        valid = valid & mask
    valid_before_head = int(valid.sum().detach().cpu().item())
    if int(domain_logits.size(1)) > 0:
        valid = valid & (targets < int(domain_logits.size(1)))
    if not bool(valid.any()):
        bucket["skipped"] = valid_before_head
        return bucket
    preds = domain_logits.detach().argmax(dim=1).view(-1).long()
    target_valid = targets[valid].detach()
    pred_valid = preds[valid].detach()
    correct = int((pred_valid == target_valid).sum().detach().cpu().item())
    total = int(target_valid.numel())
    examples = []
    valid_indices = torch.nonzero(valid.detach().cpu(), as_tuple=False).view(-1).tolist()
    for offset, original_idx in enumerate(valid_indices[: max(0, int(max_examples))]):
        item = {
            "idx": int(original_idx),
            "target": int(target_valid[offset].detach().cpu().item()),
            "pred": int(pred_valid[offset].detach().cpu().item()),
        }
        if sources is not None and 0 <= int(original_idx) < len(sources):
            item["source"] = str(sources[int(original_idx)])
        examples.append(item)
    bucket.update(
        {
            "correct": correct,
            "total": total,
            "skipped": max(0, valid_before_head - total),
            "target_hist": _histogram_dict(target_valid),
            "pred_hist": _histogram_dict(pred_valid),
            "examples": examples,
        }
    )
    return _finalize_style_zdom_bucket(bucket)


def _merge_hist_dict(dst: Dict[str, int], src: Mapping[str, Any]) -> Dict[str, int]:
    out = dict(dst)
    for key, value in (src or {}).items():
        try:
            out[str(key)] = int(out.get(str(key), 0)) + int(value)
        except (TypeError, ValueError):
            continue
    return out


def _merge_style_zdom_probe(
    acc: Optional[Mapping[str, Any]],
    probe: Optional[Mapping[str, Any]],
    *,
    max_examples: int = 4,
) -> Optional[Dict[str, Any]]:
    if not probe:
        return dict(acc) if acc else None
    merged: Dict[str, Any] = dict(acc or {})
    merged["mode"] = str(merged.get("mode") or probe.get("mode") or "")
    merged["head"] = str(merged.get("head") or probe.get("head") or "")
    merged["label_semantics"] = str(merged.get("label_semantics") or probe.get("label_semantics") or "")
    merged["clients"] = sorted(set(list(merged.get("clients", []) or []) + [str(probe.get("client_id", ""))]))
    for bucket_name in _STYLE_ZDOM_BUCKETS:
        dst = _finalize_style_zdom_bucket(merged.get(bucket_name, _empty_style_zdom_bucket()))
        src = _finalize_style_zdom_bucket(probe.get(bucket_name, _empty_style_zdom_bucket()))
        examples = list(dst.get("examples", []) or [])
        for item in src.get("examples", []) or []:
            if len(examples) >= max(0, int(max_examples)):
                break
            examples.append(dict(item))
        bucket = {
            "correct": int(dst.get("correct", 0)) + int(src.get("correct", 0)),
            "total": int(dst.get("total", 0)) + int(src.get("total", 0)),
            "skipped": int(dst.get("skipped", 0) or 0) + int(src.get("skipped", 0) or 0),
            "target_hist": _merge_hist_dict(dict(dst.get("target_hist", {}) or {}), src.get("target_hist", {}) or {}),
            "pred_hist": _merge_hist_dict(dict(dst.get("pred_hist", {}) or {}), src.get("pred_hist", {}) or {}),
            "examples": examples,
        }
        merged[bucket_name] = _finalize_style_zdom_bucket(bucket)
    return merged


def _style_zdom_probe_flat_metrics(probe: Optional[Mapping[str, Any]]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if not probe:
        return out
    for bucket_name in _STYLE_ZDOM_BUCKETS:
        bucket = _finalize_style_zdom_bucket(probe.get(bucket_name, _empty_style_zdom_bucket()))
        out[f"style_zdom_{bucket_name}_acc"] = float(bucket.get("acc", float("nan")))
        out[f"style_zdom_{bucket_name}_total"] = float(bucket.get("total", 0))
    return out


def _style_zdom_probe_text(bucket: Mapping[str, Any]) -> str:
    finalized = _finalize_style_zdom_bucket(bucket)
    total = int(finalized.get("total", 0))
    if total <= 0:
        return "nan%(0/0)"
    return f"{float(finalized.get('acc', float('nan'))):.2f}%({int(finalized.get('correct', 0))}/{total})"


_MAIN_TEST_KEYS = [
    "test_unseen_day_seen_rx",
    "test_seen_day_unseen_rx",
    "test_unseen_day_unseen_rx",
]

_ROUND_BEGIN_SEPARATOR = "=" * 120
_ROUND_END_SEPARATOR = "-" * 120


def _round_tag(round_idx: int, total_rounds: int) -> str:
    return f"R{int(round_idx):03d}/{max(1, int(total_rounds)):03d}"


def _expand_sat_eval_on_for_log(eval_sat_on: str) -> list[str]:
    names = [item.strip() for item in str(eval_sat_on or "").split(",") if item.strip()]
    expanded: list[str] = []
    for name in names:
        key = name.lower()
        if key == "main":
            expanded.extend(_MAIN_TEST_KEYS)
        elif key == "all":
            expanded.append("all")
        else:
            expanded.append(name)
    deduped: list[str] = []
    for name in expanded:
        if name not in deduped:
            deduped.append(name)
    return deduped


def _test_subset_label(name: str, meta: Mapping[str, Any]) -> str:
    if name == "test_unseen_day_seen_rx":
        return f"unseen_day_seen_rx(days={meta.get('days_label', [])}, rxs={meta.get('rxs_idx', [])})"
    if name == "test_seen_day_unseen_rx":
        return f"seen_day_unseen_rx(days={meta.get('days_label', [])}, rxs={meta.get('rxs_idx', [])})"
    if name == "test_unseen_day_unseen_rx":
        return f"unseen_day_unseen_rx(days={meta.get('days_label', [])}, rxs={meta.get('rxs_idx', [])})"
    if name.startswith("test_day_"):
        return f"day={meta.get('days_label', ['?'])[0]} on seen_rxs={meta.get('rxs_idx', [])}"
    if name.startswith("test_unseen_day_rx_"):
        return f"rx={meta.get('rxs_idx', ['?'])[0]} on unseen_days={meta.get('days_label', [])}"
    if name.startswith("test_rx_"):
        return f"rx={meta.get('rxs_idx', ['?'])[0]} on seen_days={meta.get('days_label', [])}"
    return name


def _ordered_named_test_keys(named_stats: Mapping[str, Mapping[str, float]]) -> list[str]:
    names = list(named_stats.keys())
    return [k for k in _MAIN_TEST_KEYS if k in named_stats] + [k for k in names if k not in _MAIN_TEST_KEYS]


def _aggregate_named_test_stats(named_stats: Mapping[str, Mapping[str, float]]) -> Dict[str, float]:
    keys = [k for k in _MAIN_TEST_KEYS if k in named_stats] or list(named_stats.keys())
    correct = 0
    total = 0
    for key in keys:
        stats = named_stats.get(key, {})
        correct += int(stats.get("tx_correct", 0))
        total += int(stats.get("tx_total", 0))
    return {
        "tx_acc": 100.0 * correct / max(1, total),
        "tx_correct": int(correct),
        "tx_total": int(total),
    }


def _format_named_test_lines(
    named_stats: Mapping[str, Mapping[str, float]],
    named_meta: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    lines = []
    for name in _ordered_named_test_keys(named_stats):
        stats = named_stats[name]
        label = _test_subset_label(name, named_meta.get(name, {}))
        tx_acc = float(stats.get("tx_acc", float("nan")))
        tx_correct = int(stats.get("tx_correct", 0))
        tx_total = int(stats.get("tx_total", 0))
        lines.append(f"          {label}: tx={tx_acc:.2f}% ({tx_correct}/{tx_total})")
    return lines


def _format_extra_test_lines(extra_tests: Mapping[str, Any]) -> list[str]:
    sat_stats = extra_tests.get("sat_channel") if isinstance(extra_tests, Mapping) else None
    if not isinstance(sat_stats, Mapping):
        return []
    lines = []
    split_order = _MAIN_TEST_KEYS
    for scenario, stats in sat_stats.items():
        if not isinstance(stats, Mapping):
            continue
        agg = stats.get("aggregate", {}) or {}
        selected = ",".join(stats.get("selected_names", []) or [])
        strict = stats.get("strict_udu", float("nan"))
        try:
            strict_text = f"{float(strict):.2f}%"
        except Exception:
            strict_text = "nan%"
        lines.append(
            f"          scenario={scenario} selected={selected} "
            f"overall_tx={agg.get('tx_acc', float('nan')):.2f}% "
            f"strict_udu={strict_text} "
            f"({int(agg.get('tx_correct', 0))}/{int(agg.get('tx_total', 0))})"
        )
        named = stats.get("named", {}) or {}
        if isinstance(named, Mapping):
            for split_name in split_order:
                split_stats = named.get(split_name)
                if not isinstance(split_stats, Mapping):
                    continue
                try:
                    tx_acc = float(split_stats.get("tx_acc", float("nan")))
                except Exception:
                    tx_acc = float("nan")
                lines.append(
                    f"          [SAT-TEST-SPLIT] scenario={scenario} {split_name}: "
                    f"tx={tx_acc:.2f}% "
                    f"({int(split_stats.get('tx_correct', 0))}/{int(split_stats.get('tx_total', 0))})"
                )
    return lines


def _named_tx_acc_summary(named_stats: Mapping[str, Mapping[str, float]]) -> Dict[str, float]:
    summary = {}
    for name, stats in named_stats.items():
        try:
            summary[name] = float(stats.get("tx_acc", float("nan")))
        except (TypeError, ValueError):
            summary[name] = float("nan")
    return summary


def _safe_iq_tensor(x: torch.Tensor, clamp: float = 8.0) -> torch.Tensor:
    return torch.nan_to_num(x, nan=0.0, posinf=float(clamp), neginf=-float(clamp)).clamp(-float(clamp), float(clamp))


def _safe_l2_normalize(x: torch.Tensor, dim: int = 1, eps: float = 1e-6) -> torch.Tensor:
    x = torch.nan_to_num(x.float(), nan=0.0, posinf=0.0, neginf=0.0)
    return x / torch.linalg.vector_norm(x, ord=2, dim=dim, keepdim=True).clamp_min(float(eps))


def _cosine_consistency_loss(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-6):
    dist = (1.0 - (_safe_l2_normalize(a, eps=eps) * _safe_l2_normalize(b, eps=eps)).sum(dim=1).clamp(-1.0, 1.0)).clamp(0.0, 2.0)
    return dist.mean(), float((1.0 - dist).mean().detach().item())


def _symmetric_kl_consistency_loss(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    pa = F.softmax(a.float(), dim=1).clamp_min(1e-8)
    pb = F.softmax(b.float(), dim=1).clamp_min(1e-8)
    log_pa = pa.log()
    log_pb = pb.log()
    return 0.5 * (
        F.kl_div(log_pa, pb.detach(), reduction="batchmean")
        + F.kl_div(log_pb, pa.detach(), reduction="batchmean")
    )


def _logit_entropy(logits: Optional[torch.Tensor]) -> float:
    if not torch.is_tensor(logits) or logits.numel() == 0:
        return float("nan")
    prob = F.softmax(logits.float(), dim=1).clamp_min(1e-8)
    ent = -(prob * prob.log()).sum(dim=1).mean()
    return float(ent.detach().item())


def _feature_variance_floor_loss(z: Optional[torch.Tensor], ref: torch.Tensor, floor: float) -> tuple[torch.Tensor, float]:
    zero = ref.new_tensor(0.0)
    if not torch.is_tensor(z) or z.dim() != 2 or int(z.size(0)) <= 1 or float(floor) <= 0.0:
        return zero, float("nan")
    z = torch.nan_to_num(z.float(), nan=0.0, posinf=0.0, neginf=0.0)
    std = z.std(dim=0, unbiased=False)
    finite_std = torch.nan_to_num(std, nan=0.0, posinf=0.0, neginf=0.0)
    return F.relu(float(floor) - finite_std).mean(), float(finite_std.mean().detach().item())


def _client_zdom_compactness_loss(z: Optional[torch.Tensor], ref: torch.Tensor) -> tuple[torch.Tensor, float]:
    zero = ref.new_tensor(0.0)
    if not torch.is_tensor(z) or z.dim() != 2 or int(z.size(0)) <= 1:
        return zero, float("nan")
    z_norm = _safe_l2_normalize(z.float(), dim=1)
    center = _safe_l2_normalize(z_norm.mean(dim=0, keepdim=True), dim=1)
    dist = (1.0 - (z_norm * center).sum(dim=1).clamp(-1.0, 1.0)).clamp(0.0, 2.0)
    return dist.mean(), float(dist.mean().detach().item())


def _concat_optional_domain(d: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if d is None:
        return None
    return torch.cat([d, d], dim=0)


def _covariance_orth_loss(z_id: Optional[torch.Tensor], z_dom: Optional[torch.Tensor], ref: torch.Tensor) -> torch.Tensor:
    if z_id is None or z_dom is None or z_id.size(0) <= 1 or z_dom.size(0) <= 1:
        return ref.new_tensor(0.0)
    z_id = torch.nan_to_num(z_id.float(), nan=0.0, posinf=0.0, neginf=0.0)
    z_dom = torch.nan_to_num(z_dom.float(), nan=0.0, posinf=0.0, neginf=0.0)
    z_id = z_id - z_id.mean(dim=0, keepdim=True)
    z_dom = z_dom - z_dom.mean(dim=0, keepdim=True)
    cov = (z_id.t() @ z_dom) / float(max(1, z_id.size(0) - 1))
    return torch.mean(cov * cov)


def _same_tx_cross_domain_consistency(z_id: Optional[torch.Tensor], y: torch.Tensor, d: Optional[torch.Tensor], ref: torch.Tensor):
    if z_id is None or d is None:
        return ref.new_tensor(0.0), float("nan")
    z = _safe_l2_normalize(z_id, dim=1)
    y = y.view(-1).long()
    d = d.view(-1).long()
    losses = []
    sims = []
    for cls in torch.unique(y):
        m_cls = y == cls
        doms = torch.unique(d[m_cls])
        if doms.numel() < 2:
            continue
        cents = []
        for dom in doms:
            m = m_cls & (d == dom)
            if bool(m.any()):
                cents.append(_safe_l2_normalize(z[m].mean(dim=0, keepdim=True), dim=1).squeeze(0))
        if len(cents) < 2:
            continue
        sim = torch.stack(cents, dim=0) @ torch.stack(cents, dim=0).t()
        iu = torch.triu_indices(sim.size(0), sim.size(1), offset=1, device=sim.device)
        pair_sim = sim[iu[0], iu[1]]
        losses.append((1.0 - pair_sim).mean())
        sims.append(float(pair_sim.mean().detach().item()))
    if not losses:
        return ref.new_tensor(0.0), float("nan")
    return torch.stack(losses).mean(), sum(sims) / max(1, len(sims))


def _fishr_logit_gradient_variance_loss(logits: torch.Tensor, y: torch.Tensor, d: Optional[torch.Tensor], *, min_domains: int = 2):
    if d is None or logits.size(0) <= 1:
        return logits.new_tensor(0.0)
    d = d.view(-1).long()
    valid = d >= 0
    if not bool(valid.any()):
        return logits.new_tensor(0.0)
    prob = F.softmax(logits.float(), dim=1)
    one_hot = F.one_hot(y.view(-1).long(), num_classes=logits.size(1)).to(prob.dtype)
    grad_proxy = prob - one_hot
    vars_by_domain = []
    for dom in torch.unique(d[valid]):
        m = valid & (d == dom)
        if int(m.sum().item()) > 1:
            vars_by_domain.append(grad_proxy[m].var(dim=0, unbiased=False))
    if len(vars_by_domain) < max(2, int(min_domains)):
        return logits.new_tensor(0.0)
    V = torch.stack(vars_by_domain, dim=0)
    target = V.mean(dim=0, keepdim=True).detach()
    return ((V - target) ** 2).mean()


def _batch_domain_count(d: Optional[torch.Tensor]) -> int:
    if d is None or d.numel() == 0:
        return 0
    vals = d.view(-1).long()
    vals = vals[vals >= 0]
    return int(torch.unique(vals).numel()) if vals.numel() else 0


def _batch_fishr_domain_count(d: Optional[torch.Tensor]) -> int:
    if d is None or d.numel() == 0:
        return 0
    vals = d.view(-1).long()
    valid = vals >= 0
    if not bool(valid.any()):
        return 0
    count = 0
    for dom in torch.unique(vals[valid]):
        if int((valid & (vals == dom)).sum().item()) > 1:
            count += 1
    return count


def _nested_tensor(out: Mapping[str, Any], key: str, group: str, subkey: str) -> Optional[torch.Tensor]:
    value = out.get(key)
    if torch.is_tensor(value):
        return value
    nested = out.get(group)
    if isinstance(nested, Mapping):
        value = nested.get(subkey)
        if torch.is_tensor(value):
            return value
    return None


def _select_generalization_feature(out: Mapping[str, Any], cfg) -> Optional[torch.Tensor]:
    name = str(getattr(cfg, "generalization_feature", "z_id") or "z_id").lower().strip()
    if name == "z_id":
        value = out.get("z_id")
        return value if torch.is_tensor(value) else None
    if name in {"id_feat_joint", "feat_joint", "joint"}:
        return _nested_tensor(out, "id_feat_joint", "aux_id", "feat_joint")
    if name in {"id_feat_pa", "feat_pa", "pa"}:
        return _nested_tensor(out, "id_feat_pa", "aux_id", "feat_pa")
    if name in {"id_feat_dac", "feat_dac", "dac"}:
        return _nested_tensor(out, "id_feat_dac", "aux_id", "feat_dac")
    return None


def _fed_proto_enabled(cfg) -> bool:
    return bool(getattr(cfg, "use_fed_proto_stats", False)) or float(getattr(cfg, "lambda_fed_proto", 0.0)) > 0.0


def _fed_coral_enabled(cfg) -> bool:
    if bool(getattr(cfg, "use_fed_coral", False)):
        return True
    return (
        _coral_weight(cfg, "lambda_fl_coral_zid_global", "lambda_fed_coral") > 0.0
        or _coral_weight(cfg, "lambda_fl_coral_zid_virtual", "lambda_fed_coral_virtual") > 0.0
        or float(getattr(cfg, "lambda_fl_coral_zdom_global", 0.0) or 0.0) > 0.0
    )


def _fed_fishr_enabled(cfg) -> bool:
    if bool(getattr(cfg, "use_fed_fishr", False)):
        return True
    return float(getattr(cfg, "lambda_fed_fishr", 0.0) or 0.0) > 0.0


def _fed_fishr_mode(cfg) -> str:
    mode = str(getattr(cfg, "fed_fishr_mode", "reweight") or "reweight").lower().strip()
    if mode in {"off", "none", "disabled"}:
        return "off"
    if mode not in {"reweight", "target_loss", "both"}:
        raise ValueError("fed_fishr_mode must be one of: reweight, target_loss, both, off")
    return mode


def _coral_weight(cfg, primary_name: str, alias_name: Optional[str] = None) -> float:
    primary = float(getattr(cfg, primary_name, 0.0) or 0.0)
    if primary != 0.0 or alias_name is None:
        return primary
    return float(getattr(cfg, alias_name, 0.0) or 0.0)


def _coral_config_value(cfg, primary_name: str, alias_name: Optional[str], default: Any) -> Any:
    primary = getattr(cfg, primary_name, default)
    if alias_name is None:
        return primary
    alias = getattr(cfg, alias_name, primary)
    return alias if primary == default and alias != default else primary


def _coral_feature_name(cfg) -> str:
    return str(_coral_config_value(cfg, "fl_coral_feature", "fed_coral_feature", "z_id") or "z_id").lower().strip()


def _merge_fed_proto_stats(accum: Optional[Dict[str, torch.Tensor]], stats: Optional[Mapping[str, torch.Tensor]]):
    if not stats:
        return accum
    if accum is None:
        return {k: v.detach().cpu().clone() for k, v in stats.items() if torch.is_tensor(v)}
    for key, value in stats.items():
        if not torch.is_tensor(value):
            continue
        value = value.detach().cpu()
        if key not in accum:
            accum[key] = value.clone()
            continue
        if accum[key].shape != value.shape:
            if accum[key].dim() != value.dim() or accum[key].shape[1:] != value.shape[1:]:
                raise ValueError(f"Federated prototype stat shape mismatch for {key}: {tuple(accum[key].shape)} vs {tuple(value.shape)}")
            rows = max(int(accum[key].shape[0]), int(value.shape[0]))
            if int(accum[key].shape[0]) < rows:
                padded = torch.zeros((rows, *accum[key].shape[1:]), dtype=accum[key].dtype)
                padded[: accum[key].shape[0]] = accum[key]
                accum[key] = padded
            if int(value.shape[0]) < rows:
                padded = torch.zeros((rows, *value.shape[1:]), dtype=value.dtype)
                padded[: value.shape[0]] = value
                value = padded
        accum[key] = accum[key] + value
    return accum


def _collect_fed_proto_stats(
    z_id: Optional[torch.Tensor],
    labels: torch.Tensor,
    domains: Optional[torch.Tensor],
    *,
    num_classes: int,
    num_domains: int,
) -> Optional[Dict[str, torch.Tensor]]:
    if z_id is None or not torch.is_tensor(z_id) or z_id.dim() != 2 or z_id.size(0) == 0:
        return None
    z = _safe_l2_normalize(z_id.detach(), dim=1).cpu()
    y = labels.detach().view(-1).long().cpu()
    if int(y.numel()) != int(z.size(0)):
        return None
    c = max(1, int(num_classes))
    feat_dim = int(z.size(1))
    class_sum = torch.zeros(c, feat_dim, dtype=torch.float32)
    class_count = torch.zeros(c, dtype=torch.float32)
    valid_y = (y >= 0) & (y < c)
    for cls in torch.unique(y[valid_y]):
        m = valid_y & (y == cls)
        if bool(m.any()):
            class_sum[int(cls.item())] = z[m].sum(dim=0)
            class_count[int(cls.item())] = float(m.sum().item())
    out = {"class_sum": class_sum, "class_count": class_count}

    if domains is not None:
        d = domains.detach().view(-1).long().cpu()
        if int(d.numel()) == int(z.size(0)):
            nd = max(1, int(num_domains))
            domain_sum = torch.zeros(nd, feat_dim, dtype=torch.float32)
            domain_count = torch.zeros(nd, dtype=torch.float32)
            valid_d = (d >= 0) & (d < nd)
            for dom in torch.unique(d[valid_d]):
                m = valid_d & (d == dom)
                if bool(m.any()):
                    domain_sum[int(dom.item())] = z[m].sum(dim=0)
                    domain_count[int(dom.item())] = float(m.sum().item())
            out["domain_sum"] = domain_sum
            out["domain_count"] = domain_count
    return out


def _finalize_fed_proto_stats(accum: Optional[Mapping[str, torch.Tensor]]) -> Optional[Dict[str, torch.Tensor]]:
    if not accum or "class_sum" not in accum or "class_count" not in accum:
        return None
    class_sum = accum["class_sum"].float()
    class_count = accum["class_count"].float()
    class_proto = class_sum / class_count.clamp_min(1.0).view(-1, 1)
    class_proto = F.normalize(class_proto, dim=1, eps=1e-6)
    out = {"class_proto": class_proto.cpu(), "class_count": class_count.cpu()}
    if "domain_sum" in accum and "domain_count" in accum:
        domain_sum = accum["domain_sum"].float()
        domain_count = accum["domain_count"].float()
        domain_proto = domain_sum / domain_count.clamp_min(1.0).view(-1, 1)
        out["domain_proto"] = F.normalize(domain_proto, dim=1, eps=1e-6).cpu()
        out["domain_count"] = domain_count.cpu()
    return out


def _blend_fed_proto_stats(old: Optional[Mapping[str, torch.Tensor]], new: Optional[Mapping[str, torch.Tensor]], momentum: float):
    if not new:
        return old
    if not old or float(momentum) <= 0.0:
        return {k: v.detach().cpu().clone() for k, v in new.items() if torch.is_tensor(v)}
    m = max(0.0, min(0.9999, float(momentum)))
    blended: Dict[str, torch.Tensor] = {}
    for key, value in new.items():
        if not torch.is_tensor(value):
            continue
        cur = value.detach().cpu().clone()
        prev = old.get(key) if isinstance(old, Mapping) else None
        if torch.is_tensor(prev) and prev.shape == cur.shape and key.endswith("_proto"):
            cur = F.normalize(prev.float() * m + cur.float() * (1.0 - m), dim=1, eps=1e-6)
        blended[key] = cur
    return blended


def _fed_proto_pull_loss(z_id: Optional[torch.Tensor], labels: torch.Tensor, global_proto: Optional[Mapping[str, torch.Tensor]], min_count: int):
    if z_id is None or not torch.is_tensor(z_id) or not global_proto or "class_proto" not in global_proto:
        if torch.is_tensor(z_id):
            return z_id.new_tensor(0.0), float("nan")
        return torch.tensor(0.0), float("nan")
    proto = global_proto["class_proto"].to(device=z_id.device, dtype=z_id.dtype)
    counts = global_proto.get("class_count", torch.ones(proto.size(0))).to(device=z_id.device)
    y = labels.view(-1).long()
    valid = (y >= 0) & (y < proto.size(0)) & (counts[y.clamp(0, proto.size(0) - 1)] >= max(1, int(min_count)))
    if not bool(valid.any()):
        return z_id.new_tensor(0.0), float("nan")
    z = _safe_l2_normalize(z_id[valid], dim=1)
    target = proto[y[valid]]
    cos = (z * target).sum(dim=1).clamp(-1.0, 1.0)
    return (1.0 - cos).mean(), float(cos.mean().detach().item())


def _fed_proto_summary(proto: Optional[Mapping[str, torch.Tensor]]) -> Dict[str, Any]:
    if not proto:
        return {"enabled": False, "class_count_nonzero": 0, "domain_count_nonzero": 0}
    class_count = proto.get("class_count")
    domain_count = proto.get("domain_count")
    return {
        "enabled": True,
        "class_count_nonzero": int((class_count > 0).sum().item()) if torch.is_tensor(class_count) else 0,
        "domain_count_nonzero": int((domain_count > 0).sum().item()) if torch.is_tensor(domain_count) else 0,
    }


def _cfg_bool(cfg, name: str, default: bool = False) -> bool:
    return bool(getattr(cfg, name, default))


def _jsonable(value: Any) -> Any:
    if torch.is_tensor(value):
        value = value.detach().cpu()
        if value.numel() == 1:
            return _jsonable(value.item())
        if value.numel() <= 2048:
            return _jsonable(value.tolist())
        return {"type": "tensor", "shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, set):
        return [_jsonable(v) for v in sorted(value, key=str)]
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value) if math.isfinite(float(value)) else None
    try:
        import numpy as np  # type: ignore

        if isinstance(value, np.generic):
            return _jsonable(value.item())
        if isinstance(value, np.ndarray):
            return _jsonable(torch.as_tensor(value))
    except Exception:
        pass
    if hasattr(value, "__dict__"):
        return _jsonable(vars(value))
    return str(value)


def _cfg_snapshot(cfg) -> Dict[str, Any]:
    if cfg is None:
        return {}
    if isinstance(cfg, Mapping):
        items = cfg.items()
    elif hasattr(cfg, "__dict__"):
        items = vars(cfg).items()
    else:
        return {"value": _jsonable(cfg)}
    return {str(k): _jsonable(v) for k, v in sorted(items, key=lambda kv: str(kv[0]))}


def _safe_len(value: Any) -> Optional[int]:
    try:
        return int(len(value))
    except Exception:
        return None


def _model_parameter_summary(model: nn.Module) -> Dict[str, Any]:
    total = int(sum(p.numel() for p in model.parameters()))
    trainable = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    return {
        "class": model.__class__.__name__,
        "parameters_total": total,
        "parameters_trainable": trainable,
        "state_dict_keys": int(len(model.state_dict())),
    }


def _style_bank_enabled(cfg) -> bool:
    return bool(getattr(cfg, "use_fed_style_bank", False)) or bool(getattr(cfg, "use_fl_style_bank_stats", False))


def _proto_evidence_enabled(cfg) -> bool:
    return bool(getattr(cfg, "use_proto_evidence_bank", True))


def _logit_anchor_enabled(cfg) -> bool:
    return bool(getattr(cfg, "use_logit_anchors", False))


def _activation_token_enabled(cfg) -> bool:
    route = str(getattr(cfg, "activation_token_route", "none") or "none").lower()
    return route not in {"", "none", "off"}


def _style_domain_entropy(d_style: Optional[torch.Tensor]) -> float:
    if d_style is None or d_style.numel() == 0:
        return float("nan")
    vals = d_style.view(-1).long()
    vals = vals[vals >= 0]
    if vals.numel() == 0:
        return float("nan")
    counts = torch.bincount(vals.cpu()).float()
    probs = counts[counts > 0] / counts.sum().clamp_min(1.0)
    return float((-(probs * probs.log()).sum()).item())


def _tensor_entropy(prob: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    p = prob.float().clamp_min(float(eps))
    return -(p * p.log()).sum(dim=1)


def _collect_proto_evidence(
    z_id: Optional[torch.Tensor],
    logits: torch.Tensor,
    labels: torch.Tensor,
    domains: Optional[torch.Tensor],
    *,
    client_id: str,
) -> list[ProtoEvidence]:
    if z_id is None or not torch.is_tensor(z_id) or z_id.dim() != 2 or z_id.size(0) == 0:
        return []
    if int(labels.numel()) != int(z_id.size(0)) or int(logits.size(0)) != int(z_id.size(0)):
        return []
    z = _safe_l2_normalize(z_id.detach(), dim=1).cpu()
    y = labels.detach().view(-1).long().cpu()
    d = domains.detach().view(-1).long().cpu() if torch.is_tensor(domains) and int(domains.numel()) == int(z.size(0)) else None
    prob = F.softmax(logits.detach().float().cpu(), dim=1)
    top2 = torch.topk(prob, k=min(2, prob.size(1)), dim=1).values
    margins = top2[:, 0] - (top2[:, 1] if top2.size(1) > 1 else 0.0)
    ent = _tensor_entropy(prob)
    items: list[ProtoEvidence] = []
    for cls in torch.unique(y[(y >= 0) & (y < prob.size(1))]):
        m = y == cls
        if not bool(m.any()):
            continue
        z_cls = z[m]
        proto = z_cls.mean(dim=0)
        if z_cls.size(0) > 1:
            intra_var = float(((z_cls - proto.view(1, -1)) ** 2).mean().item())
        else:
            intra_var = 0.0
        style_id = None
        if d is not None:
            vals = d[m]
            vals = vals[vals >= 0]
            if vals.numel() > 0:
                style_id = int(torch.mode(vals).values.item())
        items.append(
            ProtoEvidence(
                class_id=int(cls.item()),
                prototype=proto,
                count=int(m.sum().item()),
                margin=float(margins[m].mean().item()),
                entropy=float(ent[m].mean().item()),
                intra_var=intra_var,
                client_drift=0.0,
                clean_sat_kl=0.0,
                client_id=str(client_id),
                style_id=style_id,
                mode="style" if style_id is not None and style_id > 0 else "clean",
            )
        )
    return items


class FederatedTrainer:
    """Single-process federated simulator for FedAvg/FedProx."""

    def __init__(
        self,
        model: nn.Module,
        train_dataset,
        val_loader,
        named_test_loaders: Mapping[str, Any],
        cfg,
        *,
        device,
        criterion: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        evaluate_loader_fn: Optional[Callable[..., Dict[str, float]]] = None,
        evaluate_named_loaders_fn: Optional[Callable[..., Dict[str, Dict[str, float]]]] = None,
        domain_label_map: Optional[Mapping[int, int]] = None,
        named_test_meta: Optional[Mapping[str, Any]] = None,
        split_info: Optional[Mapping[str, Any]] = None,
        augment_fn: Optional[Callable[..., torch.Tensor]] = None,
        sat_transform_fn: Optional[Callable[..., torch.Tensor]] = None,
        extra_eval_fn: Optional[Callable[..., Dict[str, Any]]] = None,
        style_batch_fn: Optional[Callable[..., Optional[StyleDomainBatch]]] = None,
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.val_loader = val_loader
        self.named_test_loaders = dict(named_test_loaders or {})
        self.cfg = cfg
        self.device = device
        self.criterion = criterion or nn.CrossEntropyLoss()
        self.evaluate_loader_fn = evaluate_loader_fn
        self.evaluate_named_loaders_fn = evaluate_named_loaders_fn
        self.domain_label_map = dict(domain_label_map or {})
        self.named_test_meta = dict(named_test_meta or {})
        self.split_info = dict(split_info or {})
        self.augment_fn = augment_fn
        self.sat_transform_fn = sat_transform_fn
        self.extra_eval_fn = extra_eval_fn
        self.style_batch_fn = style_batch_fn

        self.train_mode = str(getattr(cfg, "train_mode", "fedavg")).lower()
        allowed_train_modes = {
            "fedavg",
            "fedprox",
            "fedcvs_vmb",
            "split_bex02",
            "fedriei",
            "fedfa",
            "fucl",
            "rafl",
        }
        if self.train_mode not in allowed_train_modes:
            raise ValueError(
                f"Unsupported federated train_mode={self.train_mode!r}; "
                "use fedavg, fedprox, fedcvs_vmb, split_bex02, fedriei, fedfa, fucl, or rafl."
            )
        self.split_bex02_enabled = self.train_mode == "split_bex02"
        self.vmb_enabled = self.train_mode in {"fedcvs_vmb", "split_bex02"}
        self.fed_cgrl = FedCGRLController.from_config(cfg)

        self.client_splits = build_client_splits(
            train_dataset,
            getattr(cfg, "fl_client_key", "receiver_day"),
            min_samples_per_client=int(getattr(cfg, "fl_min_samples_per_client", 1)),
            drop_small=bool(getattr(cfg, "fl_drop_small_clients", False)),
            verbose=bool(getattr(cfg, "fl_verbose_clients", True)),
        )
        self.client_loaders = build_client_loaders(
            train_dataset,
            self.client_splits,
            batch_size=int(getattr(cfg, "batch_size", 128)),
            num_workers=int(getattr(cfg, "fl_num_workers", 0)),
            sampler_cfg={
                "shuffle": True,
                "drop_last": False,
                "pin_memory": (getattr(device, "type", "") == "cuda"),
            },
        )
        self.client_num_samples = {cid: len(indices) for cid, indices in self.client_splits.items()}
        self.vmb_client_domains = {cid: client_domain_from_id(str(cid)) for cid in self.client_splits}
        self.global_state = OrderedDict((k, v.detach().cpu().clone()) for k, v in self.model.state_dict().items())
        self.global_proto_stats: Optional[Dict[str, torch.Tensor]] = None
        self.global_coral_bank = (
            FedCVSCoralStatsBank(
                num_classes=int(getattr(cfg, "num_classes", 1)),
                momentum=float(_coral_config_value(cfg, "fl_coral_momentum", "fed_coral_momentum", 0.95)),
                mode=str(_coral_config_value(cfg, "fl_coral_cov_mode", "fed_coral_mode", "diag") or "diag"),
            )
            if _fed_coral_enabled(cfg)
            else None
        )
        self.fed_fishr_mode = _fed_fishr_mode(cfg)
        self.fed_fishr_bank = (
            FedFishrBank(
                min_clients=int(getattr(cfg, "fed_fishr_min_clients", 2)),
                min_count=int(getattr(cfg, "fed_fishr_min_count", 2)),
                momentum=float(getattr(cfg, "fed_fishr_momentum", 0.0) or 0.0),
            )
            if _fed_fishr_enabled(cfg) and self.fed_fishr_mode != "off"
            else None
        )
        self.global_fed_fishr_summary: Dict[str, Any] = {
            "enabled": bool(self.fed_fishr_bank is not None),
            "active": False,
            "inactive_reason": "not_updated" if self.fed_fishr_bank is not None else "disabled",
        }
        if self.global_coral_bank is not None:
            coral_feature = _coral_feature_name(cfg)
            zdom_weight = float(getattr(cfg, "lambda_fl_coral_zdom_global", 0.0) or 0.0)
            zid_weight = _coral_weight(cfg, "lambda_fl_coral_zid_global", "lambda_fed_coral")
            if zdom_weight > 0.0 and coral_feature != "z_dom":
                raise ValueError(
                    "--lambda_fl_coral_zdom_global is a negative-control setting and requires "
                    "--fl_coral_feature z_dom so the server bank is built from z_dom statistics."
                )
            if zdom_weight > 0.0 and zid_weight > 0.0:
                raise ValueError(
                    "Do not combine z_id global CORAL and z_dom negative-control CORAL in one run; "
                    "use separate runs with --fl_coral_feature z_id or z_dom."
                )
        self.groupdro_state = SmoothGroupDROState(momentum=float(getattr(cfg, "groupdro_momentum", 0.95)))
        self.vmb_proto_bank = (
            FedCVSVMBPrototypeBank(
                num_classes=int(getattr(cfg, "num_classes", 1)),
                ema_alpha=float(getattr(cfg, "fl_vmb_prototype_ema", 0.95)),
                clip_norm=float(getattr(cfg, "fl_vmb_prototype_clip_norm", 1.0)),
            )
            if self.vmb_enabled
            else None
        )
        self.vmb_server_optimizer_state: Dict[str, torch.Tensor] = {}
        self.logit_anchor_bank = (
            LogitAnchorBank(
                num_classes=int(getattr(cfg, "num_classes", 1)),
                ema_alpha=float(getattr(cfg, "kd_anchor_ema", 0.9)),
            )
            if _logit_anchor_enabled(cfg)
            else None
        )
        token_route = str(getattr(cfg, "activation_token_route", "none") or "none").lower()
        self.activation_token_codec = (
            ActivationTokenCodec(
                route=token_route,
                quant_bits=int(getattr(cfg, "token_quant_bits", 8)),
                sketch_dim=int(getattr(cfg, "token_sketch_dim", 64)),
                rank=int(getattr(cfg, "token_rank", 8)),
                seed=int(getattr(cfg, "seed", 0) or 0),
            )
            if (_activation_token_enabled(cfg) or self.split_bex02_enabled)
            else None
        )
        self.proto_evidence_bank = (
            ProtoEvidenceBank(max_per_class=int(getattr(cfg, "proto_max_per_class", 8)))
            if _proto_evidence_enabled(cfg)
            else None
        )
        self.style_extractor = RFStyleExtractor(
            sample_rate_hz=float(getattr(cfg, "sample_rate_hz", 0.0) or 0.0)
        ) if _style_bank_enabled(cfg) else None
        self.style_bank = (
            FederatedStyleBank(
                momentum=float(getattr(cfg, "fl_style_bank_momentum", 0.5)),
                max_centroids=int(getattr(cfg, "fl_style_bank_max_centroids", 64)),
                merge_radius=float(getattr(cfg, "fl_style_bank_merge_radius", 0.0)),
            )
            if self.style_extractor is not None
            else None
        )
        self.global_style_summary: Dict[str, Any] = {"enabled": bool(self.style_bank is not None)}
        self.style_transform = (
            StyleConditionedReceiverDG(
                max_gain_delta=float(getattr(cfg, "fl_style_phys_max_gain_delta", 0.05)),
                max_noise_std=float(getattr(cfg, "fl_style_phys_max_noise_std", 0.01)),
                sample_rate_hz=float(getattr(cfg, "sample_rate_hz", 25_000_000.0) or 25_000_000.0),
                style_jitter_scale=float(getattr(cfg, "fl_style_phys_jitter_scale", 0.25)),
                max_cfo_hz=float(getattr(cfg, "fl_style_phys_max_cfo_hz", 5000.0)),
                max_sro_ppm=float(getattr(cfg, "fl_style_phys_max_sro_ppm", 25.0)),
                max_iq_gain_db=float(getattr(cfg, "fl_style_phys_max_iq_gain_db", 0.5)),
                max_iq_phase_deg=float(getattr(cfg, "fl_style_phys_max_iq_phase_deg", 0.5)),
                max_phase_noise_std=float(getattr(cfg, "fl_style_phys_max_phase_noise_std", 0.0005)),
                min_awgn_snr_db=float(getattr(cfg, "fl_style_phys_min_awgn_snr_db", 20.0)),
                p_lowpass=float(getattr(cfg, "fl_style_phys_p_lowpass", 0.2)),
                p_multipath=float(getattr(cfg, "fl_style_phys_p_multipath", 0.2)),
                max_multipath_taps=int(getattr(cfg, "fl_style_phys_max_multipath_taps", 3)),
            )
            if self.style_bank is not None
            else None
        )
        self.virtual_domain_sampler = VirtualDomainSampler(clean_style_id=0)

        self.output_dir = str(getattr(cfg, "output_dir", "") or os.path.dirname(str(getattr(cfg, "latest_save_path", ""))) or ".")
        self.log_dir = str(getattr(cfg, "log_dir", "") or self.output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        self.logs_jsonl = os.path.join(self.log_dir, "logs.jsonl")
        self.metrics_csv = os.path.join(self.log_dir, "metrics.csv")
        self.config_json = os.path.join(self.log_dir, "federated_config.json")
        self.summary_json = os.path.join(self.log_dir, "summary.json")
        self.best_checkpoint = os.path.join(self.output_dir, "best_checkpoint.pt")
        self.last_checkpoint = os.path.join(self.output_dir, "last_checkpoint.pt")
        self._config_written = False
        self.last_feature_probe_summary: Dict[str, Any] = {"enabled": False, "num_samples": 0}

    def _update_global_proto_stats(self, client_proto_stats: Mapping[str, Optional[Mapping[str, torch.Tensor]]]) -> Dict[str, Any]:
        if not _fed_proto_enabled(self.cfg):
            return _fed_proto_summary(None)
        accum = None
        for stats in client_proto_stats.values():
            accum = _merge_fed_proto_stats(accum, stats)
        finalized = _finalize_fed_proto_stats(accum)
        self.global_proto_stats = _blend_fed_proto_stats(
            self.global_proto_stats,
            finalized,
            float(getattr(self.cfg, "fed_proto_momentum", 0.0)),
        )
        return _fed_proto_summary(self.global_proto_stats)

    def _update_global_coral_stats(self, client_coral_stats: Mapping[str, Any]) -> Dict[str, Any]:
        if getattr(self, "global_coral_bank", None) is None:
            return {"enabled": False, "class_count_nonzero": 0, "payload_bytes": 0}
        merged = merge_coral_stats(client_coral_stats.values())
        return self.global_coral_bank.update(merged)

    def _base_aggregation_weights(self, selected, *, agg_weight: Optional[str] = None, domain_ids: Optional[Mapping[str, str]] = None) -> OrderedDict[str, float]:
        selected = [str(cid) for cid in selected]
        mode = str(agg_weight or getattr(self.cfg, "fl_agg_weight", "num_samples") or "num_samples")
        if not selected:
            return OrderedDict()
        if mode == "uniform":
            return OrderedDict((cid, 1.0 / float(len(selected))) for cid in selected)
        if mode in {"domain_uniform", "domain_balanced"}:
            domains = domain_ids or {cid: client_domain_from_id(cid) for cid in selected}
            return OrderedDict((cid, float(w)) for cid, w in domain_balanced_weights(selected, domains).items())
        total = float(sum(max(0, int(self.client_num_samples[cid])) for cid in selected))
        if total <= 0:
            return OrderedDict((cid, 1.0 / float(len(selected))) for cid in selected)
        return OrderedDict((cid, max(0, int(self.client_num_samples[cid])) / total) for cid in selected)

    def _update_global_fed_fishr_stats(
        self,
        client_fed_fishr_stats: Mapping[str, Any],
        *,
        base_weights: Mapping[str, float],
        round_idx: int,
    ) -> OrderedDict[str, float]:
        base = OrderedDict((str(cid), float(w)) for cid, w in (base_weights or {}).items())
        if self.fed_fishr_bank is None:
            self.global_fed_fishr_summary = {"enabled": False, "active": False, "inactive_reason": "disabled"}
            return base
        raw_summary = self.fed_fishr_bank.update(client_fed_fishr_stats)
        mode = self.fed_fishr_mode
        start_round = int(getattr(self.cfg, "fed_fishr_start_round", 1) or 1)
        reweight_active = False
        reweighted = base
        reweight_summary: Dict[str, Any] = {
            "active": False,
            "inactive_reason": "before_start_round" if int(round_idx) < start_round else "mode_without_reweight",
            "weights": dict(base),
            "base_weights": dict(base),
            "max_delta": 0.0,
        }
        if int(round_idx) >= start_round and mode in {"reweight", "both"} and bool(raw_summary.get("active", False)):
            reweighted, reweight_summary = fed_fishr_reweight(
                base,
                raw_summary.get("client_mismatch", {}),
                alpha=float(getattr(self.cfg, "lambda_fed_fishr", 0.0) or 0.0),
                floor=float(getattr(self.cfg, "fed_fishr_reweight_floor", 0.0) or 0.0),
                cap=float(getattr(self.cfg, "fed_fishr_reweight_cap", 1.0) or 1.0),
            )
            reweight_active = bool(reweight_summary.get("active", False))
        log_summary = fed_fishr_log_summary(raw_summary)
        log_summary.update(
            {
                "mode": mode,
                "start_round": int(start_round),
                "reweight_active": bool(reweight_active),
                "weights": dict(reweight_summary.get("weights", reweighted)),
                "base_weights": dict(reweight_summary.get("base_weights", base)),
                "weight_min": float(reweight_summary.get("weight_min", float("nan"))),
                "weight_max": float(reweight_summary.get("weight_max", float("nan"))),
                "weight_max_delta": float(reweight_summary.get("max_delta", 0.0) or 0.0),
                "reweight_inactive_reason": str(reweight_summary.get("inactive_reason", "")),
            }
        )
        self.global_fed_fishr_summary = log_summary
        return reweighted

    def _update_logit_anchor_bank(self, client_anchor_stats: Mapping[str, Any]) -> Dict[str, Any]:
        if self.logit_anchor_bank is None:
            return {"enabled": False, "anchor_count_nonzero": 0, "payload_bytes": 0}
        merged = merge_logit_anchor_stats(client_anchor_stats.values())
        return self.logit_anchor_bank.update(merged)

    def _style_code_dim(self) -> int:
        return max(0, int(getattr(self.cfg, "fl_style_code_dim", 0) or 0))

    def _attach_style_code(self, packet: StylePacket) -> StylePacket:
        dim = self._style_code_dim()
        if dim <= 0:
            return packet
        return StylePacket(
            client_id=packet.client_id,
            round_idx=packet.round_idx,
            count=packet.count,
            stats=packet.stats,
            style_id=packet.style_id,
            metadata=packet.metadata,
            style_code=style_code_from_stats(packet.stats, dim=dim),
        )

    def _extract_style_packet(
        self,
        client_id: str,
        round_idx: int,
        x: torch.Tensor,
        y: torch.Tensor,
        d_raw: Optional[torch.Tensor],
    ) -> Optional[StylePacket]:
        if self.style_extractor is None:
            return None
        with torch.no_grad():
            style_meta: Dict[str, Any] = {}
            mapped_target = self._mode_domain_label(d_raw)
            if mapped_target is not None:
                style_meta["target_domain_label"] = int(mapped_target)
                style_meta["source_domain_label"] = int(mapped_target)
            if d_raw is not None:
                raw_vals = d_raw.detach().view(-1).long()
                raw_vals = raw_vals[raw_vals >= 0]
                if raw_vals.numel() > 0:
                    style_meta["raw_target_domain_label"] = int(torch.mode(raw_vals.cpu()).values.item())
            return self._attach_style_code(
                self.style_extractor.extract(
                    x,
                    y,
                    client_id=str(client_id),
                    round_idx=int(round_idx),
                    d_raw=d_raw,
                    metadata=style_meta,
                )
            )

    @staticmethod
    def _style_packet_payload_bytes(style_packets: Optional[list[StylePacket]]) -> int:
        return int(sum(int(packet.size_bytes()) for packet in (style_packets or [])))

    def _split_feature_tensor(self, out: Mapping[str, Any]) -> Optional[torch.Tensor]:
        key = str(getattr(self.cfg, "split_layer", "z_id") or "z_id").lower()
        if key in {"concat", "z_id_z_dom", "zt_zr"}:
            z_id = out.get("z_id")
            z_dom = out.get("z_dom")
            if torch.is_tensor(z_id) and torch.is_tensor(z_dom) and int(z_id.size(0)) == int(z_dom.size(0)):
                return torch.cat([z_id, z_dom], dim=1)
            return z_id if torch.is_tensor(z_id) else None
        value = out.get(key)
        if torch.is_tensor(value):
            return value
        if key in {"zt", "z_t"}:
            return out.get("z_id") if torch.is_tensor(out.get("z_id")) else None
        if key in {"zr", "z_r"}:
            return out.get("z_dom") if torch.is_tensor(out.get("z_dom")) else None
        return out.get("z_id") if torch.is_tensor(out.get("z_id")) else None

    def _feature_probe_enabled(self, round_idx: int, batch_idx: int) -> bool:
        every = int(getattr(self.cfg, "fl_probe_every", 0) or 0)
        export = str(getattr(self.cfg, "feature_probe_export", "") or "").strip()
        if every <= 0 or export == "":
            return False
        return int(round_idx) % every == 0 and int(batch_idx) == 0

    def _feature_probe_item(
        self,
        client_id: str,
        out: Mapping[str, Any],
        y: torch.Tensor,
        d_probe: Optional[torch.Tensor],
        round_idx: int,
        batch_idx: int,
    ) -> Optional[Dict[str, Any]]:
        if not self._feature_probe_enabled(round_idx, batch_idx):
            return None
        z_t = out.get("z_id")
        z_r = out.get("z_dom")
        if not torch.is_tensor(z_t) or not torch.is_tensor(z_r):
            return None
        n = min(int(z_t.size(0)), int(z_r.size(0)), int(y.numel()))
        max_samples = int(getattr(self.cfg, "probe_max_samples", 0) or 0)
        if max_samples > 0:
            n = min(n, max_samples)
        if n <= 0:
            return None
        tx = y.detach().view(-1)[:n].cpu().long()
        if torch.is_tensor(d_probe):
            rx = d_probe.detach().view(-1)[:n].cpu().long()
            if int(rx.numel()) < n:
                rx = torch.full((n,), -1, dtype=torch.long)
        else:
            rx = torch.full((n,), -1, dtype=torch.long)
        return {
            "client_id": str(client_id),
            "round": int(round_idx),
            "batch_idx": int(batch_idx),
            "z_t": z_t.detach()[:n].cpu().float(),
            "z_r": z_r.detach()[:n].cpu().float(),
            "tx": tx,
            "rx": rx,
        }

    def _feature_probe_export_path(self, round_idx: int) -> str:
        raw = str(getattr(self.cfg, "feature_probe_export", "") or "").strip()
        if not raw:
            return ""
        if "{round}" in raw:
            rel = raw.format(round=int(round_idx), round_idx=int(round_idx))
        else:
            root, ext = os.path.splitext(raw)
            if ext:
                rel = f"{root}_r{int(round_idx):03d}{ext}"
            else:
                rel = os.path.join(raw, f"features_r{int(round_idx):03d}.pt")
        return rel if os.path.isabs(rel) else os.path.join(self.log_dir, rel)

    def _write_feature_probe_export(self, round_idx: int, client_items: Mapping[str, list[Mapping[str, Any]]]) -> Dict[str, Any]:
        export_path = self._feature_probe_export_path(round_idx)
        if not export_path:
            self.last_feature_probe_summary = {"enabled": False, "num_samples": 0}
            return dict(self.last_feature_probe_summary)
        items = []
        for entries in client_items.values():
            items.extend(entries or [])
        if not items:
            self.last_feature_probe_summary = {"enabled": True, "num_samples": 0, "path": export_path, "wrote": False}
            return dict(self.last_feature_probe_summary)
        os.makedirs(os.path.dirname(export_path) or ".", exist_ok=True)
        payload = {
            "round": int(round_idx),
            "client_ids": [str(item.get("client_id", "")) for item in items],
            "batch_indices": [int(item.get("batch_idx", 0)) for item in items],
            "z_t": torch.cat([item["z_t"] for item in items], dim=0),
            "z_r": torch.cat([item["z_r"] for item in items], dim=0),
            "tx": torch.cat([item["tx"] for item in items], dim=0),
            "rx": torch.cat([item["rx"] for item in items], dim=0),
            "semantics": {
                "z_t": "transmitter_identity_feature_z_id",
                "z_r": "receiver_domain_feature_z_dom",
                "rx": "mapped_receiver_or_constructed_style_domain_label",
            },
        }
        torch.save(payload, export_path)
        summary = {
            "enabled": True,
            "wrote": True,
            "path": export_path,
            "num_clients": int(len(client_items)),
            "num_samples": int(payload["tx"].numel()),
            "z_t_dim": int(payload["z_t"].view(int(payload["tx"].numel()), -1).size(1)),
            "z_r_dim": int(payload["z_r"].view(int(payload["tx"].numel()), -1).size(1)),
        }
        self.last_feature_probe_summary = summary
        return dict(summary)

    def _vmb_stage(self, round_idx: int) -> str:
        return vmb_stage_for_round(
            str(getattr(self.cfg, "fl_vmb_stage", "stage2")),
            int(round_idx),
            int(getattr(self.cfg, "fl_vmb_pretrain_rounds", 0)),
        )

    def _vmb_stage1_objective(self) -> str:
        return str(getattr(self.cfg, "fl_vmb_stage1_objective", "ce") or "ce").lower()

    def _vmb_stage1_ce_only(self, round_idx: int) -> bool:
        return (
            self.vmb_enabled
            and self._vmb_stage(round_idx) == "stage1"
            and self._vmb_stage1_objective() == "ce"
            and not bool(getattr(self.cfg, "fl_vmb_stage1_use_aux_losses", False))
        )

    def _vmb_domain_id(self, client_id: str) -> str:
        return str(self.vmb_client_domains.get(str(client_id), client_domain_from_id(str(client_id))))

    def _vmb_set_trainability(self, stage: str) -> list[str]:
        freeze_rx = bool(getattr(self.cfg, "fl_vmb_freeze_rx_stage2", True)) and str(stage) == "stage2"
        domain_scope_stage1 = (
            str(stage) == "stage1"
            and self._vmb_stage1_objective() == "domain_unsup_pretrain"
            and str(getattr(self.cfg, "fl_domain_pretrain_train_scope", "all") or "all").lower() == "domain"
        )
        active: list[str] = []
        for name, param in self.model.named_parameters():
            if domain_scope_stage1:
                enabled = name.startswith(("dom_backbone.", "dom_head.", "dom_enhancer."))
            else:
                enabled = not (freeze_rx and name.startswith(("dom_backbone.", "dom_head.", "dom_enhancer.")))
            param.requires_grad_(enabled)
            if enabled:
                active.append(str(name))
        return active

    def _vmb_restore_trainability(self) -> None:
        for param in self.model.parameters():
            param.requires_grad_(True)

    def _vmb_build_balanced_batch(self, client_id: str, round_idx: int, batch_idx: int):
        if not bool(getattr(self.cfg, "fl_vmb_transmitter_balanced_batch", True)):
            return None
        indices = self.client_splits.get(str(client_id), [])
        selected = select_transmitter_balanced_indices(
            indices,
            lambda idx: get_sample_metadata(self.train_dataset, int(idx)),
            batch_size=int(getattr(self.cfg, "batch_size", 128)),
            seed=int(getattr(self.cfg, "seed", 0) or 0),
            round_idx=int(round_idx),
            batch_idx=int(batch_idx),
        )
        if not selected:
            return None
        xs = []
        ys = []
        ds = []
        for idx in selected:
            try:
                sample = self.train_dataset[int(idx)]
                x, y, d = self._sample_to_xyd_values(sample)
                if x is None or y is None:
                    continue
                xs.append(torch.as_tensor(x).detach().clone())
                ys.append(int(torch.as_tensor(y).view(-1)[0].item()))
                if d is None:
                    match = re.match(r"^rx(\d+)", str(client_id))
                    ds.append(int(match.group(1)) if match else -1)
                else:
                    ds.append(int(torch.as_tensor(d).view(-1)[0].item()))
            except Exception:
                continue
        if not xs:
            return None
        return (
            torch.stack(xs, dim=0).to(self.device),
            torch.as_tensor(ys, device=self.device).long(),
            torch.as_tensor(ds, device=self.device).long(),
        )

    def _proto_evidence_summary(self) -> Dict[str, Any]:
        if self.proto_evidence_bank is None:
            return {"enabled": False, "num_classes": 0, "num_prototypes": 0}
        return {"enabled": True, **self.proto_evidence_bank.summary()}

    def _update_proto_evidence_bank(self, client_items: Mapping[str, list[ProtoEvidence]]) -> Dict[str, Any]:
        if self.proto_evidence_bank is None:
            return {"enabled": False, "num_classes": 0, "num_prototypes": 0}
        self.proto_evidence_bank.age_one_round()
        items = []
        for values in client_items.values():
            items.extend(values or [])
        if items:
            self.proto_evidence_bank.update(items)
        summary = self.proto_evidence_bank.summary()
        return {"enabled": True, "last_added": len(items), **summary}

    def _proto_posterior(self, z_id: torch.Tensor, num_classes: int) -> tuple[torch.Tensor, float]:
        if self.proto_evidence_bank is None:
            return torch.full((z_id.size(0), int(num_classes)), 1.0 / max(1, int(num_classes)), device=z_id.device), 0.0
        z = _safe_l2_normalize(z_id.detach(), dim=1)
        top_m = max(1, int(getattr(self.cfg, "proto_top_m", 4)))
        temp = max(1e-4, float(getattr(self.cfg, "proto_temperature", 0.1)))
        scores = z.new_full((z.size(0), int(num_classes)), -1e9)
        reliabilities = []
        for cls in range(int(num_classes)):
            entries = sorted(
                self.proto_evidence_bank.get_class(cls),
                key=lambda item: (float(item.reliability), int(item.count), -int(item.age)),
                reverse=True,
            )[:top_m]
            if not entries:
                continue
            proto = torch.stack([item.prototype.to(device=z.device, dtype=z.dtype) for item in entries], dim=0)
            rel = torch.tensor([float(item.reliability) for item in entries], device=z.device, dtype=z.dtype).clamp_min(1e-6)
            sim = z @ _safe_l2_normalize(proto, dim=1).t()
            weighted = sim + rel.log().view(1, -1)
            scores[:, cls] = torch.logsumexp(weighted / temp, dim=1)
            reliabilities.extend(float(item.reliability) for item in entries)
        missing = scores.max(dim=1).values < -1e8
        posterior = torch.softmax(scores, dim=1)
        if bool(missing.any()):
            posterior[missing] = 1.0 / max(1, int(num_classes))
        reliability = float(sum(reliabilities) / max(1, len(reliabilities))) if reliabilities else 0.0
        return normalize_probabilities(posterior), reliability

    def _evaluate_proto_fusion(self) -> Dict[str, Any]:
        if self.proto_evidence_bank is None or not bool(getattr(self.cfg, "proto_fusion_eval", True)):
            return {}
        if not self.named_test_loaders:
            return {}
        self.model.eval()
        per_split: Dict[str, Any] = {}
        aggregate = {"total": 0, "base_correct": 0, "fused_correct": 0, "rescue": 0, "harm": 0, "net_gain": 0}
        max_batches = int(getattr(self.cfg, "eval_max_batches", 0))
        with torch.no_grad():
            for name, loader in self.named_test_loaders.items():
                split = {"total": 0, "base_correct": 0, "fused_correct": 0, "rescue": 0, "harm": 0, "net_gain": 0}
                for batch_idx, batch in enumerate(loader):
                    if max_batches > 0 and batch_idx >= max_batches:
                        break
                    x, y, d_raw = _batch_to_xyd(batch, self.device)
                    out = self._forward_outputs(x, y, d_raw)
                    logits = out.get("tx_logits", out.get("logits"))
                    z_id = out.get("z_id")
                    if not torch.is_tensor(logits) or not torch.is_tensor(z_id):
                        continue
                    p_base = F.softmax(logits.float(), dim=1)
                    p_proto, reliability = self._proto_posterior(z_id, int(logits.size(1)))
                    p_fused = conservative_probability_fusion(
                        p_base,
                        p_proto,
                        rho=float(getattr(self.cfg, "proto_rho_max", 0.05)),
                        max_rho=float(getattr(self.cfg, "proto_rho_max", 0.05)),
                        reliability=reliability,
                    )
                    report = harm_rescue_report(p_base, p_fused, y)
                    for key in split:
                        split[key] += int(report.get(key, 0))
                per_split[name] = dict(split)
                for key in aggregate:
                    aggregate[key] += int(split.get(key, 0))
        return {
            "aggregate": aggregate,
            "splits": per_split,
            "rho_max": float(getattr(self.cfg, "proto_rho_max", 0.05)),
            "top_m": int(getattr(self.cfg, "proto_top_m", 4)),
        }

    def _heavy_eval_interval(self) -> int:
        return max(0, int(getattr(self.cfg, "fl_test_eval_interval", 0) or 0))

    def _heavy_eval_last_n(self) -> int:
        return max(0, int(getattr(self.cfg, "fl_test_eval_last_n", 0) or 0))

    def _heavy_eval_final_offsets(self) -> tuple:
        raw = getattr(self.cfg, "fl_test_eval_final_offsets", "5,3,1")
        if raw is None:
            return tuple()
        if isinstance(raw, str):
            parts = raw.replace(";", ",").split(",")
        else:
            parts = list(raw)
        offsets = set()
        for part in parts:
            try:
                offset = int(str(part).strip())
            except (TypeError, ValueError):
                continue
            if offset > 0:
                offsets.add(offset)
        return tuple(sorted(offsets, reverse=True))

    def _heavy_eval_final_rounds(self) -> set:
        total_rounds = max(1, int(getattr(self.cfg, "fl_rounds", 1)))
        return {
            total_rounds - offset + 1
            for offset in self._heavy_eval_final_offsets()
            if 1 <= offset <= total_rounds
        }

    def _heavy_eval_runs_every_round(self) -> bool:
        return self._heavy_eval_interval() == 1

    def _should_run_heavy_eval(self, round_idx: int) -> bool:
        total_rounds = max(1, int(getattr(self.cfg, "fl_rounds", 1)))
        round_idx = int(round_idx)
        if round_idx in self._heavy_eval_final_rounds():
            return True
        last_n = self._heavy_eval_last_n()
        if last_n > 0 and round_idx > max(0, total_rounds - last_n):
            return True
        interval = self._heavy_eval_interval()
        if interval == 1:
            return True
        if round_idx == total_rounds:
            return True
        if interval > 1:
            return (round_idx % interval) == 0
        return False

    def _next_heavy_eval_round(self, round_idx: int) -> Optional[int]:
        total_rounds = max(1, int(getattr(self.cfg, "fl_rounds", 1)))
        for candidate in range(int(round_idx) + 1, total_rounds + 1):
            if self._should_run_heavy_eval(candidate):
                return candidate
        return None

    def _style_collab_enabled(self) -> bool:
        return bool(getattr(self.cfg, "use_style_collab_eval", False))

    def _style_packet_reliability(self, style: StylePacket) -> float:
        metadata = dict(getattr(style, "metadata", {}) or {})
        for key in ("reliability", "mean_reliability", "style_reliability"):
            if key in metadata:
                try:
                    value = float(metadata[key])
                    if math.isfinite(value):
                        return max(0.0, min(1.0, value))
                except (TypeError, ValueError):
                    pass
        count = max(0, int(getattr(style, "count", 0) or 0))
        count_rel = min(1.0, math.log1p(count) / math.log1p(32.0))
        age = 0
        try:
            age = max(0, int(metadata.get("age", 0)))
        except (TypeError, ValueError):
            age = 0
        age_rel = 1.0 / (1.0 + 0.15 * float(age))
        return max(0.0, min(1.0, count_rel * age_rel))

    def _style_collab_view_packets(self) -> tuple[StylePacket, ...]:
        if self.style_bank is None:
            return tuple()
        k = max(1, int(getattr(self.cfg, "style_collab_views", 2)))
        return self.style_bank.sample_remote_styles(exclude_client_id="__style_collab_eval__", k=k)

    @staticmethod
    def _finalize_fusion_counts(counts: Mapping[str, Any]) -> Dict[str, Any]:
        out = {k: int(counts.get(k, 0)) for k in ["total", "base_correct", "fused_correct", "rescue", "harm", "net_gain"]}
        total = max(1, int(out["total"]))
        out["base_tx_acc"] = 100.0 * float(out["base_correct"]) / float(total)
        out["fused_tx_acc"] = 100.0 * float(out["fused_correct"]) / float(total)
        return out

    def _evaluate_style_collab_fusion(self) -> Dict[str, Any]:
        if not self._style_collab_enabled():
            return {}
        if self.style_bank is None:
            return {"enabled": False, "reason": "stylebank_disabled"}
        if self.style_transform is None:
            return {"enabled": False, "reason": "style_transform_missing"}
        if not self.named_test_loaders:
            return {"enabled": False, "reason": "no_named_test_loaders"}
        styles = self._style_collab_view_packets()
        if not styles:
            return {"enabled": False, "reason": "no_style_packets"}

        self.model.eval()
        per_split: Dict[str, Any] = {}
        aggregate = {"total": 0, "base_correct": 0, "fused_correct": 0, "rescue": 0, "harm": 0, "net_gain": 0}
        max_batches = int(getattr(self.cfg, "eval_max_batches", 0))
        fusion_mode = str(getattr(self.cfg, "style_collab_fusion", "adaptive") or "adaptive").lower()
        base_weight = float(getattr(self.cfg, "style_collab_base_weight", 1.0))
        max_aux_weight = float(getattr(self.cfg, "style_collab_max_aux_weight", 1.0))
        style_reliabilities = torch.tensor(
            [self._style_packet_reliability(style) for style in styles],
            device=self.device,
            dtype=torch.float32,
        )
        with torch.no_grad():
            for name, loader in self.named_test_loaders.items():
                split = {"total": 0, "base_correct": 0, "fused_correct": 0, "rescue": 0, "harm": 0, "net_gain": 0}
                for batch_idx, batch in enumerate(loader):
                    if max_batches > 0 and batch_idx >= max_batches:
                        break
                    x, y, d_raw = _batch_to_xyd(batch, self.device)
                    out = self._forward_outputs(x, y, d_raw)
                    logits = out.get("tx_logits", out.get("logits"))
                    if not torch.is_tensor(logits):
                        continue
                    p_base = F.softmax(logits.float(), dim=1)
                    aux_probs = []
                    for style in styles:
                        x_style = _safe_iq_tensor(self.style_transform.transform(x, style))
                        out_style = self._forward_outputs(x_style, y, d_raw)
                        style_logits = out_style.get("tx_logits", out_style.get("logits"))
                        if torch.is_tensor(style_logits) and tuple(style_logits.shape) == tuple(logits.shape):
                            aux_probs.append(F.softmax(style_logits.float(), dim=1))
                    if not aux_probs:
                        continue
                    p_fused = collaborative_probability_fusion(
                        p_base,
                        torch.stack(aux_probs, dim=0),
                        mode=fusion_mode,
                        aux_reliabilities=style_reliabilities[: len(aux_probs)],
                        base_weight=base_weight,
                        max_aux_weight=max_aux_weight,
                    )
                    report = harm_rescue_report(p_base, p_fused, y)
                    for key in split:
                        split[key] += int(report.get(key, 0))
                finalized = self._finalize_fusion_counts(split)
                per_split[name] = finalized
                for key in aggregate:
                    aggregate[key] += int(split.get(key, 0))

        return {
            "enabled": True,
            "fusion": fusion_mode,
            "views": int(len(styles)),
            "base_weight": base_weight,
            "max_aux_weight": max_aux_weight,
            "style_reliability_mean": float(style_reliabilities.mean().detach().cpu().item()) if style_reliabilities.numel() else 0.0,
            "aggregate": self._finalize_fusion_counts(aggregate),
            "splits": per_split,
        }

    def _update_style_bank(self, client_style_packets: Mapping[str, list[StylePacket]]) -> Dict[str, Any]:
        if self.style_bank is None:
            self.global_style_summary = {"enabled": False}
            return dict(self.global_style_summary)
        packets = []
        for items in client_style_packets.values():
            packets.extend(items or [])
        update = self.style_bank.update(packets) if packets else self.style_bank.diagnostics()
        diag = self.style_bank.diagnostics()
        self.global_style_summary = {"enabled": True, **diag, **{f"last_{k}": v for k, v in update.items()}}
        return dict(self.global_style_summary)

    def _remap_domain_tensor(self, d_raw: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if d_raw is None:
            return None
        d_raw = d_raw.view(-1).long().to(self.device)
        if not self.domain_label_map:
            return d_raw
        mapped = torch.full_like(d_raw, -1)
        for raw, mapped_id in self.domain_label_map.items():
            mapped[d_raw == int(raw)] = int(mapped_id)
        return mapped

    def _num_domains(self, d: Optional[torch.Tensor]) -> int:
        if self.domain_label_map:
            vals = [int(v) for v in self.domain_label_map.values() if int(v) >= 0]
            return max(vals) + 1 if vals else 0
        if d is None or d.numel() == 0:
            return 0
        valid = d[d >= 0]
        return int(valid.max().item()) + 1 if valid.numel() else 0

    def _remap_domain_value(self, value: int) -> int:
        raw = int(value)
        if not self.domain_label_map:
            return raw
        return int(self.domain_label_map.get(raw, -1))

    def _mode_domain_label(self, d_raw: Optional[torch.Tensor]) -> Optional[int]:
        if d_raw is None:
            return None
        mapped = self._remap_domain_tensor(d_raw)
        if mapped is None:
            return None
        vals = mapped.detach().view(-1).long()
        vals = vals[vals >= 0]
        if vals.numel() == 0:
            return None
        return int(torch.mode(vals.cpu()).values.item())

    def _style_packet_target_domain(self, style: StylePacket) -> Optional[int]:
        meta = dict(style.metadata or {})
        for key in ("target_domain_label", "source_domain_label", "mapped_target_domain_label"):
            if key in meta:
                try:
                    label = int(meta[key])
                except (TypeError, ValueError):
                    continue
                return label if label >= 0 else None
        for key in ("raw_target_domain_label", "raw_domain_label"):
            if key in meta:
                try:
                    label = self._remap_domain_value(int(meta[key]))
                except (TypeError, ValueError):
                    continue
                return label if label >= 0 else None
        match = re.match(r"^rx(\d+)$", str(style.client_id))
        if match:
            label = self._remap_domain_value(int(match.group(1)))
            return label if label >= 0 else None
        return None

    def _style_domain_label_mode(self) -> str:
        mode = str(getattr(self.cfg, "fl_style_domain_label_mode", "constructed") or "constructed").lower()
        if mode in {"target", "target_receiver", "receiver", "domain", "target_domain"}:
            return "target_receiver"
        return "constructed"

    def _style_target_domain_tensor(self, style_batch: Optional[StyleDomainBatch]) -> Optional[torch.Tensor]:
        if style_batch is None or style_batch.d_raw is None:
            return None
        d = style_batch.d_raw.to(self.device).view(-1).long()
        if int(d.numel()) != int(style_batch.y.numel()):
            return None
        semantics = str((style_batch.metadata or {}).get("d_raw_semantics", "") or "").lower()
        if "mapped" in semantics:
            return d
        return self._remap_domain_tensor(d)

    def _style_batch_sources_for_samples(self, style_batch: StyleDomainBatch) -> list[str]:
        style_ids = style_batch.d_style.detach().view(-1).cpu().long().tolist()
        sources = list(style_batch.sources or ())
        out = []
        for style_id in style_ids:
            idx = int(style_id)
            out.append(str(sources[idx]) if 0 <= idx < len(sources) else f"style:{idx}")
        return out

    def _style_zdom_probe_enabled(self, round_idx: int, batch_idx: int) -> bool:
        every = int(getattr(self.cfg, "fl_style_zdom_probe_every", 0) or 0)
        if every <= 0:
            return False
        if int(round_idx) % every != 0:
            return False
        return int(batch_idx) == 0

    def _sample_to_xyd_values(self, sample):
        if isinstance(sample, Mapping):
            x = sample.get("iq", sample.get("x"))
            y = sample.get("label", sample.get("y"))
            d = sample.get("domain", sample.get("d", sample.get("receiver", sample.get("rx_i", None))))
            meta = sample.get("meta", sample)
        else:
            x = sample[0]
            y = sample[1]
            d = sample[2] if len(sample) > 2 else None
            meta = sample[3] if len(sample) > 3 and isinstance(sample[3], Mapping) else {}
        if d is None and isinstance(meta, Mapping):
            d = meta.get("domain", meta.get("d", meta.get("receiver", meta.get("rx_i", None))))
        return x, y, d

    def _sample_real_domain_probe_batch(
        self,
        client_id: str,
        *,
        count: int,
        round_idx: int,
        batch_idx: int,
    ):
        count = int(count)
        if count <= 0:
            return None
        candidates = []
        for cid, indices in self.client_splits.items():
            if str(cid) == str(client_id):
                continue
            candidates.extend(int(idx) for idx in indices)
        if not candidates:
            return None
        rng = random.Random(
            int(getattr(self.cfg, "seed", 0) or 0)
            + int(round_idx) * 1000003
            + int(batch_idx) * 9176
            + 7919
        )
        k = min(count, len(candidates))
        selected = rng.sample(candidates, k=k) if len(candidates) > k else list(candidates)
        xs = []
        ys = []
        ds = []
        for idx in selected:
            try:
                sample = self.train_dataset[int(idx)]
                x, y, d = self._sample_to_xyd_values(sample)
                if x is None or y is None or d is None:
                    continue
                xs.append(torch.as_tensor(x).detach().clone())
                ys.append(int(torch.as_tensor(y).view(-1)[0].item()))
                ds.append(int(torch.as_tensor(d).view(-1)[0].item()))
            except Exception:
                continue
        if not xs:
            return None
        return (
            torch.stack(xs, dim=0).to(self.device),
            torch.as_tensor(ys, device=self.device).long(),
            torch.as_tensor(ds, device=self.device).long(),
        )

    def _style_zdom_probe(
        self,
        client_id: str,
        style_batch: Optional[StyleDomainBatch],
        out: Mapping[str, Any],
        style_targets: Optional[torch.Tensor],
        round_idx: int,
        batch_idx: int,
    ) -> Optional[Dict[str, Any]]:
        if style_batch is None or style_targets is None:
            return None
        if not self._style_zdom_probe_enabled(round_idx, batch_idx):
            return None
        dom_logits = out.get("dom_logits")
        max_examples = int(getattr(self.cfg, "fl_style_zdom_probe_max_examples", 4) or 4)
        d_style = style_batch.d_style.to(self.device).view(-1).long()
        clean_id = int(getattr(self.virtual_domain_sampler, "clean_style_id", 0))
        sources = self._style_batch_sources_for_samples(style_batch)
        clean_mask = d_style == clean_id
        virtual_mask = d_style != clean_id
        all_mask = torch.ones_like(d_style, dtype=torch.bool)
        probe: Dict[str, Any] = {
            "client_id": str(client_id),
            "round": int(round_idx),
            "batch_idx": int(batch_idx),
            "mode": self._style_domain_label_mode(),
            "head": "dom_logits",
            "label_semantics": "mapped_target_domain_label_per_style_view",
            "clean": _domain_probe_stats(dom_logits, style_targets, mask=clean_mask, max_examples=max_examples, sources=sources),
            "virtual": _domain_probe_stats(dom_logits, style_targets, mask=virtual_mask, max_examples=max_examples, sources=sources),
            "all_style": _domain_probe_stats(dom_logits, style_targets, mask=all_mask, max_examples=max_examples, sources=sources),
            "real": _empty_style_zdom_bucket(),
        }
        real_samples = int(getattr(self.cfg, "fl_style_zdom_probe_real_samples", 0) or 0)
        real_batch = self._sample_real_domain_probe_batch(
            client_id,
            count=real_samples,
            round_idx=round_idx,
            batch_idx=batch_idx,
        )
        if real_batch is not None:
            x_real, y_real, d_real_raw = real_batch
            was_training = bool(self.model.training)
            try:
                self.model.eval()
                with torch.no_grad():
                    out_real = self._forward_outputs(x_real, y_real, d_real_raw)
                real_targets = self._remap_domain_tensor(d_real_raw)
                probe["real"] = _domain_probe_stats(
                    out_real.get("dom_logits"),
                    real_targets,
                    max_examples=max_examples,
                    sources=["real_other_domain"] * int(y_real.numel()),
                )
            finally:
                self.model.train(was_training)
        return probe

    def _domain_stats(self, d: Optional[torch.Tensor], y: torch.Tensor) -> Dict[str, Any]:
        valid = None if d is None else d.view(-1).long() >= 0
        num_domains = self._num_domains(d)
        if d is None or valid is None or not bool(valid.any()):
            return {"valid": valid, "num_valid": 0, "num_domains": 0, "domain_frac": 0.0, "has_cross_pairs": False}
        d_valid = d.view(-1).long()[valid]
        y_valid = y.view(-1).long()[valid]
        has_cross_pairs = False
        for cls in torch.unique(y_valid):
            if torch.unique(d_valid[y_valid == cls]).numel() >= 2:
                has_cross_pairs = True
                break
        return {
            "valid": valid,
            "num_valid": int(d_valid.numel()),
            "num_domains": int(torch.unique(d_valid).numel()),
            "domain_frac": float(torch.unique(d_valid).numel()) / float(max(1, num_domains)),
            "has_cross_pairs": bool(has_cross_pairs),
        }

    def _domain_gates(self, stats: Mapping[str, Any], d: Optional[torch.Tensor]) -> Dict[str, bool]:
        num_domains = self._num_domains(d)
        min_domains = int(getattr(self.cfg, "min_batch_domains_for_domain_loss", 2))
        min_frac = float(getattr(self.cfg, "min_batch_domain_frac", 0.15))
        enough_domains = (
            num_domains > 1
            and int(stats.get("num_domains", 0)) >= max(2, min_domains)
            and float(stats.get("domain_frac", 0.0)) >= min_frac
        )
        return {
            "dom": enough_domains,
            "adv": enough_domains,
            "cons": enough_domains and bool(stats.get("has_cross_pairs", False)),
            "group_ce": enough_domains,
        }

    def _forward_outputs(self, x: torch.Tensor, y: torch.Tensor, d: Optional[torch.Tensor]) -> Dict[str, Any]:
        try:
            out = self.model(
                x,
                y_tx=y,
                grl_lambda=float(getattr(self.cfg, "grl_lambda", 1.0)),
                return_aux=True,
                domain_labels=d,
            )
        except TypeError:
            out = self.model(x)
        if isinstance(out, Mapping):
            return dict(out)
        return {"tx_logits": out}

    def _maybe_augment(self, x: torch.Tensor, y: torch.Tensor, round_idx: int, batch_idx: int) -> torch.Tensor:
        if self.augment_fn is None or not bool(getattr(self.cfg, "use_aug", False)):
            return _safe_iq_tensor(x)
        with torch.no_grad():
            return _safe_iq_tensor(self.augment_fn(x, y, round_idx, batch_idx))

    def _domain_preserving_perturb(self, x: torch.Tensor, round_idx: int, batch_idx: int) -> torch.Tensor:
        x = _safe_iq_tensor(x)
        noise_std = max(0.0, float(getattr(self.cfg, "domain_unsup_noise_std", 0.01) or 0.0))
        amp_jitter = max(0.0, float(getattr(self.cfg, "domain_unsup_amp_jitter", 0.03) or 0.0))
        out = x
        if amp_jitter > 0.0:
            shape = [int(x.size(0))] + [1] * max(0, x.dim() - 1)
            amp = 1.0 + (torch.rand(shape, device=x.device, dtype=x.dtype) * 2.0 - 1.0) * amp_jitter
            out = out * amp
        if noise_std > 0.0:
            reduce_dims = tuple(range(1, x.dim()))
            scale = x.detach().float().pow(2).mean(dim=reduce_dims, keepdim=True).sqrt().clamp_min(1e-4)
            out = out + torch.randn_like(out) * scale.to(dtype=out.dtype) * noise_std
        max_shift = max(0, int(getattr(self.cfg, "domain_unsup_max_shift", 0) or 0))
        if max_shift > 0 and out.dim() >= 3:
            shift_rng = random.Random(int(getattr(self.cfg, "seed", 0) or 0) + int(round_idx) * 104729 + int(batch_idx) * 8191)
            shift = shift_rng.randint(-max_shift, max_shift)
            if shift != 0:
                out = torch.roll(out, shifts=shift, dims=-1)
        return _safe_iq_tensor(out)

    def _compute_domain_unsup_pretrain(
        self,
        x_clean: torch.Tensor,
        y_clean: torch.Tensor,
        d_raw_clean: Optional[torch.Tensor],
        round_idx: int,
        batch_idx: int,
        ref: torch.Tensor,
    ) -> Dict[str, Any]:
        zero = ref.new_tensor(0.0)
        metrics: Dict[str, Any] = {
            "loss_domain_unsup_pretrain": zero,
            "loss_domain_unsup_metadata_ce": zero,
            "loss_domain_unsup_var": zero,
            "stage1_domain_pretrain_active": 0.0,
            "domain_unsup_active": 0.0,
            "domain_unsup_view_count": 0.0,
            "domain_unsup_zdom_cos": float("nan"),
            "domain_unsup_client_compact": float("nan"),
            "domain_unsup_client_radius": float("nan"),
            "domain_unsup_dom_entropy": float("nan"),
            "domain_unsup_dom_acc": float("nan"),
            "domain_unsup_zdom_var": float("nan"),
        }
        if self._vmb_stage1_objective() != "domain_unsup_pretrain":
            return metrics
        metrics["stage1_domain_pretrain_active"] = 1.0
        x_clean = _safe_iq_tensor(x_clean.to(self.device))
        y_clean = y_clean.to(self.device).long()
        d_model = None if d_raw_clean is None else d_raw_clean.to(self.device).long()
        out_clean = self._forward_outputs(x_clean, y_clean, d_model)
        out_view = self._forward_outputs(
            self._domain_preserving_perturb(x_clean, round_idx, batch_idx),
            y_clean,
            d_model,
        )
        z_clean = out_clean.get("z_dom")
        z_view = out_view.get("z_dom")
        dom_logits = out_clean.get("dom_logits")
        if torch.is_tensor(z_clean) and torch.is_tensor(z_view) and tuple(z_clean.shape) == tuple(z_view.shape):
            loss_cons, cos = _cosine_consistency_loss(z_clean, z_view)
            metrics["loss_domain_unsup_pretrain"] = loss_cons
            metrics["domain_unsup_zdom_cos"] = cos
            metrics["domain_unsup_active"] = 1.0
            metrics["domain_unsup_view_count"] = 2.0
            compact_weight = max(0.0, float(getattr(self.cfg, "domain_unsup_client_compact_weight", 0.0) or 0.0))
            if compact_weight > 0.0:
                compact_loss, compact_radius = _client_zdom_compactness_loss(torch.cat([z_clean, z_view], dim=0), ref)
                metrics["loss_domain_unsup_pretrain"] = metrics["loss_domain_unsup_pretrain"] + compact_weight * compact_loss
                metrics["domain_unsup_client_compact"] = compact_loss
                metrics["domain_unsup_client_radius"] = compact_radius
                metrics["domain_unsup_active"] = 1.0
        logit_weight = max(0.0, float(getattr(self.cfg, "domain_unsup_logit_cons_weight", 0.0) or 0.0))
        clean_logits = out_clean.get("tx_logits", out_clean.get("logits"))
        view_logits = out_view.get("tx_logits", out_view.get("logits"))
        if logit_weight > 0.0 and torch.is_tensor(clean_logits) and torch.is_tensor(view_logits) and tuple(clean_logits.shape) == tuple(view_logits.shape):
            metrics["loss_domain_unsup_pretrain"] = metrics["loss_domain_unsup_pretrain"] + logit_weight * _symmetric_kl_consistency_loss(clean_logits, view_logits)
            metrics["domain_unsup_active"] = 1.0
        var_loss, z_var = _feature_variance_floor_loss(
            z_clean,
            ref,
            float(getattr(self.cfg, "domain_unsup_var_floor", 0.02) or 0.0),
        )
        metrics["loss_domain_unsup_var"] = var_loss
        metrics["domain_unsup_zdom_var"] = z_var
        metrics["domain_unsup_dom_entropy"] = _logit_entropy(dom_logits)
        method = str(getattr(self.cfg, "domain_unsup_pretrain_method", "consistency") or "consistency").lower()
        if method in {"metadata_consistency", "metadata", "hybrid"} and torch.is_tensor(dom_logits):
            d_loss = self._remap_domain_tensor(d_model)
            if d_loss is not None:
                valid = d_loss.view(-1).long() >= 0
                dom_slice, dom_targets = self._domain_logits_and_targets(dom_logits, d_loss, valid)
                if dom_slice is not None and dom_targets is not None:
                    metrics["loss_domain_unsup_metadata_ce"] = F.cross_entropy(dom_slice.float(), dom_targets)
                    metrics["domain_unsup_dom_acc"] = float((dom_slice.argmax(dim=1) == dom_targets).float().mean().detach().item() * 100.0)
                    metrics["domain_unsup_active"] = 1.0
        return metrics

    def _style_replay_ready(self, client_id: str, round_idx: int) -> bool:
        if self.style_bank is None or self.style_transform is None:
            return False
        if int(round_idx) < int(getattr(self.cfg, "fl_style_replay_start_round", 20)):
            return False
        if int(round_idx) < int(getattr(self.cfg, "fl_style_phys_start_round", 20)):
            return False
        diag = self.style_bank.diagnostics()
        if int(diag.get("num_centroids", 0)) < int(getattr(self.cfg, "fl_style_min_remote_centroids", 1)):
            return False
        if self.style_bank.sample_remote_style(exclude_client_id=str(client_id)) is None:
            return False
        return True

    def _style_dg_ready(self, style_batch: Optional[StyleDomainBatch], round_idx: int) -> bool:
        if style_batch is None:
            return False
        if int(round_idx) < int(getattr(self.cfg, "fl_style_dg_start_round", 40)):
            return False
        if int(style_batch.num_style_domains) < int(getattr(self.cfg, "fl_style_dg_min_domains", 2)):
            return False
        min_accept = float(getattr(self.cfg, "style_gate_min_accept_rate", 0.0) or 0.0)
        if min_accept > 0.0:
            meta = dict(getattr(style_batch, "metadata", {}) or {})
            requested = float(meta.get("requested_remote_views", 0.0) or 0.0)
            appended = float(meta.get("appended_remote_views", 0.0) or 0.0)
            if requested <= 0.0 or appended / max(requested, 1.0) < min_accept:
                return False
        return True

    def _style_sat_scenario(self, round_idx: int, batch_idx: int) -> str:
        raw = str(getattr(self.cfg, "sat_train_scenarios", "") or "").strip()
        if not raw:
            raw = str(getattr(self.cfg, "sat_train_scenario", "mixed_orbit") or "mixed_orbit")
        scenarios = [item.strip() for item in raw.split(",") if item.strip()]
        if not scenarios:
            return "mixed_orbit"
        return scenarios[(int(round_idx) + int(batch_idx)) % len(scenarios)]

    def _build_default_style_batch(
        self,
        client_id: str,
        x_main: torch.Tensor,
        y: torch.Tensor,
        d_raw: Optional[torch.Tensor],
        round_idx: int,
        batch_idx: int,
        *,
        ignore_replay_probability: bool = False,
    ) -> Optional[StyleDomainBatch]:
        if not self._style_replay_ready(client_id, round_idx):
            return None
        clean_domain = self._remap_domain_tensor(d_raw)
        clean_target = self._mode_domain_label(d_raw)
        if clean_domain is None or clean_target is None:
            return None
        prob = float(getattr(self.cfg, "fl_style_replay_prob", 0.25))
        if (not ignore_replay_probability) and prob < 1.0:
            rng = random.Random(int(getattr(self.cfg, "seed", 0)) + int(round_idx) * 1000003 + int(batch_idx) * 9176)
            if rng.random() > max(0.0, prob):
                return None
        assert self.style_bank is not None and self.style_transform is not None
        requested_views = max(1, int(getattr(self.cfg, "fl_style_max_views", 1)))
        sample_k = requested_views
        if self._style_domain_label_mode() == "target_receiver":
            sample_k = max(requested_views, requested_views * 3)
        try:
            styles = self.style_bank.sample_remote_styles(
                exclude_client_id=str(client_id),
                k=sample_k,
                policy=str(getattr(self.cfg, "fl_style_sampling_policy", "diverse") or "diverse"),
            )
        except TypeError:
            styles = self.style_bank.sample_remote_styles(
                exclude_client_id=str(client_id),
                k=sample_k,
            )
        clean_style_id = int(getattr(self.virtual_domain_sampler, "clean_style_id", 0))
        xs = [x_main]
        ys = [y]
        d_raw_parts = [clean_domain.view(-1).long()]
        d_style_parts = [
            torch.full((int(y.numel()),), clean_style_id, dtype=torch.long, device=y.device)
        ]
        sources = ["clean"]
        raw_style_ids = [clean_style_id]
        target_domain_labels = [int(clean_target)]
        appended_remote_views = 0
        for style in styles:
            if appended_remote_views >= requested_views:
                break
            target = self._style_packet_target_domain(style)
            if target is None:
                continue
            if int(target) == int(clean_target):
                continue
            x_style = self.style_transform.transform(x_main, style)
            mix_alpha = max(0.0, min(1.0, float(getattr(self.cfg, "fl_style_transform_mix_alpha", 1.0))))
            if mix_alpha < 1.0:
                x_style = x_main + float(mix_alpha) * (_safe_iq_tensor(x_style) - x_main)
            target_vec = torch.full((int(y.numel()),), int(target), dtype=torch.long, device=y.device)
            style_id = len(sources)
            xs.append(_safe_iq_tensor(x_style))
            ys.append(y)
            d_raw_parts.append(target_vec)
            d_style_parts.append(torch.full((int(y.numel()),), int(style_id), dtype=torch.long, device=y.device))
            sources.append(f"remote_style:{style.client_id}")
            raw_style_ids.append(int(style.style_id if style.style_id is not None else -1))
            target_domain_labels.append(int(target))
            appended_remote_views += 1
        use_sat_view = (
            bool(getattr(self.cfg, "use_fed_style_sat_view", False))
            and bool(getattr(self.cfg, "use_sat_consistency", False))
            and self.sat_transform_fn is not None
            and int(round_idx) >= int(getattr(self.cfg, "sat_cons_start_epoch", 1))
        )
        if use_sat_view:
            scenario = self._style_sat_scenario(round_idx, batch_idx)
            with torch.no_grad():
                x_sat = _safe_iq_tensor(self.sat_transform_fn(x_main, scenario, round_idx, batch_idx))
            xs.append(x_sat)
            ys.append(y)
            d_raw_parts.append(clean_domain.view(-1).long())
            d_style_parts.append(torch.full((int(y.numel()),), int(len(sources)), dtype=torch.long, device=y.device))
            sources.append(f"sat:{scenario}")
            raw_style_ids.append(-1000)
            target_domain_labels.append(int(clean_target))
        real_mix_samples = int(getattr(self.cfg, "fl_style_real_mix_samples", 0) or 0)
        real_mix_start = int(
            getattr(
                self.cfg,
                "fl_style_real_mix_start_round",
                int(getattr(self.cfg, "fl_style_replay_start_round", 20)),
            )
            or 0
        )
        if real_mix_samples > 0 and int(round_idx) >= real_mix_start:
            real_batch = self._sample_real_domain_probe_batch(
                client_id,
                count=real_mix_samples,
                round_idx=round_idx,
                batch_idx=batch_idx,
            )
            if real_batch is not None:
                x_real, y_real, d_real_raw = real_batch
                real_domain = self._remap_domain_tensor(d_real_raw)
                if real_domain is not None and int(real_domain.numel()) == int(y_real.numel()):
                    style_id = len(sources)
                    xs.append(_safe_iq_tensor(x_real.to(device=y.device)))
                    ys.append(y_real.to(device=y.device).long())
                    d_raw_parts.append(real_domain.view(-1).to(device=y.device).long())
                    d_style_parts.append(torch.full((int(y_real.numel()),), int(style_id), dtype=torch.long, device=y.device))
                    sources.append("real_other_domain")
                    raw_style_ids.append(-2000)
                    valid_real = real_domain.view(-1).long()
                    valid_real = valid_real[valid_real >= 0]
                    target_domain_labels.append(int(torch.mode(valid_real.cpu()).values.item()) if valid_real.numel() else -1)
        if len(xs) <= 1:
            return None
        return StyleDomainBatch(
            x=torch.cat(xs, dim=0),
            y=torch.cat(ys, dim=0).long(),
            d_raw=torch.cat(d_raw_parts, dim=0).long(),
            d_style=torch.cat(d_style_parts, dim=0).long(),
            sources=tuple(sources),
            metadata={
                "num_views": len(sources),
                "requested_remote_views": int(requested_views),
                "appended_remote_views": int(appended_remote_views),
                "raw_style_ids": tuple(raw_style_ids),
                "target_domain_labels": tuple(target_domain_labels),
                "source_domain_labels": tuple(target_domain_labels),
                "style_domain_semantics": "constructed_style_view_id",
                "d_style_semantics": "0=clean_view,1..K=remote_or_sat_style_view_order",
                "d_raw_semantics": "mapped_target_receiver_domain_for_each_view",
            },
        )

    def _maybe_build_style_batch(
        self,
        client_id: str,
        x_main: torch.Tensor,
        y: torch.Tensor,
        d_raw: Optional[torch.Tensor],
        round_idx: int,
        batch_idx: int,
    ) -> Optional[StyleDomainBatch]:
        if self.style_batch_fn is None:
            batch = self._build_default_style_batch(client_id, x_main, y, d_raw, round_idx, batch_idx)
        else:
            batch = self.style_batch_fn(x_main, y, d_raw, round_idx, batch_idx, self)
        if batch is None:
            return None
        if not isinstance(batch, StyleDomainBatch):
            raise TypeError("style_batch_fn must return federated.style_packet.StyleDomainBatch or None.")
        if int(batch.x.size(0)) != int(batch.y.numel()) or int(batch.d_style.numel()) != int(batch.y.numel()):
            raise ValueError("StyleDomainBatch x, y, and d_style batch sizes must match.")
        return batch

    def _build_forced_style_probe_batch(
        self,
        client_id: str,
        x_main: torch.Tensor,
        y: torch.Tensor,
        d_raw: Optional[torch.Tensor],
        round_idx: int,
        batch_idx: int,
    ) -> Optional[StyleDomainBatch]:
        if not bool(getattr(self.cfg, "fl_style_zdom_probe_force_batch", False)):
            return None
        if not self._style_zdom_probe_enabled(round_idx, batch_idx):
            return None
        if self.style_batch_fn is not None:
            return None
        batch = self._build_default_style_batch(
            client_id,
            x_main,
            y,
            d_raw,
            round_idx,
            batch_idx,
            ignore_replay_probability=True,
        )
        if batch is None:
            return None
        if not isinstance(batch, StyleDomainBatch):
            raise TypeError("forced style probe batch must return federated.style_packet.StyleDomainBatch or None.")
        if int(batch.x.size(0)) != int(batch.y.numel()) or int(batch.d_style.numel()) != int(batch.y.numel()):
            raise ValueError("Forced StyleDomainBatch x, y, and d_style batch sizes must match.")
        return batch

    def _compute_group_ce(self, logits: torch.Tensor, y: torch.Tensor, d: Optional[torch.Tensor], gates: Mapping[str, bool]) -> torch.Tensor:
        if d is None or not gates.get("group_ce", False):
            return logits.new_tensor(0.0)
        mode = str(getattr(self.cfg, "group_ce_mode", "hard")).lower()
        if mode not in {"mean"}:
            loss, _ = groupdro_or_hard_domain_ce_loss(
                logits,
                y,
                d,
                self.groupdro_state,
                mode=mode,
                label_smoothing=float(getattr(self.cfg, "label_smoothing", 0.0)),
                top_frac=float(getattr(self.cfg, "group_ce_top_frac", 0.35)),
                min_domains=int(getattr(self.cfg, "group_ce_min_domains", 2)),
                tau=float(getattr(self.cfg, "groupdro_tau", 0.5)),
                cap=float(getattr(self.cfg, "groupdro_cap", 0.65)),
                rx_day_num_days=int(getattr(self.cfg, "groupdro_num_days", 4)),
            )
            return loss
        losses = []
        for dom in torch.unique(d[d >= 0]):
            m = d == dom
            if bool(m.any()):
                losses.append(self.criterion(logits[m].float(), y[m]))
        if len(losses) < max(2, int(getattr(self.cfg, "group_ce_min_domains", 2))):
            return logits.new_tensor(0.0)
        return torch.stack(losses).mean()

    def _domain_logits_and_targets(
        self,
        domain_logits: Optional[torch.Tensor],
        d: Optional[torch.Tensor],
        valid: Optional[torch.Tensor],
    ):
        if not torch.is_tensor(domain_logits) or d is None or valid is None or not bool(valid.any()):
            return None, None
        targets = d.view(-1).long()[valid]
        if targets.numel() == 0:
            return None, None
        if int(targets.min().item()) < 0 or int(targets.max().item()) >= int(domain_logits.size(1)):
            return None, None
        return domain_logits[valid], targets

    def _fed_coral_stage_active(self, round_idx: int) -> bool:
        if getattr(self, "global_coral_bank", None) is None:
            return False
        start = int(_coral_config_value(self.cfg, "fl_coral_start_round", "fed_coral_start_round", 1) or 1)
        if int(round_idx) < max(1, start):
            return False
        stage = str(getattr(self.cfg, "fl_coral_stage", "stage1") or "stage1").lower().strip()
        if stage in {"all", "any", "*"}:
            return True
        current = self._vmb_stage(round_idx) if self.vmb_enabled else "off"
        return stage == current

    def _coral_feature_from_outputs(self, out: Mapping[str, Any], feature_name: str) -> Optional[torch.Tensor]:
        name = str(feature_name or "z_id").lower().strip()
        if name == "z_dom":
            value = out.get("z_dom")
        else:
            value = _select_generalization_feature(out, SimpleNamespace(generalization_feature=name))
        return value if torch.is_tensor(value) and value.dim() == 2 else None

    def _compute_virtual_zid_coral(
        self,
        z_id: Optional[torch.Tensor],
        labels: torch.Tensor,
        d_style: Optional[torch.Tensor],
        clean_style_id: int,
    ):
        ref = z_id if torch.is_tensor(z_id) else torch.tensor(0.0, device=self.device)
        zero = ref.new_tensor(0.0)
        if not torch.is_tensor(z_id) or d_style is None:
            return zero, {"active_classes": 0, "mean_dist": float("nan"), "cov_dist": float("nan"), "skip_rate": 1.0}
        if int(z_id.size(0)) != int(labels.numel()) or int(d_style.numel()) != int(labels.numel()):
            return zero, {"active_classes": 0, "mean_dist": float("nan"), "cov_dist": float("nan"), "skip_rate": 1.0}
        clean_mask = d_style.view(-1).long() == int(clean_style_id)
        virtual_mask = ~clean_mask
        if not bool(clean_mask.any()) or not bool(virtual_mask.any()):
            return zero, {"active_classes": 0, "mean_dist": float("nan"), "cov_dist": float("nan"), "skip_rate": 1.0}
        mode = str(_coral_config_value(self.cfg, "fl_coral_cov_mode", "fed_coral_mode", "diag") or "diag")
        num_classes = int(getattr(self.cfg, "num_classes", 1))
        clean_stats = build_class_conditional_coral_stats(
            z_id[clean_mask].detach(),
            labels[clean_mask],
            num_classes=num_classes,
            mode=mode,
        )
        return class_conditional_coral_loss(
            z_id[virtual_mask],
            labels[virtual_mask],
            clean_stats,
            min_count=int(_coral_config_value(self.cfg, "fl_coral_min_count", "fed_coral_min_count", 2)),
            shrinkage=float(getattr(self.cfg, "fl_coral_shrinkage", 0.0) or 0.0),
        )

    def _compute_pairwise_zid_coral(
        self,
        clean_z: Optional[torch.Tensor],
        clean_y: torch.Tensor,
        virtual_z: Optional[torch.Tensor],
        virtual_y: torch.Tensor,
    ):
        ref = virtual_z if torch.is_tensor(virtual_z) else (clean_z if torch.is_tensor(clean_z) else torch.tensor(0.0, device=self.device))
        zero = ref.new_tensor(0.0)
        if not torch.is_tensor(clean_z) or not torch.is_tensor(virtual_z):
            return zero, {"active_classes": 0, "mean_dist": float("nan"), "cov_dist": float("nan"), "skip_rate": 1.0}
        if clean_z.dim() != 2 or virtual_z.dim() != 2:
            return zero, {"active_classes": 0, "mean_dist": float("nan"), "cov_dist": float("nan"), "skip_rate": 1.0}
        if int(clean_z.size(0)) != int(clean_y.numel()) or int(virtual_z.size(0)) != int(virtual_y.numel()):
            return zero, {"active_classes": 0, "mean_dist": float("nan"), "cov_dist": float("nan"), "skip_rate": 1.0}
        mode = str(_coral_config_value(self.cfg, "fl_coral_cov_mode", "fed_coral_mode", "diag") or "diag")
        num_classes = int(getattr(self.cfg, "num_classes", 0) or 0)
        if num_classes <= 0:
            max_label = int(torch.cat([clean_y.detach().view(-1), virtual_y.detach().view(-1)]).max().item()) if int(clean_y.numel() + virtual_y.numel()) > 0 else 0
            num_classes = max(1, max_label + 1)
        clean_stats = build_class_conditional_coral_stats(
            clean_z.detach(),
            clean_y.detach(),
            num_classes=num_classes,
            mode=mode,
        )
        return class_conditional_coral_loss(
            virtual_z,
            virtual_y,
            clean_stats,
            min_count=int(_coral_config_value(self.cfg, "fl_coral_min_count", "fed_coral_min_count", 2)),
            shrinkage=float(getattr(self.cfg, "fl_coral_shrinkage", 0.0) or 0.0),
        )

    @staticmethod
    def _combine_coral_metric_dicts(metrics: list[Mapping[str, Any]]) -> Dict[str, float]:
        active = [m for m in metrics if int(m.get("active_classes", 0) or 0) > 0]
        if not active:
            return {"active_classes": 0, "mean_dist": float("nan"), "cov_dist": float("nan"), "skip_rate": 1.0}
        def finite_avg(name: str) -> float:
            vals = []
            for item in active:
                try:
                    value = float(item.get(name, float("nan")))
                except (TypeError, ValueError):
                    continue
                if math.isfinite(value):
                    vals.append(value)
            return float(sum(vals) / max(1, len(vals))) if vals else float("nan")
        return {
            "active_classes": float(sum(float(m.get("active_classes", 0) or 0) for m in active)),
            "mean_dist": finite_avg("mean_dist"),
            "cov_dist": finite_avg("cov_dist"),
            "skip_rate": float(sum(float(m.get("skip_rate", 1.0) or 1.0) for m in active) / max(1, len(active))),
        }

    def _compute_local_objective(
        self,
        client_id: str,
        x: torch.Tensor,
        y: torch.Tensor,
        d_raw: Optional[torch.Tensor],
        round_idx: int,
        batch_idx: int,
        global_params: Mapping[str, torch.Tensor],
        mu: float,
        exclude_for_prox,
        fed_cgrl_lambda_rx_adv: Optional[float] = None,
        fed_cgrl_metrics: Optional[Mapping[str, float]] = None,
    ) -> Dict[str, Any]:
        d_raw_clean = d_raw
        y_clean = y
        x_main = self._maybe_augment(x, y, round_idx, batch_idx)
        style_batch = self._maybe_build_style_batch(client_id, x_main, y, d_raw, round_idx, batch_idx)
        d_style: Optional[torch.Tensor] = None
        style_targets: Optional[torch.Tensor] = None
        style_domain_label_mode = self._style_domain_label_mode()
        style_dg_ready = False
        if style_batch is not None:
            x_main = _safe_iq_tensor(style_batch.x.to(self.device))
            y = style_batch.y.to(self.device).long()
            d_raw = None if style_batch.d_raw is None else style_batch.d_raw.to(self.device).long()
            d_style = style_batch.d_style.to(self.device).long()
            style_targets = self._style_target_domain_tensor(style_batch)
            style_dg_ready = self._style_dg_ready(style_batch, round_idx)

        if d_style is not None:
            if style_domain_label_mode == "target_receiver" and style_targets is not None:
                d_loss = style_targets if style_dg_ready else None
                d_model = style_targets
            else:
                d_loss = d_style if style_dg_ready else None
                d_model = d_style
        else:
            d_loss = self._remap_domain_tensor(d_raw)
            d_model = d_raw
        out = self._forward_outputs(x_main, y, d_model)
        logits = out.get("tx_logits", out.get("logits"))
        if logits is None:
            raise KeyError("Federated local objective requires tx_logits or logits from the model.")

        vmb_stage = self._vmb_stage(round_idx) if self.vmb_enabled else "off"
        stage1_objective = self._vmb_stage1_objective() if self.vmb_enabled and vmb_stage == "stage1" else "same"
        stage1_objective_for_loss = stage1_objective
        if stage1_objective == "receiver_style_pretrain":
            stage1_objective_for_loss = "receiver_agnostic_bex02"
        elif stage1_objective == "domain_unsup_pretrain":
            stage1_objective_for_loss = "receiver_agnostic_bex02"
        elif stage1_objective == "cen_a31_lite":
            stage1_objective_for_loss = "local_virtual_bex02"
        stage1_aux_losses = (
            self.vmb_enabled
            and vmb_stage == "stage1"
            and (
                bool(getattr(self.cfg, "fl_vmb_stage1_use_aux_losses", False))
                or stage1_objective in {"receiver_style_pretrain", "cen_a31_lite", "domain_unsup_pretrain"}
            )
        )
        stage1_objective_is_ce = stage1_objective_for_loss == "ce"
        stage1_ce_only = stage1_objective_is_ce and not stage1_aux_losses
        objective = str(getattr(self.cfg, "fl_local_objective", "ce")).lower()
        if stage1_objective_for_loss not in {"", "same"}:
            objective = stage1_objective_for_loss
        sat_aug_mode = str(getattr(self.cfg, "fl_sat_aug_mode", "baseline_view")).lower()
        baseline_sat_ce_only = bool(getattr(self.cfg, "fl_baseline_view_ce_only", False))
        dg_objectives = {"bex02_dg", "strong_dg", "dg", "receiver_agnostic_bex02", "ra_bex02", "local_virtual_bex02"}
        stage1_baseline_sat_ce = (
            self.vmb_enabled
            and vmb_stage == "stage1"
            and stage1_objective_is_ce
            and baseline_sat_ce_only
        )
        use_baseline_sat_view = (
            (objective in dg_objectives or stage1_baseline_sat_ce)
            and sat_aug_mode == "baseline_view"
            and bool(getattr(self.cfg, "use_sat_consistency", False))
            and self.sat_transform_fn is not None
            and round_idx >= int(getattr(self.cfg, "sat_cons_start_epoch", 1))
            and (d_style is None or baseline_sat_ce_only)
        )
        loss_cls = self.criterion(logits.float(), y)
        terms: Dict[str, torch.Tensor] = {
            "loss_cls": loss_cls,
            "loss_rx_adv": logits.new_tensor(0.0),
            "loss_dom": logits.new_tensor(0.0),
            "loss_adv": logits.new_tensor(0.0),
            "loss_orth": logits.new_tensor(0.0),
            "loss_cons": logits.new_tensor(0.0),
            "loss_group_ce": logits.new_tensor(0.0),
            "loss_fishr": logits.new_tensor(0.0),
            "loss_fed_fishr": logits.new_tensor(0.0),
            "loss_sat_cls": logits.new_tensor(0.0),
            "loss_sat_cons": logits.new_tensor(0.0),
            "loss_fed_proto": logits.new_tensor(0.0),
            "loss_baseline_sat_view": logits.new_tensor(0.0),
            "loss_vmb_tx_proto": logits.new_tensor(0.0),
            "loss_vmb_rx_proto": logits.new_tensor(0.0),
            "loss_tx_adv_r": logits.new_tensor(0.0),
            "loss_logit_kd": logits.new_tensor(0.0),
            "loss_supcon_id": logits.new_tensor(0.0),
            "loss_coral_zid_global": logits.new_tensor(0.0),
            "loss_coral_zid_virtual": logits.new_tensor(0.0),
            "loss_coral_zdom_global": logits.new_tensor(0.0),
            "loss_domain_unsup_pretrain": logits.new_tensor(0.0),
            "loss_domain_unsup_metadata_ce": logits.new_tensor(0.0),
            "loss_domain_unsup_var": logits.new_tensor(0.0),
        }
        extra_metrics: Dict[str, float] = {
            "cons_cos": float("nan"),
            "sat_cos": float("nan"),
            "fed_proto_cos": float("nan"),
            "vmb_tx_proto_cos": float("nan"),
            "vmb_rx_proto_cos": float("nan"),
            "vmb_tx_proto_active": 0.0,
            "vmb_rx_proto_active": 0.0,
            "vmb_stage": vmb_stage,
            "tx_adv_r_acc": float("nan"),
            "dom_acc": float("nan"),
            "zdom_target_acc": float("nan"),
            "grl_target_acc": float("nan"),
            "style_num_domains": float(style_batch.num_style_domains) if style_batch is not None else float("nan"),
            "style_batch_views": float(len(style_batch.sources)) if style_batch is not None else float("nan"),
            "style_domain_entropy": _style_domain_entropy(d_style),
            "style_dg_ready": 1.0 if style_dg_ready else 0.0,
            "style_domain_label_mode": style_domain_label_mode,
            "style_target_domain_count": float(_batch_domain_count(style_targets)),
            "style_requested_remote_views": float((style_batch.metadata or {}).get("requested_remote_views", 0) if style_batch is not None else 0),
            "style_appended_remote_views": float((style_batch.metadata or {}).get("appended_remote_views", 0) if style_batch is not None else 0),
            "style_real_mix_active": 1.0 if (style_batch is not None and "real_other_domain" in set(style_batch.sources or ())) else 0.0,
            "style_gate_value": (
                float((style_batch.metadata or {}).get("appended_remote_views", 0) or 0)
                / max(1.0, float((style_batch.metadata or {}).get("requested_remote_views", 0) or 0))
            ) if style_batch is not None else float("nan"),
            "stage1_domain_pretrain_active": 0.0,
            "domain_unsup_active": 0.0,
            "domain_unsup_view_count": 0.0,
            "domain_unsup_zdom_cos": float("nan"),
            "domain_unsup_client_compact": float("nan"),
            "domain_unsup_client_radius": float("nan"),
            "domain_unsup_dom_entropy": float("nan"),
            "domain_unsup_dom_acc": float("nan"),
            "domain_unsup_zdom_var": float("nan"),
            "fishr_gate_value": 0.0,
            "diag_domain_count": float(_batch_domain_count(d_loss)),
            "diag_fishr_domain_count": float(_batch_fishr_domain_count(d_loss)),
            "diag_domain_loss_active": 0.0,
            "diag_adv_active": 0.0,
            "diag_cons_active": 0.0,
            "diag_group_ce_active": 0.0,
            "diag_fishr_active": 0.0,
            "diag_rx_adv_active": 0.0,
            "diag_sat_aug_active": 0.0,
            "diag_baseline_sat_view_active": 0.0,
            "diag_sat_cls_active": 0.0,
            "diag_sat_cons_active": 0.0,
            "diag_style_batch_active": 1.0 if style_batch is not None else 0.0,
            "diag_style_domain_count": float(style_batch.num_style_domains) if style_batch is not None else 0.0,
            "diag_stage1_aux_active": 1.0 if stage1_aux_losses else 0.0,
            "diag_coral_global_active": 0.0,
            "diag_coral_virtual_active": 0.0,
            "diag_coral_zdom_active": 0.0,
            "kd_active": 0.0,
            "logit_anchor_count": 0.0,
            "logit_anchor_payload_bytes": 0.0,
            "activation_token_payload_bytes": 0.0,
            "activation_token_compression_ratio": float("nan"),
            "activation_token_quant_error": float("nan"),
            "feature_probe_samples": 0.0,
            "coral_zid_global_active_classes": 0.0,
            "coral_zid_global_mean_dist": float("nan"),
            "coral_zid_global_cov_dist": float("nan"),
            "coral_zid_global_skip_rate": float("nan"),
            "coral_zid_virtual_active_classes": 0.0,
            "coral_zid_virtual_mean_dist": float("nan"),
            "coral_zid_virtual_cov_dist": float("nan"),
            "coral_zid_virtual_skip_rate": float("nan"),
            "coral_zdom_global_active_classes": 0.0,
            "coral_zdom_global_mean_dist": float("nan"),
            "coral_zdom_global_cov_dist": float("nan"),
            "coral_zdom_global_skip_rate": float("nan"),
            "fed_fishr_active": 0.0,
            "fed_fishr_active_classes": 0.0,
            "fed_fishr_var_dist": float("nan"),
            "fed_fishr_skip_rate": float("nan"),
            "fed_fishr_payload_bytes": 0.0,
            "fed_fishr_target_ready": 0.0,
        }
        if fed_cgrl_metrics:
            for metric_name, metric_value in fed_cgrl_metrics.items():
                try:
                    extra_metrics[str(metric_name)] = float(metric_value)
                except (TypeError, ValueError):
                    continue

        def refresh_domain_diag() -> None:
            extra_metrics["diag_domain_count"] = float(_batch_domain_count(d_loss))
            extra_metrics["diag_fishr_domain_count"] = float(_batch_fishr_domain_count(d_loss))

        out_sat_for_coral: Optional[Mapping[str, Any]] = None
        sat_y_for_coral: Optional[torch.Tensor] = None
        if use_baseline_sat_view:
            with torch.no_grad():
                scenario = self._style_sat_scenario(round_idx, batch_idx)
            x_sat = _safe_iq_tensor(self.sat_transform_fn(x, scenario, round_idx, batch_idx))
            if baseline_sat_ce_only:
                out_sat = self._forward_outputs(x_sat, y_clean, d_raw_clean)
                out_sat_for_coral = out_sat
                sat_y_for_coral = y_clean
                sat_logits = out_sat.get("tx_logits", out_sat.get("logits"))
                if sat_logits is None:
                    raise KeyError("Federated baseline-view CE-only SAT objective requires tx_logits or logits from the model.")
                terms["loss_baseline_sat_view"] = self.criterion(sat_logits.float(), y_clean)
                extra_metrics["diag_sat_cls_active"] = 1.0
            else:
                x_view = torch.cat([x_main, x_sat], dim=0)
                y_view = torch.cat([y, y], dim=0)
                d_raw_view = _concat_optional_domain(d_raw)
                d_view = self._remap_domain_tensor(d_raw_view)
                out = self._forward_outputs(x_view, y_view, d_raw_view)
                logits = out.get("tx_logits", out.get("logits"))
                if logits is None:
                    raise KeyError("Federated baseline-view SAT objective requires tx_logits or logits from the model.")
                loss_cls = self.criterion(logits.float(), y_view)
                terms["loss_cls"] = loss_cls
                terms["loss_baseline_sat_view"] = loss_cls
                y = y_view
                d_raw = d_raw_view
                d_loss = d_view
                d_model = d_raw_view
                refresh_domain_diag()
            extra_metrics["diag_sat_aug_active"] = 1.0
            extra_metrics["diag_baseline_sat_view_active"] = 1.0

        if objective in {"bex02_dg", "strong_dg", "dg", "receiver_agnostic_bex02", "ra_bex02", "local_virtual_bex02"}:
            stats = self._domain_stats(d_loss, y)
            gates = self._domain_gates(stats, d_loss)
            valid = stats.get("valid", None)
            extra_metrics["diag_domain_count"] = float(stats.get("num_domains", _batch_domain_count(d_loss)))
            if d_loss is not None and valid is not None and bool(valid.any()):
                d_valid = d_loss[valid].long()
                dom_logits = out.get("dom_logits")
                adv_dom_logits = out.get("adv_dom_logits")
                rx_logits = out.get("rx_logits")
                rx_uses_adv_head = False
                if rx_logits is None and objective in {"receiver_agnostic_bex02", "ra_bex02"}:
                    rx_logits = adv_dom_logits
                    rx_uses_adv_head = adv_dom_logits is not None
                rx_slice, rx_targets = self._domain_logits_and_targets(rx_logits, d_loss, valid)
                dom_slice, dom_targets = self._domain_logits_and_targets(dom_logits, d_loss, valid)
                adv_slice, adv_targets = self._domain_logits_and_targets(adv_dom_logits, d_loss, valid)
                if objective in {"receiver_agnostic_bex02", "ra_bex02"} and rx_slice is not None:
                    terms["loss_rx_adv"] = F.cross_entropy(rx_slice.float(), rx_targets)
                    extra_metrics["grl_target_acc"] = float((rx_slice.argmax(dim=1) == rx_targets).float().mean().detach().item() * 100.0)
                    extra_metrics["diag_rx_adv_active"] = 1.0
                if gates.get("dom", False) and dom_slice is not None:
                    terms["loss_dom"] = F.cross_entropy(dom_slice.float(), dom_targets)
                    extra_metrics["dom_acc"] = float((dom_slice.argmax(dim=1) == dom_targets).float().mean().detach().item() * 100.0)
                    extra_metrics["zdom_target_acc"] = extra_metrics["dom_acc"]
                    extra_metrics["diag_domain_loss_active"] = 1.0
                if gates.get("adv", False) and adv_slice is not None and not rx_uses_adv_head:
                    terms["loss_adv"] = F.cross_entropy(adv_slice.float(), adv_targets)
                    if not math.isfinite(extra_metrics["grl_target_acc"]):
                        extra_metrics["grl_target_acc"] = float((adv_slice.argmax(dim=1) == adv_targets).float().mean().detach().item() * 100.0)
                    extra_metrics["diag_adv_active"] = 1.0
                if gates.get("cons", False):
                    terms["loss_cons"], extra_metrics["cons_cos"] = _same_tx_cross_domain_consistency(out.get("z_id"), y, d_loss, logits)
                    extra_metrics["diag_cons_active"] = 1.0
                terms["loss_group_ce"] = self._compute_group_ce(logits, y, d_loss, gates)
                if gates.get("group_ce", False):
                    extra_metrics["diag_group_ce_active"] = 1.0

            terms["loss_orth"] = _covariance_orth_loss(out.get("z_id"), out.get("z_dom"), logits)
            fishr_domain_count = _batch_fishr_domain_count(d_loss)
            fishr_min_domains = int(getattr(self.cfg, "fishr_min_domains", 2))
            extra_metrics["diag_fishr_domain_count"] = float(fishr_domain_count)
            extra_metrics["diag_fishr_active"] = 1.0 if fishr_domain_count >= max(2, fishr_min_domains) else 0.0
            extra_metrics["fishr_gate_value"] = extra_metrics["diag_fishr_active"]
            terms["loss_fishr"] = _fishr_logit_gradient_variance_loss(
                logits,
                y,
                d_loss,
                min_domains=fishr_min_domains,
            )

        if self.vmb_enabled and vmb_stage == "stage1" and stage1_objective == "domain_unsup_pretrain":
            domain_pretrain = self._compute_domain_unsup_pretrain(
                x,
                y_clean,
                d_raw_clean,
                round_idx,
                batch_idx,
                logits,
            )
            for name in ("loss_domain_unsup_pretrain", "loss_domain_unsup_metadata_ce", "loss_domain_unsup_var"):
                value = domain_pretrain.pop(name, None)
                if torch.is_tensor(value):
                    terms[name] = value
            for name, value in domain_pretrain.items():
                try:
                    extra_metrics[str(name)] = float(value)
                except (TypeError, ValueError):
                    continue

            use_sat = bool(getattr(self.cfg, "use_sat_consistency", False)) and self.sat_transform_fn is not None
            use_sat = use_sat and round_idx >= int(getattr(self.cfg, "sat_cons_start_epoch", 1))
            use_sat = use_sat and sat_aug_mode != "baseline_view"
            if use_sat and (float(getattr(self.cfg, "lambda_sat_cls", 0.0)) > 0.0 or float(getattr(self.cfg, "lambda_sat_cons", 0.0)) > 0.0):
                extra_metrics["diag_sat_aug_active"] = 1.0
                with torch.no_grad():
                    scenario = self._style_sat_scenario(round_idx, batch_idx)
                    x_sat = _safe_iq_tensor(self.sat_transform_fn(x, scenario, round_idx, batch_idx))
                out_sat = self._forward_outputs(x_sat, y_clean, d_raw_clean)
                sat_logits = out_sat.get("tx_logits", out_sat.get("logits"))
                if torch.is_tensor(sat_logits):
                    out_sat_for_coral = out_sat
                    sat_y_for_coral = y_clean
                    terms["loss_sat_cls"] = self.criterion(sat_logits.float(), y_clean)
                    if float(getattr(self.cfg, "lambda_sat_cls", 0.0)) > 0.0:
                        extra_metrics["diag_sat_cls_active"] = 1.0
                z_ref = out.get("z_id")
                if torch.is_tensor(z_ref) and d_style is not None:
                    clean_count = int(y_clean.numel())
                    if int(z_ref.size(0)) >= clean_count:
                        z_ref = z_ref[:clean_count]
                if torch.is_tensor(z_ref) and torch.is_tensor(out_sat.get("z_id")):
                    terms["loss_sat_cons"], extra_metrics["sat_cos"] = _cosine_consistency_loss(out_sat["z_id"], z_ref.detach())
                    if float(getattr(self.cfg, "lambda_sat_cons", 0.0)) > 0.0:
                        extra_metrics["diag_sat_cons_active"] = 1.0

        if (not stage1_ce_only) and float(getattr(self.cfg, "lambda_supcon_id", 0.0)) > 0.0:
            dg_feat = _select_generalization_feature(out, self.cfg)
            supcon_d = d_style if d_style is not None else d_loss
            if torch.is_tensor(dg_feat) and int(dg_feat.size(0)) == int(y.numel()):
                terms["loss_supcon_id"] = domain_aware_supcon_loss(
                    dg_feat,
                    y,
                    supcon_d,
                    temperature=float(getattr(self.cfg, "supcon_temp", 0.12)),
                )

        batch_proto_stats = None
        if (not stage1_ce_only) and _fed_proto_enabled(self.cfg) and torch.is_tensor(out.get("z_id")):
            terms["loss_fed_proto"], extra_metrics["fed_proto_cos"] = _fed_proto_pull_loss(
                out.get("z_id"),
                y,
                self.global_proto_stats,
                int(getattr(self.cfg, "fed_proto_min_count", 2)),
            )
            batch_proto_stats = _collect_fed_proto_stats(
                out.get("z_id"),
                y,
                d_loss,
                num_classes=int(logits.size(1)),
                num_domains=max(1, self._num_domains(d_loss)),
            )
        vmb_proto_stats = None
        if (not stage1_ce_only) and self.vmb_enabled and self.vmb_proto_bank is not None:
            tx_proto, tx_count = self.vmb_proto_bank.tx_tensors()
            if tx_proto is not None and tx_count is not None and torch.is_tensor(out.get("z_id")):
                loss_tx_proto, tx_metrics = prototype_contrastive_loss(
                    out.get("z_id"),
                    y,
                    tx_proto.to(self.device),
                    tx_count.to(self.device),
                    temperature=float(getattr(self.cfg, "tau_vmb_tx", 0.1)),
                    min_count=int(getattr(self.cfg, "fed_proto_min_count", 2)),
                )
                terms["loss_vmb_tx_proto"] = loss_tx_proto
                extra_metrics["vmb_tx_proto_cos"] = float(tx_metrics.get("target_cos", float("nan")))
                extra_metrics["vmb_tx_proto_active"] = float(tx_metrics.get("active_prototypes", 0))
            rx_proto, rx_count, rx_clients = self.vmb_proto_bank.rx_tensors()
            if rx_proto is not None and rx_count is not None and torch.is_tensor(out.get("z_dom")) and str(client_id) in rx_clients:
                target_idx = rx_clients.index(str(client_id))
                z_dom_for_rx = out.get("z_dom")
                if d_style is not None:
                    clean_style_id = int(getattr(self.virtual_domain_sampler, "clean_style_id", 0))
                    clean_mask = d_style.view(-1).long() == clean_style_id
                    if bool(clean_mask.any()):
                        z_dom_for_rx = z_dom_for_rx[clean_mask]
                rx_targets = torch.full((int(z_dom_for_rx.size(0)),), int(target_idx), device=self.device, dtype=torch.long)
                loss_rx_proto, rx_metrics = prototype_contrastive_loss(
                    z_dom_for_rx,
                    rx_targets,
                    rx_proto.to(self.device),
                    rx_count.to(self.device),
                    temperature=float(getattr(self.cfg, "tau_vmb_rx", 0.1)),
                    min_count=int(getattr(self.cfg, "fed_proto_min_count", 2)),
                )
                terms["loss_vmb_rx_proto"] = loss_rx_proto
                extra_metrics["vmb_rx_proto_cos"] = float(rx_metrics.get("target_cos", float("nan")))
                extra_metrics["vmb_rx_proto_active"] = float(rx_metrics.get("active_prototypes", 0))
            tx_adv_logits = out.get("tx_adv_logits")
            if torch.is_tensor(tx_adv_logits) and int(tx_adv_logits.size(0)) == int(y.numel()):
                terms["loss_tx_adv_r"] = F.cross_entropy(tx_adv_logits.float(), y)
                extra_metrics["tx_adv_r_acc"] = float((tx_adv_logits.argmax(dim=1) == y).float().mean().detach().item() * 100.0)
            vmb_proto_stats = build_prototype_stats(
                out.get("z_id")[: int(y_clean.numel())] if torch.is_tensor(out.get("z_id")) and d_style is not None else out.get("z_id"),
                out.get("z_dom")[: int(y_clean.numel())] if torch.is_tensor(out.get("z_dom")) and d_style is not None else out.get("z_dom"),
                y_clean if d_style is not None else y,
                client_id=str(client_id),
                num_classes=int(logits.size(1)),
            )
        proto_evidence_items = []
        if (not stage1_ce_only) and self.proto_evidence_bank is not None and torch.is_tensor(out.get("z_id")):
            proto_evidence_items = _collect_proto_evidence(
                out.get("z_id"),
                logits,
                y,
                d_style if d_style is not None else d_loss,
                client_id=str(client_id),
            )

        logit_anchor_stats = None
        if (not stage1_ce_only) and self.logit_anchor_bank is not None:
            anchor_logits, anchor_counts = self.logit_anchor_bank.tensors()
            terms["loss_logit_kd"], kd_metrics = logit_anchor_kd_loss(
                logits,
                y,
                anchor_logits.to(self.device),
                anchor_counts.to(self.device),
                temperature=float(getattr(self.cfg, "kd_temperature", 2.0)),
                min_count=int(getattr(self.cfg, "kd_min_count", 1)),
            )
            extra_metrics["kd_active"] = float(kd_metrics.get("kd_active", 0.0))
            extra_metrics["logit_anchor_count"] = float(kd_metrics.get("anchor_count", 0.0))
            logit_anchor_stats = build_logit_anchor_stats(
                logits.detach(),
                y.detach(),
                confidence_min=float(getattr(self.cfg, "kd_reliability_gate", 0.0)),
                margin_min=float(getattr(self.cfg, "kd_margin_min", 0.0)),
                require_correct=True,
            )
            extra_metrics["logit_anchor_payload_bytes"] = float(logit_anchor_stats_payload_size_bytes(logit_anchor_stats))

        coral_stats = None
        if getattr(self, "global_coral_bank", None) is not None:
            coral_feature_name = _coral_feature_name(self.cfg)
            coral_feature = self._coral_feature_from_outputs(out, coral_feature_name)
            num_classes = int(logits.size(1))
            clean_count = int(y_clean.numel())
            if torch.is_tensor(coral_feature):
                coral_collect = str(getattr(self.cfg, "fl_coral_collect_views", "clean") or "clean").lower()
                if d_style is not None and coral_collect == "clean":
                    coral_collect_feat = coral_feature[:clean_count]
                    coral_collect_y = y_clean
                else:
                    coral_collect_feat = coral_feature
                    coral_collect_y = y
                coral_stats = build_class_conditional_coral_stats(
                    coral_collect_feat.detach(),
                    coral_collect_y,
                    num_classes=num_classes,
                    mode=str(_coral_config_value(self.cfg, "fl_coral_cov_mode", "fed_coral_mode", "diag") or "diag"),
                )
            if self._fed_coral_stage_active(round_idx):
                min_count = int(_coral_config_value(self.cfg, "fl_coral_min_count", "fed_coral_min_count", 2))
                global_bank = getattr(self, "global_coral_bank", None)
                global_stats = global_bank.stats if global_bank is not None else None
                zid_weight = _coral_weight(self.cfg, "lambda_fl_coral_zid_global", "lambda_fed_coral")
                zdom_weight = float(getattr(self.cfg, "lambda_fl_coral_zdom_global", 0.0) or 0.0)
                virtual_weight = _coral_weight(self.cfg, "lambda_fl_coral_zid_virtual", "lambda_fed_coral_virtual")
                if zid_weight > 0.0 and torch.is_tensor(coral_feature) and str(coral_feature_name).lower() != "z_dom":
                    terms["loss_coral_zid_global"], coral_metrics = class_conditional_coral_loss(
                        coral_feature,
                        y,
                        global_stats,
                        min_count=min_count,
                        shrinkage=float(getattr(self.cfg, "fl_coral_shrinkage", 0.0) or 0.0),
                    )
                    extra_metrics["diag_coral_global_active"] = 1.0 if int(coral_metrics.get("active_classes", 0)) > 0 else 0.0
                    extra_metrics["coral_zid_global_active_classes"] = float(coral_metrics.get("active_classes", 0))
                    extra_metrics["coral_zid_global_mean_dist"] = float(coral_metrics.get("mean_dist", float("nan")))
                    extra_metrics["coral_zid_global_cov_dist"] = float(coral_metrics.get("cov_dist", float("nan")))
                    extra_metrics["coral_zid_global_skip_rate"] = float(coral_metrics.get("skip_rate", float("nan")))
                if zdom_weight > 0.0 and coral_feature_name == "z_dom" and torch.is_tensor(coral_feature):
                    terms["loss_coral_zdom_global"], coral_metrics = class_conditional_coral_loss(
                        coral_feature,
                        y,
                        global_stats,
                        min_count=min_count,
                        shrinkage=float(getattr(self.cfg, "fl_coral_shrinkage", 0.0) or 0.0),
                    )
                    extra_metrics["diag_coral_zdom_active"] = 1.0 if int(coral_metrics.get("active_classes", 0)) > 0 else 0.0
                    extra_metrics["coral_zdom_global_active_classes"] = float(coral_metrics.get("active_classes", 0))
                    extra_metrics["coral_zdom_global_mean_dist"] = float(coral_metrics.get("mean_dist", float("nan")))
                    extra_metrics["coral_zdom_global_cov_dist"] = float(coral_metrics.get("cov_dist", float("nan")))
                    extra_metrics["coral_zdom_global_skip_rate"] = float(coral_metrics.get("skip_rate", float("nan")))
                if virtual_weight > 0.0 and torch.is_tensor(out.get("z_id")):
                    virtual_losses = []
                    virtual_metric_items = []
                    if d_style is not None:
                        clean_style_id = int(getattr(self.virtual_domain_sampler, "clean_style_id", 0))
                        style_loss, style_metrics = self._compute_virtual_zid_coral(
                            out.get("z_id"),
                            y,
                            d_style,
                            clean_style_id,
                        )
                        if int(style_metrics.get("active_classes", 0) or 0) > 0:
                            virtual_losses.append(style_loss)
                        virtual_metric_items.append(style_metrics)
                    if out_sat_for_coral is not None and sat_y_for_coral is not None and torch.is_tensor(out_sat_for_coral.get("z_id")):
                        z_clean_for_sat = out.get("z_id")
                        if d_style is not None:
                            clean_count = int(y_clean.numel())
                            z_clean_for_sat = z_clean_for_sat[:clean_count]
                        sat_loss, sat_metrics = self._compute_pairwise_zid_coral(
                            z_clean_for_sat,
                            y_clean,
                            out_sat_for_coral.get("z_id"),
                            sat_y_for_coral,
                        )
                        if int(sat_metrics.get("active_classes", 0) or 0) > 0:
                            virtual_losses.append(sat_loss)
                        virtual_metric_items.append(sat_metrics)
                    if virtual_losses:
                        terms["loss_coral_zid_virtual"] = torch.stack(virtual_losses).mean()
                    coral_metrics = self._combine_coral_metric_dicts(virtual_metric_items)
                    extra_metrics["diag_coral_virtual_active"] = 1.0 if int(coral_metrics.get("active_classes", 0)) > 0 else 0.0
                    extra_metrics["coral_zid_virtual_active_classes"] = float(coral_metrics.get("active_classes", 0))
                    extra_metrics["coral_zid_virtual_mean_dist"] = float(coral_metrics.get("mean_dist", float("nan")))
                    extra_metrics["coral_zid_virtual_cov_dist"] = float(coral_metrics.get("cov_dist", float("nan")))
                    extra_metrics["coral_zid_virtual_skip_rate"] = float(coral_metrics.get("skip_rate", float("nan")))

        batch_fed_fishr_stats = None
        if self.fed_fishr_bank is not None:
            fishr_scope = str(getattr(self.cfg, "fed_fishr_gradient_scope", "classifier_head") or "classifier_head").lower().strip()
            fishr_feature = _select_generalization_feature(out, self.cfg)
            if fishr_scope in {"classifier_head", "head", "linear_head"} and not torch.is_tensor(fishr_feature):
                fishr_scope = "logit"
            fishr_logits = logits
            fishr_y = y
            if int(y.numel()) != int(y_clean.numel()) and int(logits.size(0)) >= int(y_clean.numel()):
                clean_count = int(y_clean.numel())
                fishr_logits = logits[:clean_count]
                fishr_y = y_clean
                if torch.is_tensor(fishr_feature) and int(fishr_feature.size(0)) >= clean_count:
                    fishr_feature = fishr_feature[:clean_count]
            try:
                batch_fed_fishr_stats = build_fed_fishr_stats(
                    fishr_logits,
                    fishr_y,
                    fishr_feature,
                    num_classes=int(logits.size(1)),
                    scope=fishr_scope,
                    min_count=int(getattr(self.cfg, "fed_fishr_min_count", 2)),
                    max_samples_per_class=int(getattr(self.cfg, "fed_fishr_max_samples_per_class", 0) or 0),
                    sketch_dim=int(getattr(self.cfg, "fed_fishr_sketch_dim", 0) or 0),
                    seed=int(getattr(self.cfg, "fed_fishr_seed", getattr(self.cfg, "seed", 0)) or 0),
                )
                active_mask = batch_fed_fishr_stats.get("active_mask")
                extra_metrics["fed_fishr_active_classes"] = float(int(active_mask.sum().item())) if torch.is_tensor(active_mask) else 0.0
                extra_metrics["fed_fishr_payload_bytes"] = float(int(batch_fed_fishr_stats.get("payload_bytes", 0) or 0))
                extra_metrics["fed_fishr_active"] = 1.0 if extra_metrics["fed_fishr_active_classes"] > 0 else 0.0
            except ValueError:
                batch_fed_fishr_stats = None
            target_var, target_mask = self.fed_fishr_bank.tensors()
            if (
                self.fed_fishr_mode in {"target_loss", "both"}
                and int(round_idx) >= int(getattr(self.cfg, "fed_fishr_start_round", 1) or 1)
                and torch.is_tensor(target_var)
                and torch.is_tensor(target_mask)
            ):
                extra_metrics["fed_fishr_target_ready"] = 1.0
                try:
                    terms["loss_fed_fishr"], fishr_metrics = fed_fishr_target_loss(
                        fishr_logits,
                        fishr_y,
                        fishr_feature,
                        target_var=target_var,
                        target_mask=target_mask,
                        scope=fishr_scope,
                        min_count=int(getattr(self.cfg, "fed_fishr_min_count", 2)),
                        max_samples_per_class=int(getattr(self.cfg, "fed_fishr_max_samples_per_class", 0) or 0),
                        sketch_dim=int(getattr(self.cfg, "fed_fishr_sketch_dim", 0) or 0),
                        seed=int(getattr(self.cfg, "fed_fishr_seed", getattr(self.cfg, "seed", 0)) or 0),
                    )
                    extra_metrics["fed_fishr_active_classes"] = float(fishr_metrics.get("active_classes", 0))
                    extra_metrics["fed_fishr_var_dist"] = float(fishr_metrics.get("mean_dist", float("nan")))
                    extra_metrics["fed_fishr_skip_rate"] = float(fishr_metrics.get("skip_rate", float("nan")))
                except ValueError:
                    terms["loss_fed_fishr"] = logits.new_tensor(0.0)

        activation_token_summary = None
        if (not stage1_ce_only) and self.activation_token_codec is not None:
            token_feature = self._split_feature_tensor(out)
            if torch.is_tensor(token_feature):
                token_packet = self.activation_token_codec.encode(token_feature.detach())
                activation_token_summary = {
                    "route": token_packet.route,
                    "split_layer": str(getattr(self.cfg, "split_layer", "z_id") or "z_id"),
                    "shape": [int(v) for v in token_packet.original_shape],
                    "payload_bytes": int(token_packet.payload_bytes),
                    "raw_bytes": int(token_packet.raw_bytes),
                    "compression_ratio": float(token_packet.compression_ratio),
                    "quant_bits": int(token_packet.quant_bits),
                    "quantization_error": float(token_packet.quantization_error),
                    "rank": int(token_packet.rank),
                    "sketch_dim": int(token_packet.sketch_dim),
                }
                extra_metrics["activation_token_payload_bytes"] = float(token_packet.payload_bytes)
                extra_metrics["activation_token_compression_ratio"] = float(token_packet.compression_ratio)
                extra_metrics["activation_token_quant_error"] = float(token_packet.quantization_error)

        total = terms["loss_cls"]
        weights = {
            "loss_rx_adv": "lambda_rx_adv",
            "loss_dom": "lambda_dom",
            "loss_adv": "lambda_adv",
            "loss_orth": "lambda_orth",
            "loss_cons": "lambda_cons",
            "loss_group_ce": "lambda_group_ce",
            "loss_fishr": "lambda_fishr",
            "loss_fed_fishr": "lambda_fed_fishr",
            "loss_sat_cls": "lambda_sat_cls",
            "loss_sat_cons": "lambda_sat_cons",
            "loss_fed_proto": "lambda_fed_proto",
            "loss_vmb_tx_proto": "lambda_vmb_tx_proto",
            "loss_vmb_rx_proto": "lambda_vmb_rx_proto",
            "loss_tx_adv_r": "lambda_tx_adv_r",
            "loss_logit_kd": "lambda_logit_kd",
            "loss_supcon_id": "lambda_supcon_id",
            "loss_coral_zid_global": "lambda_fl_coral_zid_global",
            "loss_coral_zid_virtual": "lambda_fl_coral_zid_virtual",
            "loss_coral_zdom_global": "lambda_fl_coral_zdom_global",
            "loss_domain_unsup_pretrain": "lambda_domain_unsup_pretrain",
            "loss_domain_unsup_metadata_ce": "lambda_domain_unsup_metadata_ce",
            "loss_domain_unsup_var": "lambda_domain_unsup_var",
        }
        for loss_name, weight_name in weights.items():
            fed_cgrl_overrides_rx = loss_name == "loss_rx_adv" and fed_cgrl_lambda_rx_adv is not None
            if fed_cgrl_overrides_rx:
                weight = float(fed_cgrl_lambda_rx_adv)
            elif loss_name == "loss_coral_zid_global":
                weight = _coral_weight(self.cfg, weight_name, "lambda_fed_coral")
            elif loss_name == "loss_coral_zid_virtual":
                weight = _coral_weight(self.cfg, weight_name, "lambda_fed_coral_virtual")
            else:
                weight = float(getattr(self.cfg, weight_name, 0.0))
            if loss_name in {"loss_rx_adv", "loss_adv", "loss_tx_adv_r"} and self.vmb_enabled and not fed_cgrl_overrides_rx:
                weight = adversarial_warmup_weight(
                    weight,
                    round_idx=int(round_idx),
                    warmup_rounds=int(getattr(self.cfg, "fl_vmb_adv_warmup_rounds", 0)),
                )
            total = total + weight * terms[loss_name]
        if use_baseline_sat_view and baseline_sat_ce_only:
            total = total + float(getattr(self.cfg, "fl_baseline_view_ce_weight", 1.0)) * terms["loss_baseline_sat_view"]
        if mu > 0.0:
            prox = compute_fedprox_loss(self.model, global_params, mu, exclude_for_prox)
            terms["loss_fedprox"] = prox
            total = total + prox
        else:
            terms["loss_fedprox"] = logits.new_tensor(0.0)
        terms["loss"] = total

        metrics = {k: float(v.detach().item()) for k, v in terms.items() if torch.is_tensor(v)}
        metrics.update(extra_metrics)
        metrics["loss"] = total
        metrics["tx_logits"] = logits
        metrics["_metric_y"] = y.detach()
        if batch_proto_stats is not None:
            metrics["_fed_proto_stats"] = batch_proto_stats
        if vmb_proto_stats is not None:
            metrics["_vmb_proto_stats"] = vmb_proto_stats
        if proto_evidence_items:
            metrics["_proto_evidence_items"] = proto_evidence_items
        if logit_anchor_stats is not None:
            metrics["_logit_anchor_stats"] = logit_anchor_stats
        if activation_token_summary is not None:
            metrics["_activation_token_summary"] = activation_token_summary
        if coral_stats is not None:
            metrics["_coral_stats"] = coral_stats
            metrics["coral_payload_bytes"] = float(coral_stats_payload_size_bytes(coral_stats))
        if batch_fed_fishr_stats is not None:
            metrics["_fed_fishr_stats"] = batch_fed_fishr_stats
        d_probe = style_targets if style_targets is not None else (d_loss if d_loss is not None else d_raw)
        feature_probe_item = self._feature_probe_item(
            client_id,
            out,
            y,
            d_probe,
            round_idx,
            batch_idx,
        )
        if feature_probe_item is not None:
            metrics["_feature_probe_item"] = feature_probe_item
            metrics["feature_probe_samples"] = float(feature_probe_item["tx"].numel())
        probe_style_batch = style_batch
        probe_style_targets = style_targets
        probe_out = out
        if probe_style_batch is None:
            probe_style_batch = self._build_forced_style_probe_batch(
                client_id,
                x_main,
                y_clean,
                d_raw_clean,
                round_idx,
                batch_idx,
            )
            if probe_style_batch is not None:
                probe_style_targets = self._style_target_domain_tensor(probe_style_batch)
                if probe_style_targets is not None:
                    probe_d_model = (
                        probe_style_targets
                        if self._style_domain_label_mode() == "target_receiver"
                        else probe_style_batch.d_style.to(self.device).long()
                    )
                    was_training = bool(self.model.training)
                    try:
                        self.model.eval()
                        with torch.no_grad():
                            probe_out = self._forward_outputs(
                                _safe_iq_tensor(probe_style_batch.x.to(self.device)),
                                probe_style_batch.y.to(self.device).long(),
                                probe_d_model,
                            )
                    finally:
                        self.model.train(was_training)
        style_zdom_probe = self._style_zdom_probe(
            client_id,
            probe_style_batch,
            probe_out,
            probe_style_targets,
            round_idx,
            batch_idx,
        )
        if style_zdom_probe:
            metrics["_style_zdom_probe"] = style_zdom_probe
        return metrics

    def _selected_clients(self, round_idx: int):
        client_ids = list(self.client_splits.keys())
        frac = float(getattr(self.cfg, "fl_clients_per_round", 1.0))
        k = len(client_ids) if frac >= 1.0 else max(1, int(math.ceil(len(client_ids) * max(0.0, frac))))
        if self.vmb_enabled and bool(getattr(self.cfg, "fl_vmb_domain_balanced_sampling", True)):
            return select_domain_balanced_clients(
                client_ids,
                self.vmb_client_domains,
                clients_per_round=k,
                seed=int(getattr(self.cfg, "seed", 0) or 0),
                round_idx=int(round_idx),
            )
        rng = random.Random(int(getattr(self.cfg, "seed", 0)) + int(round_idx) * 1009)
        return sorted(rng.sample(client_ids, k=k))

    def _round_lr(self, round_idx: int) -> float:
        lr = float(getattr(self.cfg, "lr", 2e-4))
        lr_min = float(getattr(self.cfg, "lr_min", lr))
        total_rounds = max(1, int(getattr(self.cfg, "fl_rounds", 1)))
        if total_rounds <= 1 or lr_min >= lr:
            return lr
        t = max(0.0, min(1.0, (int(round_idx) - 1) / float(max(1, total_rounds - 1))))
        return float(lr_min + 0.5 * (lr - lr_min) * (1.0 + math.cos(math.pi * t)))

    def _make_optimizer(self, model: nn.Module, round_idx: int):
        return torch.optim.AdamW(
            model.parameters(),
            lr=self._round_lr(round_idx),
            weight_decay=float(getattr(self.cfg, "wd", 1e-4)),
        )

    def _configure_mixstyle_for_round(self, round_idx: int) -> Dict[str, Any]:
        raw_model = getattr(self.model, "_orig_mod", self.model)
        id_backbone = getattr(raw_model, "id_backbone", raw_model)
        mix = getattr(id_backbone, "mixstyle", None)
        if mix is None:
            return {"enabled": False, "p": 0.0, "strength": 0.0, "phase": "missing", "anneal_t": 0.0}
        if not bool(getattr(self.cfg, "use_mixstyle", False)):
            setattr(id_backbone, "mixstyle_on", False)
            return {"enabled": False, "p": 0.0, "strength": 0.0, "phase": "disabled", "anneal_t": 0.0}

        late_start = int(getattr(self.cfg, "mixstyle_late_start", 0))
        if late_start <= 0:
            late_start = int(getattr(self.cfg, "late_stable_start", 0))
        ramp_epochs = int(getattr(self.cfg, "mixstyle_late_ramp_epochs", 0))
        if ramp_epochs <= 0:
            ramp_epochs = int(getattr(self.cfg, "late_stable_ramp_epochs", 1))
        state = compute_mixstyle_epoch_state(
            epoch=int(round_idx),
            base_p=float(getattr(self.cfg, "mixstyle_p", getattr(mix, "p", 0.0))),
            base_strength=float(getattr(self.cfg, "mixstyle_strength", getattr(mix, "strength", 0.0))),
            late_start=late_start,
            ramp_epochs=ramp_epochs,
            min_p=float(getattr(self.cfg, "mixstyle_late_min_p", -1.0)),
            min_strength=float(getattr(self.cfg, "mixstyle_late_min_strength", -1.0)),
            stop_epoch=int(getattr(self.cfg, "mixstyle_stop_epoch", 0)),
        )
        setattr(mix, "p", float(state["p"]))
        setattr(mix, "strength", float(state["strength"]))
        setattr(id_backbone, "mixstyle_on", bool(state["enabled"]))
        return dict(state)

    def _load_client_model(self, client_id: str):
        self.model.load_state_dict(self.global_state, strict=False)
        self.model.to(self.device)

    def _fed_cgrl_for_client(self, client_id: str, round_idx: int) -> tuple[Optional[float], Dict[str, Any]]:
        if not self.fed_cgrl.enabled:
            return None, {}
        decision = self.fed_cgrl.lambda_for_client(client_id, round_idx)
        metrics = decision.as_metrics()
        metrics["fed_cgrl_enabled"] = 1.0
        return float(decision.lambda_rx_adv), metrics

    def _fed_cgrl_empty_conflict_summary(self, source: str) -> Dict[str, Any]:
        return {
            "source": str(source or "none"),
            "conflict_mode": "diagnostic_only",
            "clients": 0,
            "gradient_keys": 0,
            "conflicts_detected": 0,
            "conflicts_resolved": 0,
            "grad_cos_pairs": 0,
            "grad_cos_mean_before": float("nan"),
            "grad_cos_min_before": float("nan"),
            "grad_cos_mean_after": float("nan"),
            "grad_cos_min_after": float("nan"),
            "conflict_signal_available": 0.0,
        }

    def _fed_cgrl_conflict_signal_available(self, summary: Optional[Mapping[str, Any]]) -> bool:
        if not summary:
            return False
        try:
            value = float(summary.get("grad_cos_min_before", float("nan")))
        except (TypeError, ValueError):
            return False
        return math.isfinite(value)

    def _fed_cgrl_conflict_summary_from_client_states(
        self,
        client_states: Mapping[str, Mapping[str, torch.Tensor]],
        selected: list[str],
    ) -> Dict[str, Any]:
        deltas: OrderedDict[str, OrderedDict[str, torch.Tensor]] = OrderedDict()
        for cid in selected:
            state = client_states.get(cid)
            if not state:
                continue
            client_delta: OrderedDict[str, torch.Tensor] = OrderedDict()
            for key, base_value in self.global_state.items():
                value = state.get(key)
                if not (torch.is_tensor(base_value) and torch.is_tensor(value)):
                    continue
                if tuple(value.shape) != tuple(base_value.shape):
                    continue
                if not (torch.is_floating_point(base_value) and torch.is_floating_point(value)):
                    continue
                client_delta[str(key)] = value.detach().cpu().float() - base_value.detach().cpu().float()
            if client_delta:
                deltas[str(cid)] = client_delta
        if len(deltas) < 2:
            return self._fed_cgrl_empty_conflict_summary("client_delta")
        common_keys = sorted(set.intersection(*(set(delta.keys()) for delta in deltas.values())))
        if not common_keys:
            return self._fed_cgrl_empty_conflict_summary("client_delta")
        filtered = OrderedDict(
            (cid, OrderedDict((key, delta[key]) for key in common_keys))
            for cid, delta in deltas.items()
        )
        cosine = gradient_cosine_summary(filtered)
        cos_min = float(cosine.get("min", float("nan")))
        return {
            "source": "client_delta",
            "conflict_mode": "diagnostic_only",
            "clients": int(len(filtered)),
            "gradient_keys": int(len(common_keys)),
            "conflicts_detected": int(1 if math.isfinite(cos_min) and cos_min < 0.0 else 0),
            "conflicts_resolved": 0,
            "grad_cos_pairs": int(cosine.get("pairs", 0)),
            "grad_cos_mean_before": float(cosine.get("mean", float("nan"))),
            "grad_cos_min_before": cos_min,
            "grad_cos_mean_after": float(cosine.get("mean", float("nan"))),
            "grad_cos_min_after": cos_min,
            "conflict_signal_available": 1.0 if math.isfinite(cos_min) else 0.0,
        }

    def _fed_cgrl_conflict_summary_from_vmb(
        self,
        vmb_conflict_summary: Mapping[str, Any],
        vmb_gradient_cosine: Mapping[str, Any],
    ) -> Dict[str, Any]:
        summary = dict(vmb_conflict_summary or {})
        if "grad_cos_min_before" not in summary and vmb_gradient_cosine:
            summary["grad_cos_pairs"] = int(vmb_gradient_cosine.get("pairs", 0))
            summary["grad_cos_mean_before"] = float(vmb_gradient_cosine.get("mean", float("nan")))
            summary["grad_cos_min_before"] = float(vmb_gradient_cosine.get("min", float("nan")))
            summary["grad_cos_mean_after"] = float(vmb_gradient_cosine.get("mean", float("nan")))
            summary["grad_cos_min_after"] = float(vmb_gradient_cosine.get("min", float("nan")))
        summary["source"] = "vmb"
        summary["conflict_signal_available"] = 1.0 if self._fed_cgrl_conflict_signal_available(summary) else 0.0
        return summary

    def _select_fed_cgrl_conflict_summary(
        self,
        *,
        client_delta_summary: Mapping[str, Any],
        vmb_conflict_summary: Mapping[str, Any],
        vmb_gradient_cosine: Mapping[str, Any],
        current_vmb_stage: str,
    ) -> Dict[str, Any]:
        source = str(getattr(self.cfg, "fed_cgrl_conflict_source", getattr(self.fed_cgrl, "conflict_source", "auto")) or "auto").lower()
        if source in {"", "none", "off"}:
            return self._fed_cgrl_empty_conflict_summary("none")
        if source == "client_delta":
            return dict(client_delta_summary or self._fed_cgrl_empty_conflict_summary("client_delta"))
        vmb_summary = self._fed_cgrl_conflict_summary_from_vmb(vmb_conflict_summary, vmb_gradient_cosine)
        if source == "vmb":
            return vmb_summary
        if self.vmb_enabled and str(current_vmb_stage) == "stage2" and self._fed_cgrl_conflict_signal_available(vmb_summary):
            return vmb_summary
        return dict(client_delta_summary or self._fed_cgrl_empty_conflict_summary("client_delta"))

    def train_one_client(self, client_id: str, round_idx: int):
        if self.vmb_enabled:
            if self._vmb_stage(round_idx) == "stage1":
                return self.train_one_client_vmb_stage1(client_id, round_idx)
            return self.train_one_client_vmb(client_id, round_idx)
        self._load_client_model(client_id)
        self.model.train()
        optimizer = self._make_optimizer(self.model, round_idx)
        global_params = {k: v.to(self.device) for k, v in self.global_state.items()}
        mu = float(getattr(self.cfg, "fedprox_mu", 0.0)) if self.train_mode == "fedprox" else 0.0

        total_loss = 0.0
        total_correct = 0
        total_seen = 0
        component_sums: Dict[str, float] = {}
        fed_cgrl_lambda, fed_cgrl_metrics = self._fed_cgrl_for_client(client_id, round_idx)
        label_hist: Optional[torch.Tensor] = None
        pred_hist: Optional[torch.Tensor] = None
        client_proto_stats = None
        client_coral_stats = None
        client_fed_fishr_stats = None
        client_logit_anchor_stats = None
        client_proto_evidence: list[ProtoEvidence] = []
        style_packets: list[StylePacket] = []
        style_zdom_probe = None
        activation_token_summary = None
        feature_probe_items: list[Mapping[str, Any]] = []
        for _local_epoch in range(max(1, int(getattr(self.cfg, "fl_local_epochs", 1)))):
            for batch_idx, batch in enumerate(self.client_loaders[client_id]):
                x, y, d_raw = _batch_to_xyd(batch, self.device)
                style_packet = self._extract_style_packet(client_id, round_idx, x, y, d_raw)
                if style_packet is not None:
                    style_packets.append(style_packet)
                optimizer.zero_grad(set_to_none=True)
                objective = self._compute_local_objective(
                    client_id,
                    x,
                    y,
                    d_raw,
                    round_idx,
                    batch_idx,
                    global_params,
                    mu,
                    set(),
                    fed_cgrl_lambda_rx_adv=fed_cgrl_lambda,
                    fed_cgrl_metrics=fed_cgrl_metrics,
                )
                client_proto_stats = _merge_fed_proto_stats(client_proto_stats, objective.pop("_fed_proto_stats", None))
                client_coral_stats = merge_coral_stats([client_coral_stats, objective.pop("_coral_stats", None)])
                client_fed_fishr_stats = merge_fed_fishr_stats(
                    [client_fed_fishr_stats, objective.pop("_fed_fishr_stats", None)],
                    min_count=int(getattr(self.cfg, "fed_fishr_min_count", 2)),
                )
                client_proto_evidence.extend(objective.pop("_proto_evidence_items", []) or [])
                client_logit_anchor_stats = merge_logit_anchor_stats(
                    [client_logit_anchor_stats, objective.pop("_logit_anchor_stats", None)]
                )
                activation_token_summary = objective.pop("_activation_token_summary", activation_token_summary)
                feature_probe_item = objective.pop("_feature_probe_item", None)
                if feature_probe_item is not None:
                    feature_probe_items.append(feature_probe_item)
                style_zdom_probe = _merge_style_zdom_probe(
                    style_zdom_probe,
                    objective.pop("_style_zdom_probe", None),
                    max_examples=int(getattr(self.cfg, "fl_style_zdom_probe_max_examples", 4) or 4),
                )
                logits = objective.pop("tx_logits")
                metric_y = objective.pop("_metric_y", None)
                loss = objective["loss"]
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(getattr(self.cfg, "grad_clip", 1.0)))
                optimizer.step()
                y_metric = metric_y.to(self.device).long() if torch.is_tensor(metric_y) else y
                if int(logits.size(0)) != int(y_metric.numel()) and int(y_metric.numel()) > 0:
                    repeat = int(logits.size(0)) // int(y_metric.numel())
                    if repeat > 1 and repeat * int(y_metric.numel()) == int(logits.size(0)):
                        y_metric = y_metric.repeat(repeat)
                bsz = int(y_metric.numel())
                total_loss += float(loss.detach().item()) * bsz
                pred = logits.argmax(dim=1)
                total_correct += int((pred == y_metric).sum().item())
                total_seen += bsz
                num_classes = int(logits.size(1))
                y_cpu = y_metric.detach().view(-1).cpu()
                pred_cpu = pred.detach().view(-1).cpu()
                if label_hist is None or int(label_hist.numel()) < num_classes:
                    new_label_hist = torch.zeros(num_classes, dtype=torch.long)
                    new_pred_hist = torch.zeros(num_classes, dtype=torch.long)
                    if label_hist is not None:
                        new_label_hist[: label_hist.numel()] = label_hist
                        new_pred_hist[: pred_hist.numel()] = pred_hist
                    label_hist = new_label_hist
                    pred_hist = new_pred_hist
                label_hist += torch.bincount(y_cpu.clamp_min(0), minlength=num_classes)[:num_classes]
                pred_hist += torch.bincount(pred_cpu.clamp_min(0), minlength=num_classes)[:num_classes]
                for k, v in objective.items():
                    if _is_train_component_metric(k):
                        if isinstance(v, float) and math.isfinite(v):
                            component_sums[k] = component_sums.get(k, 0.0) + float(v) * bsz
                        elif isinstance(v, (int, float)):
                            continue

        state = OrderedDict((k, v.detach().cpu().clone()) for k, v in self.model.state_dict().items())
        result = {
            "state": state,
            "loss": total_loss / max(1, total_seen),
            "acc": 100.0 * total_correct / max(1, total_seen),
            "seen": total_seen,
            "label_hist": label_hist.tolist() if label_hist is not None else [],
            "pred_hist": pred_hist.tolist() if pred_hist is not None else [],
            "style_domain_label_mode": self._style_domain_label_mode(),
        }
        if client_proto_stats is not None:
            result["fed_proto_stats"] = client_proto_stats
        if client_coral_stats is not None:
            result["coral_stats"] = client_coral_stats
        if client_fed_fishr_stats is not None:
            result["fed_fishr_stats"] = client_fed_fishr_stats
        if client_logit_anchor_stats is not None:
            result["logit_anchor_stats"] = client_logit_anchor_stats
        if client_proto_evidence:
            result["proto_evidence_items"] = client_proto_evidence
        if activation_token_summary is not None:
            result["activation_token_summary"] = activation_token_summary
        if style_packets:
            result["style_packets"] = style_packets
            result["style_packet_payload_bytes"] = self._style_packet_payload_bytes(style_packets)
        if feature_probe_items:
            result["feature_probe_items"] = feature_probe_items
        if style_zdom_probe:
            result["style_zdom_probe"] = style_zdom_probe
            result.update(_style_zdom_probe_flat_metrics(style_zdom_probe))
        for k, v in sorted(component_sums.items()):
            result[k] = v / max(1, total_seen)
        return result

    def train_one_client_vmb_stage1(self, client_id: str, round_idx: int):
        self._load_client_model(client_id)
        self.model.train()
        stage = self._vmb_stage(round_idx)
        active_param_names = set(self._vmb_set_trainability(stage))
        global_params = {k: v.to(self.device) for k, v in self.global_state.items()}
        opt = self._make_optimizer(self.model, round_idx)
        stage1_lr_mult = float(getattr(self.cfg, "fl_vmb_stage1_lr_mult", 1.0))
        if math.isfinite(stage1_lr_mult) and stage1_lr_mult > 0.0 and stage1_lr_mult != 1.0:
            for group in opt.param_groups:
                group["lr"] = float(group.get("lr", self._round_lr(round_idx))) * stage1_lr_mult

        total_loss = 0.0
        total_correct = 0
        total_seen = 0
        component_sums: Dict[str, float] = {}
        fed_cgrl_lambda, fed_cgrl_metrics = self._fed_cgrl_for_client(client_id, round_idx)
        label_hist: Optional[torch.Tensor] = None
        pred_hist: Optional[torch.Tensor] = None
        client_vmb_proto_stats = None
        client_coral_stats = None
        client_fed_fishr_stats = None
        client_logit_anchor_stats = None
        client_proto_evidence: list[ProtoEvidence] = []
        activation_token_summary = None
        style_packets: list[StylePacket] = []
        style_zdom_probe = None
        feature_probe_items: list[Mapping[str, Any]] = []
        local_steps = max(1, int(getattr(self.cfg, "fl_vmb_stage1_local_steps", getattr(self.cfg, "fl_local_epochs", 1))))
        loader_iter = iter(self.client_loaders[client_id])
        steps_used = 0

        for local_batch_idx in range(local_steps):
            balanced = self._vmb_build_balanced_batch(client_id, round_idx, local_batch_idx)
            if balanced is None:
                try:
                    batch = next(loader_iter)
                except StopIteration:
                    break
                x, y, d_raw = _batch_to_xyd(batch, self.device)
            else:
                x, y, d_raw = balanced
            style_packet = self._extract_style_packet(client_id, round_idx, x, y, d_raw)
            if style_packet is not None:
                style_packets.append(style_packet)
            opt.zero_grad(set_to_none=True)
            objective = self._compute_local_objective(
                client_id,
                x,
                y,
                d_raw,
                round_idx,
                local_batch_idx,
                global_params,
                0.0,
                set(),
                fed_cgrl_lambda_rx_adv=fed_cgrl_lambda,
                fed_cgrl_metrics=fed_cgrl_metrics,
            )
            client_vmb_proto_stats = merge_prototype_stats(
                [client_vmb_proto_stats, objective.pop("_vmb_proto_stats", None)]
            )
            client_coral_stats = merge_coral_stats([client_coral_stats, objective.pop("_coral_stats", None)])
            client_fed_fishr_stats = merge_fed_fishr_stats(
                [client_fed_fishr_stats, objective.pop("_fed_fishr_stats", None)],
                min_count=int(getattr(self.cfg, "fed_fishr_min_count", 2)),
            )
            client_proto_evidence.extend(objective.pop("_proto_evidence_items", []) or [])
            objective.pop("_fed_proto_stats", None)
            client_logit_anchor_stats = merge_logit_anchor_stats(
                [client_logit_anchor_stats, objective.pop("_logit_anchor_stats", None)]
            )
            activation_token_summary = objective.pop("_activation_token_summary", activation_token_summary)
            feature_probe_item = objective.pop("_feature_probe_item", None)
            if feature_probe_item is not None:
                feature_probe_items.append(feature_probe_item)
            style_zdom_probe = _merge_style_zdom_probe(
                style_zdom_probe,
                objective.pop("_style_zdom_probe", None),
                max_examples=int(getattr(self.cfg, "fl_style_zdom_probe_max_examples", 4) or 4),
            )
            logits = objective.pop("tx_logits")
            metric_y = objective.pop("_metric_y", None)
            loss = objective["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad and p.grad is not None],
                float(getattr(self.cfg, "grad_clip", 1.0)),
            )
            opt.step()
            steps_used += 1

            y_metric = metric_y.to(self.device).long() if torch.is_tensor(metric_y) else y
            if int(logits.size(0)) != int(y_metric.numel()) and int(y_metric.numel()) > 0:
                repeat = int(logits.size(0)) // int(y_metric.numel())
                if repeat > 1 and repeat * int(y_metric.numel()) == int(logits.size(0)):
                    y_metric = y_metric.repeat(repeat)
            bsz = int(y_metric.numel())
            total_loss += float(loss.detach().item()) * bsz
            pred = logits.argmax(dim=1)
            total_correct += int((pred == y_metric).sum().item())
            total_seen += bsz
            num_classes = int(logits.size(1))
            y_cpu = y_metric.detach().view(-1).cpu()
            pred_cpu = pred.detach().view(-1).cpu()
            if label_hist is None or int(label_hist.numel()) < num_classes:
                new_label_hist = torch.zeros(num_classes, dtype=torch.long)
                new_pred_hist = torch.zeros(num_classes, dtype=torch.long)
                if label_hist is not None:
                    new_label_hist[: label_hist.numel()] = label_hist
                    new_pred_hist[: pred_hist.numel()] = pred_hist
                label_hist = new_label_hist
                pred_hist = new_pred_hist
            label_hist += torch.bincount(y_cpu.clamp_min(0), minlength=num_classes)[:num_classes]
            pred_hist += torch.bincount(pred_cpu.clamp_min(0), minlength=num_classes)[:num_classes]
            for k, v in objective.items():
                if _is_train_component_metric(k):
                    if isinstance(v, float) and math.isfinite(v):
                        component_sums[k] = component_sums.get(k, 0.0) + float(v) * bsz

        state = OrderedDict((k, v.detach().cpu().clone()) for k, v in self.model.state_dict().items())
        drift_sq = 0.0
        update_bytes = 0
        for name, value in state.items():
            if torch.is_tensor(value):
                update_bytes += int(value.numel()) * int(value.element_size())
            if name not in active_param_names:
                continue
            ref = self.global_state.get(name)
            if torch.is_tensor(value) and torch.is_tensor(ref) and torch.is_floating_point(value):
                diff = value.float() - ref.detach().cpu().float()
                drift_sq += float(torch.sum(diff * diff).item())
        self._vmb_restore_trainability()

        result = {
            "state": state,
            "loss": total_loss / max(1, total_seen),
            "acc": 100.0 * total_correct / max(1, total_seen),
            "seen": total_seen,
            "label_hist": label_hist.tolist() if label_hist is not None else [],
            "pred_hist": pred_hist.tolist() if pred_hist is not None else [],
            "style_domain_label_mode": self._style_domain_label_mode(),
            "vmb_stage": stage,
            "vmb_batches_used": steps_used,
            "vmb_stage1_objective": self._vmb_stage1_objective(),
            "vmb_stage1_use_aux_losses": bool(getattr(self.cfg, "fl_vmb_stage1_use_aux_losses", False)),
            "vmb_stage1_lr_mult": stage1_lr_mult,
            "vmb_domain_id": self._vmb_domain_id(client_id),
            "vmb_client_drift_norm": math.sqrt(max(0.0, drift_sq)),
            "vmb_client_payload_bytes": (
                update_bytes
                + prototype_stats_payload_size_bytes(client_vmb_proto_stats)
                + coral_stats_payload_size_bytes(client_coral_stats)
                + logit_anchor_stats_payload_size_bytes(client_logit_anchor_stats)
                + int((activation_token_summary or {}).get("payload_bytes", 0) or 0)
                + self._style_packet_payload_bytes(style_packets)
            ),
        }
        if client_vmb_proto_stats is not None:
            result["vmb_proto_stats"] = client_vmb_proto_stats
        if client_coral_stats is not None:
            result["coral_stats"] = client_coral_stats
        if client_fed_fishr_stats is not None:
            result["fed_fishr_stats"] = client_fed_fishr_stats
        if client_logit_anchor_stats is not None:
            result["logit_anchor_stats"] = client_logit_anchor_stats
        if client_proto_evidence:
            result["proto_evidence_items"] = client_proto_evidence
        if activation_token_summary is not None:
            result["activation_token_summary"] = activation_token_summary
        if style_packets:
            result["style_packets"] = style_packets
            result["style_packet_payload_bytes"] = self._style_packet_payload_bytes(style_packets)
        if feature_probe_items:
            result["feature_probe_items"] = feature_probe_items
        if style_zdom_probe:
            result["style_zdom_probe"] = style_zdom_probe
            result.update(_style_zdom_probe_flat_metrics(style_zdom_probe))
        for k, v in sorted(component_sums.items()):
            result[k] = v / max(1, total_seen)
        return result

    def train_one_client_vmb(self, client_id: str, round_idx: int):
        self._load_client_model(client_id)
        self.model.train()
        stage = self._vmb_stage(round_idx)
        active_param_names = set(self._vmb_set_trainability(stage))
        global_params = {k: v.to(self.device) for k, v in self.global_state.items()}

        total_loss = 0.0
        total_correct = 0
        total_seen = 0
        component_sums: Dict[str, float] = {}
        fed_cgrl_lambda, fed_cgrl_metrics = self._fed_cgrl_for_client(client_id, round_idx)
        label_hist: Optional[torch.Tensor] = None
        pred_hist: Optional[torch.Tensor] = None
        client_vmb_proto_stats = None
        client_coral_stats = None
        client_fed_fishr_stats = None
        client_logit_anchor_stats = None
        client_proto_evidence: list[ProtoEvidence] = []
        activation_token_summary = None
        style_packets: list[StylePacket] = []
        style_zdom_probe = None
        feature_probe_items: list[Mapping[str, Any]] = []
        max_batches = max(1, int(getattr(self.cfg, "fl_vmb_batches_per_client", 1)))

        self.model.zero_grad(set_to_none=True)
        loader_iter = iter(self.client_loaders[client_id])
        batches_used = 0
        for local_batch_idx in range(max_batches):
            balanced = self._vmb_build_balanced_batch(client_id, round_idx, local_batch_idx)
            if balanced is None:
                try:
                    batch = next(loader_iter)
                except StopIteration:
                    break
                x, y, d_raw = _batch_to_xyd(batch, self.device)
            else:
                x, y, d_raw = balanced
            style_packet = self._extract_style_packet(client_id, round_idx, x, y, d_raw)
            if style_packet is not None:
                style_packets.append(style_packet)
            objective = self._compute_local_objective(
                client_id,
                x,
                y,
                d_raw,
                round_idx,
                local_batch_idx,
                global_params,
                0.0,
                set(),
                fed_cgrl_lambda_rx_adv=fed_cgrl_lambda,
                fed_cgrl_metrics=fed_cgrl_metrics,
            )
            client_vmb_proto_stats = merge_prototype_stats(
                [client_vmb_proto_stats, objective.pop("_vmb_proto_stats", None)]
            )
            client_coral_stats = merge_coral_stats([client_coral_stats, objective.pop("_coral_stats", None)])
            client_fed_fishr_stats = merge_fed_fishr_stats(
                [client_fed_fishr_stats, objective.pop("_fed_fishr_stats", None)],
                min_count=int(getattr(self.cfg, "fed_fishr_min_count", 2)),
            )
            client_proto_evidence.extend(objective.pop("_proto_evidence_items", []) or [])
            objective.pop("_fed_proto_stats", None)
            client_logit_anchor_stats = merge_logit_anchor_stats(
                [client_logit_anchor_stats, objective.pop("_logit_anchor_stats", None)]
            )
            activation_token_summary = objective.pop("_activation_token_summary", activation_token_summary)
            feature_probe_item = objective.pop("_feature_probe_item", None)
            if feature_probe_item is not None:
                feature_probe_items.append(feature_probe_item)
            style_zdom_probe = _merge_style_zdom_probe(
                style_zdom_probe,
                objective.pop("_style_zdom_probe", None),
                max_examples=int(getattr(self.cfg, "fl_style_zdom_probe_max_examples", 4) or 4),
            )
            logits = objective.pop("tx_logits")
            metric_y = objective.pop("_metric_y", None)
            loss = objective["loss"]
            (loss / float(max_batches)).backward()
            batches_used += 1

            y_metric = metric_y.to(self.device).long() if torch.is_tensor(metric_y) else y
            if int(logits.size(0)) != int(y_metric.numel()) and int(y_metric.numel()) > 0:
                repeat = int(logits.size(0)) // int(y_metric.numel())
                if repeat > 1 and repeat * int(y_metric.numel()) == int(logits.size(0)):
                    y_metric = y_metric.repeat(repeat)
            bsz = int(y_metric.numel())
            total_loss += float(loss.detach().item()) * bsz
            pred = logits.argmax(dim=1)
            total_correct += int((pred == y_metric).sum().item())
            total_seen += bsz
            num_classes = int(logits.size(1))
            y_cpu = y_metric.detach().view(-1).cpu()
            pred_cpu = pred.detach().view(-1).cpu()
            if label_hist is None or int(label_hist.numel()) < num_classes:
                new_label_hist = torch.zeros(num_classes, dtype=torch.long)
                new_pred_hist = torch.zeros(num_classes, dtype=torch.long)
                if label_hist is not None:
                    new_label_hist[: label_hist.numel()] = label_hist
                    new_pred_hist[: pred_hist.numel()] = pred_hist
                label_hist = new_label_hist
                pred_hist = new_pred_hist
            label_hist += torch.bincount(y_cpu.clamp_min(0), minlength=num_classes)[:num_classes]
            pred_hist += torch.bincount(pred_cpu.clamp_min(0), minlength=num_classes)[:num_classes]
            for k, v in objective.items():
                if _is_train_component_metric(k):
                    if isinstance(v, float) and math.isfinite(v):
                        component_sums[k] = component_sums.get(k, 0.0) + float(v) * bsz
                    elif isinstance(v, (int, float)):
                        continue

        if 0 < batches_used < max_batches:
            scale = float(max_batches) / float(batches_used)
            for param in self.model.parameters():
                if param.grad is not None:
                    param.grad.mul_(scale)
        torch.nn.utils.clip_grad_norm_(
            [p for p in self.model.parameters() if p.requires_grad and p.grad is not None],
            float(getattr(self.cfg, "grad_clip", 1.0)),
        )
        grads = OrderedDict()
        for name, param in self.model.named_parameters():
            if name not in active_param_names or param.grad is None:
                continue
            grads[name] = param.grad.detach().cpu().clone()
        self._vmb_restore_trainability()

        result = {
            "grads": grads,
            "loss": total_loss / max(1, total_seen),
            "acc": 100.0 * total_correct / max(1, total_seen),
            "seen": total_seen,
            "label_hist": label_hist.tolist() if label_hist is not None else [],
            "pred_hist": pred_hist.tolist() if pred_hist is not None else [],
            "style_domain_label_mode": self._style_domain_label_mode(),
            "vmb_stage": stage,
            "vmb_batches_used": batches_used,
            "vmb_domain_id": self._vmb_domain_id(client_id),
        }
        result["vmb_client_drift_norm"] = gradient_norm(grads)
        result["vmb_client_payload_bytes"] = (
            gradient_payload_size_bytes(grads)
            + prototype_stats_payload_size_bytes(client_vmb_proto_stats)
            + coral_stats_payload_size_bytes(client_coral_stats)
            + logit_anchor_stats_payload_size_bytes(client_logit_anchor_stats)
            + int((activation_token_summary or {}).get("payload_bytes", 0) or 0)
            + self._style_packet_payload_bytes(style_packets)
        )
        if client_vmb_proto_stats is not None:
            result["vmb_proto_stats"] = client_vmb_proto_stats
        if client_coral_stats is not None:
            result["coral_stats"] = client_coral_stats
        if client_fed_fishr_stats is not None:
            result["fed_fishr_stats"] = client_fed_fishr_stats
        if client_logit_anchor_stats is not None:
            result["logit_anchor_stats"] = client_logit_anchor_stats
        if client_proto_evidence:
            result["proto_evidence_items"] = client_proto_evidence
        if activation_token_summary is not None:
            result["activation_token_summary"] = activation_token_summary
        if style_packets:
            result["style_packets"] = style_packets
            result["style_packet_payload_bytes"] = self._style_packet_payload_bytes(style_packets)
        if feature_probe_items:
            result["feature_probe_items"] = feature_probe_items
        if style_zdom_probe:
            result["style_zdom_probe"] = style_zdom_probe
            result.update(_style_zdom_probe_flat_metrics(style_zdom_probe))
        for k, v in sorted(component_sums.items()):
            result[k] = v / max(1, total_seen)
        return result

    def _evaluate(self, round_idx: int):
        eval_t0 = time.perf_counter()
        self.model.load_state_dict(self.global_state, strict=False)
        self.model.to(self.device)
        val_stats = {}
        named_stats = {}
        extra_tests: Dict[str, Any] = {}
        val_time_s = 0.0
        test_time_s = 0.0
        extra_eval_time_s = 0.0
        proto_fusion_time_s = 0.0
        style_collab_time_s = 0.0
        heavy_eval_ran = self._should_run_heavy_eval(round_idx)
        self._current_eval_round = int(round_idx)
        if self.evaluate_loader_fn is not None and self.val_loader is not None:
            val_t0 = time.perf_counter()
            val_stats = self.evaluate_loader_fn(
                self.model,
                self.val_loader,
                self.device,
                domain_label_map=self.domain_label_map,
                max_batches=int(getattr(self.cfg, "eval_max_batches", 0)),
            )
            val_time_s = time.perf_counter() - val_t0
        if heavy_eval_ran and self.evaluate_named_loaders_fn is not None and self.named_test_loaders:
            test_t0 = time.perf_counter()
            named_stats = self.evaluate_named_loaders_fn(
                self.model,
                self.named_test_loaders,
                self.device,
                domain_label_map=self.domain_label_map,
                max_batches=int(getattr(self.cfg, "eval_max_batches", 0)),
            )
            test_time_s = time.perf_counter() - test_t0
        if heavy_eval_ran and self.extra_eval_fn is not None:
            extra_eval_t0 = time.perf_counter()
            extra_tests = self.extra_eval_fn(self.model, self.device, round_idx) or {}
            extra_eval_time_s = time.perf_counter() - extra_eval_t0
        proto_fusion = {}
        style_collab_fusion = {}
        if heavy_eval_ran:
            proto_fusion_t0 = time.perf_counter()
            proto_fusion = self._evaluate_proto_fusion()
            proto_fusion_time_s = time.perf_counter() - proto_fusion_t0
            style_collab_t0 = time.perf_counter()
            style_collab_fusion = self._evaluate_style_collab_fusion()
            style_collab_time_s = time.perf_counter() - style_collab_t0
        eval_time_s = time.perf_counter() - eval_t0
        test_overall = _aggregate_named_test_stats(named_stats) if named_stats else {"tx_acc": float("nan"), "tx_correct": 0, "tx_total": 0}
        strict = float(named_stats.get("test_unseen_day_unseen_rx", {}).get("tx_acc", float("nan"))) if named_stats else float("nan")
        return {
            "test_eval_ran": bool(heavy_eval_ran),
            "next_test_eval_round": self._next_heavy_eval_round(round_idx) if not heavy_eval_ran else None,
            "val": val_stats,
            "named": named_stats,
            "named_tx_acc": _named_tx_acc_summary(named_stats),
            "test_overall": test_overall,
            "strict_udu_acc": strict,
            "extra_tests": extra_tests,
            "proto_fusion": proto_fusion,
            "style_collab_fusion": style_collab_fusion,
            "timing": {
                "eval_time_s": eval_time_s,
                "val_time_s": val_time_s,
                "test_time_s": test_time_s,
                "extra_eval_time_s": extra_eval_time_s,
                "proto_fusion_time_s": proto_fusion_time_s,
                "style_collab_time_s": style_collab_time_s,
            },
        }

    def _save_checkpoint(self, path: str, round_idx: int, metrics: Mapping[str, Any]):
        payload = {
            "round": int(round_idx),
            "global_shared_state": self.global_state,
            "cfg": vars(self.cfg) if hasattr(self.cfg, "__dict__") else dict(self.cfg),
            "metrics": dict(metrics),
            "client_splits_summary": summarize_client_splits(self.train_dataset, self.client_splits),
        }
        torch.save(payload, path)

    def _build_config_snapshot(self) -> Dict[str, Any]:
        cfg = self.cfg
        client_summary = summarize_client_splits(self.train_dataset, self.client_splits)
        train_len = _safe_len(self.train_dataset)
        val_batches = _safe_len(self.val_loader)
        domain_values = sorted({int(v) for v in self.domain_label_map.values()}) if self.domain_label_map else []
        num_domains = (max(domain_values) + 1) if domain_values else int(getattr(cfg, "num_domains", 0) or 0)
        sat_train_scenarios = getattr(cfg, "sat_train_scenario_list", None)
        if sat_train_scenarios is None:
            raw_scenarios = str(getattr(cfg, "sat_train_scenarios", "") or "").strip()
            sat_train_scenarios = [s.strip() for s in raw_scenarios.split(",") if s.strip()]
            if not sat_train_scenarios:
                sat_train_scenarios = [str(getattr(cfg, "sat_train_scenario", "mixed_orbit"))]
        eval_sat_scenarios = getattr(cfg, "eval_sat_scenario_list", None)
        if eval_sat_scenarios is None:
            raw_eval = str(getattr(cfg, "eval_sat_scenarios", "") or "").strip()
            eval_sat_scenarios = [s.strip() for s in raw_eval.split(",") if s.strip()]

        return {
            "event": "fed_config",
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "run": {
                "run_name": getattr(cfg, "run_name", ""),
                "output_dir": self.output_dir,
                "log_dir": self.log_dir,
                "logs_jsonl": self.logs_jsonl,
                "metrics_csv": self.metrics_csv,
                "config_json": self.config_json,
                "summary_json": self.summary_json,
                "device": str(self.device),
                "seed": int(getattr(cfg, "seed", 0) or 0),
            },
            "runtime": snapshot_thread_runtime(),
            "data": {
                "dataset": getattr(cfg, "dataset", ""),
                "wisig_pkl": getattr(cfg, "wisig_pkl", ""),
                "wisig_domain": getattr(cfg, "wisig_domain", ""),
                "wisig_train_ratio": getattr(cfg, "wisig_train_ratio", None),
                "wisig_train_days": getattr(cfg, "wisig_train_days", ""),
                "wisig_test_days": getattr(cfg, "wisig_test_days", ""),
                "wisig_train_rxs": getattr(cfg, "wisig_train_rxs", ""),
                "wisig_test_rxs": getattr(cfg, "wisig_test_rxs", ""),
                "train_samples": train_len,
                "val_batches": val_batches,
                "named_tests": list(self.named_test_loaders.keys()),
                "named_test_meta": self.named_test_meta,
                "split_info": self.split_info,
                "domain_label_map": self.domain_label_map,
                "num_domains": num_domains,
            },
            "model": _model_parameter_summary(self.model),
            "federated": {
                "train_mode": self.train_mode,
                "fl_local_objective": str(getattr(cfg, "fl_local_objective", "ce")),
                "fl_rounds": int(getattr(cfg, "fl_rounds", 100)),
                "fl_local_epochs": int(getattr(cfg, "fl_local_epochs", 1)),
                "fl_clients_per_round": getattr(cfg, "fl_clients_per_round", 1.0),
                "fl_client_key": str(getattr(cfg, "fl_client_key", "receiver_day")),
                "fl_agg_weight": str(getattr(cfg, "fl_agg_weight", "num_samples")),
                "fl_min_samples_per_client": int(getattr(cfg, "fl_min_samples_per_client", 1)),
                "fl_drop_small_clients": bool(getattr(cfg, "fl_drop_small_clients", False)),
                "fl_local_exclude_keys": str(getattr(cfg, "fl_local_exclude_keys", "") or ""),
                "fl_local_exclude_prefixes": str(getattr(cfg, "fl_local_exclude_prefixes", "") or ""),
                "batch_size": int(getattr(cfg, "batch_size", 128)),
                "num_workers": int(getattr(cfg, "num_workers", 0)),
                "fl_num_workers": int(getattr(cfg, "fl_num_workers", 0)),
                "lr": float(getattr(cfg, "lr", 0.0)),
                "wd": float(getattr(cfg, "wd", 0.0)),
                "grad_clip": float(getattr(cfg, "grad_clip", 0.0)),
                "fedprox_mu": float(getattr(cfg, "fedprox_mu", 0.0)),
            },
            "client_splits": client_summary,
            "stylebank": {
                "enabled": bool(self.style_bank is not None),
                "use_fed_style_bank": bool(getattr(cfg, "use_fed_style_bank", _style_bank_enabled(cfg))),
                "use_fl_style_bank_stats": bool(getattr(cfg, "use_fl_style_bank_stats", _style_bank_enabled(cfg))),
                "style_extractor": self.style_extractor.__class__.__name__ if self.style_extractor is not None else None,
                "style_transform": self.style_transform.__class__.__name__ if self.style_transform is not None else None,
                "style_domain_semantics": "constructed_style_view_id",
                "d_style_semantics": "0=clean_view,1..K=remote_or_sat_style_view_order",
                "d_raw_semantics": "mapped_target_receiver_domain_for_each_view",
                "fl_style_domain_label_mode": self._style_domain_label_mode(),
                "fl_style_zdom_probe_every": int(getattr(cfg, "fl_style_zdom_probe_every", 0)),
                "fl_style_zdom_probe_force_batch": bool(getattr(cfg, "fl_style_zdom_probe_force_batch", False)),
                "fl_style_zdom_probe_real_samples": int(getattr(cfg, "fl_style_zdom_probe_real_samples", 0)),
                "fl_style_zdom_probe_max_examples": int(getattr(cfg, "fl_style_zdom_probe_max_examples", 4)),
                "fl_style_sampling_policy": str(getattr(cfg, "fl_style_sampling_policy", "diverse") or "diverse"),
                "fl_style_transform_mix_alpha": float(getattr(cfg, "fl_style_transform_mix_alpha", 1.0)),
                "fl_style_real_mix_samples": int(getattr(cfg, "fl_style_real_mix_samples", 0)),
                "fl_style_real_mix_start_round": int(getattr(cfg, "fl_style_real_mix_start_round", 0)),
                "fl_style_code_dim": int(getattr(cfg, "fl_style_code_dim", 0)),
                "clean_style_id": int(getattr(self.virtual_domain_sampler, "clean_style_id", 0)),
                "fl_style_replay_start_round": int(getattr(cfg, "fl_style_replay_start_round", 20)),
                "fl_style_phys_start_round": int(getattr(cfg, "fl_style_phys_start_round", 20)),
                "fl_style_dg_start_round": int(getattr(cfg, "fl_style_dg_start_round", 40)),
                "fl_style_dg_min_domains": int(getattr(cfg, "fl_style_dg_min_domains", 2)),
                "style_gate_min_accept_rate": float(getattr(cfg, "style_gate_min_accept_rate", 0.0)),
                "fl_style_min_remote_centroids": int(getattr(cfg, "fl_style_min_remote_centroids", 1)),
                "fl_style_max_views": int(getattr(cfg, "fl_style_max_views", 1)),
                "fl_style_replay_prob": float(getattr(cfg, "fl_style_replay_prob", 0.25)),
                "fl_style_phys_max_gain_delta": float(getattr(cfg, "fl_style_phys_max_gain_delta", 0.05)),
                "fl_style_phys_max_noise_std": float(getattr(cfg, "fl_style_phys_max_noise_std", 0.01)),
                "fl_style_phys_jitter_scale": float(getattr(cfg, "fl_style_phys_jitter_scale", 0.25)),
                "fl_style_phys_max_cfo_hz": float(getattr(cfg, "fl_style_phys_max_cfo_hz", 5000.0)),
                "fl_style_phys_max_sro_ppm": float(getattr(cfg, "fl_style_phys_max_sro_ppm", 25.0)),
                "fl_style_phys_max_iq_gain_db": float(getattr(cfg, "fl_style_phys_max_iq_gain_db", 0.5)),
                "fl_style_phys_max_iq_phase_deg": float(getattr(cfg, "fl_style_phys_max_iq_phase_deg", 0.5)),
                "fl_style_phys_max_phase_noise_std": float(getattr(cfg, "fl_style_phys_max_phase_noise_std", 0.0005)),
                "fl_style_phys_min_awgn_snr_db": float(getattr(cfg, "fl_style_phys_min_awgn_snr_db", 20.0)),
                "fl_style_phys_p_lowpass": float(getattr(cfg, "fl_style_phys_p_lowpass", 0.2)),
                "fl_style_phys_p_multipath": float(getattr(cfg, "fl_style_phys_p_multipath", 0.2)),
                "fl_style_phys_max_multipath_taps": int(getattr(cfg, "fl_style_phys_max_multipath_taps", 3)),
                "use_fed_style_sat_view": bool(getattr(cfg, "use_fed_style_sat_view", False)),
                "fl_style_bank_momentum": float(getattr(cfg, "fl_style_bank_momentum", 0.5)),
                "fl_style_bank_max_centroids": int(getattr(cfg, "fl_style_bank_max_centroids", 64)),
                "fl_style_bank_merge_radius": float(getattr(cfg, "fl_style_bank_merge_radius", 0.0)),
                "style_batch_fn": getattr(self.style_batch_fn, "__name__", str(self.style_batch_fn)) if self.style_batch_fn is not None else None,
            },
            "domain_pretrain": {
                "enabled": str(getattr(cfg, "fl_vmb_stage1_objective", "ce") or "ce").lower() == "domain_unsup_pretrain",
                "stage1_objective": str(getattr(cfg, "fl_vmb_stage1_objective", "ce") or "ce"),
                "method": str(getattr(cfg, "domain_unsup_pretrain_method", "consistency") or "consistency"),
                "train_scope": str(getattr(cfg, "fl_domain_pretrain_train_scope", "all") or "all"),
                "lambda_domain_unsup_pretrain": float(getattr(cfg, "lambda_domain_unsup_pretrain", 0.0)),
                "lambda_domain_unsup_metadata_ce": float(getattr(cfg, "lambda_domain_unsup_metadata_ce", 0.0)),
                "lambda_domain_unsup_var": float(getattr(cfg, "lambda_domain_unsup_var", 0.0)),
                "domain_unsup_noise_std": float(getattr(cfg, "domain_unsup_noise_std", 0.01)),
                "domain_unsup_amp_jitter": float(getattr(cfg, "domain_unsup_amp_jitter", 0.03)),
                "domain_unsup_max_shift": int(getattr(cfg, "domain_unsup_max_shift", 0)),
                "domain_unsup_logit_cons_weight": float(getattr(cfg, "domain_unsup_logit_cons_weight", 0.0)),
                "domain_unsup_client_compact_weight": float(getattr(cfg, "domain_unsup_client_compact_weight", 0.0)),
                "domain_unsup_var_floor": float(getattr(cfg, "domain_unsup_var_floor", 0.02)),
                "semantics": "receiver_preserving_view_consistency_plus_client_local_receiver_cluster_compactness",
            },
            "protobank": {
                "fed_proto_enabled": bool(_fed_proto_enabled(cfg)),
                "use_fed_proto_stats": bool(getattr(cfg, "use_fed_proto_stats", False)),
                "lambda_fed_proto": float(getattr(cfg, "lambda_fed_proto", 0.0)),
                "fed_proto_min_count": int(getattr(cfg, "fed_proto_min_count", 2)),
                "fed_proto_momentum": float(getattr(cfg, "fed_proto_momentum", 0.0)),
                "evidence_bank_enabled": bool(self.proto_evidence_bank is not None),
                "use_proto_evidence_bank": bool(getattr(cfg, "use_proto_evidence_bank", True)),
                "proto_max_per_class": int(getattr(cfg, "proto_max_per_class", 8)),
                "proto_top_m": int(getattr(cfg, "proto_top_m", 4)),
                "proto_temperature": float(getattr(cfg, "proto_temperature", 0.10)),
                "proto_rho_max": float(getattr(cfg, "proto_rho_max", 0.05)),
                "proto_fusion_eval": bool(getattr(cfg, "proto_fusion_eval", True)),
                "fusion_policy": "base_anchored_conservative_harm_rescue",
            },
            "coral_alignment": {
                "enabled": bool(_fed_coral_enabled(cfg)),
                "use_fed_coral": bool(getattr(cfg, "use_fed_coral", False)),
                "feature": _coral_feature_name(cfg),
                "stage": str(getattr(cfg, "fl_coral_stage", "stage1") or "stage1"),
                "start_round": int(_coral_config_value(cfg, "fl_coral_start_round", "fed_coral_start_round", 1) or 1),
                "cov_mode": str(_coral_config_value(cfg, "fl_coral_cov_mode", "fed_coral_mode", "diag") or "diag"),
                "min_count": int(_coral_config_value(cfg, "fl_coral_min_count", "fed_coral_min_count", 2) or 2),
                "momentum": float(_coral_config_value(cfg, "fl_coral_momentum", "fed_coral_momentum", 0.95) or 0.95),
                "shrinkage": float(getattr(cfg, "fl_coral_shrinkage", 0.05) or 0.0),
                "collect_views": str(getattr(cfg, "fl_coral_collect_views", "clean") or "clean"),
                "lambda_fed_coral": float(getattr(cfg, "lambda_fed_coral", 0.0) or 0.0),
                "lambda_fed_coral_virtual": float(getattr(cfg, "lambda_fed_coral_virtual", 0.0) or 0.0),
                "lambda_fl_coral_zid_global": _coral_weight(cfg, "lambda_fl_coral_zid_global", "lambda_fed_coral"),
                "lambda_fl_coral_zid_virtual": _coral_weight(cfg, "lambda_fl_coral_zid_virtual", "lambda_fed_coral_virtual"),
                "lambda_fl_coral_zdom_global": float(getattr(cfg, "lambda_fl_coral_zdom_global", 0.0) or 0.0),
                "scope_compat": str(getattr(cfg, "fed_coral_scope", "zid_global") or "zid_global"),
                "claim": "opt_in_class_conditional_zid_feature_alignment_from_uploaded_statistics",
            },
            "fed_fishr": {
                "enabled": bool(self.fed_fishr_bank is not None),
                "use_fed_fishr": bool(getattr(cfg, "use_fed_fishr", False)),
                "mode": self.fed_fishr_mode,
                "lambda_fed_fishr": float(getattr(cfg, "lambda_fed_fishr", 0.0) or 0.0),
                "gradient_scope": str(getattr(cfg, "fed_fishr_gradient_scope", "classifier_head") or "classifier_head"),
                "start_round": int(getattr(cfg, "fed_fishr_start_round", 1) or 1),
                "min_clients": int(getattr(cfg, "fed_fishr_min_clients", 2) or 2),
                "min_count": int(getattr(cfg, "fed_fishr_min_count", 2) or 2),
                "max_samples_per_class": int(getattr(cfg, "fed_fishr_max_samples_per_class", 0) or 0),
                "sketch_dim": int(getattr(cfg, "fed_fishr_sketch_dim", 0) or 0),
                "momentum": float(getattr(cfg, "fed_fishr_momentum", 0.0) or 0.0),
                "reweight_floor": float(getattr(cfg, "fed_fishr_reweight_floor", 0.0) or 0.0),
                "reweight_cap": float(getattr(cfg, "fed_fishr_reweight_cap", 1.0) or 1.0),
                "semantics": "server_merges_class_conditional_gradient_variance_stats_across_single_receiver_clients",
                "client_privacy_boundary": "uploads_sum_sq_sum_count_variance_by_tx_class_not_per_sample_gradients",
            },
            "fedcvs_vmb": {
                "enabled": bool(self.vmb_enabled),
                "method": "FedCVS-RFFI-VMB",
                "stage": str(getattr(cfg, "fl_vmb_stage", "stage2")),
                "pretrain_rounds": int(getattr(cfg, "fl_vmb_pretrain_rounds", 0)),
                "stage1_local_steps": int(getattr(cfg, "fl_vmb_stage1_local_steps", getattr(cfg, "fl_local_epochs", 1))),
                "stage1_objective": str(getattr(cfg, "fl_vmb_stage1_objective", "ce") or "ce"),
                "stage1_use_aux_losses": bool(getattr(cfg, "fl_vmb_stage1_use_aux_losses", False)),
                "stage1_lr_mult": float(getattr(cfg, "fl_vmb_stage1_lr_mult", 1.0)),
                "batches_per_client": int(getattr(cfg, "fl_vmb_batches_per_client", 1)),
                "server_lr": float(getattr(cfg, "fl_vmb_server_lr", 0.01)),
                "server_momentum": float(getattr(cfg, "fl_vmb_server_momentum", 0.9)),
                "server_weight_decay": float(getattr(cfg, "fl_vmb_weight_decay", 0.0)),
                "domain_balanced_sampling": bool(getattr(cfg, "fl_vmb_domain_balanced_sampling", True)),
                "domain_balanced_aggregation": bool(getattr(cfg, "fl_vmb_domain_balanced_aggregation", True)),
                "transmitter_balanced_batch": bool(getattr(cfg, "fl_vmb_transmitter_balanced_batch", True)),
                "freeze_rx_stage2": bool(getattr(cfg, "fl_vmb_freeze_rx_stage2", True)),
                "prototype_ema": float(getattr(cfg, "fl_vmb_prototype_ema", 0.95)),
                "prototype_clip_norm": float(getattr(cfg, "fl_vmb_prototype_clip_norm", 1.0)),
                "tau_tx": float(getattr(cfg, "tau_vmb_tx", 0.1)),
                "tau_rx": float(getattr(cfg, "tau_vmb_rx", 0.1)),
                "lambda_tx_proto": float(getattr(cfg, "lambda_vmb_tx_proto", 0.0)),
                "lambda_rx_proto": float(getattr(cfg, "lambda_vmb_rx_proto", 0.0)),
                "lambda_tx_adv_r": float(getattr(cfg, "lambda_tx_adv_r", 0.0)),
                "adv_warmup_rounds": int(getattr(cfg, "fl_vmb_adv_warmup_rounds", 0)),
                "client_domain_ids": self.vmb_client_domains,
                "final_classifier": "C_t(z_t)",
                "receiver_feature_role": "prototype_sampling_orthogonal_adversarial_only",
                "diagnostics": [
                    "prototype_counts",
                    "gradient_norm",
                    "gradient_cosine",
                    "client_drift_norm",
                    "per_domain_loss_variance",
                    "communication_payload_bytes",
                ],
                "approximation_label": (
                    "compressed_split_bex02_approximation"
                    if self.split_bex02_enabled
                    else "vmb_prototype_gradient_approximation"
                ),
            },
            "distillation": {
                "enabled": bool(self.logit_anchor_bank is not None),
                "use_logit_anchors": bool(getattr(cfg, "use_logit_anchors", False)),
                "lambda_logit_kd": float(getattr(cfg, "lambda_logit_kd", 0.0)),
                "kd_temperature": float(getattr(cfg, "kd_temperature", 2.0)),
                "kd_reliability_gate": float(getattr(cfg, "kd_reliability_gate", 0.0)),
                "kd_margin_min": float(getattr(cfg, "kd_margin_min", 0.0)),
                "kd_anchor_ema": float(getattr(cfg, "kd_anchor_ema", 0.9)),
                "kd_min_count": int(getattr(cfg, "kd_min_count", 1)),
                "claim": "confidence_gated_logit_anchor_distillation",
            },
            "compression": {
                "enabled": bool(self.activation_token_codec is not None),
                "train_mode": self.train_mode,
                "activation_token_route": str(getattr(cfg, "activation_token_route", "none") or "none"),
                "split_layer": str(getattr(cfg, "split_layer", "z_id") or "z_id"),
                "token_quant_bits": int(getattr(cfg, "token_quant_bits", 8)),
                "token_sketch_dim": int(getattr(cfg, "token_sketch_dim", 64)),
                "token_rank": int(getattr(cfg, "token_rank", 8)),
                "claim": "compressed_feature_token_accounting_not_raw_iq",
            },
            "conflict_aggregation": {
                "mode": str(getattr(cfg, "fl_conflict_agg", "none") or "none"),
                "claim": "stage2_gradient_conflict_diagnostics_and_optional_projection",
            },
            "feature_probe": {
                "probe_every": int(getattr(cfg, "fl_probe_every", 0)),
                "feature_probe_export": str(getattr(cfg, "feature_probe_export", "") or ""),
                "probe_max_samples": int(getattr(cfg, "probe_max_samples", 0)),
            },
            "style_collab": {
                "enabled": bool(self._style_collab_enabled()),
                "views": int(getattr(cfg, "style_collab_views", 2)),
                "fusion": str(getattr(cfg, "style_collab_fusion", "adaptive")),
                "base_weight": float(getattr(cfg, "style_collab_base_weight", 1.0)),
                "max_aux_weight": float(getattr(cfg, "style_collab_max_aux_weight", 1.0)),
                "source": "StyleBank virtual receiver views",
                "policy": "paper_inspired_virtual_collaborative_inference",
            },
            "grl": {
                "objective": str(getattr(cfg, "fl_local_objective", "ce")),
                "grl_lambda": float(getattr(cfg, "grl_lambda", 1.0)),
                "lambda_rx_adv": float(getattr(cfg, "lambda_rx_adv", getattr(cfg, "rx_weight", 0.0))),
                "lambda_dom": float(getattr(cfg, "lambda_dom", 0.0)),
                "lambda_adv": float(getattr(cfg, "lambda_adv", 0.0)),
                "receiver_agnostic_alias_rx_weight": getattr(cfg, "rx_weight", None),
                "domain_labels_for_style_batches": (
                    "mapped_target_receiver_domain_when_style_dg_ready"
                    if self._style_domain_label_mode() == "target_receiver"
                    else "d_style_when_style_dg_ready_else_none"
                ),
                "target_metrics": ["zdom_target_acc", "grl_target_acc"],
                "role": "remove_receiver_channel_shortcut_so_zid_keeps_transmitter_fingerprint",
            },
            "fed_cgrl": self.fed_cgrl.config_snapshot(),
            "losses": {
                "label_smoothing": float(getattr(cfg, "label_smoothing", 0.0)),
                "fedprox_mu": float(getattr(cfg, "fedprox_mu", 0.0)),
                "lambda_rx_adv": float(getattr(cfg, "lambda_rx_adv", getattr(cfg, "rx_weight", 0.0))),
                "lambda_dom": float(getattr(cfg, "lambda_dom", 0.0)),
                "lambda_adv": float(getattr(cfg, "lambda_adv", 0.0)),
                "lambda_orth": float(getattr(cfg, "lambda_orth", 0.0)),
                "lambda_cons": float(getattr(cfg, "lambda_cons", 0.0)),
                "lambda_group_ce": float(getattr(cfg, "lambda_group_ce", 0.0)),
                "lambda_fishr": float(getattr(cfg, "lambda_fishr", 0.0)),
                "lambda_fed_fishr": float(getattr(cfg, "lambda_fed_fishr", 0.0)),
                "fishr_min_domains": int(getattr(cfg, "fishr_min_domains", 2)),
                "group_ce_min_domains": int(getattr(cfg, "group_ce_min_domains", 2)),
                "group_ce_top_frac": float(getattr(cfg, "group_ce_top_frac", 0.25)),
                "group_ce_mode": str(getattr(cfg, "group_ce_mode", "hard")),
                "groupdro_tau": float(getattr(cfg, "groupdro_tau", 0.5)),
                "groupdro_cap": float(getattr(cfg, "groupdro_cap", 0.65)),
                "lambda_sat_cls": float(getattr(cfg, "lambda_sat_cls", 0.0)),
                "lambda_sat_cons": float(getattr(cfg, "lambda_sat_cons", 0.0)),
                "lambda_fed_proto": float(getattr(cfg, "lambda_fed_proto", 0.0)),
                "lambda_supcon_id": float(getattr(cfg, "lambda_supcon_id", 0.0)),
                "supcon_temp": float(getattr(cfg, "supcon_temp", 0.12)),
                "generalization_feature": str(getattr(cfg, "generalization_feature", "z_id") or "z_id"),
                "lambda_domain_unsup_pretrain": float(getattr(cfg, "lambda_domain_unsup_pretrain", 0.0)),
                "lambda_domain_unsup_metadata_ce": float(getattr(cfg, "lambda_domain_unsup_metadata_ce", 0.0)),
                "lambda_domain_unsup_var": float(getattr(cfg, "lambda_domain_unsup_var", 0.0)),
                "domain_unsup_client_compact_weight": float(getattr(cfg, "domain_unsup_client_compact_weight", 0.0)),
            },
            "satellite": {
                "use_sat_consistency": bool(getattr(cfg, "use_sat_consistency", False)),
                "sat_cons_start_epoch": int(getattr(cfg, "sat_cons_start_epoch", 1)),
                "fl_sat_aug_mode": str(getattr(cfg, "fl_sat_aug_mode", "baseline_view")),
                "fl_baseline_view_ce_only": bool(getattr(cfg, "fl_baseline_view_ce_only", False)),
                "fl_baseline_view_ce_weight": float(getattr(cfg, "fl_baseline_view_ce_weight", 1.0)),
                "sat_train_scenario": str(getattr(cfg, "sat_train_scenario", "mixed_orbit")),
                "sat_train_scenarios": sat_train_scenarios,
                "sat_view_schedule": str(getattr(cfg, "sat_view_schedule", "") or ""),
                "use_fed_style_sat_view": bool(getattr(cfg, "use_fed_style_sat_view", False)),
                "per_round_satellite_eval": bool(self.extra_eval_fn is not None and self._heavy_eval_runs_every_round()),
                "scheduled_satellite_eval": bool(self.extra_eval_fn is not None),
                "eval_sat_channel": bool(getattr(cfg, "eval_sat_channel", False)),
                "eval_sat_scenarios": eval_sat_scenarios,
                "eval_sat_on": str(getattr(cfg, "eval_sat_on", "")),
                "sat_eval_max_batches": int(getattr(cfg, "sat_eval_max_batches", 0)),
            },
            "evaluation": {
                "eval_max_batches": int(getattr(cfg, "eval_max_batches", 0)),
                "test_eval_policy": str(getattr(cfg, "test_eval_policy", "")),
                "primary_udu_weight": float(getattr(cfg, "primary_udu_weight", 0.0)),
                "named_test_loaders": list(self.named_test_loaders.keys()),
                "extra_eval": bool(self.extra_eval_fn is not None),
                "heavy_eval_interval": self._heavy_eval_interval(),
                "heavy_eval_last_n": self._heavy_eval_last_n(),
                "heavy_eval_final_offsets": self._heavy_eval_final_offsets(),
                "heavy_eval_final_rounds": sorted(self._heavy_eval_final_rounds()),
                "heavy_eval_scope": "named_test+extra_eval+proto_fusion+style_collab",
            },
            "full_args": _cfg_snapshot(cfg),
        }

    def _print_config_snapshot(self, snapshot: Mapping[str, Any]):
        run = snapshot.get("run", {}) or {}
        runtime = snapshot.get("runtime", {}) or {}
        data = snapshot.get("data", {}) or {}
        fed = snapshot.get("federated", {}) or {}
        clients = snapshot.get("client_splits", {}) or {}
        style = snapshot.get("stylebank", {}) or {}
        proto = snapshot.get("protobank", {}) or {}
        coral = snapshot.get("coral_alignment", {}) or {}
        fed_fishr = snapshot.get("fed_fishr", {}) or {}
        vmb = snapshot.get("fedcvs_vmb", {}) or {}
        distill = snapshot.get("distillation", {}) or {}
        compression = snapshot.get("compression", {}) or {}
        conflict = snapshot.get("conflict_aggregation", {}) or {}
        feature_probe = snapshot.get("feature_probe", {}) or {}
        style_collab = snapshot.get("style_collab", {}) or {}
        grl = snapshot.get("grl", {}) or {}
        fed_cgrl = snapshot.get("fed_cgrl", {}) or {}
        sat = snapshot.get("satellite", {}) or {}
        eval_cfg = snapshot.get("evaluation", {}) or {}
        losses = snapshot.get("losses", {}) or {}
        print("[FED-CONFIG-BEGIN]", flush=True)
        print(
            f"[FED-CONFIG-RUN] seed={run.get('seed')} device={run.get('device')} "
            f"output_dir={run.get('output_dir')}",
            flush=True,
        )
        print(
            f"[FED-CONFIG-RUNTIME] torch_threads={runtime.get('torch_num_threads')} "
            f"torch_interop={runtime.get('torch_num_interop_threads')} "
            f"omp={runtime.get('omp_num_threads')} mkl={runtime.get('mkl_num_threads')} "
            f"openblas={runtime.get('openblas_num_threads')} numexpr={runtime.get('numexpr_num_threads')}",
            flush=True,
        )
        print(
            f"[FED-CONFIG-DATA] dataset={data.get('dataset')} wisig_train_ratio={data.get('wisig_train_ratio')} "
            f"train_samples={data.get('train_samples')} val_batches={data.get('val_batches')} "
            f"named_tests={','.join(data.get('named_tests') or [])}",
            flush=True,
        )
        print(
            f"[FED-CONFIG-FL] mode={fed.get('train_mode')} objective={fed.get('fl_local_objective')} "
            f"rounds={fed.get('fl_rounds')} local_epochs={fed.get('fl_local_epochs')} "
            f"clients={clients.get('num_clients')} client_key={fed.get('fl_client_key')} "
            f"clients_per_round={fed.get('fl_clients_per_round')} agg={fed.get('fl_agg_weight')} "
            f"batch={fed.get('batch_size')} lr={fed.get('lr')} wd={fed.get('wd')} fedprox_mu={fed.get('fedprox_mu')}",
            flush=True,
        )
        print(
            f"[FED-CONFIG-STYLEBANK] enabled={int(bool(style.get('enabled')))} "
            f"replay_start={style.get('fl_style_replay_start_round')} phys_start={style.get('fl_style_phys_start_round')} "
            f"dg_start={style.get('fl_style_dg_start_round')} dg_min_domains={style.get('fl_style_dg_min_domains')} "
            f"max_views={style.get('fl_style_max_views')} replay_prob={style.get('fl_style_replay_prob')} "
            f"label_mode={style.get('fl_style_domain_label_mode')} "
            f"zdom_probe_every={style.get('fl_style_zdom_probe_every')} "
            f"force_probe={int(bool(style.get('fl_style_zdom_probe_force_batch')))} "
            f"real_probe={style.get('fl_style_zdom_probe_real_samples')} "
            f"sampling={style.get('fl_style_sampling_policy')} mix_alpha={style.get('fl_style_transform_mix_alpha')} "
            f"real_mix={style.get('fl_style_real_mix_samples')}@R{style.get('fl_style_real_mix_start_round')} "
            f"style_code_dim={style.get('fl_style_code_dim')} "
            f"bank_max={style.get('fl_style_bank_max_centroids')} merge_radius={style.get('fl_style_bank_merge_radius')} "
            f"style_domain={style.get('style_domain_semantics')} d_style={style.get('d_style_semantics')} "
            f"d_raw={style.get('d_raw_semantics')} sat_view={int(bool(style.get('use_fed_style_sat_view')))}",
            flush=True,
        )
        print(
            f"[FED-CONFIG-GRL] objective={grl.get('objective')} grl_lambda={grl.get('grl_lambda')} "
            f"lambda_rx_adv={grl.get('lambda_rx_adv')} lambda_dom={grl.get('lambda_dom')} "
            f"lambda_adv={grl.get('lambda_adv')} d_style={grl.get('domain_labels_for_style_batches')} "
            f"role={grl.get('role')}",
            flush=True,
        )
        print(
            f"[FED-CONFIG-CGRL] enabled={int(bool(fed_cgrl.get('enabled')))} "
            f"base={fed_cgrl.get('base_lambda')} min={fed_cgrl.get('min_lambda')} "
            f"max={fed_cgrl.get('max_lambda')} warmup={fed_cgrl.get('warmup_rounds')} "
            f"leak_target={fed_cgrl.get('leak_target_acc')} leak_gain={fed_cgrl.get('leak_gain')} "
            f"leak_stat={fed_cgrl.get('leak_stat')} tx_guard={fed_cgrl.get('tx_loss_guard')} "
            f"tx_release={fed_cgrl.get('tx_guard_release_rounds')} "
            f"conflict_source={fed_cgrl.get('conflict_source')} "
            f"conflict_threshold={fed_cgrl.get('conflict_threshold')} "
            f"ema={fed_cgrl.get('ema')}",
            flush=True,
        )
        print(
            f"[FED-CONFIG-PROTOBANK] evidence_enabled={int(bool(proto.get('evidence_bank_enabled')))} "
            f"fed_proto_enabled={int(bool(proto.get('fed_proto_enabled')))} "
            f"lambda_fed_proto={proto.get('lambda_fed_proto')} max_per_class={proto.get('proto_max_per_class')} "
            f"top_m={proto.get('proto_top_m')} rho_max={proto.get('proto_rho_max')} "
            f"fusion={proto.get('fusion_policy')}",
            flush=True,
        )
        print(
            f"[FED-CONFIG-CORAL] enabled={int(bool(coral.get('enabled')))} "
            f"use_fed_coral={int(bool(coral.get('use_fed_coral')))} "
            f"feature={coral.get('feature')} stage={coral.get('stage')} "
            f"start={coral.get('start_round')} mode={coral.get('cov_mode')} "
            f"min_count={coral.get('min_count')} momentum={coral.get('momentum')} "
            f"collect={coral.get('collect_views')} "
            f"lambda_zid_global={coral.get('lambda_fl_coral_zid_global')} "
            f"lambda_zid_virtual={coral.get('lambda_fl_coral_zid_virtual')} "
            f"lambda_zdom_global={coral.get('lambda_fl_coral_zdom_global')}",
            flush=True,
        )
        print(
            f"[FED-CONFIG-FEDFISHR] enabled={int(bool(fed_fishr.get('enabled')))} "
            f"use={int(bool(fed_fishr.get('use_fed_fishr')))} mode={fed_fishr.get('mode')} "
            f"lambda={fed_fishr.get('lambda_fed_fishr')} scope={fed_fishr.get('gradient_scope')} "
            f"start={fed_fishr.get('start_round')} min_clients={fed_fishr.get('min_clients')} "
            f"min_count={fed_fishr.get('min_count')} max_per_class={fed_fishr.get('max_samples_per_class')} "
            f"sketch={fed_fishr.get('sketch_dim')} momentum={fed_fishr.get('momentum')} "
            f"floor={fed_fishr.get('reweight_floor')} cap={fed_fishr.get('reweight_cap')}",
            flush=True,
        )
        print(
            f"[FED-CONFIG-VMB] enabled={int(bool(vmb.get('enabled')))} method={vmb.get('method')} "
            f"stage={vmb.get('stage')} pretrain_rounds={vmb.get('pretrain_rounds')} "
            f"stage1_local_steps={vmb.get('stage1_local_steps')} "
            f"stage1_objective={vmb.get('stage1_objective')} "
            f"stage1_lr_mult={vmb.get('stage1_lr_mult')} "
            f"batches_per_client={vmb.get('batches_per_client')} "
            f"server_lr={vmb.get('server_lr')} momentum={vmb.get('server_momentum')} "
            f"domain_sampling={int(bool(vmb.get('domain_balanced_sampling')))} "
            f"domain_aggregation={int(bool(vmb.get('domain_balanced_aggregation')))} "
            f"tx_balanced_batch={int(bool(vmb.get('transmitter_balanced_batch')))} "
            f"freeze_rx_stage2={int(bool(vmb.get('freeze_rx_stage2')))} "
            f"lambda_tx_proto={vmb.get('lambda_tx_proto')} lambda_rx_proto={vmb.get('lambda_rx_proto')} "
            f"lambda_tx_adv_r={vmb.get('lambda_tx_adv_r')} final={vmb.get('final_classifier')} "
            f"approx={vmb.get('approximation_label')}",
            flush=True,
        )
        print(
            f"[FED-CONFIG-DISTILL] enabled={int(bool(distill.get('enabled')))} "
            f"lambda_logit_kd={distill.get('lambda_logit_kd')} temp={distill.get('kd_temperature')} "
            f"conf_gate={distill.get('kd_reliability_gate')} margin={distill.get('kd_margin_min')} "
            f"ema={distill.get('kd_anchor_ema')} min_count={distill.get('kd_min_count')} "
            f"claim={distill.get('claim')}",
            flush=True,
        )
        print(
            f"[FED-CONFIG-SPLIT] enabled={int(bool(compression.get('enabled')))} "
            f"mode={compression.get('train_mode')} route={compression.get('activation_token_route')} "
            f"split_layer={compression.get('split_layer')} bits={compression.get('token_quant_bits')} "
            f"sketch={compression.get('token_sketch_dim')} rank={compression.get('token_rank')} "
            f"conflict_agg={conflict.get('mode')} probe_every={feature_probe.get('probe_every')} "
            f"feature_export={feature_probe.get('feature_probe_export')} claim={compression.get('claim')}",
            flush=True,
        )
        print(
            f"[FED-CONFIG-STYLE-COLLAB] enabled={int(bool(style_collab.get('enabled')))} "
            f"views={style_collab.get('views')} fusion={style_collab.get('fusion')} "
            f"base_weight={style_collab.get('base_weight')} max_aux_weight={style_collab.get('max_aux_weight')} "
            f"source={style_collab.get('source')}",
            flush=True,
        )
        print(
            f"[FED-CONFIG-SAT] use_sat={int(bool(sat.get('use_sat_consistency')))} "
            f"mode={sat.get('fl_sat_aug_mode')} train={','.join(sat.get('sat_train_scenarios') or [])} "
            f"baseline_ce_only={int(bool(sat.get('fl_baseline_view_ce_only')))} "
            f"baseline_ce_weight={sat.get('fl_baseline_view_ce_weight')} "
            f"lambda_cls={losses.get('lambda_sat_cls')} lambda_cons={losses.get('lambda_sat_cons')} "
            f"per_round_eval={int(bool(sat.get('per_round_satellite_eval')))} "
            f"eval_sat={int(bool(sat.get('eval_sat_channel')))} eval_on={sat.get('eval_sat_on')} "
            f"eval={','.join(sat.get('eval_sat_scenarios') or [])}",
            flush=True,
        )
        eval_sat_on = str(sat.get("eval_sat_on") or "")
        active_sat_splits = _expand_sat_eval_on_for_log(eval_sat_on)
        print(
            f"[FED-CONFIG-SAT-SPLITS] count={len(_MAIN_TEST_KEYS)} splits={','.join(_MAIN_TEST_KEYS)} "
            f"eval_on={eval_sat_on or 'none'} active_count={len(active_sat_splits)} "
            f"active_splits={','.join(active_sat_splits) or 'none'}",
            flush=True,
        )
        print(
            f"[FED-CONFIG-EVAL] eval_max_batches={eval_cfg.get('eval_max_batches')} "
            f"primary_udu_weight={eval_cfg.get('primary_udu_weight')} extra_eval={int(bool(eval_cfg.get('extra_eval')))} "
            f"heavy_interval={eval_cfg.get('heavy_eval_interval')} heavy_last_n={eval_cfg.get('heavy_eval_last_n')} "
            f"heavy_final_offsets={eval_cfg.get('heavy_eval_final_offsets')} "
            f"heavy_final_rounds={eval_cfg.get('heavy_eval_final_rounds')} "
            f"scope={eval_cfg.get('heavy_eval_scope')}",
            flush=True,
        )
        print(f"[FED-CONFIG-END] path={run.get('config_json')}", flush=True)

    def _write_config_snapshot(self) -> Dict[str, Any]:
        if self._config_written and os.path.exists(self.config_json):
            with open(self.config_json, "r", encoding="utf-8") as f:
                return json.load(f)
        snapshot = _jsonable(self._build_config_snapshot())
        with open(self.config_json, "w", encoding="utf-8") as f:
            json.dump(snapshot, f, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        with open(self.logs_jsonl, "a", encoding="utf-8") as f:
            f.write(json.dumps(snapshot, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
        self._print_config_snapshot(snapshot)
        self._config_written = True
        return snapshot

    def _append_logs(self, row: Mapping[str, Any]):
        with open(self.logs_jsonl, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        exists = os.path.exists(self.metrics_csv)
        flat = {
            "round": row.get("round"),
            "round_time_s": row.get("round_time_s"),
            "round_train_time_s": row.get("round_train_time_s"),
            "round_eval_time_s": row.get("round_eval_time_s"),
            "round_val_time_s": row.get("round_val_time_s"),
            "round_test_time_s": row.get("round_test_time_s"),
            "round_extra_eval_time_s": row.get("round_extra_eval_time_s"),
            "round_other_time_s": row.get("round_other_time_s"),
            "test_eval_ran": row.get("global_test_eval_ran"),
            "next_test_eval_round": row.get("global_next_test_eval_round"),
            "test_eval_interval": row.get("global_test_eval_interval"),
            "test_eval_last_n": row.get("global_test_eval_last_n"),
            "test_eval_final_offsets": row.get("global_test_eval_final_offsets"),
            "test_eval_final_rounds": row.get("global_test_eval_final_rounds"),
            "train_loss": row.get("client_train_loss_avg"),
            "train_loss_cls": row.get("client_loss_cls_avg"),
            "train_loss_rx_adv": row.get("client_loss_rx_adv_avg"),
            "fed_cgrl_lambda_rx_adv": row.get("client_fed_cgrl_lambda_rx_adv_avg"),
            "fed_cgrl_base_lambda": row.get("client_fed_cgrl_base_lambda_avg"),
            "fed_cgrl_warmup_gate": row.get("client_fed_cgrl_warmup_gate_avg"),
            "fed_cgrl_leak_gate": row.get("client_fed_cgrl_leak_gate_avg"),
            "fed_cgrl_tx_gate": row.get("client_fed_cgrl_tx_gate_avg"),
            "fed_cgrl_conflict_gate": row.get("client_fed_cgrl_conflict_gate_avg"),
            "fed_cgrl_unclamped_lambda": row.get("client_fed_cgrl_unclamped_lambda_avg"),
            "fed_cgrl_enabled": row.get("client_fed_cgrl_enabled_avg"),
            "fed_cgrl_global_lambda_rx_adv_avg": (row.get("global_fed_cgrl_summary") or {}).get("lambda_rx_adv_avg"),
            "fed_cgrl_global_lambda_rx_adv_min": (row.get("global_fed_cgrl_summary") or {}).get("lambda_rx_adv_min"),
            "fed_cgrl_global_lambda_rx_adv_max": (row.get("global_fed_cgrl_summary") or {}).get("lambda_rx_adv_max"),
            "fed_cgrl_global_lambda_rx_adv_p90": (row.get("global_fed_cgrl_summary") or {}).get("lambda_rx_adv_p90"),
            "fed_cgrl_global_grl_target_acc_min": (row.get("global_fed_cgrl_summary") or {}).get("grl_target_acc_min"),
            "fed_cgrl_global_grl_target_acc_max": (row.get("global_fed_cgrl_summary") or {}).get("grl_target_acc_max"),
            "fed_cgrl_global_grl_target_acc_p90": (row.get("global_fed_cgrl_summary") or {}).get("grl_target_acc_p90"),
            "fed_cgrl_global_worst_client_id": (row.get("global_fed_cgrl_summary") or {}).get("grl_target_acc_worst_client"),
            "fed_cgrl_global_loss_cls_max": (row.get("global_fed_cgrl_summary") or {}).get("loss_cls_max"),
            "fed_cgrl_global_loss_cls_worst_client": (row.get("global_fed_cgrl_summary") or {}).get("loss_cls_worst_client"),
            "fed_cgrl_global_conflict_cos_min": (row.get("global_fed_cgrl_summary") or {}).get("conflict_cos_min"),
            "fed_cgrl_global_conflict_source": (row.get("global_fed_cgrl_summary") or {}).get("conflict_source"),
            "fed_cgrl_global_conflict_signal_available": (row.get("global_fed_cgrl_summary") or {}).get("conflict_signal_available"),
            "train_loss_baseline_sat_view": row.get("client_loss_baseline_sat_view_avg"),
            "train_loss_fed_proto": row.get("client_loss_fed_proto_avg"),
            "train_loss_vmb_tx_proto": row.get("client_loss_vmb_tx_proto_avg"),
            "train_loss_vmb_rx_proto": row.get("client_loss_vmb_rx_proto_avg"),
            "train_loss_tx_adv_r": row.get("client_loss_tx_adv_r_avg"),
            "train_loss_logit_kd": row.get("client_loss_logit_kd_avg"),
            "train_loss_supcon_id": row.get("client_loss_supcon_id_avg"),
            "train_loss_coral_zid_global": row.get("client_loss_coral_zid_global_avg"),
            "train_loss_coral_zid_virtual": row.get("client_loss_coral_zid_virtual_avg"),
            "train_loss_coral_zdom_global": row.get("client_loss_coral_zdom_global_avg"),
            "coral_zid_global_active_classes": row.get("client_coral_zid_global_active_classes_avg"),
            "coral_zid_global_mean_dist": row.get("client_coral_zid_global_mean_dist_avg"),
            "coral_zid_global_cov_dist": row.get("client_coral_zid_global_cov_dist_avg"),
            "coral_zid_global_skip_rate": row.get("client_coral_zid_global_skip_rate_avg"),
            "coral_zid_virtual_active_classes": row.get("client_coral_zid_virtual_active_classes_avg"),
            "coral_zid_virtual_mean_dist": row.get("client_coral_zid_virtual_mean_dist_avg"),
            "coral_zid_virtual_cov_dist": row.get("client_coral_zid_virtual_cov_dist_avg"),
            "coral_zid_virtual_skip_rate": row.get("client_coral_zid_virtual_skip_rate_avg"),
            "coral_zdom_global_active_classes": row.get("client_coral_zdom_global_active_classes_avg"),
            "coral_zdom_global_mean_dist": row.get("client_coral_zdom_global_mean_dist_avg"),
            "coral_zdom_global_cov_dist": row.get("client_coral_zdom_global_cov_dist_avg"),
            "coral_zdom_global_skip_rate": row.get("client_coral_zdom_global_skip_rate_avg"),
            "coral_payload_bytes": row.get("coral_payload_bytes"),
            "coral_global_classes": (row.get("global_coral_summary") or {}).get("class_count_nonzero"),
            "coral_global_count": (row.get("global_coral_summary") or {}).get("total_count"),
            "train_loss_fedprox": row.get("client_loss_fedprox_avg"),
            "fedprox_ratio": row.get("client_fedprox_ratio"),
            "kd_active": row.get("client_kd_active_rate"),
            "logit_anchor_count": row.get("client_logit_anchor_count_avg"),
            "logit_anchor_payload_bytes": row.get("logit_anchor_payload_bytes"),
            "activation_token_payload_bytes": row.get("activation_token_payload_bytes"),
            "activation_token_compression_ratio": row.get("activation_token_compression_ratio"),
            "activation_token_quant_error": row.get("activation_token_quant_error"),
            "feature_probe_samples": row.get("feature_probe_samples"),
            "feature_probe_path": (row.get("global_feature_probe_summary") or {}).get("path"),
            "fed_proto_cos": row.get("client_fed_proto_cos_avg"),
            "vmb_tx_proto_cos": row.get("client_vmb_tx_proto_cos_avg"),
            "vmb_rx_proto_cos": row.get("client_vmb_rx_proto_cos_avg"),
            "vmb_tx_adv_r_acc": row.get("client_tx_adv_r_acc_avg"),
            "vmb_stage": row.get("vmb_stage"),
            "vmb_grad_norm": (row.get("vmb_server_update") or {}).get("grad_norm"),
            "vmb_grad_cos_mean": (row.get("vmb_gradient_cosine") or {}).get("mean"),
            "vmb_conflicts_detected": (row.get("vmb_conflict_summary") or {}).get("conflicts_detected"),
            "vmb_conflicts_resolved": (row.get("vmb_conflict_summary") or {}).get("conflicts_resolved"),
            "vmb_missing_gradient_entries": (row.get("vmb_conflict_summary") or {}).get("missing_gradient_entries"),
            "vmb_grad_cos_mean_before_conflict": (row.get("vmb_conflict_summary") or {}).get("grad_cos_mean_before"),
            "vmb_grad_cos_mean_after_conflict": (row.get("vmb_conflict_summary") or {}).get("grad_cos_mean_after"),
            "vmb_client_drift_norm": row.get("vmb_client_drift_norm_avg"),
            "vmb_loss_domain_variance": row.get("vmb_loss_domain_variance"),
            "vmb_comm_payload_bytes": row.get("vmb_comm_payload_bytes"),
            "vmb_tx_proto_classes": (row.get("global_vmb_proto_summary") or {}).get("tx_count_nonzero"),
            "vmb_rx_proto_clients": (row.get("global_vmb_proto_summary") or {}).get("rx_count_nonzero"),
            "fed_proto_classes": (row.get("global_proto_summary") or {}).get("class_count_nonzero"),
            "proto_evidence_prototypes": (row.get("global_proto_evidence_summary") or {}).get("num_prototypes"),
            "proto_evidence_reliability": (row.get("global_proto_evidence_summary") or {}).get("mean_reliability"),
            "proto_rescue": ((row.get("global_proto_fusion") or {}).get("aggregate") or {}).get("rescue"),
            "proto_harm": ((row.get("global_proto_fusion") or {}).get("aggregate") or {}).get("harm"),
            "proto_net_gain": ((row.get("global_proto_fusion") or {}).get("aggregate") or {}).get("net_gain"),
            "style_collab_rescue": ((row.get("global_style_collab_fusion") or {}).get("aggregate") or {}).get("rescue"),
            "style_collab_harm": ((row.get("global_style_collab_fusion") or {}).get("aggregate") or {}).get("harm"),
            "style_collab_net_gain": ((row.get("global_style_collab_fusion") or {}).get("aggregate") or {}).get("net_gain"),
            "style_collab_base_tx_acc": ((row.get("global_style_collab_fusion") or {}).get("aggregate") or {}).get("base_tx_acc"),
            "style_collab_fused_tx_acc": ((row.get("global_style_collab_fusion") or {}).get("aggregate") or {}).get("fused_tx_acc"),
            "zdom_target_acc": row.get("client_zdom_target_acc_avg"),
            "grl_target_acc": row.get("client_grl_target_acc_avg"),
            "style_bank_centroids": (row.get("global_style_summary") or {}).get("num_centroids"),
            "style_bank_bytes": (row.get("global_style_summary") or {}).get("size_bytes"),
            "style_replay_enabled": row.get("style_replay_enabled"),
            "style_phys_enabled": row.get("style_phys_enabled"),
            "style_dg_enabled": row.get("style_dg_enabled"),
            "style_bank_remote_sample_accept_rate": row.get("style_bank_remote_sample_accept_rate"),
            "style_num_domains": row.get("client_style_num_domains_avg"),
            "style_batch_views": row.get("client_style_batch_views_avg"),
            "style_requested_remote_views": row.get("client_style_requested_remote_views_avg"),
            "style_appended_remote_views": row.get("client_style_appended_remote_views_avg"),
            "style_real_mix_active": row.get("client_style_real_mix_active_rate"),
            "style_domain_entropy": row.get("client_style_domain_entropy_avg"),
            "style_dg_ready": row.get("client_style_dg_ready_rate"),
            "style_domain_label_mode": row.get("client_style_domain_label_mode"),
            "style_zdom_clean_acc": row.get("client_style_zdom_clean_acc"),
            "style_zdom_clean_total": row.get("client_style_zdom_clean_total"),
            "style_zdom_virtual_acc": row.get("client_style_zdom_virtual_acc"),
            "style_zdom_virtual_total": row.get("client_style_zdom_virtual_total"),
            "style_zdom_all_style_acc": row.get("client_style_zdom_all_style_acc"),
            "style_zdom_all_style_total": row.get("client_style_zdom_all_style_total"),
            "style_zdom_real_acc": row.get("client_style_zdom_real_acc"),
            "style_zdom_real_total": row.get("client_style_zdom_real_total"),
            "loss_domain_unsup_pretrain": row.get("client_loss_domain_unsup_pretrain_avg"),
            "loss_domain_unsup_metadata_ce": row.get("client_loss_domain_unsup_metadata_ce_avg"),
            "loss_domain_unsup_var": row.get("client_loss_domain_unsup_var_avg"),
            "train_loss_fishr": row.get("client_loss_fishr_avg"),
            "train_loss_fed_fishr": row.get("client_loss_fed_fishr_avg"),
            "fed_fishr_client_active_rate": row.get("client_fed_fishr_active_rate"),
            "fed_fishr_client_active_classes": row.get("client_fed_fishr_active_classes_avg"),
            "fed_fishr_client_var_dist": row.get("client_fed_fishr_var_dist_avg"),
            "fed_fishr_client_skip_rate": row.get("client_fed_fishr_skip_rate_avg"),
            "fed_fishr_target_ready_rate": row.get("client_fed_fishr_target_ready_rate"),
            "fed_fishr_payload_bytes": row.get("fed_fishr_payload_bytes"),
            "global_fed_fishr_active": bool((row.get("global_fed_fishr_summary") or {}).get("active", False)),
            "global_fed_fishr_reweight_active": bool((row.get("global_fed_fishr_summary") or {}).get("reweight_active", False)),
            "global_fed_fishr_active_classes": (row.get("global_fed_fishr_summary") or {}).get("active_classes"),
            "global_fed_fishr_client_count": (row.get("global_fed_fishr_summary") or {}).get("client_count"),
            "global_fed_fishr_mismatch_mean": (row.get("global_fed_fishr_summary") or {}).get("mismatch_mean"),
            "global_fed_fishr_mismatch_max": (row.get("global_fed_fishr_summary") or {}).get("mismatch_max"),
            "global_fed_fishr_target_var_mean": (row.get("global_fed_fishr_summary") or {}).get("target_var_mean"),
            "global_fed_fishr_weight_max_delta": (row.get("global_fed_fishr_summary") or {}).get("weight_max_delta"),
            "domain_unsup_active": row.get("client_domain_unsup_active_rate"),
            "stage1_domain_pretrain_active": row.get("client_stage1_domain_pretrain_active_rate"),
            "domain_unsup_zdom_cos": row.get("client_domain_unsup_zdom_cos_avg"),
            "domain_unsup_client_compact": row.get("client_domain_unsup_client_compact_avg"),
            "domain_unsup_client_radius": row.get("client_domain_unsup_client_radius_avg"),
            "domain_unsup_dom_entropy": row.get("client_domain_unsup_dom_entropy_avg"),
            "domain_unsup_dom_acc": row.get("client_domain_unsup_dom_acc_avg"),
            "domain_unsup_zdom_var": row.get("client_domain_unsup_zdom_var_avg"),
            "diag_domain_count": row.get("client_diag_domain_count_avg"),
            "diag_fishr_domain_count": row.get("client_diag_fishr_domain_count_avg"),
            "diag_domain_loss_active": row.get("client_diag_domain_loss_active_rate"),
            "diag_adv_active": row.get("client_diag_adv_active_rate"),
            "diag_cons_active": row.get("client_diag_cons_active_rate"),
            "diag_group_ce_active": row.get("client_diag_group_ce_active_rate"),
            "diag_fishr_active": row.get("client_diag_fishr_active_rate"),
            "diag_rx_adv_active": row.get("client_diag_rx_adv_active_rate"),
            "diag_sat_aug_active": row.get("client_diag_sat_aug_active_rate"),
            "diag_baseline_sat_view_active": row.get("client_diag_baseline_sat_view_active_rate"),
            "diag_sat_cls_active": row.get("client_diag_sat_cls_active_rate"),
            "diag_sat_cons_active": row.get("client_diag_sat_cons_active_rate"),
            "diag_style_batch_active": row.get("client_diag_style_batch_active_rate"),
            "diag_style_domain_count": row.get("client_diag_style_domain_count_avg"),
            "diag_stage1_aux_active": row.get("client_diag_stage1_aux_active_rate"),
            "diag_coral_global_active": row.get("client_diag_coral_global_active_rate"),
            "diag_coral_virtual_active": row.get("client_diag_coral_virtual_active_rate"),
            "diag_coral_zdom_active": row.get("client_diag_coral_zdom_active_rate"),
            "train_acc": row.get("client_train_acc_avg"),
            "val_tx_acc": row.get("global_eval_acc"),
            "test_overall_tx_acc": row.get("global_test_overall_acc"),
            "strict_udu_acc": row.get("global_strict_udu_acc"),
            "named_test_tx_acc_json": json.dumps(row.get("global_named_test_tx_acc", {}), ensure_ascii=False, default=str),
            "extra_tests_json": json.dumps(row.get("global_extra_tests", {}), ensure_ascii=False, default=str),
            "lr": row.get("lr"),
            "mixstyle_phase": (row.get("mixstyle_state") or {}).get("phase"),
            "mixstyle_p": (row.get("mixstyle_state") or {}).get("p"),
            "mixstyle_strength": (row.get("mixstyle_state") or {}).get("strength"),
            "train_mode": row.get("train_mode"),
            "fedprox_mu": row.get("fedprox_mu"),
            "fl_local_objective": row.get("fl_local_objective"),
            "fl_sat_aug_mode": row.get("fl_sat_aug_mode"),
        }
        with open(self.metrics_csv, "a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(flat.keys()))
            if not exists:
                writer.writeheader()
            writer.writerow(flat)

    def train(self):
        config_snapshot = self._write_config_snapshot()
        best_score = -float("inf")
        best_round = -1
        best_eval_stats: Dict[str, Any] = {}
        last_eval_stats: Dict[str, Any] = {}
        total_rounds = int(getattr(self.cfg, "fl_rounds", 100))
        for round_idx in range(1, total_rounds + 1):
            round_t0 = time.perf_counter()
            print("", flush=True)
            print(_ROUND_BEGIN_SEPARATOR, flush=True)
            print(
                f"[FED-ROUND-BEGIN][{_round_tag(round_idx, total_rounds)}] "
                f"mode={self.train_mode} objective={getattr(self.cfg, 'fl_local_objective', 'ce')} "
                f"client_key={getattr(self.cfg, 'fl_client_key', 'receiver')}",
                flush=True,
            )
            print(_ROUND_BEGIN_SEPARATOR, flush=True)
            mixstyle_state = self._configure_mixstyle_for_round(round_idx)
            round_lr = self._round_lr(round_idx)
            selected = self._selected_clients(round_idx)
            current_vmb_stage = self._vmb_stage(round_idx) if self.vmb_enabled else "off"
            client_results = OrderedDict()
            client_states = OrderedDict()
            client_gradients = OrderedDict()
            client_proto_stats = OrderedDict()
            client_coral_stats = OrderedDict()
            client_fed_fishr_stats = OrderedDict()
            client_vmb_proto_stats = OrderedDict()
            client_logit_anchor_stats = OrderedDict()
            client_proto_evidence = OrderedDict()
            client_style_packets = OrderedDict()
            client_activation_tokens = OrderedDict()
            client_feature_probe_items = OrderedDict()
            for cid in selected:
                result = self.train_one_client(cid, round_idx)
                vmb_proto_stats = result.pop("vmb_proto_stats", None)
                if vmb_proto_stats is not None:
                    client_vmb_proto_stats[cid] = vmb_proto_stats
                proto_stats = result.pop("fed_proto_stats", None)
                if proto_stats is not None:
                    client_proto_stats[cid] = proto_stats
                coral_stats = result.pop("coral_stats", None)
                if coral_stats is not None:
                    client_coral_stats[cid] = coral_stats
                fed_fishr_stats = result.pop("fed_fishr_stats", None)
                if fed_fishr_stats is not None:
                    client_fed_fishr_stats[cid] = fed_fishr_stats
                logit_stats = result.pop("logit_anchor_stats", None)
                if logit_stats is not None:
                    client_logit_anchor_stats[cid] = logit_stats
                proto_items = result.pop("proto_evidence_items", None)
                if proto_items is not None:
                    client_proto_evidence[cid] = proto_items
                token_summary = result.pop("activation_token_summary", None)
                if token_summary is not None:
                    client_activation_tokens[cid] = token_summary
                style_packets = result.pop("style_packets", None)
                if style_packets is not None:
                    client_style_packets[cid] = style_packets
                feature_probe_items = result.pop("feature_probe_items", None)
                if feature_probe_items is not None:
                    client_feature_probe_items[cid] = feature_probe_items
                if self.vmb_enabled:
                    client_results[cid] = {k: v for k, v in result.items() if k not in {"grads", "state"}}
                    if "grads" in result:
                        client_gradients[cid] = result["grads"]
                    if "state" in result:
                        client_states[cid] = result["state"]
                else:
                    client_results[cid] = {k: v for k, v in result.items() if k != "state"}
                    client_states[cid] = result["state"]

            server_update = {}
            vmb_gradient_cosine = {}
            vmb_domain_weights = {}
            vmb_conflict_summary = {"conflict_mode": str(getattr(self.cfg, "fl_conflict_agg", "none")), "conflicts_detected": 0, "conflicts_resolved": 0}
            fed_cgrl_client_delta_conflict_summary = (
                self._fed_cgrl_conflict_summary_from_client_states(client_states, selected)
                if self.fed_cgrl.enabled
                else self._fed_cgrl_empty_conflict_summary("disabled")
            )
            vmb_proto_summary = {"enabled": False}
            if self.vmb_enabled:
                exclude_keys = set()
                if bool(getattr(self.cfg, "fl_vmb_domain_balanced_aggregation", True)):
                    vmb_domain_weights = domain_balanced_weights(selected, self.vmb_client_domains)
                else:
                    if str(getattr(self.cfg, "fl_agg_weight", "num_samples")) == "uniform":
                        vmb_domain_weights = {cid: 1.0 / float(max(1, len(selected))) for cid in selected}
                    else:
                        total = float(sum(max(0, int(self.client_num_samples[cid])) for cid in selected))
                        vmb_domain_weights = {cid: max(0, int(self.client_num_samples[cid])) / max(total, 1.0) for cid in selected}
                vmb_domain_weights = self._update_global_fed_fishr_stats(
                    client_fed_fishr_stats,
                    base_weights=vmb_domain_weights,
                    round_idx=round_idx,
                )
                if current_vmb_stage == "stage1":
                    exclude_keys = resolve_exclude_keys(
                        next(iter(client_states.values())),
                        exact_keys=str(getattr(self.cfg, "fl_local_exclude_keys", "") or "").replace(";", ",").split(","),
                        prefixes=str(getattr(self.cfg, "fl_local_exclude_prefixes", "") or "").replace(";", ",").split(","),
                    )
                    aggregated = aggregate_state_dicts(
                        client_states,
                        {cid: self.client_num_samples[cid] for cid in selected},
                        exclude_keys=exclude_keys,
                        agg_weight="domain_balanced" if bool(getattr(self.cfg, "fl_vmb_domain_balanced_aggregation", True)) else str(getattr(self.cfg, "fl_agg_weight", "num_samples")),
                        domain_ids=self.vmb_client_domains,
                        client_weights=vmb_domain_weights,
                    )
                    self.global_state.update(aggregated)
                    server_update = {
                        "stage": "stage1",
                        "mode": "state_average",
                        "updated_keys": float(len(aggregated)),
                        "server_lr": float(round_lr),
                    }
                    vmb_gradient_cosine = {"pairs": 0, "mean": float("nan"), "min": float("nan"), "max": float("nan")}
                else:
                    conflict_mode = str(getattr(self.cfg, "fl_conflict_agg", "none") or "none").lower()
                    if conflict_mode in {"", "none", "off"}:
                        aggregated_grads = aggregate_gradients(client_gradients, vmb_domain_weights)
                        vmb_conflict_summary = {
                            "conflict_mode": "none",
                            "conflicts_detected": 0,
                            "conflicts_resolved": 0,
                        }
                    else:
                        aggregated_grads, vmb_conflict_summary = conflict_aware_aggregate_gradients(
                            client_gradients,
                            vmb_domain_weights,
                            mode=conflict_mode,
                        )
                    self.global_state, self.vmb_server_optimizer_state, server_update = apply_server_gradient_step(
                        self.global_state,
                        aggregated_grads,
                        lr=float(getattr(self.cfg, "fl_vmb_server_lr", 0.01)),
                        momentum=float(getattr(self.cfg, "fl_vmb_server_momentum", 0.9)),
                        weight_decay=float(getattr(self.cfg, "fl_vmb_weight_decay", 0.0)),
                        optimizer_state=self.vmb_server_optimizer_state,
                    )
                    server_update["stage"] = "stage2"
                    server_update["mode"] = "gradient_sgd"
                    server_update["conflict_agg"] = vmb_conflict_summary
                    vmb_gradient_cosine = gradient_cosine_summary(client_gradients)
                if self.vmb_proto_bank is not None:
                    vmb_proto_summary = self.vmb_proto_bank.update(merge_prototype_stats(client_vmb_proto_stats.values()))
                proto_summary = self._update_global_proto_stats(client_proto_stats)
            else:
                exclude_keys = resolve_exclude_keys(
                    next(iter(client_states.values())),
                    exact_keys=str(getattr(self.cfg, "fl_local_exclude_keys", "") or "").replace(";", ",").split(","),
                    prefixes=str(getattr(self.cfg, "fl_local_exclude_prefixes", "") or "").replace(";", ",").split(","),
                )
                fed_fishr_weights = self._update_global_fed_fishr_stats(
                    client_fed_fishr_stats,
                    base_weights=self._base_aggregation_weights(selected, agg_weight=str(getattr(self.cfg, "fl_agg_weight", "num_samples"))),
                    round_idx=round_idx,
                )
                aggregated = aggregate_state_dicts(
                    client_states,
                    {cid: self.client_num_samples[cid] for cid in selected},
                    exclude_keys=exclude_keys,
                    agg_weight=str(getattr(self.cfg, "fl_agg_weight", "num_samples")),
                    client_weights=fed_fishr_weights,
                )
                self.global_state.update(aggregated)
                proto_summary = self._update_global_proto_stats(client_proto_stats)
            coral_summary = self._update_global_coral_stats(client_coral_stats)
            proto_evidence_summary = self._update_proto_evidence_bank(client_proto_evidence)
            logit_anchor_summary = self._update_logit_anchor_bank(client_logit_anchor_stats)
            style_summary = self._update_style_bank(client_style_packets)
            feature_probe_summary = self._write_feature_probe_export(round_idx, client_feature_probe_items)
            fed_cgrl_conflict_summary = self._select_fed_cgrl_conflict_summary(
                client_delta_summary=fed_cgrl_client_delta_conflict_summary,
                vmb_conflict_summary=vmb_conflict_summary,
                vmb_gradient_cosine=vmb_gradient_cosine,
                current_vmb_stage=current_vmb_stage,
            )
            self.fed_cgrl.update_after_round(
                client_results,
                round_idx=round_idx,
                conflict_summary=fed_cgrl_conflict_summary,
            )
            fed_cgrl_summary = self.fed_cgrl.round_summary()
            round_train_time_s = time.perf_counter() - round_t0
            eval_stats = self._evaluate(round_idx)
            last_eval_stats = eval_stats
            val_acc = float(eval_stats.get("val", {}).get("tx_acc", float("nan")))
            named_stats = eval_stats.get("named", {}) or {}
            test_overall = eval_stats.get("test_overall", {}) or {"tx_acc": float("nan"), "tx_correct": 0, "tx_total": 0}
            strict = float(eval_stats.get("strict_udu_acc", float("nan")))
            extra_tests = eval_stats.get("extra_tests", {}) or {}
            eval_timing = eval_stats.get("timing", {}) or {}
            score = strict if math.isfinite(strict) else val_acc
            if math.isfinite(score) and score > best_score:
                best_score = score
                best_round = round_idx
                best_eval_stats = eval_stats
                self._save_checkpoint(self.best_checkpoint, round_idx, eval_stats)
            self._save_checkpoint(self.last_checkpoint, round_idx, eval_stats)
            round_eval_time_s = float(eval_timing.get("eval_time_s", 0.0) or 0.0)
            round_time_s = time.perf_counter() - round_t0
            round_other_time_s = max(0.0, round_time_s - round_train_time_s - round_eval_time_s)

            train_loss_avg = _client_seen_weighted_avg(client_results, "loss")
            train_acc_avg = _client_seen_weighted_avg(client_results, "acc")
            loss_cls_avg = _client_seen_weighted_avg(client_results, "loss_cls")
            loss_fishr_avg = _client_seen_weighted_avg(client_results, "loss_fishr")
            loss_fed_fishr_avg = _client_seen_weighted_avg(client_results, "loss_fed_fishr")
            loss_rx_adv_avg = _client_seen_weighted_avg(client_results, "loss_rx_adv")
            loss_baseline_sat_view_avg = _client_seen_weighted_avg(client_results, "loss_baseline_sat_view")
            loss_fed_proto_avg = _client_seen_weighted_avg(client_results, "loss_fed_proto")
            loss_vmb_tx_proto_avg = _client_seen_weighted_avg(client_results, "loss_vmb_tx_proto")
            loss_vmb_rx_proto_avg = _client_seen_weighted_avg(client_results, "loss_vmb_rx_proto")
            loss_tx_adv_r_avg = _client_seen_weighted_avg(client_results, "loss_tx_adv_r")
            loss_logit_kd_avg = _client_seen_weighted_avg(client_results, "loss_logit_kd")
            loss_supcon_id_avg = _client_seen_weighted_avg(client_results, "loss_supcon_id")
            loss_coral_zid_global_avg = _client_seen_weighted_avg(client_results, "loss_coral_zid_global")
            loss_coral_zid_virtual_avg = _client_seen_weighted_avg(client_results, "loss_coral_zid_virtual")
            loss_coral_zdom_global_avg = _client_seen_weighted_avg(client_results, "loss_coral_zdom_global")
            loss_domain_unsup_pretrain_avg = _client_seen_weighted_avg(client_results, "loss_domain_unsup_pretrain")
            loss_domain_unsup_metadata_ce_avg = _client_seen_weighted_avg(client_results, "loss_domain_unsup_metadata_ce")
            loss_domain_unsup_var_avg = _client_seen_weighted_avg(client_results, "loss_domain_unsup_var")
            domain_unsup_active_rate = _client_seen_weighted_avg(client_results, "domain_unsup_active")
            stage1_domain_pretrain_active_rate = _client_seen_weighted_avg(client_results, "stage1_domain_pretrain_active")
            domain_unsup_zdom_cos_avg = _client_seen_weighted_avg(client_results, "domain_unsup_zdom_cos")
            domain_unsup_client_compact_avg = _client_seen_weighted_avg(client_results, "domain_unsup_client_compact")
            domain_unsup_client_radius_avg = _client_seen_weighted_avg(client_results, "domain_unsup_client_radius")
            domain_unsup_dom_entropy_avg = _client_seen_weighted_avg(client_results, "domain_unsup_dom_entropy")
            domain_unsup_dom_acc_avg = _client_seen_weighted_avg(client_results, "domain_unsup_dom_acc")
            domain_unsup_zdom_var_avg = _client_seen_weighted_avg(client_results, "domain_unsup_zdom_var")
            coral_zid_global_active_classes_avg = _client_seen_weighted_avg(client_results, "coral_zid_global_active_classes")
            coral_zid_global_mean_dist_avg = _client_seen_weighted_avg(client_results, "coral_zid_global_mean_dist")
            coral_zid_global_cov_dist_avg = _client_seen_weighted_avg(client_results, "coral_zid_global_cov_dist")
            coral_zid_global_skip_rate_avg = _client_seen_weighted_avg(client_results, "coral_zid_global_skip_rate")
            coral_zid_virtual_active_classes_avg = _client_seen_weighted_avg(client_results, "coral_zid_virtual_active_classes")
            coral_zid_virtual_mean_dist_avg = _client_seen_weighted_avg(client_results, "coral_zid_virtual_mean_dist")
            coral_zid_virtual_cov_dist_avg = _client_seen_weighted_avg(client_results, "coral_zid_virtual_cov_dist")
            coral_zid_virtual_skip_rate_avg = _client_seen_weighted_avg(client_results, "coral_zid_virtual_skip_rate")
            coral_zdom_global_active_classes_avg = _client_seen_weighted_avg(client_results, "coral_zdom_global_active_classes")
            coral_zdom_global_mean_dist_avg = _client_seen_weighted_avg(client_results, "coral_zdom_global_mean_dist")
            coral_zdom_global_cov_dist_avg = _client_seen_weighted_avg(client_results, "coral_zdom_global_cov_dist")
            coral_zdom_global_skip_rate_avg = _client_seen_weighted_avg(client_results, "coral_zdom_global_skip_rate")
            fed_fishr_active_rate = _client_seen_weighted_avg(client_results, "fed_fishr_active")
            fed_fishr_active_classes_avg = _client_seen_weighted_avg(client_results, "fed_fishr_active_classes")
            fed_fishr_var_dist_avg = _client_seen_weighted_avg(client_results, "fed_fishr_var_dist")
            fed_fishr_skip_rate_avg = _client_seen_weighted_avg(client_results, "fed_fishr_skip_rate")
            fed_fishr_target_ready_rate = _client_seen_weighted_avg(client_results, "fed_fishr_target_ready")
            fed_fishr_payload_bytes = int(
                sum(int(float(result.get("fed_fishr_payload_bytes", 0) or 0)) for result in client_results.values())
            )
            fed_proto_cos_avg = _client_seen_weighted_avg(client_results, "fed_proto_cos")
            vmb_tx_proto_cos_avg = _client_seen_weighted_avg(client_results, "vmb_tx_proto_cos")
            vmb_rx_proto_cos_avg = _client_seen_weighted_avg(client_results, "vmb_rx_proto_cos")
            tx_adv_r_acc_avg = _client_seen_weighted_avg(client_results, "tx_adv_r_acc")
            kd_active_rate = _client_seen_weighted_avg(client_results, "kd_active")
            logit_anchor_count_avg = _client_seen_weighted_avg(client_results, "logit_anchor_count")
            logit_anchor_payload_bytes = int(
                sum(int(float(result.get("logit_anchor_payload_bytes", 0) or 0)) for result in client_results.values())
            )
            coral_payload_bytes = int(sum(coral_stats_payload_size_bytes(stats) for stats in client_coral_stats.values()))
            activation_token_payload_bytes = int(
                sum(int(float(result.get("activation_token_payload_bytes", 0) or 0)) for result in client_results.values())
            )
            activation_token_compression_ratio = _client_seen_weighted_avg(client_results, "activation_token_compression_ratio")
            activation_token_quant_error = _client_seen_weighted_avg(client_results, "activation_token_quant_error")
            feature_probe_samples = int(sum(int(float(result.get("feature_probe_samples", 0) or 0)) for result in client_results.values()))
            loss_fedprox_avg = _client_seen_weighted_avg(client_results, "loss_fedprox")
            zdom_target_acc_avg = _client_seen_weighted_avg(client_results, "zdom_target_acc")
            grl_target_acc_avg = _client_seen_weighted_avg(client_results, "grl_target_acc")
            fed_cgrl_lambda_rx_adv_avg = _client_seen_weighted_avg(client_results, "fed_cgrl_lambda_rx_adv")
            fed_cgrl_base_lambda_avg = _client_seen_weighted_avg(client_results, "fed_cgrl_base_lambda")
            fed_cgrl_warmup_gate_avg = _client_seen_weighted_avg(client_results, "fed_cgrl_warmup_gate")
            fed_cgrl_leak_gate_avg = _client_seen_weighted_avg(client_results, "fed_cgrl_leak_gate")
            fed_cgrl_tx_gate_avg = _client_seen_weighted_avg(client_results, "fed_cgrl_tx_gate")
            fed_cgrl_conflict_gate_avg = _client_seen_weighted_avg(client_results, "fed_cgrl_conflict_gate")
            fed_cgrl_unclamped_lambda_avg = _client_seen_weighted_avg(client_results, "fed_cgrl_unclamped_lambda")
            fed_cgrl_enabled_avg = _client_seen_weighted_avg(client_results, "fed_cgrl_enabled")
            style_domains_avg = _client_seen_weighted_avg(client_results, "style_num_domains")
            style_views_avg = _client_seen_weighted_avg(client_results, "style_batch_views")
            style_entropy_avg = _client_seen_weighted_avg(client_results, "style_domain_entropy")
            style_dg_ready_rate = _client_seen_weighted_avg(client_results, "style_dg_ready")
            style_requested_remote_views_avg = _client_seen_weighted_avg(client_results, "style_requested_remote_views")
            style_appended_remote_views_avg = _client_seen_weighted_avg(client_results, "style_appended_remote_views")
            style_real_mix_active_rate = _client_seen_weighted_avg(client_results, "style_real_mix_active")
            style_zdom_probe = None
            for result in client_results.values():
                style_zdom_probe = _merge_style_zdom_probe(
                    style_zdom_probe,
                    result.get("style_zdom_probe"),
                    max_examples=int(getattr(self.cfg, "fl_style_zdom_probe_max_examples", 4) or 4),
                )
            style_zdom_flat = _style_zdom_probe_flat_metrics(style_zdom_probe)
            dg_diag = OrderedDict(
                (row_name, _client_seen_weighted_avg(client_results, metric_name))
                for metric_name, row_name in _DG_DIAG_FIELD_MAP.items()
            )
            vmb_client_drift_norm_avg = _client_seen_weighted_avg(client_results, "vmb_client_drift_norm")
            vmb_loss_domain_variance = _domain_metric_variance(client_results, "loss", "vmb_domain_id")
            vmb_comm_payload_bytes = int(
                sum(
                    int(float(result.get("vmb_client_payload_bytes", 0) or 0))
                    for result in client_results.values()
                )
            )
            prox_ratio = (
                loss_fedprox_avg / max(abs(loss_cls_avg), 1e-12)
                if math.isfinite(loss_fedprox_avg) and math.isfinite(loss_cls_avg)
                else float("nan")
            )
            label_hist_sum = _sum_client_hist(client_results, "label_hist")
            pred_hist_sum = _sum_client_hist(client_results, "pred_hist")
            row = {
                "round": round_idx,
                "round_time_s": round_time_s,
                "round_train_time_s": round_train_time_s,
                "round_eval_time_s": round_eval_time_s,
                "round_val_time_s": float(eval_timing.get("val_time_s", 0.0) or 0.0),
                "round_test_time_s": float(eval_timing.get("test_time_s", 0.0) or 0.0),
                "round_extra_eval_time_s": float(eval_timing.get("extra_eval_time_s", 0.0) or 0.0),
                "round_other_time_s": round_other_time_s,
                "train_mode": self.train_mode,
                "fl_local_objective": str(getattr(self.cfg, "fl_local_objective", "ce")),
                "selected_clients": selected,
                "client_num_samples": {cid: self.client_num_samples[cid] for cid in selected},
                "client_train": client_results,
                "client_train_loss_avg": train_loss_avg,
                "client_loss_cls_avg": loss_cls_avg,
                "client_loss_fishr_avg": loss_fishr_avg,
                "client_loss_fed_fishr_avg": loss_fed_fishr_avg,
                "client_loss_rx_adv_avg": loss_rx_adv_avg,
                "client_loss_baseline_sat_view_avg": loss_baseline_sat_view_avg,
                "client_loss_fed_proto_avg": loss_fed_proto_avg,
                "client_loss_vmb_tx_proto_avg": loss_vmb_tx_proto_avg,
                "client_loss_vmb_rx_proto_avg": loss_vmb_rx_proto_avg,
                "client_loss_tx_adv_r_avg": loss_tx_adv_r_avg,
                "client_loss_logit_kd_avg": loss_logit_kd_avg,
                "client_loss_supcon_id_avg": loss_supcon_id_avg,
                "client_loss_coral_zid_global_avg": loss_coral_zid_global_avg,
                "client_loss_coral_zid_virtual_avg": loss_coral_zid_virtual_avg,
                "client_loss_coral_zdom_global_avg": loss_coral_zdom_global_avg,
                "client_loss_domain_unsup_pretrain_avg": loss_domain_unsup_pretrain_avg,
                "client_loss_domain_unsup_metadata_ce_avg": loss_domain_unsup_metadata_ce_avg,
                "client_loss_domain_unsup_var_avg": loss_domain_unsup_var_avg,
                "client_domain_unsup_active_rate": domain_unsup_active_rate,
                "client_stage1_domain_pretrain_active_rate": stage1_domain_pretrain_active_rate,
                "client_domain_unsup_zdom_cos_avg": domain_unsup_zdom_cos_avg,
                "client_domain_unsup_client_compact_avg": domain_unsup_client_compact_avg,
                "client_domain_unsup_client_radius_avg": domain_unsup_client_radius_avg,
                "client_domain_unsup_dom_entropy_avg": domain_unsup_dom_entropy_avg,
                "client_domain_unsup_dom_acc_avg": domain_unsup_dom_acc_avg,
                "client_domain_unsup_zdom_var_avg": domain_unsup_zdom_var_avg,
                "client_coral_zid_global_active_classes_avg": coral_zid_global_active_classes_avg,
                "client_coral_zid_global_mean_dist_avg": coral_zid_global_mean_dist_avg,
                "client_coral_zid_global_cov_dist_avg": coral_zid_global_cov_dist_avg,
                "client_coral_zid_global_skip_rate_avg": coral_zid_global_skip_rate_avg,
                "client_coral_zid_virtual_active_classes_avg": coral_zid_virtual_active_classes_avg,
                "client_coral_zid_virtual_mean_dist_avg": coral_zid_virtual_mean_dist_avg,
                "client_coral_zid_virtual_cov_dist_avg": coral_zid_virtual_cov_dist_avg,
                "client_coral_zid_virtual_skip_rate_avg": coral_zid_virtual_skip_rate_avg,
                "client_coral_zdom_global_active_classes_avg": coral_zdom_global_active_classes_avg,
                "client_coral_zdom_global_mean_dist_avg": coral_zdom_global_mean_dist_avg,
                "client_coral_zdom_global_cov_dist_avg": coral_zdom_global_cov_dist_avg,
                "client_coral_zdom_global_skip_rate_avg": coral_zdom_global_skip_rate_avg,
                "client_fed_fishr_active_rate": fed_fishr_active_rate,
                "client_fed_fishr_active_classes_avg": fed_fishr_active_classes_avg,
                "client_fed_fishr_var_dist_avg": fed_fishr_var_dist_avg,
                "client_fed_fishr_skip_rate_avg": fed_fishr_skip_rate_avg,
                "client_fed_fishr_target_ready_rate": fed_fishr_target_ready_rate,
                "fed_fishr_payload_bytes": fed_fishr_payload_bytes,
                "client_fed_proto_cos_avg": fed_proto_cos_avg,
                "client_vmb_tx_proto_cos_avg": vmb_tx_proto_cos_avg,
                "client_vmb_rx_proto_cos_avg": vmb_rx_proto_cos_avg,
                "client_tx_adv_r_acc_avg": tx_adv_r_acc_avg,
                "client_kd_active_rate": kd_active_rate,
                "client_logit_anchor_count_avg": logit_anchor_count_avg,
                "logit_anchor_payload_bytes": logit_anchor_payload_bytes,
                "coral_payload_bytes": coral_payload_bytes,
                "activation_token_payload_bytes": activation_token_payload_bytes,
                "activation_token_compression_ratio": activation_token_compression_ratio,
                "activation_token_quant_error": activation_token_quant_error,
                "feature_probe_samples": feature_probe_samples,
                "client_loss_fedprox_avg": loss_fedprox_avg,
                "client_fedprox_ratio": prox_ratio,
                "client_zdom_target_acc_avg": zdom_target_acc_avg,
                "client_grl_target_acc_avg": grl_target_acc_avg,
                "client_fed_cgrl_lambda_rx_adv_avg": fed_cgrl_lambda_rx_adv_avg,
                "client_fed_cgrl_base_lambda_avg": fed_cgrl_base_lambda_avg,
                "client_fed_cgrl_warmup_gate_avg": fed_cgrl_warmup_gate_avg,
                "client_fed_cgrl_leak_gate_avg": fed_cgrl_leak_gate_avg,
                "client_fed_cgrl_tx_gate_avg": fed_cgrl_tx_gate_avg,
                "client_fed_cgrl_conflict_gate_avg": fed_cgrl_conflict_gate_avg,
                "client_fed_cgrl_unclamped_lambda_avg": fed_cgrl_unclamped_lambda_avg,
                "client_fed_cgrl_enabled_avg": fed_cgrl_enabled_avg,
                "client_train_acc_avg": train_acc_avg,
                "client_label_hist_sum": label_hist_sum,
                "client_pred_hist_sum": pred_hist_sum,
                "global_test_eval_ran": bool(eval_stats.get("test_eval_ran", True)),
                "global_next_test_eval_round": eval_stats.get("next_test_eval_round"),
                "global_test_eval_interval": self._heavy_eval_interval(),
                "global_test_eval_last_n": self._heavy_eval_last_n(),
                "global_test_eval_final_offsets": self._heavy_eval_final_offsets(),
                "global_test_eval_final_rounds": sorted(self._heavy_eval_final_rounds()),
                "global_eval_acc": val_acc,
                "global_test_overall_acc": float(test_overall.get("tx_acc", float("nan"))),
                "global_test_overall": test_overall,
                "global_strict_udu_acc": strict,
                "global_named_test": named_stats,
                "global_named_test_tx_acc": eval_stats.get("named_tx_acc", _named_tx_acc_summary(named_stats)),
                "global_extra_tests": extra_tests,
                "global_eval_timing": eval_timing,
                "global_proto_fusion": eval_stats.get("proto_fusion", {}),
                "global_style_collab_fusion": eval_stats.get("style_collab_fusion", {}),
                "global_proto_summary": proto_summary,
                "global_coral_summary": coral_summary,
                "global_fed_fishr_summary": self.global_fed_fishr_summary,
                "global_vmb_proto_summary": vmb_proto_summary,
                "global_logit_anchor_summary": logit_anchor_summary,
                "global_fed_cgrl_summary": fed_cgrl_summary,
                "global_feature_probe_summary": feature_probe_summary,
                "vmb_server_update": server_update,
                "vmb_gradient_cosine": vmb_gradient_cosine,
                "vmb_conflict_summary": vmb_conflict_summary,
                "fed_cgrl_conflict_summary": fed_cgrl_conflict_summary,
                "vmb_domain_weights": vmb_domain_weights,
                "vmb_stage": current_vmb_stage,
                "vmb_client_drift_norm_avg": vmb_client_drift_norm_avg,
                "vmb_loss_domain_variance": vmb_loss_domain_variance,
                "vmb_comm_payload_bytes": vmb_comm_payload_bytes,
                "global_proto_evidence_summary": proto_evidence_summary,
                "global_style_summary": style_summary,
                "style_replay_enabled": bool(self.style_bank is not None and round_idx >= int(getattr(self.cfg, "fl_style_replay_start_round", 20))),
                "style_phys_enabled": bool(self.style_transform is not None and round_idx >= int(getattr(self.cfg, "fl_style_phys_start_round", 20))),
                "style_dg_enabled": bool(round_idx >= int(getattr(self.cfg, "fl_style_dg_start_round", 40))),
                "style_bank_remote_sample_accept_rate": _client_seen_weighted_avg(client_results, "diag_style_batch_active"),
                "client_style_num_domains_avg": style_domains_avg,
                "client_style_batch_views_avg": style_views_avg,
                "client_style_domain_entropy_avg": style_entropy_avg,
                "client_style_dg_ready_rate": style_dg_ready_rate,
                "client_style_requested_remote_views_avg": style_requested_remote_views_avg,
                "client_style_appended_remote_views_avg": style_appended_remote_views_avg,
                "client_style_real_mix_active_rate": style_real_mix_active_rate,
                "client_style_domain_label_mode": self._style_domain_label_mode(),
                "client_style_zdom_probe": style_zdom_probe or {},
                **{f"client_{k}": v for k, v in style_zdom_flat.items()},
                **dg_diag,
                "fedprox_mu": float(getattr(self.cfg, "fedprox_mu", 0.0)),
                "fl_sat_aug_mode": str(getattr(self.cfg, "fl_sat_aug_mode", "baseline_view")),
                "lr": round_lr,
                "mixstyle_state": mixstyle_state,
                "num_shared_keys": len(self.global_state),
                "num_excluded_keys": len(exclude_keys),
            }
            self._append_logs(row)
            prox_part = ""
            if self.train_mode == "fedprox":
                prox_part = (
                    f" cls={loss_cls_avg:.4f} prox={loss_fedprox_avg:.6g} "
                    f"prox/cls={prox_ratio:.3e} mu={row['fedprox_mu']:.3g}"
                )
            print(
                f"[FED][R{round_idx:03d}] mode={self.train_mode} clients={len(selected)}/{len(self.client_splits)} "
                f"loss={row['client_train_loss_avg']:.4f}{prox_part} train_acc={row['client_train_acc_avg']:.2f}% "
                f"val={val_acc:.2f}% test={test_overall.get('tx_acc', float('nan')):.2f}% strict_udu={strict:.2f}% "
                f"lr={round_lr:.3e} time={round_time_s:.1f}s",
                flush=True,
            )
            print(
                f"[FED-TIME][R{round_idx:03d}] total={round_time_s:.1f}s "
                f"train={round_train_time_s:.1f}s eval={round_eval_time_s:.1f}s "
                f"val={float(eval_timing.get('val_time_s', 0.0) or 0.0):.1f}s "
                f"test={float(eval_timing.get('test_time_s', 0.0) or 0.0):.1f}s "
                f"extra={float(eval_timing.get('extra_eval_time_s', 0.0) or 0.0):.1f}s "
                f"other={round_other_time_s:.1f}s",
                flush=True,
            )
            if not bool(row.get("global_test_eval_ran", True)):
                next_eval = row.get("global_next_test_eval_round")
                next_text = f"R{int(next_eval):03d}" if next_eval is not None else "none"
                print(
                    f"[FED-TEST-SKIP][R{round_idx:03d}] next={next_text} "
                    f"interval={self._heavy_eval_interval()} last_n={self._heavy_eval_last_n()} "
                    f"final_offsets={self._heavy_eval_final_offsets()} "
                    f"scope=named_test+extra_eval+proto_fusion+style_collab",
                    flush=True,
                )
            dg_objective = str(row.get("fl_local_objective", "ce")).lower() in {
                "bex02_dg",
                "strong_dg",
                "dg",
                "receiver_agnostic_bex02",
                "ra_bex02",
                "local_virtual_bex02",
            }
            if dg_objective or float(row.get("client_diag_sat_aug_active_rate", 0.0) or 0.0) > 0.0:
                print(
                    f"[FED-DG-DIAG][R{round_idx:03d}] "
                    f"domains={float(row.get('client_diag_domain_count_avg', float('nan'))):.2f} "
                    f"fishr_domains={float(row.get('client_diag_fishr_domain_count_avg', float('nan'))):.2f} "
                    f"fishr_active={float(row.get('client_diag_fishr_active_rate', float('nan'))):.2f} "
                    f"dom_active={float(row.get('client_diag_domain_loss_active_rate', float('nan'))):.2f} "
                    f"cons_active={float(row.get('client_diag_cons_active_rate', float('nan'))):.2f} "
                    f"rx_adv_active={float(row.get('client_diag_rx_adv_active_rate', float('nan'))):.2f} "
                    f"grl_rx_adv_active={float(row.get('client_diag_rx_adv_active_rate', float('nan'))):.2f} "
                    f"zdom_acc={float(row.get('client_zdom_target_acc_avg', float('nan'))):.2f} "
                    f"grl_acc={float(row.get('client_grl_target_acc_avg', float('nan'))):.2f} "
                    f"sat_aug={float(row.get('client_diag_sat_aug_active_rate', float('nan'))):.2f} "
                    f"baseline_view={float(row.get('client_diag_baseline_sat_view_active_rate', float('nan'))):.2f} "
                    f"sat_cls={float(row.get('client_diag_sat_cls_active_rate', float('nan'))):.2f} "
                    f"sat_cons={float(row.get('client_diag_sat_cons_active_rate', float('nan'))):.2f} "
                    f"style_batch={float(row.get('client_diag_style_batch_active_rate', float('nan'))):.2f}",
                    flush=True,
                )
            if bool((row.get("global_fed_fishr_summary") or {}).get("enabled", False)):
                fishr_info = row.get("global_fed_fishr_summary") or {}
                print(
                    f"[FED-FISHR][R{round_idx:03d}] "
                    f"mode={fishr_info.get('mode', self.fed_fishr_mode)} "
                    f"active={int(bool(fishr_info.get('active', False)))} "
                    f"reweight={int(bool(fishr_info.get('reweight_active', False)))} "
                    f"classes={int(fishr_info.get('active_classes', 0) or 0)} "
                    f"clients={int(fishr_info.get('client_count', 0) or 0)} "
                    f"mismatch_mean={float(fishr_info.get('mismatch_mean', float('nan'))):.6g} "
                    f"mismatch_max={float(fishr_info.get('mismatch_max', float('nan'))):.6g} "
                    f"target_var={float(fishr_info.get('target_var_mean', float('nan'))):.6g} "
                    f"w_delta={float(fishr_info.get('weight_max_delta', 0.0) or 0.0):.4f} "
                    f"target_loss={loss_fed_fishr_avg:.6g} "
                    f"payload={int(row.get('fed_fishr_payload_bytes', 0) or 0)}",
                    flush=True,
                )
            if self.fed_cgrl.enabled:
                cgrl = row.get("global_fed_cgrl_summary") or {}
                print(
                    f"[FED-CGRL][R{round_idx:03d}] "
                    f"lambda={float(row.get('client_fed_cgrl_lambda_rx_adv_avg', float('nan'))):.4f} "
                    f"warmup={float(row.get('client_fed_cgrl_warmup_gate_avg', float('nan'))):.3f} "
                    f"leak={float(row.get('client_fed_cgrl_leak_gate_avg', float('nan'))):.3f} "
                    f"tx={float(row.get('client_fed_cgrl_tx_gate_avg', float('nan'))):.3f} "
                    f"conflict={float(row.get('client_fed_cgrl_conflict_gate_avg', float('nan'))):.3f} "
                    f"grl_acc={float(row.get('client_grl_target_acc_avg', float('nan'))):.2f} "
                    f"grl_p90={float(cgrl.get('grl_target_acc_p90', float('nan'))):.2f} "
                    f"grl_max={float(cgrl.get('grl_target_acc_max', float('nan'))):.2f} "
                    f"worst={cgrl.get('grl_target_acc_worst_client', '')} "
                    f"lambda_p90={float(cgrl.get('lambda_rx_adv_p90', float('nan'))):.4f} "
                    f"conflict_src={cgrl.get('conflict_source', 'none')} "
                    f"conflict_cos={float(cgrl.get('conflict_cos_min', float('nan'))):.4f} "
                    f"loss_cls={float(row.get('client_loss_cls_avg', float('nan'))):.4f} "
                    f"state_clients={int(cgrl.get('client_count', 0) or 0)}",
                    flush=True,
                )
            if bool(getattr(self.cfg, "use_mixstyle", False)) or mixstyle_state.get("phase") not in {"missing", "disabled"}:
                print(
                    f"[FED-MIXSTYLE][R{round_idx:03d}] phase={mixstyle_state.get('phase', 'unknown')} "
                    f"enabled={int(bool(mixstyle_state.get('enabled', False)))} "
                    f"p={float(mixstyle_state.get('p', 0.0)):.3f} "
                    f"strength={float(mixstyle_state.get('strength', 0.0)):.3f} "
                    f"anneal_t={float(mixstyle_state.get('anneal_t', 0.0)):.3f}",
                    flush=True,
                )
            if bool((row.get("global_proto_summary") or {}).get("enabled", False)):
                proto_info = row["global_proto_summary"]
                print(
                    f"[FED-PROTO][R{round_idx:03d}] "
                    f"classes={int(proto_info.get('class_count_nonzero', 0))} "
                    f"domains={int(proto_info.get('domain_count_nonzero', 0))} "
                    f"loss={loss_fed_proto_avg:.6g} cos={fed_proto_cos_avg:.4f}",
                    flush=True,
                )
            if bool((row.get("global_coral_summary") or {}).get("enabled", False)):
                coral_info = row["global_coral_summary"]
                print(
                    f"[FED-CORAL][R{round_idx:03d}] "
                    f"classes={int(coral_info.get('class_count_nonzero', 0))} "
                    f"count={float(coral_info.get('total_count', 0.0) or 0.0):.1f} "
                    f"zid_global_loss={loss_coral_zid_global_avg:.6g} "
                    f"zid_virtual_loss={loss_coral_zid_virtual_avg:.6g} "
                    f"zdom_global_loss={loss_coral_zdom_global_avg:.6g} "
                    f"active_global={float(row.get('client_diag_coral_global_active_rate', float('nan'))):.2f} "
                    f"active_virtual={float(row.get('client_diag_coral_virtual_active_rate', float('nan'))):.2f} "
                    f"active_zdom={float(row.get('client_diag_coral_zdom_active_rate', float('nan'))):.2f} "
                    f"zid_global_mean={coral_zid_global_mean_dist_avg:.6g} "
                    f"zid_virtual_mean={coral_zid_virtual_mean_dist_avg:.6g} "
                    f"bytes={int(row.get('coral_payload_bytes', 0) or 0)}",
                    flush=True,
                )
            if self.vmb_enabled:
                vmb_proto = row.get("global_vmb_proto_summary") or {}
                vmb_update = row.get("vmb_server_update") or {}
                vmb_cos = row.get("vmb_gradient_cosine") or {}
                print(
                    f"[FED-VMB][R{round_idx:03d}] stage={row.get('vmb_stage')} "
                    f"grad_norm={float(vmb_update.get('grad_norm', float('nan'))):.6g} "
                    f"grad_cos_mean={float(vmb_cos.get('mean', float('nan'))):.4f} "
                    f"tx_proto={int(vmb_proto.get('tx_count_nonzero', 0))} "
                    f"rx_proto={int(vmb_proto.get('rx_count_nonzero', 0))} "
                    f"tx_proto_loss={loss_vmb_tx_proto_avg:.6g} "
                    f"rx_proto_loss={loss_vmb_rx_proto_avg:.6g} "
                    f"tx_adv_r={loss_tx_adv_r_avg:.6g} "
                    f"tx_adv_r_acc={tx_adv_r_acc_avg:.2f} "
                    f"client_drift={float(row.get('vmb_client_drift_norm_avg', float('nan'))):.6g} "
                    f"loss_domain_var={float(row.get('vmb_loss_domain_variance', float('nan'))):.6g} "
                    f"comm_bytes={int(row.get('vmb_comm_payload_bytes', 0) or 0)}",
                    flush=True,
                )
            if float(row.get("client_stage1_domain_pretrain_active_rate", 0.0) or 0.0) > 0.0:
                print(
                    f"[FED-DOMAIN-PRETRAIN][R{round_idx:03d}] "
                    f"active={float(row.get('client_domain_unsup_active_rate', float('nan'))):.2f} "
                    f"loss={float(row.get('client_loss_domain_unsup_pretrain_avg', float('nan'))):.6g} "
                    f"metadata_ce={float(row.get('client_loss_domain_unsup_metadata_ce_avg', float('nan'))):.6g} "
                    f"var={float(row.get('client_loss_domain_unsup_var_avg', float('nan'))):.6g} "
                    f"zdom_cos={float(row.get('client_domain_unsup_zdom_cos_avg', float('nan'))):.4f} "
                    f"client_compact={float(row.get('client_domain_unsup_client_compact_avg', float('nan'))):.6g} "
                    f"client_radius={float(row.get('client_domain_unsup_client_radius_avg', float('nan'))):.6g} "
                    f"dom_entropy={float(row.get('client_domain_unsup_dom_entropy_avg', float('nan'))):.4f} "
                    f"dom_acc={float(row.get('client_domain_unsup_dom_acc_avg', float('nan'))):.2f} "
                    f"zdom_var={float(row.get('client_domain_unsup_zdom_var_avg', float('nan'))):.4f}",
                    flush=True,
                )
            if bool((row.get("global_proto_evidence_summary") or {}).get("enabled", False)):
                proto_ev = row["global_proto_evidence_summary"]
                fusion = (row.get("global_proto_fusion") or {}).get("aggregate", {}) or {}
                print(
                    f"[FED-PROTOBANK][R{round_idx:03d}] "
                    f"classes={int(proto_ev.get('num_classes', 0))} "
                    f"prototypes={int(proto_ev.get('num_prototypes', 0))} "
                    f"mean_rel={float(proto_ev.get('mean_reliability', 0.0)):.3f} "
                    f"rescue={int(fusion.get('rescue', 0))} harm={int(fusion.get('harm', 0))} "
                    f"net={int(fusion.get('net_gain', 0))}",
                    flush=True,
                )
            if bool((row.get("global_style_summary") or {}).get("enabled", False)):
                style_info = row["global_style_summary"]
                print(
                    f"[FED-STYLE][R{round_idx:03d}] "
                    f"centroids={int(style_info.get('num_centroids', 0))} "
                    f"packets={int(style_info.get('num_packets_seen', 0))} "
                    f"bytes={int(style_info.get('size_bytes', 0))} "
                    f"mean_l2={float(style_info.get('mean_pairwise_l2', 0.0)):.4f} "
                    f"views={float(row.get('client_style_batch_views_avg', float('nan'))):.2f} "
                    f"remote_views={float(row.get('client_style_appended_remote_views_avg', float('nan'))):.2f}/"
                    f"{float(row.get('client_style_requested_remote_views_avg', float('nan'))):.2f} "
                    f"domains={float(row.get('client_style_num_domains_avg', float('nan'))):.2f} "
                    f"entropy={float(row.get('client_style_domain_entropy_avg', float('nan'))):.3f} "
                    f"dg_ready={float(row.get('client_style_dg_ready_rate', float('nan'))):.2f} "
                    f"real_mix={float(row.get('client_style_real_mix_active_rate', float('nan'))):.2f}",
                    flush=True,
                )
            feature_probe = row.get("global_feature_probe_summary") or {}
            if bool(feature_probe.get("enabled", False)):
                print(
                    f"[FED-FEATURE-PROBE][R{round_idx:03d}] "
                    f"wrote={int(bool(feature_probe.get('wrote', False)))} "
                    f"samples={int(feature_probe.get('num_samples', 0) or 0)} "
                    f"path={feature_probe.get('path', '')}",
                    flush=True,
                )
            style_probe = row.get("client_style_zdom_probe") or {}
            if isinstance(style_probe, Mapping) and any(
                int((style_probe.get(name, {}) or {}).get("total", 0) or 0) > 0
                for name in _STYLE_ZDOM_BUCKETS
            ):
                virtual_examples = json.dumps(
                    (style_probe.get("virtual", {}) or {}).get("examples", []) or [],
                    ensure_ascii=False,
                    default=str,
                )
                real_examples = json.dumps(
                    (style_probe.get("real", {}) or {}).get("examples", []) or [],
                    ensure_ascii=False,
                    default=str,
                )
                print(
                    f"[FED-STYLE-ZDOM][R{round_idx:03d}] "
                    f"mode={style_probe.get('mode', row.get('client_style_domain_label_mode'))} "
                    f"clean={_style_zdom_probe_text(style_probe.get('clean', {}))} "
                    f"virtual={_style_zdom_probe_text(style_probe.get('virtual', {}))} "
                    f"real={_style_zdom_probe_text(style_probe.get('real', {}))} "
                    f"virtual_targets={(style_probe.get('virtual', {}) or {}).get('target_hist', {})} "
                    f"virtual_preds={(style_probe.get('virtual', {}) or {}).get('pred_hist', {})} "
                    f"real_targets={(style_probe.get('real', {}) or {}).get('target_hist', {})} "
                    f"real_preds={(style_probe.get('real', {}) or {}).get('pred_hist', {})} "
                    f"virtual_examples={virtual_examples} real_examples={real_examples}",
                    flush=True,
                )
            if bool((row.get("global_style_collab_fusion") or {}).get("enabled", False)):
                collab = row["global_style_collab_fusion"]
                fusion = collab.get("aggregate", {}) or {}
                print(
                    f"[FED-STYLE-COLLAB][R{round_idx:03d}] "
                    f"fusion={collab.get('fusion')} views={int(collab.get('views', 0))} "
                    f"base={float(fusion.get('base_tx_acc', float('nan'))):.2f}% "
                    f"fused={float(fusion.get('fused_tx_acc', float('nan'))):.2f}% "
                    f"rescue={int(fusion.get('rescue', 0))} harm={int(fusion.get('harm', 0))} "
                    f"net={int(fusion.get('net_gain', 0))}",
                    flush=True,
                )
            if named_stats:
                print(
                    f"[FED-TEST][R{round_idx:03d}] overall_tx={test_overall.get('tx_acc', float('nan')):.2f}% "
                    f"({int(test_overall.get('tx_correct', 0))}/{int(test_overall.get('tx_total', 0))})",
                    flush=True,
                )
                print(f"[FED-TEST-SPLIT][R{round_idx:03d}]", flush=True)
                for line in _format_named_test_lines(named_stats, self.named_test_meta):
                    print(line, flush=True)
            extra_lines = _format_extra_test_lines(extra_tests)
            if extra_lines:
                print(f"[FED-SAT-TEST][R{round_idx:03d}]", flush=True)
                for line in extra_lines:
                    print(line, flush=True)
            if round_idx <= 3 or row["client_train_acc_avg"] <= 0.0:
                print(
                    f"[FED-DIAG][R{round_idx:03d}] label_hist={label_hist_sum} pred_hist={pred_hist_sum}",
                    flush=True,
                )
            print(_ROUND_END_SEPARATOR, flush=True)
            print(
                f"[FED-ROUND-END][{_round_tag(round_idx, total_rounds)}] "
                f"best_round={best_round} best_score={best_score:.4f}",
                flush=True,
            )
            print(_ROUND_END_SEPARATOR, flush=True)

        summary = {
            "train_mode": self.train_mode,
            "fl_local_objective": str(getattr(self.cfg, "fl_local_objective", "ce")),
            "best_round": best_round,
            "best_score": best_score,
            "num_clients": len(self.client_splits),
            "best_checkpoint": self.best_checkpoint,
            "last_checkpoint": self.last_checkpoint,
            "config_json": self.config_json,
            "best_eval": best_eval_stats,
            "last_eval": last_eval_stats,
            "global_proto_summary": _fed_proto_summary(self.global_proto_stats),
            "global_coral_summary": self.global_coral_bank.summary() if self.global_coral_bank is not None else {"enabled": False},
            "global_fed_fishr_summary": dict(self.global_fed_fishr_summary),
            "global_vmb_proto_summary": self.vmb_proto_bank.summary() if self.vmb_proto_bank is not None else {"enabled": False},
            "global_logit_anchor_summary": self.logit_anchor_bank.summary() if self.logit_anchor_bank is not None else {"enabled": False},
            "global_fed_cgrl_summary": self.fed_cgrl.round_summary(),
            "global_feature_probe_summary": dict(self.last_feature_probe_summary),
            "global_proto_evidence_summary": self._proto_evidence_summary(),
            "global_style_summary": dict(self.global_style_summary),
            "config_event": {
                "timestamp": config_snapshot.get("timestamp"),
                "path": self.config_json,
            },
        }
        with open(self.summary_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
        return summary
