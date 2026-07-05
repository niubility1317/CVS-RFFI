#!/usr/bin/env python
"""OSPR-CI source-heldout prototype repair for Stage2-C qknn8.

OSPR-CI keeps target_unknown rows evaluation-only. It trains a compact residual
adapter from source old, source-heldout old calibration, source-side proxy
unknown, and target old/seen-new K-shot support, then delegates collaborative
M=1..R scoring to the existing qknn8 ENPC/SLEV backends.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))
SCRIPTS_ROOT = CODE_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from phase2_collaborative_open_set_qknn_eval import (  # noqa: E402
    PROXY_UNKNOWN_ROLE,
    UNKNOWN_ROLE,
    _normalize_rows,
    load_feature_npz,
)
from phase2_old_floor_hnfr_adapter_ci_eval import (  # noqa: E402
    _true_margin_mean,
    _virtual_boundary_samples,
    old_floor_margin_loss,
)
from phase2_proxy_adapter_ci_eval import (  # noqa: E402
    LowRankResidualAdapter,
    _backend_common_argv,
    _build_prototypes,
    _proto_logits,
    _state_bytes,
    apply_adapter,
    build_training_plan,
    save_adapted_npz,
)
from phase2_orbit_enpc_ci_eval import (  # noqa: E402
    _write_csv as _write_enpc_csv,
    parse_args as parse_enpc_args,
    run_enpc_ci,
)
from phase2_orbit_slev_ci_eval import (  # noqa: E402
    _write_csv as _write_slev_csv,
    parse_args as parse_slev_args,
    run_slev_ci,
)


@dataclass(frozen=True)
class OsprTrainingPlan:
    source_fit_indices: list[int]
    source_holdout_indices: list[int]
    proxy_unknown_indices: list[int]
    support_indices: list[int]
    support_labels: list[str]
    target_unknown_indices: list[int]
    target_receivers: list[str]
    old_labels: list[str]
    seen_new_labels: list[str]
    unknown_labels: list[str]
    source_holdout_per_class: int

    @property
    def target_unknown_training_count(self) -> int:
        return 0

    @property
    def source_holdout_calibration_count(self) -> int:
        return len(self.source_holdout_indices)


def build_ospr_training_plan(
    payload: Mapping[str, Any],
    *,
    k_shot: int,
    query_per_class: int,
    seed: int,
    support_selection_policy: str,
    source_holdout_per_class: int,
) -> OsprTrainingPlan:
    base = build_training_plan(
        payload,
        k_shot=int(k_shot),
        query_per_class=int(query_per_class),
        seed=int(seed),
        support_selection_policy=str(support_selection_policy),
    )
    tx_ids = np.asarray(payload["tx_ids"]).astype(str)
    by_label: dict[str, list[int]] = {}
    for idx in base.source_old_indices:
        by_label.setdefault(str(tx_ids[int(idx)]), []).append(int(idx))

    rng = np.random.default_rng(int(seed))
    source_fit: list[int] = []
    source_holdout: list[int] = []
    holdout_n = int(source_holdout_per_class)
    if holdout_n <= 0:
        raise RuntimeError("LOCAL_PROTOCOL_REPAIR_REQUIRED: OSPR-CI requires source_holdout_per_class > 0")
    for label in sorted(by_label):
        values = sorted(by_label[label])
        if len(values) <= holdout_n:
            raise RuntimeError(
                "LOCAL_DATASET_EXTENSION_REQUIRED: insufficient source-old rows for source-heldout calibration; "
                f"label={label}, available={len(values)}, holdout={holdout_n}"
            )
        perm = values.copy()
        rng.shuffle(perm)
        chosen = sorted(perm[:holdout_n])
        fit = sorted(perm[holdout_n:])
        source_holdout.extend(chosen)
        source_fit.extend(fit)

    forbidden = set(base.target_unknown_indices)
    training_like = set(source_fit) | set(source_holdout) | set(base.proxy_unknown_indices) | set(base.support_indices)
    leak = sorted(forbidden & training_like)
    if leak:
        raise RuntimeError(f"LOCAL_PROTOCOL_REPAIR_REQUIRED: target_unknown leaked into OSPR-CI training: {leak[:5]}")

    return OsprTrainingPlan(
        source_fit_indices=source_fit,
        source_holdout_indices=source_holdout,
        proxy_unknown_indices=list(base.proxy_unknown_indices),
        support_indices=list(base.support_indices),
        support_labels=list(base.support_labels),
        target_unknown_indices=list(base.target_unknown_indices),
        target_receivers=list(base.target_receivers),
        old_labels=list(base.old_labels),
        seen_new_labels=list(base.seen_new_labels),
        unknown_labels=list(base.unknown_labels),
        source_holdout_per_class=holdout_n,
    )


def _labels_in_mask(labels: torch.Tensor, allowed_positions: torch.Tensor) -> torch.Tensor:
    if int(labels.numel()) == 0 or int(allowed_positions.numel()) == 0:
        return torch.zeros_like(labels, dtype=torch.bool)
    return (labels.view(-1, 1) == allowed_positions.view(1, -1)).any(dim=1)


def _ospr_state_bytes(dim: int, rank: int, class_count: int, support_count: int) -> dict[str, int]:
    state = _state_bytes(dim, rank, class_count)
    state["qknn8_support_int8_bytes"] = int(dim) * int(support_count)
    state["total_fp16_state_bytes"] = int(state["total_fp16_state_bytes"]) + int(state["qknn8_support_int8_bytes"])
    return state


def train_ospr_adapter(
    payload: Mapping[str, Any],
    plan: OsprTrainingPlan,
    args: argparse.Namespace,
) -> tuple[LowRankResidualAdapter, dict[str, Any]]:
    requested_device = str(args.device)
    device = torch.device(requested_device if torch.cuda.is_available() or not requested_device.startswith("cuda") else "cpu")
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    features_np = _normalize_rows(np.asarray(payload["features"], dtype=np.float32))
    tx_ids = np.asarray(payload["tx_ids"]).astype(str)
    dim = int(features_np.shape[1])
    source_fit_labels = [str(tx_ids[i]) for i in plan.source_fit_indices]
    source_holdout_labels = [str(tx_ids[i]) for i in plan.source_holdout_indices]
    support_labels = [str(v) for v in plan.support_labels]
    proto_labels, proto_np = _build_prototypes(
        features_np[np.asarray(plan.source_fit_indices + plan.support_indices, dtype=int)],
        source_fit_labels + support_labels,
    )
    label_to_pos = {label: i for i, label in enumerate(proto_labels)}
    fit_y = np.asarray([label_to_pos[str(tx_ids[i])] for i in plan.source_fit_indices], dtype=np.int64)
    holdout_y = np.asarray([label_to_pos[str(tx_ids[i])] for i in plan.source_holdout_indices], dtype=np.int64)
    support_y = np.asarray([label_to_pos[str(v)] for v in support_labels], dtype=np.int64)
    old_positions_np = np.asarray([label_to_pos[label] for label in plan.old_labels if label in label_to_pos], dtype=np.int64)
    seen_positions_np = np.asarray([label_to_pos[label] for label in plan.seen_new_labels if label in label_to_pos], dtype=np.int64)

    x = torch.as_tensor(features_np, dtype=torch.float32, device=device)
    fit_idx = torch.as_tensor(plan.source_fit_indices, dtype=torch.long, device=device)
    holdout_idx = torch.as_tensor(plan.source_holdout_indices, dtype=torch.long, device=device)
    support_idx = torch.as_tensor(plan.support_indices, dtype=torch.long, device=device)
    proxy_idx_all = torch.as_tensor(plan.proxy_unknown_indices, dtype=torch.long, device=device)
    y_by_global = torch.full((int(features_np.shape[0]),), -1, dtype=torch.long, device=device)
    y_by_global.index_copy_(0, fit_idx, torch.as_tensor(fit_y, dtype=torch.long, device=device))
    y_by_global.index_copy_(0, holdout_idx, torch.as_tensor(holdout_y, dtype=torch.long, device=device))
    y_by_global.index_copy_(0, support_idx, torch.as_tensor(support_y, dtype=torch.long, device=device))
    old_positions = torch.as_tensor(old_positions_np, dtype=torch.long, device=device)
    seen_positions = torch.as_tensor(seen_positions_np, dtype=torch.long, device=device)
    prototypes = torch.as_tensor(proto_np, dtype=torch.float32, device=device)

    adapter = LowRankResidualAdapter(dim, int(args.adapter_rank), float(args.adapter_alpha), float(args.dropout)).to(device)
    opt = torch.optim.AdamW(adapter.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    rng = np.random.default_rng(int(args.seed))
    losses: list[float] = []
    start_time = time.perf_counter()
    fit_count = int(fit_idx.numel())
    holdout_count = int(holdout_idx.numel())
    support_count = int(support_idx.numel())
    proxy_count = int(proxy_idx_all.numel())
    batch = max(1, int(args.batch_size))

    for _epoch in range(int(args.adapter_epochs)):
        adapter.train()
        fit_order = fit_idx[torch.randperm(fit_count, device=device)]
        holdout_order = holdout_idx[torch.randperm(holdout_count, device=device)]
        support_order = support_idx[torch.randperm(support_count, device=device)]
        epoch_loss = 0.0
        steps = 0
        max_steps = max(math.ceil(fit_count / batch), math.ceil(support_count / batch), 1)
        for step in range(max_steps):
            start = step * batch
            fit_batch = fit_order[start % fit_count : min(start % fit_count + batch, fit_count)]
            if fit_batch.numel() == 0:
                fit_batch = fit_order[: min(batch, fit_count)]
            hold_batch = holdout_order[start % holdout_count : min(start % holdout_count + batch, holdout_count)]
            if hold_batch.numel() == 0:
                hold_batch = holdout_order[: min(batch, holdout_count)]
            sup_batch = support_order[start % support_count : min(start % support_count + batch, support_count)]
            if sup_batch.numel() == 0:
                sup_batch = support_order[: min(batch, support_count)]
            proxy_take = min(max(int(fit_batch.numel()), int(sup_batch.numel()), int(args.virtual_count)), proxy_count)
            proxy_np = rng.choice(proxy_count, size=proxy_take, replace=proxy_take > proxy_count)
            proxy_batch = proxy_idx_all[torch.as_tensor(proxy_np, dtype=torch.long, device=device)]

            z_fit0 = x.index_select(0, fit_batch)
            z_hold0 = x.index_select(0, hold_batch)
            z_sup0 = x.index_select(0, sup_batch)
            z_proxy0 = x.index_select(0, proxy_batch)
            virtual0 = _virtual_boundary_samples(
                prototypes,
                z_proxy0,
                count=int(args.virtual_count),
                mix_low=float(args.virtual_mix_low),
                mix_high=float(args.virtual_mix_high),
                noise_scale=float(args.virtual_noise_scale),
            )
            z_fit = adapter(z_fit0)
            z_hold = adapter(z_hold0)
            z_sup = adapter(z_sup0)
            z_proxy = adapter(z_proxy0)
            z_virtual = adapter(virtual0)

            fit_y_t = y_by_global.index_select(0, fit_batch)
            hold_y_t = y_by_global.index_select(0, hold_batch)
            sup_y_t = y_by_global.index_select(0, sup_batch)
            sup_old_mask = _labels_in_mask(sup_y_t, old_positions)
            sup_seen_mask = _labels_in_mask(sup_y_t, seen_positions)

            fit_logits = _proto_logits(z_fit, prototypes.detach(), float(args.proto_temperature))
            hold_logits = _proto_logits(z_hold, prototypes.detach(), float(args.proto_temperature))
            sup_logits = _proto_logits(z_sup, prototypes.detach(), float(args.proto_temperature))
            proxy_logits = _proto_logits(z_proxy, prototypes.detach(), float(args.proto_temperature))
            virtual_logits = _proto_logits(z_virtual, prototypes.detach(), float(args.proto_temperature))
            base_fit_logits = _proto_logits(z_fit0, prototypes.detach(), float(args.proto_temperature)).detach()
            base_hold_logits = _proto_logits(z_hold0, prototypes.detach(), float(args.proto_temperature)).detach()
            base_sup_logits = _proto_logits(z_sup0, prototypes.detach(), float(args.proto_temperature)).detach()

            fit_ce = F.cross_entropy(fit_logits, fit_y_t)
            hold_ce = F.cross_entropy(hold_logits, hold_y_t)
            sup_ce = F.cross_entropy(sup_logits, sup_y_t)
            fit_preserve = F.kl_div(F.log_softmax(fit_logits, dim=1), F.softmax(base_fit_logits, dim=1), reduction="batchmean")
            hold_preserve = F.kl_div(F.log_softmax(hold_logits, dim=1), F.softmax(base_hold_logits, dim=1), reduction="batchmean")
            sup_preserve = F.kl_div(F.log_softmax(sup_logits, dim=1), F.softmax(base_sup_logits, dim=1), reduction="batchmean")
            fit_floor = old_floor_margin_loss(fit_logits, fit_y_t, float(args.old_floor_margin))
            hold_floor = old_floor_margin_loss(hold_logits, hold_y_t, float(args.source_holdout_margin))
            if bool(sup_old_mask.any().item()):
                old_sup_floor = old_floor_margin_loss(sup_logits[sup_old_mask], sup_y_t[sup_old_mask], float(args.old_floor_margin))
                old_sup_residual = ((z_sup[sup_old_mask] - z_sup0[sup_old_mask]) ** 2).mean()
            else:
                old_sup_floor = sup_logits.sum() * 0.0
                old_sup_residual = sup_logits.sum() * 0.0
            if bool(sup_seen_mask.any().item()):
                seen_floor = old_floor_margin_loss(sup_logits[sup_seen_mask], sup_y_t[sup_seen_mask], float(args.seen_floor_margin))
            else:
                seen_floor = sup_logits.sum() * 0.0
            proxy_open = F.softplus(proxy_logits.max(dim=1).values - float(args.proxy_open_margin)).mean()
            virtual_open = F.softplus(virtual_logits.max(dim=1).values - float(args.virtual_open_margin)).mean()
            hold_compact = (1.0 - F.cosine_similarity(z_hold, prototypes.index_select(0, hold_y_t), dim=1)).mean()
            sup_compact = (1.0 - F.cosine_similarity(z_sup, prototypes.index_select(0, sup_y_t), dim=1)).mean()
            residual = (
                ((z_fit - z_fit0) ** 2).mean()
                + ((z_hold - z_hold0) ** 2).mean()
                + ((z_sup - z_sup0) ** 2).mean()
                + ((z_proxy - z_proxy0) ** 2).mean()
            )
            loss = (
                float(args.source_fit_cls_weight) * fit_ce
                + float(args.source_holdout_cls_weight) * hold_ce
                + float(args.support_cls_weight) * sup_ce
                + float(args.source_preserve_weight) * fit_preserve
                + float(args.source_holdout_preserve_weight) * hold_preserve
                + float(args.support_preserve_weight) * sup_preserve
                + float(args.old_floor_weight) * (fit_floor + hold_floor + old_sup_floor)
                + float(args.seen_floor_weight) * seen_floor
                + float(args.proxy_open_weight) * proxy_open
                + float(args.virtual_open_weight) * virtual_open
                + float(args.source_holdout_compact_weight) * hold_compact
                + float(args.support_compact_weight) * sup_compact
                + float(args.residual_weight) * residual
                + float(args.old_core_residual_weight) * old_sup_residual
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), float(args.grad_clip))
            opt.step()
            epoch_loss += float(loss.detach().item())
            steps += 1
        losses.append(epoch_loss / max(steps, 1))

    adapter.eval()
    with torch.no_grad():
        z_fit = adapter(x.index_select(0, fit_idx))
        z_hold = adapter(x.index_select(0, holdout_idx))
        z_support = adapter(x.index_select(0, support_idx))
        z_proxy = adapter(x.index_select(0, proxy_idx_all))
        fit_y_t = y_by_global.index_select(0, fit_idx)
        hold_y_t = y_by_global.index_select(0, holdout_idx)
        support_y_t = y_by_global.index_select(0, support_idx)
        support_old_mask_all = _labels_in_mask(support_y_t, old_positions)
        support_seen_mask_all = _labels_in_mask(support_y_t, seen_positions)
        fit_before = _proto_logits(x.index_select(0, fit_idx), prototypes, float(args.proto_temperature))
        fit_after = _proto_logits(z_fit, prototypes, float(args.proto_temperature))
        hold_before = _proto_logits(x.index_select(0, holdout_idx), prototypes, float(args.proto_temperature))
        hold_after = _proto_logits(z_hold, prototypes, float(args.proto_temperature))
        support_before = _proto_logits(x.index_select(0, support_idx), prototypes, float(args.proto_temperature))
        support_after = _proto_logits(z_support, prototypes, float(args.proto_temperature))
        proxy_before = _proto_logits(x.index_select(0, proxy_idx_all), prototypes, float(args.proto_temperature)).max(dim=1).values
        proxy_after = _proto_logits(z_proxy, prototypes, float(args.proto_temperature)).max(dim=1).values

        def _acc(logits: torch.Tensor, labels: torch.Tensor) -> float:
            return float((logits.argmax(dim=1) == labels).float().mean().item())

        def _masked_acc(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> float:
            if not bool(mask.any().item()):
                return float("nan")
            return float((logits[mask].argmax(dim=1) == labels[mask]).float().mean().item())

        metrics = {
            "adapter_train_seconds": time.perf_counter() - start_time,
            "device": str(device),
            "final_loss": losses[-1] if losses else None,
            "loss_trace_tail": losses[-5:],
            "source_fit_proto_acc_before": _acc(fit_before, fit_y_t),
            "source_fit_proto_acc_after": _acc(fit_after, fit_y_t),
            "source_holdout_proto_acc_before": _acc(hold_before, hold_y_t),
            "source_holdout_proto_acc_after": _acc(hold_after, hold_y_t),
            "support_proto_acc_before": _acc(support_before, support_y_t),
            "support_proto_acc_after": _acc(support_after, support_y_t),
            "target_old_support_proto_acc_before": _masked_acc(support_before, support_y_t, support_old_mask_all),
            "target_old_support_proto_acc_after": _masked_acc(support_after, support_y_t, support_old_mask_all),
            "seen_new_support_proto_acc_before": _masked_acc(support_before, support_y_t, support_seen_mask_all),
            "seen_new_support_proto_acc_after": _masked_acc(support_after, support_y_t, support_seen_mask_all),
            "source_holdout_margin_before_mean": _true_margin_mean(hold_before, hold_y_t),
            "source_holdout_margin_after_mean": _true_margin_mean(hold_after, hold_y_t),
            "target_old_support_margin_before_mean": _true_margin_mean(support_before[support_old_mask_all], support_y_t[support_old_mask_all]),
            "target_old_support_margin_after_mean": _true_margin_mean(support_after[support_old_mask_all], support_y_t[support_old_mask_all]),
            "proxy_max_logit_before_mean": float(proxy_before.mean().item()),
            "proxy_max_logit_after_mean": float(proxy_after.mean().item()),
            "mean_source_holdout_residual_norm": float((z_hold - x.index_select(0, holdout_idx)).norm(dim=1).mean().item()),
            "mean_support_residual_norm": float((z_support - x.index_select(0, support_idx)).norm(dim=1).mean().item()),
            "mean_proxy_residual_norm": float((z_proxy - x.index_select(0, proxy_idx_all)).norm(dim=1).mean().item()),
            "prototype_labels": proto_labels,
            "training_counts": {
                "source_fit": len(plan.source_fit_indices),
                "source_holdout_calibration": len(plan.source_holdout_indices),
                "proxy_unknown": len(plan.proxy_unknown_indices),
                "target_support": len(plan.support_indices),
                "target_unknown_eval_only": len(plan.target_unknown_indices),
                "target_unknown_training_count": 0,
            },
            "state_bytes": _ospr_state_bytes(dim, int(args.adapter_rank), len(proto_labels), len(plan.support_indices)),
        }
    return adapter, metrics


def save_ospr_adapted_npz(src_path: Path, dst_path: Path, adapted_features: np.ndarray, metadata: Mapping[str, Any]) -> None:
    save_adapted_npz(src_path, dst_path, adapted_features, metadata)


def run_ospr_backends(args: argparse.Namespace, adapted_npz: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    requested = {part.strip().lower() for part in str(args.backend).split(",") if part.strip()}
    results: dict[str, Any] = {}
    if "both" in requested:
        requested.update({"enpc", "slev"})
    if "enpc" in requested:
        argv = ["--profiles", str(args.enpc_profiles), *_backend_common_argv(args, adapted_npz, output_dir, "ospr_ci_enpc")]
        enpc_args = parse_enpc_args(argv)
        result = run_enpc_ci(enpc_args)
        evidence = result.pop("_evidence_rows")
        Path(enpc_args.output_json).write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        _write_enpc_csv(enpc_args.output_summary_csv, result["summary_rows"])
        _write_enpc_csv(enpc_args.output_evidence_csv, evidence)
        results["enpc"] = result
    if "slev" in requested:
        argv = [
            "--profiles",
            str(args.slev_profiles),
            "--slev_energy_support_quantile",
            str(args.slev_energy_support_quantile),
            "--slev_logit_temperature",
            str(args.slev_logit_temperature),
            "--slev_energy_risk_temperature",
            str(args.slev_energy_risk_temperature),
            "--slev_energy_risk_margin",
            str(args.slev_energy_risk_margin),
            *_backend_common_argv(args, adapted_npz, output_dir, "ospr_ci_slev"),
        ]
        slev_args = parse_slev_args(argv)
        result = run_slev_ci(slev_args)
        evidence = result.pop("_evidence_rows")
        Path(slev_args.output_json).write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        _write_slev_csv(slev_args.output_summary_csv, result["summary_rows"])
        _write_slev_csv(slev_args.output_evidence_csv, evidence)
        results["slev"] = result
    return results


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--feature_npz", type=Path, required=True)
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--adapter_npz", type=Path, default=None)
    p.add_argument("--backend", default="both", help="Comma list: enpc,slev,both")
    p.add_argument("--collab_counts", default="all")
    p.add_argument("--collab_group_policy", default="same_max_budget", choices=["exact_k", "available_up_to_k", "same_max_budget"])
    p.add_argument("--partial_collab_min_receivers", type=int, default=1)
    p.add_argument("--k_shot", type=int, default=8)
    p.add_argument("--query_per_class", type=int, default=20)
    p.add_argument("--qknn_k", type=int, default=8)
    p.add_argument("--seed", type=int, default=4070505)
    p.add_argument("--source_holdout_per_class", type=int, default=32)
    p.add_argument("--support_selection_policy", default="stable_first", choices=["stable_first", "centroid", "scenario_diverse", "strict_event_query_preserve"])
    p.add_argument("--event_alignment_policy", default="receiver_domain_ranked", choices=["strict_event_key", "receiver_domain_ranked"])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--adapter_epochs", type=int, default=90)
    p.add_argument("--adapter_rank", type=int, default=12)
    p.add_argument("--adapter_alpha", type=float, default=0.12)
    p.add_argument("--dropout", type=float, default=0.02)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=9e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--proto_temperature", type=float, default=0.08)
    p.add_argument("--old_floor_margin", type=float, default=0.85)
    p.add_argument("--source_holdout_margin", type=float, default=0.90)
    p.add_argument("--seen_floor_margin", type=float, default=0.65)
    p.add_argument("--proxy_open_margin", type=float, default=0.14)
    p.add_argument("--virtual_open_margin", type=float, default=0.08)
    p.add_argument("--source_fit_cls_weight", type=float, default=1.0)
    p.add_argument("--source_holdout_cls_weight", type=float, default=1.0)
    p.add_argument("--support_cls_weight", type=float, default=1.0)
    p.add_argument("--source_preserve_weight", type=float, default=5.0)
    p.add_argument("--source_holdout_preserve_weight", type=float, default=7.0)
    p.add_argument("--support_preserve_weight", type=float, default=5.0)
    p.add_argument("--old_floor_weight", type=float, default=2.5)
    p.add_argument("--seen_floor_weight", type=float, default=1.6)
    p.add_argument("--proxy_open_weight", type=float, default=0.28)
    p.add_argument("--virtual_open_weight", type=float, default=0.28)
    p.add_argument("--source_holdout_compact_weight", type=float, default=0.45)
    p.add_argument("--support_compact_weight", type=float, default=0.35)
    p.add_argument("--residual_weight", type=float, default=0.16)
    p.add_argument("--old_core_residual_weight", type=float, default=0.75)
    p.add_argument("--virtual_count", type=int, default=256)
    p.add_argument("--virtual_mix_low", type=float, default=0.30)
    p.add_argument("--virtual_mix_high", type=float, default=0.68)
    p.add_argument("--virtual_noise_scale", type=float, default=0.01)
    p.add_argument("--enpc_profiles", default="all")
    p.add_argument("--slev_profiles", default="all")
    p.add_argument("--slev_energy_support_quantile", type=float, default=0.92)
    p.add_argument("--slev_logit_temperature", type=float, default=1.0)
    p.add_argument("--slev_energy_risk_temperature", type=float, default=0.75)
    p.add_argument("--slev_energy_risk_margin", type=float, default=0.0)
    p.add_argument("--max_event_bytes", type=float, default=1152.0)
    p.add_argument("--max_event_latency_ms", type=float, default=20.0)
    p.add_argument("--target_old_acc", type=float, default=0.99)
    p.add_argument("--target_min_old", type=float, default=0.95)
    p.add_argument("--target_seen_new_acc", type=float, default=0.97)
    p.add_argument("--target_min_seen", type=float, default=0.93)
    p.add_argument("--target_unknown_reject", type=float, default=0.99)
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = load_feature_npz(args.feature_npz)
    plan = build_ospr_training_plan(
        payload,
        k_shot=int(args.k_shot),
        query_per_class=int(args.query_per_class),
        seed=int(args.seed),
        support_selection_policy=str(args.support_selection_policy),
        source_holdout_per_class=int(args.source_holdout_per_class),
    )
    adapter, train_metrics = train_ospr_adapter(payload, plan, args)
    adapted = apply_adapter(payload, adapter, str(args.device))
    adapted_npz = args.adapter_npz or (args.output_dir / "ospr_ci_adapted_features.npz")
    adapter_metadata = {
        "algorithm": "OSPR-CI",
        "adapter": "source_holdout_open_space_low_rank_residual_adapter",
        "target_unknown_eval_only": True,
        "training_roles": ["source_fit", "source_holdout_calibration", PROXY_UNKNOWN_ROLE, "target_old_support", "target_new_support"],
        "forbidden_roles": [UNKNOWN_ROLE],
        "plan": asdict(plan),
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "train_metrics": train_metrics,
    }
    save_ospr_adapted_npz(args.feature_npz, adapted_npz, adapted, adapter_metadata)
    backend_results = run_ospr_backends(args, adapted_npz, args.output_dir)
    summary = {
        "algorithm": "OSPR-CI",
        "feature_npz": str(args.feature_npz),
        "adapted_feature_npz": str(adapted_npz),
        "target_unknown_eval_only": True,
        "training_counts": train_metrics["training_counts"],
        "state_bytes": train_metrics["state_bytes"],
        "target_receivers": plan.target_receivers,
        "collab_counts_requested": str(args.collab_counts),
        "backend_results": {
            key: {
                "best_joint_row": value.get("best_joint_row"),
                "summary_row_count": len(value.get("summary_rows", [])),
                "target_gates": value.get("target_gates", {}),
            }
            for key, value in backend_results.items()
        },
        "adapter_metadata": adapter_metadata,
    }
    (args.output_dir / "ospr_ci_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps({"adapted_feature_npz": str(adapted_npz), "target_receivers": plan.target_receivers}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
