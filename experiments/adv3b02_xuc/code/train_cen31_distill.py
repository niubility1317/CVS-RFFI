from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_dual_cvsincnet import build_dual_model
from post_stage_common import (
    add_common_data_args,
    build_standard_data,
    load_baseline_from_checkpoint,
    resolve_device,
)
from training_controls import parse_sat_scenarios
from training_test_eval import aggregate_named_stats, format_named_test_lines
from baseline_origin_sat_view import BaselineOriginSatViewAugment
from cvsrffi.checkpoint import save_checkpoint
from cvsrffi.eval import (
    accuracy_from_logits,
    apply_sat_channel_for_scenario,
    evaluate_loader,
    evaluate_named_loaders,
    evaluate_sat_scenarios,
    format_sat_test_lines,
    make_loader,
    metric_or_neg_inf,
)
from cvsrffi.losses import SmoothGroupDROState, groupdro_or_hard_domain_ce_loss, one_way_kl_from_teacher
from cvsrffi.tensors import (
    build_domain_label_map,
    extract_domain_from_extra,
    remap_domain_tensor,
    safe_cosine_similarity,
    safe_iq_tensor,
    set_seed,
    unpack_batch,
)


def count_unique_params(module: nn.Module) -> int:
    seen: set[int] = set()
    total = 0
    for p in module.parameters():
        ident = id(p)
        if ident in seen:
            continue
        seen.add(ident)
        total += int(p.numel())
    return total


def build_student(args, data_ctx: Dict[str, Any], device: torch.device) -> nn.Module:
    model = build_dual_model(
        int(args.num_classes),
        int(data_ctx["num_domains"]),
        model_size=str(args.model_size),
        dataset=str(args.dataset),
        input_len=int(data_ctx["input_len"]),
        sample_rate_hz=float(args.sample_rate_hz),
        model_variant=str(args.model_variant),
        branch_ablation=str(args.branch_ablation),
        domain_branch_ablation=str(args.domain_branch_ablation),
        domain_enhancer=str(args.domain_enhancer),
        domain_enhancer_strength=float(args.domain_enhancer_strength),
        id_time_stability_mode=str(args.id_time_stability_mode),
        id_freq_stability_mode=str(args.id_freq_stability_mode),
        domain_time_stability_mode=str(args.domain_time_stability_mode),
        domain_freq_stability_mode=str(args.domain_freq_stability_mode),
        time_stability_channels=int(args.time_stability_channels),
        freq_stability_channels=int(args.freq_stability_channels),
        fast_infer_when_no_aux=True,
        arch_family=str(args.arch_family),
        mixstyle_on=False,
    )
    return model.to(device)


class FeatureProjector(nn.Module):
    def __init__(self, student_dim: int, teacher_dim: int):
        super().__init__()
        if int(student_dim) == int(teacher_dim):
            self.net = nn.Identity()
        else:
            self.net = nn.Sequential(
                nn.Linear(int(student_dim), int(teacher_dim), bias=False),
                nn.LayerNorm(int(teacher_dim)),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def teacher_reliability_mask(
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    conf_min: float,
    margin_min: float,
    require_correct: bool,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    probs = F.softmax(teacher_logits.float(), dim=1)
    top2 = torch.topk(probs, k=min(2, probs.size(1)), dim=1)
    conf = top2.values[:, 0]
    if probs.size(1) >= 2:
        margin = top2.values[:, 0] - top2.values[:, 1]
    else:
        margin = torch.ones_like(conf)
    pred = top2.indices[:, 0]
    mask = (conf >= float(conf_min)) & (margin >= float(margin_min))
    if require_correct:
        mask = mask & (pred == labels.view(-1).long())
    return mask, {
        "teacher_conf": float(conf.mean().item()),
        "teacher_margin": float(margin.mean().item()),
        "teacher_acc": float((pred == labels.view(-1).long()).float().mean().item() * 100.0),
        "kd_active_frac": float(mask.float().mean().item()),
    }


def masked_logit_kd(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    mask: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    if mask is None or not bool(mask.any()):
        return student_logits.new_tensor(0.0)
    return one_way_kl_from_teacher(student_logits[mask], teacher_logits[mask], temperature=float(temperature))


def feature_kd_loss(student_z: torch.Tensor, teacher_z: torch.Tensor, projector: nn.Module, mask: Optional[torch.Tensor]) -> torch.Tensor:
    if mask is not None and not bool(mask.any()):
        return student_z.new_tensor(0.0)
    if mask is not None:
        student_z = student_z[mask]
        teacher_z = teacher_z[mask]
    if int(student_z.numel()) == 0:
        return teacher_z.new_tensor(0.0)
    projected = projector(student_z.float())
    return (1.0 - safe_cosine_similarity(projected, teacher_z.float(), dim=1)).mean()


def relation_kd_loss(student_z: torch.Tensor, teacher_z: torch.Tensor, projector: nn.Module, mask: Optional[torch.Tensor]) -> torch.Tensor:
    if mask is not None and not bool(mask.any()):
        return student_z.new_tensor(0.0)
    if mask is not None:
        student_z = student_z[mask]
        teacher_z = teacher_z[mask]
    if int(student_z.size(0)) < 2:
        return teacher_z.new_tensor(0.0)
    s = F.normalize(projector(student_z.float()), dim=1)
    t = F.normalize(teacher_z.float(), dim=1)
    return F.smooth_l1_loss(s @ s.t(), t @ t.t())


def compute_sat_view_distill_losses(
    student_out: Dict[str, torch.Tensor],
    teacher_out: Dict[str, torch.Tensor],
    labels: torch.Tensor,
    projector: nn.Module,
    mask: Optional[torch.Tensor],
    ce: nn.Module,
    args,
    domain_labels: Optional[torch.Tensor] = None,
    groupdro_state: Optional[SmoothGroupDROState] = None,
) -> Dict[str, torch.Tensor]:
    """Distill clean teacher predictions onto a student satellite/augmented view."""
    loss_ce = ce(student_out["tx_logits"].float(), labels)
    loss_kd = masked_logit_kd(student_out["tx_logits"], teacher_out["tx_logits"], mask, float(args.kd_temperature))
    loss_feat = feature_kd_loss(student_out["z_id"], teacher_out["z_id"], projector, mask)
    loss_rel = relation_kd_loss(student_out["z_id"], teacher_out["z_id"], projector, mask)
    if float(getattr(args, "lambda_sat_view_group_ce", 0.0)) > 0.0:
        loss_group_ce, group_hard = groupdro_or_hard_domain_ce_loss(
            student_out["tx_logits"],
            labels,
            domain_labels,
            groupdro_state,
            mode=str(getattr(args, "group_ce_mode", "smooth_dro_capped")),
            label_smoothing=float(getattr(args, "label_smoothing", 0.0)),
            top_frac=float(getattr(args, "group_ce_top_frac", 0.35)),
            min_domains=int(getattr(args, "group_ce_min_domains", 2)),
            tau=float(getattr(args, "groupdro_tau", 0.5)),
            cap=float(getattr(args, "groupdro_cap", 0.65)),
        )
    else:
        loss_group_ce = student_out["tx_logits"].new_tensor(0.0)
        group_hard = float("nan")
    loss = (
        float(args.lambda_sat_view_ce) * loss_ce
        + float(args.lambda_sat_view_kd) * loss_kd
        + float(args.lambda_sat_view_feature_kd) * loss_feat
        + float(args.lambda_sat_view_relation_kd) * loss_rel
        + float(getattr(args, "lambda_sat_view_group_ce", 0.0)) * loss_group_ce
    )
    return {
        "loss": loss,
        "ce": loss_ce,
        "kd": loss_kd,
        "feat": loss_feat,
        "rel": loss_rel,
        "group_ce": loss_group_ce,
        "group_hard": group_hard,
    }


def sat_view_loss_scale(epoch: int, args) -> float:
    start = int(getattr(args, "sat_view_loss_start_epoch", 1))
    ramp = int(getattr(args, "sat_view_loss_ramp_epochs", 0))
    if int(epoch) < start:
        return 0.0
    if ramp <= 1:
        return 1.0
    return max(0.0, min(1.0, float(int(epoch) - start + 1) / float(ramp)))


def build_sat_view_augment(args) -> Optional[BaselineOriginSatViewAugment]:
    if not bool(args.use_sat_view_kd):
        return None
    scenarios = parse_sat_scenarios(args.sat_train_scenarios)
    return BaselineOriginSatViewAugment(
        scenarios=scenarios,
        schedule=str(args.sat_view_schedule or ""),
        p=float(args.sat_view_prob),
        seed=int(args.sat_seed),
        apply_fn=apply_sat_channel_for_scenario,
    )


def worst_receiver_metrics(named: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    seen_rx = []
    unseen_rx = []
    for name, stats in named.items():
        tx_acc = metric_or_neg_inf(stats, "tx_acc")
        if str(name).startswith("test_rx_"):
            seen_rx.append((float(tx_acc), str(name)))
        elif str(name).startswith("test_unseen_day_rx_"):
            unseen_rx.append((float(tx_acc), str(name)))

    def _worst(items: list[tuple[float, str]]) -> tuple[float, str]:
        if not items:
            return float("nan"), ""
        return min(items, key=lambda item: item[0])

    worst_seen, worst_seen_name = _worst(seen_rx)
    worst_unseen, worst_unseen_name = _worst(unseen_rx)
    rx8_seen = metric_or_neg_inf(named.get("test_rx_8", {}), "tx_acc")
    rx8_unseen = metric_or_neg_inf(named.get("test_unseen_day_rx_8", {}), "tx_acc")
    floor_values = [v for v in (worst_seen, worst_unseen, rx8_seen, rx8_unseen) if math.isfinite(float(v))]
    return {
        "worst_rx": float(worst_seen),
        "worst_rx_name": worst_seen_name,
        "worst_unseen_rx": float(worst_unseen),
        "worst_unseen_rx_name": worst_unseen_name,
        "rx8_seen": float(rx8_seen),
        "rx8_unseen": float(rx8_unseen),
        "receiver_floor": float(min(floor_values)) if floor_values else float("nan"),
    }


def sat_scenario_primary(stats: Dict[str, Any]) -> float:
    aggregate = stats.get("aggregate", {})
    overall = float(aggregate.get("tx_acc", float("nan")))
    strict = float(stats.get("strict_udu", float("nan")))
    if not math.isfinite(overall) and not math.isfinite(strict):
        return float("nan")
    if not math.isfinite(overall):
        overall = strict
    if not math.isfinite(strict):
        strict = overall
    return 0.30 * overall + 0.70 * strict


def summarize_sat_selection(sat_stats: Optional[Dict[str, Dict[str, Any]]]) -> Dict[str, float]:
    if not sat_stats:
        return {"sat_primary_mean": float("nan"), "sat_floor": float("nan")}
    primaries = []
    floors = []
    for stats in sat_stats.values():
        primary = sat_scenario_primary(stats)
        strict = float(stats.get("strict_udu", float("nan")))
        if math.isfinite(primary):
            primaries.append(primary)
        if math.isfinite(strict):
            floors.append(strict)
    return {
        "sat_primary_mean": float(statistics.fmean(primaries)) if primaries else float("nan"),
        "sat_floor": float(min(floors)) if floors else float("nan"),
    }


def compute_balanced_selection_score(
    clean_primary: float,
    receiver_floor: float,
    sat_primary_mean: float,
    sat_floor: float,
    args,
) -> float:
    clean = float(clean_primary)
    receiver = float(receiver_floor)
    sat_mean = float(sat_primary_mean)
    sat_min = float(sat_floor)
    if not math.isfinite(receiver):
        receiver = clean
    if not math.isfinite(sat_mean):
        sat_mean = clean
    if not math.isfinite(sat_min):
        sat_min = sat_mean
    return (
        float(getattr(args, "best_clean_weight", 0.55)) * clean
        + float(getattr(args, "best_receiver_floor_weight", 0.10)) * receiver
        + float(getattr(args, "best_sat_mean_weight", 0.25)) * sat_mean
        + float(getattr(args, "best_sat_floor_weight", 0.10)) * sat_min
    )


def clean_guard_allows_balanced_update(clean_primary: float, best_primary_seen: float, guard_drop: float) -> bool:
    if not math.isfinite(float(best_primary_seen)):
        return True
    return float(clean_primary) >= float(best_primary_seen) - max(0.0, float(guard_drop))


def should_run_sat_selection_eval(epoch: int, args) -> bool:
    if not bool(getattr(args, "eval_sat_channel", False)):
        return False
    if str(getattr(args, "best_select_metric", "primary")) != "clean_sat_joint":
        return False
    interval = int(getattr(args, "sat_select_eval_interval", 0))
    if interval <= 0:
        return True
    return int(epoch) == int(args.epochs) or int(epoch) % interval == 0


def sat_selection_max_batches(args) -> int:
    value = int(getattr(args, "sat_select_max_batches", -1))
    if value >= 0:
        return value
    return int(getattr(args, "sat_eval_max_batches", 0))


def format_sat_selection_fields(sat_summary: Dict[str, float]) -> str:
    return (
        f"sat_mean={sat_summary.get('sat_primary_mean', float('nan')):.2f} "
        f"sat_floor={sat_summary.get('sat_floor', float('nan')):.2f}"
    )


def print_sat_eval(prefix: str, sat_stats: Dict[str, Dict[str, Any]]) -> None:
    for scenario, stats in sat_stats.items():
        agg = stats.get("aggregate", {})
        strict = float(stats.get("strict_udu", float("nan")))
        selected = ",".join(stats.get("selected_names", []))
        primary = sat_scenario_primary(stats)
        print(
            f"[{prefix}] scenario={scenario} selected={selected} "
            f"overall_tx={agg.get('tx_acc', float('nan')):.2f}% "
            f"strict_udu={strict:.2f}% primary={primary:.2f} "
            f"({int(agg.get('tx_correct', 0))}/{int(agg.get('tx_total', 0))})",
            flush=True,
        )


def profile_latency(model: nn.Module, device: torch.device, input_len: int, num_classes: int, num_domains: int) -> Dict[str, Any]:
    rows = []
    model.eval()
    for batch_size in (1, 256):
        x = torch.randn(batch_size, 2, int(input_len), device=device)
        y = torch.randint(0, int(num_classes), (batch_size,), device=device)
        d = torch.randint(0, max(1, int(num_domains)), (batch_size,), device=device)
        with torch.no_grad():
            for _ in range(5):
                model(x, y_tx=y, domain_labels=d, return_aux=False)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            samples = []
            for _ in range(20):
                t0 = time.perf_counter()
                model(x, y_tx=y, domain_labels=d, return_aux=False)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                samples.append((time.perf_counter() - t0) * 1000.0)
        rows.append(
            {
                "batch_size": int(batch_size),
                "deploy_latency_ms_median": float(statistics.median(samples)),
                "deploy_latency_ms_per_sample": float(statistics.median(samples) / float(batch_size)),
            }
        )
    return {
        "device": str(device),
        "params_train": count_unique_params(model),
        "params_deploy": count_unique_params(getattr(model, "id_backbone", model)),
        "rows": rows,
    }


def evaluate_summary(model: nn.Module, data_ctx: Dict[str, Any], args, device: torch.device) -> Dict[str, Any]:
    val_stats = evaluate_loader(
        model,
        data_ctx["val_loader"],
        device,
        domain_label_map=data_ctx["domain_label_map"],
        max_batches=int(args.eval_max_batches),
    )
    named = evaluate_named_loaders(
        model,
        data_ctx["named_test_loaders"],
        device,
        domain_label_map=data_ctx["domain_label_map"],
        max_batches=int(args.eval_max_batches),
    )
    main_keys = [k for k in ["test_unseen_day_seen_rx", "test_seen_day_unseen_rx", "test_unseen_day_unseen_rx"] if k in named]
    if not main_keys:
        main_keys = list(named.keys())
    aggregate = aggregate_named_stats(named, main_keys)
    strict_udu = metric_or_neg_inf(named.get("test_unseen_day_unseen_rx", {}), "tx_acc")
    primary = 0.30 * float(aggregate.get("tx_acc", float("nan"))) + 0.70 * float(strict_udu)
    receiver = worst_receiver_metrics(named)
    return {
        "val": val_stats,
        "named": named,
        "aggregate": aggregate,
        "strict_udu": strict_udu,
        "primary_score": primary,
        **receiver,
    }


def print_eval(prefix: str, summary: Dict[str, Any], named_meta: Dict[str, Dict[str, Any]]) -> None:
    print(
        f"[{prefix}] val_tx={summary['val'].get('tx_acc', float('nan')):.2f}% "
        f"overall={summary['aggregate'].get('tx_acc', float('nan')):.2f}% "
        f"strict_udu={summary['strict_udu']:.2f}% primary={summary['primary_score']:.2f} "
        f"rx8_seen={summary.get('rx8_seen', float('nan')):.2f}% "
        f"rx8_unseen={summary.get('rx8_unseen', float('nan')):.2f}% "
        f"receiver_floor={summary.get('receiver_floor', float('nan')):.2f}%",
        flush=True,
    )
    for line in format_named_test_lines(summary["named"], named_meta):
        print(f"[{prefix}] {line}", flush=True)


_TEACHER_CLI_ARG_KEYS = {
    "dataset",
    "dataset_dir",
    "run_name",
    "wisig_pkl",
    "wisig_equalized",
    "wisig_domain",
    "wisig_out_len",
    "wisig_train_ratio",
    "wisig_val_ratio",
    "wisig_guard_gap",
    "wisig_train_days",
    "wisig_test_days",
    "wisig_train_rxs",
    "wisig_test_rxs",
    "wisig_max_day123_per_combo",
    "wisig_max_train_per_combo",
    "wisig_max_val_per_combo",
    "wisig_max_test_per_combo",
    "num_classes",
    "batch_size",
    "eval_batch_size",
    "num_workers",
    "prefetch_factor",
    "device",
    "seed",
    "eval_max_batches",
    "sample_rate_hz",
}


def build_teacher_cli_args(args) -> SimpleNamespace:
    return SimpleNamespace(**{key: getattr(args, key) for key in _TEACHER_CLI_ARG_KEYS if hasattr(args, key)})


def train(args) -> None:
    set_seed(int(args.seed))
    device = resolve_device(args.device)
    if float(args.sample_rate_hz) <= 0.0:
        args.sample_rate_hz = 25e6 if str(args.dataset).lower() == "wisig" else 5e6
    data_ctx = build_standard_data(args, device)
    teacher, teacher_ckpt, teacher_model_args = load_baseline_from_checkpoint(
        args.teacher_ckpt,
        build_teacher_cli_args(args),
        data_ctx,
        device,
        freeze=True,
    )
    student = build_student(args, data_ctx, device)
    projector = FeatureProjector(int(getattr(student, "emb_dim", 1)), int(getattr(teacher, "emb_dim", 1))).to(device)
    optimizer = torch.optim.AdamW(
        list(student.parameters()) + list(projector.parameters()),
        lr=float(args.lr),
        weight_decay=float(args.wd),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, int(args.epochs)), eta_min=float(args.lr_min))
    scaler = torch.cuda.amp.GradScaler(enabled=bool(args.amp and device.type == "cuda"))
    ce = nn.CrossEntropyLoss(label_smoothing=float(args.label_smoothing))
    sat_view_aug = build_sat_view_augment(args)
    groupdro_state = SmoothGroupDROState(momentum=float(args.groupdro_momentum))
    sat_groupdro_state = SmoothGroupDROState(momentum=float(args.groupdro_momentum))
    run_dir = Path(args.output_dir or Path(args.latest_save_path).parent or ".")
    run_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[DISTILL-CONFIG] teacher={args.teacher_ckpt} teacher_variant={getattr(teacher_model_args, 'model_variant', 'unknown')} "
        f"student_arch={args.arch_family} student_variant={args.model_variant} branch={args.branch_ablation} "
        f"lambda_kd={args.lambda_kd} lambda_feature_kd={args.lambda_feature_kd} lambda_relation_kd={args.lambda_relation_kd} "
        f"lambda_group_ce={args.lambda_group_ce} group_ce_mode={args.group_ce_mode}",
        flush=True,
    )
    if sat_view_aug is not None:
        print(
            f"[DISTILL-SAT-VIEW] enabled=1 scenarios={args.sat_train_scenarios} schedule={args.sat_view_schedule or '<none>'} "
            f"p={float(args.sat_view_prob):.3f} lambda_ce={float(args.lambda_sat_view_ce):.3f} "
            f"lambda_kd={float(args.lambda_sat_view_kd):.3f} "
            f"lambda_feature_kd={float(args.lambda_sat_view_feature_kd):.3f} "
            f"lambda_relation_kd={float(args.lambda_sat_view_relation_kd):.3f} "
            f"lambda_group_ce={float(args.lambda_sat_view_group_ce):.3f} "
            f"loss_start={int(args.sat_view_loss_start_epoch)} ramp_epochs={int(args.sat_view_loss_ramp_epochs)}",
            flush=True,
        )
    print(
        f"[DISTILL-SELECTION] metric={args.best_select_metric} best_balanced={args.best_balanced_save_path} "
        f"weights clean={float(args.best_clean_weight):.3f} receiver={float(args.best_receiver_floor_weight):.3f} "
        f"sat_mean={float(args.best_sat_mean_weight):.3f} sat_floor={float(args.best_sat_floor_weight):.3f} "
        f"clean_guard_drop={float(args.best_clean_guard_drop):.3f} sat_select_interval={int(args.sat_select_eval_interval)} "
        f"sat_select_max_batches={int(args.sat_select_max_batches)}",
        flush=True,
    )
    print(
        f"[DISTILL-MODEL] teacher_params={count_unique_params(teacher):,} "
        f"student_train_params={count_unique_params(student):,} "
        f"student_deploy_params={count_unique_params(getattr(student, 'id_backbone', student)):,}",
        flush=True,
    )

    best_primary = float("-inf")
    best_epoch = 0
    best_balanced = float("-inf")
    best_balanced_epoch = 0
    for epoch in range(1, int(args.epochs) + 1):
        student.train()
        projector.train()
        losses = []
        ce_vals = []
        kd_vals = []
        feat_vals = []
        rel_vals = []
        group_ce_vals = []
        group_hard_vals = []
        acc_vals = []
        active_vals = []
        teacher_acc_vals = []
        sat_ce_vals = []
        sat_kd_vals = []
        sat_feat_vals = []
        sat_rel_vals = []
        sat_group_ce_vals = []
        sat_scale_vals = []
        sat_active_vals = []
        for batch_idx, batch in enumerate(data_ctx["train_loader"], start=1):
            x, y, extra = unpack_batch(batch)
            x = safe_iq_tensor(x.to(device, non_blocking=True))
            y = y.to(device, non_blocking=True).long()
            d_raw = extract_domain_from_extra(extra, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=bool(args.amp and device.type == "cuda")):
                teacher_out = teacher(x, y_tx=None, domain_labels=d_raw, return_aux=True)
            sat_view = sat_view_aug.transform(x, args=args, epoch=epoch, batch_idx=batch_idx) if sat_view_aug is not None else None
            with torch.cuda.amp.autocast(enabled=bool(args.amp and device.type == "cuda")):
                student_out = student(x, y_tx=y, domain_labels=d_raw, return_aux=True)
                mask, kd_stats = teacher_reliability_mask(
                    teacher_out["tx_logits"],
                    y,
                    conf_min=float(args.kd_conf_min),
                    margin_min=float(args.kd_margin_min),
                    require_correct=bool(args.kd_require_correct),
                )
                loss_ce = ce(student_out["tx_logits"].float(), y)
                loss_kd = masked_logit_kd(student_out["tx_logits"], teacher_out["tx_logits"], mask, float(args.kd_temperature))
                loss_feat = feature_kd_loss(student_out["z_id"], teacher_out["z_id"], projector, mask)
                loss_rel = relation_kd_loss(student_out["z_id"], teacher_out["z_id"], projector, mask)
                if float(args.lambda_group_ce) > 0.0:
                    loss_group_ce, group_hard = groupdro_or_hard_domain_ce_loss(
                        student_out["tx_logits"],
                        y,
                        d_raw,
                        groupdro_state,
                        mode=str(args.group_ce_mode),
                        label_smoothing=float(args.label_smoothing),
                        top_frac=float(args.group_ce_top_frac),
                        min_domains=int(args.group_ce_min_domains),
                        tau=float(args.groupdro_tau),
                        cap=float(args.groupdro_cap),
                    )
                else:
                    loss_group_ce = student_out["tx_logits"].new_tensor(0.0)
                    group_hard = float("nan")
                loss = (
                    loss_ce
                    + float(args.lambda_kd) * loss_kd
                    + float(args.lambda_feature_kd) * loss_feat
                    + float(args.lambda_relation_kd) * loss_rel
                    + float(args.lambda_group_ce) * loss_group_ce
                )
                if sat_view_aug is not None:
                    x_sat = safe_iq_tensor(sat_view.x)
                    student_sat_out = student(x_sat, y_tx=y, domain_labels=d_raw, return_aux=True)
                    sat_losses = compute_sat_view_distill_losses(
                        student_sat_out,
                        teacher_out,
                        y,
                        projector,
                        mask,
                        ce,
                        args,
                        domain_labels=d_raw,
                        groupdro_state=sat_groupdro_state,
                    )
                    sat_scale = sat_view_loss_scale(epoch, args)
                    loss = loss + float(sat_scale) * sat_losses["loss"]
                else:
                    sat_losses = None
                    sat_scale = 0.0
            scaler.scale(loss).backward()
            if float(args.clip_grad) > 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(list(student.parameters()) + list(projector.parameters()), float(args.clip_grad))
            scaler.step(optimizer)
            scaler.update()

            losses.append(float(loss.detach().item()))
            ce_vals.append(float(loss_ce.detach().item()))
            kd_vals.append(float(loss_kd.detach().item()))
            feat_vals.append(float(loss_feat.detach().item()))
            rel_vals.append(float(loss_rel.detach().item()))
            group_ce_vals.append(float(loss_group_ce.detach().item()))
            if math.isfinite(float(group_hard)):
                group_hard_vals.append(float(group_hard))
            acc_vals.append(float(accuracy_from_logits(student_out["tx_logits"].detach(), y)))
            active_vals.append(float(kd_stats["kd_active_frac"]))
            teacher_acc_vals.append(float(kd_stats["teacher_acc"]))
            if sat_losses is not None:
                sat_ce_vals.append(float(sat_losses["ce"].detach().item()))
                sat_kd_vals.append(float(sat_losses["kd"].detach().item()))
                sat_feat_vals.append(float(sat_losses["feat"].detach().item()))
                sat_rel_vals.append(float(sat_losses["rel"].detach().item()))
                sat_group_ce_vals.append(float(sat_losses["group_ce"].detach().item()))
                sat_scale_vals.append(float(sat_scale))
                sat_active_vals.append(1.0 if bool(getattr(sat_view, "applied", False)) else 0.0)
        scheduler.step()

        msg = (
            f"[DISTILL-E{epoch:03d}] loss={statistics.fmean(losses):.4f} ce={statistics.fmean(ce_vals):.4f} "
            f"kd={statistics.fmean(kd_vals):.4f} feat={statistics.fmean(feat_vals):.4f} rel={statistics.fmean(rel_vals):.4f} "
            f"group_ce={statistics.fmean(group_ce_vals):.4f} "
            f"train_acc={statistics.fmean(acc_vals):.2f}% teacher_acc={statistics.fmean(teacher_acc_vals):.2f}% "
            f"kd_active={statistics.fmean(active_vals):.3f}"
        )
        if group_hard_vals:
            msg += f" group_hard={statistics.fmean(group_hard_vals):.4f}"
        if sat_ce_vals:
            msg += (
                f" sat_ce={statistics.fmean(sat_ce_vals):.4f}"
                f" sat_kd={statistics.fmean(sat_kd_vals):.4f}"
                f" sat_feat={statistics.fmean(sat_feat_vals):.4f}"
                f" sat_rel={statistics.fmean(sat_rel_vals):.4f}"
                f" sat_group_ce={statistics.fmean(sat_group_ce_vals):.4f}"
                f" sat_scale={statistics.fmean(sat_scale_vals):.3f}"
                f" sat_active={statistics.fmean(sat_active_vals):.3f}"
            )
        print(msg, flush=True)

        should_eval = epoch == int(args.epochs) or (int(args.eval_interval) > 0 and epoch % int(args.eval_interval) == 0)
        if should_eval:
            summary = evaluate_summary(student, data_ctx, args, device)
            print_eval(f"DISTILL-EVAL-E{epoch:03d}", summary, data_ctx["named_test_meta"])
            clean_primary = float(summary["primary_score"])
            sat_stats = None
            sat_summary = summarize_sat_selection(None)
            if should_run_sat_selection_eval(epoch, args):
                sat_stats = evaluate_sat_scenarios(
                    student,
                    data_ctx["named_test_loaders"],
                    device,
                    domain_label_map=data_ctx["domain_label_map"],
                    scenario_names=parse_sat_scenarios(args.eval_sat_scenarios),
                    args=args,
                    max_batches=sat_selection_max_batches(args),
                )
                sat_summary = summarize_sat_selection(sat_stats)
                print_sat_eval(f"DISTILL-EVAL-SAT-E{epoch:03d}", sat_stats)
            balanced_score = compute_balanced_selection_score(
                clean_primary,
                float(summary.get("receiver_floor", float("nan"))),
                float(sat_summary.get("sat_primary_mean", float("nan"))),
                float(sat_summary.get("sat_floor", float("nan"))),
                args,
            )
            clean_allowed = clean_guard_allows_balanced_update(clean_primary, best_primary, float(args.best_clean_guard_drop))
            primary_improved = clean_primary > best_primary
            balanced_improved = (
                str(args.best_select_metric) == "clean_sat_joint"
                and sat_stats is not None
                and clean_allowed
                and balanced_score > best_balanced
            )
            print(
                f"[DISTILL-SELECT-E{epoch:03d}] clean_primary={clean_primary:.2f} "
                f"strict_udu={summary['strict_udu']:.2f} rx8_seen={summary.get('rx8_seen', float('nan')):.2f} "
                f"rx8_unseen={summary.get('rx8_unseen', float('nan')):.2f} "
                f"receiver_floor={summary.get('receiver_floor', float('nan')):.2f} "
                f"{format_sat_selection_fields(sat_summary)} composite={balanced_score:.2f} "
                f"clean_guard={int(clean_allowed)} best_primary={int(primary_improved)} "
                f"best_composite={int(balanced_improved)}",
                flush=True,
            )
            if primary_improved:
                best_primary = clean_primary
                best_epoch = int(epoch)
                save_checkpoint(
                    args.best_save_path,
                    model=student,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    epoch=epoch,
                    args=args,
                    split_info=data_ctx["split_info"],
                    stats=summary,
                )
                print(f"[DISTILL-CKPT] best_primary={best_primary:.2f} epoch={best_epoch} -> {args.best_save_path}", flush=True)
            if balanced_improved:
                best_balanced = float(balanced_score)
                best_balanced_epoch = int(epoch)
                balanced_stats = dict(summary)
                balanced_stats["sat_selection"] = sat_stats
                balanced_stats["sat_primary_mean"] = float(sat_summary.get("sat_primary_mean", float("nan")))
                balanced_stats["sat_floor"] = float(sat_summary.get("sat_floor", float("nan")))
                balanced_stats["balanced_score"] = float(balanced_score)
                save_checkpoint(
                    args.best_balanced_save_path,
                    model=student,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    epoch=epoch,
                    args=args,
                    split_info=data_ctx["split_info"],
                    stats=balanced_stats,
                )
                print(
                    f"[DISTILL-CKPT-BALANCED] best_composite={best_balanced:.2f} "
                    f"clean_primary={clean_primary:.2f} epoch={best_balanced_epoch} -> {args.best_balanced_save_path}",
                    flush=True,
                )
        save_checkpoint(
            args.latest_save_path,
            model=student,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            args=args,
            split_info=data_ctx["split_info"],
            stats={"epoch": int(epoch)},
        )

    if args.latency_profile_json:
        payload = profile_latency(student, device, int(data_ctx["input_len"]), int(args.num_classes), int(data_ctx["num_domains"]))
        out = Path(args.latency_profile_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[DISTILL-LATENCY] wrote {out}", flush=True)

    def evaluate_final_view(prefix: str, *, sat_prefix: str, compat_sat_lines: bool = False) -> None:
        final_summary = evaluate_summary(student, data_ctx, args, device)
        print_eval(prefix, final_summary, data_ctx["named_test_meta"])
        final_sat_stats = None
        final_sat_summary = summarize_sat_selection(None)
        if bool(args.eval_sat_channel):
            final_sat_stats = evaluate_sat_scenarios(
                student,
                data_ctx["named_test_loaders"],
                device,
                domain_label_map=data_ctx["domain_label_map"],
                scenario_names=parse_sat_scenarios(args.eval_sat_scenarios),
                args=args,
                max_batches=int(args.sat_eval_max_batches),
            )
            if compat_sat_lines:
                for line in format_sat_test_lines(final_sat_stats):
                    print(line, flush=True)
            print_sat_eval(sat_prefix, final_sat_stats)
            final_sat_summary = summarize_sat_selection(final_sat_stats)
        final_composite = compute_balanced_selection_score(
            float(final_summary["primary_score"]),
            float(final_summary.get("receiver_floor", float("nan"))),
            float(final_sat_summary.get("sat_primary_mean", float("nan"))),
            float(final_sat_summary.get("sat_floor", float("nan"))),
            args,
        )
        print(
            f"[{prefix}-SELECT] clean_primary={float(final_summary['primary_score']):.2f} "
            f"strict_udu={float(final_summary['strict_udu']):.2f} "
            f"rx8_unseen={float(final_summary.get('rx8_unseen', float('nan'))):.2f} "
            f"receiver_floor={float(final_summary.get('receiver_floor', float('nan'))):.2f} "
            f"{format_sat_selection_fields(final_sat_summary)} composite={final_composite:.2f}",
            flush=True,
        )

    evaluate_final_view("DISTILL-FINAL-LATEST", sat_prefix="DISTILL-FINAL-LATEST-SAT", compat_sat_lines=True)

    for label, ckpt_path in (
        ("DISTILL-FINAL-PRIMARY", args.best_save_path),
        ("DISTILL-FINAL-BALANCED", args.best_balanced_save_path),
    ):
        path = Path(ckpt_path)
        if not path.exists():
            print(f"[WARN] {label} checkpoint missing: {path}", flush=True)
            continue
        try:
            ckpt = torch.load(path, map_location=device)
            state = ckpt.get("model", ckpt)
            student.load_state_dict(state, strict=False)
            evaluate_final_view(label, sat_prefix=f"{label}-SAT", compat_sat_lines=False)
        except Exception as exc:
            print(f"[WARN] {label} evaluation failed: {exc}", flush=True)

    print(
        f"[DISTILL-DONE] best_primary={best_primary:.2f} best_epoch={best_epoch} "
        f"best_composite={best_balanced:.2f} best_composite_epoch={best_balanced_epoch}",
        flush=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Distill low-latency CVS-RFFI students from the CEN31 teacher.")
    add_common_data_args(parser)
    parser.add_argument("--teacher_ckpt", required=True)
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--latest_save_path", default="latest_student.pth")
    parser.add_argument("--best_save_path", default="best_student_primary.pth")
    parser.add_argument("--best_balanced_save_path", default="best_student_balanced.pth")
    parser.add_argument("--latency_profile_json", default="")
    parser.add_argument("--arch_family", default="cvsincnet", choices=["cvsincnet", "resnet18_1d", "cvcnn", "sinc_cvcnn"])
    parser.add_argument("--model_variant", default="lite_f", choices=["base", "lite_a", "lite_b", "lite_c", "lite_d", "lite_e", "lite_f", "lite_g", "lite_h"])
    parser.add_argument("--model_size", default="M")
    parser.add_argument("--branch_ablation", default="no_dac,no_stats")
    parser.add_argument("--domain_branch_ablation", default="no_stats")
    parser.add_argument("--domain_enhancer", default="rcn_stats")
    parser.add_argument("--domain_enhancer_strength", type=float, default=0.20)
    parser.add_argument("--id_time_stability_mode", default="off")
    parser.add_argument("--id_freq_stability_mode", default="off")
    parser.add_argument("--domain_time_stability_mode", default="off")
    parser.add_argument("--domain_freq_stability_mode", default="off")
    parser.add_argument("--time_stability_channels", type=int, default=4)
    parser.add_argument("--freq_stability_channels", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=90)
    parser.add_argument("--eval_interval", type=int, default=10)
    parser.add_argument("--lr", type=float, default=4e-4)
    parser.add_argument("--lr_min", type=float, default=1e-6)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--label_smoothing", type=float, default=0.01)
    parser.add_argument("--lambda_kd", type=float, default=0.70)
    parser.add_argument("--lambda_feature_kd", type=float, default=0.20)
    parser.add_argument("--lambda_relation_kd", type=float, default=0.05)
    parser.add_argument("--lambda_group_ce", type=float, default=0.0)
    parser.add_argument("--group_ce_mode", default="smooth_dro_capped")
    parser.add_argument("--group_ce_top_frac", type=float, default=0.35)
    parser.add_argument("--group_ce_min_domains", type=int, default=2)
    parser.add_argument("--groupdro_tau", type=float, default=0.5)
    parser.add_argument("--groupdro_cap", type=float, default=0.65)
    parser.add_argument("--groupdro_momentum", type=float, default=0.95)
    parser.add_argument("--kd_temperature", type=float, default=3.0)
    parser.add_argument("--kd_conf_min", type=float, default=0.60)
    parser.add_argument("--kd_margin_min", type=float, default=0.05)
    parser.add_argument("--kd_require_correct", action="store_true")
    parser.add_argument("--no_kd_require_correct", dest="kd_require_correct", action="store_false")
    parser.set_defaults(kd_require_correct=True)
    parser.add_argument("--clip_grad", type=float, default=1.0)
    parser.add_argument("--amp", dest="amp", action="store_true")
    parser.add_argument("--no_amp", dest="amp", action="store_false")
    parser.set_defaults(amp=True)
    parser.add_argument("--eval_sat_channel", action="store_true")
    parser.add_argument("--no_eval_sat_channel", dest="eval_sat_channel", action="store_false")
    parser.set_defaults(eval_sat_channel=False)
    parser.add_argument("--eval_sat_scenarios", default="clear_leo,low_elev_leo,rain_leo")
    parser.add_argument("--eval_sat_on", default="test_unseen_day_unseen_rx")
    parser.add_argument("--sat_eval_max_batches", type=int, default=0)
    parser.add_argument("--sat_seed", type=int, default=2027)
    parser.add_argument("--sat_fs_hz", type=float, default=25e6)
    parser.add_argument("--sat_fc_hz", type=float, default=2.462e9)
    parser.add_argument("--use_sat_view_kd", action="store_true")
    parser.add_argument("--no_use_sat_view_kd", dest="use_sat_view_kd", action="store_false")
    parser.set_defaults(use_sat_view_kd=False)
    parser.add_argument("--sat_train_scenarios", default="clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit")
    parser.add_argument("--sat_view_prob", type=float, default=1.0)
    parser.add_argument("--sat_view_schedule", default="")
    parser.add_argument("--sat_view_loss_start_epoch", type=int, default=1)
    parser.add_argument("--sat_view_loss_ramp_epochs", type=int, default=0)
    parser.add_argument("--lambda_sat_view_ce", type=float, default=0.0)
    parser.add_argument("--lambda_sat_view_kd", type=float, default=0.0)
    parser.add_argument("--lambda_sat_view_feature_kd", type=float, default=0.0)
    parser.add_argument("--lambda_sat_view_relation_kd", type=float, default=0.0)
    parser.add_argument("--lambda_sat_view_group_ce", type=float, default=0.0)
    parser.add_argument("--best_select_metric", choices=["primary", "clean_sat_joint"], default="primary")
    parser.add_argument("--best_clean_weight", type=float, default=0.55)
    parser.add_argument("--best_receiver_floor_weight", type=float, default=0.10)
    parser.add_argument("--best_sat_mean_weight", type=float, default=0.25)
    parser.add_argument("--best_sat_floor_weight", type=float, default=0.10)
    parser.add_argument("--best_clean_guard_drop", type=float, default=1.0)
    parser.add_argument("--sat_select_eval_interval", type=int, default=0)
    parser.add_argument("--sat_select_max_batches", type=int, default=-1)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.output_dir:
        out = Path(args.output_dir)
        if str(args.latest_save_path) == "latest_student.pth":
            args.latest_save_path = str(out / "latest_student.pth")
        if str(args.best_save_path) == "best_student_primary.pth":
            args.best_save_path = str(out / "best_student_primary.pth")
        if str(args.best_balanced_save_path) == "best_student_balanced.pth":
            args.best_balanced_save_path = str(out / "best_student_balanced.pth")
        if not args.latency_profile_json:
            args.latency_profile_json = str(out / "latency_profile.json")
    train(args)


if __name__ == "__main__":
    main()
