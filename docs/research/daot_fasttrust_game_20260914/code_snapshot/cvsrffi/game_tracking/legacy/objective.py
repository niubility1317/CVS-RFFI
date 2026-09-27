from __future__ import annotations
import math
from typing import Dict
import torch
import torch.nn.functional as F
from .losses import *
from .options import _stage_gate_scale

def labeled_terms(out_l, y_l, d_l, args, epoch, batch_idx, cur_w, proto_bank):
    domain_stats = {"valid": (d_l >= 0) if d_l is not None else None}
    domain_gates = {
        "dom": d_l is not None and "dom_logits" in out_l and cur_w["dom"] > 0.0,
        "adv": d_l is not None and "adv_dom_logits" in out_l and cur_w["adv"] > 0.0,
        "cons": d_l is not None and cur_w["cons"] > 0.0,
        "group_ce": d_l is not None and cur_w["group_ce"] > 0.0,
    }
    core_losses = compute_core_losses(
        out_l,
        y_l,
        d_l,
        domain_stats,
        domain_gates,
        lambda logits, target: F.cross_entropy(logits, target, label_smoothing=float(args.label_smoothing)),
        lambda logits, target: F.cross_entropy(logits, target),
        label_smoothing=float(args.label_smoothing),
        group_top_frac=float(args.group_ce_top_frac),
        group_min_domains=int(args.group_ce_min_domains),
        group_ce_mode=str(args.group_ce_mode),
    )
    loss_tx_l = core_losses["loss_cls"]
    loss_dom_l = core_losses["loss_dom"]
    loss_adv_l = core_losses["loss_adv"]
    loss_cons_l = core_losses["loss_cons"]
    loss_orth_l = core_losses["loss_orth"] if cur_w["orth"] > 0.0 else out_l["tx_logits"].sum() * 0.0
    loss_group_ce_l = core_losses["loss_group_ce"]
    if d_l is not None and cur_w["fishr"] > 0.0:
        loss_fishr_l = fishr_logit_gradient_variance_loss(
            out_l["tx_logits"],
            y_l,
            d_l,
            min_domains=int(args.fishr_min_domains),
        )
    else:
        loss_fishr_l = out_l["tx_logits"].sum() * 0.0
    z_id_l = out_l["z_id"]
    loss_proto_l = z_id_l.sum() * 0.0
    proto_info: Dict[str, float] = {
        "proto_pull_cos": float("nan"),
        "proto_push": 0.0,
        "proto_active_classes": 0.0,
    }
    if proto_bank is not None:
        loss_proto_l, proto_info = proto_bank.loss(z_id_l, y_l, d_l)
        if proto_bank.class_count is not None:
            active = proto_bank.class_count >= int(args.proto_min_count)
            proto_info["proto_active_classes"] = float(int(active.sum().detach().item()))
    loss_open_world_feat_l = z_id_l.sum() * 0.0
    ow_feat_stage_scale = _stage_gate_scale(
        epoch,
        start_epoch=int(getattr(args, "ow_feat_start_epoch", 1)),
        warmup_epochs=int(getattr(args, "ow_feat_warmup_epochs", 0)),
    )
    ow_feat_info: Dict[str, float] = {
        "compact": 0.0,
        "inter": 0.0,
        "sample_margin": 0.0,
        "domain_align": 0.0,
        "active_classes": 0.0,
        "pos_angle_deg": float("nan"),
        "min_inter_angle_deg": float("nan"),
        "pos_angle_p50_deg": float("nan"),
        "pos_angle_p95_deg": float("nan"),
        "pos_angle_p99_deg": float("nan"),
        "pos_angle_max_deg": float("nan"),
        "tail_loss": 0.0,
        "tail_cvar_deg": float("nan"),
        "tail_frac_gt_3sigma": 0.0,
        "tail_radius_3sigma_deg": float("nan"),
        "vacuum_loss": 0.0,
        "vacuum_violation_rate": 0.0,
        "vacuum_min_neg_angle_deg": float("nan"),
        "vacuum_margin_deg": float("nan"),
        "vacuum_boundary_deg": float("nan"),
    }
    if float(args.lambda_open_world_feat) > 0.0 and ow_feat_stage_scale > 0.0:
        if open_world_feature_space_loss is None:
            raise ImportError("cvsrffi.losses.open_world_feature_space_loss is required for --lambda_open_world_feat")
        loss_open_world_feat_l, ow_feat_info = open_world_feature_space_loss(
            z_id_l,
            y_l,
            d_l,
            radius_rad=math.radians(float(args.ow_feat_radius_deg)),
            inter_margin_rad=math.radians(float(args.ow_feat_inter_margin_deg)),
            sample_margin_rad=math.radians(float(args.ow_feat_sample_margin_deg)),
            domain_align_weight=float(args.ow_feat_domain_align_weight),
            min_classes=int(args.ow_feat_min_classes),
            min_samples_per_class=int(args.ow_feat_min_samples_per_class),
            tail_mode=str(args.ow_feat_tail_mode),
            tail_weight=float(args.ow_feat_tail_weight),
            cvar_alpha=float(args.ow_feat_cvar_alpha),
            vacuum_weight=float(args.ow_feat_vacuum_weight),
            vacuum_width_rad=math.radians(float(args.ow_feat_vacuum_width_deg)),
            vacuum_hard_k=int(args.ow_feat_vacuum_hard_k),
        )
    loss_zid_compact_l = z_id_l.sum() * 0.0
    zid_compact_info: Dict[str, float] = {
        "supcon": 0.0,
        "radius": 0.0,
        "tail_cvar": 0.0,
        "active_classes": 0.0,
        "pos_angle_p50_deg": float("nan"),
        "pos_angle_p95_deg": float("nan"),
        "pos_angle_p99_deg": float("nan"),
        "tail_cvar_deg": float("nan"),
    }
    zid_warm = _stage_gate_scale(
        epoch,
        start_epoch=int(getattr(args, "zid_compact_start_epoch", 1)),
        warmup_epochs=int(getattr(args, "zid_compact_warmup_epochs", 0)),
    )
    if float(args.lambda_zid_compact) > 0.0 and zid_warm > 0.0:
        if zid_compactness_loss is None:
            raise ImportError("cvsrffi.losses.zid_compactness_loss is required for --lambda_zid_compact")
        loss_zid_compact_l, zid_compact_info = zid_compactness_loss(
            z_id_l,
            y_l,
            d_l,
            radius_rad=math.radians(float(args.zid_compact_radius_deg)),
            cvar_alpha=float(args.zid_compact_cvar_alpha),
            supcon_weight=float(args.zid_compact_supcon_weight),
            radius_weight=float(args.zid_compact_radius_weight),
            cvar_weight=float(args.zid_compact_cvar_weight),
            domain_aware=bool(args.zid_compact_domain_aware),
        )
    loss_proxy_unknown_l = z_id_l.sum() * 0.0
    proxy_unknown_info: Dict[str, float] = {
        "active": 0.0,
        "known_count": 0.0,
        "proxy_unknown_count": 0.0,
        "virtual_count": 0.0,
        "core_count": 0.0,
        "tail_count": 0.0,
        "overflow_count": 0.0,
        "energy_known": float("nan"),
        "energy_proxy": float("nan"),
        "energy_virtual": float("nan"),
        "energy_margin": float("nan"),
        "accept_energy_threshold": float("nan"),
        "core_energy_threshold": float("nan"),
        "vaccept_surrogate": 0.0,
        "core_accept_loss": 0.0,
        "component_gate_unknown": 0.0,
        "component_gate_accept_prob": float("nan"),
        "component_gate_accept_prob_max": float("nan"),
        "tail_quarantine_loss": 0.0,
        "source_safe_loss": 0.0,
        "proxy_unknown_auc": float("nan"),
        "virtual_accept_rate": float("nan"),
        "virtual_accept_rate_core": float("nan"),
        "proxy_accept_rate": float("nan"),
        "hard_proxy_accept_rate": float("nan"),
        "shell_accept_rate": float("nan"),
        "bridge_accept_rate": float("nan"),
        "outward_accept_rate": float("nan"),
        "vacuum_loss": 0.0,
        "vacuum_violation_rate": 0.0,
        "vacuum_margin_deg": float("nan"),
        "vacuum_min_angle_deg": float("nan"),
    }
    proxy_stage_scale = _stage_gate_scale(
        epoch,
        start_epoch=int(getattr(args, "proxy_unknown_start_epoch", 40)),
        warmup_epochs=int(getattr(args, "proxy_unknown_warmup_epochs", 0)),
    )
    source_episode_stage_scale = _stage_gate_scale(
        epoch,
        start_epoch=int(getattr(args, "source_episode_start_epoch", 1)),
        warmup_epochs=int(getattr(args, "source_episode_warmup_epochs", 0)),
    )
    soft_mixup_start_epoch = int(getattr(args, "soft_unknown_mixup_start_epoch", -1))
    if soft_mixup_start_epoch <= 0:
        soft_mixup_start_epoch = int(getattr(args, "proxy_unknown_start_epoch", 40))
    soft_mixup_warmup_epochs = int(getattr(args, "soft_unknown_mixup_warmup_epochs", -1))
    if soft_mixup_warmup_epochs < 0:
        soft_mixup_warmup_epochs = int(getattr(args, "proxy_unknown_warmup_epochs", 0))
    loss_soft_unknown_mixup_l = z_id_l.sum() * 0.0
    soft_unknown_mixup_info: Dict[str, float] = {
        "soft_unknown_mixup_count": 0.0,
        "soft_unknown_mixup_order": float(max(2, int(getattr(args, "soft_unknown_mixup_order", 3)))),
        "soft_unknown_mixup_ce": 0.0,
        "soft_unknown_mixup_energy": 0.0,
        "soft_unknown_mixup_vacuum": 0.0,
        "soft_unknown_mixup_virtual_accept_rate": 0.0,
        "soft_unknown_mixup_vacuum_violation": 0.0,
    }
    soft_unknown_mixup_batch = None
    soft_unknown_mixup_stage_scale = _stage_gate_scale(
        epoch,
        start_epoch=soft_mixup_start_epoch,
        warmup_epochs=soft_mixup_warmup_epochs,
    )
    soft_mixup_needed = (
        (float(getattr(args, "lambda_soft_unknown_mixup", 0.0)) > 0.0 and soft_unknown_mixup_stage_scale > 0.0)
        or (
            float(args.lambda_source_episode) > 0.0
            and source_episode_stage_scale > 0.0
            and float(getattr(args, "source_episode_mixup_weight", 0.0)) > 0.0
        )
    )
    if soft_mixup_needed:
        if make_soft_unknown_mixup is None:
            raise ImportError("cvsrffi.losses.make_soft_unknown_mixup is required for soft unknown mixup")
        soft_unknown_mixup_batch = make_soft_unknown_mixup(
            z_id_l,
            y_l,
            mixup_count=int(args.soft_unknown_mixup_count),
            mixup_order=int(args.soft_unknown_mixup_order),
            alpha=float(args.soft_unknown_mixup_alpha),
        )
    proxy_active = float(args.lambda_proxy_unknown) > 0.0 and proxy_stage_scale > 0.0
    if proxy_active:
        if proxy_unknown_energy_loss is None:
            raise ImportError("cvsrffi.losses.proxy_unknown_energy_loss is required for --lambda_proxy_unknown")
        valid_y = torch.unique(y_l[y_l >= 0])
        if valid_y.numel() > 0:
            holdout_idx = (int(epoch) + int(batch_idx)) % int(valid_y.numel())
            holdout_label = int(valid_y[holdout_idx].detach().item())
        else:
            holdout_label = None
        loss_proxy_unknown_l, proxy_unknown_info = proxy_unknown_energy_loss(
            z_id_l,
            y_l,
            holdout_label=holdout_label,
            virtual_count=int(args.proxy_unknown_virtual_count),
            virtual_mode=str(args.proxy_unknown_virtual_mode),
            energy_margin=float(args.proxy_unknown_energy_margin),
            energy_temperature=float(args.proxy_unknown_energy_temperature),
            placeholder_weight=float(args.proxy_unknown_placeholder_weight),
            virtual_detach=bool(args.proxy_unknown_virtual_detach),
            vacuum_weight=float(args.proxy_unknown_vacuum_weight),
            vacuum_width_rad=math.radians(float(args.proxy_unknown_vacuum_width_deg)),
            vacuum_hard_k=int(args.proxy_unknown_vacuum_hard_k),
            vacuum_radius_rad=math.radians(float(args.proxy_unknown_vacuum_radius_deg)),
            core_quantile=float(args.proxy_unknown_core_quantile),
            accept_quantile=float(args.proxy_unknown_accept_quantile),
            tail_quantile=float(args.proxy_unknown_tail_quantile),
            overflow_quantile=float(args.proxy_unknown_overflow_quantile),
            vaccept_weight=float(args.proxy_unknown_vaccept_weight),
            core_accept_weight=float(args.proxy_unknown_core_accept_weight),
            component_gate_weight=float(args.proxy_unknown_component_gate_weight),
            tail_quarantine_weight=float(args.proxy_unknown_tail_quarantine_weight),
            source_safe_weight=float(args.proxy_unknown_source_safe_weight),
            vaccept_cvar_alpha=float(args.proxy_unknown_vaccept_cvar_alpha),
            unknown_margin=float(args.proxy_unknown_unknown_margin),
            known_margin=float(args.proxy_unknown_known_margin),
            energy_softplus_temperature=float(args.proxy_unknown_energy_softplus_temperature),
            component_temperature_rad=math.radians(float(args.proxy_unknown_component_temperature_deg)),
            component_margin_rad=math.radians(float(args.proxy_unknown_component_margin_deg)),
            component_margin_temperature_rad=math.radians(float(args.proxy_unknown_component_margin_temperature_deg)),
            shell_width_rad=math.radians(float(args.proxy_unknown_shell_width_deg)),
        )
    if float(getattr(args, "lambda_soft_unknown_mixup", 0.0)) > 0.0 and soft_unknown_mixup_stage_scale > 0.0:
        if soft_unknown_mixup_loss is None:
            raise ImportError("cvsrffi.losses.soft_unknown_mixup_loss is required for --lambda_soft_unknown_mixup")
        loss_soft_unknown_mixup_l, soft_unknown_mixup_info = soft_unknown_mixup_loss(
            z_id_l,
            y_l,
            logits=out_l["tx_logits"],
            mixup=soft_unknown_mixup_batch,
            mixup_count=int(args.soft_unknown_mixup_count),
            mixup_order=int(args.soft_unknown_mixup_order),
            alpha=float(args.soft_unknown_mixup_alpha),
            energy_margin=float(args.soft_unknown_mixup_energy_margin),
            ce_weight=float(args.soft_unknown_mixup_ce_weight),
            energy_weight=float(args.soft_unknown_mixup_energy_weight),
            vacuum_weight=float(args.soft_unknown_mixup_vacuum_weight),
            vacuum_width_rad=math.radians(float(args.soft_unknown_mixup_vacuum_width_deg)),
            vacuum_hard_k=int(args.soft_unknown_mixup_vacuum_hard_k),
            detach_mixup=bool(args.soft_unknown_mixup_detach),
        )
    loss_source_episode_l = z_id_l.sum() * 0.0
    source_episode_info: Dict[str, float] = {
        "source_episode_loss": 0.0,
        "source_episode_overflow_rate": 0.0,
        "source_episode_radius_3sigma_deg": float("nan"),
        "source_episode_val_angle_deg": float("nan"),
        "source_episode_classes": 0.0,
        "source_episode_domains": 0.0,
        "source_episode_mixup_count": 0.0,
        "source_episode_mixup_order": float(max(2, int(getattr(args, "soft_unknown_mixup_order", 3)))),
        "source_episode_mixup_loss": 0.0,
        "source_episode_mixup_overflow_rate": 0.0,
        "source_episode_mixup_margin_deg": float("nan"),
    }
    if float(args.lambda_source_episode) > 0.0 and source_episode_stage_scale > 0.0:
        if source_episode_three_sigma_loss is None:
            raise ImportError("cvsrffi.losses.source_episode_three_sigma_loss is required for --lambda_source_episode")
        loss_source_episode_l, source_episode_info = source_episode_three_sigma_loss(
            z_id_l,
            y_l,
            d_l,
            min_domains=int(args.source_episode_min_domains),
            radius_cap_rad=math.radians(float(args.source_episode_radius_cap_deg)),
            mixup_features=soft_unknown_mixup_batch.features if soft_unknown_mixup_batch is not None else None,
            mixup_weight=float(args.source_episode_mixup_weight),
            mixup_order=int(args.soft_unknown_mixup_order),
            mixup_hard_k=int(args.source_episode_mixup_hard_k),
        )
    terms = {
        "tx": loss_tx_l, "dom": loss_dom_l, "adv": loss_adv_l,
        "orth": loss_orth_l, "cons": loss_cons_l, "group_ce": loss_group_ce_l,
        "fishr": loss_fishr_l, "proto": loss_proto_l,
        "open_world_feat": loss_open_world_feat_l * ow_feat_stage_scale,
        "zid_compact": loss_zid_compact_l * zid_warm,
        "proxy_unknown": loss_proxy_unknown_l * proxy_stage_scale,
        "soft_unknown_mixup": loss_soft_unknown_mixup_l * soft_unknown_mixup_stage_scale,
        "source_episode": loss_source_episode_l * source_episode_stage_scale,
    }
    total = terms["tx"] + sum(cur_w[k] * v for k, v in terms.items() if k != "tx")
    return total, terms
