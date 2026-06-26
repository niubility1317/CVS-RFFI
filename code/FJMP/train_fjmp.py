from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from post_stage_cli import add_common_data_args, add_sat_eval_args, str2bool
from post_stage_eval import resolve_sat_eval_max_batches, summarize_post_stage_tests
from training_controls import parse_sat_scenarios, sat_channel_config_for_scenario

try:
    import torch
    import torch.nn as nn
    from torch.cuda.amp import GradScaler, autocast

    from FJMP.frozen_joint_prototype_head import (
        CalibratedFusion,
        ConfidenceGate,
        FrozenJointPrototypeClassifier,
        build_fjmp_checkpoint_payload,
        compute_fjmp_loss,
        compute_ood_distance,
        compute_relative_harm_metrics,
        forward_frozen_backbone,
        init_prototypes_by_class_domain,
    )
    from FJMP.fjmp_v2_losses import compute_fjmp_v2_loss, get_fjmp_v2_stage_weights
    from FJMP.fjmp_v2_proto_head import SafeResidualProtoHead
    from FJMP.prototype_metrics import compute_fjmp_v2_metrics
    from FJMP.base_protected_fusion import BaseProtectedFusion, stage_rho_max
    from FJMP.star_ground_view import StarGroundViewGenerator, estimate_sat_reliability
    from FJMP.sgv_bp_losses import compute_sgv_bp_losses, sgv_bp_stage_config
    from post_stage_common import (
        build_standard_data,
        domain_from_extra,
        ensure_dir,
        evaluate_post_model,
        load_baseline_from_checkpoint,
        mean_logs,
        move_batch,
        resolve_device,
        save_payload,
        set_seed,
    )
except ModuleNotFoundError:
    torch = None
    nn = None
    GradScaler = autocast = None
    CalibratedFusion = ConfidenceGate = FrozenJointPrototypeClassifier = None
    BaseProtectedFusion = stage_rho_max = None
    StarGroundViewGenerator = estimate_sat_reliability = compute_sgv_bp_losses = sgv_bp_stage_config = None
    build_fjmp_checkpoint_payload = compute_fjmp_loss = forward_frozen_backbone = None
    compute_ood_distance = compute_relative_harm_metrics = None
    init_prototypes_by_class_domain = None
    SafeResidualProtoHead = compute_fjmp_v2_loss = get_fjmp_v2_stage_weights = compute_fjmp_v2_metrics = None
    build_standard_data = domain_from_extra = ensure_dir = evaluate_post_model = None
    load_baseline_from_checkpoint = mean_logs = move_batch = resolve_device = None
    save_payload = set_seed = None


class FJMPArgumentParser(argparse.ArgumentParser):
    def parse_args(self, args=None, namespace=None):
        raw_args = list(args) if args is not None else sys.argv[1:]
        parsed = super().parse_args(args, namespace)
        parsed.aggregation = parsed.prototype_aggregation
        _apply_fjmp_v2_defaults(parsed, raw_args)
        return parsed


def _option_was_supplied(raw_args, *names: str) -> bool:
    names = set(names)
    for item in raw_args:
        if item in names:
            return True
        if any(item.startswith(f"{name}=") for name in names):
            return True
    return False


def _set_v2_default(parsed, attr: str, value, raw_args, *option_names: str) -> None:
    if not _option_was_supplied(raw_args, *option_names):
        setattr(parsed, attr, value)


def _apply_fjmp_v2_defaults(parsed, raw_args) -> None:
    version = str(getattr(parsed, "fjmp_version", "v1")).lower().strip()
    if version not in {"v2", "v3"}:
        return

    default_name = "FJMP_V3_AGGRESSIVE_RESCUE" if version == "v3" else "FJMP_V2_K3_SAFE_RESIDUAL"
    _set_v2_default(parsed, "model_name", default_name, raw_args, "--model_name")
    _set_v2_default(parsed, "num_prototypes", 3, raw_args, "--num_prototypes")
    _set_v2_default(parsed, "proto_dim", 256, raw_args, "--proto_dim")
    _set_v2_default(parsed, "init_scale", 10.0, raw_args, "--init_scale")
    _set_v2_default(parsed, "prototype_aggregation", "logsumexp", raw_args, "--prototype_aggregation", "--aggregation")
    parsed.aggregation = parsed.prototype_aggregation
    _set_v2_default(parsed, "zdom_mode", "zero", raw_args, "--zdom_mode")
    _set_v2_default(parsed, "init_zdom_mode", "zero", raw_args, "--init_zdom_mode")
    _set_v2_default(parsed, "epochs", 12, raw_args, "--epochs")
    _set_v2_default(parsed, "freeze_backbone", True, raw_args, "--freeze_backbone")
    _set_v2_default(parsed, "strict_raw", True, raw_args, "--strict_raw")
    _set_v2_default(parsed, "rho_max", 0.30 if version == "v3" else 0.15, raw_args, "--rho_max")
    _set_v2_default(parsed, "delta_clip", 5.0 if version == "v3" else 3.0, raw_args, "--delta_clip")
    _set_v2_default(parsed, "proto_dropout", 0.10, raw_args, "--proto_dropout")
    _set_v2_default(parsed, "lambda_ce_trim", 1.5 if version == "v3" else 1.0, raw_args, "--lambda_ce_trim")
    _set_v2_default(parsed, "lambda_kd_selective", 0.1 if version == "v3" else 0.3, raw_args, "--lambda_kd_selective")
    _set_v2_default(parsed, "lambda_proto_div", 0.01, raw_args, "--lambda_proto_div")
    _set_v2_default(parsed, "lambda_proto_usage", 0.005, raw_args, "--lambda_proto_usage")
    _set_v2_default(parsed, "lambda_assign_entropy", 0.001, raw_args, "--lambda_assign_entropy")
    _set_v2_default(parsed, "lambda_delta", 0.005 if version == "v3" else 0.01, raw_args, "--lambda_delta")
    _set_v2_default(parsed, "lambda_logit_residual", 0.003 if version == "v3" else 0.01, raw_args, "--lambda_logit_residual")
    _set_v2_default(parsed, "lambda_gate", 0.003, raw_args, "--lambda_gate")
    _set_v2_default(parsed, "kd_selective_temperature", 2.0, raw_args, "--kd_selective_temperature")
    _set_v2_default(parsed, "ce_trim_margin", 5.0 if version == "v3" else 3.0, raw_args, "--ce_trim_margin")
    _set_v2_default(parsed, "ce_trim_tau", 1.25 if version == "v3" else 0.75, raw_args, "--ce_trim_tau")
    _set_v2_default(parsed, "ce_trim_rescue_weight", 1.5 if version == "v3" else 0.5, raw_args, "--ce_trim_rescue_weight")
    _set_v2_default(parsed, "tau_div", 0.55, raw_args, "--tau_div")
    _set_v2_default(parsed, "tau_assign", 0.75, raw_args, "--tau_assign")
    _set_v2_default(parsed, "target_entropy_ratio", 0.5, raw_args, "--target_entropy_ratio")
    _set_v2_default(parsed, "delta_ratio_max", 0.15 if version == "v3" else 0.10, raw_args, "--delta_ratio_max")
    _set_v2_default(parsed, "delta_bound_weight", 5.0, raw_args, "--delta_bound_weight")
    _set_v2_default(parsed, "logit_residual_target_norm", 5.0 if version == "v3" else 3.0, raw_args, "--logit_residual_target_norm")
    _set_v2_default(parsed, "proto_warmup_epochs", 3 if version == "v3" else 0, raw_args, "--proto_warmup_epochs")
    _set_v2_default(parsed, "lambda_ce_proto_warmup", 1.0 if version == "v3" else 0.0, raw_args, "--lambda_ce_proto_warmup")
    _set_v2_default(parsed, "kd_conf_threshold", 0.95 if version == "v3" else 0.0, raw_args, "--kd_conf_threshold")
    _set_v2_default(parsed, "kd_margin_threshold", 5.0 if version == "v3" else -1.0, raw_args, "--kd_margin_threshold")
    _set_v2_default(parsed, "dynamic_rho_cap", version == "v3", raw_args, "--dynamic_rho_cap")
    _set_v2_default(parsed, "rho_easy_cap", 0.03, raw_args, "--rho_easy_cap")
    _set_v2_default(parsed, "rho_boundary_cap", 0.30 if version == "v3" else 0.15, raw_args, "--rho_boundary_cap")

    # FJMP-v2 replaces these old protective losses with structural safeguards.
    parsed.use_sgv = False
    parsed.lambda_margin_preserve = 0.0
    parsed.lambda_pres_clean = 0.0
    parsed.lambda_pres_sat = 0.0
    parsed.lambda_harm = 0.0
    parsed.lambda_sgv_safe = 0.0
    parsed.lambda_sgv_margin = 0.0
    parsed.lambda_proto_sgv = 0.0
    parsed.lambda_worst_domain_view = 0.0
    parsed.lambda_gate_view_gap = 0.0


def _is_fjmp_v2(args) -> bool:
    return str(getattr(args, "fjmp_version", "v1")).lower().strip() in {"v2", "v3"}


def _fjmp_v2_loss_cfg(args) -> dict:
    return {
        "K": int(args.num_prototypes),
        "rho_max": float(args.rho_max),
        "lambda_ce_trim": float(args.lambda_ce_trim),
        "lambda_kd_selective": float(args.lambda_kd_selective),
        "lambda_proto_div": float(args.lambda_proto_div),
        "lambda_proto_usage": float(args.lambda_proto_usage),
        "lambda_assign_entropy": float(args.lambda_assign_entropy),
        "lambda_delta": float(args.lambda_delta),
        "lambda_logit_residual": float(args.lambda_logit_residual),
        "lambda_gate": float(args.lambda_gate),
        "ce_trim_margin": float(args.ce_trim_margin),
        "ce_trim_tau": float(args.ce_trim_tau),
        "ce_trim_rescue_weight": float(args.ce_trim_rescue_weight),
        "kd_selective_temperature": float(args.kd_selective_temperature),
        "tau_div": float(args.tau_div),
        "tau_assign": float(args.tau_assign),
        "target_entropy_ratio": float(args.target_entropy_ratio),
        "delta_ratio_max": float(args.delta_ratio_max),
        "delta_bound_weight": float(args.delta_bound_weight),
        "logit_residual_target_norm": float(args.logit_residual_target_norm),
        "proto_warmup_epochs": int(args.proto_warmup_epochs),
        "lambda_ce_proto_warmup": float(args.lambda_ce_proto_warmup),
        "kd_conf_threshold": float(args.kd_conf_threshold),
        "kd_margin_threshold": float(args.kd_margin_threshold),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = FJMPArgumentParser(description="Train a frozen joint multi-prototype head over a Stable-SAT baseline.")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "baseline_eval"])
    parser.add_argument("--baseline_ckpt", type=str, required=True)
    parser.add_argument("--fjmp_version", type=str, default="v1", choices=["v1", "v2", "v3"])
    parser.add_argument("--model_name", type=str, default="FJMP")
    parser.add_argument("--feature_input", type=str, default="z_id")
    parser.add_argument("--num_prototypes", type=int, default=2)
    parser.add_argument("--proto_dim", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=0)
    parser.add_argument("--init_scale", type=float, default=8.0)
    parser.add_argument("--max_scale", type=float, default=30.0)
    parser.add_argument("--scale_reg", type=str2bool, default=False)
    parser.add_argument("--dom_drop_prob", "--zdom_drop_prob", dest="dom_drop_prob", type=float, default=0.3)
    parser.add_argument("--init_res_scale", type=float, default=0.1)
    parser.add_argument("--max_res_scale", type=float, default=0.5)
    parser.add_argument("--prototype_init", type=str, default="class_domain_center")
    parser.add_argument("--prototype_aggregation", "--aggregation", dest="prototype_aggregation", type=str, default="logsumexp")
    parser.add_argument("--zdom_usage", type=str, default="")
    parser.add_argument("--freeze_backbone", type=str2bool, default=True)
    parser.add_argument("--strict_raw", type=str2bool, default=True)
    parser.add_argument("--zdom_mode", type=str, default="zero", choices=["normal", "zero", "mean", "shuffled", "dropout"])
    parser.add_argument("--init_zdom_mode", type=str, default=None, choices=["normal", "zero", "mean", "shuffled", "dropout"])
    parser.add_argument("--fusion_mode", type=str, default="calibrated_logit")
    parser.add_argument("--logit_calibration", type=str, default="")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--fusion_learnable", type=str2bool, default=True)
    parser.add_argument("--probability_weight", type=float, default=0.5)
    parser.add_argument("--eta", type=float, default=0.05)
    parser.add_argument("--eta_max", type=float, default=0.10)
    parser.add_argument("--rho_init", type=float, default=0.03)
    parser.add_argument("--rho_max_stage1", type=float, default=0.10)
    parser.add_argument("--rho_max_stage2", type=float, default=0.25)
    parser.add_argument("--rho_max_stage3", type=float, default=0.30)
    parser.add_argument("--rho_max", type=float, default=0.15)
    parser.add_argument("--delta_clip", type=float, default=3.0)
    parser.add_argument("--proto_dropout", type=float, default=0.10)
    parser.add_argument("--dynamic_rho_cap", type=str2bool, default=False)
    parser.add_argument("--rho_easy_cap", type=float, default=0.03)
    parser.add_argument("--rho_boundary_cap", type=float, default=0.30)
    parser.add_argument("--rho_easy_conf", type=float, default=0.95)
    parser.add_argument("--rho_easy_margin", type=float, default=5.0)
    parser.add_argument("--rho_boundary_margin", type=float, default=3.0)
    parser.add_argument("--max_delta_norm", type=float, default=3.0)
    parser.add_argument("--center_proto", type=str2bool, default=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr_projector", type=float, default=1e-3)
    parser.add_argument("--lr_proto", type=float, default=5e-4)
    parser.add_argument("--lr_calib", type=float, default=5e-4)
    parser.add_argument("--lr_gate", type=float, default=1e-3)
    parser.add_argument("--lr_adapter", type=float, default=1e-3)
    parser.add_argument("--lr_rho", type=float, default=3e-4)
    parser.add_argument("--prototype_lr_decay_epoch", type=int, default=16)
    parser.add_argument("--prototype_lr_decay", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--ce_on", type=str, default="auto", choices=["auto", "none", "fused", "proto", "both"])
    parser.add_argument("--ce_proto_weight", type=float, default=1.0)
    parser.add_argument("--ce_fused_weight", type=float, default=1.0)
    parser.add_argument("--kd_on", type=str, default="auto", choices=["auto", "none", "fused", "proto", "both"])
    parser.add_argument("--lambda_kd", type=float, default=0.3)
    parser.add_argument("--kd_temperature", type=float, default=4.0)
    parser.add_argument("--lambda_dnh", type=float, default=0.0)
    parser.add_argument("--dnh_margin", type=float, default=0.0)
    parser.add_argument("--lambda_margin_preserve", type=float, default=0.0)
    parser.add_argument("--margin_preserve_delta", type=float, default=0.0)
    parser.add_argument("--lambda_proxy_cons", type=float, default=0.0)
    parser.add_argument("--use_sgv", nargs="?", const=True, type=str2bool, default=False)
    parser.add_argument("--sgv_train_strength", type=str, default="")
    parser.add_argument("--sgv_eval_strength", type=str, default="")
    parser.add_argument("--use_sat_reliability", nargs="?", const=True, type=str2bool, default=False)
    parser.add_argument("--lambda_ce_head_clean", type=float, default=0.30)
    parser.add_argument("--lambda_ce_head_sat", type=float, default=0.15)
    parser.add_argument("--lambda_ce_safe_clean", type=float, default=0.05)
    parser.add_argument("--lambda_ce_safe_sat", type=float, default=0.02)
    parser.add_argument("--lambda_pres_clean", type=float, default=3.0)
    parser.add_argument("--lambda_pres_sat", type=float, default=1.5)
    parser.add_argument("--lambda_harm", type=float, default=2.0)
    parser.add_argument("--lambda_kd_easy", type=float, default=1.5)
    parser.add_argument("--lambda_kd_mid", type=float, default=0.5)
    parser.add_argument("--lambda_kd_hard_low_margin", type=float, default=0.05)
    parser.add_argument("--lambda_sgv_head", type=float, default=0.5)
    parser.add_argument("--lambda_sgv_safe", type=float, default=1.0)
    parser.add_argument("--lambda_sgv_margin", type=float, default=0.3)
    parser.add_argument("--lambda_proto_sgv", type=float, default=0.2)
    parser.add_argument("--lambda_worst_domain_view", type=float, default=0.3)
    parser.add_argument("--lambda_gate_easy", type=float, default=0.08)
    parser.add_argument("--lambda_gate_view_gap", type=float, default=0.03)
    parser.add_argument("--proxy_aug", type=str, default="none", choices=["none", "weak_rf", "sat07", "mixed_orbit", "receiver_mix", "cfo_phase_noise"])
    parser.add_argument("--enable_conf_gate", type=str2bool, default=False)
    parser.add_argument("--gate_type", type=str, default="none", choices=["none", "margin", "entropy", "ood", "combined"])
    parser.add_argument("--gate_threshold_score", type=float, default=0.0)
    parser.add_argument("--gate_threshold_margin", type=float, default=0.0)
    parser.add_argument("--gate_threshold_entropy", type=float, default=1.5)
    parser.add_argument("--gate_threshold_ood", type=float, default=0.35)
    parser.add_argument("--log_base_proto_fused", type=str2bool, default=True)
    parser.add_argument("--save_epoch_metrics_csv", type=str2bool, default=True)
    parser.add_argument("--metrics_csv", type=str, default="")
    parser.add_argument("--save_checkpoints", type=str2bool, default=True)
    parser.add_argument("--init_only_eval", type=str2bool, default=False)
    parser.add_argument("--check_backbone_grad", type=str2bool, default=False)
    parser.add_argument("--repeat_seeds", type=str, default="")
    parser.add_argument("--dnh_split", type=str, default="train")
    parser.add_argument("--consistency_target", type=str, default="")
    parser.add_argument("--zdom_to_classifier", type=str, default="")
    parser.add_argument("--zdom_to_gate", type=str, default="")
    parser.add_argument("--diagnostic", type=str, default="")
    parser.add_argument("--selection_metric", type=str, default="")
    parser.add_argument("--sat_scenario", type=str, default="")
    parser.add_argument("--sat_strength", type=str, default="")
    parser.add_argument("--fusion_batch_index", type=int, default=-1)
    parser.add_argument("--loss_batch_index", type=int, default=-1)
    parser.add_argument("--geometry_batch_index", type=int, default=-1)
    parser.add_argument("--zdom_batch_index", type=int, default=-1)
    parser.add_argument("--proxy_batch_index", type=int, default=-1)
    parser.add_argument("--lambda_sep", type=float, default=0.01)
    parser.add_argument("--lambda_div", type=float, default=0.003)
    parser.add_argument("--lambda_usage", type=float, default=0.003)
    parser.add_argument("--lambda_delta", type=float, default=0.0005)
    parser.add_argument("--lambda_cov", type=float, default=0.0)
    parser.add_argument("--lambda_ce_trim", type=float, default=1.0)
    parser.add_argument("--lambda_kd_selective", type=float, default=0.3)
    parser.add_argument("--lambda_proto_div", type=float, default=0.01)
    parser.add_argument("--lambda_proto_usage", type=float, default=0.005)
    parser.add_argument("--lambda_assign_entropy", type=float, default=0.001)
    parser.add_argument("--lambda_logit_residual", type=float, default=0.01)
    parser.add_argument("--lambda_gate", type=float, default=0.003)
    parser.add_argument("--ce_trim_margin", type=float, default=3.0)
    parser.add_argument("--ce_trim_tau", type=float, default=0.75)
    parser.add_argument("--ce_trim_rescue_weight", type=float, default=0.5)
    parser.add_argument("--kd_selective_temperature", type=float, default=2.0)
    parser.add_argument("--tau_div", type=float, default=0.55)
    parser.add_argument("--tau_assign", type=float, default=0.75)
    parser.add_argument("--target_entropy_ratio", type=float, default=0.5)
    parser.add_argument("--delta_ratio_max", type=float, default=0.10)
    parser.add_argument("--delta_bound_weight", type=float, default=5.0)
    parser.add_argument("--logit_residual_target_norm", type=float, default=3.0)
    parser.add_argument("--proto_warmup_epochs", type=int, default=0)
    parser.add_argument("--lambda_ce_proto_warmup", type=float, default=0.0)
    parser.add_argument("--kd_conf_threshold", type=float, default=0.0)
    parser.add_argument("--kd_margin_threshold", type=float, default=-1.0)
    add_sat_eval_args(parser)
    parser.add_argument("--sat_fs_hz", type=float, default=25e6)
    parser.add_argument("--sat_fc_hz", type=float, default=2.462e9)
    parser.add_argument("--amp", type=str2bool, default=True)
    parser.add_argument("--dry_run", action="store_true")
    add_common_data_args(parser)
    return parser


def _id_key(feature_input: str) -> str:
    aliases = {
        "z_id": "z_id",
        "z_id_zdom": "z_id",
        "id_feat_joint": "id_feat_joint",
        "feat_joint": "id_feat_joint",
        "id_feat_pa": "id_feat_pa",
        "id_feat_dac": "id_feat_dac",
    }
    return aliases.get(str(feature_input), str(feature_input))


class FJMPWrappedModel(nn.Module if nn is not None else object):
    def __init__(
        self,
        frozen_model: nn.Module,
        proto_model: FrozenJointPrototypeClassifier,
        fusion: CalibratedFusion,
        gate: ConfidenceGate | None = None,
        *,
        id_key: str,
        strict_raw: bool,
        zdom_mode: str,
        fusion_args=None,
    ) -> None:
        super().__init__()
        self.frozen_model = frozen_model
        self.proto_model = proto_model
        self.fusion = fusion
        self.gate = gate
        self.id_key = id_key
        self.strict_raw = strict_raw
        self.zdom_mode = zdom_mode
        self.fusion_args = fusion_args

    def forward(self, x, y_tx=None, grl_lambda: float = 1.0, return_aux: bool = True, domain_labels=None):
        with torch.no_grad():
            base_out = self.frozen_model(x, y_tx=None, grl_lambda=grl_lambda, return_aux=True)
            feats = forward_frozen_backbone(
                self.frozen_model,
                x,
                id_key=self.id_key,
                strict_raw=self.strict_raw,
                y_tx=None,
                grl_lambda=grl_lambda,
            )
        if isinstance(self.proto_model, SafeResidualProtoHead):
            fused_logits, aux = self.proto_model(feats["z_id_raw"], feats["base_logits"])
            out = dict(base_out) if isinstance(base_out, dict) else {}
            out["tx_logits"] = fused_logits
            out["head_logits"] = aux["proto_logits"]
            out["proto_logits"] = aux["proto_logits"]
            out["fused_logits"] = fused_logits
            out["safe_logits"] = fused_logits
            out["base_logits"] = feats["base_logits"]
            out["rho"] = aux["rho"]
            out["delta"] = aux["delta"]
            return out
        proto_out = self.proto_model(feats["z_id_raw"], feats["z_dom"], zdom_mode=self.zdom_mode)
        gate_out = {}
        accept_proto = None
        if self.gate is not None:
            ood_distance = compute_ood_distance(proto_out["z_joint"], self.proto_model.head.prototypes)
            gate_out = self.gate(
                proto_logits=proto_out["logits"],
                proto_scores=proto_out["proto_scores"],
                nearest_proto_score=proto_out["nearest_proto_score"],
                ood_distance=ood_distance,
            )
            accept_proto = gate_out["accept_proto"]
        fused = _fusion_forward(
            self.fusion,
            base_logits=feats["base_logits"],
            proto_logits=proto_out["logits"],
            accept_proto=accept_proto,
            args=self.fusion_args,
        )
        out = dict(base_out) if isinstance(base_out, dict) else {}
        out["tx_logits"] = fused["logits"]
        out["head_logits"] = proto_out["logits"]
        out["proto_logits"] = proto_out["logits"]
        out["fused_logits"] = fused["logits"]
        out["safe_logits"] = fused.get("safe_logits", fused["logits"])
        out["base_logits"] = feats["base_logits"]
        if "rho" in fused:
            out["rho"] = fused["rho"]
        if "delta" in fused:
            out["delta"] = fused["delta"]
        if "gate" in fused:
            out["gate"] = fused["gate"]
        if gate_out:
            out["accept_proto"] = gate_out["accept_proto"]
        return out


def _initialize_prototypes(
    proto_model,
    frozen_model,
    loader,
    device,
    id_key: str,
    strict_raw: bool,
    domain_label_map,
    max_batches: int = 8,
    init_zdom_mode: str = "normal",
) -> None:
    with torch.no_grad():
        features = []
        labels = []
        domains = []
        for bi, batch in enumerate(loader):
            x, y, extra = move_batch(batch, device)
            feats = forward_frozen_backbone(frozen_model, x, id_key=id_key, strict_raw=strict_raw)
            out = proto_model(feats["z_id_raw"], feats["z_dom"], zdom_mode=str(init_zdom_mode))
            features.append(out["z_joint"].detach())
            labels.append(y.detach())
            d = domain_from_extra(extra, domain_label_map, device)
            domains.append((torch.zeros_like(y) if d is None else d.clamp_min(0)).detach())
            if bi + 1 >= max_batches:
                break
    if features:
        init_prototypes_by_class_domain(
            proto_model.head,
            torch.cat(features, dim=0),
            torch.cat(labels, dim=0),
            torch.cat(domains, dim=0),
            num_domains=max(1, len(domain_label_map)),
        )


def _build_confidence_gate(args):
    if not bool(args.enable_conf_gate) and str(args.fusion_mode).lower() != "confidence_gated":
        return None
    threshold_ood = None
    if str(args.gate_type).lower() in {"ood", "combined"}:
        threshold_ood = float(args.gate_threshold_ood)
    return ConfidenceGate(
        threshold_score=float(args.gate_threshold_score),
        threshold_margin=float(args.gate_threshold_margin),
        threshold_entropy=float(args.gate_threshold_entropy),
        threshold_ood=threshold_ood,
    )


def _build_fusion(args, num_classes: int):
    mode = str(args.fusion_mode).lower().strip()
    if mode == "base_protected_residual":
        return BaseProtectedFusion(
            num_classes=int(num_classes),
            rho_init=float(args.rho_init),
            rho_max=float(args.rho_max_stage1),
            max_delta_norm=float(args.max_delta_norm),
            learn_temperature=str(args.logit_calibration).lower().strip() in {"", "centered_temperature"},
        )
    return CalibratedFusion(
        alpha=float(args.alpha),
        beta=float(args.beta),
        mode=str(args.fusion_mode),
        learnable=bool(args.fusion_learnable),
        probability_weight=float(args.probability_weight),
        eta=float(args.eta),
        eta_max=float(args.eta_max),
        center_proto=bool(args.center_proto),
    )


def _fusion_forward(fusion, *, base_logits, proto_logits, accept_proto=None, epoch: int = 1, args=None):
    if isinstance(fusion, BaseProtectedFusion):
        rho_cap = stage_rho_max(
            epoch,
            stage1=float(args.rho_max_stage1),
            stage2=float(args.rho_max_stage2),
            stage3=float(args.rho_max_stage3),
        ) if args is not None else None
        if rho_cap is not None:
            fusion.set_rho_max(rho_cap)
        return fusion(base_logits=base_logits.detach(), head_logits=proto_logits, accept_proto=accept_proto, rho_max=rho_cap)
    return fusion(base_logits=base_logits, proto_logits=proto_logits, accept_proto=accept_proto)


def _build_optimizer(args, proto_model, fusion):
    if isinstance(proto_model, SafeResidualProtoHead):
        return torch.optim.AdamW(
            [param for param in proto_model.parameters() if param.requires_grad],
            lr=float(args.lr),
            weight_decay=float(args.weight_decay),
        )
    if str(getattr(args, "model_name", "")).upper() == "SGV-BP-FJMP" or str(args.fusion_mode).lower() == "base_protected_residual":
        groups = [
            {"params": list(proto_model.projector.id_proj.parameters()), "lr": float(args.lr_projector)},
            {"params": list(proto_model.projector.delta_net.parameters()), "lr": float(args.lr_adapter)},
            {"params": list(proto_model.projector.gate_net.parameters()), "lr": float(args.lr_gate)},
            {"params": [proto_model.head.prototypes], "lr": float(args.lr_proto)},
            {"params": [proto_model.head.log_scale], "lr": float(args.lr_calib)},
        ]
        if isinstance(fusion, BaseProtectedFusion):
            groups.extend(
                [
                    {"params": list(fusion.head_calibrator.parameters()) + list(fusion.base_calibrator.parameters()), "lr": float(args.lr_calib)},
                    {"params": [fusion.raw_rho_bias], "lr": float(args.lr_rho)},
                ]
            )
            if fusion.gate is not None:
                groups.append({"params": list(fusion.gate.parameters()), "lr": float(args.lr_gate)})
        else:
            groups.append({"params": list(fusion.parameters()), "lr": float(args.lr_calib)})
        groups = [group for group in groups if len(group["params"]) > 0]
        return torch.optim.AdamW(groups, weight_decay=float(args.weight_decay))
    return torch.optim.AdamW(
        list(proto_model.parameters()) + list(fusion.parameters()),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )


def _scalar_logs(prefix: str, values) -> dict:
    out = {}
    for key, value in values.items():
        if torch.is_tensor(value):
            if value.numel() == 1:
                out[f"{prefix}/{key}"] = value.detach()
        else:
            out[f"{prefix}/{key}"] = value
    return out


FJMP_LOSS_KEYS = (
    "fjmp_loss",
    "loss_ce",
    "loss_ce_proto",
    "loss_ce_fused",
    "loss_kd",
    "loss_kd_proto",
    "loss_kd_fused",
    "loss_dnh",
    "loss_margin_preserve",
    "loss_sep",
    "loss_div",
    "loss_usage",
    "loss_delta",
    "loss_cov",
    "loss_ce_proto_warmup",
    "loss_ce_trim",
    "loss_kd_selective",
    "loss_proto_div",
    "loss_proto_usage",
    "loss_assign_entropy",
    "loss_logit_residual",
    "loss_gate",
    "w_ce_mean",
    "w_kd_mean",
)

SGV_LOSS_KEYS = (
    "sgv_loss",
    "sgv_ce_head_clean",
    "sgv_ce_safe_clean",
    "sgv_pres_clean",
    "sgv_harm",
    "sgv_kd_easy",
    "sgv_kd_mid",
    "sgv_kd_hard_low_margin",
    "sgv_gate_easy",
    "sgv_delta",
    "sgv_rho_reg",
    "sgv_ce_head_sat",
    "sgv_ce_safe_sat",
    "sgv_pres_sat",
    "sgv_sgv_head",
    "sgv_sgv_safe",
    "sgv_sgv_margin",
    "sgv_gate_view_gap",
    "sgv_proto_sgv",
    "sgv_worst",
)


def _finite_float(value) -> float | None:
    if value is None:
        return None
    if torch is not None and torch.is_tensor(value):
        if value.numel() != 1:
            return None
        value = value.detach().cpu()
    try:
        val = float(value)
    except Exception:
        return None
    return val if math.isfinite(val) else None


def _format_loss_line(label: str, logs: dict, keys, *, prefix: str = "train/") -> str:
    parts = []
    for key in keys:
        val = _finite_float(logs.get(f"{prefix}{key}"))
        if val is not None:
            parts.append(f"{key}={val:.4f}")
    return f"{label} " + (" ".join(parts) if parts else "none")


def _format_weight_line(label: str, weights: dict, keys, *, source_prefix: str = "") -> str:
    parts = []
    for key in keys:
        weight_key = key[len(source_prefix) :] if source_prefix and key.startswith(source_prefix) else key
        if weight_key not in weights:
            continue
        val = _finite_float(weights.get(weight_key))
        if val is not None:
            parts.append(f"{key}={val:.4g}")
    return f"{label} " + (" ".join(parts) if parts else "none")


def _format_top_loss_contributors(logs: dict, *, top_k: int = 6) -> str:
    items = []
    for key, value in logs.items():
        if not str(key).startswith("train_weighted/"):
            continue
        name = str(key).split("/", 1)[1]
        if name in {"loss", "fjmp_loss", "sgv_loss"}:
            continue
        val = _finite_float(value)
        if val is None:
            continue
        items.append((name, val))
    items.sort(key=lambda item: abs(item[1]), reverse=True)
    if not items:
        return "[LOSS-TOP] none"
    return "[LOSS-TOP] " + " ".join(f"{name}={val:.4f}" for name, val in items[: max(1, int(top_k))])


def _print_config_group(label: str, args, keys) -> None:
    values = []
    for key in keys:
        if hasattr(args, key):
            values.append(f"{key}={getattr(args, key)}")
    print(f"{label} " + " ".join(values), flush=True)


def _print_experiment_config(args, legacy_ce_on: str, legacy_kd_on: str, out_dir) -> None:
    print("[CONFIG-BEGIN]", flush=True)
    print(
        f"[CONFIG-RUN] model_name={args.model_name} output_dir={out_dir} "
        f"baseline_ckpt={args.baseline_ckpt} epochs={args.epochs} amp={args.amp} seed={args.seed}",
        flush=True,
    )
    _print_config_group(
        "[CONFIG-DATA]",
        args,
        (
            "dataset",
            "wisig_pkl",
            "wisig_equalized",
            "wisig_domain",
            "wisig_train_days",
            "wisig_test_days",
            "wisig_train_rxs",
            "wisig_test_rxs",
            "batch_size",
            "eval_batch_size",
        ),
    )
    _print_config_group(
        "[CONFIG-MODEL]",
        args,
        (
            "fjmp_version",
            "model_name",
            "feature_input",
            "num_prototypes",
            "proto_dim",
            "prototype_init",
            "prototype_aggregation",
            "zdom_usage",
            "zdom_mode",
            "fusion_mode",
            "logit_calibration",
            "rho_max",
            "delta_clip",
            "proto_dropout",
            "dynamic_rho_cap",
            "rho_easy_cap",
            "rho_boundary_cap",
            "rho_easy_margin",
            "rho_boundary_margin",
            "rho_init",
            "rho_max_stage1",
            "rho_max_stage2",
            "rho_max_stage3",
            "max_delta_norm",
        ),
    )
    _print_config_group(
        "[CONFIG-OPT]",
        args,
        (
            "lr",
            "lr_projector",
            "lr_proto",
            "lr_calib",
            "lr_gate",
            "lr_adapter",
            "lr_rho",
            "weight_decay",
            "grad_clip",
            "prototype_lr_decay_epoch",
            "prototype_lr_decay",
        ),
    )
    print(
        f"[CONFIG-LOSS] ce_on={legacy_ce_on} kd_on={legacy_kd_on} "
        f"ce_proto_weight={args.ce_proto_weight} ce_fused_weight={args.ce_fused_weight} "
        f"lambda_kd={args.lambda_kd} lambda_sep={args.lambda_sep} lambda_div={args.lambda_div} "
        f"lambda_usage={args.lambda_usage} lambda_delta={args.lambda_delta} lambda_cov={args.lambda_cov} "
        f"lambda_ce_trim={args.lambda_ce_trim} lambda_kd_selective={args.lambda_kd_selective} "
        f"lambda_proto_div={args.lambda_proto_div} lambda_proto_usage={args.lambda_proto_usage} "
        f"lambda_assign_entropy={args.lambda_assign_entropy} lambda_logit_residual={args.lambda_logit_residual} "
        f"lambda_gate={args.lambda_gate} lambda_ce_proto_warmup={args.lambda_ce_proto_warmup} "
        f"proto_warmup_epochs={args.proto_warmup_epochs} kd_conf_threshold={args.kd_conf_threshold} "
        f"kd_margin_threshold={args.kd_margin_threshold}",
        flush=True,
    )
    _print_config_group(
        "[CONFIG-SGV-LOSS]",
        args,
        (
            "use_sgv",
            "sgv_train_strength",
            "sgv_eval_strength",
            "use_sat_reliability",
            "lambda_ce_head_clean",
            "lambda_ce_head_sat",
            "lambda_ce_safe_clean",
            "lambda_ce_safe_sat",
            "lambda_pres_clean",
            "lambda_pres_sat",
            "lambda_harm",
            "lambda_kd_easy",
            "lambda_kd_mid",
            "lambda_kd_hard_low_margin",
            "lambda_sgv_head",
            "lambda_sgv_safe",
            "lambda_sgv_margin",
            "lambda_proto_sgv",
            "lambda_worst_domain_view",
            "lambda_gate_easy",
            "lambda_gate_view_gap",
        ),
    )
    _print_config_group(
        "[CONFIG-EVAL]",
        args,
        (
            "selection_metric",
            "log_base_proto_fused",
            "save_epoch_metrics_csv",
            "metrics_csv",
            "save_checkpoints",
            "eval_max_batches",
            "eval_sat_channel",
            "eval_sat_scenarios",
            "sat_eval_max_batches",
        ),
    )
    print("[CONFIG-END]", flush=True)


def _legacy_loss_component_weights(args, ce_on: str, kd_on: str) -> dict:
    ce_on = str(ce_on or "none").lower().strip()
    kd_on = str(kd_on or "none").lower().strip()
    return {
        "loss_ce": 1.0,
        "loss_ce_proto": float(args.ce_proto_weight) if ce_on in {"proto", "both"} else 0.0,
        "loss_ce_fused": float(args.ce_fused_weight) if ce_on in {"fused", "both"} else 0.0,
        "loss_kd": float(args.lambda_kd),
        "loss_kd_proto": float(args.lambda_kd) if kd_on in {"proto", "both"} else 0.0,
        "loss_kd_fused": float(args.lambda_kd) if kd_on in {"fused", "both"} else 0.0,
        "loss_dnh": float(args.lambda_dnh),
        "loss_margin_preserve": float(args.lambda_margin_preserve),
        "loss_sep": float(args.lambda_sep),
        "loss_div": float(args.lambda_div),
        "loss_usage": float(args.lambda_usage),
        "loss_delta": float(args.lambda_delta),
        "loss_cov": float(args.lambda_cov),
    }


def _fjmp_v2_loss_component_weights(args, epoch: int) -> dict:
    weights = get_fjmp_v2_stage_weights(epoch, _fjmp_v2_loss_cfg(args))
    return {
        "loss_ce_proto_warmup": float(args.lambda_ce_proto_warmup)
        if epoch <= int(args.proto_warmup_epochs)
        else 0.0,
        "loss_ce_trim": weights.ce_trim,
        "loss_kd_selective": weights.kd,
        "loss_proto_div": weights.div,
        "loss_proto_usage": weights.usage,
        "loss_assign_entropy": weights.entropy,
        "loss_delta": weights.delta,
        "loss_logit_residual": weights.logit,
        "loss_gate": weights.gate,
    }


def _sgv_loss_weights_for_epoch(args, epoch: int) -> dict:
    weights = {}
    if sgv_bp_stage_config is not None:
        weights.update(sgv_bp_stage_config(epoch).weights)
    weights.update(_sgv_weight_overrides(args))
    return weights


def _weighted_loss_logs(losses: dict, weights: dict, *, source_prefix: str = "", output_prefix: str = "") -> dict:
    out = {}
    for name, weight in weights.items():
        key = f"{source_prefix}{name}"
        if key not in losses:
            continue
        value = losses[key]
        if torch is not None and torch.is_tensor(value):
            out[f"{output_prefix}{key}"] = value.detach() * float(weight)
            continue
        val = _finite_float(value)
        if val is not None:
            out[f"{output_prefix}{key}"] = val * float(weight)
    return out


def _sgv_weight_overrides(args) -> dict:
    return {
        "ce_head_clean": float(args.lambda_ce_head_clean),
        "ce_head_sat": float(args.lambda_ce_head_sat),
        "ce_safe_clean": float(args.lambda_ce_safe_clean),
        "ce_safe_sat": float(args.lambda_ce_safe_sat),
        "pres_clean": float(args.lambda_pres_clean),
        "pres_sat": float(args.lambda_pres_sat),
        "harm": float(args.lambda_harm),
        "kd_easy": float(args.lambda_kd_easy),
        "kd_mid": float(args.lambda_kd_mid),
        "kd_hard_low_margin": float(args.lambda_kd_hard_low_margin),
        "sgv_head": float(args.lambda_sgv_head),
        "sgv_safe": float(args.lambda_sgv_safe),
        "sgv_margin": float(args.lambda_sgv_margin),
        "proto_sgv": float(args.lambda_proto_sgv),
        "worst": float(args.lambda_worst_domain_view),
        "gate_easy": float(args.lambda_gate_easy),
        "gate_view_gap": float(args.lambda_gate_view_gap),
        "delta": float(args.lambda_delta),
    }


def _sgv_strength_for_epoch(args, epoch: int) -> str:
    strengths = [item.strip().replace("sat_", "") for item in str(args.sgv_train_strength or "low,mid").split(",") if item.strip()]
    if not strengths:
        return "mid"
    if epoch <= 5 and "low" in strengths:
        return "low"
    if "mid" in strengths:
        return "mid"
    return strengths[-1]


def _resolve_legacy_loss_targets(args) -> tuple[str, str]:
    """Keep old FJMP defaults, but do not double-apply CE/KD in SGV-BP mode."""

    is_sgv_bp = bool(args.use_sgv) or str(getattr(args, "model_name", "")).upper() == "SGV-BP-FJMP"
    ce_on = str(args.ce_on or "auto").lower().strip()
    kd_on = str(args.kd_on or "auto").lower().strip()
    if ce_on == "auto":
        ce_on = "none" if is_sgv_bp else "fused"
    if kd_on == "auto":
        kd_on = "none" if is_sgv_bp else "fused"
    return ce_on, kd_on


def _params_are_finite(modules) -> bool:
    for module in modules:
        for param in module.parameters():
            if param.grad is not None and not bool(torch.isfinite(param.grad).all()):
                return False
    return True


def _format_nonfinite_losses(losses) -> str:
    parts = []
    for key, value in losses.items():
        if torch.is_tensor(value) and value.numel() == 1:
            try:
                val = float(value.detach().cpu())
            except Exception:
                continue
            if not math.isfinite(val):
                parts.append(f"{key}={val}")
    return ",".join(parts) if parts else "unknown"


def _append_epoch_metrics_csv(path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    fieldnames = sorted(row.keys())
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fieldnames})


def evaluate_fjmp_loader_heads(
    baseline,
    proto_model,
    fusion,
    gate,
    loader,
    device,
    *,
    id_key: str,
    strict_raw: bool,
    zdom_mode: str,
    max_batches: int = 0,
    fusion_args=None,
) -> dict:
    baseline.eval()
    proto_model.eval()
    if fusion is not None:
        fusion.eval()
    total = 0
    base_correct = 0
    proto_correct = 0
    fused_correct = 0
    changed = 0
    rescue = 0
    harm = 0
    harm_conf_sum = 0.0
    harm_count = 0
    accept_count = 0
    ood_reject_count = 0
    accept_seen = False
    for bi, batch in enumerate(loader):
        if int(max_batches) > 0 and bi >= int(max_batches):
            break
        x, y, _ = move_batch(batch, device)
        feats = forward_frozen_backbone(baseline, x, id_key=id_key, strict_raw=strict_raw)
        if isinstance(proto_model, SafeResidualProtoHead):
            fused_logits, aux = proto_model(feats["z_id_raw"], feats["base_logits"])
            proto_logits = aux["proto_logits"]
            fused = {"logits": fused_logits}
            accept_proto = None
        else:
            proto_out = proto_model(feats["z_id_raw"], feats["z_dom"], zdom_mode=zdom_mode)
            proto_logits = proto_out["logits"]
            accept_proto = None
            if gate is not None:
                ood_distance = compute_ood_distance(proto_out["z_joint"], proto_model.head.prototypes)
                gate_out = gate(
                    proto_logits=proto_out["logits"],
                    proto_scores=proto_out["proto_scores"],
                    nearest_proto_score=proto_out["nearest_proto_score"],
                    ood_distance=ood_distance,
                )
                accept_proto = gate_out["accept_proto"]
                accept_seen = True
                accept_count += int(accept_proto.sum().item())
                ood_reject_count += int((~accept_proto).sum().item())
            fused = _fusion_forward(
                fusion,
                base_logits=feats["base_logits"],
                proto_logits=proto_out["logits"],
                accept_proto=accept_proto,
                args=fusion_args,
            )
        y = y.long()
        pred_base = feats["base_logits"].argmax(dim=1)
        pred_proto = proto_logits.argmax(dim=1)
        pred_fused = fused["logits"].argmax(dim=1)
        base_ok = pred_base.eq(y)
        proto_ok = pred_proto.eq(y)
        fused_ok = pred_fused.eq(y)
        changed_mask = pred_fused.ne(pred_base)
        rescue_mask = (~base_ok) & fused_ok
        harm_mask = base_ok & (~fused_ok)
        conf = torch.softmax(fused["logits"].float(), dim=1).max(dim=1).values
        n = int(y.numel())
        total += n
        base_correct += int(base_ok.sum().item())
        proto_correct += int(proto_ok.sum().item())
        fused_correct += int(fused_ok.sum().item())
        changed += int(changed_mask.sum().item())
        rescue += int(rescue_mask.sum().item())
        harm += int(harm_mask.sum().item())
        if bool(harm_mask.any()):
            harm_conf_sum += float(conf[harm_mask].sum().item())
            harm_count += int(harm_mask.sum().item())
    denom = max(1, total)
    return {
        "base_tx_acc": 100.0 * base_correct / denom,
        "proto_tx_acc": 100.0 * proto_correct / denom,
        "fused_tx_acc": 100.0 * fused_correct / denom,
        "changed_pred_rate": changed / denom,
        "rescue_rate": rescue / denom,
        "harm_rate": harm / denom,
        "net_gain_rate": (rescue - harm) / denom,
        "harm_conf_mean": harm_conf_sum / max(1, harm_count),
        "proto_accept_rate": (accept_count / denom) if accept_seen else float("nan"),
        "ood_reject_rate": (ood_reject_count / denom) if accept_seen else float("nan"),
        "tx_total": total,
    }


def evaluate_fjmp_named_heads(
    baseline,
    proto_model,
    fusion,
    gate,
    loaders,
    device,
    *,
    id_key: str,
    strict_raw: bool,
    zdom_mode: str,
    max_batches: int = 0,
    fusion_args=None,
) -> dict:
    return {
        name: evaluate_fjmp_loader_heads(
            baseline,
            proto_model,
            fusion,
            gate,
            loader,
            device,
            id_key=id_key,
            strict_raw=strict_raw,
            zdom_mode=zdom_mode,
            max_batches=max_batches,
            fusion_args=fusion_args,
        )
        for name, loader in loaders.items()
    }


def train(args) -> int:
    if torch is None:
        raise ModuleNotFoundError("PyTorch is required to run train_fjmp.py training.")
    from cvsrffi.eval import SatSimConfig, evaluate_sat_scenarios, format_sat_test_lines

    set_seed(int(args.seed))
    device = resolve_device(args.device)
    out_dir = ensure_dir(args.output_dir)
    metrics_csv = Path(args.metrics_csv) if str(args.metrics_csv or "").strip() else out_dir / "metrics_epoch.csv"
    data_ctx = build_standard_data(args, device)
    baseline, ckpt, model_args = load_baseline_from_checkpoint(
        args.baseline_ckpt,
        args,
        data_ctx,
        device,
        freeze=bool(args.freeze_backbone),
    )
    id_key = _id_key(args.feature_input)

    first_x, _, _ = move_batch(next(iter(data_ctx["train_loader"])), device)
    feats = forward_frozen_backbone(baseline, first_x, id_key=id_key, strict_raw=bool(args.strict_raw))
    if _is_fjmp_v2(args):
        proto_model = SafeResidualProtoHead(
            in_dim=int(feats["z_id_raw"].size(1)),
            proto_dim=int(args.proto_dim),
            num_classes=int(args.num_classes),
            K=int(args.num_prototypes),
            rho_max=float(args.rho_max),
            delta_clip=float(args.delta_clip),
            proto_dropout=float(args.proto_dropout),
            logit_scale_init=float(args.init_scale),
            dynamic_rho_cap=bool(args.dynamic_rho_cap),
            rho_easy_cap=float(args.rho_easy_cap),
            rho_boundary_cap=float(args.rho_boundary_cap),
            rho_easy_conf=float(args.rho_easy_conf),
            rho_easy_margin=float(args.rho_easy_margin),
            rho_boundary_margin=float(args.rho_boundary_margin),
        ).to(device)
        fusion = CalibratedFusion(mode="base_only", learnable=False).to(device)
        gate = None
    else:
        proto_model = FrozenJointPrototypeClassifier(
            id_dim=int(feats["z_id_raw"].size(1)),
            dom_dim=int(feats["z_dom"].size(1)),
            num_classes=int(args.num_classes),
            num_prototypes=int(args.num_prototypes),
            proto_dim=int(args.proto_dim),
            hidden_dim=None if int(args.hidden_dim) <= 0 else int(args.hidden_dim),
            init_scale=float(args.init_scale),
            dom_drop_prob=float(args.dom_drop_prob),
            init_res_scale=float(args.init_res_scale),
            max_res_scale=float(args.max_res_scale),
            aggregation=str(args.prototype_aggregation),
        ).to(device)
        fusion = _build_fusion(args, int(args.num_classes)).to(device)
        gate = _build_confidence_gate(args)
        if gate is not None:
            gate = gate.to(device)
    sgv_generator = StarGroundViewGenerator(sample_rate_hz=float(args.sat_fs_hz)).to(device) if bool(args.use_sgv) else None
    init_zdom_mode = str(args.init_zdom_mode or args.zdom_mode)
    if not _is_fjmp_v2(args):
        _initialize_prototypes(
            proto_model,
            baseline,
            data_ctx["train_loader"],
            device,
            id_key,
            bool(args.strict_raw),
            data_ctx["domain_label_map"],
            init_zdom_mode=init_zdom_mode,
        )

    optimizer = _build_optimizer(args, proto_model, fusion)
    scaler = GradScaler(enabled=bool(args.amp and device.type == "cuda"))
    args.eval_sat_scenario_list = parse_sat_scenarios(args.eval_sat_scenarios) if bool(args.eval_sat_channel) else []
    if bool(args.eval_sat_channel) and SatSimConfig is None:
        raise ImportError("sat_channel.py is required when --eval_sat_channel is enabled.")
    if bool(args.eval_sat_channel):
        for scenario in args.eval_sat_scenario_list:
            sat_channel_config_for_scenario(scenario)
    legacy_ce_on, legacy_kd_on = _resolve_legacy_loss_targets(args)
    print(
        f"[FJMP-TRAIN] baseline={args.baseline_ckpt} id_key={id_key} zdom_mode={args.zdom_mode} "
        f"init_zdom_mode={init_zdom_mode} K={args.num_prototypes} agg={args.prototype_aggregation} "
        f"fusion={args.fusion_mode} ce_on={legacy_ce_on} kd_on={legacy_kd_on} "
        f"sgv={bool(args.use_sgv)} epochs={args.epochs} output={out_dir}",
        flush=True,
    )
    trainable_backbone = sum(p.numel() for p in baseline.parameters() if p.requires_grad)
    print(f"[CHECK] backbone_trainable_params={trainable_backbone}", flush=True)
    print(f"[CHECK] feature_keys={','.join(sorted(feats.keys()))}", flush=True)
    print(f"[CHECK] id_key_actual={id_key}", flush=True)
    _print_experiment_config(args, legacy_ce_on, legacy_kd_on, out_dir)
    if bool(args.save_epoch_metrics_csv):
        print(f"[METRICS] csv={metrics_csv}", flush=True)
    if bool(args.eval_sat_channel):
        print(
            f"[SAT-EVAL] enabled scenarios={','.join(args.eval_sat_scenario_list)} "
            f"on={args.eval_sat_on} max_batches={args.sat_eval_max_batches}",
            flush=True,
        )
    if args.dry_run:
        print("[DRY-RUN] Parsed arguments, built data/model context, and skipped optimization.", flush=True)
        return 0

    if str(args.mode).lower() == "baseline_eval":
        val_stats, named_stats = evaluate_post_model(
            baseline,
            data_ctx,
            device,
            max_batches=int(args.eval_max_batches),
        )
        test_stats, test_lines = summarize_post_stage_tests(named_stats, data_ctx, dataset=str(args.dataset))
        strict_udu = float(named_stats.get("test_unseen_day_unseen_rx", {}).get("tx_acc", float("nan")))
        row = {
            "epoch": 0,
            "val_source": float(val_stats.get("tx_acc", float("nan"))),
            "test_overall": float(test_stats.get("tx_acc", float("nan"))),
            "unseen_day_unseen_rx": strict_udu,
            "mode": "baseline_eval",
        }
        if bool(args.save_epoch_metrics_csv):
            _append_epoch_metrics_csv(metrics_csv, row)
        print(
            f"[BASELINE-EVAL] val_tx={val_stats['tx_acc']:.2f}% "
            f"test_tx={test_stats['tx_acc']:.2f}% udu={strict_udu:.2f}%",
            flush=True,
        )
        for line in test_lines:
            print(line, flush=True)
        return 0

    if int(args.epochs) <= 0 or bool(args.init_only_eval):
        wrapped = FJMPWrappedModel(
            baseline,
            proto_model,
            fusion,
            gate,
            id_key=id_key,
            strict_raw=bool(args.strict_raw),
            zdom_mode=str(args.zdom_mode),
            fusion_args=args,
        ).to(device)
        val_stats, named_stats = evaluate_post_model(
            wrapped,
            data_ctx,
            device,
            max_batches=int(args.eval_max_batches),
        )
        head_val_stats = {}
        head_named_stats = {}
        if bool(args.log_base_proto_fused):
            head_val_stats = evaluate_fjmp_loader_heads(
                baseline,
                proto_model,
                fusion,
                gate,
                data_ctx["val_loader"],
                device,
                id_key=id_key,
                strict_raw=bool(args.strict_raw),
                zdom_mode=str(args.zdom_mode),
                max_batches=int(args.eval_max_batches),
                fusion_args=args,
            )
            head_named_stats = evaluate_fjmp_named_heads(
                baseline,
                proto_model,
                fusion,
                gate,
                data_ctx["named_test_loaders"],
                device,
                id_key=id_key,
                strict_raw=bool(args.strict_raw),
                zdom_mode=str(args.zdom_mode),
                max_batches=int(args.eval_max_batches),
                fusion_args=args,
            )
        test_stats, test_lines = summarize_post_stage_tests(named_stats, data_ctx, dataset=str(args.dataset))
        strict_udu = float(named_stats.get("test_unseen_day_unseen_rx", {}).get("tx_acc", float("nan")))
        row = {
            "epoch": 0,
            "val_source": float(val_stats.get("tx_acc", float("nan"))),
            "test_overall": float(test_stats.get("tx_acc", float("nan"))),
            "unseen_day_unseen_rx": strict_udu,
            "mode": "init_only_eval" if bool(args.init_only_eval) else "epoch0_eval",
            "alpha": float(fusion.alpha().detach().cpu()),
            "beta": float(fusion.beta().detach().cpu()),
            "eta": float(fusion.eta().detach().cpu()),
        }
        if bool(args.save_epoch_metrics_csv):
            _append_epoch_metrics_csv(metrics_csv, row)
        print(
            f"[E000] val_tx={val_stats['tx_acc']:.2f}% test_tx={test_stats['tx_acc']:.2f}% "
            f"udu={strict_udu:.2f}% mode={row['mode']}",
            flush=True,
        )
        for line in test_lines:
            print(line, flush=True)
        return 0

    best_val = float("-inf")
    best_udu = float("-inf")
    best_udu_epoch = 0
    for epoch in range(1, int(args.epochs) + 1):
        v2_stage = None
        if _is_fjmp_v2(args):
            v2_stage = get_fjmp_v2_stage_weights(epoch, _fjmp_v2_loss_cfg(args))
            proto_model.set_rho_max(v2_stage.rho_max)
            proto_model.set_dynamic_rho_cap(bool(args.dynamic_rho_cap))
            proto_model.set_stage3_gate_only(v2_stage.train_gate_only)
            optimizer = _build_optimizer(args, proto_model, fusion)
        if (not _is_fjmp_v2(args)) and epoch == int(args.prototype_lr_decay_epoch) and float(args.prototype_lr_decay) > 0:
            for group in optimizer.param_groups:
                params = group.get("params", [])
                if any(param is proto_model.head.prototypes for param in params):
                    group["lr"] = float(group["lr"]) * float(args.prototype_lr_decay)
                    print(f"[SCHEDULE] prototype_lr_decay epoch={epoch} lr={group['lr']:.6g}", flush=True)
        proto_model.train()
        fusion.train()
        epoch_logs = []
        for batch in data_ctx["train_loader"]:
            x, y, extra = move_batch(batch, device)
            sgv_group = domain_from_extra(extra, data_ctx["domain_label_map"], device)
            with torch.no_grad():
                feats = forward_frozen_backbone(baseline, x, id_key=id_key, strict_raw=bool(args.strict_raw))
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=bool(args.amp and device.type == "cuda")):
                if _is_fjmp_v2(args):
                    fused_logits, aux = proto_model(feats["z_id_raw"], feats["base_logits"])
                    proto_out = {
                        "logits": aux["proto_logits"],
                        "proto_logits": aux["proto_logits"],
                        "proto_scores": aux["sim"],
                        "z_joint": aux["h"],
                    }
                    fused = {"logits": fused_logits, "rho": aux["rho"], "delta": aux["delta"]}
                    losses = compute_fjmp_v2_loss(
                        fused_logits,
                        feats["base_logits"],
                        y,
                        aux,
                        epoch=epoch,
                        cfg=_fjmp_v2_loss_cfg(args),
                    )
                    v2_metrics = compute_fjmp_v2_metrics(
                        prototypes=aux["prototypes"],
                        usage=losses["usage"],
                        rho=aux["rho"],
                        delta_logits=aux["delta"],
                        base_logits=feats["base_logits"],
                        fused_logits=fused_logits,
                        y=y,
                    )
                    losses.update(v2_metrics)
                    accept_proto = None
                else:
                    proto_out = proto_model(feats["z_id_raw"], feats["z_dom"], zdom_mode=str(args.zdom_mode))
                    gate_out = {}
                    accept_proto = None
                    if gate is not None:
                        ood_distance = compute_ood_distance(proto_out["z_joint"], proto_model.head.prototypes)
                        gate_out = gate(
                            proto_logits=proto_out["logits"],
                            proto_scores=proto_out["proto_scores"],
                            nearest_proto_score=proto_out["nearest_proto_score"],
                            ood_distance=ood_distance,
                        )
                        accept_proto = gate_out["accept_proto"]
                    fused = _fusion_forward(
                        fusion,
                        base_logits=feats["base_logits"],
                        proto_logits=proto_out["logits"],
                        accept_proto=accept_proto,
                        epoch=epoch,
                        args=args,
                    )
                    loss_out = dict(proto_out)
                    loss_out["proto_logits"] = proto_out["logits"]
                    loss_out["fused_logits"] = fused["logits"]
                    loss_out["safe_logits"] = fused.get("safe_logits", fused["logits"])
                    loss_out["head_logits"] = proto_out["logits"]
                    loss_out["logits"] = fused["logits"]
                    losses = compute_fjmp_loss(
                        loss_out,
                        y,
                        base_logits=feats["base_logits"],
                        ce_on=legacy_ce_on,
                        ce_proto_weight=float(args.ce_proto_weight),
                        ce_fused_weight=float(args.ce_fused_weight),
                        kd_on=legacy_kd_on,
                        lambda_kd=float(args.lambda_kd),
                        kd_temperature=float(args.kd_temperature),
                        lambda_dnh=float(args.lambda_dnh),
                        dnh_margin=float(args.dnh_margin),
                        lambda_margin_preserve=float(args.lambda_margin_preserve),
                        margin_preserve_delta=float(args.margin_preserve_delta),
                        lambda_sep=float(args.lambda_sep),
                        lambda_div=float(args.lambda_div),
                        lambda_usage=float(args.lambda_usage),
                        lambda_delta=float(args.lambda_delta),
                        lambda_cov=float(args.lambda_cov),
                        prototypes=proto_model.head.prototypes,
                        num_classes=int(args.num_classes),
                    )
                losses = dict(losses)
                losses["fjmp_loss"] = losses["loss"]
                sgv_weights = None
                if sgv_generator is not None:
                    strength = _sgv_strength_for_epoch(args, epoch)
                    x_sat, sat_params = sgv_generator(x, strength=strength)
                    with torch.no_grad():
                        feats_sat = forward_frozen_backbone(baseline, x_sat, id_key=id_key, strict_raw=bool(args.strict_raw))
                    proto_out_sat = proto_model(feats_sat["z_id_raw"], feats_sat["z_dom"], zdom_mode=str(args.zdom_mode))
                    fused_sat = _fusion_forward(
                        fusion,
                        base_logits=feats_sat["base_logits"],
                        proto_logits=proto_out_sat["logits"],
                        accept_proto=None,
                        epoch=epoch,
                        args=args,
                    )
                    sat_reliability = (
                        estimate_sat_reliability(
                            x_clean=x,
                            x_sat=x_sat,
                            base_clean=feats["base_logits"],
                            base_sat=feats_sat["base_logits"],
                            sat_params=sat_params,
                        )
                        if bool(args.use_sat_reliability)
                        else None
                    )
                    clean_view = {
                        "base_logits": feats["base_logits"].detach(),
                        "head_logits": proto_out["logits"],
                        "safe_logits": fused.get("safe_logits", fused["logits"]),
                        "gate": fused.get("gate", torch.zeros_like(y, dtype=torch.float32)),
                        "rho": fused.get("rho", torch.zeros_like(y, dtype=torch.float32)),
                        "delta": fused.get("delta", torch.zeros_like(fused["logits"])),
                        "proto_scores": proto_out["proto_scores"],
                    }
                    sat_view = {
                        "base_logits": feats_sat["base_logits"].detach(),
                        "head_logits": proto_out_sat["logits"],
                        "safe_logits": fused_sat.get("safe_logits", fused_sat["logits"]),
                        "gate": fused_sat.get("gate", torch.zeros_like(y, dtype=torch.float32)),
                        "rho": fused_sat.get("rho", torch.zeros_like(y, dtype=torch.float32)),
                        "delta": fused_sat.get("delta", torch.zeros_like(fused_sat["logits"])),
                        "proto_scores": proto_out_sat["proto_scores"],
                    }
                    sgv_losses = compute_sgv_bp_losses(
                        clean_view,
                        sat_view,
                        y,
                        epoch=epoch,
                        weights=_sgv_weight_overrides(args),
                        sat_reliability=sat_reliability,
                        group=sgv_group,
                    )
                    losses.update({f"sgv_{key}": value for key, value in sgv_losses.items() if key != "loss"})
                    losses["sgv_loss"] = sgv_losses["loss"]
                    losses["loss"] = losses["fjmp_loss"] + sgv_losses["loss"]
                    sgv_weights = _sgv_loss_weights_for_epoch(args, epoch)
                loss = losses["loss"]
                if not bool(torch.isfinite(loss).all()):
                    print(
                        f"[WARN] nonfinite_loss epoch={epoch} components={_format_nonfinite_losses(losses)}; "
                        "skipping batch before backward",
                        flush=True,
                    )
                    optimizer.zero_grad(set_to_none=True)
                    continue
            scaler.scale(loss).backward()
            if len(epoch_logs) == 0:
                backbone_grad_sq = 0.0
                for param in baseline.parameters():
                    if param.grad is not None:
                        backbone_grad_sq += float(param.grad.detach().float().pow(2).sum().cpu())
                print(f"[CHECK] backbone_grad_norm={backbone_grad_sq ** 0.5:.6f}", flush=True)
            if float(args.grad_clip) > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(list(proto_model.parameters()) + list(fusion.parameters()), float(args.grad_clip))
            if not _params_are_finite([proto_model, fusion]):
                print(f"[WARN] nonfinite_grad epoch={epoch}; skipping optimizer step", flush=True)
                optimizer.zero_grad(set_to_none=True)
                scaler.update()
                continue
            scaler.step(optimizer)
            scaler.update()
            train_log = {f"train/{k}": v for k, v in losses.items()}
            weighted_logs = {"fjmp_loss": losses["fjmp_loss"]}
            fjmp_weights = (
                _fjmp_v2_loss_component_weights(args, epoch)
                if _is_fjmp_v2(args)
                else _legacy_loss_component_weights(args, legacy_ce_on, legacy_kd_on)
            )
            weighted_logs.update(_weighted_loss_logs(losses, fjmp_weights))
            if sgv_weights is not None and "sgv_loss" in losses:
                weighted_logs["sgv_loss"] = losses["sgv_loss"]
                weighted_logs.update(_weighted_loss_logs(losses, sgv_weights, source_prefix="sgv_"))
            train_log.update({f"train_weighted/{k}": v for k, v in weighted_logs.items()})
            train_log.update(_scalar_logs("fusion", fused))
            if bool(args.log_base_proto_fused):
                rel = compute_relative_harm_metrics(
                    feats["base_logits"],
                    fused["logits"],
                    y,
                    proto_logits=proto_out["logits"],
                    accept_proto=accept_proto,
                    ood_reject=(~accept_proto) if accept_proto is not None else None,
                )
                train_log.update(_scalar_logs("train_relative", rel))
            epoch_logs.append(train_log)

        wrapped = FJMPWrappedModel(
            baseline,
            proto_model,
            fusion,
            gate,
            id_key=id_key,
            strict_raw=bool(args.strict_raw),
            zdom_mode=str(args.zdom_mode),
            fusion_args=args,
        ).to(device)
        val_stats, named_stats = evaluate_post_model(
            wrapped,
            data_ctx,
            device,
            max_batches=int(args.eval_max_batches),
        )
        test_stats, test_lines = summarize_post_stage_tests(named_stats, data_ctx, dataset=str(args.dataset))
        sat_test_stats = {}
        if bool(args.eval_sat_channel) and len(getattr(args, "eval_sat_scenario_list", [])) > 0:
            sat_test_stats = evaluate_sat_scenarios(
                wrapped,
                data_ctx["named_test_loaders"],
                device,
                domain_label_map=data_ctx["domain_label_map"],
                scenario_names=args.eval_sat_scenario_list,
                args=args,
                max_batches=resolve_sat_eval_max_batches(args.sat_eval_max_batches, args.eval_max_batches),
            )
        head_val_stats = {}
        head_named_stats = {}
        if bool(args.log_base_proto_fused):
            head_val_stats = evaluate_fjmp_loader_heads(
                baseline,
                proto_model,
                fusion,
                gate,
                data_ctx["val_loader"],
                device,
                id_key=id_key,
                strict_raw=bool(args.strict_raw),
                zdom_mode=str(args.zdom_mode),
                max_batches=int(args.eval_max_batches),
                fusion_args=args,
            )
            head_named_stats = evaluate_fjmp_named_heads(
                baseline,
                proto_model,
                fusion,
                gate,
                data_ctx["named_test_loaders"],
                device,
                id_key=id_key,
                strict_raw=bool(args.strict_raw),
                zdom_mode=str(args.zdom_mode),
                max_batches=int(args.eval_max_batches),
                fusion_args=args,
            )
        train_logs = mean_logs(epoch_logs)
        stats = {
            "train": train_logs,
            "val": val_stats,
            "test": test_stats,
            "named_test": named_stats,
            "head_val": head_val_stats,
            "head_named_test": head_named_stats,
            "sat_test_named": sat_test_stats,
        }
        strict_udu = float(named_stats.get("test_unseen_day_unseen_rx", {}).get("tx_acc", float("nan")))
        unseen_day_seen_rx = float(named_stats.get("test_unseen_day_seen_rx", {}).get("tx_acc", float("nan")))
        seen_day_unseen_rx = float(named_stats.get("test_seen_day_unseen_rx", {}).get("tx_acc", float("nan")))
        epoch_row = {
            "epoch": epoch,
            "val_source": float(val_stats.get("tx_acc", float("nan"))),
            "test_overall": float(test_stats.get("tx_acc", float("nan"))),
            "unseen_day_seen_rx": unseen_day_seen_rx,
            "seen_day_unseen_rx": seen_day_unseen_rx,
            "unseen_day_unseen_rx": strict_udu,
            "changed_pred_rate": train_logs.get("train_relative/changed_pred_rate", float("nan")),
            "rescue_rate": train_logs.get("train_relative/rescue_rate", float("nan")),
            "harm_rate": train_logs.get("train_relative/harm_rate", float("nan")),
            "net_gain_rate": train_logs.get("train_relative/net_gain_rate", float("nan")),
            "proto_accept_rate": train_logs.get("train_relative/proto_accept_rate", float("nan")),
            "ood_reject_rate": train_logs.get("train_relative/ood_reject_rate", float("nan")),
            "alpha": train_logs.get("fusion/alpha", float("nan")),
            "beta": train_logs.get("fusion/beta", float("nan")),
            "eta": train_logs.get("fusion/eta", float("nan")),
            "loss": train_logs.get("train/loss", float("nan")),
            "loss_ce_proto": train_logs.get("train/loss_ce_proto", float("nan")),
            "loss_ce_fused": train_logs.get("train/loss_ce_fused", float("nan")),
            "loss_kd_proto": train_logs.get("train/loss_kd_proto", float("nan")),
            "loss_kd_fused": train_logs.get("train/loss_kd_fused", float("nan")),
            "loss_dnh": train_logs.get("train/loss_dnh", float("nan")),
            "loss_margin_preserve": train_logs.get("train/loss_margin_preserve", float("nan")),
            "loss_ce_trim": train_logs.get("train/loss_ce_trim", float("nan")),
            "loss_kd_selective": train_logs.get("train/loss_kd_selective", float("nan")),
            "loss_logit_residual": train_logs.get("train/loss_logit_residual", float("nan")),
            "loss_gate": train_logs.get("train/loss_gate", float("nan")),
            "rho_mean": train_logs.get("train/rho_mean", float("nan")),
            "rho_p95": train_logs.get("train/rho_p95", float("nan")),
            "delta_ratio_mean": train_logs.get("train/delta_ratio_mean", float("nan")),
            "delta_ratio_p95": train_logs.get("train/delta_ratio_p95", float("nan")),
            "proto_pairwise_cos_mean": train_logs.get("train/proto_pairwise_cos_mean", float("nan")),
            "proto_pairwise_cos_max": train_logs.get("train/proto_pairwise_cos_max", float("nan")),
            "usage_entropy_mean": train_logs.get("train/usage_entropy_mean", float("nan")),
            "dead_proto_rate": train_logs.get("train/dead_proto_rate", float("nan")),
            "val_base_tx": head_val_stats.get("base_tx_acc", float("nan")),
            "val_proto_tx": head_val_stats.get("proto_tx_acc", float("nan")),
            "val_fused_tx": head_val_stats.get("fused_tx_acc", float("nan")),
            "udu_base_tx": head_named_stats.get("test_unseen_day_unseen_rx", {}).get("base_tx_acc", float("nan")),
            "udu_proto_tx": head_named_stats.get("test_unseen_day_unseen_rx", {}).get("proto_tx_acc", float("nan")),
            "udu_fused_tx": head_named_stats.get("test_unseen_day_unseen_rx", {}).get("fused_tx_acc", float("nan")),
        }
        for loss_key in FJMP_LOSS_KEYS + SGV_LOSS_KEYS:
            raw_val = _finite_float(train_logs.get(f"train/{loss_key}"))
            if raw_val is not None:
                epoch_row[f"raw_{loss_key}"] = raw_val
            weighted_val = _finite_float(train_logs.get(f"train_weighted/{loss_key}"))
            if weighted_val is not None:
                epoch_row[f"weighted_{loss_key}"] = weighted_val
        if bool(args.save_epoch_metrics_csv):
            _append_epoch_metrics_csv(metrics_csv, epoch_row)
        if _is_fjmp_v2(args):
            payload = {
                "proto_model": proto_model.state_dict(),
                "args": vars(args),
                "baseline_checkpoint": args.baseline_ckpt,
                "num_classes": int(args.num_classes),
                "id_dim": int(feats["z_id_raw"].size(1)),
                "num_prototypes": int(args.num_prototypes),
                "proto_dim": int(args.proto_dim),
                "feature_source": "z_id_raw only; z_dom disabled for prototype path",
                "fusion": "stopgrad(base_logits) + rho * clipped_residual",
                "best_stats": stats,
                "diagnostics": {"epoch": epoch, "epoch_row": epoch_row},
            }
        else:
            payload = build_fjmp_checkpoint_payload(
                proto_model,
                baseline_checkpoint=args.baseline_ckpt,
                args=vars(args),
                best_stats=stats,
                calibration_params={"fusion_mode": args.fusion_mode},
                diagnostics={"epoch": epoch, "epoch_row": epoch_row},
            )
        payload["fusion"] = fusion.state_dict()
        payload["epoch"] = epoch
        payload["split_info"] = data_ctx["split_info"]
        payload["baseline_args"] = vars(model_args)
        if bool(args.save_checkpoints):
            save_payload(out_dir / "latest_fjmp.pth", payload)
        if float(val_stats["tx_acc"]) > best_val:
            best_val = float(val_stats["tx_acc"])
            if bool(args.save_checkpoints):
                save_payload(out_dir / "best_val_fjmp.pth", payload)
        if math.isfinite(strict_udu) and strict_udu > best_udu:
            best_udu = strict_udu
            best_udu_epoch = epoch
            payload["diagnostic_test_selection"] = {
                "metric": "test_unseen_day_unseen_rx",
                "epoch": epoch,
                "value": best_udu,
                "note": "Diagnostic only: uses test split to localize losses, not for unbiased model selection.",
            }
            if bool(args.save_checkpoints):
                save_payload(out_dir / "best_udu_fjmp.pth", payload)
                best_udu_path = str(out_dir / "best_udu_fjmp.pth")
            else:
                best_udu_path = "disabled"
            print(
                f"[BEST-UDU] diagnostic_test_selection epoch={best_udu_epoch} "
                f"unseen_day_unseen_rx={best_udu:.2f}% path={best_udu_path}",
                flush=True,
            )
        print(
            f"[E{epoch:03d}] loss={train_logs.get('train/loss', float('nan')):.4f} "
            f"val_tx={val_stats['tx_acc']:.2f}% test_tx={test_stats['tx_acc']:.2f}% "
            f"udu={strict_udu:.2f}% best_val={best_val:.2f}% "
            f"harm={epoch_row['harm_rate']:.4f} rescue={epoch_row['rescue_rate']:.4f} "
            f"net={epoch_row['net_gain_rate']:.4f}",
            flush=True,
        )
        print(_format_loss_line("[LOSS-FJMP-RAW]", train_logs, FJMP_LOSS_KEYS, prefix="train/"), flush=True)
        print(_format_loss_line("[LOSS-FJMP-W]", train_logs, FJMP_LOSS_KEYS, prefix="train_weighted/"), flush=True)
        print(
            _format_weight_line(
                "[LOSS-FJMP-WEIGHT]",
                _fjmp_v2_loss_component_weights(args, epoch)
                if _is_fjmp_v2(args)
                else _legacy_loss_component_weights(args, legacy_ce_on, legacy_kd_on),
                FJMP_LOSS_KEYS,
            ),
            flush=True,
        )
        if bool(args.use_sgv):
            print(_format_loss_line("[LOSS-SGV-RAW]", train_logs, SGV_LOSS_KEYS, prefix="train/"), flush=True)
            print(_format_loss_line("[LOSS-SGV-W]", train_logs, SGV_LOSS_KEYS, prefix="train_weighted/"), flush=True)
            print(
                _format_weight_line(
                    "[LOSS-SGV-WEIGHT]",
                    _sgv_loss_weights_for_epoch(args, epoch),
                    SGV_LOSS_KEYS,
                    source_prefix="sgv_",
                ),
                flush=True,
            )
        print(_format_top_loss_contributors(train_logs), flush=True)
        print(
            f"[LOGIT] alpha={epoch_row['alpha']:.6f} beta={epoch_row['beta']:.6f} "
            f"eta={epoch_row['eta']:.6f}",
            flush=True,
        )
        print(
            f"[RELATIVE] changed_pred={epoch_row['changed_pred_rate']:.6f} "
            f"rescue={epoch_row['rescue_rate']:.6f} harm={epoch_row['harm_rate']:.6f} "
            f"net_gain={epoch_row['net_gain_rate']:.6f}",
            flush=True,
        )
        if bool(args.log_base_proto_fused):
            print(
                f"[SPLIT-BASE] val_source={epoch_row['val_base_tx']:.2f}% "
                f"unseen_day_unseen_rx={epoch_row['udu_base_tx']:.2f}%",
                flush=True,
            )
            print(
                f"[SPLIT-PROTO] val_source={epoch_row['val_proto_tx']:.2f}% "
                f"unseen_day_unseen_rx={epoch_row['udu_proto_tx']:.2f}%",
                flush=True,
            )
            print(
                f"[SPLIT-FUSED] val_source={epoch_row['val_fused_tx']:.2f}% "
                f"unseen_day_unseen_rx={epoch_row['udu_fused_tx']:.2f}%",
                flush=True,
            )
        for line in test_lines:
            print(line, flush=True)
        for line in format_sat_test_lines(sat_test_stats):
            print(line, flush=True)
    return 0


def main() -> int:
    args = build_arg_parser().parse_args()
    return train(args)


if __name__ == "__main__":
    raise SystemExit(main())
