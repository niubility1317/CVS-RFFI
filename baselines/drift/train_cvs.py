from __future__ import annotations

import argparse
from pathlib import Path
import sys

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import torch

from baselines.common.augmentation import add_sat_channel_view_args, build_sat_channel_view_augment, supervised_sat_view_batch
from baselines.common.cvs_data import add_cvs_data_args, build_cvs_loaders
from baselines.common.cvs_sat_eval import (
    add_cvs_sat_eval_args,
    evaluate_sat_scenarios,
    parse_and_validate_sat_scenarios,
)
from baselines.common.cvs_trainer import run_validation_gated_training
from baselines.common.io import set_seed
from baselines.common.paper_protocol import compact_receiver_targets, train_receiver_count, train_receiver_indices
from baselines.common.pseudo_labels import add_pseudo_label_args, build_pseudo_label_config, build_pseudo_step_fn
from baselines.drift.losses import ReceiverCenterEMA, compute_drift_loss
from baselines.drift.model import DRIFTModel
from baselines.drift.grl import dann_lambda


def add_drift_method_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--embedding_dim", type=int, default=512)
    parser.add_argument("--split_dim", type=int, default=256)
    parser.add_argument("--lambda_grl", type=float, default=1.0)
    parser.add_argument("--grl_coeff", type=float, default=1.0)
    parser.add_argument("--lambda_center", type=float, default=0.01)
    parser.add_argument("--center_mode", type=str, default="ema", choices=["batch", "ema"])
    parser.add_argument("--center_momentum", type=float, default=0.95)
    parser.add_argument("--lambda_mse", type=float, default=0.02)
    parser.add_argument("--normalize_features_for_mse", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--mse_reduction", type=str, default="sum", choices=["sum", "mean"])
    parser.add_argument("--mse_cap", type=float, default=0.0)
    parser.add_argument("--lambda_feature_norm", type=float, default=0.0)
    parser.add_argument("--feature_norm_target", type=float, default=0.0)
    parser.add_argument("--use_resnet_projection", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--domain_discriminator_layers", type=int, default=2, choices=[2, 3])
    parser.add_argument("--grl_schedule", type=str, default="constant", choices=["constant", "dann"])
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "adamw"])
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_clip_norm", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, default="baseline_runs/drift")
    return parser


def main() -> None:
    parser = argparse.ArgumentParser(description="DRIFT CVS-RFFI training")
    add_cvs_data_args(parser)
    add_cvs_sat_eval_args(parser)
    add_pseudo_label_args(parser)
    add_sat_channel_view_args(parser)
    parser.set_defaults(batch_size=64, eval_batch_size=256)
    add_drift_method_args(parser)
    args = parser.parse_args()
    sat_scenarios = parse_and_validate_sat_scenarios(args) if args.eval_sat_channel else []

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    print(
        f"[START] method=drift seed={args.seed} device={device} epochs={args.epochs} "
        f"sat_eval={int(bool(sat_scenarios))} output_dir={args.output_dir}",
        flush=True,
    )
    loaders = build_cvs_loaders(args, device)
    sat_view_aug = build_sat_channel_view_augment(args)
    num_train_receivers = train_receiver_count(loaders.split.split_info, loaders.split.num_receivers)
    train_receivers = train_receiver_indices(loaders.split.split_info)
    receiver_mapping = {int(raw): int(idx) for idx, raw in enumerate(train_receivers)}
    print(
        f"[CONFIG-PAPER] method=drift protocol={args.wisig_protocol} lr={args.lr:.3e} "
        f"batch_size={args.batch_size} eval_batch_size={args.eval_batch_size} "
        f"embedding_dim={args.embedding_dim} split_dim={args.split_dim} "
        f"lambda_grl={args.lambda_grl:.4f} grl_coeff={args.grl_coeff:.4f} "
        f"lambda_center={args.lambda_center:.4f} center_mode={args.center_mode} "
        f"center_momentum={args.center_momentum:.4f} "
        f"lambda_mse={args.lambda_mse:.4f} grl_schedule={args.grl_schedule} "
        f"normalize_features_for_mse={int(bool(args.normalize_features_for_mse))} "
        f"paper_eval_last_n={args.paper_eval_last_n} paper_eval_name={args.paper_eval_name} "
        f"test_eval_interval={args.test_eval_interval} test_eval_start_epoch={args.test_eval_start_epoch} "
        f"test_on_val_improve={int(bool(args.test_on_val_improve))}",
        flush=True,
    )
    print(
        f"[CONFIG-OPT] method=drift mse_reduction={args.mse_reduction} "
        f"mse_cap={args.mse_cap:.6g} lambda_feature_norm={args.lambda_feature_norm:.6g} "
        f"feature_norm_target={args.feature_norm_target:.6g} optimizer={args.optimizer} "
        f"weight_decay={args.weight_decay:.6g} grad_clip_norm={args.grad_clip_norm:.6g} "
        f"use_resnet_projection={int(bool(args.use_resnet_projection))} "
        f"domain_discriminator_layers={args.domain_discriminator_layers}",
        flush=True,
    )
    print(
        f"[CONFIG-DOMAINS] train_receiver_count={num_train_receivers} "
        f"train_days={loaders.split.split_info.get('train_days_label', [])} "
        f"test_days={loaders.split.split_info.get('test_days_label', [])} "
        f"train_receivers_raw={train_receivers} compact_receiver_mapping={receiver_mapping} "
        f"train_receivers_label={loaders.split.split_info.get('train_rxs_label', [])} "
        f"test_receivers_raw={loaders.split.split_info.get('test_rxs_idx', [])} "
        f"test_receivers_label={loaders.split.split_info.get('test_rxs_label', [])} "
        f"split_info={loaders.split.split_info}",
        flush=True,
    )
    model = DRIFTModel(
        loaders.split.num_classes,
        num_train_receivers,
        embedding_dim=args.embedding_dim,
        split_dim=args.split_dim,
        dropout=args.dropout,
        encoder_use_projection=args.use_resnet_projection,
        domain_discriminator_layers=args.domain_discriminator_layers,
    ).to(device)
    optimizer_cls = torch.optim.AdamW if args.optimizer == "adamw" else torch.optim.Adam
    optimizer = optimizer_cls(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = max(1, int(args.epochs) * max(1, len(loaders.train)))
    center_memory = None
    if args.center_mode == "ema":
        center_memory = ReceiverCenterEMA(
            num_receivers=num_train_receivers,
            feature_dim=model.embedding_dim - model.split_dim,
            momentum=args.center_momentum,
        ).to(device)

    def train_step(model, batch, device, epoch, step):
        batch = supervised_sat_view_batch(batch, device, sat_view_aug)
        progress = float(step) / float(total_steps)
        grl = dann_lambda(progress) if args.grl_schedule == "dann" else args.grl_coeff
        out = model(batch["iq"], grl_lambda=grl)
        receiver_target = compact_receiver_targets(batch["receiver"].to(device), loaders.split.split_info)
        losses = compute_drift_loss(
            out,
            batch["label"].to(device),
            receiver_target,
            lambda_grl=args.lambda_grl,
            lambda_center=args.lambda_center,
            lambda_mse=args.lambda_mse,
            normalize_features_for_mse=args.normalize_features_for_mse,
            mse_reduction=args.mse_reduction,
            mse_cap=args.mse_cap,
            lambda_feature_norm=args.lambda_feature_norm,
            feature_norm_target=args.feature_norm_target,
            center_mode=args.center_mode,
            center_memory=center_memory,
        )
        optimizer.zero_grad()
        losses["loss"].backward()
        if float(args.grad_clip_norm) > 0.0:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip_norm))
            losses["grad_norm"] = grad_norm.detach() if torch.is_tensor(grad_norm) else torch.tensor(float(grad_norm))
        optimizer.step()
        return {k: float(v.detach().cpu()) for k, v in losses.items()}

    def forward_eval(model, batch, device):
        return model(batch["iq"].to(device), grl_lambda=0.0)

    pseudo_cfg = build_pseudo_label_config(args)
    pseudo_loader = loaders.unlabeled if loaders.unlabeled is not None else loaders.train
    pseudo_step = (
        build_pseudo_step_fn(cfg=pseudo_cfg, loader=pseudo_loader, optimizer=optimizer, forward_fn=forward_eval)
        if pseudo_cfg.enabled
        else None
    )

    def extra_test(model, device):
        if not sat_scenarios:
            return {}
        sat_stats = evaluate_sat_scenarios(
            model,
            loaders.named_tests,
            device,
            scenario_names=sat_scenarios,
            args=args,
            forward_fn=forward_eval,
            max_batches=max(0, int(args.sat_eval_max_batches)),
        )
        return {"sat_channel": sat_stats}

    run_validation_gated_training(
        model=model,
        train_loader=loaders.train,
        val_loader=loaders.val,
        named_test_loaders=loaders.named_tests,
        device=device,
        epochs=args.epochs,
        optimizer=optimizer,
        train_step_fn=train_step,
        pseudo_step_fn=pseudo_step,
        forward_eval_fn=forward_eval,
        extra_test_fn=extra_test,
        paper_eval_last_n=args.paper_eval_last_n,
        paper_eval_name=args.paper_eval_name,
        test_eval_interval=args.test_eval_interval,
        test_eval_start_epoch=args.test_eval_start_epoch,
        test_on_val_improve=args.test_on_val_improve,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
