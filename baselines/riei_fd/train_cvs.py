from __future__ import annotations

import argparse
from pathlib import Path
import sys

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import torch

from baselines.common.augmentation import add_sat_channel_view_args, build_sat_channel_view_augment, supervised_sat_view_batch
from baselines.common.consistency import (
    add_augmentation_consistency_args,
    build_augmentation_consistency_config,
    build_augmentation_consistency_step_fn,
)
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
from baselines.riei_fd.train import alternating_training_step
from baselines.riei_fd.model import RIEIModel


def build_riei_optimizer(name: str, parameters, *, lr: float, weight_decay: float = 0.0, momentum: float = 0.0):
    optimizer_name = str(name).strip().lower()
    params = list(parameters)
    if optimizer_name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    if optimizer_name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unsupported RIEI optimizer: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="RIEI-FD CVS-RFFI training")
    add_cvs_data_args(parser)
    add_cvs_sat_eval_args(parser)
    add_pseudo_label_args(parser)
    add_augmentation_consistency_args(parser)
    add_sat_channel_view_args(parser)
    parser.set_defaults(batch_size=64, eval_batch_size=256)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr_all", type=float, default=1e-4)
    parser.add_argument("--lr_fed", type=float, default=1e-4)
    parser.add_argument("--lambda_mi", type=float, default=1.2)
    parser.add_argument("--lambda_ie", type=float, default=1.2)
    parser.add_argument("--feature_dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--use_resnet_projection", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--fed_variant", type=str, default="imagenet1d", choices=["imagenet1d", "short_stem1d"])
    parser.add_argument("--mi_mode", type=str, default="cosine_abs", choices=["cosine_abs", "cosine_square", "cross_cov"])
    parser.add_argument("--ie_temperature", type=float, default=1.0)
    parser.add_argument("--ce_reduction", type=str, default="mean", choices=["mean", "sum"])
    parser.add_argument("--mi_reduction", type=str, default="mean", choices=["mean", "sum"])
    parser.add_argument("--ie_reduction", type=str, default="mean", choices=["mean", "sum"])
    parser.add_argument("--disentangle_steps", type=int, default=1)
    parser.add_argument("--weight_decay_all", type=float, default=0.0)
    parser.add_argument("--weight_decay_fed", type=float, default=0.0)
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "sgd"])
    parser.add_argument("--sgd_momentum", type=float, default=0.0)
    parser.add_argument("--grad_clip_norm", type=float, default=0.0)
    parser.add_argument("--lambda_feature_norm", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, default="baseline_runs/riei_fd")
    args = parser.parse_args()
    sat_scenarios = parse_and_validate_sat_scenarios(args) if args.eval_sat_channel else []

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    print(
        f"[START] method=riei_fd seed={args.seed} device={device} epochs={args.epochs} "
        f"sat_eval={int(bool(sat_scenarios))} output_dir={args.output_dir}",
        flush=True,
    )
    loaders = build_cvs_loaders(args, device)
    sat_view_aug = build_sat_channel_view_augment(args)
    if args.use_pseudo_labels and args.use_augmentation_consistency:
        raise ValueError("Pseudo-label and augmentation-consistency routes must run separately.")
    num_train_receivers = train_receiver_count(loaders.split.split_info, loaders.split.num_receivers)
    train_receivers = train_receiver_indices(loaders.split.split_info)
    receiver_mapping = {int(raw): int(idx) for idx, raw in enumerate(train_receivers)}
    print(
        f"[CONFIG-PAPER] method=riei_fd protocol={args.wisig_protocol} "
        f"lr_all={args.lr_all:.3e} lr_fed={args.lr_fed:.3e} "
        f"batch_size={args.batch_size} eval_batch_size={args.eval_batch_size} "
        f"feature_dim={args.feature_dim} lambda_mi={args.lambda_mi:.4f} "
        f"lambda_ie={args.lambda_ie:.4f} dropout={args.dropout:.4f} "
        f"paper_eval_last_n={args.paper_eval_last_n} paper_eval_name={args.paper_eval_name} "
        f"test_eval_interval={args.test_eval_interval} test_eval_start_epoch={args.test_eval_start_epoch} "
        f"test_on_val_improve={int(bool(args.test_on_val_improve))}",
        flush=True,
    )
    print(
        f"[CONFIG-OPT] method=riei_fd mi_mode={args.mi_mode} "
        f"ie_temperature={args.ie_temperature:.6g} ce_reduction={args.ce_reduction} "
        f"mi_reduction={args.mi_reduction} ie_reduction={args.ie_reduction} "
        f"disentangle_steps={args.disentangle_steps} "
        f"optimizer={args.optimizer} sgd_momentum={args.sgd_momentum:.6g} "
        f"weight_decay_all={args.weight_decay_all:.6g} weight_decay_fed={args.weight_decay_fed:.6g} "
        f"grad_clip_norm={args.grad_clip_norm:.6g} lambda_feature_norm={args.lambda_feature_norm:.6g} "
        f"use_resnet_projection={int(bool(args.use_resnet_projection))} fed_variant={args.fed_variant}",
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
    model = RIEIModel(
        loaders.split.num_classes,
        num_train_receivers,
        feature_dim=args.feature_dim,
        dropout=args.dropout,
        encoder_use_projection=args.use_resnet_projection,
        fed_variant=args.fed_variant,
    ).to(device)
    classifier_parameters = list(model.ec.parameters()) + list(model.rc.parameters())
    opt_all = build_riei_optimizer(
        args.optimizer,
        classifier_parameters,
        lr=args.lr_all,
        weight_decay=args.weight_decay_all,
        momentum=args.sgd_momentum,
    )
    opt_fed = build_riei_optimizer(
        args.optimizer,
        model.fed.parameters(),
        lr=args.lr_fed,
        weight_decay=args.weight_decay_fed,
        momentum=args.sgd_momentum,
    )

    def train_step(model, batch, device, epoch, step):
        batch = supervised_sat_view_batch(batch, device, sat_view_aug)
        batch = dict(batch)
        batch["receiver_target"] = compact_receiver_targets(batch["receiver"].to(device), loaders.split.split_info)
        return alternating_training_step(
            model,
            batch,
            opt_all,
            opt_fed,
            lambda_mi=args.lambda_mi,
            lambda_ie=args.lambda_ie,
            device=device,
            mi_mode=args.mi_mode,
            ie_temperature=args.ie_temperature,
            ce_reduction=args.ce_reduction,
            mi_reduction=args.mi_reduction,
            ie_reduction=args.ie_reduction,
            disentangle_steps=args.disentangle_steps,
            grad_clip_norm=args.grad_clip_norm,
            lambda_feature_norm=args.lambda_feature_norm,
        )

    def forward_eval(model, batch, device):
        return model(batch["iq"].to(device))

    pseudo_cfg = build_pseudo_label_config(args)
    consistency_cfg = build_augmentation_consistency_config(args)
    unlabeled_loader = loaders.unlabeled
    if (pseudo_cfg.enabled or consistency_cfg.enabled) and unlabeled_loader is None:
        raise ValueError("Unlabeled training requires --use_source_ssl_split.")
    pseudo_step = (
        build_pseudo_step_fn(cfg=pseudo_cfg, loader=unlabeled_loader, optimizer=opt_all, forward_fn=forward_eval)
        if pseudo_cfg.enabled
        else None
    )
    consistency_step = (
        build_augmentation_consistency_step_fn(
            cfg=consistency_cfg,
            loader=unlabeled_loader,
            optimizer=opt_all,
            sat_augment=sat_view_aug,
            forward_fn=forward_eval,
        )
        if consistency_cfg.enabled
        else None
    )
    unlabeled_step = pseudo_step if pseudo_step is not None else consistency_step
    print(
        f"[CONFIG-UNLABELED] route={'pseudo_label' if pseudo_cfg.enabled else ('augmentation_consistency' if consistency_cfg.enabled else 'none')} "
        f"labeled_ratio={args.wisig_labeled_ratio:.6g} unlabeled_ratio={args.wisig_unlabeled_ratio:.6g} "
        f"source_val_ratio={args.wisig_source_val_ratio:.6g}",
        flush=True,
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
        optimizer=opt_all,
        train_step_fn=train_step,
        pseudo_step_fn=unlabeled_step,
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
