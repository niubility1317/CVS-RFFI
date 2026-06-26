from __future__ import annotations

import argparse
from pathlib import Path
import sys

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import torch

from baselines.common.augmentation import (
    OnlineRFChannelAugment,
    add_sat_channel_view_args,
    build_sat_channel_view_augment,
    supervised_sat_view_batch,
)
from baselines.common.cvs_data import add_cvs_data_args, build_cvs_loaders
from baselines.common.cvs_sat_eval import (
    add_cvs_sat_eval_args,
    evaluate_sat_scenarios,
    parse_and_validate_sat_scenarios,
)
from baselines.common.cvs_trainer import (
    ValidationLossPlateauController,
    cross_entropy_val_loss,
    run_validation_gated_training,
)
from baselines.common.io import set_seed
from baselines.common.pseudo_labels import add_pseudo_label_args, build_pseudo_label_config, build_pseudo_step_fn
from baselines.ra_collab.cvs_collaborative import evaluate_collaborative_tx
from baselines.ra_collab.losses import ra_collab_adversarial_loss
from baselines.ra_collab.model import RACollabRFFI
from baselines.ra_collab.spectrogram import SpectrogramTransform


def main() -> None:
    parser = argparse.ArgumentParser(description="RA-Collab RFFI CVS training")
    add_cvs_data_args(parser)
    add_cvs_sat_eval_args(parser)
    add_pseudo_label_args(parser)
    add_sat_channel_view_args(parser)
    parser.set_defaults(batch_size=64, eval_batch_size=256)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--lr_reduce_factor", type=float, default=0.2)
    parser.add_argument("--lr_patience", type=int, default=10)
    parser.add_argument("--early_stop_patience", type=int, default=20)
    parser.add_argument("--feature_dim", type=int, default=256)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--grl_lambda", type=float, default=1.0)
    parser.add_argument("--rx_weight", type=float, default=1.0)
    parser.add_argument("--sample_rate", type=float, default=1_000_000.0)
    parser.add_argument("--n_fft", type=int, default=64)
    parser.add_argument("--hop_length", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--collaborative_fusion", type=str, default="soft", choices=["soft", "adaptive"])
    parser.add_argument("--output_dir", type=str, default="baseline_runs/ra_collab")
    args = parser.parse_args()
    sat_scenarios = parse_and_validate_sat_scenarios(args) if args.eval_sat_channel else []

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    print(
        f"[START] method=ra_collab seed={args.seed} device={device} epochs={args.epochs} "
        f"sat_eval={int(bool(sat_scenarios))} output_dir={args.output_dir}",
        flush=True,
    )
    spec = SpectrogramTransform(n_fft=args.n_fft, hop_length=args.hop_length, win_length=args.n_fft)
    aug = OnlineRFChannelAugment(sample_rate=args.sample_rate, max_taps=4, p=1.0)
    loaders = build_cvs_loaders(args, device)
    sat_view_aug = build_sat_channel_view_augment(args)
    model = RACollabRFFI(
        loaders.split.num_classes,
        loaders.split.num_receivers,
        feature_dim=args.feature_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    plateau = ValidationLossPlateauController(
        optimizer,
        lr_factor=args.lr_reduce_factor,
        lr_patience=args.lr_patience,
        early_stop_patience=args.early_stop_patience,
    )

    def train_step(model, batch, device, epoch, step):
        batch = supervised_sat_view_batch(batch, device, sat_view_aug)
        x = spec(aug(batch["iq"]))
        out = model(x, grl_lambda=args.grl_lambda)
        losses = ra_collab_adversarial_loss(
            out,
            batch["label"].to(device),
            batch["receiver"].to(device),
            rx_weight=args.rx_weight,
        )
        optimizer.zero_grad()
        losses["loss"].backward()
        optimizer.step()
        return {k: float(v.detach().cpu()) for k, v in losses.items()}

    def forward_eval(model, batch, device):
        return model(spec(batch["iq"].to(device)), grl_lambda=0.0, return_rx=False)

    pseudo_cfg = build_pseudo_label_config(args)
    pseudo_step = (
        build_pseudo_step_fn(cfg=pseudo_cfg, loader=loaders.train, optimizer=optimizer, forward_fn=forward_eval)
        if pseudo_cfg.enabled
        else None
    )

    def collaborative_eval(model, loader, device):
        return evaluate_collaborative_tx(
            model,
            loader,
            device,
            forward_fn=forward_eval,
            fusion=args.collaborative_fusion,
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
        plateau_controller=plateau,
        forward_eval_fn=forward_eval,
        val_loss_fn=cross_entropy_val_loss,
        best_metric="loss",
        test_evaluate_fn=collaborative_eval,
        extra_test_fn=extra_test,
        paper_eval_last_n=args.paper_eval_last_n,
        paper_eval_name=args.paper_eval_name,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
