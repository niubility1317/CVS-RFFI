from __future__ import annotations

import argparse
import os

import torch
from torch.utils.data import DataLoader

from baselines.common.augmentation import OnlineRFChannelAugment
from baselines.common.cvs_data import add_cvs_data_args, build_cvs_loaders, build_cvs_split
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
from baselines.common.io import ensure_dir, set_seed
from baselines.tifs2025_channel_receiver_rffi.data import PretrainDataset, SiamesePairDataset, SpectrogramTransform
from baselines.tifs2025_channel_receiver_rffi.losses import NTXentLoss, siamese_contrastive_ce_loss
from baselines.tifs2025_channel_receiver_rffi.models import ProjectionHead, ResNetRFF, SiameseRFF


def main() -> None:
    parser = argparse.ArgumentParser(description="TIFS2025 channel/receiver robust CVS training")
    add_cvs_data_args(parser)
    add_cvs_sat_eval_args(parser)
    parser.set_defaults(batch_size=32, eval_batch_size=256)
    parser.add_argument("--pretrain_epochs", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--pretrain_lr", type=float, default=3e-4)
    parser.add_argument("--lr_reduce_factor", type=float, default=0.5)
    parser.add_argument("--lr_patience", type=int, default=10)
    parser.add_argument("--early_stop_patience", type=int, default=30)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--feature_dim", type=int, default=256)
    parser.add_argument("--projection_dim", type=int, default=128)
    parser.add_argument("--ce_weight", type=float, default=1.0)
    parser.add_argument("--contrastive_weight", type=float, default=1.0)
    parser.add_argument("--sample_rate", type=float, default=1_000_000.0)
    parser.add_argument("--n_fft", type=int, default=128)
    parser.add_argument("--hop_length", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, default="baseline_runs/tifs2025_channel_receiver_rffi")
    args = parser.parse_args()
    sat_scenarios = parse_and_validate_sat_scenarios(args) if args.eval_sat_channel else []

    set_seed(args.seed)
    ensure_dir(args.output_dir)
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    print(
        f"[START] method=tifs2025 seed={args.seed} device={device} "
        f"pretrain_epochs={args.pretrain_epochs} epochs={args.epochs} "
        f"sat_eval={int(bool(sat_scenarios))} output_dir={args.output_dir}",
        flush=True,
    )
    spec = SpectrogramTransform(n_fft=args.n_fft, hop_length=args.hop_length, win_length=args.n_fft)
    augment = OnlineRFChannelAugment(sample_rate=args.sample_rate, max_taps=8, p=1.0)
    raw_split = build_cvs_split(args)
    eval_loaders = build_cvs_loaders(args, device, transform_train=spec, transform_eval=spec)
    raw_eval_loaders = build_cvs_loaders(args, device) if sat_scenarios else None

    backbone = ResNetRFF(num_classes=raw_split.num_classes, feature_dim=args.feature_dim).to(device)
    projector = ProjectionHead(in_dim=args.feature_dim, projection_dim=args.projection_dim).to(device)
    pretrain_ds = PretrainDataset(raw_split.train, augment=augment, spec_transform=spec)
    pretrain_loader = DataLoader(pretrain_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
    pretrain_opt = torch.optim.Adam(list(backbone.parameters()) + list(projector.parameters()), lr=args.pretrain_lr)
    ntxent = NTXentLoss(temperature=args.temperature)

    for epoch in range(1, int(args.pretrain_epochs) + 1):
        backbone.train()
        projector.train()
        loss_sum = 0.0
        n_batches = 0
        for v1, v2 in pretrain_loader:
            x = torch.cat([v1, v2], dim=0).to(device)
            _, z = backbone(x)
            loss = ntxent(projector(z))
            pretrain_opt.zero_grad()
            loss.backward()
            pretrain_opt.step()
            loss_sum += float(loss.detach().cpu())
            n_batches += 1
        print(f"[PRETRAIN {epoch:03d}/{int(args.pretrain_epochs):03d}] ntxent={loss_sum / max(1, n_batches):.4f}", flush=True)
    torch.save(
        {"model": backbone.state_dict(), "projector": projector.state_dict(), "epoch": int(args.pretrain_epochs)},
        os.path.join(args.output_dir, "pretrain_last.pt"),
    )

    siamese = SiameseRFF(backbone).to(device)
    pair_ds = SiamesePairDataset(raw_split.train, augment=augment, spec_transform=spec, seed=args.seed)
    print(
        f"[TIFS-PAIRS] mode={pair_ds.pair_mode} aligned_pairs={pair_ds.aligned_pair_count} total_pairs={len(pair_ds)}",
        flush=True,
    )
    pair_loader = DataLoader(pair_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
    opt = torch.optim.Adam(siamese.parameters(), lr=args.lr)
    plateau = ValidationLossPlateauController(
        opt,
        lr_factor=args.lr_reduce_factor,
        lr_patience=args.lr_patience,
        early_stop_patience=args.early_stop_patience,
    )

    def train_step(model, batch, device, epoch, step):
        x1, x2, y = batch
        losses = siamese_contrastive_ce_loss(
            *siamese(x1.to(device), x2.to(device)),
            y.to(device),
            ce_weight=args.ce_weight,
            contrastive_weight=args.contrastive_weight,
            temperature=args.temperature,
        )
        opt.zero_grad()
        losses["loss"].backward()
        opt.step()
        return {k: float(v.detach().cpu()) for k, v in losses.items()}

    def forward_eval(model, batch, device):
        return model(batch["iq"].to(device))

    def extra_test(model, device):
        if not sat_scenarios or raw_eval_loaders is None:
            return {}
        sat_stats = evaluate_sat_scenarios(
            model,
            raw_eval_loaders.named_tests,
            device,
            scenario_names=sat_scenarios,
            args=args,
            forward_fn=forward_eval,
            input_transform=spec,
            max_batches=max(0, int(args.sat_eval_max_batches)),
        )
        return {"sat_channel": sat_stats}

    run_validation_gated_training(
        model=backbone,
        train_loader=pair_loader,
        val_loader=eval_loaders.val,
        named_test_loaders=eval_loaders.named_tests,
        device=device,
        epochs=args.epochs,
        optimizer=opt,
        train_step_fn=train_step,
        plateau_controller=plateau,
        forward_eval_fn=forward_eval,
        val_loss_fn=cross_entropy_val_loss,
        best_metric="loss",
        extra_test_fn=extra_test,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
