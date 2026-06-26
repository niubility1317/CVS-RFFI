from __future__ import annotations

import argparse
from pathlib import Path
import sys

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import torch
import torch.nn.functional as F

from baselines.common.augmentation import add_sat_channel_view_args, build_sat_channel_view_augment, supervised_sat_view_batch
from baselines.common.cvs_data import add_cvs_data_args, build_cvs_loaders
from baselines.common.cvs_sat_eval import (
    add_cvs_sat_eval_args,
    evaluate_sat_scenarios,
    parse_and_validate_sat_scenarios,
)
from baselines.common.cvs_trainer import run_validation_gated_training
from baselines.common.io import set_seed
from baselines.common.pseudo_labels import add_pseudo_label_args, build_pseudo_label_config, build_pseudo_step_fn
from baselines.cvcnn_ce.model import BasicCVCNN, SincCVCNN


def main() -> None:
    parser = argparse.ArgumentParser(description="CVCNN-CE CVS-RFFI baseline")
    add_cvs_data_args(parser)
    add_cvs_sat_eval_args(parser)
    add_pseudo_label_args(parser)
    add_sat_channel_view_args(parser)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lr_min", type=float, default=1e-6)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--base_channels", type=int, default=32)
    parser.add_argument("--embedding_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--front_end", type=str, default="conv", choices=["conv", "sinc"])
    parser.add_argument("--sinc_kernel_size", type=int, default=79)
    parser.add_argument("--sample_rate_hz", type=float, default=25e6)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, default="baseline_runs/cvcnn_ce")
    args = parser.parse_args()
    if args.front_end == "sinc" and args.output_dir == "baseline_runs/cvcnn_ce":
        args.output_dir = "baseline_runs/cvcnn_sinc_ce"
    sat_scenarios = parse_and_validate_sat_scenarios(args) if args.eval_sat_channel else []

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    print(
        f"[START] method=cvcnn_ce front_end={args.front_end} seed={args.seed} device={device} epochs={args.epochs} "
        f"sat_eval={int(bool(sat_scenarios))} output_dir={args.output_dir}",
        flush=True,
    )
    loaders = build_cvs_loaders(args, device)
    sat_view_aug = build_sat_channel_view_augment(args)
    model_cls = SincCVCNN if args.front_end == "sinc" else BasicCVCNN
    model_kwargs = {
        "num_classes": loaders.split.num_classes,
        "input_len": loaders.split.input_len,
        "base_channels": args.base_channels,
        "embedding_dim": args.embedding_dim,
        "dropout": args.dropout,
    }
    if args.front_end == "sinc":
        model_kwargs.update({
            "sinc_kernel_size": args.sinc_kernel_size,
            "sample_rate_hz": args.sample_rate_hz,
        })
    model = model_cls(**model_kwargs).to(device)
    total_params = sum(int(p.numel()) for p in model.parameters())
    trainable_params = sum(int(p.numel()) for p in model.parameters() if p.requires_grad)
    print(f"[MODEL] {model.__class__.__name__} params={total_params:,} trainable={trainable_params:,}", flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs), eta_min=args.lr_min)
    pseudo_cfg = build_pseudo_label_config(args)
    pseudo_step = (
        build_pseudo_step_fn(cfg=pseudo_cfg, loader=loaders.train, optimizer=optimizer)
        if pseudo_cfg.enabled
        else None
    )

    def train_step(model, batch, device, epoch, step):
        batch = supervised_sat_view_batch(batch, device, sat_view_aug)
        x = batch["iq"]
        y = batch["label"].to(device)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return {"loss": float(loss.detach().cpu())}

    def extra_test(model, device):
        if not sat_scenarios:
            return {}
        sat_stats = evaluate_sat_scenarios(
            model,
            loaders.named_tests,
            device,
            scenario_names=sat_scenarios,
            args=args,
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
        scheduler=scheduler,
        train_step_fn=train_step,
        pseudo_step_fn=pseudo_step,
        extra_test_fn=extra_test,
        paper_eval_last_n=args.paper_eval_last_n,
        paper_eval_name=args.paper_eval_name,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
