from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F

from baselines.common.cvs_data import add_cvs_data_args, build_cvs_loaders
from baselines.common.cvs_sat_eval import (
    add_cvs_sat_eval_args,
    evaluate_sat_scenarios,
    format_sat_test_lines,
    parse_and_validate_sat_scenarios,
)
from baselines.common.cvs_trainer import run_validation_gated_training
from baselines.common.io import set_seed
from baselines.cvcnn.model import BasicCVCNN


def main() -> None:
    parser = argparse.ArgumentParser(description="Basic CVCNN CVS-RFFI baseline")
    add_cvs_data_args(parser)
    add_cvs_sat_eval_args(parser)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lr_min", type=float, default=1e-6)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--base_channels", type=int, default=32)
    parser.add_argument("--embedding_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, default="baseline_runs/cvcnn")
    args = parser.parse_args()
    sat_scenarios = parse_and_validate_sat_scenarios(args) if args.eval_sat_channel else []

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    loaders = build_cvs_loaders(args, device)
    model = BasicCVCNN(
        num_classes=loaders.split.num_classes,
        input_len=loaders.split.input_len,
        base_channels=args.base_channels,
        embedding_dim=args.embedding_dim,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs), eta_min=args.lr_min)

    def train_step(model, batch, device, epoch, step):
        x = batch["iq"].to(device)
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
        for line in format_sat_test_lines(sat_stats):
            print(line, flush=True)
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
        extra_test_fn=extra_test,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
