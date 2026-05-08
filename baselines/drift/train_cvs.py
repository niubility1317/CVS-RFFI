from __future__ import annotations

import argparse

import torch

from baselines.common.cvs_data import add_cvs_data_args, build_cvs_loaders
from baselines.common.cvs_sat_eval import (
    add_cvs_sat_eval_args,
    evaluate_sat_scenarios,
    format_sat_test_lines,
    parse_and_validate_sat_scenarios,
)
from baselines.common.cvs_trainer import run_validation_gated_training
from baselines.common.grl import dann_lambda
from baselines.common.io import set_seed
from baselines.drift.losses import compute_drift_loss
from baselines.drift.model import DRIFTModel


def main() -> None:
    parser = argparse.ArgumentParser(description="DRIFT CVS-RFFI training")
    add_cvs_data_args(parser)
    add_cvs_sat_eval_args(parser)
    parser.set_defaults(batch_size=64, eval_batch_size=256)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--embedding_dim", type=int, default=512)
    parser.add_argument("--split_dim", type=int, default=256)
    parser.add_argument("--lambda_grl", type=float, default=1.0)
    parser.add_argument("--lambda_center", type=float, default=0.01)
    parser.add_argument("--lambda_mse", type=float, default=0.02)
    parser.add_argument("--grl_schedule", type=str, default="constant", choices=["constant", "dann"])
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, default="baseline_runs/drift")
    args = parser.parse_args()
    sat_scenarios = parse_and_validate_sat_scenarios(args) if args.eval_sat_channel else []

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    loaders = build_cvs_loaders(args, device)
    model = DRIFTModel(
        loaders.split.num_classes,
        loaders.split.num_receivers,
        embedding_dim=args.embedding_dim,
        split_dim=args.split_dim,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    total_steps = max(1, int(args.epochs) * len(loaders.train))

    def train_step(model, batch, device, epoch, step):
        progress = float(step) / float(total_steps)
        grl = dann_lambda(progress) if args.grl_schedule == "dann" else args.lambda_grl
        out = model(batch["iq"].to(device), grl_lambda=grl)
        losses = compute_drift_loss(
            out,
            batch["label"].to(device),
            batch["receiver"].to(device),
            lambda_grl=args.lambda_grl,
            lambda_center=args.lambda_center,
            lambda_mse=args.lambda_mse,
        )
        optimizer.zero_grad()
        losses["loss"].backward()
        optimizer.step()
        return {k: float(v.detach().cpu()) for k, v in losses.items()}

    def forward_eval(model, batch, device):
        return model(batch["iq"].to(device), grl_lambda=0.0)

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
        train_step_fn=train_step,
        forward_eval_fn=forward_eval,
        extra_test_fn=extra_test,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
