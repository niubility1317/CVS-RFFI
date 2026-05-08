from __future__ import annotations

import argparse

import torch

from baselines.common.cvs_data import add_cvs_data_args, build_cvs_loaders
from baselines.common.cvs_sat_eval import (
    add_cvs_sat_eval_args,
    evaluate_sat_scenarios,
    parse_and_validate_sat_scenarios,
)
from baselines.common.cvs_trainer import run_validation_gated_training
from baselines.common.io import set_seed
from baselines.riei.train import alternating_training_step
from baselines.riei.model import RIEIModel


def main() -> None:
    parser = argparse.ArgumentParser(description="RIEI CVS-RFFI training")
    add_cvs_data_args(parser)
    add_cvs_sat_eval_args(parser)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr_all", type=float, default=1e-4)
    parser.add_argument("--lr_fed", type=float, default=1e-4)
    parser.add_argument("--lambda_mi", type=float, default=0.1)
    parser.add_argument("--lambda_ie", type=float, default=0.1)
    parser.add_argument("--feature_dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, default="baseline_runs/riei")
    args = parser.parse_args()
    sat_scenarios = parse_and_validate_sat_scenarios(args) if args.eval_sat_channel else []

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    print(
        f"[START] method=riei seed={args.seed} device={device} epochs={args.epochs} "
        f"sat_eval={int(bool(sat_scenarios))} output_dir={args.output_dir}",
        flush=True,
    )
    loaders = build_cvs_loaders(args, device)
    model = RIEIModel(
        loaders.split.num_classes,
        loaders.split.num_receivers,
        feature_dim=args.feature_dim,
        dropout=args.dropout,
    ).to(device)
    opt_all = torch.optim.Adam(model.parameters(), lr=args.lr_all)
    opt_fed = torch.optim.Adam(model.fed.parameters(), lr=args.lr_fed)

    def train_step(model, batch, device, epoch, step):
        return alternating_training_step(
            model,
            batch,
            opt_all,
            opt_fed,
            lambda_mi=args.lambda_mi,
            lambda_ie=args.lambda_ie,
            device=device,
        )

    def forward_eval(model, batch, device):
        return model(batch["iq"].to(device))

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
        forward_eval_fn=forward_eval,
        extra_test_fn=extra_test,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
