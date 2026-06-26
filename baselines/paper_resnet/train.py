from __future__ import annotations

import argparse
from pathlib import Path
import sys

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import torch
import torch.nn.functional as F
from torch import nn

from baselines.common.augmentation import add_sat_channel_view_args, build_sat_channel_view_augment, supervised_sat_view_batch
from baselines.common.cvs_data import add_cvs_data_args, build_cvs_loaders
from baselines.common.cvs_sat_eval import (
    add_cvs_sat_eval_args,
    evaluate_sat_scenarios,
    parse_and_validate_sat_scenarios,
)
from baselines.common.cvs_trainer import run_validation_gated_training
from baselines.common.grl import dann_lambda, gradient_reverse
from baselines.common.io import set_seed
from baselines.common.paper_protocol import compact_receiver_targets, train_receiver_count
from baselines.common.pseudo_labels import add_pseudo_label_args, build_pseudo_label_config, build_pseudo_step_fn
from baselines.common.resnet1d import MLPClassifier, ResNet1DEncoder


class PaperResNetModel(nn.Module):
    """Unified ResNet18-1D encoder with paper Table-I classifier heads."""

    def __init__(
        self,
        *,
        num_tx: int,
        num_rx: int,
        embedding_dim: int = 512,
        classifier_hidden_dim: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.encoder = ResNet1DEncoder(embedding_dim=int(embedding_dim), dropout=float(dropout))
        self.tx_classifier = MLPClassifier(int(embedding_dim), int(num_tx), int(classifier_hidden_dim), float(dropout))
        self.rx_classifier = MLPClassifier(int(embedding_dim), int(num_rx), int(classifier_hidden_dim), float(dropout))
        self.domain_discriminator = MLPClassifier(
            int(embedding_dim),
            int(num_rx),
            int(classifier_hidden_dim),
            float(dropout),
        )

    def forward(self, x: torch.Tensor, *, grl_lambda: float = 0.0) -> dict[str, torch.Tensor]:
        z = self.encoder(x)
        return {
            "z": z,
            "tx_logits": self.tx_classifier(z),
            "rx_logits": self.rx_classifier(z),
            "domain_logits": self.domain_discriminator(gradient_reverse(z, grl_lambda)),
        }


def add_paper_resnet_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--paper_method", type=str, default="erm", choices=["erm", "dann", "mtl"])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--embedding_dim", type=int, default=512)
    parser.add_argument("--classifier_hidden_dim", type=int, default=256)
    parser.add_argument("--lambda_domain", type=float, default=1.0)
    parser.add_argument("--grl_schedule", type=str, default="dann", choices=["constant", "dann"])
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, default="baseline_runs/paper_resnet")
    return parser


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified ResNet18-1D CVS/WiSig paper baselines")
    add_cvs_data_args(parser)
    add_cvs_sat_eval_args(parser)
    add_pseudo_label_args(parser)
    add_sat_channel_view_args(parser)
    parser.set_defaults(batch_size=64, eval_batch_size=256)
    add_paper_resnet_args(parser)
    args = parser.parse_args()
    sat_scenarios = parse_and_validate_sat_scenarios(args) if args.eval_sat_channel else []

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    print(
        f"[START] method=paper_resnet.{args.paper_method} seed={args.seed} device={device} "
        f"epochs={args.epochs} sat_eval={int(bool(sat_scenarios))} output_dir={args.output_dir}",
        flush=True,
    )

    loaders = build_cvs_loaders(args, device)
    sat_view_aug = build_sat_channel_view_augment(args)
    num_train_receivers = train_receiver_count(loaders.split.split_info, loaders.split.num_receivers)
    model = PaperResNetModel(
        num_tx=loaders.split.num_classes,
        num_rx=num_train_receivers,
        embedding_dim=args.embedding_dim,
        classifier_hidden_dim=args.classifier_hidden_dim,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    total_steps = max(1, int(args.epochs) * len(loaders.train))

    def train_step(model, batch, device, epoch, step):
        batch = supervised_sat_view_batch(batch, device, sat_view_aug)
        progress = float(step) / float(total_steps)
        grl = dann_lambda(progress) if args.grl_schedule == "dann" else args.lambda_domain
        out = model(batch["iq"], grl_lambda=grl)
        y = batch["label"].to(device)
        loss_tx = F.cross_entropy(out["tx_logits"], y)
        losses = {"loss_tx": loss_tx}
        loss = loss_tx
        if args.paper_method == "mtl":
            receiver_target = compact_receiver_targets(batch["receiver"].to(device), loaders.split.split_info)
            loss_rx = F.cross_entropy(out["rx_logits"], receiver_target)
            losses["loss_rx"] = loss_rx
            loss = loss + loss_rx
        elif args.paper_method == "dann":
            receiver_target = compact_receiver_targets(batch["receiver"].to(device), loaders.split.split_info)
            loss_domain = F.cross_entropy(out["domain_logits"], receiver_target)
            losses["loss_domain"] = loss_domain
            loss = loss + float(args.lambda_domain) * loss_domain
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses["loss"] = loss
        return {k: float(v.detach().cpu()) for k, v in losses.items()}

    def forward_eval(model, batch, device):
        return model(batch["iq"].to(device), grl_lambda=0.0)

    pseudo_cfg = build_pseudo_label_config(args)
    pseudo_step = (
        build_pseudo_step_fn(cfg=pseudo_cfg, loader=loaders.train, optimizer=optimizer, forward_fn=forward_eval)
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
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
