from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn

from .losses import base_training_loss, incremental_calibration_loss
from .model import SixBlockConv1DEncoder, class_mean_weights
from .pseudo_targets import assign_base_targets, make_simplex_pseudo_targets, perturb_pseudo_targets


def load_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def run_dry_run(config: dict, *, device: str = "cpu") -> dict[str, float | int | str]:
    seed = int(config.get("seed", 1337))
    torch.manual_seed(seed)
    shot = int(config.get("shot", 1))
    if shot <= 0:
        raise ValueError("shot must be positive")
    feature_dim = int(config.get("embedding_dim", 16))
    num_targets = int(config.get("pseudo_targets", min(feature_dim + 1, 8)))
    base_classes = int(config.get("base_classes", min(3, num_targets)))
    if base_classes > num_targets:
        raise ValueError("base_classes must be <= pseudo_targets")

    dev = torch.device(device)
    encoder = SixBlockConv1DEncoder(input_channels=2, embedding_dim=feature_dim).to(dev)
    x = torch.randn(base_classes * max(shot, 2), 2, int(config.get("input_length", 256)), device=dev)
    labels = torch.arange(base_classes, device=dev).repeat_interleave(max(shot, 2))
    features = encoder(x)
    targets = make_simplex_pseudo_targets(num_targets=num_targets, feature_dim=feature_dim, device=dev)
    perturbed = perturb_pseudo_targets(targets, noise_range=float(config.get("noise_range", 0.01)), seed=seed)
    assigned = assign_base_targets(range(base_classes), targets)
    base_loss, _ = base_training_loss(features, labels, assigned, targets, perturbed)

    old_weights = torch.stack([assigned[index] for index in range(base_classes)], dim=0).to(dev)
    new_x = torch.randn(4, 2, int(config.get("input_length", 256)), device=dev)
    new_labels = torch.tensor([base_classes, base_classes, base_classes + 1, base_classes + 1], device=dev)
    new_features = encoder(new_x).detach()
    new_weights_init, new_class_ids = class_mean_weights(new_features, new_labels)
    new_weights = nn.Parameter(new_weights_init.detach().clone())
    optimizer = torch.optim.SGD([new_weights], lr=float(config.get("increment_lr", 0.08)))
    optimizer.zero_grad(set_to_none=True)
    inc_loss, inc_terms = incremental_calibration_loss(
        new_features,
        new_labels,
        old_weights,
        new_weights,
        new_class_ids=new_class_ids,
        prototypes=new_weights.detach(),
        top_k=int(config.get("top_k", 4)),
        margin=float(config.get("margin", 0.2)),
        tau_fuse=float(config.get("tau_fuse", 0.01)),
        lambda_align=float(config.get("lambda_align", 1.6)),
    )
    inc_loss.backward()
    grad_norm = float(new_weights.grad.detach().norm().cpu().item()) if new_weights.grad is not None else 0.0
    optimizer.step()
    encoder_grad = 0.0
    for parameter in encoder.parameters():
        if parameter.grad is not None:
            encoder_grad += float(parameter.grad.detach().abs().sum().cpu().item())
    return {
        "mode": "dry-run",
        "seed": seed,
        "base_loss": float(base_loss.detach().cpu().item()),
        "incremental_loss": float(inc_loss.detach().cpu().item()),
        "hard_count": int(inc_terms["hard_count"].detach().cpu().item()),
        "incremental_grad_norm": grad_norm,
        "encoder_grad_after_increment": encoder_grad,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Paper-faithful OSC-FSCIL SEI reproduction entrypoint")
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if not args.dry_run:
        raise SystemExit("formal training is not implemented in this scaffold; use --dry-run for wiring verification")
    print(json.dumps(run_dry_run(config, device=args.device), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
