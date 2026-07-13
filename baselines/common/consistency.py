from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

import torch
import torch.nn.functional as F

from baselines.common.cvs_trainer import logits_from_output
from baselines.common.pseudo_labels import CyclingLoader


@dataclass(frozen=True)
class AugmentationConsistencyConfig:
    enabled: bool = False
    start_epoch: int = 1
    temperature: float = 1.0
    weight: float = 1.0


@dataclass
class AugmentationConsistencyBatchResult:
    loss: torch.Tensor
    active: bool
    total: int = 0
    metrics: Dict[str, float] = field(default_factory=dict)


def add_augmentation_consistency_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--use_augmentation_consistency",
        action="store_true",
        help="Use soft clean-to-satellite-view consistency on source-unlabeled samples.",
    )
    parser.add_argument("--consistency_start_epoch", type=int, default=1)
    parser.add_argument("--consistency_temperature", type=float, default=1.0)
    parser.add_argument("--lambda_consistency", type=float, default=1.0)
    return parser


def build_augmentation_consistency_config(args) -> AugmentationConsistencyConfig:
    return AugmentationConsistencyConfig(
        enabled=bool(getattr(args, "use_augmentation_consistency", False)),
        start_epoch=int(getattr(args, "consistency_start_epoch", 1)),
        temperature=float(getattr(args, "consistency_temperature", 1.0)),
        weight=float(getattr(args, "lambda_consistency", 1.0)),
    )


def _forward_logits(model, batch, device, forward_fn):
    if forward_fn is None:
        output = model(batch["iq"].to(device))
    else:
        output = forward_fn(model, batch, device)
    return logits_from_output(output)


def compute_augmentation_consistency_loss(
    model,
    batch: Dict[str, Any],
    device,
    cfg: AugmentationConsistencyConfig,
    *,
    epoch: int,
    sat_augment,
    forward_fn: Optional[Callable[[Any, Dict[str, Any], torch.device], Any]] = None,
) -> AugmentationConsistencyBatchResult:
    if not cfg.enabled:
        return AugmentationConsistencyBatchResult(loss=torch.zeros((), device=device), active=False)
    if sat_augment is None:
        raise ValueError("augmentation consistency requires --use_sat_channel_view_aug")

    active = int(epoch) >= int(cfg.start_epoch)
    x_clean = batch["iq"].to(device)
    clean_batch = dict(batch)
    clean_batch["iq"] = x_clean
    with torch.no_grad():
        clean_logits = _forward_logits(model, clean_batch, device, forward_fn)

    strong_batch = dict(batch)
    strong_batch["iq"] = sat_augment(x_clean)
    strong_logits = _forward_logits(model, strong_batch, device, forward_fn)

    temperature = max(float(cfg.temperature), 1e-6)
    clean_prob = (clean_logits.detach() / temperature).softmax(dim=1)
    strong_log_prob = (strong_logits / temperature).log_softmax(dim=1)
    raw_loss = F.kl_div(strong_log_prob, clean_prob, reduction="batchmean") * (temperature**2)
    loss = float(cfg.weight) * raw_loss if active else strong_logits.sum() * 0.0
    agreement = clean_logits.argmax(dim=1).eq(strong_logits.argmax(dim=1)).float().mean()
    confidence = clean_prob.max(dim=1).values.mean()
    total = int(x_clean.size(0))
    metrics = {
        "consistency/enabled": 1.0,
        "consistency/active": 1.0 if active else 0.0,
        "consistency/start_epoch": float(cfg.start_epoch),
        "consistency/temperature": temperature,
        "consistency/weight": float(cfg.weight),
        "consistency/loss": float(loss.detach().cpu()),
        "consistency/agreement": float(agreement.detach().cpu()),
        "consistency/clean_confidence": float(confidence.detach().cpu()),
        "consistency/total": float(total),
    }
    return AugmentationConsistencyBatchResult(loss=loss, active=active, total=total, metrics=metrics)


def build_augmentation_consistency_step_fn(
    *,
    cfg: AugmentationConsistencyConfig,
    loader,
    optimizer,
    sat_augment,
    forward_fn: Optional[Callable[[Any, Dict[str, Any], torch.device], Any]] = None,
):
    cursor = CyclingLoader(loader)

    def consistency_step(model, device, epoch: int, step: int) -> Dict[str, float]:
        del step
        batch = cursor.next()
        if batch is None:
            return {"loss": 0.0, "consistency/active": 0.0}
        result = compute_augmentation_consistency_loss(
            model,
            batch,
            device,
            cfg,
            epoch=epoch,
            sat_augment=sat_augment,
            forward_fn=forward_fn,
        )
        if result.active:
            optimizer.zero_grad()
            result.loss.backward()
            optimizer.step()
        metrics = dict(result.metrics)
        metrics["loss"] = float(result.loss.detach().cpu())
        return metrics

    return consistency_step
