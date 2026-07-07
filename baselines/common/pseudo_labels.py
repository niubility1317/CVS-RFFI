from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

import torch
import torch.nn.functional as F

from baselines.common.cvs_trainer import logits_from_output


@dataclass(frozen=True)
class PseudoLabelConfig:
    enabled: bool = False
    start_epoch: int = 150
    threshold: float = 0.90
    margin: float = 0.0
    weight: float = 1.0


@dataclass
class PseudoLabelBatchResult:
    loss: torch.Tensor
    active: bool
    total: int = 0
    selected: int = 0
    metrics: Dict[str, float] = field(default_factory=dict)


class CyclingLoader:
    def __init__(self, loader):
        self.loader = loader
        self._iter = None

    def next(self):
        if self.loader is None:
            return None
        if self._iter is None:
            self._iter = iter(self.loader)
        try:
            return next(self._iter)
        except StopIteration:
            self._iter = iter(self.loader)
            try:
                return next(self._iter)
            except StopIteration:
                return None


def add_pseudo_label_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--use_pseudo_labels", action="store_true", help="Enable opt-in pseudo-label self-training.")
    parser.add_argument("--pseudo_start_epoch", type=int, default=150)
    parser.add_argument("--pseudo_threshold", type=float, default=0.90)
    parser.add_argument("--pseudo_margin", type=float, default=0.0)
    parser.add_argument("--lambda_pseudo", type=float, default=1.0)
    return parser


def build_pseudo_label_config(args) -> PseudoLabelConfig:
    return PseudoLabelConfig(
        enabled=bool(getattr(args, "use_pseudo_labels", False)),
        start_epoch=int(getattr(args, "pseudo_start_epoch", 150)),
        threshold=float(getattr(args, "pseudo_threshold", 0.90)),
        margin=float(getattr(args, "pseudo_margin", 0.0)),
        weight=float(getattr(args, "lambda_pseudo", 1.0)),
    )


def _model_zero(model, device) -> torch.Tensor:
    return torch.zeros((), device=device)


def _move_batch_tensor(value: Any, device):
    return value.to(device) if torch.is_tensor(value) else value


def compute_pseudo_label_loss(
    model,
    batch: Dict[str, Any],
    device,
    cfg: PseudoLabelConfig,
    *,
    epoch: int,
    forward_fn: Optional[Callable[[Any, Dict[str, Any], torch.device], Any]] = None,
) -> PseudoLabelBatchResult:
    if not cfg.enabled or int(epoch) < int(cfg.start_epoch):
        return PseudoLabelBatchResult(loss=_model_zero(model, device), active=False)

    if forward_fn is None:
        output = model(batch["iq"].to(device))
    else:
        output = forward_fn(model, batch, device)
    logits = logits_from_output(output)
    prob = logits.softmax(dim=1)
    conf, pseudo = prob.max(dim=1)
    if prob.size(1) > 1:
        top2 = prob.topk(k=2, dim=1).values
        margin = top2[:, 0] - top2[:, 1]
    else:
        margin = torch.ones_like(conf)
    mask = (conf >= float(cfg.threshold)) & (margin >= float(cfg.margin))
    selected = int(mask.sum().detach().item())
    total = int(mask.numel())
    if selected > 0:
        raw_loss = F.cross_entropy(logits[mask], pseudo.detach()[mask])
    else:
        raw_loss = logits.sum() * 0.0
    loss = float(cfg.weight) * raw_loss

    metrics: Dict[str, float] = {
        "pseudo/active": 1.0,
        "pseudo/loss": float(loss.detach().cpu()),
        "pseudo/coverage": float(mask.float().mean().detach().cpu()) if total else 0.0,
        "pseudo/confidence": float(conf[mask].mean().detach().cpu()) if selected else 0.0,
        "pseudo/margin": float(margin[mask].mean().detach().cpu()) if selected else 0.0,
        "pseudo/total": float(total),
        "pseudo/selected": float(selected),
    }
    label_key = "label"
    if "true_label" in batch and torch.is_tensor(batch["true_label"]):
        true_y = _move_batch_tensor(batch["true_label"], device).long()
        if bool((true_y >= 0).any().detach().item()):
            label_key = "true_label"
    if label_key in batch and torch.is_tensor(batch[label_key]):
        y = _move_batch_tensor(batch[label_key], device).long()
        metrics["pseudo/precision"] = float(pseudo[mask].eq(y[mask]).float().mean().detach().cpu()) if selected else 0.0
    return PseudoLabelBatchResult(loss=loss, active=True, total=total, selected=selected, metrics=metrics)


def build_pseudo_step_fn(
    *,
    cfg: PseudoLabelConfig,
    loader,
    optimizer,
    forward_fn: Optional[Callable[[Any, Dict[str, Any], torch.device], Any]] = None,
):
    cursor = CyclingLoader(loader)

    def pseudo_step(model, device, epoch: int, step: int) -> Dict[str, float]:
        del step
        if not cfg.enabled:
            return {"loss": 0.0, "pseudo/active": 0.0}
        batch = cursor.next()
        if batch is None:
            return {"loss": 0.0, "pseudo/active": 0.0}
        result = compute_pseudo_label_loss(model, batch, device, cfg, epoch=epoch, forward_fn=forward_fn)
        if result.active:
            optimizer.zero_grad()
            result.loss.backward()
            optimizer.step()
        metrics = dict(result.metrics)
        metrics["loss"] = float(result.loss.detach().cpu())
        if not result.active:
            metrics["pseudo/active"] = 0.0
        return metrics

    return pseudo_step
