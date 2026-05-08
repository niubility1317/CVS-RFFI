from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Optional

import torch.nn.functional as F
import torch


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def save_json(obj: Dict[str, Any], path: str) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def logits_from_output(output: Any, preferred_key: str = "tx_logits") -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    if isinstance(output, dict):
        for key in (preferred_key, "emitter_logits", "logits"):
            if key in output:
                return output[key]
    if isinstance(output, (tuple, list)) and output:
        return output[0]
    raise TypeError(f"Cannot extract logits from output type {type(output)!r}")


def accuracy_counts(logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    pred = logits.argmax(dim=1)
    correct = int((pred == labels).sum().item())
    total = int(labels.numel())
    return {"tx_acc": 100.0 * correct / max(1, total), "tx_correct": correct, "tx_total": total}


@torch.no_grad()
def evaluate_tx(
    model,
    loader,
    device,
    *,
    forward_fn: Optional[Callable[[Any, Dict[str, Any], torch.device], Any]] = None,
    loss_fn: Optional[Callable[[torch.Tensor, torch.Tensor, Dict[str, Any], Any], torch.Tensor]] = None,
    max_batches: int = 0,
) -> Dict[str, float]:
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0
    for batch_i, batch in enumerate(loader):
        if max_batches and batch_i >= int(max_batches):
            break
        y = batch["label"].to(device)
        if forward_fn is None:
            out = model(batch["iq"].to(device))
        else:
            out = forward_fn(model, batch, device)
        logits = logits_from_output(out)
        if loss_fn is not None:
            loss = loss_fn(logits, y, batch, out)
            loss_sum += float(loss.detach().cpu()) * int(y.numel())
        counts = accuracy_counts(logits, y)
        correct += int(counts["tx_correct"])
        total += int(counts["tx_total"])
    stats = {"tx_acc": 100.0 * correct / max(1, total), "tx_correct": correct, "tx_total": total}
    if loss_fn is not None:
        stats["tx_loss"] = loss_sum / max(1, total)
    return stats


def cross_entropy_val_loss(logits: torch.Tensor, labels: torch.Tensor, batch: Dict[str, Any], output: Any) -> torch.Tensor:
    del batch, output
    return F.cross_entropy(logits, labels)


def aggregate_named_stats(named_stats: Dict[str, Dict[str, float]], keys: Iterable[str]) -> Dict[str, float]:
    correct = 0
    total = 0
    for key in keys:
        stats = named_stats.get(key)
        if not stats:
            continue
        correct += int(stats.get("tx_correct", 0))
        total += int(stats.get("tx_total", 0))
    return {"tx_acc": 100.0 * correct / max(1, total), "tx_correct": correct, "tx_total": total}


def format_named_test_lines(named_stats: Dict[str, Dict[str, float]]) -> list[str]:
    priority = ["test_unseen_day_seen_rx", "test_seen_day_unseen_rx", "test_unseen_day_unseen_rx"]
    ordered = [k for k in priority if k in named_stats] + [k for k in named_stats if k not in priority]
    lines = []
    for name in ordered:
        stats = named_stats[name]
        lines.append(
            f"[TEST-NAMED] {name}: tx={stats.get('tx_acc', float('nan')):.2f}% "
            f"({int(stats.get('tx_correct', 0))}/{int(stats.get('tx_total', 0))})"
        )
    return lines


def format_extra_test_lines(extra_tests: Dict[str, Any]) -> list[str]:
    sat_stats = extra_tests.get("sat_channel") if isinstance(extra_tests, dict) else None
    if not isinstance(sat_stats, dict):
        return []
    lines = []
    for scenario, stats in sat_stats.items():
        if not isinstance(stats, dict):
            continue
        agg = stats.get("aggregate", {})
        all_agg = stats.get("all_named_aggregate", {})
        selected = ",".join(stats.get("selected_names", []))
        strict = stats.get("strict_udu", float("nan"))
        try:
            strict_text = f"{float(strict):.2f}%"
        except Exception:
            strict_text = "nan%"
        lines.append(
            f"[SAT-TEST] scenario={scenario} selected={selected} "
            f"overall_tx={agg.get('tx_acc', float('nan')):.2f}% "
            f"all_named_tx={all_agg.get('tx_acc', float('nan')):.2f}% "
            f"strict_udu={strict_text} "
            f"({int(agg.get('tx_correct', 0))}/{int(agg.get('tx_total', 0))})"
        )
        named = stats.get("named", {})
        if isinstance(named, dict):
            priority = ["test_unseen_day_seen_rx", "test_seen_day_unseen_rx", "test_unseen_day_unseen_rx"]
            ordered = [k for k in priority if k in named] + [k for k in named if k not in priority]
            for name in ordered:
                cur = named[name]
                lines.append(
                    f"[SAT-TEST-SPLIT] scenario={scenario} {name}: "
                    f"tx={cur.get('tx_acc', float('nan')):.2f}% "
                    f"({int(cur.get('tx_correct', 0))}/{int(cur.get('tx_total', 0))})"
                )
    return lines


@dataclass
class BestValTestGate:
    best_val: float = -math.inf
    best_epoch: int = -1
    mode: str = "max"

    def __post_init__(self) -> None:
        self.mode = str(self.mode).lower()
        if self.mode not in {"max", "min"}:
            raise ValueError("BestValTestGate mode must be 'max' or 'min'.")
        if self.best_val == -math.inf and self.mode == "min":
            self.best_val = math.inf

    def should_test(self, val_metric: float, epoch: Optional[int] = None) -> bool:
        val = float(val_metric)
        improved = val > self.best_val if self.mode == "max" else val < self.best_val
        if not improved:
            return False
        self.best_val = val
        if epoch is not None:
            self.best_epoch = int(epoch)
        return True


@dataclass
class PlateauStep:
    improved: bool
    lr_reduced: bool
    stop_training: bool
    best_loss: float
    bad_epochs: int


class ValidationLossPlateauController:
    """Validation-loss LR decay and early stopping used by paper baselines."""

    def __init__(
        self,
        optimizer,
        *,
        lr_factor: float,
        lr_patience: int,
        early_stop_patience: int,
        min_delta: float = 0.0,
        min_lr: float = 0.0,
    ):
        self.optimizer = optimizer
        self.lr_factor = float(lr_factor)
        self.lr_patience = int(lr_patience)
        self.early_stop_patience = int(early_stop_patience)
        self.min_delta = float(min_delta)
        self.min_lr = float(min_lr)
        self.best_loss = math.inf
        self.best_epoch = -1
        self.bad_epochs = 0

    def step(self, val_loss: float, epoch: Optional[int] = None) -> PlateauStep:
        loss = float(val_loss)
        improved = loss < (self.best_loss - self.min_delta)
        lr_reduced = False
        if improved:
            self.best_loss = loss
            self.best_epoch = -1 if epoch is None else int(epoch)
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
            if self.lr_patience > 0 and self.bad_epochs > 0 and self.bad_epochs % self.lr_patience == 0:
                for group in self.optimizer.param_groups:
                    old_lr = float(group.get("lr", 0.0))
                    group["lr"] = max(self.min_lr, old_lr * self.lr_factor)
                lr_reduced = True
        stop = self.early_stop_patience > 0 and self.bad_epochs >= self.early_stop_patience
        return PlateauStep(
            improved=improved,
            lr_reduced=lr_reduced,
            stop_training=stop,
            best_loss=float(self.best_loss),
            bad_epochs=int(self.bad_epochs),
        )


@dataclass
class TrainHistory:
    epochs: list[Dict[str, Any]] = field(default_factory=list)
    best: Dict[str, Any] = field(default_factory=dict)


def run_validation_gated_training(
    *,
    model,
    train_loader,
    val_loader,
    named_test_loaders: Dict[str, Any],
    device,
    epochs: int,
    optimizer,
    train_step_fn: Callable[[Any, Dict[str, Any], torch.device, int, int], Dict[str, Any]],
    output_dir: str,
    scheduler=None,
    plateau_controller: Optional[ValidationLossPlateauController] = None,
    forward_eval_fn: Optional[Callable[[Any, Dict[str, Any], torch.device], Any]] = None,
    val_loss_fn: Optional[Callable[[torch.Tensor, torch.Tensor, Dict[str, Any], Any], torch.Tensor]] = None,
    best_metric: str = "acc",
    test_evaluate_fn: Optional[Callable[[Any, Any, torch.device], Dict[str, float]]] = None,
    extra_test_fn: Optional[Callable[[Any, torch.device], Dict[str, Any]]] = None,
    test_keys: Optional[list[str]] = None,
    checkpoint_name: str = "best_by_val.pt",
) -> TrainHistory:
    ensure_dir(output_dir)
    best_metric = str(best_metric).lower()
    if best_metric not in {"acc", "loss"}:
        raise ValueError("best_metric must be 'acc' or 'loss'.")
    gate = BestValTestGate(mode="min" if best_metric == "loss" else "max")
    history = TrainHistory()
    test_keys = test_keys or ["test_unseen_day_seen_rx", "test_seen_day_unseen_rx", "test_unseen_day_unseen_rx"]
    best_path = os.path.join(output_dir, checkpoint_name)
    step = 0
    for epoch in range(1, int(epochs) + 1):
        print(f"[Epoch {epoch:03d}/{int(epochs):03d}][START]", flush=True)
        model.train()
        loss_sum = 0.0
        n_batches = 0
        for batch in train_loader:
            metrics = train_step_fn(model, batch, device, epoch, step)
            loss_sum += float(metrics.get("loss", 0.0))
            n_batches += 1
            step += 1
        if scheduler is not None and plateau_controller is None:
            scheduler.step()
        val_stats = evaluate_tx(model, val_loader, device, forward_fn=forward_eval_fn, loss_fn=val_loss_fn)
        plateau_stats = None
        if plateau_controller is not None:
            if "tx_loss" not in val_stats:
                raise ValueError("plateau_controller requires val_loss_fn so validation loss is available.")
            plateau_stats = plateau_controller.step(val_stats["tx_loss"], epoch=epoch)
        epoch_stats: Dict[str, Any] = {
            "epoch": epoch,
            "train_loss": loss_sum / max(1, n_batches),
            "val": val_stats,
            "tested": False,
        }
        if plateau_stats is not None:
            epoch_stats["plateau"] = {
                "improved": plateau_stats.improved,
                "lr_reduced": plateau_stats.lr_reduced,
                "stop_training": plateau_stats.stop_training,
                "bad_epochs": plateau_stats.bad_epochs,
                "best_loss": plateau_stats.best_loss,
                "lr": [float(group.get("lr", 0.0)) for group in optimizer.param_groups],
            }
        gate_value = val_stats["tx_loss"] if best_metric == "loss" else val_stats["tx_acc"]
        if gate.should_test(gate_value, epoch):
            named_stats = {
                name: (
                    test_evaluate_fn(model, loader, device)
                    if test_evaluate_fn is not None
                    else evaluate_tx(model, loader, device, forward_fn=forward_eval_fn)
                )
                for name, loader in named_test_loaders.items()
            }
            aggregate = aggregate_named_stats(named_stats, test_keys)
            epoch_stats.update({
                "tested": True,
                "test_named": named_stats,
                "test_overall": aggregate,
            })
        print(
            f"[Epoch {epoch:03d}/{int(epochs):03d}] train_loss={epoch_stats['train_loss']:.4f} "
            f"val_tx={val_stats['tx_acc']:.2f}% tested={int(epoch_stats['tested'])}",
            flush=True,
        )
        if epoch_stats.get("tested"):
            print(f"[BEST-VAL-TEST] overall_tx={epoch_stats['test_overall']['tx_acc']:.2f}%", flush=True)
            for line in format_named_test_lines(epoch_stats["test_named"]):
                print(line, flush=True)
            extra_tests = extra_test_fn(model, device) if extra_test_fn is not None else {}
            epoch_stats["extra_tests"] = extra_tests
            for line in format_extra_test_lines(extra_tests):
                print(line, flush=True)
            history.best = {
                "epoch": epoch,
                "best_rule": "val_tx_loss" if best_metric == "loss" else "val_tx_acc",
                "val": val_stats,
                "test_named": epoch_stats["test_named"],
                "test_overall": epoch_stats["test_overall"],
                "extra_tests": epoch_stats.get("extra_tests", {}),
                "checkpoint": best_path,
            }
            torch.save({"model": model.state_dict(), "epoch": epoch, "stats": history.best}, best_path)
        history.epochs.append(epoch_stats)
        save_json({"epochs": history.epochs, "best": history.best}, os.path.join(output_dir, "metrics.json"))
        if plateau_stats is not None and plateau_stats.stop_training:
            print(
                f"[EARLY-STOP] validation loss did not improve for {plateau_stats.bad_epochs} epochs.",
                flush=True,
            )
            break
    return history
