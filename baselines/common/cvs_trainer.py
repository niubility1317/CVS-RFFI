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


def evaluate_named_tests(
    *,
    model,
    named_test_loaders: Dict[str, Any],
    device,
    forward_eval_fn: Optional[Callable[[Any, Dict[str, Any], torch.device], Any]] = None,
    test_evaluate_fn: Optional[Callable[[Any, Any, torch.device], Dict[str, float]]] = None,
    test_keys: Optional[list[str]] = None,
) -> tuple[Dict[str, Dict[str, float]], Dict[str, float]]:
    named_stats = {
        name: (
            test_evaluate_fn(model, loader, device)
            if test_evaluate_fn is not None
            else evaluate_tx(model, loader, device, forward_fn=forward_eval_fn)
        )
        for name, loader in named_test_loaders.items()
    }
    aggregate = aggregate_named_stats(
        named_stats,
        test_keys or ["test_unseen_day_seen_rx", "test_seen_day_unseen_rx", "test_unseen_day_unseen_rx"],
    )
    return named_stats, aggregate


def format_named_test_lines(named_stats: Dict[str, Dict[str, float]], *, prefix: str = "[TEST-NAMED]") -> list[str]:
    priority = ["test_unseen_day_seen_rx", "test_seen_day_unseen_rx", "test_unseen_day_unseen_rx"]
    ordered = [k for k in priority if k in named_stats] + [k for k in named_stats if k not in priority]
    lines = []
    for name in ordered:
        stats = named_stats[name]
        lines.append(
            f"{prefix} {name}: tx={stats.get('tx_acc', float('nan')):.2f}% "
            f"({int(stats.get('tx_correct', 0))}/{int(stats.get('tx_total', 0))})"
        )
    return lines


def evaluate_primary_and_obs_tests(
    *,
    model,
    named_test_loaders: Dict[str, Any],
    device,
    forward_eval_fn: Optional[Callable[[Any, Dict[str, Any], torch.device], Any]] = None,
    test_evaluate_fn: Optional[Callable[[Any, Any, torch.device], Dict[str, float]]] = None,
    test_keys: Optional[list[str]] = None,
) -> Dict[str, Any]:
    """Evaluate the configured baseline protocol and, when needed, OBS too.

    Receiver-collaborative baselines pass ``test_evaluate_fn`` to collapse
    receiver observations into packet groups. For fair CVS-RFFI comparison, we
    also report the observation-level protocol over the same named loaders.
    """

    named_stats, aggregate = evaluate_named_tests(
        model=model,
        named_test_loaders=named_test_loaders,
        device=device,
        forward_eval_fn=forward_eval_fn,
        test_evaluate_fn=test_evaluate_fn,
        test_keys=test_keys,
    )
    result: Dict[str, Any] = {
        "test_named": named_stats,
        "test_overall": aggregate,
    }
    if test_evaluate_fn is not None:
        obs_named_stats, obs_aggregate = evaluate_named_tests(
            model=model,
            named_test_loaders=named_test_loaders,
            device=device,
            forward_eval_fn=forward_eval_fn,
            test_evaluate_fn=None,
            test_keys=test_keys,
        )
        result.update(
            {
                "test_named_collab": named_stats,
                "test_overall_collab": aggregate,
                "test_named_obs": obs_named_stats,
                "test_overall_obs": obs_aggregate,
                "test_protocols": {
                    "primary": "collaborative_group",
                    "comparison": "observation",
                },
            }
        )
    return result


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


def _mean_std(values: list[float]) -> Dict[str, float]:
    vals = [float(v) for v in values]
    if not vals:
        return {"mean": float("nan"), "std": float("nan")}
    mean = sum(vals) / len(vals)
    var = sum((v - mean) ** 2 for v in vals) / len(vals)
    return {"mean": mean, "std": math.sqrt(var)}


def summarize_paper_eval_window(records: list[Dict[str, Any]], *, name: str) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "name": str(name),
        "n": len(records),
        "epochs": [int(r.get("epoch", -1)) for r in records],
    }
    overall = [float(r.get("test_overall", {}).get("tx_acc", float("nan"))) for r in records]
    summary["test_overall_tx_acc"] = _mean_std([v for v in overall if not math.isnan(v)])
    named_keys = sorted({k for r in records for k in r.get("test_named", {})})
    named: Dict[str, Any] = {}
    for key in named_keys:
        vals = [
            float(r.get("test_named", {}).get(key, {}).get("tx_acc", float("nan")))
            for r in records
            if key in r.get("test_named", {})
        ]
        named[key] = {"tx_acc": _mean_std([v for v in vals if not math.isnan(v)])}
    summary["test_named"] = named
    return summary


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
    pseudo_step_fn: Optional[Callable[[Any, torch.device, int, int], Dict[str, Any]]] = None,
    scheduler=None,
    plateau_controller: Optional[ValidationLossPlateauController] = None,
    forward_eval_fn: Optional[Callable[[Any, Dict[str, Any], torch.device], Any]] = None,
    val_loss_fn: Optional[Callable[[torch.Tensor, torch.Tensor, Dict[str, Any], Any], torch.Tensor]] = None,
    best_metric: str = "acc",
    test_evaluate_fn: Optional[Callable[[Any, Any, torch.device], Dict[str, float]]] = None,
    extra_test_fn: Optional[Callable[[Any, torch.device], Dict[str, Any]]] = None,
    test_keys: Optional[list[str]] = None,
    paper_eval_last_n: int = 0,
    paper_eval_name: str = "",
    test_on_val_improve: bool = True,
    checkpoint_name: str = "best_by_val.pt",
) -> TrainHistory:
    ensure_dir(output_dir)
    best_metric = str(best_metric).lower()
    if best_metric not in {"acc", "loss"}:
        raise ValueError("best_metric must be 'acc' or 'loss'.")
    gate = BestValTestGate(mode="min" if best_metric == "loss" else "max")
    history = TrainHistory()
    test_keys = test_keys or ["test_unseen_day_seen_rx", "test_seen_day_unseen_rx", "test_unseen_day_unseen_rx"]
    paper_eval_last_n = max(0, int(paper_eval_last_n))
    paper_eval_name = str(paper_eval_name or f"last{paper_eval_last_n}")
    test_on_val_improve = bool(test_on_val_improve)
    paper_window_records: list[Dict[str, Any]] = []
    best_path = os.path.join(output_dir, checkpoint_name)
    step = 0
    for epoch in range(1, int(epochs) + 1):
        print(f"[Epoch {epoch:03d}/{int(epochs):03d}][START]", flush=True)
        model.train()
        loss_sum = 0.0
        n_batches = 0
        metric_sums: Dict[str, float] = {}
        metric_counts: Dict[str, int] = {}
        pseudo_sums: Dict[str, float] = {}
        pseudo_count = 0
        for batch in train_loader:
            metrics = train_step_fn(model, batch, device, epoch, step)
            batch_loss = float(metrics.get("loss", 0.0))
            for key, value in metrics.items():
                if key == "loss":
                    continue
                try:
                    if hasattr(value, "detach"):
                        metric_value = float(value.detach().cpu())
                    else:
                        metric_value = float(value)
                except Exception:
                    continue
                if not math.isfinite(metric_value):
                    continue
                metric_sums[key] = metric_sums.get(key, 0.0) + metric_value
                metric_counts[key] = metric_counts.get(key, 0) + 1
            if pseudo_step_fn is not None:
                pseudo_metrics = pseudo_step_fn(model, device, epoch, step)
                batch_loss += float(pseudo_metrics.get("loss", 0.0))
                for key, value in pseudo_metrics.items():
                    if key == "loss":
                        continue
                    pseudo_sums[key] = pseudo_sums.get(key, 0.0) + float(value)
                pseudo_count += 1
            loss_sum += batch_loss
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
        if metric_counts:
            epoch_stats["train_metrics"] = {
                key: metric_sums[key] / max(1, metric_counts[key])
                for key in sorted(metric_sums.keys())
            }
        if pseudo_count:
            pseudo_stats: Dict[str, float] = {}
            for key, value in pseudo_sums.items():
                if key in {"pseudo/total", "pseudo/selected"}:
                    pseudo_stats[key] = value
                else:
                    pseudo_stats[key] = value / max(1, pseudo_count)
            epoch_stats["train_pseudo"] = pseudo_stats
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
        val_improved = gate.should_test(gate_value, epoch)
        if val_improved and not test_on_val_improve:
            history.best = {
                "epoch": epoch,
                "best_rule": "val_tx_loss" if best_metric == "loss" else "val_tx_acc",
                "val": val_stats,
                "test_on_val_improve": False,
                "checkpoint": best_path,
            }
            torch.save({"model": model.state_dict(), "epoch": epoch, "stats": history.best}, best_path)
        if val_improved and test_on_val_improve:
            test_result = evaluate_primary_and_obs_tests(
                model=model,
                named_test_loaders=named_test_loaders,
                device=device,
                forward_eval_fn=forward_eval_fn,
                test_evaluate_fn=test_evaluate_fn,
                test_keys=test_keys,
            )
            epoch_stats.update({
                "tested": True,
                **test_result,
            })
        print(
            f"[Epoch {epoch:03d}/{int(epochs):03d}] train_loss={epoch_stats['train_loss']:.4f} "
            f"val_tx={val_stats['tx_acc']:.2f}% tested={int(epoch_stats['tested'])}",
            flush=True,
        )
        if epoch_stats.get("train_metrics"):
            parts = [f"{key}={value:.6g}" for key, value in epoch_stats["train_metrics"].items()]
            print("[TRAIN-METRICS] " + " ".join(parts), flush=True)
        if epoch_stats.get("tested"):
            if "test_named_obs" in epoch_stats:
                print(
                    f"[BEST-VAL-TEST-COLLAB] overall_tx={epoch_stats['test_overall_collab']['tx_acc']:.2f}%",
                    flush=True,
                )
                for line in format_named_test_lines(
                    epoch_stats["test_named_collab"],
                    prefix="[TEST-NAMED-COLLAB]",
                ):
                    print(line, flush=True)
                print(
                    f"[BEST-VAL-TEST-OBS] overall_tx={epoch_stats['test_overall_obs']['tx_acc']:.2f}%",
                    flush=True,
                )
                for line in format_named_test_lines(epoch_stats["test_named_obs"], prefix="[TEST-NAMED-OBS]"):
                    print(line, flush=True)
            else:
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
                "test_named_collab": epoch_stats.get("test_named_collab"),
                "test_overall_collab": epoch_stats.get("test_overall_collab"),
                "test_named_obs": epoch_stats.get("test_named_obs"),
                "test_overall_obs": epoch_stats.get("test_overall_obs"),
                "test_protocols": epoch_stats.get("test_protocols"),
                "extra_tests": epoch_stats.get("extra_tests", {}),
                "checkpoint": best_path,
            }
            torch.save({"model": model.state_dict(), "epoch": epoch, "stats": history.best}, best_path)
        elif val_improved and not test_on_val_improve:
            print(f"[BEST-VAL] epoch={epoch} checkpoint={best_path} test_on_val_improve=0", flush=True)
        if paper_eval_last_n and epoch > max(0, int(epochs) - paper_eval_last_n):
            paper_result = evaluate_primary_and_obs_tests(
                model=model,
                named_test_loaders=named_test_loaders,
                device=device,
                forward_eval_fn=forward_eval_fn,
                test_evaluate_fn=test_evaluate_fn,
                test_keys=test_keys,
            )
            paper_record = {
                "epoch": epoch,
                **paper_result,
            }
            epoch_stats["paper_eval"] = paper_record
            paper_window_records.append(paper_record)
            print(
                f"[PAPER-EVAL] name={paper_eval_name} epoch={epoch} "
                f"overall_tx={paper_record['test_overall']['tx_acc']:.2f}%",
                flush=True,
            )
        history.epochs.append(epoch_stats)
        save_json({"epochs": history.epochs, "best": history.best}, os.path.join(output_dir, "metrics.json"))
        if plateau_stats is not None and plateau_stats.stop_training:
            print(
                f"[EARLY-STOP] validation loss did not improve for {plateau_stats.bad_epochs} epochs.",
                flush=True,
            )
            break
    if history.epochs:
        last_epoch = history.epochs[-1]
        test_result = evaluate_primary_and_obs_tests(
            model=model,
            named_test_loaders=named_test_loaders,
            device=device,
            forward_eval_fn=forward_eval_fn,
            test_evaluate_fn=test_evaluate_fn,
            test_keys=test_keys,
        )
        extra_tests = extra_test_fn(model, device) if extra_test_fn is not None else {}
        final = {
            "epoch": last_epoch.get("epoch"),
            "reason": "post_training",
            "last_epoch_tested": bool(last_epoch.get("tested")),
            **test_result,
            "extra_tests": extra_tests,
        }
        if paper_window_records:
            final["paper_eval_window"] = summarize_paper_eval_window(paper_window_records, name=paper_eval_name)
            window = final["paper_eval_window"]["test_overall_tx_acc"]
            print(
                f"[PAPER-EVAL-SUMMARY] name={paper_eval_name} n={len(paper_window_records)} "
                f"overall_tx_mean={window['mean']:.2f}% overall_tx_std={window['std']:.2f}%",
                flush=True,
            )
        if "test_named_obs" in final:
            print(f"[FINAL-TEST-COLLAB] overall_tx={final['test_overall_collab']['tx_acc']:.2f}%", flush=True)
            for line in format_named_test_lines(final["test_named_collab"], prefix="[FINAL-TEST-NAMED-COLLAB]"):
                print(line, flush=True)
            print(f"[FINAL-TEST-OBS] overall_tx={final['test_overall_obs']['tx_acc']:.2f}%", flush=True)
            for line in format_named_test_lines(final["test_named_obs"], prefix="[FINAL-TEST-NAMED-OBS]"):
                print(line, flush=True)
        else:
            print(f"[FINAL-TEST] overall_tx={final['test_overall']['tx_acc']:.2f}%", flush=True)
            for line in format_named_test_lines(final["test_named"]):
                print(line.replace("[TEST-NAMED]", "[FINAL-TEST-NAMED]", 1), flush=True)
        for line in format_extra_test_lines(extra_tests):
            print(line.replace("[SAT-TEST]", "[FINAL-SAT-TEST]", 1), flush=True)
        save_json(
            {"epochs": history.epochs, "best": history.best, "final": final},
            os.path.join(output_dir, "metrics.json"),
        )
    return history
