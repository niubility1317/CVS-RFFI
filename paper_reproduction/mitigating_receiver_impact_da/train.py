from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F

from paper_reproduction.common.config import load_json_config
from paper_reproduction.common.wisig_runtime import set_seed, tx_accuracy, write_json
from paper_reproduction.mitigating_receiver_impact_da.algorithm import PseudoLabelState, gada_batch_step
from paper_reproduction.mitigating_receiver_impact_da.data import build_manysig_task_loaders, load_wisig_compact_pkl
from paper_reproduction.mitigating_receiver_impact_da.model import ReceiverImpactGADNet
from paper_reproduction.mitigating_receiver_impact_da.protocol import (
    PAPER_TASKS,
    build_paper_task_plan,
    validate_paper_faithful_config,
)


def build_dry_run_payload(config: dict) -> dict:
    checked = validate_paper_faithful_config(config)
    return {
        "method_id": "mitigating_receiver_impact_da",
        "paper": "Mitigating Receiver Impact on Radio Frequency Fingerprint Identification via Domain Adaptation",
        "citation": "Liu Yang, Qiang Li, Xiaoyang Ren, Yi Fang, and Shafei Wang, IEEE Internet of Things Journal, 2024",
        "algorithm": "GAD adversarial training with DV-KL domain alignment and adaptive pseudo-labeling",
        "scope": checked["claim_boundary"],
        "dataset": checked["dataset"],
        "capture_days": checked["capture_days"],
        "source_target_tasks": checked["source_target_tasks"],
        "target_labels_scope": "evaluation_only",
        "paper_task_plan": build_paper_task_plan(checked),
        "paper_reported_hyperparameters": checked["paper_reported_hyperparameters"],
        "paper_unspecified_fields": checked["paper_unspecified_fields"],
        "paper_evidence_targets": {
            "Table II": "task/display-method plan only",
            "Table III": "not reproduced in dry-run",
            "Table IV": "not reproduced in dry-run",
            "Fig.5-7": "not reproduced in dry-run",
        },
        "claim_blocks": [
            "not CVS Stage2-A/B/C",
            "not satellite/LEO deployment evidence",
            "not open-set or new-class registration",
            "target labels are evaluation-only in paper-faithful DA",
        ],
    }


def _batch_tensor(batch: Any, key: str, fallback_index: int) -> torch.Tensor:
    if isinstance(batch, dict):
        value = batch[key]
    else:
        value = batch[fallback_index]
    if not isinstance(value, torch.Tensor):
        value = torch.as_tensor(value)
    return value


def _next_cycling(iterator: Iterable[Any], current_iterator: Any, *, name: str) -> tuple[Any, Any]:
    try:
        return next(current_iterator), current_iterator
    except StopIteration:
        current_iterator = iter(iterator)
    try:
        return next(current_iterator), current_iterator
    except StopIteration as exc:
        raise ValueError(f"{name} batches cannot be empty") from exc


def _state_payload(state: PseudoLabelState) -> dict[str, Any]:
    return {
        "pseudo_counts": [float(v) for v in state.pseudo_counts.tolist()],
        "predicted_counts": [float(v) for v in state.predicted_counts.tolist()],
        "total_seen": int(state.total_seen),
    }


def run_gada_training_loop(
    model: ReceiverImpactGADNet,
    source_batches: Iterable[Any],
    target_batches: Iterable[Any],
    *,
    optimizer_t: Any,
    optimizer_ec: Any,
    epochs: int,
    checkpoint_path: Path | str | None = None,
    device: torch.device | str | None = None,
    estimate_steps: int = 7,
    base_tau: float = 0.7,
    mu: float = 0.5,
    kl_weight: float = 0.005,
    class_prior: torch.Tensor | None = None,
    max_batches_per_epoch: int | None = None,
) -> dict[str, Any]:
    """Execute the Algorithm 1 GAD loop over caller-provided source/target batches.

    This is intentionally data-loader agnostic: paper-faithful WiSig task construction
    remains outside this helper, while this function owns the update ordering,
    pseudo-label state, metrics, and checkpoint surface.
    """
    if int(epochs) <= 0:
        raise ValueError("epochs must be positive")
    if max_batches_per_epoch is not None and int(max_batches_per_epoch) <= 0:
        raise ValueError("max_batches_per_epoch must be positive when provided")

    resolved_device = torch.device(device) if device is not None else next(model.parameters()).device
    model.to(resolved_device)
    state = PseudoLabelState(num_classes=int(model.num_tx))
    target_iterator = iter(target_batches)
    history: list[dict[str, float | int]] = []
    total_batches = 0

    for epoch_index in range(int(epochs)):
        epoch_batches = 0
        epoch_loss = 0.0
        epoch_loss_source = 0.0
        epoch_loss_target = 0.0
        epoch_loss_kl = 0.0
        epoch_conf = 0.0
        epoch_selected = 0
        epoch_weight_min = float("inf")
        epoch_weight_max = float("-inf")
        source_iterator = iter(source_batches)
        for source_batch in source_iterator:
            if max_batches_per_epoch is not None and epoch_batches >= int(max_batches_per_epoch):
                break
            target_batch, target_iterator = _next_cycling(target_batches, target_iterator, name="target")
            source_x = _batch_tensor(source_batch, "iq", 0).to(resolved_device)
            source_y = _batch_tensor(source_batch, "label", 1).long().to(resolved_device)
            target_x = _batch_tensor(target_batch, "iq", 0).to(resolved_device)
            result = gada_batch_step(
                model,
                source_x,
                source_y,
                target_x,
                state=state,
                optimizer_t=optimizer_t,
                optimizer_ec=optimizer_ec,
                estimate_steps=estimate_steps,
                base_tau=base_tau,
                mu=mu,
                kl_weight=kl_weight,
                class_prior=class_prior,
            )
            epoch_batches += 1
            total_batches += 1
            epoch_loss += float(result["loss"].item())
            epoch_loss_source += float(result["loss_source"].item())
            epoch_loss_target += float(result["loss_target"].item())
            epoch_loss_kl += float(result["loss_kl"].item())
            epoch_conf += float(result["target_conf_mean"].item())
            epoch_selected += int(result["target_selected"].item())
            epoch_weight_min = min(epoch_weight_min, float(result["class_weight_min"].item()))
            epoch_weight_max = max(epoch_weight_max, float(result["class_weight_max"].item()))

        if epoch_batches == 0:
            raise ValueError("source batches cannot be empty")
        history.append(
            {
                "epoch": epoch_index + 1,
                "batches": epoch_batches,
                "loss_mean": epoch_loss / float(epoch_batches),
                "loss_source_mean": epoch_loss_source / float(epoch_batches),
                "loss_target_mean": epoch_loss_target / float(epoch_batches),
                "loss_kl_mean": epoch_loss_kl / float(epoch_batches),
                "target_conf_mean": epoch_conf / float(epoch_batches),
                "class_weight_min": epoch_weight_min,
                "class_weight_max": epoch_weight_max,
                "target_selected": epoch_selected,
                "target_seen_total": int(state.total_seen),
            }
        )

    payload: dict[str, Any] = {
        "paper": "Mitigating Receiver Impact on Radio Frequency Fingerprint Identification via Domain Adaptation",
        "algorithm": "Algorithm 1 GAD training loop",
        "epochs": int(epochs),
        "batches": total_batches,
        "estimate_steps": int(estimate_steps),
        "base_tau": float(base_tau),
        "mu": float(mu),
        "kl_weight": float(kl_weight),
        "history": history,
        "state": _state_payload(state),
    }
    if checkpoint_path is not None:
        path = Path(checkpoint_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                **payload,
                "epoch": int(epochs),
                "model_state_dict": model.state_dict(),
            },
            path,
        )
        payload["checkpoint_path"] = str(path)
    return payload


def _limited_batches(loader: Iterable[Any], max_batches: int | None) -> Iterable[Any]:
    if max_batches is None:
        return loader
    limit = int(max_batches)

    def _generator() -> Iterable[Any]:
        for index, batch in enumerate(loader):
            if index >= limit:
                break
            yield batch

    return _generator()


def _evaluate_target_accuracy(model: ReceiverImpactGADNet, loader: Iterable[Any], *, device: torch.device | str) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            x = _batch_tensor(batch, "iq", 0).to(device)
            y = _batch_tensor(batch, "label", 1).long().to(device)
            logits = model.classify(x)
            correct += int((logits.argmax(dim=1) == y).sum().item())
            total += int(y.numel())
    return 0.0 if total == 0 else correct / float(total)


def _train_source_only(
    model: ReceiverImpactGADNet,
    source_loader: Iterable[Any],
    *,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    device: torch.device | str,
    max_batches_per_epoch: int | None,
) -> dict[str, Any]:
    history: list[dict[str, float | int]] = []
    for epoch_index in range(int(epochs)):
        model.train()
        loss_sum = 0.0
        acc_sum = 0.0
        batches = 0
        for batch in _limited_batches(source_loader, max_batches_per_epoch):
            x = _batch_tensor(batch, "iq", 0).to(device)
            y = _batch_tensor(batch, "label", 1).long().to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model.classify(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach().cpu())
            acc_sum += tx_accuracy(logits.detach().cpu(), y.detach().cpu())
            batches += 1
        if batches == 0:
            raise ValueError("source batches cannot be empty")
        history.append(
            {
                "epoch": epoch_index + 1,
                "batches": batches,
                "loss_mean": loss_sum / float(batches),
                "source_batch_acc_mean": acc_sum / float(batches),
            }
        )
    return {"history": history}


def _task_slug(task: str) -> str:
    return str(task).replace("->", "_to_").replace("/", "_")


def run_table2_reproduction(
    compact_or_path: dict[str, Any] | str | Path,
    *,
    tasks: list[str] | None = None,
    methods: list[str] | None = None,
    output_dir: Path | str,
    epochs: int,
    batch_size: int,
    learning_rate: float = 0.0006,
    max_samples_per_combo: int | None = None,
    max_batches_per_epoch: int | None = None,
    source_pretrain_epochs: int | None = None,
    seed: int = 0,
    device: torch.device | str | None = None,
    num_workers: int = 0,
) -> dict[str, Any]:
    set_seed(int(seed))
    resolved_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    requested_tasks = list(PAPER_TASKS if tasks is None else tasks)
    requested_methods = [str(method).lower() for method in (methods or ["source_only", "proposed"])]
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    compact = load_wisig_compact_pkl(str(compact_or_path)) if isinstance(compact_or_path, (str, Path)) else compact_or_path
    rows: list[dict[str, Any]] = []

    for task in requested_tasks:
        loaders = build_manysig_task_loaders(
            compact,
            task=task,
            batch_size=batch_size,
            max_samples_per_combo=max_samples_per_combo,
            seed=seed,
            num_workers=num_workers,
        )
        for method in requested_methods:
            if method not in {"source_only", "proposed"}:
                rows.append(
                    {
                        "task": task,
                        "method": method,
                        "status": "not_implemented",
                        "target_labels_scope": "evaluation_only",
                    }
                )
                continue
            model = ReceiverImpactGADNet(num_tx=6).to(resolved_device)
            checkpoint_path = output_path / f"{_task_slug(task)}_{method}.pt"
            if method == "source_only":
                optimizer = torch.optim.Adam(
                    list(model.feature_extractor.parameters()) + list(model.classifier.parameters()),
                    lr=float(learning_rate),
                )
                train_result = _train_source_only(
                    model,
                    loaders["source"],
                    optimizer=optimizer,
                    epochs=epochs,
                    device=resolved_device,
                    max_batches_per_epoch=max_batches_per_epoch,
                )
                torch.save(
                    {
                        "paper": "Mitigating Receiver Impact on Radio Frequency Fingerprint Identification via Domain Adaptation",
                        "method": method,
                        "task": task,
                        "model_state_dict": model.inference_state_dict(),
                        "history": train_result["history"],
                    },
                    checkpoint_path,
                )
            else:
                resolved_pretrain_epochs = int(epochs if source_pretrain_epochs is None else source_pretrain_epochs)
                if resolved_pretrain_epochs < 0:
                    raise ValueError("source_pretrain_epochs must be non-negative")
                source_pretrain_result: dict[str, Any] | None = None
                if resolved_pretrain_epochs > 0:
                    source_optimizer = torch.optim.Adam(
                        list(model.feature_extractor.parameters()) + list(model.classifier.parameters()),
                        lr=float(learning_rate),
                    )
                    source_pretrain_result = _train_source_only(
                        model,
                        loaders["source"],
                        optimizer=source_optimizer,
                        epochs=resolved_pretrain_epochs,
                        device=resolved_device,
                        max_batches_per_epoch=max_batches_per_epoch,
                    )
                optimizer_t = torch.optim.Adam(model.estimate_network.parameters(), lr=float(learning_rate))
                optimizer_ec = torch.optim.Adam(
                    list(model.feature_extractor.parameters()) + list(model.classifier.parameters()),
                    lr=float(learning_rate),
                )
                train_result = run_gada_training_loop(
                    model,
                    loaders["source"],
                    loaders["target_train"],
                    optimizer_t=optimizer_t,
                    optimizer_ec=optimizer_ec,
                    epochs=epochs,
                    checkpoint_path=None,
                    device=resolved_device,
                    max_batches_per_epoch=max_batches_per_epoch,
                )
                if source_pretrain_result is not None:
                    train_result["source_pretrain_history"] = source_pretrain_result["history"]
                torch.save(
                    {
                        "paper": "Mitigating Receiver Impact on Radio Frequency Fingerprint Identification via Domain Adaptation",
                        "method": method,
                        "task": task,
                        "model_state_dict": model.inference_state_dict(),
                        "history": train_result["history"],
                        "source_pretrain_history": train_result.get("source_pretrain_history", []),
                        "adaptation": {
                            "algorithm": train_result["algorithm"],
                            "epochs": train_result["epochs"],
                            "estimate_steps": train_result["estimate_steps"],
                            "base_tau": train_result["base_tau"],
                            "mu": train_result["mu"],
                            "kl_weight": train_result["kl_weight"],
                            "state": train_result["state"],
                        },
                    },
                    checkpoint_path,
                )
                train_result["checkpoint_path"] = str(checkpoint_path)
            target_accuracy = _evaluate_target_accuracy(model, loaders["target_eval"], device=resolved_device)
            row = {
                "task": task,
                "method": method,
                "status": "completed",
                "target_accuracy": float(target_accuracy),
                "target_labels_scope": "evaluation_only",
                "target_label_role": loaders["meta"]["target_label_role"],
                "checkpoint_path": str(checkpoint_path),
                "history": train_result["history"],
                "task_meta": loaders["meta"],
            }
            if method == "proposed":
                row["source_pretrain_history"] = train_result.get("source_pretrain_history", [])
            rows.append(row)

    return {
        "method_id": "mitigating_receiver_impact_da",
        "paper": "Mitigating Receiver Impact on Radio Frequency Fingerprint Identification via Domain Adaptation",
        "artifact_type": "table2_reproduction_run",
        "dataset": "WiSig ManySig",
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "max_samples_per_combo": max_samples_per_combo,
        "max_batches_per_epoch": max_batches_per_epoch,
        "source_pretrain_epochs": int(epochs if source_pretrain_epochs is None else source_pretrain_epochs),
        "seed": int(seed),
        "device": str(resolved_device),
        "result_claim_status": "smoke_or_formal_metrics_depend_on_dataset",
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Paper-faithful dry-run entrypoint for the IoTJ 2024 receiver-impact DA RFFI paper.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true", help="Validate config and print the reproduction matrix.")
    parser.add_argument("--run-table2", action="store_true", help="Run Table II source-only/proposed rows on a WiSig ManySig pkl.")
    parser.add_argument("--manysig-pkl", type=Path, default=None)
    parser.add_argument("--methods", type=str, default="source_only,proposed")
    parser.add_argument("--tasks", type=str, default="")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.0006)
    parser.add_argument("--max-samples-per-combo", type=int, default=None)
    parser.add_argument("--max-batches-per-epoch", type=int, default=None)
    parser.add_argument("--source-pretrain-epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("paper_reproduction/runs/mitigating_receiver_impact_da"))
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path for dry-run payload.")
    args = parser.parse_args()

    config = load_json_config(args.config)
    if args.run_table2:
        if args.manysig_pkl is None:
            raise SystemExit("--manysig-pkl is required with --run-table2")
        tasks = [token.strip() for token in args.tasks.split(",") if token.strip()] or None
        methods = [token.strip() for token in args.methods.split(",") if token.strip()]
        payload = run_table2_reproduction(
            args.manysig_pkl,
            tasks=tasks,
            methods=methods,
            output_dir=args.checkpoint_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            max_samples_per_combo=args.max_samples_per_combo,
            max_batches_per_epoch=args.max_batches_per_epoch,
            source_pretrain_epochs=args.source_pretrain_epochs,
            seed=args.seed,
            device=args.device,
            num_workers=args.num_workers,
        )
        if args.output is not None:
            write_json(args.output, payload)
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        return 0

    if not args.dry_run:
        raise SystemExit("formal WiSig training CLI is intentionally gated; use --dry-run or --run-table2")

    payload = build_dry_run_payload(config)
    if args.output is not None:
        write_json(args.output, payload)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
