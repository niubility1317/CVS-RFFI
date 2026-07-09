from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F

from paper_reproduction.common.config import load_json_config
from paper_reproduction.common.wisig_runtime import set_seed, tx_accuracy, write_json
from paper_reproduction.dadda_cross_receiver.data import PAPER_TABLE2_TASKS, build_manysig_task_loaders, load_wisig_compact_pkl
from paper_reproduction.dadda_cross_receiver.experiment_plans import (
    IMPLEMENTED_TABLE2_METHODS,
    PAPER_TABLE2_METHODS,
    build_paper_artifact_plan,
    build_pending_paper_artifacts,
)
from paper_reproduction.dadda_cross_receiver.losses import dadda_objective
from paper_reproduction.dadda_cross_receiver.model import DADDANet


PAPER_TITLE = "Cross-Receiver Radio Frequency Fingerprint Identification Based on Domain Adaptation With Dynamic Distribution Alignment"
PAPER_EVIDENCE_TARGETS = {
    "Table II": "twelve receiver-transfer tasks; full formal reproduction also requires DANN/DAN/DSAN/WD/DCORAL/CDAN baselines and real WiSig runs",
    "Table III": "module ablation matrix not produced by dry-run/smoke",
    "Table IV": "dynamic alpha ablation not produced by dry-run/smoke",
    "Table V": "kernel sensitivity, parameter count, and FLOPs sweep not produced by dry-run/smoke",
    "Table VI": "per-epoch training/testing timing not produced by dry-run/smoke",
    "Fig.5": "SNR robustness not produced by dry-run/smoke",
    "Fig.6-8": "A-distance, t-SNE, and confusion matrix visualizations not produced by dry-run/smoke",
}
CLAIM_BLOCKS = [
    "not CVS Stage2-A/B/C",
    "not target-new enrollment",
    "not unknown/open-set rejection evidence",
    "not satellite/LEO deployment evidence",
]


def build_dry_run_payload(config: dict[str, Any]) -> dict[str, Any]:
    validate_paper_faithful_config(config)
    tasks = list(config.get("source_target_tasks") or PAPER_TABLE2_TASKS)
    return {
        "method_id": "dadda_cross_receiver",
        "paper": PAPER_TITLE,
        "citation": "Junhao Feng, Shengliang Fang, and Youchen Fan, IEEE Internet of Things Journal, 2025",
        "algorithm": "DADDA: modified ResNet18-style G_f + 2-D paper-shaped multiscale G_m + shared-kernel MMD/LMMD + dynamic adaptive factor",
        "paper_scope": "paper_faithful_closed_set_single_source_UDA",
        "cvs_extension": False,
        "dataset": config.get("dataset", "WiSig ManySig"),
        "tx_count": int(config.get("tx_count", 6)),
        "total_receivers": int(config.get("total_receivers", 12)),
        "capture_days": int(config.get("capture_days", 4)),
        "source_target_tasks": tasks,
        "target_labels_scope": "evaluation_only",
        "paper_task_plan": [
            {
                "task": task,
                "source": task.split("->", 1)[0],
                "target": task.split("->", 1)[1],
                "target_label_role": "hidden_for_UDA_training_available_for_final_accuracy_only",
                "compare_method_ids": ["source_only", "dann", "dan", "dsan", "wd", "dcoral", "cdan", "dadda"],
            }
            for task in tasks
        ],
        "paper_reported_hyperparameters": {
            "epochs": 100,
            "batch_size": 128,
            "samples_per_tx_receiver_domain": 4000,
            "samples_per_receiver_domain_in_manysig": "6 x 4000 = 24000",
            "optimizer": "SGD",
            "momentum": 0.9,
            "weight_decay": 0.0005,
            "lr_schedule": "lr_p=0.0001/(1+10p)^0.75",
            "lambda_schedule": "lambda_p=2/(1+exp(-10p))-1",
            "classifier_hidden": [512, 128],
            "model_variant": "conv2d_paper",
            "normalize": "paper uses upstream energy normalization; rerun configs may set runtime normalize false for ablation",
        },
        "paper_evidence_targets": dict(PAPER_EVIDENCE_TARGETS),
        "pending_paper_artifacts": build_pending_paper_artifacts(),
        "claim_blocks": list(CLAIM_BLOCKS),
    }


def lambda_schedule(progress: float, beta: float = 10.0) -> float:
    p = float(progress)
    return float(2.0 / (1.0 + torch.exp(torch.tensor(-float(beta) * p)).item()) - 1.0)


def learning_rate_schedule(progress: float, *, base_lr: float = 0.0001) -> float:
    return float(base_lr) / float((1.0 + 10.0 * float(progress)) ** 0.75)


def _batch_tensor(batch: Any, key: str, fallback_index: int) -> torch.Tensor:
    if isinstance(batch, dict):
        value = batch[key]
    else:
        value = batch[fallback_index]
    if not isinstance(value, torch.Tensor):
        value = torch.as_tensor(value)
    return value


def _limited_batches(loader: Iterable[Any], max_batches: int | None) -> Iterable[Any]:
    if max_batches is None:
        return loader

    def _generator() -> Iterable[Any]:
        for index, batch in enumerate(loader):
            if index >= int(max_batches):
                break
            yield batch

    return _generator()


def _evaluate_target_accuracy(model: DADDANet, loader: Iterable[Any], *, device: torch.device | str) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            x = _batch_tensor(batch, "iq", 0).to(device)
            y = _batch_tensor(batch, "label", 1).long().to(device)
            pred = model.classify(x).argmax(dim=1)
            correct += int((pred == y).sum().item())
            total += int(y.numel())
    return 0.0 if total == 0 else correct / float(total)


def _set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = float(lr)


def _sha256_file(path: Path | None) -> str | None:
    if path is None:
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _config_value(config: dict[str, Any], key: str, default: Any, cast: Any) -> Any:
    return cast(config.get(key, default))


def validate_paper_faithful_config(config: dict[str, Any]) -> None:
    if config.get("cvs_extension") is not False:
        raise ValueError("DADDA paper-faithful config must set cvs_extension=false")
    if config.get("target_labels_scope", "evaluation_only") != "evaluation_only":
        raise ValueError("target labels are evaluation-only in the paper-faithful UDA protocol")
    if config.get("normalize", True) is False and not config.get("allow_unnormalized_ablation", False):
        raise ValueError(
            "DADDA paper-faithful config must keep energy/RMS normalization enabled; "
            "set allow_unnormalized_ablation=true only for explicitly labeled ablations"
        )


def resolve_table2_run_settings(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    return {
        "epochs": int(args.epochs) if args.epochs is not None else _config_value(config, "epochs", 100, int),
        "batch_size": int(args.batch_size) if args.batch_size is not None else _config_value(config, "batch_size", 128, int),
        "learning_rate": float(args.learning_rate)
        if args.learning_rate is not None
        else 0.0001,
        "momentum": float(args.momentum) if args.momentum is not None else _config_value(config, "momentum", 0.9, float),
        "weight_decay": float(args.weight_decay)
        if args.weight_decay is not None
        else _config_value(config, "weight_decay", 0.0005, float),
        "paper_domain_sample_count": int(getattr(args, "paper_domain_sample_count", None))
        if getattr(args, "paper_domain_sample_count", None) is not None
        else config.get("paper_domain_sample_count"),
        "normalize": getattr(args, "normalize", None) if getattr(args, "normalize", None) is not None else config.get("normalize"),
        "crop_mode": getattr(args, "crop_mode", None) if getattr(args, "crop_mode", None) is not None else config.get("crop_mode"),
        "detach_target_probabilities": bool(
            getattr(args, "detach_target_probabilities", None)
            if getattr(args, "detach_target_probabilities", None) is not None
            else config.get("detach_target_probabilities", False)
        ),
        "alpha_mode": str(getattr(args, "alpha_mode", None) or config.get("alpha_mode", "dynamic")),
        "fixed_alpha": float(
            getattr(args, "fixed_alpha", None)
            if getattr(args, "fixed_alpha", None) is not None
            else config.get("fixed_alpha", 0.5)
        ),
    }


def validate_formal_or_smoke_settings(
    *,
    config: dict[str, Any],
    settings: dict[str, Any],
    smoke: bool,
    max_samples_per_combo: int | None,
    max_batches_per_epoch: int | None,
) -> None:
    paper_epochs = _config_value(config, "epochs", 100, int)
    if smoke:
        return
    if settings.get("normalize") is False and not config.get("allow_unnormalized_ablation", False):
        raise SystemExit("--no-normalize is not paper-faithful; use --smoke or set allow_unnormalized_ablation=true for an ablation")
    if int(settings["epochs"]) < paper_epochs:
        raise SystemExit("--smoke is required when --epochs is below the paper config epochs")
    if max_samples_per_combo is not None or max_batches_per_epoch is not None:
        raise SystemExit("--smoke is required when limiting samples or batches")


def _train_source_only(
    model: DADDANet,
    source_loader: Iterable[Any],
    *,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    device: torch.device | str,
    max_batches_per_epoch: int | None,
    base_lr: float,
) -> dict[str, Any]:
    history = []
    for epoch in range(int(epochs)):
        model.train()
        progress = epoch / float(max(1, int(epochs)))
        lr = learning_rate_schedule(progress, base_lr=base_lr)
        _set_optimizer_lr(optimizer, lr)
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
                "epoch": epoch + 1,
                "batches": batches,
                "lr": lr,
                "lambda": 0.0,
                "loss_mean": loss_sum / float(batches),
                "source_batch_acc_mean": acc_sum / float(batches),
            }
        )
    return {"history": history}


def run_dadda_training_loop(
    model: DADDANet,
    source_loader: Iterable[Any],
    target_loader: Iterable[Any],
    *,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    device: torch.device | str,
    max_batches_per_epoch: int | None = None,
    base_lr: float = 0.0001,
    bandwidth: float | None = None,
    detach_target_probabilities: bool = False,
    alpha_mode: str = "dynamic",
    fixed_alpha: float = 0.5,
) -> dict[str, Any]:
    history = []
    total_batches = 0
    for epoch in range(int(epochs)):
        model.train()
        progress = epoch / float(max(1, int(epochs)))
        lr = learning_rate_schedule(progress, base_lr=base_lr)
        tradeoff = lambda_schedule(progress)
        _set_optimizer_lr(optimizer, lr)
        batches = 0
        loss_sum = 0.0
        ce_sum = 0.0
        mmd_sum = 0.0
        lmmd_sum = 0.0
        alpha_sum = 0.0
        for source_batch, target_batch in zip(_limited_batches(source_loader, max_batches_per_epoch), target_loader):
            source_x = _batch_tensor(source_batch, "iq", 0).to(device)
            source_y = _batch_tensor(source_batch, "label", 1).long().to(device)
            target_x = _batch_tensor(target_batch, "iq", 0).to(device)
            optimizer.zero_grad(set_to_none=True)
            source_outputs = model(source_x)
            target_outputs = model(target_x)
            terms = dadda_objective(
                source_outputs,
                target_outputs,
                source_y,
                tradeoff_lambda=tradeoff,
                bandwidth=bandwidth,
                detach_target_probabilities=detach_target_probabilities,
                alpha_mode=alpha_mode,
                fixed_alpha=fixed_alpha,
            )
            terms["loss"].backward()
            optimizer.step()
            batches += 1
            total_batches += 1
            loss_sum += float(terms["loss"].detach().cpu())
            ce_sum += float(terms["cross_entropy"].cpu())
            mmd_sum += float(terms["mmd"].cpu())
            lmmd_sum += float(terms["lmmd"].cpu())
            alpha_sum += float(terms["alpha"].cpu())
        if batches == 0:
            raise ValueError("source/target batches cannot be empty")
        history.append(
            {
                "epoch": epoch + 1,
                "batches": batches,
                "lr": lr,
                "lambda": tradeoff,
                "loss_mean": loss_sum / float(batches),
                "cross_entropy_mean": ce_sum / float(batches),
                "mmd_mean": mmd_sum / float(batches),
                "lmmd_mean": lmmd_sum / float(batches),
                "alpha_mean": alpha_sum / float(batches),
            }
        )
    return {
        "algorithm": "DADDA Algorithm 1 smoke loop",
        "alpha_mode": alpha_mode,
        "fixed_alpha": float(fixed_alpha),
        "epochs": int(epochs),
        "batches": total_batches,
        "history": history,
    }


def _task_slug(task: str) -> str:
    return task.replace("->", "_to_").replace("/", "_")


def _make_model(config: dict[str, Any], *, device: torch.device) -> DADDANet:
    return DADDANet(
        num_classes=int(config.get("num_classes", 6)),
        feature_dim=int(config.get("feature_dim", 128)),
        multiscale_dim=int(config.get("multiscale_dim", 128)),
        base_channels=int(config.get("base_channels", 16)),
        classifier_hidden1=int(config.get("classifier_hidden1", 512)),
        classifier_hidden2=int(config.get("classifier_hidden2", 128)),
        model_variant=str(config.get("model_variant", "conv1d")),
    ).to(device)


def run_table2_reproduction(
    compact_or_path: dict[str, Any] | str | Path,
    *,
    tasks: list[str] | None = None,
    methods: list[str] | None = None,
    output_dir: Path | str,
    epochs: int,
    batch_size: int,
    learning_rate: float = 0.0001,
    momentum: float = 0.9,
    weight_decay: float = 0.0005,
    max_samples_per_combo: int | None = None,
    paper_domain_sample_count: int | None = None,
    normalize: bool | None = None,
    crop_mode: str | None = None,
    max_batches_per_epoch: int | None = None,
    seed: int = 0,
    device: torch.device | str | None = None,
    num_workers: int = 0,
    model_config: dict[str, Any] | None = None,
    detach_target_probabilities: bool = False,
    alpha_mode: str = "dynamic",
    fixed_alpha: float = 0.5,
    smoke: bool | None = None,
    config_path: Path | str | None = None,
) -> dict[str, Any]:
    set_seed(int(seed))
    resolved_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    requested_tasks = list(PAPER_TABLE2_TASKS if tasks is None else tasks)
    requested_methods = [method.lower() for method in (methods or ["source_only", "dadda"])]
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    dataset_path = Path(compact_or_path) if isinstance(compact_or_path, (str, Path)) else None
    compact = load_wisig_compact_pkl(str(compact_or_path)) if dataset_path is not None else compact_or_path
    is_smoke = bool(smoke) if smoke is not None else int(epochs) < 100 or max_samples_per_combo is not None or max_batches_per_epoch is not None
    rows = []
    for task in requested_tasks:
        loaders = build_manysig_task_loaders(
            compact,
            task=task,
            batch_size=batch_size,
            max_samples_per_combo=max_samples_per_combo,
            paper_domain_sample_count=paper_domain_sample_count,
            normalize=normalize,
            crop_mode=crop_mode,
            seed=seed,
            num_workers=num_workers,
        )
        for method in requested_methods:
            config = dict(model_config or {})
            task_seed = int(seed)
            set_seed(task_seed)
            model = _make_model(config, device=resolved_device)
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=float(learning_rate),
                momentum=float(momentum),
                weight_decay=float(weight_decay),
            )
            checkpoint_path = output_path / f"{_task_slug(task)}_{method}.pt"
            if method == "source_only":
                train_result = _train_source_only(
                    model,
                    loaders["source"],
                    optimizer=optimizer,
                    epochs=epochs,
                    device=resolved_device,
                    max_batches_per_epoch=max_batches_per_epoch,
                    base_lr=learning_rate,
                )
            elif method in {"dadda", "proposed"}:
                train_result = run_dadda_training_loop(
                    model,
                    loaders["source"],
                    loaders["target_train"],
                    optimizer=optimizer,
                    epochs=epochs,
                    device=resolved_device,
                    max_batches_per_epoch=max_batches_per_epoch,
                    base_lr=learning_rate,
                    detach_target_probabilities=detach_target_probabilities,
                    alpha_mode=alpha_mode,
                    fixed_alpha=fixed_alpha,
                )
            else:
                rows.append(
                    {
                        "task": task,
                        "method": method,
                        "status": "not_implemented",
                        "result_claim_status": "missing_required_paper_baseline"
                        if method in PAPER_TABLE2_METHODS
                        else "unsupported_method_not_paper_table2",
                        "paper_table2_required": method in PAPER_TABLE2_METHODS,
                        "missing_reason": "baseline runner is not implemented in this DADDA module",
                        "target_labels_scope": "evaluation_only",
                        "claim_blocks": list(CLAIM_BLOCKS),
                    }
                )
                continue
            torch.save(
                {
                    "paper": PAPER_TITLE,
                    "method": method,
                    "task": task,
                    "model_state_dict": model.state_dict(),
                    "history": train_result["history"],
                },
                checkpoint_path,
            )
            checkpoint_sha256 = _sha256_file(checkpoint_path)
            rows.append(
                {
                    "task": task,
                    "method": method,
                    "status": "completed",
                    "result_claim_status": "smoke_only_not_paper_formal" if is_smoke else "formal_run_partial_table2_requires_missing_baselines",
                    "target_accuracy": _evaluate_target_accuracy(model, loaders["target_eval"], device=resolved_device),
                    "target_labels_scope": "evaluation_only",
                    "target_label_role": loaders["meta"]["target_label_role"],
                    "checkpoint_path": str(checkpoint_path),
                    "checkpoint_sha256": checkpoint_sha256,
                    "history": train_result["history"],
                    "task_meta": loaders["meta"],
                    "source_sample_count": len(loaders["source"].dataset),
                    "target_train_sample_count": len(loaders["target_train"].dataset),
                    "target_eval_sample_count": len(loaders["target_eval"].dataset),
                    "source_loader_batches": len(loaders["source"]),
                    "target_train_loader_batches": len(loaders["target_train"]),
                    "task_seed": task_seed,
                    "drop_last": False,
                    "partial_batch_policy": "DataLoader drop_last=False; Algorithm 1 pairing stops at the shorter source/target stream",
                }
            )
    completed_methods_by_task: dict[str, list[str]] = {}
    for row in rows:
        if row["status"] == "completed":
            completed_methods_by_task.setdefault(str(row["task"]), []).append(str(row["method"]))
    requested_task_set = set(requested_tasks)
    expected_task_set = set(PAPER_TABLE2_TASKS)
    return {
        "method_id": "dadda_cross_receiver",
        "paper": PAPER_TITLE,
        "artifact_type": "table2_reproduction_run",
        "paper_scope": "paper_faithful_closed_set_single_source_UDA",
        "cvs_extension": False,
        "not_cvs_stage2": True,
        "not_leo_deployment_evidence": True,
        "not_open_set_evidence": True,
        "result_claim_status": "smoke_only_not_paper_formal" if is_smoke else "formal_run_partial_table2_requires_missing_baselines",
        "expected_table2_tasks": len(PAPER_TABLE2_TASKS),
        "requested_task_count": len(requested_tasks),
        "completed_task_count": len(completed_methods_by_task),
        "missing_task_ids": [task for task in PAPER_TABLE2_TASKS if task not in requested_task_set],
        "unexpected_task_ids": [task for task in requested_tasks if task not in expected_task_set],
        "completed_method_ids_by_task": completed_methods_by_task,
        "paper_table2_required_methods": list(PAPER_TABLE2_METHODS),
        "implemented_methods": list(IMPLEMENTED_TABLE2_METHODS),
        "not_implemented_paper_baselines": [method for method in PAPER_TABLE2_METHODS if method not in {"source_only", "dadda"}],
        "paper_evidence_targets": dict(PAPER_EVIDENCE_TARGETS),
        "pending_paper_artifacts": build_pending_paper_artifacts(),
        "claim_blocks": list(CLAIM_BLOCKS),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "optimizer": "SGD",
        "learning_rate": float(learning_rate),
        "momentum": float(momentum),
        "weight_decay": float(weight_decay),
        "lr_schedule": "lr_p=0.0001/(1+10p)^0.75",
        "lambda_schedule": "lambda_p=2/(1+exp(-10p))-1",
        "max_samples_per_combo": max_samples_per_combo,
        "paper_domain_sample_count": paper_domain_sample_count,
        "normalize": normalize,
        "crop_mode": crop_mode,
        "detach_target_probabilities": bool(detach_target_probabilities),
        "alpha_mode": alpha_mode,
        "fixed_alpha": float(fixed_alpha),
        "max_batches_per_epoch": max_batches_per_epoch,
        "config_path": str(config_path) if config_path is not None else None,
        "config_sha256": _sha256_file(Path(config_path)) if config_path is not None else None,
        "dataset_path": str(dataset_path) if dataset_path is not None else None,
        "dataset_sha256": _sha256_file(dataset_path),
        "drop_last": False,
        "partial_batch_policy": "DataLoader drop_last=False; Algorithm 1 pairing stops at the shorter source/target stream",
        "seed": int(seed),
        "seed_policy": "reset base seed before each task/method model initialization so Table II rows are independent of lane order",
        "device": str(resolved_device),
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Paper-faithful DADDA cross-receiver RFFI reproduction entrypoint.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--formal", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--run-table2", action="store_true")
    parser.add_argument("--plan-paper-artifacts", action="store_true")
    parser.add_argument("--manysig-pkl", type=Path, default=None)
    parser.add_argument("--tasks", type=str, default="")
    parser.add_argument("--methods", type=str, default="source_only,dadda")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--momentum", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--max-samples-per-combo", type=int, default=None)
    parser.add_argument("--paper-domain-sample-count", type=int, default=None)
    parser.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--crop-mode", type=str, default=None, choices=["left", "center"])
    parser.add_argument("--detach-target-probabilities", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--alpha-mode", type=str, default=None, choices=["dynamic", "fixed"])
    parser.add_argument("--fixed-alpha", type=float, default=None)
    parser.add_argument("--max-batches-per-epoch", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("paper_reproduction/runs/dadda_cross_receiver"))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    config = load_json_config(args.config)
    validate_paper_faithful_config(config)
    if args.dry_run:
        payload = build_dry_run_payload(config)
        if args.output is not None:
            write_json(args.output, payload)
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        return 0
    if args.plan_paper_artifacts:
        payload = build_paper_artifact_plan()
        if args.output is not None:
            write_json(args.output, payload)
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        return 0
    if args.run_table2:
        if not args.formal:
            raise SystemExit("--formal is required for real DADDA Table II runs")
        if args.manysig_pkl is None:
            raise SystemExit("--manysig-pkl is required with --run-table2")
        settings = resolve_table2_run_settings(config, args)
        validate_formal_or_smoke_settings(
            config=config,
            settings=settings,
            smoke=args.smoke,
            max_samples_per_combo=args.max_samples_per_combo,
            max_batches_per_epoch=args.max_batches_per_epoch,
        )
        tasks = [token.strip() for token in args.tasks.split(",") if token.strip()] or None
        methods = [token.strip() for token in args.methods.split(",") if token.strip()]
        model_config = dict(config.get("model_config", {}))
        if "model_variant" in config:
            model_config["model_variant"] = config["model_variant"]
        payload = run_table2_reproduction(
            args.manysig_pkl,
            tasks=tasks,
            methods=methods,
            output_dir=args.checkpoint_dir,
            epochs=settings["epochs"],
            batch_size=settings["batch_size"],
            learning_rate=settings["learning_rate"],
            momentum=settings["momentum"],
            weight_decay=settings["weight_decay"],
            max_samples_per_combo=args.max_samples_per_combo,
            paper_domain_sample_count=settings["paper_domain_sample_count"],
            normalize=settings["normalize"],
            crop_mode=settings["crop_mode"],
            max_batches_per_epoch=args.max_batches_per_epoch,
            seed=args.seed,
            device=args.device,
            num_workers=args.num_workers,
            smoke=args.smoke,
            config_path=args.config,
            model_config=model_config,
            detach_target_probabilities=settings["detach_target_probabilities"],
            alpha_mode=settings["alpha_mode"],
            fixed_alpha=settings["fixed_alpha"],
        )
        if args.output is not None:
            write_json(args.output, payload)
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        return 0
    raise SystemExit("use --dry-run, or --formal --run-table2 with --manysig-pkl after local verification")


if __name__ == "__main__":
    raise SystemExit(main())
