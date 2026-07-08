from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from paper_reproduction.common.config import load_json_config
from paper_reproduction.common.wisig_runtime import make_loader, set_seed, write_json
from paper_reproduction.receiver_agnostic_twostage_uda.data import build_manysig_receiver_uda_datasets
from paper_reproduction.receiver_agnostic_twostage_uda.model import ReceiverAgnosticUDANet
from paper_reproduction.receiver_agnostic_twostage_uda.protocol import (
    build_receiver_ratio_plan,
    validate_paper_faithful_config,
)
from paper_reproduction.receiver_agnostic_twostage_uda.steps import (
    compose_fig8_finetune_batch,
    dann_stage1_train_step,
    fig8_finetune_train_step,
    lmmd_stage2_train_step,
    select_fig8_labeled_target_indices,
)

ROOT = Path(__file__).resolve().parents[2]
CODE_DIR = ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from dataset_wisig import load_wisig_compact_pkl


PAPER_TITLE = "Receiver-Agnostic Radio Frequency Fingerprinting Based on Two-stage Unsupervised Domain Adaptation and Fine-tuning"
CLAIM_BLOCKS = [
    "not CVS Stage2-C",
    "not satellite/LEO deployment evidence",
    "not open-set or new-class registration",
]


def build_dry_run_payload(config: dict) -> dict:
    checked = validate_paper_faithful_config(config)
    return {
        "artifact_type": "dry_run_only",
        "schema_version": 2,
        "paper": PAPER_TITLE,
        "scope": checked["claim_boundary"],
        "dataset": checked["dataset"],
        "formal_training_status": "blocked",
        "result_claim_status": "no_formal_metrics",
        "synthetic_smoke_allowed": True,
        "requires_real_manysig_for_fig7_table_i_fig8": True,
        "receiver_ratio_plan": build_receiver_ratio_plan(checked),
        "paper_unspecified_fields": checked["paper_unspecified_fields"],
        "preprocessing": checked["preprocessing"],
        "claim_blocks": CLAIM_BLOCKS,
    }


def _as_device(value: str) -> torch.device:
    if value.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(value)


def _loader(dataset, *, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    return make_loader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def _split_indices(n: int, *, eval_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    if n < 2:
        raise ValueError("formal training requires at least two target samples for adaptation/evaluation split")
    generator = torch.Generator().manual_seed(int(seed))
    perm = torch.randperm(int(n), generator=generator).tolist()
    eval_n = max(1, int(round(float(eval_fraction) * int(n))))
    eval_n = min(eval_n, int(n) - 1)
    eval_ids = sorted(int(i) for i in perm[:eval_n])
    adapt_ids = sorted(int(i) for i in perm[eval_n:])
    return adapt_ids, eval_ids


def _target_adapt_eval_indices(
    n: int,
    *,
    eval_fraction: float,
    seed: int,
    transductive_target_eval: bool,
) -> tuple[list[int], list[int], str]:
    if bool(transductive_target_eval):
        if n < 1:
            raise ValueError("formal training requires at least one target sample")
        ids = list(range(int(n)))
        return ids, ids, "transductive_all_target_unlabeled_for_UDA_and_eval"
    adapt_ids, eval_ids = _split_indices(n, eval_fraction=eval_fraction, seed=seed)
    return adapt_ids, eval_ids, "heldout_target_eval_split"


def _cycle(loader: DataLoader):
    while True:
        for batch in loader:
            yield batch


def _supervised_step(
    model: torch.nn.Module,
    batch: dict[str, Any],
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    iq = batch["iq"].to(device).float()
    labels = batch["label"].to(device).long()
    outputs = model(iq)
    loss = F.cross_entropy(outputs["tx_logits"], labels)
    loss.backward()
    optimizer.step()
    return {"loss": float(loss.detach().cpu())}


def _run_steps(step_fn, *, steps: int, progress_every: int) -> dict[str, float]:
    last: dict[str, float] = {}
    for step_idx in range(1, int(steps) + 1):
        last = step_fn()
        if progress_every > 0 and (step_idx == 1 or step_idx % int(progress_every) == 0 or step_idx == int(steps)):
            print(json.dumps({"event": "train_step", "step": step_idx, "steps": int(steps), **last}, sort_keys=True), flush=True)
    return last


def _csv_floats(value: str) -> list[float]:
    return [float(part.strip()) for part in str(value).split(",") if part.strip()]


def _csv_ints(value: str) -> list[int]:
    return [int(part.strip()) for part in str(value).split(",") if part.strip()]


def _csv_strings(value: str) -> list[str]:
    return [part.strip() for part in str(value).split(",") if part.strip()]


@torch.no_grad()
def _evaluate(model: torch.nn.Module, loader: DataLoader, *, device: torch.device) -> dict[str, Any]:
    model.eval()
    total = 0
    correct = 0
    by_rx: dict[str, dict[str, int]] = {}
    for batch in loader:
        iq = batch["iq"].to(device).float()
        labels = batch["label"].to(device).long()
        logits = model(iq)["tx_logits"]
        pred = logits.argmax(dim=1)
        ok = (pred == labels).detach().cpu()
        total += int(labels.numel())
        correct += int(ok.sum().item())
        for idx, meta in enumerate(batch.get("meta", [])):
            rx = str(meta.get("rx", meta.get("rx_i", "unknown")))
            slot = by_rx.setdefault(rx, {"correct": 0, "total": 0})
            slot["correct"] += int(ok[idx].item())
            slot["total"] += 1
    return {
        "accuracy": float(correct / max(1, total)),
        "correct": int(correct),
        "total": int(total),
        "per_target_receiver": {
            rx: {
                "accuracy": float(v["correct"] / max(1, v["total"])),
                "correct": int(v["correct"]),
                "total": int(v["total"]),
            }
            for rx, v in sorted(by_rx.items())
        },
    }


@torch.no_grad()
def _collect_logits_and_samples(model: torch.nn.Module, loader: DataLoader, *, device: torch.device) -> dict[str, torch.Tensor]:
    model.eval()
    logits_parts: list[torch.Tensor] = []
    iq_parts: list[torch.Tensor] = []
    label_parts: list[torch.Tensor] = []
    for batch in loader:
        iq = batch["iq"].to(device).float()
        logits_parts.append(model(iq)["tx_logits"].detach().cpu())
        iq_parts.append(batch["iq"].detach().cpu())
        label_parts.append(batch["label"].detach().cpu().long())
    return {
        "logits": torch.cat(logits_parts, dim=0),
        "iq": torch.cat(iq_parts, dim=0),
        "label": torch.cat(label_parts, dim=0),
    }


def _fit_base_model(
    method: str,
    *,
    loaders: dict[str, DataLoader],
    device: torch.device,
    num_tx: int,
    lr: float,
    weight_decay: float,
    supervised_steps: int,
    stage1_steps: int,
    stage2_steps: int,
    stage2_lr: float,
    reset_stage2_optimizer: bool,
    lmmd_lambda: float,
    lmmd_layers: str,
    progress_every: int,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    model = ReceiverAgnosticUDANet(num_tx=num_tx).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    source_iter = _cycle(loaders["source_train"])
    target_iter = _cycle(loaders["target_adapt"])
    train_meta: dict[str, Any] = {}

    if method == "source_only":
        train_meta["source_supervised_last"] = _run_steps(
            lambda: _supervised_step(model, next(source_iter), optimizer, device=device),
            steps=supervised_steps,
            progress_every=progress_every,
        )
    elif method == "target_labeled_upper":
        train_meta["target_labeled_supervised_last"] = _run_steps(
            lambda: _supervised_step(model, next(target_iter), optimizer, device=device),
            steps=supervised_steps,
            progress_every=progress_every,
        )
    elif method in {"dann", "dann_lmmd"}:
        train_meta["stage1_dann_last"] = _run_steps(
            lambda: dann_stage1_train_step(
                model,
                next(source_iter),
                next(target_iter),
                optimizer,
                domain_weight=1.0,
                grl_lambda=1.0,
                device=device,
            ),
            steps=stage1_steps,
            progress_every=progress_every,
        )
        if method == "dann_lmmd":
            stage2_optimizer = optimizer
            if reset_stage2_optimizer:
                stage2_optimizer = torch.optim.Adam(model.parameters(), lr=float(stage2_lr), weight_decay=float(weight_decay))
                train_meta["stage2_optimizer_policy"] = "reset_after_stage1"
            elif float(stage2_lr) != float(lr):
                for group in stage2_optimizer.param_groups:
                    group["lr"] = float(stage2_lr)
                train_meta["stage2_optimizer_policy"] = "reuse_after_stage1_lr_changed"
            else:
                train_meta["stage2_optimizer_policy"] = "reuse_after_stage1"
            train_meta["stage2_lmmd_last"] = _run_steps(
                lambda: lmmd_stage2_train_step(
                    model,
                    next(source_iter),
                    next(target_iter),
                    stage2_optimizer,
                    num_classes=num_tx,
                    lmmd_lambda=lmmd_lambda,
                    lmmd_layers=lmmd_layers,
                    device=device,
                ),
                steps=stage2_steps,
                progress_every=progress_every,
            )
    else:
        raise ValueError(f"unknown method: {method}")
    return model, train_meta


def _fit_stage1_dann_model(
    *,
    loaders: dict[str, DataLoader],
    device: torch.device,
    num_tx: int,
    lr: float,
    weight_decay: float,
    stage1_steps: int,
    progress_every: int,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    model = ReceiverAgnosticUDANet(num_tx=num_tx).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    source_iter = _cycle(loaders["source_train"])
    target_iter = _cycle(loaders["target_adapt"])
    train_meta = {
        "stage1_dann_last": _run_steps(
            lambda: dann_stage1_train_step(
                model,
                next(source_iter),
                next(target_iter),
                optimizer,
                domain_weight=1.0,
                grl_lambda=1.0,
                device=device,
            ),
            steps=stage1_steps,
            progress_every=progress_every,
        )
    }
    return model, train_meta


def _run_lmmd_grid(
    base_model: torch.nn.Module,
    *,
    loaders: dict[str, DataLoader],
    device: torch.device,
    num_tx: int,
    weight_decay: float,
    lambdas: list[float],
    layers: list[str],
    steps_options: list[int],
    lrs: list[float],
    progress_every: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for lmmd_layers in layers:
        for lmmd_lambda in lambdas:
            for stage2_steps in steps_options:
                for stage2_lr in lrs:
                    model = copy.deepcopy(base_model).to(device)
                    optimizer = torch.optim.Adam(model.parameters(), lr=float(stage2_lr), weight_decay=float(weight_decay))
                    source_iter = _cycle(loaders["source_train"])
                    target_iter = _cycle(loaders["target_adapt"])
                    train_last = _run_steps(
                        lambda: lmmd_stage2_train_step(
                            model,
                            next(source_iter),
                            next(target_iter),
                            optimizer,
                            num_classes=num_tx,
                            lmmd_lambda=float(lmmd_lambda),
                            lmmd_layers=str(lmmd_layers),
                            device=device,
                        ),
                        steps=int(stage2_steps),
                        progress_every=progress_every,
                    )
                    rows.append(
                        {
                            "method": "dann_lmmd_grid",
                            "lmmd_lambda": float(lmmd_lambda),
                            "lmmd_layers": str(lmmd_layers),
                            "stage2_steps": int(stage2_steps),
                            "stage2_lr": float(stage2_lr),
                            "optimizer_policy": "reset_after_stage1",
                            "train": {"stage2_lmmd_last": train_last},
                            "target_eval": _evaluate(model, loaders["target_eval"], device=device),
                        }
                    )
    return rows


def _run_fig8(
    base_model: torch.nn.Module,
    *,
    loaders: dict[str, DataLoader],
    device: torch.device,
    strategies: list[str],
    iterations: list[int],
    lr: float,
    weight_decay: float,
    source_replay_per_class: int,
    seed: int,
) -> list[dict[str, Any]]:
    target_cache = _collect_logits_and_samples(base_model, loaders["target_adapt"], device=device)
    source_cache = _collect_logits_and_samples(base_model, loaders["source_train"], device=device)
    curves: list[dict[str, Any]] = []
    max_iter = max(int(i) for i in iterations)
    report_points = set(int(i) for i in iterations)
    for strategy in strategies:
        selected = select_fig8_labeled_target_indices(
            target_cache["logits"],
            strategy=strategy,
            denominator=50,
            seed=seed,
        )
        model = copy.deepcopy(base_model).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
        batch = compose_fig8_finetune_batch(
            target_cache["iq"].to(device),
            target_cache["label"].to(device),
            selected["selected"].to(device),
            source_cache["iq"].to(device),
            source_cache["label"].to(device),
            source_replay_per_class=int(source_replay_per_class),
            seed=int(seed),
        )
        points: list[dict[str, Any]] = []
        if 0 in report_points:
            points.append({"iteration": 0, "target_eval": _evaluate(model, loaders["target_eval"], device=device)})
        for i in range(1, max_iter + 1):
            metrics = fig8_finetune_train_step(model, batch, optimizer, device=device)
            if i in report_points:
                points.append({"iteration": int(i), "train": metrics, "target_eval": _evaluate(model, loaders["target_eval"], device=device)})
        curves.append(
            {
                "strategy": strategy,
                "budget": int(selected["budget"]),
                "budget_denominator": 50,
                "source_replay_per_class": int(source_replay_per_class),
                "points": points,
            }
        )
    return curves


def _write_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def run_formal(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    checked = validate_paper_faithful_config(config)
    manysig_pkl = args.manysig_pkl or config.get("manysig_pkl")
    if not manysig_pkl:
        raise SystemExit("formal training requires --manysig-pkl or config.manysig_pkl")
    set_seed(int(args.seed))
    compact = load_wisig_compact_pkl(str(manysig_pkl))
    rx_labels = list(compact.get("rx_list", []))
    if len(rx_labels) != int(config.get("total_receivers", 12)):
        raise ValueError(f"expected 12 ManySig receivers, got {len(rx_labels)}")

    ratio_counts = [int(x) for x in (args.source_receiver_counts or config.get("source_receiver_counts", []))]
    if args.limit_ratios > 0:
        ratio_counts = ratio_counts[: int(args.limit_ratios)]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    fig8_strategies = _csv_strings(args.fig8_strategies)
    fig8_iterations = _csv_ints(args.fig8_iterations)
    lmmd_grid_lambdas = _csv_floats(args.lmmd_grid_lambdas)
    lmmd_grid_layers = _csv_strings(args.lmmd_grid_layers)
    lmmd_grid_steps = _csv_ints(args.lmmd_grid_steps)
    lmmd_grid_lrs = _csv_floats(args.lmmd_grid_lrs)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "results.jsonl"
    summary_path = output_dir / "summary.json"
    if jsonl_path.exists() and not args.resume:
        jsonl_path.unlink()
    device = _as_device(args.device)
    rows: list[dict[str, Any]] = []
    completed_keys: set[tuple[int, str]] = set()
    if args.resume and jsonl_path.exists():
        for line in jsonl_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            completed_keys.add((int(row["source_receiver_count"]), str(row["method"])))
            rows.append(row)

    for source_count in ratio_counts:
        source_ids = list(range(int(source_count)))
        target_ids = list(range(int(source_count), len(rx_labels)))
        if not target_ids:
            raise ValueError("target receiver set cannot be empty")
        row_seed = int(args.seed) + int(source_count) * 1009
        datasets = build_manysig_receiver_uda_datasets(
            compact,
            source_receivers=source_ids,
            target_receivers=target_ids,
            seed=row_seed,
            max_samples_per_combo=args.max_samples_per_combo,
        )
        target_adapt_ids, target_eval_ids, target_eval_protocol = _target_adapt_eval_indices(
            len(datasets["target"]),
            eval_fraction=float(args.target_eval_fraction),
            seed=row_seed,
            transductive_target_eval=bool(args.transductive_target_eval),
        )
        loaders = {
            "source_train": _loader(datasets["source"], batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers),
            "target_adapt": _loader(Subset(datasets["target"], target_adapt_ids), batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers),
            "target_eval": _loader(Subset(datasets["target"], target_eval_ids), batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers),
        }
        source_steps = int(args.source_epochs) * max(1, math.ceil(len(datasets["source"]) / int(args.batch_size)))
        target_steps = int(args.source_epochs) * max(1, math.ceil(len(target_adapt_ids) / int(args.batch_size)))
        supervised_steps_by_method = {
            "source_only": source_steps,
            "target_labeled_upper": target_steps,
        }
        stage1_steps = int(args.stage1_epochs) * max(1, min(len(loaders["source_train"]), len(loaders["target_adapt"])))
        stage2_steps = int(args.stage2_epochs) * max(1, min(len(loaders["source_train"]), len(loaders["target_adapt"])))
        if args.max_train_steps > 0:
            supervised_steps_by_method = {
                key: max(1, min(int(value), int(args.max_train_steps))) for key, value in supervised_steps_by_method.items()
            }
            stage1_steps = max(1, min(stage1_steps, int(args.max_train_steps)))
            stage2_steps = max(1, min(stage2_steps, int(args.max_train_steps)))
        if args.max_stage2_steps > 0:
            stage2_steps = max(1, min(stage2_steps, int(args.max_stage2_steps)))
        print(
            json.dumps(
                {
                    "event": "ratio_start",
                    "source_receiver_count": int(source_count),
                    "target_receiver_count": len(target_ids),
                    "source_receivers": datasets["meta"]["source_receiver_labels"],
                    "target_receivers": datasets["meta"]["target_receiver_labels"],
                    "source_size": len(datasets["source"]),
                    "target_adapt_size": len(target_adapt_ids),
                    "target_eval_size": len(target_eval_ids),
                    "target_eval_protocol": target_eval_protocol,
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            flush=True,
        )
        if args.lmmd_grid:
            base_model, base_train_meta = _fit_stage1_dann_model(
                loaders=loaders,
                device=device,
                num_tx=int(config.get("tx_count", 6)),
                lr=float(args.lr),
                weight_decay=float(args.weight_decay),
                stage1_steps=stage1_steps,
                progress_every=int(args.progress_every),
            )
            base_row: dict[str, Any] = {
                "artifact_type": "formal_training_result",
                "paper_scope": checked["claim_boundary"],
                "paper": PAPER_TITLE,
                "dataset": checked["dataset"],
                "method": "dann_grid_base",
                "source_receiver_count": int(source_count),
                "target_receiver_count": len(target_ids),
                "source_receiver_ids": source_ids,
                "target_receiver_ids": target_ids,
                "source_receiver_labels": datasets["meta"]["source_receiver_labels"],
                "target_receiver_labels": datasets["meta"]["target_receiver_labels"],
                "preprocessing": datasets["meta"]["preprocessing"],
                "target_adapt_size": len(target_adapt_ids),
                "target_eval_size": len(target_eval_ids),
                "target_eval_protocol": target_eval_protocol,
                "seed": row_seed,
                "hyperparameters": {
                    "batch_size": int(args.batch_size),
                    "eval_batch_size": int(args.eval_batch_size),
                    "lr": float(args.lr),
                    "weight_decay": float(args.weight_decay),
                    "stage1_epochs": int(args.stage1_epochs),
                    "stage1_steps": int(stage1_steps),
                    "target_eval_fraction": float(args.target_eval_fraction),
                    "transductive_target_eval": bool(args.transductive_target_eval),
                    "lmmd_grid": True,
                    "paper_status": "paper-unspecified LMMD lambda/layer/optimizer choices; grid rows are diagnostics",
                },
                "train": base_train_meta,
                "target_eval": _evaluate(base_model, loaders["target_eval"], device=device),
                "claim_blocks": CLAIM_BLOCKS,
            }
            _write_jsonl(jsonl_path, base_row)
            rows.append(base_row)
            print(
                json.dumps(
                    {
                        "event": "row_complete",
                        "method": base_row["method"],
                        "source_receiver_count": int(source_count),
                        "target_accuracy": base_row["target_eval"]["accuracy"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            grid_results = _run_lmmd_grid(
                base_model,
                loaders=loaders,
                device=device,
                num_tx=int(config.get("tx_count", 6)),
                weight_decay=float(args.weight_decay),
                lambdas=lmmd_grid_lambdas,
                layers=lmmd_grid_layers,
                steps_options=lmmd_grid_steps,
                lrs=lmmd_grid_lrs,
                progress_every=int(args.progress_every),
            )
            for grid in grid_results:
                row = {
                    **base_row,
                    "method": grid["method"],
                    "variant_id": (
                        f"layers={grid['lmmd_layers']}|lambda={grid['lmmd_lambda']}"
                        f"|steps={grid['stage2_steps']}|lr={grid['stage2_lr']}"
                    ),
                    "hyperparameters": {
                        **base_row["hyperparameters"],
                        "lmmd_lambda": grid["lmmd_lambda"],
                        "lmmd_layers": grid["lmmd_layers"],
                        "stage2_steps": grid["stage2_steps"],
                        "stage2_lr": grid["stage2_lr"],
                        "optimizer_policy": grid["optimizer_policy"],
                    },
                    "train": {**base_train_meta, **grid["train"]},
                    "target_eval": grid["target_eval"],
                }
                _write_jsonl(jsonl_path, row)
                rows.append(row)
                print(
                    json.dumps(
                        {
                            "event": "row_complete",
                            "method": row["method"],
                            "variant_id": row["variant_id"],
                            "source_receiver_count": int(source_count),
                            "target_accuracy": row["target_eval"]["accuracy"],
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            continue
        for method in methods:
            if (int(source_count), method) in completed_keys:
                continue
            model, train_meta = _fit_base_model(
                method,
                loaders=loaders,
                device=device,
                num_tx=int(config.get("tx_count", 6)),
                lr=float(args.lr),
                weight_decay=float(args.weight_decay),
                supervised_steps=supervised_steps_by_method.get(method, source_steps),
                stage1_steps=stage1_steps,
                stage2_steps=stage2_steps,
                stage2_lr=float(args.stage2_lr),
                reset_stage2_optimizer=bool(args.reset_stage2_optimizer),
                lmmd_lambda=float(args.lmmd_lambda),
                lmmd_layers=str(args.lmmd_layers),
                progress_every=int(args.progress_every),
            )
            row: dict[str, Any] = {
                "artifact_type": "formal_training_result",
                "paper_scope": checked["claim_boundary"],
                "paper": PAPER_TITLE,
                "dataset": checked["dataset"],
                "method": method,
                "source_receiver_count": int(source_count),
                "target_receiver_count": len(target_ids),
                "source_receiver_ids": source_ids,
                "target_receiver_ids": target_ids,
                "source_receiver_labels": datasets["meta"]["source_receiver_labels"],
                "target_receiver_labels": datasets["meta"]["target_receiver_labels"],
                "preprocessing": datasets["meta"]["preprocessing"],
                "target_adapt_size": len(target_adapt_ids),
                "target_eval_size": len(target_eval_ids),
                "target_eval_protocol": target_eval_protocol,
                "seed": row_seed,
                "hyperparameters": {
                    "batch_size": int(args.batch_size),
                    "eval_batch_size": int(args.eval_batch_size),
                    "lr": float(args.lr),
                    "weight_decay": float(args.weight_decay),
                    "source_epochs": int(args.source_epochs),
                    "stage1_epochs": int(args.stage1_epochs),
                    "stage2_epochs": int(args.stage2_epochs),
                    "stage2_steps": int(stage2_steps),
                    "stage2_lr": float(args.stage2_lr),
                    "reset_stage2_optimizer": bool(args.reset_stage2_optimizer),
                    "target_eval_fraction": float(args.target_eval_fraction),
                    "transductive_target_eval": bool(args.transductive_target_eval),
                    "lmmd_lambda": float(args.lmmd_lambda),
                    "lmmd_layers": str(args.lmmd_layers),
                    "max_samples_per_combo": args.max_samples_per_combo,
                    "paper_status": "paper-unspecified optimizer/batch/epoch choices; recorded for reproducibility",
                },
                "train": train_meta,
                "target_eval": _evaluate(model, loaders["target_eval"], device=device),
                "claim_blocks": CLAIM_BLOCKS,
            }
            if args.run_fig8 and method == "dann_lmmd" and int(source_count) < 4:
                row["fig8_finetune"] = _run_fig8(
                    model,
                    loaders=loaders,
                    device=device,
                    strategies=fig8_strategies,
                    iterations=fig8_iterations,
                    lr=float(args.finetune_lr),
                    weight_decay=float(args.weight_decay),
                    source_replay_per_class=int(args.source_replay_per_class),
                    seed=row_seed,
                )
            if not args.no_save_checkpoints:
                ckpt_dir = output_dir / "checkpoints"
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                ckpt_path = ckpt_dir / f"source{int(source_count)}_{method}.pt"
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "method": method,
                        "source_receiver_count": int(source_count),
                        "target_receiver_count": len(target_ids),
                        "source_receiver_labels": datasets["meta"]["source_receiver_labels"],
                        "target_receiver_labels": datasets["meta"]["target_receiver_labels"],
                        "seed": row_seed,
                        "hyperparameters": row["hyperparameters"],
                    },
                    ckpt_path,
                )
                row["checkpoint"] = str(ckpt_path)
            _write_jsonl(jsonl_path, row)
            rows.append(row)
            print(
                json.dumps(
                    {
                        "event": "row_complete",
                        "method": method,
                        "source_receiver_count": int(source_count),
                        "target_accuracy": row["target_eval"]["accuracy"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    summary = {
        "artifact_type": "formal_training_summary",
        "paper_scope": checked["claim_boundary"],
        "dataset": checked["dataset"],
        "manysig_pkl": str(manysig_pkl),
        "output_dir": str(output_dir),
        "results_jsonl": str(jsonl_path),
        "rows": rows,
        "table_i_rows": [
            row
            for row in rows
            if int(row.get("source_receiver_count", -1)) == 6 and row.get("method") == "dann_lmmd"
        ],
        "claim_blocks": CLAIM_BLOCKS,
    }
    write_json(summary_path, summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Paper-faithful entrypoint for Bao et al. two-stage UDA RFFI.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true", help="Validate config and print the reproduction matrix.")
    parser.add_argument("--formal", action="store_true", help="Run formal WiSig/ManySig paper-faithful training.")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path for dry-run payload.")
    parser.add_argument("--output-dir", type=Path, default=Path("local_artifacts/receiver_agnostic_twostage_uda_formal"))
    parser.add_argument("--manysig-pkl", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260708)
    parser.add_argument("--source-receiver-counts", nargs="*", type=int, default=None)
    parser.add_argument("--limit-ratios", type=int, default=0)
    parser.add_argument("--methods", type=str, default="source_only,target_labeled_upper,dann,dann_lmmd")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--source-epochs", type=int, default=20)
    parser.add_argument("--stage1-epochs", type=int, default=20)
    parser.add_argument("--stage2-epochs", type=int, default=10)
    parser.add_argument("--stage2-lr", type=float, default=1e-3)
    parser.add_argument("--reset-stage2-optimizer", action="store_true")
    parser.add_argument("--max-train-steps", type=int, default=0)
    parser.add_argument("--max-stage2-steps", type=int, default=0)
    parser.add_argument("--max-samples-per-combo", type=int, default=None)
    parser.add_argument("--target-eval-fraction", type=float, default=0.5)
    parser.add_argument(
        "--transductive-target-eval",
        action="store_true",
        help="Use all target receiver samples as unlabeled UDA data and evaluate on the same target pool, matching transductive UDA reporting.",
    )
    parser.add_argument("--lmmd-lambda", type=float, default=1.0)
    parser.add_argument(
        "--lmmd-layers",
        choices=["activations", "features", "features_and_activations"],
        default="activations",
    )
    parser.add_argument("--lmmd-grid", action="store_true", help="Train one DANN base model and evaluate a reset-optimizer LMMD diagnostic grid.")
    parser.add_argument("--lmmd-grid-lambdas", type=str, default="0.005,0.01,0.02")
    parser.add_argument("--lmmd-grid-layers", type=str, default="features,activations")
    parser.add_argument("--lmmd-grid-steps", type=str, default="500,1000")
    parser.add_argument("--lmmd-grid-lrs", type=str, default="0.0001")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--run-fig8", action="store_true")
    parser.add_argument("--fig8-strategies", type=str, default="random,entropy")
    parser.add_argument("--fig8-iterations", type=str, default="0,25,50,75,100")
    parser.add_argument("--finetune-lr", type=float, default=1e-4)
    parser.add_argument("--source-replay-per-class", type=int, default=20)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-save-checkpoints", action="store_true")
    args = parser.parse_args()

    config = load_json_config(args.config)
    if args.formal:
        summary = run_formal(config, args)
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
        return 0
    if not args.dry_run:
        raise SystemExit("choose --dry-run or --formal")
    payload = build_dry_run_payload(config)
    if args.output is not None:
        write_json(args.output, payload)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
