from __future__ import annotations

import argparse
import copy
import csv
import json
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from paper_reproduction.common.config import load_json_config
from paper_reproduction.common.wisig_runtime import load_wisig_compact_pkl, set_seed, write_json
from paper_reproduction.cvs_aligned.class_incremental import (
    _compact_source_labels,
    _cycle_batches,
    _detailed_breakdown,
    _source_loader,
    _trace_loss,
)
from paper_reproduction.cvs_aligned.evaluate import (
    _apply_scenario,
    _build_stage2_tensors,
    _select_target_sets,
    _train_model,
)
from paper_reproduction.cvs_aligned.supervised_da import (
    dadda_sda_objective,
    mrior_sda_objective,
    validate_supervised_da_manifest,
)
from paper_reproduction.DADDA.model import DADDANet
from paper_reproduction.mitigating_receiver_impact_da.model import ReceiverImpactGADNet


METHODS = {"protonet_cda", "mrior_sda", "dadda_sda"}


def _parametric_optimizer(
    config: dict[str, Any],
    model: nn.Module,
    *,
    method: str,
    phase: str,
) -> tuple[torch.optim.Optimizer, dict[str, Any]]:
    prefix = f"{method.removesuffix('_sda')}_{phase}"
    if method == "dadda_sda":
        learning_rate = float(config.get(f"{prefix}_learning_rate", 1.0e-4))
        momentum = float(config.get("dadda_momentum", 0.9))
        weight_decay = float(config.get("dadda_weight_decay", 5.0e-4))
        optimizer = torch.optim.SGD(
            model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay
        )
        profile = {
            "optimizer": "SGD",
            "learning_rate": learning_rate,
            "momentum": momentum,
            "weight_decay": weight_decay,
            "schedule": "inverse_(1+10p)^-0.75",
        }
    elif method == "mrior_sda":
        learning_rate = float(config.get(f"{prefix}_learning_rate", 6.0e-4))
        weight_decay = float(config.get("mrior_weight_decay", 0.0))
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        profile = {
            "optimizer": "Adam",
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "schedule": "constant",
        }
    else:
        raise ValueError(f"no parametric optimizer for {method}")
    return optimizer, profile


def _set_method_learning_rate(
    optimizer: torch.optim.Optimizer,
    profile: dict[str, Any],
    *,
    step: int,
    total_steps: int,
) -> float:
    base_lr = float(profile["learning_rate"])
    if str(profile["schedule"]).startswith("inverse_"):
        progress = float(step - 1) / float(max(1, total_steps - 1))
        learning_rate = base_lr / ((1.0 + 10.0 * progress) ** 0.75)
    else:
        learning_rate = base_lr
    for group in optimizer.param_groups:
        group["lr"] = learning_rate
    return learning_rate


def _validate_config(config: dict[str, Any]) -> None:
    method = str(config.get("method_id", "")).lower()
    if method not in METHODS:
        raise ValueError(f"method_id must be one of {sorted(METHODS)}")
    if str(config.get("stage", "")) not in {"Stage2-B", "B"}:
        raise ValueError("supervised domain adaptation must use Stage2-B")
    if config.get("target_new_tx_labels") or config.get("target_unknown_tx_labels"):
        raise ValueError("Stage2-B supervised DA permits target-old classes only")
    if len(config.get("target_receiver_labels", [])) != 1:
        raise ValueError("each Stage2-B run must adapt exactly one target receiver")
    if int(config.get("base_steps", 0)) <= 0:
        raise ValueError("base_steps must be positive")
    if int(config.get("adapt_steps", 0)) <= 0 and method != "protonet_cda":
        raise ValueError("adapt_steps must be positive for MRIOR-SDA and DADDA-SDA")
    scenarios = [str(value) for value in config.get("target_channel_scenarios", [])]
    if not scenarios or any(value == "clean" for value in scenarios):
        raise ValueError("formal Stage2-B tests must use non-clean satellite scenarios")


def _nearest_prototype(
    support_features: torch.Tensor,
    support_labels: torch.Tensor,
    query_features: torch.Tensor,
) -> torch.Tensor:
    class_ids = torch.unique(support_labels, sorted=True)
    prototypes = torch.stack(
        [support_features[support_labels == class_id].mean(dim=0) for class_id in class_ids],
        dim=0,
    )
    return class_ids[torch.cdist(query_features.float(), prototypes.float()).argmin(dim=1)]


def _source_prototypes(
    model: nn.Module,
    loader: DataLoader,
    source_ids: list[int],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    features: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            features.append(model(batch["iq"].to(device)).detach().cpu())
            labels.append(_compact_source_labels(batch["label"], source_ids, device).cpu())
    return torch.cat(features, dim=0), torch.cat(labels, dim=0)


def _train_base(
    config: dict[str, Any],
    manysig: dict[str, Any],
    loader: DataLoader,
    source_ids: list[int],
    device: torch.device,
) -> tuple[nn.Module, dict[str, Any]]:
    method = str(config["method_id"]).lower()
    base_steps = int(config["base_steps"])
    if method == "protonet_cda":
        model, info = _train_model(
            {
                **config,
                "baseline": "protonet_cda",
                "max_steps": base_steps,
                "train_channel_view": str(config.get("source_train_channel_view", "clean")),
            },
            manysig,
            device,
        )
        return model, info
    if method == "mrior_sda":
        model = ReceiverImpactGADNet(
            num_tx=len(source_ids),
            feature_dim=int(config.get("feature_dim", 128)),
            hidden_dim=int(config.get("hidden_dim", 128)),
            model_profile=str(config.get("mrior_model_profile", "standard_resnet18")),
        ).to(device)
    else:
        model = DADDANet(
            num_classes=len(source_ids),
            feature_dim=int(config.get("feature_dim", 128)),
            multiscale_dim=int(config.get("multiscale_dim", 128)),
            base_channels=int(config.get("base_channels", 16)),
            model_variant=str(config.get("dadda_model_variant", "conv1d")),
        ).to(device)
    optimizer, optimizer_profile = _parametric_optimizer(
        config, model, method=method, phase="base"
    )
    last_loss = float("nan")
    loss_trace: list[dict[str, Any]] = []
    model.train()
    for step, batch in enumerate(_cycle_batches(loader, base_steps), start=1):
        learning_rate = _set_method_learning_rate(
            optimizer, optimizer_profile, step=step, total_steps=base_steps
        )
        x = batch["iq"].to(device)
        y = _compact_source_labels(batch["label"], source_ids, device)
        outputs = model(x)
        logits = outputs["tx_logits"] if method == "mrior_sda" else outputs["logits"]
        loss = F.cross_entropy(logits, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        last_loss = float(loss.detach().cpu())
        _trace_loss(
            loss_trace,
            {**config, "method": method, "_active_scenario": "source_base"},
            phase="base",
            step=step,
            total_steps=base_steps,
            losses={"loss": loss, "learning_rate": learning_rate},
        )
    return model, {
        "base_steps": base_steps,
        "base_final_source_ce": last_loss,
        "source_train_channel_view": str(config.get("source_train_channel_view", "clean")),
        "training_origin": "source_supervised_random_init",
        "optimizer_profile": optimizer_profile,
        "loss_trace": loss_trace,
    }


def _adapt_parametric(
    config: dict[str, Any],
    base_model: nn.Module,
    loader: DataLoader,
    source_ids: list[int],
    support_x: torch.Tensor,
    support_y: torch.Tensor,
    query_x: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    method = str(config["method_id"]).lower()
    model = copy.deepcopy(base_model).to(device)
    model.eval()
    with torch.no_grad():
        before_outputs = model(query_x.to(device))
        before_logits = (
            before_outputs["tx_logits"] if method == "mrior_sda" else before_outputs["logits"]
        )
        before = before_logits.argmax(dim=1).cpu()
    target_loader = DataLoader(
        TensorDataset(support_x, support_y),
        batch_size=min(int(config.get("target_batch_size", 64)), int(support_y.numel())),
        shuffle=True,
        generator=torch.Generator().manual_seed(int(config.get("seed", 1337))),
    )
    source_batches = _cycle_batches(loader, int(config["adapt_steps"]))
    target_batches = _cycle_batches(target_loader, int(config["adapt_steps"]))
    optimizer, optimizer_profile = _parametric_optimizer(
        config, model, method=method, phase="adapt"
    )
    last: dict[str, float] = {}
    loss_trace: list[dict[str, Any]] = []
    model.train()
    for step, (source_batch, (target_x, target_y)) in enumerate(
        zip(source_batches, target_batches), start=1
    ):
        learning_rate = _set_method_learning_rate(
            optimizer,
            optimizer_profile,
            step=step,
            total_steps=int(config["adapt_steps"]),
        )
        source_x = source_batch["iq"].to(device)
        source_y = _compact_source_labels(source_batch["label"], source_ids, device)
        target_x = target_x.to(device)
        target_y = target_y.to(device)
        if method == "mrior_sda":
            losses = mrior_sda_objective(
                model(source_x),
                model(target_x),
                source_labels=source_y,
                target_support_labels=target_y,
                target_ce_weight=float(config.get("target_ce_weight", 1.0)),
                dvkl_weight=float(config.get("dvkl_weight", 0.005)),
                mu=float(config.get("mrior_mu", 0.5)),
                class_balance_smoothing=float(config.get("class_balance_smoothing", 0.0)),
            )
        else:
            losses = dadda_sda_objective(
                model(source_x),
                model(target_x),
                source_labels=source_y,
                target_support_labels=target_y,
                target_ce_weight=float(config.get("target_ce_weight", 1.0)),
                alignment_weight=float(config.get("alignment_weight", 1.0)),
                bandwidth=config.get("bandwidth"),
            )
        optimizer.zero_grad(set_to_none=True)
        losses["loss"].backward()
        optimizer.step()
        last = {
            key: float(value.detach().cpu())
            for key, value in losses.items()
            if isinstance(value, torch.Tensor) and value.numel() == 1
        }
        _trace_loss(
            loss_trace,
            {**config, "method": method},
            phase="target_support_adaptation",
            step=step,
            total_steps=int(config["adapt_steps"]),
            losses={
                **{key: value for key, value in losses.items() if value.numel() == 1},
                "learning_rate": learning_rate,
            },
        )
    model.eval()
    with torch.no_grad():
        outputs = model(query_x.to(device))
        logits = outputs["tx_logits"] if method == "mrior_sda" else outputs["logits"]
        predicted = logits.argmax(dim=1).cpu()
    return predicted, before, {
        "adapt_steps": int(config["adapt_steps"]),
        "adaptation_objective": "mrior_gad_true_support_class_weighting+dvkl"
        if method == "mrior_sda"
        else "source_ce+target_support_ce+dynamic_mmd_lmmd",
        "final_adaptation_losses": last,
        "optimizer_profile": optimizer_profile,
        "loss_trace": loss_trace,
    }


def _accuracy(predicted: torch.Tensor, truth: torch.Tensor) -> float:
    return float((predicted.cpu() == truth.cpu()).float().mean().item())


def run(config: dict[str, Any], *, run_dir: Path, device: torch.device) -> dict[str, Any]:
    _validate_config(config)
    seed = int(config.get("seed", 1337))
    set_seed(seed)
    manysig = load_wisig_compact_pkl(str(config["manysig_pkl"]))
    manytx = load_wisig_compact_pkl(str(config["manytx_pkl"]))
    target_info = _select_target_sets(config, manysig, manytx)
    tensors = _build_stage2_tensors(config, manysig, manytx)
    if tensors["new_class_ids"] or tensors["support_query_overlap"]:
        raise ValueError("Stage2-B requires old-only, disjoint target support/query")
    loader, source_ids = _source_loader(config, manysig)
    base_model, base_info = _train_base(config, manysig, loader, source_ids, device)
    loss_trace_rows = list(base_info.pop("loss_trace", []))
    method = str(config["method_id"]).lower()
    source_features = source_labels = None
    if method == "protonet_cda":
        source_features, source_labels = _source_prototypes(base_model, loader, source_ids, device)

    scenarios = [str(value) for value in config["target_channel_scenarios"]]
    metrics_by_scenario: dict[str, dict[str, Any]] = {}
    predictions: dict[str, torch.Tensor] = {}
    detailed_rows: list[dict[str, Any]] = []
    for scenario_index, scenario in enumerate(scenarios):
        set_seed(seed)
        support_x = _apply_scenario(
            tensors["support_x"].to(device), scenario, seed=seed + 1000 + scenario_index
        ).cpu()
        query_x = _apply_scenario(
            tensors["query_x"].to(device), scenario, seed=seed + 2000 + scenario_index
        ).cpu()
        started = time.perf_counter()
        if method == "protonet_cda":
            base_model.eval()
            with torch.no_grad():
                support_z = base_model(support_x.to(device)).cpu()
                query_z = base_model(query_x.to(device)).cpu()
            predicted = _nearest_prototype(support_z, tensors["support_y"], query_z)
            before = _nearest_prototype(source_features, source_labels, query_z)
            method_info = {
                "adaptation_objective": "labeled_target_support_prototype_registration",
                "adapt_steps": 0,
            }
        else:
            predicted, before, method_info = _adapt_parametric(
                {**config, "_active_scenario": scenario},
                base_model,
                loader,
                source_ids,
                support_x,
                tensors["support_y"],
                query_x,
                device,
            )
        loss_trace_rows.extend(method_info.pop("loss_trace", []))
        elapsed = time.perf_counter() - started
        after_acc = _accuracy(predicted, tensors["query_y"])
        before_acc = _accuracy(before, tensors["query_y"])
        metrics_by_scenario[scenario] = {
            "target_old_accuracy": after_acc,
            "target_old_accuracy_before_adaptation": before_acc,
            "target_old_accuracy_delta": after_acc - before_acc,
            "adaptation_latency_sec": elapsed,
            "latency_per_query_ms": elapsed * 1000.0 / int(tensors["query_y"].numel()),
            **method_info,
        }
        predictions[scenario] = predicted
        detailed_rows.extend(
            _detailed_breakdown(predicted, tensors["query_y"], tensors["query_meta"], scenario=scenario)
        )

    support_ids = [row["sample_id"] for row in tensors["support_meta"]]
    query_ids = [row["sample_id"] for row in tensors["query_meta"]]
    manifest = validate_supervised_da_manifest(
        {
            **config,
            **target_info,
            "method_id": method,
            "stage": "Stage2-B",
            "cvs_extension": True,
            "target_old_support_sample_ids": support_ids,
            "target_old_query_sample_ids": query_ids,
            "target_labels_scope": "registered_support_only",
            "target_query_used_for_training": False,
            "target_query_used_for_model_selection": False,
        }
    )
    manifest.update(
        {
            "seed": seed,
            "split_seed": int(config.get("split_seed", seed)),
            "all_tests_satellite_augmented": all(value != "clean" for value in scenarios),
            "support_query_overlap": bool(set(support_ids) & set(query_ids)),
            "base_training": base_info,
        }
    )
    aggregate = {
        key + "_mean": float(sum(float(row[key]) for row in metrics_by_scenario.values()) / len(scenarios))
        for key in (
            "target_old_accuracy",
            "target_old_accuracy_before_adaptation",
            "target_old_accuracy_delta",
            "adaptation_latency_sec",
        )
    }
    result = {
        "experiment_id": config.get("experiment_id", f"{method}_cvs_stage2b_{seed}"),
        "method_id": method,
        "seed": seed,
        "target_receiver_label": config["target_receiver_labels"][0],
        "target_channel_scenarios": scenarios,
        "metrics": aggregate,
        "metrics_by_scenario": metrics_by_scenario,
        "detailed_result_rows": detailed_rows,
        "split_manifest": manifest,
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "metrics.json", result)
    write_json(run_dir / "split_manifest.json", manifest)
    write_json(run_dir / "resolved_config.json", config)
    write_json(run_dir / "detailed_metrics.json", detailed_rows)
    write_json(run_dir / "loss_trace.json", loss_trace_rows)
    with (run_dir / "detailed_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(detailed_rows[0].keys()))
        writer.writeheader()
        writer.writerows(detailed_rows)
    trace_fields = sorted({key for row in loss_trace_rows for key in row})
    with (run_dir / "loss_trace.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=trace_fields)
        writer.writeheader()
        writer.writerows(loss_trace_rows)
    with (run_dir / "score_table.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = [
            "sample_id", "receiver_label", "transmitter_label", "day_i", "sig_i", "role",
            "true_label", "predicted_label", "correct", "scenario",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for scenario, predicted in predictions.items():
            for meta, truth, prediction in zip(
                tensors["query_meta"], tensors["query_y"].tolist(), predicted.tolist()
            ):
                writer.writerow(
                    {
                        "sample_id": meta["sample_id"],
                        "receiver_label": meta.get("rx_label", ""),
                        "transmitter_label": meta.get("tx_label", ""),
                        "day_i": meta.get("day_i", ""),
                        "sig_i": meta.get("sig_i", ""),
                        "role": meta["role"],
                        "true_label": truth,
                        "predicted_label": prediction,
                        "correct": int(truth == prediction),
                        "scenario": scenario,
                    }
                )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="CVS Stage2-B supervised DA comparison runner")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--experiment-id", default=None)
    parser.add_argument("--method", choices=sorted(METHODS), default=None)
    parser.add_argument("--target-receiver", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--split-seed", type=int, default=None)
    parser.add_argument("--k-shot", type=int, default=None)
    parser.add_argument("--base-steps", type=int, default=None)
    parser.add_argument("--adapt-steps", type=int, default=None)
    args = parser.parse_args()
    config = load_json_config(args.config)
    overrides = {
        "experiment_id": args.experiment_id,
        "method_id": args.method,
        "seed": args.seed,
        "split_seed": args.split_seed,
        "k_shot": args.k_shot,
        "base_steps": args.base_steps,
        "adapt_steps": args.adapt_steps,
    }
    config.update({key: value for key, value in overrides.items() if value is not None})
    if args.target_receiver is not None:
        config["target_receiver_labels"] = [args.target_receiver]
    _validate_config(config)
    if args.dry_run:
        print(json.dumps(config, ensure_ascii=False, sort_keys=True))
        return 0
    result = run(config, run_dir=args.run_dir, device=torch.device(args.device))
    print(
        json.dumps(
            {
                "experiment_id": result["experiment_id"],
                "method_id": result["method_id"],
                "seed": result["seed"],
                "target_receiver_label": result["target_receiver_label"],
                "metrics": result["metrics"],
                "support_query_overlap": result["split_manifest"]["support_query_overlap"],
                "all_tests_satellite_augmented": result["split_manifest"]["all_tests_satellite_augmented"],
                "detailed_result_row_count": len(result["detailed_result_rows"]),
                "run_dir": str(args.run_dir),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
