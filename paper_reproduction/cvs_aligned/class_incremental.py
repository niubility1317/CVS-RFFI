"""CVS Stage2-C adapters for CSIL, MoPC-HR, and Orthogonal Incremental SEI.

This module deliberately keeps paper-faithful source reproductions separate.
Every result emitted here is a ``cvs_extension`` using registered target-old
and target-new support samples from one target receiver domain.
"""

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
from torch.utils.data import DataLoader

from paper_reproduction.common.config import load_json_config
from paper_reproduction.common.wisig_runtime import collate_wisig, load_wisig_compact_pkl, set_seed, write_json
from paper_reproduction.CSIL.losses import compute_csil_loss
from paper_reproduction.CSIL.model import CSILClassifier, csil_masked_sgd_step
from paper_reproduction.cvs_aligned.evaluate import (
    _apply_scenario,
    _build_stage2_tensors,
    _make_source_dataset,
    _select_target_sets,
)
from paper_reproduction.cvs_aligned.protocol import validate_stage2_protocol_payload
from paper_reproduction.mopc_hr_non_exemplar_cil_sei import (
    compute_class_prototypes,
    correct_old_prototypes,
    mopc_hr_incremental_objective,
    prototype_augmentation,
)
from paper_reproduction.orthogonal_incremental_sei.losses import (
    base_training_loss,
    incremental_calibration_loss,
)
from paper_reproduction.orthogonal_incremental_sei.model import SixBlockConv1DEncoder
from paper_reproduction.orthogonal_incremental_sei.pseudo_targets import (
    assign_base_targets,
    make_simplex_pseudo_targets,
    perturb_pseudo_targets,
)


METHODS = {"csil", "mopc_hr", "orthogonal_incremental"}


def validate_class_incremental_manifest(payload: dict[str, Any]) -> dict[str, Any]:
    checked = validate_stage2_protocol_payload({**payload, "stage": "Stage2-C"})
    method = str(payload.get("method", "")).lower()
    if method not in METHODS:
        raise ValueError(f"method must be one of {sorted(METHODS)}")
    if bool(payload.get("unknown_rejection_enabled", False)):
        raise ValueError("Stage2-C comparison excludes unknown/open-set rejection")
    if payload.get("target_unknown_tx_labels"):
        raise ValueError("Stage2-C comparison must not include unknown TX labels")
    if int(payload.get("query_per_tx", 0)) <= 0:
        raise ValueError("query_per_tx must be positive")
    if int(payload.get("base_steps", 0)) <= 0 or int(payload.get("increment_steps", 0)) <= 0:
        raise ValueError("base_steps and increment_steps must be positive")
    checked.update(
        {
            "method": method,
            "cvs_extension": True,
            "claim_boundary": "cvs_stage2c_supervised_class_incremental_extension",
            "target_labels_scope": "registered_old_and_new_support_only",
            "query_used_for_training": False,
            "query_used_for_model_selection": False,
            "unknown_rejection_enabled": False,
        }
    )
    return checked


def _compact_source_labels(labels: torch.Tensor, source_ids: list[int], device: torch.device) -> torch.Tensor:
    mapping = {int(value): index for index, value in enumerate(source_ids)}
    try:
        compact = [mapping[int(value)] for value in labels.detach().cpu().tolist()]
    except KeyError as exc:
        raise ValueError(f"source batch contains TX outside target-old set: {exc}") from exc
    return torch.tensor(compact, dtype=torch.long, device=device)


def _source_loader(config: dict[str, Any], manysig: dict[str, Any]) -> tuple[DataLoader, list[int]]:
    dataset = _make_source_dataset(config, manysig)
    tx_lookup = {str(label): index for index, label in enumerate(manysig.get("tx_list", []))}
    source_ids = [tx_lookup[str(label)] for label in config["target_old_tx_labels"]]
    loader = DataLoader(
        dataset,
        batch_size=int(config.get("batch_size", 64)),
        shuffle=True,
        collate_fn=collate_wisig,
        drop_last=False,
    )
    return loader, source_ids


def _cycle_batches(loader: DataLoader, steps: int):
    iterator = iter(loader)
    for _ in range(int(steps)):
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(loader)
            yield next(iterator)


def _accuracy(predicted: torch.Tensor, truth: torch.Tensor, ids: set[int]) -> float:
    mask = torch.tensor([int(value) in ids for value in truth.detach().cpu().tolist()], dtype=torch.bool)
    if int(mask.sum()) == 0:
        return float("nan")
    return float((predicted.detach().cpu()[mask] == truth.detach().cpu()[mask]).float().mean().item())


def _stage_metrics(
    predicted: torch.Tensor,
    truth: torch.Tensor,
    old_ids: set[int],
    new_ids: set[int],
    *,
    old_accuracy_before_increment: float,
) -> dict[str, float]:
    old_acc = _accuracy(predicted, truth, old_ids)
    new_acc = _accuracy(predicted, truth, new_ids)
    harmonic = 0.0 if old_acc + new_acc <= 0 else 2.0 * old_acc * new_acc / (old_acc + new_acc)
    old_mask = torch.tensor([int(v) in old_ids for v in truth.tolist()], dtype=torch.bool)
    new_mask = torch.tensor([int(v) in new_ids for v in truth.tolist()], dtype=torch.bool)
    pred_old = torch.tensor([int(v) in old_ids for v in predicted.tolist()], dtype=torch.bool)
    pred_new = torch.tensor([int(v) in new_ids for v in predicted.tolist()], dtype=torch.bool)
    return {
        "old_acc": old_acc,
        "seen_new_acc": new_acc,
        "H_old_new": harmonic,
        "old_to_seen_new_rate": float(pred_new[old_mask].float().mean().item()),
        "seen_new_to_old_rate": float(pred_old[new_mask].float().mean().item()),
        "old_acc_before_increment": old_accuracy_before_increment,
        "average_forgetting": float(old_accuracy_before_increment - old_acc),
    }


def _detailed_breakdown(
    predicted: torch.Tensor,
    truth: torch.Tensor,
    metadata: list[dict[str, Any]],
    *,
    scenario: str,
) -> list[dict[str, Any]]:
    if len(metadata) != int(truth.numel()) or predicted.shape != truth.shape:
        raise ValueError("metadata, truth, and predictions must align")
    groups: dict[tuple[str, str, str, str, str], list[int]] = {}
    for index, row in enumerate(metadata):
        rx = str(row.get("rx_label", ""))
        tx = str(row.get("tx_label", ""))
        day = str(row.get("day_i", ""))
        role = str(row.get("role", ""))
        for key in (
            ("per_receiver", rx, "ALL", "ALL", role),
            ("per_transmitter", "ALL", tx, "ALL", role),
            ("per_receiver_transmitter", rx, tx, "ALL", role),
            ("per_receiver_transmitter_day", rx, tx, day, role),
        ):
            groups.setdefault(key, []).append(index)
    rows: list[dict[str, Any]] = []
    for (group_type, receiver, transmitter, day, role), indices in sorted(groups.items()):
        index_tensor = torch.tensor(indices, dtype=torch.long)
        group_truth = truth[index_tensor].detach().cpu()
        group_predicted = predicted[index_tensor].detach().cpu()
        correct = int((group_truth == group_predicted).sum().item())
        confusion: dict[str, int] = {}
        for true_value, predicted_value in zip(group_truth.tolist(), group_predicted.tolist()):
            key = f"{int(true_value)}->{int(predicted_value)}"
            confusion[key] = confusion.get(key, 0) + 1
        rows.append(
            {
                "scenario": scenario,
                "group_type": group_type,
                "receiver_label": receiver,
                "transmitter_label": transmitter,
                "day": day,
                "role": role,
                "sample_count": len(indices),
                "correct_count": correct,
                "accuracy": correct / len(indices),
                "confusion_json": json.dumps(confusion, ensure_ascii=False, sort_keys=True),
            }
        )
    return rows


def _train_csil(
    config: dict[str, Any],
    loader: DataLoader,
    source_ids: list[int],
    support_x: torch.Tensor,
    support_y: torch.Tensor,
    query_x: torch.Tensor,
    query_y: torch.Tensor,
    old_ids: set[int],
    new_ids: set[int],
    device: torch.device,
) -> tuple[torch.Tensor, float, dict[str, Any]]:
    old_count, new_count = len(old_ids), len(new_ids)
    model = CSILClassifier(
        input_dim=int(support_x[0].numel()),
        embedding_dim=int(config.get("csil_embedding_dim", 64)),
        num_classes=old_count,
    ).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=float(config.get("learning_rate", 0.01)),
        momentum=float(config.get("momentum", 0.9)),
        weight_decay=float(config.get("weight_decay", 0.01)),
    )
    model.train()
    for batch in _cycle_batches(loader, int(config["base_steps"])):
        x = batch["iq"].to(device).flatten(1)
        y = _compact_source_labels(batch["label"], source_ids, device)
        optimizer.zero_grad(set_to_none=True)
        F.cross_entropy(model(x), y).backward()
        optimizer.step()

    old_mask = torch.tensor([int(v) in old_ids for v in support_y.tolist()], dtype=torch.bool)
    old_x = support_x[old_mask].to(device).flatten(1)
    old_y = support_y[old_mask].to(device)
    for _ in range(int(config.get("old_support_steps", config["increment_steps"]))):
        optimizer.zero_grad(set_to_none=True)
        F.cross_entropy(model(old_x), old_y).backward()
        optimizer.step()
    with torch.no_grad():
        pre_pred = model(query_x.to(device).flatten(1)).argmax(dim=1).cpu()
    pre_old = _accuracy(pre_pred, query_y, old_ids)

    previous = copy.deepcopy(model).eval()
    model.expand_for_stage(
        new_classes=new_count,
        added_embedding_dim=int(config.get("csil_added_embedding_dim", 32)),
        stage_id=1,
    )
    new_mask = torch.tensor([int(v) in new_ids for v in support_y.tolist()], dtype=torch.bool)
    new_x = support_x[new_mask].to(device).flatten(1)
    new_y = support_y[new_mask].to(device)
    velocity: dict[str, torch.Tensor] = {}
    for _ in range(int(config["increment_steps"])):
        model.zero_grad(set_to_none=True)
        logits = model(new_x)
        with torch.no_grad():
            previous_response = previous(new_x)
        loss = compute_csil_loss(
            logits=logits,
            labels=new_y,
            current_old_response=logits[:, :old_count],
            previous_old_response=previous_response,
            kd_weight=float(config.get("csil_kd_weight", 1.0)),
        )
        loss.total.backward()
        velocity = csil_masked_sgd_step(
            model,
            lr=float(config.get("learning_rate", 0.01)),
            momentum=float(config.get("momentum", 0.9)),
            weight_decay=float(config.get("weight_decay", 0.01)),
            state=velocity,
        )
    with torch.no_grad():
        predicted = model(query_x.to(device).flatten(1)).argmax(dim=1).cpu()
    return predicted, pre_old, {
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "prototype_storage": 0,
        "paper_mechanisms": ["zero_bias_cosine", "channel_separation", "old_block_gradient_mask", "KD"],
    }


class _MoPCModel(nn.Module):
    def __init__(self, embedding_dim: int, total_classes: int) -> None:
        super().__init__()
        self.encoder = SixBlockConv1DEncoder(input_channels=2, embedding_dim=embedding_dim)
        self.classifier = nn.Linear(embedding_dim, total_classes, bias=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feature = self.encoder(x)
        return self.classifier(feature), feature


def _train_mopc(
    config: dict[str, Any],
    loader: DataLoader,
    source_ids: list[int],
    support_x: torch.Tensor,
    support_y: torch.Tensor,
    query_x: torch.Tensor,
    query_y: torch.Tensor,
    old_ids: set[int],
    new_ids: set[int],
    device: torch.device,
) -> tuple[torch.Tensor, float, dict[str, Any]]:
    old_count = len(old_ids)
    model = _MoPCModel(int(config.get("embedding_dim", 64)), old_count + len(new_ids)).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=float(config.get("learning_rate", 0.01)),
        momentum=float(config.get("momentum", 0.9)),
        weight_decay=float(config.get("weight_decay", 2e-4)),
    )
    for batch in _cycle_batches(loader, int(config["base_steps"])):
        x = batch["iq"].to(device)
        y = _compact_source_labels(batch["label"], source_ids, device)
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(x)
        F.cross_entropy(logits[:, :old_count], y).backward()
        optimizer.step()

    old_mask = torch.tensor([int(v) in old_ids for v in support_y.tolist()], dtype=torch.bool)
    old_x = support_x[old_mask].to(device)
    old_y = support_y[old_mask].to(device)
    for _ in range(int(config.get("old_support_steps", config["increment_steps"]))):
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(old_x)
        F.cross_entropy(logits[:, :old_count], old_y).backward()
        optimizer.step()
    with torch.no_grad():
        pre_pred = model(query_x.to(device))[0][:, :old_count].argmax(dim=1).cpu()
        old_features = model(old_x)[1]
        old_prototypes, old_class_ids = compute_class_prototypes(old_features, old_y)
    pre_old = _accuracy(pre_pred, query_y, old_ids)

    previous = copy.deepcopy(model).eval()
    previous_parameters = {name: value.detach().clone() for name, value in model.named_parameters()}
    new_mask = torch.tensor([int(v) in new_ids for v in support_y.tolist()], dtype=torch.bool)
    new_x = support_x[new_mask].to(device)
    new_y = support_y[new_mask].to(device)
    generator = torch.Generator(device=device).manual_seed(int(config.get("seed", 0)) + 991)
    for _ in range(int(config["increment_steps"])):
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(new_x)
        augmented, augmented_y = prototype_augmentation(
            old_prototypes,
            old_class_ids,
            num_samples=max(1, int(new_x.size(0))),
            noise_std=float(config.get("prototype_noise_std", 0.05)),
            generator=generator,
        )
        prototype_logits = model.classifier(augmented)
        loss = mopc_hr_incremental_objective(
            logits,
            new_y,
            prototype_logits,
            augmented_y,
            dict(model.named_parameters()),
            previous_parameters,
            beta=float(config.get("mopc_beta", 1.0)),
            lambda_max=float(config.get("mopc_lambda_max", 1.0)),
        )
        loss.total.backward()
        optimizer.step()

    with torch.no_grad():
        new_previous = previous(new_x)[1]
        new_current = model(new_x)[1]
        previous_prototypes, new_class_ids = compute_class_prototypes(new_previous, new_y)
        current_prototypes, _ = compute_class_prototypes(new_current, new_y, new_class_ids)
        corrected = correct_old_prototypes(
            old_prototypes,
            previous_prototypes,
            current_prototypes,
            alpha=float(config.get("prototype_momentum", 0.97)),
            similarity_mode=str(config.get("mopc_similarity_mode", "paper_cosine")),
        )
        predicted = model(query_x.to(device))[0].argmax(dim=1).cpu()
    return predicted, pre_old, {
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "prototype_storage": int(corrected.numel() + current_prototypes.numel()),
        "paper_mechanisms": ["prototype_augmentation", "hierarchical_regularization", "momentum_prototype_correction"],
        "mopc_similarity_mode": str(config.get("mopc_similarity_mode", "paper_cosine")),
    }


def _train_orthogonal(
    config: dict[str, Any],
    loader: DataLoader,
    source_ids: list[int],
    support_x: torch.Tensor,
    support_y: torch.Tensor,
    query_x: torch.Tensor,
    query_y: torch.Tensor,
    old_ids: set[int],
    new_ids: set[int],
    device: torch.device,
) -> tuple[torch.Tensor, float, dict[str, Any]]:
    old_count = len(old_ids)
    total_count = old_count + len(new_ids)
    embedding_dim = int(config.get("embedding_dim", 64))
    encoder = SixBlockConv1DEncoder(input_channels=2, embedding_dim=embedding_dim).to(device)
    targets = make_simplex_pseudo_targets(num_targets=total_count, feature_dim=embedding_dim).to(device)
    assigned = assign_base_targets(list(range(old_count)), targets)
    perturbed = perturb_pseudo_targets(
        targets,
        noise_range=float(config.get("orthogonal_noise_range", 0.05)),
        seed=int(config.get("seed", 0)),
    )
    optimizer = torch.optim.SGD(
        encoder.parameters(),
        lr=float(config.get("learning_rate", 0.01)),
        momentum=float(config.get("momentum", 0.9)),
        weight_decay=float(config.get("weight_decay", 5e-4)),
    )
    for batch in _cycle_batches(loader, int(config["base_steps"])):
        x = batch["iq"].to(device)
        y = _compact_source_labels(batch["label"], source_ids, device)
        optimizer.zero_grad(set_to_none=True)
        features = encoder(x)
        loss, _ = base_training_loss(features, y, assigned, targets, perturbed)
        loss.backward()
        optimizer.step()

    old_mask = torch.tensor([int(v) in old_ids for v in support_y.tolist()], dtype=torch.bool)
    old_x = support_x[old_mask].to(device)
    old_y = support_y[old_mask].to(device)
    for _ in range(int(config.get("old_support_steps", config["increment_steps"]))):
        optimizer.zero_grad(set_to_none=True)
        loss, _ = base_training_loss(encoder(old_x), old_y, assigned, targets, perturbed)
        loss.backward()
        optimizer.step()
    old_weights = torch.stack([assigned[index] for index in range(old_count)]).to(device)
    with torch.no_grad():
        pre_features = F.normalize(encoder(query_x.to(device)), dim=1)
        pre_pred = (pre_features @ F.normalize(old_weights, dim=1).t()).argmax(dim=1).cpu()
    pre_old = _accuracy(pre_pred, query_y, old_ids)

    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    new_mask = torch.tensor([int(v) in new_ids for v in support_y.tolist()], dtype=torch.bool)
    new_x = support_x[new_mask].to(device)
    new_y = support_y[new_mask].to(device)
    with torch.no_grad():
        new_features = encoder(new_x)
        prototypes, _ = compute_class_prototypes(new_features, new_y)
    new_weights = nn.Parameter(prototypes.detach().clone())
    increment_optimizer = torch.optim.SGD([new_weights], lr=float(config.get("increment_learning_rate", 0.01)))
    for _ in range(int(config["increment_steps"])):
        increment_optimizer.zero_grad(set_to_none=True)
        loss, _ = incremental_calibration_loss(
            new_features,
            new_y,
            old_weights,
            new_weights,
            new_class_ids=torch.tensor(sorted(new_ids), device=device),
            prototypes=prototypes,
            top_k=int(config.get("orthogonal_top_k", 2)),
            margin=float(config.get("orthogonal_margin", 0.2)),
            tau_fuse=float(config.get("orthogonal_tau_fuse", 0.5)),
            lambda_align=float(config.get("orthogonal_lambda_align", 1.6)),
        )
        loss.backward()
        increment_optimizer.step()
    weights = torch.cat([old_weights, new_weights.detach()], dim=0)
    with torch.no_grad():
        features = F.normalize(encoder(query_x.to(device)), dim=1)
        predicted = (features @ F.normalize(weights, dim=1).t()).argmax(dim=1).cpu()
    return predicted, pre_old, {
        "trainable_parameters": int(new_weights.numel()),
        "prototype_storage": int(weights.numel() + targets.numel()),
        "paper_mechanisms": ["orthogonal_pseudo_targets", "frozen_encoder", "incremental_calibration"],
    }


RUNNERS = {
    "csil": _train_csil,
    "mopc_hr": _train_mopc,
    "orthogonal_incremental": _train_orthogonal,
}


def run(config: dict[str, Any], *, run_dir: Path, device: torch.device) -> dict[str, Any]:
    checked = validate_class_incremental_manifest(config)
    seed = int(config.get("seed", 1337))
    set_seed(seed)
    manysig = load_wisig_compact_pkl(str(config["manysig_pkl"]))
    manytx = load_wisig_compact_pkl(str(config["manytx_pkl"]))
    target_info = _select_target_sets(config, manysig, manytx)
    tensors = _build_stage2_tensors(config, manysig, manytx)
    if tensors["support_query_overlap"]:
        raise ValueError("support/query overlap detected")
    loader, source_ids = _source_loader(config, manysig)

    scenarios = [str(value) for value in config["target_channel_scenarios"]]
    metrics_by_scenario: dict[str, dict[str, Any]] = {}
    predictions_by_scenario: dict[str, torch.Tensor] = {}
    detailed_rows: list[dict[str, Any]] = []
    total_elapsed = 0.0
    for scenario_index, scenario in enumerate(scenarios):
        set_seed(seed)
        support_x = _apply_scenario(
            tensors["support_x"].to(device), scenario, seed=seed + 1000 + scenario_index
        ).cpu()
        query_x = _apply_scenario(
            tensors["query_x"].to(device), scenario, seed=seed + 2000 + scenario_index
        ).cpu()
        started = time.perf_counter()
        predicted, pre_old, method_info = RUNNERS[checked["method"]](
            {**config, **checked},
            loader,
            source_ids,
            support_x,
            tensors["support_y"],
            query_x,
            tensors["query_y"],
            tensors["old_class_ids"],
            tensors["new_class_ids"],
            device,
        )
        elapsed = time.perf_counter() - started
        total_elapsed += elapsed
        scenario_metrics = _stage_metrics(
            predicted,
            tensors["query_y"],
            tensors["old_class_ids"],
            tensors["new_class_ids"],
            old_accuracy_before_increment=pre_old,
        )
        scenario_metrics.update(
            {
                "adaptation_latency_sec": elapsed,
                "latency_per_query_ms": elapsed * 1000.0 / max(1, int(tensors["query_y"].numel())),
                **method_info,
            }
        )
        metrics_by_scenario[scenario] = scenario_metrics
        predictions_by_scenario[scenario] = predicted
        detailed_rows.extend(
            _detailed_breakdown(
                predicted,
                tensors["query_y"],
                tensors["query_meta"],
                scenario=scenario,
            )
        )

    aggregate: dict[str, Any] = {
        "adaptation_latency_sec_total": total_elapsed,
        "scenario_count": len(scenarios),
    }
    for key in (
        "old_acc",
        "seen_new_acc",
        "H_old_new",
        "old_to_seen_new_rate",
        "seen_new_to_old_rate",
        "old_acc_before_increment",
        "average_forgetting",
    ):
        values = [float(row[key]) for row in metrics_by_scenario.values()]
        aggregate[f"{key}_mean"] = float(sum(values) / len(values))
    support_ids = [row["sample_id"] for row in tensors["support_meta"]]
    query_ids = [row["sample_id"] for row in tensors["query_meta"]]
    split_manifest = {
        **checked,
        **target_info,
        "seed": seed,
        "target_channel_scenarios": scenarios,
        "all_tests_satellite_augmented": all(value != "clean" for value in scenarios),
        "support_sample_ids": support_ids,
        "query_sample_ids": query_ids,
        "support_query_overlap": bool(set(support_ids) & set(query_ids)),
        "old_class_ids": sorted(tensors["old_class_ids"]),
        "new_class_ids": sorted(tensors["new_class_ids"]),
        "class_map": tensors["class_map"],
    }
    if split_manifest["support_query_overlap"]:
        raise ValueError("support/query overlap detected after manifest construction")
    result = {
        "experiment_id": config.get("experiment_id", f"cvs_stage2c_{checked['method']}_{seed}"),
        "method": checked["method"],
        "seed": seed,
        "target_channel_scenarios": scenarios,
        "cvs_extension": True,
        "claim_boundary": checked["claim_boundary"],
        "metrics": aggregate,
        "metrics_by_scenario": metrics_by_scenario,
        "detailed_result_rows": detailed_rows,
        "split_manifest": split_manifest,
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "metrics.json", result)
    write_json(run_dir / "split_manifest.json", split_manifest)
    write_json(run_dir / "resolved_config.json", {**config, **checked})
    write_json(run_dir / "detailed_metrics.json", detailed_rows)
    with (run_dir / "detailed_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(detailed_rows[0].keys()))
        writer.writeheader()
        writer.writerows(detailed_rows)
    with (run_dir / "score_table.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sample_id",
                "receiver_label",
                "transmitter_label",
                "day_i",
                "sig_i",
                "role",
                "true_label",
                "predicted_label",
                "correct",
                "scenario",
            ],
        )
        writer.writeheader()
        for scenario, predicted in predictions_by_scenario.items():
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
    parser = argparse.ArgumentParser(description="CVS Stage2-C class-incremental comparison runner")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = load_json_config(args.config)
    checked = validate_class_incremental_manifest(config)
    if args.dry_run:
        print(json.dumps(checked, ensure_ascii=False, sort_keys=True))
        return 0
    result = run(config, run_dir=args.run_dir, device=torch.device(args.device))
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
