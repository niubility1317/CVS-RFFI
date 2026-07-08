from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from paper_reproduction.common.wisig_runtime import load_wisig_compact_pkl, set_seed, write_json
from dataset_wisig import WiSigCompactDataset
from .losses import base_training_loss, incremental_calibration_loss
from .metrics import average_incremental_metrics, top1_accuracy
from .model import SixBlockConv1DEncoder, class_mean_weights, concat_classifier_weights
from .pseudo_targets import assign_base_targets, optimize_pseudo_targets, perturb_pseudo_targets


FORMAL_PROTOCOL_FIELDS = {
    "shot_grid",
    "base_train_ratio",
    "base_test_ratio",
    "same_receiver_only",
    "min_samples_per_transmitter",
    "base_epochs",
    "increment_epochs",
    "batch_size",
    "optimizer",
    "base_lr",
    "early_stop_patience",
    "increment_classes_per_session",
    "num_increment_sessions",
    "cvs_extension",
}


def load_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _paper_float(config: dict, implementation_key: str, paper_key: str, *, default: float) -> float:
    if paper_key in config:
        return float(config[paper_key])
    return float(config.get(implementation_key, default))


def _build_pseudo_targets(config: dict, *, device: str | torch.device) -> tuple[torch.Tensor, int]:
    feature_dim = int(config.get("embedding_dim", 16))
    num_targets = int(config.get("pseudo_targets", min(feature_dim + 1, 8)))
    base_classes = int(config.get("base_classes", min(3, num_targets)))
    steps = int(config.get("pseudo_target_steps", 0))
    targets = optimize_pseudo_targets(
        num_targets=num_targets,
        feature_dim=feature_dim,
        total_classes=base_classes,
        temperature=_paper_float(config, "pseudo_target_temperature", "tau_c", default=0.01),
        steps=steps,
        seed=int(config.get("seed", 1337)),
        device=device,
    )
    return targets, steps


def _unsupported_config_fields(config: dict) -> list[str]:
    return sorted(field for field in FORMAL_PROTOCOL_FIELDS if field in config)


def _resolve_label_indices(available: list[Any], requested: list[Any] | None, *, limit: int) -> list[int]:
    if requested is None:
        if len(available) < limit:
            raise ValueError(f"dataset has {len(available)} transmitters, need at least {limit}")
        return list(range(limit))
    lookup = {str(value): idx for idx, value in enumerate(available)}
    out: list[int] = []
    missing: list[Any] = []
    for value in requested:
        if isinstance(value, int) and 0 <= value < len(available):
            out.append(int(value))
            continue
        key = str(value)
        if key not in lookup:
            missing.append(value)
        else:
            out.append(lookup[key])
    if missing:
        raise ValueError(f"requested transmitter labels not found: {missing[:8]}")
    if len(out) < limit:
        raise ValueError(f"requested transmitter list has {len(out)} entries, need {limit}")
    return out[:limit]


def _resolve_all_requested_label_indices(available: list[Any], requested: list[Any]) -> list[int]:
    lookup = {str(value): idx for idx, value in enumerate(available)}
    out: list[int] = []
    missing: list[Any] = []
    for value in requested:
        if isinstance(value, int) and 0 <= value < len(available):
            out.append(int(value))
            continue
        key = str(value)
        if key not in lookup:
            missing.append(value)
        else:
            out.append(lookup[key])
    if missing:
        raise ValueError(f"requested transmitter labels not found: {missing[:8]}")
    return out


def _resolve_single_receiver(available: list[Any], requested: Any | None) -> int:
    if requested is None:
        if not available:
            raise ValueError("dataset receiver list is empty")
        return 0
    if isinstance(requested, int) and 0 <= requested < len(available):
        return int(requested)
    lookup = {str(value): idx for idx, value in enumerate(available)}
    key = str(requested)
    if key not in lookup:
        raise ValueError(f"receiver label not found: {requested!r}")
    return int(lookup[key])


def _resolve_config_labels(config: dict) -> list[Any] | None:
    labels = config.get("tx_labels") or config.get("labels")
    if labels is not None:
        return labels
    preset = str(config.get("label_preset", "") or "").strip().lower()
    if preset:
        raise ValueError(f"label_preset={preset} requires an explicit labels list in this standalone runner")
    return None


def _base_requested_labels(config: dict, *, base_classes: int) -> list[Any] | None:
    labels = _resolve_config_labels(config)
    if labels is None:
        return None
    if len(labels) < base_classes:
        raise ValueError(f"labels list has {len(labels)} entries, need at least base_classes={base_classes}")
    return list(labels[:base_classes])


def _split_class_samples(
    tensors: list[torch.Tensor],
    *,
    train_ratio: float,
    seed: int,
    class_id: int,
    min_samples: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if len(tensors) < min_samples:
        raise ValueError(f"class {class_id} has {len(tensors)} samples, below min_samples={min_samples}")
    generator = torch.Generator().manual_seed(int(seed) + int(class_id) * 1009)
    order = torch.randperm(len(tensors), generator=generator).tolist()
    shuffled = [tensors[int(i)] for i in order]
    train_n = int(math.floor(len(shuffled) * float(train_ratio)))
    train_n = min(max(1, train_n), len(shuffled) - 1)
    return torch.stack(shuffled[:train_n], dim=0), torch.stack(shuffled[train_n:], dim=0)


def _split_incremental_samples(
    tensors: list[torch.Tensor],
    *,
    shot: int,
    seed: int,
    class_id: int,
    min_samples: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if shot <= 0:
        raise ValueError("shot must be positive")
    if len(tensors) < min_samples:
        raise ValueError(f"class {class_id} has {len(tensors)} samples, below min_samples={min_samples}")
    if len(tensors) <= shot:
        raise ValueError(f"class {class_id} has {len(tensors)} samples, need more than shot={shot}")
    generator = torch.Generator().manual_seed(int(seed) + int(class_id) * 1009)
    order = torch.randperm(len(tensors), generator=generator).tolist()
    shuffled = [tensors[int(i)] for i in order]
    return torch.stack(shuffled[:shot], dim=0), torch.stack(shuffled[shot:], dim=0)


def _load_wisig_fscil_tensors(config: dict, *, wisig_pkl: str, seed: int) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor], dict[str, Any]]:
    ds = load_wisig_compact_pkl(wisig_pkl)
    base_classes = int(config.get("base_classes", 60))
    total_classes = base_classes + int(config.get("increment_classes_per_session", 10)) * int(config.get("num_increment_sessions", 7))
    tx_labels = list(ds.get("tx_list", []))
    rx_labels = list(ds.get("rx_list", []))
    selected_tx = _resolve_label_indices(tx_labels, _resolve_config_labels(config), limit=total_classes)
    receiver_key = config.get("receiver_label", config.get("receiver", config.get("target_receiver")))
    receiver = _resolve_single_receiver(rx_labels, receiver_key)
    compact = WiSigCompactDataset(
        ds,
        out_len=int(config.get("input_length", 256)),
        crop_mode=str(config.get("crop_mode", "center")),
        normalize=bool(config.get("rms_normalize", True)),
        equalized=config.get("equalized", 1),
        tx_keep=selected_tx,
        rx_keep=[receiver],
        day_keep=config.get("day_indices"),
        domain="day",
        max_samples_per_combo=config.get("max_samples_per_combo"),
        sample_strategy=str(config.get("sample_strategy", "front")),
        seed=seed,
        build_index=True,
    )
    tx_to_compact = {int(tx_i): compact_i for compact_i, tx_i in enumerate(selected_tx)}
    grouped: dict[int, list[torch.Tensor]] = {idx: [] for idx in range(total_classes)}
    for idx in range(len(compact)):
        x, _label, _domain, meta = compact[idx]
        grouped[tx_to_compact[int(meta["tx_i"])]].append(x.float())
    train_by_class: dict[int, torch.Tensor] = {}
    query_by_class: dict[int, torch.Tensor] = {}
    min_samples = int(config.get("min_samples_per_transmitter", 50))
    train_ratio = float(config.get("base_train_ratio", 0.8))
    shot = int(config.get("shot", config.get("k_shot", 5)))
    for class_id in range(total_classes):
        if class_id < base_classes:
            train_x, query_x = _split_class_samples(
                grouped[class_id],
                train_ratio=train_ratio,
                seed=seed,
                class_id=class_id,
                min_samples=min_samples,
            )
        else:
            train_x, query_x = _split_incremental_samples(
                grouped[class_id],
                shot=shot,
                seed=seed,
                class_id=class_id,
                min_samples=min_samples,
            )
        train_by_class[class_id] = train_x
        query_by_class[class_id] = query_x
    split_info = {
        "wisig_pkl": wisig_pkl,
        "receiver_index": receiver,
        "receiver_label": rx_labels[receiver] if receiver < len(rx_labels) else receiver,
        "tx_indices": selected_tx,
        "tx_labels": [tx_labels[i] if i < len(tx_labels) else i for i in selected_tx],
        "total_classes": total_classes,
        "train_samples_per_class": {str(k): int(v.size(0)) for k, v in train_by_class.items()},
        "query_samples_per_class": {str(k): int(v.size(0)) for k, v in query_by_class.items()},
        "claim_boundary": "formal_wisig_closed_set_fscil_not_adsb_not_cvs_stage2",
    }
    return train_by_class, query_by_class, split_info


def _load_wisig_new_label_scan_tensors(
    config: dict,
    *,
    wisig_pkl: str,
    seed: int,
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor], dict[int, tuple[torch.Tensor, torch.Tensor]], dict[str, Any]]:
    ds = load_wisig_compact_pkl(wisig_pkl)
    base_classes = int(config.get("base_classes", 60))
    tx_labels = list(ds.get("tx_list", []))
    rx_labels = list(ds.get("rx_list", []))
    base_requested = _base_requested_labels(config, base_classes=base_classes)
    base_tx = _resolve_label_indices(tx_labels, base_requested, limit=base_classes)
    if config.get("candidate_labels") is not None:
        candidate_tx = _resolve_all_requested_label_indices(tx_labels, list(config["candidate_labels"]))
    else:
        candidate_tx = [idx for idx in range(len(tx_labels)) if idx not in set(base_tx)]
    candidate_tx = [idx for idx in candidate_tx if idx not in set(base_tx)]
    if not candidate_tx:
        raise ValueError("new-label scan requires at least one candidate transmitter")
    selected_tx = base_tx + candidate_tx
    receiver_key = config.get("receiver_label", config.get("receiver", config.get("target_receiver")))
    receiver = _resolve_single_receiver(rx_labels, receiver_key)
    compact = WiSigCompactDataset(
        ds,
        out_len=int(config.get("input_length", 256)),
        crop_mode=str(config.get("crop_mode", "center")),
        normalize=bool(config.get("rms_normalize", True)),
        equalized=config.get("equalized", 1),
        tx_keep=selected_tx,
        rx_keep=[receiver],
        day_keep=config.get("day_indices"),
        domain="day",
        max_samples_per_combo=config.get("max_samples_per_combo"),
        sample_strategy=str(config.get("sample_strategy", "front")),
        seed=seed,
        build_index=True,
    )
    tx_to_local = {int(tx_i): local_i for local_i, tx_i in enumerate(selected_tx)}
    grouped: dict[int, list[torch.Tensor]] = {idx: [] for idx in range(len(selected_tx))}
    for idx in range(len(compact)):
        x, _label, _domain, meta = compact[idx]
        grouped[tx_to_local[int(meta["tx_i"])]].append(x.float())
    train_by_class: dict[int, torch.Tensor] = {}
    query_by_class: dict[int, torch.Tensor] = {}
    candidate_by_class: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    min_samples = int(config.get("min_samples_per_transmitter", 50))
    train_ratio = float(config.get("base_train_ratio", 0.8))
    shot = int(config.get("shot", config.get("k_shot", 5)))
    skipped: list[dict[str, Any]] = []
    for class_id in range(base_classes):
        train_x, query_x = _split_class_samples(
            grouped[class_id],
            train_ratio=train_ratio,
            seed=seed,
            class_id=class_id,
            min_samples=min_samples,
        )
        train_by_class[class_id] = train_x
        query_by_class[class_id] = query_x
    for offset, tx_index in enumerate(candidate_tx):
        class_id = base_classes + offset
        local_id = base_classes + offset
        try:
            train_x, query_x = _split_incremental_samples(
                grouped[local_id],
                shot=shot,
                seed=seed,
                class_id=class_id,
                min_samples=min_samples,
            )
        except ValueError as exc:
            skipped.append(
                {
                    "tx_index": int(tx_index),
                    "tx_label": tx_labels[tx_index] if tx_index < len(tx_labels) else tx_index,
                    "reason": str(exc),
                }
            )
            continue
        candidate_by_class[class_id] = (train_x, query_x)
    split_info = {
        "wisig_pkl": wisig_pkl,
        "receiver_index": receiver,
        "receiver_label": rx_labels[receiver] if receiver < len(rx_labels) else receiver,
        "base_tx_indices": base_tx,
        "base_tx_labels": [tx_labels[i] if i < len(tx_labels) else i for i in base_tx],
        "candidate_tx_indices": candidate_tx,
        "candidate_tx_labels": [tx_labels[i] if i < len(tx_labels) else i for i in candidate_tx],
        "candidate_class_ids": sorted(candidate_by_class),
        "skipped_candidates": skipped,
        "shot": shot,
        "claim_boundary": "new_label_scan_diagnostic_not_full_paper_reproduction",
    }
    return train_by_class, query_by_class, candidate_by_class, split_info


def _make_tensor_dataset(by_class: dict[int, torch.Tensor], class_ids: list[int], *, limit_per_class: int = 0) -> TensorDataset:
    xs: list[torch.Tensor] = []
    ys: list[torch.Tensor] = []
    for class_id in class_ids:
        data = by_class[int(class_id)]
        if limit_per_class > 0:
            data = data[: int(limit_per_class)]
        xs.append(data)
        ys.append(torch.full((data.size(0),), int(class_id), dtype=torch.long))
    return TensorDataset(torch.cat(xs, dim=0), torch.cat(ys, dim=0))


def _extract_features(encoder: nn.Module, x: torch.Tensor, *, device: torch.device, batch_size: int) -> torch.Tensor:
    features: list[torch.Tensor] = []
    loader = DataLoader(TensorDataset(x), batch_size=batch_size, shuffle=False)
    encoder.eval()
    with torch.no_grad():
        for (xb,) in loader:
            features.append(encoder(xb.to(device)).detach().cpu())
    return torch.cat(features, dim=0)


def _fit_base_classifier_weights(
    encoder: nn.Module,
    train_by_class: dict[int, torch.Tensor],
    *,
    base_classes: int,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    base_dataset = _make_tensor_dataset(train_by_class, list(range(base_classes)))
    base_x, base_y = base_dataset.tensors
    base_features = _extract_features(encoder, base_x, device=device, batch_size=batch_size).to(device)
    weights, class_ids = class_mean_weights(base_features, base_y.to(device))
    expected = torch.arange(base_classes, device=class_ids.device)
    if not torch.equal(class_ids, expected):
        raise ValueError("base class ids must be contiguous and sorted")
    return weights.detach().cpu()


def _assigned_base_target_weights(assigned_targets: dict[int, torch.Tensor], *, base_classes: int) -> torch.Tensor:
    missing = [class_id for class_id in range(base_classes) if class_id not in assigned_targets]
    if missing:
        raise ValueError(f"missing assigned pseudo targets for base classes: {missing[:8]}")
    return torch.stack([assigned_targets[class_id].detach().cpu() for class_id in range(base_classes)], dim=0)


def _base_classifier_weights(
    assigned_targets: dict[int, torch.Tensor],
    feature_mean_weights: torch.Tensor,
    *,
    base_classes: int,
    source: str = "paper_pseudo_targets",
) -> tuple[torch.Tensor, str]:
    source_key = str(source or "paper_pseudo_targets").strip().lower()
    if source_key in {"paper_pseudo_targets", "pseudo_targets", "paper"}:
        return _assigned_base_target_weights(assigned_targets, base_classes=base_classes), "paper_pseudo_targets"
    if source_key in {"class_mean_features", "feature_means", "class_means"}:
        if feature_mean_weights.size(0) != base_classes:
            raise ValueError("feature_mean_weights must contain one row per base class")
        return feature_mean_weights.detach().cpu(), "class_mean_features"
    raise ValueError("base_weight_source must be paper_pseudo_targets or class_mean_features")


def _predict_with_weights(encoder: nn.Module, dataset: TensorDataset, weights: torch.Tensor, *, device: torch.device, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    preds: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    encoder.eval()
    with torch.no_grad():
        for xb, yb in loader:
            feats = encoder(xb.to(device))
            logits = F.normalize(feats, dim=1) @ F.normalize(weights.to(device), dim=1).t()
            pred_rows = logits.argmax(dim=1).detach().cpu()
            preds.append(pred_rows)
            labels.append(yb.detach().cpu())
    return torch.cat(preds, dim=0), torch.cat(labels, dim=0)


def _evaluate_sessions(
    encoder: nn.Module,
    query_by_class: dict[int, torch.Tensor],
    weights: torch.Tensor,
    *,
    seen_class_ids: list[int],
    task_class_ids: list[list[int]],
    device: torch.device,
    batch_size: int,
) -> tuple[float, list[float]]:
    dataset = _make_tensor_dataset(query_by_class, seen_class_ids)
    pred_rows, labels = _predict_with_weights(encoder, dataset, weights, device=device, batch_size=batch_size)
    row_to_class = torch.tensor(seen_class_ids, dtype=torch.long)
    preds = row_to_class[pred_rows]
    session_acc = top1_accuracy(preds, labels)
    task_accs: list[float] = []
    for class_ids in task_class_ids:
        mask = torch.isin(labels, torch.tensor(class_ids, dtype=torch.long))
        if bool(mask.any()):
            task_accs.append(top1_accuracy(preds[mask], labels[mask]))
        else:
            task_accs.append(float("nan"))
    return session_acc, task_accs


def _evaluate_class_group_accuracy(
    encoder: nn.Module,
    query_by_class: dict[int, torch.Tensor],
    weights: torch.Tensor,
    *,
    seen_class_ids: list[int],
    eval_class_ids: list[int],
    device: torch.device,
    batch_size: int,
) -> float:
    dataset = _make_tensor_dataset(query_by_class, eval_class_ids)
    pred_rows, labels = _predict_with_weights(encoder, dataset, weights, device=device, batch_size=batch_size)
    row_to_class = torch.tensor(seen_class_ids, dtype=torch.long)
    preds = row_to_class[pred_rows]
    return top1_accuracy(preds, labels)


def _early_stop_should_break(*, current_loss: float, best_loss: float, stale_epochs: int, patience: int, min_delta: float) -> tuple[float, int, bool]:
    if patience < 0:
        return min(best_loss, current_loss), 0, False
    if current_loss < best_loss - min_delta:
        return current_loss, 0, False
    stale_epochs += 1
    return best_loss, stale_epochs, stale_epochs > patience


def _train_base_encoder(
    config: dict,
    train_by_class: dict[int, torch.Tensor],
    *,
    base_classes: int,
    pseudo_targets: torch.Tensor,
    perturbed: torch.Tensor,
    assigned: dict[int, torch.Tensor],
    encoder: nn.Module,
    device: torch.device,
) -> list[dict[str, Any]]:
    batch_size = int(config.get("batch_size", 128))
    base_epochs = int(config.get("base_epochs", 100))
    early_stop_patience = int(config.get("early_stop_patience", -1))
    early_stop_min_delta = float(config.get("early_stop_min_delta", 0.0))
    base_dataset = _make_tensor_dataset(train_by_class, list(range(base_classes)))
    base_loader = DataLoader(base_dataset, batch_size=batch_size, shuffle=True)
    opt_name = str(config.get("optimizer", "SGD")).lower()
    if opt_name == "sgd":
        optimizer = torch.optim.SGD(encoder.parameters(), lr=float(config.get("base_lr", 0.01)))
    elif opt_name == "adam":
        optimizer = torch.optim.Adam(encoder.parameters(), lr=float(config.get("base_lr", 0.001)))
    else:
        raise ValueError(f"unsupported optimizer: {config.get('optimizer')}")
    history: list[dict[str, Any]] = []
    best_loss = float("inf")
    stale_epochs = 0
    for epoch in range(1, base_epochs + 1):
        encoder.train()
        losses: list[float] = []
        for xb, yb in base_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss, _terms = base_training_loss(
                encoder(xb),
                yb,
                assigned,
                pseudo_targets,
                perturbed,
                contrast_temperature=_paper_float(config, "contrast_temperature", "tau_s", default=0.1),
                center_temperature=_paper_float(config, "center_temperature", "tau_c", default=0.1),
            )
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
        epoch_loss = float(sum(losses) / max(1, len(losses)))
        history.append({"phase": "base", "epoch": epoch, "loss": epoch_loss})
        best_loss, stale_epochs, should_stop = _early_stop_should_break(
            current_loss=epoch_loss,
            best_loss=best_loss,
            stale_epochs=stale_epochs,
            patience=early_stop_patience,
            min_delta=early_stop_min_delta,
        )
        if should_stop:
            history[-1]["early_stop"] = True
            history[-1]["best_loss"] = best_loss
            break
    return history


def run_formal_wisig(config: dict, *, wisig_pkl: str, run_dir: str | Path, device: str = "cuda:0") -> dict[str, Any]:
    seed = int(config.get("seed", 1337))
    set_seed(seed)
    dev = torch.device(device)
    train_by_class, query_by_class, split_info = _load_wisig_fscil_tensors(config, wisig_pkl=wisig_pkl, seed=seed)
    feature_dim = int(config.get("embedding_dim", 256))
    base_classes = int(config.get("base_classes", 60))
    inc_classes = int(config.get("increment_classes_per_session", 10))
    inc_sessions = int(config.get("num_increment_sessions", 7))
    batch_size = int(config.get("batch_size", 128))
    eval_batch_size = int(config.get("eval_batch_size", batch_size))
    base_epochs = int(config.get("base_epochs", 100))
    increment_epochs = int(config.get("increment_epochs", 50))
    early_stop_patience = int(config.get("early_stop_patience", -1))
    early_stop_min_delta = float(config.get("early_stop_min_delta", 0.0))
    encoder = SixBlockConv1DEncoder(input_channels=2, embedding_dim=feature_dim).to(dev)
    pseudo_targets, pseudo_target_steps = _build_pseudo_targets(config, device=dev)
    perturbed = perturb_pseudo_targets(pseudo_targets, noise_range=float(config.get("noise_range", 0.01)), seed=seed).to(dev)
    assigned = assign_base_targets(range(base_classes), pseudo_targets)
    base_dataset = _make_tensor_dataset(train_by_class, list(range(base_classes)))
    base_loader = DataLoader(base_dataset, batch_size=batch_size, shuffle=True)
    opt_name = str(config.get("optimizer", "SGD")).lower()
    if opt_name == "sgd":
        optimizer = torch.optim.SGD(encoder.parameters(), lr=float(config.get("base_lr", 0.01)))
    elif opt_name == "adam":
        optimizer = torch.optim.Adam(encoder.parameters(), lr=float(config.get("base_lr", 0.001)))
    else:
        raise ValueError(f"unsupported optimizer: {config.get('optimizer')}")
    history: list[dict[str, Any]] = []
    early_stop_info: dict[str, Any] = {"base": {"stopped_epoch": 0}, "increment": {}}
    best_loss = float("inf")
    stale_epochs = 0
    for epoch in range(1, base_epochs + 1):
        encoder.train()
        losses: list[float] = []
        for xb, yb in base_loader:
            xb = xb.to(dev)
            yb = yb.to(dev)
            optimizer.zero_grad(set_to_none=True)
            loss, _terms = base_training_loss(
                encoder(xb),
                yb,
                assigned,
                pseudo_targets,
                perturbed,
                contrast_temperature=_paper_float(config, "contrast_temperature", "tau_s", default=0.1),
                center_temperature=_paper_float(config, "center_temperature", "tau_c", default=0.1),
            )
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
        epoch_loss = float(sum(losses) / max(1, len(losses)))
        history.append({"phase": "base", "epoch": epoch, "loss": epoch_loss})
        best_loss, stale_epochs, should_stop = _early_stop_should_break(
            current_loss=epoch_loss,
            best_loss=best_loss,
            stale_epochs=stale_epochs,
            patience=early_stop_patience,
            min_delta=early_stop_min_delta,
        )
        if should_stop:
            early_stop_info["base"] = {"stopped_epoch": epoch, "best_loss": best_loss}
            break

    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    feature_mean_base_weights = _fit_base_classifier_weights(
        encoder,
        train_by_class,
        base_classes=base_classes,
        device=dev,
        batch_size=eval_batch_size,
    )
    base_weights, base_weight_source = _base_classifier_weights(
        assigned,
        feature_mean_base_weights,
        base_classes=base_classes,
        source=str(config.get("base_weight_source", "paper_pseudo_targets")),
    )
    learned_new_weights: list[torch.Tensor] = []
    seen_class_ids = list(range(base_classes))
    task_class_ids: list[list[int]] = [list(range(base_classes))]
    session_accuracies: list[float] = []
    old_accuracies: list[float] = []
    new_accuracies: list[float] = []
    accuracy_rows: list[list[float]] = []
    weights = base_weights
    base_acc, base_task_accs = _evaluate_sessions(
        encoder,
        query_by_class,
        weights,
        seen_class_ids=seen_class_ids,
        task_class_ids=task_class_ids,
        device=dev,
        batch_size=eval_batch_size,
    )
    session_accuracies.append(base_acc)
    old_accuracies.append(base_acc)
    new_accuracies.append(base_acc)
    accuracy_rows.append(base_task_accs)

    for session in range(1, inc_sessions + 1):
        new_class_ids = list(range(base_classes + (session - 1) * inc_classes, base_classes + session * inc_classes))
        support_dataset = _make_tensor_dataset(train_by_class, new_class_ids, limit_per_class=int(config.get("shot", config.get("k_shot", 5))))
        support_x, support_y = support_dataset.tensors
        support_features = _extract_features(encoder, support_x, device=dev, batch_size=eval_batch_size).to(dev)
        support_y = support_y.to(dev)
        init_weights, class_ids_tensor = class_mean_weights(support_features, support_y)
        new_weights = nn.Parameter(init_weights.detach().clone())
        inc_optimizer = torch.optim.SGD([new_weights], lr=float(config.get("increment_lr", 0.08)))
        current_old_weights = concat_classifier_weights(base_weights, torch.cat(learned_new_weights, dim=0)) if learned_new_weights else base_weights
        current_old_weights = current_old_weights.to(dev)
        best_loss = float("inf")
        stale_epochs = 0
        for epoch in range(1, increment_epochs + 1):
            inc_optimizer.zero_grad(set_to_none=True)
            inc_loss, inc_terms = incremental_calibration_loss(
                support_features,
                support_y,
                current_old_weights,
                new_weights,
                new_class_ids=class_ids_tensor,
                prototypes=init_weights.detach(),
                top_k=int(config.get("top_k", 60)),
                margin=_paper_float(config, "margin", "q", default=0.2),
                tau_fuse=float(config.get("tau_fuse", 0.01)),
                lambda_align=_paper_float(config, "lambda_align", "lambda_a", default=1.6),
            )
            inc_loss.backward()
            inc_optimizer.step()
            history.append(
                {
                    "phase": "increment",
                    "session": session,
                    "epoch": epoch,
                    "loss": float(inc_loss.detach().cpu().item()),
                    "hard_count": float(inc_terms["hard_count"].detach().cpu().item()),
                }
            )
            best_loss, stale_epochs, should_stop = _early_stop_should_break(
                current_loss=float(inc_loss.detach().cpu().item()),
                best_loss=best_loss,
                stale_epochs=stale_epochs,
                patience=early_stop_patience,
                min_delta=early_stop_min_delta,
            )
            if should_stop:
                early_stop_info["increment"][str(session)] = {"stopped_epoch": epoch, "best_loss": best_loss}
                break
        learned_new_weights.append(new_weights.detach().cpu())
        seen_class_ids.extend(new_class_ids)
        task_class_ids.append(new_class_ids)
        weights = concat_classifier_weights(base_weights, torch.cat(learned_new_weights, dim=0))
        session_acc, task_accs = _evaluate_sessions(
            encoder,
            query_by_class,
            weights,
            seen_class_ids=seen_class_ids,
            task_class_ids=task_class_ids,
            device=dev,
            batch_size=eval_batch_size,
        )
        session_accuracies.append(session_acc)
        old_accuracies.append(
            _evaluate_class_group_accuracy(
                encoder,
                query_by_class,
                weights,
                seen_class_ids=seen_class_ids,
                eval_class_ids=seen_class_ids[: -len(new_class_ids)],
                device=dev,
                batch_size=eval_batch_size,
            )
        )
        new_accuracies.append(
            _evaluate_class_group_accuracy(
                encoder,
                query_by_class,
                weights,
                seen_class_ids=seen_class_ids,
                eval_class_ids=new_class_ids,
                device=dev,
                batch_size=eval_batch_size,
            )
        )
        accuracy_rows.append(task_accs)

    total_sessions = 1 + inc_sessions
    matrix = torch.full((total_sessions, total_sessions), float("nan"))
    for row, values in enumerate(accuracy_rows):
        matrix[row, : len(values)] = torch.tensor(values, dtype=torch.float32)
    summary = average_incremental_metrics(
        session_accuracies=session_accuracies,
        old_accuracies=old_accuracies,
        new_accuracies=new_accuracies,
        accuracy_matrix=matrix,
        average_denominator=str(config.get("average_denominator", "incremental_sessions")),
        forgetting_denominator=str(config.get("forgetting_denominator", "incremental_sessions")),
    )
    result = {
        "method_id": "orthogonal_incremental_sei",
        "mode": "formal_wisig_fscil",
        "claim_boundary": "formal_wisig_closed_set_fscil_not_adsb_not_cvs_stage2",
        "seed": seed,
        "pseudo_target_steps": pseudo_target_steps,
        "base_classes": base_classes,
        "increment_classes_per_session": inc_classes,
        "num_increment_sessions": inc_sessions,
        "shot": int(config.get("shot", config.get("k_shot", 5))),
        "session_accuracies": session_accuracies,
        "old_accuracies": old_accuracies,
        "new_accuracies": new_accuracies,
        "accuracy_matrix": matrix.tolist(),
        "summary": summary,
        "base_weight_source": base_weight_source,
        "early_stop": early_stop_info,
        "split_info": split_info,
        "history": history,
    }
    out_dir = Path(run_dir)
    write_json(out_dir / "metrics.json", result)
    write_json(out_dir / "resolved_config.json", config)
    write_json(out_dir / "split_manifest.json", split_info)
    return result


def run_new_label_scan(config: dict, *, wisig_pkl: str, run_dir: str | Path, device: str = "cuda:0") -> dict[str, Any]:
    seed = int(config.get("seed", 1337))
    set_seed(seed)
    dev = torch.device(device)
    train_by_class, query_by_class, candidate_by_class, split_info = _load_wisig_new_label_scan_tensors(config, wisig_pkl=wisig_pkl, seed=seed)
    feature_dim = int(config.get("embedding_dim", 256))
    base_classes = int(config.get("base_classes", 60))
    eval_batch_size = int(config.get("eval_batch_size", int(config.get("batch_size", 128))))
    encoder = SixBlockConv1DEncoder(input_channels=2, embedding_dim=feature_dim).to(dev)
    pseudo_targets, pseudo_target_steps = _build_pseudo_targets(config, device=dev)
    perturbed = perturb_pseudo_targets(pseudo_targets, noise_range=float(config.get("noise_range", 0.01)), seed=seed).to(dev)
    assigned = assign_base_targets(range(base_classes), pseudo_targets)
    history = _train_base_encoder(
        config,
        train_by_class,
        base_classes=base_classes,
        pseudo_targets=pseudo_targets,
        perturbed=perturbed,
        assigned=assigned,
        encoder=encoder,
        device=dev,
    )
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    feature_mean_base_weights = _fit_base_classifier_weights(
        encoder,
        train_by_class,
        base_classes=base_classes,
        device=dev,
        batch_size=eval_batch_size,
    )
    base_weights, base_weight_source = _base_classifier_weights(
        assigned,
        feature_mean_base_weights,
        base_classes=base_classes,
        source=str(config.get("base_weight_source", "paper_pseudo_targets")),
    )
    base_acc = _evaluate_class_group_accuracy(
        encoder,
        query_by_class,
        base_weights,
        seen_class_ids=list(range(base_classes)),
        eval_class_ids=list(range(base_classes)),
        device=dev,
        batch_size=eval_batch_size,
    )
    ranked: list[dict[str, Any]] = []
    tx_labels = split_info["candidate_tx_labels"]
    tx_indices = split_info["candidate_tx_indices"]
    scan_epochs = int(config.get("scan_increment_epochs", config.get("increment_epochs", 50)))
    for rank_offset, class_id in enumerate(split_info["candidate_class_ids"]):
        candidate_offset = int(class_id) - base_classes
        support_x, query_x = candidate_by_class[int(class_id)]
        support_y = torch.full((support_x.size(0),), int(class_id), dtype=torch.long)
        support_features = _extract_features(encoder, support_x, device=dev, batch_size=eval_batch_size).to(dev)
        support_y = support_y.to(dev)
        init_weights, class_ids_tensor = class_mean_weights(support_features, support_y)
        new_weights = nn.Parameter(init_weights.detach().clone())
        inc_optimizer = torch.optim.SGD([new_weights], lr=float(config.get("increment_lr", 0.08)))
        old_weights = base_weights.to(dev)
        last_loss = 0.0
        last_hard_count = 0.0
        for _epoch in range(1, scan_epochs + 1):
            inc_optimizer.zero_grad(set_to_none=True)
            inc_loss, inc_terms = incremental_calibration_loss(
                support_features,
                support_y,
                old_weights,
                new_weights,
                new_class_ids=class_ids_tensor,
                prototypes=init_weights.detach(),
                top_k=int(config.get("top_k", 60)),
                margin=_paper_float(config, "margin", "q", default=0.2),
                tau_fuse=float(config.get("tau_fuse", 0.01)),
                lambda_align=_paper_float(config, "lambda_align", "lambda_a", default=1.6),
            )
            inc_loss.backward()
            inc_optimizer.step()
            last_loss = float(inc_loss.detach().cpu().item())
            last_hard_count = float(inc_terms["hard_count"].detach().cpu().item())
        scan_query_by_class = dict(query_by_class)
        scan_query_by_class[int(class_id)] = query_x
        scan_weights = concat_classifier_weights(base_weights, new_weights.detach().cpu())
        seen_class_ids = list(range(base_classes)) + [int(class_id)]
        candidate_acc = _evaluate_class_group_accuracy(
            encoder,
            scan_query_by_class,
            scan_weights,
            seen_class_ids=seen_class_ids,
            eval_class_ids=[int(class_id)],
            device=dev,
            batch_size=eval_batch_size,
        )
        old_acc = _evaluate_class_group_accuracy(
            encoder,
            scan_query_by_class,
            scan_weights,
            seen_class_ids=seen_class_ids,
            eval_class_ids=list(range(base_classes)),
            device=dev,
            batch_size=eval_batch_size,
        )
        combined_acc, _task_accs = _evaluate_sessions(
            encoder,
            scan_query_by_class,
            scan_weights,
            seen_class_ids=seen_class_ids,
            task_class_ids=[list(range(base_classes)), [int(class_id)]],
            device=dev,
            batch_size=eval_batch_size,
        )
        ranked.append(
            {
                "candidate_rank_input": int(rank_offset),
                "class_id": int(class_id),
                "tx_index": int(tx_indices[candidate_offset]),
                "tx_label": tx_labels[candidate_offset],
                "support_count": int(support_x.size(0)),
                "query_count": int(query_x.size(0)),
                "candidate_acc": candidate_acc,
                "old_acc_with_candidate": old_acc,
                "combined_acc": combined_acc,
                "last_increment_loss": last_loss,
                "last_hard_count": last_hard_count,
            }
        )
    ranked.sort(key=lambda row: (float(row["candidate_acc"]), float(row["combined_acc"])), reverse=True)
    recommended_count = int(config.get("recommend_new_label_count", 70))
    result = {
        "method_id": "orthogonal_incremental_sei_new_label_scan",
        "mode": "new_label_scan",
        "claim_boundary": "diagnostic_candidate_tx_scan_not_full_incremental_reproduction",
        "seed": seed,
        "pseudo_target_steps": pseudo_target_steps,
        "base_classes": base_classes,
        "shot": int(config.get("shot", config.get("k_shot", 5))),
        "base_weight_source": base_weight_source,
        "base_acc": base_acc,
        "base_labels": split_info["base_tx_labels"],
        "ranked_candidates": ranked,
        "recommended_new_labels": [row["tx_label"] for row in ranked[:recommended_count]],
        "split_info": split_info,
        "history": history,
    }
    out_dir = Path(run_dir)
    write_json(out_dir / "new_label_scan.json", result)
    write_json(out_dir / "resolved_config.json", config)
    return result


def run_dry_run(config: dict, *, device: str = "cpu") -> dict[str, object]:
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
    targets, pseudo_target_steps = _build_pseudo_targets(config, device=dev)
    perturbed = perturb_pseudo_targets(targets, noise_range=float(config.get("noise_range", 0.01)), seed=seed)
    assigned = assign_base_targets(range(base_classes), targets)
    base_loss, _ = base_training_loss(
        features,
        labels,
        assigned,
        targets,
        perturbed,
        contrast_temperature=_paper_float(config, "contrast_temperature", "tau_s", default=0.1),
        center_temperature=_paper_float(config, "center_temperature", "tau_c", default=0.1),
    )

    old_weights = torch.stack([assigned[index] for index in range(base_classes)], dim=0).to(dev)
    new_x = torch.randn(4, 2, int(config.get("input_length", 256)), device=dev)
    new_labels = torch.tensor([base_classes, base_classes, base_classes + 1, base_classes + 1], device=dev)
    for parameter in encoder.parameters():
        parameter.grad = None
        parameter.requires_grad_(False)
    encoder.eval()
    with torch.no_grad():
        new_features = encoder(new_x)
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
        margin=_paper_float(config, "margin", "q", default=0.2),
        tau_fuse=float(config.get("tau_fuse", 0.01)),
        lambda_align=_paper_float(config, "lambda_align", "lambda_a", default=1.6),
    )
    inc_loss.backward()
    grad_norm = float(new_weights.grad.detach().norm().cpu().item()) if new_weights.grad is not None else 0.0
    optimizer.step()
    encoder_grad = 0.0
    for parameter in encoder.parameters():
        if parameter.grad is not None:
            encoder_grad += float(parameter.grad.detach().abs().sum().cpu().item())
    encoder_trainable = sum(1 for parameter in encoder.parameters() if parameter.requires_grad)
    return {
        "mode": "dry-run",
        "claim_boundary": "synthetic_dry_run_not_formal_reproduction",
        "unsupported_config_fields": _unsupported_config_fields(config),
        "seed": seed,
        "pseudo_target_steps": pseudo_target_steps,
        "base_loss": float(base_loss.detach().cpu().item()),
        "incremental_loss": float(inc_loss.detach().cpu().item()),
        "hard_count": int(inc_terms["hard_count"].detach().cpu().item()),
        "incremental_grad_norm": grad_norm,
        "encoder_grad_after_increment": encoder_grad,
        "encoder_trainable_after_increment": encoder_trainable,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Paper-faithful OSC-FSCIL SEI reproduction entrypoint")
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--formal", action="store_true")
    parser.add_argument("--scan-new-labels", action="store_true")
    parser.add_argument("--wisig-pkl", default="")
    parser.add_argument("--run-dir", default="runs/orthogonal_incremental_sei")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.dry_run:
        print(json.dumps(run_dry_run(config, device=args.device), ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.formal:
        wisig_pkl = args.wisig_pkl or str(config.get("wisig_pkl", ""))
        if not wisig_pkl:
            raise SystemExit("formal WiSig training requires --wisig-pkl or config.wisig_pkl")
        print(
            json.dumps(
                run_formal_wisig(config, wisig_pkl=wisig_pkl, run_dir=args.run_dir, device=args.device),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.scan_new_labels:
        wisig_pkl = args.wisig_pkl or str(config.get("wisig_pkl", ""))
        if not wisig_pkl:
            raise SystemExit("new-label scan requires --wisig-pkl or config.wisig_pkl")
        print(
            json.dumps(
                run_new_label_scan(config, wisig_pkl=wisig_pkl, run_dir=args.run_dir, device=args.device),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    raise SystemExit("use --dry-run for wiring verification or --formal for WiSig FSCIL training")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
