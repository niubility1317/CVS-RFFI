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


def _load_wisig_fscil_tensors(config: dict, *, wisig_pkl: str, seed: int) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor], dict[str, Any]]:
    ds = load_wisig_compact_pkl(wisig_pkl)
    total_classes = int(config.get("base_classes", 60)) + int(config.get("increment_classes_per_session", 10)) * int(config.get("num_increment_sessions", 7))
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
    for class_id in range(total_classes):
        train_x, query_x = _split_class_samples(
            grouped[class_id],
            train_ratio=train_ratio,
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
        history.append({"phase": "base", "epoch": epoch, "loss": float(sum(losses) / max(1, len(losses)))})

    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    base_weights = torch.stack([assigned[i].detach().cpu() for i in range(base_classes)], dim=0)
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
        previous_tasks = task_accs[:-1]
        old_accuracies.append(float(sum(previous_tasks) / max(1, len(previous_tasks))))
        new_accuracies.append(float(task_accs[-1]))
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
        "split_info": split_info,
        "history": history,
    }
    out_dir = Path(run_dir)
    write_json(out_dir / "metrics.json", result)
    write_json(out_dir / "resolved_config.json", config)
    write_json(out_dir / "split_manifest.json", split_info)
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
    raise SystemExit("use --dry-run for wiring verification or --formal for WiSig FSCIL training")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
