#!/usr/bin/env python
"""Ground-pretrain a tiny ADV3B02 adapter with exact rx_light5 supervision.

Only source-role rows are consumed.  Six source receivers train the adapter and
one preregistered source receiver is held out for checkpoint selection.  Each
formal LEO scenario is followed by the exact historical five receive-side TTA
views (base, shift -2/+2, CFO -1e-4/+1e-4).  Ground training sees all five views
of one rotating scenario per optimizer step; onboard deployment still defaults
to one view.  The exported adapter contains no optimizer state and initializes
the separate support-only onboard trainer.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = REPO_ROOT / "code"
for candidate in (str(REPO_ROOT), str(CODE_ROOT)):
    while candidate in sys.path:
        sys.path.remove(candidate)
for candidate in (str(REPO_ROOT), str(CODE_ROOT)):
    sys.path.insert(0, candidate)

from cvsrffi.checkpoint_loading import build_exact_ssdg_model_from_checkpoint
from export_spaceborne_features import _satellite_tta_views
from paper_reproduction.cvs_aligned.cvs_method_runner import SCENARIOS
from paper_reproduction.scripts.train_export_cvs_micro_iq_adapter import (
    _feature_forward,
    _json_safe,
    _load_npz,
    _norm_rows,
    _numpy_to_tensor_compat,
    _sha256_file,
    _write_trace,
)
from paper_reproduction.scripts.train_export_cvs_support_lora_adapter import (
    enable_late_key_layer_finetune,
    inject_late_channel_film,
)


RX_LIGHT5_VIEW_NAMES = (
    "rx_base",
    "rx_shift_m2",
    "rx_shift_p2",
    "rx_cfo_m1e4",
    "rx_cfo_p1e4",
)


def select_source_ground_split(
    arrays: dict[str, np.ndarray],
    *,
    source_receivers: Sequence[str],
    val_receiver: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Return source-only train/validation positions and a role audit."""

    roles = arrays["dataset_role"].astype(str)
    receivers = arrays["rx_ids"].astype(str)
    source_mask = roles == "source"
    observed_source_receivers = set(receivers[source_mask].tolist())
    expected_receivers = {str(value) for value in source_receivers}
    if observed_source_receivers != expected_receivers:
        raise ValueError(
            "source receiver mismatch: "
            f"observed={sorted(observed_source_receivers)}, "
            f"expected={sorted(expected_receivers)}"
        )
    if str(val_receiver) not in expected_receivers:
        raise ValueError("val_receiver must be one of source_receiver_labels")
    train_mask = source_mask & (receivers != str(val_receiver))
    val_mask = source_mask & (receivers == str(val_receiver))
    train_positions = np.flatnonzero(train_mask).astype(np.int64)
    val_positions = np.flatnonzero(val_mask).astype(np.int64)
    if not len(train_positions) or not len(val_positions):
        raise ValueError("source ground split is empty")
    consumed_roles = set(roles[np.concatenate([train_positions, val_positions])].tolist())
    if consumed_roles != {"source"}:
        raise ValueError(f"ground pretraining consumed non-source roles: {consumed_roles}")
    return train_positions, val_positions, {
        "train_receiver_labels": sorted(expected_receivers - {str(val_receiver)}),
        "validation_receiver_label": str(val_receiver),
        "train_count": int(len(train_positions)),
        "validation_count": int(len(val_positions)),
        "consumed_roles": sorted(consumed_roles),
        "target_row_count": 0,
        "target_query_row_count": 0,
    }


def _load_exact_rx_light5_source_views(
    scenario_arrays: dict[str, dict[str, np.ndarray]],
    source_positions: np.ndarray,
) -> tuple[torch.Tensor, tuple[str, ...], dict[str, Any]]:
    """Apply exact rx_light5 to each cached formal LEO observation."""

    views: list[torch.Tensor] = []
    names: list[str] = []
    audit: dict[str, Any] = {}
    reference_shape: tuple[int, ...] | None = None
    for scenario in SCENARIOS:
        arrays = scenario_arrays[str(scenario)]
        rows = _numpy_to_tensor_compat(
            arrays["raw_iq"][source_positions],
            numpy_dtype=np.dtype(np.float32),
            torch_dtype=torch.float32,
        )
        if reference_shape is None:
            reference_shape = tuple(rows.shape)
        elif tuple(rows.shape) != reference_shape:
            raise ValueError(
                f"formal scenario raw_iq shape mismatch for {scenario}: "
                f"{tuple(rows.shape)} != {reference_shape}"
            )
        generated = _satellite_tta_views(rows, "rx_light5")
        observed_names = tuple(name for name, _ in generated)
        if observed_names != RX_LIGHT5_VIEW_NAMES:
            raise ValueError(
                f"rx_light5 definition drift: {observed_names} != "
                f"{RX_LIGHT5_VIEW_NAMES}"
            )
        audit[str(scenario)] = {
            "physical_rows": int(rows.shape[0]),
            "tta_policy": "rx_light5",
            "tta_view_count": 5,
            "view_names": list(observed_names),
        }
        for view_name, view_rows in generated:
            if not bool(torch.isfinite(view_rows).all()):
                raise FloatingPointError(
                    f"non-finite source view: {scenario}/{view_name}"
                )
            names.append(f"{scenario}/{view_name}")
            views.append(view_rows.detach().cpu().float())
    return torch.stack(views, dim=0), tuple(names), audit


@torch.no_grad()
def _forward_views(
    model: nn.Module,
    views: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    feature_views: list[torch.Tensor] = []
    logit_views: list[torch.Tensor] = []
    model.eval()
    for view in views:
        features: list[torch.Tensor] = []
        logits: list[torch.Tensor] = []
        for start in range(0, int(view.shape[0]), int(batch_size)):
            batch = view[start : start + int(batch_size)].to(device)
            z, score = _feature_forward(model, batch)
            features.append(z.detach().cpu().float())
            logits.append(score.detach().cpu().float())
        feature_views.append(torch.cat(features, dim=0))
        logit_views.append(torch.cat(logits, dim=0))
    return torch.stack(feature_views, dim=0), torch.stack(logit_views, dim=0)


@torch.no_grad()
def evaluate_source_views(
    model: nn.Module,
    views: torch.Tensor,
    labels: torch.Tensor,
    positions: torch.Tensor,
    teacher_mean: torch.Tensor,
    view_names: Sequence[str],
    *,
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    accuracies: dict[str, float] = {}
    teacher_cosines: dict[str, float] = {}
    if len(view_names) != int(views.shape[0]):
        raise ValueError("view_names length does not match source view tensor")
    for view_index, view_name in enumerate(view_names):
        correct = 0
        total = 0
        cosine_sum = 0.0
        for start in range(0, int(positions.numel()), int(batch_size)):
            batch_positions = positions[start : start + int(batch_size)]
            rows = views[view_index, batch_positions].to(device)
            truth = labels[batch_positions].to(device)
            teacher = teacher_mean[batch_positions].to(device)
            z, logits = _feature_forward(model, rows)
            correct += int((logits.argmax(dim=1) == truth).sum())
            total += int(truth.numel())
            cosine_sum += float(
                torch.sum(torch.sum(_norm_rows(z) * teacher, dim=1)).detach()
            )
        accuracies[view_name] = float(correct / max(1, total))
        teacher_cosines[view_name] = float(cosine_sum / max(1, total))
    return {
        "accuracy_by_view": accuracies,
        "mean_accuracy": float(sum(accuracies.values()) / len(accuracies)),
        "min_accuracy": float(min(accuracies.values())),
        "teacher_cosine_by_view": teacher_cosines,
        "mean_teacher_cosine": float(
            sum(teacher_cosines.values()) / len(teacher_cosines)
        ),
    }


def train_ground_source_film(
    model: nn.Module,
    views: torch.Tensor,
    labels: torch.Tensor,
    train_positions: torch.Tensor,
    val_positions: torch.Tensor,
    teacher_mean: torch.Tensor,
    *,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    teacher_weight: float,
    batch_size: int,
    grad_clip: float,
    max_optimizer_steps: int,
    view_names: Sequence[str],
    multiview_consistency_weight: float,
    seed: int,
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, torch.Tensor], dict[str, Any]]:
    """Train only FiLM tensors and select solely on held-out source receiver."""

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    trainable_count = int(sum(parameter.numel() for parameter in trainable))
    if trainable_count not in {1_280, 31_200}:
        raise ValueError(
            "ground source trainer requires exactly 1,280 FiLM or 31,200 "
            f"late-key parameters, got {trainable_count}"
        )
    if int(views.shape[0]) != len(SCENARIOS) * len(RX_LIGHT5_VIEW_NAMES):
        raise ValueError("ground trainer requires 3 formal scenarios x exact rx_light5")
    if len(view_names) != int(views.shape[0]):
        raise ValueError("view_names length does not match source view tensor")
    optimizer = torch.optim.AdamW(
        trainable, lr=float(learning_rate), weight_decay=float(weight_decay)
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    trace: list[dict[str, Any]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_key = (-float("inf"), -float("inf"), -float("inf"))
    optimizer_steps = 0
    started = time.perf_counter()
    peak_memory = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch_index in range(int(epochs)):
        epoch_started = time.perf_counter()
        scenario_index = int(epoch_index % len(SCENARIOS))
        view_start = scenario_index * len(RX_LIGHT5_VIEW_NAMES)
        view_stop = view_start + len(RX_LIGHT5_VIEW_NAMES)
        order = train_positions[torch.randperm(train_positions.numel(), generator=generator)]
        totals = {
            "loss": 0.0,
            "ce": 0.0,
            "teacher": 0.0,
            "multiview_consistency": 0.0,
            "correct": 0.0,
        }
        seen = 0
        gradient_norm = 0.0
        model.eval()
        for start in range(0, int(order.numel()), int(batch_size)):
            if int(max_optimizer_steps) > 0 and optimizer_steps >= int(max_optimizer_steps):
                break
            batch_positions = order[start : start + int(batch_size)]
            rows = views[view_start:view_stop, batch_positions].to(device)
            view_count, physical_count = int(rows.shape[0]), int(rows.shape[1])
            truth_physical = labels[batch_positions].to(device)
            truth = truth_physical.repeat(view_count)
            teacher = teacher_mean[batch_positions].to(device)
            z_flat, logits = _feature_forward(
                model, rows.reshape(view_count * physical_count, *rows.shape[2:])
            )
            z = _norm_rows(z_flat).reshape(view_count, physical_count, -1)
            if int(logits.shape[1]) <= int(truth.max()):
                raise ValueError("source raw_labels exceed the frozen classifier width")
            ce = F.cross_entropy(logits, truth)
            teacher_loss = torch.mean(
                1.0 - torch.sum(z * teacher.unsqueeze(0), dim=2)
            )
            consensus = _norm_rows(z.mean(dim=0))
            consistency = torch.mean(
                1.0 - torch.sum(z * consensus.detach().unsqueeze(0), dim=2)
            )
            loss = (
                ce
                + float(teacher_weight) * teacher_loss
                + float(multiview_consistency_weight) * consistency
            )
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("non-finite ground source loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = float(
                torch.nn.utils.clip_grad_norm_(trainable, max_norm=float(grad_clip))
            )
            optimizer.step()
            optimizer_steps += 1
            count = int(truth.numel())
            totals["loss"] += float(loss.detach()) * count
            totals["ce"] += float(ce.detach()) * count
            totals["teacher"] += float(teacher_loss.detach()) * count
            totals["multiview_consistency"] += float(consistency.detach()) * count
            totals["correct"] += float((logits.argmax(dim=1) == truth).sum())
            seen += count
        validation = evaluate_source_views(
            model,
            views,
            labels,
            val_positions,
            teacher_mean,
            view_names,
            batch_size=int(batch_size),
            device=device,
        )
        key = (
            float(validation["min_accuracy"]),
            float(validation["mean_accuracy"]),
            float(validation["mean_teacher_cosine"]),
        )
        if key > best_key:
            best_key = key
            best_state = {
                name: parameter.detach().cpu().clone()
                for name, parameter in model.named_parameters()
                if parameter.requires_grad
            }
        row = {
            "epoch": int(epoch_index + 1),
            "train_scenario_index": int(scenario_index),
            "train_scenario": SCENARIOS[scenario_index],
            "train_tta_policy": "rx_light5",
            "train_view_count_per_physical_sample": len(RX_LIGHT5_VIEW_NAMES),
            "loss": float(totals["loss"] / max(1, seen)),
            "source_ce": float(totals["ce"] / max(1, seen)),
            "teacher_loss": float(totals["teacher"] / max(1, seen)),
            "multiview_consistency_loss": float(
                totals["multiview_consistency"] / max(1, seen)
            ),
            "train_accuracy": float(totals["correct"] / max(1, seen)),
            "validation_min_accuracy": float(validation["min_accuracy"]),
            "validation_mean_accuracy": float(validation["mean_accuracy"]),
            "validation_mean_teacher_cosine": float(
                validation["mean_teacher_cosine"]
            ),
            "validation_accuracy_by_view": json.dumps(
                validation["accuracy_by_view"], sort_keys=True
            ),
            "gradient_norm": float(gradient_norm),
            "optimizer_steps": int(optimizer_steps),
            "epoch_seconds": float(time.perf_counter() - epoch_started),
        }
        if not all(
            math.isfinite(float(value))
            for value in row.values()
            if isinstance(value, (int, float))
        ):
            raise FloatingPointError(f"non-finite ground trace: {row}")
        trace.append(row)
        print("[GROUND-FILM-EPOCH] " + json.dumps(row, sort_keys=True), flush=True)
        if int(max_optimizer_steps) > 0 and optimizer_steps >= int(max_optimizer_steps):
            break
    if best_state is None:
        raise RuntimeError("ground source training did not produce a checkpoint")
    with torch.no_grad():
        named = dict(model.named_parameters())
        for name, value in best_state.items():
            named[name].copy_(value.to(device=named[name].device, dtype=named[name].dtype))
    if device.type == "cuda":
        peak_memory = int(torch.cuda.max_memory_allocated(device))
    runtime = {
        "ground_wall_seconds": float(time.perf_counter() - started),
        "optimizer": "adamw_ground_only",
        "optimizer_steps": int(optimizer_steps),
        "max_optimizer_steps": int(max_optimizer_steps),
        "peak_cuda_memory_bytes": int(peak_memory),
        "deployment_optimizer_state_required": False,
        "deployment_optimizer_state_bytes": 0,
        "selection_key": {
            "validation_min_accuracy": float(best_key[0]),
            "validation_mean_accuracy": float(best_key[1]),
            "validation_mean_teacher_cosine": float(best_key[2]),
        },
    }
    return trace, best_state, runtime


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--out_root", type=Path, required=True)
    parser.add_argument("--source_cache", type=Path, default=None)
    parser.add_argument("--val_receiver", default="2-19")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=1.0e-3)
    parser.add_argument("--weight_decay", type=float, default=1.0e-4)
    parser.add_argument("--teacher_weight", type=float, default=0.25)
    parser.add_argument("--multiview_consistency_weight", type=float, default=0.5)
    parser.add_argument(
        "--adapter_type", choices=("late_film", "late_key_ft"), default="late_key_ft"
    )
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--max_optimizer_steps", type=int, default=400)
    parser.add_argument("--seed", type=int, default=713101)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    if not 1 <= int(args.epochs) <= 30:
        raise ValueError("ground epochs must be in [1,30]")
    if not 1 <= int(args.max_optimizer_steps) <= 500:
        raise ValueError("ground max_optimizer_steps must be in [1,500]")
    if float(args.teacher_weight) < 0.0:
        raise ValueError("teacher_weight must be nonnegative")
    if float(args.multiview_consistency_weight) < 0.0:
        raise ValueError("multiview_consistency_weight must be nonnegative")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = json.loads(args.config.read_text(encoding="utf-8-sig"))
    source_receivers = [str(value) for value in config["source_receiver_labels"]]
    mapping = dict(config.get("feature_npz_by_scenario", {}))
    if set(mapping) != set(SCENARIOS):
        raise ValueError(f"config must map exactly the formal scenarios: {SCENARIOS}")
    source_cache = Path(args.source_cache or mapping[SCENARIOS[0]])
    arrays, source_manifest = _load_npz(source_cache)
    scenario_arrays: dict[str, dict[str, np.ndarray]] = {}
    scenario_manifests: dict[str, dict[str, Any]] = {}
    for scenario in SCENARIOS:
        scenario_path = Path(mapping[str(scenario)])
        scenario_arrays[str(scenario)], scenario_manifests[str(scenario)] = _load_npz(
            scenario_path
        )
        candidate = scenario_arrays[str(scenario)]
        for key in ("dataset_role", "rx_ids", "raw_labels"):
            if not np.array_equal(candidate[key], arrays[key]):
                raise ValueError(f"formal scenario row alignment mismatch: {scenario}/{key}")
    train_positions_raw, val_positions_raw, split_audit = select_source_ground_split(
        arrays,
        source_receivers=source_receivers,
        val_receiver=str(args.val_receiver),
    )
    source_positions = np.concatenate([train_positions_raw, val_positions_raw])
    source_lookup = {int(position): index for index, position in enumerate(source_positions)}
    train_positions = torch.as_tensor(
        [source_lookup[int(position)] for position in train_positions_raw],
        dtype=torch.long,
    )
    val_positions = torch.as_tensor(
        [source_lookup[int(position)] for position in val_positions_raw],
        dtype=torch.long,
    )
    clean_rows = _numpy_to_tensor_compat(
        arrays["raw_iq"][source_positions],
        numpy_dtype=np.dtype(np.float32),
        torch_dtype=torch.float32,
    )
    labels = _numpy_to_tensor_compat(
        arrays["raw_labels"][source_positions],
        numpy_dtype=np.dtype(np.int64),
        torch_dtype=torch.int64,
    )
    unique_labels = sorted(int(value) for value in torch.unique(labels).tolist())
    if unique_labels != list(range(len(unique_labels))):
        raise ValueError(f"source raw_labels must be contiguous: {unique_labels}")
    device = torch.device(str(args.device) if torch.cuda.is_available() else "cpu")
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed) % (2**32))
    views, view_names, view_audit = _load_exact_rx_light5_source_views(
        scenario_arrays,
        source_positions,
    )
    checkpoint = torch.load(args.ckpt, map_location="cpu")
    model, checkpoint_load_audit = build_exact_ssdg_model_from_checkpoint(
        checkpoint, input_len=int(clean_rows.shape[-1]), device=device
    )
    if str(getattr(model, "id_feature_key", "")) != "feat_joint":
        raise ValueError("ground FiLM pretraining requires feat_joint ADV3B02")
    base_features, base_logits = _forward_views(
        model,
        views,
        batch_size=int(args.batch_size),
        device=device,
    )
    if int(base_logits.shape[-1]) != len(unique_labels):
        raise ValueError(
            f"source classifier width mismatch: {base_logits.shape[-1]} vs {len(unique_labels)}"
        )
    teacher_mean = _norm_rows(_norm_rows(base_features).mean(dim=0))
    baseline_validation = {
        "accuracy_by_view": {
            view_names[index]: float(
                (base_logits[index, val_positions].argmax(dim=1) == labels[val_positions])
                .float()
                .mean()
            )
            for index in range(len(view_names))
        }
    }
    baseline_validation["mean_accuracy"] = float(
        sum(baseline_validation["accuracy_by_view"].values()) / len(view_names)
    )
    baseline_validation["min_accuracy"] = float(
        min(baseline_validation["accuracy_by_view"].values())
    )
    if str(args.adapter_type) == "late_film":
        resources = inject_late_channel_film(model)
    else:
        resources = enable_late_key_layer_finetune(model)
    model.to(device).eval()
    trace, best_state, runtime = train_ground_source_film(
        model,
        views,
        labels,
        train_positions,
        val_positions,
        teacher_mean,
        epochs=int(args.epochs),
        learning_rate=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        teacher_weight=float(args.teacher_weight),
        batch_size=int(args.batch_size),
        grad_clip=float(args.grad_clip),
        max_optimizer_steps=int(args.max_optimizer_steps),
        view_names=view_names,
        multiview_consistency_weight=float(args.multiview_consistency_weight),
        seed=int(args.seed),
        device=device,
    )
    final_validation = evaluate_source_views(
        model,
        views,
        labels,
        val_positions,
        teacher_mean,
        view_names,
        batch_size=int(args.batch_size),
        device=device,
    )
    run_id = (
        f"ground_source_rxlight5_{args.adapter_type}_seed_{args.seed}"
        f"_valrx_{args.val_receiver}"
    )
    run_dir = args.out_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_trace(run_dir / "loss_trace.json", trace)
    state_path = run_dir / f"ground_{args.adapter_type}_state_fp16.pt"
    torch.save({name: value.half() for name, value in best_state.items()}, state_path)
    resources["adapter_state_file_bytes_fp16_pt"] = int(state_path.stat().st_size)
    manifest = {
        "method": f"ground_source_rxlight5_{args.adapter_type}_v1",
        "stage": "ground_pretraining",
        "source_only": True,
        "target_rows_used": False,
        "target_query_rows_used": False,
        "target_labels_used": False,
        "old_new_role_used_by_optimizer": False,
        "class_quota_used": False,
        "deployment_query_view_count": 1,
        "ground_teacher_view_count": len(view_names),
        "ground_formal_scenario_count": len(SCENARIOS),
        "ground_rx_light5_view_count_per_scenario": len(RX_LIGHT5_VIEW_NAMES),
        "ground_train_views_per_sample_per_epoch": len(RX_LIGHT5_VIEW_NAMES),
        "ground_view_names": list(view_names),
        "split": split_audit,
        "view_generation": view_audit,
        "baseline_source_validation": baseline_validation,
        "selected_source_validation": final_validation,
        "hyperparameters": {
            "epochs": int(args.epochs),
            "learning_rate": float(args.learning_rate),
            "weight_decay": float(args.weight_decay),
            "teacher_weight": float(args.teacher_weight),
            "multiview_consistency_weight": float(
                args.multiview_consistency_weight
            ),
            "adapter_type": str(args.adapter_type),
            "batch_size": int(args.batch_size),
            "grad_clip": float(args.grad_clip),
            "max_optimizer_steps": int(args.max_optimizer_steps),
            "seed": int(args.seed),
        },
        "resources": resources,
        "runtime": runtime,
        "checkpoint": str(args.ckpt),
        "checkpoint_sha256": _sha256_file(args.ckpt),
        "checkpoint_load_audit": checkpoint_load_audit,
        "source_cache": str(source_cache),
        "source_cache_sha256": _sha256_file(source_cache),
        "source_cache_manifest": source_manifest,
        "formal_scenario_manifests": scenario_manifests,
        "adapter_state": str(state_path),
        "adapter_state_sha256": _sha256_file(state_path),
        "loss_trace_json": str(run_dir / "loss_trace.json"),
        "loss_trace_csv": str(run_dir / "loss_trace.csv"),
    }
    manifest_path = run_dir / "ground_training_manifest.json"
    manifest_path.write_text(
        json.dumps(_json_safe(manifest), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "run_id": run_id,
                "run_dir": str(run_dir),
                "adapter_state": str(state_path),
                "resources": resources,
                "baseline_source_validation": baseline_validation,
                "selected_source_validation": final_validation,
                "runtime": runtime,
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
