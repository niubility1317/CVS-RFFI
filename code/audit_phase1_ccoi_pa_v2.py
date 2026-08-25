"""Source-only causal audit for a frozen CCOI-PA-V2 C4 sidecar."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from cvsrffi.ccoi_causal_audit import (
    build_factor_indices,
    complementarity_table,
    group_paired_bootstrap,
    pair_relation_sweep,
    token_code_audit,
)
from cvsrffi.ccoi_pa import CCOIPASidecar, PAChallengeEncoder
from cvsrffi.checkpoint_loading import build_exact_ssdg_model_from_checkpoint
from cvsrffi.leakage_probe import frozen_ridge_linear_probe
from train_phase1_ccoi_pa import (
    FrozenCore90CCOI,
    SCENARIOS,
    _infer_base_dimensions,
    _limited,
    _meta_value,
    _move_batch,
    _prepare_ssdg_args,
    _satellite_view,
    _torch_load,
    freeze_base_model,
    validate_source_roles,
)


SIDECAR_SCHEMA = "cvs.phase1.ccoi_pa_sidecar.v2"
SOURCE_ROLE_RATIOS = (0.07, 0.63, 0.15, 0.15)


def _probe_metrics(prediction: Tensor, labels: Tensor, classes: Tensor) -> dict[str, Any]:
    prediction = prediction.detach().view(-1).long().cpu()
    labels = labels.detach().view(-1).long().cpu()
    if prediction.numel() != labels.numel() or labels.numel() == 0:
        raise ValueError("probe prediction and labels must have the same non-zero size")
    class_accuracy = []
    for value in classes.tolist():
        mask = labels.eq(int(value))
        if bool(mask.any()):
            class_accuracy.append(prediction[mask].eq(labels[mask]).float().mean())
    balanced = float(torch.stack(class_accuracy).mean().item())
    chance = 1.0 / float(max(1, len(class_accuracy)))
    return {
        "eval_count": int(labels.numel()),
        "class_count": int(len(class_accuracy)),
        "accuracy": float(prediction.eq(labels).float().mean().item()),
        "balanced_accuracy": balanced,
        "balanced_chance_accuracy": chance,
        "normalized_gain": (balanced - chance) / max(1e-12, 1.0 - chance),
    }


def _prepare_probe_data(
    train_features: Tensor,
    train_labels: Tensor,
    eval_features: Tensor,
    eval_labels: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    train_x = train_features.detach().float().cpu()
    eval_x = eval_features.detach().float().cpu()
    train_y = train_labels.detach().view(-1).long().cpu()
    eval_y = eval_labels.detach().view(-1).long().cpu()
    if train_x.ndim != 2 or eval_x.ndim != 2 or train_x.size(1) != eval_x.size(1):
        raise ValueError("probe features must share two-dimensional geometry")
    if train_x.size(0) != train_y.numel() or eval_x.size(0) != eval_y.numel():
        raise ValueError("probe features and labels must align")
    classes = torch.unique(train_y[train_y >= 0], sorted=True)
    if classes.numel() < 2:
        raise ValueError("probe training requires at least two classes")
    train_valid = train_y >= 0
    eval_known = torch.isin(eval_y, classes)
    train_x, train_y = train_x[train_valid], train_y[train_valid]
    eval_x, eval_y = eval_x[eval_known], eval_y[eval_known]
    if eval_y.numel() == 0:
        raise ValueError("probe evaluation has no labels represented in training")
    return train_x, train_y, eval_x, eval_y, classes


def fit_torch_probe(
    train_features: Tensor,
    train_labels: Tensor,
    eval_features: Tensor,
    eval_labels: Tensor,
    *,
    steps: int,
    seed: int,
    hidden_dim: int = 64,
    batch_size: int = 512,
    device: Optional[torch.device] = None,
    eval_groups: Optional[Tensor] = None,
    bootstrap_resamples: int = 0,
) -> dict[str, Any]:
    """Fit a bounded frozen-feature probe without updating the audited encoder."""

    original_eval_labels = eval_labels.detach().view(-1).long().cpu()
    train_x, train_y, eval_x, eval_y, classes = _prepare_probe_data(
        train_features, train_labels, eval_features, eval_labels
    )
    mean = train_x.mean(dim=0, keepdim=True)
    scale = train_x.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-5)
    train_x = (train_x - mean) / scale
    eval_x = (eval_x - mean) / scale
    class_to_index = {int(value): index for index, value in enumerate(classes.tolist())}
    train_index = torch.tensor([class_to_index[int(value)] for value in train_y.tolist()], dtype=torch.long)
    device = device or torch.device("cpu")
    torch.manual_seed(int(seed))
    if int(hidden_dim) > 0:
        model = nn.Sequential(
            nn.Linear(train_x.size(1), int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), int(classes.numel())),
        ).to(device)
    else:
        model = nn.Linear(train_x.size(1), int(classes.numel())).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(int(seed) + 17)
    model.train()
    for _ in range(max(1, int(steps))):
        indices = torch.randint(
            train_x.size(0),
            (min(int(batch_size), train_x.size(0)),),
            generator=generator,
        )
        logits = model(train_x[indices].to(device))
        loss = F.cross_entropy(logits, train_index[indices].to(device))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        pred_index = model(eval_x.to(device)).argmax(dim=1).cpu()
    result = _probe_metrics(classes[pred_index], eval_y, classes)
    result.update(
        {
            "status": "COMPLETE",
            "probe": "mlp" if int(hidden_dim) > 0 else "linear_sgd",
            "train_count": int(train_x.size(0)),
            "steps": max(1, int(steps)),
            "hidden_dim": int(hidden_dim),
        }
    )
    if eval_groups is not None and int(bootstrap_resamples) > 0:
        known = torch.isin(original_eval_labels, classes)
        groups = eval_groups.detach().cpu()[known]
        result.update(
            _probe_gain_bootstrap(
                classes[pred_index],
                eval_y,
                groups,
                resamples=int(bootstrap_resamples),
                seed=int(seed) + 911,
            )
        )
    return result


def fit_knn_probe(
    train_features: Tensor,
    train_labels: Tensor,
    eval_features: Tensor,
    eval_labels: Tensor,
    *,
    neighbors: int = 5,
    chunk_size: int = 1024,
    eval_groups: Optional[Tensor] = None,
    bootstrap_resamples: int = 0,
    seed: int = 0,
) -> dict[str, Any]:
    original_eval_labels = eval_labels.detach().view(-1).long().cpu()
    train_x, train_y, eval_x, eval_y, classes = _prepare_probe_data(
        train_features, train_labels, eval_features, eval_labels
    )
    mean = train_x.mean(dim=0, keepdim=True)
    scale = train_x.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-5)
    train_x = F.normalize((train_x - mean) / scale, dim=1, eps=1e-8)
    eval_x = F.normalize((eval_x - mean) / scale, dim=1, eps=1e-8)
    class_to_index = {int(value): index for index, value in enumerate(classes.tolist())}
    train_index = torch.tensor([class_to_index[int(value)] for value in train_y.tolist()], dtype=torch.long)
    k = max(1, min(int(neighbors), int(train_x.size(0))))
    predictions = []
    for start in range(0, eval_x.size(0), max(1, int(chunk_size))):
        similarity = eval_x[start : start + int(chunk_size)] @ train_x.T
        nearest = similarity.topk(k, dim=1).indices
        votes = F.one_hot(train_index[nearest], num_classes=int(classes.numel())).sum(dim=1)
        predictions.append(classes[votes.argmax(dim=1)])
    prediction = torch.cat(predictions)
    result = _probe_metrics(prediction, eval_y, classes)
    result.update(
        {
            "status": "COMPLETE",
            "probe": "cosine_knn",
            "train_count": int(train_x.size(0)),
            "neighbors": k,
        }
    )
    if eval_groups is not None and int(bootstrap_resamples) > 0:
        known = torch.isin(original_eval_labels, classes)
        result.update(
            _probe_gain_bootstrap(
                prediction,
                eval_y,
                eval_groups.detach().cpu()[known],
                resamples=int(bootstrap_resamples),
                seed=int(seed) + 977,
            )
        )
    return result


def _probe_gain_bootstrap(
    prediction: Tensor,
    labels: Tensor,
    groups: Tensor,
    *,
    resamples: int,
    seed: int,
) -> dict[str, float | int]:
    prediction = prediction.detach().view(-1).long().cpu()
    labels = labels.detach().view(-1).long().cpu()
    groups = groups.detach().cpu()
    if groups.ndim == 1:
        groups = groups[:, None]
    if prediction.numel() != labels.numel() or groups.size(0) != labels.numel():
        raise ValueError("probe bootstrap groups must align with evaluation predictions")
    unique_groups, inverse = torch.unique(groups, dim=0, return_inverse=True)
    classes = torch.unique(labels, sorted=True)
    chance = 1.0 / float(classes.numel())
    class_index = torch.searchsorted(classes, labels)
    group_count = int(unique_groups.size(0))
    class_count = int(classes.numel())
    flat_index = inverse * class_count + class_index
    totals = torch.zeros(group_count * class_count, dtype=torch.float32)
    correct = torch.zeros_like(totals)
    totals.scatter_add_(0, flat_index, torch.ones_like(flat_index, dtype=torch.float32))
    correct.scatter_add_(0, flat_index, prediction.eq(labels).float())
    totals = totals.reshape(group_count, class_count)
    correct = correct.reshape(group_count, class_count)
    generator = torch.Generator().manual_seed(int(seed))
    draw_count = max(1, int(resamples))
    weights = torch.zeros((draw_count, group_count), dtype=torch.float32)
    for draw in range(draw_count):
        selected = torch.randint(group_count, (group_count,), generator=generator)
        weights[draw] = torch.bincount(selected, minlength=group_count).float()
    sampled_totals = weights @ totals
    sampled_correct = weights @ correct
    active = sampled_totals > 0
    class_accuracy = sampled_correct / sampled_totals.clamp_min(1.0)
    balanced = (class_accuracy * active).sum(dim=1) / active.sum(dim=1).clamp_min(1)
    distribution = (balanced - chance) / max(1e-12, 1.0 - chance)
    return {
        "bootstrap_group_count": int(unique_groups.size(0)),
        "normalized_gain_ci95_low": float(torch.quantile(distribution, 0.025).item()),
        "normalized_gain_ci95_high": float(torch.quantile(distribution, 0.975).item()),
    }


class _HoldoutHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(input_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), int(output_dim)),
        )

    def forward(self, value: Tensor) -> Tensor:
        return self.net(value)


def _factor_input(features: Mapping[str, Tensor], row: str, indices: Mapping[str, Tensor]) -> tuple[Tensor, Tensor]:
    q = features["q_holdout"].detach().float().reshape(features["q_holdout"].size(0), -1).cpu()
    theta = features["support_theta"].detach().float().reshape(features["support_theta"].size(0), -1).cpu()
    if row == "H0":
        return torch.cat((q, torch.zeros_like(theta)), dim=1), torch.ones(q.size(0), dtype=torch.bool)
    if row == "H1":
        return torch.cat((torch.zeros_like(q), theta), dim=1), torch.ones(q.size(0), dtype=torch.bool)
    selected = indices[row]
    valid = selected >= 0
    safe = selected.clamp_min(0)
    return torch.cat((q, theta[safe]), dim=1), valid


def run_holdout_factorization(
    train_features: Mapping[str, Tensor],
    eval_features: Mapping[str, Tensor],
    *,
    device: torch.device,
    steps: int,
    batch_size: int,
    seed: int,
    bootstrap_resamples: int,
    hidden_dim: int = 64,
) -> dict[str, Any]:
    """Train capacity-matched H0-H6 heads and an explicitly two-stage HR diagnostic."""

    train_target = train_features["heldout_target"].detach().float().reshape(
        train_features["heldout_target"].size(0), -1
    ).cpu()
    eval_target = eval_features["heldout_target"].detach().float().reshape(
        eval_features["heldout_target"].size(0), -1
    ).cpu()
    target_mean = train_target.mean(dim=0, keepdim=True)
    target_scale = train_target.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-5)
    normalized_train_target = (train_target - target_mean) / target_scale
    train_indices = build_factor_indices(
        train_features["tx"],
        train_features["receiver"],
        train_features["day"],
        seed=int(seed) + 101,
        q=train_features["q_holdout"],
    )
    eval_indices = build_factor_indices(
        eval_features["tx"],
        eval_features["receiver"],
        eval_features["day"],
        seed=int(seed) + 202,
        q=eval_features["q_holdout"],
    )
    predictions: dict[str, Tensor] = {}
    valid_masks: dict[str, Tensor] = {}
    heads: dict[str, nn.Module] = {}

    def fit_head(train_x: Tensor, target: Tensor, valid: Tensor, row_seed: int) -> nn.Module:
        torch.manual_seed(int(row_seed))
        head = _HoldoutHead(train_x.size(1), int(hidden_dim), train_target.size(1)).to(device)
        optimizer = torch.optim.AdamW(head.parameters(), lr=3e-3, weight_decay=1e-4)
        valid_train_indices = torch.nonzero(valid, as_tuple=False).flatten()
        if valid_train_indices.numel() < 2:
            raise ValueError("holdout head requires at least two valid training samples")
        generator = torch.Generator().manual_seed(int(row_seed) + 701)
        head.train()
        for _ in range(max(1, int(steps))):
            positions = torch.randint(
                valid_train_indices.numel(),
                (min(int(batch_size), int(valid_train_indices.numel())),),
                generator=generator,
            )
            selected = valid_train_indices[positions]
            output = head(train_x[selected].to(device))
            loss = F.mse_loss(output, target[selected].to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        head.eval()
        return head

    for row_index, row in enumerate(("H0", "H1", "H2", "H3", "H4", "H5", "H6")):
        train_x, train_valid = _factor_input(train_features, row, train_indices)
        eval_x, eval_valid = _factor_input(eval_features, row, eval_indices)
        if int(train_valid.sum().item()) < 2 or int(eval_valid.sum().item()) < 1:
            valid_masks[row] = eval_valid
            continue
        head = fit_head(
            train_x,
            normalized_train_target,
            train_valid,
            row_seed=int(seed) + 1000 + row_index,
        )
        with torch.no_grad():
            normalized_prediction = head(eval_x.to(device)).cpu()
        predictions[row] = normalized_prediction * target_scale + target_mean
        valid_masks[row] = eval_valid
        heads[row] = head

    if "H0" in heads and "H2" in heads:
        train_h0_x, train_h0_valid = _factor_input(train_features, "H0", train_indices)
        train_h2_x, train_h2_valid = _factor_input(train_features, "H2", train_indices)
        eval_h2_x, eval_h2_valid = _factor_input(eval_features, "H2", eval_indices)
        train_groups = torch.stack(
            (
                train_features["tx"].view(-1).long().cpu(),
                train_features["receiver"].view(-1).long().cpu(),
                train_features["day"].view(-1).long().cpu(),
            ),
            dim=1,
        )
        unique_groups, group_inverse = torch.unique(train_groups, dim=0, return_inverse=True)
        if unique_groups.size(0) < 2:
            raise ValueError("cross-fitted common response requires at least two TX-receiver-day groups")
        group_generator = torch.Generator().manual_seed(int(seed) + 2900)
        group_order = torch.randperm(unique_groups.size(0), generator=group_generator)
        group_fold = torch.empty(unique_groups.size(0), dtype=torch.long)
        group_fold[group_order] = torch.arange(unique_groups.size(0)).remainder(2)
        sample_fold = group_fold[group_inverse]
        common_train_normalized = torch.empty_like(normalized_train_target)
        for fold in (0, 1):
            fit_mask = train_h0_valid & sample_fold.ne(fold)
            predict_mask = train_h0_valid & sample_fold.eq(fold)
            common_head = fit_head(
                train_h0_x,
                normalized_train_target,
                fit_mask,
                row_seed=int(seed) + 2910 + fold,
            )
            with torch.no_grad():
                common_train_normalized[predict_mask] = common_head(
                    train_h0_x[predict_mask].to(device)
                ).cpu()
        with torch.no_grad():
            common_train = common_train_normalized * target_scale + target_mean
        residual_target = (train_target - common_train) / target_scale
        residual_head = fit_head(
            train_h2_x,
            residual_target,
            train_h0_valid & train_h2_valid,
            row_seed=int(seed) + 3000,
        )
        with torch.no_grad():
            residual_eval = residual_head(eval_h2_x.to(device)).cpu() * target_scale
        predictions["HR"] = predictions["H0"] + residual_eval
        valid_masks["HR"] = eval_h2_valid

    rows: dict[str, Any] = {}
    squared_errors: dict[str, Tensor] = {}
    target_energy_rows = eval_target.square().sum(dim=1)
    for row, prediction in predictions.items():
        valid = valid_masks[row]
        error = (prediction - eval_target).square().sum(dim=1)
        squared_errors[row] = error
        error_sum = float(error[valid].sum().item())
        energy_sum = float(target_energy_rows[valid].sum().item())
        nmse = error_sum / max(1e-12, energy_sum)
        rows[row] = {
            "sample_count": int(valid.sum().item()),
            "squared_error": error_sum,
            "target_energy": energy_sum,
            "nmse": nmse,
            "normalized_energy_fit_score": 1.0 - nmse,
        }

    groups = torch.stack(
        (
            eval_features["tx"].view(-1).long().cpu(),
            eval_features["receiver"].view(-1).long().cpu(),
            eval_features["day"].view(-1).long().cpu(),
        ),
        dim=1,
    )
    comparisons: dict[str, Any] = {}
    for name, reference_row, candidate_row in (
        ("h2_vs_h0", "H0", "H2"),
        ("h2_vs_shuffle", "H3", "H2"),
        ("h2_vs_other_tx", "H4", "H2"),
        ("h5_vs_other_tx", "H4", "H5"),
        ("h6_vs_other_tx", "H4", "H6"),
    ):
        if reference_row not in squared_errors or candidate_row not in squared_errors:
            comparisons[name] = {"status": "UNAVAILABLE"}
            continue
        valid = valid_masks[reference_row] & valid_masks[candidate_row]
        if int(valid.sum().item()) < 2:
            comparisons[name] = {"status": "UNAVAILABLE", "sample_count": int(valid.sum().item())}
            continue
        comparison = group_paired_bootstrap(
            squared_errors[reference_row][valid],
            squared_errors[candidate_row][valid],
            groups[valid],
            resamples=int(bootstrap_resamples),
            seed=int(seed) + len(comparisons) * 31,
        )
        comparison["status"] = "COMPLETE"
        comparisons[name] = comparison
    return {
        "status": "COMPLETE",
        "fit_role": "L_s",
        "eval_role": "V_select",
        "same_capacity_rows": ["H0", "H1", "H2", "H3", "H4", "H5", "H6"],
        "hr_interpretation": "two_stage_exploratory_common_plus_residual_not_same_capacity",
        "hr_common_cross_fitted": True,
        "hr_common_cross_fit_folds": 2,
        "steps_per_head": max(1, int(steps)),
        "hidden_dim": int(hidden_dim),
        "rows": rows,
        "comparisons": comparisons,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CCOI-PA-V2 frozen causal audit")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--sidecar", required=True)
    parser.add_argument("--wisig_pkl", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--fit_role", default="L_s", choices=("L_s",))
    parser.add_argument("--eval_role", default="V_select", choices=("V_select",))
    parser.add_argument("--target_or_query_access", action="store_true", default=False)
    parser.add_argument("--probe_epochs", type=int, default=40)
    parser.add_argument("--holdout_epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--eval_batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--sat_seed", type=int, default=20260824)
    parser.add_argument("--eval_sat_on", default="main")
    parser.add_argument("--min_match_cosine", type=float, default=0.70)
    parser.add_argument("--pair_thresholds", default="0.50,0.70,0.80,0.90,0.95,0.98,0.99")
    parser.add_argument("--probe_steps", type=int, default=400)
    parser.add_argument("--holdout_steps", type=int, default=800)
    parser.add_argument("--bootstrap_resamples", type=int, default=1000)
    parser.add_argument("--max_eval_batches", type=int, default=0)
    parser.add_argument("--max_probe_tokens", type=int, default=50000)
    parser.add_argument("--smoke_only", action="store_true")
    parser.add_argument("--synthetic_smoke", action="store_true")
    return parser


def validate_audit_output_root(args: argparse.Namespace) -> Path:
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing causal-audit output: {output}")
    return output


def validate_sidecar_payload(payload: Mapping[str, Any]) -> None:
    if str(payload.get("schema", "")) != SIDECAR_SCHEMA:
        raise ValueError(f"sidecar schema must be {SIDECAR_SCHEMA}")
    if str(payload.get("row", "")) != "C4":
        raise ValueError("causal audit requires the frozen C4 sidecar")
    if bool(payload.get("sample_level_source_state_included", True)):
        raise ValueError("causal audit rejects sidecars containing sample-level source state")


def build_sidecar_from_state(
    *,
    pa_channels: int,
    num_classes: int,
    num_domains: int,
    state_dict: Mapping[str, Tensor],
    device: torch.device,
) -> CCOIPASidecar:
    """Reconstruct the exact saved C4 structure before a strict state load."""

    def _shape(key: str) -> tuple[int, ...]:
        value = state_dict.get(key)
        if not isinstance(value, Tensor):
            raise ValueError(f"C4 sidecar state is missing tensor {key}")
        return tuple(int(size) for size in value.shape)

    q_head_shape = _shape("challenge_encoder.q_head.weight")
    code_head_shape = _shape("challenge_encoder.code_head.weight")
    pa_proj_shape = _shape("response_head.pa_proj.weight")
    operator_value_shape = _shape("operator_pool.value.weight")
    if len(q_head_shape) != 2 or len(code_head_shape) != 2:
        raise ValueError("C4 challenge encoder weights have invalid geometry")
    q_dim, hidden_dim = q_head_shape
    codebook_size, code_q_dim = code_head_shape
    if code_q_dim != q_dim:
        raise ValueError("C4 code head and q head dimensions disagree")

    tx_weight = state_dict.get("challenge_encoder.tx_probe.weight")
    rx_weight = state_dict.get("challenge_encoder.rx_probe.weight")
    num_tx = int(tx_weight.shape[0]) if isinstance(tx_weight, Tensor) else 0
    num_rx = int(rx_weight.shape[0]) if isinstance(rx_weight, Tensor) else 0
    if num_tx not in (0, int(num_classes)):
        raise ValueError(f"C4 TX probe classes {num_tx} != expected {num_classes}")
    if num_rx not in (0, int(num_domains)):
        raise ValueError(f"C4 RX probe classes {num_rx} != expected {num_domains}")

    challenge_encoder = PAChallengeEncoder(
        q_dim=q_dim,
        codebook_size=codebook_size,
        hidden_dim=hidden_dim,
        num_tx=num_tx,
        num_rx=num_rx,
    )
    sidecar = CCOIPASidecar(
        pa_channels=int(pa_channels),
        num_classes=int(num_classes),
        challenge_encoder=challenge_encoder,
        q_dim=q_dim,
        response_dim=int(pa_proj_shape[0]),
        operator_dim=int(operator_value_shape[0]),
    ).to(device)
    sidecar.load_state_dict(state_dict, strict=True)
    return sidecar


def evaluate_stop_rules(metrics: Mapping[str, Any]) -> dict[str, Any]:
    required = (
        "q_tx_normalized_gain",
        "q_rx_normalized_gain",
        "negative_anchor_coverage",
        "h2_vs_h0_relative_gain",
        "h2_vs_shuffle_relative_gain",
        "h2_vs_other_tx_relative_gain",
        "h2_vs_h0_ci_low",
        "h2_vs_shuffle_ci_low",
        "h2_vs_other_tx_ci_low",
        "cross_rx_stability_pass",
        "cross_day_stability_pass",
        "source_leo_oracle_gain_pp",
        "rescue_minus_harm",
    )
    missing = [key for key in required if key not in metrics]
    if missing:
        raise ValueError(f"missing stop-rule metrics: {missing}")
    reasons = []
    if float(metrics["q_tx_normalized_gain"]) > 0.10:
        reasons.append("Q_TX_LEAKAGE")
    if float(metrics["q_rx_normalized_gain"]) > 0.10:
        reasons.append("Q_RX_LEAKAGE")
    if float(metrics["negative_anchor_coverage"]) < 0.80:
        reasons.append("NEGATIVE_ANCHOR_COVERAGE_LT_0.80")
    if float(metrics["h2_vs_h0_relative_gain"]) < 0.05:
        reasons.append("H2_NOT_BETTER_THAN_H0_BY_5PCT")
    if float(metrics["h2_vs_shuffle_relative_gain"]) < 0.05:
        reasons.append("H2_NOT_BETTER_THAN_SHUFFLE_BY_5PCT")
    if float(metrics["h2_vs_other_tx_relative_gain"]) < 0.05:
        reasons.append("H2_NOT_BETTER_THAN_OTHER_TX_BY_5PCT")
    if float(metrics["h2_vs_h0_ci_low"]) <= 0.0:
        reasons.append("H2_VS_H0_CI_CROSSES_ZERO")
    if float(metrics["h2_vs_shuffle_ci_low"]) <= 0.0:
        reasons.append("H2_VS_SHUFFLE_CI_CROSSES_ZERO")
    if float(metrics["h2_vs_other_tx_ci_low"]) <= 0.0:
        reasons.append("H2_VS_OTHER_TX_CI_CROSSES_ZERO")
    if not bool(metrics["cross_rx_stability_pass"]):
        reasons.append("CROSS_RX_STABILITY_FAILED")
    if not bool(metrics["cross_day_stability_pass"]):
        reasons.append("CROSS_DAY_STABILITY_FAILED")
    if float(metrics["source_leo_oracle_gain_pp"]) < 0.30:
        reasons.append("SOURCE_LEO_ORACLE_GAIN_LT_0.30PP")
    if int(metrics["rescue_minus_harm"]) <= 0:
        reasons.append("RESCUE_NOT_GREATER_THAN_HARM")
    return {
        "promotable": not reasons,
        "stop_reasons": reasons,
        "thresholds": {
            "q_normalized_gain_max": 0.10,
            "negative_anchor_coverage_min": 0.80,
            "holdout_relative_gain_min": 0.05,
            "source_leo_oracle_gain_pp_min": 0.30,
            "rescue_minus_harm_min_exclusive": 0,
        },
    }


def _json_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _seed_all(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _metadata_tensor(extra: Any, key: str, count: int) -> Tensor:
    values = []
    for index in range(int(count)):
        value = _meta_value(extra, key, index, None)
        try:
            values.append(int(value))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"missing source metadata {key} at batch position {index}") from exc
    return torch.tensor(values, dtype=torch.long)


def _collect_frozen_features(model, loader, ssdg, data_ctx, device, max_batches: int) -> dict[str, Tensor]:
    model.eval()
    rows: dict[str, list[Tensor]] = {
        key: []
        for key in (
            "q",
            "code_prob",
            "theta",
            "support_theta",
            "q_holdout",
            "heldout_target",
            "base_logits",
            "operator_logits",
            "fused_logits",
            "tx",
            "receiver",
            "day",
            "base_index",
        )
    }
    with torch.no_grad():
        for _, batch in _limited(loader, int(max_batches)):
            x, y, domain, extra = _move_batch(ssdg, batch, device, data_ctx["domain_label_map"])
            out = model(x, return_aux=True, domain_labels=domain)
            ccoi = out["ccoi"]
            count = int(x.size(0))
            tensor_values = {
                "q": ccoi["q"],
                "code_prob": ccoi["code_prob"],
                "theta": ccoi["theta"],
                "support_theta": ccoi["support_theta"],
                "q_holdout": ccoi["q_holdout"],
                "heldout_target": ccoi["heldout_target"],
                "base_logits": out["base_tx_logits"],
                "operator_logits": ccoi["logit_correction"],
                "fused_logits": out["tx_logits"],
                "tx": y,
                "receiver": _metadata_tensor(extra, "rx_i", count),
                "day": _metadata_tensor(extra, "day_i", count),
                "base_index": _metadata_tensor(extra, "base_index", count),
            }
            for key, value in tensor_values.items():
                rows[key].append(value.detach().float().cpu() if value.is_floating_point() else value.detach().long().cpu())
    if not rows["q"]:
        raise RuntimeError("source feature audit produced zero batches")
    return {key: torch.cat(values, dim=0) for key, values in rows.items()}


def _collect_scenario_features(
    model,
    loader,
    scenario: str,
    ssdg,
    data_ctx,
    data_args,
    args,
    device,
) -> dict[str, Tensor]:
    model.eval()
    rows = {key: [] for key in ("code_prob", "q", "base_prediction", "operator_prediction", "fused_prediction", "truth")}
    generator = ssdg.make_torch_generator(device, int(args.sat_seed) + 7001 + SCENARIOS.index(scenario) * 101)
    with torch.no_grad():
        for _, batch in _limited(loader, int(args.max_eval_batches)):
            x, y, domain, _extra = _move_batch(ssdg, batch, device, data_ctx["domain_label_map"])
            view = x if scenario == "clean" else _satellite_view(ssdg, x, scenario, data_args, generator)
            base_out = model.base(view, return_aux=True, domain_labels=domain)
            pa_map = (base_out.get("aux_id", {}) or {}).get("pa_token_map")
            if not torch.is_tensor(pa_map):
                raise KeyError("Core90 scenario audit is missing pa_token_map")
            side = model.sidecar(view, pa_map.detach(), conditioned=True)
            base_logits = base_out["tx_logits"]
            operator_logits = side["logit_correction"]
            fused_logits = (1.0 - model.fusion_alpha) * base_logits + (
                model.fusion_alpha * model.fusion_scale * operator_logits
            )
            values = {
                "code_prob": side["code_prob"],
                "q": side["q"],
                "base_prediction": base_logits.argmax(dim=1),
                "operator_prediction": operator_logits.argmax(dim=1),
                "fused_prediction": fused_logits.argmax(dim=1),
                "truth": y,
            }
            for key, value in values.items():
                rows[key].append(value.detach().float().cpu() if value.is_floating_point() else value.detach().long().cpu())
    if not rows["truth"]:
        raise RuntimeError(f"scenario audit produced zero samples for {scenario}")
    return {key: torch.cat(value, dim=0) for key, value in rows.items()}


def _subsample_tokens(q: Tensor, labels: Tensor, limit: int, seed: int) -> tuple[Tensor, Tensor]:
    features = q.detach().float().reshape(-1, q.size(-1)).cpu()
    labels = labels.detach().view(-1).long().cpu()
    if features.size(0) != labels.numel():
        raise ValueError("token features and labels must align")
    if int(limit) > 0 and features.size(0) > int(limit):
        generator = torch.Generator().manual_seed(int(seed))
        indices = torch.randperm(features.size(0), generator=generator)[: int(limit)]
        features, labels = features[indices], labels[indices]
    return features, labels


def _run_probe_audit(train: Mapping[str, Tensor], evaluate: Mapping[str, Tensor], args, device) -> dict[str, Any]:
    train_mean = train["q"].mean(dim=1)
    eval_mean = evaluate["q"].mean(dim=1)
    train_sequence = train["q"].reshape(train["q"].size(0), -1)
    eval_sequence = evaluate["q"].reshape(evaluate["q"].size(0), -1)
    result: dict[str, Any] = {
        "status": "COMPLETE",
        "fit_role": "L_s",
        "eval_role": "V_select",
        "target_or_query_access": False,
        "representations": {},
    }
    eval_groups = torch.stack((evaluate["tx"], evaluate["receiver"], evaluate["day"]), dim=1)
    probe_steps = max(1, int(args.probe_steps))
    for offset, label_name in enumerate(("tx", "receiver", "day")):
        train_labels = train[label_name]
        eval_labels = evaluate[label_name]
        linear = frozen_ridge_linear_probe(train_mean, train_labels, eval_mean, eval_labels, ridge=0.01)
        linear["normalized_gain"] = (
            float(linear["balanced_accuracy"]) - float(linear["balanced_chance_accuracy"])
        ) / max(1e-12, 1.0 - float(linear["balanced_chance_accuracy"]))
        result["representations"][label_name] = {
            "packet_mean_linear": linear,
            "packet_mean_mlp": fit_torch_probe(
                train_mean,
                train_labels,
                eval_mean,
                eval_labels,
                steps=probe_steps,
                seed=int(args.seed) + 100 + offset,
                hidden_dim=64,
                device=device,
                eval_groups=eval_groups,
                bootstrap_resamples=int(args.bootstrap_resamples),
            ),
            "packet_mean_knn": fit_knn_probe(
                train_mean,
                train_labels,
                eval_mean,
                eval_labels,
                neighbors=5,
                eval_groups=eval_groups,
                bootstrap_resamples=int(args.bootstrap_resamples),
                seed=int(args.seed) + 150 + offset,
            ),
            "token_sequence_mlp": fit_torch_probe(
                train_sequence,
                train_labels,
                eval_sequence,
                eval_labels,
                steps=probe_steps,
                seed=int(args.seed) + 200 + offset,
                hidden_dim=64,
                device=device,
                eval_groups=eval_groups,
                bootstrap_resamples=int(args.bootstrap_resamples),
            ),
        }
        train_token_labels = train_labels.repeat_interleave(train["q"].size(1))
        eval_token_labels = eval_labels.repeat_interleave(evaluate["q"].size(1))
        train_token_x, train_token_y = _subsample_tokens(
            train["q"], train_token_labels, int(args.max_probe_tokens), int(args.seed) + 301 + offset
        )
        eval_token_x, eval_token_y = _subsample_tokens(
            evaluate["q"], eval_token_labels, int(args.max_probe_tokens), int(args.seed) + 401 + offset
        )
        token_linear = frozen_ridge_linear_probe(
            train_token_x, train_token_y, eval_token_x, eval_token_y, ridge=0.01
        )
        token_linear["normalized_gain"] = (
            float(token_linear["balanced_accuracy"]) - float(token_linear["balanced_chance_accuracy"])
        ) / max(1e-12, 1.0 - float(token_linear["balanced_chance_accuracy"]))
        result["representations"][label_name]["token_linear"] = token_linear

    train_position = torch.arange(train["q"].size(1)).repeat(train["q"].size(0))
    eval_position = torch.arange(evaluate["q"].size(1)).repeat(evaluate["q"].size(0))
    train_token_x, train_position = _subsample_tokens(
        train["q"], train_position, int(args.max_probe_tokens), int(args.seed) + 501
    )
    eval_token_x, eval_position = _subsample_tokens(
        evaluate["q"], eval_position, int(args.max_probe_tokens), int(args.seed) + 502
    )
    result["representations"]["position"] = {
        "token_mlp": fit_torch_probe(
            train_token_x,
            train_position,
            eval_token_x,
            eval_position,
            steps=probe_steps,
            seed=int(args.seed) + 503,
            hidden_dim=64,
            device=device,
        )
    }
    for label_name in ("tx", "receiver"):
        complete = [
            probe
            for probe in result["representations"][label_name].values()
            if str(probe.get("status", "")) == "COMPLETE"
        ]
        result[f"q_{label_name}_normalized_gain_raw_max"] = max(
            float(probe.get("normalized_gain", float("-inf")))
            for probe in complete
        )
        confirmed = [
            float(probe["normalized_gain"])
            for probe in complete
            if float(probe.get("normalized_gain_ci95_low", float("-inf"))) > 0.0
        ]
        result[f"q_{label_name}_normalized_gain"] = max(confirmed) if confirmed else 0.0
    return result


def _feature_and_complementarity_audits(
    model,
    clean_features,
    data_ctx,
    ssdg,
    data_args,
    args,
    device,
) -> tuple[dict[str, Any], dict[str, Any]]:
    scenario_data = {}
    feature = {
        "status": "COMPLETE",
        "truth_scope": "source_V_select_only",
        "target_or_query_access": False,
        "clean_code": token_code_audit(clean_features["code_prob"]),
        "scenario_code": {},
        "clean_satellite_token_assignment_consistency": {},
        "clean_satellite_q_cosine": {},
    }
    clean_code_hard = clean_features["code_prob"].argmax(dim=-1)
    clean_q = F.normalize(clean_features["q"].float(), dim=-1, eps=1e-8)
    for scenario in SCENARIOS:
        scenario_data[scenario] = _collect_scenario_features(
            model,
            data_ctx["val_loader"],
            scenario,
            ssdg,
            data_ctx,
            data_args,
            args,
            device,
        )
        feature["scenario_code"][scenario] = token_code_audit(scenario_data[scenario]["code_prob"])
        if scenario != "clean":
            feature["clean_satellite_token_assignment_consistency"][scenario] = float(
                clean_code_hard.eq(scenario_data[scenario]["code_prob"].argmax(dim=-1)).float().mean().item()
            )
            satellite_q = F.normalize(scenario_data[scenario]["q"].float(), dim=-1, eps=1e-8)
            feature["clean_satellite_q_cosine"][scenario] = float((clean_q * satellite_q).sum(dim=-1).mean().item())

    complementarity = {
        "status": "COMPLETE",
        "truth_scope": "source_V_select_only",
        "target_or_query_access": False,
        "scenarios": {},
    }
    leo_oracle_gains = []
    leo_rescue_minus_harm = 0
    for scenario, values in scenario_data.items():
        base_operator = complementarity_table(
            values["base_prediction"], values["operator_prediction"], values["truth"]
        )
        base_fused = complementarity_table(
            values["base_prediction"], values["fused_prediction"], values["truth"]
        )
        oracle_gain_pp = 100.0 * (base_operator["oracle_accuracy"] - base_operator["base_accuracy"])
        complementarity["scenarios"][scenario] = {
            "base_vs_operator": base_operator,
            "base_vs_fused": base_fused,
            "operator_oracle_gain_pp": oracle_gain_pp,
        }
        if scenario != "clean":
            leo_oracle_gains.append(oracle_gain_pp)
            leo_rescue_minus_harm += int(base_operator["rescue_minus_harm"])
    complementarity["source_leo_oracle_gain_pp"] = float(sum(leo_oracle_gains) / max(1, len(leo_oracle_gains)))
    complementarity["source_leo_rescue_minus_harm"] = int(leo_rescue_minus_harm)
    return feature, complementarity


def _parse_thresholds(spec: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in str(spec).split(",") if item.strip())
    if not values:
        raise ValueError("pair_thresholds must not be empty")
    return values


def _real_run(args: argparse.Namespace, output: Path) -> int:
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    sidecar_path = Path(args.sidecar).expanduser().resolve()
    wisig_path = Path(args.wisig_pkl).expanduser().resolve()
    for path in (checkpoint_path, sidecar_path, wisig_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    output.mkdir(parents=True, exist_ok=False)
    _seed_all(args.seed)
    device = torch.device(args.device)
    checkpoint = _torch_load(checkpoint_path, device)
    sidecar_payload = _torch_load(sidecar_path, device)
    validate_sidecar_payload(sidecar_payload)
    ssdg, data_args = _prepare_ssdg_args(args, checkpoint)
    data_ctx = ssdg._build_ssdg_wisig_data(data_args, device)
    validate_source_roles(data_args, data_ctx["split_info"])
    base, checkpoint_audit = build_exact_ssdg_model_from_checkpoint(
        checkpoint,
        input_len=int(data_ctx["input_len"]),
        device=device,
        ssdg_module=ssdg,
    )
    base = freeze_base_model(base)
    pa_channels, num_classes, base_smoke = _infer_base_dimensions(base, data_ctx, ssdg, device)
    sidecar = build_sidecar_from_state(
        pa_channels=pa_channels,
        num_classes=num_classes,
        num_domains=int(data_ctx["num_domains"]),
        state_dict=sidecar_payload["state_dict"],
        device=device,
    )
    sidecar.freeze_challenge_encoder()
    for parameter in sidecar.parameters():
        parameter.requires_grad = False
    model = FrozenCore90CCOI(
        base,
        sidecar,
        row="C4",
        fusion_alpha=float(sidecar_payload["fusion_alpha"]),
        fusion_scale=float(sidecar_payload["fusion_scale"]),
    ).to(device)
    model.eval()
    _, smoke_batch = next(iter(_limited(data_ctx["val_loader"], 1)))
    smoke_x, _smoke_y, smoke_domain, _smoke_extra = _move_batch(
        ssdg, smoke_batch, device, data_ctx["domain_label_map"]
    )
    with torch.no_grad():
        smoke_out = model(smoke_x, return_aux=True, domain_labels=smoke_domain)
    smoke = {
        "base": base_smoke,
        "c4_finite_logits": bool(torch.isfinite(smoke_out["tx_logits"]).all().item()),
        "q_shape": list(smoke_out["ccoi"]["q"].shape),
        "holdout_target_shape": list(smoke_out["ccoi"]["heldout_target"].shape),
    }
    if not smoke["c4_finite_logits"]:
        raise FloatingPointError("real C4 no-query smoke produced non-finite logits")
    protocol = {
        "status": "PASS",
        "protocol": "Phase1_source_only_causal_audit",
        "source_roles": data_ctx["split_info"],
        "fit_role": "L_s",
        "eval_role": "V_select",
        "target_or_query_access": False,
        "checkpoint_audit": checkpoint_audit,
        "sidecar_schema": sidecar_payload["schema"],
        "sidecar_row": sidecar_payload["row"],
        "sample_level_source_state_included": bool(sidecar_payload.get("sample_level_source_state_included", True)),
        "real_checkpoint_no_query_smoke": smoke,
    }
    _json_write(output / "protocol_and_smoke.json", protocol)
    if args.smoke_only:
        print(f"[CCOI-CAUSAL-SMOKE] PASS output={output}", flush=True)
        return 0

    train_features = _collect_frozen_features(
        model, data_ctx["probe_train_loader"], ssdg, data_ctx, device, int(args.max_eval_batches)
    )
    eval_features = _collect_frozen_features(
        model, data_ctx["val_loader"], ssdg, data_ctx, device, int(args.max_eval_batches)
    )
    feature_audit, complementarity = _feature_and_complementarity_audits(
        model, eval_features, data_ctx, ssdg, data_args, args, device
    )
    _json_write(output / "feature_audit.json", feature_audit)
    probe_audit = _run_probe_audit(train_features, eval_features, args, device)
    _json_write(output / "probe_audit.json", probe_audit)
    thresholds = _parse_thresholds(args.pair_thresholds)
    pair_geometry = {
        "status": "COMPLETE",
        "truth_scope": "source_V_select_only",
        "target_or_query_access": False,
        "batch_local": False,
        "threshold_sweep": pair_relation_sweep(
            eval_features["q"],
            eval_features["tx"],
            eval_features["receiver"],
            thresholds=thresholds,
        ),
    }
    _json_write(output / "pair_geometry.json", pair_geometry)
    holdout = run_holdout_factorization(
        train_features,
        eval_features,
        device=device,
        steps=int(args.holdout_steps),
        batch_size=int(args.batch_size),
        seed=int(args.seed),
        bootstrap_resamples=int(args.bootstrap_resamples),
    )
    _json_write(output / "holdout_factorization.json", holdout)
    _json_write(output / "complementarity.json", complementarity)

    threshold_key = f"{float(args.min_match_cosine):.3f}"
    if threshold_key not in pair_geometry["threshold_sweep"]:
        raise ValueError("min_match_cosine must be included in pair_thresholds")
    comparisons = holdout["comparisons"]
    required_comparisons = ("h2_vs_h0", "h2_vs_shuffle", "h2_vs_other_tx")
    if any(comparisons.get(name, {}).get("status") != "COMPLETE" for name in required_comparisons):
        raise RuntimeError("required holdout comparison is unavailable")
    cross_rx = comparisons.get("h5_vs_other_tx", {})
    cross_day = comparisons.get("h6_vs_other_tx", {})
    stop_inputs = {
        "q_tx_normalized_gain": probe_audit["q_tx_normalized_gain"],
        "q_rx_normalized_gain": probe_audit["q_receiver_normalized_gain"],
        "negative_anchor_coverage": pair_geometry["threshold_sweep"][threshold_key]["negative_anchor_coverage"],
        "h2_vs_h0_relative_gain": comparisons["h2_vs_h0"]["relative_gain"],
        "h2_vs_shuffle_relative_gain": comparisons["h2_vs_shuffle"]["relative_gain"],
        "h2_vs_other_tx_relative_gain": comparisons["h2_vs_other_tx"]["relative_gain"],
        "h2_vs_h0_ci_low": comparisons["h2_vs_h0"]["ci95_low"],
        "h2_vs_shuffle_ci_low": comparisons["h2_vs_shuffle"]["ci95_low"],
        "h2_vs_other_tx_ci_low": comparisons["h2_vs_other_tx"]["ci95_low"],
        "cross_rx_stability_pass": cross_rx.get("status") == "COMPLETE"
        and float(cross_rx.get("relative_gain", float("-inf"))) >= 0.05
        and float(cross_rx.get("ci95_low", float("-inf"))) > 0.0,
        "cross_day_stability_pass": cross_day.get("status") == "COMPLETE"
        and float(cross_day.get("relative_gain", float("-inf"))) >= 0.05
        and float(cross_day.get("ci95_low", float("-inf"))) > 0.0,
        "source_leo_oracle_gain_pp": complementarity["source_leo_oracle_gain_pp"],
        "rescue_minus_harm": complementarity["source_leo_rescue_minus_harm"],
    }
    verdict = evaluate_stop_rules(stop_inputs)
    manifest = {
        "status": "ANALYZED",
        "schema": "cvs.phase1.ccoi_pa_v2_causal_audit.v1",
        "target_or_query_access": False,
        "fit_role": "L_s",
        "eval_role": "V_select",
        "train_sample_count": int(train_features["tx"].numel()),
        "eval_sample_count": int(eval_features["tx"].numel()),
        "stop_rule_inputs": stop_inputs,
        "verdict": verdict,
        "next_route": "STOP_PA_M2" if not verdict["promotable"] else "DESIGN_RESIDUAL_V3",
        "artifacts": [
            "protocol_and_smoke.json",
            "feature_audit.json",
            "probe_audit.json",
            "pair_geometry.json",
            "holdout_factorization.json",
            "complementarity.json",
            "audit_manifest.json",
        ],
        "sample_level_source_features_persisted": False,
    }
    _json_write(output / "audit_manifest.json", manifest)
    print(f"[CCOI-CAUSAL-AUDIT] ANALYZED output={output} next={manifest['next_route']}", flush=True)
    return 0


def run(args: argparse.Namespace) -> int:
    if bool(args.target_or_query_access):
        raise ValueError("target/query access is forbidden for this source-only causal audit")
    output = validate_audit_output_root(args)
    if args.synthetic_smoke:
        output.mkdir(parents=True, exist_ok=False)
        names = (
            "protocol_and_smoke.json",
            "feature_audit.json",
            "probe_audit.json",
            "pair_geometry.json",
            "holdout_factorization.json",
            "complementarity.json",
            "audit_manifest.json",
        )
        for name in names:
            payload = {
                "status": "SYNTHETIC_SMOKE",
                "schema": "cvs.phase1.ccoi_pa_v2_causal_audit.v1",
                "target_or_query_access": False,
            }
            (output / name).write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
                newline="\n",
            )
        return 0
    return _real_run(args, output)


def main(argv: Optional[Sequence[str]] = None) -> int:
    return run(build_arg_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
