#!/usr/bin/env python
"""Validate a source-trained ADV3B02 LoRA before any target-query access.

The validator holds out one source receiver and uses only the three formal
``leo_weak`` scenarios.  It source-locks one symmetric head rule, simulates
nested K=1/5/10/20 enrollment on source-only physical samples, and calibrates
the deployed 1->3->5 gate from that exact head.  Calibration and evaluation use
disjoint physical samples.  A promotion manifest is written only when every
resource, provenance, fixed-view, class-floor, and adaptive-view gate passes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = REPO_ROOT / "code"
CODE_SCRIPTS_ROOT = CODE_ROOT / "scripts"
for candidate in (str(REPO_ROOT), str(CODE_ROOT), str(CODE_SCRIPTS_ROOT)):
    while candidate in sys.path:
        sys.path.remove(candidate)
for candidate in (str(REPO_ROOT), str(CODE_ROOT), str(CODE_SCRIPTS_ROOT)):
    sys.path.insert(0, candidate)

from cvsrffi.eval import apply_sat_channel_for_scenario  # noqa: E402
from cvsrffi.tensors import make_torch_generator  # noqa: E402
from export_spaceborne_features import (  # noqa: E402
    _build_wisig_dataset,
    _satellite_tta_views,
    _spectral_logmag_sketch_batch,
)
from paper_reproduction.cvs_aligned.adaptive_rxlight_tta import (  # noqa: E402
    AdaptiveTTAThresholds,
    apply_adaptive_rxlight_tta,
    calibrate_adaptive_rxlight_tta,
)
from paper_reproduction.cvs_aligned.extreme_light_adapter import (  # noqa: E402
    concatenate_registered_features,
)
from paper_reproduction.cvs_aligned.k1_symmetric_head import (  # noqa: E402
    calibrate_symmetric_k1_head,
    fit_locked_symmetric_support_head,
    score_symmetric_head,
)
from paper_reproduction.scripts.benchmark_cvs_adaptive_rxlight_tta import (  # noqa: E402
    BASE_MARGIN_GRID,
    DISAGREEMENT_GRID,
    SHIFT3_MARGIN_GRID,
    apply_fp16_lora_state,
)
from train_apply_phase1_iq_preadapter_20260703 import (  # noqa: E402
    _build_model,
    _feature_forward,
)
from training_controls import parse_sat_scenarios  # noqa: E402


FORMAL_LEO_WEAK_SCENARIOS = (
    "leo_clear_weak",
    "leo_low_elev_weak",
    "leo_rain_weak",
)
FORMAL_TARGET_MANYSIG_RX_INDICES = {"7", "8", "9", "10", "11"}
FORMAL_K = (1, 5, 10, 20)


def validate_formal_scenarios(scenarios: Sequence[str]) -> tuple[str, ...]:
    values = tuple(str(value) for value in scenarios)
    if values != FORMAL_LEO_WEAK_SCENARIOS:
        raise ValueError("source validation must use the exact formal leo_weak scenarios")
    return values


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _serializable(value: Any) -> Any:
    if is_dataclass(value):
        return _serializable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serializable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _receiver_tokens(raw: str) -> set[str]:
    return {item.strip() for item in str(raw).split(",") if item.strip()}


def validate_receiver_holdout(train_receivers: str, val_receivers: str) -> dict[str, Any]:
    train = _receiver_tokens(train_receivers)
    val = _receiver_tokens(val_receivers)
    if not train or not val:
        raise ValueError("source train/validation receiver sets must be non-empty")
    overlap = sorted(train & val)
    if overlap:
        raise ValueError(f"source train/validation receiver overlap: {overlap}")
    target_overlap = sorted((train | val) & FORMAL_TARGET_MANYSIG_RX_INDICES)
    if target_overlap:
        raise ValueError(
            f"source train/validation receivers overlap formal target domain: {target_overlap}"
        )
    return {
        "source_train_receivers": sorted(train),
        "source_validation_receivers": sorted(val),
        "overlap": overlap,
        "formal_target_manysig_indices": sorted(FORMAL_TARGET_MANYSIG_RX_INDICES),
        "formal_target_overlap": target_overlap,
        "disjoint": True,
    }


def stratified_physical_split(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    truth = np.asarray(labels, dtype=np.int64).reshape(-1)
    calibration: list[int] = []
    evaluation: list[int] = []
    for class_index in sorted(np.unique(truth).tolist()):
        indices = np.flatnonzero(truth == int(class_index)).astype(np.int64)
        if len(indices) < 4:
            raise ValueError(
                f"source validation class {class_index} needs at least four samples"
            )
        calibration.extend(indices[::2].tolist())
        evaluation.extend(indices[1::2].tolist())
    return (
        np.asarray(sorted(calibration), dtype=np.int64),
        np.asarray(sorted(evaluation), dtype=np.int64),
    )


def _metric_row(predictions: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    pred = np.asarray(predictions, dtype=np.int64)
    truth = np.asarray(labels, dtype=np.int64)
    per_class = {
        str(class_index): float(np.mean(pred[truth == class_index] == class_index))
        for class_index in sorted(np.unique(truth).tolist())
    }
    return {
        "accuracy": float(np.mean(pred == truth)),
        "min_class_accuracy": float(min(per_class.values())),
        "per_class_accuracy": per_class,
        "sample_count": int(len(truth)),
    }


def _fixed_metrics(view_scores: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    scores = np.asarray(view_scores, dtype=np.float32)
    return {
        "fixed1": _metric_row(np.argmax(scores[:, 0], axis=1), labels),
        "fixed3": _metric_row(np.argmax(scores[:, :3].mean(axis=1), axis=1), labels),
        "fixed5": _metric_row(np.argmax(scores.mean(axis=1), axis=1), labels),
    }


def _dataset(
    args: argparse.Namespace, *, receivers: str, role: str, max_samples_per_tx: int, seed: int
):
    return _build_wisig_dataset(
        pkl_path=str(args.wisig_pkl),
        tx_spec=str(args.source_tx_ids),
        role=role,
        equalized=str(args.wisig_equalized),
        out_len=int(args.wisig_out_len),
        domain=str(args.wisig_domain),
        days=None,
        rxs=str(receivers),
        max_samples_per_combo=0,
        max_samples_per_tx=int(max_samples_per_tx),
        seed=int(seed),
    )


def _joint_feature_tensor(
    primary: torch.Tensor,
    raw_iq: torch.Tensor,
    *,
    fft_dim: int = 96,
    fft_weight: float = 2.0,
) -> torch.Tensor:
    primary_np = primary.detach().cpu().numpy().astype(np.float32)
    raw_np = raw_iq.detach().cpu().numpy().astype(np.float32)
    fft = _spectral_logmag_sketch_batch(raw_np, dim=int(fft_dim))
    joint = concatenate_registered_features(
        primary_np, fft, auxiliary_weight=float(fft_weight)
    )
    return torch.from_numpy(joint).to(primary.device, dtype=torch.float32)


@torch.no_grad()
def _leo_source_prototypes(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    class_count: int,
    scenarios: Sequence[str],
    args: argparse.Namespace,
    device: torch.device,
) -> torch.Tensor:
    """Build source prototypes without exposing any clean waveform to scoring."""

    features: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    generator = make_torch_generator(device, int(args.seed) + 40_000)
    for batch_index, (x, y, _domain, _meta) in enumerate(loader):
        x_device = x.to(device)
        scenario = str(scenarios[int(batch_index) % len(scenarios)])
        x_sat, _ = apply_sat_channel_for_scenario(
            x_device, scenario, args, gen=generator, return_meta=False
        )
        z, _ = _feature_forward(model, x_sat, "z_id")
        joint = _joint_feature_tensor(z, x_sat)
        features.append(F.normalize(joint.float(), dim=1))
        labels.append(y.to(device).long())
    all_features = torch.cat(features, dim=0)
    all_labels = torch.cat(labels, dim=0)
    prototypes = []
    for class_index in range(int(class_count)):
        mask = all_labels == int(class_index)
        if not bool(mask.any()):
            raise ValueError(f"missing source prototype class {class_index}")
        prototypes.append(F.normalize(all_features[mask].mean(dim=0), dim=0))
    return torch.stack(prototypes, dim=0)


@torch.no_grad()
def _heldout_scores(
    base_model: torch.nn.Module,
    adapted_model: torch.nn.Module,
    loader: DataLoader,
    prototypes: torch.Tensor,
    *,
    scenarios: Sequence[str],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, Any], np.ndarray]:
    by_model: dict[str, list[np.ndarray]] = {"base": [], "ground_lora": []}
    label_parts: list[np.ndarray] = []
    adapted_feature_blocks: list[np.ndarray] = []
    view_names: tuple[str, ...] | None = None
    for scenario_index, scenario in enumerate(scenarios):
        generator = make_torch_generator(
            device, int(args.seed) + 50_000 + int(scenario_index)
        )
        scenario_labels: list[np.ndarray] = []
        scenario_scores: dict[str, list[np.ndarray]] = {
            "base": [],
            "ground_lora": [],
        }
        scenario_adapted_features: list[np.ndarray] = []
        for x, y, _domain, _meta in loader:
            x = x.to(device)
            x_sat, _ = apply_sat_channel_for_scenario(
                x, str(scenario), args, gen=generator, return_meta=False
            )
            views = _satellite_tta_views(x_sat, "rx_light5")
            names = tuple(name for name, _value in views)
            if view_names is None:
                view_names = names
            elif names != view_names:
                raise RuntimeError("rx_light5 view ordering drift")
            for model_name, model in (
                ("base", base_model),
                ("ground_lora", adapted_model),
            ):
                view_scores = []
                view_features = []
                for _view_name, x_view in views:
                    z, _ = _feature_forward(model, x_view, "z_id")
                    joint = _joint_feature_tensor(z, x_view)
                    normalized_joint = F.normalize(joint.float(), dim=1)
                    score = normalized_joint @ prototypes.t()
                    view_scores.append(score.detach().cpu().numpy())
                    view_features.append(
                        normalized_joint.detach().cpu().numpy().astype(np.float32)
                    )
                scenario_scores[model_name].append(
                    np.stack(view_scores, axis=1).astype(np.float32)
                )
                if model_name == "ground_lora":
                    scenario_adapted_features.append(
                        np.stack(view_features, axis=1).astype(np.float32)
                    )
            scenario_labels.append(y.detach().cpu().numpy().astype(np.int64))
        labels = np.concatenate(scenario_labels)
        if scenario_index == 0:
            label_parts.append(labels)
        elif not np.array_equal(label_parts[0], labels):
            raise RuntimeError("held-out source label ordering drift across scenarios")
        for model_name in by_model:
            by_model[model_name].append(np.concatenate(scenario_scores[model_name]))
        adapted_feature_blocks.append(
            np.concatenate(scenario_adapted_features).astype(np.float32)
        )
    physical_labels = label_parts[0]
    flattened = {
        name: np.concatenate(blocks, axis=0).astype(np.float32)
        for name, blocks in by_model.items()
    }
    repeated_labels = np.tile(physical_labels, len(scenarios)).astype(np.int64)
    return flattened, repeated_labels, {
        "view_names": list(view_names or ()),
        "physical_validation_samples": int(len(physical_labels)),
        "scenario_count": int(len(scenarios)),
        "scored_rows": int(len(repeated_labels)),
    }, np.concatenate(adapted_feature_blocks).astype(np.float32)


def _expanded_indices(
    physical_indices: np.ndarray, *, physical_count: int, scenario_count: int
) -> np.ndarray:
    return np.concatenate(
        [physical_indices + scenario_index * int(physical_count) for scenario_index in range(int(scenario_count))]
    ).astype(np.int64)


def build_source_symmetric_head_lock(
    adapted_features: np.ndarray,
    physical_labels: np.ndarray,
    calibration_physical: np.ndarray,
    *,
    physical_count: int,
    scenario_count: int,
    source_mean: np.ndarray,
    source_std: np.ndarray,
    max_episodes: int = 20,
) -> dict[str, Any]:
    """Select one role-free symmetric head on source receiver holdout only."""

    features = np.asarray(adapted_features, dtype=np.float32)
    labels = np.asarray(physical_labels, dtype=np.int64).reshape(-1)
    calibration = np.asarray(calibration_physical, dtype=np.int64).reshape(-1)
    if features.ndim != 3 or features.shape[0] != int(physical_count) * int(
        scenario_count
    ):
        raise ValueError("adapted feature layout must be [scenario*N,5,D]")
    if features.shape[1] != 5 or len(labels) != int(physical_count):
        raise ValueError("source head lock requires rx_light5 features and labels")
    if int(scenario_count) != 3:
        raise ValueError("source head lock requires three formal leo_weak scenarios")
    class_ids = sorted(np.unique(labels).tolist())
    by_class = {
        int(class_id): calibration[labels[calibration] == int(class_id)]
        for class_id in class_ids
    }
    if any(len(indices) == 0 for indices in by_class.values()):
        raise ValueError("source head lock is missing a calibration class")
    episode_count = min(
        int(max_episodes), min(len(indices) for indices in by_class.values())
    )
    if episode_count < 1:
        raise ValueError("source head lock has no valid K1 episodes")

    aggregate: dict[tuple[bool, str, float | None], list[dict[str, Any]]] = {}
    episode_support_hashes: list[str] = []
    for episode_index in range(episode_count):
        selected_physical = np.asarray(
            [by_class[int(class_id)][episode_index] for class_id in class_ids],
            dtype=np.int64,
        )
        support = np.stack(
            [
                features[
                    int(scenario_index) * int(physical_count) + selected_physical,
                    0,
                    :,
                ]
                for scenario_index in range(int(scenario_count))
            ],
            axis=0,
        ).astype(np.float32)
        episode_support_hashes.append(
            hashlib.sha256(selected_physical.tobytes()).hexdigest()
        )
        episode = calibrate_symmetric_k1_head(
            support,
            source_mean=source_mean,
            source_std=source_std,
        )
        for row in episode["candidates"]:
            key = (
                bool(row["use_alignment"]),
                str(row["prototype_rule"]),
                None if row["ridge"] is None else float(row["ridge"]),
            )
            aggregate.setdefault(key, []).append(dict(row))

    candidates: list[dict[str, Any]] = []
    for (use_alignment, prototype_rule, ridge), rows in aggregate.items():
        if len(rows) != episode_count:
            raise RuntimeError("source head candidate coverage drift across episodes")
        candidates.append(
            {
                "use_alignment": use_alignment,
                "prototype_rule": prototype_rule,
                "ridge": ridge,
                "mean_accuracy": float(np.mean([row["accuracy"] for row in rows])),
                "worst_episode_accuracy": float(
                    np.min([row["accuracy"] for row in rows])
                ),
                "mean_min_class_accuracy": float(
                    np.mean([row["min_class_accuracy"] for row in rows])
                ),
                "mean_true_class_rank": float(
                    np.mean([row["mean_true_class_rank"] for row in rows])
                ),
                "complexity_rank": int(rows[0]["complexity_rank"]),
            }
        )
    selected = max(
        candidates,
        key=lambda row: (
            float(row["mean_accuracy"]),
            float(row["worst_episode_accuracy"]),
            float(row["mean_min_class_accuracy"]),
            -float(row["mean_true_class_rank"]),
            -int(row["complexity_rank"]),
        ),
    )
    identity = next(
        row
        for row in candidates
        if row["use_alignment"] is False
        and row["prototype_rule"] == "mean"
        and row["ridge"] is None
    )
    return {
        "selection_source": "disjoint_source_receiver_holdout_k1_episodes",
        "support_view_policy": "three_leo_weak_scenario_base_views",
        "support_receive_views_per_physical_sample": 3,
        "allowed_k": [1, 5, 10, 20],
        "episode_count": int(episode_count),
        "source_class_count": int(len(class_ids)),
        "selected": dict(selected),
        "identity_reference": dict(identity),
        "candidates": candidates,
        "episode_support_hashes": episode_support_hashes,
        "target_support_used_for_selection": False,
        "target_query_features_used": False,
        "target_query_labels_used": False,
        "old_new_role_oracle_used": False,
        "class_quota_used": False,
    }


def build_locked_nested_k_source_scores(
    adapted_features: np.ndarray,
    physical_labels: np.ndarray,
    calibration_physical: np.ndarray,
    evaluation_physical: np.ndarray,
    *,
    physical_count: int,
    scenario_count: int,
    selected: dict[str, Any],
    source_mean: np.ndarray,
    source_std: np.ndarray,
    k_values: Sequence[int] = FORMAL_K,
) -> dict[str, Any]:
    """Score source-only nested-K episodes with the exact deployed head rule."""

    features = np.asarray(adapted_features, dtype=np.float32)
    labels = np.asarray(physical_labels, dtype=np.int64).reshape(-1)
    calibration = np.asarray(calibration_physical, dtype=np.int64).reshape(-1)
    evaluation = np.asarray(evaluation_physical, dtype=np.int64).reshape(-1)
    values_k = tuple(int(value) for value in k_values)
    if (
        features.ndim != 3
        or features.shape[0] != int(physical_count) * int(scenario_count)
        or features.shape[1] != 5
        or len(labels) != int(physical_count)
        or int(scenario_count) != 3
        or tuple(sorted(values_k)) != tuple(values_k)
        or not values_k
    ):
        raise ValueError("invalid source nested-K feature layout or K grid")
    class_ids = tuple(int(value) for value in sorted(np.unique(labels).tolist()))
    class_to_position = {value: index for index, value in enumerate(class_ids)}
    support_pool: dict[int, np.ndarray] = {}
    calibration_query_parts: list[np.ndarray] = []
    max_k = int(max(values_k))
    for class_id in class_ids:
        indices = calibration[labels[calibration] == int(class_id)]
        if len(indices) <= max_k:
            raise ValueError(
                f"source class {class_id} needs >{max_k} calibration samples"
            )
        support_pool[int(class_id)] = indices[:max_k]
        calibration_query_parts.append(indices[max_k:])
    calibration_query = np.asarray(
        sorted(np.concatenate(calibration_query_parts).tolist()), dtype=np.int64
    )
    if len(calibration_query) == 0 or len(evaluation) == 0:
        raise ValueError("source nested-K calibration/evaluation query split is empty")

    def support_observations(k_shot: int) -> np.ndarray:
        by_class = []
        for class_id in class_ids:
            indices = support_pool[int(class_id)][: int(k_shot)]
            by_scenario = np.stack(
                [
                    features[
                        int(scenario_index) * int(physical_count) + indices,
                        0,
                        :,
                    ]
                    for scenario_index in range(int(scenario_count))
                ],
                axis=0,
            )
            by_class.append(by_scenario)
        # [C,3,K,D] -> [3*K,C,D]
        stacked = np.stack(by_class, axis=0)
        return np.transpose(stacked, (1, 2, 0, 3)).reshape(
            3 * int(k_shot), len(class_ids), features.shape[-1]
        )

    def query_block(indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        rows = np.concatenate(
            [
                features[
                    int(scenario_index) * int(physical_count) + indices,
                    :,
                    :,
                ]
                for scenario_index in range(int(scenario_count))
            ],
            axis=0,
        ).astype(np.float32)
        raw_labels = np.tile(labels[indices], int(scenario_count))
        mapped = np.asarray(
            [class_to_position[int(value)] for value in raw_labels], dtype=np.int64
        )
        return rows, mapped

    calibration_rows, calibration_labels = query_block(calibration_query)
    evaluation_rows, evaluation_labels = query_block(evaluation)
    identity_selected = {
        "use_alignment": False,
        "prototype_rule": "mean",
        "ridge": None,
    }
    calibration_score_parts: list[np.ndarray] = []
    evaluation_scores: dict[str, np.ndarray] = {}
    identity_evaluation_scores: dict[str, np.ndarray] = {}
    support_indices_by_k: dict[str, list[int]] = {}
    for k_shot in values_k:
        observations = support_observations(int(k_shot))
        head = fit_locked_symmetric_support_head(
            observations,
            physical_shots_per_class=int(k_shot),
            selected=selected,
            source_mean=source_mean,
            source_std=source_std,
        )
        identity_head = fit_locked_symmetric_support_head(
            observations,
            physical_shots_per_class=int(k_shot),
            selected=identity_selected,
            source_mean=source_mean,
            source_std=source_std,
        )
        calibration_score_parts.append(
            score_symmetric_head(calibration_rows, head).astype(np.float32)
        )
        evaluation_scores[str(k_shot)] = score_symmetric_head(
            evaluation_rows, head
        ).astype(np.float32)
        identity_evaluation_scores[str(k_shot)] = score_symmetric_head(
            evaluation_rows, identity_head
        ).astype(np.float32)
        support_indices_by_k[str(k_shot)] = sorted(
            np.concatenate(
                [support_pool[int(class_id)][: int(k_shot)] for class_id in class_ids]
            ).astype(np.int64).tolist()
        )
    return {
        "calibration_scores": np.concatenate(calibration_score_parts, axis=0),
        "calibration_labels": np.tile(calibration_labels, len(values_k)),
        "evaluation_scores_by_k": evaluation_scores,
        "identity_evaluation_scores_by_k": identity_evaluation_scores,
        "evaluation_labels": evaluation_labels,
        "support_indices_by_k": support_indices_by_k,
        "calibration_query_count": int(len(calibration_query)),
        "evaluation_query_count": int(len(evaluation)),
        "scenario_count": int(scenario_count),
        "k_values": list(values_k),
        "target_rows_used": False,
        "role_labels_used": False,
        "class_quota_used": False,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--adapter_state", type=Path, required=True)
    parser.add_argument("--training_manifest", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--wisig_pkl", type=Path, required=True)
    parser.add_argument("--source_tx_ids", required=True)
    parser.add_argument("--source_train_rxs", default="0,1,2,3,4,5")
    parser.add_argument("--source_val_rxs", default="6")
    parser.add_argument("--wisig_equalized", default="1")
    parser.add_argument("--wisig_domain", default="rx_day")
    parser.add_argument("--wisig_out_len", type=int, default=256)
    parser.add_argument("--num_old_classes", type=int, default=6)
    parser.add_argument("--max_train_samples_per_tx", type=int, default=400)
    parser.add_argument("--max_val_samples_per_tx", type=int, default=120)
    parser.add_argument(
        "--sat_scenarios",
        default="leo_clear_weak,leo_low_elev_weak,leo_rain_weak",
    )
    parser.add_argument("--star_ground_channel_impl", default="simplified_leo_residual")
    parser.add_argument("--sat_fs_hz", type=float, default=25e6)
    parser.add_argument("--sat_fc_hz", type=float, default=2.462e9)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--max_accuracy_drop_pp", type=float, default=0.5)
    parser.add_argument("--max_fixed1_drop_pp", type=float, default=1.0)
    parser.add_argument("--max_floor_drop_pp", type=float, default=2.0)
    parser.add_argument("--max_mean_backbone_forwards", type=float, default=3.0)
    parser.add_argument("--min_extra_view_rate", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=4070391)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    if not 1 <= int(args.batch_size) <= 4096:
        raise ValueError("batch_size must be in [1,4096]")
    if int(args.num_old_classes) < 2:
        raise ValueError("source validation requires at least two classes")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    receiver_audit = validate_receiver_holdout(
        args.source_train_rxs, args.source_val_rxs
    )
    manifest = json.loads(args.training_manifest.read_text(encoding="utf-8-sig"))
    if manifest.get("method") not in {
        "ground_source_full_feature_lora_v1",
        "ground_source_effective_feature_lora_v1",
    }:
        raise ValueError("training manifest is not a ground source LoRA")
    if manifest.get("source_only") is not True:
        raise ValueError("ground LoRA manifest must be source-only")
    if manifest.get("source_validation_pass") is not False:
        raise ValueError("training manifest was already promoted or mutated")
    if str(manifest.get("source_receiver_scope", "")) != str(
        args.source_train_rxs
    ):
        raise ValueError("source receiver scope does not match training manifest")
    if str(manifest.get("source_tx_scope", "")) != str(args.source_tx_ids):
        raise ValueError("source TX scope does not match training manifest")
    source_dataset_path = Path(str(manifest.get("source_dataset", "")))
    if (
        not source_dataset_path.is_file()
        or source_dataset_path.resolve() != args.wisig_pkl.resolve()
        or str(manifest.get("source_dataset_sha256", ""))
        != _sha256_file(source_dataset_path)
    ):
        raise ValueError("source dataset path/fingerprint mismatch")
    if manifest.get("method") == "ground_source_effective_feature_lora_v1" and (
        manifest.get("proxy_data_used_for_training") is not False
        or int(manifest.get("proxy_training_rows", -1)) != 0
        or float(manifest.get("proxy_loss_weight_sum", -1.0)) != 0.0
        or manifest.get("clean_samples_used_for_training") is not False
        or manifest.get("formal_training_view") != "leo_weak_only"
        or manifest.get("teacher_reference_view") != "same_leo_weak_base_view"
    ):
        raise ValueError(
            "effective ground LoRA must be leo_weak-only without clean/proxy rows"
        )
    if str(manifest.get("adapter_state_sha256", "")) != _sha256_file(
        args.adapter_state
    ):
        raise ValueError("ground LoRA state hash mismatch")
    if str(manifest.get("checkpoint_sha256", "")) != _sha256_file(args.ckpt):
        raise ValueError("checkpoint hash mismatch")

    scenarios = parse_sat_scenarios(str(args.sat_scenarios))
    validate_formal_scenarios(scenarios)
    training_scenarios = tuple(
        str(value)
        for value in dict(manifest.get("hyperparameters", {})).get(
            "sat_scenarios", []
        )
    )
    if training_scenarios != FORMAL_LEO_WEAK_SCENARIOS:
        raise ValueError("ground training scenario tuple differs from formal leo_weak")
    device = torch.device(str(args.device) if torch.cuda.is_available() else "cpu")
    train_ds, train_info = _dataset(
        args,
        receivers=str(args.source_train_rxs),
        role="source_train_for_prototypes",
        max_samples_per_tx=int(args.max_train_samples_per_tx),
        seed=int(args.seed) + 101,
    )
    val_ds, val_info = _dataset(
        args,
        receivers=str(args.source_val_rxs),
        role="source_receiver_holdout",
        max_samples_per_tx=int(args.max_val_samples_per_tx),
        seed=int(args.seed) + 211,
    )
    train_loader = DataLoader(
        train_ds, batch_size=int(args.batch_size), shuffle=False, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=int(args.batch_size), shuffle=False, num_workers=0
    )
    base_model = _build_model(args, train_ds, device, freeze=True)
    adapted_model = _build_model(args, train_ds, device, freeze=True)
    state = torch.load(args.adapter_state, map_location="cpu")
    if not isinstance(state, dict):
        raise TypeError("ground LoRA state must be a tensor dictionary")
    hp = dict(manifest["hyperparameters"])
    adapter_audit = apply_fp16_lora_state(
        adapted_model,
        state,
        scope=str(hp["scope"]),
        rank=int(hp["rank"]),
        alpha=float(hp["alpha"]),
    )
    base_model.to(device).eval()
    adapted_model.to(device).eval()
    prototypes = _leo_source_prototypes(
        base_model,
        train_loader,
        class_count=int(args.num_old_classes),
        scenarios=scenarios,
        args=args,
        device=device,
    )
    scores, repeated_labels, extraction_audit, adapted_features = _heldout_scores(
        base_model,
        adapted_model,
        val_loader,
        prototypes,
        scenarios=scenarios,
        args=args,
        device=device,
    )
    physical_labels = repeated_labels[: int(len(val_ds))]
    calibration_physical, evaluation_physical = stratified_physical_split(
        physical_labels
    )
    calibration_indices = _expanded_indices(
        calibration_physical,
        physical_count=int(len(val_ds)),
        scenario_count=len(scenarios),
    )
    evaluation_indices = _expanded_indices(
        evaluation_physical,
        physical_count=int(len(val_ds)),
        scenario_count=len(scenarios),
    )
    calibration_feature_rows = adapted_features[calibration_indices].reshape(
        -1, adapted_features.shape[-1]
    )
    source_feature_mean = calibration_feature_rows.mean(axis=0).astype(np.float32)
    source_feature_std = np.maximum(
        calibration_feature_rows.std(axis=0), 0.05
    ).astype(np.float32)
    symmetric_head_lock = build_source_symmetric_head_lock(
        adapted_features,
        physical_labels,
        calibration_physical,
        physical_count=int(len(val_ds)),
        scenario_count=len(scenarios),
        source_mean=source_feature_mean,
        source_std=source_feature_std,
    )
    nested_source = build_locked_nested_k_source_scores(
        adapted_features,
        physical_labels,
        calibration_physical,
        evaluation_physical,
        physical_count=int(len(val_ds)),
        scenario_count=len(scenarios),
        selected=dict(symmetric_head_lock["selected"]),
        source_mean=source_feature_mean,
        source_std=source_feature_std,
    )
    calibration = calibrate_adaptive_rxlight_tta(
        nested_source["calibration_scores"],
        nested_source["calibration_labels"],
        base_margin_grid=BASE_MARGIN_GRID,
        shift3_margin_grid=SHIFT3_MARGIN_GRID,
        disagreement_grid=DISAGREEMENT_GRID,
        max_accuracy_drop_pp=float(args.max_accuracy_drop_pp),
    )
    threshold_payload = calibration["selected"]["thresholds"]
    if isinstance(threshold_payload, AdaptiveTTAThresholds):
        thresholds = threshold_payload
    else:
        thresholds = AdaptiveTTAThresholds(**dict(threshold_payload))
    nested_eval_scores = np.concatenate(
        [nested_source["evaluation_scores_by_k"][str(k)] for k in FORMAL_K],
        axis=0,
    )
    nested_eval_labels = np.tile(
        nested_source["evaluation_labels"], len(FORMAL_K)
    )
    adaptive = apply_adaptive_rxlight_tta(nested_eval_scores, thresholds)
    eval_labels = repeated_labels[evaluation_indices]
    locked_head_by_k: dict[str, Any] = {}
    identity_head_by_k: dict[str, Any] = {}
    for k_shot in FORMAL_K:
        key = str(k_shot)
        k_scores = nested_source["evaluation_scores_by_k"][key]
        identity_scores = nested_source["identity_evaluation_scores_by_k"][key]
        k_adaptive = apply_adaptive_rxlight_tta(k_scores, thresholds)
        locked_head_by_k[key] = {
            **_fixed_metrics(k_scores, nested_source["evaluation_labels"]),
            "adaptive": {
                **_metric_row(
                    k_adaptive["predictions"], nested_source["evaluation_labels"]
                ),
                "mean_backbone_forwards": float(
                    k_adaptive["mean_backbone_forwards"]
                ),
                "p95_backbone_forwards": float(
                    k_adaptive["p95_backbone_forwards"]
                ),
                "trigger_rates": k_adaptive["trigger_rates"],
            },
        }
        identity_head_by_k[key] = _fixed_metrics(
            identity_scores, nested_source["evaluation_labels"]
        )
    metrics = {
        "base_checkpoint": _fixed_metrics(
            scores["base"][evaluation_indices], eval_labels
        ),
        "ground_lora": _fixed_metrics(
            scores["ground_lora"][evaluation_indices], eval_labels
        ),
        "ground_lora_adaptive": {
            **_metric_row(adaptive["predictions"], nested_eval_labels),
            "mean_backbone_forwards": float(adaptive["mean_backbone_forwards"]),
            "p95_backbone_forwards": float(adaptive["p95_backbone_forwards"]),
            "trigger_rates": adaptive["trigger_rates"],
        },
        "deployed_locked_head_by_k": locked_head_by_k,
        "identity_mean_head_by_k": identity_head_by_k,
    }
    base_fixed1 = metrics["base_checkpoint"]["fixed1"]
    lora_fixed1 = metrics["ground_lora"]["fixed1"]
    locked_fixed1_accuracy = float(
        np.mean(
            [locked_head_by_k[str(k)]["fixed1"]["accuracy"] for k in FORMAL_K]
        )
    )
    locked_fixed5_accuracy = float(
        np.mean(
            [locked_head_by_k[str(k)]["fixed5"]["accuracy"] for k in FORMAL_K]
        )
    )
    lora_adaptive = metrics["ground_lora_adaptive"]
    extra_view_rate = float(
        1.0 - lora_adaptive["trigger_rates"]["view1_rate"]
    )
    gates = {
        "receiver_holdout_disjoint": receiver_audit["disjoint"] is True,
        "no_target_data_in_training": manifest.get(
            "target_receiver_data_used_for_training"
        )
        is False,
        "fixed1_no_material_drop": float(lora_fixed1["accuracy"])
        >= float(base_fixed1["accuracy"])
        - float(args.max_fixed1_drop_pp) / 100.0,
        "class_floor_no_material_drop": float(lora_fixed1["min_class_accuracy"])
        >= float(base_fixed1["min_class_accuracy"])
        - float(args.max_floor_drop_pp) / 100.0,
        "five_view_not_worse_than_one": locked_fixed5_accuracy
        + 1.0e-12
        >= locked_fixed1_accuracy,
        "adaptive_not_worse_than_one": float(lora_adaptive["accuracy"])
        + 1.0e-12
        >= locked_fixed1_accuracy,
        "adaptive_mean_forward_cap": float(
            lora_adaptive["mean_backbone_forwards"]
        )
        <= float(args.max_mean_backbone_forwards),
        "adaptive_extra_views_are_exercised": extra_view_rate
        >= float(args.min_extra_view_rate),
        "combined_state_within_cap": manifest.get("resources", {}).get(
            "combined_persistent_state_within_cap"
        )
        is True,
        "symmetric_head_not_worse_than_identity": float(
            symmetric_head_lock["selected"]["mean_accuracy"]
        )
        + 1.0e-12
        >= float(symmetric_head_lock["identity_reference"]["mean_accuracy"]),
        "locked_head_each_k_not_worse_than_identity": all(
            float(locked_head_by_k[str(k)]["fixed1"]["accuracy"]) + 1.0e-12
            >= float(identity_head_by_k[str(k)]["fixed1"]["accuracy"])
            for k in FORMAL_K
        ),
    }
    failed = [name for name, passed in gates.items() if not passed]
    source_validation_pass = not failed
    args.out_dir.mkdir(parents=True, exist_ok=False)
    source_stats_path = args.out_dir / "source_joint_feature_stats_fp32.npz"
    np.savez(
        source_stats_path,
        mean=source_feature_mean,
        std=source_feature_std,
        feature_dim=np.asarray([source_feature_mean.size], dtype=np.int64),
        fft_dim=np.asarray([96], dtype=np.int64),
        fft_weight=np.asarray([2.0], dtype=np.float32),
    )
    result = _serializable({
        "schema": "cvs_ground_source_lora_multiview_validation_v1",
        "checkpoint_sha256": _sha256_file(args.ckpt),
        "adapter_state_sha256": _sha256_file(args.adapter_state),
        "training_manifest_sha256": _sha256_file(args.training_manifest),
        "source_feature_statistics": {
            "path": str(source_stats_path),
            "sha256": _sha256_file(source_stats_path),
            "feature_kind": "normalized_z_id_plus_fft96_weight2",
            "feature_dim": int(source_feature_mean.size),
            "fft_dim": 96,
            "fft_weight": 2.0,
            "row_count": int(len(calibration_feature_rows)),
            "target_rows_used": False,
        },
        "source_validation_pass": source_validation_pass,
        "failed_gates": failed,
        "gates": gates,
        "receiver_holdout": receiver_audit,
        "source_train_dataset": train_info,
        "source_validation_dataset": val_info,
        "physical_split": {
            "calibration_physical_count": int(len(calibration_physical)),
            "evaluation_physical_count": int(len(evaluation_physical)),
            "same_physical_sample_cross_split": False,
        },
        "scenarios": scenarios,
        "clean_samples_used_for_validation": False,
        "prototype_reference_view": "rotating_formal_leo_weak_only",
        "metrics": metrics,
        "calibration": calibration,
        "symmetric_head_lock": symmetric_head_lock,
        "nested_k_source_lock": {
            key: value
            for key, value in nested_source.items()
            if key
            not in {
                "calibration_scores",
                "calibration_labels",
                "evaluation_scores_by_k",
                "identity_evaluation_scores_by_k",
                "evaluation_labels",
            }
        },
        "adapter_audit": adapter_audit,
        "extraction_audit": extraction_audit,
        "permissions": {
            "source_validation_labels_used": True,
            "target_support_used": False,
            "target_query_features_used": False,
            "target_query_labels_used": False,
            "old_new_role_oracle_used": False,
            "class_quota_used": False,
        },
    })
    validation_path = args.out_dir / "source_validation.json"
    validation_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    promoted = dict(manifest)
    promoted.update(
        {
            "source_validation_pass": source_validation_pass,
            "source_validation_manifest": str(validation_path),
            "source_validation_manifest_sha256": _sha256_file(validation_path),
            "training_manifest_sha256": _sha256_file(args.training_manifest),
            "source_validation_failed_gates": failed,
            "source_validation_receiver_scope": str(args.source_val_rxs),
            "source_validation_permissions": result["permissions"],
            "source_feature_statistics": result["source_feature_statistics"],
            "symmetric_head_lock": result["symmetric_head_lock"],
        }
    )
    promotion_path = args.out_dir / "promotion_manifest.json"
    promotion_path.write_text(
        json.dumps(promoted, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "source_validation_pass": source_validation_pass,
                "failed_gates": failed,
                "metrics": metrics,
                "promotion_manifest": str(promotion_path),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return 0 if source_validation_pass else 3


if __name__ == "__main__":
    raise SystemExit(main())
