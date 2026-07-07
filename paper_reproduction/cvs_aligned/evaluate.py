from __future__ import annotations

import argparse
import csv
import inspect
import json
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from paper_reproduction.common.config import contains_unresolved_placeholder, contains_unspecified, load_json_config
from paper_reproduction.common.wisig_runtime import collate_wisig, load_wisig_compact_pkl, set_seed, write_json
from paper_reproduction.cvs_aligned.metrics import compute_cvs_stage2_metrics
from paper_reproduction.cvs_aligned.protocol import validate_stage2_protocol_payload
from paper_reproduction.feature_separation_crossrx.losses import feature_separation_loss
from paper_reproduction.feature_separation_crossrx.model import FeatureSeparationNet, build_wisig_fusion_representation
from paper_reproduction.protonet_cda.train import ProtoEmbeddingNet, _group_indices, _sample_episode

from baselines.drift.losses import ReceiverCenterEMA, compute_drift_loss
from baselines.drift.model import DRIFTModel
from baselines.riei_fd.model import RIEIModel
from baselines.riei_fd.train import alternating_training_step


ROOT = Path(__file__).resolve().parents[2]
CODE_DIR = ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from cvsrffi.eval import apply_sat_channel_for_scenario
from dataset_wisig import WiSigCompactDataset, _pad_or_crop_2t, _rms_normalize_iq


def _idx(labels: list[Any], label: Any) -> int:
    lookup = {str(v): i for i, v in enumerate(labels)}
    key = str(label)
    if key not in lookup:
        raise ValueError(f"label {label!r} not found in {labels[:8]}...")
    return int(lookup[key])


def _filter_kwargs_for_callable(fn, kwargs: dict[str, Any]) -> dict[str, Any]:
    accepted = inspect.signature(fn).parameters
    return {key: value for key, value in kwargs.items() if key in accepted}


def _eq_idx(ds: dict[str, Any], equalized: int) -> int:
    eq = list(ds.get("equalized_list", [0]))
    if int(equalized) not in eq:
        raise ValueError(f"equalized={equalized} not found in {eq}")
    return int(eq.index(int(equalized)))


def _iq_from_raw(raw: Any, *, out_len: int = 256) -> torch.Tensor:
    arr = np.asarray(raw, dtype=np.float32)
    x_2t = arr if arr.ndim == 2 and arr.shape[0] == 2 else arr.T
    x_2t = _pad_or_crop_2t(x_2t, out_len, mode="center")
    x_2t = _rms_normalize_iq(x_2t)
    return torch.tensor(x_2t, dtype=torch.float32)


def _available_count(ds: dict[str, Any], *, tx_label: str, rx_label: str, day_i: int, equalized: int) -> int:
    tx_i = _idx(list(ds.get("tx_list", [])), tx_label)
    rx_i = _idx(list(ds.get("rx_list", [])), rx_label)
    eq_i = _eq_idx(ds, equalized)
    arr = ds["data"][tx_i][rx_i][int(day_i)][eq_i]
    return 0 if arr is None else int(arr.shape[0])


def _take_samples(
    ds: dict[str, Any],
    *,
    tx_label: str,
    rx_label: str,
    day_i: int,
    equalized: int,
    count: int,
    offset: int,
    class_label: int,
    role: str,
) -> tuple[list[torch.Tensor], list[int], list[dict[str, Any]]]:
    tx_i = _idx(list(ds.get("tx_list", [])), tx_label)
    rx_i = _idx(list(ds.get("rx_list", [])), rx_label)
    eq_i = _eq_idx(ds, equalized)
    arr = ds["data"][tx_i][rx_i][int(day_i)][eq_i]
    if arr is None or int(arr.shape[0]) < int(offset) + int(count):
        got = 0 if arr is None else int(arr.shape[0])
        raise ValueError(f"insufficient samples for tx={tx_label} rx={rx_label}: need {offset + count}, got {got}")
    xs: list[torch.Tensor] = []
    ys: list[int] = []
    meta: list[dict[str, Any]] = []
    for sig_i in range(int(offset), int(offset) + int(count)):
        xs.append(_iq_from_raw(arr[sig_i]))
        ys.append(int(class_label))
        meta.append(
            {
                "sample_id": f"{tx_label}|{rx_label}|day{day_i}|eq{equalized}|sig{sig_i}",
                "tx_label": str(tx_label),
                "rx_label": str(rx_label),
                "day_i": int(day_i),
                "sig_i": int(sig_i),
                "role": str(role),
            }
        )
    return xs, ys, meta


def _make_source_dataset(config: dict[str, Any], ds: dict[str, Any]):
    tx_labels = [str(v) for v in config["target_old_tx_labels"]]
    rx_labels = [str(v) for v in config["source_receiver_labels"]]
    day_keep = [int(v) for v in config.get("source_days", [0, 1])]
    tx_keep = [_idx(list(ds.get("tx_list", [])), label) for label in tx_labels]
    rx_keep = [_idx(list(ds.get("rx_list", [])), label) for label in rx_labels]
    return WiSigCompactDataset(
        ds,
        **_filter_kwargs_for_callable(
            WiSigCompactDataset,
            {
                "out_len": 256,
                "equalized": int(config.get("equalized", 1)),
                "tx_keep": tx_keep,
                "rx_keep": rx_keep,
                "day_keep": day_keep,
                "domain": "rx",
                "max_samples_per_combo": int(config.get("source_train_samples_per_combo", 30)),
                "sample_strategy": "front",
                "seed": int(config.get("seed", 1337)),
            },
        ),
    )


def _target_scenarios(config: dict[str, Any]) -> list[str]:
    scenarios = [str(v) for v in config.get("target_channel_scenarios", ["clean"])]
    return scenarios or ["clean"]


def _train_scenarios(config: dict[str, Any]) -> list[str]:
    if str(config.get("train_channel_view", config.get("target_channel_view", "clean"))) == "clean":
        return ["clean"]
    scenarios = [str(v) for v in config.get("train_channel_scenarios", config.get("target_channel_scenarios", []))]
    return scenarios or ["leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak"]


def _scenario_counts_for_steps(scenarios: list[str], steps: int) -> dict[str, int]:
    counts = {scenario: 0 for scenario in scenarios}
    for step in range(max(0, int(steps))):
        scenario = scenarios[step % len(scenarios)]
        counts[scenario] = counts.get(scenario, 0) + 1
    return counts


def _compact_receiver_targets(raw_rx: torch.Tensor, mapping: dict[int, int], *, device: torch.device) -> torch.Tensor:
    values = [int(mapping[int(v)]) for v in raw_rx.detach().cpu().reshape(-1).tolist()]
    return torch.tensor(values, dtype=torch.long, device=device)


def _apply_scenario(x: torch.Tensor, scenario: str, *, seed: int) -> torch.Tensor:
    if str(scenario) == "clean":
        return x
    args = SimpleNamespace(sat_fs_hz=25e6, sat_fc_hz=2.462e9)
    gen = torch.Generator(device=x.device).manual_seed(int(seed))
    y, _ = apply_sat_channel_for_scenario(x, scenario, args, gen=gen, return_meta=False)
    return y


def _load_checkpoint_state(model: nn.Module, checkpoint_path: str, device: torch.device) -> dict[str, Any]:
    path = Path(checkpoint_path)
    if not checkpoint_path:
        return {}
    ckpt = torch.load(path, map_location=device)
    if not isinstance(ckpt, dict):
        raise ValueError(f"checkpoint must be a dict: {checkpoint_path}")
    state = ckpt.get("model") or ckpt.get("model_state_dict") or ckpt.get("state_dict")
    if not isinstance(state, dict):
        raise ValueError(f"checkpoint has no model state dict: {checkpoint_path}")
    model.load_state_dict(state, strict=True)
    return {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_epoch": ckpt.get("epoch"),
        "checkpoint_stats": ckpt.get("stats", {}),
        "checkpoint_loaded_strict": True,
    }


def _train_model(config: dict[str, Any], manysig: dict[str, Any], device: torch.device):
    baseline = str(config.get("baseline", "protonet_cda"))
    source_ds = _make_source_dataset(config, manysig)
    seed = int(config.get("seed", 1337))
    max_steps = int(config.get("max_steps", 10))
    if baseline == "protonet_cda":
        model = ProtoEmbeddingNet(embedding_dim=int(config.get("embedding_dim", 128))).to(device)
        opt = torch.optim.SGD(model.parameters(), lr=float(config.get("learning_rate", 0.01)))
        groups = _group_indices(source_ds)
        rng = random.Random(seed)
        n_way = min(int(config.get("n_way", 6)), len(groups))
        scenarios = _train_scenarios(config)
        scenario_counts = {scenario: 0 for scenario in scenarios}
        for step in range(max_steps):
            sx, sy, qx, qy = _sample_episode(
                source_ds,
                groups,
                n_way=n_way,
                k_shot=min(int(config.get("source_episode_k", 5)), 5),
                query_per_class=int(config.get("source_episode_query", 5)),
                rng=rng,
                device=device,
            )
            scenario = scenarios[step % len(scenarios)]
            scenario_counts[scenario] = scenario_counts.get(scenario, 0) + 1
            sx = _apply_scenario(sx, scenario, seed=seed + 10_000 + step)
            qx = _apply_scenario(qx, scenario, seed=seed + 20_000 + step)
            opt.zero_grad(set_to_none=True)
            support = model(sx)
            query = model(qx)
            class_ids = torch.unique(sy).sort().values
            prototypes = torch.stack([support[sy == label].mean(dim=0) for label in class_ids], dim=0)
            logits = -torch.cdist(query, prototypes)
            compact = torch.empty_like(qy)
            for i, label in enumerate(class_ids):
                compact[qy == label] = int(i)
            loss = F.cross_entropy(logits, compact)
            loss.backward()
            opt.step()
        return model, {
            "source_train_size": len(source_ds),
            "steps": max_steps,
            "train_channel_view": str(config.get("train_channel_view", config.get("target_channel_view", "clean"))),
            "train_channel_scenarios": scenarios,
            "train_channel_scenario_counts": scenario_counts,
            "satellite_train_augmentation_enabled": any(scenario != "clean" for scenario in scenarios),
            "training_origin": "paper_baseline_random_init",
        }

    if baseline == "feature_separation_crossrx":
        model = FeatureSeparationNet(
            input_channels=3,
            input_length=256,
            num_tx=len(manysig.get("tx_list", [])),
            num_rx=len(manysig.get("rx_list", [])),
        ).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=float(config.get("learning_rate", 0.005)))
        loader = DataLoader(
            source_ds,
            batch_size=int(config.get("batch_size", 64)),
            shuffle=True,
            collate_fn=collate_wisig,
            drop_last=False,
        )
        steps = 0
        scenarios = _train_scenarios(config)
        scenario_counts = {scenario: 0 for scenario in scenarios}
        while steps < max_steps:
            for batch in loader:
                scenario = scenarios[steps % len(scenarios)]
                scenario_counts[scenario] = scenario_counts.get(scenario, 0) + 1
                iq = _apply_scenario(batch["iq"].to(device), scenario, seed=seed + 30_000 + steps)
                x = build_wisig_fusion_representation(iq)
                y = batch["label"].to(device)
                r = batch["domain"].to(device)
                opt.zero_grad(set_to_none=True)
                out = model(x)
                loss, _ = feature_separation_loss(
                    out,
                    y,
                    r,
                    lambda_similarity=float(config.get("lambda_similarity", 1.0)),
                    lambda_tx_entropy=float(config.get("lambda_tx_entropy", 1.0)),
                    lambda_rx_entropy=float(config.get("lambda_rx_entropy", 1.0)),
                )
                loss.backward()
                opt.step()
                steps += 1
                if steps >= max_steps:
                    break
        return model, {
            "source_train_size": len(source_ds),
            "steps": steps,
            "train_channel_view": str(config.get("train_channel_view", config.get("target_channel_view", "clean"))),
            "train_channel_scenarios": scenarios,
            "train_channel_scenario_counts": scenario_counts,
            "satellite_train_augmentation_enabled": any(scenario != "clean" for scenario in scenarios),
            "training_origin": "paper_baseline_random_init",
            "feature_separation_loss": {
                "lambda_similarity": float(config.get("lambda_similarity", 1.0)),
                "lambda_tx_entropy": float(config.get("lambda_tx_entropy", 1.0)),
                "lambda_rx_entropy": float(config.get("lambda_rx_entropy", 1.0)),
                "entropy_target": "cross_branch_logits",
            },
        }

    if baseline == "riei_fd":
        rx_labels = [str(v) for v in config["source_receiver_labels"]]
        rx_mapping = {_idx(list(manysig.get("rx_list", [])), label): i for i, label in enumerate(rx_labels)}
        model = RIEIModel(
            num_emitters=len(manysig.get("tx_list", [])),
            num_receivers=len(rx_mapping),
            feature_dim=int(config.get("feature_dim", 512)),
            dropout=float(config.get("dropout", 0.0)),
            encoder_use_projection=bool(config.get("use_resnet_projection", False)),
        ).to(device)
        checkpoint_info = _load_checkpoint_state(
            model,
            str(config.get("source_checkpoint_path") or config.get("checkpoint_path") or ""),
            device,
        )
        if checkpoint_info:
            return model, {
                "source_train_size": len(source_ds),
                "steps": 0,
                "train_channel_view": str(config.get("train_channel_view", config.get("target_channel_view", "clean"))),
                "train_channel_scenarios": _train_scenarios(config),
                "satellite_train_augmentation_enabled": any(scenario != "clean" for scenario in _train_scenarios(config)),
                "training_origin": str(config.get("training_origin", "pretrained_source_checkpoint")),
                "dg_reproduction_method": "RIEI",
                "method_version": str(config.get("method_version", "paper_original_finaltest")),
                "phase2_adapter": str(config.get("phase2_adapter", "ProtoNet-CDA")),
                "identity_embedding": "z_e",
                "source_checkpoint_run_id": str(config.get("source_checkpoint_run_id", "")),
                "source_train_ratio": float(config.get("rho_label", config.get("source_train_ratio", 0.0))),
                **checkpoint_info,
            }
        opt_all = torch.optim.Adam(
            model.parameters(),
            lr=float(config.get("lr_all", config.get("learning_rate", 1.0e-4))),
            weight_decay=float(config.get("weight_decay_all", 0.0)),
        )
        opt_fed = torch.optim.Adam(
            model.fed.parameters(),
            lr=float(config.get("lr_fed", config.get("learning_rate", 1.0e-4))),
            weight_decay=float(config.get("weight_decay_fed", 0.0)),
        )
        loader = DataLoader(
            source_ds,
            batch_size=int(config.get("batch_size", 64)),
            shuffle=True,
            collate_fn=collate_wisig,
            drop_last=False,
        )
        steps = 0
        scenarios = _train_scenarios(config)
        scenario_counts = {scenario: 0 for scenario in scenarios}
        while steps < max_steps:
            for batch in loader:
                scenario = scenarios[steps % len(scenarios)]
                scenario_counts[scenario] = scenario_counts.get(scenario, 0) + 1
                iq = _apply_scenario(batch["iq"].to(device), scenario, seed=seed + 40_000 + steps)
                receiver_target = _compact_receiver_targets(batch["domain"], rx_mapping, device=device)
                train_batch = {
                    "iq": iq,
                    "label": batch["label"].to(device),
                    "receiver": batch["domain"].to(device),
                    "receiver_target": receiver_target,
                }
                alternating_training_step(
                    model,
                    train_batch,
                    opt_all,
                    opt_fed,
                    lambda_mi=float(config.get("lambda_mi", 1.2)),
                    lambda_ie=float(config.get("lambda_ie", 1.2)),
                    device=device,
                    mi_mode=str(config.get("mi_mode", "cosine_abs")),
                    ie_temperature=float(config.get("ie_temperature", 1.0)),
                    ce_reduction=str(config.get("ce_reduction", "mean")),
                    mi_reduction=str(config.get("mi_reduction", "mean")),
                    ie_reduction=str(config.get("ie_reduction", "mean")),
                    disentangle_steps=int(config.get("disentangle_steps", 1)),
                    grad_clip_norm=float(config.get("grad_clip_norm", 0.0)),
                    lambda_feature_norm=float(config.get("lambda_feature_norm", 0.0)),
                )
                steps += 1
                if steps >= max_steps:
                    break
        return model, {
            "source_train_size": len(source_ds),
            "steps": steps,
            "train_channel_view": str(config.get("train_channel_view", config.get("target_channel_view", "clean"))),
            "train_channel_scenarios": scenarios,
            "train_channel_scenario_counts": scenario_counts,
            "satellite_train_augmentation_enabled": any(scenario != "clean" for scenario in scenarios),
            "training_origin": str(config.get("training_origin", "source_domain_dg_reproduction_random_init")),
            "dg_reproduction_method": "RIEI",
            "method_version": str(config.get("method_version", "fix_optimized")),
            "phase2_adapter": str(config.get("phase2_adapter", "ProtoNet-CDA")),
            "identity_embedding": "z_e",
            "lambda_mi": float(config.get("lambda_mi", 1.2)),
            "lambda_ie": float(config.get("lambda_ie", 1.2)),
            "lambda_feature_norm": float(config.get("lambda_feature_norm", 0.0)),
        }

    if baseline == "drift":
        rx_labels = [str(v) for v in config["source_receiver_labels"]]
        rx_mapping = {_idx(list(manysig.get("rx_list", [])), label): i for i, label in enumerate(rx_labels)}
        model = DRIFTModel(
            num_tx=len(manysig.get("tx_list", [])),
            num_rx=len(rx_mapping),
            embedding_dim=int(config.get("embedding_dim", 512)),
            split_dim=int(config.get("split_dim", 256)),
            dropout=float(config.get("dropout", 0.0)),
            encoder_use_projection=bool(config.get("use_resnet_projection", False)),
            domain_discriminator_layers=int(config.get("domain_discriminator_layers", 2)),
        ).to(device)
        checkpoint_info = _load_checkpoint_state(
            model,
            str(config.get("source_checkpoint_path") or config.get("checkpoint_path") or ""),
            device,
        )
        if checkpoint_info:
            return model, {
                "source_train_size": len(source_ds),
                "steps": 0,
                "train_channel_view": str(config.get("train_channel_view", config.get("target_channel_view", "clean"))),
                "train_channel_scenarios": _train_scenarios(config),
                "satellite_train_augmentation_enabled": any(scenario != "clean" for scenario in _train_scenarios(config)),
                "training_origin": str(config.get("training_origin", "pretrained_source_checkpoint")),
                "dg_reproduction_method": "DRIFT",
                "method_version": str(config.get("method_version", "paper_original_finaltest")),
                "phase2_adapter": str(config.get("phase2_adapter", "ProtoNet-CDA")),
                "identity_embedding": "z_tx",
                "source_checkpoint_run_id": str(config.get("source_checkpoint_run_id", "")),
                "source_train_ratio": float(config.get("rho_label", config.get("source_train_ratio", 0.0))),
                **checkpoint_info,
            }
        opt_cls = torch.optim.AdamW if str(config.get("optimizer", "adam")) == "adamw" else torch.optim.Adam
        opt = opt_cls(
            model.parameters(),
            lr=float(config.get("learning_rate", config.get("lr", 1.0e-4))),
            weight_decay=float(config.get("weight_decay", 0.0)),
        )
        center_memory = None
        if str(config.get("center_mode", "ema")) == "ema":
            center_memory = ReceiverCenterEMA(
                num_receivers=len(rx_mapping),
                feature_dim=model.embedding_dim - model.split_dim,
                momentum=float(config.get("center_momentum", 0.95)),
            ).to(device)
        loader = DataLoader(
            source_ds,
            batch_size=int(config.get("batch_size", 64)),
            shuffle=True,
            collate_fn=collate_wisig,
            drop_last=False,
        )
        steps = 0
        scenarios = _train_scenarios(config)
        scenario_counts = {scenario: 0 for scenario in scenarios}
        while steps < max_steps:
            for batch in loader:
                scenario = scenarios[steps % len(scenarios)]
                scenario_counts[scenario] = scenario_counts.get(scenario, 0) + 1
                iq = _apply_scenario(batch["iq"].to(device), scenario, seed=seed + 50_000 + steps)
                receiver_target = _compact_receiver_targets(batch["domain"], rx_mapping, device=device)
                out = model(iq, grl_lambda=float(config.get("grl_coeff", 1.0)))
                losses = compute_drift_loss(
                    out,
                    batch["label"].to(device),
                    receiver_target,
                    lambda_grl=float(config.get("lambda_grl", 1.0)),
                    lambda_center=float(config.get("lambda_center", 0.01)),
                    lambda_mse=float(config.get("lambda_mse", 0.02)),
                    normalize_features_for_mse=bool(config.get("normalize_features_for_mse", False)),
                    mse_reduction=str(config.get("mse_reduction", "sum")),
                    mse_cap=float(config.get("mse_cap", 0.0)),
                    lambda_feature_norm=float(config.get("lambda_feature_norm", 0.0)),
                    feature_norm_target=float(config.get("feature_norm_target", 0.0)),
                    center_mode=str(config.get("center_mode", "ema")),
                    center_memory=center_memory,
                )
                opt.zero_grad(set_to_none=True)
                losses["loss"].backward()
                if float(config.get("grad_clip_norm", 0.0)) > 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.get("grad_clip_norm", 0.0)))
                opt.step()
                steps += 1
                if steps >= max_steps:
                    break
        return model, {
            "source_train_size": len(source_ds),
            "steps": steps,
            "train_channel_view": str(config.get("train_channel_view", config.get("target_channel_view", "clean"))),
            "train_channel_scenarios": scenarios,
            "train_channel_scenario_counts": scenario_counts,
            "satellite_train_augmentation_enabled": any(scenario != "clean" for scenario in scenarios),
            "training_origin": str(config.get("training_origin", "source_domain_dg_reproduction_random_init")),
            "dg_reproduction_method": "DRIFT",
            "method_version": str(config.get("method_version", "fix_optimized")),
            "phase2_adapter": str(config.get("phase2_adapter", "ProtoNet-CDA")),
            "identity_embedding": "z_tx",
            "lambda_grl": float(config.get("lambda_grl", 1.0)),
            "lambda_center": float(config.get("lambda_center", 0.01)),
            "lambda_mse": float(config.get("lambda_mse", 0.02)),
            "mse_cap": float(config.get("mse_cap", 0.0)),
        }
    raise ValueError(f"unsupported baseline: {baseline}")


def _embed(model, baseline: str, x: torch.Tensor, device: torch.device) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        x = x.to(device)
        if baseline == "feature_separation_crossrx":
            return model(build_wisig_fusion_representation(x))["tx_features"].detach().cpu()
        if baseline == "riei_fd":
            return model(x)["z_e"].detach().cpu()
        if baseline == "drift":
            return model(x, grl_lambda=0.0)["z_tx"].detach().cpu()
        return model(x).detach().cpu()


def _select_target_sets(config: dict[str, Any], manysig: dict[str, Any], manytx: dict[str, Any]) -> dict[str, Any]:
    target_rxs = [str(v) for v in config["target_receiver_labels"]]
    equalized = int(config.get("equalized", 1))
    day_i = int(config.get("target_day", 0))
    k = int(config["k_shot"])
    q = int(config.get("query_per_tx", 20))
    old_labels = [str(v) for v in config["target_old_tx_labels"]]
    needed_old = k + q
    for target_rx in target_rxs:
        for label in old_labels:
            _available_count(manysig, tx_label=label, rx_label=target_rx, day_i=day_i, equalized=equalized)
            if _available_count(manysig, tx_label=label, rx_label=target_rx, day_i=day_i, equalized=equalized) < needed_old:
                raise ValueError(f"insufficient old target samples for {label} on {target_rx}")
    return {
        "target_receiver_label": target_rxs[0],
        "target_receiver_labels": target_rxs,
        "manysig_target_receiver_indices": {
            target_rx: _idx(list(manysig.get("rx_list", [])), target_rx) for target_rx in target_rxs
        },
        "manytx_receiver_indices": {
            target_rx: _idx(list(manytx.get("rx_list", [])), target_rx) for target_rx in target_rxs
        },
    }


def _build_stage2_tensors(config: dict[str, Any], manysig: dict[str, Any], manytx: dict[str, Any]) -> dict[str, Any]:
    target_rxs = [str(v) for v in config["target_receiver_labels"]]
    day_i = int(config.get("target_day", 0))
    equalized = int(config.get("equalized", 1))
    k = int(config["k_shot"])
    q = int(config.get("query_per_tx", 20))
    old_labels = [str(v) for v in config["target_old_tx_labels"]]
    new_labels = [str(v) for v in config.get("target_new_tx_labels", [])]
    unknown_labels = [str(v) for v in config.get("target_unknown_tx_labels", [])]
    class_map = {label: i for i, label in enumerate(old_labels)}
    class_map.update({label: len(old_labels) + i for i, label in enumerate(new_labels)})

    support_x: list[torch.Tensor] = []
    support_y: list[int] = []
    query_x: list[torch.Tensor] = []
    query_y: list[int] = []
    support_meta: list[dict[str, Any]] = []
    query_meta: list[dict[str, Any]] = []

    for target_rx in target_rxs:
        for label in old_labels:
            xs, ys, meta = _take_samples(
                manysig,
                tx_label=label,
                rx_label=target_rx,
                day_i=day_i,
                equalized=equalized,
                count=k,
                offset=0,
                class_label=class_map[label],
                role="target_old_support",
            )
            support_x.extend(xs)
            support_y.extend(ys)
            support_meta.extend(meta)
            xs, ys, meta = _take_samples(
                manysig,
                tx_label=label,
                rx_label=target_rx,
                day_i=day_i,
                equalized=equalized,
                count=q,
                offset=k,
                class_label=class_map[label],
                role="target_old_query",
            )
            query_x.extend(xs)
            query_y.extend(ys)
            query_meta.extend(meta)

        for label in new_labels:
            xs, ys, meta = _take_samples(
                manytx,
                tx_label=label,
                rx_label=target_rx,
                day_i=day_i,
                equalized=equalized,
                count=k,
                offset=0,
                class_label=class_map[label],
                role="target_new_support",
            )
            support_x.extend(xs)
            support_y.extend(ys)
            support_meta.extend(meta)
            xs, ys, meta = _take_samples(
                manytx,
                tx_label=label,
                rx_label=target_rx,
                day_i=day_i,
                equalized=equalized,
                count=q,
                offset=k,
                class_label=class_map[label],
                role="target_new_query",
            )
            query_x.extend(xs)
            query_y.extend(ys)
            query_meta.extend(meta)

        for label in unknown_labels:
            xs, ys, meta = _take_samples(
                manytx,
                tx_label=label,
                rx_label=target_rx,
                day_i=day_i,
                equalized=equalized,
                count=q,
                offset=0,
                class_label=-1,
                role="target_unknown_query",
            )
            query_x.extend(xs)
            query_y.extend(ys)
            query_meta.extend(meta)

    support_ids = {m["sample_id"] for m in support_meta}
    query_ids = {m["sample_id"] for m in query_meta}
    if support_ids & query_ids:
        raise ValueError("support/query overlap detected")
    return {
        "support_x": torch.stack(support_x, dim=0),
        "support_y": torch.tensor(support_y, dtype=torch.long),
        "query_x": torch.stack(query_x, dim=0),
        "query_y": torch.tensor(query_y, dtype=torch.long),
        "support_meta": support_meta,
        "query_meta": query_meta,
        "class_map": class_map,
        "old_class_ids": set(class_map[label] for label in old_labels),
        "new_class_ids": set(class_map[label] for label in new_labels),
        "support_query_overlap": False,
    }


def _prototype_predict(
    support_z: torch.Tensor,
    support_y: torch.Tensor,
    query_z: torch.Tensor,
    *,
    margin: float,
    metric: str = "cosine",
    rejection_enabled: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    labels = torch.unique(support_y).sort().values
    metric = str(metric).lower()
    if metric == "cosine":
        support_n = F.normalize(support_z.float(), dim=1)
        query_n = F.normalize(query_z.float(), dim=1)
        protos = torch.stack([support_n[support_y == label].mean(dim=0) for label in labels], dim=0)
        protos = F.normalize(protos, dim=1)
        sim = query_n @ protos.t()
        max_sim, argmax = sim.max(dim=1)
        support_sim = (support_n @ protos.t()).max(dim=1).values
        threshold = float(torch.quantile(support_sim, 0.05).item()) - float(margin)
        pred = labels[argmax].clone()
        if rejection_enabled:
            pred[max_sim < threshold] = -1
        return pred.cpu(), (-max_sim).cpu(), {
            "gate_method": "prototype_cosine_support_quantile" if rejection_enabled else "prototype_cosine_no_rejection",
            "unknown_score_kind": "negative_max_similarity",
            "threshold": threshold,
            "unknown_rejection_enabled": bool(rejection_enabled),
        }
    if metric == "euclidean":
        support_f = support_z.float()
        query_f = query_z.float()
        protos = torch.stack([support_f[support_y == label].mean(dim=0) for label in labels], dim=0)
        query_dist = torch.cdist(query_f, protos)
        min_dist, argmin = query_dist.min(dim=1)
        support_dist = torch.cdist(support_f, protos).min(dim=1).values
        threshold = float(torch.quantile(support_dist, 0.95).item()) + float(margin)
        pred = labels[argmin].clone()
        if rejection_enabled:
            pred[min_dist > threshold] = -1
        return pred.cpu(), min_dist.cpu(), {
            "gate_method": "prototype_euclidean_support_quantile" if rejection_enabled else "prototype_euclidean_no_rejection",
            "unknown_score_kind": "min_euclidean_distance",
            "threshold": threshold,
            "unknown_rejection_enabled": bool(rejection_enabled),
        }
    raise ValueError(f"unsupported prototype_metric: {metric}")


def _support_head_finetune_predict(
    support_z: torch.Tensor,
    support_y: torch.Tensor,
    query_z: torch.Tensor,
    *,
    margin: float,
    steps: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
    rejection_enabled: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    labels = torch.unique(support_y).sort().values
    compact = torch.empty_like(support_y)
    for i, label in enumerate(labels):
        compact[support_y == label] = int(i)
    x = support_z.float().to(device)
    y = compact.to(device)
    head = nn.Linear(int(support_z.shape[1]), int(labels.numel())).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    for _ in range(max(1, int(steps))):
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(head(x), y)
        loss.backward()
        opt.step()
    with torch.no_grad():
        support_conf = F.softmax(head(x), dim=1).max(dim=1).values
        threshold = float(torch.quantile(support_conf.detach().cpu(), 0.05).item()) - float(margin)
        probs = F.softmax(head(query_z.float().to(device)), dim=1)
        max_prob, argmax = probs.max(dim=1)
    pred = labels[argmax.detach().cpu()].clone()
    if rejection_enabled:
        pred[max_prob.detach().cpu() < threshold] = -1
    return pred.cpu(), (-max_prob.detach().cpu()), {
        "gate_method": "support_head_confidence_quantile" if rejection_enabled else "support_head_no_rejection",
        "unknown_score_kind": "negative_head_confidence",
        "threshold": threshold,
        "support_finetune_steps": int(steps),
        "unknown_rejection_enabled": bool(rejection_enabled),
    }


def _write_score_table(path: Path, *, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _run(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    seed = int(args.seed if args.seed is not None else config.get("seed", 1337))
    set_seed(seed)
    device = torch.device(args.device)
    manysig = load_wisig_compact_pkl(str(args.manysig_pkl or config["manysig_pkl"]))
    manytx = load_wisig_compact_pkl(str(args.manytx_pkl or config["manytx_pkl"]))
    target_info = _select_target_sets(config, manysig, manytx)
    model, train_info = _train_model(config, manysig, device)
    tensors = _build_stage2_tensors(config, manysig, manytx)

    baseline = str(config["baseline"])
    target_scenarios = _target_scenarios(config)
    support_query_satellite = any(scenario != "clean" for scenario in target_scenarios)
    unknown_rejection_enabled = bool(config.get("unknown_rejection_enabled", True))
    metrics_by_scenario: dict[str, Any] = {}
    score_rows: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    for si, scenario in enumerate(target_scenarios):
        support_x = _apply_scenario(tensors["support_x"].to(device), str(scenario), seed=seed + 1000 + si).cpu()
        query_x = _apply_scenario(tensors["query_x"].to(device), str(scenario), seed=seed + 2000 + si).cpu()
        support_z = _embed(model, baseline, support_x, device)
        query_z = _embed(model, baseline, query_x, device)
        adaptation_mode = str(config.get("adaptation_mode", "support_prototype_registration"))
        if adaptation_mode == "support_head_finetune":
            pred, unknown_scores, gate_info = _support_head_finetune_predict(
                support_z,
                tensors["support_y"],
                query_z,
                margin=float(config.get("threshold_margin", 0.02)),
                steps=int(config.get("support_finetune_steps", 100)),
                lr=float(config.get("support_finetune_lr", 0.05)),
                weight_decay=float(config.get("support_finetune_weight_decay", 0.0)),
                device=device,
                rejection_enabled=unknown_rejection_enabled,
            )
        else:
            pred, unknown_scores, gate_info = _prototype_predict(
                support_z,
                tensors["support_y"],
                query_z,
                margin=float(config.get("threshold_margin", 0.02)),
                metric=str(config.get("prototype_metric", "cosine")),
                rejection_enabled=unknown_rejection_enabled,
            )
        metrics = compute_cvs_stage2_metrics(
            true_labels=tensors["query_y"],
            predicted_labels=pred,
            unknown_scores=unknown_scores,
            old_labels=tensors["old_class_ids"],
            new_labels=tensors["new_class_ids"],
        )
        metrics["unknown_score_kind"] = str(gate_info["unknown_score_kind"])
        metrics_by_scenario[str(scenario)] = metrics
        for i, meta in enumerate(tensors["query_meta"]):
            score_rows.append(
                {
                    **meta,
                    "scenario": str(scenario),
                    "true_label": int(tensors["query_y"][i].item()),
                    "predicted_label": int(pred[i].item()),
                    "accepted": bool(int(pred[i].item()) >= 0),
                    "unknown_score": float(unknown_scores[i].item()),
                    "adaptation_mode": adaptation_mode,
                    "gate_method": gate_info["gate_method"],
                }
            )
    elapsed = max(1e-9, time.perf_counter() - t0)

    metric_keys = ["old_acc", "target_old_accepted_acc", "target_old_coverage"]
    if config.get("target_new_tx_labels"):
        metric_keys.extend(["seen_new_acc", "H_old_new"])
    if unknown_rejection_enabled:
        metric_keys.extend(["unknown_FAR", "FPR95", "AUROC"])
    aggregate: dict[str, Any] = {}
    for key in metric_keys:
        vals = [float(m[key]) for m in metrics_by_scenario.values() if key in m and np.isfinite(float(m[key]))]
        if vals:
            aggregate[f"{key}_mean"] = float(np.mean(vals))

    protocol = validate_stage2_protocol_payload(
        {
            **config,
            **target_info,
            "target_old_tx_labels": config["target_old_tx_labels"],
            "target_new_tx_labels": config.get("target_new_tx_labels", []),
            "target_unknown_tx_labels": config.get("target_unknown_tx_labels", []),
            "unknown_rejection_enabled": unknown_rejection_enabled,
        }
    )
    split_manifest = {
        **protocol,
        **target_info,
        "support_count": len(tensors["support_meta"]),
        "query_count": len(tensors["query_meta"]),
        "old_support_sample_ids": [m["sample_id"] for m in tensors["support_meta"] if m["role"] == "target_old_support"],
        "old_query_sample_ids": [m["sample_id"] for m in tensors["query_meta"] if m["role"] == "target_old_query"],
        "new_support_sample_ids": [m["sample_id"] for m in tensors["support_meta"] if m["role"] == "target_new_support"],
        "new_query_sample_ids": [m["sample_id"] for m in tensors["query_meta"] if m["role"] == "target_new_query"],
        "unknown_query_sample_ids": [m["sample_id"] for m in tensors["query_meta"] if m["role"] == "target_unknown_query"],
        "support_query_overlap": tensors["support_query_overlap"],
        "all_support_query_from_R_t": True,
        "support_query_channel_view": str(config.get("target_channel_view", "clean")),
        "support_query_channel_scenarios": target_scenarios,
        "support_query_satellite_augmentation_enabled": support_query_satellite,
        "target_support_satellite_augmentation_enabled": support_query_satellite,
        "target_query_satellite_augmentation_enabled": support_query_satellite,
        "gate_method": str(gate_info["gate_method"]),
        "threshold_fit_scope": config.get("threshold_scope", "support_only_no_unknown_query"),
        "label_scope": "old_only_support_query" if not config.get("target_new_tx_labels") else "old_and_seen_new_support; unknown_query_eval_only",
        "unknown_rejection_enabled": unknown_rejection_enabled,
        "prototype_metric": config.get("prototype_metric", ""),
        "support_finetune_steps": config.get("support_finetune_steps", 0),
    }
    result = {
        "experiment_id": config.get("experiment_id", "cvs_aligned_stage2"),
        "baseline_name": baseline,
        "cvs_extension": True,
        "stage": protocol["stage"],
        "seed": seed,
        "train_info": train_info,
        "protocol": protocol,
        "split_manifest": split_manifest,
        "metrics_by_scenario": metrics_by_scenario,
        "metrics": {
            **aggregate,
            "latency_sec": float(elapsed),
            "latency_per_query_ms": float(elapsed * 1000.0 / max(1, len(score_rows))),
            "prototype_storage": int(len(tensors["support_y"].unique()) * support_z.shape[1]),
            "memory": {"embedding_dim": int(support_z.shape[1]), "support_count": int(tensors["support_y"].numel())},
            "required_metric_bundle": metric_keys,
            "gate_method": str(gate_info["gate_method"]),
            "unknown_score_kind": str(gate_info["unknown_score_kind"]),
            "unknown_rejection_enabled": unknown_rejection_enabled,
            "support_query_channel_view": str(config.get("target_channel_view", "clean")),
            "support_query_channel_scenarios": target_scenarios,
            "support_query_satellite_augmentation_enabled": support_query_satellite,
            "target_support_satellite_augmentation_enabled": support_query_satellite,
            "target_query_satellite_augmentation_enabled": support_query_satellite,
        },
    }
    out_dir = Path(args.run_dir)
    write_json(out_dir / "metrics.json", result)
    write_json(out_dir / "split_manifest.json", split_manifest)
    write_json(out_dir / "resolved_config.json", config)
    _write_score_table(out_dir / "score_table.csv", rows=score_rows)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="CVS-aligned Stage2 metrics for paper reproduction baselines.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--formal", action="store_true")
    parser.add_argument("--manysig-pkl", default="")
    parser.add_argument("--manytx-pkl", default="")
    parser.add_argument("--run-dir", default="runs/paper_reproduction_cvs_aligned")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    config = load_json_config(args.config)
    if args.formal and contains_unspecified(config):
        raise ValueError("formal CVS-aligned config still contains paper-unspecified")
    if args.formal and contains_unresolved_placeholder(config):
        raise ValueError("formal CVS-aligned config still contains unresolved placeholder")
    checked = validate_stage2_protocol_payload(config)
    expected_metrics = ["old_acc", "target_old_accepted_acc", "target_old_coverage"]
    if config.get("target_new_tx_labels"):
        expected_metrics.extend(["seen_new_acc", "H_old_new"])
    if bool(config.get("unknown_rejection_enabled", True)):
        expected_metrics.extend(["unknown_FAR", "FPR95", "AUROC"])
    if args.dry_run:
        print(
            json.dumps(
                {
                    "baseline": config.get("baseline"),
                    "stage": checked["stage"],
                    "cvs_extension": True,
                    "protocol": checked,
                    "expected_metrics": expected_metrics,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0
    result = _run({**config, **checked}, args)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
