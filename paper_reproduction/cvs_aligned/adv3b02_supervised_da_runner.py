"""LEO_weak-only Stage2-B DA methods sharing the exact ADV3B02 backbone.

The Phase2 process accepts only sealed post-channel IQ cache sets. It exposes
no WiSig/ManySig dataset path and never constructs a clean/raw dataset or a
satellite-channel overlay. Cache construction is a separate Phase1/offline
process handled by ``code/scripts/build_cvs_leo_weak_iq_cache.py``.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "code"
for value in (str(CODE_ROOT), str(PROJECT_ROOT)):
    if value not in sys.path:
        sys.path.insert(0, value)

from SSDG import train_ssdg as ssdg_mod  # noqa: E402
from cvsrffi.leo_weak_cache import (  # noqa: E402
    FORMAL_LEO_WEAK_SCENARIOS,
    PHASE2_SAMPLE_VIEW_POLICY,
    load_verified_leo_weak_cache_set,
)
from model_dual_cvsincnet import backbone_forward_compat  # noqa: E402
from eval_feature_diagnosis import infer_num_domains, strip_module_prefix  # noqa: E402
from paper_reproduction.common.config import load_json_config  # noqa: E402
from paper_reproduction.common.wisig_runtime import set_seed, write_json  # noqa: E402
from paper_reproduction.cvs_aligned.class_incremental import (  # noqa: E402
    _cycle_batches,
    _detailed_breakdown,
    _trace_loss,
)
from paper_reproduction.cvs_aligned.supervised_da import (  # noqa: E402
    dadda_sda_objective,
    mrior_sda_batch_step,
    validate_supervised_da_manifest,
)


METHODS = {"protonet_cda", "mrior_sda", "dadda_sda"}
SCENARIOS = tuple(FORMAL_LEO_WEAK_SCENARIOS)
QUERY_POLICY = "per_sample_all_registered_classes"
PRETRAINED_POLICY = "sealed_phase1_checkpoint_only"
FORBIDDEN_CONFIG_KEYS = {
    "manysig_pkl",
    "manytx_pkl",
    "dataset_path",
    "source_dataset",
    "target_dataset",
    "source_train_channel_view",
    "train_channel_view",
    "clean_cache",
    "clean_control",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_receiver(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", str(value)).strip("_")


def _target_cache_manifest_path(config: dict[str, Any]) -> Path:
    root = Path(str(config["target_leo_weak_cache_root"]))
    receiver = _safe_receiver(str(config["target_receiver_labels"][0]))
    seed = int(config["split_seed"])
    return root / f"rx_{receiver}" / f"seed_{seed}" / "cache_set.json"


def _exact_adv3b02(checkpoint_path: Path, *, device: torch.device) -> tuple[nn.Module, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = strip_module_prefix(checkpoint["model"])
    checkpoint_args = dict(checkpoint.get("args") or {})
    num_domains = infer_num_domains(
        None, state=state, split_info={}, ckpt_args=checkpoint_args, cli_num_domains=None
    )
    parser = ssdg_mod.build_arg_parser()
    model_args = parser.parse_args(["--output_dir", str(PROJECT_ROOT / ".tmp_adv3b02_da")])
    for key, value in checkpoint_args.items():
        setattr(model_args, key, value)
    model_args.device = str(device)
    merged = ssdg_mod.merge_checkpoint_args(
        checkpoint, model_args, input_len=256, num_domains=int(num_domains)
    )
    merged = ssdg_mod._apply_model_cli_args(merged, model_args)
    model = ssdg_mod.build_baseline_model(merged, device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise ValueError(
            f"strict ADV3B02 reconstruction failed: missing={list(missing)} "
            f"unexpected={list(unexpected)}"
        )
    if not hasattr(model, "id_backbone") or not callable(getattr(model, "_pick_z_id", None)):
        raise ValueError("ADV3B02 checkpoint does not expose the identity backbone/z_id interface")
    return model, {
        "checkpoint_sha256": _sha256(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_load_strict": True,
        "checkpoint_load_audit": {"missing_keys": 0, "unexpected_keys": 0, "shape_mismatch": 0},
        "num_domains": int(num_domains),
        "checkpoint_args": json.loads(json.dumps(checkpoint_args, default=str)),
    }


class ADV3B02MethodModel(nn.Module):
    def __init__(self, exact_model: nn.Module, *, method: str, feature_dim: int) -> None:
        super().__init__()
        self.method = str(method)
        self.id_backbone = copy.deepcopy(exact_model.id_backbone)
        self.feature_key = str(exact_model.id_feature_key)
        self.estimate_network = (
            nn.Sequential(
                nn.Linear(feature_dim, feature_dim), nn.ELU(),
                nn.Linear(feature_dim, feature_dim), nn.ELU(), nn.Linear(feature_dim, 1),
            )
            if self.method == "mrior_sda" else None
        )

    def _identity(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        aux = backbone_forward_compat(
            self.id_backbone, x, y=None, return_aux=True, domain_labels=None
        )
        keys = [self.feature_key, "feat_joint", "feat_cls", "feat_con", "base"]
        features = next((aux[key] for key in keys if torch.is_tensor(aux.get(key))), None)
        if not torch.is_tensor(features) or not torch.is_tensor(aux.get("logits")):
            raise KeyError(f"ADV3B02 identity output misses z_id/logits; keys={sorted(aux)}")
        return features, aux["logits"], aux

    def forward(self, x: torch.Tensor) -> Any:
        features, logits, aux = self._identity(x)
        if self.method == "protonet_cda":
            return features
        if self.method == "mrior_sda":
            assert self.estimate_network is not None
            return {
                "features": features, "tx_logits": logits,
                "estimate_logits": self.estimate_network(features),
            }
        local_parts = [
            aux[key] for key in ("feat_cls", "feat_dac", "feat_pa", "feat_imp")
            if torch.is_tensor(aux.get(key))
        ]
        if not local_parts:
            local_parts = [features]
        return {
            "global_features": features,
            "local_features": torch.cat(local_parts, dim=1),
            "logits": logits,
        }


def _validate_config(config: dict[str, Any]) -> None:
    method = str(config.get("method_id", "")).lower()
    if method not in METHODS:
        raise ValueError(f"method_id must be one of {sorted(METHODS)}")
    if str(config.get("stage")) != "Stage2-B":
        raise ValueError("ADV3B02 supervised DA requires Stage2-B")
    if config.get("target_new_tx_labels") or config.get("target_unknown_tx_labels"):
        raise ValueError("Stage2-B permits target-old classes only")
    if len(config.get("target_receiver_labels", [])) != 1:
        raise ValueError("each run must adapt exactly one target receiver")
    if tuple(config.get("target_channel_scenarios", [])) != SCENARIOS:
        raise ValueError(f"formal scenarios must be exactly {SCENARIOS}")
    if int(config.get("k_shot", 0)) <= 0:
        raise ValueError("k_shot must be positive")
    if int(config.get("support_pool_max_k", 0)) < int(config.get("k_shot", 0)):
        raise ValueError("support_pool_max_k must cover k_shot")
    if int(config.get("query_per_tx", 0)) <= 0:
        raise ValueError("query_per_tx must be positive")
    if method != "protonet_cda" and int(config.get("adapt_steps", 0)) <= 0:
        raise ValueError("parametric DA methods require positive adapt_steps")
    expected = {
        "phase2_sample_view_policy": PHASE2_SAMPLE_VIEW_POLICY,
        "clean_sample_access": False,
        "clean_derived_signal_access": False,
        "phase2_clean_dataset_reachable": False,
        "phase2_clean_cache_reachable": False,
        "phase2_clean_control_flow_reachable": False,
        "phase2_pretrained_artifact_policy": PRETRAINED_POLICY,
        "target_channel_view": "leo_weak_only",
        "phase2_query_decision_policy": QUERY_POLICY,
        "phase2_query_role_oracle_access": False,
        "phase2_query_true_batch_class_count_access": False,
        "phase2_query_class_quota_access": False,
        "phase2_query_batch_global_assignment": False,
    }
    failed = [key for key, value in expected.items() if config.get(key) != value]
    if failed:
        raise ValueError(f"LOCAL_PROTOCOL_REPAIR_REQUIRED: invalid or missing fields={failed}")
    present_forbidden = sorted(key for key in FORBIDDEN_CONFIG_KEYS if key in config)
    if present_forbidden:
        raise ValueError(
            "LOCAL_PROTOCOL_REPAIR_REQUIRED: Phase2 config exposes raw/clean inputs: "
            f"{present_forbidden}"
        )
    for key in ("source_leo_weak_cache_set_manifest", "target_leo_weak_cache_root"):
        if not str(config.get(key, "")).strip():
            raise ValueError(f"LOCAL_PROTOCOL_REPAIR_REQUIRED: missing sealed cache field={key}")
    if not config.get("target_old_tx_labels"):
        raise ValueError("target_old_tx_labels must be nonempty")


def _nearest_prototype(support: torch.Tensor, labels: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
    class_ids = torch.unique(labels, sorted=True)
    prototypes = torch.stack([support[labels == class_id].mean(dim=0) for class_id in class_ids])
    return class_ids[torch.cdist(query.float(), prototypes.float()).argmin(dim=1)]


def _compact_labels(tx_ids: np.ndarray, class_labels: list[str]) -> torch.Tensor:
    mapping = {str(label): index for index, label in enumerate(class_labels)}
    try:
        values = [mapping[str(value)] for value in np.asarray(tx_ids).astype(str).tolist()]
    except KeyError as exc:
        raise ValueError(f"cache contains TX outside registered target-old classes: {exc}") from exc
    return torch.tensor(values, dtype=torch.long)


def _source_loader_from_cache(
    arrays: dict[str, np.ndarray], config: dict[str, Any], *, scenario: str
) -> DataLoader:
    roles = np.asarray(arrays["dataset_role"]).astype(str)
    if set(roles.tolist()) != {"source"}:
        raise ValueError(f"source cache role drift in {scenario}")
    expected_receivers = {str(value) for value in config["source_receiver_labels"]}
    observed_receivers = set(np.asarray(arrays["rx_ids"]).astype(str).tolist())
    if observed_receivers != expected_receivers:
        raise ValueError(
            f"source cache receiver drift in {scenario}: "
            f"{sorted(observed_receivers)} != {sorted(expected_receivers)}"
        )
    class_labels = [str(value) for value in config["target_old_tx_labels"]]
    labels = _compact_labels(arrays["tx_ids"], class_labels)
    iq = torch.from_numpy(np.asarray(arrays["leo_weak_iq"], dtype=np.float32))
    return DataLoader(
        TensorDataset(iq, labels),
        batch_size=int(config.get("batch_size", 128)),
        shuffle=True,
        generator=torch.Generator().manual_seed(int(config["seed"])),
        drop_last=False,
    )


def _target_split_from_cache(
    arrays: dict[str, np.ndarray], config: dict[str, Any]
) -> dict[str, Any]:
    roles = np.asarray(arrays["dataset_role"]).astype(str)
    if set(roles.tolist()) != {"target_old"}:
        raise ValueError("Stage2-B target cache must expose only target_old rows")
    receiver = str(config["target_receiver_labels"][0])
    receivers = np.asarray(arrays["rx_ids"]).astype(str)
    if set(receivers.tolist()) != {receiver}:
        raise ValueError("target cache receiver does not match the matrix row")
    tx_ids = np.asarray(arrays["tx_ids"]).astype(str)
    class_labels = [str(value) for value in config["target_old_tx_labels"]]
    if set(tx_ids.tolist()) != set(class_labels):
        raise ValueError("target cache TX set does not match registered target-old classes")
    max_k = int(config["support_pool_max_k"])
    k_shot = int(config["k_shot"])
    query_per_tx = int(config["query_per_tx"])
    support_indices: list[int] = []
    query_indices: list[int] = []
    support_labels: list[int] = []
    query_labels: list[int] = []
    for class_index, label in enumerate(class_labels):
        indices = np.flatnonzero(tx_ids == label).astype(np.int64).tolist()
        required = max_k + query_per_tx
        if len(indices) < required:
            raise ValueError(
                f"target cache has insufficient rows for TX={label}: {len(indices)}<{required}"
            )
        support_indices.extend(indices[:k_shot])
        query_indices.extend(indices[max_k : max_k + query_per_tx])
        support_labels.extend([class_index] * k_shot)
        query_labels.extend([class_index] * query_per_tx)
    sample_ids = np.asarray(arrays["sample_ids"]).astype(str)
    if set(sample_ids[support_indices].tolist()) & set(sample_ids[query_indices].tolist()):
        raise ValueError("support/query overlap in sealed target cache")
    day_ids = np.asarray(arrays["day_ids"]).astype(str)
    sig_ids = np.asarray(arrays["sig_ids"]).astype(str)
    query_meta = [
        {
            "sample_id": str(sample_ids[index]),
            "rx_label": str(receivers[index]),
            "tx_label": str(tx_ids[index]),
            "day_i": str(day_ids[index]),
            "sig_i": str(sig_ids[index]),
            "role": "target_old",
        }
        for index in query_indices
    ]
    return {
        "support_indices": np.asarray(support_indices, dtype=np.int64),
        "query_indices": np.asarray(query_indices, dtype=np.int64),
        "support_y": torch.tensor(support_labels, dtype=torch.long),
        "query_y": torch.tensor(query_labels, dtype=torch.long),
        "support_sample_ids": sample_ids[support_indices].tolist(),
        "query_sample_ids": sample_ids[query_indices].tolist(),
        "query_meta": query_meta,
    }


def _predict_logits(model: ADV3B02MethodModel, x: torch.Tensor, device: torch.device) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        _, logits, _ = model._identity(x.to(device))
        return logits.argmax(dim=1).cpu()


def _adapt(
    config: dict[str, Any], model: ADV3B02MethodModel, source_loader: DataLoader,
    support_x: torch.Tensor, support_y: torch.Tensor, *, scenario: str, device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    method = str(config["method_id"])
    steps = int(config["adapt_steps"])
    target_loader = DataLoader(
        TensorDataset(support_x, support_y),
        batch_size=min(int(config.get("target_batch_size", 64)), int(support_y.numel())),
        shuffle=True,
        generator=torch.Generator().manual_seed(int(config["seed"])),
    )
    source_batches = _cycle_batches(source_loader, steps)
    target_batches = _cycle_batches(target_loader, steps)
    trace: list[dict[str, Any]] = []
    if method == "mrior_sda":
        learning_rate = float(config.get("mrior_adapt_learning_rate", 6.0e-4))
        optimizer_ec = torch.optim.Adam(model.id_backbone.parameters(), lr=learning_rate)
        assert model.estimate_network is not None
        optimizer_t = torch.optim.Adam(model.estimate_network.parameters(), lr=learning_rate)
        optimizer = None
    else:
        learning_rate = float(config.get("dadda_adapt_learning_rate", 1.0e-4))
        optimizer = torch.optim.SGD(
            model.parameters(), lr=learning_rate,
            momentum=float(config.get("dadda_momentum", 0.9)),
            weight_decay=float(config.get("dadda_weight_decay", 5.0e-4)),
        )
    last: dict[str, float] = {}
    for step, ((source_x, source_y), (target_x, target_y)) in enumerate(
        zip(source_batches, target_batches), start=1
    ):
        source_x, source_y = source_x.to(device), source_y.to(device)
        target_x, target_y = target_x.to(device), target_y.to(device)
        if method == "mrior_sda":
            losses = mrior_sda_batch_step(
                model, source_x, source_y, target_x, target_y,
                optimizer_t=optimizer_t, optimizer_ec=optimizer_ec,
                estimate_steps=int(config.get("mrior_estimate_steps", 7)),
                target_ce_weight=float(config.get("target_ce_weight", 1.0)),
                dvkl_weight=float(config.get("dvkl_weight", 0.005)),
                mu=float(config.get("mrior_mu", 0.5)),
                class_balance_smoothing=float(config.get("class_balance_smoothing", 0.0)),
            )
        else:
            progress = float(step - 1) / float(max(1, steps - 1))
            current_lr = learning_rate / ((1.0 + 10.0 * progress) ** 0.75)
            assert optimizer is not None
            for group in optimizer.param_groups:
                group["lr"] = current_lr
            losses = dadda_sda_objective(
                model(source_x), model(target_x), source_labels=source_y,
                target_support_labels=target_y,
                target_ce_weight=float(config.get("target_ce_weight", 1.0)),
                alignment_weight=float(config.get("alignment_weight", 1.0)),
                bandwidth=config.get("bandwidth"), detach_dynamic_alpha=True,
            )
            optimizer.zero_grad(set_to_none=True)
            losses["loss"].backward()
            optimizer.step()
        last = {
            key: float(value.detach().cpu()) for key, value in losses.items()
            if isinstance(value, torch.Tensor) and value.numel() == 1
        }
        _trace_loss(
            trace, {**config, "method": method, "_active_scenario": scenario},
            phase="target_support_adaptation", step=step, total_steps=steps,
            losses={key: value for key, value in losses.items() if value.numel() == 1},
        )
    return trace, {
        "adapt_steps": steps, "final_adaptation_losses": last,
        "optimizer": "Adam_minimax" if method == "mrior_sda" else "SGD_inverse",
        "learning_rate": learning_rate,
        "adv3b02_gradient_updates": steps,
    }


def _accuracy(predicted: torch.Tensor, truth: torch.Tensor) -> float:
    return float((predicted.cpu() == truth.cpu()).float().mean())


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run(config: dict[str, Any], *, run_dir: Path, device: torch.device) -> dict[str, Any]:
    _validate_config(config)
    seed = int(config["seed"])
    set_seed(seed)
    method = str(config["method_id"])

    target_cache_path = _target_cache_manifest_path(config)
    target_arrays, target_cache_manifest, target_cache_audit = load_verified_leo_weak_cache_set(
        target_cache_path,
        expected_scope="stage2_target_old",
        allowed_roles={"target_old"},
    )
    split = _target_split_from_cache(target_arrays[SCENARIOS[0]], config)

    source_arrays: dict[str, dict[str, np.ndarray]] | None = None
    source_cache_manifest: dict[str, Any] | None = None
    source_cache_audit: dict[str, Any] | None = None
    if method != "protonet_cda":
        source_arrays, source_cache_manifest, source_cache_audit = load_verified_leo_weak_cache_set(
            config["source_leo_weak_cache_set_manifest"],
            expected_scope="source_train",
            allowed_roles={"source"},
        )

    checkpoint_path = Path(config["adv3b02_checkpoint"])
    exact_model, checkpoint_info = _exact_adv3b02(checkpoint_path, device=device)
    feature_dim = int(config.get("adv3b02_feature_dim", getattr(exact_model, "emb_dim", 160)))
    template = ADV3B02MethodModel(exact_model, method=method, feature_dim=feature_dim).cpu()
    del exact_model

    scenarios: dict[str, dict[str, Any]] = {}
    score_rows: list[dict[str, Any]] = []
    detailed: list[dict[str, Any]] = []
    trace: list[dict[str, Any]] = []
    updates = 0
    for scenario in SCENARIOS:
        set_seed(seed)
        arrays = target_arrays[scenario]
        iq = np.asarray(arrays["leo_weak_iq"], dtype=np.float32)
        support_x = torch.from_numpy(iq[split["support_indices"]])
        query_x = torch.from_numpy(iq[split["query_indices"]])
        model = copy.deepcopy(template).to(device)
        before = _predict_logits(model, query_x, device)
        started = time.perf_counter()
        if method == "protonet_cda":
            model.eval()
            with torch.no_grad():
                support_z = model(support_x.to(device)).cpu()
                query_z = model(query_x.to(device)).cpu()
            predicted = _nearest_prototype(support_z, split["support_y"], query_z)
            method_info = {
                "adapt_steps": 0, "adv3b02_gradient_updates": 0,
                "adaptation_objective": "labeled_target_support_prototype_registration",
            }
            trace.append({
                "method": method, "scenario": scenario, "phase": "support_prototype_registration",
                "step": 1, "total_steps": 1, "loss": 0.0, "gradient_updates": 0,
            })
        else:
            assert source_arrays is not None
            source_loader = _source_loader_from_cache(source_arrays[scenario], config, scenario=scenario)
            scenario_trace, method_info = _adapt(
                config, model, source_loader, support_x, split["support_y"],
                scenario=scenario, device=device,
            )
            trace.extend(scenario_trace)
            updates += int(method_info["adv3b02_gradient_updates"])
            predicted = _predict_logits(model, query_x, device)
        elapsed = time.perf_counter() - started
        after_acc = _accuracy(predicted, split["query_y"])
        before_acc = _accuracy(before, split["query_y"])
        scenarios[scenario] = {
            "target_old_accuracy": after_acc,
            "target_old_accuracy_before_adaptation": before_acc,
            "target_old_accuracy_delta": after_acc - before_acc,
            "adaptation_latency_sec": elapsed,
            "latency_per_query_ms": elapsed * 1000.0 / int(split["query_y"].numel()),
            "role_oracle_used": False,
            "equal_class_quota_used": False,
            "query_query_graph_used": False,
            "query_batch_state_required": False,
            **method_info,
        }
        detailed.extend(_detailed_breakdown(
            predicted, split["query_y"], split["query_meta"], scenario=scenario
        ))
        for meta, truth, prediction in zip(
            split["query_meta"], split["query_y"].tolist(), predicted.tolist()
        ):
            score_rows.append({
                "sample_id": meta["sample_id"], "receiver_label": meta["rx_label"],
                "transmitter_label": meta["tx_label"], "day_i": meta["day_i"],
                "sig_i": meta["sig_i"], "role": meta["role"],
                "true_label": truth, "predicted_label": prediction,
                "correct": int(truth == prediction), "scenario": scenario,
            })

    manifest = validate_supervised_da_manifest({
        **config,
        "method_id": method,
        "stage": "Stage2-B",
        "cvs_extension": True,
        "target_old_support_sample_ids": split["support_sample_ids"],
        "target_old_query_sample_ids": split["query_sample_ids"],
        "target_labels_scope": "registered_support_only",
        "target_query_used_for_training": False,
        "target_query_used_for_model_selection": False,
    })
    manifest.update({
        **checkpoint_info,
        "feature_extractor": "ADV3B02 identity backbone",
        "adv3b02_feature_dim": feature_dim,
        "adv3b02_frozen": method == "protonet_cda",
        "adv3b02_gradient_updates": updates,
        "method_architecture_claim": "ADV3B02-backbone CVS extension",
        "paper_faithful_architecture": False,
        "split_seed": int(config["split_seed"]),
        "support_query_overlap": False,
        "all_tests_satellite_augmented": True,
        "overlay_applied_before_phase2": True,
        "target_leo_weak_cache_set_manifest": str(target_cache_path),
        "target_leo_weak_cache_manifest": target_cache_manifest,
        "target_leo_weak_cache_audit": target_cache_audit,
        "source_leo_weak_cache_used": method != "protonet_cda",
        "source_leo_weak_cache_manifest": source_cache_manifest,
        "source_leo_weak_cache_audit": source_cache_audit,
        "source_cache_declared_but_not_opened": method == "protonet_cda",
        "query_used_for_joint_decision": False,
        "query_used_for_transductive_inference": False,
        "resource_profile": (
            "frozen_backbone_prototype_comparison" if method == "protonet_cda"
            else "non_lightweight_full_backbone_da_comparison"
        ),
        "deployment_resource_claim_allowed": method == "protonet_cda",
        "claim_boundary": "Stage2-B target-old LEO_weak-only adaptation comparison",
    })
    aggregate = {
        key + "_mean": float(sum(float(row[key]) for row in scenarios.values()) / len(scenarios))
        for key in (
            "target_old_accuracy", "target_old_accuracy_before_adaptation",
            "target_old_accuracy_delta", "adaptation_latency_sec",
        )
    }
    result = {
        "schema": "adv3b02_stage2b_supervised_da_v2",
        "experiment_id": config["experiment_id"],
        "method_id": method,
        "seed": seed,
        "target_receiver_label": config["target_receiver_labels"][0],
        "k_shot": int(config["k_shot"]),
        "metrics": aggregate,
        "metrics_by_scenario": scenarios,
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    for filename, payload in (
        ("metrics.json", result), ("split_manifest.json", manifest),
        ("resolved_config.json", config), ("detailed_metrics.json", detailed),
        ("loss_trace.json", trace),
    ):
        write_json(run_dir / filename, payload)
    _write_csv(run_dir / "score_table.csv", score_rows)
    _write_csv(run_dir / "detailed_metrics.csv", detailed)
    _write_csv(run_dir / "loss_trace.csv", trace)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--experiment-id", default=None)
    parser.add_argument("--method", choices=sorted(METHODS), default=None)
    parser.add_argument("--target-receiver", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--split-seed", type=int, default=None)
    parser.add_argument("--k-shot", type=int, default=None)
    parser.add_argument("--adapt-steps", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = load_json_config(args.config)
    for key, value in (
        ("experiment_id", args.experiment_id), ("method_id", args.method),
        ("seed", args.seed), ("split_seed", args.split_seed), ("k_shot", args.k_shot),
        ("adapt_steps", args.adapt_steps),
    ):
        if value is not None:
            config[key] = value
    if args.target_receiver is not None:
        config["target_receiver_labels"] = [args.target_receiver]
    _validate_config(config)
    if args.dry_run:
        print(json.dumps({
            "status": "dry_run_pass",
            "target_cache_set": str(_target_cache_manifest_path(config)),
            "config": config,
        }, ensure_ascii=False, default=str))
        return 0
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    result = run(config, run_dir=args.run_dir, device=device)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
