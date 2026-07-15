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
import random
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
    while value in sys.path:
        sys.path.remove(value)
for value in (str(PROJECT_ROOT), str(CODE_ROOT)):
    sys.path.insert(0, value)

from cvsrffi.leo_weak_cache import (  # noqa: E402
    FORMAL_LEO_WEAK_SCENARIOS,
    PHASE2_SAMPLE_VIEW_POLICY,
    load_verified_leo_weak_cache_set,
    sha256_file,
)
from cvsrffi.stage2_predictor_bundle import (  # noqa: E402
    load_verified_stage2_predictor_bundle,
    preflight_stage2_predictor_package,
)
from cvsrffi.phase2_runtime_contract import (  # noqa: E402
    PHASE2_FULL_CONTRACT,
    validate_phase2_contract,
    validate_predictor_request,
)
from model_dual_cvsincnet import backbone_forward_compat, build_dual_model  # noqa: E402
from paper_reproduction.common.config import load_json_config  # noqa: E402
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
    "target_leo_weak_cache_root",
    "truth_sidecar",
    "scoring_manifest",
    "adv3b02_checkpoint",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def _cycle_batches(loader: DataLoader, steps: int):
    iterator = iter(loader)
    for _ in range(int(steps)):
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(loader)
            yield next(iterator)


def _trace_loss(
    trace: list[dict[str, Any]], config: dict[str, Any], *, phase: str,
    step: int, total_steps: int, losses: dict[str, torch.Tensor | float],
) -> None:
    every = max(1, int(config.get("loss_trace_every", 20)))
    if int(step) not in {1, int(total_steps)} and int(step) % every != 0:
        return
    row: dict[str, Any] = {
        "method": str(config.get("method", "")),
        "scenario": str(config.get("_active_scenario", "")),
        "phase": str(phase), "step": int(step), "total_steps": int(total_steps),
    }
    for key, value in losses.items():
        row[str(key)] = (
            float(value.detach().cpu()) if isinstance(value, torch.Tensor) else float(value)
        )
    if "loss" not in row and "total" in row:
        row["loss"] = row["total"]
    numeric = [
        float(value) for key, value in row.items()
        if key not in {"method", "scenario", "phase"}
    ]
    if not all(math.isfinite(value) for value in numeric):
        raise FloatingPointError(f"non-finite loss trace row: {row}")
    trace.append(row)
    print("[LOSS-TRACE] " + json.dumps(row, ensure_ascii=False, sort_keys=True), flush=True)


def _strip_module_prefix(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        (key[7:] if key.startswith("module.") else key): value
        for key, value in state.items()
    }


def _infer_num_domains(state: dict[str, torch.Tensor]) -> int:
    for key in (
        "dom_head.net.3.bias", "dom_head.net.3.weight",
        "adv_head.net.3.bias", "adv_head.net.3.weight",
    ):
        value = state.get(key)
        if torch.is_tensor(value) and value.ndim >= 1:
            return int(value.shape[0])
    raise ValueError("strict ADV3B02 reconstruction cannot infer num_domains from checkpoint")


def _safe_receiver(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", str(value)).strip("_")


def _tensor_from_array(value: Any, *, numpy_dtype: Any, torch_dtype: torch.dtype) -> torch.Tensor:
    array = np.ascontiguousarray(value, dtype=numpy_dtype)
    return torch.frombuffer(memoryview(array), dtype=torch_dtype).reshape(array.shape).clone()


def _target_predictor_bundle_path(config: dict[str, Any]) -> Path:
    root = Path(str(config["target_predictor_bundle_root"]))
    receiver = _safe_receiver(str(config["target_receiver_labels"][0]))
    seed = int(config["split_seed"])
    return root / f"rx_{receiver}" / f"seed_{seed}"


def _target_predictor_seal_path(config: dict[str, Any]) -> Path:
    root = Path(str(config["target_predictor_seal_root"]))
    receiver = _safe_receiver(str(config["target_receiver_labels"][0]))
    seed = int(config["split_seed"])
    return root / f"rx_{receiver}" / f"seed_{seed}" / "seal.json"


def _runtime_evidence_path(config: dict[str, Any]) -> Path:
    root = Path(str(config["phase2_runtime_isolation_evidence_root"]))
    receiver = _safe_receiver(str(config["target_receiver_labels"][0]))
    seed = int(config["split_seed"])
    return root / f"rx_{receiver}" / f"seed_{seed}" / "runtime_isolation_evidence.json"


def _exact_adv3b02(checkpoint_path: Path, *, device: torch.device) -> tuple[nn.Module, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = _strip_module_prefix(checkpoint["model"])
    checkpoint_args = dict(checkpoint.get("args") or {})
    num_domains = _infer_num_domains(state)
    sample_rate_hz = float(checkpoint_args.get("sample_rate_hz", 0.0))
    if sample_rate_hz <= 0.0:
        sample_rate_hz = 25e6
    model = build_dual_model(
        int(checkpoint_args["num_classes"]), int(num_domains),
        model_size=str(checkpoint_args.get("model_size", "M")),
        dataset=str(checkpoint_args.get("dataset", "wisig")),
        input_len=256,
        sample_rate_hz=sample_rate_hz,
        id_feature_key=str(checkpoint_args.get("id_feature_key", "feat_joint")),
        dom_feature_key=str(checkpoint_args.get("dom_feature_key", "feat_imp")),
        model_variant=str(checkpoint_args.get("model_variant", "lite_c")),
        branch_ablation=str(checkpoint_args.get("branch_ablation", "none")),
        mixstyle_on=bool(checkpoint_args.get("use_mixstyle", False)),
        mixstyle_p=float(checkpoint_args.get("mixstyle_p", 0.3)),
        mixstyle_alpha=float(checkpoint_args.get("mixstyle_alpha", 0.1)),
        mixstyle_eps=float(checkpoint_args.get("mixstyle_eps", 1e-6)),
        mixstyle_layers=str(checkpoint_args.get("mixstyle_layers", "time_down,t1")),
        mixstyle_use_domain_label=bool(
            checkpoint_args.get("mixstyle_use_domain_label", True)
        ),
        mixstyle_mix=str(checkpoint_args.get("mixstyle_mix", "crossdomain")),
        mixstyle_strength=float(checkpoint_args.get("mixstyle_strength", 1.0)),
        mixstyle_fallback=str(checkpoint_args.get("mixstyle_fallback", "random")),
        domain_branch_ablation=str(
            checkpoint_args.get("domain_branch_ablation", "same")
        ),
        domain_enhancer=str(checkpoint_args.get("domain_enhancer", "rcn_stats")),
        domain_enhancer_strength=float(
            checkpoint_args.get("domain_enhancer_strength", 0.35)
        ),
        id_time_stability_mode=str(checkpoint_args.get("id_time_stability_mode", "off")),
        id_freq_stability_mode=str(checkpoint_args.get("id_freq_stability_mode", "off")),
        domain_time_stability_mode=str(
            checkpoint_args.get("domain_time_stability_mode", "off")
        ),
        domain_freq_stability_mode=str(
            checkpoint_args.get("domain_freq_stability_mode", "off")
        ),
        time_stability_channels=int(checkpoint_args.get("time_stability_channels", 8)),
        freq_stability_channels=int(checkpoint_args.get("freq_stability_channels", 4)),
        fast_infer_when_no_aux=bool(
            checkpoint_args.get("fast_infer_when_no_aux", True)
        ),
        arch_family=str(checkpoint_args.get("arch_family", "cvsincnet")),
    ).to(device)
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
    for key in (
        "source_leo_weak_cache_set_manifest", "target_predictor_bundle_root",
        "target_predictor_seal_root",
    ):
        if not str(config.get(key, "")).strip():
            raise ValueError(f"LOCAL_PROTOCOL_REPAIR_REQUIRED: missing sealed cache field={key}")
    if not config.get("target_old_tx_labels"):
        raise ValueError("target_old_tx_labels must be nonempty")
    if not str(config.get("phase2_runtime_isolation_evidence_root", "")).strip():
        raise ValueError("LOCAL_PROTOCOL_REPAIR_REQUIRED: runtime isolation evidence root missing")
    evidence_path = _runtime_evidence_path(config)
    if not evidence_path.is_file() or evidence_path.is_symlink():
        raise ValueError("LOCAL_PROTOCOL_REPAIR_REQUIRED: runtime isolation evidence unavailable")
    evidence = json.loads(evidence_path.read_text(encoding="utf-8-sig"))
    contract = dict(config)
    contract["phase2_runtime_isolation_evidence"] = evidence
    validate_phase2_contract(contract, require_runtime_evidence=True)
    config["_verified_phase2_runtime_isolation_evidence"] = evidence


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
    iq = _tensor_from_array(
        arrays["leo_weak_iq"], numpy_dtype=np.float32, torch_dtype=torch.float32
    )
    return DataLoader(
        TensorDataset(iq, labels),
        batch_size=int(config.get("batch_size", 128)),
        shuffle=True,
        generator=torch.Generator().manual_seed(int(config["seed"])),
        drop_last=False,
    )


def _select_registered_support(arrays: dict[str, np.ndarray], config: dict[str, Any]):
    labels = np.asarray(arrays["support_pool_class_indices"], dtype=np.int64)
    k_shot = int(config["k_shot"])
    class_count = len(config["target_old_tx_labels"])
    indices: list[int] = []
    for class_index in range(class_count):
        candidates = np.flatnonzero(labels == class_index).astype(np.int64).tolist()
        if len(candidates) < k_shot:
            raise ValueError(f"support pool has fewer than K rows for registered class={class_index}")
        indices.extend(candidates[:k_shot])
    selected = np.asarray(indices, dtype=np.int64)
    iq = _tensor_from_array(
        np.asarray(arrays["support_pool_leo_weak_iq"])[selected],
        numpy_dtype=np.float32,
        torch_dtype=torch.float32,
    )
    y = _tensor_from_array(
        labels[selected], numpy_dtype=np.int64, torch_dtype=torch.int64
    ).long()
    ids = np.asarray(arrays["support_pool_tokens"]).astype(str)[selected].tolist()
    return iq, y, ids


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


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


def run(config: dict[str, Any], *, run_dir: Path, device: torch.device) -> dict[str, Any]:
    _validate_config(config)
    seed = int(config["seed"])
    set_seed(seed)
    method = str(config["method_id"])

    predictor_bundle_path = _target_predictor_bundle_path(config)
    predictor_seal_path = _target_predictor_seal_path(config)
    predictor_seal_sha = _sha256(predictor_seal_path)
    if predictor_seal_sha != config[
        "_verified_phase2_runtime_isolation_evidence"
    ]["sealed_inference_package_sha256"]:
        raise ValueError("predictor package seal/evidence digest mismatch")
    preflight_manifest, _preflight_seal, _preflight_audit = preflight_stage2_predictor_package(
        predictor_bundle_path,
        detached_seal_path=predictor_seal_path,
        expected_seal_sha256=predictor_seal_sha,
    )
    members = {item["artifact_role"]: item for item in preflight_manifest["members"]}
    def request_descriptor(item: dict[str, Any]) -> dict[str, Any]:
        return {
            "relative_path": item["relative_path"], "sha256": item["sha256"],
            "size_bytes": item["size_bytes"], "artifact_role": item["artifact_role"],
            "schema": item["schema"],
        }
    evidence = config["_verified_phase2_runtime_isolation_evidence"]
    for scenario in SCENARIOS:
        validate_predictor_request({
            "schema_version": "cvs.phase2.predict_request.v2",
            "request_id": f"{config['experiment_id']}:{scenario}",
            "row_id": str(config["experiment_id"]),
            "stage": "stage2b",
            "receiver": str(config["target_receiver_labels"][0]),
            "scenario": scenario,
            "k_shot": int(config["k_shot"]),
            "satellite_seed": int(config["split_seed"]),
            "candidate_lock_sha256": preflight_manifest["candidate_lock_sha256"],
            "package_root_sha256": preflight_manifest["package_root_sha256"],
            "runtime_code_sha256": evidence["runtime_code_sha256"],
            "registered_class_count": preflight_manifest["registered_class_count"],
            "registered_classes": preflight_manifest["registered_classes"],
            "support_artifact": request_descriptor(members[f"support:{scenario}"]),
            "query_artifact": request_descriptor(members[f"query:{scenario}"]),
            "checkpoint_artifact": request_descriptor(members["checkpoint"]),
            "adapter_artifact": request_descriptor(members["adapter"]),
            "head_artifact": request_descriptor(members["head"]),
            "tta_policy": {"mode": "single_view", "views": 1},
            "tta_policy_sha256": members["tta_policy"]["sha256"],
            "output_contract": {
                "schema": "cvs.phase2.prediction.v2",
                "relative_path": "prediction_artifact.npz",
                "sealed_immutable_required": True,
            },
            "phase2_runtime_isolation_evidence": evidence,
            **{key: config[key] for key in PHASE2_FULL_CONTRACT},
        })
    support_arrays, query_arrays, predictor_bundle_manifest, predictor_bundle_audit = (
        load_verified_stage2_predictor_bundle(
            predictor_bundle_path,
            detached_seal_path=predictor_seal_path,
            expected_seal_sha256=predictor_seal_sha,
        )
    )
    if int(predictor_bundle_manifest["registered_class_count"]) != len(
        config["target_old_tx_labels"]
    ):
        raise ValueError("predictor package registered class count does not match config")
    if predictor_bundle_manifest["receiver"] != str(config["target_receiver_labels"][0]):
        raise ValueError("predictor package receiver does not match config")
    if int(predictor_bundle_manifest["seed"]) != int(config["split_seed"]):
        raise ValueError("predictor package seed does not match config")

    source_arrays: dict[str, dict[str, np.ndarray]] | None = None
    source_cache_manifest: dict[str, Any] | None = None
    source_cache_audit: dict[str, Any] | None = None
    if method != "protonet_cda":
        source_arrays, source_cache_manifest, source_cache_audit = load_verified_leo_weak_cache_set(
            config["source_leo_weak_cache_set_manifest"],
            expected_scope="source_train",
            allowed_roles={"source"},
        )

    checkpoint_path = predictor_bundle_path / str(members["checkpoint"]["relative_path"])
    exact_model, checkpoint_info = _exact_adv3b02(checkpoint_path, device=device)
    feature_dim = int(config.get("adv3b02_feature_dim", getattr(exact_model, "emb_dim", 160)))
    template = ADV3B02MethodModel(exact_model, method=method, feature_dim=feature_dim).cpu()
    del exact_model

    prediction_rows: list[dict[str, Any]] = []
    scenario_runtime: dict[str, dict[str, Any]] = {}
    trace: list[dict[str, Any]] = []
    updates = 0
    reference_support_ids: list[str] | None = None
    reference_query_ids: list[str] | None = None
    for scenario in SCENARIOS:
        set_seed(seed)
        support_x, support_y, support_ids = _select_registered_support(
            support_arrays[scenario], config
        )
        query_input = query_arrays[scenario]
        query_x = _tensor_from_array(
            query_input["query_leo_weak_iq"],
            numpy_dtype=np.float32,
            torch_dtype=torch.float32,
        )
        query_ids = np.asarray(query_input["query_tokens"]).astype(str).tolist()
        if reference_support_ids is None:
            reference_support_ids, reference_query_ids = support_ids, query_ids
        elif support_ids != reference_support_ids or query_ids != reference_query_ids:
            raise ValueError("predictor inputs drift across LEO scenarios")
        model = copy.deepcopy(template).to(device)
        before = _predict_logits(model, query_x, device)
        started = time.perf_counter()
        if method == "protonet_cda":
            model.eval()
            with torch.no_grad():
                support_z = model(support_x.to(device)).cpu()
                query_z = model(query_x.to(device)).cpu()
            predicted = _nearest_prototype(support_z, support_y, query_z)
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
                config, model, source_loader, support_x, support_y,
                scenario=scenario, device=device,
            )
            trace.extend(scenario_trace)
            updates += int(method_info["adv3b02_gradient_updates"])
            predicted = _predict_logits(model, query_x, device)
        elapsed = time.perf_counter() - started
        scenario_runtime[scenario] = {
            "adaptation_latency_sec": elapsed,
            "latency_per_query_ms": elapsed * 1000.0 / int(len(query_ids)),
            "role_oracle_used": False,
            "true_batch_class_count_used": False,
            "class_quota_used": False,
            "query_query_graph_used": False,
            "query_batch_state_required": False,
            **method_info,
        }
        for sample_id, before_label, predicted_label in zip(
            query_ids, before.tolist(), predicted.tolist()
        ):
            prediction_rows.append({
                "sample_id": sample_id,
                "scenario": scenario,
                "before_predicted_label": int(before_label),
                "predicted_label": int(predicted_label),
            })

    manifest = validate_supervised_da_manifest({
        **config,
        "method_id": method,
        "stage": "Stage2-B",
        "cvs_extension": True,
        "target_old_support_sample_ids": reference_support_ids,
        "target_old_query_sample_ids": reference_query_ids,
        "target_labels_scope": "registered_support_only",
        "target_query_used_for_training": False,
        "target_query_used_for_model_selection": False,
        "predictor_query_truth_access": False,
        "predictor_query_role_access": False,
        "predictor_query_true_batch_class_count_access": False,
        "predictor_query_class_quota_access": False,
        "prediction_scoring_process_isolated": True,
        "scorer_output_must_not_feed_predictor": True,
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
        "target_predictor_bundle_manifest": str(predictor_bundle_path),
        "target_predictor_package_seal": str(predictor_seal_path),
        "target_predictor_package_seal_sha256": predictor_seal_sha,
        "target_predictor_bundle_contract": predictor_bundle_manifest,
        "target_predictor_bundle_audit": predictor_bundle_audit,
        "source_leo_weak_cache_used": method != "protonet_cda",
        "source_leo_weak_cache_manifest": source_cache_manifest,
        "source_leo_weak_cache_audit": source_cache_audit,
        "source_cache_declared_but_not_opened": method == "protonet_cda",
        "query_used_for_joint_decision": False,
        "query_used_for_transductive_inference": False,
        "predictor_query_truth_access": False,
        "predictor_query_role_access": False,
        "predictor_query_true_batch_class_count_access": False,
        "predictor_query_class_quota_access": False,
        "prediction_scoring_process_isolated": True,
        "scorer_output_must_not_feed_predictor": True,
        "phase2_runtime_isolation_evidence": config[
            "_verified_phase2_runtime_isolation_evidence"
        ],
        "resource_profile": (
            "frozen_backbone_prototype_comparison" if method == "protonet_cda"
            else "non_lightweight_full_backbone_da_comparison"
        ),
        "deployment_resource_claim_allowed": method == "protonet_cda",
        "claim_boundary": "Stage2-B target-old LEO_weak-only adaptation comparison",
    })

    run_dir.mkdir(parents=True, exist_ok=True)
    prediction_npz = run_dir / "prediction_artifact.npz"
    np.savez(
        prediction_npz,
        sample_ids=np.asarray([row["sample_id"] for row in prediction_rows]),
        scenarios=np.asarray([row["scenario"] for row in prediction_rows]),
        before_predicted_labels=np.asarray(
            [row["before_predicted_label"] for row in prediction_rows], dtype=np.int64
        ),
        predicted_labels=np.asarray(
            [row["predicted_label"] for row in prediction_rows], dtype=np.int64
        ),
    )
    prediction_manifest = {
        "schema": "adv3b02_stage2b_prediction_artifact_v1",
        "experiment_id": config["experiment_id"],
        "method_id": method,
        "seed": seed,
        "target_receiver_label": config["target_receiver_labels"][0],
        "k_shot": int(config["k_shot"]),
        "prediction_npz": prediction_npz.name,
        "prediction_npz_sha256": sha256_file(prediction_npz),
        "prediction_row_count": len(prediction_rows),
        "scenario_runtime": scenario_runtime,
        "predictor_input_schema": {
            "allowed": [
                "LEO query IQ", "query sample ID", "overlay provenance",
                "registered support IQ/labels", "registered class list", "sealed source LEO cache",
            ],
            "query_truth": False,
            "query_role": False,
            "query_true_batch_class_count": False,
            "query_class_quota": False,
            "query_ordering_hint": False,
        },
        "scoring_completed_inside_predictor": False,
    }
    for filename, payload in (
        ("prediction_manifest.json", prediction_manifest),
        ("split_manifest.json", manifest), ("resolved_config.json", config),
        ("loss_trace.json", trace),
    ):
        write_json(run_dir / filename, payload)
    _write_csv(run_dir / "loss_trace.csv", trace)
    return prediction_manifest


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
            "target_predictor_bundle": str(_target_predictor_bundle_path(config)),
            "config": config,
        }, ensure_ascii=False, default=str))
        return 0
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    result = run(config, run_dir=args.run_dir, device=device)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
