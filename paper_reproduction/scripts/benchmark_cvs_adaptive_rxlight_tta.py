#!/usr/bin/env python
"""Benchmark fixed and adaptive rx_light5 with one deployed ADV3B02 state.

The benchmark is deliberately non-transductive.  Registered support builds one
prototype bank per receive view and leave-one-out support scores calibrate a
single preregistered 1->3->5 early-exit gate.  Query labels are loaded only
after the gate is frozen and are used exclusively for reporting metrics.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import sys
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = REPO_ROOT / "code"
for candidate in (str(REPO_ROOT), str(CODE_ROOT)):
    while candidate in sys.path:
        sys.path.remove(candidate)
for candidate in (str(REPO_ROOT), str(CODE_ROOT)):
    sys.path.insert(0, candidate)

from cvsrffi.checkpoint_loading import build_exact_ssdg_model_from_checkpoint
from export_spaceborne_features import (
    _satellite_tta_views,
    _spectral_logmag_sketch_batch,
)
from paper_reproduction.cvs_aligned.adaptive_rxlight_tta import (
    RX_LIGHT5_ORDER,
    apply_adaptive_rxlight_tta,
    apply_adaptive_rxlight_tta_lazy,
    calibrate_adaptive_rxlight_tta,
)
from paper_reproduction.cvs_aligned.cvs_method_runner import SCENARIOS, _select_split
from paper_reproduction.cvs_aligned.extreme_light_adapter import (
    concatenate_registered_features,
)
from paper_reproduction.cvs_aligned.k1_symmetric_head import (
    SymmetricK1Head,
    fit_locked_symmetric_support_head,
    persist_and_reload_symmetric_head_fp16,
    quantize_symmetric_head_fp16,
    score_symmetric_head,
)
from paper_reproduction.scripts.train_export_cvs_micro_iq_adapter import (
    _feature_forward,
    _json_safe,
    _load_npz,
    _numpy_to_tensor_compat,
    _sha256_file,
    _tensor_to_numpy_compat,
)
from paper_reproduction.scripts.train_export_cvs_support_lora_adapter import (
    LATE_KEY_FT_TARGETS,
    inject_feat_joint_lora,
    merge_feat_joint_lora,
)
from paper_reproduction.scripts.build_cvs_stage2c_candidate_lock import (
    verify_candidate_lock,
)


BASE_MARGIN_GRID = (0.00, 0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.40)
SHIFT3_MARGIN_GRID = (0.00, 0.01, 0.02, 0.05, 0.10, 0.20, 0.30)
DISAGREEMENT_GRID = (0.0, 1.0 / 3.0, 2.0 / 3.0)


def _norm(rows: np.ndarray) -> np.ndarray:
    values = np.asarray(rows, dtype=np.float32)
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1.0e-8)


def _serializable(value: Any) -> Any:
    if is_dataclass(value):
        return _serializable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serializable(item) for item in value]
    return _json_safe(value)


def _sample_id(arrays: dict[str, np.ndarray], index: int) -> str:
    return "|".join(
        str(arrays[key][index])
        for key in ("dataset_role", "tx_ids", "rx_ids", "day_ids", "eq_ids", "sig_ids")
    )


def _ids_sha256(values: Sequence[str]) -> str:
    payload = "\n".join(str(value) for value in values).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_trusted_class_id_to_tx(path: Path) -> list[str]:
    """Load an explicit checkpoint-class mapping from a hashed JSON artifact."""

    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    candidates = [
        payload.get("class_id_to_tx"),
        payload.get("direct_adv3b02_class_id_to_tx"),
        dict(payload.get("dataset", {})).get("tx_labels"),
    ]
    for candidate in candidates:
        if isinstance(candidate, list) and candidate:
            labels = [str(value) for value in candidate]
            if len(labels) != len(set(labels)):
                raise ValueError("trusted class mapping contains duplicate TX labels")
            return labels
    raise ValueError("trusted JSON artifact does not contain class_id_to_tx")


def apply_fp16_checkpoint_delta(
    model: torch.nn.Module,
    delta_state: dict[str, torch.Tensor],
) -> dict[str, Any]:
    """Merge the exact six-tensor FP16 late-key patch into a strict checkpoint."""

    expected = {
        f"{module_name}.{suffix}"
        for module_name in LATE_KEY_FT_TARGETS
        for suffix in ("weight", "bias")
    }
    if set(delta_state) != expected:
        raise ValueError(
            "late-key delta key mismatch: "
            f"observed={sorted(delta_state)}, expected={sorted(expected)}"
        )
    parameters = dict(model.named_parameters())
    element_count = 0
    with torch.no_grad():
        for name in sorted(expected):
            if name not in parameters:
                raise ValueError(f"checkpoint model is missing delta parameter {name}")
            delta = delta_state[name].detach().cpu()
            parameter = parameters[name]
            if tuple(delta.shape) != tuple(parameter.shape):
                raise ValueError(
                    f"delta shape mismatch for {name}: {tuple(delta.shape)} != "
                    f"{tuple(parameter.shape)}"
                )
            if not bool(torch.isfinite(delta).all()):
                raise FloatingPointError(f"non-finite delta tensor: {name}")
            parameter.add_(delta.to(device=parameter.device, dtype=parameter.dtype))
            element_count += int(delta.numel())
    if element_count != 31_200:
        raise ValueError(f"late-key delta element budget drift: {element_count}")
    return {
        "format": "fp16_delta_from_strict_checkpoint",
        "tensor_count": len(expected),
        "element_count": int(element_count),
        "tensor_bytes_fp16": int(element_count * 2),
        "target_modules": list(LATE_KEY_FT_TARGETS),
        "merged_added_macs_per_query": 0,
    }


def apply_fp16_lora_state(
    model: torch.nn.Module,
    state: dict[str, torch.Tensor],
    *,
    scope: str,
    rank: int,
    alpha: float,
) -> dict[str, Any]:
    """Inject and strictly load a compact support-trained LoRA state."""

    resources = inject_feat_joint_lora(
        model, rank=int(rank), alpha=float(alpha), scope=str(scope)
    )
    trainable = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    if set(state) != set(trainable):
        raise ValueError(
            "LoRA state key mismatch: "
            f"observed={sorted(state)}, expected={sorted(trainable)}"
        )
    element_count = 0
    with torch.no_grad():
        for name, parameter in trainable.items():
            value = state[name].detach().cpu()
            if tuple(value.shape) != tuple(parameter.shape):
                raise ValueError(
                    f"LoRA state shape mismatch for {name}: "
                    f"{tuple(value.shape)} != {tuple(parameter.shape)}"
                )
            if not bool(torch.isfinite(value).all()):
                raise FloatingPointError(f"non-finite LoRA state tensor: {name}")
            parameter.copy_(value.to(device=parameter.device, dtype=parameter.dtype))
            element_count += int(value.numel())
    if element_count != int(resources["trainable_parameters"]):
        raise ValueError(
            "LoRA state element count drift: "
            f"{element_count}!={resources['trainable_parameters']}"
        )
    return {
        "format": "fp16_trainable_state",
        "scope": str(scope),
        "rank": int(rank),
        "alpha": float(alpha),
        "tensor_count": int(len(trainable)),
        "element_count": int(element_count),
        "tensor_bytes_fp16": int(element_count * 2),
        "dynamic_added_macs_per_backbone_forward": int(
            resources["adapter_macs_per_query"]
        ),
        "mergeable_into_base_linear_weights": True,
        "merged_added_macs_per_query": 0,
        "target_modules": [
            str(row["module"]) for row in resources["target_modules"]
        ],
    }


def audit_adapter_manifest(
    manifest: dict[str, Any], *, adapter_state: Path
) -> dict[str, Any]:
    """Fail closed on support/source adapter provenance before query scoring."""

    resources = dict(manifest.get("resources", {}))
    sparse_key_methods = {
        "support_only_late_key_ft_source_init_v1",
        "support_only_late_key_ft_source_init_rx_shift_pair_v1",
    }
    support_lora_methods = {"support_only_full_feature_lora_v1"}
    ground_lora_methods = {
        "ground_source_full_feature_lora_v1",
        "ground_source_effective_feature_lora_v1",
    }
    lora_methods = support_lora_methods | ground_lora_methods
    method = str(manifest.get("method", ""))
    common_checks = {
        "known_method": method in sparse_key_methods | lora_methods,
        "query_update_forbidden": manifest.get("query_update_forbidden") is True,
        "no_query_labels": manifest.get("query_labels_used_for_training") is False,
        "no_role_oracle": manifest.get("old_new_role_used_by_optimizer") is False,
        "no_class_quota": manifest.get("class_quota_used_at_inference") is False,
        "state_hash": str(manifest.get("adapter_state_sha256", ""))
        == _sha256_file(adapter_state),
    }
    if method in sparse_key_methods:
        method_checks = {
            "support_only": manifest.get("support_only") is True,
            "epoch_cap": 1 <= int(manifest.get("epochs", -1)) <= 5,
            "delta_format": manifest.get("adapter_state_format")
            == "fp16_delta_from_strict_checkpoint",
            "parameter_count": int(resources.get("trainable_parameters", -1))
            == 31_200,
            "delta_tensor_bytes": int(
                resources.get("adapter_state_bytes_fp16", -1)
            )
            == 62_400,
            "merged_added_macs": int(
                resources.get("deployment_added_macs_per_query_after_merge", -1)
            )
            == 0,
        }
    elif method in support_lora_methods:
        hyperparameters = dict(manifest.get("hyperparameters", {}))
        parameter_count = int(resources.get("trainable_parameters", -1))
        method_checks = {
            "support_only": manifest.get("support_only") is True,
            "relaxed_resource_tier": manifest.get("resource_tier")
            == "performance_relaxed",
            "epoch_cap": 1 <= int(manifest.get("epochs", -1)) <= 40,
            "state_format": manifest.get("adapter_state_format")
            == "fp16_trainable_state",
            "scope": hyperparameters.get("scope") == "full_feature",
            "parameter_count": 50_000 < parameter_count <= 100_000,
            "state_tensor_bytes": int(
                resources.get("adapter_state_bytes_fp16", -1)
            )
            == 2 * parameter_count,
            "combined_state_within_cap": resources.get(
                "combined_persistent_state_within_cap"
            )
            is True,
        }
    elif method in ground_lora_methods:
        hyperparameters = dict(manifest.get("hyperparameters", {}))
        parameter_count = int(resources.get("trainable_parameters", -1))
        validation_permissions = dict(
            manifest.get("source_validation_permissions", {})
        )
        validation_path = Path(str(manifest.get("source_validation_manifest", "")))
        expected_validation_hash = str(
            manifest.get("source_validation_manifest_sha256", "")
        )
        validation_hash_ok = (
            validation_path.is_file()
            and bool(expected_validation_hash)
            and _sha256_file(validation_path) == expected_validation_hash
        )
        validation_payload: dict[str, Any] = {}
        if validation_hash_ok:
            try:
                loaded_validation = json.loads(
                    validation_path.read_text(encoding="utf-8-sig")
                )
                if isinstance(loaded_validation, dict):
                    validation_payload = loaded_validation
            except (OSError, UnicodeError, json.JSONDecodeError):
                validation_payload = {}
        artifact_permissions = dict(validation_payload.get("permissions", {}))
        receiver_holdout = dict(validation_payload.get("receiver_holdout", {}))
        validation_gates = dict(validation_payload.get("gates", {}))
        head_lock = dict(validation_payload.get("symmetric_head_lock", {}))
        stats_meta = dict(validation_payload.get("source_feature_statistics", {}))
        stats_path = Path(str(stats_meta.get("path", "")))
        method_checks = {
            "ground_source_only": manifest.get("source_only") is True
            and manifest.get("support_only") is False,
            "target_data_not_used_for_training": manifest.get(
                "target_receiver_data_used_for_training"
            )
            is False,
            "source_validation_pass": manifest.get("source_validation_pass")
            is True,
            "source_validation_artifact_hash": validation_hash_ok,
            "source_validation_schema": validation_payload.get("schema")
            == "cvs_ground_source_lora_multiview_validation_v1",
            "source_validation_artifact_pass": validation_payload.get(
                "source_validation_pass"
            )
            is True,
            "leo_weak_only_ground_training": (
                manifest.get("clean_samples_used_for_training") is False
                and manifest.get("formal_training_view") == "leo_weak_only"
                and manifest.get("proxy_data_used_for_training") is False
                and int(manifest.get("proxy_training_rows", -1)) == 0
            )
            if method == "ground_source_effective_feature_lora_v1"
            else True,
            "leo_weak_only_source_validation": validation_payload.get(
                "clean_samples_used_for_validation"
            )
            is False
            if method == "ground_source_effective_feature_lora_v1"
            else True,
            "source_validation_adapter_hash": str(
                validation_payload.get("adapter_state_sha256", "")
            )
            == _sha256_file(adapter_state),
            "source_validation_checkpoint_hash": str(
                validation_payload.get("checkpoint_sha256", "")
            )
            == str(manifest.get("checkpoint_sha256", "")),
            "source_validation_training_manifest_hash": str(
                validation_payload.get("training_manifest_sha256", "")
            )
            == str(manifest.get("training_manifest_sha256", "")),
            "source_validation_all_gates_pass": bool(validation_gates)
            and all(bool(value) for value in validation_gates.values())
            and not validation_payload.get("failed_gates"),
            "source_validation_receiver_disjoint": receiver_holdout.get(
                "disjoint"
            )
            is True
            and not receiver_holdout.get("overlap"),
            "source_validation_no_target_support": validation_permissions.get(
                "target_support_used"
            )
            is False
            and artifact_permissions.get("target_support_used") is False,
            "source_validation_no_target_query_features": validation_permissions.get(
                "target_query_features_used"
            )
            is False
            and artifact_permissions.get("target_query_features_used") is False,
            "source_validation_no_target_query_labels": validation_permissions.get(
                "target_query_labels_used"
            )
            is False
            and artifact_permissions.get("target_query_labels_used") is False,
            "source_validation_no_role_or_quota": artifact_permissions.get(
                "old_new_role_oracle_used"
            )
            is False
            and artifact_permissions.get("class_quota_used") is False,
            "source_head_lock": head_lock.get("selection_source")
            == "disjoint_source_receiver_holdout_k1_episodes"
            and head_lock.get("support_view_policy")
            == "three_leo_weak_scenario_base_views"
            and int(head_lock.get("support_receive_views_per_physical_sample", -1))
            == 3
            and head_lock.get("target_support_used_for_selection") is False
            and head_lock.get("target_query_features_used") is False
            and head_lock.get("old_new_role_oracle_used") is False
            and head_lock.get("class_quota_used") is False,
            "source_feature_statistics": stats_meta.get("feature_kind")
            == "normalized_z_id_plus_fft96_weight2"
            and int(stats_meta.get("fft_dim", -1)) == 96
            and float(stats_meta.get("fft_weight", -1.0)) == 2.0
            and int(stats_meta.get("feature_dim", -1)) > 96
            and stats_meta.get("target_rows_used") is False
            and stats_path.is_file()
            and _sha256_file(stats_path) == str(stats_meta.get("sha256", "")),
            "epoch_cap": 1
            <= int(manifest.get("epochs", -1))
            <= (
                20
                if method == "ground_source_effective_feature_lora_v1"
                else 40
            ),
            "state_format": manifest.get("adapter_state_format")
            == "fp16_trainable_state",
            "scope": hyperparameters.get("scope")
            == (
                "effective_feature"
                if method == "ground_source_effective_feature_lora_v1"
                else "full_feature"
            ),
            "parameter_count": 0 < parameter_count <= 100_000,
            "resource_tier": (
                manifest.get("resource_tier") == "preferred"
                if parameter_count <= 50_000
                else manifest.get("resource_tier") == "performance_relaxed"
            ),
            "state_tensor_bytes": int(
                resources.get("adapter_state_bytes_fp16", -1)
            )
            == 2 * parameter_count,
            "combined_state_within_cap": resources.get(
                "combined_persistent_state_within_cap"
            )
            is True,
        }
    else:
        method_checks = {"known_method_branch": False}
    checks = {**common_checks, **method_checks}
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"invalid support adapter manifest: {failed}")
    return {
        "method": method,
        "checks": checks,
        "support_view_policy": str(manifest.get("support_view_policy", "")),
        "source_validation_pass": manifest.get("source_validation_pass"),
        "epochs": int(manifest["epochs"]),
        "optimizer_steps": int(manifest.get("runtime", {}).get("optimizer_steps", -1)),
    }


def build_view_prototypes(
    support_features: np.ndarray,
    support_labels: np.ndarray,
    classes: Sequence[str],
) -> np.ndarray:
    """Return independently normalized prototypes shaped [5,C,D]."""

    features = np.asarray(support_features, dtype=np.float32)
    labels = np.asarray(support_labels).astype(str)
    if features.ndim != 3 or features.shape[0] != 5 or features.shape[1] != len(labels):
        raise ValueError("support_features must have shape [5,N,D] aligned to labels")
    normalized = _norm(features.reshape(-1, features.shape[-1])).reshape(features.shape)
    prototypes = []
    for view_index in range(5):
        view_rows = []
        for label in classes:
            selected = normalized[view_index, labels == str(label)]
            if len(selected) < 2:
                raise ValueError(f"view prototype requires >=2 support rows for {label}")
            view_rows.append(_norm(selected.mean(axis=0, keepdims=True))[0])
        prototypes.append(np.stack(view_rows).astype(np.float32))
    # The five banks are persisted in FP16 so the adapter patch plus prototype
    # state remains below the strict 128 KiB deployment cap.  Scoring promotes
    # the stored values to FP32.
    return np.stack(prototypes).astype(np.float16)


def build_single_view_prototypes(
    support_features: np.ndarray,
    support_labels: np.ndarray,
    classes: Sequence[str],
) -> np.ndarray:
    """Identity-only single-qKNN prototypes, valid from K=1 upward."""

    features = _norm(np.asarray(support_features, dtype=np.float32))
    labels = np.asarray(support_labels).astype(str)
    if features.ndim != 2 or len(features) != len(labels):
        raise ValueError("single-view support features must have shape [N,D]")
    prototypes = []
    for label in classes:
        selected = features[labels == str(label)]
        if len(selected) < 1:
            raise ValueError(f"single-qKNN is missing support for {label}")
        prototypes.append(_norm(selected.mean(axis=0, keepdims=True))[0])
    return np.stack(prototypes).astype(np.float32)


def score_views(query_features: np.ndarray, prototypes: np.ndarray) -> np.ndarray:
    """Score matching query/prototype views and return [N,5,C]."""

    query = np.asarray(query_features, dtype=np.float32)
    banks = np.asarray(prototypes, dtype=np.float32)
    if query.ndim != 3 or query.shape[0] != 5 or banks.ndim != 3 or banks.shape[0] != 5:
        raise ValueError("query features and prototypes must each start with five views")
    if query.shape[2] != banks.shape[2]:
        raise ValueError("query/prototype feature dimensions differ")
    return np.stack(
        [_norm(query[v]) @ _norm(banks[v]).T for v in range(5)], axis=1
    ).astype(np.float32)


def leave_one_out_support_scores(
    support_features: np.ndarray,
    support_labels: np.ndarray,
    classes: Sequence[str],
) -> np.ndarray:
    """Build legal support-only calibration scores without self-prototypes."""

    features = np.asarray(support_features, dtype=np.float32)
    labels = np.asarray(support_labels).astype(str)
    if features.ndim != 3 or features.shape[0] != 5 or features.shape[1] != len(labels):
        raise ValueError("support_features must have shape [5,N,D]")
    normalized = _norm(features.reshape(-1, features.shape[-1])).reshape(features.shape)
    class_to_index = {str(label): index for index, label in enumerate(classes)}
    scores = np.empty((len(labels), 5, len(classes)), dtype=np.float32)
    for view_index in range(5):
        full_sums = {
            str(label): normalized[view_index, labels == str(label)].sum(axis=0)
            for label in classes
        }
        counts = {str(label): int(np.sum(labels == str(label))) for label in classes}
        if min(counts.values()) < 2:
            raise ValueError("leave-one-out calibration requires >=2 support rows per class")
        for row_index, row_label in enumerate(labels):
            banks = []
            for label in classes:
                label = str(label)
                total = full_sums[label]
                count = counts[label]
                if label == row_label:
                    total = total - normalized[view_index, row_index]
                    count -= 1
                banks.append(_norm((total / count)[None, :])[0])
            scores[row_index, view_index] = (
                normalized[view_index, row_index] @ np.stack(banks).T
            )
    if set(labels) - set(class_to_index):
        raise ValueError("support labels are outside the registered class list")
    return scores


@torch.no_grad()
def extract_joint_rxlight_views(
    model: torch.nn.Module,
    raw_iq: np.ndarray,
    *,
    view_names: Sequence[str],
    batch_size: int,
    device: torch.device,
    fft_dim: int = 96,
    fft_weight: float = 2.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Extract only the requested same-view ADV3B02+FFT receive features."""

    rows = _numpy_to_tensor_compat(
        raw_iq,
        numpy_dtype=np.dtype(np.float32),
        torch_dtype=torch.float32,
    )
    generated = _satellite_tta_views(rows, "rx_light5")
    names = tuple(name for name, _ in generated)
    if names != RX_LIGHT5_ORDER:
        raise ValueError(f"rx_light5 definition drift: {names} != {RX_LIGHT5_ORDER}")
    requested = tuple(str(value) for value in view_names)
    if not requested or len(set(requested)) != len(requested):
        raise ValueError("view_names must be nonempty and unique")
    if any(name not in RX_LIGHT5_ORDER for name in requested):
        raise ValueError(f"view_names are outside rx_light5: {requested}")
    generated_by_name = {name: value for name, value in generated}
    outputs: list[np.ndarray] = []
    timings: dict[str, float] = {}
    model.eval()
    for name in requested:
        view_rows = generated_by_name[name]
        started = time.perf_counter()
        primary_parts: list[np.ndarray] = []
        for start in range(0, int(view_rows.shape[0]), int(batch_size)):
            batch = view_rows[start : start + int(batch_size)].to(device)
            primary, _ = _feature_forward(model, batch)
            primary_parts.append(
                _tensor_to_numpy_compat(primary, dtype=np.dtype(np.float32))
            )
        primary_np = np.concatenate(primary_parts, axis=0)
        raw_np = view_rows.detach().cpu().numpy().astype(np.float32)
        fft = _spectral_logmag_sketch_batch(raw_np, dim=int(fft_dim))
        outputs.append(
            concatenate_registered_features(
                primary_np,
                fft,
                auxiliary_weight=float(fft_weight),
            )
        )
        timings[name] = float(time.perf_counter() - started)
    return np.stack(outputs).astype(np.float32), {
        "view_names": list(requested),
        "physical_rows": int(len(raw_iq)),
        "joint_feature_dim": int(outputs[0].shape[1]),
        "seconds_by_view": timings,
        "total_seconds": float(sum(timings.values())),
    }


def extract_joint_rxlight5(
    model: torch.nn.Module,
    raw_iq: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
    fft_dim: int = 96,
    fft_weight: float = 2.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Extract all five views for support calibration or offline upper bounds."""

    return extract_joint_rxlight_views(
        model,
        raw_iq,
        view_names=RX_LIGHT5_ORDER,
        batch_size=int(batch_size),
        device=device,
        fft_dim=int(fft_dim),
        fft_weight=float(fft_weight),
    )


@torch.no_grad()
def predict_direct_adv3b02_base_view(
    model: torch.nn.Module,
    raw_iq: np.ndarray,
    *,
    class_labels: Sequence[str],
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Strict old-head baseline: no support, adapter, FFT, or extra view."""

    rows = _numpy_to_tensor_compat(
        raw_iq,
        numpy_dtype=np.dtype(np.float32),
        torch_dtype=torch.float32,
    )
    generated = _satellite_tta_views(rows, "rx_light5")
    if tuple(name for name, _value in generated) != RX_LIGHT5_ORDER:
        raise ValueError("rx_light5 definition drift for direct ADV3B02")
    base_rows = generated[0][1]
    predictions: list[np.ndarray] = []
    logit_width: int | None = None
    started = time.perf_counter()
    model.eval()
    for start in range(0, int(base_rows.shape[0]), int(batch_size)):
        batch = base_rows[start : start + int(batch_size)].to(device)
        _features, logits = _feature_forward(model, batch)
        if logits.ndim != 2:
            raise ValueError("direct ADV3B02 logits must have shape [N,C]")
        logit_width = int(logits.shape[1])
        predictions.append(
            _tensor_to_numpy_compat(
                torch.argmax(logits, dim=1), dtype=np.dtype(np.int64)
            )
        )
    if int(logit_width or -1) != len(class_labels):
        raise ValueError(
            "direct ADV3B02 class width does not match configured old labels: "
            f"{logit_width}!={len(class_labels)}"
        )
    indices = np.concatenate(predictions).astype(np.int64)
    labels = np.asarray(class_labels, dtype=object)[indices].astype(str)
    return labels, {
        "method": "strict_direct_ADV3B02_tx_logits",
        "support_rows_used": 0,
        "adapter_used": False,
        "fft_used": False,
        "tta_view_count": 1,
        "query_batch_state_used": False,
        "class_labels": list(class_labels),
        "query_rows": int(len(labels)),
        "elapsed_seconds": float(time.perf_counter() - started),
    }


def score_named_views(
    features: np.ndarray,
    prototypes: np.ndarray,
    *,
    view_indices: Sequence[int],
) -> np.ndarray:
    """Score an explicitly requested subset of matching prototype banks."""

    rows = np.asarray(features, dtype=np.float32)
    banks = np.asarray(prototypes, dtype=np.float32)
    indices = tuple(int(value) for value in view_indices)
    if rows.ndim != 3 or len(rows) != len(indices):
        raise ValueError("features must have shape [requested_views,N,D]")
    if banks.ndim != 3 or banks.shape[0] != 5:
        raise ValueError("prototypes must have shape [5,C,D]")
    return np.stack(
        [_norm(rows[local]) @ _norm(banks[view]).T for local, view in enumerate(indices)],
        axis=1,
    ).astype(np.float32)


def order_k1_support_views(
    support_features: np.ndarray,
    support_labels: np.ndarray,
    classes: Sequence[str],
) -> np.ndarray:
    """Return requested views as ``[V,C,D]`` while enforcing one physical shot."""

    features = np.asarray(support_features, dtype=np.float32)
    labels = np.asarray(support_labels).astype(str)
    if features.ndim != 3 or features.shape[0] < 1 or features.shape[1] != len(labels):
        raise ValueError("support_features must have shape [V>=1,N,D]")
    ordered = []
    for label in classes:
        positions = np.flatnonzero(labels == str(label))
        if len(positions) != 1:
            raise ValueError(
                f"K1 symmetric head requires exactly one physical support for {label}"
            )
        ordered.append(features[:, int(positions[0]), :])
    return np.stack(ordered, axis=1).astype(np.float32)


def order_k_support_observations(
    support_features: np.ndarray,
    support_labels: np.ndarray,
    classes: Sequence[str],
    *,
    k_shot: int,
) -> np.ndarray:
    """Return role-symmetric observations as ``[V*K,C,D]`` for any locked K."""

    features = np.asarray(support_features, dtype=np.float32)
    labels = np.asarray(support_labels).astype(str)
    k = int(k_shot)
    if k < 1 or features.ndim != 3 or features.shape[1] != len(labels):
        raise ValueError("support_features must have shape [V,N,D] and K>=1")
    ordered_classes: list[np.ndarray] = []
    for label in classes:
        positions = np.flatnonzero(labels == str(label))
        if len(positions) != k:
            raise ValueError(
                f"symmetric head requires exactly K={k} physical supports for {label}"
            )
        ordered_classes.append(features[:, positions, :])
    # [C,V,K,D] -> [V*K,C,D]
    stacked = np.stack(ordered_classes, axis=0)
    return np.transpose(stacked, (1, 2, 0, 3)).reshape(
        features.shape[0] * k, len(classes), features.shape[-1]
    ).astype(np.float32)


def score_symmetric_named_views(
    features: np.ndarray, head: SymmetricK1Head
) -> np.ndarray:
    """Return symmetric-head scores as ``[N,requested_views,C]``."""

    rows = np.asarray(features, dtype=np.float32)
    if rows.ndim != 3:
        raise ValueError("features must have shape [requested_views,N,D]")
    return np.transpose(score_symmetric_head(rows, head), (1, 0, 2)).astype(
        np.float32
    )


def _split_indices(
    arrays: dict[str, np.ndarray], config: dict[str, Any], scenario: str
) -> tuple[list[int], list[int]]:
    k_shot = int(config["k_shot"])
    support_pool_max_k = int(config["support_pool_max_k"])
    if support_pool_max_k < k_shot:
        raise ValueError(
            "support_pool_max_k must cover K before query selection: "
            f"{support_pool_max_k}<{k_shot}"
        )
    common = {
        "receiver": str(config["target_receiver_labels"][0]),
        "seed": int(config["seed"]),
        "k_shot": k_shot,
        "support_pool_max_k": support_pool_max_k,
        "query_per_tx": int(config["query_per_tx"]),
        "scenario": str(scenario),
    }
    old_support, old_query = _select_split(
        arrays,
        role="target_old",
        tx_labels=[str(value) for value in config["target_old_tx_labels"]],
        **common,
    )
    new_support, new_query = _select_split(
        arrays,
        role="target_new",
        tx_labels=[str(value) for value in config["target_new_tx_labels"]],
        **common,
    )
    support = old_support + new_support
    query = old_query + new_query
    if len(support) != len(set(support)) or len(query) != len(set(query)):
        raise ValueError("support/query indices contain duplicates")
    overlap = sorted(set(support) & set(query))
    if overlap:
        raise ValueError(
            f"support/query overlap after split selection: {len(overlap)} rows"
        )
    labels = np.asarray(arrays["tx_ids"]).astype(str)
    query_per_tx = int(config["query_per_tx"])
    for label in [
        *[str(value) for value in config["target_old_tx_labels"]],
        *[str(value) for value in config["target_new_tx_labels"]],
    ]:
        support_count = int(np.sum(labels[support] == label))
        query_count = int(np.sum(labels[query] == label))
        if support_count != k_shot or query_count != query_per_tx:
            raise ValueError(
                f"split count drift for {label}: support={support_count}/{k_shot}, "
                f"query={query_count}/{query_per_tx}"
            )
    return support, query


def _metric_row(
    predictions: np.ndarray,
    truth: np.ndarray,
    roles: np.ndarray,
    old_labels: Sequence[str],
    new_labels: Sequence[str],
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    pred = np.asarray(predictions).astype(str)
    y = np.asarray(truth).astype(str)
    role_values = np.asarray(roles).astype(str)
    per_class: list[dict[str, Any]] = []
    for label in list(old_labels) + list(new_labels):
        mask = y == str(label)
        per_class.append(
            {
                "tx_label": str(label),
                "evaluation_role": "target_old" if str(label) in set(old_labels) else "target_new",
                "query_count": int(mask.sum()),
                "accuracy": float(np.mean(pred[mask] == y[mask])),
            }
        )
    old_mask = role_values == "target_old"
    new_mask = role_values == "target_new"
    old_acc = float(np.mean(pred[old_mask] == y[old_mask]))
    new_acc = float(np.mean(pred[new_mask] == y[new_mask]))
    harmonic = float(2.0 * old_acc * new_acc / max(old_acc + new_acc, 1.0e-12))
    old_rows = [row for row in per_class if row["evaluation_role"] == "target_old"]
    new_rows = [row for row in per_class if row["evaluation_role"] == "target_new"]
    return {
        "old_accuracy": old_acc,
        "min_old_class_accuracy": float(min(row["accuracy"] for row in old_rows)),
        "new_accuracy": new_acc,
        "min_new_class_accuracy": float(min(row["accuracy"] for row in new_rows)),
        "harmonic_mean": harmonic,
        "overall_accuracy": float(np.mean(pred == y)),
    }, per_class


def _reference_parity(
    reference_path: Path | None,
    arrays: dict[str, np.ndarray],
    selected_indices: Sequence[int],
    generated_base: np.ndarray,
) -> dict[str, Any]:
    if reference_path is None:
        return {"checked": False}
    # Adapted feature exports intentionally omit raw_iq; the training loader
    # enforces raw_iq and therefore cannot be reused for this read-only parity
    # cache.  Keep this loader narrow and reject pickle/object arrays.
    with np.load(reference_path, allow_pickle=False) as payload:
        reference = {key: payload[key] for key in payload.files}
    required = {
        "features",
        "fft_logmag_features",
        "dataset_role",
        "tx_ids",
        "rx_ids",
        "day_ids",
        "eq_ids",
        "sig_ids",
    }
    if not required.issubset(reference):
        raise KeyError(
            f"reference feature cache is missing keys: {sorted(required - set(reference))}"
        )
    expected_ids = [_sample_id(arrays, index) for index in selected_indices]
    reference_lookup = {_sample_id(reference, index): index for index in range(len(reference["tx_ids"]))}
    positions = [reference_lookup[value] for value in expected_ids]
    expected = concatenate_registered_features(
        reference["features"][positions],
        reference["fft_logmag_features"][positions],
        auxiliary_weight=2.0,
    )
    cosine = np.sum(_norm(generated_base) * _norm(expected), axis=1)
    return {
        "checked": True,
        "reference_path": str(reference_path),
        "row_count": int(len(positions)),
        "mean_cosine": float(np.mean(cosine)),
        "min_cosine": float(np.min(cosine)),
        "mean_absolute_difference": float(np.mean(np.abs(generated_base - expected))),
        "max_absolute_difference": float(np.max(np.abs(generated_base - expected))),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Raw-IQ protocol config")
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--adapter_state", type=Path, required=True)
    parser.add_argument("--adapter_manifest", type=Path, default=None)
    parser.add_argument("--candidate_lock", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--reference_config", type=Path, default=None)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--max_accuracy_drop_pp", type=float, default=1.0)
    parser.add_argument(
        "--head_mode",
        choices=("matching_view", "k1_symmetric", "symmetric_locked"),
        default="symmetric_locked",
    )
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    if not 1 <= int(args.batch_size) <= 4096:
        raise ValueError("batch_size must be in [1,4096]")
    if not 0.0 <= float(args.max_accuracy_drop_pp) <= 3.0:
        raise ValueError("max_accuracy_drop_pp must be in [0,3]")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = json.loads(args.config.read_text(encoding="utf-8-sig"))
    k_shot = int(config["k_shot"])
    if k_shot not in (1, 5, 10, 20):
        raise ValueError("formal Stage2-C requires K in {1,5,10,20}")
    if str(args.head_mode) == "k1_symmetric" and k_shot != 1:
        raise ValueError("legacy k1_symmetric diagnostic is restricted to K1")
    reference_config = None
    if args.reference_config is not None:
        reference_config = json.loads(args.reference_config.read_text(encoding="utf-8-sig"))
    device = torch.device(str(args.device) if torch.cuda.is_available() else "cpu")
    # Project checkpoints contain the trusted SatViewStage enum in addition to tensors.
    checkpoint = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model, checkpoint_audit = build_exact_ssdg_model_from_checkpoint(
        checkpoint,
        input_len=int(config.get("raw_iq_input_len", 4800)),
        device=device,
    )
    direct_model = copy.deepcopy(model).to(device).eval()
    adapter_state = torch.load(args.adapter_state, map_location="cpu")
    if not isinstance(adapter_state, dict):
        raise TypeError("adapter_state must be a tensor dictionary")
    adapter_manifest = None
    adapter_manifest_audit = None
    source_validation_result = None
    source_feature_mean: np.ndarray | None = None
    source_feature_std: np.ndarray | None = None
    if args.adapter_manifest is not None:
        adapter_manifest = json.loads(
            args.adapter_manifest.read_text(encoding="utf-8-sig")
        )
        adapter_manifest_audit = audit_adapter_manifest(
            adapter_manifest, adapter_state=args.adapter_state
        )
        if str(adapter_manifest.get("method", "")) in {
            "ground_source_full_feature_lora_v1",
            "ground_source_effective_feature_lora_v1",
        }:
            source_validation_path = Path(
                str(adapter_manifest["source_validation_manifest"])
            )
            source_validation_result = json.loads(
                source_validation_path.read_text(encoding="utf-8-sig")
            )
            if source_validation_result.get("source_validation_pass") is not True:
                raise ValueError("ground source validation result is not PASS")
            stats_meta = dict(
                source_validation_result.get("source_feature_statistics", {})
            )
            stats_path = Path(str(stats_meta.get("path", "")))
            if (
                not stats_path.is_file()
                or _sha256_file(stats_path) != str(stats_meta.get("sha256", ""))
                or stats_meta.get("target_rows_used") is not False
                or stats_meta.get("feature_kind")
                != "normalized_z_id_plus_fft96_weight2"
                or int(stats_meta.get("fft_dim", -1)) != 96
                or float(stats_meta.get("fft_weight", -1.0)) != 2.0
            ):
                raise ValueError("source feature-statistics provenance is invalid")
            with np.load(stats_path, allow_pickle=False) as stats_payload:
                source_feature_mean = np.asarray(
                    stats_payload["mean"], dtype=np.float32
                ).reshape(-1)
                source_feature_std = np.asarray(
                    stats_payload["std"], dtype=np.float32
                ).reshape(-1)
            if (
                source_feature_mean.shape != source_feature_std.shape
                or not np.isfinite(source_feature_mean).all()
                or not np.isfinite(source_feature_std).all()
            ):
                raise ValueError("source feature statistics are malformed")
    if args.adapter_manifest is None:
        raise ValueError("formal candidate lock requires an adapter promotion manifest")
    candidate_lock = verify_candidate_lock(
        args.candidate_lock,
        checkpoint=args.ckpt,
        adapter_state=args.adapter_state,
        promotion_manifest=args.adapter_manifest,
        config=config,
    )
    merge_audit: dict[str, Any] | None = None
    if (
        adapter_manifest is not None
        and str(adapter_manifest.get("method", ""))
        in {
            "support_only_full_feature_lora_v1",
            "ground_source_full_feature_lora_v1",
            "ground_source_effective_feature_lora_v1",
        }
    ):
        hyperparameters = dict(adapter_manifest.get("hyperparameters", {}))
        delta_audit = apply_fp16_lora_state(
            model,
            adapter_state,
            scope=str(hyperparameters["scope"]),
            rank=int(hyperparameters["rank"]),
            alpha=float(hyperparameters["alpha"]),
        )
        merge_audit = merge_feat_joint_lora(model)
    else:
        delta_audit = apply_fp16_checkpoint_delta(model, adapter_state)
    model.to(device).eval()
    if (
        str(args.head_mode) in {"k1_symmetric", "symmetric_locked"}
        and source_validation_result is None
    ):
        raise ValueError(
            "symmetric deployment requires a source-validated ground LoRA and head lock"
        )

    old_labels = [str(value) for value in config["target_old_tx_labels"]]
    new_labels = [str(value) for value in config["target_new_tx_labels"]]
    old_classes = sorted(old_labels)
    direct_class_labels = [
        str(value) for value in config.get("direct_adv3b02_class_id_to_tx", [])
    ]
    if (
        len(direct_class_labels) != len(old_labels)
        or len(set(direct_class_labels)) != len(direct_class_labels)
        or set(direct_class_labels) != set(old_labels)
    ):
        raise ValueError(
            "config must provide trusted direct_adv3b02_class_id_to_tx aligned "
            "to checkpoint logit indices; sorting old labels is forbidden"
        )
    direct_mapping_source = str(
        config.get("direct_adv3b02_class_mapping_source", "")
    ).strip()
    direct_mapping_sha256 = str(
        config.get("direct_adv3b02_class_mapping_sha256", "")
    ).strip()
    direct_mapping_path = Path(direct_mapping_source)
    if (
        not direct_mapping_source
        or len(direct_mapping_sha256) != 64
        or not direct_mapping_path.is_file()
        or _sha256_file(direct_mapping_path) != direct_mapping_sha256
    ):
        raise ValueError(
            "direct ADV3B02 class mapping requires an existing trusted source "
            "artifact with matching SHA256"
        )
    if load_trusted_class_id_to_tx(direct_mapping_path) != direct_class_labels:
        raise ValueError(
            "configured direct ADV3B02 class mapping differs from trusted artifact"
        )
    classes = sorted(old_labels + new_labels)
    class_to_index = {label: index for index, label in enumerate(classes)}
    scenario_payloads: dict[str, dict[str, Any]] = {}
    calibration_scores: list[np.ndarray] = []
    calibration_labels: list[np.ndarray] = []
    old_calibration_scores: list[np.ndarray] = []
    old_calibration_labels: list[np.ndarray] = []
    symmetric_support_blocks: list[np.ndarray] = []
    protocol_audit: dict[str, Any] = {}
    extraction_audit: dict[str, Any] = {}
    parity_audit: dict[str, Any] = {}
    for scenario in SCENARIOS:
        raw_path = Path(config["feature_npz_by_scenario"][scenario])
        arrays, raw_manifest = _load_npz(raw_path)
        if "raw_iq" not in arrays:
            raise ValueError(f"raw-IQ cache is missing raw_iq: {raw_path}")
        support_idx, query_idx = _split_indices(arrays, config, scenario)
        selected_idx = support_idx + query_idx
        if "sat_scenarios" not in arrays or "channel_views" not in arrays:
            raise ValueError("formal support/query cache lacks LEO view provenance")
        selected_scenarios = arrays["sat_scenarios"][selected_idx].astype(str)
        selected_channel_views = arrays["channel_views"][selected_idx].astype(str)
        if not np.all(selected_scenarios == str(scenario)):
            raise ValueError("support/query rows are not from the configured leo_weak scenario")
        if any("clean" in value.lower() for value in selected_channel_views):
            raise ValueError("clean support/query rows are forbidden in formal Stage2-C")
        if str(args.head_mode) in {"k1_symmetric", "symmetric_locked"}:
            support_features, support_extract = extract_joint_rxlight_views(
                model,
                arrays["raw_iq"][support_idx],
                view_names=(RX_LIGHT5_ORDER[0],),
                batch_size=int(args.batch_size),
                device=device,
            )
        else:
            support_features, support_extract = extract_joint_rxlight5(
                model,
                arrays["raw_iq"][support_idx],
                batch_size=int(args.batch_size),
                device=device,
            )
        support_y = arrays["tx_ids"][support_idx].astype(str)
        truth = arrays["tx_ids"][query_idx].astype(str)
        roles = arrays["dataset_role"][query_idx].astype(str)
        support_ids = [_sample_id(arrays, i) for i in support_idx]
        query_ids = [_sample_id(arrays, i) for i in query_idx]
        identity_support_features, identity_support_extract = extract_joint_rxlight_views(
            direct_model,
            arrays["raw_iq"][support_idx],
            view_names=(RX_LIGHT5_ORDER[0],),
            batch_size=int(args.batch_size),
            device=device,
        )
        identity_prototypes = build_single_view_prototypes(
            identity_support_features[0], support_y, classes
        )
        identity_old_mask = np.isin(support_y, np.asarray(old_classes))
        identity_old_prototypes = build_single_view_prototypes(
            identity_support_features[0, identity_old_mask],
            support_y[identity_old_mask],
            old_classes,
        )
        if str(args.head_mode) == "matching_view":
            prototypes = build_view_prototypes(support_features, support_y, classes)
            loo_scores = leave_one_out_support_scores(
                support_features, support_y, classes
            )
            calibration_scores.append(loo_scores)
            calibration_labels.append(
                np.asarray(
                    [class_to_index[label] for label in support_y], dtype=np.int64
                )
            )
            old_support_mask = np.isin(support_y, np.asarray(old_classes))
            old_prototypes = build_view_prototypes(
                support_features[:, old_support_mask],
                support_y[old_support_mask],
                old_classes,
            )
            old_loo_scores = leave_one_out_support_scores(
                support_features[:, old_support_mask],
                support_y[old_support_mask],
                old_classes,
            )
            old_class_to_index = {
                label: index for index, label in enumerate(old_classes)
            }
            old_calibration_scores.append(old_loo_scores)
            old_calibration_labels.append(
                np.asarray(
                    [old_class_to_index[label] for label in support_y[old_support_mask]],
                    dtype=np.int64,
                )
            )
        else:
            prototypes = None
            old_prototypes = None
            ordered_support = order_k_support_observations(
                support_features, support_y, classes, k_shot=k_shot
            )
            # One base receive view from each of the three formal leo_weak
            # scenarios: exactly three enrollment observations per physical K1
            # sample, within the project support-view cap.
            symmetric_support_blocks.append(ordered_support)
        protocol_audit[scenario] = {
            "raw_cache": str(raw_path),
            "raw_cache_sha256": _sha256_file(raw_path),
            "raw_cache_manifest": raw_manifest,
            "support_count": int(len(support_idx)),
            "query_count": int(len(query_idx)),
            "support_ids": support_ids,
            "query_ids": query_ids,
            "support_ids_sha256": _ids_sha256(support_ids),
            "query_ids_sha256": _ids_sha256(query_ids),
            "support_query_overlap": int(len(set(support_idx) & set(query_idx))),
            "clean_support_query_rows": 0,
            "support_query_view": "leo_weak_only",
            "sat_scenario_values": sorted(set(selected_scenarios.tolist())),
            "channel_view_values": sorted(set(selected_channel_views.tolist())),
            "support_roles": sorted(set(arrays["dataset_role"][support_idx].astype(str).tolist())),
            "query_roles": sorted(set(roles.tolist())),
        }
        scenario_payloads[scenario] = {
            "arrays": arrays,
            "selected_idx": selected_idx,
            "support_features": support_features,
            "support_extract": support_extract,
            "identity_support_extract": identity_support_extract,
            "identity_prototypes": identity_prototypes,
            "identity_old_prototypes": identity_old_prototypes,
            "prototypes": prototypes,
            "old_prototypes": old_prototypes,
            "query_raw_iq": arrays["raw_iq"][query_idx],
            "truth": truth,
            "roles": roles,
            "query_ids": query_ids,
            "reference_path": (
                Path(reference_config["feature_npz_by_scenario"][scenario])
                if reference_config is not None
                else None
            ),
        }

    reference_scenario = str(SCENARIOS[0])
    reference_support_ids = protocol_audit[reference_scenario]["support_ids"]
    reference_query_ids = protocol_audit[reference_scenario]["query_ids"]
    for scenario in SCENARIOS[1:]:
        if protocol_audit[str(scenario)]["support_ids"] != reference_support_ids:
            raise ValueError("physical support IDs drift across leo_weak scenarios")
        if protocol_audit[str(scenario)]["query_ids"] != reference_query_ids:
            raise ValueError("physical query IDs drift across leo_weak scenarios")

    k1_head: SymmetricK1Head | None = None
    k1_old_head: SymmetricK1Head | None = None
    head_state_path: Path | None = None
    head_state_reload_audit: dict[str, Any] | None = None
    old_calibration: dict[str, Any] | None = None
    if str(args.head_mode) in {"k1_symmetric", "symmetric_locked"}:
        if len(symmetric_support_blocks) != len(SCENARIOS):
            raise RuntimeError("symmetric support-view scenario count drift")
        all_support_views = np.concatenate(symmetric_support_blocks, axis=0)
        head_lock = dict(source_validation_result.get("symmetric_head_lock", {}))
        if (
            head_lock.get("selection_source")
            != "disjoint_source_receiver_holdout_k1_episodes"
            or k_shot not in [int(value) for value in head_lock.get("allowed_k", [])]
            or head_lock.get("target_support_used_for_selection") is not False
            or head_lock.get("target_query_features_used") is not False
        ):
            raise ValueError("source symmetric-head lock is missing or invalid")
        locked_selected = dict(head_lock["selected"])
        k1_head = quantize_symmetric_head_fp16(
            fit_locked_symmetric_support_head(
                all_support_views,
                physical_shots_per_class=k_shot,
                selected=locked_selected,
                source_mean=source_feature_mean,
                source_std=source_feature_std,
            )
        )
        old_positions = np.asarray(
            [class_to_index[str(label)] for label in old_labels], dtype=np.int64
        )
        old_support_views = all_support_views[:, old_positions, :]
        k1_old_head = quantize_symmetric_head_fp16(
            fit_locked_symmetric_support_head(
                old_support_views,
                physical_shots_per_class=k_shot,
                selected=locked_selected,
                source_mean=source_feature_mean,
                source_std=source_feature_std,
            )
        )
        head_state_path = args.out_dir / "symmetric_locked_head_state_fp16.npz"
        k1_head, head_state_reload_audit = persist_and_reload_symmetric_head_fp16(
            k1_head, head_state_path
        )
        calibration = dict(source_validation_result["calibration"])
        old_calibration = calibration
    else:
        if source_validation_result is not None:
            calibration = dict(source_validation_result["calibration"])
            old_calibration = calibration
        else:
            calibration = calibrate_adaptive_rxlight_tta(
                np.concatenate(calibration_scores, axis=0),
                np.concatenate(calibration_labels, axis=0),
                base_margin_grid=BASE_MARGIN_GRID,
                shift3_margin_grid=SHIFT3_MARGIN_GRID,
                disagreement_grid=DISAGREEMENT_GRID,
                max_accuracy_drop_pp=float(args.max_accuracy_drop_pp),
            )
            # Legacy support-adapter diagnostic only.  Formal ground-source
            # candidates above always reuse one source-locked threshold tuple
            # before and after class registration.
            old_calibration = calibration
    thresholds = calibration["selected"]["thresholds"]
    for scenario, payload in scenario_payloads.items():
        prototypes = payload["prototypes"]
        raw_query = payload["query_raw_iq"]
        direct_old_mask = payload["roles"] == "target_old"
        direct_old_prediction, direct_audit = predict_direct_adv3b02_base_view(
            direct_model,
            raw_query[direct_old_mask],
            class_labels=direct_class_labels,
            batch_size=int(args.batch_size),
            device=device,
        )
        direct_full_prediction = np.full(len(raw_query), "", dtype=object)
        direct_full_prediction[direct_old_mask] = direct_old_prediction
        payload["direct_adv3b02_prediction"] = direct_full_prediction.astype(str)
        payload["direct_adv3b02_audit"] = direct_audit
        base_features, base_extract = extract_joint_rxlight_views(
            model,
            raw_query,
            view_names=(RX_LIGHT5_ORDER[0],),
            batch_size=int(args.batch_size),
            device=device,
        )
        identity_query_features, identity_query_extract = extract_joint_rxlight_views(
            direct_model,
            raw_query,
            view_names=(RX_LIGHT5_ORDER[0],),
            batch_size=int(args.batch_size),
            device=device,
        )
        payload["identity_single_qknn_predictions"] = np.asarray(classes, dtype=object)[
            np.argmax(
                _norm(identity_query_features[0])
                @ _norm(payload["identity_prototypes"]).T,
                axis=1,
            )
        ].astype(str)
        identity_old_mask = payload["roles"] == "target_old"
        payload["identity_single_qknn_old_before_predictions"] = np.asarray(
            old_classes, dtype=object
        )[
            np.argmax(
                _norm(identity_query_features[0, identity_old_mask])
                @ _norm(payload["identity_old_prototypes"]).T,
                axis=1,
            )
        ].astype(str)
        if k1_head is None:
            base_scores = score_named_views(
                base_features, prototypes, view_indices=(0,)
            )[:, 0, :]
        else:
            base_scores = score_symmetric_named_views(
                base_features, k1_head
            )[:, 0, :]
        lazy_extract: dict[str, Any] = {"base": base_extract}

        def shift_provider(indices: np.ndarray) -> np.ndarray:
            features, audit = extract_joint_rxlight_views(
                model,
                raw_query[indices],
                view_names=RX_LIGHT5_ORDER[1:3],
                batch_size=int(args.batch_size),
                device=device,
            )
            lazy_extract["shift_pair"] = audit
            if k1_head is None:
                return score_named_views(
                    features, prototypes, view_indices=(1, 2)
                )
            return score_symmetric_named_views(features, k1_head)

        def cfo_provider(indices: np.ndarray) -> np.ndarray:
            features, audit = extract_joint_rxlight_views(
                model,
                raw_query[indices],
                view_names=RX_LIGHT5_ORDER[3:5],
                batch_size=int(args.batch_size),
                device=device,
            )
            lazy_extract["cfo_pair"] = audit
            if k1_head is None:
                return score_named_views(
                    features, prototypes, view_indices=(3, 4)
                )
            return score_symmetric_named_views(features, k1_head)

        adaptive = apply_adaptive_rxlight_tta_lazy(
            base_scores, shift_provider, cfo_provider, thresholds
        )
        full_query_features, full_extract = extract_joint_rxlight5(
            model,
            raw_query,
            batch_size=int(args.batch_size),
            device=device,
        )
        query_scores = (
            score_views(full_query_features, prototypes)
            if k1_head is None
            else score_symmetric_named_views(full_query_features, k1_head)
        )
        eager_check = apply_adaptive_rxlight_tta(query_scores, thresholds)
        if not np.array_equal(
            adaptive["view_budgets"], eager_check["view_budgets"]
        ) or not np.array_equal(adaptive["predictions"], eager_check["predictions"]):
            raise ValueError(
                f"lazy/eager adaptive TTA parity failed for {scenario}"
            )
        generated_base = np.concatenate(
            [payload["support_features"][0], full_query_features[0]], axis=0
        )
        parity_audit[scenario] = _reference_parity(
            payload["reference_path"],
            payload["arrays"],
            payload["selected_idx"],
            generated_base,
        )
        if parity_audit[scenario]["checked"] and (
            float(parity_audit[scenario]["mean_cosine"]) < 0.9999
            or float(parity_audit[scenario]["min_cosine"]) < 0.999
        ):
            raise ValueError(
                "deployed adapter base-view extraction does not reproduce "
                f"the registered adapted cache for {scenario}: "
                f"{parity_audit[scenario]}"
            )
        query_count = int(len(raw_query))
        shift_count = int(adaptive["shift_rows_requested"])
        cfo_count = int(adaptive["cfo_rows_requested"])
        support_forward_rows = int(
            payload["support_extract"]["physical_rows"]
            * len(payload["support_extract"]["view_names"])
        )
        deployed_query_forward_rows = int(
            query_count + 2 * shift_count + 2 * cfo_count
        )
        offline_fixed5_forward_rows = int(5 * query_count)
        direct_baseline_forward_rows = int(np.sum(direct_old_mask))
        identity_support_forward_rows = int(
            payload["identity_support_extract"]["physical_rows"]
            * len(payload["identity_support_extract"]["view_names"])
        )
        identity_query_forward_rows = int(
            identity_query_extract["physical_rows"]
            * len(identity_query_extract["view_names"])
        )
        extraction_audit[scenario] = {
            "support_enrollment_extraction": payload["support_extract"],
            "identity_single_qknn_support_extraction": payload[
                "identity_support_extract"
            ],
            "identity_single_qknn_query_extraction": identity_query_extract,
            "deployed_adaptive_lazy": {
                **lazy_extract,
                "query_rows": query_count,
                "shift_rows_requested": shift_count,
                "cfo_rows_requested": cfo_count,
                "actual_backbone_forward_rows": deployed_query_forward_rows,
                "mean_backbone_forwards": float(
                    adaptive["mean_backbone_forwards"]
                ),
            },
            "offline_fixed5_upper_bound_only": full_extract,
            "resource_accounting": {
                "deployment_one_time_support_forward_rows": support_forward_rows,
                "deployment_query_forward_rows": deployed_query_forward_rows,
                "benchmark_offline_fixed5_forward_rows": offline_fixed5_forward_rows,
                "benchmark_direct_baseline_forward_rows": direct_baseline_forward_rows,
                "benchmark_identity_support_forward_rows": identity_support_forward_rows,
                "benchmark_identity_query_forward_rows": identity_query_forward_rows,
                "benchmark_total_executed_forward_rows": int(
                    support_forward_rows
                    + deployed_query_forward_rows
                    + offline_fixed5_forward_rows
                    + direct_baseline_forward_rows
                    + identity_support_forward_rows
                    + identity_query_forward_rows
                ),
                "deployment_logical_and_benchmark_total_are_separate": True,
            },
            "lazy_eager_prediction_parity": True,
        }
        payload["scores"] = query_scores
        payload["adaptive"] = adaptive
        if old_calibration is not None:
            old_mask = payload["roles"] == "target_old"
            if k1_old_head is not None:
                old_scores = score_symmetric_named_views(
                    full_query_features[:, old_mask, :], k1_old_head
                )
                old_label_order = old_labels
            else:
                old_scores = score_views(
                    full_query_features[:, old_mask, :],
                    payload["old_prototypes"],
                )
                old_label_order = old_classes
            old_adaptive = apply_adaptive_rxlight_tta(
                old_scores, old_calibration["selected"]["thresholds"]
            )
            payload["old_before_label_order"] = list(old_label_order)
            payload["old_before_predictions"] = {
                "fixed1": np.argmax(old_scores[:, 0], axis=1),
                "fixed3": np.argmax(old_scores[:, :3].mean(axis=1), axis=1),
                "fixed5": np.argmax(old_scores.mean(axis=1), axis=1),
                "adaptive1to3to5": old_adaptive["predictions"],
            }

    rows: list[dict[str, Any]] = []
    per_class_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    aggregate: dict[str, dict[str, list[np.ndarray]]] = {
        name: {
            "pred": [],
            "truth": [],
            "roles": [],
            "old_before": [],
            "direct_old": [],
            "identity_after": [],
            "identity_before": [],
        }
        for name in ("fixed1", "fixed3", "fixed5", "adaptive1to3to5")
    }
    for scenario, payload in scenario_payloads.items():
        scores = payload["scores"]
        truth = payload["truth"]
        roles = payload["roles"]
        fixed_indices = {
            "fixed1": np.argmax(scores[:, 0], axis=1),
            "fixed3": np.argmax(scores[:, :3].mean(axis=1), axis=1),
            "fixed5": np.argmax(scores.mean(axis=1), axis=1),
        }
        adaptive = payload["adaptive"]
        fixed_indices["adaptive1to3to5"] = adaptive["predictions"]
        for method_name, indices in fixed_indices.items():
            predicted = np.asarray(classes, dtype=object)[indices].astype(str)
            metrics, class_rows = _metric_row(
                predicted, truth, roles, old_labels, new_labels
            )
            old_mask = roles == "target_old"
            direct_old = payload["direct_adv3b02_prediction"][old_mask]
            direct_old_accuracy = float(np.mean(direct_old == truth[old_mask]))
            metrics["direct_adv3b02_old_accuracy"] = direct_old_accuracy
            metrics["delta_vs_direct_adv3b02"] = float(
                metrics["old_accuracy"] - direct_old_accuracy
            )
            identity_after = payload["identity_single_qknn_predictions"][old_mask]
            identity_before = payload[
                "identity_single_qknn_old_before_predictions"
            ]
            identity_after_acc = float(np.mean(identity_after == truth[old_mask]))
            identity_before_acc = float(np.mean(identity_before == truth[old_mask]))
            metrics["identity_old_acc_before_increment"] = identity_before_acc
            metrics["identity_old_acc_after_increment"] = identity_after_acc
            metrics["identity_average_forgetting"] = float(
                identity_before_acc - identity_after_acc
            )
            if "old_before_predictions" in payload:
                old_before = np.asarray(
                    payload["old_before_label_order"], dtype=object
                )[
                    payload["old_before_predictions"][method_name]
                ].astype(str)
                old_before_accuracy = float(
                    np.mean(old_before == truth[old_mask])
                )
                metrics["old_accuracy_before_increment"] = old_before_accuracy
                metrics["old_adaptation_gain"] = float(
                    metrics["old_accuracy"] - old_before_accuracy
                )
                metrics["average_forgetting"] = float(
                    old_before_accuracy - metrics["old_accuracy"]
                )
            if method_name == "fixed1":
                resources = {"mean_backbone_forwards": 1.0, "p95_backbone_forwards": 1.0,
                             "view1_rate": 1.0, "view3_rate": 0.0, "view5_rate": 0.0}
            elif method_name == "fixed3":
                resources = {"mean_backbone_forwards": 3.0, "p95_backbone_forwards": 3.0,
                             "view1_rate": 0.0, "view3_rate": 1.0, "view5_rate": 0.0}
            elif method_name == "fixed5":
                resources = {"mean_backbone_forwards": 5.0, "p95_backbone_forwards": 5.0,
                             "view1_rate": 0.0, "view3_rate": 0.0, "view5_rate": 1.0}
            else:
                resources = {
                    "mean_backbone_forwards": float(adaptive["mean_backbone_forwards"]),
                    "p95_backbone_forwards": float(adaptive["p95_backbone_forwards"]),
                    **adaptive["trigger_rates"],
                }
            rows.append({"scenario": scenario, "method": method_name, **metrics, **resources})
            per_class_rows.extend(
                {"scenario": scenario, "method": method_name, **row}
                for row in class_rows
            )
            aggregate[method_name]["pred"].append(predicted)
            aggregate[method_name]["truth"].append(truth)
            aggregate[method_name]["roles"].append(roles)
            aggregate[method_name]["direct_old"].append(direct_old)
            aggregate[method_name]["identity_after"].append(identity_after)
            aggregate[method_name]["identity_before"].append(identity_before)
            if "old_before_predictions" in payload:
                aggregate[method_name]["old_before"].append(old_before)
            budgets = (
                adaptive["view_budgets"]
                if method_name == "adaptive1to3to5"
                else np.full(len(truth), int(method_name[-1]), dtype=np.int64)
            )
            prediction_rows.extend(
                {
                    "scenario": scenario,
                    "method": method_name,
                    "query_id": payload["query_ids"][index],
                    "truth": str(truth[index]),
                    "prediction": str(predicted[index]),
                    "evaluation_role": str(roles[index]),
                    "view_budget": int(budgets[index]),
                    "correct": int(predicted[index] == truth[index]),
                    "direct_adv3b02_prediction": str(
                        payload["direct_adv3b02_prediction"][index]
                    ),
                    "direct_adv3b02_correct": (
                        int(
                            payload["direct_adv3b02_prediction"][index]
                            == truth[index]
                        )
                        if str(roles[index]) == "target_old"
                        else ""
                    ),
                }
                for index in range(len(truth))
            )
    for method_name, blocks in aggregate.items():
        pred = np.concatenate(blocks["pred"])
        truth = np.concatenate(blocks["truth"])
        roles = np.concatenate(blocks["roles"])
        metrics, class_rows = _metric_row(pred, truth, roles, old_labels, new_labels)
        direct_old = np.concatenate(blocks["direct_old"])
        old_truth = truth[roles == "target_old"]
        direct_old_accuracy = float(np.mean(direct_old == old_truth))
        metrics["direct_adv3b02_old_accuracy"] = direct_old_accuracy
        metrics["delta_vs_direct_adv3b02"] = float(
            metrics["old_accuracy"] - direct_old_accuracy
        )
        identity_after = np.concatenate(blocks["identity_after"])
        identity_before = np.concatenate(blocks["identity_before"])
        identity_after_acc = float(np.mean(identity_after == old_truth))
        identity_before_acc = float(np.mean(identity_before == old_truth))
        metrics["identity_old_acc_before_increment"] = identity_before_acc
        metrics["identity_old_acc_after_increment"] = identity_after_acc
        metrics["identity_average_forgetting"] = float(
            identity_before_acc - identity_after_acc
        )
        if blocks["old_before"]:
            old_before = np.concatenate(blocks["old_before"])
            old_before_accuracy = float(np.mean(old_before == old_truth))
            metrics["old_accuracy_before_increment"] = old_before_accuracy
            metrics["old_adaptation_gain"] = float(
                metrics["old_accuracy"] - old_before_accuracy
            )
            metrics["average_forgetting"] = float(
                old_before_accuracy - metrics["old_accuracy"]
            )
        method_scenario_rows = [row for row in rows if row["method"] == method_name]
        resources = {
            key: float(np.mean([row[key] for row in method_scenario_rows]))
            for key in ("mean_backbone_forwards", "p95_backbone_forwards", "view1_rate", "view3_rate", "view5_rate")
        }
        rows.append({"scenario": "ALL", "method": method_name, **metrics, **resources})
        per_class_rows.extend(
            {"scenario": "ALL", "method": method_name, **row}
            for row in class_rows
        )

    fixed1_parity = [row for row in rows if row["scenario"] == "ALL" and row["method"] == "fixed1"][0]
    candidate_id = str(
        candidate_lock["locked_candidate"].get("candidate_id", "")
    )
    candidate_lock_file_sha256 = _sha256_file(args.candidate_lock)
    formal_rows: list[dict[str, Any]] = []
    for scenario in SCENARIOS:
        metric = next(
            row
            for row in rows
            if row["scenario"] == scenario
            and row["method"] == "adaptive1to3to5"
        )
        audit = protocol_audit[str(scenario)]
        formal_rows.append(
            {
                "candidate_id": candidate_id,
                "candidate_lock_sha256": candidate_lock_file_sha256,
                "receiver": str(config["target_receiver_labels"][0]),
                "seed": int(config["seed"]),
                "scenario": str(scenario),
                "new_class_count": int(len(new_labels)),
                "k_shot": int(k_shot),
                "registered_class_count": int(len(classes)),
                "query_per_tx": int(config["query_per_tx"]),
                "support_ids_json": json.dumps(audit["support_ids"]),
                "query_ids_json": json.dumps(audit["query_ids"]),
                "old_tx_labels_json": json.dumps(old_labels),
                "new_tx_labels_json": json.dumps(new_labels),
                "support_query_view": "leo_weak_only",
                "clean_support_query_rows": 0,
                "old_acc_before_increment": float(
                    metric["old_accuracy_before_increment"]
                ),
                "old_acc_after_increment": float(metric["old_accuracy"]),
                "average_forgetting": float(metric["average_forgetting"]),
                "old_adaptation_gain": float(metric["old_adaptation_gain"]),
                "min_old_class_acc": float(metric["min_old_class_accuracy"]),
                "seen_new_acc": float(metric["new_accuracy"]),
                "h_old_new": float(metric["harmonic_mean"]),
                "identity_average_forgetting": float(
                    metric["identity_average_forgetting"]
                ),
                "direct_adv3b02_old_acc": float(
                    metric["direct_adv3b02_old_accuracy"]
                ),
            }
        )
    formal_prediction_rows = [
        {
            "candidate_id": candidate_id,
            "candidate_lock_sha256": candidate_lock_file_sha256,
            "receiver": str(config["target_receiver_labels"][0]),
            "seed": int(config["seed"]),
            "scenario": str(row["scenario"]),
            "new_class_count": int(len(new_labels)),
            "k_shot": int(k_shot),
            "query_id": str(row["query_id"]),
            "evaluation_role": str(row["evaluation_role"]),
            "candidate_correct": int(row["correct"]),
            "direct_correct": row["direct_adv3b02_correct"],
        }
        for row in prediction_rows
        if row["method"] == "adaptive1to3to5"
    ]
    adapter_tensor_bytes = int(delta_audit["tensor_bytes_fp16"])
    if k1_head is not None:
        prototype_tensor_bytes = int(k1_head.prototypes.size * 2)
        score_transform_bytes = int(k1_head.score_transform.size * 2)
        alignment_tensor_bytes = int(
            k1_head.persistent_state_bytes_fp16
            - prototype_tensor_bytes
            - score_transform_bytes
        )
        head_tensor_bytes = int(k1_head.persistent_state_bytes_fp16)
        head_extra_macs = int(k1_head.extra_macs_per_query)
        head_prototype_macs = int(k1_head.class_count * k1_head.feature_dim)
    else:
        first_prototypes = next(iter(scenario_payloads.values()))["prototypes"]
        prototype_tensor_bytes = int(np.asarray(first_prototypes).nbytes)
        score_transform_bytes = 0
        alignment_tensor_bytes = 0
        head_tensor_bytes = prototype_tensor_bytes
        head_extra_macs = 0
        head_prototype_macs = int(
            len(classes) * np.asarray(first_prototypes).shape[-1]
        )
    threshold_bytes = 12
    total_state_bytes = int(
        adapter_tensor_bytes + head_tensor_bytes + threshold_bytes
    )
    state_cap_bytes = int(
        config.get("extreme_light_max_persistent_state_bytes", 256 * 1024)
    )
    if total_state_bytes > state_cap_bytes:
        raise ValueError(
            "adaptive multiview state exceeds configured cap: "
            f"{total_state_bytes}>{state_cap_bytes}"
        )
    adaptive_all_row = next(
        row
        for row in rows
        if row["scenario"] == "ALL" and row["method"] == "adaptive1to3to5"
    )
    head_total_macs_per_view = int(head_prototype_macs + head_extra_macs)
    adaptive_mean_views = float(adaptive_all_row["mean_backbone_forwards"])
    adaptive_p95_views = float(adaptive_all_row["p95_backbone_forwards"])
    support_unique_rows = int(len(classes) * k_shot)
    support_view_rows = int(3 * support_unique_rows)
    feature_dim = int(
        k1_head.feature_dim
        if k1_head is not None
        else np.asarray(next(iter(scenario_payloads.values()))["prototypes"]).shape[-1]
    )
    enrollment_prototype_accumulation_macs = int(support_view_rows * feature_dim)
    enrollment_gram_solve_macs_upper = int(len(classes) ** 3)
    support_extraction_seconds = float(
        sum(
            payload["support_extract"]["total_seconds"]
            for payload in scenario_payloads.values()
        )
    )
    manifest = {
        "method": (
            "support_symmetric_locked_adaptive_rxlight5_v1"
            if k1_head is not None
            else "support_prototype_adaptive_rxlight5_v1"
        ),
        "stage": "Stage2-C_recovery_diagnostic",
        "deployment_default": "lazy_adaptive_1to3to5",
        "base_view_is_default": True,
        "extra_views_requested_only_after_low_confidence": True,
        "fixed5_is_offline_upper_bound_only": True,
        "decision_rule": "per_sample_argmax_view_score_mean",
        "view_prototype_rule": (
            "three_leo_base_views_robust_spherical_plus_support_gram"
            if k1_head is not None
            else "matching_view_support_prototype"
        ),
        "support_head_calibration": (
            k1_head.calibration if k1_head is not None else None
        ),
        "calibration": calibration,
        "calibration_scope": (
            "disjoint_source_receiver_holdout_only"
            if source_validation_result is not None
            else "registered_support_leave_one_out_across_three_scenarios_only"
        ),
        "support_enrollment": {
            "physical_shots_per_class": int(k_shot),
            "augmented_view_count_per_physical_sample": (
                3 if k1_head is not None else 5
            ),
            "registered_support_views": (
                [f"{scenario}:rx_base" for scenario in SCENARIOS]
                if k1_head is not None
                else None
            ),
            "same_physical_id_augmented_views_not_counted_as_extra_shots": True,
            "clean_samples_used": False,
            "unique_physical_rows": support_unique_rows,
            "backbone_forward_rows": support_view_rows,
            "fft96_extraction_rows": support_view_rows,
            "prototype_accumulation_macs": enrollment_prototype_accumulation_macs,
            "gram_solve_macs_upper_bound": enrollment_gram_solve_macs_upper,
            "measured_feature_extraction_seconds": support_extraction_seconds,
        },
        "query_labels_used_for_calibration": False,
        "query_features_used_for_calibration": False,
        "old_new_role_used_for_decision": False,
        "class_quota_used": False,
        "query_batch_state_required": False,
        "threshold_grid": {
            "base_margin": list(BASE_MARGIN_GRID),
            "shift3_margin": list(SHIFT3_MARGIN_GRID),
            "shift3_disagreement": list(DISAGREEMENT_GRID),
        },
        "classes": classes,
        "candidate_lock": {
            "path": str(args.candidate_lock),
            "sha256": _sha256_file(args.candidate_lock),
            "locked_candidate_sha256": candidate_lock["locked_candidate_sha256"],
        },
        "protocol_audit": protocol_audit,
        "extraction_audit": extraction_audit,
        "reference_base_view_parity": parity_audit,
        "fixed1_deployed_adapter_metrics": fixed1_parity,
        "fixed1_deployed_fp16_delta_metrics": fixed1_parity,
        "checkpoint": str(args.ckpt),
        "checkpoint_sha256": _sha256_file(args.ckpt),
        "checkpoint_load_audit": checkpoint_audit,
        "direct_adv3b02_baseline": {
            "checkpoint_is_same_strict_base": True,
            "scenario_audits": {
                scenario: payload["direct_adv3b02_audit"]
                for scenario, payload in scenario_payloads.items()
            },
            "support_used": False,
            "adapter_used": False,
            "fft_used": False,
            "tta_used": False,
            "claim_scope": "target_old_only",
            "class_id_to_tx": direct_class_labels,
            "class_mapping_source": direct_mapping_source,
            "class_mapping_sha256": direct_mapping_sha256,
        },
        "adapter_state": str(args.adapter_state),
        "adapter_state_sha256": _sha256_file(args.adapter_state),
        "adapter_delta_audit": delta_audit,
        "adapter_merge_audit": merge_audit,
        "adapter_manifest": (
            str(args.adapter_manifest) if args.adapter_manifest is not None else None
        ),
        "adapter_manifest_audit": adapter_manifest_audit,
        "persistent_state": {
            "adapter_tensor_bytes_fp16": adapter_tensor_bytes,
            "prototype_tensor_bytes_fp16": prototype_tensor_bytes,
            "score_transform_tensor_bytes_fp16": score_transform_bytes,
            "diagonal_alignment_tensor_bytes_fp16": alignment_tensor_bytes,
            "head_tensor_bytes_fp16": head_tensor_bytes,
            "head_state_path": str(head_state_path) if head_state_path else None,
            "head_state_sha256": (
                _sha256_file(head_state_path) if head_state_path else None
            ),
            "head_state_reload_audit": head_state_reload_audit,
            "threshold_bytes_fp32": threshold_bytes,
            "total_bytes": total_state_bytes,
            "configured_cap_bytes": state_cap_bytes,
            "headroom_to_configured_cap_bytes": int(
                state_cap_bytes - total_state_bytes
            ),
            "within_configured_cap": total_state_bytes <= state_cap_bytes,
            "class_label_strings_excluded_from_tensor_state_accounting": True,
            "storage_dtype": "fp16",
            "compute_dtype_claim": "fp32_scoring_after_fp16_state_roundtrip",
            "native_fp16_kernel_validated": False,
        },
        "adapter_compute": {
            "dynamic_added_macs_per_backbone_forward": int(
                delta_audit.get("dynamic_added_macs_per_backbone_forward", 0)
            ),
            "mergeable_into_base_linear_weights": bool(
                delta_audit.get("mergeable_into_base_linear_weights", True)
            ),
            "merged_added_macs_per_query": int(
                delta_audit.get("merged_added_macs_per_query", 0)
            ),
            "support_head_extra_macs_per_view_over_prototype_cosine": head_extra_macs,
            "support_head_prototype_cosine_macs_per_view": head_prototype_macs,
            "support_head_total_macs_per_view": head_total_macs_per_view,
            "support_head_mean_macs_per_physical_query": int(
                round(head_total_macs_per_view * adaptive_mean_views)
            ),
            "support_head_p95_macs_per_physical_query": int(
                round(head_total_macs_per_view * adaptive_p95_views)
            ),
            "support_head_fixed5_macs_per_physical_query": int(
                5 * head_total_macs_per_view
            ),
            "adaptive_mean_view_count": adaptive_mean_views,
            "adaptive_p95_view_count": adaptive_p95_views,
        },
        "old_before_increment_diagnostic": {
            "uses_separate_old_only_head": True,
            "old_only_head_state_is_not_deployed_or_counted_in_persistent_state": True,
            "adaptive_thresholds": old_calibration,
        },
        "results": rows,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "manifest.json").write_text(
        json.dumps(_serializable(manifest), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    for filename, fieldnames, data in (
        ("summary.csv", list(rows[0]), rows),
        ("per_class.csv", list(per_class_rows[0]), per_class_rows),
        ("predictions.csv", list(prediction_rows[0]), prediction_rows),
        ("formal_rows.csv", list(formal_rows[0]), formal_rows),
        (
            "formal_predictions.csv",
            list(formal_prediction_rows[0]),
            formal_prediction_rows,
        ),
    ):
        with (args.out_dir / filename).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)
    print(json.dumps(_serializable({
        "out_dir": str(args.out_dir),
        "selected_calibration": calibration["selected"],
        "aggregate_results": [row for row in rows if row["scenario"] == "ALL"],
        "adapter_delta_audit": delta_audit,
    }), indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
