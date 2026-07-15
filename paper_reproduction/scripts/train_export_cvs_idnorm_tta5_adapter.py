#!/usr/bin/env python
"""Train one task-specific 289,685-parameter ADV3B02 adapter and export TTA5 caches.

The optimizer is fail-closed to the registered target receiver's LEO support
rows.  Clean, source, proxy and query samples are forbidden.  Query features
and FFT96 descriptors are the mean of the same fixed ``rx_light5`` views.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
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
from export_spaceborne_features import (
    _satellite_tta_views,
    _spectral_logmag_sketch_batch,
)
from paper_reproduction.cvs_aligned.cvs_method_runner import SCENARIOS
from paper_reproduction.scripts.train_export_cvs_micro_iq_adapter import (
    _feature_forward,
    _filter_export_rows,
    _json_safe,
    _load_npz,
    _numpy_to_tensor_compat,
    _sha256_file,
    _tensor_to_numpy_compat,
    assemble_support_views,
)
from scripts.train_apply_phase1_iq_preadapter_20260703 import _configure_model_adapter


EXPECTED_TRAINABLE_PARAMETERS = 289_685
SUPPORTED_EPOCHS = (1, 2, 5, 10, 20, 30, 60)
SUPPORTED_K_SHOTS = (1, 2, 5, 10, 20)
PHASE2_VIEW_POLICY = "leo_weak_only_no_clean_access"
PAYLOAD_SOURCE = "cvs_stage2c_support_only_id_norm_late_feature_tta5_v1"
ADAPTER_METHOD = "support_only_id_norm_late_feature_tta5_v1"
EPS = 1.0e-8


def _norm_rows(value: torch.Tensor) -> torch.Tensor:
    return F.normalize(value.float(), dim=-1, eps=EPS)


def _strict_load_ok(audit: dict[str, Any]) -> bool:
    def count(*names: str) -> int:
        for name in names:
            if name in audit:
                value = audit[name]
                return len(value) if isinstance(value, (list, tuple, dict)) else int(value)
        return 0

    return (
        count("missing_key_count", "missing_keys") == 0
        and count("unexpected_key_count", "unexpected_keys") == 0
        and count("shape_mismatch_count", "shape_mismatches", "skipped_mismatch") == 0
    )


def _tensor_to_numpy_fast(value: torch.Tensor, *, dtype: np.dtype) -> np.ndarray:
    """Use DLPack for the PyTorch-old/NumPy-2 bridge without Python-list copies."""
    cpu = value.detach().cpu().contiguous()
    try:
        return np.asarray(np.from_dlpack(cpu), dtype=dtype)
    except (AttributeError, BufferError, RuntimeError, TypeError):
        return _tensor_to_numpy_compat(cpu, dtype=dtype)


def _assert_cache_is_leo_only(
    arrays: dict[str, np.ndarray], manifest: dict[str, Any], *, scenario: str
) -> None:
    required = {
        "raw_iq",
        "dataset_role",
        "channel_views",
        "sat_scenarios",
        "tx_ids",
        "rx_ids",
    }
    missing = sorted(required - set(arrays))
    if missing:
        raise ValueError(f"cache {scenario} is missing required arrays: {missing}")
    roles = arrays["dataset_role"].astype(str)
    target = np.isin(roles, ["target_old", "target_new"])
    if not bool(target.any()):
        raise ValueError(f"cache {scenario} contains no target rows")
    views = set(arrays["channel_views"][target].astype(str).tolist())
    observed_scenarios = set(arrays["sat_scenarios"][target].astype(str).tolist())
    if observed_scenarios != {scenario}:
        raise ValueError(
            f"target scenario mismatch for {scenario}: {sorted(observed_scenarios)}"
        )
    if not views or any("clean" in value.lower() for value in views):
        raise ValueError(f"clean or missing LEO target view in {scenario}: {sorted(views)}")
    if manifest.get("raw_iq_included") is not True:
        raise ValueError(f"raw IQ provenance is not explicit for {scenario}")
    if int(manifest.get("satellite_tta_view_count", -1)) != 1:
        raise ValueError(f"base cache must contain one physical LEO view: {scenario}")


def _forward_rows(
    model: nn.Module,
    rows: torch.Tensor,
    *,
    batch_size: int,
    require_grad: bool,
) -> torch.Tensor:
    chunks: list[torch.Tensor] = []
    context = torch.enable_grad if require_grad else torch.no_grad
    with context():
        for start in range(0, int(rows.shape[0]), int(batch_size)):
            z, _ = _feature_forward(model, rows[start : start + int(batch_size)])
            chunks.append(z.float())
    return torch.cat(chunks, dim=0)


def _prototypes(
    model: nn.Module,
    rows: torch.Tensor,
    labels: torch.Tensor,
    *,
    class_count: int,
    batch_size: int,
) -> torch.Tensor:
    features = _norm_rows(
        _forward_rows(model, rows, batch_size=batch_size, require_grad=False)
    )
    result: list[torch.Tensor] = []
    for class_index in range(int(class_count)):
        mask = labels == class_index
        if not bool(mask.any()):
            raise ValueError(f"support class {class_index} is empty")
        result.append(_norm_rows(features[mask].mean(dim=0, keepdim=True))[0])
    return torch.stack(result).detach()


def train_support_only(
    model: nn.Module,
    support_rows: np.ndarray,
    support_labels: np.ndarray,
    *,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    temperature: float,
    feature_anchor_weight: float,
    parameter_delta_weight: float,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, torch.Tensor]]:
    rows = _numpy_to_tensor_compat(
        support_rows, numpy_dtype=np.dtype(np.float32), torch_dtype=torch.float32
    ).to(device)
    labels = _numpy_to_tensor_compat(
        support_labels, numpy_dtype=np.dtype(np.int64), torch_dtype=torch.int64
    ).to(device)
    trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    if sum(parameter.numel() for _, parameter in trainable) != EXPECTED_TRAINABLE_PARAMETERS:
        raise ValueError("the ADV3B02 id_norm_late_feature selection is not exactly 289,685 parameters")
    initial = {name: parameter.detach().cpu().clone() for name, parameter in trainable}
    model.eval()
    with torch.no_grad():
        base_features = _norm_rows(
            _forward_rows(model, rows, batch_size=batch_size, require_grad=False)
        ).detach()
    optimizer = torch.optim.AdamW(
        [parameter for _, parameter in trainable],
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )
    rng = np.random.default_rng(int(seed))
    class_count = int(labels.max().item()) + 1
    trace: list[dict[str, Any]] = []
    gradient_updates = 0
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(1, int(epochs) + 1):
        epoch_started = time.perf_counter()
        prototypes = _prototypes(
            model,
            rows,
            labels,
            class_count=class_count,
            batch_size=batch_size,
        )
        totals = {"loss": 0.0, "ce": 0.0, "anchor": 0.0, "delta": 0.0, "correct": 0.0}
        seen = 0
        order = rng.permutation(int(rows.shape[0]))
        for start in range(0, int(rows.shape[0]), int(batch_size)):
            positions = torch.as_tensor(
                order[start : start + int(batch_size)],
                device=device,
                dtype=torch.long,
            )
            if not int(positions.numel()):
                continue
            optimizer.zero_grad(set_to_none=True)
            z, _ = _feature_forward(model, rows[positions])
            z = _norm_rows(z)
            scores = float(temperature) * (z @ prototypes.T)
            ce = F.cross_entropy(scores, labels[positions])
            anchor = (1.0 - torch.sum(z * base_features[positions], dim=1)).mean()
            delta = torch.stack(
                [
                    torch.mean((parameter - initial[name].to(device=device, dtype=parameter.dtype)) ** 2)
                    for name, parameter in trainable
                ]
            ).mean()
            loss = ce + float(feature_anchor_weight) * anchor + float(parameter_delta_weight) * delta
            loss.backward()
            torch.nn.utils.clip_grad_norm_([parameter for _, parameter in trainable], 5.0)
            optimizer.step()
            gradient_updates += 1
            count = int(positions.numel())
            seen += count
            totals["loss"] += float(loss.detach()) * count
            totals["ce"] += float(ce.detach()) * count
            totals["anchor"] += float(anchor.detach()) * count
            totals["delta"] += float(delta.detach()) * count
            totals["correct"] += float((scores.argmax(dim=1) == labels[positions]).sum().detach())
        row = {
            "epoch": epoch,
            "loss": totals["loss"] / max(1, seen),
            "prototype_ce": totals["ce"] / max(1, seen),
            "leo_support_feature_anchor": totals["anchor"] / max(1, seen),
            "selected_parameter_delta_mse": totals["delta"] / max(1, seen),
            "support_train_acc": totals["correct"] / max(1, seen),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "gradient_updates_cumulative": gradient_updates,
            "epoch_seconds": time.perf_counter() - epoch_started,
        }
        trace.append(row)
        print("[IDNORM-EPOCH] " + json.dumps(row, sort_keys=True), flush=True)
    delta_state = {
        name: (parameter.detach().cpu() - initial[name]).half()
        for name, parameter in trainable
    }
    runtime = {
        "adaptation_wall_seconds": time.perf_counter() - started,
        "peak_cuda_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
        "gradient_updates": gradient_updates,
        "optimizer_state_deployment_required": False,
    }
    return trace, runtime, delta_state


def _renormalize_fft(value: np.ndarray) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32)
    result = result - result.mean(axis=1, keepdims=True, dtype=np.float64).astype(np.float32)
    norm = np.linalg.norm(result, axis=1, keepdims=True)
    return (result / np.maximum(norm, 1.0e-8)).astype(np.float32)


@torch.no_grad()
def _tta5_forward(
    model: nn.Module,
    raw: torch.Tensor,
    *,
    fft_dim: int,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    feature_chunks: list[np.ndarray] = []
    logit_chunks: list[np.ndarray] = []
    fft_chunks: list[np.ndarray] = []
    if raw.is_cuda:
        torch.cuda.synchronize(raw.device)
    started = time.perf_counter()
    for start in range(0, int(raw.shape[0]), int(batch_size)):
        batch = raw[start : start + int(batch_size)]
        views = _satellite_tta_views(batch, "rx_light5")
        if len(views) != 5:
            raise RuntimeError("rx_light5 must return exactly five views")
        features: list[torch.Tensor] = []
        logits: list[torch.Tensor] = []
        fft: list[np.ndarray] = []
        for _, view in views:
            z, score = _feature_forward(model, view)
            features.append(z.float())
            logits.append(score.float())
            fft.append(
                _spectral_logmag_sketch_batch(
                    _tensor_to_numpy_fast(view, dtype=np.dtype(np.float32)), dim=int(fft_dim)
                )
            )
        feature_chunks.append(
            _tensor_to_numpy_fast(torch.stack(features).mean(dim=0), dtype=np.dtype(np.float32))
        )
        logit_chunks.append(
            _tensor_to_numpy_fast(torch.stack(logits).mean(dim=0), dtype=np.dtype(np.float32))
        )
        fft_chunks.append(_renormalize_fft(np.stack(fft).mean(axis=0)))
    if raw.is_cuda:
        torch.cuda.synchronize(raw.device)
    return (
        np.concatenate(feature_chunks),
        np.concatenate(logit_chunks),
        np.concatenate(fft_chunks),
        time.perf_counter() - started,
    )


def export_cache(
    arrays: dict[str, np.ndarray],
    source_manifest: dict[str, Any],
    *,
    model: nn.Module,
    receiver: str,
    old_labels: list[str],
    new_labels: list[str],
    scenario: str,
    batch_size: int,
    device: torch.device,
    out_path: Path,
    adaptation_manifest: dict[str, Any],
) -> dict[str, Any]:
    keep = _filter_export_rows(
        arrays, receiver=receiver, old_labels=old_labels, new_labels=new_labels
    )
    raw = _numpy_to_tensor_compat(
        arrays["raw_iq"][keep], numpy_dtype=np.dtype(np.float32), torch_dtype=torch.float32
    ).to(device)
    fft_dim = int(arrays.get("fft_logmag_features", np.empty((0, 96))).shape[1] or 96)
    features, logits, fft, elapsed = _tta5_forward(
        model, raw, fft_dim=fft_dim, batch_size=batch_size
    )
    payload: dict[str, np.ndarray] = {}
    total_rows = int(arrays["raw_iq"].shape[0])
    excluded = {
        "features", "tx_logits", "raw_iq", "fft_logmag_features", "rf_stat_features", "fft_rf_features"
    }
    for key, value in arrays.items():
        if key not in excluded and value.ndim >= 1 and int(value.shape[0]) == total_rows:
            payload[key] = value[keep]
    payload["channel_views"] = np.full(int(len(keep)), "rx_light5_mean")
    payload["features"] = features.astype(np.float32)
    payload["tx_logits"] = logits.astype(np.float32)
    payload["fft_logmag_features"] = fft.astype(np.float32)
    manifest = dict(source_manifest)
    manifest.update(
        {
            "payload_source": PAYLOAD_SOURCE,
            "target_receiver": receiver,
            "target_old_tx_ids": old_labels,
            "new_tx_ids": new_labels,
            "target_channel_scenarios": [scenario],
            "phase2_sample_view_policy": PHASE2_VIEW_POLICY,
            "clean_sample_access": False,
            "satellite_tta_policy": "rx_light5",
            "satellite_tta_view_count": 5,
            "satellite_tta_aggregation": "feature_logit_fft_mean_per_physical_sample",
            "aux_fft_view_alignment": "same_five_post_channel_views_as_backbone",
            "raw_iq_included": False,
            "query_update_forbidden": True,
            "query_labels_used_for_training": False,
            "adapter": adaptation_manifest,
            "export_row_count": int(len(keep)),
            "backbone_forward_count_per_export_row": 5,
            "adapter_plus_backbone_latency_ms_per_row_export_batch": float(
                elapsed * 1000.0 / max(1, len(keep))
            ),
        }
    )
    payload["manifest_json"] = np.asarray(json.dumps(_json_safe(manifest), sort_keys=True))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **payload)
    return {
        "path": str(out_path),
        "sha256": _sha256_file(out_path),
        "rows": int(len(keep)),
        "latency_ms_per_row_export_batch": float(elapsed * 1000.0 / max(1, len(keep))),
    }


def _write_trace(path: Path, trace: list[dict[str, Any]]) -> None:
    path.write_text(json.dumps(_json_safe(trace), indent=2) + "\n", encoding="utf-8")
    with path.with_suffix(".csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(trace[0]))
        writer.writeheader()
        writer.writerows(trace)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--out_root", type=Path, required=True)
    parser.add_argument("--receiver", required=True)
    parser.add_argument("--new_count", type=int, choices=(2,), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--k_shot", type=int, choices=SUPPORTED_K_SHOTS, required=True)
    parser.add_argument("--epochs", type=int, choices=SUPPORTED_EPOCHS, required=True)
    parser.add_argument("--learning_rate", type=float, default=1.0e-4)
    parser.add_argument("--weight_decay", type=float, default=1.0e-4)
    parser.add_argument("--temperature", type=float, default=18.0)
    parser.add_argument("--feature_anchor_weight", type=float, default=0.05)
    parser.add_argument("--parameter_delta_weight", type=float, default=1.0e-4)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = json.loads(args.config.read_text(encoding="utf-8-sig"))
    if config.get("phase2_sample_view_policy") != PHASE2_VIEW_POLICY:
        raise ValueError("config must declare phase2_sample_view_policy=leo_weak_only_no_clean_access")
    if config.get("clean_sample_access") is not False:
        raise ValueError("config must declare clean_sample_access=false")
    old_labels = [str(value) for value in config["target_old_tx_labels"]]
    new_labels = [str(value) for value in config["target_new_tx_labels"][: int(args.new_count)]]
    if len(old_labels) != 6 or len(new_labels) != 2:
        raise ValueError("this matrix is fixed to six old plus two registered new classes")
    mapping = config.get("feature_npz_by_scenario", {})
    if tuple(mapping) != tuple(SCENARIOS):
        raise ValueError(f"config must map the ordered scenarios {SCENARIOS}")
    caches: dict[str, dict[str, np.ndarray]] = {}
    source_manifests: dict[str, dict[str, Any]] = {}
    cache_hashes: dict[str, str] = {}
    for scenario in SCENARIOS:
        path = Path(mapping[scenario])
        caches[scenario], source_manifests[scenario] = _load_npz(path)
        _assert_cache_is_leo_only(caches[scenario], source_manifests[scenario], scenario=scenario)
        cache_hashes[scenario] = _sha256_file(path)
    support_rows, support_labels, split_manifest = assemble_support_views(
        caches,
        receiver=str(args.receiver),
        old_labels=old_labels,
        new_labels=new_labels,
        seed=int(args.seed),
        k_shot=int(args.k_shot),
        support_pool_max_k=int(config["support_pool_max_k"]),
        query_per_tx=int(config["query_per_tx"]),
    )
    device = torch.device(str(args.device) if torch.cuda.is_available() else "cpu")
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed) % (2**32))
    checkpoint = torch.load(args.ckpt, map_location="cpu")
    model, checkpoint_load_audit = build_exact_ssdg_model_from_checkpoint(
        checkpoint, input_len=int(support_rows.shape[-1]), device=device
    )
    if not _strict_load_ok(checkpoint_load_audit):
        raise ValueError(f"ADV3B02 did not load strictly: {checkpoint_load_audit}")
    selection = _configure_model_adapter(model, "id_norm_late_feature")
    if int(selection.get("trainable_parameters", -1)) != EXPECTED_TRAINABLE_PARAMETERS:
        raise ValueError(f"expected exactly 289,685 trainable parameters, got {selection}")
    model.eval()
    trace, runtime, delta_state = train_support_only(
        model,
        support_rows,
        support_labels,
        epochs=int(args.epochs),
        learning_rate=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        temperature=float(args.temperature),
        feature_anchor_weight=float(args.feature_anchor_weight),
        parameter_delta_weight=float(args.parameter_delta_weight),
        batch_size=int(args.batch_size),
        seed=int(args.seed),
        device=device,
    )
    run_id = (
        f"idnorm_tta5_rx_{args.receiver}_new_2_seed_{args.seed}_"
        f"k_{args.k_shot}_e_{args.epochs}"
    )
    run_dir = args.out_root / run_id
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty task directory: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_trace(run_dir / "loss_trace.json", trace)
    torch.save(delta_state, run_dir / "adapter_delta_fp16.pt")
    resources = {
        "trainable_parameters": EXPECTED_TRAINABLE_PARAMETERS,
        "original_checkpoint_trainable_parameters": EXPECTED_TRAINABLE_PARAMETERS,
        "original_checkpoint_gradient_updates": int(runtime["gradient_updates"]),
        "adapter_state_bytes_fp16": EXPECTED_TRAINABLE_PARAMETERS * 2,
        "adapter_state_file_bytes_fp16_pt": int((run_dir / "adapter_delta_fp16.pt").stat().st_size),
        "adapter_macs_per_query": 0,
        "deployment_added_macs_per_query_after_merge": 0,
        "backbone_forward_count_per_query": 5,
        "tta_compute_multiplier": 5,
        "full_model_finetune": False,
        "checkpoint_update_target_modules": selection["trainable_tensors"],
        "checkpoint_update_tensor_count": len(selection["trainable_tensors"]),
    }
    adaptation_manifest = {
        "method": ADAPTER_METHOD,
        "adapter_selection_mode": "id_norm_late_feature",
        "adapter_state_format": "fp16_delta_from_strict_checkpoint",
        "receiver": str(args.receiver),
        "new_count": 2,
        "seed": int(args.seed),
        "k_shot": int(args.k_shot),
        "support_view_count": 3,
        "query_view_count": 5,
        "support_only": True,
        "query_update_forbidden": True,
        "query_labels_used_for_training": False,
        "old_new_role_used_by_optimizer": False,
        "class_quota_used_at_inference": False,
        "epochs": int(args.epochs),
        "phase2_sample_view_policy": PHASE2_VIEW_POLICY,
        "clean_sample_access": False,
        "resource_tier": "non_extreme_light_large_adapter_diagnostic",
        "diagnostic_only": True,
        "optimizer_sample_contract": {
            "roles": ["target_old_support", "target_new_support"],
            "channel_view": "leo_weak_only",
            "phase2_sample_view_policy": PHASE2_VIEW_POLICY,
            "clean_samples_used": False,
            "source_samples_used": False,
            "proxy_samples_used": False,
            "query_samples_used": False,
            "physical_support_count": len(split_manifest["physical_support_ids"]),
            "support_view_count": int(split_manifest["support_view_count"]),
        },
        "loss_definition": (
            "prototype_cross_entropy_on_target_receiver_leo_support + "
            "leo_support_feature_anchor + selected_parameter_delta_l2"
        ),
        "hyperparameters": {
            "learning_rate": float(args.learning_rate),
            "weight_decay": float(args.weight_decay),
            "temperature": float(args.temperature),
            "feature_anchor_weight": float(args.feature_anchor_weight),
            "parameter_delta_weight": float(args.parameter_delta_weight),
            "batch_size": int(args.batch_size),
        },
        "resources": resources,
        "runtime": runtime,
        "split": split_manifest,
        "input_cache_sha256": cache_hashes,
        "checkpoint": str(args.ckpt),
        "checkpoint_sha256": _sha256_file(args.ckpt),
        "checkpoint_load_audit": checkpoint_load_audit,
    }
    export_audit: dict[str, Any] = {}
    output_mapping: dict[str, str] = {}
    for scenario in SCENARIOS:
        out_path = run_dir / f"{scenario}.npz"
        export_audit[scenario] = export_cache(
            caches[scenario],
            source_manifests[scenario],
            model=model,
            receiver=str(args.receiver),
            old_labels=old_labels,
            new_labels=new_labels,
            scenario=scenario,
            batch_size=int(args.batch_size),
            device=device,
            out_path=out_path,
            adaptation_manifest=adaptation_manifest,
        )
        output_mapping[scenario] = str(out_path)
    resolved = dict(config)
    resolved.update(
        {
            "experiment_id": run_id,
            "feature_npz_by_scenario": output_mapping,
            "target_receiver_labels": [str(args.receiver)],
            "target_new_tx_labels": new_labels,
            "split_seed": int(args.seed),
            "seed": int(args.seed),
            "k_shot": int(args.k_shot),
            "qknnv42_expected_tta_view_count": 5,
            "input_adapter_method": ADAPTER_METHOD,
            "input_adapter_manifest": str(run_dir / "training_manifest.json"),
            "adapter_training_epochs": int(args.epochs),
            "adapter_resource_tier": "non_extreme_light_large_adapter_diagnostic",
            "resource_diagnostic_only": True,
        }
    )
    resolved_path = run_dir / "resolved_qknn_config.json"
    resolved_path.write_text(
        json.dumps(_json_safe(resolved), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    training_manifest = {
        **adaptation_manifest,
        "loss_trace_json": str(run_dir / "loss_trace.json"),
        "loss_trace_csv": str(run_dir / "loss_trace.csv"),
        "adapter_state": str(run_dir / "adapter_delta_fp16.pt"),
        "export_audit": export_audit,
        "resolved_qknn_config": str(resolved_path),
    }
    (run_dir / "training_manifest.json").write_text(
        json.dumps(_json_safe(training_manifest), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "run_id": run_id,
                "run_dir": str(run_dir),
                "resolved_qknn_config": str(resolved_path),
                "resources": resources,
                "last_epoch": trace[-1],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
