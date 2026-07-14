#!/usr/bin/env python
"""Train and export a support-only micro IQ adapter for formal CVS Stage2-C.

The frozen ADV3B02 identity backbone receives gradients only with respect to the
adapter input.  The optimizer sees registered target support labels and up to
three preregistered LEO support views; target query rows are never passed to the
training loop.  Each exported query is scored from one physical IQ view.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

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
from cvsrffi.identity_only_forward import identity_only_feature_forward
from eval_feature_diagnosis import collect_feature_dict
from export_spaceborne_features import _rf_statistics_batch, _spectral_logmag_sketch_batch
from paper_reproduction.cvs_aligned.cvs_method_runner import SCENARIOS, _sample_id, _select_split


EPS = 1.0e-8


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _norm_rows(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x.float(), dim=-1, eps=EPS)


class MicroIQResidualAdapter(nn.Module):
    """Identity-initialized depthwise IQ residual with a sub-kilobyte state."""

    def __init__(self, *, hidden: int = 8, kernel_size: int = 5, alpha: float = 0.20) -> None:
        super().__init__()
        hidden = max(2, int(hidden))
        kernel_size = max(3, int(kernel_size))
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")
        pad = kernel_size // 2
        self.hidden = hidden
        self.kernel_size = kernel_size
        self.alpha = float(alpha)
        self.in_conv = nn.Conv1d(2, hidden, kernel_size, padding=pad)
        self.depthwise = nn.Conv1d(
            hidden, hidden, kernel_size, padding=pad, groups=hidden
        )
        self.out_conv = nn.Conv1d(hidden, 2, 1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = F.gelu(self.in_conv(x))
        residual = F.gelu(self.depthwise(residual))
        residual = torch.tanh(self.out_conv(residual))
        return x + self.alpha * residual

    def macs_per_sample(self, sequence_length: int) -> int:
        length = int(sequence_length)
        return int(
            length
            * (
                2 * self.hidden * self.kernel_size
                + self.hidden * self.kernel_size
                + self.hidden * 2
            )
        )


def adapter_resource_audit(adapter: nn.Module, *, sequence_length: int) -> dict[str, Any]:
    params = [parameter for parameter in adapter.parameters() if parameter.requires_grad]
    count = int(sum(parameter.numel() for parameter in params))
    fp32_bytes = int(sum(parameter.numel() * 4 for parameter in params))
    fp16_bytes = int(sum(parameter.numel() * 2 for parameter in params))
    macs = (
        int(adapter.macs_per_sample(sequence_length))
        if callable(getattr(adapter, "macs_per_sample", None))
        else 0
    )
    return {
        "trainable_parameters": count,
        "adapter_state_bytes_fp32": fp32_bytes,
        "adapter_state_bytes_fp16": fp16_bytes,
        "adapter_macs_per_query": macs,
        "query_view_count": 1,
    }


def _load_npz(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    with np.load(path, allow_pickle=False) as data:
        if "raw_iq" not in data.files:
            raise KeyError(f"raw IQ cache is missing raw_iq: {path}")
        arrays = {
            key: np.asarray(data[key])
            for key in data.files
            if key != "manifest_json"
        }
        manifest = (
            json.loads(str(data["manifest_json"].item()))
            if "manifest_json" in data.files
            else {}
        )
    raw = arrays["raw_iq"]
    if raw.ndim != 3 or raw.shape[1] != 2:
        raise ValueError(f"raw_iq must be [N,2,T], got {raw.shape} from {path}")
    return arrays, manifest


def _feature_forward(model: nn.Module, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    identity = identity_only_feature_forward(model, x, "z_id")
    if identity is not None:
        return identity
    out = model(x, y_tx=None, grl_lambda=1.0, return_aux=True)
    features = collect_feature_dict(out)
    if "z_id" not in features:
        raise KeyError("frozen checkpoint does not expose z_id")
    logits = out.get("tx_logits", out.get("logits")) if isinstance(out, dict) else None
    if not torch.is_tensor(logits):
        raise KeyError("frozen checkpoint does not expose tx_logits")
    return features["z_id"].float(), logits.float()


def _batched_feature_forward(
    model: nn.Module,
    adapter: nn.Module,
    rows: torch.Tensor,
    *,
    batch_size: int,
    require_grad: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    features: list[torch.Tensor] = []
    logits: list[torch.Tensor] = []
    adapted: list[torch.Tensor] = []
    context = torch.enable_grad if require_grad else torch.no_grad
    with context():
        for start in range(0, int(rows.shape[0]), int(batch_size)):
            batch = rows[start : start + int(batch_size)]
            batch_adapted = adapter(batch)
            z, score = _feature_forward(model, batch_adapted)
            features.append(z.float())
            logits.append(score.float())
            adapted.append(batch_adapted.float())
    return torch.cat(features), torch.cat(logits), torch.cat(adapted)


def _split_for_cache(
    arrays: dict[str, np.ndarray],
    *,
    receiver: str,
    old_labels: list[str],
    new_labels: list[str],
    seed: int,
    k_shot: int,
    support_pool_max_k: int,
    query_per_tx: int,
    scenario: str,
) -> tuple[list[int], list[int]]:
    old_support, old_query = _select_split(
        arrays,
        role="target_old",
        tx_labels=old_labels,
        receiver=receiver,
        seed=seed,
        k_shot=k_shot,
        support_pool_max_k=support_pool_max_k,
        query_per_tx=query_per_tx,
        scenario=scenario,
    )
    new_support, new_query = _select_split(
        arrays,
        role="target_new",
        tx_labels=new_labels,
        receiver=receiver,
        seed=seed,
        k_shot=k_shot,
        support_pool_max_k=support_pool_max_k,
        query_per_tx=query_per_tx,
        scenario=scenario,
    )
    return old_support + new_support, old_query + new_query


def assemble_support_views(
    caches: dict[str, dict[str, np.ndarray]],
    *,
    receiver: str,
    old_labels: list[str],
    new_labels: list[str],
    seed: int,
    k_shot: int,
    support_pool_max_k: int,
    query_per_tx: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    class_order = list(old_labels) + list(new_labels)
    class_to_index = {label: index for index, label in enumerate(class_order)}
    view_rows: list[np.ndarray] = []
    view_labels: list[int] = []
    reference_support_ids: list[str] | None = None
    reference_query_ids: list[str] | None = None
    scenario_audit: dict[str, Any] = {}
    for scenario in SCENARIOS:
        arrays = caches[scenario]
        support_idx, query_idx = _split_for_cache(
            arrays,
            receiver=receiver,
            old_labels=old_labels,
            new_labels=new_labels,
            seed=seed,
            k_shot=k_shot,
            support_pool_max_k=support_pool_max_k,
            query_per_tx=query_per_tx,
            scenario=scenario,
        )
        support_ids = [_sample_id(arrays, index) for index in support_idx]
        query_ids = [_sample_id(arrays, index) for index in query_idx]
        if reference_support_ids is None:
            reference_support_ids = support_ids
            reference_query_ids = query_ids
        elif support_ids != reference_support_ids or query_ids != reference_query_ids:
            raise ValueError(f"physical split is not matched across scenarios: {scenario}")
        labels = arrays["tx_ids"][support_idx].astype(str).tolist()
        view_rows.append(arrays["raw_iq"][support_idx].astype(np.float32, copy=False))
        view_labels.extend(class_to_index[label] for label in labels)
        scenario_audit[scenario] = {
            "support_count": len(support_idx),
            "query_count": len(query_idx),
            "support_ids_sha256": hashlib.sha256(
                "\n".join(support_ids).encode("utf-8")
            ).hexdigest(),
            "query_ids_sha256": hashlib.sha256(
                "\n".join(query_ids).encode("utf-8")
            ).hexdigest(),
        }
    return (
        np.concatenate(view_rows, axis=0),
        np.asarray(view_labels, dtype=np.int64),
        {
            "class_order": class_order,
            "physical_support_ids": reference_support_ids or [],
            "physical_query_ids": reference_query_ids or [],
            "support_view_count": len(SCENARIOS),
            "scenario_audit": scenario_audit,
        },
    )


def _class_prototypes(
    model: nn.Module,
    adapter: nn.Module,
    rows: torch.Tensor,
    labels: torch.Tensor,
    *,
    class_count: int,
    batch_size: int,
) -> torch.Tensor:
    features, _, _ = _batched_feature_forward(
        model, adapter, rows, batch_size=batch_size, require_grad=False
    )
    features = _norm_rows(features)
    prototypes = []
    for class_index in range(int(class_count)):
        mask = labels == class_index
        if not bool(mask.any()):
            raise ValueError(f"support class {class_index} is empty")
        prototypes.append(_norm_rows(features[mask].mean(dim=0, keepdim=True))[0])
    return torch.stack(prototypes, dim=0).detach()


def train_support_only_adapter(
    model: nn.Module,
    adapter: nn.Module,
    support_rows: np.ndarray,
    support_labels: np.ndarray,
    *,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    temperature: float,
    feature_anchor_weight: float,
    residual_weight: float,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if int(epochs) > 20:
        raise ValueError("formal extreme-light adaptation is capped at 20 epochs")
    rows = torch.from_numpy(np.asarray(support_rows, dtype=np.float32)).to(device)
    labels = torch.from_numpy(np.asarray(support_labels, dtype=np.int64)).to(device)
    class_count = int(labels.max().item()) + 1
    with torch.no_grad():
        base_features, _, _ = _batched_feature_forward(
            model, nn.Identity(), rows, batch_size=batch_size, require_grad=False
        )
        base_features = _norm_rows(base_features).detach()
    optimizer = torch.optim.AdamW(
        adapter.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay)
    )
    rng = np.random.default_rng(int(seed))
    trace: list[dict[str, Any]] = []
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(1, int(epochs) + 1):
        epoch_started = time.perf_counter()
        prototypes = _class_prototypes(
            model,
            adapter,
            rows,
            labels,
            class_count=class_count,
            batch_size=batch_size,
        )
        order = rng.permutation(int(rows.shape[0]))
        totals = {"loss": 0.0, "ce": 0.0, "anchor": 0.0, "residual": 0.0, "correct": 0.0}
        seen = 0
        adapter.train()
        model.eval()
        for start in range(0, len(order), int(batch_size)):
            positions = torch.as_tensor(
                order[start : start + int(batch_size)], device=device, dtype=torch.long
            )
            x = rows[positions]
            y = labels[positions]
            optimizer.zero_grad(set_to_none=True)
            x_adapted = adapter(x)
            z, _ = _feature_forward(model, x_adapted)
            z = _norm_rows(z)
            scores = float(temperature) * (z @ prototypes.T)
            ce = F.cross_entropy(scores, y)
            anchor = (1.0 - torch.sum(z * base_features[positions], dim=1)).mean()
            residual = F.mse_loss(x_adapted, x)
            loss = ce + float(feature_anchor_weight) * anchor + float(residual_weight) * residual
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), 5.0)
            optimizer.step()
            count = int(y.numel())
            seen += count
            totals["loss"] += float(loss.detach()) * count
            totals["ce"] += float(ce.detach()) * count
            totals["anchor"] += float(anchor.detach()) * count
            totals["residual"] += float(residual.detach()) * count
            totals["correct"] += float((scores.argmax(dim=1) == y).sum().detach())
        trace.append(
            {
                "epoch": epoch,
                "loss": totals["loss"] / max(1, seen),
                "prototype_ce": totals["ce"] / max(1, seen),
                "feature_anchor": totals["anchor"] / max(1, seen),
                "input_residual_mse": totals["residual"] / max(1, seen),
                "support_train_acc": totals["correct"] / max(1, seen),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "epoch_seconds": time.perf_counter() - epoch_started,
            }
        )
        print("[MICRO-IQ-EPOCH] " + json.dumps(trace[-1], sort_keys=True), flush=True)
    runtime = {
        "adaptation_wall_seconds": time.perf_counter() - started,
        "peak_cuda_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
        "optimizer_state_deployment_required": False,
    }
    return trace, runtime


def _filter_export_rows(
    arrays: dict[str, np.ndarray],
    *,
    receiver: str,
    old_labels: Iterable[str],
    new_labels: Iterable[str],
) -> np.ndarray:
    roles = arrays["dataset_role"].astype(str)
    tx = arrays["tx_ids"].astype(str)
    rx = arrays["rx_ids"].astype(str)
    old_set = set(old_labels)
    new_set = set(new_labels)
    return np.flatnonzero(
        (rx == str(receiver))
        & (((roles == "target_old") & np.isin(tx, list(old_set))) | ((roles == "target_new") & np.isin(tx, list(new_set))))
    )


def export_adapted_cache(
    arrays: dict[str, np.ndarray],
    source_manifest: dict[str, Any],
    *,
    model: nn.Module,
    adapter: nn.Module,
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
    raw = torch.from_numpy(arrays["raw_iq"][keep].astype(np.float32, copy=False)).to(device)
    adapter.eval()
    model.eval()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    features, logits, adapted_iq = _batched_feature_forward(
        model, adapter, raw, batch_size=batch_size, require_grad=False
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    payload: dict[str, np.ndarray] = {}
    total_rows = int(arrays["raw_iq"].shape[0])
    for key, value in arrays.items():
        if key in {"features", "tx_logits", "raw_iq", "fft_logmag_features", "rf_stat_features", "fft_rf_features"}:
            continue
        if value.ndim >= 1 and int(value.shape[0]) == total_rows:
            payload[key] = value[keep]
    payload["features"] = features.detach().cpu().numpy().astype(np.float32)
    payload["tx_logits"] = logits.detach().cpu().numpy().astype(np.float32)
    adapted_np = adapted_iq.detach().cpu().numpy().astype(np.float32)
    if "fft_logmag_features" in arrays:
        dim = int(arrays["fft_logmag_features"].shape[1])
        payload["fft_logmag_features"] = _spectral_logmag_sketch_batch(adapted_np, dim=dim)
    if "rf_stat_features" in arrays:
        payload["rf_stat_features"] = _rf_statistics_batch(adapted_np)
    if "fft_logmag_features" in payload and "rf_stat_features" in payload:
        payload["fft_rf_features"] = np.concatenate(
            [payload["fft_logmag_features"], payload["rf_stat_features"]], axis=1
        ).astype(np.float32)
    manifest = dict(source_manifest)
    manifest.update(
        {
            "payload_source": "cvs_stage2c_support_only_micro_iq_adapter_v1",
            "target_receiver": receiver,
            "target_old_tx_ids": old_labels,
            "new_tx_ids": new_labels,
            "target_channel_scenarios": [scenario],
            "satellite_tta_policy": "none",
            "satellite_tta_view_count": 1,
            "raw_iq_included": False,
            "query_update_forbidden": True,
            "query_labels_used_for_training": False,
            "adapter": adaptation_manifest,
            "export_row_count": int(len(keep)),
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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(trace), indent=2) + "\n", encoding="utf-8")
    if trace:
        with path.with_suffix(".csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(trace[0].keys()))
            writer.writeheader()
            writer.writerows(trace)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--out_root", type=Path, required=True)
    parser.add_argument("--receiver", required=True)
    parser.add_argument("--new_count", type=int, choices=(5, 10, 20), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--k_shot", type=int, choices=(5, 10), default=10)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--hidden", type=int, default=8)
    parser.add_argument("--kernel_size", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.20)
    parser.add_argument("--learning_rate", type=float, default=5.0e-4)
    parser.add_argument("--weight_decay", type=float, default=1.0e-4)
    parser.add_argument("--temperature", type=float, default=18.0)
    parser.add_argument("--feature_anchor_weight", type=float, default=0.05)
    parser.add_argument("--residual_weight", type=float, default=0.02)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = json.loads(args.config.read_text(encoding="utf-8-sig"))
    old_labels = [str(value) for value in config["target_old_tx_labels"]]
    all_new = [str(value) for value in config["target_new_tx_labels"]]
    new_labels = all_new[: int(args.new_count)]
    mapping = config.get("feature_npz_by_scenario", {})
    if set(mapping) != set(SCENARIOS):
        raise ValueError(f"config must map exactly the formal scenarios: {SCENARIOS}")
    caches: dict[str, dict[str, np.ndarray]] = {}
    source_manifests: dict[str, dict[str, Any]] = {}
    cache_hashes: dict[str, str] = {}
    for scenario in SCENARIOS:
        path = Path(mapping[scenario])
        caches[scenario], source_manifests[scenario] = _load_npz(path)
        cache_hashes[scenario] = _sha256_file(path)
        roles = caches[scenario]["dataset_role"].astype(str)
        target_mask = np.isin(roles, ["target_old", "target_new"])
        observed = set(
            caches[scenario]["sat_scenarios"][target_mask].astype(str).tolist()
        )
        if observed != {scenario}:
            raise ValueError(f"cache scenario mismatch for {scenario}: {sorted(observed)}")
    support_rows, support_labels, split_manifest = assemble_support_views(
        caches,
        receiver=str(args.receiver),
        old_labels=old_labels,
        new_labels=new_labels,
        seed=int(args.seed),
        k_shot=int(args.k_shot),
        support_pool_max_k=int(config.get("support_pool_max_k", 10)),
        query_per_tx=int(config.get("query_per_tx", 20)),
    )
    device = torch.device(str(args.device) if torch.cuda.is_available() else "cpu")
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed) % (2**32))
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model, checkpoint_load_audit = build_exact_ssdg_model_from_checkpoint(
        ckpt, input_len=int(support_rows.shape[-1]), device=device
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    adapter = MicroIQResidualAdapter(
        hidden=int(args.hidden), kernel_size=int(args.kernel_size), alpha=float(args.alpha)
    ).to(device)
    resources = adapter_resource_audit(adapter, sequence_length=int(support_rows.shape[-1]))
    if resources["trainable_parameters"] > 50_000:
        raise ValueError(f"adapter exceeds 50k parameter cap: {resources}")
    if resources["adapter_state_bytes_fp16"] > 131_072:
        raise ValueError(f"adapter exceeds 128KB state cap: {resources}")
    trace, runtime = train_support_only_adapter(
        model,
        adapter,
        support_rows,
        support_labels,
        epochs=int(args.epochs),
        learning_rate=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        temperature=float(args.temperature),
        feature_anchor_weight=float(args.feature_anchor_weight),
        residual_weight=float(args.residual_weight),
        batch_size=int(args.batch_size),
        seed=int(args.seed),
        device=device,
    )
    run_id = (
        f"micro_iq_rx_{args.receiver}_new_{args.new_count}_seed_{args.seed}_k_{args.k_shot}"
    )
    run_dir = args.out_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_trace(run_dir / "loss_trace.json", trace)
    fp16_state = {key: value.detach().cpu().half() for key, value in adapter.state_dict().items()}
    torch.save(fp16_state, run_dir / "adapter_state_fp16.pt")
    state_file_bytes = int((run_dir / "adapter_state_fp16.pt").stat().st_size)
    resources["adapter_state_file_bytes_fp16_pt"] = state_file_bytes
    resources["backbone_trainable_parameters"] = 0
    resources["backbone_gradient_updates"] = 0
    adaptation_manifest = {
        "method": "support_only_micro_iq_residual_v1",
        "receiver": str(args.receiver),
        "new_count": int(args.new_count),
        "seed": int(args.seed),
        "k_shot": int(args.k_shot),
        "support_view_count": 3,
        "query_view_count": 1,
        "support_only": True,
        "query_update_forbidden": True,
        "query_labels_used_for_training": False,
        "old_new_role_used_by_optimizer": False,
        "class_quota_used_at_inference": False,
        "epochs": int(args.epochs),
        "hyperparameters": {
            "hidden": int(args.hidden),
            "kernel_size": int(args.kernel_size),
            "alpha": float(args.alpha),
            "learning_rate": float(args.learning_rate),
            "weight_decay": float(args.weight_decay),
            "temperature": float(args.temperature),
            "feature_anchor_weight": float(args.feature_anchor_weight),
            "residual_weight": float(args.residual_weight),
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
        export_audit[scenario] = export_adapted_cache(
            caches[scenario],
            source_manifests[scenario],
            model=model,
            adapter=adapter,
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
            "qknnv42_expected_tta_view_count": 1,
            "input_adapter_method": "support_only_micro_iq_residual_v1",
            "input_adapter_manifest": str(run_dir / "training_manifest.json"),
        }
    )
    resolved_path = run_dir / "resolved_qknn_config.json"
    resolved_path.write_text(
        json.dumps(_json_safe(resolved), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    training_manifest = {
        **adaptation_manifest,
        "loss_trace_json": str(run_dir / "loss_trace.json"),
        "loss_trace_csv": str(run_dir / "loss_trace.csv"),
        "adapter_state": str(run_dir / "adapter_state_fp16.pt"),
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
                "last_epoch": trace[-1] if trace else {},
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
