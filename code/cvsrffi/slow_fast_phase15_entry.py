"""Ground-cache construction and bundle fitting for CVS Phase1.5."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import torch
from torch import Tensor, nn

from .slow_fast_adapter import SlowFastCandidate
from .slow_fast_bundle import save_slow_fast_bundle
from .slow_fast_cache import GroundFeatureCache
from .slow_fast_phase15 import train_slow_fast_basis
from .stage2_structured_late_block_adaptation import _identity_features


_VIEWS = ("clean", "leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak")


def _materialize_source_view(
    x: Tensor,
    *,
    physical_sample_id: str,
    view: str,
    seed: int,
) -> Tensor:
    from .meta_phase1_entry import _materialize_ref_view

    ref = SimpleNamespace(view=view, physical_sample_id=physical_sample_id)
    return _materialize_ref_view(
        x,
        ref,
        view_seed=int(seed),
    )


def _flush_features(model: nn.Module, rows: list[Tensor], device: torch.device) -> Tensor:
    batch = torch.stack(rows).to(device)
    with torch.no_grad():
        features = _identity_features(model, batch)
    if (
        features.ndim != 2
        or features.shape[0] != batch.shape[0]
        or not bool(torch.isfinite(features).all())
    ):
        raise ValueError("frozen ADV3B02 returned invalid Phase1.5 z_id features")
    return features.detach().cpu().float()


def build_ground_feature_cache(
    model: nn.Module,
    source_dataset: Any,
    *,
    class_id_to_row: Mapping[int, int],
    seed: int,
    device: str | torch.device,
    batch_size: int = 128,
) -> GroundFeatureCache:
    """Extract clean and three LEO views from L_s without retaining IQ."""

    if int(batch_size) < 1:
        raise ValueError("batch_size must be positive")
    if not class_id_to_row or len(set(class_id_to_row.values())) != len(class_id_to_row):
        raise ValueError("frozen class mapping must be nonempty and one-to-one")
    target_device = torch.device(device)
    model = model.to(target_device)
    model.eval()
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("Phase1.5 feature extractor must be fully frozen")

    pending: list[Tensor] = []
    feature_parts: list[Tensor] = []
    labels: list[int] = []
    receivers: list[object] = []
    days: list[object] = []
    scenes: list[str] = []
    physical_ids: list[str] = []
    views: list[str] = []

    def flush() -> None:
        if pending:
            feature_parts.append(_flush_features(model, pending, target_device))
            pending.clear()

    for dataset_index in range(len(source_dataset)):
        item = source_dataset[dataset_index]
        if not isinstance(item, (tuple, list)) or len(item) < 3:
            raise ValueError("L_s row must contain IQ, class and metadata")
        x = item[0] if torch.is_tensor(item[0]) else torch.as_tensor(item[0])
        class_id = int(item[1])
        metadata = item[-1]
        if class_id not in class_id_to_row:
            raise ValueError("L_s class is outside the frozen class mapping")
        if not isinstance(metadata, Mapping):
            raise ValueError("L_s row metadata must be a mapping")
        if any(key not in metadata for key in ("rx_i", "day_i", "physical_sample_id")):
            raise ValueError("L_s metadata lacks receiver/day/physical_sample_id")
        physical_id = str(metadata["physical_sample_id"])
        if not physical_id:
            raise ValueError("L_s physical_sample_id must be nonempty")
        for view in _VIEWS:
            pending.append(
                _materialize_source_view(
                    x.detach().float(),
                    physical_sample_id=physical_id,
                    view=view,
                    seed=int(seed),
                )
            )
            labels.append(int(class_id_to_row[class_id]))
            receivers.append(metadata["rx_i"])
            days.append(metadata["day_i"])
            scenes.append(view)
            physical_ids.append(physical_id)
            views.append(view)
            if len(pending) >= int(batch_size):
                flush()
    flush()
    if not feature_parts:
        raise ValueError("L_s source dataset is empty")
    return GroundFeatureCache(
        features=torch.cat(feature_parts, dim=0),
        labels=torch.tensor(labels, dtype=torch.long),
        receivers=tuple(receivers),
        days=tuple(days),
        scenes=tuple(scenes),
        physical_sample_ids=tuple(physical_ids),
        views=tuple(views),
        roles=tuple("L_s" for _ in labels),
    )


def fit_and_save_slow_fast_bundles(
    cache: GroundFeatureCache,
    prototypes: Tensor,
    class_ids: Tensor,
    output_root: str | Path,
    *,
    base_checkpoint_id: str,
    steps: int = 200,
    learning_rate: float = 1.0e-2,
    seed: int = 392002,
    rho: float = 0.1,
    support_logit_scale: float = 8.0,
    fast_step_size: float = 0.02,
    trust_radius: float = 0.15,
    device: str | torch.device = "cpu",
    meta_steps: int | None = None,
) -> dict[str, Any]:
    """Fit all three candidates and persist only aggregate deployment bundles."""

    root = Path(output_root)
    if root.exists() or root.is_symlink():
        raise FileExistsError(f"immutable Phase1.5 output already exists: {root}")
    if class_ids.ndim != 1 or prototypes.shape != (class_ids.numel(), cache.feature_dim):
        raise ValueError("frozen prototypes/class IDs must match the ground cache")
    root.mkdir(parents=True, exist_ok=False)
    cache_path = root / "ground_feature_cache.pt"
    torch.save(
        {
            "features": cache.features,
            "labels": cache.labels,
            "receivers": cache.receivers,
            "days": cache.days,
            "scenes": cache.scenes,
            "physical_sample_ids": cache.physical_sample_ids,
            "views": cache.views,
            "roles": cache.roles,
        },
        cache_path,
    )
    shared_metadata = {
        "base_checkpoint_id": str(base_checkpoint_id),
        "class_ids": class_ids.detach().cpu().long(),
        "prototypes": prototypes.detach().cpu().float(),
        "support_logit_scale": float(support_logit_scale),
        "trust_radius": float(trust_radius),
    }
    candidates: dict[str, Any] = {}
    for candidate in SlowFastCandidate:
        state, audit = train_slow_fast_basis(
            cache,
            prototypes,
            candidate=candidate,
            steps=int(steps),
            learning_rate=float(learning_rate),
            seed=int(seed),
            rho=float(rho),
            trust_radius=float(trust_radius),
            device=device,
            meta_steps=meta_steps,
            fast_step_size=float(fast_step_size),
        )
        bundle_path = root / f"{candidate.value}.pt"
        metadata = {
            **shared_metadata,
            "fast_step_size": float(
                audit.get("learned_fast_step_size", fast_step_size)
            ),
        }
        save_slow_fast_bundle(bundle_path, state, metadata)
        candidates[candidate.value] = {
            "bundle_path": str(bundle_path.resolve()),
            "rank": state.rank,
            "fast_parameter_count": state.fast_parameter_count,
            "training": audit,
        }
    summary = {
        "status": "ARTIFACTS_COMPLETE",
        "schema": "cvs.cached_slow_fast.phase15.v1",
        "base_checkpoint_id": str(base_checkpoint_id),
        "seed": int(seed),
        "source_role": "L_s",
        "source_physical_sample_count": len(set(cache.physical_sample_ids)),
        "cached_feature_row_count": int(cache.features.shape[0]),
        "feature_dim": cache.feature_dim,
        "ground_cache_path": str(cache_path.resolve()),
        "ground_cache_in_deployment_bundle": False,
        "candidates": candidates,
    }
    (root / "phase15_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


__all__ = ["build_ground_feature_cache", "fit_and_save_slow_fast_bundles"]
