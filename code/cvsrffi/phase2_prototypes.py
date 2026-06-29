from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

import torch
import torch.nn.functional as F

from cvsrffi.tensors import extract_domain_from_extra, get_nested_tensor, unpack_batch


def _normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    x = torch.nan_to_num(x.float(), nan=0.0, posinf=0.0, neginf=0.0)
    return F.normalize(x, dim=dim, eps=float(eps))


def _graph_zero(ref: torch.Tensor) -> torch.Tensor:
    if torch.is_tensor(ref) and ref.requires_grad:
        return torch.nan_to_num(ref.float(), nan=0.0, posinf=0.0, neginf=0.0).sum() * 0.0
    return torch.zeros((), device=ref.device if torch.is_tensor(ref) else None, dtype=torch.float32)


@dataclass
class PrototypeGeometrySummary:
    initialized: int
    radius_p95_mean_deg: float
    min_interclass_angle_deg: float
    margin_violation_pairs: int


class BalancedPrototypeBank:
    """Momentum prototype bank with optional group-balanced updates.

    With group labels, each active group contributes one local center before
    averaging. This matches the Phase2 design requirement that large receiver
    or day groups do not dominate transmitter/domain prototypes.
    """

    def __init__(
        self,
        num_items: int,
        feat_dim: int,
        *,
        momentum: float = 0.95,
        min_count_per_update: int = 2,
        device=None,
        dtype: torch.dtype = torch.float32,
    ):
        self.num_items = int(num_items)
        self.feat_dim = int(feat_dim)
        self.momentum = float(momentum)
        self.min_count_per_update = int(min_count_per_update)
        self.prototypes = torch.zeros(self.num_items, self.feat_dim, device=device, dtype=dtype)
        self.counts = torch.zeros(self.num_items, device=device, dtype=torch.long)

    def to(self, device=None, dtype: Optional[torch.dtype] = None) -> "BalancedPrototypeBank":
        self.prototypes = self.prototypes.to(device=device, dtype=dtype or self.prototypes.dtype)
        self.counts = self.counts.to(device=device)
        return self

    def initialized_mask(self) -> torch.Tensor:
        return self.counts > 0

    def get(self, labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        proto = _normalize(self.prototypes, dim=1)
        if labels is None:
            return proto
        return proto[labels.to(device=proto.device).long()]

    @torch.no_grad()
    def update_from_features(
        self,
        z: torch.Tensor,
        labels: torch.Tensor,
        group_labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        if z.ndim != 2:
            raise ValueError("z must have shape [N, D]")
        if z.size(1) != self.feat_dim:
            raise ValueError(f"feature dim mismatch: expected {self.feat_dim}, got {z.size(1)}")
        z_norm = _normalize(z.detach(), dim=1)
        labels = labels.view(-1).long().to(z_norm.device)
        groups = group_labels.view(-1).long().to(z_norm.device) if group_labels is not None else None
        if labels.numel() != z_norm.size(0):
            raise ValueError("labels must have one value per feature")
        if groups is not None and groups.numel() != z_norm.size(0):
            raise ValueError("group_labels must have one value per feature")

        m = max(0.0, min(0.9999, float(self.momentum)))
        updates = 0
        skipped = 0
        for item in torch.unique(labels[(labels >= 0) & (labels < self.num_items)]):
            item_int = int(item.item())
            item_mask = labels == item_int
            if int(item_mask.sum().item()) < max(1, self.min_count_per_update):
                skipped += 1
                continue
            if groups is None:
                center = z_norm[item_mask].mean(dim=0, keepdim=True)
            else:
                local_centers = []
                for grp in torch.unique(groups[item_mask]):
                    cell = item_mask & (groups == grp)
                    if bool(cell.any()):
                        local_centers.append(z_norm[cell].mean(dim=0))
                if not local_centers:
                    skipped += 1
                    continue
                center = torch.stack(local_centers, dim=0).mean(dim=0, keepdim=True)
            center = _normalize(center, dim=1).squeeze(0)
            if int(self.counts[item_int].item()) <= 0:
                self.prototypes[item_int].copy_(center)
            else:
                self.prototypes[item_int].mul_(m).add_(center, alpha=1.0 - m)
                self.prototypes[item_int].copy_(_normalize(self.prototypes[item_int].view(1, -1), dim=1).squeeze(0))
            self.counts[item_int] += int(item_mask.sum().item())
            updates += 1
        return {
            "updated": float(updates),
            "skipped": float(skipped),
            "initialized": float(int(self.initialized_mask().sum().item())),
        }

    def prototype_pull_margin_loss(
        self,
        z: torch.Tensor,
        labels: torch.Tensor,
        *,
        margin: float = 0.20,
        alpha_margin: float = 1.0,
    ) -> tuple[torch.Tensor, Dict[str, float]]:
        if z.ndim != 2:
            raise ValueError("z must have shape [N, D]")
        labels = labels.view(-1).long().to(z.device)
        valid = (labels >= 0) & (labels < self.num_items)
        initialized = self.initialized_mask().to(z.device)
        valid = valid & initialized[labels.clamp(0, self.num_items - 1)]
        if not bool(valid.any()):
            return _graph_zero(z), {"pull": float("nan"), "margin": float("nan"), "active": float(initialized.sum().item())}

        feat = _normalize(z[valid], dim=1)
        proto = self.get().to(device=z.device, dtype=feat.dtype)
        scores = feat @ proto.t()
        pos = scores[torch.arange(scores.size(0), device=z.device), labels[valid]]
        loss_pull = (1.0 - pos).mean()
        scores_neg = scores.clone()
        scores_neg[torch.arange(scores.size(0), device=z.device), labels[valid]] = -1.0e6
        neg = scores_neg.max(dim=1).values
        loss_margin = torch.relu(float(margin) + neg - pos).mean()
        loss = loss_pull + float(alpha_margin) * loss_margin
        return loss, {
            "pull": float(loss_pull.detach().item()),
            "margin": float(loss_margin.detach().item()),
            "pos_cos": float(pos.detach().mean().item()),
            "neg_cos": float(neg.detach().mean().item()),
            "active": float(initialized.sum().item()),
        }

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {"prototypes": self.prototypes.detach().clone(), "counts": self.counts.detach().clone()}

    def load_state_dict(self, state: Mapping[str, torch.Tensor]) -> None:
        self.prototypes.copy_(state["prototypes"].to(device=self.prototypes.device, dtype=self.prototypes.dtype))
        self.counts.copy_(state["counts"].to(device=self.counts.device, dtype=self.counts.dtype))


class TxDomainPrototypeBank:
    """Local transmitter-domain prototype bank P_tx_dom[t, d]."""

    def __init__(
        self,
        num_tx: int,
        num_domains: int,
        feat_dim: int,
        *,
        momentum: float = 0.95,
        min_count_per_update: int = 1,
        device=None,
        dtype: torch.dtype = torch.float32,
    ):
        self.num_tx = int(num_tx)
        self.num_domains = int(num_domains)
        self.feat_dim = int(feat_dim)
        self.momentum = float(momentum)
        self.min_count_per_update = int(min_count_per_update)
        self.prototypes = torch.zeros(self.num_tx, self.num_domains, self.feat_dim, device=device, dtype=dtype)
        self.counts = torch.zeros(self.num_tx, self.num_domains, device=device, dtype=torch.long)

    def initialized_mask(self) -> torch.Tensor:
        return self.counts > 0

    def local_proto(self, tx: int | torch.Tensor, domain: int | torch.Tensor) -> torch.Tensor:
        return _normalize(self.prototypes[tx, domain], dim=-1)

    @torch.no_grad()
    def update(self, z_tx: torch.Tensor, y_tx: torch.Tensor, d: torch.Tensor) -> Dict[str, float]:
        if z_tx.ndim != 2:
            raise ValueError("z_tx must have shape [N, D]")
        z_norm = _normalize(z_tx.detach(), dim=1)
        y = y_tx.view(-1).long().to(z_norm.device)
        dom = d.view(-1).long().to(z_norm.device)
        if y.numel() != z_norm.size(0) or dom.numel() != z_norm.size(0):
            raise ValueError("y_tx and d must have one value per feature")
        m = max(0.0, min(0.9999, float(self.momentum)))
        updates = 0
        for tx in torch.unique(y[(y >= 0) & (y < self.num_tx)]):
            tx_i = int(tx.item())
            tx_mask = y == tx_i
            for domain in torch.unique(dom[tx_mask & (dom >= 0) & (dom < self.num_domains)]):
                d_i = int(domain.item())
                cell = tx_mask & (dom == d_i)
                if int(cell.sum().item()) < max(1, int(self.min_count_per_update)):
                    continue
                center = _normalize(z_norm[cell].mean(dim=0, keepdim=True), dim=1).squeeze(0)
                if int(self.counts[tx_i, d_i].item()) <= 0:
                    self.prototypes[tx_i, d_i].copy_(center)
                else:
                    self.prototypes[tx_i, d_i].mul_(m).add_(center, alpha=1.0 - m)
                    self.prototypes[tx_i, d_i].copy_(
                        _normalize(self.prototypes[tx_i, d_i].view(1, -1), dim=1).squeeze(0)
                    )
                self.counts[tx_i, d_i] += int(cell.sum().item())
                updates += 1
        return {"updated": float(updates), "initialized": float(int(self.initialized_mask().sum().item()))}

    def compute_domain_shifts(self, tx_bank: BalancedPrototypeBank) -> Dict[str, torch.Tensor]:
        tx_proto = tx_bank.get().to(device=self.prototypes.device, dtype=self.prototypes.dtype)
        local = _normalize(self.prototypes, dim=2)
        mask = self.initialized_mask().to(local.device)
        delta = local - tx_proto[:, None, :]
        shifts = torch.zeros(self.num_domains, self.feat_dim, device=local.device, dtype=local.dtype)
        domain_counts = torch.zeros(self.num_domains, device=local.device, dtype=torch.long)
        for domain in range(self.num_domains):
            active = mask[:, domain]
            if bool(active.any()):
                shifts[domain] = delta[active, domain].mean(dim=0)
                domain_counts[domain] = int(active.sum().item())
        interaction = delta - shifts[None, :, :]
        return {
            "delta": delta,
            "domain_shift": shifts,
            "interaction": interaction,
            "mask": mask,
            "domain_counts": domain_counts,
        }


class PrototypeRadiusTracker:
    """Bounded per-class angular radius tracker."""

    def __init__(self, num_classes: int, *, max_samples_per_class: int = 4096):
        self.num_classes = int(num_classes)
        self.max_samples_per_class = int(max_samples_per_class)
        self._angles: Dict[int, list[float]] = defaultdict(list)

    @torch.no_grad()
    def update(self, z_tx: torch.Tensor, y_tx: torch.Tensor, tx_proto: torch.Tensor) -> Dict[str, float]:
        z = _normalize(z_tx.detach(), dim=1)
        proto = _normalize(tx_proto.detach().to(device=z.device, dtype=z.dtype), dim=1)
        y = y_tx.view(-1).long().to(z.device)
        updates = 0
        for cls in torch.unique(y[(y >= 0) & (y < self.num_classes)]):
            cls_i = int(cls.item())
            m = y == cls_i
            if not bool(m.any()):
                continue
            cos = (z[m] * proto[cls_i].view(1, -1)).sum(dim=1).clamp(-1.0, 1.0)
            angles = torch.arccos(cos).detach().cpu().tolist()
            store = self._angles[cls_i]
            store.extend(float(v) for v in angles if math.isfinite(float(v)))
            if len(store) > self.max_samples_per_class:
                del store[: len(store) - self.max_samples_per_class]
            updates += len(angles)
        return {"updated_samples": float(updates), "tracked_classes": float(len(self._angles))}

    def radius(self, class_id: int, quantile: str | float = "p95") -> float:
        vals = self._angles.get(int(class_id), [])
        if not vals:
            return float("nan")
        q_name = str(quantile).lower()
        if q_name == "p95":
            q = 0.95
        elif q_name == "p99":
            q = 0.99
        elif q_name in ("max", "p100"):
            q = 1.0
        else:
            q = float(quantile)
        t = torch.tensor(vals, dtype=torch.float32)
        return float(torch.quantile(t, max(0.0, min(1.0, q))).item())

    def sigma(self, class_id: int) -> float:
        vals = self._angles.get(int(class_id), [])
        if len(vals) <= 1:
            return float("nan")
        return float(torch.tensor(vals, dtype=torch.float32).std(unbiased=False).item())

    def radii_tensor(self, *, quantile: str | float = "p95", device=None) -> torch.Tensor:
        vals = [self.radius(i, quantile=quantile) for i in range(self.num_classes)]
        return torch.tensor(vals, dtype=torch.float32, device=device)

    def sigma_tensor(self, *, device=None) -> torch.Tensor:
        vals = [self.sigma(i) for i in range(self.num_classes)]
        return torch.tensor(vals, dtype=torch.float32, device=device)


def prototype_geometry_summary(
    prototypes: torch.Tensor,
    radii: torch.Tensor,
    *,
    gamma_open_rad: float = 0.087,
    initialized: Optional[torch.Tensor] = None,
) -> PrototypeGeometrySummary:
    proto = _normalize(prototypes, dim=1)
    if initialized is None:
        initialized = torch.ones(proto.size(0), device=proto.device, dtype=torch.bool)
    initialized = initialized.to(device=proto.device).bool()
    active_idx = torch.where(initialized)[0]
    if active_idx.numel() == 0:
        return PrototypeGeometrySummary(0, float("nan"), float("nan"), 0)
    r = radii.to(device=proto.device, dtype=proto.dtype)
    active_r = r[active_idx]
    radius_mean = float(torch.nanmean(active_r).item()) if active_r.numel() else float("nan")
    if active_idx.numel() <= 1:
        return PrototypeGeometrySummary(int(active_idx.numel()), math.degrees(radius_mean), float("nan"), 0)
    P = proto[active_idx]
    sim = (P @ P.t()).clamp(-1.0, 1.0)
    angles = torch.arccos(sim)
    iu = torch.triu_indices(angles.size(0), angles.size(1), offset=1, device=angles.device)
    pair_angles = angles[iu[0], iu[1]]
    rsum = active_r[iu[0]] + active_r[iu[1]]
    safety = pair_angles - rsum
    violations = int((safety <= float(gamma_open_rad)).sum().item())
    return PrototypeGeometrySummary(
        initialized=int(active_idx.numel()),
        radius_p95_mean_deg=math.degrees(radius_mean) if math.isfinite(radius_mean) else float("nan"),
        min_interclass_angle_deg=math.degrees(float(pair_angles.min().item())),
        margin_violation_pairs=violations,
    )


def _select_phase2_feature(out: Mapping[str, Any], feature_key: str) -> torch.Tensor:
    key = str(feature_key or "z_id").strip()
    if key in out and torch.is_tensor(out[key]):
        return out[key]
    name = key.lower()
    if name in ("id_feat_joint", "feat_joint", "joint"):
        return get_nested_tensor(dict(out), "id_feat_joint", "aux_id", "feat_joint")
    if name in ("id_feat_pa", "feat_pa", "pa"):
        return get_nested_tensor(dict(out), "id_feat_pa", "aux_id", "feat_pa")
    if name in ("id_feat_dac", "feat_dac", "dac"):
        return get_nested_tensor(dict(out), "id_feat_dac", "aux_id", "feat_dac")
    raise KeyError(f"Phase2 feature key not found in model output: {feature_key}")


@torch.no_grad()
def extract_phase2_features(
    model,
    loader: Iterable,
    *,
    device=None,
    feature_key: str = "z_id",
    max_batches: int = 0,
    grl_lambda: float = 1.0,
) -> Dict[str, torch.Tensor | str]:
    """Extract Phase2 features through the existing CVS auxiliary forward path.

    This intentionally reuses ``unpack_batch`` and ``extract_domain_from_extra``
    from the training/evaluation stack so Phase2 export follows the same batch
    contract as ``train.py``. The model is always called with ``return_aux=True``;
    ``return_aux=False`` does not guarantee that ``z_id`` exists.
    """

    dev = torch.device(device) if device is not None else next(model.parameters()).device
    was_training = bool(getattr(model, "training", False))
    model.eval()
    feats = []
    labels = []
    domains = []
    try:
        for batch_idx, batch in enumerate(loader):
            if int(max_batches) > 0 and batch_idx >= int(max_batches):
                break
            x, y, extra = unpack_batch(batch)
            x = x.to(dev, non_blocking=True)
            y = y.to(dev, non_blocking=True).view(-1).long()
            d = extract_domain_from_extra(extra, dev)
            out = model(x, y_tx=y, grl_lambda=float(grl_lambda), return_aux=True, domain_labels=d)
            z = _select_phase2_feature(out, feature_key)
            if z.ndim != 2:
                raise ValueError(f"Phase2 feature {feature_key} must have shape [N, D], got {tuple(z.shape)}")
            if z.size(0) != y.numel():
                raise ValueError("Phase2 feature batch size must match labels")
            if d is None:
                d_cpu = torch.full((y.numel(),), -1, dtype=torch.long)
            else:
                d_cpu = d.detach().view(-1).long().cpu()
                if d_cpu.numel() != y.numel():
                    raise ValueError("domain_labels must have one value per sample")
            feats.append(z.detach().float().cpu())
            labels.append(y.detach().cpu())
            domains.append(d_cpu)
    finally:
        if was_training:
            model.train()
    if not feats:
        raise ValueError("No batches were available for Phase2 feature extraction")
    return {
        "features": torch.cat(feats, dim=0),
        "labels": torch.cat(labels, dim=0).long(),
        "domains": torch.cat(domains, dim=0).long(),
        "feature_key": str(feature_key),
    }


def _valid_label_count(labels: torch.Tensor) -> int:
    valid = labels.view(-1).long()
    valid = valid[valid >= 0]
    if valid.numel() == 0:
        raise ValueError("Phase2 prototype export requires at least one non-negative TX label")
    return int(valid.max().item()) + 1


def _valid_domain_count(domains: Optional[torch.Tensor]) -> int:
    if domains is None:
        return 0
    valid = domains.view(-1).long()
    valid = valid[valid >= 0]
    return int(valid.max().item()) + 1 if valid.numel() else 0


def _as_jsonable(obj: Any) -> Any:
    if torch.is_tensor(obj):
        return obj.detach().cpu().tolist()
    if isinstance(obj, PrototypeGeometrySummary):
        return {
            "initialized": obj.initialized,
            "radius_p95_mean_deg": obj.radius_p95_mean_deg,
            "min_interclass_angle_deg": obj.min_interclass_angle_deg,
            "margin_violation_pairs": obj.margin_violation_pairs,
        }
    if isinstance(obj, Mapping):
        return {str(k): _as_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_as_jsonable(v) for v in obj]
    return obj


def build_phase2_prototype_export(
    features: torch.Tensor,
    labels: torch.Tensor,
    domains: Optional[torch.Tensor] = None,
    *,
    feature_key: str = "z_id",
    metadata: Optional[Mapping[str, Any]] = None,
    min_count_per_update: int = 1,
    max_samples_per_class: int = 4096,
    robust_sigma_scale: float = 1.0,
) -> Dict[str, Any]:
    """Build an offline Phase2 prototype package from extracted ``z_id`` features.

    ``features`` are used for TX identity geometry. Domain labels only balance
    source prototype updates and build ``P_tx_dom`` diagnostics; ``z_dom`` is not
    part of the TX distance surface.
    """

    if features.ndim != 2:
        raise ValueError("features must have shape [N, D]")
    y = labels.view(-1).long()
    if y.numel() != features.size(0):
        raise ValueError("labels must have one value per feature")
    d = domains.view(-1).long() if domains is not None else None
    if d is not None and d.numel() != features.size(0):
        raise ValueError("domains must have one value per feature")

    feat = features.detach().float().cpu()
    y_cpu = y.detach().cpu()
    d_cpu = d.detach().cpu() if d is not None else None
    num_tx = _valid_label_count(y_cpu)
    num_domains = _valid_domain_count(d_cpu)
    feat_dim = int(feat.size(1))

    tx_bank = BalancedPrototypeBank(
        num_items=num_tx,
        feat_dim=feat_dim,
        momentum=0.0,
        min_count_per_update=min_count_per_update,
    )
    group_labels = d_cpu if num_domains > 0 else None
    tx_stats = tx_bank.update_from_features(feat, y_cpu, group_labels=group_labels)

    if num_domains > 0:
        tx_domain_bank = TxDomainPrototypeBank(
            num_tx=num_tx,
            num_domains=num_domains,
            feat_dim=feat_dim,
            momentum=0.0,
            min_count_per_update=1,
        )
        tx_domain_stats = tx_domain_bank.update(feat, y_cpu, d_cpu)
        tx_domain_prototypes = tx_domain_bank.prototypes.detach().clone()
        tx_domain_counts = tx_domain_bank.counts.detach().clone()
        domain_shifts = tx_domain_bank.compute_domain_shifts(tx_bank)
    else:
        tx_domain_stats = {"updated": 0.0, "initialized": 0.0}
        tx_domain_prototypes = torch.zeros(num_tx, 0, feat_dim, dtype=feat.dtype)
        tx_domain_counts = torch.zeros(num_tx, 0, dtype=torch.long)
        domain_shifts = {}

    tracker = PrototypeRadiusTracker(num_classes=num_tx, max_samples_per_class=max_samples_per_class)
    tracker.update(feat, y_cpu, tx_bank.get())
    radii_p95 = tracker.radii_tensor(quantile="p95")
    radii_p99 = tracker.radii_tensor(quantile="p99")
    radii_max = tracker.radii_tensor(quantile="max")
    sigma = tracker.sigma_tensor()
    robust = torch.minimum(
        radii_max,
        torch.nan_to_num(radii_p99, nan=0.0) + float(robust_sigma_scale) * torch.nan_to_num(sigma, nan=0.0),
    )
    geometry = prototype_geometry_summary(
        tx_bank.get(),
        radii_p95,
        initialized=tx_bank.initialized_mask(),
    )

    meta = dict(metadata or {})
    meta.update(
        {
            "feature_key": str(feature_key),
            "num_samples": int(feat.size(0)),
            "num_tx": int(num_tx),
            "num_domains": int(num_domains),
            "feat_dim": int(feat_dim),
            "default_training_behavior_changed": False,
        }
    )
    return {
        "schema_version": 1,
        "feature_key": str(feature_key),
        "prototypes": tx_bank.get().detach().clone(),
        "prototype_counts": tx_bank.counts.detach().clone(),
        "tx_domain_prototypes": tx_domain_prototypes,
        "tx_domain_counts": tx_domain_counts,
        "domain_shifts": {k: v.detach().clone() if torch.is_tensor(v) else v for k, v in domain_shifts.items()},
        "radii": {
            "p95": radii_p95,
            "p99": radii_p99,
            "max": radii_max,
            "robust_max": robust,
        },
        "radius_sigma": sigma,
        "geometry": _as_jsonable(geometry),
        "stats": {
            "tx_bank": tx_stats,
            "tx_domain_bank": tx_domain_stats,
        },
        "metadata": meta,
    }


def save_phase2_prototype_export(package: Mapping[str, Any], output_path: str | Path) -> Dict[str, str]:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(package), out)
    sidecar = out.with_suffix(".json")
    with sidecar.open("w", encoding="utf-8") as f:
        json.dump(_as_jsonable(package), f, ensure_ascii=False, indent=2, sort_keys=True)
    return {"pt_path": str(out), "json_path": str(sidecar)}


def export_phase2_prototypes(
    model,
    loader: Iterable,
    *,
    output_path: str | Path | None = None,
    device=None,
    feature_key: str = "z_id",
    metadata: Optional[Mapping[str, Any]] = None,
    max_batches: int = 0,
    grl_lambda: float = 1.0,
) -> Dict[str, Any]:
    extracted = extract_phase2_features(
        model,
        loader,
        device=device,
        feature_key=feature_key,
        max_batches=max_batches,
        grl_lambda=grl_lambda,
    )
    package = build_phase2_prototype_export(
        extracted["features"],
        extracted["labels"],
        extracted["domains"],
        feature_key=str(extracted["feature_key"]),
        metadata=metadata,
    )
    if output_path is not None and str(output_path).strip() != "":
        paths = save_phase2_prototype_export(package, output_path)
        package = dict(package)
        package["paths"] = paths
    return package
