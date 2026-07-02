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


@dataclass(frozen=True)
class PrototypeFusionConfig:
    max_components_per_tx: int = 4
    merge_angle_deg: float = 6.0
    radius_cap_deg: float = 25.0
    tail_abs_deg: float = 30.0
    accept_policy: str = "local_component"
    accept_radius_key: str = "p95"
    max_p95_increase_deg: float = 2.0
    keep_tail_sentinel: bool = True
    global_ball_accept: bool = False
    tail_auto_accept: bool = False
    min_count: int = 1
    eps: float = 1e-6


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
        if q_name == "p50":
            q = 0.50
        elif q_name == "p80":
            q = 0.80
        elif q_name == "p90":
            q = 0.90
        elif q_name == "p95":
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

    def robust_stats(self, class_id: int) -> Dict[str, float]:
        vals = self._angles.get(int(class_id), [])
        if not vals:
            return {
                "median": float("nan"),
                "mad": float("nan"),
                "iqr": float("nan"),
                "robust_sigma": float("nan"),
                "r_1sigma": float("nan"),
                "r_2sigma": float("nan"),
                "r_3sigma": float("nan"),
                "tail_count_gt_3sigma": 0.0,
                "tail_frac_gt_3sigma": float("nan"),
                "p50": float("nan"),
                "p80": float("nan"),
                "p90": float("nan"),
                "p95": float("nan"),
                "p99": float("nan"),
                "max": float("nan"),
            }
        t = torch.tensor(vals, dtype=torch.float32)
        median = torch.quantile(t, 0.50)
        mad = torch.quantile((t - median).abs(), 0.50)
        q25 = torch.quantile(t, 0.25)
        q75 = torch.quantile(t, 0.75)
        iqr = q75 - q25
        mad_sigma = 1.4826 * mad
        iqr_sigma = 0.7413 * iqr
        if float(mad_sigma.item()) > 0.0:
            robust_sigma = mad_sigma
        elif float(iqr_sigma.item()) > 0.0:
            robust_sigma = iqr_sigma
        else:
            robust_sigma = t.std(unbiased=False) if t.numel() > 1 else t.new_tensor(0.0)
        p95 = torch.quantile(t, 0.95)
        p99 = torch.quantile(t, 0.99)
        max_v = torch.quantile(t, 1.0)
        r1 = median + robust_sigma
        r2 = median + 2.0 * robust_sigma
        r3 = torch.minimum(max_v, median + 3.0 * robust_sigma)
        tail = t > r3
        return {
            "median": float(median.item()),
            "mad": float(mad.item()),
            "iqr": float(iqr.item()),
            "robust_sigma": float(robust_sigma.item()),
            "r_1sigma": float(torch.minimum(max_v, r1).item()),
            "r_2sigma": float(torch.minimum(max_v, r2).item()),
            "r_3sigma": float(r3.item()),
            "tail_count_gt_3sigma": float(int(tail.sum().item())),
            "tail_frac_gt_3sigma": float(tail.float().mean().item()),
            "p50": float(torch.quantile(t, 0.50).item()),
            "p80": float(torch.quantile(t, 0.80).item()),
            "p90": float(torch.quantile(t, 0.90).item()),
            "p95": float(p95.item()),
            "p99": float(p99.item()),
            "max": float(max_v.item()),
        }

    def radii_tensor(self, *, quantile: str | float = "p95", device=None) -> torch.Tensor:
        vals = [self.radius(i, quantile=quantile) for i in range(self.num_classes)]
        return torch.tensor(vals, dtype=torch.float32, device=device)

    def sigma_tensor(self, *, device=None) -> torch.Tensor:
        vals = [self.sigma(i) for i in range(self.num_classes)]
        return torch.tensor(vals, dtype=torch.float32, device=device)

    def robust_stats_tensor(self, key: str, *, device=None) -> torch.Tensor:
        vals = [self.robust_stats(i).get(str(key), float("nan")) for i in range(self.num_classes)]
        return torch.tensor(vals, dtype=torch.float32, device=device)

    def robust_stats_table(self) -> Dict[str, list[float]]:
        keys = [
            "median",
            "mad",
            "iqr",
            "robust_sigma",
            "r_1sigma",
            "r_2sigma",
            "r_3sigma",
            "tail_count_gt_3sigma",
            "tail_frac_gt_3sigma",
            "p50",
            "p80",
            "p90",
            "p95",
            "p99",
            "max",
        ]
        return {key: [self.robust_stats(i).get(key, float("nan")) for i in range(self.num_classes)] for key in keys}


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


def _angle_between(a: torch.Tensor, b: torch.Tensor) -> float:
    aa = _normalize(a.view(1, -1), dim=1).squeeze(0)
    bb = _normalize(b.view(1, -1), dim=1).squeeze(0)
    cos = float((aa * bb).sum().clamp(-1.0, 1.0).item())
    return float(math.acos(cos))


def _component_center(vectors: torch.Tensor, counts: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    weights = torch.sqrt(torch.clamp(counts.float(), min=1.0)).to(device=vectors.device, dtype=vectors.dtype)
    center = (vectors * weights.view(-1, 1)).sum(dim=0, keepdim=True) / weights.sum().clamp_min(float(eps))
    return _normalize(center, dim=1).squeeze(0)


def fuse_tx_domain_prototypes(package: Mapping[str, Any], config: PrototypeFusionConfig | None = None) -> Dict[str, Any]:
    """Compress redundant per-domain TX prototypes into local angular components.

    The fusion operates only on exported ``z_id`` prototype tensors. Domain ids
    are used as grouping metadata; no ``z_dom`` representation is read.
    """
    cfg = config or PrototypeFusionConfig()
    tx_domain = package.get("tx_domain_prototypes")
    tx_counts = package.get("tx_domain_counts")
    global_proto = package.get("prototypes")
    if not torch.is_tensor(tx_domain) or tx_domain.ndim != 3:
        raise ValueError("package must contain tx_domain_prototypes with shape [num_tx, num_domains, feat_dim]")
    if not torch.is_tensor(tx_counts) or tx_counts.shape[:2] != tx_domain.shape[:2]:
        raise ValueError("package must contain tx_domain_counts matching tx_domain_prototypes")
    if not torch.is_tensor(global_proto) or global_proto.ndim != 2 or global_proto.size(0) != tx_domain.size(0):
        raise ValueError("package must contain prototypes with shape [num_tx, feat_dim]")

    local = _normalize(tx_domain.detach().float().cpu(), dim=2)
    counts = tx_counts.detach().long().cpu()
    gproto = _normalize(global_proto.detach().float().cpu(), dim=1)
    num_tx, _num_domains, feat_dim = local.shape
    max_components = max(1, int(cfg.max_components_per_tx))
    merge_angle = math.radians(max(0.0, float(cfg.merge_angle_deg)))
    tail_abs = math.radians(max(0.0, float(cfg.tail_abs_deg)))
    cap = math.radians(max(0.0, float(cfg.radius_cap_deg)))
    min_count = max(1, int(cfg.min_count))

    radii_obj = package.get("radii", {}) if isinstance(package.get("radii", {}), Mapping) else {}
    base_radius = radii_obj.get("r_3sigma", radii_obj.get("p99", radii_obj.get("p95")))
    accept_base = radii_obj.get(str(cfg.accept_radius_key), radii_obj.get("p95", base_radius))
    if torch.is_tensor(base_radius) and base_radius.numel() >= num_tx:
        base_radius_t = base_radius.detach().float().cpu()
    else:
        base_radius_t = torch.zeros(num_tx, dtype=torch.float32)
    if torch.is_tensor(accept_base) and accept_base.numel() >= num_tx:
        accept_base_t = accept_base.detach().float().cpu()
    else:
        accept_base_t = base_radius_t.clone()

    fused_proto = torch.zeros(num_tx, max_components, feat_dim, dtype=local.dtype)
    fused_radii = torch.zeros(num_tx, max_components, dtype=torch.float32)
    fused_accept_radii = torch.zeros(num_tx, max_components, dtype=torch.float32)
    fused_evidence_radii = torch.zeros(num_tx, max_components, dtype=torch.float32)
    fused_counts = torch.zeros(num_tx, max_components, dtype=torch.long)
    fused_mask = torch.zeros(num_tx, max_components, dtype=torch.bool)
    domain_to_component = torch.full((num_tx, local.size(1)), -1, dtype=torch.long)
    components: list[list[Dict[str, Any]]] = []

    for tx in range(num_tx):
        active_domains = [int(i) for i in torch.where(counts[tx] >= min_count)[0].tolist()]
        if not active_domains:
            components.append([])
            continue
        parent = {d: d for d in active_domains}

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra = find(a)
            rb = find(b)
            if ra != rb:
                parent[rb] = ra

        for i, dom_i in enumerate(active_domains):
            for dom_j in active_domains[i + 1:]:
                angle = _angle_between(local[tx, dom_i], local[tx, dom_j])
                if angle <= merge_angle:
                    union(dom_i, dom_j)
        grouped: Dict[int, list[int]] = defaultdict(list)
        for dom in active_domains:
            grouped[find(dom)].append(dom)

        comp_rows = []
        for domains in grouped.values():
            doms = sorted(int(d) for d in domains)
            vecs = local[tx, doms]
            cts = counts[tx, doms]
            center = _component_center(vecs, cts, eps=float(cfg.eps))
            domain_angles = [_angle_between(local[tx, d], center) for d in doms]
            global_angle = _angle_between(center, gproto[tx])
            tail_sentinel = bool(global_angle > tail_abs)
            max_domain_angle = max(domain_angles) if domain_angles else 0.0
            evidence_radius = min(cap, float(base_radius_t[tx].item()) + max_domain_angle)
            accept_radius = min(cap, float(accept_base_t[tx].item()), float(evidence_radius))
            p95_delta = max(0.0, float(evidence_radius) - float(accept_base_t[tx].item()))
            over_fused = bool(math.degrees(p95_delta) > float(cfg.max_p95_increase_deg))
            comp_rows.append(
                {
                    "domains": doms,
                    "count": int(cts.sum().item()),
                    "center": center,
                    "radius": float(evidence_radius),
                    "evidence_radius": float(evidence_radius),
                    "accept_radius": float(accept_radius),
                    "p95_delta": float(p95_delta),
                    "over_fused": bool(over_fused),
                    "tail_sentinel": tail_sentinel,
                    "global_angle": float(global_angle),
                    "max_domain_angle": float(max_domain_angle),
                }
            )
        comp_rows.sort(key=lambda row: (bool(row["tail_sentinel"]), -int(row["count"]), row["domains"][0]))
        kept = comp_rows[:max_components]
        if bool(cfg.keep_tail_sentinel):
            kept_ids = {tuple(row["domains"]) for row in kept}
            tail_rows = [row for row in comp_rows if bool(row["tail_sentinel"])]
            if tail_rows and not any(bool(row["tail_sentinel"]) for row in kept):
                kept[-1] = tail_rows[0]
            elif tail_rows:
                for tail_row in tail_rows:
                    if tuple(tail_row["domains"]) not in kept_ids and len(kept) < max_components:
                        kept.append(tail_row)
        tx_components = []
        for comp_idx, row in enumerate(kept):
            fused_proto[tx, comp_idx] = row["center"]
            fused_radii[tx, comp_idx] = float(row["radius"])
            fused_evidence_radii[tx, comp_idx] = float(row["evidence_radius"])
            fused_accept_radii[tx, comp_idx] = float(row["accept_radius"])
            fused_counts[tx, comp_idx] = int(row["count"])
            fused_mask[tx, comp_idx] = True
            for dom in row["domains"]:
                domain_to_component[tx, int(dom)] = int(comp_idx)
            tx_components.append(
                {
                    "component_id": int(comp_idx),
                    "source_domains": list(row["domains"]),
                    "n_samples": int(row["count"]),
                    "mu": row["center"].detach().clone(),
                    "r_core_deg": math.degrees(float(row["accept_radius"])) * 0.8,
                    "r_accept_deg": math.degrees(float(row["accept_radius"])),
                    "r_tail_deg": math.degrees(float(row["radius"])),
                    "r_vac_deg": max(math.degrees(float(row["radius"])), math.degrees(float(row["accept_radius"])) + 4.0),
                    "density_p05": None,
                    "density_p10": None,
                    "nll_p95": None,
                    "nll_tail_p95": None,
                    "nearest_other_deg": None,
                    "accept_enabled": not bool(row["tail_sentinel"]),
                    "domains": list(row["domains"]),
                    "count": int(row["count"]),
                    "radius_deg": math.degrees(float(row["radius"])),
                    "component_evidence_radius_deg": math.degrees(float(row["evidence_radius"])),
                    "component_accept_radius_deg": math.degrees(float(row["accept_radius"])),
                    "evidence_radius_deg": math.degrees(float(row["evidence_radius"])),
                    "accept_radius_deg": math.degrees(float(row["accept_radius"])),
                    "component_p95_delta_deg": math.degrees(float(row["p95_delta"])),
                    "over_fused": bool(row["over_fused"]),
                    "tail_sentinel": bool(row["tail_sentinel"]),
                    "component_tail_sentinel": bool(row["tail_sentinel"]),
                    "global_angle_deg": math.degrees(float(row["global_angle"])),
                    "max_domain_angle_deg": math.degrees(float(row["max_domain_angle"])),
                }
            )
        components.append(tx_components)

    active_centers: list[tuple[int, int, torch.Tensor]] = []
    for tx in range(num_tx):
        for comp_idx in range(max_components):
            if bool(fused_mask[tx, comp_idx].item()):
                active_centers.append((tx, comp_idx, fused_proto[tx, comp_idx]))
    for tx, comp_idx, center in active_centers:
        nearest = None
        for other_tx, _other_idx, other_center in active_centers:
            if int(other_tx) == int(tx):
                continue
            angle = math.degrees(_angle_between(center, other_center))
            nearest = angle if nearest is None else min(nearest, angle)
        if comp_idx < len(components[tx]):
            components[tx][comp_idx]["nearest_other_deg"] = nearest

    fused_package = dict(package)
    fused_package.update(
        {
            "fused_tx_prototypes": fused_proto,
            "fused_tx_radii": fused_radii,
            "fused_tx_accept_radii": fused_accept_radii,
            "fused_tx_evidence_radii": fused_evidence_radii,
            "fused_tx_counts": fused_counts,
            "fused_tx_mask": fused_mask,
            "fusion_components": components,
            "domain_to_fused_component": domain_to_component,
            "fusion_accept_policy": str(cfg.accept_policy),
            "global_fused_radius_is_accept_region": bool(cfg.global_ball_accept),
            "fusion_config": {
                "enabled": True,
                "method": "tx_domain_to_local_components",
                "mode": "tx_domain_to_local_components",
                "accept_policy": str(cfg.accept_policy),
                "global_ball_accept": bool(cfg.global_ball_accept),
                "tail_auto_accept": False,
                "tail_auto_accept_requested": bool(cfg.tail_auto_accept),
                "tail_auto_accept_effective": False,
                "feature_key": str(package.get("feature_key", "z_id")),
                "max_components_per_class": int(max_components),
                "max_components_per_tx": int(max_components),
                "min_component_samples": int(min_count),
                "radius_quantile_core": 0.80,
                "accept_radius_key": str(cfg.accept_radius_key),
                "keep_tail_sentinel": bool(cfg.keep_tail_sentinel),
            },
            "fusion_metadata": {
                "schema": "tx_domain_prototype_fusion_v2",
                "feature_key": str(package.get("feature_key", "z_id")),
                "max_components_per_tx": int(max_components),
                "merge_angle_deg": float(cfg.merge_angle_deg),
                "radius_cap_deg": float(cfg.radius_cap_deg),
                "tail_abs_deg": float(cfg.tail_abs_deg),
                "accept_policy": str(cfg.accept_policy),
                "accept_radius_key": str(cfg.accept_radius_key),
                "max_p95_increase_deg": float(cfg.max_p95_increase_deg),
                "keep_tail_sentinel": bool(cfg.keep_tail_sentinel),
                "global_fused_radius_is_accept_region": bool(cfg.global_ball_accept),
                "tail_auto_accept": False,
                "tail_auto_accept_requested": bool(cfg.tail_auto_accept),
                "tail_auto_accept_effective": False,
                "fused_tx_radii_semantics": "legacy_evidence_radius_not_accept_region",
                "default_training_behavior_changed": False,
            },
        }
    )
    return fused_package


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
    radii_p50 = tracker.radii_tensor(quantile="p50")
    radii_p80 = tracker.radii_tensor(quantile="p80")
    radii_p90 = tracker.radii_tensor(quantile="p90")
    radii_p99 = tracker.radii_tensor(quantile="p99")
    radii_max = tracker.radii_tensor(quantile="max")
    sigma = tracker.sigma_tensor()
    robust_sigma = tracker.robust_stats_tensor("robust_sigma")
    r_1sigma = tracker.robust_stats_tensor("r_1sigma")
    r_2sigma = tracker.robust_stats_tensor("r_2sigma")
    r_3sigma = tracker.robust_stats_tensor("r_3sigma")
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
            "p50": radii_p50,
            "p80": radii_p80,
            "p90": radii_p90,
            "p95": radii_p95,
            "p99": radii_p99,
            "max": radii_max,
            "robust_max": robust,
            "r_1sigma": r_1sigma,
            "r_2sigma": r_2sigma,
            "r_3sigma": r_3sigma,
        },
        "radius_sigma": sigma,
        "radius_robust_sigma": robust_sigma,
        "radius_tail_stats": tracker.robust_stats_table(),
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
