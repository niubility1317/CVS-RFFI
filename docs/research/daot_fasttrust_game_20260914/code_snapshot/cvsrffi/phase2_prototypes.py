from __future__ import annotations

import json
import hashlib
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


class Phase1CalibrationError(ValueError):
    """A source-calibration failure with one non-aliased machine code."""

    def __init__(self, code: str, details: Mapping[str, Any]):
        self.code = str(code)
        self.details = dict(details)
        super().__init__(f"{self.code}: {json.dumps(self.details, sort_keys=True)}")


def audit_identity_feature_contract(
    z_id: torch.Tensor,
    feat_joint: torch.Tensor,
    labels: torch.Tensor,
    domains: torch.Tensor,
    logits: torch.Tensor,
    *,
    expected_classes: int,
    min_class_samples: int = 4,
) -> Dict[str, Any]:
    """Audit the classification and export identity spaces before calibration."""

    expected = int(expected_classes)
    minimum = max(1, int(min_class_samples))
    y = torch.as_tensor(labels).detach().view(-1).long().cpu()
    d = torch.as_tensor(domains).detach().view(-1).long().cpu()
    logit = torch.as_tensor(logits).detach().float().cpu()
    spaces = {
        "z_id": torch.as_tensor(z_id).detach().float().cpu(),
        "feat_joint": torch.as_tensor(feat_joint).detach().float().cpu(),
    }
    details: Dict[str, Any] = {
        "class_count": expected,
        "sample_count": int(y.numel()),
        "logit_class_order": list(range(expected)),
    }
    if expected < 1 or logit.ndim != 2 or int(logit.shape[1]) != expected:
        details["logit_width"] = int(logit.shape[1]) if logit.ndim == 2 else -1
        raise Phase1CalibrationError("CLASS_ORDER_MISMATCH", details)
    if d.numel() != y.numel() or any(
        value.ndim != 2 or int(value.shape[0]) != y.numel()
        for value in spaces.values()
    ):
        details["domain_count"] = int(d.numel())
        raise Phase1CalibrationError("CLASS_ORDER_MISMATCH", details)
    if not torch.isfinite(logit).all() or any(
        not torch.isfinite(value).all() for value in spaces.values()
    ):
        raise Phase1CalibrationError("NONFINITE_FEATURE", details)

    class_counts = {
        str(class_id): int((y == class_id).sum().item())
        for class_id in range(expected)
    }
    details["class_counts"] = class_counts
    missing = [class_id for class_id in range(expected) if class_counts[str(class_id)] == 0]
    if missing:
        details["missing_classes"] = missing
        raise Phase1CalibrationError("MISSING_CLASS_IN_V_CAL", details)
    insufficient = {
        key: count for key, count in class_counts.items() if count < minimum
    }
    if insufficient:
        details["insufficient_classes"] = insufficient
        details["min_class_samples"] = minimum
        raise Phase1CalibrationError("INSUFFICIENT_CLASS_SAMPLES", details)

    space_audit: Dict[str, Any] = {}
    for name, value in spaces.items():
        norms = torch.linalg.vector_norm(value, dim=1)
        zero = norms <= 1e-8
        zero_count = int(zero.sum().item())
        audit = {
            "feature_dim": int(value.shape[1]),
            "zero_count": zero_count,
            "norm_min": float(norms.min().item()),
            "norm_p50": float(torch.quantile(norms, 0.50).item()),
            "norm_p95": float(torch.quantile(norms, 0.95).item()),
            "norm_max": float(norms.max().item()),
        }
        space_audit[name] = audit
        if zero_count:
            details["space"] = name
            details.update(audit)
            raise Phase1CalibrationError("ZERO_DIRECTION_FEATURE", details)

    centers = []
    normalized_z = F.normalize(spaces["z_id"], dim=1, eps=1e-8)
    for class_id in range(expected):
        centers.append(F.normalize(normalized_z[y == class_id].mean(0), dim=0, eps=1e-8))
    center_matrix = torch.stack(centers)
    similarity = center_matrix @ center_matrix.t()
    off_diagonal = similarity[~torch.eye(expected, dtype=torch.bool)]
    min_interclass_angle = (
        float(torch.rad2deg(torch.acos(off_diagonal.clamp(-1.0 + 1e-6, 1.0 - 1e-6))).min().item())
        if off_diagonal.numel()
        else 180.0
    )
    return {
        "status": "PASS",
        "class_count": expected,
        "class_counts": class_counts,
        "class_coverage_pass": True,
        "finite_feature_pass": True,
        "nonzero_direction_pass": True,
        "interclass_geometry_pass": min_interclass_angle > 0.0,
        "min_interclass_angle_deg": min_interclass_angle,
        "feature_key_contract_pass": True,
        "classification_feature_key": "feat_joint",
        "prototype_feature_key": "z_id",
        "open_set_geometry_feature_key": "z_id",
        "phase2_export_feature_key": "z_id",
        "runtime_inference_feature_key": "z_id",
        "spaces": space_audit,
        "domain_count": int(torch.unique(d[d >= 0]).numel()),
        "logit_class_order": list(range(expected)),
    }


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


def _weighted_quantile_1d(values: torch.Tensor, weights: torch.Tensor, q: float) -> float:
    values = torch.nan_to_num(values.detach().float().view(-1), nan=0.0, posinf=0.0, neginf=0.0)
    weights = torch.nan_to_num(weights.detach().float().view(-1), nan=0.0, posinf=0.0, neginf=0.0)
    if values.numel() == 0:
        return 0.0
    if weights.numel() != values.numel():
        weights = torch.ones_like(values)
    weights = weights.clamp_min(0.0)
    if float(weights.sum().item()) <= 0.0:
        weights = torch.ones_like(values)
    order = torch.argsort(values)
    sorted_values = values[order]
    sorted_weights = weights[order]
    total = sorted_weights.sum().clamp_min(1e-6)
    threshold = float(max(0.0, min(1.0, q))) * float(total.item())
    idx = int(torch.searchsorted(sorted_weights.cumsum(dim=0), torch.tensor(threshold, dtype=sorted_weights.dtype)).item())
    idx = max(0, min(idx, int(sorted_values.numel()) - 1))
    return float(sorted_values[idx].item())


def _component_density_nll_stats(domain_angles: Iterable[float], counts: torch.Tensor, accept_radius: float) -> Dict[str, float]:
    angles = torch.tensor([float(a) for a in domain_angles], dtype=torch.float32)
    if angles.numel() == 0:
        return {"density_p05": 1.0, "density_p10": 1.0, "nll_p95": 0.0, "nll_tail_p95": 0.0}
    weights = torch.clamp(counts.detach().float().cpu().view(-1), min=1.0)
    if weights.numel() != angles.numel():
        weights = torch.ones_like(angles)
    scale = max(float(accept_radius), math.radians(1.0), 1e-6)
    nll = (angles / scale).clamp_min(0.0).clamp_max(50.0)
    density = torch.exp(-nll).clamp_min(1e-12)
    tail_cut = _weighted_quantile_1d(angles, weights, 0.80)
    tail_mask = angles >= float(tail_cut)
    if bool(tail_mask.any()):
        tail_nll = nll[tail_mask]
        tail_weights = weights[tail_mask]
    else:
        tail_nll = nll
        tail_weights = weights
    return {
        "density_p05": _weighted_quantile_1d(density, weights, 0.05),
        "density_p10": _weighted_quantile_1d(density, weights, 0.10),
        "nll_p95": _weighted_quantile_1d(nll, weights, 0.95),
        "nll_tail_p95": _weighted_quantile_1d(tail_nll, tail_weights, 0.95),
    }


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
            density_stats = _component_density_nll_stats(domain_angles, cts, float(accept_radius))
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
                    "density_p05": float(density_stats["density_p05"]),
                    "density_p10": float(density_stats["density_p10"]),
                    "nll_p95": float(density_stats["nll_p95"]),
                    "nll_tail_p95": float(density_stats["nll_tail_p95"]),
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
                    "density_p05": float(row["density_p05"]),
                    "density_p10": float(row["density_p10"]),
                    "nll_p95": float(row["nll_p95"]),
                    "nll_tail_p95": float(row["nll_tail_p95"]),
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


@torch.no_grad()
def extract_endpoint_calibration_features(
    model,
    loader: Iterable,
    *,
    device=None,
    feature_key: str = "z_id",
    max_batches: int = 0,
    grl_lambda: float = 1.0,
    require_identity_contract: bool = False,
) -> Dict[str, torch.Tensor | str]:
    """Extract source-validation geometry and logits for endpoint calibration."""

    dev = torch.device(device) if device is not None else next(model.parameters()).device
    was_training = bool(getattr(model, "training", False))
    model.eval()
    feats, labels, domains, logits = [], [], [], []
    z_id_features, feat_joint_features = [], []
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
            if bool(require_identity_contract):
                z_id = _select_phase2_feature(out, "z_id")
                feat_joint = _select_phase2_feature(out, "feat_joint")
                if z_id.ndim != 2 or feat_joint.ndim != 2:
                    raise ValueError("identity feature contract requires rank-2 z_id and feat_joint")
                if z_id.size(0) != y.numel() or feat_joint.size(0) != y.numel():
                    raise ValueError("identity feature contract batch cardinality mismatch")
                z_id_features.append(z_id.detach().float().cpu())
                feat_joint_features.append(feat_joint.detach().float().cpu())
            tx_logits = out.get("tx_logits")
            if not torch.is_tensor(tx_logits) or tx_logits.ndim != 2:
                raise ValueError("endpoint_accept_v1 calibration requires rank-2 tx_logits")
            if z.ndim != 2 or z.size(0) != y.numel() or tx_logits.size(0) != y.numel():
                raise ValueError("endpoint_accept_v1 calibration feature/logit batch mismatch")
            d_cpu = (
                torch.full((y.numel(),), -1, dtype=torch.long)
                if d is None
                else d.detach().view(-1).long().cpu()
            )
            feats.append(z.detach().float().cpu())
            labels.append(y.detach().cpu())
            domains.append(d_cpu)
            logits.append(tx_logits.detach().float().cpu())
    finally:
        if was_training:
            model.train()
    if not feats:
        raise ValueError("No source-validation batches were available for endpoint calibration")
    result: Dict[str, torch.Tensor | str | bool] = {
        "features": torch.cat(feats, dim=0),
        "labels": torch.cat(labels, dim=0).long(),
        "domains": torch.cat(domains, dim=0).long(),
        "logits": torch.cat(logits, dim=0).float(),
        "feature_key": str(feature_key),
        "identity_feature_contract_required": bool(require_identity_contract),
    }
    if bool(require_identity_contract):
        result["z_id_features"] = torch.cat(z_id_features, dim=0)
        result["feat_joint_features"] = torch.cat(feat_joint_features, dim=0)
    return result


def _finite_quantile(values: torch.Tensor, q: float) -> float:
    finite = values.detach().float().view(-1)
    finite = finite[torch.isfinite(finite)]
    if finite.numel() == 0:
        return float("nan")
    return float(torch.quantile(finite, max(0.0, min(1.0, float(q)))).item())


def calibrate_endpoint_accept_v1(
    package: Mapping[str, Any],
    calibration_features: torch.Tensor,
    calibration_labels: torch.Tensor,
    calibration_logits: torch.Tensor,
    *,
    min_component_samples: int = 4,
    min_class_samples: int = 4,
    core_quantile: float = 0.80,
    accept_quantile: float = 0.95,
    tail_quantile: float = 0.99,
    max_zero_direction_fraction: float = 0.001,
) -> Dict[str, Any]:
    """Calibrate one deterministic local-component endpoint on source-val only."""

    out = dict(package)
    fused = torch.as_tensor(out.get("fused_tx_prototypes")).detach().float().cpu()
    mask = torch.as_tensor(out.get("fused_tx_mask")).detach().bool().cpu()
    components = out.get("fusion_components")
    if fused.ndim != 3 or mask.shape != fused.shape[:2] or not isinstance(components, (list, tuple)):
        raise ValueError("endpoint_accept_v1 calibration requires fused local components")
    raw_z = calibration_features.detach().float().cpu()
    y = calibration_labels.detach().view(-1).long().cpu()
    logits = calibration_logits.detach().float().cpu()
    if raw_z.ndim != 2 or y.numel() != raw_z.size(0) or logits.ndim != 2 or logits.size(0) != raw_z.size(0):
        raise ValueError("endpoint_accept_v1 calibration tensors have incompatible shapes")
    if not torch.isfinite(raw_z).all():
        raise ValueError("endpoint_accept_v1 calibration features must be finite")
    if not torch.isfinite(logits).all():
        raise ValueError("endpoint_accept_v1 calibration logits must be finite")
    if logits.size(1) != fused.size(0):
        raise ValueError("endpoint_accept_v1 calibration logits must exactly match known-class order")
    if y.numel() == 0 or int(y.min().item()) < 0 or int(y.max().item()) >= int(fused.size(0)):
        raise ValueError("endpoint_accept_v1 calibration labels must match known-class order")
    max_zero_direction_fraction = float(max_zero_direction_fraction)
    if (
        not math.isfinite(max_zero_direction_fraction)
        or max_zero_direction_fraction < 0.0
        or max_zero_direction_fraction > 0.01
    ):
        raise ValueError("endpoint_accept_v1 max zero-direction fraction must be in [0,0.01]")
    input_num_samples = int(raw_z.size(0))
    feature_norms = torch.linalg.vector_norm(raw_z, dim=1)
    if not torch.isfinite(feature_norms).all():
        raise ValueError("endpoint_accept_v1 calibration feature norms must be finite")
    zero_direction = feature_norms <= 1e-8
    zero_direction_count = int(zero_direction.sum().item())
    zero_direction_fraction = (
        float(zero_direction_count) / float(input_num_samples)
        if input_num_samples > 0
        else 1.0
    )
    zero_direction_by_class: Dict[str, int] = {}
    zero_direction_fraction_by_class: Dict[str, float] = {}
    for class_id in range(fused.size(0)):
        class_mask = y == class_id
        class_count = int(class_mask.sum().item())
        class_zero_count = int((zero_direction & class_mask).sum().item())
        class_zero_fraction = (
            float(class_zero_count) / float(class_count)
            if class_count > 0
            else 1.0
        )
        zero_direction_by_class[str(class_id)] = class_zero_count
        zero_direction_fraction_by_class[str(class_id)] = class_zero_fraction
        if class_zero_fraction > max_zero_direction_fraction + 1e-12:
            raise ValueError(
                "endpoint_accept_v1 zero-direction fraction exceeds per-class limit: "
                f"class={class_id} fraction={class_zero_fraction:.9f} "
                f"limit={max_zero_direction_fraction:.9f}"
            )
    if zero_direction_fraction > max_zero_direction_fraction + 1e-12:
        raise ValueError(
            "endpoint_accept_v1 zero-direction fraction exceeds overall limit: "
            f"fraction={zero_direction_fraction:.9f} "
            f"limit={max_zero_direction_fraction:.9f}"
        )
    directional = ~zero_direction
    if not bool(directional.any()):
        raise ValueError("endpoint_accept_v1 calibration has no directional features")
    raw_z = raw_z[directional]
    y = y[directional]
    logits = logits[directional]
    z = F.normalize(raw_z, dim=1)

    calibrated_components = [[dict(row) for row in (rows or [])] for rows in components]
    accept_radii = torch.zeros_like(mask, dtype=torch.float32)
    evidence_radii = torch.zeros_like(mask, dtype=torch.float32)
    class_sample_counts: Dict[str, int] = {}
    component_sample_counts: Dict[str, int] = {}
    sample_own_distance = torch.full((z.size(0),), float("nan"), dtype=torch.float32)
    sample_geo_margin = torch.full((z.size(0),), float("nan"), dtype=torch.float32)
    sample_core = torch.zeros(z.size(0), dtype=torch.bool)
    enabled_by_class = [0 for _ in range(fused.size(0))]
    max_radius_to_inter_ratio = 0.50
    calibration_radius_guard_ratio = (
        max_radius_to_inter_ratio - 1e-6
    )
    interclass_guard_contracted_components = 0

    all_active_centers = []
    all_active_classes = []
    for class_id in range(fused.size(0)):
        for comp_id in torch.where(mask[class_id])[0].tolist():
            all_active_centers.append(_normalize(fused[class_id, comp_id].view(1, -1), dim=1).squeeze(0))
            all_active_classes.append(class_id)
    if not all_active_centers:
        raise ValueError("endpoint_accept_v1 calibration found no active component centers")
    all_centers = torch.stack(all_active_centers, dim=0)
    all_center_classes = torch.tensor(all_active_classes, dtype=torch.long)

    for class_id in range(fused.size(0)):
        sample_idx = torch.where(y == class_id)[0]
        class_sample_counts[str(class_id)] = int(sample_idx.numel())
        if sample_idx.numel() < max(1, int(min_class_samples)):
            raise ValueError(
                "endpoint_accept_v1 source-val calibration has "
                f"insufficient true-class samples for class {class_id}"
            )
        active_ids = torch.where(mask[class_id])[0]
        if active_ids.numel() == 0:
            for row in calibrated_components[class_id]:
                row["accept_enabled"] = False
                row["calibration_status"] = "insufficient_class_source_val"
            continue
        centers = _normalize(fused[class_id, active_ids], dim=1)
        class_z = z[sample_idx]
        angles = torch.rad2deg(torch.acos(torch.clamp(class_z @ centers.t(), -1.0 + 1e-6, 1.0 - 1e-6)))
        nearest_angle, nearest_pos = angles.min(dim=1)
        other_mask = all_center_classes != class_id
        if bool(other_mask.any()):
            other_angles = torch.rad2deg(
                torch.acos(torch.clamp(class_z @ all_centers[other_mask].t(), -1.0 + 1e-6, 1.0 - 1e-6))
            )
            nearest_other = other_angles.min(dim=1).values
            sample_geo_margin[sample_idx] = nearest_other - nearest_angle
        sample_own_distance[sample_idx] = nearest_angle

        row_by_id = {int(row.get("component_id", idx)): row for idx, row in enumerate(calibrated_components[class_id])}
        for local_pos, comp_id_obj in enumerate(active_ids.tolist()):
            comp_id = int(comp_id_obj)
            assigned = nearest_pos == int(local_pos)
            comp_angles = nearest_angle[assigned]
            count = int(comp_angles.numel())
            component_sample_counts[f"{class_id}:{comp_id}"] = count
            row = row_by_id.get(comp_id)
            if row is None:
                continue
            row["pre_calibration_r_core_deg"] = float(row.get("r_core_deg", 0.0))
            row["pre_calibration_r_accept_deg"] = float(row.get("r_accept_deg", row.get("accept_radius_deg", 0.0)))
            row["pre_calibration_r_tail_deg"] = float(row.get("r_tail_deg", row.get("radius_deg", 0.0)))
            row["source_val_count"] = count
            if count < max(1, int(min_component_samples)):
                row["accept_enabled"] = False
                row["calibration_status"] = "insufficient_component_source_val"
                continue
            raw_r_core = _finite_quantile(comp_angles, core_quantile)
            raw_r_accept = max(
                raw_r_core,
                _finite_quantile(comp_angles, accept_quantile),
            )
            raw_r_tail = max(
                raw_r_accept,
                _finite_quantile(comp_angles, tail_quantile),
            )
            if not all(
                math.isfinite(value)
                for value in (raw_r_core, raw_r_accept, raw_r_tail)
            ):
                row["accept_enabled"] = False
                row["calibration_status"] = "nonfinite_source_val_geometry"
                continue
            center = _normalize(
                fused[class_id, comp_id].view(1, -1),
                dim=1,
            ).squeeze(0)
            other_center_mask = all_center_classes != class_id
            if not bool(other_center_mask.any()):
                row["accept_enabled"] = False
                row["calibration_status"] = "missing_other_class_component"
                continue
            nearest_other_component_deg = math.degrees(
                float(
                    torch.acos(
                        torch.clamp(
                            all_centers[other_center_mask] @ center,
                            -1.0 + 1e-6,
                            1.0 - 1e-6,
                        )
                    )
                    .min()
                    .item()
                )
            )
            safe_accept_cap_deg = (
                calibration_radius_guard_ratio
                * nearest_other_component_deg
            )
            r_accept = min(raw_r_accept, safe_accept_cap_deg)
            r_core = min(raw_r_core, r_accept)
            r_tail = max(r_accept, raw_r_tail)
            contracted = raw_r_accept > safe_accept_cap_deg + 1e-8
            interclass_guard_contracted_components += int(contracted)
            density = torch.exp(-torch.square(comp_angles / max(r_accept, 1e-6)))
            nll = 0.5 * torch.square(comp_angles / max(r_accept, 1e-6))
            core_values = comp_angles <= r_core
            row.update(
                {
                    "r_core_deg": float(r_core),
                    "r_accept_deg": float(r_accept),
                    "r_tail_deg": float(r_tail),
                    "r_vac_deg": float(max(r_tail, r_accept + 4.0)),
                    "component_accept_radius_deg": float(r_accept),
                    "accept_radius_deg": float(r_accept),
                    "component_evidence_radius_deg": float(r_tail),
                    "evidence_radius_deg": float(r_tail),
                    "density_p05": _finite_quantile(density[core_values] if bool(core_values.any()) else density, 0.05),
                    "density_p10": _finite_quantile(density, 0.10),
                    "nll_p95": _finite_quantile(nll[core_values] if bool(core_values.any()) else nll, 0.95),
                    "nll_tail_p95": _finite_quantile(nll, 0.95),
                    "source_val_raw_r_core_deg": float(raw_r_core),
                    "source_val_raw_r_accept_deg": float(raw_r_accept),
                    "source_val_raw_r_tail_deg": float(raw_r_tail),
                    "nearest_other_component_deg": float(
                        nearest_other_component_deg
                    ),
                    "interclass_safe_r_accept_cap_deg": float(
                        safe_accept_cap_deg
                    ),
                    "max_radius_to_inter_ratio": float(
                        max_radius_to_inter_ratio
                    ),
                    "calibration_radius_guard_ratio": float(
                        calibration_radius_guard_ratio
                    ),
                    "radius_contracted_by_interclass_guard": bool(
                        contracted
                    ),
                    "accept_enabled": True,
                    "calibration_status": "source_val_calibrated",
                }
            )
            accept_radii[class_id, comp_id] = math.radians(r_accept)
            evidence_radii[class_id, comp_id] = math.radians(r_tail)
            sample_core[sample_idx[assigned]] = comp_angles <= r_core
            enabled_by_class[class_id] += 1

    missing_classes = [idx for idx, count in enumerate(enabled_by_class) if count <= 0]
    if missing_classes:
        raise ValueError(f"endpoint_accept_v1 source-val calibration missing enabled classes: {missing_classes}")

    pred = logits.argmax(dim=1)
    correct = pred == y
    top2 = torch.topk(logits, k=min(2, logits.size(1)), dim=1).values
    margins = top2[:, 0] - top2[:, 1] if top2.size(1) > 1 else torch.full((logits.size(0),), float("inf"))
    energies = -torch.logsumexp(logits, dim=1)
    energy_max_by_class: Dict[str, float] = {}
    energy_correct_sample_counts: Dict[str, int] = {}
    energy_all_true_sample_counts: Dict[str, int] = {}
    energy_calibration_source_by_class: Dict[str, str] = {}
    for class_id in range(fused.size(0)):
        true_mask = y == class_id
        correct_values = energies[true_mask & correct]
        all_true_values = energies[true_mask]
        correct_count = int(correct_values.numel())
        all_true_count = int(all_true_values.numel())
        energy_correct_sample_counts[str(class_id)] = correct_count
        energy_all_true_sample_counts[str(class_id)] = all_true_count
        required_count = max(1, int(min_class_samples))
        if correct_count >= required_count:
            values = correct_values
            source = "correct_true_class_source_val"
        elif all_true_count >= required_count:
            values = all_true_values
            source = "all_true_class_source_val_fallback"
        else:
            raise ValueError(
                "endpoint_accept_v1 source-val calibration has "
                f"insufficient true-class samples for class {class_id}"
            )
        energy_calibration_source_by_class[str(class_id)] = source
        energy_max_by_class[str(class_id)] = _finite_quantile(values, 0.95)
    core_known = correct & sample_core
    accepted_known = correct & torch.isfinite(sample_own_distance)
    if core_known.sum().item() < max(1, int(min_class_samples)) or accepted_known.sum().item() < max(1, int(min_class_samples)):
        raise ValueError("endpoint_accept_v1 source-val calibration lacks core/accepted known evidence")
    core_geo = sample_geo_margin[core_known & torch.isfinite(sample_geo_margin)]
    tail_geo = sample_geo_margin[accepted_known & ~sample_core & torch.isfinite(sample_geo_margin)]
    if tail_geo.numel() == 0:
        tail_geo = sample_geo_margin[accepted_known & torch.isfinite(sample_geo_margin)]
    gate_thresholds = {
        "energy_max_by_class": energy_max_by_class,
        "energy_temperature": 1.0,
        "energy_formula_id": "negative_logsumexp_temperature_v1",
        "density_formula_id": "exp_neg_sq_normalized_angle_v1",
        "nll_formula_id": "half_sq_normalized_angle_v1",
        "logit_margin_core_min": max(0.0, _finite_quantile(margins[core_known], 0.05)),
        "logit_margin_tail_min": max(0.0, _finite_quantile(margins[accepted_known], 0.10)),
        "geo_margin_core_min_deg": max(2.0, _finite_quantile(core_geo, 0.05)),
        "geo_margin_tail_min_deg": max(4.0, _finite_quantile(tail_geo, 0.05)),
        "allow_tail_auto_accept": False,
        "use_density_gate": True,
        "use_nll_gate": True,
        "use_energy_gate": True,
        "use_geo_margin_gate": True,
        "reject_nan": True,
        "reject_zero_direction": True,
        "max_radius_to_inter_ratio": max_radius_to_inter_ratio,
    }
    calibration = {
        "schema": "endpoint_accept_v1_source_val_calibration_v1",
        "threshold_source": "source_val_only",
        "calibration_split": "source_val",
        "input_num_samples": input_num_samples,
        "directional_num_samples": int(z.size(0)),
        "num_samples": int(z.size(0)),
        "zero_direction_excluded_samples": zero_direction_count,
        "zero_direction_excluded_fraction": zero_direction_fraction,
        "zero_direction_excluded_by_class": zero_direction_by_class,
        "zero_direction_excluded_fraction_by_class": zero_direction_fraction_by_class,
        "zero_direction_policy": "force_reject_exclude_from_angular_calibration_v1",
        "max_zero_direction_fraction": max_zero_direction_fraction,
        "correct_samples": int(correct.sum().item()),
        "energy_correct_sample_counts": energy_correct_sample_counts,
        "energy_all_true_sample_counts": energy_all_true_sample_counts,
        "energy_calibration_source_by_class": (
            energy_calibration_source_by_class
        ),
        "class_sample_counts": class_sample_counts,
        "component_sample_counts": component_sample_counts,
        "enabled_components_by_class": {str(i): int(v) for i, v in enumerate(enabled_by_class)},
        "min_component_samples": int(min_component_samples),
        "min_class_samples": int(min_class_samples),
        "core_quantile": float(core_quantile),
        "accept_quantile": float(accept_quantile),
        "tail_quantile": float(tail_quantile),
        "max_radius_to_inter_ratio": float(
            max_radius_to_inter_ratio
        ),
        "calibration_radius_guard_ratio": float(
            calibration_radius_guard_ratio
        ),
        "interclass_guard_contracted_components": int(
            interclass_guard_contracted_components
        ),
    }
    out["fusion_components"] = calibrated_components
    out["fused_tx_accept_radii"] = accept_radii
    out["fused_tx_evidence_radii"] = evidence_radii
    out["endpoint_gate_thresholds"] = gate_thresholds
    out["endpoint_calibration"] = calibration
    metadata = dict(out.get("metadata", {}) or {})
    metadata.update(
        {
            "endpoint_threshold_source": "source_val_only",
            "endpoint_calibration_split": "source_val",
            "endpoint_calibration_schema": calibration["schema"],
        }
    )
    out["metadata"] = metadata
    return out


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


_ENDPOINT_REASON_CODES = (
    "ACCEPT_KNOWN_CORE",
    "ACCEPT_KNOWN_TAIL_STRICT",
    "REVIEW_KNOWN_TAIL",
    "REJECT_OUTSIDE_RADIUS",
    "REJECT_LOW_GEO_MARGIN",
    "REJECT_LOW_DENSITY",
    "REJECT_HIGH_NLL",
    "REJECT_HIGH_ENERGY",
    "REJECT_LOW_LOGIT_MARGIN",
    "REJECT_INVALID_LOGITS",
    "REJECT_INVALID_FEATURE",
    "REJECT_ENERGY_MISMATCH",
    "REJECT_NAN",
)
_SUPPORTED_ENDPOINT_BOUNDARY_VERSIONS = {"endpoint_accept_v1.1"}


def _endpoint_boundary_spec(
    package: Mapping[str, Any], *, require_runtime_parity: bool = True
) -> Dict[str, Any]:
    components = package.get("fusion_components")
    fused = package.get("fused_tx_prototypes")
    if not isinstance(components, (list, tuple)) or fused is None:
        raise ValueError("endpoint_accept_v1 requires fused local components and component centers")
    if str(package.get("fusion_accept_policy", "")) != "local_component":
        raise ValueError("endpoint_accept_v1 requires local_component fusion acceptance")
    if package.get("global_fused_radius_is_accept_region") is not False:
        raise ValueError("endpoint_accept_v1 forbids global-ball acceptance")
    fused_tensor = torch.as_tensor(fused).detach().float().cpu().contiguous()
    if fused_tensor.ndim != 3 or fused_tensor.size(0) < 2 or not torch.isfinite(fused_tensor).all():
        raise ValueError("endpoint_accept_v1 requires finite centers for at least two known classes")
    metadata = package.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise ValueError("endpoint_accept_v1 requires inference identity metadata")
    feature_key = str(package.get("feature_key", ""))
    class_id_to_tx = metadata.get("class_id_to_tx")
    logit_class_order = metadata.get("logit_class_order")
    checkpoint_sha256 = str(metadata.get("source_checkpoint_sha256", ""))
    inference_identity = {
        "feature_key": feature_key,
        "feature_dim": int(fused_tensor.size(-1)),
        "known_class_count": int(metadata.get("known_class_count", 0) or 0),
        "class_id_to_tx": list(class_id_to_tx) if isinstance(class_id_to_tx, (list, tuple)) else [],
        "logit_class_order": list(logit_class_order) if isinstance(logit_class_order, (list, tuple)) else [],
        "source_checkpoint_sha256": checkpoint_sha256,
        "run_id": str(metadata.get("run_id", "")),
        "candidate_id": str(metadata.get("candidate_id", "")),
        "classification_head_contract": str(metadata.get("classification_head_contract", "")),
        "checkpoint_load_strict": metadata.get("checkpoint_load_strict"),
        "runtime_entry_parity_digest": str(metadata.get("endpoint_runtime_entry_parity_digest", "")),
        "runtime_entry_parity_sample_count": int(
            metadata.get("endpoint_runtime_entry_parity_sample_count", 0) or 0
        ),
    }
    if feature_key != "z_id":
        raise ValueError("endpoint_accept_v1 requires feature_key=z_id")
    if inference_identity["known_class_count"] != int(fused_tensor.size(0)):
        raise ValueError("endpoint_accept_v1 known-class count mismatch")
    if len(inference_identity["class_id_to_tx"]) != int(fused_tensor.size(0)):
        raise ValueError("endpoint_accept_v1 class-to-TX mapping is incomplete")
    if inference_identity["logit_class_order"] != list(range(int(fused_tensor.size(0)))):
        raise ValueError("endpoint_accept_v1 logit class order mismatch")
    if len(checkpoint_sha256) != 64 or any(ch not in "0123456789abcdefABCDEF" for ch in checkpoint_sha256):
        raise ValueError("endpoint_accept_v1 source checkpoint SHA256 is invalid")
    if not inference_identity["run_id"] or not inference_identity["candidate_id"]:
        raise ValueError("endpoint_accept_v1 requires run_id and candidate_id")
    if not inference_identity["classification_head_contract"]:
        raise ValueError("endpoint_accept_v1 classification-head contract is missing")
    if inference_identity["checkpoint_load_strict"] is not True:
        raise ValueError("endpoint_accept_v1 requires strict checkpoint loading")
    parity_digest = inference_identity["runtime_entry_parity_digest"]
    if require_runtime_parity and (
        len(parity_digest) != 64
        or any(ch not in "0123456789abcdefABCDEF" for ch in parity_digest)
        or inference_identity["runtime_entry_parity_sample_count"] <= 0
    ):
        raise ValueError("endpoint_accept_v1 runtime entry parity evidence is missing")
    centers_hash = hashlib.sha256(fused_tensor.numpy().tobytes()).hexdigest()
    component_rows = []
    enabled_by_class = [0 for _ in range(fused_tensor.size(0))]
    for class_id, rows in enumerate(components):
        for row_idx, row in enumerate(rows or []):
            component_id = int(row.get("component_id", row_idx))
            mu_obj = row.get("mu", fused_tensor[class_id, component_id])
            mu_raw = torch.as_tensor(mu_obj).detach().float().cpu().view(-1)
            if (
                mu_raw.numel() != fused_tensor.size(-1)
                or not torch.isfinite(mu_raw).all()
                or float(mu_raw.norm().item()) <= 1e-8
            ):
                raise ValueError(f"endpoint_accept_v1 component center invalid: {class_id}:{component_id}")
            mu = F.normalize(mu_raw.view(1, -1), dim=1).squeeze(0)
            radii = {
                "r_core_deg": float(row.get("r_core_deg", float("nan"))),
                "r_accept_deg": float(row.get("r_accept_deg", row.get("accept_radius_deg", float("nan")))),
                "r_tail_deg": float(row.get("r_tail_deg", row.get("radius_deg", float("nan")))),
                "r_vac_deg": float(row.get("r_vac_deg", float("nan"))),
            }
            enabled_obj = row.get("accept_enabled", False)
            if type(enabled_obj) is not bool:
                raise ValueError(f"endpoint_accept_v1 component accept_enabled must be bool: {class_id}:{component_id}")
            enabled = enabled_obj
            if enabled:
                values = tuple(radii.values())
                if not all(math.isfinite(value) for value in values):
                    raise ValueError(f"endpoint_accept_v1 component radii missing: {class_id}:{component_id}")
                if not (0.0 <= radii["r_core_deg"] <= radii["r_accept_deg"] <= radii["r_tail_deg"] <= radii["r_vac_deg"]):
                    raise ValueError(f"endpoint_accept_v1 component radius order invalid: {class_id}:{component_id}")
                if radii["r_accept_deg"] <= 0.0 or radii["r_vac_deg"] > 180.0:
                    raise ValueError(f"endpoint_accept_v1 component radius range invalid: {class_id}:{component_id}")
                for key in ("density_p05", "density_p10", "nll_p95", "nll_tail_p95"):
                    value = float(row.get(key, float("nan")))
                    if not math.isfinite(value):
                        raise ValueError(f"endpoint_accept_v1 component {key} missing: {class_id}:{component_id}")
                    if key.startswith("density_") and not 0.0 <= value <= 1.0:
                        raise ValueError(f"endpoint_accept_v1 component {key} outside [0,1]: {class_id}:{component_id}")
                    if key.startswith("nll_") and value < 0.0:
                        raise ValueError(f"endpoint_accept_v1 component {key} must be non-negative: {class_id}:{component_id}")
                enabled_by_class[class_id] += 1
            component_rows.append(
                {
                    "class_id": int(class_id),
                    "component_id": component_id,
                    "center": [float(value) for value in mu.tolist()],
                    "source_domains": [int(v) for v in row.get("source_domains", row.get("domains", []))],
                    **radii,
                    "density_p05": row.get("density_p05"),
                    "density_p10": row.get("density_p10"),
                    "nll_p95": row.get("nll_p95"),
                    "nll_tail_p95": row.get("nll_tail_p95"),
                    "nearest_other_deg": row.get("nearest_other_deg"),
                    "accept_enabled": enabled,
                }
            )
    if not component_rows:
        raise ValueError("endpoint_accept_v1 requires at least one local component")
    missing_classes = [class_id for class_id, count in enumerate(enabled_by_class) if count <= 0]
    if missing_classes:
        raise ValueError(f"endpoint_accept_v1 has no enabled component for classes: {missing_classes}")
    calibration = package.get("endpoint_calibration")
    gate_thresholds = package.get("endpoint_gate_thresholds")
    if not isinstance(calibration, Mapping):
        raise ValueError("endpoint_accept_v1 requires source-val calibration evidence")
    if str(calibration.get("threshold_source", "")) != "source_val_only" or str(
        calibration.get("calibration_split", "")
    ) != "source_val":
        raise ValueError("endpoint_accept_v1 calibration must be source_val_only/source_val")
    if not isinstance(gate_thresholds, Mapping) or not gate_thresholds:
        raise ValueError("endpoint_accept_v1 requires calibrated gate thresholds")
    energy = gate_thresholds.get("energy_max_by_class")
    if not isinstance(energy, Mapping) or any(str(class_id) not in energy for class_id in range(fused_tensor.size(0))):
        raise ValueError("endpoint_accept_v1 requires per-class energy thresholds")
    if any(not math.isfinite(float(energy[str(class_id)])) for class_id in range(fused_tensor.size(0))):
        raise ValueError("endpoint_accept_v1 energy thresholds must be finite")
    for key in (
        "energy_temperature",
        "logit_margin_core_min",
        "logit_margin_tail_min",
        "geo_margin_core_min_deg",
        "geo_margin_tail_min_deg",
    ):
        if not math.isfinite(float(gate_thresholds.get(key, float("nan")))):
            raise ValueError(f"endpoint_accept_v1 gate threshold missing: {key}")
    if float(gate_thresholds["energy_temperature"]) <= 0.0:
        raise ValueError("endpoint_accept_v1 energy_temperature must be positive")
    if str(gate_thresholds.get("energy_formula_id", "")) != "negative_logsumexp_temperature_v1":
        raise ValueError("endpoint_accept_v1 energy formula mismatch")
    max_radius_ratio = float(gate_thresholds.get("max_radius_to_inter_ratio", float("nan")))
    if not math.isfinite(max_radius_ratio) or not 0.0 < max_radius_ratio <= 0.5:
        raise ValueError("endpoint_accept_v1 max radius-to-inter ratio must be in (0,0.5]")
    for key in (
        "logit_margin_core_min",
        "logit_margin_tail_min",
        "geo_margin_core_min_deg",
        "geo_margin_tail_min_deg",
    ):
        if float(gate_thresholds[key]) < 0.0:
            raise ValueError(f"endpoint_accept_v1 gate threshold must be non-negative: {key}")
    for key in (
        "use_density_gate",
        "use_nll_gate",
        "use_energy_gate",
        "use_geo_margin_gate",
        "reject_nan",
        "reject_zero_direction",
    ):
        if gate_thresholds.get(key) is not True:
            raise ValueError(f"endpoint_accept_v1 requires {key}=true")
    if gate_thresholds.get("allow_tail_auto_accept") is not False:
        raise ValueError("endpoint_accept_v1 requires allow_tail_auto_accept=false")
    if str(gate_thresholds.get("density_formula_id", "")) != "exp_neg_sq_normalized_angle_v1":
        raise ValueError("endpoint_accept_v1 density formula mismatch")
    if str(gate_thresholds.get("nll_formula_id", "")) != "half_sq_normalized_angle_v1":
        raise ValueError("endpoint_accept_v1 NLL formula mismatch")
    if str(calibration.get("zero_direction_policy", "")) != (
        "force_reject_exclude_from_angular_calibration_v1"
    ):
        raise ValueError("endpoint_accept_v1 zero-direction policy mismatch")
    try:
        input_num_samples = int(calibration["input_num_samples"])
        directional_num_samples = int(calibration["directional_num_samples"])
        num_samples = int(calibration["num_samples"])
        zero_direction_samples = int(calibration["zero_direction_excluded_samples"])
        zero_direction_fraction = float(calibration["zero_direction_excluded_fraction"])
        max_zero_direction_fraction = float(calibration["max_zero_direction_fraction"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("endpoint_accept_v1 zero-direction audit is incomplete") from exc
    if (
        input_num_samples <= 0
        or directional_num_samples <= 0
        or num_samples != directional_num_samples
        or input_num_samples != directional_num_samples + zero_direction_samples
        or zero_direction_samples < 0
    ):
        raise ValueError("endpoint_accept_v1 zero-direction sample accounting mismatch")
    expected_zero_fraction = float(zero_direction_samples) / float(input_num_samples)
    if (
        not math.isfinite(zero_direction_fraction)
        or abs(zero_direction_fraction - expected_zero_fraction) > 1e-12
        or not math.isfinite(max_zero_direction_fraction)
        or not 0.0 <= max_zero_direction_fraction <= 0.01
        or zero_direction_fraction > max_zero_direction_fraction + 1e-12
    ):
        raise ValueError("endpoint_accept_v1 zero-direction fraction audit mismatch")
    excluded_by_class = calibration.get("zero_direction_excluded_by_class")
    excluded_fraction_by_class = calibration.get("zero_direction_excluded_fraction_by_class")
    class_sample_counts = calibration.get("class_sample_counts")
    if not all(
        isinstance(value, Mapping)
        for value in (excluded_by_class, excluded_fraction_by_class, class_sample_counts)
    ):
        raise ValueError("endpoint_accept_v1 zero-direction class audit is incomplete")
    class_excluded_total = 0
    for class_id in range(fused_tensor.size(0)):
        key = str(class_id)
        try:
            class_directional = int(class_sample_counts[key])
            class_excluded = int(excluded_by_class[key])
            class_fraction = float(excluded_fraction_by_class[key])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("endpoint_accept_v1 zero-direction class audit is incomplete") from exc
        class_input = class_directional + class_excluded
        expected_class_fraction = (
            float(class_excluded) / float(class_input)
            if class_input > 0
            else 1.0
        )
        if (
            class_directional <= 0
            or class_excluded < 0
            or not math.isfinite(class_fraction)
            or abs(class_fraction - expected_class_fraction) > 1e-12
            or class_fraction > max_zero_direction_fraction + 1e-12
        ):
            raise ValueError(
                f"endpoint_accept_v1 zero-direction class audit mismatch: class={class_id}"
            )
        class_excluded_total += class_excluded
    if class_excluded_total != zero_direction_samples:
        raise ValueError("endpoint_accept_v1 zero-direction class totals mismatch")
    enabled_rows = [row for row in component_rows if row["accept_enabled"]]
    for row in enabled_rows:
        own = torch.tensor(row["center"], dtype=torch.float32)
        other_rows = [candidate for candidate in enabled_rows if candidate["class_id"] != row["class_id"]]
        if not other_rows:
            raise ValueError("endpoint_accept_v1 requires another-class component for every class")
        other = torch.tensor([candidate["center"] for candidate in other_rows], dtype=torch.float32)
        nearest_inter = math.degrees(
            float(torch.acos((other @ own).clamp(-1.0 + 1e-6, 1.0 - 1e-6)).min().item())
        )
        ratio = float(row["r_accept_deg"]) / max(1e-8, nearest_inter)
        if ratio > max_radius_ratio + 1e-8:
            raise ValueError(
                "endpoint_accept_v1 component radius-to-inter ratio unsafe: "
                f"class={row['class_id']} component={row['component_id']} ratio={ratio:.6f}"
            )
        row["nearest_other_deg"] = nearest_inter
        row["radius_to_inter_ratio"] = ratio
    return {
        "inference_identity": inference_identity,
        "accept_policy": "local_component",
        "component_radius_key": "r_accept_deg",
        "global_ball_accept": False,
        "centers_hash": centers_hash,
        "components": component_rows,
        "gate_thresholds": _as_jsonable(gate_thresholds),
        "calibration": _as_jsonable(calibration),
        "reason_codes": list(_ENDPOINT_REASON_CODES),
    }


def attach_endpoint_accept_v1_manifest(
    package: Mapping[str, Any],
    *,
    threshold_source: str = "source_val_only",
    calibration_split: str = "source_val",
    boundary_version: str = "endpoint_accept_v1.1",
    require_runtime_parity: bool = True,
) -> Dict[str, Any]:
    """Attach one fail-closed boundary contract shared by export/eval/runtime."""

    if str(boundary_version) not in _SUPPORTED_ENDPOINT_BOUNDARY_VERSIONS:
        raise ValueError(f"unsupported endpoint_accept_v1 boundary version: {boundary_version}")
    out = dict(package)
    boundary = _endpoint_boundary_spec(out, require_runtime_parity=bool(require_runtime_parity))
    canonical = json.dumps(boundary, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    boundary_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    entry = {"boundary_version": str(boundary_version), "boundary_hash": boundary_hash}
    manifest = {
        "schema_version": 1,
        "policy_id": "endpoint_accept_v1",
        "boundary_version": str(boundary_version),
        "boundary_hash": boundary_hash,
        "threshold_source": str(threshold_source),
        "calibration_split": str(calibration_split),
        "fail_closed": True,
        "loss_gate_exported": False,
        "accept_policy": str(boundary["accept_policy"]),
        "component_radius_key": str(boundary["component_radius_key"]),
        "global_ball_accept": bool(boundary["global_ball_accept"]),
        "inference_identity": _as_jsonable(boundary["inference_identity"]),
        "reason_codes": list(_ENDPOINT_REASON_CODES),
        "gate_thresholds": _as_jsonable(boundary["gate_thresholds"]),
        "calibration_evidence": _as_jsonable(boundary["calibration"]),
        "entry_points": {
            "train_export": dict(entry),
            "offline_eval": dict(entry),
            "runtime_inference": dict(entry),
        },
    }
    out["endpoint_accept_v1"] = manifest
    metadata = dict(out.get("metadata", {}) or {})
    metadata.update(
        {
            "endpoint_policy_id": "endpoint_accept_v1",
            "endpoint_boundary_version": str(boundary_version),
            "endpoint_boundary_hash": boundary_hash,
            "endpoint_threshold_source": str(threshold_source),
            "endpoint_calibration_split": str(calibration_split),
            "endpoint_entry_parity": True,
            "endpoint_accept_boundary_exported": True,
            "final_reject_boundary": False,
            "true_unknown_validated": False,
            "loss_gate_exported": False,
        }
    )
    out["metadata"] = metadata
    out["schema_version"] = max(2, int(out.get("schema_version", 1) or 1))
    return out


def verify_endpoint_accept_v1_manifest(package: Mapping[str, Any]) -> Dict[str, Any]:
    manifest = package.get("endpoint_accept_v1", {})
    if not isinstance(manifest, Mapping):
        raise ValueError("endpoint_accept_v1 manifest is missing")
    if str(manifest.get("policy_id", "")) != "endpoint_accept_v1":
        raise ValueError("endpoint_accept_v1 policy id mismatch")
    if type(manifest.get("schema_version")) is not int or manifest.get("schema_version") != 1:
        raise ValueError("endpoint_accept_v1 schema version mismatch")
    if manifest.get("fail_closed") is not True or manifest.get("loss_gate_exported") is not False:
        raise ValueError("endpoint_accept_v1 must be fail-closed and must not export a loss gate")
    if str(manifest.get("threshold_source", "")) != "source_val_only" or str(
        manifest.get("calibration_split", "")
    ) != "source_val":
        raise ValueError("endpoint_accept_v1 threshold provenance mismatch")
    if str(manifest.get("accept_policy", "")) != "local_component" or manifest.get("global_ball_accept") is not False:
        raise ValueError("endpoint_accept_v1 must use local components without global-ball acceptance")
    if str(manifest.get("component_radius_key", "")) != "r_accept_deg":
        raise ValueError("endpoint_accept_v1 component radius key mismatch")
    if not isinstance(manifest.get("inference_identity"), Mapping):
        raise ValueError("endpoint_accept_v1 inference identity is missing")
    version = str(manifest.get("boundary_version", ""))
    if version not in _SUPPORTED_ENDPOINT_BOUNDARY_VERSIONS:
        raise ValueError(f"unsupported endpoint_accept_v1 boundary version: {version or '<missing>'}")
    if tuple(manifest.get("reason_codes", ())) != tuple(_ENDPOINT_REASON_CODES):
        raise ValueError("endpoint_accept_v1 reason-code schema mismatch")
    boundary = _endpoint_boundary_spec(package)
    canonical = json.dumps(boundary, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    expected_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    actual_hash = str(manifest.get("boundary_hash", ""))
    if actual_hash != expected_hash:
        raise ValueError("endpoint_accept_v1 boundary hash mismatch")
    manifest_gate = json.dumps(manifest.get("gate_thresholds", {}), ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    package_gate = json.dumps(boundary["gate_thresholds"], ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    manifest_calibration = json.dumps(
        manifest.get("calibration_evidence", {}), ensure_ascii=True, sort_keys=True, separators=(",", ":")
    )
    package_calibration = json.dumps(
        boundary["calibration"], ensure_ascii=True, sort_keys=True, separators=(",", ":")
    )
    manifest_identity = json.dumps(
        manifest.get("inference_identity", {}), ensure_ascii=True, sort_keys=True, separators=(",", ":")
    )
    package_identity = json.dumps(
        boundary["inference_identity"], ensure_ascii=True, sort_keys=True, separators=(",", ":")
    )
    if (
        manifest_gate != package_gate
        or manifest_calibration != package_calibration
        or manifest_identity != package_identity
    ):
        raise ValueError("endpoint_accept_v1 manifest/package threshold evidence mismatch")
    for entry in ("train_export", "offline_eval", "runtime_inference"):
        row = manifest.get("entry_points", {}).get(entry, {})
        if str(row.get("boundary_hash", "")) != expected_hash or str(row.get("boundary_version", "")) != version:
            raise ValueError(f"endpoint_accept_v1 entry parity mismatch: {entry}")
    return dict(manifest)


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
