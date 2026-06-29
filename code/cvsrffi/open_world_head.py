from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional

import torch
from torch import nn
import torch.nn.functional as F


UNKNOWN_LABEL = -1


def _norm(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    x = torch.nan_to_num(x.float(), nan=0.0, posinf=0.0, neginf=0.0)
    return F.normalize(x, dim=dim, eps=1e-6)


@dataclass
class OpenWorldDecision:
    predicted_labels: torch.Tensor
    candidate_labels: torch.Tensor
    accepted: torch.Tensor
    decisions: List[str]
    gate_reasons: List[str]
    diagnostics: Dict[str, torch.Tensor]


class OpenWorldMultiPrototypeHead(nn.Module):
    """Multi-prototype cosine head with radii and unknown rejection scores.

    Domain prototypes are intentionally absent from class scoring. Domain shift
    may be supplied as an additive correction to old/source prototypes, but
    class distances are always computed in the TX identity feature space.
    """

    def __init__(self, feat_dim: int, *, score_temperature: float = 0.10, energy_temperature: float = 1.0):
        super().__init__()
        self.feat_dim = int(feat_dim)
        self.score_temperature = float(score_temperature)
        self.energy_temperature = float(energy_temperature)
        self._prototypes: Dict[int, torch.Tensor] = {}
        self._radii: Dict[int, float] = {}
        self._groups: Dict[int, str] = {}

    @property
    def class_ids(self) -> List[int]:
        return sorted(self._prototypes)

    @classmethod
    def from_phase2_export(
        cls,
        package: Mapping[str, Any],
        *,
        radius_key: str = "robust_max",
        score_temperature: float = 0.10,
        energy_temperature: float = 1.0,
    ) -> "OpenWorldMultiPrototypeHead":
        prototypes = package.get("prototypes")
        if not torch.is_tensor(prototypes) or prototypes.ndim != 2:
            raise ValueError("Phase2 export package must contain prototypes with shape [C, D]")
        radii_obj = package.get("radii", {})
        if isinstance(radii_obj, Mapping):
            radii = radii_obj.get(radius_key)
            if radii is None:
                raise KeyError(f"Phase2 export radii missing key: {radius_key}")
        else:
            radii = radii_obj
        if not torch.is_tensor(radii):
            radii = torch.as_tensor(radii, dtype=torch.float32)
        head = cls(
            feat_dim=int(prototypes.size(1)),
            score_temperature=score_temperature,
            energy_temperature=energy_temperature,
        )
        head.add_old_classes(prototypes.float(), radii.float())
        return head

    def add_old_classes(self, prototypes: torch.Tensor, radii: torch.Tensor, sigmas: Optional[torch.Tensor] = None) -> None:
        if prototypes.ndim != 2:
            raise ValueError("prototypes must have shape [C, D]")
        if prototypes.size(1) != self.feat_dim:
            raise ValueError(f"feature dim mismatch: expected {self.feat_dim}, got {prototypes.size(1)}")
        r = radii.view(-1).float()
        if r.numel() != prototypes.size(0):
            raise ValueError("radii must have one value per class")
        for class_id in range(prototypes.size(0)):
            self.add_target_prototypes(
                [class_id],
                prototypes[class_id : class_id + 1],
                radii=r[class_id : class_id + 1],
                proto_type="source_old",
            )

    def add_target_prototypes(
        self,
        class_ids: Iterable[int],
        prototypes: torch.Tensor,
        radii: Optional[torch.Tensor] = None,
        *,
        proto_type: str = "target",
    ) -> None:
        ids = [int(v) for v in class_ids]
        if prototypes.ndim != 2 or prototypes.size(0) != len(ids) or prototypes.size(1) != self.feat_dim:
            raise ValueError("prototypes must have shape [len(class_ids), feat_dim]")
        if radii is None:
            r = torch.full((len(ids),), 0.35, dtype=torch.float32, device=prototypes.device)
        else:
            r = radii.view(-1).float().to(prototypes.device)
            if r.numel() != len(ids):
                raise ValueError("radii must have one value per class_id")
        for row, cid in enumerate(ids):
            p = _norm(prototypes[row : row + 1], dim=1).detach().cpu()
            if cid in self._prototypes:
                self._prototypes[cid] = torch.cat([self._prototypes[cid], p], dim=0)
                self._radii[cid] = max(float(self._radii[cid]), float(r[row].detach().cpu().item()))
            else:
                self._prototypes[cid] = p
                self._radii[cid] = float(r[row].detach().cpu().item())
            self._groups[cid] = str(proto_type)

    def register_new_class(
        self,
        class_id: int,
        support_features: torch.Tensor,
        support_aug_features: Optional[torch.Tensor] = None,
        radius_prior: Optional[float] = None,
        *,
        shrinkage: float = 0.50,
        overlap_margin: Optional[float] = None,
    ) -> Dict[str, float]:
        if support_features.ndim != 2 or support_features.size(1) != self.feat_dim:
            raise ValueError("support_features must have shape [K, feat_dim]")
        if support_features.size(0) <= 0:
            raise ValueError("support_features must contain at least one sample")
        feats = support_features
        if support_aug_features is not None:
            if support_aug_features.ndim != 2 or support_aug_features.size(1) != self.feat_dim:
                raise ValueError("support_aug_features must have shape [K_aug, feat_dim]")
            feats = torch.cat([feats, support_aug_features], dim=0)
        z = _norm(feats, dim=1)
        proto = _norm(z.mean(dim=0, keepdim=True), dim=1)
        angles = torch.arccos((z * proto).sum(dim=1).clamp(-1.0, 1.0))
        empirical = float(torch.quantile(angles, 0.95).detach().item()) if angles.numel() > 1 else 0.25
        if radius_prior is None:
            radius = empirical
        else:
            radius = float(shrinkage) * empirical + (1.0 - float(shrinkage)) * float(radius_prior)
        overlap_class = -1
        overlap_clearance = float("nan")
        if overlap_margin is not None and self._prototypes:
            class_ids, old_proto, old_radii = self._stack(proto.device, proto.dtype)
            old_scores = (proto @ old_proto.t()).squeeze(0).clamp(-1.0, 1.0)
            old_angles = torch.arccos(old_scores)
            clearance = old_angles - old_radii - float(radius)
            min_clearance, min_idx = clearance.min(dim=0)
            overlap_clearance = float(min_clearance.detach().cpu().item())
            overlap_class = int(class_ids[min_idx].detach().cpu().item())
            if overlap_clearance <= float(overlap_margin):
                return {
                    "support_count": float(support_features.size(0)),
                    "radius": float(radius),
                    "empirical_radius": float(empirical),
                    "status": "rejected_overlap",
                    "overlap_class": float(overlap_class),
                    "overlap_clearance": float(overlap_clearance),
                }
        self.add_target_prototypes([int(class_id)], proto, radii=torch.tensor([radius], device=proto.device), proto_type="seen_new")
        return {
            "support_count": float(support_features.size(0)),
            "radius": float(radius),
            "empirical_radius": float(empirical),
            "status": "confirmed",
            "overlap_class": float(overlap_class),
            "overlap_clearance": float(overlap_clearance),
        }

    def _stack(self, device, dtype) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self._prototypes:
            raise ValueError("head has no prototypes")
        class_ids = []
        proto_rows = []
        radius_rows = []
        for cid in self.class_ids:
            p = self._prototypes[cid].to(device=device, dtype=dtype)
            proto_rows.append(_norm(p, dim=1))
            class_ids.extend([cid] * p.size(0))
            radius_rows.extend([self._radii[cid]] * p.size(0))
        return (
            torch.tensor(class_ids, device=device, dtype=torch.long),
            torch.cat(proto_rows, dim=0),
            torch.tensor(radius_rows, device=device, dtype=dtype),
        )

    def forward(
        self,
        z: torch.Tensor,
        domain_shift: Optional[torch.Tensor] = None,
        *,
        return_details: bool = True,
    ) -> Mapping[str, torch.Tensor]:
        if z.ndim != 2 or z.size(1) != self.feat_dim:
            raise ValueError(f"z must have shape [N, {self.feat_dim}]")
        class_ids, proto, radii = self._stack(z.device, z.dtype)
        if domain_shift is not None:
            shift = domain_shift.to(device=z.device, dtype=z.dtype)
            if shift.ndim == 1:
                shift = shift.view(1, -1)
            if shift.size(-1) != self.feat_dim:
                raise ValueError("domain_shift feature dimension mismatch")
            proto = _norm(proto + shift.expand_as(proto), dim=1)
        feat = _norm(z, dim=1)
        proto_scores = feat @ proto.t()
        best_proto_score, best_proto_idx = proto_scores.max(dim=1)
        candidate_labels = class_ids[best_proto_idx]
        angle = torch.arccos(best_proto_score.clamp(-1.0, 1.0))
        best_radius = radii[best_proto_idx]
        radius_margin = best_radius - angle
        class_scores = torch.full((z.size(0), len(self.class_ids)), -1.0e6, device=z.device, dtype=z.dtype)
        for col, cid in enumerate(self.class_ids):
            m = class_ids == int(cid)
            class_scores[:, col] = proto_scores[:, m].max(dim=1).values
        energy = -torch.logsumexp(class_scores / max(1e-4, float(self.energy_temperature)), dim=1)
        out = {
            "class_scores": class_scores,
            "class_ids": torch.tensor(self.class_ids, device=z.device, dtype=torch.long),
            "candidate_labels": candidate_labels,
            "best_proto_score": best_proto_score,
            "best_angle": angle,
            "best_radius": best_radius,
            "radius_margin": radius_margin,
            "energy": energy,
        }
        return out if return_details else class_scores

    def unknown_scores(self, details: Mapping[str, torch.Tensor], view_stats: Optional[Mapping[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        radius_margin = details["radius_margin"]
        energy = details["energy"]
        score = (-radius_margin) + energy
        if view_stats and "agreement" in view_stats:
            score = score + (1.0 - view_stats["agreement"].to(score.device, score.dtype))
        return {"unknown_score": score, "radius_margin": radius_margin, "energy": energy}

    def decide(
        self,
        z_or_details,
        thresholds: Mapping[str, float],
    ) -> OpenWorldDecision:
        details = self.forward(z_or_details) if torch.is_tensor(z_or_details) else z_or_details
        candidate = details["candidate_labels"]
        best_score = details["best_proto_score"]
        radius_margin = details["radius_margin"]
        energy = details["energy"]
        min_cosine = float(thresholds.get("min_cosine", -1.0))
        min_radius_margin = float(thresholds.get("min_radius_margin", -1.0e6))
        max_energy = float(thresholds.get("max_energy", 1.0e6))
        accepted = (best_score >= min_cosine) & (radius_margin >= min_radius_margin) & (energy <= max_energy)
        predicted = torch.where(accepted, candidate, torch.full_like(candidate, UNKNOWN_LABEL))
        decisions = ["accept" if bool(v) else "reject" for v in accepted.detach().cpu().tolist()]
        reasons = []
        for ok, cos, rad, ene in zip(
            accepted.detach().cpu().tolist(),
            best_score.detach().cpu().tolist(),
            radius_margin.detach().cpu().tolist(),
            energy.detach().cpu().tolist(),
        ):
            if ok:
                reasons.append("multi_proto_accept")
            elif float(cos) < min_cosine:
                reasons.append("cosine_below_threshold")
            elif float(rad) < min_radius_margin:
                reasons.append("outside_class_radius")
            elif float(ene) > max_energy:
                reasons.append("energy_above_threshold")
            else:
                reasons.append("unknown_reject")
        return OpenWorldDecision(
            predicted_labels=predicted,
            candidate_labels=candidate,
            accepted=accepted,
            decisions=decisions,
            gate_reasons=reasons,
            diagnostics={
                "best_proto_score": best_score,
                "radius_margin": radius_margin,
                "energy": energy,
            },
        )

    def state_dict(self, *args, **kwargs):  # type: ignore[override]
        return {
            "feat_dim": self.feat_dim,
            "score_temperature": self.score_temperature,
            "energy_temperature": self.energy_temperature,
            "prototypes": {cid: p.clone() for cid, p in self._prototypes.items()},
            "radii": dict(self._radii),
            "groups": dict(self._groups),
        }

    def load_state_dict(self, state_dict, strict: bool = True):  # type: ignore[override]
        self._prototypes = {int(cid): tensor.clone().detach().cpu() for cid, tensor in state_dict["prototypes"].items()}
        self._radii = {int(cid): float(v) for cid, v in state_dict["radii"].items()}
        self._groups = {int(cid): str(v) for cid, v in state_dict.get("groups", {}).items()}
        return nn.modules.module._IncompatibleKeys([], [])
