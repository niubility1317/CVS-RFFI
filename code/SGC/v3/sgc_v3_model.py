from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from .feature_adapter import IdentityPreservingFeatureAdapter
from .logit_calibrator import BaseAnchoredLogitCalibrator
from .physical_canonicalizer import PhysicalSafeCanonicalizer, PhysicalSafeCanonicalizerConfig
from .prototype_bank import PrototypeBank
from .satellite_evidence_encoder import SatelliteEvidenceEncoder


@dataclass
class SGCv3Config:
    num_classes: int
    feature_dim: int
    scenario_dim: int = 16
    num_experts: int = 5
    adapter_rank: int = 16
    adapter_hidden_dim: int = 128
    epsilon_z: float = 0.02
    logit_hidden_dim: int = 128
    topk_only: int = 3
    epsilon_logit: float = 0.5
    high_conf_prob: float = 0.9
    high_conf_margin: float = 1.0
    uncertainty_margin: float = 0.5
    uncertainty_entropy: float = 0.8
    psc_cfo_betas: tuple[float, ...] = (0.0, 0.25, 0.5, -0.25, -0.5)
    psc_shifts: tuple[int, ...] = (-1, 0, 1)
    psc_envelope_gammas: tuple[float, ...] = (0.0, 0.15)

    @classmethod
    def from_mapping(cls, cfg: Mapping[str, object]) -> "SGCv3Config":
        data = dict(cfg)
        for key in ("psc_cfo_betas", "psc_shifts", "psc_envelope_gammas"):
            if key in data:
                data[key] = tuple(data[key])  # type: ignore[arg-type]
        allowed = {field.name for field in cls.__dataclass_fields__.values()}
        return cls(**{key: value for key, value in data.items() if key in allowed})


def _extract_feature(teacher: nn.Module, x: torch.Tensor) -> torch.Tensor:
    if hasattr(teacher, "extract_feature"):
        return teacher.extract_feature(x)
    out = teacher(x, return_aux=True)
    if isinstance(out, dict):
        for key in ("z_id_raw", "z_id", "feat_joint", "feat_cls", "base"):
            value = out.get(key)
            if torch.is_tensor(value):
                return value
    raise AttributeError("base teacher must expose extract_feature(x) or feature aux keys.")


def _classify(teacher: nn.Module, x: torch.Tensor, feat: torch.Tensor | None = None) -> torch.Tensor:
    if hasattr(teacher, "classify"):
        try:
            return teacher.classify(x, feat)
        except TypeError:
            return teacher.classify(x)
    classifier = getattr(teacher, "classifier", None)
    if classifier is not None:
        if feat is None:
            feat = _extract_feature(teacher, x)
        return classifier(feat)
    out = teacher(x)
    if torch.is_tensor(out):
        return out
    if isinstance(out, dict):
        for key in ("logits", "tx_logits", "base_logits"):
            value = out.get(key)
            if torch.is_tensor(value):
                return value
    raise AttributeError("base teacher must expose classify(x), classifier(feat), or logits output.")


def _prob_stats(logits: torch.Tensor) -> torch.Tensor:
    prob = F.softmax(logits, dim=-1)
    top2 = prob.topk(min(2, prob.size(-1)), dim=-1).values
    margin = top2[:, 0] - (top2[:, 1] if top2.size(-1) > 1 else 0.0)
    entropy = -(prob * (prob + 1e-8).log()).sum(dim=-1) / max(float(torch.log(torch.tensor(prob.size(-1), device=prob.device))), 1e-6)
    return torch.stack([prob.max(dim=-1).values, margin, entropy, logits.float().norm(dim=-1)], dim=-1)


class SGCv3Model(nn.Module):
    def __init__(
        self,
        base_teacher: nn.Module,
        cfg: SGCv3Config | Mapping[str, object],
        prototype_bank: PrototypeBank | None = None,
    ) -> None:
        super().__init__()
        if isinstance(cfg, Mapping):
            cfg = SGCv3Config.from_mapping(cfg)
        self.cfg = cfg
        self.base_teacher = base_teacher
        self.base_teacher.eval()
        for param in self.base_teacher.parameters():
            param.requires_grad = False

        psc_cfg = PhysicalSafeCanonicalizerConfig(
            cfo_betas=cfg.psc_cfo_betas,
            shifts=cfg.psc_shifts,
            envelope_gammas=cfg.psc_envelope_gammas,
        )
        self.psc = PhysicalSafeCanonicalizer(psc_cfg)
        self.evidence = SatelliteEvidenceEncoder(
            num_views=self.psc.num_candidate_views,
            scenario_dim=cfg.scenario_dim,
            num_experts=cfg.num_experts,
        )
        self.feature_adapter = IdentityPreservingFeatureAdapter(
            feature_dim=cfg.feature_dim,
            scenario_dim=cfg.scenario_dim,
            rank=cfg.adapter_rank,
            hidden_dim=cfg.adapter_hidden_dim,
            epsilon_z=cfg.epsilon_z,
        )
        self.logit_calibrator = BaseAnchoredLogitCalibrator(
            feature_dim=cfg.feature_dim,
            scenario_dim=cfg.scenario_dim,
            num_classes=cfg.num_classes,
            topk_only=cfg.topk_only,
            epsilon_logit=cfg.epsilon_logit,
            hidden_dim=cfg.logit_hidden_dim,
        )
        self.prototype_bank = prototype_bank

    def _teacher_views(self, views: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, V, C, T = views.shape
        flat = views.reshape(B * V, C, T)
        with torch.no_grad():
            z = _extract_feature(self.base_teacher, flat)
            logits = _classify(self.base_teacher, flat, z)
        return z.reshape(B, V, -1), logits.reshape(B, V, -1)

    def _gate(self, logits_base: torch.Tensor, sat_score: torch.Tensor) -> torch.Tensor:
        prob = F.softmax(logits_base, dim=-1)
        top2 = prob.topk(min(2, prob.size(-1)), dim=-1).values
        margin = top2[:, 0] - (top2[:, 1] if top2.size(-1) > 1 else 0.0)
        entropy = -(prob * (prob + 1e-8).log()).sum(dim=-1, keepdim=True)
        uncertain = torch.sigmoid((float(self.cfg.uncertainty_margin) - margin).view(-1, 1) + (entropy - float(self.cfg.uncertainty_entropy)))
        high_conf = (prob.max(dim=-1, keepdim=True).values > float(self.cfg.high_conf_prob)) & (
            margin.view(-1, 1) > float(self.cfg.high_conf_margin)
        )
        return (sat_score * uncertain).masked_fill(high_conf, 0.0).clamp(0.0, 1.0)

    def forward(self, x: torch.Tensor) -> dict[str, object]:
        psc_out = self.psc(x)
        views = psc_out["views"]
        evidence = self.evidence(x)
        view_weights = evidence["view_weights"].to(dtype=views.dtype)
        z_views, logits_views = self._teacher_views(views)
        w = view_weights.view(view_weights.size(0), view_weights.size(1), 1)
        z_base = (z_views * w).sum(dim=1)
        logits_base = (logits_views * w).sum(dim=1)
        p_stats = _prob_stats(logits_base)
        alpha_raw = self._gate(logits_base, evidence["sat_score"])

        z_probe, aux_z_probe = self.feature_adapter(z_base, evidence["scenario_code"], p_stats, gate=torch.ones_like(alpha_raw))
        logits_probe, aux_l_probe = self.logit_calibrator(
            z_probe, evidence["scenario_code"], logits_base, p_stats, gate=torch.ones_like(alpha_raw)
        )
        proto_dist = None
        if self.prototype_bank is not None:
            proto_dist = self.prototype_bank.prototype_distance(z_probe).view(-1, 1)
            proto_safe = (proto_dist <= proto_dist.detach().quantile(0.95).clamp_min(1e-6)).float()
        else:
            proto_dist = logits_base.new_zeros(logits_base.size(0), 1)
            proto_safe = torch.ones_like(proto_dist)
        safe = (
            (aux_z_probe["delta_z_ratio"].view(-1, 1) <= float(self.cfg.epsilon_z) + 1e-6).float()
            * (aux_l_probe["delta_logit_norm"].view(-1, 1) <= float(self.cfg.epsilon_logit) + 1e-6).float()
            * proto_safe
        )
        alpha = (alpha_raw * safe).clamp(0.0, 1.0)
        z_sgc, aux_z = self.feature_adapter(z_base, evidence["scenario_code"], p_stats, gate=alpha)
        logits_sgc, aux_l = self.logit_calibrator(z_sgc, evidence["scenario_code"], logits_base, p_stats, gate=alpha)
        logits_final = (1.0 - alpha) * logits_base + alpha * logits_sgc
        prob_final = F.softmax(logits_final, dim=-1)
        pseudo_weight, pseudo_y = prob_final.max(dim=-1)
        best_idx = view_weights.argmax(dim=-1)
        gather_idx = best_idx.view(-1, 1, 1, 1).expand(-1, 1, views.size(2), views.size(3))
        x_phys_best = views.gather(1, gather_idx).squeeze(1)

        metrics = {
            "delta_z_ratio_mean": aux_z["delta_z_ratio"].detach().mean(),
            "delta_z_ratio_p95": torch.quantile(aux_z["delta_z_ratio"].detach().float(), 0.95),
            "delta_logit_norm_mean": aux_l["delta_logit_norm"].detach().mean(),
            "gate_mean": alpha.detach().mean(),
            "prototype_dist_mean": proto_dist.detach().mean(),
        }
        return {
            "logits_base": logits_base,
            "logits_sgc": logits_sgc,
            "logits_final": logits_final,
            "prob_final": prob_final,
            "z_base": z_base,
            "z_sgc": z_sgc,
            "scenario_code": evidence["scenario_code"],
            "gate": alpha,
            "view_weights": view_weights,
            "expert_weights": evidence["expert_weights"],
            "sat_logit": evidence["sat_logit"],
            "sat_score": evidence["sat_score"],
            "pseudo_y": pseudo_y,
            "pseudo_weight": pseudo_weight,
            "x_phys_best": x_phys_best,
            "x_phys_views": views,
            "psc_stats": psc_out["stats"],
            "channel_stats": evidence["channel_stats"],
            "metrics": metrics,
            "delta_z": aux_z["delta_z"],
            "delta_z_ratio": aux_z["delta_z_ratio"],
            "delta_logits": aux_l["delta_logits"],
            "delta_logit_norm": aux_l["delta_logit_norm"],
            "prototype_distance": proto_dist.squeeze(-1),
        }
