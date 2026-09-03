from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

import torch
import torch.nn.functional as F

from .orbit_teacher import (
    anchored_spherical_orbit_target,
    orbit_feature_loss,
    orbit_logit_distillation_loss,
    orbit_prototype_distillation_loss,
    orbit_relation_loss,
    robust_spherical_orbit_target,
)
from .selective_tangent import (
    angular_sensitivity,
    chordal_sensitivity,
    fingerprint_keep_loss,
    heteroscedastic_nuisance_loss,
    fingerprint_selectivity,
    selective_tangent_loss,
)


def _zero(reference: torch.Tensor) -> torch.Tensor:
    return reference.sum() * 0.0


def _teacher_consensus_and_target_logits(
    logits: torch.Tensor,
    feature_weights: torch.Tensor,
    *,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    probabilities = F.softmax(logits.float() / float(temperature), dim=-1)
    confidence = probabilities.max(dim=-1).values
    predictions = probabilities.argmax(dim=-1)
    consensus = predictions.eq(predictions[:, :1]).all(dim=1)
    logit_weights = feature_weights * confidence
    logit_weights = logit_weights / logit_weights.sum(dim=1, keepdim=True).clamp_min(1e-12)
    target_probability = (logit_weights.unsqueeze(-1) * probabilities).sum(dim=1)
    target_logits = torch.log(target_probability.clamp_min(1e-8)) * float(temperature)
    return consensus, confidence.mean(dim=1), target_logits


def compute_daot_batch_objective(
    *,
    student_clean: Mapping[str, torch.Tensor],
    student_channel: Mapping[str, torch.Tensor],
    teacher_views: Sequence[Mapping[str, torch.Tensor]],
    reliability: torch.Tensor,
    importance: torch.Tensor,
    recoverability: torch.Tensor,
    orbit_scale: float,
    tangent_scale: float,
    weights: Mapping[str, float],
    coverage_floor: float,
    huber_beta_min: float,
    temperature: float,
    prototype_matrix: Optional[torch.Tensor] = None,
    tangent_perturbed: Optional[Mapping[str, torch.Tensor]] = None,
    tangent_delta: float = 0.05,
    tangent_budget: Optional[torch.Tensor] = None,
    tangent_valid: Optional[torch.Tensor] = None,
    nuisance_target: Optional[torch.Tensor] = None,
    nuisance_valid: Optional[torch.Tensor] = None,
    fingerprint_perturbed: Optional[Mapping[str, torch.Tensor]] = None,
    fingerprint_minimum: float = 0.50,
    relation_pairs: Optional[torch.Tensor] = None,
    loss_normalizer=None,
    rx_v2: bool = False,
    teacher_prior: Optional[torch.Tensor] = None,
    anchor_strength: float = 0.50,
    component_scales: Optional[Mapping[str, float]] = None,
    extra_components: Optional[Mapping[str, torch.Tensor]] = None,
    logit_valid: Optional[torch.Tensor] = None,
) -> dict[str, Any]:
    """Compose DAOT-STN losses without consulting TX labels or pseudo labels."""

    if len(teacher_views) < 2:
        raise ValueError("DAOT requires at least two teacher views")
    student_z = student_channel["z_id"]
    teacher_features = torch.stack([view["z_id"].detach() for view in teacher_views], dim=1)
    teacher_logits = torch.stack([view["tx_logits"].detach() for view in teacher_views], dim=1)
    if bool(rx_v2):
        if teacher_prior is None:
            teacher_prior = torch.ones(
                teacher_features.shape[1], device=teacher_features.device, dtype=torch.float32
            )
            teacher_prior[0] = float(max(1, teacher_features.shape[1] - 1))
        target_z, feature_weights, aggregate_diag = anchored_spherical_orbit_target(
            teacher_features,
            reliability=reliability,
            importance=importance,
            prior=teacher_prior,
            coverage_floor=float(coverage_floor),
            huber_beta_min=float(huber_beta_min),
            anchor_strength=float(anchor_strength),
        )
    else:
        target_z, feature_weights, aggregate_diag = robust_spherical_orbit_target(
            teacher_features,
            reliability=reliability,
            importance=importance,
            coverage_floor=float(coverage_floor),
            huber_beta_min=float(huber_beta_min),
        )
    consensus, confidence, target_logits = _teacher_consensus_and_target_logits(
        teacher_logits,
        feature_weights,
        temperature=float(temperature),
    )
    if logit_valid is not None:
        logit_valid = logit_valid.to(device=consensus.device, dtype=torch.bool).reshape(-1)
        if logit_valid.numel() != consensus.numel():
            raise ValueError("logit_valid must align with the batch")
        consensus = consensus & logit_valid
    recoverability = recoverability.to(device=student_z.device, dtype=torch.float32).clamp(0.0, 1.0)
    loss_z = 0.5 * (
        orbit_feature_loss(student_clean["z_id"], target_z, recoverability=recoverability)
        + orbit_feature_loss(student_channel["z_id"], target_z, recoverability=recoverability)
    )
    loss_logit = orbit_logit_distillation_loss(
        student_channel["tx_logits"],
        target_logits,
        consensus=consensus,
        confidence_weight=confidence,
        temperature=float(temperature),
    )
    loss_proto = _zero(student_z)
    if prototype_matrix is not None and int(prototype_matrix.numel()) > 0:
        prototypes = F.normalize(prototype_matrix.detach().float(), dim=-1)
        student_similarity = F.normalize(student_z.float(), dim=-1) @ prototypes.t()
        teacher_similarity = F.normalize(target_z.detach().float(), dim=-1) @ prototypes.t()
        loss_proto = orbit_prototype_distillation_loss(
            student_similarity,
            teacher_similarity,
            consensus=consensus,
            temperature=float(temperature),
        )
    loss_relation = _zero(student_z)
    if relation_pairs is not None and int(relation_pairs.numel()) > 0:
        loss_relation = orbit_relation_loss(
            student_channel["z_id"],
            target_z,
            pairs=relation_pairs,
        )

    loss_tangent = _zero(student_z)
    nuisance_sensitivity = student_z.new_zeros((student_z.shape[0],))
    if tangent_perturbed is not None:
        sensitivity_fn = chordal_sensitivity if bool(rx_v2) else angular_sensitivity
        nuisance_sensitivity = sensitivity_fn(
            student_channel["z_id"],
            tangent_perturbed["z_id"],
            delta=float(tangent_delta),
        )
        budgets = tangent_budget if tangent_budget is not None else torch.zeros_like(nuisance_sensitivity)
        valid = tangent_valid if tangent_valid is not None else torch.ones_like(nuisance_sensitivity, dtype=torch.bool)
        loss_tangent = selective_tangent_loss(nuisance_sensitivity, budgets=budgets, valid=valid)

    loss_nuisance = _zero(student_z)
    if nuisance_target is not None and nuisance_valid is not None:
        mean = student_channel.get("daot_nuisance_mean")
        log_variance = student_channel.get("daot_nuisance_log_variance")
        if torch.is_tensor(mean) and torch.is_tensor(log_variance):
            loss_nuisance = heteroscedastic_nuisance_loss(
                mean,
                log_variance,
                nuisance_target,
                valid=nuisance_valid,
            )

    loss_fingerprint = _zero(student_z)
    fingerprint_sensitivity_value = student_z.new_zeros((student_z.shape[0],))
    if fingerprint_perturbed is not None:
        sensitivity_fn = chordal_sensitivity if bool(rx_v2) else angular_sensitivity
        fingerprint_sensitivity_value = sensitivity_fn(
            student_clean["z_id"],
            fingerprint_perturbed["z_id"],
            delta=float(tangent_delta),
        )
        minimum = torch.full_like(fingerprint_sensitivity_value, float(fingerprint_minimum))
        loss_fingerprint = fingerprint_keep_loss(
            fingerprint_sensitivity_value,
            minimum=minimum,
        )

    components = {
        "orbit_z": loss_z,
        "orbit_logit": loss_logit,
        "orbit_proto": loss_proto,
        "orbit_relation": loss_relation,
        "tangent": loss_tangent,
        "nuisance": loss_nuisance,
        "fingerprint": loss_fingerprint,
        "route": _zero(student_z),
        "rx": _zero(student_z),
        "tail": _zero(student_z),
        "clean_anchor": orbit_feature_loss(
            student_clean["z_id"],
            teacher_views[0]["z_id"].detach(),
            recoverability=torch.ones_like(recoverability),
        ),
        "subspace": _zero(student_z),
    }
    if extra_components is not None:
        unknown = set(extra_components) - set(components)
        if unknown:
            raise ValueError(f"unknown DAOT component(s): {sorted(unknown)}")
        components.update({str(name): value for name, value in extra_components.items()})
    normalized_components = components
    loss_scales: dict[str, float] = {}
    if loss_normalizer is not None:
        normalized_components, loss_scales = loss_normalizer.normalize(
            components,
            active={
                name: float(weights.get(name, 0.0)) > 0.0
                and (
                    float((component_scales or {}).get(name, 0.0)) > 0.0
                    if bool(rx_v2)
                    else (float(tangent_scale) > 0.0 if name == "tangent" else float(orbit_scale) > 0.0)
                )
                for name in components
            },
        )
    if bool(rx_v2):
        component_scales = dict(component_scales or {})
        weighted_components = {
            name: float(component_scales.get(name, 0.0))
            * float(weights.get(name, 0.0))
            * normalized_components[name]
            for name in components
        }
        total = sum(weighted_components.values())
    else:
        total = float(orbit_scale) * (
            float(weights.get("orbit_z", 0.0)) * normalized_components["orbit_z"]
            + float(weights.get("orbit_logit", 0.0)) * normalized_components["orbit_logit"]
            + float(weights.get("orbit_proto", 0.0)) * normalized_components["orbit_proto"]
            + float(weights.get("orbit_relation", 0.0)) * normalized_components["orbit_relation"]
            + float(weights.get("nuisance", 0.0)) * normalized_components["nuisance"]
            + float(weights.get("fingerprint", 0.0)) * normalized_components["fingerprint"]
        ) + float(tangent_scale) * float(weights.get("tangent", 0.0)) * normalized_components["tangent"]
        weighted_components = {
            name: (
                float(tangent_scale) if name == "tangent" else float(orbit_scale)
            )
            * float(weights.get(name, 0.0))
            * normalized_components[name]
            for name in components
        }
    diagnostics = {
        **aggregate_diag,
        "consensus_mask": consensus,
        "teacher_confidence": confidence,
        "nuisance_sensitivity": nuisance_sensitivity,
        "fingerprint_sensitivity": fingerprint_sensitivity_value,
        "fingerprint_selectivity": fingerprint_selectivity(
            fingerprint_sensitivity_value,
            nuisance_sensitivity,
        ),
        **{f"loss_scale_{name}": float(scale) for name, scale in loss_scales.items()},
    }
    return {
        "loss": total,
        "components": components,
        "normalized_components": normalized_components,
        "weighted_components": weighted_components,
        "diagnostics": diagnostics,
        "target_z": target_z,
    }
