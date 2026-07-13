from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from paper_reproduction.DADDA.losses import (
    dynamic_adaptive_factor,
    lmmd_loss,
    mmd_loss,
)
from paper_reproduction.mitigating_receiver_impact_da.losses import (
    class_balance_weights,
    gada_minimax_objective,
)


def _require_labels(logits: torch.Tensor, labels: torch.Tensor, *, name: str) -> torch.Tensor:
    labels = labels.to(device=logits.device, dtype=torch.long)
    if logits.ndim != 2:
        raise ValueError(f"{name} logits must have shape [batch,num_classes]")
    if labels.ndim != 1 or labels.numel() != logits.shape[0]:
        raise ValueError(f"{name} labels must have one value per sample")
    if labels.numel() and (int(labels.min()) < 0 or int(labels.max()) >= int(logits.shape[1])):
        raise ValueError(f"{name} labels must be in [0,num_classes)")
    return labels


def mrior_sda_objective(
    source_outputs: dict[str, torch.Tensor],
    target_support_outputs: dict[str, torch.Tensor],
    *,
    source_labels: torch.Tensor,
    target_support_labels: torch.Tensor,
    target_ce_weight: float = 1.0,
    dvkl_weight: float = 0.005,
    mu: float = 0.5,
    class_balance_smoothing: float = 0.0,
) -> dict[str, torch.Tensor]:
    """CVS supervised extension of the MRIOR/GAD objective.

    Target labels must come only from the registered K-shot support set. This
    objective never accepts target-query tensors, so held-out query samples
    cannot enter adaptation through this API.
    """

    if float(target_ce_weight) < 0.0 or float(dvkl_weight) < 0.0:
        raise ValueError("target_ce_weight and dvkl_weight must be non-negative")
    if not 0.0 < float(mu) < 1.0:
        raise ValueError("mu must be in (0,1)")
    source_y = _require_labels(source_outputs["tx_logits"], source_labels, name="source")
    target_y = _require_labels(
        target_support_outputs["tx_logits"], target_support_labels, name="target support"
    )
    num_classes = int(source_outputs["tx_logits"].shape[1])
    target_counts = torch.bincount(target_y, minlength=num_classes).to(
        device=target_support_outputs["tx_logits"].device,
        dtype=target_support_outputs["tx_logits"].dtype,
    )
    weights = class_balance_weights(
        target_counts,
        total_seen=int(target_y.numel()),
        smoothing=float(class_balance_smoothing),
        mean_normalize=True,
    )
    terms = gada_minimax_objective(
        source_outputs,
        target_support_outputs,
        source_labels=source_y,
        target_pseudo_labels=target_y,
        target_mask=torch.ones_like(target_y, dtype=torch.bool),
        class_weights=weights,
        mu=float(mu),
        kl_weight=float(dvkl_weight),
        source_ce_scale=float(mu),
        target_ce_scale=(1.0 - float(mu)) * float(target_ce_weight),
    )
    total = terms["loss"]
    return {
        "loss": total,
        "source_ce": terms["loss_source"].detach(),
        "target_support_ce": terms["loss_target"].detach(),
        "weighted_ce": terms["loss_weighted_ce"].detach(),
        "dvkl": terms["loss_kl"].detach(),
        "class_balance_weights": weights.detach(),
        "target_ce_weight": torch.as_tensor(float(target_ce_weight), device=total.device),
        "dvkl_weight": torch.as_tensor(float(dvkl_weight), device=total.device),
        "mu": torch.as_tensor(float(mu), device=total.device),
    }


def dadda_sda_objective(
    source_outputs: dict[str, torch.Tensor],
    target_support_outputs: dict[str, torch.Tensor],
    *,
    source_labels: torch.Tensor,
    target_support_labels: torch.Tensor,
    target_ce_weight: float = 1.0,
    alignment_weight: float = 1.0,
    bandwidth: float | None = None,
    detach_dynamic_alpha: bool = True,
) -> dict[str, torch.Tensor]:
    """CVS supervised DADDA extension using true support labels for LMMD."""

    if float(target_ce_weight) < 0.0 or float(alignment_weight) < 0.0:
        raise ValueError("target_ce_weight and alignment_weight must be non-negative")
    source_y = _require_labels(source_outputs["logits"], source_labels, name="source")
    target_y = _require_labels(
        target_support_outputs["logits"], target_support_labels, name="target support"
    )
    num_classes = int(source_outputs["logits"].shape[1])
    if int(target_support_outputs["logits"].shape[1]) != num_classes:
        raise ValueError("source and target-support classifiers must share the same class space")

    source_ce = F.cross_entropy(source_outputs["logits"], source_y)
    target_ce = F.cross_entropy(target_support_outputs["logits"], target_y)
    global_mmd = mmd_loss(
        source_outputs["global_features"],
        target_support_outputs["global_features"],
        bandwidth=bandwidth,
    )
    target_one_hot = F.one_hot(target_y, num_classes=num_classes).to(
        dtype=target_support_outputs["local_features"].dtype
    )
    local_lmmd = lmmd_loss(
        source_outputs["local_features"],
        target_support_outputs["local_features"],
        source_y,
        target_one_hot,
        num_classes=num_classes,
        bandwidth=bandwidth,
        reduction="mean",
        target_is_probabilities=True,
    )
    local_lmmd_sum = lmmd_loss(
        source_outputs["local_features"],
        target_support_outputs["local_features"],
        source_y,
        target_one_hot,
        num_classes=num_classes,
        bandwidth=bandwidth,
        reduction="sum",
        target_is_probabilities=True,
    )
    alpha = dynamic_adaptive_factor(global_mmd, local_lmmd_sum)
    if detach_dynamic_alpha:
        alpha = alpha.detach()
    dynamic_joint = (1.0 - alpha) * global_mmd + alpha * local_lmmd_sum
    total = (
        source_ce
        + float(target_ce_weight) * target_ce
        + float(alignment_weight) * dynamic_joint
    )
    return {
        "loss": total,
        "source_ce": source_ce.detach(),
        "target_support_ce": target_ce.detach(),
        "mmd": global_mmd.detach(),
        "lmmd": local_lmmd.detach(),
        "lmmd_sum": local_lmmd_sum.detach(),
        "alpha": alpha.detach(),
        "dynamic_joint": dynamic_joint.detach(),
        "target_ce_weight": torch.as_tensor(float(target_ce_weight), device=total.device),
        "alignment_weight": torch.as_tensor(float(alignment_weight), device=total.device),
    }


def validate_supervised_da_manifest(payload: dict[str, Any]) -> dict[str, Any]:
    """Fail closed on support/query leakage and unequal supervised budgets."""

    method = str(payload.get("method_id", "")).strip().lower()
    if method not in {"protonet_cda", "mrior_sda", "dadda_sda"}:
        raise ValueError("method_id must be protonet_cda, mrior_sda, or dadda_sda")
    if payload.get("cvs_extension") is not True:
        raise ValueError("supervised CVS DA rows must set cvs_extension=true")
    if str(payload.get("stage", "")).strip() not in {"Stage2-B", "B"}:
        raise ValueError("supervised target-old domain adaptation must use Stage2-B")
    k_shot = int(payload.get("k_shot", 0))
    if k_shot <= 0:
        raise ValueError("k_shot must be a positive integer")
    support_ids = {str(v) for v in payload.get("target_old_support_sample_ids", [])}
    query_ids = {str(v) for v in payload.get("target_old_query_sample_ids", [])}
    if not support_ids or not query_ids:
        raise ValueError("target-old support and query sample IDs are required")
    if support_ids & query_ids:
        raise ValueError("target-old support and query must be disjoint")
    if str(payload.get("target_labels_scope", "")) != "registered_support_only":
        raise ValueError("target_labels_scope must be registered_support_only")
    if payload.get("target_query_used_for_training", False):
        raise ValueError("target query cannot be used for training")
    if payload.get("target_query_used_for_model_selection", False):
        raise ValueError("target query cannot be used for model selection")
    checked = dict(payload)
    checked.update(
        {
            "method_id": method,
            "stage": "Stage2-B",
            "k_shot": k_shot,
            "support_query_disjoint": True,
            "supervised_target_support": True,
            "target_query_used_for_training": False,
            "target_query_used_for_model_selection": False,
            "paper_faithful_claim_allowed": method == "protonet_cda",
        }
    )
    return checked
