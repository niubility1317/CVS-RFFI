from __future__ import annotations

from typing import Mapping

import torch
import torch.nn.functional as F


def extract_feature(base_model, x: torch.Tensor) -> torch.Tensor:
    if hasattr(base_model, "extract_feature"):
        return base_model.extract_feature(x)
    out = base_model(x, return_aux=True)
    if isinstance(out, Mapping):
        for key in ("z_id_raw", "z_id", "feat_joint", "feat_cls", "base"):
            value = out.get(key)
            if torch.is_tensor(value):
                return value
    raise AttributeError("base_model must expose extract_feature(x) or known aux feature keys.")


def classify_feature(base_model, x: torch.Tensor, feat: torch.Tensor | None = None) -> torch.Tensor:
    if hasattr(base_model, "classify"):
        try:
            return base_model.classify(x, feat)
        except TypeError:
            return base_model.classify(x)
    classifier = getattr(base_model, "classifier", None)
    if classifier is not None:
        if feat is None:
            feat = extract_feature(base_model, x)
        return classifier(feat)
    out = base_model(x)
    if torch.is_tensor(out):
        return out
    if isinstance(out, Mapping):
        for key in ("logits", "tx_logits", "base_logits"):
            value = out.get(key)
            if torch.is_tensor(value):
                return value
    raise AttributeError("base_model must expose classify(x), classifier(feat), or logits output.")


def freeze_module(module) -> None:
    module.eval()
    for param in module.parameters():
        param.requires_grad = False


def identity_preservation_loss(
    base_model,
    x_hat: torch.Tensor,
    x_clean: torch.Tensor,
    labels: torch.Tensor | None = None,
    *,
    ce_weight: float = 0.5,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    base_model.eval()
    with torch.no_grad():
        z_clean = extract_feature(base_model, x_clean).detach()
    z_hat = extract_feature(base_model, x_hat)
    cos = F.cosine_similarity(z_hat, z_clean, dim=-1)
    loss_feat = 1.0 - cos.mean()
    loss_ce = x_hat.new_tensor(0.0)
    if labels is not None:
        logits_hat = classify_feature(base_model, x_hat, z_hat)
        loss_ce = F.cross_entropy(logits_hat, labels.long().view(-1))
    loss = loss_feat + float(ce_weight) * loss_ce
    return loss, {
        "loss_id_feat": loss_feat.detach(),
        "loss_id_ce": loss_ce.detach(),
        "id_feature_cos": cos.detach().mean(),
    }
