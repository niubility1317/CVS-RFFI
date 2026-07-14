"""Identity-only inference for qKNN feature export from dual CVS checkpoints."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from model_dual_cvsincnet import backbone_forward_compat


def can_use_identity_only_forward(model: nn.Module, feature_name: str) -> bool:
    return (
        str(feature_name).strip().lower() == "z_id"
        and hasattr(model, "id_backbone")
        and callable(getattr(model, "_pick_z_id", None))
    )


def identity_only_feature_forward(
    model: nn.Module, x: torch.Tensor, feature_name: str
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Return the exact z_id/logits without executing the unused domain branch.

    Returns ``None`` for models or feature names that require the ordinary full
    forward path.  qKNN consumes only z_id and therefore does not need z_dom,
    domain logits, GRL heads, or domain-enhancer statistics during inference.
    """
    if not can_use_identity_only_forward(model, feature_name):
        return None
    aux_id: dict[str, Any] = backbone_forward_compat(
        model.id_backbone,
        x,
        y=None,
        return_aux=True,
        domain_labels=None,
    )
    z_id = model._pick_z_id(aux_id)
    logits = aux_id.get("logits")
    if not torch.is_tensor(z_id) or not torch.is_tensor(logits):
        raise KeyError("identity backbone did not return tensor z_id/logits")
    return z_id.float(), logits.float()
