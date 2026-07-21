"""Exact dual-feature inference for sealed ADV3B02 checkpoints.

This path exposes only the identity feature, the domain feature, and the
identity classifier logits required by future C-dom/C-joint research.  It
deliberately bypasses ``DualCVSincNetDisentangle.forward`` so ``dom_head``,
domain logits, gradient-reversal heads, and receiver-ID decisions can never
enter the deployment output.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from model_dual_cvsincnet import backbone_forward_compat


FEATURE_DIM = 160
ID_FEATURE_KEY = "feat_joint"
DOM_FEATURE_KEY = "feat_imp"


class DualFeatureForwardError(ValueError):
    """Raised when the strict ADV3B02 dual-feature contract drifts."""


def _require_model_contract(model: nn.Module) -> None:
    if model.training:
        raise DualFeatureForwardError("dual feature inference requires eval mode")
    if str(getattr(model, "id_feature_key", "")) != ID_FEATURE_KEY:
        raise DualFeatureForwardError("ADV3B02 z_id feature key must be feat_joint")
    if str(getattr(model, "dom_feature_key", "")) != DOM_FEATURE_KEY:
        raise DualFeatureForwardError("ADV3B02 z_dom feature key must be feat_imp")
    for name in ("id_backbone", "dom_backbone", "dom_enhancer"):
        if not isinstance(getattr(model, name, None), nn.Module):
            raise DualFeatureForwardError(f"ADV3B02 dual feature module missing: {name}")
    for name in ("_pick_z_id", "_pick_z_dom"):
        if not callable(getattr(model, name, None)):
            raise DualFeatureForwardError(f"ADV3B02 dual feature selector missing: {name}")


def _require_input(x: torch.Tensor) -> None:
    if not torch.is_tensor(x):
        raise DualFeatureForwardError("received IQ must be a tensor")
    if x.dtype != torch.float32:
        raise DualFeatureForwardError("received IQ must use float32")
    if x.ndim != 3 or int(x.shape[0]) < 1 or int(x.shape[1]) != 2:
        raise DualFeatureForwardError("received IQ must have shape [N,2,T]")
    if not bool(torch.isfinite(x).all().item()):
        raise DualFeatureForwardError("received IQ must be finite")


def _require_output(
    value: Any,
    *,
    rows: int,
    width: int | None,
    name: str,
) -> torch.Tensor:
    if not torch.is_tensor(value):
        raise DualFeatureForwardError(f"{name} must be a tensor")
    if value.dtype != torch.float32:
        raise DualFeatureForwardError(f"{name} must use float32")
    if (
        value.ndim != 2
        or int(value.shape[0]) != int(rows)
        or (width is not None and int(value.shape[1]) != int(width))
        or (width is None and int(value.shape[1]) < 2)
    ):
        expected = f"[N,{width}]" if width is not None else "[N,C>=2]"
        raise DualFeatureForwardError(f"{name} must have shape {expected}")
    if not bool(torch.isfinite(value).all().item()):
        raise DualFeatureForwardError(f"{name} must be finite")
    return value


def dual_feature_forward(
    model: nn.Module, x: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return exact ``z_id160``, ``z_dom160``, and TX logits from one IQ call.

    The two frozen backbones each execute exactly once on the same tensor.
    ``z_dom`` follows the training-time path
    ``dom_backbone.feat_imp -> dom_enhancer(feat_imp, x)``.  No domain head,
    domain label, receiver identifier, clean/source record, or query truth is
    accepted by this API.
    """

    _require_model_contract(model)
    _require_input(x)
    with torch.no_grad():
        aux_id: dict[str, Any] = backbone_forward_compat(
            model.id_backbone,
            x,
            y=None,
            return_aux=True,
            domain_labels=None,
        )
        aux_dom: dict[str, Any] = backbone_forward_compat(
            model.dom_backbone,
            x,
            y=None,
            return_aux=True,
            domain_labels=None,
        )
        if not isinstance(aux_id, dict) or not isinstance(aux_dom, dict):
            raise DualFeatureForwardError("ADV3B02 backbones must return auxiliary mappings")
        exact_z_id = aux_id.get(ID_FEATURE_KEY)
        exact_z_dom = aux_dom.get(DOM_FEATURE_KEY)
        if not torch.is_tensor(exact_z_id):
            raise DualFeatureForwardError("id backbone must expose exact feat_joint")
        if not torch.is_tensor(exact_z_dom):
            raise DualFeatureForwardError("dom backbone must expose exact feat_imp")
        z_id = model._pick_z_id(aux_id)
        z_dom_raw = model._pick_z_dom(aux_dom)
        if z_id is not exact_z_id:
            raise DualFeatureForwardError("z_id selector must return exact feat_joint")
        if z_dom_raw is not exact_z_dom:
            raise DualFeatureForwardError("z_dom selector must return exact feat_imp")
        enhanced = model.dom_enhancer(z_dom_raw, x)
        if not isinstance(enhanced, (tuple, list)) or len(enhanced) != 2:
            raise DualFeatureForwardError("dom_enhancer output contract drift")
        z_dom = enhanced[0]
        tx_logits = aux_id.get("logits")

    count = int(x.shape[0])
    return (
        _require_output(z_id, rows=count, width=FEATURE_DIM, name="z_id"),
        _require_output(z_dom, rows=count, width=FEATURE_DIM, name="z_dom"),
        _require_output(tx_logits, rows=count, width=None, name="tx_logits"),
    )


def dual_feature_components_forward(
    id_backbone: nn.Module,
    dom_backbone: nn.Module,
    dom_enhancer: nn.Module,
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the exact dual path from an explicitly stripped component set.

    Export wrappers use this variant so forbidden heads are not merely skipped
    at runtime: they are absent from the serialized module tree altogether.
    """

    if id_backbone.training or dom_backbone.training or dom_enhancer.training:
        raise DualFeatureForwardError("dual feature components require eval mode")
    _require_input(x)
    with torch.no_grad():
        aux_id = backbone_forward_compat(
            id_backbone,
            x,
            y=None,
            return_aux=True,
            domain_labels=None,
        )
        aux_dom = backbone_forward_compat(
            dom_backbone,
            x,
            y=None,
            return_aux=True,
            domain_labels=None,
        )
        if not isinstance(aux_id, dict) or not isinstance(aux_dom, dict):
            raise DualFeatureForwardError("ADV3B02 backbones must return auxiliary mappings")
        z_id = aux_id.get(ID_FEATURE_KEY)
        z_dom_raw = aux_dom.get(DOM_FEATURE_KEY)
        if not torch.is_tensor(z_id):
            raise DualFeatureForwardError("id backbone must expose exact feat_joint")
        if not torch.is_tensor(z_dom_raw):
            raise DualFeatureForwardError("dom backbone must expose exact feat_imp")
        enhanced = dom_enhancer(z_dom_raw, x)
        if not isinstance(enhanced, (tuple, list)) or len(enhanced) != 2:
            raise DualFeatureForwardError("dom_enhancer output contract drift")
        z_dom = enhanced[0]
        tx_logits = aux_id.get("logits")
    count = int(x.shape[0])
    return (
        _require_output(z_id, rows=count, width=FEATURE_DIM, name="z_id"),
        _require_output(z_dom, rows=count, width=FEATURE_DIM, name="z_dom"),
        _require_output(tx_logits, rows=count, width=None, name="tx_logits"),
    )


__all__ = [
    "DOM_FEATURE_KEY",
    "DualFeatureForwardError",
    "FEATURE_DIM",
    "ID_FEATURE_KEY",
    "dual_feature_components_forward",
    "dual_feature_forward",
]
