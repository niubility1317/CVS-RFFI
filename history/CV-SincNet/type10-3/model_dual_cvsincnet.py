
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

try:
    from model_modified import build_model as build_single_model
except Exception:
    from model import build_model as build_single_model


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = float(lambd)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


def grad_reverse(x: torch.Tensor, lambd: float = 1.0) -> torch.Tensor:
    return GradReverse.apply(x, lambd)


class MLPHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: Optional[int] = None, drop: float = 0.1):
        super().__init__()
        if hidden is None:
            hidden = max(64, in_dim // 2)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(drop),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DualCVSincNetDisentangle(nn.Module):
    """
    双 CV-SincNet 解耦模型。

    这版与原版相比不改 state_dict 结构，只额外在 return_aux=True 时
    暴露 id/dom 分支的关键中间特征，便于训练脚本做：
      - PA robustness consistency on id_feat_joint / id_feat_imp
      - DAC / PA strength supervision
      - branch-selective sensitivity ranking
    """
    def __init__(
        self,
        num_classes: int,
        num_domains: int,
        model_size: str = "S",
        dataset: str = "wisig",
        input_len: int = 256,
        sample_rate_hz: float = 25e6,
        drop: float = 0.1,
        id_feature_key: str = "feat_joint",
        dom_feature_key: str = "feat_imp",
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.num_domains = int(max(1, num_domains))
        self.id_feature_key = str(id_feature_key)
        self.dom_feature_key = str(dom_feature_key)

        self.id_backbone = build_single_model(
            num_classes=num_classes,
            model_size=model_size,
            dataset=dataset,
            input_len=input_len,
            sample_rate_hz=sample_rate_hz,
        )
        self.dom_backbone = build_single_model(
            num_classes=num_classes,
            model_size=model_size,
            dataset=dataset,
            input_len=input_len,
            sample_rate_hz=sample_rate_hz,
        )

        self.emb_dim = self._infer_emb_dim(self.id_backbone)
        self.dom_head = MLPHead(self.emb_dim, self.num_domains, hidden=max(64, self.emb_dim // 2), drop=drop)
        self.adv_head = MLPHead(self.emb_dim, self.num_domains, hidden=max(64, self.emb_dim // 2), drop=drop)
        self.probe_head = MLPHead(self.emb_dim, self.num_domains, hidden=max(32, self.emb_dim // 3), drop=0.05)

    @staticmethod
    def _infer_emb_dim(backbone: nn.Module) -> int:
        cls_head = getattr(backbone, "cls_head", None)
        if cls_head is not None:
            head = getattr(cls_head, "head", None)
            weight = getattr(head, "weight", None)
            if weight is not None and hasattr(weight, "shape") and len(weight.shape) == 2:
                return int(weight.shape[1])
        for name in ("emb_dim", "embed_dim"):
            if hasattr(backbone, name):
                return int(getattr(backbone, name))
        return 256

    @staticmethod
    def _pick_from_keys(aux: Dict[str, torch.Tensor], preferred_key: Optional[str], fallback_keys) -> torch.Tensor:
        keys = []
        if preferred_key is not None and str(preferred_key).strip() != "":
            keys.append(str(preferred_key))
        keys.extend([k for k in fallback_keys if k not in keys])
        for key in keys:
            v = aux.get(key, None)
            if torch.is_tensor(v):
                return v
        raise KeyError(f"Cannot find feature from keys={keys}; available={list(aux.keys())}")

    def _pick_z_id(self, aux: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self._pick_from_keys(aux, self.id_feature_key, ("feat_joint", "feat_cls", "feat_con", "base"))

    def _pick_z_dom(self, aux: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self._pick_from_keys(aux, self.dom_feature_key, ("feat_imp", "feat_pa", "feat_dac", "base", "feat_con", "feat_cls", "feat_joint"))

    def forward(
        self,
        x: torch.Tensor,
        y_tx: Optional[torch.Tensor] = None,
        grl_lambda: float = 1.0,
        return_aux: bool = False,
    ):
        aux_id = self.id_backbone(x, y=y_tx, return_aux=True)
        aux_dom = self.dom_backbone(x, y=None, return_aux=True)

        tx_logits = aux_id["logits"]
        z_id = self._pick_z_id(aux_id)
        z_dom = self._pick_z_dom(aux_dom)

        dom_logits = self.dom_head(z_dom)
        adv_dom_logits = self.adv_head(grad_reverse(z_id, grl_lambda))
        probe_dom_logits = self.probe_head(z_id.detach())

        if not return_aux:
            return tx_logits

        out = {
            "tx_logits": tx_logits,
            "dom_logits": dom_logits,
            "adv_dom_logits": adv_dom_logits,
            "probe_dom_logits": probe_dom_logits,
            "z_id": z_id,
            "z_dom": z_dom,
            "z_id_key": self.id_feature_key,
            "z_dom_key": self.dom_feature_key,
            "aux_id": aux_id,
            "aux_dom": aux_dom,
        }

        # top-level aliases: no parameter changes, only easier access
        alias_map = {
            "id_feat_cls": ("aux_id", "feat_cls"),
            "id_feat_imp": ("aux_id", "feat_imp"),
            "id_feat_dac": ("aux_id", "feat_dac"),
            "id_feat_pa": ("aux_id", "feat_pa"),
            "id_feat_joint": ("aux_id", "feat_joint"),
            "id_feat_con": ("aux_id", "feat_con"),
            "id_base": ("aux_id", "base"),
            "id_dac_pred": ("aux_id", "dac_pred"),
            "id_pa_pred": ("aux_id", "pa_pred"),
            "dom_feat_cls": ("aux_dom", "feat_cls"),
            "dom_feat_imp": ("aux_dom", "feat_imp"),
            "dom_feat_dac": ("aux_dom", "feat_dac"),
            "dom_feat_pa": ("aux_dom", "feat_pa"),
            "dom_feat_joint": ("aux_dom", "feat_joint"),
            "dom_feat_con": ("aux_dom", "feat_con"),
            "dom_base": ("aux_dom", "base"),
            "dom_dac_pred": ("aux_dom", "dac_pred"),
            "dom_pa_pred": ("aux_dom", "pa_pred"),
        }
        for name, (g, k) in alias_map.items():
            v = out[g].get(k, None)
            if torch.is_tensor(v):
                out[name] = v
        return out


def build_dual_model(
    num_classes: int,
    num_domains: int,
    model_size: str = "S",
    dataset: str = "wisig",
    input_len: int = 256,
    sample_rate_hz: float = 25e6,
    id_feature_key: str = "feat_joint",
    dom_feature_key: str = "feat_imp",
) -> DualCVSincNetDisentangle:
    return DualCVSincNetDisentangle(
        num_classes=num_classes,
        num_domains=num_domains,
        model_size=model_size,
        dataset=dataset,
        input_len=input_len,
        sample_rate_hz=sample_rate_hz,
        id_feature_key=id_feature_key,
        dom_feature_key=dom_feature_key,
    )
