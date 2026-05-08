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
    方案 A：双 CV-SincNet 的“指纹—域因素解耦”架构。

    - id_backbone: 提取发射机身份相关特征 z_id
    - dom_backbone: 提取日期/接收机/信道等域因素特征 z_dom
    - tx_head:     由 id_backbone 内部 cls head 产生
    - dom_head:    对 z_dom 做域分类
    - adv_head:    通过 GRL 让 z_id 域不可分
    - probe_head:  仅用于监控 z_id 里残余域信息，不回传到 z_id
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
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.num_domains = int(max(1, num_domains))

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
    def _pick_z_id(aux: Dict[str, torch.Tensor]) -> torch.Tensor:
        for key in ("feat_cls", "feat_con", "base", "feat_joint"):
            v = aux.get(key, None)
            if torch.is_tensor(v):
                return v
        raise KeyError(f"Cannot find z_id in keys={list(aux.keys())}")

    @staticmethod
    def _pick_z_dom(aux: Dict[str, torch.Tensor]) -> torch.Tensor:
        for key in ("feat_imp", "feat_pa", "feat_dac", "base", "feat_con", "feat_cls", "feat_joint"):
            v = aux.get(key, None)
            if torch.is_tensor(v):
                return v
        raise KeyError(f"Cannot find z_dom in keys={list(aux.keys())}")

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

        return {
            "tx_logits": tx_logits,
            "dom_logits": dom_logits,
            "adv_dom_logits": adv_dom_logits,
            "probe_dom_logits": probe_dom_logits,
            "z_id": z_id,
            "z_dom": z_dom,
            "aux_id": aux_id,
            "aux_dom": aux_dom,
        }


def build_dual_model(
    num_classes: int,
    num_domains: int,
    model_size: str = "S",
    dataset: str = "wisig",
    input_len: int = 256,
    sample_rate_hz: float = 25e6,
) -> DualCVSincNetDisentangle:
    return DualCVSincNetDisentangle(
        num_classes=num_classes,
        num_domains=num_domains,
        model_size=model_size,
        dataset=dataset,
        input_len=input_len,
        sample_rate_hz=sample_rate_hz,
    )
