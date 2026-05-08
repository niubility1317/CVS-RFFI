from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

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
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class StyleConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int, dilation: int = 1, pool: int = 1, drop: float = 0.1):
        super().__init__()
        pad = ((k - 1) // 2) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=pad, dilation=dilation, bias=False),
            nn.GroupNorm(max(1, min(8, out_ch // 8)), out_ch),
            nn.GELU(),
            nn.Conv1d(out_ch, out_ch, kernel_size=1, bias=False),
            nn.GroupNorm(max(1, min(8, out_ch // 8)), out_ch),
            nn.GELU(),
            nn.AvgPool1d(pool) if pool > 1 else nn.Identity(),
            nn.Dropout(drop),
        )
        self.skip = None
        if in_ch != out_ch or pool > 1:
            self.skip = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False),
                nn.AvgPool1d(pool) if pool > 1 else nn.Identity(),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.net(x)
        s = x if self.skip is None else self.skip(x)
        return y + s


class StatisticsPooling(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,C,L)
        mu = x.mean(dim=-1)
        std = x.std(dim=-1, unbiased=False).clamp_min(1e-5)
        xc = x - mu.unsqueeze(-1)
        skew = (xc.pow(3).mean(dim=-1) / (std.pow(3) + 1e-6)).clamp(-8.0, 8.0)
        kurt = (xc.pow(4).mean(dim=-1) / (std.pow(4) + 1e-6)).clamp(0.0, 50.0)
        return torch.cat([mu, std, skew, kurt], dim=1)


class ReceiverStyleBackbone(nn.Module):
    """
    Light, asymmetric, style-biased nuisance encoder.
    It prefers larger kernels + wider receptive field + statistics pooling,
    instead of duplicating the sharp device-discriminative CV-SincNet backbone.
    """
    def __init__(self, emb_dim: int = 128, stem_ch: int = 32, drop: float = 0.1):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(2, stem_ch, kernel_size=15, padding=7, bias=False),
            nn.GroupNorm(max(1, min(8, stem_ch // 4)), stem_ch),
            nn.GELU(),
            nn.AvgPool1d(2),
        )
        self.b1 = StyleConvBlock(stem_ch, 64, k=15, dilation=1, pool=2, drop=drop)
        self.b2 = StyleConvBlock(64, 96, k=11, dilation=2, pool=2, drop=drop)
        self.b3 = StyleConvBlock(96, 128, k=9, dilation=4, pool=1, drop=drop)
        self.pool = StatisticsPooling()
        stat_dim = 128 * 4
        self.proj = nn.Sequential(
            nn.Linear(stat_dim, max(128, emb_dim)),
            nn.LayerNorm(max(128, emb_dim)),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(max(128, emb_dim), emb_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nan_to_num(x.float(), nan=0.0, posinf=0.0, neginf=0.0)
        h = self.stem(x)
        h = self.b1(h)
        h = self.b2(h)
        h = self.b3(h)
        stats = self.pool(h)
        z = self.proj(stats)
        return z


class DualCVSincNetDisentangle(nn.Module):
    def __init__(
        self,
        num_classes: int,
        num_domains: int,
        model_size: str = "S",
        dataset: str = "wisig",
        input_len: int = 256,
        sample_rate_hz: float = 25e6,
        drop: float = 0.1,
        nuisance_dim: int = 128,
        **backbone_kwargs,
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
            **backbone_kwargs,
        )
        self.emb_dim = self._infer_emb_dim(self.id_backbone)
        self.nuisance_backbone = ReceiverStyleBackbone(emb_dim=max(int(nuisance_dim), self.emb_dim // 2), drop=drop)
        self.nuisance_dim = max(int(nuisance_dim), self.emb_dim // 2)

        self.dom_head = MLPHead(self.nuisance_dim, self.num_domains, hidden=max(64, self.nuisance_dim), drop=drop)
        self.adv_head = MLPHead(self.emb_dim, self.num_domains, hidden=max(64, self.emb_dim // 2), drop=drop)
        self.probe_head = MLPHead(self.emb_dim, self.num_domains, hidden=max(32, self.emb_dim // 3), drop=0.05)
        self.style_tx_probe_head = MLPHead(self.nuisance_dim, self.num_classes, hidden=max(32, self.nuisance_dim // 2), drop=0.05)

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
    def _pick(aux: Dict[str, torch.Tensor], keys) -> torch.Tensor:
        for key in keys:
            v = aux.get(key, None)
            if torch.is_tensor(v):
                return v
        raise KeyError(f"Cannot find tensor in keys={keys}, available={list(aux.keys())}")

    def forward(
        self,
        x: torch.Tensor,
        y_tx: Optional[torch.Tensor] = None,
        grl_lambda: float = 1.0,
        return_aux: bool = False,
    ):
        aux_id = self.id_backbone(x, y=y_tx, return_aux=True)
        tx_logits = aux_id["logits"]
        z_pa = self._pick(aux_id, ("z_pa", "feat_cls", "feat_con", "base"))
        z_id = self._pick(aux_id, ("z_id", "feat_joint", "feat_cls", "feat_con"))
        z_txc = self._pick(aux_id, ("z_txc", "feat_dac", "feat_imp"))

        z_rxc = self.nuisance_backbone(x)
        dom_logits = self.dom_head(z_rxc)
        adv_dom_logits = self.adv_head(grad_reverse(z_pa, grl_lambda))
        probe_dom_logits = self.probe_head(z_pa.detach())
        style_tx_probe_logits = self.style_tx_probe_head(z_rxc.detach())

        if not return_aux:
            return tx_logits

        return {
            "tx_logits": tx_logits,
            "dom_logits": dom_logits,
            "adv_dom_logits": adv_dom_logits,
            "probe_dom_logits": probe_dom_logits,
            "style_tx_probe_logits": style_tx_probe_logits,
            "z_pa": z_pa,
            "z_txc": z_txc,
            "z_id": z_id,
            "z_rxc": z_rxc,
            "aux_id": aux_id,
        }


def build_dual_model(
    num_classes: int,
    num_domains: int,
    model_size: str = "S",
    dataset: str = "wisig",
    input_len: int = 256,
    sample_rate_hz: float = 25e6,
    **backbone_kwargs,
) -> DualCVSincNetDisentangle:
    return DualCVSincNetDisentangle(
        num_classes=num_classes,
        num_domains=num_domains,
        model_size=model_size,
        dataset=dataset,
        input_len=input_len,
        sample_rate_hz=sample_rate_hz,
        **backbone_kwargs,
    )
