from __future__ import annotations

from typing import Dict, Optional, Sequence

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
        hidden = int(hidden) if hidden is not None else max(64, in_dim // 2)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DSConvBlock1d(nn.Module):
    def __init__(self, cin: int, cout: int, k: int = 5, dilation: int = 1, pool: int = 2, drop: float = 0.1):
        super().__init__()
        pad = ((k - 1) // 2) * dilation
        gn = max(1, min(8, cout // 8))
        self.dw = nn.Conv1d(cin, cin, kernel_size=k, padding=pad, dilation=dilation, groups=cin, bias=False)
        self.pw = nn.Conv1d(cin, cout, kernel_size=1, bias=False)
        self.norm = nn.GroupNorm(gn, cout)
        self.act = nn.GELU()
        self.pool = nn.AvgPool1d(pool) if pool and pool > 1 else nn.Identity()
        self.drop = nn.Dropout(drop) if drop > 0 else nn.Identity()
        if cin != cout or (pool and pool > 1):
            self.skip = nn.Sequential(
                nn.Conv1d(cin, cout, kernel_size=1, bias=False),
                nn.AvgPool1d(pool) if pool and pool > 1 else nn.Identity(),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.dw(x)
        y = self.pw(y)
        y = self.norm(y)
        y = self.act(y)
        y = self.pool(y)
        y = self.drop(y)
        return y + self.skip(x)


class StatsPool(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu = x.mean(dim=-1)
        std = x.std(dim=-1, unbiased=False).clamp_min(1e-5)
        mx = x.abs().amax(dim=-1)
        return torch.cat([mu, std, mx], dim=1)


def _rms_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    p = (x[:, 0:1, :] ** 2 + x[:, 1:2, :] ** 2).mean(dim=-1, keepdim=True)
    s = torch.sqrt(p + eps)
    return x / s


class StaticBiasStream(nn.Module):
    def __init__(self, out_dim: int = 64, drop: float = 0.1):
        super().__init__()
        self.b1 = DSConvBlock1d(8, 32, k=9, dilation=1, pool=2, drop=drop)
        self.b2 = DSConvBlock1d(32, 48, k=7, dilation=1, pool=2, drop=drop)
        self.b3 = DSConvBlock1d(48, 64, k=5, dilation=1, pool=2, drop=drop)
        self.pool = StatsPool()
        self.proj = nn.Sequential(
            nn.Linear(64 * 3, 96),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(96, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nan_to_num(x.float(), nan=0.0, posinf=0.0, neginf=0.0)
        xn = _rms_norm(x)
        i = xn[:, 0:1, :]
        q = xn[:, 1:2, :]
        mu_i = i.mean(dim=-1, keepdim=True).expand_as(i)
        mu_q = q.mean(dim=-1, keepdim=True).expand_as(q)
        amp = torch.sqrt((i * i + q * q).clamp_min(1e-8))
        image_proxy_i = q
        image_proxy_q = -i
        iq_bal = torch.abs(i.abs().mean(dim=-1, keepdim=True) - q.abs().mean(dim=-1, keepdim=True)).expand_as(i)
        feat = torch.cat([i, q, mu_i, mu_q, amp, image_proxy_i, image_proxy_q, iq_bal], dim=1)
        h = self.b1(feat)
        h = self.b2(h)
        h = self.b3(h)
        return self.proj(self.pool(h))


class PolyphaseInterleaveStream(nn.Module):
    def __init__(self, out_dim: int = 64, drop: float = 0.1):
        super().__init__()
        self.b1 = DSConvBlock1d(8, 32, k=5, dilation=1, pool=2, drop=drop)
        self.b2 = DSConvBlock1d(32, 48, k=5, dilation=2, pool=2, drop=drop)
        self.b3 = DSConvBlock1d(48, 64, k=3, dilation=2, pool=1, drop=drop)
        self.pool = StatsPool()
        self.proj = nn.Sequential(
            nn.Linear(64 * 3, 96),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(96, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nan_to_num(x.float(), nan=0.0, posinf=0.0, neginf=0.0)
        xn = _rms_norm(x)
        x0 = xn[..., 0::2]
        x1 = xn[..., 1::2]
        m = min(x0.size(-1), x1.size(-1))
        x0 = x0[..., :m]
        x1 = x1[..., :m]
        dx = x0 - x1
        sx = x0 + x1
        feat = torch.cat([x0, x1, dx, sx], dim=1)
        h = self.b1(feat)
        h = self.b2(h)
        h = self.b3(h)
        return self.proj(self.pool(h))


class NonlinearSaturationStream(nn.Module):
    def __init__(self, out_dim: int = 64, drop: float = 0.1, gamma: float = 1.8):
        super().__init__()
        self.gamma = float(gamma)
        self.avg = nn.AvgPool1d(5, stride=1, padding=2)
        self.b1 = DSConvBlock1d(8, 32, k=5, dilation=1, pool=2, drop=drop)
        self.b2 = DSConvBlock1d(32, 48, k=5, dilation=1, pool=2, drop=drop)
        self.b3 = DSConvBlock1d(48, 64, k=3, dilation=1, pool=1, drop=drop)
        self.pool = StatsPool()
        self.proj = nn.Sequential(
            nn.Linear(64 * 3, 96),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(96, out_dim),
        )

    @staticmethod
    def _diff1(x: torch.Tensor) -> torch.Tensor:
        return F.pad(x[..., 1:] - x[..., :-1], (1, 0))

    @staticmethod
    def _diff2(x: torch.Tensor) -> torch.Tensor:
        d1 = NonlinearSaturationStream._diff1(x)
        return F.pad(d1[..., 1:] - d1[..., :-1], (1, 0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nan_to_num(x.float(), nan=0.0, posinf=0.0, neginf=0.0)
        xn = _rms_norm(x)
        hf = xn - self.avg(xn)
        d1 = self._diff1(xn)
        d2 = self._diff2(xn)
        sat = torch.tanh(self.gamma * xn) - xn
        feat = torch.cat([hf, d1, d2, sat], dim=1)
        h = self.b1(feat)
        h = self.b2(h)
        h = self.b3(h)
        return self.proj(self.pool(h))


class SpectrumStyleStream(nn.Module):
    def __init__(self, out_dim: int = 64, drop: float = 0.1):
        super().__init__()
        self.b1 = DSConvBlock1d(4, 32, k=7, dilation=1, pool=2, drop=drop)
        self.b2 = DSConvBlock1d(32, 48, k=5, dilation=1, pool=2, drop=drop)
        self.b3 = DSConvBlock1d(48, 64, k=3, dilation=1, pool=1, drop=drop)
        self.pool = StatsPool()
        self.proj = nn.Sequential(
            nn.Linear(64 * 3, 96),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(96, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.cuda.amp.autocast(enabled=False):
            x = torch.nan_to_num(x.float(), nan=0.0, posinf=0.0, neginf=0.0)
            xn = _rms_norm(x)
            z = torch.complex(xn[:, 0, :], xn[:, 1, :])
            X = torch.fft.fft(z, dim=-1)
            mag = torch.log1p(torch.abs(X)).float()
            K = max(8, mag.size(-1) // 2)
            pos = mag[:, :K]
            neg = torch.flip(mag[:, -K:], dims=[1])
            mirror = pos - neg
            edge_bins = max(1, K // 8)
            low = pos[:, :edge_bins].mean(dim=1, keepdim=True)
            high = pos[:, -edge_bins:].mean(dim=1, keepdim=True)
            edge_ratio = (high - low).expand(-1, K)
            noise_floor = pos[:, -max(1, K // 16):].mean(dim=1, keepdim=True).expand(-1, K)
            feat = torch.stack([pos, mirror, edge_ratio, noise_floor], dim=1)
        h = self.b1(feat)
        h = self.b2(h)
        h = self.b3(h)
        return self.proj(self.pool(h))


class ReceiverStatisticsHead(nn.Module):
    def __init__(self, out_dim: int = 64, hidden: int = 64, drop: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(16, hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.cuda.amp.autocast(enabled=False):
            x = torch.nan_to_num(x.float(), nan=0.0, posinf=0.0, neginf=0.0)
            xn = _rms_norm(x)
            i = xn[:, 0, :]
            q = xn[:, 1, :]
            amp = torch.sqrt((i * i + q * q).clamp_min(1e-8))
            even = amp[:, 0::2]
            odd = amp[:, 1::2]
            m = min(even.size(-1), odd.size(-1))
            even = even[:, :m]
            odd = odd[:, :m]
            corr_iq = ((i - i.mean(dim=1, keepdim=True)) * (q - q.mean(dim=1, keepdim=True))).mean(dim=1)
            z = torch.complex(i, q)
            X = torch.fft.fft(z, dim=-1)
            mag = torch.abs(X)
            K = max(8, mag.size(-1) // 2)
            pos = mag[:, :K]
            edge_bins = max(1, K // 8)
            inband = pos[:, edge_bins:-edge_bins].mean(dim=1) if K > 2 * edge_bins else pos.mean(dim=1)
            edge = 0.5 * (pos[:, :edge_bins].mean(dim=1) + pos[:, -edge_bins:].mean(dim=1))
            clip_ratio = (amp > 1.8).float().mean(dim=1)
            feats = torch.stack([
                i.mean(dim=1),
                q.mean(dim=1),
                i.std(dim=1, unbiased=False),
                q.std(dim=1, unbiased=False),
                torch.sqrt((amp * amp).mean(dim=1).clamp_min(1e-8)),
                amp.amax(dim=1) / amp.mean(dim=1).clamp_min(1e-6),
                corr_iq,
                amp.mean(dim=1),
                (amp * amp).mean(dim=1),
                (amp.pow(4)).mean(dim=1),
                (i.abs().mean(dim=1) - q.abs().mean(dim=1)).abs(),
                (even.mean(dim=1) - odd.mean(dim=1)).abs(),
                (even.var(dim=1, unbiased=False) - odd.var(dim=1, unbiased=False)).abs(),
                edge / inband.clamp_min(1e-6),
                clip_ratio,
                (torch.angle(X[:, 1:]) - torch.angle(X[:, :-1])).abs().mean(dim=1),
            ], dim=1)
            feats = torch.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
        return self.mlp(feats)


class RxADCBackbone(nn.Module):
    def __init__(self, out_dim: int = 128, branch_dim: int = 64, drop: float = 0.1):
        super().__init__()
        self.static_stream = StaticBiasStream(out_dim=branch_dim, drop=drop)
        self.poly_stream = PolyphaseInterleaveStream(out_dim=branch_dim, drop=drop)
        self.nl_stream = NonlinearSaturationStream(out_dim=branch_dim, drop=drop)
        self.spec_stream = SpectrumStyleStream(out_dim=branch_dim, drop=drop)
        self.stat_head = ReceiverStatisticsHead(out_dim=branch_dim, drop=drop)
        fusion_dim = 5 * branch_dim
        hid = max(128, out_dim)
        self.proj = nn.Sequential(
            nn.Linear(fusion_dim, hid),
            nn.LayerNorm(hid),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hid, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        es = self.static_stream(x)
        ep = self.poly_stream(x)
        en = self.nl_stream(x)
        ef = self.spec_stream(x)
        est = self.stat_head(x)
        z = torch.cat([es, ep, en, ef, est], dim=1)
        return self.proj(z)


class DualRxADCDisentangle(nn.Module):
    def __init__(
        self,
        num_classes: int,
        num_domains: int,
        model_size: str = "S",
        dataset: str = "wisig",
        input_len: int = 256,
        sample_rate_hz: float = 25e6,
        rx_dim: int = 128,
        rx_branch_dim: int = 64,
        drop: float = 0.1,
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
        self.tx_dim = self._infer_tx_dim(self.id_backbone)
        self.rx_dim = int(rx_dim)
        self.rx_backbone = RxADCBackbone(out_dim=self.rx_dim, branch_dim=rx_branch_dim, drop=drop)

        self.rx_head = MLPHead(self.rx_dim, self.num_domains, hidden=max(64, self.rx_dim), drop=drop)
        self.adv_rx_on_tx = MLPHead(self.tx_dim, self.num_domains, hidden=max(64, self.tx_dim // 2), drop=drop)
        self.adv_tx_on_rx = MLPHead(self.rx_dim, self.num_classes, hidden=max(64, self.rx_dim // 2), drop=drop)
        self.probe_rx_on_tx = MLPHead(self.tx_dim, self.num_domains, hidden=max(32, self.tx_dim // 3), drop=0.05)
        self.probe_tx_on_rx = MLPHead(self.rx_dim, self.num_classes, hidden=max(32, self.rx_dim // 2), drop=0.05)

    @staticmethod
    def _infer_tx_dim(backbone: nn.Module) -> int:
        cls_head = getattr(backbone, "cls_head", None)
        if cls_head is not None:
            head = getattr(cls_head, "head", None)
            weight = getattr(head, "weight", None)
            if weight is not None and hasattr(weight, "shape") and len(weight.shape) == 2:
                return int(weight.shape[1])
        return int(getattr(backbone, "emb_dim", 256))

    @staticmethod
    def _pick(aux: Dict[str, torch.Tensor], keys: Sequence[str]) -> torch.Tensor:
        for key in keys:
            v = aux.get(key, None)
            if torch.is_tensor(v):
                return v
        raise KeyError(f"Cannot find tensor in keys={keys}, available={list(aux.keys())}")

    def forward(
        self,
        x: torch.Tensor,
        y_tx: Optional[torch.Tensor] = None,
        grl_rx_on_tx: float = 0.0,
        grl_tx_on_rx: float = 0.0,
        return_aux: bool = False,
    ):
        aux_id = self.id_backbone(x, y=y_tx, return_aux=True)
        tx_logits = aux_id["logits"]
        z_tx = self._pick(aux_id, ("feat_cls", "z_pa", "feat_con", "base"))
        z_joint = self._pick(aux_id, ("z_id", "feat_joint", "feat_con"))
        z_txc = self._pick(aux_id, ("z_txc", "feat_dac", "feat_imp"))

        z_rx = self.rx_backbone(x)
        rx_logits = self.rx_head(z_rx)
        adv_rx_logits = self.adv_rx_on_tx(grad_reverse(z_tx, grl_rx_on_tx))
        adv_tx_logits = self.adv_tx_on_rx(grad_reverse(z_rx, grl_tx_on_rx))
        probe_rx_logits = self.probe_rx_on_tx(z_tx.detach())
        probe_tx_logits = self.probe_tx_on_rx(z_rx.detach())

        if not return_aux:
            return tx_logits

        return {
            "tx_logits": tx_logits,
            "rx_logits": rx_logits,
            "adv_rx_logits": adv_rx_logits,
            "adv_tx_logits": adv_tx_logits,
            "probe_rx_logits": probe_rx_logits,
            "probe_tx_logits": probe_tx_logits,
            "z_tx": z_tx,
            "z_rx": z_rx,
            "z_joint": z_joint,
            "z_txc": z_txc,
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
) -> DualRxADCDisentangle:
    return DualRxADCDisentangle(
        num_classes=num_classes,
        num_domains=num_domains,
        model_size=model_size,
        dataset=dataset,
        input_len=input_len,
        sample_rate_hz=sample_rate_hz,
        **backbone_kwargs,
    )
