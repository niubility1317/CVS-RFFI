
from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

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
    if x.ndim != 3 or x.size(1) < 2:
        return x
    p = (x[:, 0:1, :] ** 2 + x[:, 1:2, :] ** 2).mean(dim=-1, keepdim=True)
    return x / torch.sqrt(p + eps)


def _autocast_disabled():
    try:
        return torch.amp.autocast(device_type="cuda", enabled=False)
    except Exception:
        return torch.cuda.amp.autocast(enabled=False)


class StaticDACBiasLite(nn.Module):
    """
    Receiver-side static imbalance / image-leakage cues.
    Deliberately avoids strong PA-like saturation features.
    """
    def __init__(self, out_dim: int = 16, drop: float = 0.1):
        super().__init__()
        self.b1 = DSConvBlock1d(8, 24, k=9, dilation=1, pool=2, drop=drop)
        self.b2 = DSConvBlock1d(24, 32, k=7, dilation=1, pool=2, drop=drop)
        self.pool = StatsPool()
        self.proj = nn.Sequential(
            nn.Linear(32 * 3, 48),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(48, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nan_to_num(x.float(), nan=0.0, posinf=0.0, neginf=0.0)
        xn = _rms_norm(x)
        i = xn[:, 0:1, :]
        q = xn[:, 1:2, :]
        mu_i = i.mean(dim=-1, keepdim=True).expand_as(i)
        mu_q = q.mean(dim=-1, keepdim=True).expand_as(q)
        amp = torch.sqrt((i * i + q * q).clamp_min(1e-8))
        iq_bal = torch.abs(i.abs().mean(dim=-1, keepdim=True) - q.abs().mean(dim=-1, keepdim=True)).expand_as(i)
        image_proxy_i = q
        image_proxy_q = -i
        feat = torch.cat([i, q, mu_i, mu_q, amp, iq_bal, image_proxy_i, image_proxy_q], dim=1)
        h = self.b1(feat)
        h = self.b2(h)
        return self.proj(self.pool(h))


class PolyphaseMismatchLite(nn.Module):
    """
    Even/odd interleaving mismatch / sampling skew style cues.
    """
    def __init__(self, out_dim: int = 16, drop: float = 0.1):
        super().__init__()
        self.b1 = DSConvBlock1d(8, 24, k=5, dilation=1, pool=2, drop=drop)
        self.b2 = DSConvBlock1d(24, 32, k=3, dilation=2, pool=2, drop=drop)
        self.pool = StatsPool()
        self.proj = nn.Sequential(
            nn.Linear(32 * 3, 48),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(48, out_dim),
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
        return self.proj(self.pool(h))


class MirrorSpectrumLite(nn.Module):
    """
    Lightweight spectral asymmetry and mirror leakage cues.
    """
    def __init__(self, out_dim: int = 16, drop: float = 0.1):
        super().__init__()
        self.b1 = DSConvBlock1d(4, 24, k=7, dilation=1, pool=2, drop=drop)
        self.b2 = DSConvBlock1d(24, 32, k=5, dilation=1, pool=2, drop=drop)
        self.pool = StatsPool()
        self.proj = nn.Sequential(
            nn.Linear(32 * 3, 48),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(48, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with _autocast_disabled():
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
        return self.proj(self.pool(h))


class SmallStatsHead(nn.Module):
    """
    Small handcrafted statistics head.
    Avoids clip ratio / amp^4 / saturation-heavy cues to reduce PA leakage.
    """
    def __init__(self, out_dim: int = 16, drop: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(12, 32),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(32, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with _autocast_disabled():
            x = torch.nan_to_num(x.float(), nan=0.0, posinf=0.0, neginf=0.0)
            xn = _rms_norm(x)
            i = xn[:, 0, :]
            q = xn[:, 1, :]
            amp = torch.sqrt((i * i + q * q).clamp_min(1e-8))
            corr_iq = (i * q).mean(dim=1) / (i.std(dim=1, unbiased=False).clamp_min(1e-6) * q.std(dim=1, unbiased=False).clamp_min(1e-6))
            even = amp[:, 0::2]
            odd = amp[:, 1::2]
            m = min(even.size(1), odd.size(1))
            even = even[:, :m]
            odd = odd[:, :m]
            z = torch.complex(i, q)
            X = torch.fft.fft(z, dim=-1)
            P = torch.abs(X).float()
            K = P.size(1) // 2
            edge_bins = max(1, K // 8)
            pos = P[:, :K]
            neg = torch.flip(P[:, -K:], dims=[1])
            mirror_energy = (pos - neg).abs().mean(dim=1)
            edge = pos[:, -edge_bins:].mean(dim=1)
            inband = pos[:, edge_bins:max(edge_bins + 1, K - edge_bins)].mean(dim=1)
            feats = torch.stack([
                i.mean(dim=1),
                q.mean(dim=1),
                i.std(dim=1, unbiased=False),
                q.std(dim=1, unbiased=False),
                amp.mean(dim=1),
                amp.std(dim=1, unbiased=False),
                corr_iq,
                (i.abs().mean(dim=1) - q.abs().mean(dim=1)).abs(),
                (even.mean(dim=1) - odd.mean(dim=1)).abs(),
                (even.var(dim=1, unbiased=False) - odd.var(dim=1, unbiased=False)).abs(),
                edge / inband.clamp_min(1e-6),
                mirror_energy,
            ], dim=1)
            feats = torch.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
        return self.mlp(feats)


class RxDACLiteBackbone(nn.Module):
    """
    Lightweight receiver-DAC overlap extractor.
    Only models low-dimensional receiver-side DAC-ish cues.
    """
    def __init__(self, out_dim: int = 64, branch_dim: int = 16, drop: float = 0.1):
        super().__init__()
        self.static_stream = StaticDACBiasLite(out_dim=branch_dim, drop=drop)
        self.poly_stream = PolyphaseMismatchLite(out_dim=branch_dim, drop=drop)
        self.spec_stream = MirrorSpectrumLite(out_dim=branch_dim, drop=drop)
        self.stats_head = SmallStatsHead(out_dim=branch_dim, drop=drop)
        fusion_dim = 4 * branch_dim
        hid = max(64, out_dim)
        self.proj = nn.Sequential(
            nn.Linear(fusion_dim, hid),
            nn.LayerNorm(hid),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hid, out_dim),
        )
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        es = self.static_stream(x)
        ep = self.poly_stream(x)
        ef = self.spec_stream(x)
        est = self.stats_head(x)
        z = torch.cat([es, ep, ef, est], dim=1)
        z = self.proj(z)
        return self.norm(z)


class DACOverlapExtractor(nn.Module):
    """
    Extract only the TX-DAC / RX-ADC common subspace, while first repelling the PA direction.
    """
    def __init__(self, tx_dim: int, rx_dim: int, drop: float = 0.1):
        super().__init__()
        self.tx_dim = int(tx_dim)
        self.rx_dim = int(rx_dim)
        self.rx_up = nn.Sequential(
            nn.Linear(rx_dim, tx_dim),
            nn.LayerNorm(tx_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(tx_dim, tx_dim),
        )
        self.pa_to_tx = nn.Sequential(
            nn.Linear(tx_dim, tx_dim),
            nn.LayerNorm(tx_dim),
        )
        self.gate = nn.Sequential(
            nn.Linear(4 * tx_dim, max(128, tx_dim)),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(max(128, tx_dim), tx_dim),
            nn.Sigmoid(),
        )
        self.overlap_norm = nn.LayerNorm(tx_dim)
        self.clean_norm = nn.LayerNorm(tx_dim)

    def forward(self, z_txc: torch.Tensor, r_dac: torch.Tensor, feat_pa: torch.Tensor, beta: float = 0.16) -> Dict[str, torch.Tensor]:
        r_up = self.rx_up(r_dac.float())
        pa_ref = self.pa_to_tx(feat_pa.float().detach())
        pa_dir = F.normalize(pa_ref, dim=1, eps=1e-4)
        coeff_pa = torch.sum(r_up * pa_dir, dim=1, keepdim=True)
        r_overlap = self.overlap_norm(r_up - coeff_pa * pa_dir)

        t_gate = F.normalize(z_txc.float().detach(), dim=1, eps=1e-4)
        r_gate = F.normalize(r_overlap.float(), dim=1, eps=1e-4)
        sim = torch.sum(t_gate * r_gate, dim=1, keepdim=True)
        pair = torch.cat([t_gate, r_gate, t_gate * r_gate, torch.abs(t_gate - r_gate)], dim=1)
        gate = self.gate(pair).float()
        common = gate * F.relu(sim) * r_gate
        clean = self.clean_norm(z_txc.float() - float(beta) * common)
        return {
            "r_up": r_up,
            "r_overlap": r_overlap,
            "pa_ref": pa_ref,
            "common": common,
            "clean": clean,
            "sim": sim.squeeze(1),
            "gate": gate,
            "coeff_pa": coeff_pa.squeeze(1),
        }


class PADominantFusionHead(nn.Module):
    """
    Keep classification PA-dominant.
    Main classification depends on PA + small ID residual, while cleaned DAC only makes a tiny correction.
    """
    def __init__(self, emb_dim: int, alpha_id: float = 0.25, default_eta_dac: float = 0.08):
        super().__init__()
        self.emb_dim = int(emb_dim)
        self.alpha_id = float(alpha_id)
        self.default_eta_dac = float(default_eta_dac)
        self.main_norm = nn.LayerNorm(emb_dim)
        self.cls_norm = nn.LayerNorm(emb_dim)

    def forward(self, feat_pa: torch.Tensor, feat_id: torch.Tensor, z_dac_clean: torch.Tensor, eta_dac: Optional[float] = None) -> Dict[str, torch.Tensor]:
        eta = self.default_eta_dac if eta_dac is None else float(eta_dac)
        z_main = self.main_norm(feat_pa.float() + self.alpha_id * feat_id.float())
        z_cls = self.cls_norm(z_main + eta * z_dac_clean.float())
        return {
            "z_main": z_main,
            "z_cls": z_cls,
            "eta_dac": eta,
        }


class DualRxDACOverlapPAAware(nn.Module):
    def __init__(
        self,
        num_classes: int,
        num_domains: int,
        model_size: str = "S",
        dataset: str = "wisig",
        input_len: int = 256,
        sample_rate_hz: float = 25e6,
        rx_dim: int = 64,
        rx_branch_dim: int = 16,
        overlap_beta: float = 0.20,
        alpha_id: float = 0.25,
        default_eta_dac: float = 0.08,
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
        self.overlap_beta = float(overlap_beta)

        self.rx_backbone = RxDACLiteBackbone(out_dim=self.rx_dim, branch_dim=rx_branch_dim, drop=drop)
        self.pa_to_rx = nn.Sequential(
            nn.Linear(self.tx_dim, self.rx_dim),
            nn.LayerNorm(self.rx_dim),
        )
        self.rx_priv_norm = nn.LayerNorm(self.rx_dim)
        self.overlap = DACOverlapExtractor(self.tx_dim, self.rx_dim, drop=drop)
        self.pa_fusion = PADominantFusionHead(self.tx_dim, alpha_id=alpha_id, default_eta_dac=default_eta_dac)

        self.rx_head = MLPHead(self.rx_dim, self.num_domains, hidden=max(64, self.rx_dim), drop=drop)
        self.adv_rx_on_tx = MLPHead(self.tx_dim, self.num_domains, hidden=max(64, self.tx_dim // 2), drop=drop)
        self.adv_tx_on_rx = MLPHead(self.rx_dim, self.num_classes, hidden=max(64, self.rx_dim), drop=drop)
        self.probe_rx_on_tx = MLPHead(self.tx_dim, self.num_domains, hidden=max(32, self.tx_dim // 3), drop=0.05)
        self.probe_tx_on_rx = MLPHead(self.rx_dim, self.num_classes, hidden=max(32, self.rx_dim), drop=0.05)

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
        dac_resid_scale: Optional[float] = None,
        overlap_beta: Optional[float] = None,
        return_aux: bool = False,
    ):
        aux_id = self.id_backbone(x, y=y_tx, return_aux=True)
        raw_tx_logits = aux_id["logits"]

        feat_id = self._pick(aux_id, ("feat_cls", "z_pa", "feat_con", "base"))
        feat_pa = self._pick(aux_id, ("feat_pa", "z_pa", "feat_joint"))
        z_txc = self._pick(aux_id, ("z_txc", "feat_dac", "feat_imp"))
        z_joint = self._pick(aux_id, ("z_id", "feat_joint", "feat_con"))

        r_raw = self.rx_backbone(x)
        pa_rx = self.pa_to_rx(feat_pa.detach().float())
        pa_rx_dir = F.normalize(pa_rx, dim=1, eps=1e-4)
        coeff_rx_pa = torch.sum(r_raw.float() * pa_rx_dir, dim=1, keepdim=True)
        r_priv = self.rx_priv_norm(r_raw.float() - coeff_rx_pa * pa_rx_dir)
        rx_logits = self.rx_head(r_priv)

        ov = self.overlap(z_txc, r_priv, feat_pa, beta=self.overlap_beta if overlap_beta is None else overlap_beta)
        fused = self.pa_fusion(feat_pa, feat_id, ov["clean"], eta_dac=dac_resid_scale)
        tx_head = self.id_backbone.cls_head.head
        tx_logits = tx_head(fused["z_cls"], labels=y_tx)

        adv_rx_logits = self.adv_rx_on_tx(grad_reverse(fused["z_main"], grl_rx_on_tx))
        adv_tx_logits = self.adv_tx_on_rx(grad_reverse(r_priv, grl_tx_on_rx))
        probe_rx_logits = self.probe_rx_on_tx(fused["z_main"].detach())
        probe_tx_logits = self.probe_tx_on_rx(r_priv.detach())

        if not return_aux:
            return tx_logits

        return {
            "tx_logits": tx_logits,
            "raw_tx_logits": raw_tx_logits,
            "rx_logits": rx_logits,
            "adv_rx_logits": adv_rx_logits,
            "adv_tx_logits": adv_tx_logits,
            "probe_rx_logits": probe_rx_logits,
            "probe_tx_logits": probe_tx_logits,
            "z_tx": fused["z_main"],
            "z_rx": r_priv,
            "z_joint": z_joint,
            "z_txc": z_txc,
            "z_txc_clean": ov["clean"],
            "common_dac": ov["common"],
            "r_dac_up": ov["r_up"],
            "r_overlap": ov["r_overlap"],
            "pa_ref_tx": ov["pa_ref"],
            "tx_rx_sim": ov["sim"],
            "gate_dac": ov["gate"],
            "r_raw": r_raw,
            "pa_rx_ref": pa_rx,
            "feat_pa": feat_pa,
            "feat_id": feat_id,
            "z_main": fused["z_main"],
            "z_cls": fused["z_cls"],
            "eta_dac": fused["eta_dac"],
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
) -> DualRxDACOverlapPAAware:
    return DualRxDACOverlapPAAware(
        num_classes=num_classes,
        num_domains=num_domains,
        model_size=model_size,
        dataset=dataset,
        input_len=input_len,
        sample_rate_hz=sample_rate_hz,
        **backbone_kwargs,
    )
