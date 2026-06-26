import math
import numpy as np
from typing import Optional, Tuple, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------- input length adapter -----------------------
def pad_crop_iq(
    x: torch.Tensor,
    target_len: int,
    mode: str = "center",
) -> torch.Tensor:
    """Pad/crop IQ tensor to target length.

    x: (B,2,L)
    target_len: desired L
    mode: 'center' or 'left'
    """
    if target_len is None or target_len <= 0:
        return x
    if x.dim() != 3 or x.size(1) != 2:
        return x
    L = int(x.size(-1))
    T = int(target_len)
    if L == T:
        return x
    if L > T:
        if mode == "left":
            return x[..., :T]
        start = (L - T) // 2
        return x[..., start : start + T]
    pad = T - L
    if mode == "left":
        out = x.new_zeros((x.size(0), 2, T))
        out[..., :L] = x
        return out
    left = pad // 2
    out = x.new_zeros((x.size(0), 2, T))
    out[..., left : left + L] = x
    return out


# ----------------------- utils -----------------------
def _hz2mel(hz: float) -> float:
    return 2595.0 * math.log10(1.0 + hz / 700.0)


def _mel2hz(mel: float) -> float:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def _pick_gn_groups(ch: int) -> int:
    for g in (16, 8, 4, 2, 1):
        if ch % g == 0:
            return g
    return 1


def _safe_div(num: torch.Tensor, den: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    den2 = torch.where(den.abs() < eps, torch.ones_like(den), den)
    return num / den2


def _parse_stability_mode(name: str, value: str, valid: Sequence[str]) -> str:
    mode = str(value or "off").lower().strip()
    if mode not in set(valid):
        raise ValueError(f"{name} must be one of {tuple(valid)}, got {value!r}")
    return mode


def _parse_odd_orders(value: Optional[Sequence[int] | str]) -> Optional[Tuple[int, ...]]:
    if value is None:
        return None
    if isinstance(value, str):
        raw = value.strip()
        if raw == "":
            return None
        items = [int(x.strip()) for x in raw.replace(";", ",").split(",") if x.strip()]
    else:
        items = [int(x) for x in value]
    orders = tuple(items)
    if not orders or any((p % 2) == 0 or p < 1 for p in orders):
        raise ValueError("pa_orders must be non-empty positive odd integers")
    return orders


def _trim_channels(value: int, scale: float, minimum: int = 1) -> int:
    return int(max(int(minimum), round(int(value) * float(scale))))


class PhaseDeltaStabilityStem(nn.Module):
    """Compact complex-IQ stability cues after the shared Sinc filterbank."""

    def __init__(self, sinc_out: int, out_channels: int = 8, eps: float = 1e-6):
        super().__init__()
        self.sinc_out = int(sinc_out)
        self.eps = float(eps)
        out_channels = int(max(1, out_channels))
        self.proj = nn.Sequential(
            nn.Conv1d(4 * self.sinc_out, out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(_pick_gn_groups(out_channels), out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, sinc_iq: torch.Tensor) -> torch.Tensor:
        if sinc_iq.dim() != 3 or int(sinc_iq.size(1)) != 2 * self.sinc_out:
            raise ValueError("PhaseDeltaStabilityStem expects [B, 2*sinc_out, T] filterbank IQ")
        B, _, L = sinc_iq.shape
        y = torch.nan_to_num(sinc_iq.float(), nan=0.0, posinf=0.0, neginf=0.0).view(B, 2, self.sinc_out, L)
        i = y[:, 0, :, :]
        q = y[:, 1, :, :]
        amp = torch.sqrt(i * i + q * q + self.eps)
        amp_norm = torch.log1p(amp / amp.mean(dim=-1, keepdim=True).clamp_min(self.eps))
        if L > 1:
            d_amp = F.pad(amp[..., 1:] - amp[..., :-1], (1, 0))
            cross = q[..., 1:] * i[..., :-1] - i[..., 1:] * q[..., :-1]
            dot = i[..., 1:] * i[..., :-1] + q[..., 1:] * q[..., :-1]
            d_phi = F.pad(torch.atan2(cross, dot + self.eps), (1, 0))
        else:
            d_amp = torch.zeros_like(amp)
            d_phi = torch.zeros_like(amp)
        raw = torch.cat(
            [
                amp_norm.clamp(0.0, 8.0),
                d_amp.clamp(-8.0, 8.0),
                torch.sin(d_phi),
                torch.cos(d_phi) - 1.0,
            ],
            dim=1,
        )
        return self.proj(raw)


class DSQFreqStabilityStem(nn.Module):
    """Differential spectral-quotient residual cues for the mirrored FFT branch."""

    def __init__(self, out_channels: int = 4):
        super().__init__()
        out_channels = int(max(1, out_channels))
        self.proj = nn.Sequential(
            nn.Conv1d(4, out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(_pick_gn_groups(out_channels), out_channels),
            nn.SiLU(inplace=True),
        )

    @staticmethod
    def _smooth(x: torch.Tensor) -> torch.Tensor:
        return F.avg_pool1d(x, kernel_size=5, stride=1, padding=2, count_include_pad=False)

    def forward(self, feat_f: torch.Tensor) -> torch.Tensor:
        if feat_f.dim() != 3 or int(feat_f.size(1)) < 4:
            raise ValueError("DSQFreqStabilityStem expects mirrored frequency features [B, >=4, K]")
        log_pos = feat_f[:, 0:1, :]
        log_neg = feat_f[:, 1:2, :]
        log_ratio = feat_f[:, 2:3, :]
        asym = feat_f[:, 3:4, :]
        raw = torch.cat(
            [
                log_pos - self._smooth(log_pos),
                log_neg - self._smooth(log_neg),
                log_ratio - self._smooth(log_ratio),
                asym - self._smooth(asym),
            ],
            dim=1,
        ).clamp(-8.0, 8.0)
        return self.proj(torch.nan_to_num(raw, nan=0.0, posinf=8.0, neginf=-8.0))


# ----------------------- SincConv -----------------------
class SincConv1d(nn.Module):
    """
    Hz-parameterized SincConv.
    Input:  (B,1,L)
    Output: (B,C,L)
    """
    def __init__(
        self,
        out_channels: int,
        kernel_size: int,
        sample_rate: float = 5e6,
        min_low_hz: float = 50.0,
        min_band_hz: float = 50.0,
        dataset: str = "unknown",
        input_len: Optional[int] = None,
        pad_crop_mode: str = "center",
    ):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd for symmetric sinc filters.")

        self.out_channels = int(out_channels)
        self.kernel_size = int(kernel_size)
        self.sample_rate = float(sample_rate)
        self.dataset = str(dataset)
        self.input_len = int(input_len) if (input_len is not None and int(input_len) > 0) else None
        self.pad_crop_mode = str(pad_crop_mode)
        self.min_low_hz = float(min_low_hz)
        self.min_band_hz = float(min_band_hz)

        low_hz = 30.0
        nyq = self.sample_rate / 2.0
        high_hz = nyq - (low_hz + self.min_band_hz + 1.0)
        if high_hz <= low_hz:
            raise ValueError("sample_rate too low or min_band_hz too large.")

        # Torch-only initialization avoids NumPy scalar/dtype inference problems
        # seen with some PyTorch + NumPy combinations, e.g.:
        #   RuntimeError: Could not infer dtype of numpy.float32
        mel_low = float(_hz2mel(low_hz))
        mel_high = float(_hz2mel(high_hz))
        mel_points = torch.linspace(
            mel_low,
            mel_high,
            steps=self.out_channels + 1,
            dtype=torch.float32,
        )
        hz_points = 700.0 * (10.0 ** (mel_points / 2595.0) - 1.0)

        low_init = hz_points[:-1].clone().detach().contiguous()
        band_init = torch.diff(hz_points).clone().detach().contiguous()

        self.low_hz_ = nn.Parameter(low_init.view(-1, 1))
        self.band_hz_ = nn.Parameter(band_init.view(-1, 1))

        n = torch.arange(self.kernel_size, dtype=torch.float32) - (self.kernel_size - 1) / 2.0
        t = n / self.sample_rate
        window = 0.54 - 0.46 * torch.cos(
            2 * math.pi * (torch.arange(self.kernel_size) / (self.kernel_size - 1))
        )
        self.register_buffer("t_", t.view(1, -1))
        self.register_buffer("window_", window.view(1, -1))
        self._filter_cache_key = None
        self._filter_cache = None

    def _filters(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        use_cache = (not self.training) and (not torch.is_grad_enabled())
        cache_key = (
            device.type,
            device.index,
            dtype,
            int(getattr(self.low_hz_, "_version", 0)),
            int(getattr(self.band_hz_, "_version", 0)),
        )
        if use_cache and self._filter_cache_key == cache_key and self._filter_cache is not None:
            return self._filter_cache

        t = self.t_.to(device=device, dtype=dtype)
        window = self.window_.to(device=device, dtype=dtype)

        nyq = self.sample_rate / 2.0
        low = self.min_low_hz + torch.abs(self.low_hz_)
        band = self.min_band_hz + torch.abs(self.band_hz_)

        low = torch.clamp(low, min=self.min_low_hz, max=nyq - self.min_band_hz - 1.0)
        min_high = low + self.min_band_hz
        max_high = torch.full_like(low, nyq - 1.0)
        high = torch.clamp(low + band, min=min_high, max=max_high)

        f1 = low.to(device=device, dtype=dtype)
        f2 = high.to(device=device, dtype=dtype)

        num = torch.sin(2.0 * math.pi * f2 * t) - torch.sin(2.0 * math.pi * f1 * t)
        den = math.pi * t
        bp = _safe_div(num, den, eps=1e-12)

        center = self.kernel_size // 2
        bp[:, center] = (2.0 * (f2 - f1)).squeeze(1)
        bp = bp * window
        bp = bp / (bp.abs().amax(dim=1, keepdim=True) + 1e-8)
        filters = bp.view(self.out_channels, 1, self.kernel_size)
        if use_cache:
            self._filter_cache_key = cache_key
            self._filter_cache = filters
        return filters

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        filters = self._filters(device=x.device, dtype=x.dtype)
        return F.conv1d(x, filters, stride=1, padding=self.kernel_size // 2, bias=None)

    def forward_iq_pair(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3 or int(x.size(1)) != 2:
            raise ValueError("forward_iq_pair expects IQ input shaped [B, 2, L]")
        B = int(x.size(0))
        filters = self._filters(device=x.device, dtype=x.dtype)
        y = F.conv1d(x.reshape(B * 2, 1, x.size(-1)), filters, stride=1, padding=self.kernel_size // 2, bias=None)
        return y.reshape(B, 2, self.out_channels, y.size(-1)).reshape(B, 2 * self.out_channels, y.size(-1))


# ----------------------- HighFreqEmphasis -----------------------
class HighFreqEmphasis(nn.Module):
    """
    Fixed 1st and 2nd difference operator on IQ to highlight DAC-induced ripples/steps.
    Input:  (B,2,L)
    Output: (B,4,L)
    """
    def __init__(self):
        super().__init__()
        k1 = torch.tensor([[-1.0, 1.0]], dtype=torch.float32).view(1, 1, 2)
        k2 = torch.tensor([[1.0, -2.0, 1.0]], dtype=torch.float32).view(1, 1, 3)
        self.register_buffer("k1", k1)
        self.register_buffer("k2", k2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        L = x.size(-1)
        w1 = self.k1.repeat(2, 1, 1)
        w2 = self.k2.repeat(2, 1, 1)
        d1 = F.conv1d(x, w1, padding=1, groups=2)[..., :L]
        d2 = F.conv1d(x, w2, padding=1, groups=2)
        return torch.cat([d1, d2], dim=1)


# ----------------------- Conv blocks -----------------------
class DSConvBlock1d(nn.Module):
    def __init__(self, cin: int, cout: int, k: int = 5, pool: int = 2, drop: float = 0.1):
        super().__init__()
        pad = k // 2
        gn = _pick_gn_groups(cout)
        self.dw = nn.Conv1d(cin, cin, kernel_size=k, padding=pad, groups=cin, bias=False)
        self.pw = nn.Conv1d(cin, cout, kernel_size=1, bias=False)
        self.norm = nn.GroupNorm(gn, cout)
        self.act = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool1d(pool) if (pool is not None and pool > 1) else nn.Identity()
        self.drop = nn.Dropout(drop) if (drop and drop > 0) else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dw(x)
        x = self.pw(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.pool(x)
        x = self.drop(x)
        return x


class DilatedConvBlock1d(nn.Module):
    def __init__(self, cin: int, cout: int, k: int = 5, dilation: int = 1, pool: int = 1, drop: float = 0.1):
        super().__init__()
        pad = (k // 2) * dilation
        gn = _pick_gn_groups(cout)
        self.conv = nn.Conv1d(cin, cout, kernel_size=k, padding=pad, dilation=dilation, bias=False)
        self.norm = nn.GroupNorm(gn, cout)
        self.act = nn.SiLU(inplace=True)
        self.pool = nn.AvgPool1d(pool) if (pool is not None and pool > 1) else nn.Identity()
        self.drop = nn.Dropout(drop) if (drop and drop > 0) else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.pool(x)
        x = self.drop(x)
        return x


# ----------------------- Freq helpers -----------------------
class FreqBandGate1d(nn.Module):
    def __init__(self, in_ch: int = 4, k: int = 5, alpha: float = 0.6):
        super().__init__()
        if k % 2 == 0:
            k += 1
        self.alpha = float(alpha)
        self.conv = nn.Conv1d(in_ch, 1, kernel_size=k, padding=k // 2, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.alpha <= 0:
            return x
        g = torch.sigmoid(self.conv(x))
        scale = 1.0 + self.alpha * (2.0 * g - 1.0)
        return x * scale


class SubbandGatedAggregator(nn.Module):
    def __init__(self, in_ch: int = 4, emb_dim: int = 128, hidden: int = 64, temperature: float = 1.0):
        super().__init__()
        self.in_ch = int(in_ch)
        self.emb_dim = int(emb_dim)
        self.temperature = float(temperature)
        self.band_mlp = nn.Sequential(
            nn.Linear(self.in_ch * 2, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )
        self.proj = nn.Sequential(
            nn.Linear(self.in_ch, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, self.emb_dim),
        )

    def forward(self, feat_f: torch.Tensor) -> torch.Tensor:
        B, C, K = feat_f.shape
        x = feat_f.permute(0, 2, 1).contiguous()
        g = feat_f.mean(dim=2)
        g2 = g.unsqueeze(1).expand(-1, K, -1)
        logits = self.band_mlp(torch.cat([x, g2], dim=-1)).squeeze(-1)
        w = torch.softmax(logits / max(1e-6, self.temperature), dim=1)
        agg = torch.sum(x * w.unsqueeze(-1), dim=1)
        return self.proj(agg)


class MixStyle1D(nn.Module):
    """MixStyle for 1D feature maps shaped [B, C, T].

    The instance statistics are computed along the temporal axis only. The
    statistics are stop-gradient as in MixStyle, and the module is completely
    inactive in eval mode.
    """
    def __init__(
        self,
        p: float = 0.3,
        alpha: float = 0.1,
        eps: float = 1e-6,
        mix: str = "crossdomain",
        strength: float = 1.0,
        fallback: str = "random",
    ):
        super().__init__()
        self.p = float(p)
        self.alpha = float(alpha)
        self.eps = float(eps)
        self.mix = str(mix).lower().strip()
        self.strength = float(strength)
        self.fallback = str(fallback).lower().strip()
        if self.mix not in ("random", "crossdomain", "same_tx", "same_tx_crossdomain"):
            raise ValueError("MixStyle1D mix must be one of: random,crossdomain,same_tx,same_tx_crossdomain")
        if self.fallback not in ("random", "skip"):
            raise ValueError("MixStyle1D fallback must be 'random' or 'skip'")

    @staticmethod
    def _label_vector(labels: Optional[torch.Tensor], batch_size: int, device) -> Optional[torch.Tensor]:
        if labels is None:
            return None
        labels = labels.to(device=device).view(-1)
        return labels if labels.numel() == batch_size else None

    def _random_perm(self, batch_size: int, device) -> torch.Tensor:
        if batch_size <= 1:
            return torch.arange(batch_size, device=device)
        return torch.randperm(batch_size, device=device)

    def _paired_perm(
        self,
        tx_labels: Optional[torch.Tensor],
        domain_labels: Optional[torch.Tensor],
        batch_size: int,
        device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        idx_all = torch.arange(batch_size, device=device)
        if batch_size <= 1:
            return idx_all, torch.zeros(batch_size, device=device, dtype=torch.bool)

        tx = self._label_vector(tx_labels, batch_size, device)
        d = self._label_vector(domain_labels, batch_size, device)
        if self.mix == "random":
            perm = self._random_perm(batch_size, device)
            return perm, perm != idx_all

        need_same_tx = self.mix in ("same_tx", "same_tx_crossdomain")
        need_diff_domain = self.mix in ("crossdomain", "same_tx_crossdomain")
        if need_same_tx and tx is None:
            perm = self._random_perm(batch_size, device)
            return (perm, perm != idx_all) if self.fallback == "random" else (idx_all, torch.zeros_like(idx_all, dtype=torch.bool))
        if need_diff_domain and (d is None or torch.unique(d).numel() <= 1):
            perm = self._random_perm(batch_size, device)
            return (perm, perm != idx_all) if self.fallback == "random" else (idx_all, torch.zeros_like(idx_all, dtype=torch.bool))

        perm = idx_all.clone()
        valid = torch.zeros(batch_size, device=device, dtype=torch.bool)
        fallback = self._random_perm(batch_size, device)
        for i in range(batch_size):
            mask = idx_all != i
            if need_same_tx:
                mask = mask & (tx == tx[i])
            if need_diff_domain:
                mask = mask & (d != d[i])
            candidates = idx_all[mask]
            if candidates.numel() > 0:
                j = torch.randint(candidates.numel(), (1,), device=device)
                perm[i] = candidates[j]
                valid[i] = True
            elif self.fallback == "random":
                perm[i] = fallback[i]
                valid[i] = perm[i] != i
        return perm, valid

    def _sample_lambda(self, batch_size: int, device, dtype: torch.dtype) -> torch.Tensor:
        a = torch.full((batch_size, 1, 1), self.alpha, device=device, dtype=torch.float32)
        beta = torch.distributions.Beta(a, a)
        lam = beta.sample().to(device=device, dtype=dtype)
        return lam.clamp(0.0, 1.0)

    def forward(
        self,
        x: torch.Tensor,
        domain_labels: Optional[torch.Tensor] = None,
        tx_labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if (not self.training) or self.p <= 0.0 or self.alpha <= 0.0 or self.strength <= 0.0:
            return x
        if x.dim() != 3 or x.size(0) <= 1:
            return x
        if torch.rand((), device=x.device) > self.p:
            return x

        mu = x.mean(dim=2, keepdim=True).detach()
        var = (x - mu).pow(2).mean(dim=2, keepdim=True)
        sigma = torch.sqrt(var + self.eps).detach()
        x_norm = (x - mu) / sigma.clamp_min(self.eps)

        perm, valid = self._paired_perm(tx_labels, domain_labels, x.size(0), x.device)
        if not bool(valid.any()):
            return x

        lam = self._sample_lambda(x.size(0), x.device, x.dtype)
        mu_mix = lam * mu + (1.0 - lam) * mu[perm]
        sigma_mix = lam * sigma + (1.0 - lam) * sigma[perm]
        mixed = x_norm * sigma_mix + mu_mix
        mixed = x + self.strength * (mixed - x)
        return torch.where(valid.view(-1, 1, 1), mixed, x)


# ----------------------- widely-linear complex conv -----------------------
class WLComplexConv1d(nn.Module):
    """
    Widely-linear complex convolution.

    Input / output layout:
      x: (B, 2*Cin, L)  -> [real(0:Cin), imag(Cin:2Cin)]
      y: (B, 2*Cout, L) -> [real, imag]

    Implements: y = W*x + V*x*
    where W and V are complex-valued kernels.
    """
    def __init__(self, cin_complex: int, cout_complex: int, k: int = 5, dilation: int = 1, bias: bool = False):
        super().__init__()
        pad = (k // 2) * dilation
        self.cin_complex = int(cin_complex)
        self.cout_complex = int(cout_complex)
        self.wr = nn.Conv1d(cin_complex, cout_complex, kernel_size=k, padding=pad, dilation=dilation, bias=bias)
        self.wi = nn.Conv1d(cin_complex, cout_complex, kernel_size=k, padding=pad, dilation=dilation, bias=bias)
        self.vr = nn.Conv1d(cin_complex, cout_complex, kernel_size=k, padding=pad, dilation=dilation, bias=bias)
        self.vi = nn.Conv1d(cin_complex, cout_complex, kernel_size=k, padding=pad, dilation=dilation, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xr = x[:, : self.cin_complex, :]
        xi = x[:, self.cin_complex :, :]

        yr = self.wr(xr) - self.wi(xi) + self.vr(xr) + self.vi(xi)
        yi = self.wr(xi) + self.wi(xr) - self.vr(xi) + self.vi(xr)
        return torch.cat([yr, yi], dim=1)


class WLComplexBlock1d(nn.Module):
    def __init__(
        self,
        cin_complex: int,
        cout_complex: int,
        k: int = 5,
        dilation: int = 1,
        pool: int = 1,
        drop: float = 0.1,
        residual: bool = True,
    ):
        super().__init__()
        self.conv = WLComplexConv1d(cin_complex, cout_complex, k=k, dilation=dilation, bias=False)
        gn = _pick_gn_groups(2 * cout_complex)
        self.norm = nn.GroupNorm(gn, 2 * cout_complex)
        self.act = nn.SiLU(inplace=True)
        self.pool = nn.AvgPool1d(pool) if (pool is not None and pool > 1) else nn.Identity()
        self.drop = nn.Dropout(drop) if (drop and drop > 0) else nn.Identity()
        self.use_res = bool(residual)
        if self.use_res and (cin_complex != cout_complex or (pool is not None and pool > 1)):
            self.res_proj = nn.Sequential(
                nn.Conv1d(2 * cin_complex, 2 * cout_complex, kernel_size=1, bias=False),
                nn.AvgPool1d(pool) if (pool is not None and pool > 1) else nn.Identity(),
            )
        else:
            self.res_proj = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x)
        y = self.norm(y)
        y = self.act(y)
        y = self.pool(y)
        y = self.drop(y)
        if self.use_res:
            y = y + self.res_proj(x)
        return y


# ----------------------- PA memory polynomial lift -----------------------
class MemoryPolynomialLift(nn.Module):
    """
    Generates delayed polynomial basis:
      x[n-m] * |x[n-m]|^(p-1), p in odd orders.

    Input : (B,2,L)
    Output: (B, 2 * memory_depth * len(orders), L)
    """
    def __init__(self, memory_depth: int = 4, orders: Sequence[int] = (1, 3, 5), clip: float = 2.0):
        super().__init__()
        self.memory_depth = int(memory_depth)
        self.orders = tuple(int(p) for p in orders)
        if any((p % 2) == 0 or p < 1 for p in self.orders):
            raise ValueError("orders must be positive odd integers")
        self.clip = float(clip)

    def _delay(self, v: torch.Tensor, m: int) -> torch.Tensor:
        if m <= 0:
            return v
        pad = v.new_zeros(v.size(0), v.size(1), m)
        return torch.cat([pad, v[..., :-m]], dim=-1)

    def _forward_loop(self, xr: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
        outs = []
        for m in range(self.memory_depth):
            ar = self._delay(xr, m)
            ai = self._delay(xi, m)
            mag2 = ar * ar + ai * ai
            mag2_safe = torch.clamp(mag2, min=1e-8)
            for p in self.orders:
                if p == 1:
                    scale = torch.ones_like(mag2)
                else:
                    scale = mag2_safe
                    for _ in range(1, (p - 1) // 2):
                        scale = scale * mag2_safe
                outs.append(ar * scale)
                outs.append(ai * scale)
        return torch.cat(outs, dim=1)

    def _forward_vectorized(self, xr: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
        B, _, L = xr.shape
        M = max(1, self.memory_depth)
        if M == 1:
            ar = xr
            ai = xi
        else:
            ar = F.pad(xr, (M - 1, 0)).unfold(-1, M, 1).flip(-1).squeeze(1).permute(0, 2, 1)
            ai = F.pad(xi, (M - 1, 0)).unfold(-1, M, 1).flip(-1).squeeze(1).permute(0, 2, 1)
        mag2 = ar * ar + ai * ai
        mag2_safe = torch.clamp(mag2, min=1e-8)
        scales = []
        for p in self.orders:
            if p == 1:
                scales.append(torch.ones_like(mag2))
            else:
                scale = mag2_safe
                for _ in range(1, (p - 1) // 2):
                    scale = scale * mag2_safe
                scales.append(scale)
        scale_stack = torch.stack(scales, dim=2)
        basis_r = ar.unsqueeze(2) * scale_stack
        basis_i = ai.unsqueeze(2) * scale_stack
        return torch.stack([basis_r, basis_i], dim=3).reshape(B, M * len(self.orders) * 2, L)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xr = torch.clamp(x[:, 0:1, :], -self.clip, self.clip)
        xi = torch.clamp(x[:, 1:2, :], -self.clip, self.clip)
        if int(x.size(0)) <= 1:
            return self._forward_loop(xr, xi)
        return self._forward_vectorized(xr, xi)


class EnvelopeGate1d(nn.Module):
    def __init__(self, out_ch: int, alpha: float = 0.5, k: int = 5):
        super().__init__()
        if k % 2 == 0:
            k += 1
        self.alpha = float(alpha)
        self.net = nn.Conv1d(1, out_ch, kernel_size=k, padding=k // 2, bias=True)

    def forward(self, feat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        amp = torch.sqrt(torch.clamp(x[:, 0:1, :] ** 2 + x[:, 1:2, :] ** 2, min=1e-8))
        g = torch.sigmoid(self.net(amp))
        scale = 1.0 + self.alpha * (2.0 * g - 1.0)
        return feat * scale


# ----------------------- heads -----------------------
class CosFaceHead(nn.Module):
    def __init__(self, in_features: int, out_features: int, s: float = 30.0, m: float = 0.35):
        super().__init__()
        self.s = float(s)
        self.m = float(m)
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.01)
        self._norm_weight_cache_key = None
        self._norm_weight_cache = None

    def forward(self, x: torch.Tensor, labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        with torch.cuda.amp.autocast(enabled=False):
            x_f = F.normalize(x.float(), dim=1, eps=1e-4)
            use_cache = labels is None and (not self.training) and (not torch.is_grad_enabled())
            cache_key = (
                self.weight.device.type,
                self.weight.device.index,
                self.weight.dtype,
                int(getattr(self.weight, "_version", 0)),
            )
            if use_cache and self._norm_weight_cache_key == cache_key and self._norm_weight_cache is not None:
                w_f = self._norm_weight_cache
            else:
                w_f = F.normalize(self.weight.float(), dim=1, eps=1e-4)
                if use_cache:
                    self._norm_weight_cache_key = cache_key
                    self._norm_weight_cache = w_f
            cos = F.linear(x_f, w_f)
            if labels is None:
                return cos * self.s
            labels = labels.view(-1).long()
            one_hot = torch.zeros_like(cos)
            one_hot.scatter_(1, labels.unsqueeze(1), 1.0)
            return (cos - one_hot * self.m) * self.s


class PhysicalAwareClassifier(nn.Module):
    """
    Classification head with explicit identity / DAC / PA embeddings.

    Inputs:
      base      : fused shared embedding
      dac_local : DAC-specialized embedding from widely-linear branch
      pa_local  : PA-specialized embedding from memory-polynomial branch
    """
    def __init__(
        self,
        base_dim: int,
        emb_dim: int,
        num_classes: int,
        drop: float = 0.25,
        margin_s: float = 30.0,
        margin_m: float = 0.35,
        gate_alpha: float = 0.35,
        use_dac: bool = True,
        use_pa: bool = True,
    ):
        super().__init__()
        self.emb_dim = int(emb_dim)
        self.gate_alpha = float(gate_alpha)
        self.use_dac = bool(use_dac)
        self.use_pa = bool(use_pa)

        self.id_proj = nn.Sequential(
            nn.Linear(base_dim, emb_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(drop * 0.5),
        )
        self.dac_proj = (
            nn.Sequential(
                nn.Linear(base_dim + emb_dim, emb_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(drop * 0.5),
            ) if self.use_dac else None
        )
        self.pa_proj = (
            nn.Sequential(
                nn.Linear(base_dim + emb_dim, emb_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(drop * 0.5),
            ) if self.use_pa else None
        )
        defect_count = int(self.use_dac) + int(self.use_pa)
        self.id_gate = (
            nn.Sequential(
                nn.Linear(defect_count * emb_dim, emb_dim),
                nn.Sigmoid(),
            ) if defect_count > 0 else None
        )
        self.joint_proj = nn.Sequential(
            nn.Linear((1 + defect_count) * emb_dim, emb_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(drop * 0.5),
        )
        self.imp_merge = (
            nn.Sequential(
                nn.Linear(defect_count * emb_dim, emb_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(drop * 0.25),
            ) if defect_count > 0 else None
        )
        self.head = CosFaceHead(emb_dim, num_classes, s=margin_s, m=margin_m)

        hid = max(16, emb_dim // 2)
        self.dac_head = (
            nn.Sequential(
                nn.Linear(emb_dim, hid),
                nn.ReLU(inplace=True),
                nn.Linear(hid, 1),
            ) if self.use_dac else None
        )
        self.pa_head = (
            nn.Sequential(
                nn.Linear(emb_dim, hid),
                nn.ReLU(inplace=True),
                nn.Linear(hid, 1),
            ) if self.use_pa else None
        )

    def _compute_features_for_head(
        self,
        base: torch.Tensor,
        dac_local: torch.Tensor,
        pa_local: torch.Tensor,
        dac_delta: Optional[torch.Tensor],
        pa_delta: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, list, torch.Tensor, torch.Tensor]:
        feat_id = self.id_proj(base)
        zero_emb = base.new_zeros((base.size(0), self.emb_dim))
        feat_dac = self.dac_proj(torch.cat([base, dac_local], dim=1)) if self.dac_proj is not None else zero_emb
        feat_pa = self.pa_proj(torch.cat([base, pa_local], dim=1)) if self.pa_proj is not None else zero_emb

        if self.use_dac and dac_delta is not None:
            feat_dac = feat_dac + dac_delta
        if self.use_pa and pa_delta is not None:
            feat_pa = feat_pa + pa_delta

        defect_feats = []
        if self.use_dac:
            defect_feats.append(feat_dac)
        if self.use_pa:
            defect_feats.append(feat_pa)
        if self.id_gate is not None and len(defect_feats) > 0:
            g = self.id_gate(torch.cat(defect_feats, dim=1))
            feat_id = feat_id * (1.0 + self.gate_alpha * g)

        feat_joint = self.joint_proj(torch.cat([feat_id] + defect_feats, dim=1))
        return feat_id, feat_dac, feat_pa, defect_feats, feat_joint, zero_emb

    def forward_logits(
        self,
        base: torch.Tensor,
        dac_local: torch.Tensor,
        pa_local: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        dac_delta: Optional[torch.Tensor] = None,
        pa_delta: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        _, _, _, _, feat_joint, _ = self._compute_features_for_head(
            base,
            dac_local=dac_local,
            pa_local=pa_local,
            dac_delta=dac_delta,
            pa_delta=pa_delta,
        )
        return self.head(feat_joint, labels=labels)

    def forward(
        self,
        base: torch.Tensor,
        dac_local: torch.Tensor,
        pa_local: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        return_emb: bool = False,
        dac_delta: Optional[torch.Tensor] = None,
        pa_delta: Optional[torch.Tensor] = None,
    ):
        feat_id, feat_dac, feat_pa, defect_feats, feat_joint, zero_emb = self._compute_features_for_head(
            base,
            dac_local=dac_local,
            pa_local=pa_local,
            dac_delta=dac_delta,
            pa_delta=pa_delta,
        )
        feat_imp = self.imp_merge(torch.cat(defect_feats, dim=1)) if self.imp_merge is not None else zero_emb
        logits = self.head(feat_joint, labels=labels)
        dac_pred = torch.sigmoid(self.dac_head(feat_dac).squeeze(-1)) if self.dac_head is not None else base.new_zeros(base.size(0))
        pa_pred = torch.sigmoid(self.pa_head(feat_pa).squeeze(-1)) if self.pa_head is not None else base.new_zeros(base.size(0))

        if not return_emb:
            return logits, dac_pred
        return logits, dac_pred, pa_pred, feat_id, feat_dac, feat_pa, feat_imp, feat_joint


# ----------------------- Main model -----------------------
class CVSincNet(nn.Module):
    """
    DAC/PA-aware CV-SincNet.

    Compared with the previous version, this model adds:
      1) DAC branch with widely-linear complex convolution.
      2) PA branch with memory-polynomial lift + envelope gating.
      3) Explicit identity / DAC / PA embedding decoupling in classifier.

    Interface compatibility:
      - forward(x, y=None, return_aux=False)
      - build_model(...)
      - return_aux contains legacy keys feat_cls / feat_imp / feat_dac / feat_con.
    """
    def __init__(
        self,
        num_classes: int = 16,
        sample_rate: float = 5e6,
        dataset: str = "oralce",
        input_len: int = 1024,
        pad_crop_mode: str = "center",
        sinc_out: int = 48,
        sinc_kernel: int = 79,
        time_bottleneck: int = 96,
        emb_dim: int = 256,
        drop: float = 0.25,
        freq_bands: int = 64,
        use_rfft_pair: bool = True,
        use_circularity: bool = True,
        eps: float = 1e-8,
        branch_drop_p: float = 0.0,
        use_freq_band_gate: bool = True,
        freq_gate_alpha: float = 0.6,
        use_freq_stats: bool = True,
        use_pa_stats: bool = True,
        use_aux_spectral_stats: bool = True,
        freq_feature_source: str = "raw_fft",
        pa_feature_source: str = "raw_iq",
        channel_trim_scale: float = 1.0,
        use_nonlinear_basis: bool = True,
        include_z_abs4: bool = False,
        nl_clip: float = 1.2,
        margin_s: float = 30.0,
        margin_m: float = 0.35,
        pa_memory_depth: int = 4,
        pa_orders: Sequence[int] = (1, 3, 5),
        pa_gate_alpha: float = 0.5,
        time_ch1: int = 128,
        time_ch2: int = 192,
        time_ch3: int = 192,
        dac_ch: int = 64,
        freq_ch1: int = 32,
        freq_ch2: int = 64,
        freq_ch3: int = 64,
        pa_ch1: int = 96,
        pa_ch2: int = 128,
        pa_ch3: int = 128,
        branch_ablation: str = "none",
        mixstyle_on: bool = False,
        mixstyle_p: float = 0.3,
        mixstyle_alpha: float = 0.1,
        mixstyle_eps: float = 1e-6,
        mixstyle_layers: str = "time_down,t1",
        mixstyle_use_domain_label: bool = True,
        mixstyle_mix: str = "crossdomain",
        mixstyle_strength: float = 1.0,
        mixstyle_fallback: str = "random",
        time_stability_mode: str = "off",
        freq_stability_mode: str = "off",
        time_stability_channels: int = 8,
        freq_stability_channels: int = 4,
    ):
        super().__init__()
        self.dataset = str(dataset)
        self.input_len = int(input_len) if int(input_len) > 0 else 1024
        self.expected_len = self.input_len
        self.emb_dim = int(emb_dim)
        self.pad_crop_mode = str(pad_crop_mode)
        self.sample_rate = float(sample_rate)
        self.freq_bands = int(freq_bands)
        self.use_rfft_pair = bool(use_rfft_pair)
        self.use_circularity = bool(use_circularity)
        self.eps = float(eps)
        self.branch_drop_p = float(branch_drop_p)
        self.use_freq_band_gate = bool(use_freq_band_gate)
        self.freq_gate_alpha = float(freq_gate_alpha)
        self.use_freq_stats = bool(use_freq_stats)
        self.use_pa_stats = bool(use_pa_stats)
        self.use_aux_spectral_stats = bool(use_aux_spectral_stats)
        self.freq_feature_source = str(freq_feature_source or "raw_fft").lower().strip()
        self.pa_feature_source = str(pa_feature_source or "raw_iq").lower().strip()
        if self.freq_feature_source not in ("raw_fft", "sinc_energy", "sinc_phase_asym"):
            raise ValueError("freq_feature_source must be one of: raw_fft,sinc_energy,sinc_phase_asym")
        if self.pa_feature_source not in ("raw_iq", "sinc_lowrank"):
            raise ValueError("pa_feature_source must be one of: raw_iq,sinc_lowrank")
        self.use_nonlinear_basis = bool(use_nonlinear_basis)
        self.include_z_abs4 = bool(include_z_abs4)
        self.nl_clip = float(nl_clip)
        self.time_stability_mode = _parse_stability_mode(
            "time_stability_mode",
            time_stability_mode,
            ("off", "phase_delta"),
        )
        self.freq_stability_mode = _parse_stability_mode(
            "freq_stability_mode",
            freq_stability_mode,
            ("off", "dsq"),
        )
        self.time_stability_channels = int(max(1, time_stability_channels))
        self.freq_stability_channels = int(max(1, freq_stability_channels))
        self.branch_ablation = self._parse_branch_ablation(branch_ablation)
        self.use_time_path = not self._ablated("no_time")
        self.use_dac_path = not self._ablated("no_dac")
        self.use_pa_path = not self._ablated("no_pa")
        self.use_freq_path = not self._ablated("no_freq")
        self.use_stats_path = self.use_freq_path and (not self._ablated("no_stats"))
        self.mixstyle_on = bool(mixstyle_on)
        self.mixstyle_layers = self._parse_mixstyle_layers(mixstyle_layers)
        self.mixstyle_use_domain_label = bool(mixstyle_use_domain_label)
        self.mixstyle = MixStyle1D(
            p=float(mixstyle_p),
            alpha=float(mixstyle_alpha),
            eps=float(mixstyle_eps),
            mix=str(mixstyle_mix),
            strength=float(mixstyle_strength),
            fallback=str(mixstyle_fallback),
        )

        self.channel_trim_scale = max(0.05, float(channel_trim_scale))
        if abs(self.channel_trim_scale - 1.0) > 1e-8:
            time_bottleneck = _trim_channels(time_bottleneck, self.channel_trim_scale, 8)
            time_ch1 = _trim_channels(time_ch1, self.channel_trim_scale, 8)
            time_ch2 = _trim_channels(time_ch2, self.channel_trim_scale, 8)
            time_ch3 = _trim_channels(time_ch3, self.channel_trim_scale, 8)
            dac_ch = _trim_channels(dac_ch, self.channel_trim_scale, 4)
            freq_ch1 = _trim_channels(freq_ch1, self.channel_trim_scale, 4)
            freq_ch2 = _trim_channels(freq_ch2, self.channel_trim_scale, 4)
            freq_ch3 = _trim_channels(freq_ch3, self.channel_trim_scale, 4)
            pa_ch1 = _trim_channels(pa_ch1, self.channel_trim_scale, 4)
            pa_ch2 = _trim_channels(pa_ch2, self.channel_trim_scale, 4)
            pa_ch3 = _trim_channels(pa_ch3, self.channel_trim_scale, 4)

        # Shared front-end is needed by time/DAC and Sinc-derived freq/PA probes.
        self.freq_uses_sinc = self.use_freq_path and self.freq_feature_source.startswith("sinc_")
        self.pa_uses_sinc = self.use_pa_path and self.pa_feature_source.startswith("sinc_")
        self.sinc = (
            SincConv1d(sinc_out, sinc_kernel, sample_rate=sample_rate, dataset=dataset, input_len=input_len, pad_crop_mode=pad_crop_mode)
            if (self.use_time_path or self.use_dac_path or self.freq_uses_sinc or self.pa_uses_sinc) else None
        )
        self.hf = HighFreqEmphasis() if (self.use_time_path or self.use_dac_path) else None

        time_in = 2 * sinc_out + 4
        if self.use_nonlinear_basis:
            time_in += 2 * sinc_out
        if self.use_nonlinear_basis and self.include_z_abs4:
            time_in += 2 * sinc_out
        self.time_stability = None
        if self.use_time_path and self.time_stability_mode == "phase_delta":
            self.time_stability = PhaseDeltaStabilityStem(
                sinc_out,
                out_channels=self.time_stability_channels,
                eps=max(1e-6, self.eps),
            )
            time_in += self.time_stability_channels

        Cb = int(time_bottleneck)
        gn = _pick_gn_groups(Cb)
        self.time_fuse = (
            nn.Sequential(
                nn.Conv1d(time_in, Cb, kernel_size=1, bias=False),
                nn.GroupNorm(gn, Cb),
                nn.ReLU(inplace=True),
            ) if self.use_time_path else None
        )
        self.time_down = nn.AvgPool1d(2) if self.use_time_path else None
        self.t1 = DSConvBlock1d(Cb, int(time_ch1), k=5, pool=2, drop=0.10) if self.use_time_path else None
        self.t2 = DSConvBlock1d(int(time_ch1), int(time_ch2), k=5, pool=2, drop=0.10) if self.use_time_path else None
        self.t3 = DSConvBlock1d(int(time_ch2), int(time_ch3), k=3, pool=1, drop=0.10) if self.use_time_path else None
        self.t_pool = nn.AdaptiveAvgPool1d(1) if self.use_time_path else None
        self.t_proj = nn.Linear(int(time_ch3), emb_dim) if self.use_time_path else None

        # DAC branch: widely-linear complex conv over filterbank channels + injected HF details
        self.dac_hf_proj = (
            nn.Sequential(
                nn.Conv1d(4, 2 * sinc_out, kernel_size=1, bias=False),
                nn.GroupNorm(_pick_gn_groups(2 * sinc_out), 2 * sinc_out),
                nn.SiLU(inplace=True),
            ) if self.use_dac_path else None
        )
        self.dac_b1 = WLComplexBlock1d(sinc_out, sinc_out, k=5, dilation=1, pool=1, drop=0.05, residual=True) if self.use_dac_path else None
        self.dac_b2 = WLComplexBlock1d(sinc_out, int(dac_ch), k=3, dilation=1, pool=1, drop=0.05, residual=True) if self.use_dac_path else None
        self.dac_b3 = WLComplexBlock1d(int(dac_ch), int(dac_ch), k=3, dilation=2, pool=2, drop=0.05, residual=True) if self.use_dac_path else None
        self.dac_pool = nn.AdaptiveAvgPool1d(1) if self.use_dac_path else None
        self.dac_proj = (
            nn.Sequential(
                nn.Linear(2 * int(dac_ch), emb_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(drop * 0.25),
            ) if self.use_dac_path else None
        )

        # frequency branch (legacy shared/freq contribution)
        self.freq_stability = None
        freq_in = 4
        if self.use_freq_path and self.freq_stability_mode == "dsq":
            self.freq_stability = DSQFreqStabilityStem(out_channels=self.freq_stability_channels)
            freq_in += self.freq_stability_channels
        self.freq_feature_channels = int(freq_in)
        self.f1 = DSConvBlock1d(freq_in, int(freq_ch1), k=5, pool=2, drop=0.05) if self.use_freq_path else None
        self.f2 = DSConvBlock1d(int(freq_ch1), int(freq_ch2), k=5, pool=2, drop=0.05) if self.use_freq_path else None
        self.f3 = DSConvBlock1d(int(freq_ch2), int(freq_ch3), k=3, pool=1, drop=0.05) if self.use_freq_path else None
        self.f_pool = nn.AdaptiveAvgPool1d(1) if self.use_freq_path else None
        self.f_proj = nn.Linear(int(freq_ch3), emb_dim) if self.use_freq_path else None
        self.freq_gate = FreqBandGate1d(freq_in, k=5, alpha=freq_gate_alpha) if (self.use_freq_path and self.use_freq_band_gate) else nn.Identity()
        self.dac_subband_agg = (
            SubbandGatedAggregator(in_ch=4, emb_dim=emb_dim, hidden=64, temperature=1.0)
            if (self.use_stats_path and self.use_dac_path) else None
        )
        self.freq_stats_proj = (
            nn.Sequential(
                nn.Linear(3, emb_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(drop * 0.25),
            ) if (self.use_stats_path and self.use_freq_stats) else None
        )
        self.pa_stats_proj = (
            nn.Sequential(
                nn.Linear(3, emb_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(drop * 0.25),
            ) if (self.use_stats_path and self.use_pa_path and self.use_pa_stats) else None
        )

        # PA branch: memory polynomial lift + envelope-aware large receptive field convs
        self.pa_lift = MemoryPolynomialLift(memory_depth=pa_memory_depth, orders=pa_orders, clip=2.0) if self.use_pa_path else None
        pa_in = 2 * pa_memory_depth * len(pa_orders)
        self.pa_gate = EnvelopeGate1d(pa_in, alpha=pa_gate_alpha, k=5) if self.use_pa_path else None
        self.pa_b1 = DilatedConvBlock1d(pa_in, int(pa_ch1), k=7, dilation=1, pool=2, drop=0.08) if self.use_pa_path else None
        self.pa_b2 = DilatedConvBlock1d(int(pa_ch1), int(pa_ch2), k=7, dilation=2, pool=2, drop=0.08) if self.use_pa_path else None
        self.pa_b3 = DilatedConvBlock1d(int(pa_ch2), int(pa_ch3), k=5, dilation=4, pool=1, drop=0.08) if self.use_pa_path else None
        self.pa_pool = nn.AdaptiveAvgPool1d(1) if self.use_pa_path else None
        self.pa_proj = (
            nn.Sequential(
                nn.Linear(int(pa_ch3), emb_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(drop * 0.25),
            ) if self.use_pa_path else None
        )

        # fuse + projection
        fuse_in = 0
        if self.use_time_path:
            fuse_in += emb_dim
        if self.use_freq_path:
            fuse_in += emb_dim
            if self.use_circularity:
                fuse_in += 1
        if fuse_in <= 0:
            fuse_in = emb_dim
        self.fuse = nn.Sequential(
            nn.Linear(fuse_in, emb_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(drop),
        )
        self.con_proj = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(drop * 0.25),
        )

        self.cls_head = PhysicalAwareClassifier(
            base_dim=emb_dim,
            emb_dim=emb_dim,
            num_classes=num_classes,
            drop=drop,
            margin_s=margin_s,
            margin_m=margin_m,
            gate_alpha=0.35,
            use_dac=self.use_dac_path,
            use_pa=self.use_pa_path,
        )

        self._init_weights()

    @staticmethod
    def _parse_branch_ablation(branch_ablation: str):
        raw = str(branch_ablation or "none").lower().replace(";", ",").replace("+", ",")
        aliases = {
            "none": "",
            "base": "",
            "off": "",
            "no_time_branch": "no_time",
            "no_dac_branch": "no_dac",
            "no_pa_branch": "no_pa",
            "no_freq_branch": "no_freq",
            "no_spectral": "no_freq",
            "no_spec": "no_freq",
            "no_stat": "no_stats",
            "no_spectral_stats": "no_stats",
            "no_dac_pa": "no_dac,no_pa",
            "no_physical": "no_dac,no_pa",
            "time_only": "no_dac,no_pa,no_freq,no_stats",
            "freq_only": "no_time,no_dac,no_pa,no_stats",
            "no_defect_branches": "no_dac,no_pa",
        }
        expanded = []
        for item in raw.split(","):
            item = item.strip()
            if item == "":
                continue
            item = aliases.get(item, item)
            expanded.extend([z.strip() for z in item.split(",") if z.strip()])
        valid = {"no_time", "no_dac", "no_pa", "no_freq", "no_stats", "no_aux_logits"}
        unknown = sorted({z for z in expanded if z not in valid})
        if unknown:
            raise ValueError(f"Unknown branch_ablation={unknown}; valid={sorted(valid)}")
        return frozenset(expanded)

    def _ablated(self, name: str) -> bool:
        return str(name) in self.branch_ablation

    @staticmethod
    def _parse_mixstyle_layers(mixstyle_layers: str):
        raw = str(mixstyle_layers or "").lower().replace(";", ",").replace("+", ",")
        layers = {z.strip() for z in raw.split(",") if z.strip()}
        valid = {"time_down", "t1", "t2"}
        unknown = sorted(layers - valid)
        if unknown:
            raise ValueError(f"Unknown mixstyle_layers={unknown}; valid={sorted(valid)}")
        return frozenset(layers)

    def _apply_mixstyle(
        self,
        x: torch.Tensor,
        layer_name: str,
        domain_labels: Optional[torch.Tensor],
        tx_labels: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if not self.mixstyle_on or layer_name not in self.mixstyle_layers:
            return x
        labels = domain_labels if self.mixstyle_use_domain_label else None
        return self.mixstyle(x, labels, tx_labels=tx_labels)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="linear")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @staticmethod
    def _to_complex(iq: torch.Tensor) -> torch.Tensor:
        return torch.complex(iq[:, 0, :], iq[:, 1, :])

    def _sinc_on_iq(self, x: torch.Tensor) -> torch.Tensor:
        return self.sinc.forward_iq_pair(x)

    def _nonlinear_basis_after_filterbank(self, sinc_iq: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, C, L = sinc_iq.shape
        S = C // 2
        y = sinc_iq.view(B, 2, S, L)
        i = torch.clamp(y[:, 0, :, :], -self.nl_clip, self.nl_clip)
        q = torch.clamp(y[:, 1, :, :], -self.nl_clip, self.nl_clip)
        a2 = i * i + q * q
        i3 = i * a2
        q3 = q * a2
        z_abs2 = torch.cat([i3, q3], dim=1)
        z_abs4 = None
        if self.include_z_abs4:
            a4 = a2 * a2
            z_abs4 = torch.cat([i * a4, q * a4], dim=1)
        return z_abs2, z_abs4

    def _fft_full_via_rfft_pair(self, x: torch.Tensor) -> torch.Tensor:
        I = x[:, 0, :]
        Q = x[:, 1, :]
        Fi = torch.fft.rfft(I, dim=-1)
        Fq = torch.fft.rfft(Q, dim=-1)
        if Fi.size(-1) <= 2:
            Fi_full = torch.fft.fft(I, dim=-1)
            Fq_full = torch.fft.fft(Q, dim=-1)
        else:
            tail_i = torch.conj(Fi[..., 1:-1]).flip(dims=[-1])
            tail_q = torch.conj(Fq[..., 1:-1]).flip(dims=[-1])
            Fi_full = torch.cat([Fi, tail_i], dim=-1)
            Fq_full = torch.cat([Fq, tail_q], dim=-1)
        return Fi_full + 1j * Fq_full

    def _sinc_lowrank_iq(self, sinc_iq: torch.Tensor) -> torch.Tensor:
        if sinc_iq is None:
            raise ValueError("sinc_iq is required for Sinc-derived PA features")
        B, C, L = sinc_iq.shape
        if C % 2 != 0:
            raise ValueError("Sinc IQ tensor must have paired I/Q channels")
        Fch = C // 2
        return sinc_iq.reshape(B, 2, Fch, L).mean(dim=2)

    def _sinc_compressed_features(self, sinc_iq: torch.Tensor):
        if sinc_iq is None:
            raise ValueError("sinc_iq is required for Sinc-derived frequency features")
        B, C, L = sinc_iq.shape
        if C % 2 != 0:
            raise ValueError("Sinc IQ tensor must have paired I/Q channels")
        eps = self.eps
        Fch = C // 2
        pair = sinc_iq.reshape(B, 2, Fch, L)
        i = pair[:, 0, :, :]
        q = pair[:, 1, :, :]
        energy = torch.nan_to_num(i * i + q * q, nan=0.0, posinf=1e6, neginf=0.0).mean(dim=-1).clamp_min(eps)
        rev = torch.flip(energy, dims=[-1])
        log_e = torch.log1p(energy.clamp_max(1e6))
        log_rev = torch.log1p(rev.clamp_max(1e6))
        log_r = (torch.log(energy + eps) - torch.log(rev + eps)).clamp(-20.0, 20.0)
        asym = torch.abs(energy - rev) / (energy + rev + eps)
        if self.freq_feature_source == "sinc_phase_asym" and L > 1:
            cross = q[:, :, 1:] * i[:, :, :-1] - i[:, :, 1:] * q[:, :, :-1]
            dot = i[:, :, 1:] * i[:, :, :-1] + q[:, :, 1:] * q[:, :, :-1]
            dphi = torch.atan2(cross, dot + eps)
            phase_std = dphi.std(dim=-1, unbiased=False).clamp_min(eps)
            log_rev = torch.log1p(phase_std.clamp_max(1e6))
        K = max(4, self.freq_bands)
        pooled = F.adaptive_avg_pool1d(
            torch.stack([log_e, log_rev, log_r, asym, energy, rev], dim=1),
            K,
        )
        p_pos = pooled[:, 0, :]
        p_neg = pooled[:, 1, :]
        p_rat = pooled[:, 2, :]
        p_asym = pooled[:, 3, :]
        feat_f = torch.stack([p_pos, p_neg, p_rat, p_asym], dim=1)
        if not self.use_aux_spectral_stats:
            return feat_f, sinc_iq.new_zeros((B, 3)), sinc_iq.new_zeros((B, 3))
        pos_lin = pooled[:, 4, :]
        neg_lin = pooled[:, 5, :]
        tot_lin = torch.nan_to_num(pos_lin + neg_lin, nan=0.0, posinf=1e6, neginf=0.0).clamp_min(eps).clamp_max(1e6)
        hf_start = max(0, int(0.75 * K))
        hf_ratio = tot_lin[:, hf_start:].sum(dim=1, keepdim=True) / (tot_lin.sum(dim=1, keepdim=True) + eps)
        asym_hf_mean = p_asym[:, hf_start:].mean(dim=1, keepdim=True)
        flatness = torch.exp(torch.mean(torch.log(tot_lin.clamp_min(eps)), dim=1, keepdim=True).clamp(-20.0, 20.0)) / (torch.mean(tot_lin, dim=1, keepdim=True) + eps)
        dac_stats = torch.cat([hf_ratio, asym_hf_mean, flatness], dim=1)
        edge_bins = max(1, int(0.20 * K))
        center_l = max(0, int(0.30 * K))
        center_r = max(center_l + 1, int(0.70 * K))
        edge_energy = tot_lin[:, -edge_bins:].sum(dim=1, keepdim=True)
        center_energy = tot_lin[:, center_l:center_r].sum(dim=1, keepdim=True) + eps
        edge_ratio = edge_energy / (tot_lin.sum(dim=1, keepdim=True) + eps)
        regrowth_ratio = edge_energy / center_energy
        mu = torch.mean(tot_lin, dim=1, keepdim=True)
        var = torch.mean((tot_lin - mu) ** 2, dim=1, keepdim=True) + eps
        spec_kurtosis = (torch.mean((tot_lin - mu).clamp(-1e3, 1e3) ** 4, dim=1, keepdim=True) / (var * var).clamp_min(eps)).clamp(0.0, 1e4)
        pa_stats = torch.cat([edge_ratio, regrowth_ratio, spec_kurtosis], dim=1)
        return feat_f, dac_stats, pa_stats

    def _mirror_compressed_features(self, x: torch.Tensor, sinc_iq: Optional[torch.Tensor] = None):
        """
        Returns:
          feat_f   : (B,4,K)  [logP_pos, logP_neg, logR, asym]
          rho      : (B,1) or None
          dac_stats: (B,3)   [hf_ratio, asym_hf_mean, flatness]
          pa_stats : (B,3)   [edge_ratio, regrowth_ratio, spec_kurtosis]
        """
        B, _, L = x.shape
        eps = self.eps
        if self.freq_feature_source != "raw_fft":
            feat_f, dac_stats, pa_stats = self._sinc_compressed_features(sinc_iq)
            rho = None
            if self.use_circularity:
                z = self._to_complex(x)
                Ez2 = torch.mean(z * z, dim=-1)
                Eabs2 = torch.mean(z.real * z.real + z.imag * z.imag, dim=-1)
                rho = (torch.abs(Ez2) / (Eabs2 + eps)).unsqueeze(1)
            return feat_f, rho, dac_stats, pa_stats
        if self.use_rfft_pair and (L % 2 == 0):
            spec = self._fft_full_via_rfft_pair(x)
        else:
            spec = torch.fft.fft(self._to_complex(x), dim=-1)

        power = spec.real * spec.real + spec.imag * spec.imag
        power = torch.nan_to_num(power, nan=0.0, posinf=1e6, neginf=0.0).clamp_min(0.0)
        half = L // 2
        if half <= 1:
            pos = power[:, :1]
            neg = power[:, :1]
        else:
            pos = power[:, 1:half]
            neg = torch.flip(power[:, half + 1:], dims=[-1])
            m = min(pos.size(-1), neg.size(-1))
            pos = pos[:, :m]
            neg = neg[:, :m]

        pos = torch.nan_to_num(pos, nan=0.0, posinf=1e6, neginf=0.0).clamp_min(0.0)
        neg = torch.nan_to_num(neg, nan=0.0, posinf=1e6, neginf=0.0).clamp_min(0.0)
        logP_pos = torch.log1p(pos.clamp_max(1e6))
        logP_neg = torch.log1p(neg.clamp_max(1e6))
        logR = (torch.log(pos + eps) - torch.log(neg + eps)).clamp(-20.0, 20.0)
        asym = torch.abs(pos - neg) / (pos + neg + eps)

        K = max(4, self.freq_bands)
        pooled = F.adaptive_avg_pool1d(
            torch.stack([logP_pos, logP_neg, logR, asym, pos, neg], dim=1),
            K,
        )
        p_pos = pooled[:, 0, :]
        p_neg = pooled[:, 1, :]
        p_rat = pooled[:, 2, :]
        p_asym = pooled[:, 3, :]
        feat_f = torch.stack([p_pos, p_neg, p_rat, p_asym], dim=1)
        if not self.use_aux_spectral_stats:
            rho = x.new_zeros((B, 1)) if self.use_circularity else None
            return feat_f, rho, x.new_zeros((B, 3)), x.new_zeros((B, 3))

        pos_lin = pooled[:, 4, :]
        neg_lin = pooled[:, 5, :]
        tot_lin = torch.nan_to_num(pos_lin + neg_lin, nan=0.0, posinf=1e6, neginf=0.0).clamp_min(eps).clamp_max(1e6)

        hf_start = max(0, int(0.75 * K))
        hf_ratio = tot_lin[:, hf_start:].sum(dim=1, keepdim=True) / (tot_lin.sum(dim=1, keepdim=True) + eps)
        asym_hf_mean = p_asym[:, hf_start:].mean(dim=1, keepdim=True)
        flatness = torch.exp(torch.mean(torch.log(tot_lin.clamp_min(eps)), dim=1, keepdim=True).clamp(-20.0, 20.0)) / (torch.mean(tot_lin, dim=1, keepdim=True) + eps)
        dac_stats = torch.cat([hf_ratio, asym_hf_mean, flatness], dim=1)

        edge_bins = max(1, int(0.20 * K))
        center_l = max(0, int(0.30 * K))
        center_r = max(center_l + 1, int(0.70 * K))
        edge_energy = tot_lin[:, -edge_bins:].sum(dim=1, keepdim=True)
        center_energy = tot_lin[:, center_l:center_r].sum(dim=1, keepdim=True) + eps
        edge_ratio = edge_energy / (tot_lin.sum(dim=1, keepdim=True) + eps)
        regrowth_ratio = edge_energy / center_energy
        mu = torch.mean(tot_lin, dim=1, keepdim=True)
        var = torch.mean((tot_lin - mu) ** 2, dim=1, keepdim=True) + eps
        spec_kurtosis = (torch.mean((tot_lin - mu).clamp(-1e3, 1e3) ** 4, dim=1, keepdim=True) / (var * var).clamp_min(eps)).clamp(0.0, 1e4)
        pa_stats = torch.cat([edge_ratio, regrowth_ratio, spec_kurtosis], dim=1)

        rho = None
        if self.use_circularity:
            z = self._to_complex(x)
            Ez2 = torch.mean(z * z, dim=-1)
            Eabs2 = torch.mean(z.real * z.real + z.imag * z.imag, dim=-1)
            rho = (torch.abs(Ez2) / (Eabs2 + eps)).unsqueeze(1)

        return feat_f, rho, dac_stats, pa_stats

    def forward(
        self,
        x: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        return_aux: bool = False,
        domain_labels: Optional[torch.Tensor] = None,
    ):
        x = pad_crop_iq(x, self.input_len, mode=self.pad_crop_mode)
        B = x.size(0)
        zero_emb = x.new_zeros((B, self.emb_dim))
        need_time = self.use_time_path
        need_dac = self.use_dac_path
        need_pa = self.use_pa_path
        need_freq = self.use_freq_path
        need_stats = self.use_stats_path
        need_sinc = need_time or need_dac or (need_freq and self.freq_uses_sinc) or (need_pa and self.pa_uses_sinc)

        sinc_iq = self._sinc_on_iq(x) if need_sinc else None
        hf = self.hf(x) if need_sinc and self.hf is not None else None

        # ----- shared time branch -----
        if need_time:
            feats_t = [sinc_iq]
            if self.use_nonlinear_basis:
                z_abs2, z_abs4 = self._nonlinear_basis_after_filterbank(sinc_iq)
                feats_t.append(z_abs2)
                if self.include_z_abs4 and z_abs4 is not None:
                    feats_t.append(z_abs4)
            feats_t.append(hf)
            if self.time_stability is not None:
                feats_t.append(self.time_stability(sinc_iq))
            t = torch.cat(feats_t, dim=1)
            t = self.time_fuse(t)
            t = self.time_down(t)
            t = self._apply_mixstyle(t, "time_down", domain_labels, y)
            t = self.t1(t)
            t = self._apply_mixstyle(t, "t1", domain_labels, y)
            t = self.t2(t)
            t = self._apply_mixstyle(t, "t2", domain_labels, y)
            t = self.t3(t)
            t = self.t_pool(t).squeeze(-1)
            t_emb = self.t_proj(t)

            if self.training and self.branch_drop_p > 0:
                drop_mask = (torch.rand((B, 1), device=t_emb.device) < self.branch_drop_p).float()
                t_emb = t_emb * (1.0 - drop_mask)
        else:
            t_emb = zero_emb

        # ----- DAC branch -----
        if need_dac:
            dac_pair = sinc_iq + 0.5 * self.dac_hf_proj(hf)
            d = self.dac_b1(dac_pair)
            d = self.dac_b2(d)
            d = self.dac_b3(d)
            d = self.dac_pool(d).squeeze(-1)
            dac_local = self.dac_proj(d)
        else:
            dac_local = zero_emb

        # ----- freq branch -----
        if need_freq:
            feat_f, rho, dac_stats, pa_stats = self._mirror_compressed_features(x, sinc_iq=sinc_iq)
            if self.freq_stability is not None:
                feat_f = torch.cat([feat_f, self.freq_stability(feat_f)], dim=1)
            feat_f = self.freq_gate(feat_f)
            dac_delta = self.dac_subband_agg(feat_f[:, :4, :]) if (need_stats and self.dac_subband_agg is not None) else zero_emb

            f = self.f1(feat_f)
            f = self.f2(f)
            f = self.f3(f)
            f = self.f_pool(f).squeeze(-1)
            f_emb = self.f_proj(f)
            if need_stats and self.use_freq_stats and (self.freq_stats_proj is not None):
                f_emb = f_emb + self.freq_stats_proj(dac_stats)
            if (not need_stats) and rho is not None:
                rho = torch.zeros_like(rho)
                dac_stats = torch.zeros_like(dac_stats)
                pa_stats = torch.zeros_like(pa_stats)
        else:
            f_emb = zero_emb
            dac_delta = zero_emb
            rho = x.new_zeros((B, 1)) if self.use_circularity else None
            dac_stats = x.new_zeros((B, 3))
            pa_stats = x.new_zeros((B, 3))
        if not need_dac:
            dac_delta = zero_emb

        # ----- PA branch -----
        if need_pa:
            pa_input = self._sinc_lowrank_iq(sinc_iq) if self.pa_feature_source == "sinc_lowrank" else x
            pa_feat = self.pa_lift(pa_input)
            pa_feat = self.pa_gate(pa_feat, pa_input)
            p = self.pa_b1(pa_feat)
            p = self.pa_b2(p)
            p = self.pa_b3(p)
            p = self.pa_pool(p).squeeze(-1)
            pa_local = self.pa_proj(p)
            pa_delta = self.pa_stats_proj(pa_stats) if (need_stats and self.pa_stats_proj is not None) else zero_emb
            pa_local = pa_local + 0.25 * pa_delta
        else:
            pa_local = zero_emb
            pa_delta = zero_emb

        base_parts = []
        if need_time:
            base_parts.append(t_emb)
        if need_freq:
            base_parts.append(f_emb)
            if rho is not None:
                base_parts.append(rho)
        if len(base_parts) == 0:
            base_parts.append(zero_emb)
        base_in = torch.cat(base_parts, dim=1)
        base = self.fuse(base_in)

        if not return_aux:
            return self.cls_head.forward_logits(
                base,
                dac_local=dac_local,
                pa_local=pa_local,
                labels=y,
                dac_delta=dac_delta,
                pa_delta=pa_delta,
            )

        feat_con = self.con_proj(base)

        logits, dac_pred, pa_pred, feat_id, feat_dac, feat_pa, feat_imp, feat_joint = self.cls_head(
            base,
            dac_local=dac_local,
            pa_local=pa_local,
            labels=y,
            return_emb=True,
            dac_delta=dac_delta,
            pa_delta=pa_delta,
        )

        return {
            'logits': logits,
            'dac_pred': dac_pred,
            'pa_pred': pa_pred,
            'feat_cls': feat_id,
            'feat_imp': feat_imp,
            'feat_dac': feat_dac,
            'feat_pa': feat_pa,
            'feat_con': feat_con,
            't_emb': t_emb,
            'f_emb': f_emb,
            'dac_local': dac_local,
            'pa_local': pa_local,
            'rho': rho,
            'f_stats': dac_stats,     # backward-compatible alias
            'dac_stats': dac_stats,
            'pa_stats': pa_stats,
            'base': base,
            'feat_joint': feat_joint,
        }


# ----------------------- factory -----------------------
def build_model(
    num_classes: int = 16,
    model_size: str = "M",
    dataset: str = "oralce",
    input_len: int = 1024,
    sample_rate_hz: float = 5e6,
    model_variant: str = "base",
    branch_ablation: str = "none",
    mixstyle_on: bool = False,
    mixstyle_p: float = 0.3,
    mixstyle_alpha: float = 0.1,
    mixstyle_eps: float = 1e-6,
    mixstyle_layers: str = "time_down,t1",
    mixstyle_use_domain_label: bool = True,
    mixstyle_mix: str = "crossdomain",
    mixstyle_strength: float = 1.0,
    mixstyle_fallback: str = "random",
    use_circularity: bool = True,
    use_freq_stats: bool = True,
    use_pa_stats: bool = True,
    use_freq_band_gate: bool = True,
    freq_feature_source: str = "raw_fft",
    pa_feature_source: str = "raw_iq",
    pa_orders: Optional[Sequence[int] | str] = None,
    use_aux_spectral_stats: bool = True,
    channel_trim_scale: float = 1.0,
    time_stability_mode: str = "off",
    freq_stability_mode: str = "off",
    time_stability_channels: int = 8,
    freq_stability_channels: int = 4,
):
    ds = str(dataset).lower()
    ms = str(model_size).upper().strip()

    if (input_len is None) or (int(input_len) <= 0):
        input_len = 256 if ds == "wisig" else 1024

    variant = str(model_variant or "base").lower().strip()

    if ms in ("S", "SMALL"):
        cfg = dict(sinc_out=32, sinc_kernel=63, time_bottleneck=96, emb_dim=192, freq_bands=48, drop=0.40)
    elif ms in ("L", "LARGE"):
        cfg = dict(sinc_out=64, sinc_kernel=79, time_bottleneck=128, emb_dim=256, freq_bands=96, drop=0.40)
    else:
        cfg = dict(sinc_out=48, sinc_kernel=79, time_bottleneck=96, emb_dim=256, freq_bands=64, drop=0.40)

    if variant == "lite_a":
        cfg.update(emb_dim=192)
    elif variant == "lite_b":
        cfg.update(
            sinc_out=max(24, int(round(cfg["sinc_out"] * 0.75))),
            time_bottleneck=max(64, int(round(cfg["time_bottleneck"] * 0.75))),
            emb_dim=192,
            freq_bands=max(32, int(round(cfg["freq_bands"] * 0.75))),
            time_ch1=96,
            time_ch2=144,
            time_ch3=144,
            dac_ch=48,
            freq_ch1=24,
            freq_ch2=48,
            freq_ch3=48,
            pa_ch1=72,
            pa_ch2=96,
            pa_ch3=96,
        )
    elif variant == "lite_c":
        cfg.update(emb_dim=192, drop=min(0.50, float(cfg["drop"]) + 0.05))
    elif variant == "lite_d":
        cfg.update(
            sinc_out=max(24, int(round(cfg["sinc_out"] * 0.50))),
            time_bottleneck=max(48, int(round(cfg["time_bottleneck"] * 0.50))),
            emb_dim=160,
            freq_bands=max(32, int(round(cfg["freq_bands"] * 0.50))),
            time_ch1=72,
            time_ch2=96,
            time_ch3=96,
            dac_ch=32,
            freq_ch1=16,
            freq_ch2=32,
            freq_ch3=32,
            pa_ch1=48,
            pa_ch2=64,
            pa_ch3=64,
            drop=min(0.50, float(cfg["drop"]) + 0.05),
        )
    elif variant == "lite_e":
        cfg.update(
            sinc_out=max(16, int(round(cfg["sinc_out"] * 0.40))),
            time_bottleneck=max(48, int(round(cfg["time_bottleneck"] * 0.42))),
            emb_dim=128,
            freq_bands=max(24, int(round(cfg["freq_bands"] * 0.38))),
            time_ch1=48,
            time_ch2=72,
            time_ch3=72,
            dac_ch=24,
            freq_ch1=16,
            freq_ch2=24,
            freq_ch3=24,
            pa_ch1=32,
            pa_ch2=48,
            pa_ch3=48,
            drop=min(0.55, float(cfg["drop"]) + 0.08),
        )
    elif variant == "lite_f":
        cfg.update(
            sinc_out=18,
            sinc_kernel=47,
            time_bottleneck=40,
            emb_dim=96,
            freq_bands=24,
            time_ch1=40,
            time_ch2=56,
            time_ch3=56,
            dac_ch=16,
            freq_ch1=12,
            freq_ch2=18,
            freq_ch3=18,
            pa_ch1=24,
            pa_ch2=32,
            pa_ch3=32,
            pa_memory_depth=3,
            pa_orders=(1, 3),
            drop=min(0.58, float(cfg["drop"]) + 0.10),
        )
    elif variant == "lite_g":
        cfg.update(
            sinc_out=16,
            sinc_kernel=39,
            time_bottleneck=32,
            emb_dim=80,
            freq_bands=16,
            time_ch1=32,
            time_ch2=44,
            time_ch3=44,
            dac_ch=12,
            freq_ch1=10,
            freq_ch2=14,
            freq_ch3=14,
            pa_ch1=18,
            pa_ch2=24,
            pa_ch3=24,
            pa_memory_depth=2,
            pa_orders=(1, 3),
            drop=min(0.60, float(cfg["drop"]) + 0.12),
        )
    elif variant == "lite_h":
        cfg.update(
            sinc_out=12,
            sinc_kernel=31,
            time_bottleneck=24,
            emb_dim=64,
            freq_bands=12,
            time_ch1=24,
            time_ch2=32,
            time_ch3=32,
            dac_ch=8,
            freq_ch1=8,
            freq_ch2=10,
            freq_ch3=10,
            pa_ch1=12,
            pa_ch2=16,
            pa_ch3=16,
            pa_memory_depth=2,
            pa_orders=(1,),
            drop=min(0.62, float(cfg["drop"]) + 0.14),
        )
    elif variant != "base":
        raise ValueError(f"Unknown model_variant={model_variant}")

    parsed_pa_orders = _parse_odd_orders(pa_orders)
    if parsed_pa_orders is not None:
        cfg["pa_orders"] = parsed_pa_orders

    return CVSincNet(
        num_classes=int(num_classes),
        sample_rate=float(sample_rate_hz),
        dataset=ds,
        input_len=int(input_len),
        pad_crop_mode="center",
        branch_ablation=branch_ablation,
        mixstyle_on=mixstyle_on,
        mixstyle_p=mixstyle_p,
        mixstyle_alpha=mixstyle_alpha,
        mixstyle_eps=mixstyle_eps,
        mixstyle_layers=mixstyle_layers,
        mixstyle_use_domain_label=mixstyle_use_domain_label,
        mixstyle_mix=mixstyle_mix,
        mixstyle_strength=mixstyle_strength,
        mixstyle_fallback=mixstyle_fallback,
        use_circularity=use_circularity,
        use_freq_stats=use_freq_stats,
        use_pa_stats=use_pa_stats,
        use_freq_band_gate=use_freq_band_gate,
        freq_feature_source=freq_feature_source,
        pa_feature_source=pa_feature_source,
        use_aux_spectral_stats=use_aux_spectral_stats,
        channel_trim_scale=channel_trim_scale,
        time_stability_mode=time_stability_mode,
        freq_stability_mode=freq_stability_mode,
        time_stability_channels=time_stability_channels,
        freq_stability_channels=freq_stability_channels,
        **cfg,
    )
