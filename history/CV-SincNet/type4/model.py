# model.py (DAC-specialized: Volterra basis + feature decoupling + full FFT)
#
# Key ideas:
# 1) Explicitly expose DAC memoryless nonlinearity via Volterra-like polynomial basis: x, x^3, x^5.
# 2) Keep identity / device-discriminative embedding (feat_id) separate from DAC embedding (feat_dac),
#    so supervised contrastive + prototype losses act on feat_id (domain-invariant), while DAC head
#    learns DAC severity from feat_dac.
# 3) Use full-spectrum complex FFT (fftshift) to capture spectral regrowth caused by nonlinearity.

import math
import numpy as np
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _hz2mel(hz: float) -> float:
    return 2595.0 * math.log10(1.0 + hz / 700.0)


def _mel2hz(mel: float) -> float:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def _pick_gn_groups(ch: int) -> int:
    for g in (8, 4, 2, 1):
        if ch % g == 0:
            return g
    return 1


def _safe_div(num: torch.Tensor, den: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    den2 = torch.where(den.abs() < eps, torch.ones_like(den), den)
    return num / den2


class SincConv1d(nn.Module):
    """
    Hz-based, vectorized, symmetric grid SincConv.
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
    ):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd for symmetric sinc filters.")

        self.out_channels = int(out_channels)
        self.kernel_size = int(kernel_size)
        self.sample_rate = float(sample_rate)
        self.min_low_hz = float(min_low_hz)
        self.min_band_hz = float(min_band_hz)

        low_hz = 30.0
        nyq = self.sample_rate / 2.0
        high_hz = nyq - (low_hz + self.min_band_hz + 1.0)
        if high_hz <= low_hz:
            raise ValueError("sample_rate too low or min_band_hz too large.")

        mel_low = _hz2mel(low_hz)
        mel_high = _hz2mel(high_hz)
        mel_points = np.linspace(mel_low, mel_high, self.out_channels + 1)
        hz_points = np.array([_mel2hz(m) for m in mel_points], dtype=np.float32)

        low_init = hz_points[:-1]
        band_init = np.diff(hz_points)

        self.low_hz_ = nn.Parameter(torch.tensor(low_init).view(-1, 1))   # (C,1)
        self.band_hz_ = nn.Parameter(torch.tensor(band_init).view(-1, 1)) # (C,1)

        # symmetric grid
        n = torch.arange(self.kernel_size, dtype=torch.float32) - (self.kernel_size - 1) / 2.0
        t = n / self.sample_rate  # seconds

        # Hamming window
        window = 0.54 - 0.46 * torch.cos(
            2 * math.pi * (torch.arange(self.kernel_size) / (self.kernel_size - 1))
        )

        self.register_buffer("t_", t.view(1, -1))              # (1,K)
        self.register_buffer("window_", window.view(1, -1))    # (1,K)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        dtype = x.dtype

        t = self.t_.to(device=device, dtype=dtype)            # (1,K)
        window = self.window_.to(device=device, dtype=dtype)  # (1,K)

        nyq = self.sample_rate / 2.0

        low = self.min_low_hz + torch.abs(self.low_hz_)       # (C,1)
        band = self.min_band_hz + torch.abs(self.band_hz_)    # (C,1)

        low = torch.clamp(low, min=self.min_low_hz, max=nyq - self.min_band_hz - 1.0)

        min_high = low + self.min_band_hz                      # (C,1)
        max_high = torch.full_like(low, nyq - 1.0)             # (C,1)
        high = torch.clamp(low + band, min=min_high, max=max_high)

        f1 = low.to(device=device, dtype=dtype)
        f2 = high.to(device=device, dtype=dtype)

        # band-pass sinc
        num = torch.sin(2.0 * math.pi * f2 * t) - torch.sin(2.0 * math.pi * f1 * t)
        den = math.pi * t
        bp = _safe_div(num, den, eps=1e-12)                    # (C,K)

        # center tap
        center = self.kernel_size // 2
        bp[:, center] = (2.0 * (f2 - f1)).squeeze(1)

        # window + normalize
        bp = bp * window
        bp = bp / (bp.abs().amax(dim=1, keepdim=True) + 1e-8)

        filters = bp.view(self.out_channels, 1, self.kernel_size)  # (C,1,K)
        return F.conv1d(x, filters, stride=1, padding=self.kernel_size // 2, bias=None)


class HighFreqEmphasis(nn.Module):
    """
    Fixed 1st/2nd difference on IQ to emphasize DAC-induced high frequency ripples.
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
        w1 = self.k1.repeat(2, 1, 1)  # (2,1,2)
        w2 = self.k2.repeat(2, 1, 1)  # (2,1,3)

        d1 = F.conv1d(x, w1, padding=1, groups=2)  # (B,2,L+1)
        d1 = d1[..., :L]                           # -> (B,2,L)
        d2 = F.conv1d(x, w2, padding=1, groups=2)  # (B,2,L)

        return torch.cat([d1, d2], dim=1)          # (B,4,L)


class ConvBlock1d(nn.Module):
    def __init__(self, cin: int, cout: int, k: int = 5, pool: int = 2, drop: float = 0.1):
        super().__init__()
        pad = k // 2
        gn = _pick_gn_groups(cout)

        layers = [
            nn.Conv1d(cin, cout, kernel_size=k, padding=pad, bias=False),
            nn.GroupNorm(gn, cout),
            nn.ReLU(inplace=True),
        ]
        if pool is not None and pool > 1:
            layers.append(nn.MaxPool1d(pool))
        if drop and drop > 0:
            layers.append(nn.Dropout(drop))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CVSincNet(nn.Module):
    """
    DAC-specialized CV-SincNet.

    forward(x, return_aux=False):
      - return_aux=False -> logits
      - return_aux=True  -> logits, dac_pred, feat_id

    Notes:
      - feat_id is the embedding for classification / contrastive / prototype.
      - dac_pred is predicted DAC strength in [0,1] from a separate DAC embedding path.
    """

    def __init__(
        self,
        num_classes: int = 16,
        sample_rate: float = 5e6,
        sinc_out: int = 64,
        sinc_kernel: int = 129,
        emb_dim: int = 256,
        drop: float = 0.25,
        freq_bins_keep: Optional[int] = None,   # None -> keep all bins after FFT
        use_full_fft: bool = True,              # True -> fftshift + full spectrum
        use_volterra: bool = True,              # True -> x, x^3, x^5 basis
        volterra_clip: float = 1.2,             # clamp before power
    ):
        super().__init__()
        self.use_full_fft = bool(use_full_fft)
        self.use_volterra = bool(use_volterra)
        self.volterra_clip = float(volterra_clip)

        # time branch
        self.sinc = SincConv1d(sinc_out, sinc_kernel, sample_rate=sample_rate)
        self.hf = HighFreqEmphasis()

        if self.use_volterra:
            # 6 channels: I,Q, I^3,Q^3, I^5,Q^5
            time_in = 6 * sinc_out + 4
        else:
            time_in = 2 * sinc_out + 4

        gn = _pick_gn_groups(time_in)
        self.time_fuse = nn.Sequential(
            nn.Conv1d(time_in, time_in, kernel_size=1, bias=False),
            nn.GroupNorm(gn, time_in),
            nn.ReLU(inplace=True),
        )

        self.t1 = ConvBlock1d(time_in, 128, k=5, pool=2, drop=0.10)
        self.t2 = ConvBlock1d(128, 256, k=5, pool=2, drop=0.10)
        self.t3 = ConvBlock1d(256, 256, k=3, pool=1, drop=0.10)
        self.t_pool = nn.AdaptiveAvgPool1d(1)
        self.t_proj = nn.Linear(256, emb_dim)

        # freq branch
        self.freq_bins_keep = freq_bins_keep
        self.f1 = ConvBlock1d(1, 64, k=5, pool=2, drop=0.05)
        self.f2 = ConvBlock1d(64, 128, k=5, pool=2, drop=0.05)
        self.f3 = ConvBlock1d(128, 128, k=3, pool=1, drop=0.05)
        self.f_pool = nn.AdaptiveAvgPool1d(1)
        self.f_proj = nn.Linear(128, emb_dim)

        # shared base embedding (before decoupling)
        self.base_fuse = nn.Sequential(
            nn.Linear(emb_dim * 2, emb_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(drop),
        )

        # decoupled projections
        self.id_proj = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(drop * 0.5),
        )
        self.dac_proj = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(drop * 0.5),
        )

        # heads
        self.cls_head = nn.Linear(emb_dim, num_classes)
        self.dac_head = nn.Sequential(
            nn.Linear(emb_dim, emb_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(emb_dim // 2, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="linear")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @staticmethod
    def _to_complex(iq: torch.Tensor) -> torch.Tensor:
        return torch.complex(iq[:, 0, :], iq[:, 1, :])  # (B,L) complex

    def _volterra_stack(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input:  x (B,2,L)
        Output: v (B,6,L) = [I,Q, I^3,Q^3, I^5,Q^5]
        """
        x0 = torch.clamp(x, -self.volterra_clip, self.volterra_clip)
        i = x0[:, 0:1, :]
        q = x0[:, 1:2, :]
        i3 = i ** 3
        q3 = q ** 3
        i5 = i ** 5
        q5 = q ** 5
        return torch.cat([i, q, i3, q3, i5, q5], dim=1)

    def _sinc_multi(self, v: torch.Tensor) -> torch.Tensor:
        """
        v: (B,C,L)
        return: (B, C*sinc_out, L)
        """
        B, C, L = v.shape
        v2 = v.reshape(B * C, 1, L)
        y = self.sinc(v2)  # (B*C, sinc_out, L)
        y = y.reshape(B, C * y.size(1), L)
        return y

    def _fft_mag(self, x: torch.Tensor) -> torch.Tensor:
        """
        Complex FFT magnitude (log1p), optionally fftshift and keep centered bins.
        Return: (B,F)
        """
        z = self._to_complex(x)                   # (B,L) complex
        spec = torch.fft.fft(z, dim=-1)           # (B,L) complex

        if self.use_full_fft:
            spec = torch.fft.fftshift(spec, dim=-1)

        mag = torch.log1p(torch.abs(spec))        # (B,L)

        if self.freq_bins_keep is not None and mag.size(-1) > int(self.freq_bins_keep):
            keep = int(self.freq_bins_keep)
            if self.use_full_fft:
                center = mag.size(-1) // 2
                start = center - keep // 2
                end = start + keep
                start = max(0, start)
                end = min(mag.size(-1), end)
                mag = mag[:, start:end]
            else:
                mag = mag[:, :keep]
        return mag

    def forward(self, x: torch.Tensor, return_aux: bool = False):
        # ---- time ----
        if self.use_volterra:
            v = self._volterra_stack(x)              # (B,6,L)
            sinc_cat = self._sinc_multi(v)           # (B,6*sinc_out,L)
        else:
            v = x
            sinc_cat = self._sinc_multi(v)           # (B,2*sinc_out,L)

        hf = self.hf(x)                               # (B,4,L)
        t = torch.cat([sinc_cat, hf], dim=1)          # (B,time_in,L)
        t = self.time_fuse(t)

        t = self.t1(t)
        t = self.t2(t)
        t = self.t3(t)
        t = self.t_pool(t).squeeze(-1)                # (B,256)
        t_emb = self.t_proj(t)                        # (B,emb_dim)

        # ---- freq ----
        mag = self._fft_mag(x)                         # (B,F)
        f = mag.unsqueeze(1)                           # (B,1,F)
        f = self.f1(f)
        f = self.f2(f)
        f = self.f3(f)
        f = self.f_pool(f).squeeze(-1)                 # (B,128)
        f_emb = self.f_proj(f)                         # (B,emb_dim)

        # ---- base / decoupling ----
        base = self.base_fuse(torch.cat([t_emb, f_emb], dim=1))  # (B,emb_dim)
        feat_id = self.id_proj(base)                              # (B,emb_dim)
        feat_dac = self.dac_proj(base)                            # (B,emb_dim)

        logits = self.cls_head(feat_id)

        if not return_aux:
            return logits

        dac_pred = torch.sigmoid(self.dac_head(feat_dac).squeeze(-1))  # (B,) in [0,1]
        return logits, dac_pred, feat_id
