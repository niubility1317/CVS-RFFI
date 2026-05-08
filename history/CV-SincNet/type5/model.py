# model.py
# DAC-specialized CV-SincNet with:
# - Time branch: SincConv + HighFreqEmphasis + (optional) Volterra basis (I,Q,I^3,Q^3,I^5,Q^5)
# - Freq branch (optimized): mirror-aware compressed spectral features:
#       logP_pos, logP_neg, log(P_pos/P_neg) over K sub-bands + circularity coefficient rho
#   This reduces compute (shorter sequence for CNN) and explicitly captures mirror frequencies.
# - Feature decoupling: feat_id (for classification/contrastive) and feat_dac (for DAC strength)

import math
import numpy as np
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


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

        self.low_hz_ = nn.Parameter(torch.tensor(low_init).view(-1, 1))    # (C,1)
        self.band_hz_ = nn.Parameter(torch.tensor(band_init).view(-1, 1))  # (C,1)

        n = torch.arange(self.kernel_size, dtype=torch.float32) - (self.kernel_size - 1) / 2.0
        t = n / self.sample_rate  # seconds

        window = 0.54 - 0.46 * torch.cos(
            2 * math.pi * (torch.arange(self.kernel_size) / (self.kernel_size - 1))
        )

        self.register_buffer("t_", t.view(1, -1))                # (1,K)
        self.register_buffer("window_", window.view(1, -1))      # (1,K)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        dtype = x.dtype
        t = self.t_.to(device=device, dtype=dtype)
        window = self.window_.to(device=device, dtype=dtype)

        nyq = self.sample_rate / 2.0
        low = self.min_low_hz + torch.abs(self.low_hz_)         # (C,1)
        band = self.min_band_hz + torch.abs(self.band_hz_)      # (C,1)

        low = torch.clamp(low, min=self.min_low_hz, max=nyq - self.min_band_hz - 1.0)
        min_high = low + self.min_band_hz
        max_high = torch.full_like(low, nyq - 1.0)
        high = torch.clamp(low + band, min=min_high, max=max_high)

        f1 = low.to(device=device, dtype=dtype)
        f2 = high.to(device=device, dtype=dtype)

        # band-pass sinc = sinc(2 f2 t) - sinc(2 f1 t)
        num = torch.sin(2.0 * math.pi * f2 * t) - torch.sin(2.0 * math.pi * f1 * t)
        den = math.pi * t
        bp = _safe_div(num, den, eps=1e-12)                     # (C,K)

        # center tap to avoid 0/0
        center = self.kernel_size // 2
        bp[:, center] = (2.0 * (f2 - f1)).squeeze(1)

        bp = bp * window
        bp = bp / (bp.abs().amax(dim=1, keepdim=True) + 1e-8)

        filters = bp.view(self.out_channels, 1, self.kernel_size)  # (C,1,K)
        return F.conv1d(x, filters, stride=1, padding=self.kernel_size // 2, bias=None)


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
        w1 = self.k1.repeat(2, 1, 1)  # (2,1,2)
        w2 = self.k2.repeat(2, 1, 1)  # (2,1,3)

        d1 = F.conv1d(x, w1, padding=1, groups=2)
        d1 = d1[..., :L]              # keep length L
        d2 = F.conv1d(x, w2, padding=1, groups=2)

        return torch.cat([d1, d2], dim=1)  # (B,4,L)


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
    forward(x, return_aux=False):
      - return_aux=False -> logits
      - return_aux=True  -> logits, dac_pred, feat_id

    Notes:
      - feat_id: embedding used for classification / contrastive / prototype
      - dac_pred: predicted DAC strength in [0,1] from a separate DAC embedding path
    """

    def __init__(
        self,
        num_classes: int = 16,
        sample_rate: float = 5e6,
        sinc_out: int = 64,
        sinc_kernel: int = 129,
        emb_dim: int = 256,
        drop: float = 0.25,

        # -------- freq branch optimization --------
        freq_bands: int = 64,                 # K: compress spectrum into K sub-bands
        use_rfft_pair: bool = True,           # compute full spectrum via rfft(I) + rfft(Q) reconstruction
        use_circularity: bool = True,         # add rho = |E[z^2]| / E[|z|^2] as an extra feature
        eps: float = 1e-8,

        # -------- DAC specialization --------
        use_volterra: bool = True,            # use (I,Q,I^3,Q^3,I^5,Q^5)
        volterra_clip: float = 1.2,
    ):
        super().__init__()
        self.use_rfft_pair = bool(use_rfft_pair)
        self.freq_bands = int(freq_bands)
        self.use_circularity = bool(use_circularity)
        self.eps = float(eps)

        self.use_volterra = bool(use_volterra)
        self.volterra_clip = float(volterra_clip)

        # ---------------- time branch ----------------
        self.sinc = SincConv1d(sinc_out, sinc_kernel, sample_rate=sample_rate)
        self.hf = HighFreqEmphasis()

        if self.use_volterra:
            # 6 channels -> each passes through sinc_out filters, plus 4 HF channels
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

        # ---------------- freq branch (mirror-aware, compressed) ----------------
        # Input to freq CNN becomes (B, 3, K) where 3=[logP_pos, logP_neg, logR]
        # Much shorter sequence length (K) reduces conv cost dramatically.
        self.f1 = ConvBlock1d(3, 32, k=5, pool=2, drop=0.05)
        self.f2 = ConvBlock1d(32, 64, k=5, pool=2, drop=0.05)
        self.f3 = ConvBlock1d(64, 64, k=3, pool=1, drop=0.05)
        self.f_pool = nn.AdaptiveAvgPool1d(1)
        self.f_proj = nn.Linear(64, emb_dim)

        # ---------------- fusion + decoupling ----------------
        fuse_in = emb_dim * 2 + (1 if self.use_circularity else 0)
        self.base_fuse = nn.Sequential(
            nn.Linear(fuse_in, emb_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(drop),
        )

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

    # ---------- helpers ----------
    @staticmethod
    def _to_complex(iq: torch.Tensor) -> torch.Tensor:
        return torch.complex(iq[:, 0, :], iq[:, 1, :])  # (B,L) complex

    def _volterra_stack(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input:  x (B,2,L)
        Output: v (B,6,L) = [I,Q,I^3,Q^3,I^5,Q^5]
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

    def _fft_full_via_rfft_pair(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct FFT(z) for z=I+jQ using rfft(I) and rfft(Q).
        x: (B,2,L), L should be even for clean Nyquist handling.
        return: spec (B,L) complex
        """
        I = x[:, 0, :]  # (B,L) real
        Q = x[:, 1, :]

        Fi = torch.fft.rfft(I, dim=-1)  # (B, L//2+1)
        Fq = torch.fft.rfft(Q, dim=-1)  # (B, L//2+1)

        # Reconstruct full real FFTs using conjugate symmetry:
        # Full length L: [0..L/2] + conj([L/2-1..1])
        if Fi.size(-1) <= 2:
            Fi_full = torch.fft.fft(I, dim=-1)
            Fq_full = torch.fft.fft(Q, dim=-1)
        else:
            tail_i = torch.conj(Fi[..., 1:-1]).flip(dims=[-1])
            tail_q = torch.conj(Fq[..., 1:-1]).flip(dims=[-1])
            Fi_full = torch.cat([Fi, tail_i], dim=-1)  # (B,L)
            Fq_full = torch.cat([Fq, tail_q], dim=-1)  # (B,L)

        spec = Fi_full + 1j * Fq_full
        return spec

    def _mirror_compressed_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Build mirror-aware compressed spectral features:
          - pos power bins: k=1..L/2-1
          - neg power bins: k=L-1..L/2+1 (reversed to align with pos)
        Then adaptive average pool to K bands.
        Return:
          feat_f: (B, 3, K) channels=[logP_pos, logP_neg, logR]
          rho: (B,1) if use_circularity else None
        """
        B, _, L = x.shape
        eps = self.eps

        # FFT of complex z
        if self.use_rfft_pair and (L % 2 == 0):
            spec = self._fft_full_via_rfft_pair(x)   # (B,L) complex
        else:
            z = self._to_complex(x)
            spec = torch.fft.fft(z, dim=-1)

        # power spectrum
        power = spec.real * spec.real + spec.imag * spec.imag  # (B,L)

        # define positive and negative frequency bins (exclude DC and Nyquist)
        half = L // 2
        if half <= 1:
            # degenerate
            pos = power[:, :1]
            neg = power[:, :1]
        else:
            # pos bins: 1..half-1 length=half-1
            pos = power[:, 1:half]
            # neg bins: L-1..half+1 length=half-1 -> take power[:, half+1:] then reverse
            neg = power[:, half + 1:]
            neg = torch.flip(neg, dims=[-1])  # align increasing |f| with pos

            # if L odd, shapes can differ; guard by min length
            m = min(pos.size(-1), neg.size(-1))
            pos = pos[:, :m]
            neg = neg[:, :m]

        # mirror-aware channels
        logP_pos = torch.log1p(pos)
        logP_neg = torch.log1p(neg)
        logR = torch.log((pos + eps) / (neg + eps))

        # compress into K sub-bands using adaptive average pooling
        K = max(4, self.freq_bands)
        def pool1d(v: torch.Tensor) -> torch.Tensor:
            # v: (B,F) -> (B,K)
            v = v.unsqueeze(1)  # (B,1,F)
            v = F.adaptive_avg_pool1d(v, K)  # (B,1,K)
            return v.squeeze(1)

        p_pos = pool1d(logP_pos)
        p_neg = pool1d(logP_neg)
        p_rat = pool1d(logR)

        feat_f = torch.stack([p_pos, p_neg, p_rat], dim=1)  # (B,3,K)

        rho = None
        if self.use_circularity:
            # circularity / improperness coefficient:
            # rho = |E[z^2]| / (E[|z|^2] + eps)
            z = self._to_complex(x)
            Ez2 = torch.mean(z * z, dim=-1)                    # (B,) complex
            Eabs2 = torch.mean(z.real * z.real + z.imag * z.imag, dim=-1)  # (B,) real
            rho_val = torch.abs(Ez2) / (Eabs2 + eps)
            rho = rho_val.unsqueeze(1)  # (B,1)

        return feat_f, rho

    # ---------- forward ----------
    def forward(self, x: torch.Tensor, return_aux: bool = False):
        # ----- time branch -----
        if self.use_volterra:
            v = self._volterra_stack(x)        # (B,6,L)
            sinc_cat = self._sinc_multi(v)     # (B,6*sinc_out,L)
        else:
            sinc_cat = self._sinc_multi(x)     # (B,2*sinc_out,L)

        hf = self.hf(x)                        # (B,4,L)
        t = torch.cat([sinc_cat, hf], dim=1)   # (B,time_in,L)
        t = self.time_fuse(t)

        t = self.t1(t)
        t = self.t2(t)
        t = self.t3(t)
        t = self.t_pool(t).squeeze(-1)         # (B,256)
        t_emb = self.t_proj(t)                 # (B,emb_dim)

        # ----- freq branch (compressed + mirror-aware) -----
        feat_f, rho = self._mirror_compressed_features(x)  # feat_f: (B,3,K), rho:(B,1) or None

        f = self.f1(feat_f)
        f = self.f2(f)
        f = self.f3(f)
        f = self.f_pool(f).squeeze(-1)         # (B,64)
        f_emb = self.f_proj(f)                 # (B,emb_dim)

        # ----- fuse -----
        if rho is not None:
            base_in = torch.cat([t_emb, f_emb, rho], dim=1)   # (B, 2*emb_dim+1)
        else:
            base_in = torch.cat([t_emb, f_emb], dim=1)        # (B, 2*emb_dim)

        base = self.base_fuse(base_in)                         # (B,emb_dim)

        # decouple
        feat_id = self.id_proj(base)                           # (B,emb_dim)
        feat_dac = self.dac_proj(base)                         # (B,emb_dim)

        logits = self.cls_head(feat_id)

        if not return_aux:
            return logits

        dac_pred = torch.sigmoid(self.dac_head(feat_dac).squeeze(-1))  # (B,)
        return logits, dac_pred, feat_id
