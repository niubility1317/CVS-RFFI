# model.py
# ------------------------------------------------------------
# DAC-specialized CV-SincNet (Lite + DAC-aware classifier)
#
# Key changes (per your requirements):
# 1) Reduce compute in time-branch:
#    - SincConv runs ONLY on raw IQ once (2*sinc_out), NOT on 6-way Volterra inputs
#    - DAC nonlinearity is captured AFTER filterbank using z|z|^2 (and optional z|z|^4)
#    - time_fuse is a bottleneck: Conv1d(time_in -> Cbottleneck, 1) where Cbottleneck=96/128
#    - early downsample by AvgPool1d(2)
#    - t1/t2/t3 designed from bottleneck (and use depthwise-separable conv)
#
# 2) Explicitly feed DAC pathway into classifier:
#    - classifier builds feat_cls and feat_dac from fused base
#    - classifier forms joint embedding from [feat_cls, feat_dac] (concat + joint_proj)
#    - optional FiLM-style gating using feat_dac
#
# 3) Projection head decoupling:
#    - feat_con is produced by con_proj(base) and is intended ONLY for SupCon/Proto
#    - classification uses feat_cls (and DAC-aware joint embedding) only
#
# 4) Margin-based softmax (CosFace / AM-Softmax):
#    - If labels are provided to model(..., y=labels), applies margin during forward.
#    - If labels=None, returns standard cosine logits (scaled).
#
# 5) Freq-branch DAC specialization:
#    - mirror-aware compressed features now include asymmetry channel:
#      asym = |pos-neg|/(pos+neg)
#
# ------------------------------------------------------------
# Quick training "fast tune" suggestions (in train.py):
# - Use cosine LR + warmup (instead of StepLR)
# - Reduce batch_size (e.g. 128/192) if you chase 99%+
# - Longer ramp: WARMUP_EPOCHS=15~20, RAMP_EPOCHS=60~100
# - Lower DAC_LAMBDA initially (1~2) then ramp
# ------------------------------------------------------------

import math
import numpy as np
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


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
        t = n / self.sample_rate

        window = 0.54 - 0.46 * torch.cos(
            2 * math.pi * (torch.arange(self.kernel_size) / (self.kernel_size - 1))
        )

        self.register_buffer("t_", t.view(1, -1))
        self.register_buffer("window_", window.view(1, -1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        dtype = x.dtype
        t = self.t_.to(device=device, dtype=dtype)
        window = self.window_.to(device=device, dtype=dtype)

        nyq = self.sample_rate / 2.0
        low = self.min_low_hz + torch.abs(self.low_hz_)    # (C,1)
        band = self.min_band_hz + torch.abs(self.band_hz_) # (C,1)

        low = torch.clamp(low, min=self.min_low_hz, max=nyq - self.min_band_hz - 1.0)
        min_high = low + self.min_band_hz
        max_high = torch.full_like(low, nyq - 1.0)
        high = torch.clamp(low + band, min=min_high, max=max_high)

        f1 = low.to(device=device, dtype=dtype)
        f2 = high.to(device=device, dtype=dtype)

        num = torch.sin(2.0 * math.pi * f2 * t) - torch.sin(2.0 * math.pi * f1 * t)
        den = math.pi * t
        bp = _safe_div(num, den, eps=1e-12)  # (C,K)

        center = self.kernel_size // 2
        bp[:, center] = (2.0 * (f2 - f1)).squeeze(1)

        bp = bp * window
        bp = bp / (bp.abs().amax(dim=1, keepdim=True) + 1e-8)

        filters = bp.view(self.out_channels, 1, self.kernel_size)
        return F.conv1d(x, filters, stride=1, padding=self.kernel_size // 2, bias=None)


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
        w1 = self.k1.repeat(2, 1, 1)  # (2,1,2)
        w2 = self.k2.repeat(2, 1, 1)  # (2,1,3)

        d1 = F.conv1d(x, w1, padding=1, groups=2)
        d1 = d1[..., :L]
        d2 = F.conv1d(x, w2, padding=1, groups=2)

        return torch.cat([d1, d2], dim=1)  # (B,4,L)


# ----------------------- Depthwise-separable ConvBlock1d -----------------------
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


# ----------------------- CosFace (AM-Softmax) head -----------------------
class CosFaceHead(nn.Module):
    """
    Margin-based softmax on cosine similarity (CosFace / AM-Softmax):
      logits = s * (cos(theta) - m * one_hot)
    If labels is None: logits = s * cos(theta)
    """
    def __init__(self, in_features: int, out_features: int, s: float = 30.0, m: float = 0.35):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.s = float(s)
        self.m = float(m)
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.01)

    def forward(self, x: torch.Tensor, labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = F.normalize(x, dim=1)
        w = F.normalize(self.weight, dim=1)
        cos = F.linear(x, w)  # (B,C)

        if labels is None:
            return self.s * cos

        labels = labels.view(-1).long()
        one_hot = torch.zeros_like(cos)
        one_hot.scatter_(1, labels.unsqueeze(1), 1.0)
        cos_m = cos - one_hot * self.m
        return self.s * cos_m


# ----------------------- DAC-aware classifier block -----------------------
class DACAwareClassifier(nn.Module):
    """
    From base embedding -> feat_cls & feat_dac -> joint embedding -> margin head.
    Also outputs dac_pred for regression.
    """
    def __init__(
        self,
        base_dim: int,
        emb_dim: int,
        num_classes: int,
        drop: float = 0.25,
        margin_s: float = 30.0,
        margin_m: float = 0.35,
        gate_alpha: float = 0.5,  # strength of FiLM-like gating
        use_gate: bool = True,
    ):
        super().__init__()
        self.use_gate = bool(use_gate)
        self.gate_alpha = float(gate_alpha)

        self.cls_proj = nn.Sequential(
            nn.Linear(base_dim, emb_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(drop * 0.5),
        )
        self.dac_proj = nn.Sequential(
            nn.Linear(base_dim, emb_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(drop * 0.5),
        )

        if self.use_gate:
            self.dac_gate = nn.Sequential(
                nn.Linear(emb_dim, emb_dim),
                nn.Sigmoid(),
            )
        else:
            self.dac_gate = None

        # Explicit DAC->classification fusion
        self.joint_proj = nn.Sequential(
            nn.Linear(emb_dim * 2, emb_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(drop * 0.5),
        )

        self.head = CosFaceHead(emb_dim, num_classes, s=margin_s, m=margin_m)

        self.dac_head = nn.Sequential(
            nn.Linear(emb_dim, max(8, emb_dim // 2)),
            nn.ReLU(inplace=True),
            nn.Linear(max(8, emb_dim // 2), 1),
        )

    def forward(
        self,
        base: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        return_emb: bool = False,
    ):
        feat_cls = self.cls_proj(base)  # (B,emb_dim)
        feat_dac = self.dac_proj(base)  # (B,emb_dim)

        if self.use_gate and self.dac_gate is not None:
            g = self.dac_gate(feat_dac)
            feat_cls = feat_cls * (1.0 + self.gate_alpha * g)

        feat_joint = self.joint_proj(torch.cat([feat_cls, feat_dac], dim=1))  # (B,emb_dim)
        logits = self.head(feat_joint, labels=labels)

        dac_pred = torch.sigmoid(self.dac_head(feat_dac).squeeze(-1))  # (B,)

        if not return_emb:
            return logits, dac_pred

        return logits, dac_pred, feat_cls, feat_dac, feat_joint


# ----------------------- Main Model -----------------------
class CVSincNet(nn.Module):
    """
    Input: x (B,2,L)

    Forward usage:
      - logits = model(x)  # inference
      - logits, dac_pred, feat_cls, feat_con = model(x, y=labels, return_aux=True)

    Notes:
      - feat_cls: classification embedding (used by CE / margin-softmax)
      - feat_con: projection embedding (use ONLY for SupCon / Proto)
      - classifier is DAC-aware: logits are computed from joint([feat_cls, feat_dac])
    """

    def __init__(
        self,
        num_classes: int = 16,
        sample_rate: float = 5e6,

        # Time branch
        sinc_out: int = 48,               # smaller than 64 for speed; try 48 first
        sinc_kernel: int = 79,            # smaller than 129 for speed; try 79/63
        time_bottleneck: int = 96,        # REQUIRED: 96 or 128

        # Embedding dims
        emb_dim: int = 256,
        drop: float = 0.25,

        # Freq branch
        freq_bands: int = 32,
        use_rfft_pair: bool = True,
        use_circularity: bool = True,
        eps: float = 1e-8,

        # DAC specialization (time)
        use_nonlinear_basis: bool = True,     # use z|z|^2 on filterbank outputs
        include_z_abs4: bool = False,         # optional z|z|^4 (more compute)
        nl_clip: float = 1.2,

        # Margin head
        margin_s: float = 30.0,
        margin_m: float = 0.35,
    ):
        super().__init__()
        self.sample_rate = float(sample_rate)

        self.freq_bands = int(freq_bands)
        self.use_rfft_pair = bool(use_rfft_pair)
        self.use_circularity = bool(use_circularity)
        self.eps = float(eps)

        self.use_nonlinear_basis = bool(use_nonlinear_basis)
        self.include_z_abs4 = bool(include_z_abs4)
        self.nl_clip = float(nl_clip)

        # -------- time branch --------
        self.sinc = SincConv1d(sinc_out, sinc_kernel, sample_rate=sample_rate)
        self.hf = HighFreqEmphasis()

        # Build time features:
        # - Sinc on IQ: 2*sinc_out
        # - Nonlinear basis after filterbank: z|z|^2 -> adds another 2*sinc_out
        #   (optional z|z|^4 adds another 2*sinc_out)
        # - HF: 4
        time_in = 2 * sinc_out + 4
        if self.use_nonlinear_basis:
            time_in += 2 * sinc_out
        if self.use_nonlinear_basis and self.include_z_abs4:
            time_in += 2 * sinc_out

        # Bottleneck fuse (your requirement)
        Cb = int(time_bottleneck)
        gn = _pick_gn_groups(Cb)
        self.time_fuse = nn.Sequential(
            nn.Conv1d(time_in, Cb, kernel_size=1, bias=False),
            nn.GroupNorm(gn, Cb),
            nn.ReLU(inplace=True),
        )

        # Early downsample to cut compute ~x2
        self.time_down = nn.AvgPool1d(2)

        # t1/t2/t3 designed from bottleneck
        self.t1 = DSConvBlock1d(Cb, 128, k=5, pool=2, drop=0.10)
        self.t2 = DSConvBlock1d(128, 192, k=5, pool=2, drop=0.10)
        self.t3 = DSConvBlock1d(192, 192, k=3, pool=1, drop=0.10)
        self.t_pool = nn.AdaptiveAvgPool1d(1)
        self.t_proj = nn.Linear(192, emb_dim)

        # -------- freq branch (mirror-aware compressed) --------
        # channels: [logP_pos, logP_neg, logR, asym]
        self.f1 = DSConvBlock1d(4, 32, k=5, pool=2, drop=0.05)
        self.f2 = DSConvBlock1d(32, 64, k=5, pool=2, drop=0.05)
        self.f3 = DSConvBlock1d(64, 64, k=3, pool=1, drop=0.05)
        self.f_pool = nn.AdaptiveAvgPool1d(1)
        self.f_proj = nn.Linear(64, emb_dim)

        # -------- fuse (for eval_and_explain compatibility: name it "fuse") --------
        fuse_in = emb_dim * 2 + (1 if self.use_circularity else 0)
        self.fuse = nn.Sequential(
            nn.Linear(fuse_in, emb_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(drop),
        )

        # -------- projection head (ONLY for SupCon/Proto) --------
        self.con_proj = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(drop * 0.25),
        )

        # -------- DAC-aware classifier --------
        self.cls_head = DACAwareClassifier(
            base_dim=emb_dim,
            emb_dim=emb_dim,
            num_classes=num_classes,
            drop=drop,
            margin_s=margin_s,
            margin_m=margin_m,
            gate_alpha=0.5,
            use_gate=True,
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

    def _sinc_on_iq(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B,2,L)
        return: (B, 2*sinc_out, L) = [Sinc(I), Sinc(Q)]
        """
        B, _, L = x.shape
        i = x[:, 0:1, :]
        q = x[:, 1:2, :]
        yi = self.sinc(i)  # (B,S,L)
        yq = self.sinc(q)  # (B,S,L)
        return torch.cat([yi, yq], dim=1)

    def _nonlinear_basis_after_filterbank(self, sinc_iq: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        sinc_iq: (B,2*S,L) -> reshape to (B,2,S,L)
        Return:
          z|z|^2: (B,2*S,L)
          z|z|^4: (B,2*S,L) optional
        """
        B, C, L = sinc_iq.shape
        S = C // 2
        y = sinc_iq.view(B, 2, S, L)  # (B,2,S,L)
        i = y[:, 0, :, :]             # (B,S,L)
        q = y[:, 1, :, :]

        # clip for stability
        i = torch.clamp(i, -self.nl_clip, self.nl_clip)
        q = torch.clamp(q, -self.nl_clip, self.nl_clip)

        a2 = i * i + q * q
        i3 = i * a2
        q3 = q * a2
        z_abs2 = torch.cat([i3, q3], dim=1)  # (B,2S,L)

        z_abs4 = None
        if self.include_z_abs4:
            a4 = a2 * a2
            i5 = i * a4
            q5 = q * a4
            z_abs4 = torch.cat([i5, q5], dim=1)  # (B,2S,L)

        return z_abs2, z_abs4

    def _fft_full_via_rfft_pair(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct FFT(z) for z=I+jQ using rfft(I) and rfft(Q).
        x: (B,2,L), L even recommended.
        return: spec (B,L) complex
        """
        I = x[:, 0, :]
        Q = x[:, 1, :]

        Fi = torch.fft.rfft(I, dim=-1)  # (B, L//2+1)
        Fq = torch.fft.rfft(Q, dim=-1)

        if Fi.size(-1) <= 2:
            Fi_full = torch.fft.fft(I, dim=-1)
            Fq_full = torch.fft.fft(Q, dim=-1)
        else:
            tail_i = torch.conj(Fi[..., 1:-1]).flip(dims=[-1])
            tail_q = torch.conj(Fq[..., 1:-1]).flip(dims=[-1])
            Fi_full = torch.cat([Fi, tail_i], dim=-1)  # (B,L)
            Fq_full = torch.cat([Fq, tail_q], dim=-1)  # (B,L)

        return Fi_full + 1j * Fq_full

    def _mirror_compressed_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Mirror-aware compressed spectral features:
          pos bins (exclude DC/Nyquist), neg bins aligned, then pool to K bands.
        Return:
          feat_f: (B, 4, K) channels=[logP_pos, logP_neg, logR, asym]
          rho: (B,1) if use_circularity else None
        """
        B, _, L = x.shape
        eps = self.eps

        if self.use_rfft_pair and (L % 2 == 0):
            spec = self._fft_full_via_rfft_pair(x)
        else:
            spec = torch.fft.fft(self._to_complex(x), dim=-1)

        power = spec.real * spec.real + spec.imag * spec.imag  # (B,L)

        half = L // 2
        if half <= 1:
            pos = power[:, :1]
            neg = power[:, :1]
        else:
            pos = power[:, 1:half]         # 1..half-1
            neg = power[:, half + 1:]      # half+1..L-1
            neg = torch.flip(neg, dims=[-1])

            m = min(pos.size(-1), neg.size(-1))
            pos = pos[:, :m]
            neg = neg[:, :m]

        logP_pos = torch.log1p(pos)
        logP_neg = torch.log1p(neg)
        logR = torch.log((pos + eps) / (neg + eps))
        asym = torch.abs(pos - neg) / (pos + neg + eps)

        K = max(4, self.freq_bands)

        def pool1d(v: torch.Tensor) -> torch.Tensor:
            v = v.unsqueeze(1)  # (B,1,F)
            v = F.adaptive_avg_pool1d(v, K)
            return v.squeeze(1)

        p_pos = pool1d(logP_pos)
        p_neg = pool1d(logP_neg)
        p_rat = pool1d(logR)
        p_asym = pool1d(asym)

        feat_f = torch.stack([p_pos, p_neg, p_rat, p_asym], dim=1)  # (B,4,K)

        rho = None
        if self.use_circularity:
            z = self._to_complex(x)
            Ez2 = torch.mean(z * z, dim=-1)  # (B,) complex
            Eabs2 = torch.mean(z.real * z.real + z.imag * z.imag, dim=-1)
            rho = (torch.abs(Ez2) / (Eabs2 + eps)).unsqueeze(1)  # (B,1)

        return feat_f, rho

    # ---------- forward ----------
    def forward(
        self,
        x: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        return_aux: bool = False,
    ):
        """
        x: (B,2,L)
        y: (B,) optional labels. If provided, margin-based softmax is applied.
        return_aux:
          False -> logits
          True  -> logits, dac_pred, feat_cls, feat_con
        """
        # ----- time branch -----
        sinc_iq = self._sinc_on_iq(x)     # (B,2S,L)

        feats_t = [sinc_iq]
        if self.use_nonlinear_basis:
            z_abs2, z_abs4 = self._nonlinear_basis_after_filterbank(sinc_iq)
            feats_t.append(z_abs2)
            if self.include_z_abs4 and z_abs4 is not None:
                feats_t.append(z_abs4)

        hf = self.hf(x)                  # (B,4,L)
        feats_t.append(hf)

        t = torch.cat(feats_t, dim=1)    # (B,time_in,L)
        t = self.time_fuse(t)            # (B,Cb,L)
        t = self.time_down(t)            # (B,Cb,L/2)

        t = self.t1(t)
        t = self.t2(t)
        t = self.t3(t)
        t = self.t_pool(t).squeeze(-1)   # (B,192)
        t_emb = self.t_proj(t)           # (B,emb_dim)

        # ----- freq branch -----
        feat_f, rho = self._mirror_compressed_features(x)  # (B,4,K), (B,1) or None

        f = self.f1(feat_f)
        f = self.f2(f)
        f = self.f3(f)
        f = self.f_pool(f).squeeze(-1)   # (B,64)
        f_emb = self.f_proj(f)           # (B,emb_dim)

        # ----- fuse -----
        if rho is not None:
            base_in = torch.cat([t_emb, f_emb, rho], dim=1)
        else:
            base_in = torch.cat([t_emb, f_emb], dim=1)

        base = self.fuse(base_in)  # (B,emb_dim)

        # ----- projection head for SupCon/Proto (ONLY) -----
        feat_con = self.con_proj(base)   # (B,emb_dim)

        # ----- DAC-aware margin classifier -----
        logits, dac_pred, feat_cls, _feat_dac, _feat_joint = self.cls_head(
            base, labels=y, return_emb=True
        )

        if not return_aux:
            return logits

        return logits, dac_pred, feat_cls, feat_con
