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


def _finite_clamp(x: torch.Tensor, min_val: Optional[float] = None, max_val: Optional[float] = None) -> torch.Tensor:
    y = torch.nan_to_num(x.float(), nan=0.0, posinf=0.0, neginf=0.0)
    if min_val is not None or max_val is not None:
        y = torch.clamp(y, min=min_val, max=max_val)
    return y


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

        # Linear-Hz initialization is more suitable for RF fingerprinting than mel-scale
        # because the task is driven by hardware impairment physics rather than human hearing.
        hz_points = np.linspace(low_hz, high_hz, self.out_channels + 1, dtype=np.float32)

        low_init = hz_points[:-1]
        band_init = np.diff(hz_points)

        self.low_hz_ = nn.Parameter(torch.tensor(low_init).view(-1, 1))
        self.band_hz_ = nn.Parameter(torch.tensor(band_init).view(-1, 1))

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
        low = self.min_low_hz + torch.abs(self.low_hz_)
        band = self.min_band_hz + torch.abs(self.band_hz_)

        low = torch.clamp(low, min=self.min_low_hz, max=nyq - self.min_band_hz - 1.0)
        min_high = low + self.min_band_hz
        max_high = torch.full_like(low, nyq - 1.0)
        high = torch.clamp(low + band, min=min_high, max=max_high)

        f1 = low.to(device=device, dtype=dtype)
        f2 = high.to(device=device, dtype=dtype)
        # Stable band-pass construction:
        # 2*f*sinc(2*f*t) is numerically safer than sin(2*pi*f*t)/(pi*t),
        # especially near t=0 where gradients for the learnable cutoffs are sensitive.
        with torch.cuda.amp.autocast(enabled=False):
            t32 = t.float()
            f132 = f1.float()
            f232 = f2.float()
            window32 = window.float()
            bp = 2.0 * f232 * torch.sinc(2.0 * f232 * t32) - 2.0 * f132 * torch.sinc(2.0 * f132 * t32)
            bp = bp * window32
            bp = bp / (bp.abs().amax(dim=1, keepdim=True).clamp_min(1e-6))
            bp = torch.nan_to_num(bp, nan=0.0, posinf=0.0, neginf=0.0).to(dtype)
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
        with torch.cuda.amp.autocast(enabled=False):
            feat_f = torch.nan_to_num(feat_f.float(), nan=0.0, posinf=0.0, neginf=0.0)
            B, C, K = feat_f.shape
            x = feat_f.permute(0, 2, 1).contiguous()
            g = feat_f.mean(dim=2)
            g2 = g.unsqueeze(1).expand(-1, K, -1)
            logits = self.band_mlp(torch.cat([x, g2], dim=-1)).squeeze(-1)
            logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)
            logits = (logits / max(1e-4, self.temperature)).clamp(min=-20.0, max=20.0)
            logits = logits - logits.max(dim=1, keepdim=True).values
            w = torch.softmax(logits, dim=1)
            agg = torch.sum(x * w.unsqueeze(-1), dim=1)
            out = self.proj(agg)
            return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).to(feat_f.dtype)


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xr = torch.clamp(x[:, 0:1, :], -self.clip, self.clip)
        xi = torch.clamp(x[:, 1:2, :], -self.clip, self.clip)
        outs = []
        for m in range(self.memory_depth):
            ar = self._delay(xr, m)
            ai = self._delay(xi, m)
            mag2 = ar * ar + ai * ai
            for p in self.orders:
                if p == 1:
                    scale = torch.ones_like(mag2)
                else:
                    scale = torch.pow(torch.clamp(mag2, min=1e-8), (p - 1) / 2.0)
                outs.append(ar * scale)
                outs.append(ai * scale)
        return torch.cat(outs, dim=1)


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

    def forward(self, x: torch.Tensor, labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        with torch.cuda.amp.autocast(enabled=False):
            x_f = F.normalize(x.float(), dim=1, eps=1e-4)
            w_f = F.normalize(self.weight.float(), dim=1, eps=1e-4)
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
        detach_imp_gate: bool = True,
    ):
        super().__init__()
        self.gate_alpha = float(gate_alpha)
        self.detach_imp_gate = bool(detach_imp_gate)

        self.id_proj = nn.Sequential(
            nn.Linear(base_dim, emb_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(drop * 0.5),
        )
        self.dac_proj = nn.Sequential(
            nn.Linear(base_dim + emb_dim, emb_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(drop * 0.5),
        )
        self.pa_proj = nn.Sequential(
            nn.Linear(base_dim + emb_dim, emb_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(drop * 0.5),
        )
        self.txc_gate = nn.Sequential(
            nn.Linear(2 * emb_dim, emb_dim),
            nn.Sigmoid(),
        )
        self.id_fuse = nn.Sequential(
            nn.Linear(2 * emb_dim, emb_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(drop * 0.5),
        )
        self.imp_merge = nn.Sequential(
            nn.Linear(2 * emb_dim, emb_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(drop * 0.25),
        )
        self.head = CosFaceHead(emb_dim, num_classes, s=margin_s, m=margin_m)

        hid = max(16, emb_dim // 2)
        self.dac_head = nn.Sequential(
            nn.Linear(emb_dim, hid),
            nn.ReLU(inplace=True),
            nn.Linear(hid, 1),
        )
        self.pa_head = nn.Sequential(
            nn.Linear(emb_dim, hid),
            nn.ReLU(inplace=True),
            nn.Linear(hid, 1),
        )

    def forward(
        self,
        base: torch.Tensor,
        dac_local: torch.Tensor,
        pa_local: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        return_emb: bool = False,
        dac_delta: Optional[torch.Tensor] = None,
        pa_delta: Optional[torch.Tensor] = None,
        detach_imp_gate: Optional[bool] = None,
    ):
        feat_id = self.id_proj(base)
        feat_dac = self.dac_proj(torch.cat([base, dac_local], dim=1))
        feat_pa = self.pa_proj(torch.cat([base, pa_local], dim=1))

        if dac_delta is not None:
            feat_dac = feat_dac + dac_delta
        if pa_delta is not None:
            feat_pa = feat_pa + pa_delta

        z_pa = feat_id
        z_txc = feat_dac
        if detach_imp_gate is None:
            do_detach = self.detach_imp_gate
        else:
            do_detach = bool(detach_imp_gate)
        txc_in = torch.cat([z_pa, z_txc.detach()], dim=1) if do_detach else torch.cat([z_pa, z_txc], dim=1)
        alpha = self.txc_gate(txc_in)
        feat_joint = self.id_fuse(torch.cat([z_pa, self.gate_alpha * alpha * z_txc], dim=1))

        feat_imp = self.imp_merge(torch.cat([feat_dac, feat_pa], dim=1))
        logits = self.head(feat_joint, labels=labels)
        dac_pred = torch.sigmoid(self.dac_head(feat_dac).squeeze(-1))
        pa_pred = torch.sigmoid(self.pa_head(feat_pa).squeeze(-1))

        if not return_emb:
            return logits, dac_pred
        return logits, dac_pred, pa_pred, z_pa, feat_dac, feat_pa, feat_imp, feat_joint


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
        use_nonlinear_basis: bool = True,
        include_z_abs4: bool = False,
        nl_clip: float = 1.2,
        margin_s: float = 30.0,
        margin_m: float = 0.35,
        pa_memory_depth: int = 4,
        pa_orders: Sequence[int] = (1, 3, 5),
        pa_gate_alpha: float = 0.5,
        detach_imp_gate: bool = True,
        disable_freq_stats_to_shared: bool = True,
    ):
        super().__init__()
        self.dataset = str(dataset)
        self.input_len = int(input_len) if int(input_len) > 0 else 1024
        self.expected_len = self.input_len
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
        self.use_nonlinear_basis = bool(use_nonlinear_basis)
        self.include_z_abs4 = bool(include_z_abs4)
        self.nl_clip = float(nl_clip)
        self.detach_imp_gate = bool(detach_imp_gate)
        self.disable_freq_stats_to_shared = bool(disable_freq_stats_to_shared)

        # shared front-end
        self.sinc = SincConv1d(sinc_out, sinc_kernel, sample_rate=sample_rate, dataset=dataset, input_len=input_len, pad_crop_mode=pad_crop_mode)
        self.hf = HighFreqEmphasis()

        time_in = 2 * sinc_out + 4
        if self.use_nonlinear_basis:
            time_in += 2 * sinc_out
        if self.use_nonlinear_basis and self.include_z_abs4:
            time_in += 2 * sinc_out

        Cb = int(time_bottleneck)
        gn = _pick_gn_groups(Cb)
        self.time_fuse = nn.Sequential(
            nn.Conv1d(time_in, Cb, kernel_size=1, bias=False),
            nn.GroupNorm(gn, Cb),
            nn.ReLU(inplace=True),
        )
        self.time_down = nn.AvgPool1d(2)
        self.t1 = DSConvBlock1d(Cb, 128, k=5, pool=2, drop=0.10)
        self.t2 = DSConvBlock1d(128, 192, k=5, pool=2, drop=0.10)
        self.t3 = DSConvBlock1d(192, 192, k=3, pool=1, drop=0.10)
        self.t_pool = nn.AdaptiveAvgPool1d(1)
        self.t_proj = nn.Linear(192, emb_dim)

        # DAC branch: widely-linear complex conv over filterbank channels + injected HF details
        self.dac_hf_proj = nn.Sequential(
            nn.Conv1d(4, 2 * sinc_out, kernel_size=1, bias=False),
            nn.GroupNorm(_pick_gn_groups(2 * sinc_out), 2 * sinc_out),
            nn.SiLU(inplace=True),
        )
        self.dac_b1 = WLComplexBlock1d(sinc_out, sinc_out, k=5, dilation=1, pool=1, drop=0.05, residual=True)
        self.dac_b2 = WLComplexBlock1d(sinc_out, 64, k=3, dilation=1, pool=1, drop=0.05, residual=True)
        self.dac_b3 = WLComplexBlock1d(64, 64, k=3, dilation=2, pool=2, drop=0.05, residual=True)
        self.dac_pool = nn.AdaptiveAvgPool1d(1)
        self.dac_proj = nn.Sequential(
            nn.Linear(2 * 64, emb_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(drop * 0.25),
        )

        # frequency branch (legacy shared/freq contribution)
        self.f1 = DSConvBlock1d(4, 32, k=5, pool=2, drop=0.05)
        self.f2 = DSConvBlock1d(32, 64, k=5, pool=2, drop=0.05)
        self.f3 = DSConvBlock1d(64, 64, k=3, pool=1, drop=0.05)
        self.f_pool = nn.AdaptiveAvgPool1d(1)
        self.f_proj = nn.Linear(64, emb_dim)
        self.freq_gate = FreqBandGate1d(4, k=5, alpha=freq_gate_alpha) if self.use_freq_band_gate else nn.Identity()
        self.dac_subband_agg = SubbandGatedAggregator(in_ch=4, emb_dim=emb_dim, hidden=64, temperature=1.0)
        self.freq_stats_proj = (
            nn.Sequential(
                nn.Linear(3, emb_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(drop * 0.25),
            ) if self.use_freq_stats else None
        )
        self.pa_stats_proj = nn.Sequential(
            nn.Linear(3, emb_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(drop * 0.25),
        )

        # PA branch: memory polynomial lift + envelope-aware large receptive field convs
        self.pa_lift = MemoryPolynomialLift(memory_depth=pa_memory_depth, orders=pa_orders, clip=2.0)
        pa_in = 2 * pa_memory_depth * len(pa_orders)
        self.pa_gate = EnvelopeGate1d(pa_in, alpha=pa_gate_alpha, k=5)
        self.pa_b1 = DilatedConvBlock1d(pa_in, 96, k=7, dilation=1, pool=2, drop=0.08)
        self.pa_b2 = DilatedConvBlock1d(96, 128, k=7, dilation=2, pool=2, drop=0.08)
        self.pa_b3 = DilatedConvBlock1d(128, 128, k=5, dilation=4, pool=1, drop=0.08)
        self.pa_pool = nn.AdaptiveAvgPool1d(1)
        self.pa_proj = nn.Sequential(
            nn.Linear(128, emb_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(drop * 0.25),
        )

        # fuse + projection
        self.aux_head_t = nn.Linear(emb_dim, num_classes)
        self.aux_head_f = nn.Linear(emb_dim, num_classes)
        fuse_in = emb_dim * 2 + (1 if self.use_circularity else 0)
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
            detach_imp_gate=detach_imp_gate,
        )

        self._init_weights()

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
        i = x[:, 0:1, :]
        q = x[:, 1:2, :]
        yi = self.sinc(i)
        yq = self.sinc(q)
        return torch.cat([yi, yq], dim=1)

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

    def _mirror_compressed_features(self, x: torch.Tensor):
        """
        Returns:
          feat_f   : (B,4,K)  [logP_pos, logP_neg, logR, asym]
          rho      : (B,1) or None
          dac_stats: (B,3)   [hf_ratio, asym_hf_mean, flatness]
          pa_stats : (B,3)   [edge_ratio, regrowth_ratio, spec_kurtosis]
        """
        with torch.cuda.amp.autocast(enabled=False):
            x = torch.nan_to_num(x.float(), nan=0.0, posinf=0.0, neginf=0.0)
            _, _, L = x.shape
            eps = self.eps
            if self.use_rfft_pair and (L % 2 == 0):
                spec = self._fft_full_via_rfft_pair(x)
            else:
                spec = torch.fft.fft(self._to_complex(x), dim=-1)

            power = _finite_clamp(spec.real * spec.real + spec.imag * spec.imag, min_val=0.0)
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

            logP_pos = _finite_clamp(torch.log1p(pos), min_val=0.0, max_val=16.0)
            logP_neg = _finite_clamp(torch.log1p(neg), min_val=0.0, max_val=16.0)
            logR = _finite_clamp(torch.log((pos + eps) / (neg + eps)), min_val=-12.0, max_val=12.0)
            asym = _finite_clamp(torch.abs(pos - neg) / (pos + neg + eps), min_val=0.0, max_val=1.0)

            K = max(4, self.freq_bands)

            def pool1d(v: torch.Tensor) -> torch.Tensor:
                return F.adaptive_avg_pool1d(v.unsqueeze(1), K).squeeze(1)

            p_pos = pool1d(logP_pos)
            p_neg = pool1d(logP_neg)
            p_rat = pool1d(logR)
            p_asym = pool1d(asym)
            feat_f = _finite_clamp(torch.stack([p_pos, p_neg, p_rat, p_asym], dim=1), min_val=-12.0, max_val=16.0)

            pos_lin = pool1d(pos)
            neg_lin = pool1d(neg)
            tot_lin = _finite_clamp(pos_lin + neg_lin, min_val=eps)

            hf_start = max(0, int(0.75 * K))
            hf_ratio = (tot_lin[:, hf_start:].sum(dim=1, keepdim=True) / (tot_lin.sum(dim=1, keepdim=True) + eps)).clamp(0.0, 1.0)
            asym_hf_mean = p_asym[:, hf_start:].mean(dim=1, keepdim=True).clamp(0.0, 1.0)
            flatness = (
                torch.exp(torch.mean(torch.log(tot_lin), dim=1, keepdim=True))
                / (torch.mean(tot_lin, dim=1, keepdim=True) + eps)
            ).clamp(0.0, 4.0)
            dac_stats = _finite_clamp(torch.cat([hf_ratio, asym_hf_mean, flatness], dim=1), min_val=0.0, max_val=4.0)

            edge_bins = max(1, int(0.20 * K))
            center_l = max(0, int(0.30 * K))
            center_r = max(center_l + 1, int(0.70 * K))
            edge_energy = tot_lin[:, -edge_bins:].sum(dim=1, keepdim=True)
            center_energy = tot_lin[:, center_l:center_r].sum(dim=1, keepdim=True) + eps
            edge_ratio = (edge_energy / (tot_lin.sum(dim=1, keepdim=True) + eps)).clamp(0.0, 1.0)
            regrowth_ratio = (edge_energy / center_energy).clamp(0.0, 8.0)
            mu = torch.mean(tot_lin, dim=1, keepdim=True)
            var = torch.mean((tot_lin - mu) ** 2, dim=1, keepdim=True) + eps
            spec_kurtosis = (torch.mean((tot_lin - mu) ** 4, dim=1, keepdim=True) / (var * var)).clamp(0.0, 50.0)
            pa_stats = _finite_clamp(torch.cat([edge_ratio, regrowth_ratio, spec_kurtosis], dim=1), min_val=0.0, max_val=50.0)

            rho = None
            if self.use_circularity:
                z = self._to_complex(x)
                Ez2 = torch.mean(z * z, dim=-1)
                Eabs2 = torch.mean(z.real * z.real + z.imag * z.imag, dim=-1)
                rho = _finite_clamp((torch.abs(Ez2) / (Eabs2 + eps)).unsqueeze(1), min_val=0.0, max_val=1.0)

            return feat_f, rho, dac_stats, pa_stats

    def forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None, return_aux: bool = False):
        x = pad_crop_iq(x, self.input_len, mode=self.pad_crop_mode)

        # ----- shared time branch -----
        sinc_iq = self._sinc_on_iq(x)
        feats_t = [sinc_iq]
        if self.use_nonlinear_basis:
            z_abs2, z_abs4 = self._nonlinear_basis_after_filterbank(sinc_iq)
            feats_t.append(z_abs2)
            if self.include_z_abs4 and z_abs4 is not None:
                feats_t.append(z_abs4)
        hf = self.hf(x)
        feats_t.append(hf)
        t = torch.cat(feats_t, dim=1)
        t = self.time_fuse(t)
        t = self.time_down(t)
        t = self.t1(t)
        t = self.t2(t)
        t = self.t3(t)
        t = self.t_pool(t).squeeze(-1)
        t_emb = self.t_proj(t)

        if self.training and self.branch_drop_p > 0:
            B = t_emb.size(0)
            drop_mask = (torch.rand((B, 1), device=t_emb.device) < self.branch_drop_p).float()
            t_emb = t_emb * (1.0 - drop_mask)

        # ----- DAC branch -----
        dac_pair = sinc_iq + 0.5 * self.dac_hf_proj(hf)
        d = self.dac_b1(dac_pair)
        d = self.dac_b2(d)
        d = self.dac_b3(d)
        d = self.dac_pool(d).squeeze(-1)
        dac_local = self.dac_proj(d)

        # ----- freq branch -----
        feat_f, rho, dac_stats, pa_stats = self._mirror_compressed_features(x)
        feat_f = self.freq_gate(feat_f)
        dac_delta = self.dac_subband_agg(feat_f)

        f = self.f1(feat_f)
        f = self.f2(f)
        f = self.f3(f)
        f = self.f_pool(f).squeeze(-1)
        f_emb = self.f_proj(f)
        if self.use_freq_stats and (self.freq_stats_proj is not None) and (not self.disable_freq_stats_to_shared):
            f_emb = f_emb + self.freq_stats_proj(dac_stats)

        # ----- PA branch -----
        pa_feat = self.pa_lift(x)
        pa_feat = self.pa_gate(pa_feat, x)
        p = self.pa_b1(pa_feat)
        p = self.pa_b2(p)
        p = self.pa_b3(p)
        p = self.pa_pool(p).squeeze(-1)
        pa_local = self.pa_proj(p)
        pa_delta = self.pa_stats_proj(pa_stats)
        pa_local = pa_local + 0.10 * pa_delta

        logits_t = self.aux_head_t(t_emb)
        logits_f = self.aux_head_f(f_emb)

        base_in = torch.cat([t_emb, f_emb, rho], dim=1) if rho is not None else torch.cat([t_emb, f_emb], dim=1)
        base = self.fuse(base_in)
        feat_con = self.con_proj(base)

        logits, dac_pred, pa_pred, feat_id, feat_dac, feat_pa, feat_imp, feat_joint = self.cls_head(
            base,
            dac_local=dac_local,
            pa_local=pa_local,
            labels=y,
            return_emb=True,
            dac_delta=dac_delta,
            pa_delta=pa_delta,
            detach_imp_gate=self.detach_imp_gate,
        )

        if not return_aux:
            return logits

        return {
            'logits': logits,
            'dac_pred': dac_pred,
            'pa_pred': pa_pred,
            'feat_cls': feat_id,
            'feat_imp': feat_imp,
            'feat_dac': feat_dac,
            'feat_pa': feat_pa,
            'feat_con': feat_con,
            'logits_t': logits_t,
            'logits_f': logits_f,
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
            'z_pa': feat_id,
            'z_txc': feat_dac,
            'z_id': feat_joint,
        }


# ----------------------- factory -----------------------
def build_model(
    num_classes: int = 16,
    model_size: str = "M",
    dataset: str = "oralce",
    input_len: int = 1024,
    sample_rate_hz: float = 5e6,
    detach_imp_gate: bool = True,
    disable_freq_stats_to_shared: bool = True,
):
    ds = str(dataset).lower()
    ms = str(model_size).upper().strip()

    if (input_len is None) or (int(input_len) <= 0):
        input_len = 256 if ds == "wisig" else 1024

    if ms in ("S", "SMALL"):
        cfg = dict(sinc_out=32, sinc_kernel=63, time_bottleneck=96, emb_dim=192, freq_bands=48, drop=0.40)
    elif ms in ("L", "LARGE"):
        cfg = dict(sinc_out=64, sinc_kernel=79, time_bottleneck=128, emb_dim=256, freq_bands=96, drop=0.40)
    else:
        cfg = dict(sinc_out=48, sinc_kernel=79, time_bottleneck=96, emb_dim=256, freq_bands=64, drop=0.40)

    return CVSincNet(
        num_classes=int(num_classes),
        sample_rate=float(sample_rate_hz),
        dataset=ds,
        input_len=int(input_len),
        pad_crop_mode="center",
        detach_imp_gate=bool(detach_imp_gate),
        disable_freq_stats_to_shared=bool(disable_freq_stats_to_shared),
        **cfg,
    )
