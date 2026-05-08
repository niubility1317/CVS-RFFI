# DataAugmentation_v2.py
# ------------------------------------------------------------
# RFFI Data Augmentation (DAC-specialized)
#
# Goals:
# 1) Provide "dac_only" view that emphasizes DAC-induced RF fingerprints:
#    - Complex-envelope AM/AM nonlinearity: z <- z*(1 + a3|z|^2 + a5|z|^4)
#    - Time-interleaving even/odd mismatch (cheap but fingerprint-like spurs)
#    - Quantization (bits decreases with strength) + optional dither
#    - Slew-rate limiting
#
# 2) Support per-class DAC "signature" (class-stable impairment parameters)
#    - If labels are provided, each class has fixed base params; each sample adds small jitter
#    - This makes DAC effects more "fingerprint-like" (stable within device/class)
#
# 3) Optional "anti-shortcut" augmentations to prevent relying on DC / band-edge artifacts:
#    - DC offsets (I/Q) and mild smoothing to randomize spectral edges
#
# API used by your train.py:
#   aug = RFFIAugmentor(...)
#   x1 = aug(x, return_dac_strength=False, no_dac=True)
#   x2, s2 = aug(x, labels=y, return_dac_strength=True, dac_only=True)
# ------------------------------------------------------------

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def _clamp(x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    return torch.clamp(x, min=lo, max=hi)


def _lerp(a: torch.Tensor, b: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    return a + (b - a) * t


def _safe_div(num: torch.Tensor, den: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return num / (den + eps)


@dataclass
class _ClassDACSig:
    # Base parameters for "strength=1" (then scaled by s)
    a3: float
    a5: float
    iq_skew: float
    inter_gain: float
    inter_off: float
    bits_bias: int
    slew_bias: float


class RFFIAugmentor:
    """
    Augment IQ signals for RF fingerprinting.

    Input x: (B,2,L) float tensor, assumed roughly normalized.
    """

    def __init__(
        self,
        sampling_rate: float = 5e6,

        # classic channel/receiver impairments
        p_phase_rotate: float = 0.5,
        p_amp_scale: float = 0.4,
        p_noise: float = 0.5,
        p_cfo: float = 0.25,
        p_iq_imbalance: float = 0.2,
        p_dac: float = 0.3,

        snr_db_range: Tuple[float, float] = (18.0, 35.0),
        cfo_max_hz: float = 800.0,

        # anti-shortcut (optional)
        p_dc_offset: float = 0.0,
        dc_offset_max: float = 0.05,
        p_bandedge_smooth: float = 0.0,
        smooth_kernel_choices: Tuple[int, ...] = (3, 5),

        # DAC strength sampling
        dac_strength_range: Tuple[float, float] = (0.1, 1.0),

        # DAC parameters
        dac_bits_range: Tuple[int, int] = (6, 12),           # bmin, bmax
        dac_poly_a3_max: float = 0.18,                       # for z*(1 + a3|z|^2 + ...)
        dac_poly_a5_max: float = 0.06,
        dac_iq_skew_max: float = 0.02,                       # extra small I/Q asymmetry (mirror cues)
        dac_interleave_gain_max: float = 0.01,               # even/odd gain mismatch
        dac_interleave_off_max: float = 0.005,               # even/odd offset mismatch
        dac_slew_delta_range: Tuple[float, float] = (0.005, 0.05),  # (dmin, dmax) in normalized units
        dac_dither: bool = True,

        # per-class signature
        enable_class_signature: bool = True,
        signature_seed: int = 2026,
        signature_jitter: float = 0.10,                      # relative jitter around base params
    ):
        self.fs = float(sampling_rate)

        self.p_phase_rotate = float(p_phase_rotate)
        self.p_amp_scale = float(p_amp_scale)
        self.p_noise = float(p_noise)
        self.p_cfo = float(p_cfo)
        self.p_iq_imbalance = float(p_iq_imbalance)
        self.p_dac = float(p_dac)

        self.snr_db_range = (float(snr_db_range[0]), float(snr_db_range[1]))
        self.cfo_max_hz = float(cfo_max_hz)

        self.p_dc_offset = float(p_dc_offset)
        self.dc_offset_max = float(dc_offset_max)
        self.p_bandedge_smooth = float(p_bandedge_smooth)
        self.smooth_kernel_choices = tuple(int(k) for k in smooth_kernel_choices)

        self.dac_strength_range = (float(dac_strength_range[0]), float(dac_strength_range[1]))

        self.dac_bits_range = (int(dac_bits_range[0]), int(dac_bits_range[1]))
        self.dac_poly_a3_max = float(dac_poly_a3_max)
        self.dac_poly_a5_max = float(dac_poly_a5_max)
        self.dac_iq_skew_max = float(dac_iq_skew_max)
        self.dac_interleave_gain_max = float(dac_interleave_gain_max)
        self.dac_interleave_off_max = float(dac_interleave_off_max)
        self.dac_slew_delta_range = (float(dac_slew_delta_range[0]), float(dac_slew_delta_range[1]))
        self.dac_dither = bool(dac_dither)

        self.enable_class_signature = bool(enable_class_signature)
        self.signature_seed = int(signature_seed)
        self.signature_jitter = float(signature_jitter)

        self._class_sigs: Dict[int, _ClassDACSig] = {}

    # -------------------- basic helpers --------------------
    @staticmethod
    def _rms(x: torch.Tensor) -> torch.Tensor:
        # x: (B,2,L) -> (B,1,1)
        p = torch.mean(x * x, dim=(1, 2), keepdim=True)
        return torch.sqrt(p + 1e-12)

    @staticmethod
    def _iq_to_complex(x: torch.Tensor) -> torch.Tensor:
        # x: (B,2,L) -> (B,L) complex
        return torch.complex(x[:, 0, :], x[:, 1, :])

    @staticmethod
    def _complex_to_iq(z: torch.Tensor) -> torch.Tensor:
        # z: (B,L) complex -> (B,2,L)
        return torch.stack([z.real, z.imag], dim=1)

    @staticmethod
    def _randu(shape, device, dtype, lo=0.0, hi=1.0) -> torch.Tensor:
        return (hi - lo) * torch.rand(shape, device=device, dtype=dtype) + lo

    # -------------------- signature --------------------
    def _make_sig_for_label(self, label: int) -> _ClassDACSig:
        rs = np.random.RandomState(self.signature_seed + int(label) * 10007)

        a3 = float(rs.uniform(0.25, 1.0) * self.dac_poly_a3_max)
        a5 = float(rs.uniform(0.10, 1.0) * self.dac_poly_a5_max)

        iq_skew = float(rs.uniform(-1.0, 1.0) * self.dac_iq_skew_max)
        inter_gain = float(rs.uniform(-1.0, 1.0) * self.dac_interleave_gain_max)
        inter_off = float(rs.uniform(-1.0, 1.0) * self.dac_interleave_off_max)

        # bias bits by -1/0/+1 to create stable but subtle quant differences
        bits_bias = int(rs.choice([-1, 0, 1]))

        # slew bias (relative scale)
        slew_bias = float(rs.uniform(-0.20, 0.20))

        return _ClassDACSig(
            a3=a3,
            a5=a5,
            iq_skew=iq_skew,
            inter_gain=inter_gain,
            inter_off=inter_off,
            bits_bias=bits_bias,
            slew_bias=slew_bias,
        )

    def _get_sigs(self, labels: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Return per-sample base params as tensors on same device/dtype as labels.
        labels: (B,)
        """
        labels_cpu = labels.detach().to("cpu").view(-1).tolist()
        for lab in set(int(x) for x in labels_cpu):
            if lab not in self._class_sigs:
                self._class_sigs[lab] = self._make_sig_for_label(lab)

        # gather
        a3 = torch.tensor([self._class_sigs[int(l)].a3 for l in labels_cpu], device=labels.device, dtype=torch.float32)
        a5 = torch.tensor([self._class_sigs[int(l)].a5 for l in labels_cpu], device=labels.device, dtype=torch.float32)
        iq = torch.tensor([self._class_sigs[int(l)].iq_skew for l in labels_cpu], device=labels.device, dtype=torch.float32)
        ig = torch.tensor([self._class_sigs[int(l)].inter_gain for l in labels_cpu], device=labels.device, dtype=torch.float32)
        io = torch.tensor([self._class_sigs[int(l)].inter_off for l in labels_cpu], device=labels.device, dtype=torch.float32)
        bb = torch.tensor([self._class_sigs[int(l)].bits_bias for l in labels_cpu], device=labels.device, dtype=torch.int64)
        sb = torch.tensor([self._class_sigs[int(l)].slew_bias for l in labels_cpu], device=labels.device, dtype=torch.float32)

        return {
            "a3": a3.view(-1, 1, 1),
            "a5": a5.view(-1, 1, 1),
            "iq_skew": iq.view(-1, 1, 1),
            "inter_gain": ig.view(-1, 1, 1),
            "inter_off": io.view(-1, 1, 1),
            "bits_bias": bb.view(-1),
            "slew_bias": sb.view(-1, 1, 1),
        }

    # -------------------- augment ops --------------------
    def _apply_phase_rotate(self, x: torch.Tensor) -> torch.Tensor:
        device, dtype = x.device, x.dtype
        B, _, L = x.shape
        theta = self._randu((B, 1), device, dtype, -math.pi, math.pi)
        n = torch.arange(L, device=device, dtype=dtype).view(1, -1)
        # constant phase rotation: exp(j theta)
        ej = torch.complex(torch.cos(theta), torch.sin(theta))  # (B,1) complex
        z = self._iq_to_complex(x) * ej
        return self._complex_to_iq(z)

    def _apply_amp_scale(self, x: torch.Tensor) -> torch.Tensor:
        device, dtype = x.device, x.dtype
        B = x.size(0)
        # log-uniform gain in [0.7, 1.3]
        logg = self._randu((B, 1, 1), device, dtype, math.log(0.7), math.log(1.3))
        g = torch.exp(logg)
        return x * g

    def _apply_awgn(self, x: torch.Tensor) -> torch.Tensor:
        device, dtype = x.device, x.dtype
        B = x.size(0)
        snr_lo, snr_hi = self.snr_db_range
        snr_db = self._randu((B, 1, 1), device, dtype, snr_lo, snr_hi)
        snr = 10.0 ** (snr_db / 10.0)

        sig_pow = torch.mean(x * x, dim=(1, 2), keepdim=True) + 1e-12
        noise_pow = sig_pow / snr
        noise = torch.randn_like(x) * torch.sqrt(noise_pow)
        return x + noise

    def _apply_cfo(self, x: torch.Tensor) -> torch.Tensor:
        device, dtype = x.device, x.dtype
        B, _, L = x.shape
        f = self._randu((B, 1), device, dtype, -self.cfo_max_hz, self.cfo_max_hz)  # Hz
        n = torch.arange(L, device=device, dtype=dtype).view(1, -1)
        phase = 2.0 * math.pi * f * n / self.fs  # (B,L)
        ej = torch.complex(torch.cos(phase), torch.sin(phase))  # (B,L)
        z = self._iq_to_complex(x) * ej
        return self._complex_to_iq(z)

    def _apply_iq_imbalance(self, x: torch.Tensor) -> torch.Tensor:
        """
        Simple gain+phase imbalance:
          [I'] = gi*( I*cos(phi) - Q*sin(phi) )
          [Q'] = gq*( I*sin(phi) + Q*cos(phi) )
        """
        device, dtype = x.device, x.dtype
        B, _, L = x.shape
        # small imbalance
        g = self._randu((B, 1, 1), device, dtype, 0.90, 1.10)
        phi = self._randu((B, 1, 1), device, dtype, -0.08, 0.08)  # radians

        I = x[:, 0:1, :]
        Q = x[:, 1:2, :]

        c = torch.cos(phi)
        s = torch.sin(phi)

        I2 = I * c - Q * s
        Q2 = I * s + Q * c

        # slight unequal gains
        gi = g
        gq = 2.0 - g  # roughly symmetric
        y = torch.cat([I2 * gi, Q2 * gq], dim=1)
        return y

    def _apply_dc_offset(self, x: torch.Tensor) -> torch.Tensor:
        device, dtype = x.device, x.dtype
        B = x.size(0)
        dc = self._randu((B, 2, 1), device, dtype, -self.dc_offset_max, self.dc_offset_max)
        return x + dc

    def _apply_bandedge_smooth(self, x: torch.Tensor) -> torch.Tensor:
        """
        Mild smoothing (depthwise conv) to randomize band-edge response.
        """
        device, dtype = x.device, x.dtype
        B, C, L = x.shape
        k = int(np.random.choice(self.smooth_kernel_choices))
        if k == 3:
            w = torch.tensor([0.25, 0.5, 0.25], device=device, dtype=dtype)
        elif k == 5:
            w = torch.tensor([0.10, 0.20, 0.40, 0.20, 0.10], device=device, dtype=dtype)
        else:
            # fallback uniform
            w = torch.ones(k, device=device, dtype=dtype) / float(k)

        w = w.view(1, 1, -1).repeat(C, 1, 1)  # (C,1,k)
        pad = k // 2
        y = F.conv1d(x, w, padding=pad, groups=C)
        return y[..., :L]

    # -------------------- DAC simulation --------------------
    def _sample_strength(self, B: int, device, dtype) -> torch.Tensor:
        lo, hi = self.dac_strength_range
        return self._randu((B,), device, dtype, lo, hi)

    def simulate_dac(
        self,
        x: torch.Tensor,
        strength: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        x: (B,2,L)
        strength: (B,) in [0,1] (or [lo,hi])
        labels: (B,) optional for per-class signature
        """
        device, dtype = x.device, x.dtype
        B, _, L = x.shape
        s = strength.view(B, 1, 1).to(device=device, dtype=dtype)

        # normalize to full-scale
        rms = self._rms(x)  # (B,1,1)
        full_scale = torch.clamp(3.0 * rms, min=1e-6)
        xn = torch.clamp(x / full_scale, -1.0, 1.0)

        # base (class signature) + jitter
        if self.enable_class_signature and labels is not None:
            sig = self._get_sigs(labels.to(device=device))
            a3_base = sig["a3"].to(device=device, dtype=dtype)
            a5_base = sig["a5"].to(device=device, dtype=dtype)
            iq_base = sig["iq_skew"].to(device=device, dtype=dtype)
            ig_base = sig["inter_gain"].to(device=device, dtype=dtype)
            io_base = sig["inter_off"].to(device=device, dtype=dtype)
            bits_bias = sig["bits_bias"].to(device=device)
            slew_bias = sig["slew_bias"].to(device=device, dtype=dtype)
        else:
            a3_base = self._randu((B, 1, 1), device, dtype, 0.25 * self.dac_poly_a3_max, self.dac_poly_a3_max)
            a5_base = self._randu((B, 1, 1), device, dtype, 0.10 * self.dac_poly_a5_max, self.dac_poly_a5_max)
            iq_base = self._randu((B, 1, 1), device, dtype, -self.dac_iq_skew_max, self.dac_iq_skew_max)
            ig_base = self._randu((B, 1, 1), device, dtype, -self.dac_interleave_gain_max, self.dac_interleave_gain_max)
            io_base = self._randu((B, 1, 1), device, dtype, -self.dac_interleave_off_max, self.dac_interleave_off_max)
            bits_bias = torch.zeros((B,), device=device, dtype=torch.int64)
            slew_bias = self._randu((B, 1, 1), device, dtype, -0.15, 0.15)

        # jitter around base
        if self.signature_jitter > 0:
            j = self.signature_jitter
            a3 = a3_base * (1.0 + j * torch.randn((B, 1, 1), device=device, dtype=dtype))
            a5 = a5_base * (1.0 + j * torch.randn((B, 1, 1), device=device, dtype=dtype))
            iq_skew = iq_base + (j * 0.5) * torch.randn((B, 1, 1), device=device, dtype=dtype) * self.dac_iq_skew_max
            inter_gain = ig_base + (j * 0.5) * torch.randn((B, 1, 1), device=device, dtype=dtype) * self.dac_interleave_gain_max
            inter_off = io_base + (j * 0.5) * torch.randn((B, 1, 1), device=device, dtype=dtype) * self.dac_interleave_off_max
        else:
            a3, a5 = a3_base, a5_base
            iq_skew = iq_base
            inter_gain = ig_base
            inter_off = io_base

        # scale by strength
        a3 = a3 * s
        a5 = a5 * s
        iq_skew = iq_skew * s
        inter_gain = inter_gain * s
        inter_off = inter_off * s

        # (1) Complex-envelope AM/AM nonlinearity: z*(1 + a3|z|^2 + a5|z|^4)
        z = self._iq_to_complex(xn)  # (B,L) complex
        a2 = (z.real * z.real + z.imag * z.imag).clamp(0.0, 4.0)  # |z|^2
        # a3/a5/iq_skew: (B,1,1) -> (B,1) 以便和 (B,L) 广播
        a3v = a3.view(B, 1)
        a5v = a5.view(B, 1)
        iqv = iq_skew.view(B, 1)

        g = 1.0 + a3v * a2 + a5v * (a2 * a2)   # (B,L)

        g_i = g * (1.0 + iqv)                  # (B,L)
        g_q = g * (1.0 - iqv)                  # (B,L)




        yI = z.real * g_i
        yQ = z.imag * g_q
        yn = torch.stack([yI, yQ], dim=1)
        yn = _clamp(yn, -1.0, 1.0)

        # (2) Time-interleaving even/odd mismatch
        n = torch.arange(L, device=device, dtype=dtype).view(1, 1, L)
        alt = (n % 2) * 2 - 1  # -1/+1
        yn = yn * (1.0 + inter_gain * alt) + inter_off * alt
        yn = _clamp(yn, -1.0, 1.0)

        # (3) Quantization: bits decreases with strength, plus per-class bias
        bmin, bmax = self.dac_bits_range
        bits_f = (bmax - (bmax - bmin) * s.squeeze(-1).squeeze(-1)).round().to(torch.int64)  # (B,)
        bits = torch.clamp(bits_f + bits_bias, min=bmin, max=bmax)  # (B,)
        levels = (2.0 ** bits.to(dtype=dtype)).view(B, 1, 1)
        step = 2.0 / (levels - 1.0 + 1e-12)

        if self.dac_dither:
            dither = (torch.rand_like(yn) - 0.5) * step
            yn = _clamp(yn + dither, -1.0, 1.0)

        q = torch.round((yn + 1.0) / step) * step - 1.0
        q = _clamp(q, -1.0, 1.0)

        # (4) Slew-rate limiting
        dmin, dmax = self.dac_slew_delta_range
        # higher strength -> smaller max_delta
        base_delta = _lerp(torch.tensor(dmax, device=device, dtype=dtype),
                           torch.tensor(dmin, device=device, dtype=dtype), s)  # (B,1,1)
        max_delta = base_delta * (1.0 + slew_bias * s)
        max_delta = _clamp(max_delta, 0.001, 0.2)

        x0 = q[:, :, 0:1]
        diff = q[:, :, 1:] - q[:, :, :-1]
        diff = torch.clamp(diff, -max_delta, max_delta)
        y = torch.cat([x0, x0 + torch.cumsum(diff, dim=-1)], dim=-1)
        y = _clamp(y, -1.0, 1.0)

        return y * full_scale

    # -------------------- public call --------------------
    def __call__(
        self,
        x: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        return_dac_strength: bool = False,
        no_dac: bool = False,
        dac_only: bool = False,
    ):
        """
        x: (B,2,L)
        labels: (B,) optional. If provided and enable_class_signature=True, DAC params become class-stable.
        no_dac: force no DAC (for channel-only view)
        dac_only: apply ONLY DAC (plus optional anti-shortcut if you enabled p_dc_offset/p_bandedge_smooth)
        """
        if x.dim() != 3 or x.size(1) != 2:
            raise ValueError("Input x must have shape (B,2,L).")

        device, dtype = x.device, x.dtype
        B = x.size(0)

        # Anti-shortcut transforms can be applied in any view if enabled
        def maybe_apply(p: float, fn, x_in: torch.Tensor) -> torch.Tensor:
            if p <= 0:
                return x_in
            if torch.rand((), device=device).item() < p:
                return fn(x_in)
            return x_in

        # if dac_only, skip all classic channel augments
        y = x

        if not dac_only:
            # phase rotation
            y = maybe_apply(self.p_phase_rotate, self._apply_phase_rotate, y)
            # amplitude scaling
            y = maybe_apply(self.p_amp_scale, self._apply_amp_scale, y)
            # IQ imbalance
            y = maybe_apply(self.p_iq_imbalance, self._apply_iq_imbalance, y)
            # CFO
            y = maybe_apply(self.p_cfo, self._apply_cfo, y)
            # AWGN
            y = maybe_apply(self.p_noise, self._apply_awgn, y)

        # anti-shortcut (optional)
        y = maybe_apply(self.p_dc_offset, self._apply_dc_offset, y)
        y = maybe_apply(self.p_bandedge_smooth, self._apply_bandedge_smooth, y)

        # DAC
        dac_strength = torch.zeros((B,), device=device, dtype=dtype)

        apply_dac = (not no_dac) and (self.p_dac > 0.0)
        if apply_dac:
            # dac_only view usually sets p_dac=1.0, so this always triggers
            if (dac_only and self.p_dac >= 1.0) or (torch.rand((), device=device).item() < self.p_dac):
                dac_strength = self._sample_strength(B, device, dtype)
                y = self.simulate_dac(y, dac_strength, labels=labels)

        if return_dac_strength:
            return y, dac_strength
        return y


__all__ = ["RFFIAugmentor"]
