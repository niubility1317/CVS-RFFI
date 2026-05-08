import math
import torch
import torch.nn.functional as F
from typing import Tuple, Optional


class RFFIAugmentor:
    """
    精简稳定版 RFFI 增强：
      - phase rotate
      - amp scale
      - AWGN (SNR sampling)
      - small CFO
      - small IQ imbalance
      - DAC impairment (quantization + mild nonlinearity + mild slew) with strength label

    输入:  x (B,2,L) 2=[I,Q]
    输出:
      return_dac_strength=False -> x_aug
      return_dac_strength=True  -> (x_aug, dac_strength)  dac_strength:(B,) in [0,1]
    """

    def __init__(
        self,
        sampling_rate: float = 5e6,

        # probabilities (keep them moderate)
        p_phase_rotate: float = 0.5,
        p_amp_scale: float = 0.4,
        p_noise: float = 0.5,
        p_cfo: float = 0.25,
        p_iq_imbalance: float = 0.2,
        p_dac: float = 0.3,

        # noise
        snr_db_range: Tuple[float, float] = (18.0, 35.0),

        # amp scale
        scale_range: float = 0.10,   # +/-10%

        # CFO
        cfo_max_hz: float = 300.0,   # 建议先 300~1000Hz，小一些更稳

        # IQ imbalance (keep small)
        amp_imbalance_max: float = 0.03,       # <=3%
        phase_imbalance_max_deg: float = 2.0,  # <=2 deg

        # DAC strength label
        dac_strength_range: Tuple[float, float] = (0.1, 1.0),  # smin>0 方便mask

        # DAC quantization bits (stronger => fewer bits)
        dac_bits_range: Tuple[int, int] = (7, 12),

        # DAC nonlinearity strength (mild)
        dac_poly_a3_max: float = 0.18,  # y = x + a3*x^3
        dac_poly_a5_max: float = 0.06,  # + a5*x^5

        # DAC slew rate (mild & stable)
        dac_slew_delta_range: Tuple[float, float] = (0.05, 0.50),  # stronger -> smaller

        # quantization dither (helps stability)
        dac_dither: bool = True,
    ):
        self.sampling_rate = float(sampling_rate)

        self.p_phase_rotate = float(p_phase_rotate)
        self.p_amp_scale = float(p_amp_scale)
        self.p_noise = float(p_noise)
        self.p_cfo = float(p_cfo)
        self.p_iq_imbalance = float(p_iq_imbalance)
        self.p_dac = float(p_dac)

        self.snr_db_range = (float(snr_db_range[0]), float(snr_db_range[1]))
        self.scale_range = float(scale_range)

        self.cfo_max = float(cfo_max_hz) / self.sampling_rate  # cycles/sample

        self.amp_imbalance_max = float(amp_imbalance_max)
        self.phase_imbalance_max_deg = float(phase_imbalance_max_deg)

        self.dac_strength_range = (float(dac_strength_range[0]), float(dac_strength_range[1]))
        self.dac_bits_range = (int(dac_bits_range[0]), int(dac_bits_range[1]))
        self.dac_poly_a3_max = float(dac_poly_a3_max)
        self.dac_poly_a5_max = float(dac_poly_a5_max)
        self.dac_slew_delta_range = (float(dac_slew_delta_range[0]), float(dac_slew_delta_range[1]))
        self.dac_dither = bool(dac_dither)

    # ---------------- utils ----------------
    @staticmethod
    def _should_apply(p: float, B: int, device) -> torch.Tensor:
        if p <= 0:
            return torch.zeros(B, 1, 1, device=device, dtype=torch.bool)
        if p >= 1:
            return torch.ones(B, 1, 1, device=device, dtype=torch.bool)
        return (torch.rand(B, 1, 1, device=device) < p)

    @staticmethod
    def _iq_to_complex(x: torch.Tensor) -> torch.Tensor:
        return torch.complex(x[:, 0, :], x[:, 1, :])  # (B,L)

    @staticmethod
    def _complex_to_iq(z: torch.Tensor) -> torch.Tensor:
        return torch.stack([z.real, z.imag], dim=1)  # (B,2,L)

    @staticmethod
    def _rms(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        return torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + eps)  # (B,2,1)

    # ---------------- basic aug ----------------
    def apply_phase_rotate(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        device = x.device
        mask = self._should_apply(self.p_phase_rotate, B, device)

        theta = (torch.rand(B, 1, 1, device=device) * 2 - 1) * math.pi
        z = self._iq_to_complex(x)
        rot = torch.exp(1j * theta.squeeze(-1).squeeze(-1))
        out = self._complex_to_iq(z * rot.unsqueeze(-1))
        return torch.where(mask, out, x)

    def apply_amp_scale(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        device = x.device
        mask = self._should_apply(self.p_amp_scale, B, device)

        scale = 1.0 + (torch.rand(B, 1, 1, device=device) * 2 - 1) * self.scale_range
        out = x * scale
        return torch.where(mask, out, x)

    def apply_noise(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        device = x.device
        mask = self._should_apply(self.p_noise, B, device)

        snr_min, snr_max = self.snr_db_range
        snr_db = torch.rand(B, 1, 1, device=device) * (snr_max - snr_min) + snr_min
        sig_power = torch.mean(x * x, dim=(1, 2), keepdim=True) + 1e-12
        snr_linear = 10 ** (snr_db / 10.0)
        noise_power = sig_power / snr_linear
        noise_std = torch.sqrt(noise_power)

        noise = torch.randn_like(x) * noise_std
        out = x + noise
        return torch.where(mask, out, x)

    def apply_cfo(self, x: torch.Tensor) -> torch.Tensor:
        B, _, L = x.shape
        device = x.device
        mask = self._should_apply(self.p_cfo, B, device)

        delta = (torch.rand(B, 1, 1, device=device) * 2 - 1) * self.cfo_max  # cycles/sample
        n = torch.arange(L, device=device, dtype=x.dtype).view(1, 1, L)
        theta = 2 * math.pi * delta * n
        z = self._iq_to_complex(x)
        rot = torch.exp(1j * theta.squeeze(1))
        out = self._complex_to_iq(z * rot)
        return torch.where(mask, out, x)

    def apply_iq_imbalance(self, x: torch.Tensor) -> torch.Tensor:
        B, _, _ = x.shape
        device = x.device
        mask = self._should_apply(self.p_iq_imbalance, B, device)

        eps = (torch.rand(B, 1, 1, device=device) * 2 - 1) * self.amp_imbalance_max
        i = x[:, 0:1, :] * (1.0 + eps)
        q = x[:, 1:2, :] * (1.0 - eps)

        phi_deg = (torch.rand(B, 1, 1, device=device) * 2 - 1) * self.phase_imbalance_max_deg
        phi = phi_deg * math.pi / 180.0
        q2 = q * torch.cos(phi) + i * torch.sin(phi)

        out = torch.cat([i, q2], dim=1)
        return torch.where(mask, out, x)

    # ---------------- DAC impairment (lite & stable) ----------------
    def simulate_dac(self, x: torch.Tensor, strength: torch.Tensor) -> torch.Tensor:
        """
        x: (B,2,L)
        strength: (B,) in [0,1]
        """
        device = x.device
        dtype = x.dtype
        B, C, L = x.shape
        s = strength.view(B, 1, 1).to(device=device, dtype=dtype)

        # normalize roughly to [-1,1]
        rms = self._rms(x)                           # (B,2,1)
        full_scale = torch.clamp(3.0 * rms, min=1e-6)
        xn = torch.clamp(x / full_scale, -1.0, 1.0)  # (B,2,L)

        # mild nonlinearity
        a3 = (torch.rand(B, 1, 1, device=device, dtype=dtype) * self.dac_poly_a3_max) * s
        a5 = (torch.rand(B, 1, 1, device=device, dtype=dtype) * self.dac_poly_a5_max) * s
        yn = xn + a3 * (xn ** 3) + a5 * (xn ** 5)
        yn = torch.clamp(yn, -1.0, 1.0)

        # bits: stronger => fewer bits
        bmin, bmax = self.dac_bits_range
        bits_f = (bmax - (bmax - bmin) * s).squeeze(-1).squeeze(-1)  # (B,)
        bits = torch.clamp(torch.round(bits_f), min=bmin, max=bmax).to(torch.int64)

        levels = (2.0 ** bits.to(dtype=dtype)).view(B, 1, 1)
        step = 2.0 / (levels - 1.0 + 1e-12)

        # optional dither
        if self.dac_dither:
            dither = (torch.rand_like(yn) - 0.5) * step
            yn = torch.clamp(yn + dither, -1.0, 1.0)

        # quantize
        q = torch.round((yn + 1.0) / step) * step - 1.0
        q = torch.clamp(q, -1.0, 1.0)

        # mild slew-rate limiting (stable)
        dmin, dmax = self.dac_slew_delta_range
        max_delta = ((1.0 - s) * dmax + s * dmin).to(dtype=dtype)  # stronger => smaller

        x0 = q[:, :, 0:1]
        diff = q[:, :, 1:] - q[:, :, :-1]
        diff = torch.clamp(diff, -max_delta, max_delta)
        y = torch.cat([x0, x0 + torch.cumsum(diff, dim=-1)], dim=-1)
        y = torch.clamp(y, -1.0, 1.0)

        # de-normalize
        return y * full_scale

    def apply_dac(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        returns (x_out, dac_strength)
        """
        B = x.size(0)
        device = x.device

        dac_strength = torch.zeros(B, device=device, dtype=torch.float32)
        if self.p_dac <= 0:
            return x, dac_strength

        mask = self._should_apply(self.p_dac, B, device).squeeze(-1).squeeze(-1)  # (B,)
        if not mask.any():
            return x, dac_strength

        smin, smax = self.dac_strength_range
        s = torch.rand(B, device=device) * (smax - smin) + smin
        s = torch.clamp(s, 0.0, 1.0)

        x_dac = self.simulate_dac(x, s)
        m = mask.view(B, 1, 1)
        x_out = torch.where(m, x_dac, x)
        dac_strength = torch.where(mask, s.to(torch.float32), dac_strength)
        return x_out, dac_strength

    # ---------------- call ----------------
    def __call__(self, x: torch.Tensor, return_dac_strength: bool = False):
        """
        x: (B,2,L)
        """
        # 先 DAC（让模型看到 DAC 纹理），但强度/概率是温和的
        x, dac_strength = self.apply_dac(x)

        # 再做几项“温和且物理合理”的扰动
        x = self.apply_phase_rotate(x)
        x = self.apply_amp_scale(x)
        x = self.apply_cfo(x)
        x = self.apply_iq_imbalance(x)
        x = self.apply_noise(x)

        if return_dac_strength:
            return x, dac_strength
        return x
