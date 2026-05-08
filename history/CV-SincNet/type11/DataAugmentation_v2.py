# DataAugmentation_v2.py
# RFFI Data Augmentation v2 (DAC-aware + domain generalization)
#
# Key features:
# - Clean view controls: no_dac / dac_only / return_dac_strength / labels(class-signature)
# - Stronger, more realistic DAC impairment simulation:
#     * Sampling jitter (time micro-warp)
#     * Memory polynomial (nonlinearity with memory)
#     * AM/AM polynomial + IQ imbalance (image term)
#     * 2/4-way interleaving mismatch (gain/offset/skew)
#     * Quantization with dither + optional INL/DNL-like warping
#     * Spur injection (clock feedthrough / narrowband tones)
#     * Slew/derivative limiting (vectorized)
# - General channel impairments for robustness:
#     * random time shift, amplitude scale, phase rotation
#     * CFO, phase noise, AWGN
#     * optional light multipath fading
# - Anti-shortcut transforms:
#     * DC offset randomization
#     * band-edge tapering (Tukey-like window)
#
# Expected input: x shape [B, 2, L] where channel 0=I, 1=Q, L typically 1024
#
# Public API:
#   aug = build_augmentor()  # or RFFIAugmentor(...)
#   x_aug = aug(x, labels=y, no_dac=True)
#   x_dac, strength = aug(x, labels=y, dac_only=True, return_dac_strength=True)

import math
import random
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn.functional as F


def _nan_to_num_(x: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def _to_complex(x_iq: torch.Tensor) -> torch.Tensor:
    # x_iq: [B,2,L] float
    # return: [B,L] complex64
    I = x_iq[:, 0, :]
    Q = x_iq[:, 1, :]
    return torch.complex(I, Q)


def _from_complex(z: torch.Tensor) -> torch.Tensor:
    # z: [B,L] complex
    return torch.stack([z.real, z.imag], dim=1)


def _rand_uniform(shape, device, low=0.0, high=1.0, dtype=torch.float32):
    return (low + (high - low) * torch.rand(shape, device=device, dtype=dtype))


def _beta_sample(shape, device, a=2.0, b=1.0, dtype=torch.float32):
    # Approx Beta(a,b) using Gamma
    ga = torch.distributions.Gamma(torch.tensor([a], device=device, dtype=dtype), torch.tensor([1.0], device=device, dtype=dtype)).sample(shape).squeeze(-1)
    gb = torch.distributions.Gamma(torch.tensor([b], device=device, dtype=dtype), torch.tensor([1.0], device=device, dtype=dtype)).sample(shape).squeeze(-1)
    return ga / (ga + gb + 1e-12)


def _tukey_window(L: int, alpha: float, device, dtype=torch.float32) -> torch.Tensor:
    # Simple Tukey-like window, alpha in [0,1]
    if alpha <= 0:
        return torch.ones(L, device=device, dtype=dtype)
    if alpha >= 1:
        return torch.hann_window(L, periodic=False, device=device, dtype=dtype)

    w = torch.ones(L, device=device, dtype=dtype)
    edge = int(alpha * (L - 1) / 2.0)
    if edge <= 1:
        return w
    n = torch.arange(edge, device=device, dtype=dtype)
    ramp = 0.5 * (1 - torch.cos(math.pi * (n / (edge - 1))))
    w[:edge] = ramp
    w[-edge:] = ramp.flip(0)
    return w


def _apply_time_shift(x: torch.Tensor, max_shift: int) -> torch.Tensor:
    # x: [B,2,L], per-sample circular shift (vectorized gather)
    if max_shift <= 0:
        return x
    B, C, L = x.shape
    shifts = torch.randint(-max_shift, max_shift + 1, (B,), device=x.device)
    base = torch.arange(L, device=x.device).view(1, L).expand(B, L)
    idx = (base - shifts.view(B, 1)) % L  # inverse mapping
    idx = idx.view(B, 1, L).expand(B, C, L)
    return torch.gather(x, dim=2, index=idx)


def _apply_awgn(z: torch.Tensor, snr_db: torch.Tensor) -> torch.Tensor:
    # z: [B,L] complex, snr_db: [B,1]
    # noise variance based on signal power
    power = (z.real ** 2 + z.imag ** 2).mean(dim=1, keepdim=True).clamp_min(1e-12)
    snr = (10.0 ** (snr_db / 10.0)).clamp_min(1e-6)
    noise_var = power / snr
    sigma = torch.sqrt(noise_var / 2.0)
    n = torch.complex(
        torch.randn_like(z.real) * sigma,
        torch.randn_like(z.imag) * sigma
    )
    return z + n


def _apply_cfo(z: torch.Tensor, f_cyc_per_sample: torch.Tensor) -> torch.Tensor:
    # z: [B,L] complex, f: [B,1] cycles/sample
    B, L = z.shape
    n = torch.arange(L, device=z.device, dtype=z.real.dtype).view(1, L)
    phase = 2.0 * math.pi * f_cyc_per_sample * n
    rot = torch.complex(torch.cos(phase), torch.sin(phase))
    return z * rot


def _apply_phase_rot(z: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    # phi: [B,1] radians
    rot = torch.complex(torch.cos(phi), torch.sin(phi))
    return z * rot


def _apply_phase_noise(z: torch.Tensor, sigma_step: torch.Tensor) -> torch.Tensor:
    # Wiener phase noise: phi[n] = phi[n-1] + e[n], e~N(0, sigma_step^2)
    B, L = z.shape
    e = torch.randn(B, L, device=z.device, dtype=z.real.dtype) * sigma_step  # [B,L]
    phi = torch.cumsum(e, dim=1)
    rot = torch.complex(torch.cos(phi), torch.sin(phi))
    return z * rot


def _apply_multipath(z: torch.Tensor, num_taps: int, max_delay: int, k_factor: float = 0.0) -> torch.Tensor:
    # Light multipath: random complex FIR taps
    # z: [B,L]
    B, L = z.shape
    # delays 0..max_delay
    delays = torch.randint(0, max_delay + 1, (num_taps,), device=z.device)
    # complex tap gains
    hr = torch.randn(num_taps, device=z.device, dtype=z.real.dtype)
    hi = torch.randn(num_taps, device=z.device, dtype=z.real.dtype)
    h = torch.complex(hr, hi)
    # optional Rician LOS component at delay 0
    if k_factor > 0:
        h[0] = h[0] + torch.complex(torch.tensor([math.sqrt(k_factor)], device=z.device, dtype=z.real.dtype),
                                    torch.tensor([0.0], device=z.device, dtype=z.real.dtype)).squeeze(0)
    h = h / (torch.sqrt((h.real**2 + h.imag**2).sum()) + 1e-12)

    # Build impulse response length
    Lh = int(delays.max().item()) + 1
    hh = torch.zeros(Lh, device=z.device, dtype=torch.complex64)
    for i in range(num_taps):
        hh[int(delays[i].item())] = hh[int(delays[i].item())] + h[i].to(torch.complex64)

    # Convolution via real/imag split conv1d
    # y = z * hh (same length, circular-ish padding)
    pad = Lh - 1
    zr = z.real.unsqueeze(1)  # [B,1,L]
    zi = z.imag.unsqueeze(1)
    hr = hh.real.to(zr.dtype).view(1, 1, Lh)
    hi = hh.imag.to(zr.dtype).view(1, 1, Lh)

    # reflect pad to reduce edge artifacts
    zr_p = F.pad(zr, (pad, 0), mode="reflect")
    zi_p = F.pad(zi, (pad, 0), mode="reflect")

    yr = F.conv1d(zr_p, hr) - F.conv1d(zi_p, hi)
    yi = F.conv1d(zr_p, hi) + F.conv1d(zi_p, hr)
    y = torch.complex(yr.squeeze(1), yi.squeeze(1))
    return y


@dataclass
class DacParams:
    # strength sampling
    strength_dist: str = "sqrt"  # "sqrt" or "beta"
    beta_a: float = 2.0
    beta_b: float = 1.0

    # memory polynomial
    p_mem_poly: float = 0.6
    mem_K: int = 5
    mem_M: int = 3
    mem_coeff: float = 0.02

    # sampling jitter
    p_jitter: float = 0.6
    jitter_max: float = 0.003  # fraction of sample interval (0.003=0.3%)

    # AM/AM polynomial (memoryless)
    p_poly: float = 0.8
    poly_a3: float = 0.18
    poly_a5: float = 0.06

    # IQ imbalance (image term model)
    p_iq_imb: float = 0.6
    iq_img_max: float = 0.06  # image coefficient max

    # interleaving mismatch
    p_interleave: float = 0.7
    interleave_modes: Tuple[int, ...] = (2, 4)  # 2-way or 4-way
    inter_gain_max: float = 0.04
    inter_off_max: float = 0.01
    inter_skew_max: float = 0.08  # neighbor mix factor

    # quantization
    p_quant: float = 0.9
    bits_min: int = 8
    bits_max: int = 12
    dither: float = 0.003
    inl_warp: float = 0.05  # emulate INL/DNL-like mild warping

    # spurs
    p_spur: float = 0.4
    spur_num_min: int = 1
    spur_num_max: int = 3
    spur_amp_max: float = 0.02

    # derivative limiting (slew)
    p_slew: float = 0.4
    slew_max: float = 0.25  # max derivative magnitude (relative)


@dataclass
class PaParams:
    # strength sampling
    strength_dist: str = "sqrt"  # "sqrt" or "beta"
    beta_a: float = 2.0
    beta_b: float = 1.2

    # PA memory polynomial (recommended)
    p_mp: float = 0.9
    mp_order_max: int = 7   # odd orders up to this (3/5/7)
    mp_memory_max: int = 3  # taps (0..M-1)
    mp_sigma: float = 0.08  # coeff std (scaled by strength)

    # PA memory effect (simple random FIR)
    p_mem_fir: float = 0.4
    mem_taps_min: int = 2
    mem_taps_max: int = 5
    mem_delay_max: int = 3
    mem_sigma: float = 0.06  # tap std (scaled by strength)

    # Rapp AM/AM compression
    p_rapp: float = 0.9
    sat_min: float = 0.6
    sat_max: float = 1.1
    rapp_p_min: float = 2.0
    rapp_p_max: float = 5.0

    # AM/PM
    p_am_pm: float = 0.6
    ampm_max: float = 0.35  # radians max (scaled by strength)

    # optional mild IQ imbalance in PA chain (small, to mimic RF chain)
    p_iq_imb: float = 0.15
    iq_img_max: float = 0.03


class RFFIAugmentor:
    def __init__(
        self,
        # DAC
        p_dac: float = 0.3,
        dac: DacParams = DacParams(),

        # PA (power amplifier)
        p_pa: float = 0.0,
        pa: PaParams = PaParams(),
        enable_class_signature: bool = True,
        class_sig_mix: float = 0.35,  # 0..1, higher => more class-stable signature
        seed: int = 1337,

        # General channel impairments
        p_time_shift: float = 0.35,
        max_time_shift: int = 96,

        p_amp_scale: float = 0.50,
        amp_min: float = 0.85,
        amp_max: float = 1.15,

        p_phase_rot: float = 0.50,

        p_cfo: float = 0.45,
        cfo_max: float = 0.0025,  # cycles/sample (0.0025 ~ 0.25% of fs)

        p_phase_noise: float = 0.30,
        phase_noise_sigma_max: float = 0.012,  # rad/step (scaled by strength)

        p_awgn: float = 0.55,
        snr_min_db: float = 18.0,
        snr_max_db: float = 38.0,

        p_multipath: float = 0.20,
        mp_taps_min: int = 2,
        mp_taps_max: int = 5,
        mp_delay_max: int = 6,

        # Anti-shortcut
        p_dc_offset: float = 0.35,
        dc_offset_max: float = 0.03,

        p_bandedge_taper: float = 0.30,
        taper_alpha_min: float = 0.04,
        taper_alpha_max: float = 0.18,

        # For defect-only views (dac_only/pa_only/dac_pa): also apply channel + anti-shortcut
        defect_apply_channel: bool = True,
        stabilize_rms: bool = True,
        stabilize_min_gain: float = 0.35,
        stabilize_max_gain: float = 2.50,
        stabilize_abs_max: float = 6.0,
    ):
        self.p_dac = float(p_dac)
        self.dac = dac
        self.p_pa = float(p_pa)
        self.pa = pa
        self.enable_class_signature = bool(enable_class_signature)
        self.class_sig_mix = float(class_sig_mix)
        self.seed = int(seed)

        # general
        self.p_time_shift = float(p_time_shift)
        self.max_time_shift = int(max_time_shift)

        self.p_amp_scale = float(p_amp_scale)
        self.amp_min = float(amp_min)
        self.amp_max = float(amp_max)

        self.p_phase_rot = float(p_phase_rot)

        self.p_cfo = float(p_cfo)
        self.cfo_max = float(cfo_max)

        self.p_phase_noise = float(p_phase_noise)
        self.phase_noise_sigma_max = float(phase_noise_sigma_max)

        self.p_awgn = float(p_awgn)
        self.snr_min_db = float(snr_min_db)
        self.snr_max_db = float(snr_max_db)

        self.p_multipath = float(p_multipath)
        self.mp_taps_min = int(mp_taps_min)
        self.mp_taps_max = int(mp_taps_max)
        self.mp_delay_max = int(mp_delay_max)

        # anti-shortcut
        self.p_dc_offset = float(p_dc_offset)
        self.dc_offset_max = float(dc_offset_max)

        self.p_bandedge_taper = float(p_bandedge_taper)
        self.taper_alpha_min = float(taper_alpha_min)
        self.taper_alpha_max = float(taper_alpha_max)

        self.defect_apply_channel = bool(defect_apply_channel)
        self.stabilize_rms = bool(stabilize_rms)
        self.stabilize_min_gain = float(stabilize_min_gain)
        self.stabilize_max_gain = float(stabilize_max_gain)
        self.stabilize_abs_max = float(stabilize_abs_max)

        # cached class signatures
        self._class_sig: Dict[int, Dict[str, Any]] = {}

    # -------- class signature --------
    def _get_class_sig(self, c: int) -> Dict[str, Any]:
        if c in self._class_sig:
            return self._class_sig[c]
        rng = random.Random(self.seed + int(c) * 10007)

        sig = {
            # base strength bias
            "s_bias": rng.random(),  # 0..1

            # IQ image term amplitude/phase
            "iq_img_base": (rng.random() * 2 - 1) * self.dac.iq_img_max,
            "iq_phase_base": rng.random() * 2 * math.pi,

            # interleaving base
            "inter_gain_base": (rng.random() * 2 - 1) * self.dac.inter_gain_max,
            "inter_off_base": (rng.random() * 2 - 1) * self.dac.inter_off_max,
            "inter_skew_base": rng.random() * self.dac.inter_skew_max,

            # spur base freqs/phases
            "spur_f": [rng.random() * 0.45 for _ in range(3)],  # cycles/sample
            "spur_phi": [rng.random() * 2 * math.pi for _ in range(3)],
        }
        self._class_sig[c] = sig
        return sig

    # -------- DAC components --------
    def _sample_strength(self, B: int, device, dtype=torch.float32) -> torch.Tensor:
        if self.dac.strength_dist == "beta":
            u = _beta_sample((B, 1), device=device, a=self.dac.beta_a, b=self.dac.beta_b, dtype=dtype)
            return u
        # default sqrt-biased (more strong cases)
        u = torch.rand((B, 1), device=device, dtype=dtype)
        return torch.sqrt(u + 1e-12)

    def _sample_strength_pa(self, B: int, device, dtype=torch.float32) -> torch.Tensor:
        if self.pa.strength_dist == "beta":
            u = _beta_sample((B, 1), device=device, a=self.pa.beta_a, b=self.pa.beta_b, dtype=dtype)
            return u
        u = torch.rand((B, 1), device=device, dtype=dtype)
        return torch.sqrt(u + 1e-12)

    @staticmethod
    def _batch_rms(x: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(torch.mean(x.float() * x.float(), dim=(1, 2), keepdim=True).clamp_min(1e-12))

    def _stabilize_view(self, x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        out = _nan_to_num_(x.float())
        ref_f = _nan_to_num_(ref.float())
        if self.stabilize_rms:
            ref_rms = self._batch_rms(ref_f)
            out_rms = self._batch_rms(out)
            scale = (ref_rms / out_rms.clamp_min(1e-12)).clamp(
                min=self.stabilize_min_gain,
                max=self.stabilize_max_gain,
            )
            out = out * scale
        if self.stabilize_abs_max > 0:
            out = out.clamp(min=-self.stabilize_abs_max, max=self.stabilize_abs_max)
        return _nan_to_num_(out)

    def _sampling_jitter(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        # x: [B,2,L], s: [B,1]
        if self.dac.jitter_max <= 0:
            return x
        B, C, L = x.shape
        # jitter scalar per sample
        jitter = torch.randn((B, 1), device=x.device, dtype=x.dtype).clamp(-3.0, 3.0)
        jitter = jitter * (self.dac.jitter_max * s)  # fraction

        t = torch.linspace(0, L - 1, L, device=x.device, dtype=x.dtype).view(1, 1, L)  # [1,1,L]
        # simple smooth warp
        t_warp = t + jitter.view(B, 1, 1) * torch.sin(2 * math.pi * t / L)

        t0 = t_warp.floor().clamp(0, L - 2).long()
        w = (t_warp - t0.float()).clamp(0, 1)

        idx0 = t0.expand(B, C, L)
        idx1 = (t0 + 1).expand(B, C, L)

        x0 = torch.gather(x, 2, idx0)
        x1 = torch.gather(x, 2, idx1)
        return x0 * (1 - w) + x1 * w

    def _memory_poly(self, z: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        # z: [B,L] complex, s: [B,1] float
        B, L = z.shape
        y = torch.zeros_like(z)

        K = int(self.dac.mem_K)
        M = int(self.dac.mem_M)
        coeff = float(self.dac.mem_coeff)

        for m in range(M):
            z_m = torch.roll(z, shifts=m, dims=1)
            r = torch.abs(z_m).clamp_min(1e-8)
            for k in range(1, K + 1):
                w = 1.0 if (k % 2 == 1) else 0.3
                sigma = (coeff * w * (1.0 / (1 + m))) * s  # [B,1]
                a = torch.randn((B, 1), device=z.device, dtype=z.real.dtype) * sigma
                y = y + a * z_m * (r ** (k - 1))
        return z + y

    def _poly_nonlinearity(self, z: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        # AM/AM-like polynomial: z + a3 z|z|^2 + a5 z|z|^4
        r2 = (z.real ** 2 + z.imag ** 2).clamp_min(1e-12)
        a3 = (self.dac.poly_a3 * s).to(z.real.dtype)
        a5 = (self.dac.poly_a5 * s).to(z.real.dtype)
        return z + a3 * z * r2 + a5 * z * (r2 ** 2)

    def _iq_imbalance(self, z: torch.Tensor, s: torch.Tensor, labels: Optional[torch.Tensor]) -> torch.Tensor:
        # Image term model: z' = z + k * exp(j*theta) * conj(z)
        B, L = z.shape
        k = _rand_uniform((B, 1), z.device, low=-1.0, high=1.0, dtype=z.real.dtype) * (self.dac.iq_img_max * s)
        theta = _rand_uniform((B, 1), z.device, low=0.0, high=2 * math.pi, dtype=z.real.dtype)

        if self.enable_class_signature and labels is not None:
            # Mix in class-stable bias
            k_base = torch.zeros((B, 1), device=z.device, dtype=z.real.dtype)
            th_base = torch.zeros((B, 1), device=z.device, dtype=z.real.dtype)
            for i in range(B):
                sig = self._get_class_sig(int(labels[i].item()))
                k_base[i, 0] = sig["iq_img_base"]
                th_base[i, 0] = sig["iq_phase_base"]
            mix = self.class_sig_mix
            k = (1 - mix) * k + mix * k_base.to(k.dtype)
            theta = (1 - mix) * theta + mix * th_base.to(theta.dtype)

        rot = torch.complex(torch.cos(theta), torch.sin(theta))
        return z + torch.complex(k, torch.zeros_like(k)) * rot * torch.conj(z)

    def _interleave_mismatch(self, z: torch.Tensor, s: torch.Tensor, labels: Optional[torch.Tensor]) -> torch.Tensor:
        # 2-way or 4-way: per-phase gain/offset, plus skew (neighbor mix)
        B, L = z.shape
        mode = random.choice(self.dac.interleave_modes)

        gain = _rand_uniform((B, mode), z.device, low=-1.0, high=1.0, dtype=z.real.dtype) * (self.dac.inter_gain_max * s)
        off = _rand_uniform((B, mode), z.device, low=-1.0, high=1.0, dtype=z.real.dtype) * (self.dac.inter_off_max * s)
        skew = _rand_uniform((B, 1), z.device, low=0.0, high=self.dac.inter_skew_max, dtype=z.real.dtype) * s

        if self.enable_class_signature and labels is not None:
            g_base = torch.zeros((B, 1), device=z.device, dtype=z.real.dtype)
            o_base = torch.zeros((B, 1), device=z.device, dtype=z.real.dtype)
            sk_base = torch.zeros((B, 1), device=z.device, dtype=z.real.dtype)
            for i in range(B):
                sig = self._get_class_sig(int(labels[i].item()))
                g_base[i, 0] = sig["inter_gain_base"]
                o_base[i, 0] = sig["inter_off_base"]
                sk_base[i, 0] = sig["inter_skew_base"]
            mix = self.class_sig_mix
            # broadcast to mode phases
            gain = (1 - mix) * gain + mix * g_base.expand(B, mode)
            off = (1 - mix) * off + mix * o_base.expand(B, mode)
            skew = (1 - mix) * skew + mix * sk_base

        idx = torch.arange(L, device=z.device) % mode  # [L]
        # apply gain/offset per phase
        g = gain[:, idx]  # [B,L]
        o = off[:, idx]
        z2 = z * (1.0 + torch.complex(g, torch.zeros_like(g))) + torch.complex(o, torch.zeros_like(o))

        # skew: neighbor mixing to approximate sampling instant mismatch
        if self.dac.inter_skew_max > 0:
            z_prev = torch.roll(z2, shifts=1, dims=1)
            z2 = (1.0 - skew) * z2 + skew * z_prev
        return z2

    def _quantize(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        # x: [B,2,L] float
        B, C, L = x.shape
        # soft clip
        x = torch.tanh(x)

        # optional INL/DNL-like mild warping
        if self.dac.inl_warp > 0:
            warp = (self.dac.inl_warp * s).view(B, 1, 1)
            x = x + warp * (x ** 3 - x)

        # bit depth random
        bits = torch.randint(self.dac.bits_min, self.dac.bits_max + 1, (B,), device=x.device)
        # quant step per sample
        # scale to [-1,1]
        # add dither
        if self.dac.dither > 0:
            x = x + torch.randn_like(x) * (self.dac.dither * s.view(B, 1, 1))
        # quantize per sample with its bits
        # step = 2/(2^bits - 1)
        levels = (2.0 ** bits.float() - 1.0).clamp_min(1.0)  # [B]
        step = (2.0 / levels).view(B, 1, 1)
        xq = torch.round((x + 1.0) / step) * step - 1.0
        return xq

    def _spur_inject(self, z: torch.Tensor, s: torch.Tensor, labels: Optional[torch.Tensor]) -> torch.Tensor:
        B, L = z.shape
        num = random.randint(self.dac.spur_num_min, self.dac.spur_num_max)
        # base freqs/phases
        f = _rand_uniform((B, num), z.device, low=0.02, high=0.45, dtype=z.real.dtype)
        phi = _rand_uniform((B, num), z.device, low=0.0, high=2 * math.pi, dtype=z.real.dtype)

        if self.enable_class_signature and labels is not None:
            f_base = torch.zeros((B, num), device=z.device, dtype=z.real.dtype)
            p_base = torch.zeros((B, num), device=z.device, dtype=z.real.dtype)
            for i in range(B):
                sig = self._get_class_sig(int(labels[i].item()))
                for j in range(num):
                    f_base[i, j] = sig["spur_f"][j % len(sig["spur_f"])]
                    p_base[i, j] = sig["spur_phi"][j % len(sig["spur_phi"])]
            mix = self.class_sig_mix
            f = (1 - mix) * f + mix * f_base
            phi = (1 - mix) * phi + mix * p_base

        amp = _rand_uniform((B, num), z.device, low=0.0, high=self.dac.spur_amp_max, dtype=z.real.dtype) * s
        n = torch.arange(L, device=z.device, dtype=z.real.dtype).view(1, L)  # [1,L]
        # sum of spurs
        spur = 0.0
        for j in range(num):
            phase = 2.0 * math.pi * f[:, j:j+1] * n + phi[:, j:j+1]
            tone = torch.complex(torch.cos(phase), torch.sin(phase))
            spur = spur + torch.complex(amp[:, j:j+1], torch.zeros_like(amp[:, j:j+1])) * tone
        return z + spur

    def _slew_limit(self, z: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        # Vectorized derivative limiting:
        # y[0]=z[0], y[n]=y[n-1] + clamp(dz, |dz|<=sr)
        dz = z[:, 1:] - z[:, :-1]  # [B,L-1]
        mag = torch.abs(dz).clamp_min(1e-12)
        sr = (self.dac.slew_max * s).clamp_min(1e-6)  # [B,1]
        scale = torch.minimum(torch.ones_like(mag), (sr / mag))  # [B,L-1]
        dz2 = dz * scale
        y = torch.zeros_like(z)
        y[:, 0] = z[:, 0]
        y[:, 1:] = z[:, 0:1] + torch.cumsum(dz2, dim=1)
        return y

    def simulate_dac(
        self,
        x: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        dac_strength: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply DAC impairment chain to x.

        Args:
            x: [B,2,L]
            labels: [B] optional, enables class-stable signature
            dac_strength: [B] or [B,1] optional fixed strength
        Returns:
            x_dac: [B,2,L]
            strength: [B]
        """
        # Work in float32 for stability
        x0 = _nan_to_num_(x.float())
        B, _, L = x0.shape
        device = x0.device
        dtype = x0.dtype

        # strength
        if dac_strength is None:
            s = self._sample_strength(B, device, dtype=dtype)  # [B,1]
        else:
            s = dac_strength.view(B, 1).to(device=device, dtype=dtype).clamp(0.0, 1.0)

        # mix in class bias to make per-class "signature" more stable if desired
        if self.enable_class_signature and labels is not None:
            s_base = torch.zeros((B, 1), device=device, dtype=dtype)
            for i in range(B):
                sig = self._get_class_sig(int(labels[i].item()))
                s_base[i, 0] = sig["s_bias"]
            mix = self.class_sig_mix
            s = (1 - mix) * s + mix * s_base

        s = s.clamp(0.0, 1.0)
        strength = s.view(B)

        # ---- chain begins ----
        # 1) sampling jitter on I/Q
        if random.random() < self.dac.p_jitter:
            x0 = self._sampling_jitter(x0, s)

        z = _to_complex(x0)  # [B,L] complex

        # 2) memory polynomial (nonlinearity with memory)
        if random.random() < self.dac.p_mem_poly:
            z = self._memory_poly(z, s)

        # 3) memoryless polynomial AM/AM
        if random.random() < self.dac.p_poly:
            z = self._poly_nonlinearity(z, s)

        # 4) IQ imbalance (image)
        if random.random() < self.dac.p_iq_imb:
            z = self._iq_imbalance(z, s, labels)

        # 5) interleaving mismatch
        if random.random() < self.dac.p_interleave:
            z = self._interleave_mismatch(z, s, labels)

        x1 = _from_complex(z).to(dtype)

        # 6) quantization + dither (+ mild INL warp)
        if random.random() < self.dac.p_quant:
            x1 = self._quantize(x1, s)

        z = _to_complex(x1)

        # 7) spur injection (after quant to mimic clock feedthrough etc.)
        if random.random() < self.dac.p_spur:
            z = self._spur_inject(z, s, labels)

        # 8) slew/derivative limiting (vectorized)
        if random.random() < self.dac.p_slew:
            z = self._slew_limit(z, s)

        out = _from_complex(z).to(dtype)
        out = self._stabilize_view(out, x0).to(dtype)
        return out, strength

    # ----------------------- PA (power amplifier) simulation -----------------------
    def _pa_apply_fir_memory(self, z: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """Simple random complex FIR to emulate PA memory effects."""
        if self.pa.mem_sigma <= 0:
            return z
        B, L = z.shape
        device = z.device
        dtype = z.real.dtype

        taps = random.randint(self.pa.mem_taps_min, self.pa.mem_taps_max)
        max_d = int(self.pa.mem_delay_max)
        delays = torch.randint(0, max_d + 1, (B, taps), device=device)

        sigma = (self.pa.mem_sigma * s).clamp_min(0.0).to(dtype)
        wr = torch.randn((B, taps), device=device, dtype=dtype) * sigma
        wi = torch.randn((B, taps), device=device, dtype=dtype) * sigma
        w = torch.complex(wr, wi)

        y = z
        for k in range(taps):
            # use per-batch median delay for vectorized roll
            d = int(delays[:, k].median().item())
            y = y + w[:, k:k+1] * torch.roll(z, shifts=d, dims=1)
        return y

    def _pa_rapp(self, z: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """Rapp AM/AM model: soft compression."""
        B, L = z.shape
        dtype = z.real.dtype
        a_sat = _rand_uniform((B, 1), z.device, low=self.pa.sat_min, high=self.pa.sat_max, dtype=dtype)
        a_eff = a_sat * (1.0 - 0.55 * s).clamp(0.15, 1.0)
        p = _rand_uniform((B, 1), z.device, low=self.pa.rapp_p_min, high=self.pa.rapp_p_max, dtype=dtype)

        r = torch.abs(z).clamp_min(1e-8)
        t = (r / a_eff.clamp_min(1e-6)) ** (2.0 * p)
        g = (1.0 + t).pow(-1.0 / (2.0 * p))
        return z * g

    def _pa_am_pm(self, z: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """AM/PM distortion: phase shift grows with amplitude."""
        B, L = z.shape
        dtype = z.real.dtype
        r = torch.abs(z).clamp_min(1e-8)
        k = _rand_uniform((B, 1), z.device, low=-1.0, high=1.0, dtype=dtype) * (float(self.pa.ampm_max) * s)
        a = torch.tanh(r)
        phi = k * (a ** 2)
        rot = torch.complex(torch.cos(phi), torch.sin(phi))
        return z * rot


    def _pa_memory_polynomial(self, z: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """Memory Polynomial PA model (baseband complex).

        y[n] = sum_{m=0..M-1} sum_{k in {1,3,5,7}} a_{k,m} x[n-m] |x[n-m]|^{k-1}

        Coefficients are random per-batch (for diversity) and scaled by strength s in [0,1].
        """
        B, L = z.shape
        device = z.device
        dtype = z.dtype
        eps = 1e-8

        max_order = int(self.pa.mp_order_max)
        orders = [1] + [k for k in (3, 5, 7) if k <= max_order]
        M = int(self.pa.mp_memory_max)
        M = max(1, min(6, M))

        def delay(x: torch.Tensor, d: int) -> torch.Tensor:
            if d <= 0:
                return x
            pad = torch.zeros((B, d), device=device, dtype=dtype)
            return torch.cat([pad, x[:, :-d]], dim=1)

        g0 = 1.0 + 0.02 * torch.randn((B, 1), device=device, dtype=torch.float32)
        g0 = g0.to(dtype)
        y = g0 * z

        sigma = float(self.pa.mp_sigma)
        s2 = s.clamp(0.0, 1.0)
        scale = (s2 * sigma).to(dtype)

        for m in range(M):
            xm = delay(z, m)
            r = torch.abs(xm).clamp_min(eps)
            for k in orders:
                if k == 1 and m == 0:
                    continue
                cr = torch.randn((B, 1), device=device, dtype=torch.float32)
                ci = torch.randn((B, 1), device=device, dtype=torch.float32)
                c = (cr + 1j * ci).to(dtype) * scale
                y = y + c * xm * (r ** (k - 1))

        y = y / (1.0 + 0.5 * s2.to(dtype))
        return y

    def simulate_pa(
        self,
        x: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        pa_strength: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply PA impairment chain to x."""
        x0 = _nan_to_num_(x.float())
        B, _, L = x0.shape
        device = x0.device
        dtype = x0.dtype

        if pa_strength is None:
            s = self._sample_strength_pa(B, device, dtype=dtype)
        else:
            s = pa_strength.view(B, 1).to(device=device, dtype=dtype).clamp(0.0, 1.0)

        s = s.clamp(0.0, 1.0)
        strength = s.view(B)

        z = _to_complex(x0)

        if random.random() < self.pa.p_mem_fir:
            z = self._pa_apply_fir_memory(z, s)

        if random.random() < self.pa.p_mp:
            z = self._pa_memory_polynomial(z, s)

        if random.random() < self.pa.p_rapp:
            z = self._pa_rapp(z, s)

        if random.random() < self.pa.p_am_pm:
            z = self._pa_am_pm(z, s)

        if random.random() < self.pa.p_iq_imb:
            saved = self.dac.iq_img_max
            try:
                self.dac.iq_img_max = float(self.pa.iq_img_max)
                z = self._iq_imbalance(z, s, labels)
            finally:
                self.dac.iq_img_max = saved

        out = _from_complex(z).to(dtype)
        out = self._stabilize_view(out, x0).to(dtype)
        return out, strength

    def simulate_dac_pa(
        self,
        x: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        dac_strength: Optional[torch.Tensor] = None,
        pa_strength: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply DAC then PA sequentially."""
        x_d, s_d = self.simulate_dac(x, labels=labels, dac_strength=dac_strength)
        x_p, s_p = self.simulate_pa(x_d, labels=labels, pa_strength=pa_strength)
        return x_p, s_d, s_p

    def _apply_antis_shortcut(self, x: torch.Tensor) -> torch.Tensor:
        B, C, L = x.shape
        out = x

        if self.p_dc_offset > 0 and torch.rand(()) < self.p_dc_offset:
            dc = _rand_uniform((B, 2, 1), x.device, low=-self.dc_offset_max, high=self.dc_offset_max, dtype=out.dtype)
            out = out + dc

        if self.p_bandedge_taper > 0 and torch.rand(()) < self.p_bandedge_taper:
            alpha = float(_rand_uniform((), x.device, low=self.taper_alpha_min, high=self.taper_alpha_max, dtype=out.dtype).item())
            w = _tukey_window(L, alpha=alpha, device=x.device, dtype=out.dtype).view(1, 1, L)
            out = out * w

        return out

    def _apply_general_channel(self, x: torch.Tensor) -> torch.Tensor:
        x_ref = _nan_to_num_(x.float())
        x = x_ref

        if self.p_time_shift > 0 and torch.rand(()) < self.p_time_shift:
            x = _apply_time_shift(x, self.max_time_shift)

        if self.p_amp_scale > 0 and torch.rand(()) < self.p_amp_scale:
            B = x.size(0)
            a = _rand_uniform((B, 1, 1), x.device, low=self.amp_min, high=self.amp_max, dtype=x.dtype)
            x = x * a

        z = _to_complex(x)

        if self.p_phase_rot > 0 and torch.rand(()) < self.p_phase_rot:
            B = z.size(0)
            phi = _rand_uniform((B, 1), z.device, low=0.0, high=2 * math.pi, dtype=z.real.dtype)
            z = _apply_phase_rot(z, phi)

        if self.p_cfo > 0 and torch.rand(()) < self.p_cfo:
            B = z.size(0)
            f = _rand_uniform((B, 1), z.device, low=-self.cfo_max, high=self.cfo_max, dtype=z.real.dtype)
            z = _apply_cfo(z, f)

        if self.p_phase_noise > 0 and torch.rand(()) < self.p_phase_noise:
            B = z.size(0)
            sigma = _rand_uniform((B, 1), z.device, low=0.0, high=self.phase_noise_sigma_max, dtype=z.real.dtype)
            z = _apply_phase_noise(z, sigma)

        if self.p_multipath > 0 and torch.rand(()) < self.p_multipath:
            taps = random.randint(self.mp_taps_min, self.mp_taps_max)
            z = _apply_multipath(z, num_taps=taps, max_delay=self.mp_delay_max, k_factor=0.0)

        if self.p_awgn > 0 and torch.rand(()) < self.p_awgn:
            B = z.size(0)
            snr = _rand_uniform((B, 1), z.device, low=self.snr_min_db, high=self.snr_max_db, dtype=z.real.dtype)
            z = _apply_awgn(z, snr)

        out = _from_complex(z).to(x.dtype)
        out = self._stabilize_view(out, x_ref).to(x.dtype)
        return out

    def __call__(
        self,
        x: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        return_dac_strength: bool = False,
        return_pa_strength: bool = False,
        return_defect_strengths: bool = False,
        no_dac: bool = False,
        no_pa: bool = True,
        dac_only: bool = False,
        pa_only: bool = False,
        dac_pa: bool = False,
    ):
        """
        x: [B,2,L]
        Special defect views:
          - dac_only: ONLY DAC chain, returns (x, s_dac)
          - pa_only:  ONLY PA chain,  returns (x, s_pa)
          - dac_pa:   DAC then PA,    returns (x, s_dac, s_pa)
        Normal view:
          anti-shortcut + general channel (+ optional DAC/PA gates)
        """
        assert x.dim() == 3 and x.size(1) == 2, "Expected x shape [B,2,L]"
        dtype_out = x.dtype
        x0 = x.float()

        if dac_only:
            x_aug, s_d = self.simulate_dac(x0, labels=labels, dac_strength=None)
            if self.defect_apply_channel:
                x_aug = self._apply_antis_shortcut(x_aug)
                x_aug = self._apply_general_channel(x_aug)
            x_aug = x_aug.to(dtype_out)
            if return_defect_strengths:
                return x_aug, s_d.to(dtype_out), torch.zeros_like(s_d).to(dtype_out)
            if return_dac_strength:
                return x_aug, s_d.to(dtype_out)
            return x_aug

        if pa_only:
            x_aug, s_p = self.simulate_pa(x0, labels=labels, pa_strength=None)
            if self.defect_apply_channel:
                x_aug = self._apply_antis_shortcut(x_aug)
                x_aug = self._apply_general_channel(x_aug)
            x_aug = x_aug.to(dtype_out)
            if return_defect_strengths:
                return x_aug, torch.zeros_like(s_p).to(dtype_out), s_p.to(dtype_out)
            if return_pa_strength:
                return x_aug, s_p.to(dtype_out)
            return x_aug

        if dac_pa:
            x_aug, s_d, s_p = self.simulate_dac_pa(x0, labels=labels, dac_strength=None, pa_strength=None)
            if self.defect_apply_channel:
                x_aug = self._apply_antis_shortcut(x_aug)
                x_aug = self._apply_general_channel(x_aug)
            x_aug = x_aug.to(dtype_out)
            if return_defect_strengths:
                return x_aug, s_d.to(dtype_out), s_p.to(dtype_out)
            if return_dac_strength and (not return_pa_strength):
                return x_aug, s_d.to(dtype_out)
            if return_pa_strength and (not return_dac_strength):
                return x_aug, s_p.to(dtype_out)
            return x_aug

        x0 = self._apply_antis_shortcut(x0)
        x0 = self._apply_general_channel(x0)

        B = x0.size(0)
        s_d = torch.zeros((B,), device=x0.device, dtype=torch.float32)
        s_p = torch.zeros((B,), device=x0.device, dtype=torch.float32)

        if (not no_dac) and (self.p_dac > 0):
            mask = (torch.rand((B,), device=x0.device) < self.p_dac)
            if mask.any():
                x_sel = x0[mask]
                y_sel = labels[mask] if labels is not None else None
                x_d, sd = self.simulate_dac(x_sel, labels=y_sel, dac_strength=None)
                x0 = x0.clone()
                x0[mask] = x_d
                s_d[mask] = sd

        if (not no_pa) and (self.p_pa > 0):
            mask = (torch.rand((B,), device=x0.device) < self.p_pa)
            if mask.any():
                x_sel = x0[mask]
                y_sel = labels[mask] if labels is not None else None
                x_p, sp = self.simulate_pa(x_sel, labels=y_sel, pa_strength=None)
                x0 = x0.clone()
                x0[mask] = x_p
                s_p[mask] = sp

        x0 = _nan_to_num_(x0).to(dtype_out)

        if return_defect_strengths:
            return x0, s_d.to(dtype_out), s_p.to(dtype_out)
        if return_dac_strength:
            return x0, s_d.to(dtype_out)
        if return_pa_strength:
            return x0, s_p.to(dtype_out)
        return x0


# Backward-compatible aliases
DataAugmentationV2 = RFFIAugmentor
DataAugmentation = RFFIAugmentor
Augmentor = RFFIAugmentor


def build_augmentor(**kwargs) -> RFFIAugmentor:
    """
    Factory used by train.py (if present).
    You can override any RFFIAugmentor args here.
    """
    return RFFIAugmentor(**kwargs)

# ============================================================
# Receiver-chain Domain Randomization (Source-only DG)
# ============================================================
def _apply_agc_softclip(z: torch.Tensor, gain_db: torch.Tensor, sat: torch.Tensor) -> torch.Tensor:
    """
    z: [B,L] complex
    gain_db: [B,1] dB
    sat: [B,1] positive scalar (soft clip level in linear amplitude)
    """
    g = (10.0 ** (gain_db / 20.0)).clamp(1e-3, 1e3)
    z = z * g
    # soft clip: sat * tanh(z/sat)
    s = sat.clamp_min(1e-3)
    zr = s * torch.tanh(z.real / s)
    zi = s * torch.tanh(z.imag / s)
    return torch.complex(zr, zi)


def _apply_sro_resample(z: torch.Tensor, ppm: torch.Tensor) -> torch.Tensor:
    """
    Simple linear resample to simulate sampling-rate offset (SRO).
    z: [B,L] complex
    ppm: [B,1] parts-per-million (positive stretches time, negative compresses)
    returns: [B,L] complex
    """
    B, L = z.shape
    # time scaling factor
    a = (1.0 + ppm * 1e-6).clamp(0.999, 1.001)  # safe range
    # sample positions in original index space
    n = torch.arange(L, device=z.device, dtype=torch.float32).view(1, L) * a.float()
    n0 = torch.floor(n).long()
    n1 = (n0 + 1).clamp(max=L - 1)
    n0 = n0.clamp(min=0, max=L - 1)
    w = (n - n0.float()).clamp(0.0, 1.0)

    # gather (vectorized)
    idx0 = n0.expand(B, L)
    idx1 = n1.expand(B, L)
    z0 = torch.gather(z, 1, idx0)
    z1 = torch.gather(z, 1, idx1)
    return z0 * (1.0 - w) + z1 * w


def _raised_cosine_lowpass_mask(L: int, fs: float, cutoff_hz: torch.Tensor, transition_hz: torch.Tensor, device, dtype=torch.float32) -> torch.Tensor:
    """
    Create lowpass mask in FFT-shifted domain for each sample.
    cutoff_hz: [B,1], transition_hz: [B,1]
    Returns mask: [B, L] float in [0,1], fftshift order (freq from -fs/2..fs/2)
    """
    B = cutoff_hz.shape[0]
    # frequency axis in Hz, fftshifted
    f = torch.fft.fftshift(torch.fft.fftfreq(L, d=1.0/fs).to(device=device)).view(1, L)  # [1,L]
    af = f.abs()  # [1,L]
    fc = cutoff_hz.clamp_min(1.0)  # [B,1]
    tw = transition_hz.clamp_min(1.0)  # [B,1]
    f1 = fc
    f2 = fc + tw
    afB = af.expand(B, L)

    mask = torch.ones((B, L), device=device, dtype=dtype)
    # stopband
    mask = torch.where(afB >= f2, torch.zeros_like(mask), mask)
    # transition band
    in_trans = (afB >= f1) & (afB < f2)
    t = ((afB - f1) / (f2 - f1 + 1e-12)).clamp(0.0, 1.0)
    # raised cosine from 1 to 0
    rc = 0.5 * (1.0 + torch.cos(math.pi * t))
    mask = torch.where(in_trans, rc.to(dtype), mask)
    return mask


def _apply_random_lowpass(z: torch.Tensor, fs: float, cutoff_hz: torch.Tensor, transition_hz: torch.Tensor) -> torch.Tensor:
    """
    FFT-domain lowpass with raised-cosine transition.
    z: [B,L] complex
    cutoff_hz, transition_hz: [B,1]
    """
    B, L = z.shape
    Z = torch.fft.fftshift(torch.fft.fft(z, dim=1), dim=1)
    mask = _raised_cosine_lowpass_mask(L, fs, cutoff_hz, transition_hz, device=z.device, dtype=Z.real.dtype)
    Z = Z * mask.to(Z.dtype)
    z_f = torch.fft.ifft(torch.fft.ifftshift(Z, dim=1), dim=1)
    return z_f


def _apply_multipath_fir(z: torch.Tensor, taps: torch.Tensor) -> torch.Tensor:
    """
    z: [B,L] complex
    taps: [B,T] complex taps
    returns: [B,L] complex (same length, 'same' convolution)
    """
    B, L = z.shape
    T = taps.shape[1]
    # conv1d expects [B,1,L]
    zr = z.real.unsqueeze(1)
    zi = z.imag.unsqueeze(1)

    # Build real/imag kernels per sample using grouped conv:
    # output real = zr*hr - zi*hi, imag = zr*hi + zi*hr
    hr = taps.real.flip(1).unsqueeze(1)  # [B,1,T]
    hi = taps.imag.flip(1).unsqueeze(1)

    # Use group conv: reshape to [1, B, L], weight [B,1,T], groups=B
    zr_g = zr.transpose(0,1)  # [1,B,L]
    zi_g = zi.transpose(0,1)

    pad = T//2
    # conv
    rr = F.conv1d(zr_g, hr, padding=pad, groups=B)
    ii = F.conv1d(zi_g, hi, padding=pad, groups=B)
    ri = F.conv1d(zr_g, hi, padding=pad, groups=B)
    ir = F.conv1d(zi_g, hr, padding=pad, groups=B)

    out_r = (rr - ii).transpose(0,1).squeeze(1)[:, :L]
    out_i = (ri + ir).transpose(0,1).squeeze(1)[:, :L]
    return torch.complex(out_r, out_i)


def sample_dg_env_params(env_id: torch.Tensor, fs: float) -> Dict[str, torch.Tensor]:
    """
    env_id: [B] int
    Returns dict of per-sample parameters for receiver-chain DR.
    """
    device = env_id.device
    B = env_id.numel()
    env = env_id.view(B,1)

    # ranges per env (0 mild -> 3 strong)
    # CFO (Hz)
    cfo_max = torch.where(env==0,  2000.0,
              torch.where(env==1,  8000.0,
              torch.where(env==2, 20000.0, 35000.0)))
    # SRO (ppm)
    sro_max = torch.where(env==0,   5.0,
              torch.where(env==1,  25.0,
              torch.where(env==2,  80.0, 150.0)))
    # Gain (dB)
    g_max = torch.where(env==0,  2.0,
            torch.where(env==1,  4.0,
            torch.where(env==2,  6.0,  8.0)))
    # Soft clip level (linear)
    sat_base = torch.where(env==0, 2.5,
               torch.where(env==1, 2.0,
               torch.where(env==2, 1.7, 1.4)))

    # Lowpass cutoff (fraction of Nyquist), transition (fraction of Nyquist)
    # Nyquist = fs/2
    nyq = fs/2.0
    cutoff_frac_lo = torch.where(env==0, 0.85,
                      torch.where(env==1, 0.75,
                      torch.where(env==2, 0.65, 0.55)))
    cutoff_frac_hi = torch.where(env==0, 0.98,
                      torch.where(env==1, 0.90,
                      torch.where(env==2, 0.80, 0.70)))
    trans_frac = torch.where(env==0, 0.03,
                 torch.where(env==1, 0.05,
                 torch.where(env==2, 0.08, 0.10)))

    # Multipath taps
    taps_len = torch.where(env==0, 3,
               torch.where(env==1, 5,
               torch.where(env==2, 7, 9))).long()

    # Sample random values
    u = torch.rand((B,1), device=device)
    cfo = (2*u - 1.0) * cfo_max.to(torch.float32)  # [-max,max]
    u2 = torch.rand((B,1), device=device)
    sro = (2*u2 - 1.0) * sro_max.to(torch.float32)
    u3 = torch.rand((B,1), device=device)
    gain_db = (2*u3 - 1.0) * g_max.to(torch.float32)

    sat = sat_base.to(torch.float32) * (0.8 + 0.4*torch.rand((B,1), device=device))
    cutoff_hz = (cutoff_frac_lo + (cutoff_frac_hi - cutoff_frac_lo) * torch.rand((B,1), device=device)) * nyq
    transition_hz = trans_frac.to(torch.float32) * nyq

    return {
        "cfo_hz": cfo,
        "sro_ppm": sro,
        "gain_db": gain_db,
        "sat": sat,
        "cutoff_hz": cutoff_hz,
        "transition_hz": transition_hz,
        "taps_len": taps_len,
    }


def apply_receiver_dg(x_iq: torch.Tensor, fs: float, env_id: torch.Tensor, p_lowpass: float = 0.7, p_multipath: float = 0.7) -> torch.Tensor:
    """
    Receiver-chain domain randomization.
    x_iq: [B,2,L] float
    env_id: [B] int in [0, dg_envs)
    """
    if x_iq is None:
        return x_iq
    B, C, L = x_iq.shape
    if B == 0:
        return x_iq

    # operate in fp32 for stability
    x_f = x_iq.float()
    z = _to_complex(x_f)  # [B,L]
    params = sample_dg_env_params(env_id, fs)

    # SRO -> CFO -> AGC/softclip -> (optional) multipath -> (optional) lowpass
    z = _apply_sro_resample(z, params["sro_ppm"])
    z = _apply_cfo(z, (params["cfo_hz"] / float(fs)).to(z.real.dtype))

    z = _apply_agc_softclip(z, params["gain_db"], params["sat"])

    # multipath
    if p_multipath > 0:
        do_mp = (torch.rand((B,1), device=z.device) < p_multipath)
        if do_mp.any():
            # build taps per sample with variable length; pad to maxT
            maxT = int(params["taps_len"].max().item())
            taps = torch.zeros((B, maxT), device=z.device, dtype=z.dtype)
            for bi in range(B):
                T = int(params["taps_len"][bi].item())
                # exponential decay random complex taps
                t = torch.arange(T, device=z.device, dtype=torch.float32)
                decay = torch.exp(-t / (0.8 + 0.4*torch.rand(1, device=z.device)))
                hr = torch.randn(T, device=z.device) * decay
                hi = torch.randn(T, device=z.device) * decay
                h = torch.complex(hr, hi)
                h = h / (h.abs().sum() + 1e-12)
                taps[bi, :T] = h
            z_mp = _apply_multipath_fir(z, taps)
            z = torch.where(do_mp.expand_as(z), z_mp, z)

    # lowpass
    if p_lowpass > 0:
        do_lp = (torch.rand((B,1), device=z.device) < p_lowpass)
        if do_lp.any():
            z_lp = _apply_random_lowpass(z, fs, params["cutoff_hz"], params["transition_hz"])
            z = torch.where(do_lp.expand_as(z), z_lp, z)

    out = _from_complex(_nan_to_num_(z))
    # keep energy roughly comparable
    return out
