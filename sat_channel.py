# sat_channel.py
# ------------------------------------------------------------
# Satellite-to-Ground channel simulator for RFFI (WiSig baseline)
# - Designed for short IQ snippets (e.g., WiSig IdSig: T=256) at Fs=25 Msps
# - Physics-consistent coupling between elevation angle and slant range
# - Domain parameters splitable into TRAIN vs TEST (to avoid parameter leakage)
# - Keeps fingerprint-relevant distortions (freq offset, phase noise, IQ imbalance, multipath)
# - Uses mild AGC so amplitude effects are bounded (compatible with per-sample RMS normalize pipelines)
# ------------------------------------------------------------
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Any

import torch

C = 299_792_458.0  # m/s
MU_EARTH = 3.986004418e14  # m^3/s^2
R_EARTH = 6_371_000.0      # m

ATM_TABLE: Dict[str, Dict[str, float]] = {
    "clear":   dict(mu_a=0.413, sigma2_a=0.00087, m_a=0.0072, eta2_a=0.00357),
    "cloudy":  dict(mu_a=0.498, sigma2_a=0.00025, m_a=0.0086, eta2_a=0.00405),
    "storm":   dict(mu_a=0.436, sigma2_a=0.01386, m_a=0.0068, eta2_a=0.00414),
    "rain":    dict(mu_a=0.413, sigma2_a=0.02000, m_a=-0.0089, eta2_a=0.03077),
}

LOO_TABLE: Dict[str, Dict[str, float]] = {
    "light":  dict(mu=1.1219, d0=1.2586, b0=0.158),
    "mid":    dict(mu=0.8914, d0=1.3799, b0=0.126),
    "severe": dict(mu=0.0200, d0=5.0128, b0=0.063),
}

def fspl_db(d_m: torch.Tensor, fc_hz: float) -> torch.Tensor:
    lam = C / float(fc_hz)
    return 20.0 * torch.log10(4.0 * math.pi * d_m / lam)

def slant_range_from_elevation(theta_deg: torch.Tensor, h_m: torch.Tensor) -> torch.Tensor:
    eps = torch.deg2rad(theta_deg.clamp(min=1e-3, max=89.999))
    Re = R_EARTH
    r = Re + h_m
    term = (r*r - (Re*torch.cos(eps))**2).clamp_min(0.0)
    rho = -Re*torch.sin(eps) + torch.sqrt(term)
    return rho.clamp_min(1.0)

def orbital_speed_circular(h_m: torch.Tensor) -> torch.Tensor:
    mu = torch.tensor(MU_EARTH, device=h_m.device, dtype=h_m.dtype)
    return torch.sqrt(mu / (R_EARTH + h_m))

def steady_probs_from_elevation(theta_deg: torch.Tensor, scenario: str = "urban"):
    c = 7.0e3 if scenario == "urban" else 1.66e4
    w1 = 1.0 - ((90.0 - theta_deg) ** 2) / c
    w1 = torch.clamp(w1, 0.0, 1.0)
    if scenario == "urban":
        w3 = (1.0 - w1) * 4.0 / 5.0
        w2 = (1.0 - w1) * 1.0 / 5.0
    else:
        w3 = (1.0 - w1) * 1.0 / 5.0
        w2 = (1.0 - w1) * 4.0 / 5.0
    return w1, w2, w3

def categorical_sample(probs: torch.Tensor, gen: Optional[torch.Generator]=None) -> torch.Tensor:
    return torch.multinomial(probs, num_samples=1, replacement=True, generator=gen).squeeze(1)

def complex_awgn_like(y: torch.Tensor, snr_db: torch.Tensor, gen: Optional[torch.Generator]=None) -> torch.Tensor:
    B, T = y.shape
    sigp = (y.real**2 + y.imag**2).mean(dim=1).clamp_min(1e-12)
    snr_lin = 10.0 ** (snr_db / 10.0)
    noise_var = sigp / snr_lin
    nr = torch.randn((B, T), device=y.device, dtype=y.real.dtype, generator=gen)
    ni = torch.randn((B, T), device=y.device, dtype=y.real.dtype, generator=gen)
    w = (noise_var/2.0).sqrt().unsqueeze(1) * (nr + 1j*ni)
    return y + w

def wiener_phase_noise(B: int, T: int, sigma_rad: torch.Tensor, device, dtype, gen: Optional[torch.Generator]=None) -> torch.Tensor:
    inc = torch.randn((B, T), device=device, dtype=dtype, generator=gen) * sigma_rad.unsqueeze(1)
    phi = torch.cumsum(inc, dim=1)
    return torch.exp(1j * phi)

def apply_iq_imbalance(y: torch.Tensor, amp_db: torch.Tensor, phase_deg: torch.Tensor) -> torch.Tensor:
    eps = 10.0 ** (amp_db / 20.0)
    phi = torch.deg2rad(phase_deg)
    alpha = 0.5 * (1.0 + eps) * torch.exp(-1j * phi / 2.0)
    beta  = 0.5 * (1.0 - eps) * torch.exp( 1j * phi / 2.0)
    return alpha.unsqueeze(1) * y + beta.unsqueeze(1) * torch.conj(y)

def apply_mild_agc(y: torch.Tensor, target_rms: float = 1.0, resid_db: torch.Tensor = None) -> torch.Tensor:
    rms = torch.sqrt((y.real**2 + y.imag**2).mean(dim=1).clamp_min(1e-12))
    y = y * (float(target_rms) / rms).unsqueeze(1)
    if resid_db is not None:
        y = y * (10.0 ** (resid_db / 20.0)).unsqueeze(1)
    return y

@dataclass
class SatSimConfig:
    fs_hz: float = 25e6
    fc_hz: float = 2.462e9  # WiFi ch13 center; override if needed
    scenario: str = "urban"
    weather: str = "clear"
    loo_level: str = "mid"
    orbit_probs: Dict[str, float] = None
    theta_deg: Tuple[float, float] = (10.0, 90.0)
    snr_db: Tuple[float, float] = (10.0, 30.0)
    cfo_std_hz: float = 200.0
    phase_noise_inc_std: Tuple[float, float] = (0.0, 2e-3)
    iq_amp_db: Tuple[float, float] = (-0.3, 0.3)
    iq_phase_deg: Tuple[float, float] = (-3.0, 3.0)
    agc_resid_db: Tuple[float, float] = (-1.0, 1.0)
    leo_h_km: Tuple[float, float] = (500.0, 2000.0)
    meo_h_km: Tuple[float, float] = (8000.0, 20000.0)
    geo_h_km: Tuple[float, float] = (35786.0, 35786.0)
    markov_alpha: float = 0.0
    K_db_range: Tuple[float, float] = (0.0, 18.0)
    enable_multipath: bool = False
    num_taps: Tuple[int, int] = (2, 5)
    max_delay_samp: int = 6
    pwr_decay: float = 0.8
    channel_model: str = "full"
    pathloss_alpha: float = 2.0
    shadow_mu_db: float = 0.0
    shadow_sigma_db: float = 4.0
    simple_ref_h_km: float = 1000.0
    simple_ref_theta_deg: float = 60.0
    simple_agc: bool = True

def _default_orbit_probs() -> Dict[str, float]:
    return {"LEO": 0.7, "MEO": 0.2, "GEO": 0.1}


def _sample_orbits_and_geometry(B: int, cfg: SatSimConfig, device, gen: Optional[torch.Generator] = None):
    probs = cfg.orbit_probs or _default_orbit_probs()
    orbit_keys = ["LEO", "MEO", "GEO"]
    p = torch.tensor([probs.get(k, 0.0) for k in orbit_keys], device=device, dtype=torch.float32)
    p = p / p.sum().clamp_min(1e-12)
    orbits = categorical_sample(p.expand(B, -1), gen=gen)

    u = torch.rand((B,), device=device, dtype=torch.float32, generator=gen)
    h_km = torch.empty((B,), device=device, dtype=torch.float32)
    m = (orbits == 0)
    if m.any():
        lo, hi = cfg.leo_h_km
        h_km[m] = lo + (hi - lo) * u[m]
    m = (orbits == 1)
    if m.any():
        lo, hi = cfg.meo_h_km
        h_km[m] = lo + (hi - lo) * u[m]
    m = (orbits == 2)
    if m.any():
        lo, hi = cfg.geo_h_km
        h_km[m] = lo + (hi - lo) * u[m]

    h_m = h_km * 1e3
    tlo, thi = cfg.theta_deg
    theta = tlo + (thi - tlo) * torch.rand((B,), device=device, dtype=torch.float32, generator=gen)
    d_m = slant_range_from_elevation(theta, h_m)
    return orbits, h_km, h_m, theta, d_m


def apply_simple_sat_channel_batch(
    x_iq: torch.Tensor,
    cfg: SatSimConfig,
    *,
    gen: Optional[torch.Generator] = None,
    return_meta: bool = False,
):
    """Simplified LEO satellite channel from the presentation formula.

    h(t) = L(t) * xi(t) * exp(j * phi(t))
    y(t) = h(t) * x(t) + n(t)

    Kept effects: distance path loss, lognormal shadowing, Doppler phase, AWGN.
    Excluded effects: Rician/Rayleigh state switching, multipath, IQ imbalance,
    phase noise, atmospheric complex fading, and receiver-chain impairments.
    """
    assert x_iq.ndim == 3 and x_iq.shape[1] == 2
    device = x_iq.device
    x = x_iq.to(torch.float32)
    B, _, T = x.shape
    fs = float(cfg.fs_hz)
    fc = float(cfg.fc_hz)
    xc = (x[:, 0] + 1j * x[:, 1]).to(torch.complex64)

    orbits, h_km, h_m, theta, d_m = _sample_orbits_and_geometry(B, cfg, device, gen=gen)
    ref_h_m = torch.full((1,), float(cfg.simple_ref_h_km) * 1e3, device=device, dtype=torch.float32)
    ref_theta = torch.full((1,), float(cfg.simple_ref_theta_deg), device=device, dtype=torch.float32)
    d0 = slant_range_from_elevation(ref_theta, ref_h_m)[0].clamp_min(1.0)

    # Figure convention: L(t)=(d0/d(t))^(alpha/2), an amplitude gain.
    L_gain = (d0 / d_m.clamp_min(1.0)).clamp(1e-4, 1e4) ** (float(cfg.pathloss_alpha) / 2.0)
    shadow_db = torch.normal(
        mean=float(cfg.shadow_mu_db),
        std=float(cfg.shadow_sigma_db),
        size=(B,),
        device=device,
        dtype=torch.float32,
        generator=gen,
    )
    xi = 10.0 ** (shadow_db / 20.0)

    v = orbital_speed_circular(h_m).to(torch.float32)
    sgn = torch.where(torch.rand((B,), device=device, generator=gen) < 0.5, -1.0, 1.0)
    vr = sgn * v * torch.cos(torch.deg2rad(theta))
    fD = (vr / C) * fc
    n = torch.arange(T, device=device, dtype=torch.float32).unsqueeze(0)
    phi = 2.0 * math.pi * fD.unsqueeze(1) * n / fs
    doppler_phase = torch.exp(1j * phi.to(torch.complex64))

    h = (L_gain * xi).to(torch.complex64).unsqueeze(1) * doppler_phase
    yc = h * xc

    if bool(cfg.simple_agc):
        ag_lo, ag_hi = cfg.agc_resid_db
        resid = ag_lo + (ag_hi - ag_lo) * torch.rand((B,), device=device, dtype=torch.float32, generator=gen)
        yc = apply_mild_agc(yc, target_rms=1.0, resid_db=resid)

    sn_lo, sn_hi = cfg.snr_db
    snr = sn_lo + (sn_hi - sn_lo) * torch.rand((B,), device=device, dtype=torch.float32, generator=gen)
    yc = complex_awgn_like(yc, snr, gen=gen)
    y_iq = torch.stack([yc.real, yc.imag], dim=1).to(torch.float32)

    meta = None
    if return_meta:
        meta = {
            "channel_model": "simple_leo",
            "orbit": orbits.detach().cpu(),
            "h_km": h_km.detach().cpu(),
            "theta_deg": theta.detach().cpu(),
            "d_km": (d_m / 1e3).detach().cpu(),
            "L_gain": L_gain.detach().cpu(),
            "shadow_db": shadow_db.detach().cpu(),
            "fD_hz": fD.detach().cpu(),
            "snr_db": snr.detach().cpu(),
        }
    return y_iq, meta, None


@torch.no_grad()
def apply_sat_gnd_channel_batch(
    x_iq: torch.Tensor,
    cfg: SatSimConfig,
    *,
    prev_state: Optional[torch.Tensor] = None,
    gen: Optional[torch.Generator] = None,
    return_meta: bool = False,
):
    assert x_iq.ndim == 3 and x_iq.shape[1] == 2
    if str(getattr(cfg, "channel_model", "full")).lower().strip() in ("simple", "simple_leo", "presentation"):
        return apply_simple_sat_channel_batch(x_iq, cfg, gen=gen, return_meta=return_meta)

    device = x_iq.device
    x = x_iq.to(torch.float32)
    B, _, T = x.shape
    fs = float(cfg.fs_hz)
    fc = float(cfg.fc_hz)

    xc = (x[:, 0] + 1j * x[:, 1]).to(torch.complex64)

    probs = cfg.orbit_probs or _default_orbit_probs()
    orbit_keys = ["LEO", "MEO", "GEO"]
    p = torch.tensor([probs.get(k, 0.0) for k in orbit_keys], device=device, dtype=torch.float32)
    p = p / p.sum().clamp_min(1e-12)
    orbits = categorical_sample(p.expand(B, -1), gen=gen)

    u = torch.rand((B,), device=device, dtype=torch.float32, generator=gen)
    h_km = torch.empty((B,), device=device, dtype=torch.float32)
    m = (orbits == 0)
    if m.any():
        lo, hi = cfg.leo_h_km
        h_km[m] = lo + (hi - lo) * u[m]
    m = (orbits == 1)
    if m.any():
        lo, hi = cfg.meo_h_km
        h_km[m] = lo + (hi - lo) * u[m]
    m = (orbits == 2)
    if m.any():
        lo, hi = cfg.geo_h_km
        h_km[m] = lo + (hi - lo) * u[m]
    h_m = h_km * 1e3

    tlo, thi = cfg.theta_deg
    theta = tlo + (thi - tlo) * torch.rand((B,), device=device, dtype=torch.float32, generator=gen)

    d_m = slant_range_from_elevation(theta, h_m)
    pl_db = fspl_db(d_m, fc)

    d_ref = slant_range_from_elevation(torch.tensor([60.0], device=device), torch.tensor([1_000_000.0], device=device))[0]
    pl_ref = fspl_db(d_ref, fc)
    g_pl = 10.0 ** (-(pl_db - pl_ref) / 20.0)

    w_los, w_loo, w_ray = steady_probs_from_elevation(theta, scenario=cfg.scenario)
    W = torch.stack([w_los, w_loo, w_ray], dim=1)
    W = W / W.sum(dim=1, keepdim=True).clamp_min(1e-12)

    if prev_state is None or cfg.markov_alpha <= 0.0:
        state = categorical_sample(W, gen=gen)
    else:
        alpha = float(cfg.markov_alpha)
        s = prev_state.to(device=device).long().clamp(0, 2)
        P = torch.zeros((B, 3, 3), device=device, dtype=torch.float32)
        for i in range(3):
            for j in range(3):
                if i != j:
                    P[:, i, j] = alpha * W[:, j]
            P[:, i, i] = 1.0 - alpha * (1.0 - W[:, i])
        probs_next = P[torch.arange(B, device=device), s]
        state = categorical_sample(probs_next, gen=gen)

    at = ATM_TABLE[cfg.weather]
    ra = torch.normal(mean=float(at["mu_a"]), std=math.sqrt(float(at["sigma2_a"])), size=(B,), device=device, generator=gen).clamp_min(1e-3)
    phia = torch.normal(mean=float(at["m_a"]), std=math.sqrt(float(at["eta2_a"])), size=(B,), device=device, generator=gen)
    a_atm = ra.to(torch.complex64) * torch.exp(1j * phia.to(torch.complex64))

    Klo, Khi = cfg.K_db_range
    theta_n = ((theta - tlo) / max(1e-6, (thi - tlo))).clamp(0.0, 1.0)
    K_db = Klo + (Khi - Klo) * theta_n
    if cfg.weather in ("rain", "storm"):
        K_db = (K_db - 3.0).clamp_min(0.0)

    h0 = torch.empty((B,), device=device, dtype=torch.complex64)

    m0 = (state == 0)
    if m0.any():
        K = 10.0 ** (K_db[m0] / 10.0)
        gr = torch.randn((m0.sum(),), device=device, dtype=torch.float32, generator=gen)
        gi = torch.randn((m0.sum(),), device=device, dtype=torch.float32, generator=gen)
        g = (gr + 1j * gi).to(torch.complex64) / math.sqrt(2.0)
        phi0 = (torch.rand((m0.sum(),), device=device, dtype=torch.float32, generator=gen) * 2*math.pi - math.pi)
        los = torch.sqrt(K/(K+1.0)).to(torch.complex64) * torch.exp(1j * phi0.to(torch.complex64))
        scat = torch.sqrt(1.0/(K+1.0)).to(torch.complex64) * g
        h0[m0] = los + scat

    m2 = (state == 2)
    if m2.any():
        gr = torch.randn((m2.sum(),), device=device, dtype=torch.float32, generator=gen)
        gi = torch.randn((m2.sum(),), device=device, dtype=torch.float32, generator=gen)
        h0[m2] = ((gr + 1j*gi).to(torch.complex64) / math.sqrt(2.0))

    m1 = (state == 1)
    if m1.any():
        lt = LOO_TABLE[cfg.loo_level]
        z = torch.exp(float(lt["mu"]) + math.sqrt(float(lt["d0"])) * torch.randn((m1.sum(),), device=device, dtype=torch.float32, generator=gen))
        phi0 = (torch.rand((m1.sum(),), device=device, dtype=torch.float32, generator=gen) * 2*math.pi - math.pi)
        los = (z * torch.exp(1j * phi0)).to(torch.complex64)
        gr = torch.randn((m1.sum(),), device=device, dtype=torch.float32, generator=gen)
        gi = torch.randn((m1.sum(),), device=device, dtype=torch.float32, generator=gen)
        scatter = math.sqrt(float(lt["b0"])) * ((gr + 1j*gi).to(torch.complex64) / math.sqrt(2.0))
        h0[m1] = los + scatter

    v = orbital_speed_circular(h_m).to(torch.float32)
    sgn = torch.where(torch.rand((B,), device=device, generator=gen) < 0.5, -1.0, 1.0)
    vr = sgn * v * torch.cos(torch.deg2rad(theta))
    fD = (vr / C) * fc

    cfo = torch.randn((B,), device=device, dtype=torch.float32, generator=gen) * float(cfg.cfo_std_hz)
    f_off = fD + cfo

    n = torch.arange(T, device=device, dtype=torch.float32).unsqueeze(0)
    ph = 2.0 * math.pi * f_off.unsqueeze(1) * n / fs
    freq_rot = torch.exp(1j * ph.to(torch.complex64))

    pn_lo, pn_hi = cfg.phase_noise_inc_std
    pn_std = pn_lo + (pn_hi - pn_lo) * torch.rand((B,), device=device, dtype=torch.float32, generator=gen)
    pn = wiener_phase_noise(B, T, pn_std, device, torch.float32, gen=gen).to(torch.complex64)

    if cfg.enable_multipath:
        Lmin, Lmax = cfg.num_taps
        L = int(torch.randint(low=Lmin, high=Lmax+1, size=(1,), device=device, generator=gen).item())
        maxD = int(cfg.max_delay_samp)
        delays = torch.randint(low=0, high=maxD+1, size=(L,), device=device, generator=gen)
        delays[0] = 0
        pwr = (cfg.pwr_decay ** torch.arange(L, device=device, dtype=torch.float32))
        pwr = pwr / pwr.sum().clamp_min(1e-12)

        taps = torch.zeros((B, L), device=device, dtype=torch.complex64)
        taps[:, 0] = h0
        if L > 1:
            gr = torch.randn((B, L-1), device=device, dtype=torch.float32, generator=gen)
            gi = torch.randn((B, L-1), device=device, dtype=torch.float32, generator=gen)
            taps[:, 1:] = ((gr + 1j*gi).to(torch.complex64) / math.sqrt(2.0))
        taps = taps * torch.sqrt(pwr).to(torch.complex64).unsqueeze(0)

        yc = torch.zeros_like(xc)
        for k in range(L):
            d = int(delays[k].item())
            xs = torch.roll(xc, shifts=d, dims=1)
            if d > 0:
                xs[:, :d] = 0
            yc = yc + taps[:, k].unsqueeze(1) * xs
    else:
        yc = h0.unsqueeze(1) * xc

    yc = (g_pl.to(torch.complex64) * a_atm).unsqueeze(1) * yc
    yc = yc * freq_rot * pn

    ag_lo, ag_hi = cfg.agc_resid_db
    resid = ag_lo + (ag_hi - ag_lo) * torch.rand((B,), device=device, dtype=torch.float32, generator=gen)
    yc = apply_mild_agc(yc, target_rms=1.0, resid_db=resid)

    sn_lo, sn_hi = cfg.snr_db
    snr = sn_lo + (sn_hi - sn_lo) * torch.rand((B,), device=device, dtype=torch.float32, generator=gen)
    yc = complex_awgn_like(yc, snr, gen=gen)

    a_lo, a_hi = cfg.iq_amp_db
    p_lo, p_hi = cfg.iq_phase_deg
    amp_db = a_lo + (a_hi - a_lo) * torch.rand((B,), device=device, dtype=torch.float32, generator=gen)
    ph_deg = p_lo + (p_hi - p_lo) * torch.rand((B,), device=device, dtype=torch.float32, generator=gen)
    yc = apply_iq_imbalance(yc, amp_db=amp_db, phase_deg=ph_deg)

    y_iq = torch.stack([yc.real, yc.imag], dim=1).to(torch.float32)

    meta = None
    if return_meta:
        meta = {
            "orbit": orbits.detach().cpu(),
            "h_km": h_km.detach().cpu(),
            "theta_deg": theta.detach().cpu(),
            "d_km": (d_m/1e3).detach().cpu(),
            "state": state.detach().cpu(),
            "pl_db": pl_db.detach().cpu(),
            "fD_hz": fD.detach().cpu(),
            "cfo_hz": cfo.detach().cpu(),
            "snr_db": snr.detach().cpu(),
            "K_db": K_db.detach().cpu(),
        }

    return y_iq, meta, state
