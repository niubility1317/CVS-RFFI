"""Receiver-side short-training-field controls for 25 Msps WiSig windows."""

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F


PERIOD = 20
# Center of the 200-sample STF, after the leading transient.
BEGIN, END = 40, 200
TONES = (-6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6)
SIGNS = (1, -1, 1, -1, -1, 1, -1, -1, 1, 1, 1, 1)
READOUTS = ('cfo', 'eq_fixed', 'eq_variance', 'eq_keep_cfo', 'repeat_wiener')


@dataclass
class Preamble:
    aligned: Tensor
    rotation: Tensor
    slope: Tensor
    coherence: Tensor
    disturbance: Tensor
    noise_power: Tensor


def power(iq):
    return iq.real.square() + iq.imag.square()


def phase(phi):
    return torch.polar(torch.ones_like(phi), phi)


def estimate(iq):
    """CFO from early repetitions and effective noise from disjoint late ones."""
    early = iq[:, BEGIN:BEGIN + 4 * PERIOD].unflatten(-1, (4, PERIOD))
    a, b = early[:, :-1], early[:, 1:]
    correlation = (a.conj() * b).sum(dim=(-2, -1))
    total = (power(a).sum(dim=(-2, -1)) * power(b).sum(dim=(-2, -1))).sqrt()
    coherence = (correlation.abs() / total.clamp_min(1e-8)).clamp(0, 1)
    slope = correlation.angle() / PERIOD
    time = torch.arange(iq.shape[-1], device=iq.device, dtype=iq.real.dtype)
    rotation = phase(slope[:, None] * (time - (BEGIN + END) / 2))
    aligned = iq * rotation.conj()
    late = aligned[:, BEGIN + 4 * PERIOD:END].unflatten(-1, (4, PERIOD))
    difference = late[:, 1::2] - late[:, ::2]
    # Mean squared difference/2 assumes independent equal-variance noise. With
    # structured interference or genuine repeat differences it is effective power.
    noise = .5 * power(difference).mean(dim=(-2, -1))
    disturbance = noise / power(late).mean(dim=(-2, -1)).clamp_min(1e-8)
    return Preamble(aligned, rotation, slope, coherence, disturbance, noise)


def channel_estimate(preamble):
    """Relative local response after removing only a dominant timing and gain gauge."""
    iq = preamble.aligned
    repeats = iq[:, BEGIN:END].unflatten(-1, (-1, PERIOD))
    tones = torch.tensor(TONES, device=iq.device)
    signed = torch.tensor(SIGNS, device=iq.device, dtype=iq.real.dtype)
    # STF sequence from the IEEE 802.11 legacy preamble, in reduced-period bins.
    references = signed * (1 + 1j)
    reference_response = torch.fft.fft(repeats.mean(dim=1), norm='ortho')[:, tones] / references
    shifts = torch.arange(PERIOD, device=iq.device, dtype=iq.real.dtype)
    frequency = tones.to(iq.real.dtype) / PERIOD
    lag_phases = phase(2 * torch.pi * frequency[:, None] * shifts)
    profile = reference_response @ lag_phases / len(TONES)
    delay = profile.abs().argmax(dim=1)
    gain = profile.gather(1, delay[:, None])
    gauged = reference_response * lag_phases[:, delay].T
    gauged = gauged * gain.conj() / power(gain).clamp_min(1e-8)

    taps = torch.arange(-2, 3, device=iq.device, dtype=iq.real.dtype)
    design = phase(-2 * torch.pi * frequency[:, None] * taps)
    identity = torch.eye(len(taps), device=iq.device, dtype=iq.dtype)
    center = (taps == 0).to(iq.dtype)
    ridge = (.04 + preamble.disturbance.clamp(0, 1))[:, None, None]
    normal = design.H @ design / len(TONES) + ridge * identity
    rhs = gauged @ design.conj() / len(TONES) + ridge[:, 0] * center
    coefficients = torch.linalg.solve_ex(normal, rhs[..., None], check_errors=False).result.squeeze(-1)
    mismatch = power(gauged - coefficients @ design.T).mean(dim=-1)
    usable_gain = power(gain[:, 0]) > 1e-8
    confidence = (preamble.coherence - .55).clamp(0, .40) / .40
    confidence = confidence * usable_gain / (1 + mismatch)
    return coefficients, taps, confidence, delay, mismatch


def inverse_delta(aligned, coefficients, taps, confidence, ridge):
    # The estimated response is limited to a short FIR and 12 observed tones.
    # Use a reflective halo to avoid wrap at the 256-sample benchmark boundary.
    edge = 32
    real_view = torch.view_as_real(aligned.contiguous()).transpose(-1, -2)
    padded = torch.view_as_complex(F.pad(real_view, (edge, edge), mode='reflect').transpose(-1, -2).contiguous())
    freq = torch.fft.fftfreq(padded.shape[-1], device=aligned.device)
    response = coefficients @ phase(-2 * torch.pi * taps[:, None] * freq)
    regularizer = ridge[:, None]
    # Quadratic shrinkage toward unchanged observed IQ avoids unstable division.
    correction = (response.conj() + regularizer) / (power(response) + regularizer) - 1
    band = ((.36 - freq.abs()) / .06).clamp(0, 1)
    diff = torch.fft.ifft(torch.fft.fft(padded) * correction * band)
    return confidence[:, None] * diff[:, edge:-edge]


def repeat_wiener(preamble, clean_floor):
    aligned = preamble.aligned
    repeats = aligned[:, BEGIN:END].unflatten(-1, (-1, PERIOD))
    mean = repeats.mean(dim=1, keepdim=True)
    noise_excess = (preamble.disturbance - clean_floor).clamp_min(0)
    # First remove the part of innovation power explainable by the expected
    # receiver noise, keeping part of the per-repeat noncommon signal.
    weight = (noise_excess / preamble.disturbance.clamp_min(1e-8)).clamp(max=.75)
    weight = weight * ((preamble.coherence - .35) / .35).clamp(0, 1)
    time = torch.arange(END - BEGIN, device=aligned.device, dtype=aligned.real.dtype)
    taper = (torch.minimum(time, (END - BEGIN - 1) - time) / PERIOD).clamp(0, 1)
    delta = (weight[:, None, None] * (mean - repeats)).flatten(1) * taper
    return F.pad(delta, (BEGIN, aligned.shape[-1] - END)) * preamble.rotation


def restore(iq, *, clean_floor):
    """Matched no-train controls and diagnostics using only the received packet."""
    if iq.ndim != 2 or iq.shape[-1] != 256 or not iq.is_complex():
        raise ValueError('Expected complex [N, 256] WiSig IQ')
    obs = estimate(iq)
    h, taps, confidence, delay, mismatch = channel_estimate(obs)
    fixed = torch.full_like(obs.disturbance, .12)
    estimated = fixed + 2 * obs.disturbance.clamp(0, 1)
    delta_fixed = inverse_delta(obs.aligned, h, taps, confidence, fixed)
    delta_estimated = inverse_delta(obs.aligned, h, taps, confidence, estimated)
    outputs = dict(cfo=obs.aligned, eq_fixed=obs.aligned + delta_fixed,
                   eq_variance=obs.aligned + delta_estimated,
                   eq_keep_cfo=iq + delta_estimated * obs.rotation,
                   repeat_wiener=iq + repeat_wiener(obs, clean_floor))
    return outputs, dict(coherence=obs.coherence, noise_power=obs.noise_power,
                         disturbance=obs.disturbance, timing_phase_bin=delay,
                         cfo_hz=obs.slope * (25e6 / (2 * torch.pi)),
                         one_window_fit=mismatch, channel_confidence=confidence)
