"""Visible-only periodic reference-residual feasibility primitive.

This operates on cached received/equalized IQ, not identifiable TX parameters.
Upstream equalization is unknown; raw-noise independence is not established.
"""

from __future__ import annotations

import math

import torch

TONES = (-24, -20, -16, -12, -8, -4, 4, 8, 12, 16, 20, 24)
SIGNS = (1, -1, 1, -1, -1, 1, -1, -1, 1, 1, 1, 1)
REFERENCE = (
    "https://raw.githubusercontent.com/bastibl/gr-ieee802-11/"
    "ad0598e4a874f4b8e1f391a1e0323e80df2b34ff/examples/wifi_phy_hier.grc"
)

VISIBLE_CYCLES = (0, 1, 3)
HOLDOUT = (120, 140)
VISIBLE_GAIN_EPS = 1e-12


def _public_bank(device):
    """Regenerate only existing public L-STF constants with Torch operations."""
    tones = torch.tensor(TONES, dtype=torch.float64, device=device)
    signs = torch.tensor(SIGNS, dtype=torch.float64, device=device)
    delays = torch.arange(320, dtype=torch.float64, device=device) / 16
    n = torch.arange(20, dtype=torch.float64, device=device)
    phase = 2 * torch.pi * (n[None, :, None] - delays[:, None, None]) * tones[None, None, :] / 80
    bank = ((1+1j) * signs[None, None, :] * torch.exp(1j*phase)).sum(dim=2) / math.sqrt(24)
    bins = torch.remainder(torch.tensor(TONES, dtype=torch.long, device=device) // 4, 20)
    return bank, bins


def _residual(wave, reference, gain, valid, bins):
    if wave.shape[0] == 0:
        return wave.real.new_empty((0, 24))
    ideal = torch.fft.fft(reference, dim=1)[:, bins]
    safe_gain = torch.where(valid, gain, torch.ones_like(gain))
    relative = torch.fft.fft(wave, dim=1)[:, bins] / ideal / safe_gain[:, None] - 1
    relative = torch.where(valid[:, None], relative, torch.zeros_like(relative))
    return torch.cat((relative.real, relative.imag), dim=1)


def repetition_residuals(x):
    """Return three baseline residuals, frozen target and visible-only calibration.

    x is finite real [B,2,256].  Hide [120:140] and use only [80:100],
    [100:120], [140:160] for every normalization/phase/template/gain operation.
    Phase slope is angle(sum(conj(cycle0)*cycle1))/20, with angle(0)=0.
    Visible normalization is homogeneous: divide by visible maximum component,
    then visible complex RMS.  No absolute energy floor, full-window RMS,
    all-cycle alignment or hidden-period quality/validity filter is used.

    The template is chosen by visible-mean correlation against the public bank;
    complex gain is visible least squares.  valid depends only on nonzero
    visible evidence and a nondegenerate visible relative reference gain.
    Arbitrarily low nonzero amplitude can remain valid after homogeneous FP64
    normalization.  Invalid rows explicitly return zero residuals.

    Each 24-real output is Re/Im of the twelve occupied-tone relative transfer
    divided by the visible gain, minus one.  Target applies detached calibration
    to the true heldout period and is fully detached.  Baseline and calibration
    outputs keep the visible input graph for leakage checks.  masked_input has
    only visible evidence, with the heldout period and all other positions zero.
    Visible mean remains an explicit comparison baseline, not an input fill.
    No labels, decoder, model, optimizer, learned parameters or RNG are accepted.
    """
    if not isinstance(x, torch.Tensor) or x.layout != torch.strided or x.ndim != 3 or x.shape[1:] != (2, 256) or not x.is_floating_point():
        raise ValueError("Expected real dense floating IQ[B,2,256]")
    if not bool(torch.isfinite(x).all().item()):
        raise ValueError("Input IQ must be finite")
    work = x.to(torch.float64)
    positions = torch.cat([torch.arange(80+cycle*20, 100+cycle*20, device=x.device) for cycle in VISIBLE_CYCLES])
    visible_iq = work.index_select(2, positions)
    visible_max = visible_iq.abs().amax(dim=(1,2))
    nonzero = visible_max > 0
    safe_max = torch.where(nonzero, visible_max, torch.ones_like(visible_max))
    scaled = visible_iq / safe_max[:, None, None]
    scaled_complex = torch.complex(scaled[:,0], scaled[:,1]).reshape(-1,3,20)
    power = scaled_complex.abs().square().mean(dim=(1,2))
    relative_rms = torch.sqrt(torch.where(nonzero, power, torch.ones_like(power)))
    visible_normalized = scaled_complex / relative_rms[:, None, None]
    phase_cross = (visible_normalized[:,0].conj()*visible_normalized[:,1]).sum(dim=1)
    # Avoid angle(0)'s undefined autograd derivative without changing its value.
    phase_cross_safe = torch.where(phase_cross.abs() > 0, phase_cross, torch.ones_like(phase_cross))
    phase_slope = torch.angle(phase_cross_safe) / 20
    local_n = torch.arange(20, dtype=torch.float64, device=x.device)
    visible_positions = torch.tensor([0,20,60], dtype=torch.float64, device=x.device)[None,:,None] + local_n[None,None,:]
    visible_corrected = visible_normalized * torch.exp(-1j*phase_slope[:,None,None]*visible_positions)
    visible_mean = visible_corrected.mean(dim=1)
    bank, bins = _public_bank(x.device)
    correlation = visible_mean @ bank.conj().T
    reference_index = correlation.abs().square().argmax(dim=1)
    reference = bank[reference_index]
    gain = (reference.conj()*visible_mean).sum(dim=1) / reference.abs().square().sum(dim=1)
    valid = nonzero & (gain.abs() > VISIBLE_GAIN_EPS)
    predictions = {
        "copy_cycle1": _residual(visible_corrected[:,1], reference, gain, valid, bins),
        "visible_mean": _residual(visible_mean, reference, gain, valid, bins),
        "known_reference": _residual(gain[:,None]*reference, reference, gain, valid, bins),
    }
    # Heldout is read only here, after the complete visible calibration exists.
    held = torch.complex(work[:,0,120:140], work[:,1,120:140])
    normalized_held = (held / safe_max.detach()[:,None]) / relative_rms.detach()[:,None]
    corrected_held = normalized_held * torch.exp(-1j*phase_slope.detach()[:,None]*(40+local_n))
    target = _residual(corrected_held, reference.detach(), gain.detach(), valid, bins).detach()
    masked_input = torch.zeros_like(work)
    for position, cycle in enumerate(VISIBLE_CYCLES):
        wave = visible_normalized[:,position]
        masked_input[:,0,80+20*cycle:100+20*cycle] = wave.real
        masked_input[:,1,80+20*cycle:100+20*cycle] = wave.imag
    masked_input = masked_input.to(x.dtype)
    if not all(bool(torch.isfinite(value).all().item()) for value in (*predictions.values(), target)):
        raise ValueError("Reference residual overflowed; no hidden-period validity filter was applied")
    return {
        "valid": valid, "predictions": predictions, "target": target, "masked_input": masked_input,
        "calibration": {
            "visible_max": visible_max, "relative_rms": relative_rms,
            "visible_normalized": visible_normalized, "visible_corrected": visible_corrected,
            "visible_mean": visible_mean, "phase_slope": phase_slope,
            "reference_index": reference_index, "reference": reference, "gain": gain,
        },
        "summary": {
            "input_count": len(x), "valid_count": int(valid.sum().item()),
            "zero_visible_count": int((~nonzero).sum().item()),
            "degenerate_visible_gain_count": int((nonzero & ~valid).sum().item()),
            "holdout": [120,140], "visible": [[80,100],[100,120],[140,160]],
            "period_samples": 20, "phase_slope_rule": "angle(sum(conj(cycle0)*cycle1))/20",
            "normalization": "visible-only homogeneous complex RMS; FP64 max rescaling",
            "absolute_low_energy_filter": False, "relative_gain_threshold": VISIBLE_GAIN_EPS,
            "target_detached": True, "calibration_uses_holdout": False,
            "heldout_input_fill": "zero", "outside_window_input_fill": "zero",
            "global_RNG_consumed": False, "new_parameters": False,
            "public_reference": REFERENCE,
            "scope": "cached received/equalized periodic response; not calibrated TX hardware",
            "upstream_equalization": "unknown", "raw_noise_independence_proven": False,
            "TX_identifiability_proven": False,
        },
    }
