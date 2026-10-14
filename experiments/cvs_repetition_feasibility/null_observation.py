"""Visible-only null-tone observations, without fitting a reference response.

The fixed null bins are outside the public L-STF occupied tones and DC. This
statement is about the periodic steady-state linear reference subspace; it does
not establish TX exclusivity, DG, or removal of actual receiver information.
"""

from __future__ import annotations

import math

import torch


PERIOD = 20
VISIBLE_CYCLES = (0, 1, 3)
NULL_BINS = tuple(range(7, 14))
OCCUPIED_SIGNED_BINS = (-6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6)


def _null_vector(wave):
    spectrum = torch.fft.fft(wave, dim=-1)[..., 7:14] / math.sqrt(PERIOD)
    return torch.cat((spectrum.real, spectrum.imag), dim=-1)


def null_evidence(x):
    """Return fixed null observations of real floating IQ ``[B, 2, 256]``.

    Only [80:100], [100:120], [140:160] set the homogeneous visible complex
    RMS and phase slope ``angle(sum(conj(cycle0)*cycle1))/20``. angle(0) is
    defined as zero with a finite derivative. No delay, complex response gain,
    template selection, hidden quality filter, labels, model, or RNG is used.

    ``cycles[B,3,14]`` contains FFT20/sqrt(20), bins 7..13: first seven real
    components, then seven imaginary components. The three predictions are
    visible mean, fixed cycle1 copy, and zero reference. ``target`` uses frozen
    visible calibration on [120:140] and is fully detached. All invalid rows
    (zero visible energy only) explicitly return zero observations and target.

    ``variance = sum_j mean_d((cycle_j-mean)^2)/(3-1)``; ``budget=4/3*variance``
    and ``signal=mean_d(mean^2)``. The budget is an observation tolerance
    motivated by IID noise, not an established noise floor of this cache.
    Computation uses FP64 max rescaling so arbitrarily low nonzero visible
    energy is valid without an amplitude-dependent threshold. The zero-masked
    input keeps RMS-normalized visible IQ before phase correction; hidden and
    all outside positions remain zero, as in the original visible-only input.
    """
    if (not isinstance(x, torch.Tensor) or x.layout != torch.strided
            or x.ndim != 3 or x.shape[0] < 1 or x.shape[1:] != (2, 256)
            or not x.is_floating_point()):
        raise ValueError("expected nonempty real dense floating IQ[B,2,256]")
    if not bool(torch.isfinite(x).all()):
        raise ValueError("input IQ must be finite; no hidden quality filtering is applied")
    work = x.to(dtype=torch.float64)
    positions = torch.cat([
        torch.arange(80 + cycle * PERIOD, 100 + cycle * PERIOD, device=x.device)
        for cycle in VISIBLE_CYCLES
    ])
    visible_iq = work.index_select(2, positions)
    visible_max = visible_iq.abs().amax(dim=(1, 2))
    valid = visible_max > 0
    safe_max = torch.where(valid, visible_max, torch.ones_like(visible_max))
    scaled = visible_iq / safe_max[:, None, None]
    scaled_wave = torch.complex(scaled[:, 0], scaled[:, 1]).reshape(-1, 3, PERIOD)
    power = scaled_wave.abs().square().mean(dim=(1, 2))
    relative_rms = torch.sqrt(torch.where(valid, power, torch.ones_like(power)))
    visible_normalized = scaled_wave / relative_rms[:, None, None]
    phase_cross = (visible_normalized[:, 0].conj() * visible_normalized[:, 1]).sum(dim=1)
    safe_cross = torch.where(phase_cross.abs() > 0, phase_cross, torch.ones_like(phase_cross))
    phase_slope = torch.angle(safe_cross) / PERIOD
    time = torch.arange(PERIOD, device=x.device, dtype=torch.float64)
    visible_time = torch.tensor([0, 20, 60], device=x.device, dtype=torch.float64)[None, :, None] + time[None, None, :]
    visible_corrected = visible_normalized * torch.exp(-1j * phase_slope[:, None, None] * visible_time)
    cycles = _null_vector(visible_corrected)
    cycles = torch.where(valid[:, None, None], cycles, torch.zeros_like(cycles))
    mean = cycles.mean(dim=1)
    copy_cycle1 = cycles[:, 1]
    zero_reference = torch.zeros_like(mean)
    variance = (cycles - mean[:, None, :]).square().mean(dim=2).sum(dim=1) / 2
    budget = (4 / 3) * variance
    signal = mean.square().mean(dim=1)

    held = torch.complex(work[:, 0, 120:140], work[:, 1, 120:140])
    held_normalized = (held / safe_max.detach()[:, None]) / relative_rms.detach()[:, None]
    held_corrected = held_normalized * torch.exp(-1j * phase_slope.detach()[:, None] * (40 + time))
    target = _null_vector(held_corrected)
    target = torch.where(valid[:, None], target, torch.zeros_like(target)).detach()
    masked_input = torch.zeros_like(work)
    for position, cycle in enumerate(VISIBLE_CYCLES):
        wave = visible_normalized[:, position]
        masked_input[:, 0, 80 + cycle * PERIOD:100 + cycle * PERIOD] = wave.real
        masked_input[:, 1, 80 + cycle * PERIOD:100 + cycle * PERIOD] = wave.imag
    masked_input = masked_input.to(dtype=x.dtype)
    if not all(bool(torch.isfinite(value).all()) for value in (cycles, target, variance, budget, signal, masked_input)):
        raise ValueError("null observation overflowed; no hidden-period validity filter was applied")
    return {
        "cycles": cycles, "mean": mean, "copy_cycle1": copy_cycle1,
        "zero_reference": zero_reference, "target": target, "valid": valid,
        "variance": variance, "budget": budget, "signal": signal,
        "masked_input": masked_input,
        "predictions": {"visible_mean": mean, "copy_cycle1": copy_cycle1, "zero_reference": zero_reference},
        "calibration": {
            "visible_max": visible_max, "relative_rms": relative_rms,
            "visible_normalized": visible_normalized, "visible_corrected": visible_corrected,
            "phase_slope": phase_slope,
        },
        "summary": {
            "input_count": len(x), "valid_count": int(valid.sum()),
            "zero_visible_count": int((~valid).sum()),
            "visible": [[80, 100], [100, 120], [140, 160]], "holdout": [120, 140],
            "period_samples": PERIOD, "occupied_signed_bins": list(OCCUPIED_SIGNED_BINS),
            "excluded_DC_bin": 0, "null_bins": list(NULL_BINS), "real_dimension": 14,
            "FFT_normalization": "FFT20/sqrt(20); seven real components then seven imaginary components",
            "normalization": "visible-only homogeneous complex RMS with FP64 max rescaling",
            "phase_slope_rule": "angle(sum(conj(cycle0)*cycle1))/20; angle(0)=0",
            "valid_rule": "nonzero visible energy only; no hidden, gain or label-based filtering",
            "absolute_low_energy_filter": False, "response_fitting": False,
            "variance_rule": "sum_j mean_d((cycle_j-mean)^2)/(3-1)", "budget_factor": 4 / 3,
            "target_detached": True, "heldout_input_fill": "zero", "outside_input_fill": "zero",
            "masked_visible_coordinates": "RMS-normalized IQ before phase correction",
            "global_RNG_consumed": False, "new_parameters": False,
            "upstream_equalization": "unknown", "IID_noise_proven": False, "noise_floor_proven": False,
            "scope": "periodic steady-state linear reference null subspace; no TX exclusivity, DG or removal of actual RX information claim",
        },
    }
