"""Full source attribution of the learned W24 edit direction by frequency."""

import numpy as np
import torch

from experiments.pamf_lora_main.dura_repair import real_iq
from experiments.pamf_lora_main.w24_edit_budget_audit import CENTER, predict, receive_center
from experiments.pamf_lora_main.w24_native_full import classify_aperture


RUN = '20261126-diagnostic-lora-w24-band-direction-source-s2026112559-r01'
BANDS = (('band_1_8', 1, 8), ('band_9_32', 9, 32), ('band_33_64', 33, 64), ('band_65_192', 65, 192))
MODES = ('direct', 'outer_half', *(name for name, _, _ in BANDS), 'all_bands')
CONTRASTS = tuple((mode, 'outer_half') for mode in MODES[2:])
METRICS = ('learned_energy', 'noise_energy', 'direction_cosine', 'remaining_rms_ratio', 'replacement_defined')


def matched_spectra(learned, noise):
    """Replace only complex direction, matching the learned energy in each band."""
    length = learned.shape[-1]
    bins = torch.arange(length, device=learned.device)
    distance = torch.minimum(bins, length - bins)
    masks = tuple((distance >= lo) & (distance <= hi) for _, lo, hi in BANDS)
    patched = learned
    measurements = []
    for mask in masks:
        original, direction = learned[..., mask], -noise[..., mask]
        original_energy = original.abs().square().sum((-2, -1))
        noise_energy = direction.abs().square().sum((-2, -1))
        available = (noise_energy > 1e-12) & (original_energy > 1e-12)
        ratio = (original_energy / noise_energy.clamp_min(1e-12)).sqrt()
        patched = torch.where(mask & available[:, None, None], -noise * ratio[:, None, None], patched)
        dot = (original * direction.conj()).real.sum((-2, -1))
        cosine = dot / (original_energy * noise_energy).clamp_min(1e-24).sqrt()
        residual = (original - direction).abs().square().sum((-2, -1))
        metrics = (original_energy, noise_energy, cosine,
                   (residual / noise_energy.clamp_min(1e-12)).sqrt(), available)
        measurements.append(torch.stack(metrics, dim=-1))
    return patched, masks, torch.stack(measurements, dim=1)


@torch.no_grad()
def render(f, g, tracker, received, signal, device):
    original, edit, noise, phase = receive_center(g, tracker, received, signal, device)
    phase = phase / phase.abs().clamp_min(1e-12)
    learned, exact_noise = (
        torch.fft.fft(value * phase.conj(), norm='ortho') for value in (edit, noise)
    )
    patched, masks, info = matched_spectra(learned, exact_noise)
    if not torch.isfinite(info).all():
        raise ValueError('Nonfinite band attribution metrics')
    controls = [classify_aperture(f, original)]
    half = original.clone()
    half[..., :CENTER] += real_iq(edit)
    controls.append(classify_aperture(f, half))
    dose = edit.abs().square().sum((-2, -1))
    diff = patched - learned
    for change in (*(diff * mask for mask in masks), diff):
        edit_difference = torch.fft.ifft(change, norm='ortho') * phase
        updated = edit + edit_difference
        torch.testing.assert_close(updated.abs().square().sum((-2, -1)), dose, rtol=5e-5, atol=1e-8)
        candidate = original.clone()
        candidate[..., :CENTER] += real_iq(updated)
        controls.append(classify_aperture(f, candidate))
    return np.stack(controls), info.cpu().numpy()


def summarize(values, is_clean):
    bands = {}
    for index, (name, _, _) in enumerate(BANDS):
        data = values[:, index]
        measured = data[:, -1].astype(bool)
        metrics = dict(
            replaced=int(measured.sum()),
            learned_energy=np.quantile(data[:, 0], [.1, .5, .9]).tolist(),
            noise_energy=np.quantile(data[:, 1], [.1, .5, .9]).tolist(),
        )
        if not is_clean and measured.any():
            quality = np.quantile(data[measured, 2:4], [.1, .5, .9], axis=0).T
            metrics.update(zip(METRICS[2:4], quality.tolist(), strict=True))
        bands[name] = metrics
    return bands


def predict_bands(rt):
    fields = dict(bands_dechirped_khz=[list(band) for band in BANDS],
                  edit_energy='matched_per_packet_and_band', zero_noise='retain_learned')
    predict(rt, modes=MODES, render_fn=render, summary_fn=summarize,
            measurement_shape=(len(BANDS), len(METRICS)), clean_reference=1, waveform_fields=fields)
