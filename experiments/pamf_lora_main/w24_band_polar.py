"""Source-only separation of 9–32 kHz edit phase, amplitude and energy."""

from pathlib import Path

import numpy as np
import torch

from experiments.cric_dura_comparison.io import read
from experiments.pamf_lora_main.dura_repair import real_iq
from experiments.pamf_lora_main.w24_band_direction import RUN as DIRECTION_RUN
from experiments.pamf_lora_main.w24_edit_budget_audit import CENTER, FULL, predict, receive_center
from experiments.pamf_lora_main.w24_native_full import VIEWS, classify_aperture


RUN = '20261126-diagnostic-lora-w24-band-polar-source-s2026112559-r01'
MODES = ('direct', 'outer_half', 'band_phase', 'band_amplitude', 'band_joint_matched', 'band_joint_exact')
CONTRASTS = (*((mode, 'outer_half') for mode in MODES[2:]),
             ('band_joint_matched', 'band_phase'), ('band_joint_matched', 'band_amplitude'),
             ('band_joint_exact', 'band_joint_matched'))
MEASURES = ('edit_to_learned_rms', 'remaining_noise_rms')
EPS = 1e-12


def band_mask(like):
    length = like.shape[-1]
    k = torch.arange(length, device=like.device)
    absolute = torch.minimum(k, length - k)
    return (absolute >= 9) & (absolute <= 32)


def energy(x):
    return x.abs().square().sum(dim=(-2, -1), keepdim=True)


def band_interventions(learned, noise):
    mask = band_mask(learned)
    old = learned * mask
    ref = -noise * mask
    old_energy, ref_energy = energy(learned[..., mask]), energy(noise[..., mask])
    use = (old_energy > EPS) & (ref_energy > EPS)
    phase_only = torch.where(ref != 0, old.abs() * ref.sgn(), old)
    scaled_amplitude = old.sgn() * ref.abs()
    amplitude_energy = energy(scaled_amplitude)
    amplitude_only = scaled_amplitude * (old_energy / amplitude_energy.clamp_min(EPS)).sqrt()
    joint = ref * (old_energy / ref_energy.clamp_min(EPS)).sqrt()
    return (torch.where(mask & use, phase_only, learned),
            torch.where(mask & use & (amplitude_energy > EPS), amplitude_only, learned),
            torch.where(mask & use, joint, learned),
            torch.where(mask & (ref_energy > EPS), ref, learned))


@torch.no_grad()
def render(f, g, tracker, received, signal, device):
    original, edit, noise, phase = receive_center(g, tracker, received, signal, device)
    phase = phase / phase.abs().clamp_min(EPS)
    learned, exact_noise = (
        torch.fft.fft(value * phase.conj(), norm='ortho') for value in (edit, noise)
    )
    mask = band_mask(learned)
    old_energy, ref_energy = energy(learned * mask), energy(exact_noise * mask)
    results = [classify_aperture(f, original)]
    info = []
    for branch, spectrum in enumerate((learned, *band_interventions(learned, exact_noise))):
        difference = torch.fft.ifft(spectrum - learned, norm='ortho') * phase
        update = edit + difference
        if branch != 4:
            torch.testing.assert_close(energy(update), energy(edit), rtol=5e-5, atol=1e-8)
        candidate = original.clone()
        candidate[..., :CENTER] += real_iq(update)
        results.append(classify_aperture(f, candidate))
        corrected = spectrum * mask
        dose = (energy(corrected) / old_energy.clamp_min(EPS)).sqrt()
        remaining = (energy(corrected + exact_noise * mask) / ref_energy.clamp_min(EPS)).sqrt()
        info.append(torch.cat((dose, remaining), dim=-1).squeeze(-2))
    metrics = torch.stack(info, dim=1)
    if not torch.isfinite(metrics).all():
        raise ValueError('Nonfinite edit/noise metrics')
    return np.stack(results), metrics.cpu().numpy()


def summarize(values, is_clean):
    if is_clean:
        return {}
    quantiles = np.quantile(values, [.1, .5, .9], axis=0).transpose(1, 2, 0)
    return {mode: dict(zip(MEASURES, quantile.tolist(), strict=True))
            for mode, quantile in zip(MODES[1:], quantiles, strict=True)}


def predict_bands(rt):
    fields = dict(band_dechirped_khz=[9, 32], edit_energy='matched_except_named_exact_arm',
                  zero_noise='retain_learned')
    predict(rt, modes=MODES, render_fn=render, summary_fn=summarize,
            measurement_shape=(len(MODES)-1, len(MEASURES)), clean_reference=1, waveform_fields=fields)


def verify_reference(rt):
    root = Path(rt['run_root'])
    current, previous = root / FULL, root.parent / DIRECTION_RUN / FULL
    mark, prior = (read(d / 'source-predictions.json') for d in (current, previous))
    fields = ('status', 'views', 'physical_ids', 'source_truth_reads', 'target_reads')
    if (tuple(mark[k] for k in fields) != tuple(prior[k] for k in fields)
            or prior['status'] != 'PREDICTIONS_FROZEN' or prior['target_reads'] or prior['source_truth_reads']):
        raise ValueError('Mismatched prior Full source panel')
    old = prior['modes'].index('band_9_32')
    new = MODES.index('band_joint_matched')
    with (np.load(current / 'predictions.npz', allow_pickle=False) as a,
          np.load(previous / 'predictions.npz', allow_pickle=False) as b):
        if any(not np.array_equal(a[scene][new], b[scene][old]) for scene in VIEWS):
            raise ValueError('Equal-energy narrow-band reference did not reproduce')
