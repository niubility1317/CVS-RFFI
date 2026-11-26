"""Partition exact synthetic source noise by frequency and by chirp coherence."""

from pathlib import Path

import numpy as np
import torch

from .dura_repair import real_iq
from .w24_edit_budget_audit import CENTER, FULL, RUN as ORACLE_RUN, predict, receive_center, score
from .w24_edit_cross import check_controls
from .w24_native_full import VIEWS, classify_aperture
from experiments.cric_dura_comparison.io import read


RUN = '20261126-diagnostic-lora-w24-noise-components-s2026112559-r01'
MODES = ('direct', 'outer_half', 'exact_near', 'exact_far', 'exact_common', 'exact_private',
         'far_with_private_near', 'near_with_private_far', 'common_matched', 'private_matched', 'exact_center')
CONTRASTS = (*((key, 'outer_half') for key in MODES[2:]),
             ('private_matched', 'common_matched'), ('exact_center', 'exact_private'),
             ('exact_center', 'far_with_private_near'), ('exact_center', 'near_with_private_far'),
             ('far_with_private_near', 'exact_private'), ('near_with_private_far', 'exact_private'))


def power(value):
    return value.abs().square().mean((-2, -1), keepdim=True)


def component_edits(noise, phase):
    """Project measured noise into orthogonal subspaces in its aligned receive basis."""
    frame = phase / phase.abs().clamp_min(1e-12)
    spec = torch.fft.fft(noise * frame.conj(), norm='ortho')
    bins = (torch.fft.fftfreq(CENTER, device=noise.device) * CENTER).round().abs()
    near = bins <= 32
    common = spec.mean(-2, keepdim=True)
    private = spec - common
    common = common.expand_as(spec)

    def inverse(x):
        return -torch.fft.ifft(x, norm='ortho') * frame

    common_edit, private_edit = inverse(common), inverse(private)
    common_power, private_power = power(common_edit), power(private_edit)
    matched_power = torch.minimum(common_power, private_power)

    def matched(edit, original):
        ratio = matched_power / torch.where(original > 0, original, torch.ones_like(original))
        return edit * ratio.sqrt()

    return (
        inverse(spec * near), inverse(spec * ~near), common_edit, private_edit,
        inverse(spec * ~near + private * near), inverse(spec * near + private * ~near),
        matched(common_edit, common_power), matched(private_edit, private_power), -noise,
    )


@torch.no_grad()
def render(f, g, tracker, received, signal, device):
    original, learned, noise, phase = receive_center(g, tracker, received, signal, device)
    scores = [classify_aperture(f, original)]
    denominator = power(noise)
    denominator = torch.where(denominator > 0, denominator, torch.ones_like(denominator))
    measures = []
    for edit in (learned, *component_edits(noise, phase)):
        candidate = original.clone()
        candidate[..., :CENTER] += real_iq(edit)
        scores.append(classify_aperture(f, candidate))
        measures.append(torch.cat((power(edit) / denominator, power(noise + edit) / denominator), dim=-1))
    diagnostics = torch.cat(measures, dim=-2)
    if not torch.isfinite(diagnostics).all():
        raise ValueError('Nonfinite source noise projection')
    return np.stack(scores), diagnostics.cpu().numpy()


def summarize(values, is_clean):
    if is_clean:
        return {}
    stats = np.quantile(values, [.1, .5, .9], axis=0).transpose(1, 2, 0)
    return {mode: dict(edit_to_noise_power=row[0].tolist(), remaining_noise_power=row[1].tolist())
            for mode, row in zip(MODES[1:], stats, strict=True)}


def predict_components(rt):
    fields = dict(frequency_split_abs_bin=32, common_projection='equal aligned chirps in receive scale',
                  equal_energy_comparison='per-packet weaker of the two components',
                  exact_noise_source_diagnostic_only=True)
    predict(rt, modes=MODES, render_fn=render, summary_fn=summarize, measurement_shape=(len(MODES)-1, 2),
            waveform_fields=fields)


def score_components(runtime):
    root = Path(read(runtime)['run_root'])
    current = root / FULL
    with np.load(current / 'predictions.npz', allow_pickle=False) as matrix:
        check_controls(root.parent / ORACLE_RUN / FULL, ('direct', 'outer_half', 'oracle_exact'),
                       [0, 1, len(MODES)-1], read(current / 'source-predictions.json')['physical_ids'],
                       {v: matrix[v] for v in VIEWS})
    score(runtime, modes=MODES, contrasts=CONTRASTS)
