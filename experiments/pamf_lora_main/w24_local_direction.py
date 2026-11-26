"""Matched source-only frequency and chirp contexts for actual W24 band edits."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import importlib
from pathlib import Path, PurePosixPath

import numpy as np
import torch
from torch import nn

if __package__ is None:
    from experiments import pamf_lora_main

    pamf_lora_main.__path__.append(str(Path(__file__).resolve().parent))

from experiments.cric_dura_comparison.io import read, write
from experiments.pamf_lora_main.dura_repair import complex_iq, real_iq
from experiments.pamf_lora_main.full_packet_views import native
from experiments.pamf_lora_main.w24_edit_budget_audit import FULL, score
from experiments.pamf_lora_main.w24_edit_cross import check_controls
from experiments.pamf_lora_main.w24_factor_audit import load_classifier, panel
from experiments.pamf_lora_main.w24_native_full import SOURCE, VIEWS, classify_aperture
from experiments.pamf_lora_main.w24_native_residual_source import (
    AUG_RUN, BATCH, DATA_RUN, TRAIN_RUN, W24_RUN, dispatch,
)
from experiments.pamf_lora_main.w24_observed_dose import source_receive
from experiments.pamf_lora_main.w24_spectral_cov import (
    BANDS, CENTER, CHIRPS, PARENT, fit_batches, norm, parent_delta, parent_model, received_aperture,
)


RUN = '20261126-phase1-lora-w24-local-direction-s2026112991-r01'
SEED = 2026112991
FIT_PACKETS, HOLD_PACKETS = 1536, 128
UPDATES, TRAIN_BATCH = 384, 48
MODES = ('direct', 'outer_half', 'chirp_context', 'frequency_context')
CONTRASTS = tuple((MODES[i], MODES[j]) for i, j in ((3, 2), (3, 1), (2, 1)))


def with_neighbors(fields, axis, width):
    moved = fields.movedim(axis, -2)
    positions = torch.arange(moved.shape[-2], device=fields.device)
    low = positions - positions.remainder(width)
    left = torch.maximum(low, positions-1)
    right = torch.minimum(low+width-1, positions+1)
    neighbors = (moved, moved.index_select(-2, left), moved.index_select(-2, right))
    return torch.cat(neighbors, dim=-1).movedim(-2, axis)


class Direction(nn.Module):
    def __init__(self, frequency):
        super().__init__()
        self.frequency = frequency
        self.net = nn.Sequential(nn.Linear(21, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU(),
                                 nn.Linear(64, 2))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, fields, unit):
        axis, width = (-2, len(BANDS)//2) if self.frequency else (-3, CHIRPS)
        features = with_neighbors(fields, axis, width)
        change = torch.view_as_complex(self.net(features).contiguous())
        estimate = unit + change
        magnitude = norm(estimate)
        direction = estimate / magnitude.clamp_min(1e-8)[:, None, None]
        return torch.where(magnitude[:, None, None] > 1e-8, direction, unit)


def fields_for(center, delta, phase):
    observed = torch.fft.fft(center*phase.conj(), norm='ortho')[..., BANDS]
    edit = torch.fft.fft(delta*phase.conj(), norm='ortho')
    band = edit[..., BANDS]
    scale = norm(center).clamp_min(1e-8)
    band_power = norm(band)
    unit = band / band_power.clamp_min(1e-8)[:, None, None]
    observed = observed / scale[:, None, None]
    fraction = (norm(delta)/scale)[:, None, None].expand_as(band.real)
    bins = band.real.new_tensor(BANDS)
    frequency = torch.where(bins <= CENTER//2, bins, bins-CENTER) / 32
    chirp = torch.linspace(-1., 1., CHIRPS, device=center.device)[None, :, None]
    fields = torch.stack((observed.real, observed.imag, unit.real, unit.imag, fraction,
                          frequency.expand_as(band.real), chirp.expand_as(band.real)), dim=-1)
    return fields, unit, edit, band_power


def restore_band(model, fields, unit, original, magnitude, phase):
    spectrum = original.clone()
    spectrum[..., BANDS] = model(fields, unit) * magnitude[:, None, None]
    return torch.fft.ifft(spectrum, norm='ortho') * phase


@torch.no_grad()
def collect(g, tracker, array, rows, generic, rng, device):
    batches = []
    for observed, reference in fit_batches(array, rows, tracker, generic, rng, device):
        center = observed[..., :CENTER]
        delta, phase = parent_delta(g, tracker, center, observed[..., CENTER:])
        fields, unit, _, _ = fields_for(center, delta, phase)
        wanted = torch.fft.fft((reference-center)*phase.conj(), norm='ortho')[..., BANDS]
        target = wanted / norm(wanted).clamp_min(1e-8)[:, None, None]
        clean = torch.arange(len(center), device=device) >= len(center)//2
        batches.append((fields, unit, torch.where(clean[:, None, None], unit, target), clean))
        if len(batches) % 32 == 0:
            print('W24_LOCAL_DIRECTION_COLLECT', len(batches), flush=True)
    return tuple(torch.cat(items) for items in zip(*batches, strict=True))


def fit(rt):
    root, source, device = Path(rt['run_root']), Path(rt['source_root']), 'cuda:0'
    g, tracker, contract = parent_model(rt, source, device)
    generic = importlib.import_module('lora_wide_dura_archive.dura_w24_train').condition_waveforms
    iq = native(source, 'U_s', contract['counts']['U_s'])
    rng = torch.Generator(device=device).manual_seed(SEED)
    smoke = collect(g, tracker, iq, [0], generic, rng, device)
    if smoke[0].shape != (2, CHIRPS, len(BANDS), 7) or not torch.isfinite(smoke[0]).all():
        raise ValueError('Frozen source G smoke failed')
    write(root/'fit-smoke.json', dict(status='PASS', F_reads=0, target_reads=0, source_packets=1))
    selected = np.random.default_rng(SEED).choice(len(iq), FIT_PACKETS+HOLD_PACKETS, replace=False)
    fields, unit, target, clean = collect(g, tracker, iq, selected[:FIT_PACKETS], generic, rng, device)
    hold = collect(g, tracker, iq, selected[FIT_PACKETS:], generic, rng, device)
    torch.manual_seed(SEED)
    models = [Direction(False).to(device), Direction(True).to(device)]
    models[1].load_state_dict(models[0].state_dict())
    optimizers = [torch.optim.AdamW(model.parameters(), lr=.001) for model in models]
    for step in range(UPDATES):
        idx = torch.randint(len(fields), (TRAIN_BATCH,), device=device, generator=rng)
        losses = []
        for model, opt in zip(models, optimizers, strict=True):
            guess = model(fields[idx], unit[idx])
            distance = norm(guess-target[idx]).square()
            loss = (distance * torch.where(clean[idx], 2., 1.)).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
        if (step+1) % 96 == 0:
            if not np.isfinite(losses).all():
                raise ValueError('Nonfinite local direction fit')
            print('W24_LOCAL_DIRECTION_STEP', step+1, losses, flush=True)
    hfields, hunit, htarget, hclean = hold
    with torch.no_grad():
        guesses = (hunit, *(model(hfields, hunit) for model in models))
        errors = [torch.stack((norm(w[~hclean]-htarget[~hclean]).square().mean(),
                               norm(w[hclean]-htarget[hclean]).square().mean())).tolist() for w in guesses]
    torch.save([model.cpu().state_dict() for model in models], root/'direction.pt')
    write(root/'fit-complete.json', dict(status='SOURCE_FROZEN', parent=PARENT, seed=SEED, updates=UPDATES,
          fit_packets=FIT_PACKETS, heldout_packets=HOLD_PACKETS, heldout_direction_error=errors,
          F_reads=0, target_reads=0, source_truth_reads=0, source_V_reads=0,
          parameters=sum(p.numel() for p in models[0].parameters())))


@torch.no_grad()
def render(f, g, tracker, models, raw, device):
    original = received_aperture(raw, device)
    center = complex_iq(original[..., :CENTER])
    edit, phase = parent_delta(g, tracker, center, complex_iq(original[..., CENTER:]))
    fields, unit, spectrum, magnitude = fields_for(center, edit, phase)
    estimated = tuple(restore_band(m, fields, unit, spectrum, magnitude, phase) for m in models)
    labels = [classify_aperture(f, original)]
    for delta in (edit, *estimated):
        updated = original.clone()
        updated[..., :CENTER] += real_iq(delta)
        labels.append(classify_aperture(f, updated))
    ratios = torch.stack([norm(value-edit)/norm(center).clamp_min(1e-8) for value in estimated], dim=-1)
    return np.stack(labels), np.asarray(ratios.tolist())


def predict(rt):
    root, source, device = Path(rt['run_root']), Path(rt['source_root']), 'cuda:0'
    proof = read(root/'fit-complete.json')
    if (proof['status'], proof['parent'], proof['updates']) != ('SOURCE_FROZEN', PARENT, UPDATES) or any(
            proof[key] for key in ('F_reads','target_reads','source_truth_reads','source_V_reads')):
        raise ValueError('Local direction must be fit and frozen before the W24 load')
    states = torch.load(root/'direction.pt', map_location=device, weights_only=True)
    models = tuple(Direction(freq).to(device).eval() for freq in (False, True))
    for model, state in zip(models, states, strict=True):
        model.load_state_dict(state)
    g, tracker, contract = parent_model(rt, source, device)
    f = load_classifier(rt, contract, device)
    u = native(source, 'U_s', contract['counts']['U_s'])[0]
    measured = np.stack([source_receive(view, (0, SOURCE[0], u)) for view in ('clean', VIEWS[-1])])
    labels, dose = render(f, g, tracker, models, measured, device)
    if labels.shape != (len(MODES), 2) or not np.isfinite(dose).all():
        raise ValueError('Real frozen W24 smoke failed')
    write(root/'smoke.json', dict(status='PASS', source_u_packets=1, source_truth_reads=0, target_reads=0))
    print('W24_LOCAL_DIRECTION_SMOKE PASS', flush=True)
    ix, rx, ids = panel(source, contract)
    v = native(source, 'V', contract['counts']['V'])[ix]
    jobs = [(int(i), SOURCE[int(j)], raw) for i, j, raw in zip(ix, rx, v, strict=True)]
    matrix, info = {}, {}
    with ProcessPoolExecutor(max_workers=rt['cpu_workers']) as pool:
        for view in VIEWS:
            stream = pool.map(partial(source_receive, view), jobs, chunksize=6)
            parts = []
            for start in range(0, len(jobs), BATCH):
                raw = np.stack([next(stream) for _ in range(min(BATCH, len(jobs)-start))])
                parts.append(render(f, g, tracker, models, raw, device))
            labels, ratios = zip(*parts, strict=True)
            matrix[view] = np.concatenate(labels, axis=1)
            info[view] = np.quantile(np.concatenate(ratios), [.1, .5, .9], axis=0).T.tolist()
            print('W24_LOCAL_DIRECTION_PREDICTED', view, len(jobs), flush=True)
    check_controls(root.parent/PARENT/FULL, ('direct', 'outer_objective_half'), [0, 1], ids, matrix)
    out = root/FULL
    out.mkdir()
    np.savez(out/'predictions.npz', **matrix)
    write(out/'source-predictions.json', dict(status='PREDICTIONS_FROZEN', modes=list(MODES), views=list(VIEWS),
          physical_ids=ids, target_reads=0, source_truth_reads=0,
          waveform=dict(route=FULL, frozen_controls_match=True, delta_q10_q50_q90=info)))


def main():
    from experiments.pamf_lora_main.publish import release, release_commit

    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    phases = ('fit-source', 'predict-source', 'score-source')
    for phase in ('dispatch', *phases):
        sub.add_parser(phase).add_argument('runtime', type=Path)
    pub = sub.add_parser('publish')
    for field in ('repo', 'context', 'ssh-config', 'output'):
        pub.add_argument('--'+field, type=Path, required=True)
    args = parser.parse_args()
    if args.action == 'dispatch':
        dispatch(args.runtime, entry=__file__, phases=phases, gpu_phases=phases[:2])
    elif args.action == 'fit-source':
        fit(read(args.runtime))
    elif args.action == 'predict-source':
        predict(read(args.runtime))
    elif args.action == 'score-source':
        score(args.runtime, modes=MODES, contrasts=CONTRASTS)
    else:
        ctx = read(args.context)
        rt = {field:ctx[field] for field in ('project', 'python', 'frozen_sources')}
        project = PurePosixPath(rt['project'])
        runs, releases = project/'runs', project/'releases'
        rt.update(run_id=RUN, runner=__spec__.name, code_commit=release_commit(args.repo),
                  release_root=str(releases/f'{RUN}-release'), run_root=str(runs/RUN),
                  source_root=str(releases/f'{DATA_RUN}-release'/'data/source'), classifier_root=str(runs/AUG_RUN),
                  w24_release_root=str(releases/f'{W24_RUN}-release'),
                  training_release_root=str(releases/f'{TRAIN_RUN}-release'),
                  allowed_gpus=[3, 7, 0, 6], cpu_workers=8)
        rt['external_releases'] = [rt[field] for field in ('source_root','classifier_root','w24_release_root',
                                                           'training_release_root')]
        rt['external_releases'].append(str(runs/PARENT))
        release(args.repo, args.output, rt, args.ssh_config)


if __name__ == '__main__':
    main()
