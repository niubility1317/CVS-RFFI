"""Source-paired complex spectral residual fitting for the frozen W24 DURA readout."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import importlib
from pathlib import Path, PurePosixPath

import numpy as np
import torch

if __package__ is None:
    from experiments import pamf_lora_main

    pamf_lora_main.__path__.append(str(Path(__file__).resolve().parent))

from experiments.cric_dura_comparison.io import read, tensor, write
from experiments.pamf_lora_main.dura_repair import complex_iq, real_iq
from experiments.pamf_lora_main.full_packet_views import native, source_contract
from experiments.pamf_lora_main.joint_unroll_source import coordinates
from experiments.pamf_lora_main.runner import activate
from experiments.pamf_lora_main.w24_edit_budget_audit import FULL, score
from experiments.pamf_lora_main.w24_edit_cross import check_controls
from experiments.pamf_lora_main.w24_factor_audit import load_classifier, panel
from experiments.pamf_lora_main.w24_native_full import SOURCE, VIEWS, classify_aperture
from experiments.pamf_lora_main.w24_native_residual_source import (
    AUG_RUN, BATCH, DATA_RUN, TRAIN_RUN, W24_RUN, dispatch,
)
from experiments.pamf_lora_main.w24_observed_dose import source_receive
from experiments.pamf_lora_main.w24_outer_objective_source import RUN as PARENT, load_candidate
from experiments.pamf_lora_main.w24_wide_transfer import load_restorer, received_aperture, training_runtime


RUN = '20261126-phase1-lora-w24-spectral-cov-source-s2026112991-r01'
SEED = 2026112991
FIT_COUNT, HOLD_COUNT, FIT_BATCH = 1024, 128, 12
CENTER, CHIRPS = 384, 8
BANDS = (*range(9, 33), *range(CENTER-32, CENTER-8))
MODES = ('direct', 'outer_half', 'independent', 'cross_chirp')
CONTRASTS = (('cross_chirp', 'independent'), ('cross_chirp', 'outer_half'), ('independent', 'outer_half'))
RIDGE = .05
GATE_RMS = .15
MAX_EXTRA = .5


def norm(x):
    return torch.linalg.vector_norm(x, dim=(-2, -1))


def solve(gram, rhs):
    """Fit the same ridge penalty with independent or jointly predicted chirps."""
    k = torch.arange(CHIRPS, device=gram.device).repeat(2)
    same = k[:, None] == k[None, :]
    output = k[:, None] == k[None, :CHIRPS]
    regularization = RIDGE * gram.diagonal(dim1=-2, dim2=-1).real.mean(-1).clamp_min(1e-12)
    eye = torch.eye(2*CHIRPS, dtype=gram.dtype, device=gram.device)
    penalty = regularization[:, None, None] * eye
    independent = torch.linalg.solve(gram * same + penalty, rhs * output)
    joint = torch.linalg.solve(gram + penalty, rhs)
    return torch.stack((independent, joint))


def spectral_inputs(observed, delta, phase):
    scale = norm(observed).clamp_min(1e-9)
    edit_fraction = norm(delta) / scale
    gate = (edit_fraction / GATE_RMS).clamp(max=1.)
    fft = torch.fft.fft(torch.cat((observed * gate[:, None, None], delta), dim=1)
                        * phase.repeat(1, 2, 1).conj(), norm='ortho')
    return (fft[..., BANDS] / scale[:, None, None]).transpose(-1, -2), scale


def apply(weights, features, phase, scale, delta):
    spectrum = torch.zeros((len(features), CHIRPS, CENTER), dtype=delta.dtype, device=delta.device)
    spectrum[..., BANDS] = torch.einsum('nfi,fio->nfo', features, weights).transpose(-1, -2)
    extra = torch.fft.ifft(spectrum, norm='ortho') * phase * scale[:, None, None]
    limiter = (MAX_EXTRA * norm(delta) / norm(extra).clamp_min(1e-12)).clamp(max=1.)
    return extra * limiter[:, None, None]


def parent_delta(g, tracker, center, edges):
    y, phase, quality, rms = coordinates(tracker, real_iq(center))
    stages, *_ = g.propose(y, quality, rms, edges)
    return ((stages[-1] - y) * phase * rms) * .5, phase


def restore_variants(g, tracker, center, edges, weights):
    delta, phase = parent_delta(g, tracker, center, edges)
    features, scale = spectral_inputs(center, delta, phase)
    extra = tuple(apply(w, features, phase, scale, delta) for w in weights)
    return delta, extra


def parent_model(rt, source, device):
    activate(rt)
    contract = source_contract(source)
    _, tracker, _, _ = load_restorer(*training_runtime(rt), contract, device)
    model, _ = load_candidate(Path(rt['run_root']).parent/PARENT, contract, device)
    return model, tracker, contract


def fit_examples(raw, tracker, generic, rng, device):
    data = tensor(np.stack((raw.real, raw.imag), axis=-1), device)
    packet = torch.view_as_complex(data)
    rms = packet.abs().square().mean(-1, keepdim=True).clamp_min(1e-12).sqrt()
    packet = packet / rms
    clean = tracker.aperture(packet)
    disturbed = tracker.aperture(generic(packet.unflatten(-1, (CHIRPS, 1024)), rng).flatten(-2))
    return torch.cat((disturbed, clean)), clean[..., :CENTER].repeat(2, 1, 1)


def fit_batches(iq, rows, tracker, generic, rng, device):
    for start in range(0, len(rows), FIT_BATCH):
        yield fit_examples(iq[rows[start:start+FIT_BATCH]], tracker, generic, rng, device)


@torch.no_grad()
def fit(rt):
    root, source, device = Path(rt['run_root']), Path(rt['source_root']), 'cuda:0'
    g, tracker, contract = parent_model(rt, source, device)
    generic = importlib.import_module('lora_wide_dura_archive.dura_w24_train').condition_waveforms
    source_u = native(source, 'U_s', contract['counts']['U_s'])
    preview, _ = fit_examples(source_u[:1], tracker, generic, torch.Generator(device=device).manual_seed(SEED), device)
    first, phase = parent_delta(g, tracker, preview[..., :CENTER], preview[..., CENTER:])
    features, _ = spectral_inputs(preview[..., :CENTER], first, phase)
    if features.shape != (2, len(BANDS), 2*CHIRPS) or not torch.isfinite(features).all():
        raise ValueError('Frozen G fit smoke failed')
    write(root/'fit-smoke.json', dict(status='PASS', source_u_packets=1, F_reads=0, target_reads=0))
    print('W24_SPECTRAL_COV_FIT_SMOKE PASS', flush=True)
    panel = np.random.default_rng(SEED).choice(len(source_u), FIT_COUNT+HOLD_COUNT, replace=False)
    rng = torch.Generator(device=device).manual_seed(SEED)
    gram = torch.zeros((len(BANDS), 2*CHIRPS, 2*CHIRPS), dtype=torch.complex128, device=device)
    rhs = torch.zeros((len(BANDS), 2*CHIRPS, CHIRPS), dtype=gram.dtype, device=device)
    for index, (observed, clean) in enumerate(fit_batches(source_u, panel[:FIT_COUNT], tracker, generic, rng, device)):
        center = observed[..., :CENTER]
        delta, phase = parent_delta(g, tracker, center, observed[..., CENTER:])
        f, scale = spectral_inputs(center, delta, phase)
        residual = torch.fft.fft((clean-center-delta) * phase.conj(), norm='ortho')
        target = (residual[..., BANDS] / scale[:, None, None]).permute(2, 0, 1)
        x = f.transpose(0, 1).to(gram.dtype)
        gram += x.mH @ x
        rhs += x.mH @ target.to(gram.dtype)
        if (index+1) % 16 == 0:
            print('W24_SPECTRAL_COV_FIT', index+1, flush=True)
    weights = solve(gram, rhs).to(torch.complex64)
    if not torch.isfinite(weights).all():
        raise ValueError('Non-finite source fit')

    errors = torch.zeros((3, 2), device=device)
    for observed, clean in fit_batches(source_u, panel[FIT_COUNT:], tracker, generic, rng, device):
        center = observed[..., :CENTER]
        delta, extra = restore_variants(g, tracker, center, observed[..., CENTER:], weights)
        scale = norm(center).clamp_min(1e-9)
        for arm, change in enumerate((delta, delta+extra[0], delta+extra[1])):
            errors[arm] += ((norm(center+change-clean)/scale).square().reshape(2, -1)).sum(-1)
    torch.save(weights.cpu(), root/'spectral-cov.pt')
    write(root/'fit-complete.json', dict(status='SOURCE_FROZEN', parent=PARENT, fit_packets=FIT_COUNT,
          heldout_packets=HOLD_COUNT, F_reads=0, source_truth_reads=0, source_V_reads=0, target_reads=0,
          pairs_per_fit_packet=['generic_same_packet', 'identity'], bands=list(BANDS), ridge=RIDGE,
          gate_rms=GATE_RMS, max_extra=MAX_EXTRA, seed=SEED,
          heldout_relative_error=(errors / HOLD_COUNT).tolist()))


@torch.no_grad()
def render(f, g, tracker, raw, weights, device):
    original = received_aperture(raw, device)
    center = complex_iq(original[..., :CENTER])
    delta, extra = restore_variants(g, tracker, center, complex_iq(original[..., CENTER:]), weights)
    scale = norm(center).clamp_min(1e-9)
    decision = [classify_aperture(f, original)]
    for edit in (delta, delta+extra[0], delta+extra[1]):
        updated = original.clone()
        updated[..., :CENTER] += real_iq(edit)
        decision.append(classify_aperture(f, updated))
    info = torch.stack((norm(delta)/scale, norm(extra[0])/scale, norm(extra[1])/scale), dim=-1)
    return np.stack(decision), np.asarray(info.tolist())


def predict(rt):
    root, source, device = Path(rt['run_root']), Path(rt['source_root']), 'cuda:0'
    proof = read(root/'fit-complete.json')
    if (proof['status'], proof['parent'], proof['bands'], proof['F_reads'], proof['target_reads'],
            proof['source_truth_reads'], proof['source_V_reads']) != ('SOURCE_FROZEN', PARENT, list(BANDS), 0, 0, 0, 0):
        raise ValueError('Source-only coefficients must be frozen before reading the W24 reader')
    weights = torch.load(root/'spectral-cov.pt', map_location=device, weights_only=True)
    if weights.shape != (2, len(BANDS), 2*CHIRPS, CHIRPS) or not torch.isfinite(weights).all():
        raise ValueError('Invalid source coefficients')
    g, tracker, contract = parent_model(rt, source, device)
    f = load_classifier(rt, contract, device)
    u = native(source, 'U_s', contract['counts']['U_s'])[0]
    raw = np.stack([source_receive(view, (0, SOURCE[0], u)) for view in ('clean', VIEWS[-1])])
    pred, info = render(f, g, tracker, raw, weights, device)
    if pred.shape != (len(MODES), 2) or not np.isfinite(info).all():
        raise ValueError('Real frozen source smoke failed')
    write(root/'smoke.json', dict(status='PASS', source_u_packets=1, target_reads=0, source_truth_reads=0))
    print('W24_SPECTRAL_COV_SMOKE PASS', flush=True)

    ix, rx, ids = panel(source, contract)
    raw = native(source, 'V', contract['counts']['V'])[ix]
    jobs = [(int(i), SOURCE[int(j)], x) for i, j, x in zip(ix, rx, raw, strict=True)]
    matrix, dose = {}, {}
    with ProcessPoolExecutor(max_workers=rt['cpu_workers']) as pool:
        for view in VIEWS:
            stream = pool.map(partial(source_receive, view), jobs, chunksize=6)
            rendered = []
            for off in range(0, len(jobs), BATCH):
                raw = np.stack([next(stream) for _ in range(min(BATCH, len(jobs)-off))])
                rendered.append(render(f, g, tracker, raw, weights, device))
            labels, strength = zip(*rendered, strict=True)
            matrix[view] = np.concatenate(labels, axis=1)
            dose[view] = np.quantile(np.concatenate(strength), [.1, .5, .9], axis=0).T.tolist()
            print('W24_SPECTRAL_COV_PREDICTED', view, len(jobs), flush=True)
    check_controls(root.parent/PARENT/FULL, ('direct', 'outer_objective_half'), [0, 1], ids, matrix)
    output = root/FULL
    output.mkdir()
    np.savez(output/'predictions.npz', **matrix)
    write(output/'source-predictions.json', dict(status='PREDICTIONS_FROZEN', modes=list(MODES),
          physical_ids=ids, views=list(VIEWS), target_reads=0, source_truth_reads=0,
          waveform=dict(route=FULL, receiver_only=True, frozen_controls_match=True,
                        order_dose=['outer_half', 'independent_extra', 'cross_chirp_extra'], dose_q10_q50_q90=dose)))


def main():
    from experiments.pamf_lora_main.publish import release, release_commit

    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest='action', required=True)
    phases = ('fit-source', 'predict-source', 'score-source')
    for phase in ('dispatch', *phases):
        actions.add_parser(phase).add_argument('runtime', type=Path)
    pub = actions.add_parser('publish')
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
                  allowed_gpus=[7, 3, 6, 0], cpu_workers=8)
        rt['external_releases'] = [rt[field] for field in ('source_root', 'classifier_root',
                                                           'w24_release_root', 'training_release_root')]
        rt['external_releases'].append(str(runs/PARENT))
        release(args.repo, args.output, rt, args.ssh_config)


if __name__ == '__main__':
    main()
