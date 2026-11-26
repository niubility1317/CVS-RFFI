"""Full source-only crossover of frozen W24 DURA edit direction and strength."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import faulthandler
from functools import partial
from pathlib import Path, PurePosixPath

import numpy as np
import torch

if __package__ is None:
    from experiments import pamf_lora_main

    pamf_lora_main.__path__.append(str(Path(__file__).resolve().parent))

from experiments.cric_dura_comparison.io import read, write
from experiments.pamf_lora_main.dura_repair import complex_iq, real_iq
from experiments.pamf_lora_main.full_packet_views import native, source_contract
from experiments.pamf_lora_main.joint_unroll_source import coordinates
from experiments.pamf_lora_main.runner import activate
from experiments.pamf_lora_main.w24_edit_budget_audit import FULL, score
from experiments.pamf_lora_main.w24_factor_audit import load_classifier, panel
from experiments.pamf_lora_main.w24_native_full import SOURCE, VIEWS, classify_aperture
from experiments.pamf_lora_main.w24_native_residual_source import (
    AUG_RUN, BATCH, DATA_RUN, TRAIN_RUN, W24_RUN, dispatch,
)
from experiments.pamf_lora_main.w24_observed_dose import source_receive
from experiments.pamf_lora_main.w24_outer_objective_source import RUN as PARENT, load_candidate as load_parent
from experiments.pamf_lora_main.w24_third_prior import HALF_PRIMARY_RUN, load_candidate
from experiments.pamf_lora_main.w24_wide_transfer import load_restorer, received_aperture, training_runtime


RUN = '20261126-diagnostic-lora-w24-edit-cross-source-s2026112559-r03'
CENTER = 384
BUCKETS = (('dc', 0, 0), ('1-8', 1, 8), ('9-32', 9, 32), ('33-64', 33, 64), ('65-192', 65, 192))
MODES = ('direct', 'outer_half', 'half_primary', 'outer_direction_total', 'half_direction_total',
         'outer_direction_bands', 'half_direction_bands')
CONTRASTS = (
    ('half_primary', 'outer_half'),
    ('outer_direction_total', 'outer_half'), ('half_direction_total', 'half_primary'),
    ('outer_direction_total', 'half_primary'), ('half_direction_total', 'outer_half'),
    ('outer_direction_bands', 'outer_direction_total'), ('half_direction_bands', 'half_direction_total'),
    ('outer_direction_bands', 'half_primary'), ('half_direction_bands', 'outer_half'),
)


def norm(x):
    return torch.linalg.vector_norm(x, dim=(-2, -1))


def match_norm(direction, strength):
    base = norm(direction)
    scale = torch.where(base > 1e-12, norm(strength) / base.clamp_min(1e-12), 0.)
    return direction * scale[..., None, None]


def crossover(outer, half, phase):
    old, new = (torch.fft.fft(edit * phase.conj(), norm='ortho') for edit in (outer, half))
    bins = torch.arange(CENTER, device=outer.device)
    bins = torch.minimum(bins, CENTER - bins)
    old_bands, new_bands = old.clone(), new.clone()
    metrics = []
    for name, low, high in BUCKETS:
        if name == 'dc':
            continue
        mask = (bins >= low) & (bins <= high)
        u, v = old[..., mask], new[..., mask]
        old_bands[..., mask], new_bands[..., mask] = match_norm(u, v), match_norm(v, u)
        denom = norm(u) * norm(v)
        cos = (u.conj() * v).real.sum(dim=(-2, -1)) / denom.clamp_min(1e-12)
        metrics.extend((norm(u), norm(v), cos))
    spectra = (match_norm(old, new), match_norm(new, old), old_bands, new_bands)
    edits = tuple(torch.fft.ifft(spectrum, norm='ortho') * phase for spectrum in spectra)
    return edits, torch.stack(metrics, dim=-1)


@torch.no_grad()
def render(f, restorers, tracker, raw, device):
    observed = received_aperture(raw, device)
    center = observed[..., :CENTER]
    y, phase, quality, rms = coordinates(tracker, center)
    guard = complex_iq(observed[..., CENTER:])
    edits = []
    for model in restorers:
        stages, *_ = model.propose(y, quality, rms, guard)
        edits.append(((stages[-1] - y) * phase * rms) * .5)
    crossed, metrics = crossover(*edits, phase)
    received_norm = norm(complex_iq(center)).clamp_min(1e-12)
    metrics[:, 0::3] /= received_norm[:, None]
    metrics[:, 1::3] /= received_norm[:, None]
    decisions = [classify_aperture(f, observed)]
    for edit in (*edits, *crossed):
        recovered = observed.clone()
        recovered[..., :CENTER] += real_iq(edit)
        decisions.append(classify_aperture(f, recovered))
    return np.stack(decisions), np.asarray(metrics.tolist(), dtype=np.float32)


def check_controls(archive, names, columns, ids, matrix):
    proof = read(archive/'source-predictions.json')
    if tuple(proof[k] for k in ('status', 'physical_ids', 'views', 'source_truth_reads', 'target_reads')) != (
            'PREDICTIONS_FROZEN', ids, list(VIEWS), 0, 0):
        raise ValueError('Archived same-packet Full source definition differs')
    previous = [proof['modes'].index(mode) for mode in names]
    with np.load(archive/'predictions.npz', allow_pickle=False) as frozen:
        if any(not np.array_equal(matrix[view][columns], frozen[view][previous]) for view in VIEWS):
            raise ValueError(f'Replayed frozen decisions differ for {names}')


def predict(rt):
    faulthandler.enable()
    activate(rt)
    root, source = Path(rt['run_root']), Path(rt['source_root'])
    device = 'cuda:0'
    contract = source_contract(source)
    print('W24_EDIT_CROSS_SOURCE_CONTRACT', flush=True)
    _, tracker, _, _ = load_restorer(*training_runtime(rt), contract, device)
    parent, half_training = root.parent/PARENT, root.parent/HALF_PRIMARY_RUN
    outer, outer_info = load_parent(parent, contract, device)
    half, half_info = load_candidate(half_training, contract, device, 'half_primary')
    print('W24_EDIT_CROSS_RESTORERS_FROZEN', flush=True)
    f = load_classifier(rt, contract, device)
    print('W24_EDIT_CROSS_READER_FROZEN', flush=True)
    u = native(source, 'U_s', contract['counts']['U_s'])[0]
    raw = np.stack([source_receive(view, (0, SOURCE[0], u)) for view in ('clean', VIEWS[-1])])
    decisions, metrics = render(f, (outer, half), tracker, raw, device)
    if decisions.shape != (len(MODES), 2) or not np.isfinite(metrics).all():
        raise ValueError('Full source-only smoke failed')
    write(root/'smoke.json', dict(status='PASS', source_packets=1, target_reads=0, source_truth_reads=0,
          parameters=[outer_info['parameters'], half_info['parameters']]))
    print('W24_EDIT_CROSS_SMOKE PASS', flush=True)

    ix, rx, ids = panel(source, contract)
    packets = native(source, 'V', contract['counts']['V'])[ix]
    jobs = [(int(i), SOURCE[int(r)], p) for i, r, p in zip(ix, rx, packets, strict=True)]
    matrix, diagnostics = {}, {}
    with ProcessPoolExecutor(max_workers=rt['cpu_workers']) as pool:
        for view in VIEWS:
            stream = pool.map(partial(source_receive, view), jobs, chunksize=6)
            rendered = []
            for off in range(0, len(jobs), BATCH):
                raw = np.stack([next(stream) for _ in range(min(BATCH, len(jobs)-off))])
                rendered.append(render(f, (outer, half), tracker, raw, device))
            decisions, metrics = zip(*rendered, strict=True)
            matrix[view] = np.concatenate(decisions, axis=1)
            values = np.concatenate(metrics)
            if not np.isfinite(values).all():
                raise ValueError(f'Non-finite crossover diagnostics for {view}')
            quantiles = np.quantile(values, [.1, .5, .9], axis=0).T.reshape(-1, 3, 3)
            diagnostics[view] = {
                name: dict(zip(('outer_norm', 'half_norm', 'cosine'), q.tolist(), strict=True))
                for (name, _, _), q in zip(BUCKETS[1:], quantiles, strict=True)
            }
            print('W24_EDIT_CROSS_PREDICTED', view, len(jobs), flush=True)

    check_controls(parent/FULL, ('direct', 'outer_objective_half'), [0, 1], ids, matrix)
    check_controls(half_training/FULL, ('direct', 'half_primary_half', 'outer_parent_half'), [0, 2, 1], ids, matrix)
    output = root/FULL
    output.mkdir()
    np.savez(output/'predictions.npz', **matrix)
    write(output/'source-predictions.json', dict(status='PREDICTIONS_FROZEN', modes=list(MODES),
          physical_ids=ids, views=list(VIEWS), target_reads=0, source_truth_reads=0,
          waveform=dict(route=FULL, source_diagnostic=True, uses_clean_reference=False,
                        frozen_controls_match=True, edit_by_view=diagnostics)))


def main():
    from experiments.pamf_lora_main.publish import release, release_commit

    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest='action', required=True)
    for phase in ('dispatch', 'predict-source', 'score-source'):
        actions.add_parser(phase).add_argument('runtime', type=Path)
    pub = actions.add_parser('publish')
    for field in ('repo', 'context', 'ssh-config', 'output'):
        pub.add_argument('--'+field, type=Path, required=True)
    args = parser.parse_args()
    if args.action == 'dispatch':
        dispatch(args.runtime, entry=__file__)
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
        rt['external_releases'] = [rt[field] for field in ('source_root', 'classifier_root',
                                                           'w24_release_root', 'training_release_root')]
        rt['external_releases'] += [str(runs/run) for run in (PARENT, HALF_PRIMARY_RUN)]
        release(args.repo, args.output, rt, args.ssh_config)


if __name__ == '__main__':
    main()
