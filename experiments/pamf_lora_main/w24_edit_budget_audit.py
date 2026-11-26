"""Isolate the magnitude and direction of the frozen W24 half readout on source IQ."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from functools import partial
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
from experiments.pamf_lora_main.w24_aperture_noise_audit import (
    CENTER, center_rms, model_iq, native_aperture,
)
from experiments.pamf_lora_main.w24_factor_audit import (
    frozen_source_panel, load_classifier, panel, score_source, trace_components,
)
from experiments.pamf_lora_main.w24_formal import wide_delta
from experiments.pamf_lora_main.w24_native_full import SOURCE, VIEWS, classify_aperture, native_full_view
from experiments.pamf_lora_main.w24_native_residual_source import BATCH, WAVEFORM_NAMESPACE
from experiments.pamf_lora_main.w24_outer_objective_source import RUN as PARENT, load_candidate
from experiments.pamf_lora_main.w24_wide_transfer import load_restorer, received_aperture, training_runtime


RUN = '20261126-diagnostic-lora-w24-edit-budget-source-s2026112559-r02'
FULL = 'full_native'
CAP = .5 * .55
MODES = ('direct', 'outer_half', 'oracle_same_dose', 'oracle_cap', 'oracle_projected_exact', 'oracle_exact')
CONTRASTS = (('oracle_same_dose', 'outer_half'), ('oracle_cap', 'oracle_same_dose'),
             ('oracle_projected_exact', 'oracle_cap'), ('oracle_exact', 'oracle_projected_exact'))
METRICS = ('learned_dose', 'full_noise_dose', 'projected_noise_dose', 'direction_cosine',
           'learned_remaining', 'same_dose_remaining', 'cap_remaining')


def rms(x):
    return x.abs().square().mean(dim=(-2, -1), keepdim=True).sqrt()


def project_noise(noise, phase):
    # The learned DURA edit removes the per-chirp DC term in its own tracked frame.
    frame = noise * phase.conj()
    return (frame - frame.mean(-1, keepdim=True)) * phase


def oracle_edits(received, noise, learned, phase):
    projected = project_noise(noise, phase)
    required = rms(projected).clamp_min(1e-12)
    same = -projected * (rms(learned)/required)
    capped = -projected * (CAP*rms(received)/required).clamp(max=1.)
    return same, capped, -projected, -noise


def diagnostics(received, noise, learned, phase, same, capped):
    projected = project_noise(noise, phase)
    observed = rms(received).clamp_min(1e-12)
    total_noise = rms(noise)
    dot = (learned.conj() * -projected).real.mean(dim=(-2, -1), keepdim=True)
    cosine = dot / (rms(learned)*rms(projected)).clamp_min(1e-12)
    remaining = tuple(rms(noise+edit) / total_noise.clamp_min(1e-12) for edit in (learned, same, capped))
    rest = (rms(learned)/observed, total_noise/observed, rms(projected)/observed, cosine, *remaining)
    return np.asarray(torch.cat(rest, dim=-1).squeeze(-2).tolist())


@torch.no_grad()
def receive_center(g, tracker, received, signal, device):
    original = received_aperture(received, device)
    learned = .5 * wide_delta(g, tracker, original)
    observed, noiseless = native_aperture(received), native_aperture(signal)
    noise = complex_iq(tensor(model_iq(observed-noiseless, center_rms(observed)), device)[..., :CENTER])
    _, phase, _, _ = coordinates(tracker, original[..., :CENTER])
    return original, complex_iq(learned), noise, phase


@torch.no_grad()
def render(f, g, tracker, received, signal, device):
    original, real_edit, noise, phase = receive_center(g, tracker, received, signal, device)
    base = complex_iq(original[..., :CENTER])
    same, capped, projected_exact, exact = oracle_edits(base, noise, real_edit, phase)
    info = diagnostics(base, noise, real_edit, phase, same, capped)
    if not np.isfinite(info).all() or (info[:, 0] > CAP + 1e-5).any():
        raise ValueError('The actual W24 half edit violates its registered RMS limit')
    results = [classify_aperture(f, original)]
    for edit in (real_edit, same, capped, projected_exact, exact):
        candidate = original.clone()
        candidate[..., :CENTER] += real_iq(edit)
        results.append(classify_aperture(f, candidate))
    return np.stack(results), info


def summarize(values, is_clean):
    n = len(values)
    quantiles = (0.1, 0.5, 0.9)
    data = values[:, :3] if is_clean else values
    return dict(count=n, q10_q50_q90={k: q.tolist() for k,q in
                zip(METRICS[:data.shape[-1]], np.quantile(data, quantiles, axis=0).T, strict=True)},
                learned_near_cap=int((values[:, 0] >= .95*CAP).sum()),
                projected_over_cap=int((values[:, 2] > CAP).sum()),
                same_dose_defined=int((values[:, 2] > 1e-12).sum()),
                learning_dose_over_projected=int(((values[:, 0] > values[:, 2]) & (values[:, 2] > 1e-12)).sum()),
                cap_insufficient_for_90percent=int((.9*values[:, 1] > CAP).sum()))


def full_components(scene, item, *, verify=False):
    row, rx, packet = item
    if scene == 'clean':
        return packet, packet
    args = (packet, scene, rx, row, WAVEFORM_NAMESPACE)
    received, signal, _ = trace_components(*args)
    if verify and not np.array_equal(received, native_full_view(*args)):
        raise ValueError('Trace replay changed the registered Full received IQ')
    return received, signal.astype(np.complex64)


def predict(rt, *, modes=MODES, render_fn=render, summary_fn=summarize,
            measurement_shape=(len(METRICS),), clean_reference=0, waveform_fields=None):
    activate(rt)
    root, source, device = Path(rt['run_root']), Path(rt['source_root']), 'cuda:0'
    contract = source_contract(source)
    f = load_classifier(rt, contract, device)
    # Import the architecture and tracker under the pinned training archive namespace.
    _, tracker, _, _ = load_restorer(*training_runtime(rt), contract, device)
    parent = root.parent / PARENT
    g, frozen = load_candidate(parent, contract, device)
    u = native(source, 'U_s', contract['counts']['U_s'])[0]
    for scene in ('clean', VIEWS[-1]):
        received, signal = full_components(scene, (0, SOURCE[0], u), verify=True)
        rows, measurements = render_fn(f, g, tracker, received[None], signal[None], device)
        if rows.shape != (len(modes), 1) or measurements.shape != (1, *measurement_shape):
            raise ValueError('Real source-only W24 smoke produced malformed results')
        if scene == 'clean' and not np.all(rows[2:] == rows[clean_reference]):
            raise ValueError('Clean oracle rows differ from the registered control')
    write(root/'smoke.json', dict(status='PASS', source_u_count=1, candidate_parent=PARENT,
          parameters=frozen['parameters'], target_reads=0, source_truth_reads=0))

    rows, rx, ids = panel(source, contract)
    raw = native(source, 'V', contract['counts']['V'])[rows]
    jobs = [(int(row), SOURCE[int(receiver)], packet) for row, receiver, packet in zip(rows, rx, raw, strict=True)]
    output, metrics = {}, {}
    workers = rt['cpu_workers']
    with (ProcessPoolExecutor(max_workers=workers) if workers > 1 else nullcontext()) as pool:
        for scene in VIEWS:
            receive = partial(full_components, scene)
            stream = pool.map(receive, jobs, chunksize=6) if pool is not None else map(receive, jobs)
            batches = []
            for offset in range(0, len(jobs), BATCH):
                chunk = [next(stream) for _ in range(min(BATCH, len(jobs)-offset))]
                received, signal = (np.stack(a) for a in zip(*chunk, strict=True))
                batches.append(render_fn(f, g, tracker, received, signal, device))
            decisions, info = zip(*batches, strict=True)
            output[scene] = np.concatenate(decisions, axis=1)
            metrics[scene] = summary_fn(np.concatenate(info), scene == 'clean')
            print('W24_EDIT_BUDGET_PREDICTED', scene, len(jobs), flush=True)

    previous = parent/FULL
    proof = read(previous/'source-predictions.json')
    if (proof['status'], proof['physical_ids'], proof['views'], proof['target_reads'], proof['source_truth_reads']) != (
            'PREDICTIONS_FROZEN', ids, list(VIEWS), 0, 0):
        raise ValueError('Mismatched prior blind Full source panel')
    idx = [proof['modes'].index(m) for m in ('direct', 'outer_objective_half')]
    with np.load(previous/'predictions.npz', allow_pickle=False) as base:
        if any(not np.array_equal(base[v][idx], output[v][:2]) for v in VIEWS):
            raise ValueError('Actual direct or frozen candidate decisions differ for Full')
    out = root/FULL
    out.mkdir()
    np.savez(out/'predictions.npz', **output)
    write(out/'source-predictions.json', dict(status='PREDICTIONS_FROZEN', modes=list(modes), views=list(VIEWS),
          physical_ids=ids, target_reads=0, source_truth_reads=0,
          waveform=dict(route=FULL, oracle_only=True, learned_parent=PARENT, projected_half_cap=CAP,
                        direct_and_learned_parity=True, dose_by_scene=metrics, **(waveform_fields or {}))))


def score(runtime, *, modes=MODES, contrasts=CONTRASTS):
    rt = read(runtime)
    root, source = Path(rt['run_root']), Path(rt['source_root'])
    frozen_source_panel(source, root/FULL, modes)
    score_source(runtime, modes=modes, output=root/FULL, contrasts=contrasts)


def main():
    from experiments.pamf_lora_main.publish import release, release_commit
    from experiments.pamf_lora_main.w24_native_residual_source import AUG_RUN, DATA_RUN, TRAIN_RUN, W24_RUN, dispatch

    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest='action', required=True)
    for phase in ('dispatch', 'predict-source', 'score-source'):
        actions.add_parser(phase).add_argument('runtime', type=Path)
    pub = actions.add_parser('publish')
    variants = pub.add_mutually_exclusive_group()
    variants.add_argument('--band-direction', action='store_true')
    variants.add_argument('--band-polar', action='store_true')
    variants.add_argument('--noise-components', action='store_true')
    for name in ('repo', 'context', 'ssh-config', 'output'):
        pub.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    if args.action == 'dispatch':
        dispatch(args.runtime, entry=__file__)
    elif args.action == 'predict-source':
        rt = read(args.runtime)
        if rt.get('band_direction'):
            from experiments.pamf_lora_main.w24_band_direction import predict_bands

            predict_bands(rt)
        elif rt.get('band_polar'):
            from experiments.pamf_lora_main.w24_band_polar import predict_bands

            predict_bands(rt)
        elif rt.get('noise_components'):
            from experiments.pamf_lora_main.w24_noise_components import predict_components

            predict_components(rt)
        else:
            predict(rt)
    elif args.action == 'score-source':
        rt = read(args.runtime)
        if rt.get('band_direction'):
            from experiments.pamf_lora_main.w24_band_direction import CONTRASTS as contrasts, MODES as modes

            score(args.runtime, modes=modes, contrasts=contrasts)
        elif rt.get('band_polar'):
            from experiments.pamf_lora_main.w24_band_polar import CONTRASTS, MODES, verify_reference

            verify_reference(rt)
            score(args.runtime, modes=MODES, contrasts=CONTRASTS)
        elif rt.get('noise_components'):
            from experiments.pamf_lora_main.w24_noise_components import score_components

            score_components(args.runtime)
        else:
            score(args.runtime)
    else:
        ctx = read(args.context)
        rt = {key:ctx[key] for key in ('project', 'python', 'frozen_sources')}
        project = PurePosixPath(rt['project'])
        runs, releases = project/'runs', project/'releases'
        run = RUN
        if args.band_direction:
            from experiments.pamf_lora_main.w24_band_direction import RUN as band_run

            run = band_run
            rt['band_direction'] = True
        elif args.band_polar:
            from experiments.pamf_lora_main.w24_band_polar import DIRECTION_RUN, RUN as polar_run

            run = polar_run
            rt['band_polar'] = True
        elif args.noise_components:
            from experiments.pamf_lora_main.w24_noise_components import RUN as components_run

            run = components_run
            rt['noise_components'] = True
        rt.update(run_id=run, runner=__spec__.name, code_commit=release_commit(args.repo),
                  release_root=str(releases/f'{run}-release'), run_root=str(runs/run),
                  source_root=str(releases/f'{DATA_RUN}-release'/'data/source'), classifier_root=str(runs/AUG_RUN),
                  w24_release_root=str(releases/f'{W24_RUN}-release'),
                  training_release_root=str(releases/f'{TRAIN_RUN}-release'),
                  allowed_gpus=[0, 3, 5, 6] if args.band_direction or args.band_polar else [5, 6, 3, 0], cpu_workers=8)
        if args.noise_components:
            rt.update(allowed_gpus=[0, 3, 2], cpu_workers=1)
        rt['external_releases'] = [rt[k] for k in ('source_root','classifier_root','w24_release_root','training_release_root')]
        rt['external_releases'].append(str(runs/PARENT))
        if args.band_polar:
            rt['external_releases'].append(str(runs / DIRECTION_RUN))
        if args.noise_components:
            rt['external_releases'].append(str(runs / RUN))
        release(args.repo, args.output, rt, args.ssh_config)


if __name__=='__main__':
    main()
