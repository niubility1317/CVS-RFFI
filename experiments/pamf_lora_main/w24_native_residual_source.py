"""Compare frozen W24 restoration on source Full and native Residual IQ."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import os
from pathlib import Path, PurePosixPath
import subprocess
from time import monotonic, perf_counter, sleep

import numpy as np
import torch

if __package__ is None:
    from experiments import pamf_lora_main

    pamf_lora_main.__path__.append(str(Path(__file__).resolve().parent))

from experiments.cric_dura_comparison.io import read, write
from experiments.pamf_lora_main.full_packet_views import native, source_contract
from experiments.pamf_lora_main.runner import activate, available
from experiments.pamf_lora_main.w24_factor_audit import AUG_RUN, frozen_source_panel, load_classifier, panel, score_source
from experiments.pamf_lora_main.w24_formal import wide_delta
from experiments.pamf_lora_main.w24_native_full import (
    DATA_RUN, SOURCE, VIEWS, W24_COMMIT, W24_RUN, classify, classify_aperture,
    load_restorers, native_full_view, recover,
)
from experiments.pamf_lora_main.w24_native_residual import ROUTE, native_residual
from experiments.pamf_lora_main.w24_wide_transfer import (
    TRAIN_RUN, aperture_batch, load_restorer, received_aperture, training_runtime,
)


RUN = '20261125-diagnostic-lora-w24-native-residual-source-s2026112559-r02'
SWEEP_RUN = '20261125-diagnostic-lora-w24-readout-sweep-source-s2026112559-r01'
CHIRP_RUN = '20261125-diagnostic-lora-w24-chirp-allocation-source-s2026112559-r01'
WAVEFORM_NAMESPACE = '20261125-diagnostic-lora-w24-native-residual-source-s2026112559-r01'
ROUTES = ('full_native', ROUTE)
MODES = ('direct', 'prior_complex_response_half', 'wide_dura_half')
SWEEP_AMPLITUDES = (.5, .75, 1.)
SWEEP_MODES = ('direct', *(f'wide_ungated_{int(100*x):03d}' for x in SWEEP_AMPLITUDES))
CHIRP_MODES = ('direct', 'all_half', 'outer_half', 'inner_half', 'prefer_outer', 'prefer_inner')
CHIRP_CONTRASTS = (('outer_half', 'inner_half'), ('prefer_outer', 'all_half'),
                   ('prefer_inner', 'all_half'), ('prefer_outer', 'prefer_inner'))
BATCH = 24


def paired_iq(scene, item):
    row, receiver, packet = item
    handle = (scene, receiver, row, WAVEFORM_NAMESPACE)
    return native_full_view(packet, *handle), native_residual(packet, *handle)


@torch.no_grad()
def batch_labels(f, baseline, g, tracker, threshold, packets, device):
    paired, n_wide = aperture_batch(f, g, tracker, threshold, packets, device)
    _, _, restored = recover(packets, baseline, device)
    active = np.any(restored != packets, axis=1)
    control = paired[0].copy()
    if active.any():
        control[active] = classify(f, restored[active], device)
    return np.stack((paired[0], control, paired[1])), [int(active.sum()), n_wide]


@torch.no_grad()
def sweep_labels(f, g, tracker, packets, device):
    original = received_aperture(packets, device)
    delta = wide_delta(g, tracker, original)
    output = [classify_aperture(f, original)]
    for amplitude in SWEEP_AMPLITUDES:
        restored = original.clone()
        restored[..., :384] += amplitude * delta
        output.append(classify_aperture(f, restored))
    return np.stack(output)


def chirp_edits(delta):
    indices = torch.arange(8, device=delta.device)
    outer = ((indices == 0) | (indices == 7))[None, :, None, None]
    half = .5 * delta
    energy = half.square().sum((1, 2, 3), keepdim=True)
    preferred = []
    for mask in (outer, ~outer):
        weighted = half * (.75 + .5 * mask)
        weighted_energy = weighted.square().sum((1, 2, 3), keepdim=True)
        scale = (energy / weighted_energy.clamp_min(torch.finfo(weighted_energy.dtype).tiny)).sqrt()
        preferred.append(weighted * scale)
    return half, half * outer, half * ~outer, *preferred


@torch.no_grad()
def chirp_labels(f, g, tracker, packets, device, *, verify=False):
    original = received_aperture(packets, device)
    edits = chirp_edits(wide_delta(g, tracker, original))
    if verify:
        half, outer, inner, *preferred = edits
        torch.testing.assert_close(outer + inner, half, rtol=0, atol=0)
        for biased in preferred:
            torch.testing.assert_close(biased.square().sum((1, 2, 3)), half.square().sum((1, 2, 3)))
    output = [classify_aperture(f, original)]
    for edit in edits:
        corrected = original.clone()
        corrected[..., :384] += edit
        output.append(classify_aperture(f, corrected))
    return np.stack(output), [len(packets)] * len(edits)


def mode_list(rt):
    if rt.get('outer_objective'):
        from experiments.pamf_lora_main.w24_outer_objective_source import MODES as OUTER_MODES

        return OUTER_MODES
    if rt.get('chirp_allocation'):
        return CHIRP_MODES
    return SWEEP_MODES if rt.get('readout_sweep') else MODES


@torch.no_grad()
def predict_source(runtime):
    rt = read(runtime)
    activate(rt)
    root, source = Path(rt['run_root']), Path(rt['source_root'])
    device = 'cuda:0'
    contract = source_contract(source)
    f = load_classifier(rt, contract, device)
    # Let the archived G resolve its absolute dependencies before the older
    # reference stack caches a namesake package with different submodules.
    archive, parent = training_runtime(rt)
    g, track, threshold, frozen = load_restorer(archive, parent, contract, device)
    learning_outer = rt.get('outer_objective', False)
    allocating = rt.get('chirp_allocation', False)
    sweeping = rt.get('readout_sweep', False)
    modes = mode_list(rt)
    if learning_outer:
        from experiments.pamf_lora_main.w24_outer_objective_source import load_candidate

        candidate, new_frozen = load_candidate(root, contract, device)

        def evaluate(packets):
            original = received_aperture(packets, device)
            decisions = [classify_aperture(f, original)]
            for model in (g, candidate):
                restored = original.clone()
                restored[..., :384] += .5 * wide_delta(model, track, original)
                decisions.append(classify_aperture(f, restored))
            return np.stack(decisions), [len(packets)] * 2
    elif allocating:
        evaluate = partial(chirp_labels, f, g, track, device=device)
    elif sweeping:
        def evaluate(packets):
            return sweep_labels(f, g, track, packets, device), [len(packets)] * len(SWEEP_AMPLITUDES)
    else:
        baseline = load_restorers(contract, root.parent, device)
        evaluate = partial(batch_labels, f, baseline, g, track, threshold, device=device)

    u = np.array(native(source, 'U_s', contract['counts']['U_s'])[:2])
    for scene in ('clean', VIEWS[-1]):
        arrivals = [paired_iq(scene, (i, SOURCE[0], packet)) for i, packet in enumerate(u)]
        for route in zip(*arrivals, strict=True):
            packets = np.stack(route)
            if allocating:
                labels, _ = chirp_labels(f, g, track, packets, device, verify=True)
            else:
                labels, _ = evaluate(packets)
            if labels.shape != (len(modes), len(u)):
                raise ValueError('Actual source-only front-end smoke failed')
    smoke = dict(status='PASS', source_packets=len(u), target_reads=0, source_truth_reads=0,
                 parent_steps=frozen['steps'], parameters=frozen['parameters'])
    if learning_outer:
        observed = received_aperture(packets[:1], device)
        b1 = {name: ungated_latency(model, track, observed)
              for name, model in zip(modes[1:], (g, candidate), strict=True)}
        smoke.update(candidate_steps=new_frozen['steps'], candidate_parameters=new_frozen['parameters'], ungated=True,
                     b1_ungated_g_only=b1)
    elif allocating:
        smoke['equal_edit_energy'] = True
    elif sweeping:
        smoke['readout_amplitudes'] = list(SWEEP_AMPLITUDES)
    else:
        smoke.update(latency(g, track, threshold, u[:1], device), threshold=threshold)
    write(root / 'smoke.json', smoke)

    rows, rx, ids = panel(source, contract)
    held = native(source, 'V', contract['counts']['V'])[rows]
    jobs = [(int(row), SOURCE[int(receiver)], packet)
            for row, receiver, packet in zip(rows, rx, held, strict=True)]
    output = {r: {} for r in ROUTES}
    activity = {r: {} for r in ROUTES}
    with ProcessPoolExecutor(max_workers=rt['cpu_workers']) as pool:
        for scene in VIEWS:
            stream = pool.map(partial(paired_iq, scene), jobs, chunksize=6)
            batches = {r: [] for r in ROUTES}
            counts = {r: np.zeros(len(modes)-1, dtype=int) for r in ROUTES}
            for offset in range(0, len(rows), BATCH):
                chunk = [next(stream) for _ in range(min(BATCH, len(rows)-offset))]
                for route, packets in zip(ROUTES, zip(*chunk, strict=True), strict=True):
                    pred, active = evaluate(np.stack(packets))
                    batches[route].append(pred)
                    counts[route] += active
            for route in ROUTES:
                output[route][scene] = np.concatenate(batches[route], axis=1)
                activity[route][scene] = counts[route].tolist()
            if scene == 'clean' and not np.array_equal(*(output[r][scene] for r in ROUTES)):
                raise ValueError('Same clean packets must produce identical decisions on both routes')
            print('W24_NATIVE_RESIDUAL_PREDICTED', scene, len(rows), flush=True)
    for route in ROUTES:
        if learning_outer or allocating or sweeping:
            previous = root.parent / (SWEEP_RUN if allocating or learning_outer else RUN) / route
            mark = read(previous/'source-predictions.json')
            if (mark['status'], mark['physical_ids'], mark['target_reads']) != ('PREDICTIONS_FROZEN', ids, 0):
                raise ValueError('Expected the prior frozen two-route source panel')
            names = ('direct', 'wide_ungated_050') if allocating or learning_outer else ('direct',)
            indices = [mark['modes'].index(name) for name in names]
            with np.load(previous/'predictions.npz', allow_pickle=False) as archive:
                if any(not np.array_equal(archive[v][indices], output[route][v][:len(indices)]) for v in VIEWS):
                    raise ValueError('The registered source readout comparators failed to reproduce')
        out = root / route
        out.mkdir()
        np.savez(out / 'predictions.npz', **output[route])
        if learning_outer:
            policy = dict(ungated=True, amplitude=.5, candidate_run=rt['run_id'], candidate_steps=new_frozen['steps'])
        elif allocating:
            policy = dict(ungated=True, fixed_position_masks=True, preferred_energy='all_half',
                          preferred_weights=[.75, 1.25])
        elif sweeping:
            policy = dict(ungated=True, amplitudes=list(SWEEP_AMPLITUDES))
        else:
            policy = dict(threshold=threshold)
        write(out / 'source-predictions.json', dict(status='PREDICTIONS_FROZEN', modes=list(modes),
              views=list(VIEWS), physical_ids=ids, target_reads=0, source_truth_reads=0,
              waveform=dict(route=route, source_frozen_G=TRAIN_RUN, activation=activity[route], **policy)))


def latency(g, track, threshold, clean, device):
    rng = np.random.default_rng(2026112559)
    noise = rng.standard_normal(clean.shape) + 1j*rng.standard_normal(clean.shape)
    x = received_aperture((clean + .75*noise*np.abs(clean).max()).astype(np.complex64), device)
    _, active = g.restore(track, x, threshold)
    if not active.all():
        raise ValueError('Expected active source-IQ B1 latency path')
    for _ in range(8):
        g.restore(track, x, threshold)
    torch.cuda.synchronize()
    samples = []
    for _ in range(30):
        start = perf_counter()
        g.restore(track, x, threshold)
        torch.cuda.synchronize()
        samples.append(1000*(perf_counter()-start))
    return dict(b1_active_g_median_ms=float(np.median(samples)), b1_active_g_p95_ms=float(np.percentile(samples, 95)))


def ungated_latency(g, tracker, observed):
    for _ in range(6):
        wide_delta(g, tracker, observed)
    torch.cuda.synchronize()
    samples = []
    for _ in range(24):
        start = perf_counter()
        wide_delta(g, tracker, observed)
        torch.cuda.synchronize()
        samples.append(1000 * (perf_counter() - start))
    return dict(median_ms=float(np.median(samples)), p95_ms=float(np.percentile(samples, 95)))


def compare_source(runtime):
    rt = read(runtime)
    root = Path(rt['run_root'])
    modes = mode_list(rt)
    if rt.get('outer_objective'):
        from experiments.pamf_lora_main.w24_outer_objective_source import CONTRASTS as OUTER_CONTRASTS

        contrasts = OUTER_CONTRASTS
    else:
        contrasts = CHIRP_CONTRASTS if rt.get('chirp_allocation') else ()
    for route in ROUTES:
        frozen_source_panel(Path(rt['source_root']), root / route, modes)
    for route in ROUTES:
        score_source(runtime, modes=modes, output=root / route, contrasts=contrasts)


def dispatch(runtime, *, entry=__file__):
    rt = read(runtime)
    root, reference = Path(rt['run_root']), Path(rt['w24_release_root'])
    root.mkdir()
    state = root / 'state.json'

    def status(**values):
        write(state, values, exclusive=False)

    try:
        if read(reference / 'submit.json')['code_commit'] != W24_COMMIT:
            raise ValueError('Exact W24 source release required')
        status(status='WAIT_GPU')
        deadline = monotonic() + 1800
        allowed = rt['allowed_gpus']
        while not (cards := available(allowed, max_compute_processes=8)):
            if monotonic() >= deadline:
                raise ValueError('GPU wait expired')
            sleep(20)
        gpu = next(i for i in allowed if i in cards)
        env = dict(os.environ, OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1', PYTHONPATH=str(reference))
        script = str(Path(entry).resolve())
        for phase in ('predict-source', 'score-source'):
            env['CUDA_VISIBLE_DEVICES'] = str(gpu) if phase == 'predict-source' else ''
            with (root / f'{phase}.log').open('x', encoding='utf-8') as log:
                job = subprocess.Popen([rt['python'], '-u', script, phase, str(runtime)], cwd=reference,
                                       env=env, stdout=log, stderr=subprocess.STDOUT)
                status(status='RUNNING', phase=phase, pid=job.pid, gpu=gpu if phase=='predict-source' else None)
                if job.wait():
                    raise RuntimeError(f'{phase} failed')
        status(status='SOURCE_SCORED')
    except Exception as error:
        status(status='FAILED', error=str(error))
        raise


def main():
    from experiments.pamf_lora_main.publish import release, release_commit

    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='action', required=True)
    for phase in ('dispatch', 'predict-source', 'score-source'):
        sub.add_parser(phase).add_argument('runtime', type=Path)
    publish = sub.add_parser('publish')
    for key in ('repo', 'context', 'ssh-config', 'output'):
        publish.add_argument('--' + key, type=Path, required=True)
    diagnostics = publish.add_mutually_exclusive_group()
    diagnostics.add_argument('--readout-sweep', action='store_true')
    diagnostics.add_argument('--chirp-allocation', action='store_true')
    args = parser.parse_args()
    if args.action == 'publish':
        ctx = read(args.context)
        rt = {k: ctx[k] for k in ('project', 'python', 'frozen_sources')}
        project = PurePosixPath(rt['project'])
        runs, releases = project / 'runs', project / 'releases'
        run = CHIRP_RUN if args.chirp_allocation else SWEEP_RUN if args.readout_sweep else RUN
        if args.chirp_allocation:
            allowed_gpus = [6, 3, 2, 7, 0, 5, 4, 1]
        elif args.readout_sweep:
            allowed_gpus = [6, 0, 5, 3, 2, 7, 4, 1]
        else:
            allowed_gpus = [0, 5, 3]
        rt.update(run_id=run, code_commit=release_commit(args.repo), runner=__spec__.name,
                  release_root=str(releases / f'{run}-release'), run_root=str(runs / run),
                  readout_sweep=args.readout_sweep, chirp_allocation=args.chirp_allocation,
                  source_root=str(releases / f'{DATA_RUN}-release' / 'data/source'),
                  classifier_root=str(runs / AUG_RUN), w24_release_root=str(releases / f'{W24_RUN}-release'),
                  training_release_root=str(releases / f'{TRAIN_RUN}-release'),
                  allowed_gpus=allowed_gpus, cpu_workers=8)
        rt['external_releases'] = [rt[k] for k in ('source_root', 'classifier_root', 'w24_release_root',
                                                  'training_release_root')]
        if args.chirp_allocation:
            rt['external_releases'].append(str(runs/SWEEP_RUN))
        elif args.readout_sweep:
            rt['external_releases'].append(str(runs/RUN))
        release(args.repo, args.output, rt, args.ssh_config)
    elif args.action == 'dispatch':
        dispatch(args.runtime)
    elif args.action == 'predict-source':
        predict_source(args.runtime)
    else:
        compare_source(args.runtime)


if __name__ == '__main__':
    main()
