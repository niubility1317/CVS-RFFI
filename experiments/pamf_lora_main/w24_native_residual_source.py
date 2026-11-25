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
from experiments.pamf_lora_main.w24_factor_audit import AUG_RUN, load_classifier, panel, score_source
from experiments.pamf_lora_main.w24_native_full import (
    DATA_RUN, SOURCE, VIEWS, W24_COMMIT, W24_RUN, classify, load_restorers, native_full_view, recover,
)
from experiments.pamf_lora_main.w24_native_residual import ROUTE, native_residual
from experiments.pamf_lora_main.w24_wide_transfer import (
    TRAIN_RUN, aperture_batch, load_restorer, received_aperture, training_runtime,
)


RUN = '20261125-diagnostic-lora-w24-native-residual-source-s2026112559-r01'
ROUTES = ('full_native', ROUTE)
MODES = ('direct', 'prior_complex_response_half', 'wide_dura_half')
BATCH = 24


def paired_iq(scene, item):
    row, receiver, packet = item
    handle = (scene, receiver, row, RUN)
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
def predict_source(runtime):
    rt = read(runtime)
    activate(rt)
    root, source = Path(rt['run_root']), Path(rt['source_root'])
    device = 'cuda:0'
    contract = source_contract(source)
    f = load_classifier(rt, contract, device)
    baseline = load_restorers(contract, root.parent, device)
    archive, parent = training_runtime(rt)
    g, track, threshold, frozen = load_restorer(archive, parent, contract, device)
    evaluate = partial(batch_labels, f, baseline, g, track, threshold, device=device)

    u = np.array(native(source, 'U_s', contract['counts']['U_s'])[:2])
    for scene in ('clean', VIEWS[-1]):
        arrivals = [paired_iq(scene, (i, SOURCE[0], packet)) for i, packet in enumerate(u)]
        for route in zip(*arrivals, strict=True):
            labels, _ = evaluate(np.stack(route))
            if labels.shape != (len(MODES), len(u)):
                raise ValueError('Actual source-only front-end smoke failed')
    rng = np.random.default_rng(2026112559)
    clean = u[:1]
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
    write(root / 'smoke.json', dict(status='PASS', source_packets=len(u), target_reads=0, source_truth_reads=0,
          b1_active_g_median_ms=float(np.median(samples)), b1_active_g_p95_ms=float(np.percentile(samples, 95)),
          parent_steps=frozen['steps'], parameters=frozen['parameters'], threshold=threshold))

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
            counts = {r: np.zeros(len(MODES)-1, dtype=int) for r in ROUTES}
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
        out = root / route
        out.mkdir()
        np.savez(out / 'predictions.npz', **output[route])
        write(out / 'source-predictions.json', dict(status='PREDICTIONS_FROZEN', modes=list(MODES),
              views=list(VIEWS), physical_ids=ids, target_reads=0, source_truth_reads=0,
              waveform=dict(route=route, source_frozen_G=TRAIN_RUN, threshold=threshold,
                            activation=activity[route])))


def compare_source(runtime):
    root = Path(read(runtime)['run_root'])
    for route in ROUTES:
        score_source(runtime, modes=MODES, output=root / route)


def dispatch(runtime):
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
        script = str(Path(__file__).resolve())
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
    args = parser.parse_args()
    if args.action == 'publish':
        ctx = read(args.context)
        rt = {k: ctx[k] for k in ('project', 'python', 'frozen_sources')}
        project = PurePosixPath(rt['project'])
        runs, releases = project / 'runs', project / 'releases'
        rt.update(run_id=RUN, code_commit=release_commit(args.repo), runner=__spec__.name,
                  release_root=str(releases / f'{RUN}-release'), run_root=str(runs / RUN),
                  source_root=str(releases / f'{DATA_RUN}-release' / 'data/source'),
                  classifier_root=str(runs / AUG_RUN), w24_release_root=str(releases / f'{W24_RUN}-release'),
                  training_release_root=str(releases / f'{TRAIN_RUN}-release'), allowed_gpus=[0, 5, 3],
                  cpu_workers=8)
        rt['external_releases'] = [rt[k] for k in ('source_root', 'classifier_root', 'w24_release_root',
                                                  'training_release_root')]
        release(args.repo, args.output, rt, args.ssh_config)
    elif args.action == 'dispatch':
        dispatch(args.runtime)
    elif args.action == 'predict-source':
        predict_source(args.runtime)
    else:
        compare_source(args.runtime)


if __name__ == '__main__':
    main()
