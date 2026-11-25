"""Audit available Residual IQ and missing W24 aperture inputs on source V."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import os
from pathlib import Path, PurePosixPath
import subprocess

import numpy as np
from scipy.signal import resample_poly
import torch

if __package__ is None:
    from experiments import pamf_lora_main

    pamf_lora_main.__path__.append(str(Path(__file__).resolve().parent))

from experiments.cric_dura_comparison.io import read, tensor, write
from experiments.pamf_lora_main.full_packet_views import native, source_contract
from experiments.pamf_lora_main.packet_physics import CHIRPS, chirp_windows, residual_view
from experiments.pamf_lora_main.runner import activate
from experiments.pamf_lora_main.w24_factor_audit import AUG_RUN, load_classifier, panel, score_source
from experiments.pamf_lora_main.w24_native_full import (
    DATA_RUN, SOURCE, VIEWS, W24_COMMIT, W24_RUN, classify_aperture, native_full_view,
)
from experiments.pamf_lora_main.w24_wide_transfer import received_aperture


RUN = '20261125-diagnostic-lora-w24-residual-aperture-source-s2026112559-r01'
MODES = ('native', 'native_without_edges', 'native_middle_only', 'full_middle_256', 'residual_middle_256')
BATCH = 24


def middle_centers(windows):
    """Resample measured centers and use zeros only for unavailable inputs."""
    if windows.ndim != 4 or windows.shape[1:] != (len(CHIRPS), 2, 256):
        raise ValueError('Expected the six registered 256 kHz LoRa chirp windows')
    observed = windows[:, :, 0] + 1j * windows[:, :, 1]
    core = resample_poly(observed, 3, 2, axis=-1)
    scale = np.sqrt(np.mean(np.abs(core)**2, axis=(1, 2), keepdims=True))
    if not np.isfinite(scale).all() or (scale <= 1e-12).any():
        raise ValueError('Empty or nonfinite measured LoRa centers')
    core = core / scale
    result = np.zeros((len(windows), 8, 2, 432), dtype=np.float32)
    result[:, CHIRPS, :, :384] = np.stack((core.real, core.imag), axis=2)
    return result


def views(scene, item):
    row, rx, native_packet = item
    full = native_full_view(native_packet, scene, rx, row, RUN)
    low = chirp_windows(full)
    residual = low if scene == 'clean' else residual_view(native_packet, scene, rx, row, RUN)
    return full, low, residual


@torch.no_grad()
def predictions(model, packets, full_centers, residual_centers):
    real = received_aperture(packets, 'cpu')
    no_edges = real.clone()
    no_edges[..., 384:] = 0
    six = no_edges.clone()
    six[:, (0, 7)] = 0
    low = tensor(middle_centers(full_centers))
    residual = tensor(middle_centers(residual_centers))
    return np.stack([classify_aperture(model, iq) for iq in (real, no_edges, six, low, residual)])


def predict_source(runtime):
    rt = read(runtime)
    activate(rt)
    root, source = Path(rt['run_root']), Path(rt['source_root'])
    contract = source_contract(source)
    classifier = load_classifier(rt, contract, 'cpu')
    raw = native(source, 'U_s', contract['counts']['U_s'])[0]
    for scene in ('clean', VIEWS[-1]):
        arrays = views(scene, (0, SOURCE[0], raw))
        labels = predictions(classifier, *(x[None] for x in arrays))
        if labels.shape != (len(MODES), 1):
            raise ValueError('Invalid residual aperture source U smoke')
    write(root / 'smoke.json', dict(status='PASS', source_examples=2, target_reads=0, source_truth_reads=0))
    rows, rx, ids = panel(source, contract)
    held = native(source, 'V', contract['counts']['V'])[rows]
    jobs = [(int(row), SOURCE[int(receiver)], packet)
            for row, receiver, packet in zip(rows, rx, held, strict=True)]
    output = {}
    with ProcessPoolExecutor(max_workers=rt['cpu_workers']) as pool:
        for scene in VIEWS:
            stream = pool.map(partial(views, scene), jobs, chunksize=6)
            batches = []
            for offset in range(0, len(rows), BATCH):
                chunk = [next(stream) for _ in range(min(BATCH, len(rows) - offset))]
                arrays = [np.stack(items) for items in zip(*chunk, strict=True)]
                batches.append(predictions(classifier, *arrays))
            output[scene] = np.concatenate(batches, axis=1)
            print('W24_RESIDUAL_APERTURE_PREDICTED', scene, len(rows), flush=True)
    np.savez(root / 'predictions.npz', **output)
    write(root / 'source-predictions.json', dict(status='PREDICTIONS_FROZEN', modes=list(MODES),
          views=list(VIEWS), physical_ids=ids, target_reads=0, source_truth_reads=0, waveform={
              'registered_residual_sample_rate': 256000, 'observed_chirps': list(CHIRPS),
              'native_edge_samples_stored_in_residual': 0,
              'native_complex_points_per_packet': 8192, 'residual_complex_points_per_packet': 1536,
          }))


def dispatch(runtime):
    rt = read(runtime)
    root, reference = Path(rt['run_root']), Path(rt['w24_release_root'])
    root.mkdir()
    state = root / 'state.json'
    try:
        if read(reference / 'submit.json')['code_commit'] != W24_COMMIT:
            raise ValueError('Exact W24 source release required')
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1',
                   PYTHONPATH=str(reference))
        for phase in ('predict-source', 'score-source'):
            script = str(Path(__file__).resolve())
            with (root / f'{phase}.log').open('x', encoding='utf-8') as log:
                job = subprocess.Popen(['nice', '-n', '8', rt['python'], '-u', script, phase, str(runtime)], cwd=reference,
                                       env=env, stdout=log, stderr=subprocess.STDOUT)
                write(state, dict(status='RUNNING', phase=phase, pid=job.pid, device='cpu'), exclusive=False)
                if job.wait():
                    raise RuntimeError(f'{phase} failed')
        write(state, dict(status='SOURCE_SCORED'), exclusive=False)
    except Exception as error:
        write(state, dict(status='FAILED', error=str(error)), exclusive=False)
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
        rt.update(run_id=RUN, code_commit=release_commit(args.repo), runner=__spec__.name,
                  release_root=str(project / 'releases' / f'{RUN}-release'),
                  run_root=str(project / 'runs' / RUN),
                  source_root=str(project / 'releases' / f'{DATA_RUN}-release' / 'data/source'),
                  classifier_root=str(project / 'runs' / AUG_RUN),
                  w24_release_root=str(project / 'releases' / f'{W24_RUN}-release'),
                  cpu_workers=6)
        rt['external_releases'] = [rt[k] for k in ('source_root', 'classifier_root', 'w24_release_root')]
        release(args.repo, args.output, rt, args.ssh_config)
    elif args.action == 'dispatch':
        dispatch(args.runtime)
    elif args.action == 'predict-source':
        predict_source(args.runtime)
    else:
        score_source(args.runtime, modes=MODES)


if __name__ == '__main__':
    main()
