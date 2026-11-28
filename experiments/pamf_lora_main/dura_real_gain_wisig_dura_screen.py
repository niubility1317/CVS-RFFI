"""Fixed DURA v1 prediction on the preregistered WiSig group screen."""

import argparse
import importlib
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys

import numpy as np
import torch

from experiments.cric_dura_comparison.io import read, write
from experiments.pamf_lora_main.dura_real_gain_fit import DEVICE, as_complex, wisig_models
from experiments.pamf_lora_main.dura_real_gain_run import free_gpus
from experiments.pamf_lora_main.dura_real_gain_screen import paired
from experiments.pamf_lora_main import dura_real_gain_wisig_group_robust as candidate
from experiments.pamf_restoration.models import complex_iq, pair


RUN = '20261128-phase1-wisig-dura-v1-group-screen-s2026113033-r01'
PARENT = candidate.RUN
R69_RUN = '20261021-phase1-restoration-R65-four-route-s2026092701-m4-r69'
R69_RELEASE = 'pamf_stage2_inference_20261021_r01'
VIEWS = ('clean', 'practical_low_urban')


def load_dura(project):
    release = project / 'releases' / R69_RELEASE
    if not release.is_dir():
        raise ValueError('Fixed DURA v1 runtime release missing')
    package = importlib.import_module('experiments')
    package.__path__ = [str(release / 'experiments'), *package.__path__]
    engine = importlib.import_module('experiments.pamf_stage2_inference.engine')
    runtime = importlib.import_module('experiments.pamf_stage2_inference.runtime')
    if not Path(engine.__file__).resolve().is_relative_to(release.resolve()):
        raise ValueError('Expected fixed archived DURA v1 engine')
    from types import SimpleNamespace
    args = SimpleNamespace(parent_run=project / 'runs' / R69_RUN)
    model, payload = runtime.load_model(args, DEVICE)
    if payload.get('data_contract') != 'original7/63/30':
        raise ValueError('DURA v1 source contract mismatch')
    return engine.InferenceEngine(model, reader=None, reuse=True, graph=True, native=False)


def selection(project):
    folder = project / 'data_caches/phase1_manysig_reference_20261019/source_cache/V'
    ids = np.load(folder / 'physical_ids.npy', mmap_mode='r', allow_pickle=False)
    rx = np.load(folder / 'rx_i.npy', mmap_mode='r', allow_pickle=False)
    day = np.load(folder / 'day_i.npy', mmap_mode='r', allow_pickle=False)
    raw = np.load(folder / 'iq.npy', mmap_mode='r', allow_pickle=False)
    old = np.load(project / 'runs/20261019-phase1-reception-restoration-selfsup-manysig-m6-r02/cache/validation_full/physical_ids.npy', allow_pickle=False)
    with np.load(project / 'runs' / candidate.PREVIOUS_SCREEN / 'screen.npz', allow_pickle=False) as previous:
        old = np.concatenate((old, previous['ids'].copy(), np.asarray(['5fa0fd382eb8af0fbc4cc08b'])))
    for run in (candidate.URBAN_PARENT, candidate.DOSE_SCREEN, candidate.AMPLITUDE_SCREEN,
                candidate.INTERNAL_SCREEN, candidate.JOINT_SCREEN, candidate.ACTION_PARENT,
                candidate.STRONG_SCREEN, candidate.CONTINUE_SCREEN, candidate.COARSE_SCREEN,
                candidate.CONTRAST_SCREEN, candidate.GATE_SCREEN, candidate.CONSENSUS_SCREEN):
        with np.load(project / 'runs' / run / 'screen.npz', allow_pickle=False) as previous:
            old = np.concatenate((old, previous['ids'].copy()))
    rows = candidate.select_fresh_rows(ids, rx, day, old)
    return ids, rx, day, raw, rows


@torch.no_grad()
def predict(rt):
    from experiments.cric_dura_comparison.bootstrap import numerical_policy
    numerical_policy()
    project, root = Path(rt['project']), Path(rt['run_root'])
    pipeline = load_dura(project)
    source, source_ids, _ = wisig_models(project, rt['frozen_sources'])
    original = importlib.import_module('experiments.pamf_unrolled_restore.wisig_anchored_window_source')
    config = read(project / 'releases/20261125-phase1-wisig-native-fir-source-m4-r01-release/runtime.json')
    reader = original.tri.reader_factory(config).eval().requires_grad_(False)

    def decisions(values):
        y = as_complex(values)
        restored, = pipeline(pair(y))
        return np.asarray(original.classify(reader, complex_iq(restored)).argmax(1).cpu().tolist(), dtype=np.int16)

    if decisions(source[source_ids[:2]]).shape != (2,):
        raise ValueError('Real checkpoint source U no-query smoke failed')
    from leo_practical.channel import normalize_input
    from experiments.pamf_reception_restore.data import shared_pair
    ids, rx, day, raw, rows = selection(project)
    clean = []
    for value in raw[rows]:
        z = normalize_input(value[0] + 1j * value[1])[0]
        clean.append(np.stack((z.real, z.imag)).astype(np.float32))
    clean = np.stack(clean)
    received = []
    for i, index in enumerate(rows):
        _, full, _ = shared_pair(clean[i], 'practical_low_urban', ids[index],
            f'rx{int(rx[index])}:day{int(day[index])}:eq1', 2026101901, 0, 2026101902,
            'restoration/validation/draw0')
        received.append(full)
    outputs = {}
    for view, values in (('clean', clean), ('practical_low_urban', np.stack(received))):
        outputs[view] = np.concatenate([decisions(values[i:i+48]) for i in range(0, len(rows), 48)])
        print('WISIG_DURA_V1_SCREEN', view, len(rows), flush=True)
    candidate_marker = project / 'runs' / PARENT / 'screen-frozen.json'
    if candidate_marker.exists():
        raise ValueError('Candidate predictions froze before DURA predictions')
    with (root / 'screen.npz').open('xb') as stream:
        np.savez(stream, ids=ids[rows], **outputs)
    write(root / 'screen-frozen.json', dict(status='PREDICTIONS_FROZEN', count=len(rows), views=list(VIEWS),
          source_RX=8, source_truth_reads=0, target_reads=0, checkpoint=R69_RUN,
          runtime=R69_RELEASE, graph=True, reuse=True, native=False))
    if candidate_marker.exists():
        raise ValueError('Candidate predictions froze during DURA finalization; preserve invalid partial result')


def score(rt):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Independent CPU scorer required')
    project, root = Path(rt['project']), Path(rt['run_root'])
    frozen = read(root / 'screen-frozen.json')
    if read(root / 'state.json')['status'] != 'PREDICTIONS_FROZEN_AWAITING_CANDIDATE':
        raise ValueError('DURA prediction run did not finalize before candidate')
    candidate_root = project / 'runs' / PARENT
    candidate_frozen = read(candidate_root / 'screen-frozen.json')
    if (frozen['status'], frozen['count'], frozen['source_truth_reads']) != ('PREDICTIONS_FROZEN', 300, 0):
        raise ValueError('DURA predictions must freeze before source truth')
    if (candidate_frozen['status'], candidate_frozen['count'], candidate_frozen['source_truth_reads']) != ('PREDICTIONS_FROZEN', 300, 0):
        raise ValueError('All candidate predictions must freeze before source truth')
    with np.load(root / 'screen.npz', allow_pickle=False) as f, np.load(candidate_root / 'screen.npz', allow_pickle=False) as g:
        ids = f['ids'].copy()
        if not np.array_equal(ids, g['ids']):
            raise ValueError('DURA and candidate physical IDs do not match')
        control = {v: f[v].copy() for v in VIEWS}
        outputs = {v: g[v].copy()[-1] for v in VIEWS}
    folder = project / 'data_caches/phase1_manysig_reference_20261019/source_cache/V'
    all_ids = np.load(folder / 'physical_ids.npy', allow_pickle=False)
    labels = np.load(folder / 'labels.npy', allow_pickle=False)
    lookup = dict(zip(all_ids.tolist(), labels.tolist(), strict=True))
    truth = np.asarray([lookup[x] for x in ids], dtype=np.int64)
    permutation = np.random.default_rng(candidate.SEED + 3).permutation(len(ids))
    comparison = paired(outputs['practical_low_urban'], control['practical_low_urban'], truth, permutation, 2)
    clean_net = int(((outputs['clean'] == truth).astype(np.int16) - (control['clean'] == truth).astype(np.int16)).sum())
    original = read(candidate_root / 'screen-score.json')
    write(root / 'screen-score.json', dict(status='SOURCE_SCREEN_SCORED', truth_last=True,
          count=300, control='DURA v1 fixed Graph-RR', candidate=PARENT,
          correct={v: dict(dura=int((control[v] == truth).sum()), candidate=int((outputs[v] == truth).sum())) for v in VIEWS},
          low_urban_pair=comparison, clean_net=clean_net,
          promote=bool(original['promote'] and comparison['passes'] and clean_net >= -2),
          formal_acceptance_complete=False, exposure='Source V and historical target benchmark previously exposed.'))


def dispatch(runtime):
    rt = read(runtime)
    root = Path(rt['run_root'])
    root.mkdir(exist_ok=False)
    try:
        available = free_gpus()
        if not available:
            raise RuntimeError('No GPU with 8 GiB free')
        gpu = next(value for value in (7, 2, 3, 6, 1, 4, 0, 5) if value in available)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='2')
        with (root / 'predict.log').open('x', encoding='utf-8') as log:
            proc = subprocess.Popen([sys.executable, '-u', '-m', __spec__.name, 'predict', str(runtime)],
                                    env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        write(root / 'state.json', dict(status='PREDICT', pid=proc.pid, gpu=gpu), exclusive=False)
        if proc.wait():
            raise RuntimeError('DURA prediction failed; preserve partial artifacts')
        write(root / 'state.json', dict(status='PREDICTIONS_FROZEN_AWAITING_CANDIDATE', gpu=gpu), exclusive=False)
    except Exception as error:
        write(root / 'state.json', dict(status='FAILED', error=str(error)), exclusive=False)
        raise


def publish(repo, output, context, ssh_config):
    from experiments.pamf_lora_main.publish import release, release_commit
    source = read(context)
    rt = {key: source[key] for key in ('project', 'python', 'frozen_sources')}
    project = PurePosixPath(rt['project'])
    runs, releases = project / 'runs', project / 'releases'
    rt.update(run_id=RUN, runner=__spec__.name, code_commit=release_commit(repo),
              release_root=str(releases / f'{RUN}-release'), run_root=str(runs / RUN),
              external_releases=[str(runs / R69_RUN), str(releases / R69_RELEASE),
                                 str(releases / '20261125-phase1-wisig-native-fir-source-m4-r01-release')])
    release(repo, output, rt, ssh_config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='action', required=True)
    for action in ('dispatch', 'predict', 'score'):
        commands.add_parser(action).add_argument('runtime', type=Path)
    pub = commands.add_parser('publish')
    for field in ('repo', 'output', 'context', 'ssh-config'):
        pub.add_argument('--' + field, type=Path, required=True)
    args = vars(parser.parse_args())
    action = args.pop('action')
    if action == 'publish':
        publish(**args)
    elif action == 'dispatch':
        dispatch(**args)
    else:
        {'predict': predict, 'score': score}[action](read(args['runtime']))
