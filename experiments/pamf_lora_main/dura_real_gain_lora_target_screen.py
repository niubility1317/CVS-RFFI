"""Frozen LoRa R07 probe fusion: preregistered target receiver screen."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path, PurePosixPath
import os
import subprocess
import sys
import zipfile

import numpy as np
import torch

from experiments.cric_dura_comparison.io import read, write
from experiments.pamf_lora_main import dura_real_gain_lora_probe_action as base
from experiments.pamf_lora_main.dura_real_gain_fit import DEVICE, LORA_ACTION, as_complex
from experiments.pamf_lora_main.dura_real_gain_amplitude import R17AmplitudeGain
from experiments.pamf_lora_main.dura_real_gain_screen import packet_scale, paired
from experiments.pamf_lora_main.dura_real_gain_unroll import fused_anchor_unroll
from experiments.pamf_lora_main.dura_real_gain_run import free_gpus
from experiments.pamf_lora_main.data import TARGET, label_ids, packet_indices, recording

RUN = '20261128-phase1-lora-r07-probe-target-s2026113145-r01'
PARENT = base.RUN
RX, SEED, COUNT = 'b200_2', 2026113145, 300
VIEWS = ('clean', 'practical_low_urban')
MODES = ('direct', 'lora_native_r17', 'real_gain', 'adapted_r69', 'joint_fusion')


@torch.no_grad()
def predict(rt):
    joint, native, old, contract, reader, _, baseline, r17 = base.boot(rt)
    from experiments.pamf_lora_main.w24_residual_pilot import r69_baseline, restore_r69
    from experiments.pamf_lora_main.w24_native_full import channel_job
    project, root, source = Path(rt['project']), Path(rt['run_root']), Path(old['source_root'])
    parent = project / 'runs' / PARENT
    checkpoint = torch.load(parent / 'gain.pt', map_location=DEVICE, weights_only=True)
    proof = read(parent / 'fit.json')
    if (checkpoint['proof'] != proof or (proof['status'], proof['seed'], proof['updates']) !=
            ('SOURCE_FROZEN', 2026113113, 1500) or not proof['F_frozen'] or
            any(proof[k] for k in ('U_truth_reads', 'V_reads', 'target_reads'))):
        raise ValueError('Expected lawful source-only LoRa probe checkpoint')
    gain = R17AmplitudeGain(1024).to(DEVICE).eval().requires_grad_(False)
    gain.load_state_dict(checkpoint['weights'], strict=True)
    candidate = native.dura_model(project / 'runs' / LORA_ACTION, contract, DEVICE).eval().requires_grad_(False)
    candidate.estimator.load_state_dict(checkpoint['estimator'], strict=True)
    r69 = r69_baseline(dict(rt, source_root=str(source)), contract, DEVICE)

    def classify(z):
        cpu = np.asarray(z.contiguous().view(torch.float32).tolist(), dtype=np.float32).view(np.complex64)
        return joint.classify_aperture(reader, joint.received_aperture(cpu, DEVICE))

    def decisions(raw):
        y = as_complex(raw)
        scale = packet_scale(y)
        normalized = y / scale
        p, _ = baseline(normalized)
        restored, _ = restore_r69(raw, *r69, DEVICE)
        values = (y, r17(y), baseline(y)[0], as_complex(restored),
                  fused_anchor_unroll(candidate, gain, normalized, p) * scale)
        return np.stack([classify(z) for z in values])

    if decisions(joint.source_role(source, contract, 'U_s')[0][:2]).shape != (len(MODES), 2):
        raise ValueError('Real-checkpoint source U no-query smoke failed')
    target = Path(rt['native_target_root'])
    joint.target_contract(target)
    indices = np.sort(np.random.default_rng(SEED + 2).choice(2000, COUNT, replace=False))
    write(root / 'panel.json', dict(status='FIXED_BEFORE_PREDICTION', receiver=RX,
          count=COUNT, seed=SEED+2, blind_order_indices=indices.tolist(), historic_formal_exposure=True))
    raw = joint.native(target, RX, 2000)
    matrix = {}
    with ProcessPoolExecutor(max_workers=6) as pool:
        for view in VIEWS:
            values = raw[indices] if view == 'clean' else np.stack(list(pool.map(
                partial(channel_job, view, RX), ((int(i), raw[i]) for i in indices), chunksize=8)))
            matrix[view] = np.concatenate([decisions(values[i:i+8]) for i in range(0, COUNT, 8)], axis=1)
            print('LORA_TARGET_SCREEN', RX, view, COUNT, flush=True)
    with (root / 'screen.npz').open('xb') as stream:
        np.savez(stream, indices=indices, **matrix)
    write(root / 'screen-frozen.json', dict(status='PREDICTIONS_FROZEN', modes=list(MODES),
          views=list(VIEWS), count=COUNT, target_RX=RX, target_truth_read=False,
          observation='registered 8192-point Full', historic_formal_exposure=True))


def score_local(predictions, dataset, output):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Independent CPU truth-side scorer required')
    root = Path(predictions)
    frozen = read(root / 'screen-frozen.json')
    if (frozen['status'], frozen['modes'], frozen['views'], frozen['count'], frozen['target_truth_read']) != (
            'PREDICTIONS_FROZEN', list(MODES), list(VIEWS), COUNT, False):
        raise ValueError('All target rows and views must freeze before truth')
    panel = read(root / 'panel.json')
    with np.load(root / 'screen.npz', allow_pickle=False) as data:
        indices = data['indices'].copy()
        outputs = {view: data[view].copy() for view in VIEWS}
    if (indices.tolist() != panel['blind_order_indices'] or len(np.unique(indices)) != COUNT or
            any(value.shape != (len(MODES), COUNT) for value in outputs.values())):
        raise ValueError('Incomplete or mismatched target predictions')
    with zipfile.ZipFile(Path(dataset) / 'multiple_receiver_test.zip') as archive:
        with recording(archive, f'{RX}_test.h5', Path(output).parent) as capture:
            labels = label_ids(capture, 2000)[packet_indices(2000, TARGET.index(RX))][indices]
    permutation = np.random.default_rng(SEED + 3).permutation(COUNT)
    pairs = {name: paired(outputs[VIEWS[-1]][-1], outputs[VIEWS[-1]][i], labels, permutation, 2)
             for i, name in enumerate(MODES[:-1])}
    clean_net = {name: int(((outputs['clean'][-1] == labels).astype(np.int16) -
                           (outputs['clean'][i] == labels).astype(np.int16)).sum())
                 for i, name in enumerate(MODES[:-1])}
    promote = all(value['passes'] for value in pairs.values()) and min(clean_net.values()) >= -2
    write(Path(output), dict(status='TARGET_SCREEN_SCORED', truth_last=True, count=COUNT,
          modes=list(MODES), correct={v: (x == labels).sum(1).tolist() for v, x in outputs.items()},
          low_urban_pairs=pairs, clean_net=clean_net, promote=promote, formal_acceptance_complete=False,
          r69_identity='LoRa source-trained R69 architecture with historical gate and half-dose; not WiSig DURA v1',
          exposure='Historic full target benchmark exposed; exclude this panel from held-out subset analysis.'))


def dispatch(runtime):
    rt = read(runtime)
    root = Path(rt['run_root'])
    root.mkdir(exist_ok=False)
    try:
        available = free_gpus()
        gpu = next(g for g in (5, 7, 2, 3, 6, 1, 4, 0) if g in available)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1')
        with (root / 'predict.log').open('x', encoding='utf-8') as log:
            proc = subprocess.Popen([sys.executable, '-u', '-m', __spec__.name, 'predict', str(runtime)],
                                    env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        write(root / 'state.json', dict(status='PREDICT', pid=proc.pid, gpu=gpu), exclusive=False)
        if proc.wait():
            raise RuntimeError('Prediction failed; preserve partial artifacts')
        write(root / 'state.json', dict(status='PREDICTIONS_FROZEN', gpu=gpu), exclusive=False)
    except Exception as error:
        write(root / 'state.json', dict(status='FAILED', error=str(error)), exclusive=False)
        raise


def publish(repo, output, context, ssh_config):
    from experiments.pamf_lora_main.publish import release, release_commit
    from experiments.pamf_lora_main.w24_residual_pilot import R69_TRAIN, R69_GATE
    source = read(context)
    rt = {key: source[key] for key in ('project', 'python', 'frozen_sources')}
    project = PurePosixPath(rt['project'])
    runs, releases = project / 'runs', project / 'releases'
    target = releases / '20261122-phase1-lora-full-aligned-data-r01-release/data/target'
    rt.update(run_id=RUN, runner=__spec__.name, code_commit=release_commit(repo),
              release_root=str(releases / f'{RUN}-release'), run_root=str(runs / RUN), native_target_root=str(target),
              external_releases=[str(runs / value) for value in (PARENT, LORA_ACTION, base.LORA_GAIN, R69_TRAIN, R69_GATE)] +
              [str(target)] + [str(releases / f'{value}-release') for value in (base.LORA_GAIN, R69_TRAIN, R69_GATE)])
    release(repo, output, rt, ssh_config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='action', required=True)
    for action in ('dispatch', 'predict'):
        commands.add_parser(action).add_argument('runtime', type=Path)
    pub = commands.add_parser('publish')
    for field in ('repo', 'output', 'context', 'ssh-config'):
        pub.add_argument('--' + field, type=Path, required=True)
    score = commands.add_parser('score-local')
    for field in ('predictions', 'dataset', 'output'):
        score.add_argument('--' + field, type=Path, required=True)
    args = vars(parser.parse_args())
    action = args.pop('action')
    if action == 'publish':
        publish(**args)
    elif action == 'score-local':
        score_local(**args)
    else:
        {'dispatch': dispatch, 'predict': predict}[action](read(args['runtime']))
