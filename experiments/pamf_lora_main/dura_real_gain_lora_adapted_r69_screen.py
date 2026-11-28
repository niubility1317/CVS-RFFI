"""Matched source screen for LoRa-adapted historical R69 and frozen fusion."""

import argparse
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys

import numpy as np
import torch

from experiments.cric_dura_comparison.io import read, write
from experiments.pamf_lora_main.dura_real_gain_fit import DEVICE, LORA_ACTION, LORA_GAIN, as_complex
from experiments.pamf_lora_main.dura_real_gain_amplitude import R17AmplitudeGain
from experiments.pamf_lora_main.dura_real_gain_r17_fit import boot
from experiments.pamf_lora_main.dura_real_gain_run import free_gpus
from experiments.pamf_lora_main.dura_real_gain_screen import packet_scale, paired
from experiments.pamf_lora_main.dura_real_gain_unroll import fused_anchor_unroll
from experiments.pamf_lora_main import dura_real_gain_lora_group_robust as parent


RUN = '20261128-phase1-lora-adapted-r69-source-screen-s2026113041-r01'
PARENT = parent.RUN
SEED, COUNT = 2026113041, 300
VIEWS = ('clean', 'practical_low_urban')
MODES = ('direct', 'lora_native_r17', 'real_gain', 'adapted_r69', 'group_fusion')


def select_rows(ids, rx, earlier_rows, prior_ids):
    eligible = np.flatnonzero((rx == 4) & ~np.isin(ids, prior_ids))
    eligible = np.setdiff1d(eligible, earlier_rows)
    if len(eligible) < COUNT:
        raise ValueError('Insufficient fresh source V packets')
    return np.sort(np.random.default_rng(SEED + 2).choice(eligible, COUNT, replace=False))


@torch.no_grad()
def predict(rt):
    joint, native, old, contract, reader, _, realgain, r17 = boot(rt)
    from experiments.pamf_lora_main.w24_residual_pilot import r69_baseline, restore_r69
    project, root, source = Path(rt['project']), Path(rt['run_root']), Path(old['source_root'])
    parent_root = project / 'runs' / PARENT
    checkpoint = torch.load(parent_root / 'gain.pt', map_location=DEVICE, weights_only=True)
    proof = read(parent_root / 'fit.json')
    if (checkpoint['proof'] != proof or (proof['status'], proof['seed'], proof['updates']) !=
            ('SOURCE_FROZEN', 2026113017, 1500) or not proof['F_frozen'] or
            any(proof[k] for k in ('U_truth_reads', 'V_reads', 'target_reads'))):
        raise ValueError('Expected frozen source-trained group candidate')
    gain = R17AmplitudeGain(1024).to(DEVICE).eval().requires_grad_(False)
    gain.load_state_dict(checkpoint['weights'], strict=True)
    candidate = native.dura_model(project / 'runs' / LORA_ACTION, contract, DEVICE).eval().requires_grad_(False)
    candidate.estimator.load_state_dict(checkpoint['estimator'], strict=True)
    old_r69 = r69_baseline(dict(rt, source_root=str(source)), contract, DEVICE)

    def classify(z):
        cpu = np.asarray(z.contiguous().view(torch.float32).tolist(), dtype=np.float32).view(np.complex64)
        return joint.classify_aperture(reader, joint.received_aperture(cpu, DEVICE))

    def decisions(raw):
        y = as_complex(raw)
        scale = packet_scale(y)
        normalized = y / scale
        p, _ = realgain(normalized)
        restored, _ = restore_r69(raw, *old_r69, DEVICE)
        values = (y, r17(y), realgain(y)[0], as_complex(restored),
                  fused_anchor_unroll(candidate, gain, normalized, p) * scale)
        return np.stack([classify(z) for z in values])

    if decisions(joint.source_role(source, contract, 'U_s')[0][:2]).shape != (len(MODES), 2):
        raise ValueError('Real checkpoint source U no-query smoke failed')
    earlier_rows, _, _ = joint.panel(source, contract)
    with np.load(source / 'V_metadata.npz', allow_pickle=False) as metadata:
        ids, rx = metadata['ids'].copy(), metadata['rx_i'].copy()
    if ids.tolist() != contract['role_ids']['V']:
        raise ValueError('Source V physical order differs')
    prior_ids = []
    for run in (parent.PREVIOUS, parent.JOINT_PREVIOUS, parent.RETENTION_PREVIOUS,
                parent.AMPLITUDE_PREVIOUS, parent.ANCHOR_PREVIOUS, parent.FULL_PREVIOUS,
                parent.ACTION_PREVIOUS, parent.PRIOR_SCREEN, parent.COARSE_SCREEN,
                parent.CONTRAST_PARENT, parent.REPLICATE_SCREEN, parent.GATE_SCREEN, PARENT):
        with np.load(project / 'runs' / run / 'screen.npz', allow_pickle=False) as previous:
            prior_ids.append(previous['ids'].copy())
    rows = select_rows(ids, rx, earlier_rows, np.concatenate(prior_ids))
    raw = joint.native(source, 'V', contract['counts']['V'])[rows]
    matrix = {}
    for view in VIEWS:
        signals = np.stack([joint.source_receive(view, (int(i), joint.SOURCE[4], value))
                            for i, value in zip(rows, raw, strict=True)])
        matrix[view] = np.concatenate([decisions(signals[i:i+8]) for i in range(0, COUNT, 8)], axis=1)
        print('LORA_ADAPTED_R69_SCREEN', view, COUNT, flush=True)
    with (root / 'screen.npz').open('xb') as stream:
        np.savez(stream, ids=ids[rows], **matrix)
    write(root / 'screen-frozen.json', dict(status='PREDICTIONS_FROZEN', modes=list(MODES), views=list(VIEWS),
          count=COUNT, source_RX='rtl_1', source_truth_reads=0, target_reads=0,
          r69='LoRa source-trained R69 architecture with historical gate and half-dose projection'))


def score(rt):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Independent CPU scorer required')
    root = Path(rt['run_root'])
    frozen = read(root / 'screen-frozen.json')
    if (frozen['status'], frozen['modes'], frozen['views'], frozen['count'], frozen['source_truth_reads']) != (
            'PREDICTIONS_FROZEN', list(MODES), list(VIEWS), COUNT, 0):
        raise ValueError('All source predictions must freeze before truth')
    with np.load(root / 'screen.npz', allow_pickle=False) as archive:
        ids = archive['ids'].copy()
        outputs = {view: archive[view].copy() for view in VIEWS}
    if len(ids) != COUNT or any(x.shape != (len(MODES), COUNT) for x in outputs.values()):
        raise ValueError('Incomplete prediction matrix')
    old = read(Path(rt['project']) / 'runs' / LORA_GAIN / 'frequency_inphase_native' / 'fit-inputs.json')
    with np.load(Path(old['source_root']) / 'V_metadata.npz', allow_pickle=False) as metadata:
        truth = dict(zip(metadata['ids'].tolist(), metadata['y'].tolist(), strict=True))
    labels = np.asarray([truth[x] for x in ids], dtype=np.int64)
    permutation = np.random.default_rng(SEED + 3).permutation(COUNT)
    pairs = {name: paired(outputs['practical_low_urban'][-1], outputs['practical_low_urban'][i], labels, permutation, 2)
             for i, name in enumerate(MODES[:-1])}
    clean_net = {name: int(((outputs['clean'][-1] == labels).astype(np.int16) -
                            (outputs['clean'][i] == labels).astype(np.int16)).sum())
                 for i, name in enumerate(MODES[:-1])}
    write(root / 'screen-score.json', dict(status='SOURCE_SCREEN_SCORED', truth_last=True, count=COUNT,
          modes=list(MODES), correct={v: (x == labels).sum(1).tolist() for v, x in outputs.items()},
          low_urban_pairs=pairs, clean_net=clean_net, promote=False,
          reason='Control characterization only; parent failed its preregistered source screen.',
          r69_identity='LoRa source-trained R69 architecture with historical gate/half-dose, distinct from fixed WiSig DURA v1 weights/runtime',
          formal_acceptance_complete=False, exposure='Fresh source V packets; parent and E104 source V selection exposed, historical target benchmark exposed.'))


def dispatch(runtime):
    rt = read(runtime)
    root = Path(rt['run_root'])
    root.mkdir(exist_ok=False)
    try:
        available = free_gpus()
        if not available:
            raise RuntimeError('No GPU with 8 GiB free')
        gpu = next(value for value in (5, 7, 2, 3, 6, 1, 4, 0) if value in available)
        for action in ('predict', 'score'):
            env = dict(os.environ, CUDA_VISIBLE_DEVICES='' if action == 'score' else str(gpu),
                       OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='2')
            with (root / f'{action}.log').open('x', encoding='utf-8') as log:
                proc = subprocess.Popen([sys.executable, '-u', '-m', __spec__.name, action, str(runtime)],
                                        env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            write(root / 'state.json', dict(status=action.upper(), pid=proc.pid, gpu=None if action == 'score' else gpu), exclusive=False)
            if proc.wait():
                raise RuntimeError(f'{action} failed; preserve partial artifacts')
        write(root / 'state.json', dict(status='SOURCE_SCREEN_SCORED', gpu=gpu, promote=False), exclusive=False)
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
    rt.update(run_id=RUN, runner=__spec__.name, code_commit=release_commit(repo),
              release_root=str(releases / f'{RUN}-release'), run_root=str(runs / RUN),
              external_releases=[str(runs / value) for value in (LORA_ACTION, LORA_GAIN, PARENT, R69_TRAIN, R69_GATE,
                  parent.PREVIOUS, parent.JOINT_PREVIOUS, parent.RETENTION_PREVIOUS, parent.AMPLITUDE_PREVIOUS,
                  parent.ANCHOR_PREVIOUS, parent.FULL_PREVIOUS, parent.ACTION_PREVIOUS, parent.PRIOR_SCREEN,
                  parent.COARSE_SCREEN, parent.CONTRAST_PARENT, parent.REPLICATE_SCREEN, parent.GATE_SCREEN)] +
                  [str(releases / f'{value}-release') for value in (LORA_GAIN, R69_TRAIN, R69_GATE)])
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
