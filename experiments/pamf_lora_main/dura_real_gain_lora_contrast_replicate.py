"""Fixed-weight cross-receiver source replication of LoRa packet contrast."""

import argparse
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
from time import monotonic

import numpy as np
import torch
from torch.nn import functional as F

from experiments.cric_dura_comparison.io import read, write
from experiments.pamf_lora_main.dura_real_gain_fit import DEVICE, LORA_ACTION, LORA_GAIN, as_complex
from experiments.pamf_lora_main.dura_real_gain_amplitude import R17AmplitudeGain
from experiments.pamf_lora_main.dura_real_gain_unroll import fused_anchor_unroll
from experiments.pamf_lora_main.dura_real_gain_wisig_packet_contrast import pair_contrastive
from experiments.pamf_lora_main.dura_real_gain_r17_fit import boot
from experiments.pamf_lora_main.dura_real_gain_run import free_gpus
from experiments.pamf_lora_main.dura_real_gain_screen import packet_scale, paired


RUN = '20261128-phase1-lora-r07-contrast-replicate-s2026113001-r01'
FROZEN_RUN = '20261128-phase1-lora-r07-packet-contrast-s2026112977-r01'
COARSE_SCREEN = '20261128-phase1-lora-r02-coarse-joint-s2026112961-r01'
PRIOR_SCREEN = '20261128-phase1-lora-r17-prior-joint-s2026112941-r01'
ACTION_PREVIOUS = '20261128-phase1-lora-r07-action-joint-s2026112929-r01'
FULL_PREVIOUS = '20261128-phase1-lora-r17-full-source-s2026112921-r01'
ANCHOR_PREVIOUS = '20261128-phase1-lora-r17-anchor-unroll-s2026112913-r01'
AMPLITUDE_PREVIOUS = '20261128-phase1-lora-r17-amplitude-gain-s2026112897-r01'
RETENTION_PREVIOUS = '20261128-phase1-lora-r17-recognition-retain-s2026112885-r01'
PREVIOUS = '20261128-phase1-lora-r17-realgain-e104-s2026112861-r01'
JOINT_PREVIOUS = '20261128-phase1-lora-joint-r17-realgain-s2026112873-r01'
SEED, STEPS, ROLE_BATCH, COUNT = 2026113001, 1500, 2, 300
VIEWS = ('clean', 'practical_low_urban')
MODES = ('direct', 'lora_native_r17', 'real_gain', 'joint_fusion')


def select_rows(ids, rx, earlier_rows, prior_ids):
    eligible = np.flatnonzero((rx == 3) & ~np.isin(ids, prior_ids))
    eligible = np.setdiff1d(eligible, earlier_rows)
    if len(eligible) < COUNT:
        raise ValueError('Insufficient unexposed source V packets for fixed RX')
    return np.sort(np.random.default_rng(SEED + 2).choice(eligible, COUNT, replace=False))


@torch.no_grad()
def predict(rt):
    joint, native, old, contract, reader, _, baseline, r17 = boot(rt)
    root, source = Path(rt['run_root']), Path(old['source_root'])
    frozen_root = Path(rt['project']) / 'runs' / FROZEN_RUN
    checkpoint = torch.load(frozen_root / 'gain.pt', map_location=DEVICE, weights_only=True)
    proof = read(frozen_root / 'fit.json')
    if (checkpoint['proof'] != proof or (proof['status'], proof['seed'], proof['updates']) !=
            ('SOURCE_FROZEN', 2026112977, STEPS) or any(proof[k] for k in ('U_truth_reads', 'V_reads', 'target_reads'))):
        raise ValueError('Expected fixed lawful source-trained joint fusion')
    gain = R17AmplitudeGain(1024).to(DEVICE).eval().requires_grad_(False)
    gain.load_state_dict(checkpoint['weights'], strict=True)
    candidate_r17 = native.dura_model(Path(rt['project']) / 'runs' / LORA_ACTION, contract, DEVICE).eval().requires_grad_(False)
    candidate_r17.estimator.load_state_dict(checkpoint['estimator'], strict=True)

    def classify(z):
        cpu = np.asarray(z.contiguous().view(torch.float32).tolist(), dtype=np.float32).view(np.complex64)
        return joint.classify_aperture(reader, joint.received_aperture(cpu, DEVICE))

    def decisions(raw):
        y = as_complex(raw)
        scale = packet_scale(y)
        normalized = y / scale
        p, _ = baseline(normalized)
        values = (y, r17(y), baseline(y)[0], fused_anchor_unroll(candidate_r17, gain, normalized, p) * scale)
        return np.stack([classify(z) for z in values])

    if decisions(joint.source_role(source, contract, 'U_s')[0][:2]).shape != (len(MODES), 2):
        raise ValueError('No-query checkpoint smoke failed')
    earlier_rows, _, _ = joint.panel(source, contract)
    with np.load(source / 'V_metadata.npz', allow_pickle=False) as metadata:
        ids, rx = metadata['ids'].copy(), metadata['rx_i'].copy()
    if ids.tolist() != contract['role_ids']['V']:
        raise ValueError('Source V physical order differs')
    with np.load(Path(rt['project']) / 'runs' / PREVIOUS / 'screen.npz', allow_pickle=False) as prior:
        prior_ids = prior['ids'].copy()
    with np.load(Path(rt['project']) / 'runs' / JOINT_PREVIOUS / 'screen.npz', allow_pickle=False) as prior:
        prior_ids = np.concatenate((prior_ids, prior['ids'].copy()))
    with np.load(Path(rt['project']) / 'runs' / RETENTION_PREVIOUS / 'screen.npz', allow_pickle=False) as prior:
        prior_ids = np.concatenate((prior_ids, prior['ids'].copy()))
    with np.load(Path(rt['project']) / 'runs' / AMPLITUDE_PREVIOUS / 'screen.npz', allow_pickle=False) as prior:
        prior_ids = np.concatenate((prior_ids, prior['ids'].copy()))
    with np.load(Path(rt['project']) / 'runs' / ANCHOR_PREVIOUS / 'screen.npz', allow_pickle=False) as prior:
        prior_ids = np.concatenate((prior_ids, prior['ids'].copy()))
    with np.load(Path(rt['project']) / 'runs' / FULL_PREVIOUS / 'screen.npz', allow_pickle=False) as prior:
        prior_ids = np.concatenate((prior_ids, prior['ids'].copy()))
    with np.load(Path(rt['project']) / 'runs' / ACTION_PREVIOUS / 'screen.npz', allow_pickle=False) as prior:
        prior_ids = np.concatenate((prior_ids, prior['ids'].copy()))
    with np.load(Path(rt['project']) / 'runs' / PRIOR_SCREEN / 'screen.npz', allow_pickle=False) as prior:
        prior_ids = np.concatenate((prior_ids, prior['ids'].copy()))
    with np.load(Path(rt['project']) / 'runs' / COARSE_SCREEN / 'screen.npz', allow_pickle=False) as prior:
        prior_ids = np.concatenate((prior_ids, prior['ids'].copy()))
    with np.load(Path(rt['project']) / 'runs' / FROZEN_RUN / 'screen.npz', allow_pickle=False) as prior:
        prior_ids = np.concatenate((prior_ids, prior['ids'].copy()))
    rows = select_rows(ids, rx, earlier_rows, prior_ids)
    raw = joint.native(source, 'V', contract['counts']['V'])[rows]
    matrix = {}
    for view in VIEWS:
        signals = np.stack([joint.source_receive(view, (int(i), joint.SOURCE[3], value))
                            for i, value in zip(rows, raw, strict=True)])
        matrix[view] = np.concatenate([decisions(signals[i:i+8]) for i in range(0, COUNT, 8)], axis=1)
        print('LORA_CONTRAST_REPLICATE_SCREEN', view, COUNT, flush=True)
    with (root / 'screen.npz').open('xb') as stream:
        np.savez(stream, ids=ids[rows], **matrix)
    write(root / 'screen-frozen.json', dict(status='PREDICTIONS_FROZEN', modes=list(MODES), views=list(VIEWS),
          count=COUNT, source_RX='pluto_1', prior_screen_overlap=0, source_truth_reads=0, target_reads=0))


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
    hard, clean = outputs[VIEWS[-1]], outputs['clean']
    pairs = {name: paired(hard[-1], hard[i], labels, permutation, 2) for i, name in enumerate(MODES[:-1])}
    clean_net = {name: int(((clean[-1] == labels).astype(np.int16) - (clean[i] == labels).astype(np.int16)).sum())
                 for i, name in enumerate(MODES[:-1])}
    promote = all(value['passes'] for value in pairs.values()) and min(clean_net.values()) >= -2
    write(root / 'screen-score.json', dict(status='SOURCE_SCREEN_SCORED', truth_last=True, count=COUNT,
          modes=list(MODES), correct={v: (x == labels).sum(1).tolist() for v, x in outputs.items()},
          low_urban_pairs=pairs, clean_net=clean_net, promote=promote,
          missing_controls=['same-contract LoRa DURA v1'], formal_acceptance_complete=False,
          exposure='Fixed checkpoint selected using an exposed source screen; fresh RX3 packets exclude previous screens, source V historically exposed.'))


def dispatch(runtime):
    rt = read(runtime)
    root = Path(rt['run_root'])
    root.mkdir(exist_ok=False)
    try:
        devices = free_gpus()
        if not devices:
            raise RuntimeError('No GPU with 8 GiB free')
        gpu = next(value for value in (5, 7, 2, 3, 6, 1, 4, 0) if value in devices)
        for action in ('predict', 'score'):
            device = None if action == 'score' else gpu
            env = dict(os.environ, CUDA_VISIBLE_DEVICES='' if device is None else str(device),
                       OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='2')
            with (root / f'{action}.log').open('x', encoding='utf-8') as log:
                proc = subprocess.Popen([sys.executable, '-u', '-m', __spec__.name, action, str(runtime)],
                                        env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            write(root / 'state.json', dict(status=action.upper(), pid=proc.pid, gpu=device), exclusive=False)
            if proc.wait():
                raise RuntimeError(f'{action} failed; preserve partial artifacts')
        write(root / 'state.json', dict(status='SOURCE_SCREEN_SCORED', gpu=gpu,
              promote=read(root / 'screen-score.json')['promote']), exclusive=False)
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
              external_releases=[str(runs / LORA_ACTION), str(runs / LORA_GAIN), str(runs / PREVIOUS), str(runs / JOINT_PREVIOUS), str(runs / RETENTION_PREVIOUS), str(runs / AMPLITUDE_PREVIOUS), str(runs / ANCHOR_PREVIOUS), str(runs / FULL_PREVIOUS), str(runs / ACTION_PREVIOUS), str(runs / PRIOR_SCREEN), str(runs / COARSE_SCREEN), str(runs / FROZEN_RUN),
                                 str(releases / f'{LORA_GAIN}-release')])
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
