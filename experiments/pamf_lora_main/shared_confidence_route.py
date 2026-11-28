"""Shared per-packet frozen-default-head confidence routing; no fitting or labels."""
import argparse
import importlib
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys

import numpy as np
import torch

from experiments.cric_dura_comparison.io import read, write
from experiments.pamf_lora_main.dura_real_gain_fit import (DEVICE, LORA_ACTION, LORA_GAIN, WISIG_CACHE,
    WISIG_PACKET, WISIG_PARENT, activate_wisig_archive, as_complex, lora_models, wisig_models, wisig_real_prior)
from experiments.pamf_lora_main.dura_real_gain_screen import load_gain, packet_scale
from experiments.pamf_lora_main.shared_dura_realgain_run import REFERENCE, VIEWS, dispatch, score

RUN = '20261128-phase1-shared-confidence-route-s2026112925-r01'
PARENT = '20261128-phase1-dura-realgain-crossdata-s2026112845-r02'


def select_by_confidence(real, r17, real_logits, r17_logits):
    if (real.shape != r17.shape or real.ndim != 2 or real_logits.ndim != 2 or
            real_logits.shape != r17_logits.shape or len(real_logits) != len(real) or
            not torch.isfinite(real_logits).all() or not torch.isfinite(r17_logits).all()):
        raise ValueError('Expected paired packets and finite matching logits')
    real_conf = real_logits.softmax(-1).amax(-1)
    r17_conf = r17_logits.softmax(-1).amax(-1)
    return torch.where((real_conf > r17_conf)[:, None], real, r17)


def prediction_classes(logits):
    # tolist avoids the project's PyTorch/NumPy ABI mismatch on N607.
    return np.asarray(logits.argmax(1).cpu().tolist(), dtype=np.int16)


@torch.no_grad()
def lora_predictions(rt):
    from experiments.cric_dura_comparison.bootstrap import activate
    activate(rt)
    project = Path(rt['project'])
    parent = read(project / 'releases' / f'{LORA_GAIN}-release' / 'runtime.json')
    if parent['run_id'] != LORA_GAIN:
        raise ValueError('Unexpected LoRa real-gain parent')
    activate_wisig_archive(project / 'releases' / f'{LORA_GAIN}-release')
    activate_wisig_archive(Path(parent['w24_release_root']))
    from experiments.pamf_lora_main import w24_joint_pamf as joint
    from experiments.pamf_lora_main import w24_native_training as native
    old = read(project / 'runs' / native.PHASE_RUN / 'frequency_inphase_native' / 'fit-inputs.json')
    source = Path(old['source_root'])
    contract, reader, _, _ = joint.boot(old, DEVICE)
    real = native.task_model(None, DEVICE)
    real.core.innovation = native.spectral_innovation('frequency_inphase_native', DEVICE)
    native.load_stage(project / 'runs' / native.PHASE_RUN, 'frequency_inphase_native', contract, real.core, DEVICE)
    real.eval().requires_grad_(False)
    r17 = native.dura_model(project / 'runs' / LORA_ACTION, contract, DEVICE).eval().requires_grad_(False)

    def logits(z):
        cpu = np.asarray(z.contiguous().view(torch.float32).tolist(), dtype=np.float32).view(np.complex64)
        out = reader(joint.received_aperture(cpu, DEVICE))[0]
        if out.ndim != 2 or out.shape[0] != len(z) or not torch.isfinite(out).all():
            raise ValueError('Invalid frozen LoRa default-head logits')
        return out

    def decisions(raw):
        y = as_complex(raw)
        a, b = real(y), r17(y)
        la, lb = logits(a), logits(b)
        fused = select_by_confidence(a, b, la, lb)
        return np.stack([prediction_classes(logits(z)) for z in (y, b, a, fused)])

    sample = joint.source_role(source, contract, 'U_s')[0][:2]
    if decisions(sample).shape != (4, 2):
        raise ValueError('Real LoRa no-query smoke failed')
    indices, receivers, physical_ids = joint.panel(source, contract)
    chosen = np.flatnonzero(receivers == 0)
    if len(chosen) != 60:
        raise ValueError('Unexpected LoRa source panel')
    rows, ids = indices[chosen], np.asarray(physical_ids)[chosen]
    raw = joint.native(source, 'V', contract['counts']['V'])[rows]
    outputs = {}
    for view in VIEWS:
        signals = np.stack([joint.source_receive(view, (int(i), joint.SOURCE[0], x))
                            for i, x in zip(rows, raw, strict=True)])
        outputs[view] = np.concatenate([decisions(signals[i:i+8]) for i in range(0, len(ids), 8)], axis=1)
        print('CONFIDENCE_FROZEN_VIEW', 'lora', view, len(ids), flush=True)
    return ids, outputs


@torch.no_grad()
def wisig_predictions(rt):
    project = Path(rt['project'])
    _, _, r17 = wisig_models(project, rt['frozen_sources'])
    original = importlib.import_module('experiments.pamf_unrolled_restore.wisig_anchored_window_source')
    config = read(project / 'releases' / '20261125-phase1-wisig-native-fir-source-m4-r01-release' / 'runtime.json')
    reader = original.tri.reader_factory(config).eval().requires_grad_(False)
    control = load_gain(project / 'runs' / PARENT, 'wisig', real=True)
    packet = wisig_real_prior(project)

    def logits(z):
        out = original.classify(reader, z)
        if out.ndim != 2 or out.shape[0] != len(z) or not torch.isfinite(out).all():
            raise ValueError('Invalid frozen WiSig default-head logits')
        return out

    def decisions(batch):
        y = as_complex(batch)
        scale = packet_scale(y)
        observed, estimate, group_scale = packet(y / scale)
        a = ((estimate + control(observed, estimate)) * group_scale).flatten(-2) * scale
        b, _ = r17(y)
        fused = select_by_confidence(a, b, logits(a), logits(b))
        return np.stack([prediction_classes(logits(z)) for z in (y, b, a, fused)])

    source, _, calibration = original.core.source_iq(Path(config['source_root']))
    if decisions(source[calibration[:2]]).shape != (4, 2):
        raise ValueError('Real WiSig no-query smoke failed')
    cache, _, _, _, ids, rx, _ = original.core.baseline.handles(config)
    chosen = np.flatnonzero(rx == 1)
    if len(chosen) != 360:
        raise ValueError('Unexpected WiSig source panel')
    outputs = {}
    for view in VIEWS:
        signals = np.load(cache / 'validation_full' / f'{view}.npy', mmap_mode='r', allow_pickle=False)
        outputs[view] = np.concatenate([decisions(signals[chosen[i:i+48]]) for i in range(0, len(chosen), 48)], axis=1)
        print('CONFIDENCE_FROZEN_VIEW', 'wisig', view, len(chosen), flush=True)
    return ids[chosen], outputs


def predict(rt, dataset):
    from experiments.cric_dura_comparison.bootstrap import numerical_policy
    numerical_policy()
    project, root = Path(rt['project']), Path(rt['run_root'])
    ids, values = (lora_predictions if dataset == 'lora' else wisig_predictions)(rt)
    if dataset == 'lora':
        old = project / 'runs' / '20261128-diagnostic-lora-r69-full-w24-s2026112895-r01'
        mark = read(old / 'screen-frozen.json')
        if (mark['status'], mark['modes'], mark['views'], mark['count'], mark['source_truth_reads'], mark['target_reads']) != (
                'PREDICTIONS_FROZEN', ['direct', 'real_gain', 'lora_r69_half'], list(VIEWS), 60, 0, 0):
            raise ValueError('Expected matched frozen LoRa R69 control')
        with np.load(old / 'screen.npz', allow_pickle=False) as archive:
            if not np.array_equal(ids, archive['ids']):
                raise ValueError('LoRa physical IDs differ')
            for view in VIEWS:
                history = archive[view]
                if not np.array_equal(history[0], values[view][0]) or not np.array_equal(history[1], values[view][2]):
                    raise ValueError('LoRa matched predictions differ')
                values[view] = np.concatenate((values[view][:-1], history[2][None], values[view][-1:]))
        modes = ['direct', 'lora_native_r17', 'real_gain', 'lora_r69_adapt', 'fusion']
    else:
        old = project / 'runs' / WISIG_PACKET / 'wisig'
        mark = read(old / 'frozen.json')
        if mark['status'] != 'PREDICTIONS_FROZEN' or mark['source_truth_reads'] or mark['target_reads']:
            raise ValueError('Expected matched frozen WiSig DURA control')
        with np.load(old / 'predictions.npz', allow_pickle=False) as archive:
            historical_ids = archive['ids'].tolist()
            if len(set(historical_ids)) != len(historical_ids):
                raise ValueError('Duplicate WiSig control IDs')
            lookup = {item: i for i, item in enumerate(historical_ids)}
            indices = [lookup[item] for item in ids.tolist()]
            for view in VIEWS:
                history = archive[view]
                if not np.array_equal(history[0, indices], values[view][0]):
                    raise ValueError('WiSig matched predictions differ')
                values[view] = np.concatenate((values[view][:-1], history[1, indices][None], values[view][-1:]))
        modes = ['direct', 'wisig_r17', 'real_gain', 'dura_v1', 'fusion']
    folder = root / dataset
    with (folder / 'screen.npz').open('xb') as stream:
        np.savez(stream, ids=ids, **values)
    write(folder / 'screen-frozen.json', dict(status='PREDICTIONS_FROZEN', views=list(VIEWS), modes=modes,
          count=len(ids), RX=0 if dataset == 'lora' else 1, source_truth_reads=0, target_reads=0))


def worker(runtime, dataset):
    predict(read(runtime), dataset)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='')
    result = subprocess.run([sys.executable, '-u', '-m', __spec__.name, 'score', str(runtime), dataset],
                            env=env, stdin=subprocess.DEVNULL, check=False)
    if result.returncode:
        raise RuntimeError('Independent source scoring failed')


def publish(repo, output, context, ssh_config):
    from experiments.pamf_lora_main.publish import release, release_commit
    source = read(context)
    rt = {k: source[k] for k in ('project', 'python', 'frozen_sources')}
    project = PurePosixPath(rt['project'])
    runs, releases = project / 'runs', project / 'releases'
    rt.update(run_id=RUN, runner=__spec__.name, code_commit=release_commit(repo),
              release_root=str(releases / f'{RUN}-release'), run_root=str(runs / RUN),
              lora_frozen_root=str(runs / PARENT), allowed_gpus=[7, 1, 6, 4],
              external_releases=[str(runs / n) for n in (LORA_ACTION, LORA_GAIN, WISIG_CACHE,
                  WISIG_PACKET, WISIG_PARENT, PARENT,
                  '20261128-diagnostic-lora-r69-full-w24-s2026112895-r01')] + [str(releases / REFERENCE)])
    release(repo, output, rt, ssh_config, archive_extras=('tools/r17_activation_factorization.py',))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='action', required=True)
    for phase in ('dispatch', 'worker', 'score'):
        child = commands.add_parser(phase)
        child.add_argument('runtime', type=Path)
        if phase != 'dispatch':
            child.add_argument('dataset', choices=('lora', 'wisig'))
    pub = commands.add_parser('publish')
    for field in ('repo', 'output', 'context', 'ssh-config'):
        pub.add_argument('--' + field, type=Path, required=True)
    args = vars(parser.parse_args())
    action = args.pop('action')
    if action == 'dispatch':
        dispatch(args['runtime'], worker_module=__spec__.name)
    elif action == 'score':
        score(read(args['runtime']), args['dataset'])
    elif action == 'worker':
        worker(**args)
    else:
        publish(**args)
