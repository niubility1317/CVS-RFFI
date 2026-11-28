"""Evaluate a LoRa-trained R69 structural adaptation on the registered W24 source panel."""
import argparse
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys

import numpy as np
import torch

from experiments.cric_dura_comparison.io import read, tensor, write
from experiments.pamf_lora_main.dura_real_gain_fit import DEVICE, LORA_GAIN, activate_wisig_archive, lora_models
from experiments.pamf_lora_main.dura_real_gain_run import free_gpus

RUN = '20261128-diagnostic-lora-r69-full-w24-s2026112895-r01'
PARENT = '20261128-phase1-dura-realgain-crossdata-s2026112845-r02'
R69 = '20261122-phase1-lora-dura-r69-aligned-m2-r01'
SEED = 2026092702
VIEWS = ('clean', 'practical_low_urban')
MODES = ('direct', 'real_gain', 'lora_r69_half')


def boot(rt):
    from experiments.cric_dura_comparison.bootstrap import activate, numerical_policy
    numerical_policy()
    activate(rt)
    project = Path(rt['project'])
    release = project / 'releases' / f'{LORA_GAIN}-release'
    activate_wisig_archive(release)
    activate_wisig_archive(Path(read(release / 'runtime.json')['w24_release_root']))
    lora_models(project)
    from experiments.pamf_lora_main import w24_joint_pamf as joint
    from experiments.pamf_lora_main.full_dura_r69_lora import load_model, load_utility
    old = read(project / 'runs' / LORA_GAIN / 'frequency_inphase_native' / 'fit-inputs.json')
    contract, reader, _, _ = joint.boot(old, DEVICE)
    root = project / 'runs' / R69
    gate = load_utility(root, contract, SEED, DEVICE)
    model = load_model(root, contract, SEED, 'student', DEVICE)
    return joint, old, contract, reader.eval().requires_grad_(False), gate, model


@torch.no_grad()
def restore_packets(packets, gate, model, *, force=False):
    from experiments.pamf_lora_main.full_dura_r69_lora import restored
    from experiments.pamf_lora_main.packet_physics import chirp_samples
    from experiments.pamf_lora_main.w24_native_full import project_delta
    chirps = np.stack([chirp_samples(q) for q in packets])
    scale = np.sqrt(np.mean(np.abs(chirps)**2, axis=(1, 2)))
    if not np.isfinite(scale).all() or (scale <= 1e-12).any():
        raise ValueError('Invalid current packet scale')
    channels = np.stack((chirps.real, chirps.imag), axis=2) / scale[:, None, None, None]
    iq = tensor(channels.astype(np.float32, copy=False), DEVICE)
    active = torch.ones(len(iq), dtype=torch.bool, device=iq.device) if force else gate.impairment(iq) > 0
    output = iq.clone()
    if active.any():
        output[active] = restored(gate, model, iq[active])[1]
    if not torch.isfinite(output).all():
        raise ValueError('Nonfinite LoRa R69 adaptation')
    delta = np.from_dlpack((output - iq).cpu().contiguous())
    return project_delta(packets, delta, scale), int(active.sum())


@torch.no_grad()
def predict(rt):
    project, root = Path(rt['project']), Path(rt['run_root'])
    joint, old, contract, reader, gate, model = boot(rt)
    source = Path(old['source_root'])

    def classify(packets):
        return joint.classify_aperture(reader, joint.received_aperture(packets, DEVICE))

    smoke = joint.source_role(source, contract, 'U_s')[0][:2]
    smoke_out, smoke_active = restore_packets(smoke, gate, model, force=True)
    if smoke_active != 2 or smoke_out.shape != smoke.shape or classify(smoke_out).shape != (2,):
        raise ValueError('Real-checkpoint no-query LoRa R69 smoke failed')
    write(root / 'smoke.json', dict(status='PASS', source='U_s', count=2, target_reads=0, TX_reads=0))
    panel, receivers, physical_ids = joint.panel(source, contract)
    selected = np.flatnonzero(receivers == 0)
    if len(selected) != 60:
        raise ValueError('Expected 60 fixed source packets on b200_1')
    rows = panel[selected]
    ids = np.asarray(physical_ids)[selected]
    parent = project / 'runs' / PARENT / 'lora'
    mark = read(parent / 'screen-frozen.json')
    if (mark['status'], mark['modes'], mark['views'], mark['count'],
            mark['source_truth_reads'], mark['target_reads']) != (
            'PREDICTIONS_FROZEN', ['direct', 'lora_native_r17', 'real_gain', 'fusion'],
            list(VIEWS), 60, 0, 0):
        raise ValueError('Historical screen must be fixed before comparison')
    with np.load(parent / 'screen.npz', allow_pickle=False) as archive:
        if not np.array_equal(ids, archive['ids']):
            raise ValueError('Historical panel IDs differ')
        historical = {view: archive[view].copy() for view in VIEWS}
    raw = joint.native(source, 'V', contract['counts']['V'])[rows]
    results, activation = {}, {}
    for view in VIEWS:
        signals = np.stack([joint.source_receive(view, (int(i), joint.SOURCE[0], x))
                            for i, x in zip(rows, raw, strict=True)])
        direct, adapted, active = [], [], 0
        for start in range(0, len(ids), 8):
            packets = signals[start:start+8]
            output, n = restore_packets(packets, gate, model)
            direct.append(classify(packets))
            adapted.append(classify(output))
            active += n
        direct, adapted = np.concatenate(direct), np.concatenate(adapted)
        if not np.array_equal(direct, historical[view][0]):
            raise ValueError('Historical direct predictions differ from replayed observations')
        results[view] = np.stack((direct, historical[view][2], adapted))
        activation[view] = active
        print('LORA_R69_W24_SCREEN', view, len(ids), active, flush=True)
    with (root / 'screen.npz').open('xb') as stream:
        np.savez(stream, ids=ids, **results)
    write(root / 'screen-frozen.json', dict(status='PREDICTIONS_FROZEN', modes=list(MODES), views=list(VIEWS),
          count=60, source_RX='b200_1', activation=activation, source_truth_reads=0, target_reads=0))


def score(rt):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Independent CPU scorer required')
    from experiments.pamf_lora_main.dura_real_gain_screen import paired
    project, root = Path(rt['project']), Path(rt['run_root'])
    frozen = read(root / 'screen-frozen.json')
    if (frozen['status'], frozen['modes'], frozen['views'], frozen['count'],
            frozen['source_truth_reads'], frozen['target_reads']) != (
            'PREDICTIONS_FROZEN', list(MODES), list(VIEWS), 60, 0, 0):
        raise ValueError('Two matched views must freeze before source truth')
    with np.load(root / 'screen.npz', allow_pickle=False) as archive:
        ids, outputs = archive['ids'].copy(), {v: archive[v].copy() for v in VIEWS}
    if len(set(ids.tolist())) != 60 or any(x.shape != (3, 60) for x in outputs.values()):
        raise ValueError('Incomplete LoRa source control')
    old = read(project / 'runs' / LORA_GAIN / 'frequency_inphase_native' / 'fit-inputs.json')
    with np.load(Path(old['source_root']) / 'V_metadata.npz', allow_pickle=False) as meta:
        lookup = dict(zip(meta['ids'].tolist(), meta['y'].tolist(), strict=True))
    labels = np.asarray([lookup[x] for x in ids.tolist()], dtype=np.int64)
    perm = np.random.default_rng(2026112852).permutation(60)
    results = {}
    for view in VIEWS:
        values = outputs[view]
        results[view] = dict(correct=(values == labels).sum(1).tolist(),
                             versus_direct=paired(values[2], values[0], labels, perm, 1),
                             versus_real_gain=paired(values[2], values[1], labels, perm, 1))
    write(root / 'screen-score.json', dict(status='SOURCE_SCREEN_SCORED', truth_last=True, modes=list(MODES),
          views=results, exposure='Previously exposed source V panel; no target access',
          identity='LoRa-trained R69 architecture adaptation, not frozen WiSig v1 checkpoint'))


def dispatch(runtime):
    rt = read(runtime)
    root = Path(rt['run_root'])
    root.mkdir(exist_ok=False)
    try:
        available = free_gpus()
        gpu = next(i for i in (7, 2, 3, 6, 1, 4) if i in available)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1')
        with (root / 'predict.log').open('x', encoding='utf-8') as log:
            proc = subprocess.Popen([sys.executable, '-u', '-m', __spec__.name, 'predict', str(runtime)],
                                    env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        write(root / 'state.json', dict(status='PREDICT', pid=proc.pid, gpu=gpu))
        if proc.wait():
            raise RuntimeError('Source prediction failed')
        env['CUDA_VISIBLE_DEVICES'] = ''
        with (root / 'score.log').open('x', encoding='utf-8') as log:
            proc = subprocess.Popen([sys.executable, '-u', '-m', __spec__.name, 'score', str(runtime)],
                                    env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        write(root / 'state.json', dict(status='SCORE', pid=proc.pid, gpu=None), exclusive=False)
        if proc.wait():
            raise RuntimeError('Independent source score failed')
        write(root / 'state.json', dict(status='SOURCE_SCREEN_SCORED'), exclusive=False)
    except Exception as error:
        write(root / 'state.json', dict(status='FAILED', error=str(error)), exclusive=False)
        raise


def publish(repo, output, context, ssh_config):
    from experiments.pamf_lora_main.publish import release, release_commit
    source = read(context)
    rt = {k: source[k] for k in ('project', 'python', 'frozen_sources')}
    project = PurePosixPath(rt['project'])
    runs, releases = project / 'runs', project / 'releases'
    rt.update(run_id=RUN, runner=__spec__.name, code_commit=release_commit(repo),
              release_root=str(releases / f'{RUN}-release'), run_root=str(runs / RUN),
              external_releases=[str(runs / name) for name in (LORA_GAIN, R69, PARENT)] +
                                [str(releases / f'{LORA_GAIN}-release')])
    release(repo, output, rt, ssh_config, archive_extras=('tools/r17_activation_factorization.py',))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='action', required=True)
    for phase in ('dispatch', 'predict', 'score'):
        commands.add_parser(phase).add_argument('runtime', type=Path)
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
        (predict if action == 'predict' else score)(read(args['runtime']))
