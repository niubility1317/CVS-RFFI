"""Jointly adapt WiSig R17 priors and real-gain fusion without TX or F."""

import argparse
from pathlib import Path, PurePosixPath
import importlib
import os
import subprocess
import sys
from time import monotonic

import numpy as np
import torch

from experiments.cric_dura_comparison.io import read, write
from experiments.pamf_lora_main.dura_real_gain_fit import DEVICE, WISIG_PARENT, as_complex, activate_wisig_archive
from experiments.pamf_lora_main.dura_real_gain_r17 import R17Gain
from experiments.pamf_lora_main.dura_real_gain_run import free_gpus
from experiments.pamf_lora_main.dura_real_gain_screen import packet_scale, paired
from experiments.pamf_lora_main.dura_real_gain_wisig_next import setup
from experiments.pamf_restoration.models import complex_iq, pair


RUN = '20261128-phase1-wisig-r17-joint-realgain-source-s2026112869-r01'
PREVIOUS = '20261128-phase1-wisig-r17-realgain-source-s2026112865-r01'
SEED, STEPS, BATCH, RX = 2026112869, 1500, 16, 4
VIEWS = ('clean', 'practical_low_urban')
MODES = ('direct', 'wisig_r17', 'real_gain', 'first_fusion', 'second_fusion', 'joint_fusion')


def configure_student(student):
    student.requires_grad_(False)
    student.prior.requires_grad_(True)
    student.train()
    student.frozen.eval()
    return student


def seeded_gain(device):
    torch.manual_seed(SEED)
    return R17Gain(32).to(device).train()


def original_student(project):
    from experiments.pamf_unrolled_restore.runtime import load_checkpoint
    path = project / 'runs' / WISIG_PARENT / 'unroll_unshared' / 'native_map_prior' / 'final.pt'
    model, proof = load_checkpoint(path, DEVICE)
    if (proof['candidate'], proof['epoch'], proof['updates']) != ('unroll_unshared', 20, 4940):
        raise ValueError('Expected the original source-selected full R17')
    return model


def fit(rt):
    from experiments.pamf_lora_main.w24_native_dura import source_pair
    source, ids, teacher, real_gain, _ = setup(rt)
    student = configure_student(original_student(Path(rt['project'])))
    gain = seeded_gain(DEVICE)
    rng = np.random.default_rng(SEED)
    generator = torch.Generator(device=DEVICE).manual_seed(SEED + 1)
    optimizer = torch.optim.AdamW([{'params': student.prior.parameters(), 'lr': 3e-5},
                                  {'params': gain.parameters(), 'lr': 3e-4}], weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, STEPS, eta_min=1e-6)
    started = monotonic()
    for step in range(1, STEPS + 1):
        clean, observed, _, _ = source_pair(as_complex(source[rng.choice(ids, BATCH, replace=False)]), generator)
        with torch.no_grad():
            old, _ = teacher(observed)
            old_clean, _ = teacher(clean)
            real = real_gain(observed)
            real_clean = real_gain(clean)
        p, _, action, _ = student.training_outputs(pair(observed))
        pc, _, clean_action, _ = student.training_outputs(pair(clean))
        p, pc = complex_iq(p), complex_iq(pc)
        restored = gain(observed, p, real, action)
        clean_output = gain(clean, pc, real_clean, clean_action)
        if step == 1:
            torch.testing.assert_close(restored, p, rtol=0, atol=0)
        error = restored - clean
        power = torch.fft.fft(clean, norm='ortho').abs().square()
        spectral = (torch.fft.fft(error, norm='ortho').abs().square() / (.03 + power)).log1p().mean()
        old_retention = (restored - old).abs().square().mean()
        clean_retention = (clean_output - old_clean).abs().square().mean()
        loss = error.abs().square().mean() + .3 * spectral + .3 * old_retention + 1.5 * clean_retention
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if step == 1:
            head = gain.net[-1].weight.grad.flatten(1).norm(dim=1)
            prior_grads = [p.grad.norm() for p in student.prior.parameters() if p.grad is not None]
            if (not torch.isfinite(head).all() or not bool((head > 0).all()) or not prior_grads
                    or not any(bool(torch.isfinite(g) and g > 0) for g in prior_grads)
                    or any(p.grad is not None for p in student.frozen.parameters())):
                raise ValueError('Head and R17 priors must receive finite gradients; R02/R07 remain frozen')
        norm = torch.nn.utils.clip_grad_norm_([*student.prior.parameters(), *gain.parameters()], 1., error_if_nonfinite=True)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite source-only loss')
        optimizer.step()
        schedule.step()
        if step == 1 or step % 50 == 0:
            print('WISIG_R17_JOINT_FIT', step, float(loss.detach()), float(norm), flush=True)
    torch.cuda.synchronize()
    proof = dict(status='SOURCE_FROZEN', dataset='wisig', seed=SEED, updates=STEPS, batch=BATCH,
                 U_exposures=STEPS * BATCH, U_pool=len(ids), F_reads=0, TX_reads=0, V_reads=0,
                 target_reads=0, R02_R07_trainable=False, R17_priors_trainable=True,
                 gain_parameters=sum(p.numel() for p in gain.parameters()),
                 R17_parameters=sum(p.numel() for p in student.parameters()),
                 seconds=monotonic()-started, peak_cuda_bytes=torch.cuda.max_memory_allocated())
    root = Path(rt['run_root'])
    with (root / 'joint.pt').open('xb') as stream:
        torch.save(dict(gain=gain.state_dict(), r17=student.state_dict(), proof=proof), stream)
    write(root / 'fit.json', proof)


@torch.no_grad()
def predict(rt):
    project, root = Path(rt['project']), Path(rt['run_root'])
    source, source_ids, teacher, real_gain, first = setup(rt)
    original = importlib.import_module('experiments.pamf_unrolled_restore.wisig_anchored_window_source')
    reference = project / 'releases' / '20261125-phase1-wisig-native-fir-source-m4-r01-release'
    config = read(reference / 'runtime.json')
    reader = original.tri.reader_factory(config).eval().requires_grad_(False)
    checkpoint = torch.load(root / 'joint.pt', map_location=DEVICE, weights_only=True)
    proof = read(root / 'fit.json')
    if (checkpoint['proof'] != proof or (proof['status'], proof['seed'], proof['updates']) !=
            ('SOURCE_FROZEN', SEED, STEPS) or any(proof[k] for k in ('TX_reads', 'V_reads', 'target_reads', 'F_reads'))):
        raise ValueError('Expected the fixed source-only joint R17 candidate')
    student = original_student(project).eval().requires_grad_(False)
    student.load_state_dict(checkpoint['r17'], strict=True)
    gain = R17Gain(32).to(DEVICE).eval().requires_grad_(False)
    gain.load_state_dict(checkpoint['gain'], strict=True)
    prior_root = project / 'runs' / PREVIOUS
    previous = torch.load(prior_root / 'gain.pt', map_location=DEVICE, weights_only=True)
    evidence = read(prior_root / 'fit.json')
    if previous['proof'] != evidence or evidence['status'] != 'SOURCE_FROZEN' or any(
            evidence[k] for k in ('F_reads', 'TX_reads', 'V_reads', 'target_reads')):
        raise ValueError('Previous F-free source prior is required')
    second = R17Gain(32).to(DEVICE).eval().requires_grad_(False)
    second.load_state_dict(previous['weights'], strict=True)

    def classify(z):
        return np.asarray(original.classify(reader, z).argmax(1).cpu().tolist(), dtype=np.int16)

    def decisions(batch):
        y = as_complex(batch)
        scale = packet_scale(y)
        n = y / scale
        original_p, action = teacher(n)
        r = real_gain(n)
        p, _, student_action, _ = student.training_outputs(pair(n))
        p = complex_iq(p)
        values = (y, teacher(y)[0], r * scale, first(n, original_p, action) * scale,
                  second(n, original_p, r, action) * scale, gain(n, p, r, student_action) * scale)
        return np.stack([classify(z) for z in values])

    if decisions(source[source_ids[:2]]).shape != (len(MODES), 2):
        raise ValueError('No-query original and trained checkpoint reader smoke failed')
    cache, _, _, _, ids, rx, _ = original.core.baseline.handles(config)
    selected = np.flatnonzero(rx == RX)
    if len(selected) != 360:
        raise ValueError('Expected all 360 source-V samples for fixed RX4')
    values = {}
    for view in VIEWS:
        signal = np.load(cache / 'validation_full' / f'{view}.npy', mmap_mode='r', allow_pickle=False)
        values[view] = np.concatenate([decisions(signal[selected[i:i+48]])
                                       for i in range(0, len(selected), 48)], axis=1)
        print('WISIG_R17_JOINT_SCREEN', view, len(selected), flush=True)
    with (root / 'screen.npz').open('xb') as stream:
        np.savez(stream, ids=ids[selected], **values)
    write(root / 'screen-frozen.json', dict(status='PREDICTIONS_FROZEN', modes=list(MODES), views=list(VIEWS),
          count=len(selected), source_RX=RX, prior_screen_RX=[1, 3], source_truth_reads=0, target_reads=0))


def score(rt):
    from experiments.cric_dura_comparison.bootstrap import activate
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Independent CPU scorer required')
    root, project = Path(rt['run_root']), Path(rt['project'])
    frozen = read(root / 'screen-frozen.json')
    if (frozen['status'], frozen['modes'], frozen['views'], frozen['count'], frozen['source_truth_reads']) != (
            'PREDICTIONS_FROZEN', list(MODES), list(VIEWS), 360, 0):
        raise ValueError('Complete source predictions must precede source truth')
    with np.load(root / 'screen.npz', allow_pickle=False) as archive:
        ids = archive['ids'].copy()
        predictions = {view: archive[view].copy() for view in VIEWS}
    if len(ids) != 360 or any(value.shape != (len(MODES), 360) for value in predictions.values()):
        raise ValueError('Incomplete source screen')
    activate(rt)
    reference = project / 'releases' / '20261125-phase1-wisig-native-fir-source-m4-r01-release'
    activate_wisig_archive(reference)
    original = importlib.import_module('experiments.pamf_unrolled_restore.wisig_anchored_window_source')
    _, _, source, _, all_ids, _, positions = original.core.baseline.handles(read(reference / 'runtime.json'))
    labels = np.load(source / 'V' / 'labels.npy', allow_pickle=False)[positions]
    truth = dict(zip(all_ids.tolist(), labels.tolist(), strict=True))
    labels = np.asarray([truth[x] for x in ids], dtype=np.int64)
    permutation = np.random.default_rng(SEED + 3).permutation(len(ids))
    hard, clean = predictions[VIEWS[-1]], predictions['clean']
    pairs = {name: paired(hard[-1], hard[i], labels, permutation, 2) for i, name in enumerate(MODES[:-1])}
    clean_net = {name: int(((clean[-1] == labels).astype(np.int16) - (clean[i] == labels).astype(np.int16)).sum())
                 for i, name in enumerate(MODES[:-1])}
    promote = all(v['passes'] for v in pairs.values()) and min(clean_net.values()) >= -2
    write(root / 'screen-score.json', dict(status='SOURCE_SCREEN_SCORED', truth_last=True, count=len(ids),
          modes=list(MODES), correct={v: (x == labels).sum(1).tolist() for v, x in predictions.items()},
          low_urban_pairs=pairs, clean_net=clean_net, promote=promote,
          missing_controls=['DURA v1 comparison deferred to formal evaluation'], formal_acceptance_complete=False,
          exposure='RX4 source V disjoint from earlier RX1/RX3 screens; source V historically exposed.'))


def dispatch(runtime):
    rt = read(runtime)
    root = Path(rt['run_root'])
    root.mkdir(exist_ok=False)
    state = root / 'state.json'
    try:
        devices = free_gpus()
        if not devices:
            raise RuntimeError('No GPU with 8 GiB free')
        gpu = next(value for value in (7, 2, 3, 6, 1, 4, 0, 5) if value in devices)
        for action in ('fit', 'predict', 'score'):
            device = None if action == 'score' else gpu
            env = dict(os.environ, CUDA_VISIBLE_DEVICES='' if device is None else str(device),
                       OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='2')
            with (root / f'{action}.log').open('x', encoding='utf-8') as log:
                proc = subprocess.Popen([sys.executable, '-u', '-m', __spec__.name, action, str(runtime)],
                                        env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            write(state, dict(status=action.upper(), pid=proc.pid, gpu=device), exclusive=False)
            print('WISIG_R17_JOINT_WORKER', action, proc.pid, device, flush=True)
            if proc.wait():
                raise RuntimeError(f'{action} failed; preserve partial artifacts')
        write(state, dict(status='SOURCE_SCREEN_SCORED', gpu=gpu, promote=read(root / 'screen-score.json')['promote']),
              exclusive=False)
    except Exception as error:
        write(state, dict(status='FAILED', error=str(error)), exclusive=False)
        raise


def publish(repo, output, context, ssh_config):
    from experiments.pamf_lora_main.publish import release, release_commit
    source = read(context)
    rt = {key: source[key] for key in ('project', 'python', 'frozen_sources')}
    project = PurePosixPath(rt['project'])
    runs, releases = project / 'runs', project / 'releases'
    rt.update(run_id=RUN, runner=__spec__.name, code_commit=release_commit(repo),
              release_root=str(releases / f'{RUN}-release'), run_root=str(runs / RUN),
              external_releases=[str(runs / PREVIOUS),
                 str(runs / '20261128-phase1-dura-realgain-crossdata-s2026112845-r02'),
                 str(releases / '20261125-phase1-wisig-native-fir-source-m4-r01-release')])
    release(repo, output, rt, ssh_config, archive_extras=('tools/r17_activation_factorization.py',))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='action', required=True)
    for action in ('dispatch', 'fit', 'predict', 'score'):
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
        {'fit': fit, 'predict': predict, 'score': score}[action](read(args['runtime']))
