"""WiSig R17 plus an identity-regularized, real spectral innovation gain."""

import argparse
from pathlib import Path, PurePosixPath
import os
import subprocess
import sys
from time import monotonic

import numpy as np
import torch

from experiments.cric_dura_comparison.io import read, write
from experiments.pamf_lora_main.dura_real_gain_fit import DEVICE, as_complex, wisig_models, wisig_real_prior
from experiments.pamf_lora_main.dura_real_gain_fusion import RealSpectralGain
from experiments.pamf_lora_main.dura_real_gain_run import free_gpus
from experiments.pamf_lora_main.dura_real_gain_screen import load_gain, packet_scale, paired


RUN = '20261128-phase1-wisig-r17-spectral-gain-s2026112877-r01'
PARENT = '20261128-phase1-dura-realgain-crossdata-s2026112845-r02'
SEED, STEPS, BATCH, RX = 2026112877, 2000, 24, 6
VIEWS = ('clean', 'practical_low_urban')
MODES = ('direct', 'wisig_r17', 'real_gain', 'prior_fusion', 'fusion')


def spectral_fusion(model, observed, prior):
    return prior + model(observed.unflatten(-1, (8, 32)), prior.unflatten(-1, (8, 32))).flatten(-2)


def setup(rt):
    from experiments.cric_dura_comparison.bootstrap import numerical_policy
    numerical_policy()
    project = Path(rt['project'])
    source, ids, r17 = wisig_models(project, rt['frozen_sources'])
    prior = wisig_real_prior(project)
    control = load_gain(project / 'runs' / PARENT, 'wisig', real=True)
    old_fusion = load_gain(project / 'runs' / PARENT, 'wisig')

    def real_gain(y):
        observed, estimate, scale = prior(y)
        return ((estimate + control(observed, estimate)) * scale).flatten(-2)

    return source, ids, r17, real_gain, old_fusion


def fit(rt):
    from experiments.pamf_lora_main.w24_native_dura import source_pair
    source, ids, r17, _, _ = setup(rt)
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    generator = torch.Generator(device=DEVICE).manual_seed(SEED + 1)
    model = RealSpectralGain(32).to(DEVICE).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, STEPS, eta_min=3e-5)
    started = monotonic()
    for step in range(1, STEPS + 1):
        clean, observed, _, _ = source_pair(as_complex(source[rng.choice(ids, BATCH, replace=False)]), generator)
        with torch.no_grad():
            p, _ = r17(observed)
            pc, _ = r17(clean)
        output = spectral_fusion(model, observed, p)
        preserved = spectral_fusion(model, clean, pc)
        if step == 1:
            torch.testing.assert_close(output, p, rtol=0, atol=0)
        error = output - clean
        target = torch.fft.fft(clean, norm='ortho').abs().square()
        spectral = (torch.fft.fft(error, norm='ortho').abs().square() / (.03 + target)).log1p().mean()
        keep = (preserved - pc).abs().square().mean()
        retention = (output - p).abs().square().mean()
        loss = error.abs().square().mean() + .3 * spectral + 1.5 * keep + 2. * retention
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradients = model.net[-1].weight.grad
        if step == 1 and (not torch.isfinite(gradients).all() or not bool(gradients.norm() > 0)):
            raise ValueError('Real spectral innovation requires finite nonzero gradient')
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite source waveform loss')
        optimizer.step()
        schedule.step()
        if step == 1 or step % 50 == 0:
            print('WISIG_R17_SPECTRAL_FIT', step, float(loss.detach()), float(norm), flush=True)
    torch.cuda.synchronize()
    proof = dict(status='SOURCE_FROZEN', dataset='wisig', seed=SEED, updates=STEPS,
                 batch=BATCH, U_exposures=STEPS * BATCH, U_pool=len(ids), F_reads=0,
                 TX_reads=0, V_reads=0, target_reads=0, parameters=sum(p.numel() for p in model.parameters()),
                 seconds=monotonic()-started, peak_cuda_bytes=torch.cuda.max_memory_allocated())
    root = Path(rt['run_root'])
    with (root / 'gain.pt').open('xb') as stream:
        torch.save(dict(weights=model.state_dict(), proof=proof), stream)
    write(root / 'fit.json', proof)


@torch.no_grad()
def predict(rt):
    import importlib
    project, root = Path(rt['project']), Path(rt['run_root'])
    source, source_ids, r17, real_gain, old_fusion = setup(rt)
    original = importlib.import_module('experiments.pamf_unrolled_restore.wisig_anchored_window_source')
    reference = project / 'releases' / '20261125-phase1-wisig-native-fir-source-m4-r01-release'
    config = read(reference / 'runtime.json')
    reader = original.tri.reader_factory(config).eval().requires_grad_(False)
    checkpoint = torch.load(root / 'gain.pt', map_location=DEVICE, weights_only=True)
    proof = read(root / 'fit.json')
    if (checkpoint['proof'] != proof or (proof['status'], proof['seed'], proof['updates']) !=
            ('SOURCE_FROZEN', SEED, STEPS) or any(proof[k] for k in ('TX_reads', 'V_reads', 'target_reads'))):
        raise ValueError('Expected the lawful frozen source-only candidate')
    model = RealSpectralGain(32).to(DEVICE).eval().requires_grad_(False)
    model.load_state_dict(checkpoint['weights'], strict=True)

    def classify(z):
        return np.asarray(original.classify(reader, z).argmax(1).cpu().tolist(), dtype=np.int16)

    def decisions(batch):
        y = as_complex(batch)
        scale = packet_scale(y)
        n = y / scale
        p, action = r17(n)
        canonical = r17(y)[0]
        return np.stack([classify(z) for z in (y, canonical, real_gain(n) * scale,
                         old_fusion(n, p, action) * scale, spectral_fusion(model, n, p) * scale)])

    if decisions(source[source_ids[:2]]).shape != (len(MODES), 2):
        raise ValueError('Real checkpoint and original reader source-only smoke failed')
    cache, _, _, _, ids, rx, _ = original.core.baseline.handles(config)
    chosen = np.flatnonzero(rx == RX)
    if len(chosen) != 360:
        raise ValueError('Expected all 360 source-V samples of RX 6')
    results = {}
    for view in VIEWS:
        signal = np.load(cache / 'validation_full' / f'{view}.npy', mmap_mode='r', allow_pickle=False)
        results[view] = np.concatenate([decisions(signal[chosen[i:i+48]]) for i in range(0, len(chosen), 48)], axis=1)
        print('WISIG_R17_SPECTRAL_SCREEN', view, len(chosen), flush=True)
    with (root / 'screen.npz').open('xb') as stream:
        np.savez(stream, ids=ids[chosen], **results)
    write(root / 'screen-frozen.json', dict(status='PREDICTIONS_FROZEN', modes=list(MODES), views=list(VIEWS),
          count=len(chosen), source_RX=RX, previous_screen_RX=[1, 3, 4], source_truth_reads=0, target_reads=0))


def score(rt):
    import importlib
    from experiments.cric_dura_comparison.bootstrap import activate
    from experiments.pamf_lora_main.dura_real_gain_fit import activate_wisig_archive
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Independent CPU scorer required')
    root, project = Path(rt['run_root']), Path(rt['project'])
    frozen = read(root / 'screen-frozen.json')
    if (frozen['status'], frozen['modes'], frozen['views'], frozen['count'], frozen['source_truth_reads']) != (
            'PREDICTIONS_FROZEN', list(MODES), list(VIEWS), 360, 0):
        raise ValueError('Two-view predictions must freeze before source truth')
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
    lookup = dict(zip(all_ids.tolist(), labels.tolist(), strict=True))
    truth = np.asarray([lookup[x] for x in ids], dtype=np.int64)
    permutation = np.random.default_rng(SEED + 3).permutation(len(ids))
    hard, clean = predictions[VIEWS[-1]], predictions['clean']
    pairs = {name: paired(hard[-1], hard[i], truth, permutation, 2) for i, name in enumerate(MODES[:-1])}
    clean_net = {name: int(((clean[-1] == truth).astype(np.int16) - (clean[i] == truth).astype(np.int16)).sum())
                 for i, name in enumerate(MODES[:-1])}
    promote = all(value['passes'] for value in pairs.values()) and min(clean_net.values()) >= -2
    write(root / 'screen-score.json', dict(status='SOURCE_SCREEN_SCORED', truth_last=True, count=len(ids),
          modes=list(MODES), correct={view: (values == truth).sum(1).tolist() for view, values in predictions.items()},
          low_urban_pairs=pairs, clean_net=clean_net, promote=promote,
          missing_controls=['DURA v1 comparison deferred to formal evaluation'], formal_acceptance_complete=False,
          exposure='RX6 source V is disjoint from previous RX1/RX3/RX4 screens; source V has historical exposure.'))


def dispatch(runtime):
    rt = read(runtime)
    root = Path(rt['run_root'])
    root.mkdir(exist_ok=False)
    state = root / 'state.json'
    try:
        available = free_gpus()
        if not available:
            raise RuntimeError('No GPU with 8 GiB available')
        gpu = next(value for value in (7, 2, 3, 6, 1, 4, 0, 5) if value in available)
        for action in ('fit', 'predict', 'score'):
            device = None if action == 'score' else gpu
            env = dict(os.environ, CUDA_VISIBLE_DEVICES='' if device is None else str(device),
                       OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='2')
            args = [sys.executable, '-u', '-m', __spec__.name, action, str(runtime)]
            with (root / f'{action}.log').open('x', encoding='utf-8') as log:
                proc = subprocess.Popen(args, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            write(state, dict(status=action.upper(), pid=proc.pid, gpu=device), exclusive=False)
            print('WISIG_R17_SPECTRAL_WORKER', action, proc.pid, device, flush=True)
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
              external_releases=[str(runs / PARENT),
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
