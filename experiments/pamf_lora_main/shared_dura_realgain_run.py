"""Train the same frozen-R17/real-gain method on two lawful source contracts."""
import argparse
import importlib
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
from time import monotonic, sleep

import numpy as np
import torch

from experiments.cric_dura_comparison.io import read, write
from experiments.pamf_lora_main.dura_real_gain_fit import (
    DEVICE, LORA_ACTION, LORA_GAIN, WISIG_CACHE, WISIG_PACKET, WISIG_PARENT,
    activate_wisig_archive, as_complex, lora_models, wisig_models,
)
from experiments.pamf_lora_main.dura_real_gain_run import free_gpus
from experiments.pamf_lora_main.shared_dura_realgain import SharedDuraRealGain
from experiments.pamf_restoration.models import pair

RUN = '20261128-phase1-shared-r17-realgain-s2026112893-r01'
PARENT = '20261128-phase1-dura-realgain-crossdata-s2026112845-r02'
REFERENCE = '20261125-phase1-wisig-native-fir-source-m4-r01-release'
SEED, STEPS = 2026112893, 2000
BATCH = {'lora': 4, 'wisig': 24}
WIDTH = {'lora': 1024, 'wisig': 32}
VIEWS = ('clean', 'practical_low_urban')


def restore_scaled(base, model, received):
    """Use the same per-packet normalization for source training and inference."""
    scale = received.abs().square().mean(-1, keepdim=True).clamp_min(1e-12).sqrt()
    normalized = received / scale
    with torch.no_grad():
        prior, action = base(normalized)
    return model(normalized, prior, action) * scale, prior * scale


def setup(rt, dataset):
    from experiments.cric_dura_comparison.bootstrap import activate, numerical_policy
    numerical_policy()
    activate(rt)
    project = Path(rt['project'])
    if dataset == 'wisig':
        return wisig_models(project, rt['frozen_sources'])
    release = project / 'releases' / f'{LORA_GAIN}-release'
    activate_wisig_archive(release)
    activate_wisig_archive(Path(read(release / 'runtime.json')['w24_release_root']))
    source, ids, _ = lora_models(project)
    from experiments.pamf_lora_main import w24_joint_pamf as joint
    from experiments.pamf_lora_main import w24_native_training as native
    old = read(project / 'runs' / LORA_GAIN / 'frequency_inphase_native' / 'fit-inputs.json')
    contract = joint.source_contract(Path(old['source_root']))
    r17 = native.dura_model(project / 'runs' / LORA_ACTION, contract, DEVICE).eval().requires_grad_(False)

    def base(y):
        _, action, _ = r17.estimate(pair(y))
        return r17(y), action

    return source, ids, base


def fit(rt, dataset):
    from experiments.pamf_lora_main.w24_native_dura import source_pair
    source, ids, base = setup(rt, dataset)
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    generator = torch.Generator(device=DEVICE).manual_seed(SEED + 1)
    model = SharedDuraRealGain(WIDTH[dataset]).to(DEVICE).train()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS, eta_min=3e-5)
    started = monotonic()
    for step in range(1, STEPS + 1):
        raw = as_complex(source[rng.choice(ids, BATCH[dataset], replace=False)])
        clean, observed, _, _ = source_pair(raw, generator)
        restored, prior = restore_scaled(base, model, observed)
        preserved, clean_prior = restore_scaled(base, model, clean)
        if step == 1:
            torch.testing.assert_close(restored, prior, rtol=0, atol=0)
        error = restored - clean
        spectrum = torch.fft.fft(clean, norm='ortho').abs().square()
        spectral = (torch.fft.fft(error, norm='ortho').abs().square() / (.03 + spectrum)).log1p().mean()
        loss = error.abs().square().mean() + .3 * spectral + 1.5 * (preserved-clean_prior).abs().square().mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if step == 1 and (model.net[-1].weight.grad is None or
                          not torch.isfinite(model.net[-1].weight.grad).all() or
                          not bool(model.net[-1].weight.grad.norm() > 0)):
            raise ValueError('Real-checkpoint source smoke needs finite nonzero gain gradient')
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite source-only objective')
        opt.step()
        schedule.step()
        if step == 1 or step % 50 == 0:
            print('SHARED_GAIN_FIT', dataset, step, float(loss.detach()), float(norm), flush=True)
    torch.cuda.synchronize()
    proof = dict(status='SOURCE_FROZEN', dataset=dataset, seed=SEED, updates=STEPS,
                 batch=BATCH[dataset], U_pool=len(ids), U_exposures=STEPS*BATCH[dataset],
                 F_reads=0, TX_reads=0, V_reads=0, target_reads=0,
                 parameters=sum(p.numel() for p in model.parameters()),
                 seconds=monotonic()-started, peak_cuda_bytes=torch.cuda.max_memory_allocated())
    folder = Path(rt['run_root']) / dataset
    with (folder / 'gain.pt').open('xb') as stream:
        torch.save(dict(weights=model.state_dict(), proof=proof), stream)
    write(folder / 'fit.json', proof)


@torch.no_grad()
def predict(rt, dataset):
    from experiments.pamf_lora_main import dura_real_gain_screen as screen
    root = Path(rt['run_root'])
    _, _, base = setup(rt, dataset)
    proof = read(root / dataset / 'fit.json')
    checkpoint = torch.load(root / dataset / 'gain.pt', map_location=DEVICE, weights_only=True)
    if (checkpoint['proof'] != proof or (proof['dataset'], proof['seed'], proof['updates']) !=
            (dataset, SEED, STEPS) or any(proof[k] for k in ('F_reads', 'TX_reads', 'V_reads', 'target_reads'))):
        raise ValueError('Expected fixed source-only shared gain checkpoint')
    model = SharedDuraRealGain(WIDTH[dataset]).to(DEVICE).eval().requires_grad_(False)
    model.load_state_dict(checkpoint['weights'], strict=True)
    old_load = screen.load_gain
    old_base = screen.lora_models

    def replacement(folder, row, *, real=False):
        return old_load(Path(rt['project']) / 'runs' / PARENT, row, real=True) if real else model

    screen.load_gain = replacement
    if dataset == 'lora':
        screen.lora_models = lambda project: (None, None, base)
    try:
        ids, values = (screen.predict_lora if dataset == 'lora' else screen.predict_wisig)(rt, root)
    finally:
        screen.load_gain, screen.lora_models = old_load, old_base
    if dataset == 'wisig':
        old = Path(rt['project']) / 'runs' / WISIG_PACKET / 'wisig'
        frozen = read(old / 'frozen.json')
        if frozen['status'] != 'PREDICTIONS_FROZEN' or frozen['source_truth_reads'] or frozen['target_reads']:
            raise ValueError('Historical DURA source predictions must already be frozen')
        with np.load(old / 'predictions.npz', allow_pickle=False) as archive:
            old_ids = archive['ids'].copy()
            if len(set(old_ids.tolist())) != len(old_ids):
                raise ValueError('Duplicate historical DURA physical IDs')
            lookup = {item: i for i, item in enumerate(old_ids.tolist())}
            indices = [lookup[item] for item in ids.tolist()]
            for view in VIEWS:
                history = archive[view]
                if not np.array_equal(history[0, indices], values[view][0]):
                    raise ValueError('Historical R69 and current input predictions differ')
                values[view] = np.concatenate((values[view][:-1], history[1, indices][None],
                                               values[view][-1:]), axis=0)
    modes = list(screen.MODES[dataset])
    if dataset == 'wisig':
        modes.insert(-1, 'dura_v1')
    folder = root / dataset
    with (folder / 'screen.npz').open('xb') as stream:
        np.savez(stream, ids=ids, **values)
    write(folder / 'screen-frozen.json', dict(status='PREDICTIONS_FROZEN', views=list(VIEWS), modes=modes,
          count=len(ids), RX=0 if dataset == 'lora' else 1, source_truth_reads=0, target_reads=0))


def score(rt, dataset):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Independent CPU scorer required')
    from experiments.pamf_lora_main.dura_real_gain_fit import activate_wisig_archive
    from experiments.pamf_lora_main.dura_real_gain_screen import paired
    root, project = Path(rt['run_root']) / dataset, Path(rt['project'])
    frozen = read(root / 'screen-frozen.json')
    count = 60 if dataset == 'lora' else 360
    modes = (['direct', 'lora_native_r17', 'real_gain', 'fusion'] if dataset == 'lora' else
             ['direct', 'wisig_r17', 'real_gain', 'dura_v1', 'fusion'])
    if (frozen['status'], frozen['views'], frozen['modes'], frozen['count'],
            frozen['source_truth_reads'], frozen['target_reads']) != (
            'PREDICTIONS_FROZEN', list(VIEWS), modes, count, 0, 0):
        raise ValueError('All same-packet views and controls must be frozen before truth')
    with np.load(root / 'screen.npz', allow_pickle=False) as archive:
        ids = archive['ids'].copy()
        outputs = {view: archive[view].copy() for view in VIEWS}
    if len(set(ids.tolist())) != count or any(x.shape != (len(modes), count) for x in outputs.values()):
        raise ValueError('Incomplete source prediction matrix')
    if dataset == 'lora':
        old = read(project / 'runs' / LORA_GAIN / 'frequency_inphase_native' / 'fit-inputs.json')
        with np.load(Path(old['source_root']) / 'V_metadata.npz', allow_pickle=False) as meta:
            lookup = dict(zip(meta['ids'].tolist(), meta['y'].tolist(), strict=True))
    else:
        from experiments.cric_dura_comparison.bootstrap import activate
        activate(rt)
        ref = project / 'releases' / REFERENCE
        activate_wisig_archive(ref)
        original = importlib.import_module('experiments.pamf_unrolled_restore.wisig_anchored_window_source')
        _, _, source, _, all_ids, _, positions = original.core.baseline.handles(read(ref / 'runtime.json'))
        labels = np.load(source / 'V' / 'labels.npy', allow_pickle=False)[positions]
        lookup = dict(zip(all_ids.tolist(), labels.tolist(), strict=True))
    labels = np.asarray([lookup[x] for x in ids.tolist()], dtype=np.int64)
    permutation = np.random.default_rng(2026112852).permutation(count)
    minimum, clean_limit = (1, 1) if dataset == 'lora' else (2, 2)
    hard, clean = outputs[VIEWS[-1]], outputs['clean']
    comparisons = {name: paired(hard[-1], hard[i], labels, permutation, minimum)
                   for i, name in enumerate(modes[:-1])}
    clean_net = {name: int(((clean[-1] == labels).astype(np.int16) -
                            (clean[i] == labels).astype(np.int16)).sum()) for i, name in enumerate(modes[:-1])}
    missing = ['same-contract LoRa DURA v1'] if dataset == 'lora' else []
    promote = (not missing and all(x['passes'] for x in comparisons.values()) and
               min(clean_net.values()) >= -clean_limit)
    write(root / 'screen-score.json', dict(status='SOURCE_SCREEN_SCORED', truth_last=True, count=count,
          modes=modes, correct={v: (x == labels).sum(1).tolist() for v, x in outputs.items()},
          low_urban_pairs=comparisons, clean_net=clean_net, promote=promote, missing_controls=missing,
          exposure='Previously exposed source V panel; no target read', formal_acceptance_complete=False))


def dispatch(runtime, worker_module=None):
    rt = read(runtime)
    root = Path(rt['run_root'])
    root.mkdir(exist_ok=False)
    for row in WIDTH:
        (root / row).mkdir()
    try:
        gpus = free_gpus()
        if len(gpus) < 2:
            raise RuntimeError('Two distinct GPUs with 8 GiB free required')
        jobs = {}
        for row, gpu in zip(WIDTH, gpus[:2], strict=True):
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='2',
                       OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='2')
            with (root / f'{row}-worker.log').open('x', encoding='utf-8') as log:
                jobs[row] = subprocess.Popen([sys.executable, '-u', '-m', worker_module or __spec__.name, 'worker',
                                               str(runtime), row], env=env, stdin=subprocess.DEVNULL,
                                              stdout=log, stderr=subprocess.STDOUT)
        write(root / 'state.json', dict(status='RUNNING', workers={k: p.pid for k, p in jobs.items()},
                                      gpus=dict(zip(WIDTH, gpus[:2], strict=True))))
        failures = {}
        while jobs:
            for row, process in list(jobs.items()):
                code = process.poll()
                if code is not None:
                    if code:
                        failures[row] = code
                    del jobs[row]
            if jobs:
                sleep(3)
        write(root / 'state.json', dict(status='FAILED' if failures else 'SOURCE_SCREEN_SCORED',
                                      failures=failures), exclusive=False)
    except Exception as error:
        write(root / 'state.json', dict(status='FAILED', error=str(error)), exclusive=False)
        raise


def worker(runtime, dataset):
    rt = read(runtime)
    fit(rt, dataset)
    predict(rt, dataset)
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
              lora_frozen_root=str(runs / PARENT),
              external_releases=[str(runs / n) for n in (LORA_ACTION, LORA_GAIN, WISIG_CACHE,
                                                          WISIG_PACKET, WISIG_PARENT, PARENT)] +
                                [str(releases / REFERENCE)])
    release(repo, output, rt, ssh_config, archive_extras=('tools/r17_activation_factorization.py',))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='action', required=True)
    for name in ('dispatch', 'worker', 'score'):
        child = commands.add_parser(name)
        child.add_argument('runtime', type=Path)
        if name != 'dispatch':
            child.add_argument('dataset', choices=tuple(WIDTH))
    pub = commands.add_parser('publish')
    for field in ('repo', 'output', 'context', 'ssh-config'):
        pub.add_argument('--' + field, type=Path, required=True)
    args = vars(parser.parse_args())
    action = args.pop('action')
    {'dispatch': dispatch, 'worker': worker, 'score': lambda runtime, dataset: score(read(runtime), dataset),
     'publish': publish}[action](**args)
