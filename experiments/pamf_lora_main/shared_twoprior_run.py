"""Shared two-prior real-gain fusion with independent source-trained weights."""
import argparse
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
from time import monotonic

import numpy as np
import torch

from experiments.cric_dura_comparison.io import read, write
from experiments.pamf_lora_main.dura_real_gain_fit import (
    DEVICE, LORA_ACTION, LORA_GAIN, WISIG_CACHE, WISIG_PACKET, WISIG_PARENT,
    as_complex, lora_models, wisig_real_prior,
)
from experiments.pamf_lora_main.dura_real_gain_screen import load_gain
from experiments.pamf_lora_main.shared_dura_realgain import SharedTwoPriorGain
from experiments.pamf_lora_main.shared_dura_realgain_run import REFERENCE, VIEWS, dispatch, score, setup

RUN = '20261128-phase1-shared-twoprior-realgain-s2026112901-r01'
PARENT = '20261128-phase1-dura-realgain-crossdata-s2026112845-r02'
SEED, STEPS = 2026112901, 2000
BATCH = {'lora': 4, 'wisig': 24}
WIDTH = {'lora': 1024, 'wisig': 32}


def experts(rt, dataset):
    source, ids, r17 = setup(rt, dataset)
    project = Path(rt['project'])
    if dataset == 'lora':
        _, _, real = lora_models(project)

        def gain(y):
            return real(y)[0]
    else:
        packet = wisig_real_prior(project)
        control = load_gain(project / 'runs' / PARENT, 'wisig', real=True)

        def gain(y):
            observed, prior, scale = packet(y)
            return ((prior + control(observed, prior)) * scale).flatten(-2)
    return source, ids, r17, gain


def restore_scaled(r17, gain, model, received):
    scale = received.abs().square().mean(-1, keepdim=True).clamp_min(1e-12).sqrt()
    normalized = received / scale
    with torch.no_grad():
        prior = gain(normalized)
        physical, action = r17(normalized)
    return model(normalized, prior, physical, action) * scale, prior * scale


def fit(rt, dataset):
    from experiments.pamf_lora_main.w24_native_dura import source_pair
    source, ids, r17, gain = experts(rt, dataset)
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    generator = torch.Generator(device=DEVICE).manual_seed(SEED + 1)
    model = SharedTwoPriorGain(WIDTH[dataset]).to(DEVICE).train()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS, eta_min=3e-5)
    started = monotonic()
    for step in range(1, STEPS + 1):
        raw = as_complex(source[rng.choice(ids, BATCH[dataset], replace=False)])
        clean, observed, _, _ = source_pair(raw, generator)
        output, prior = restore_scaled(r17, gain, model, observed)
        preserved, clean_prior = restore_scaled(r17, gain, model, clean)
        if step == 1:
            torch.testing.assert_close(output, prior, rtol=0, atol=0)
        error = output - clean
        spectrum = torch.fft.fft(clean, norm='ortho').abs().square()
        spectral = (torch.fft.fft(error, norm='ortho').abs().square() / (.03+spectrum)).log1p().mean()
        loss = error.abs().square().mean() + .3*spectral + 1.5*(preserved-clean_prior).abs().square().mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if step == 1 and (model.net[-1].weight.grad is None or
                          not torch.isfinite(model.net[-1].weight.grad).all() or
                          not bool(model.net[-1].weight.grad.norm() > 0)):
            raise ValueError('Real-checkpoint source smoke requires finite nonzero gradients')
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite shared source objective')
        opt.step()
        schedule.step()
        if step == 1 or step % 50 == 0:
            print('SHARED_TWOPRIOR_FIT', dataset, step, float(loss.detach()), float(norm), flush=True)
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
    root, project = Path(rt['run_root']), Path(rt['project'])
    _, _, r17, real = experts(rt, dataset)
    proof = read(root / dataset / 'fit.json')
    checkpoint = torch.load(root / dataset / 'gain.pt', map_location=DEVICE, weights_only=True)
    if (checkpoint['proof'] != proof or (proof['dataset'], proof['seed'], proof['updates']) !=
            (dataset, SEED, STEPS) or any(proof[k] for k in ('F_reads', 'TX_reads', 'V_reads', 'target_reads'))):
        raise ValueError('Expected the fixed source-only shared gain checkpoint')
    model = SharedTwoPriorGain(WIDTH[dataset]).to(DEVICE).eval().requires_grad_(False)
    model.load_state_dict(checkpoint['weights'], strict=True)
    old_load = screen.load_gain

    def candidate(y, base, action):
        if dataset == 'lora':
            physical, _ = r17(y)
            return model(y, base, physical, action)
        return model(y, real(y), base, action)

    def replacement(folder, row, *, real=False):
        return old_load(project / 'runs' / PARENT, row, real=True) if real else candidate

    screen.load_gain = replacement
    try:
        ids, values = (screen.predict_lora if dataset == 'lora' else screen.predict_wisig)(rt, root)
    finally:
        screen.load_gain = old_load
    if dataset == 'wisig':
        old = project / 'runs' / WISIG_PACKET / 'wisig'
        mark = read(old / 'frozen.json')
        if mark['status'] != 'PREDICTIONS_FROZEN' or mark['source_truth_reads'] or mark['target_reads']:
            raise ValueError('Historical DURA predictions must already be frozen')
        with np.load(old / 'predictions.npz', allow_pickle=False) as archive:
            old_ids = archive['ids'].copy()
            if len(set(old_ids.tolist())) != len(old_ids):
                raise ValueError('Duplicate historical physical IDs')
            lookup = {item: i for i, item in enumerate(old_ids.tolist())}
            indices = [lookup[item] for item in ids.tolist()]
            for view in VIEWS:
                history = archive[view]
                if not np.array_equal(history[0, indices], values[view][0]):
                    raise ValueError('Historical R69 and input predictions differ')
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
    if action == 'dispatch':
        dispatch(args['runtime'], worker_module=__spec__.name)
    elif action == 'score':
        score(read(args['runtime']), args['dataset'])
    elif action == 'worker':
        worker(**args)
    else:
        publish(**args)
