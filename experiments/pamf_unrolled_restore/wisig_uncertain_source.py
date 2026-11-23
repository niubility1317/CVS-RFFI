"""Matched fresh WiSig restorers from TX-hidden source waveforms only."""

import argparse
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
from time import perf_counter

import numpy as np
import torch

from experiments.cric_dura_comparison.io import read, tensor, write
from experiments.pamf_lora_main.publish import release, release_commit
from experiments.pamf_lora_main.runner import activate

from . import wisig_short_training_source as baseline
from .wisig_uncertain_prior import UncertaintyPrior, as_complex, as_iq, fit_loss, received_features


RUN = '20261123-phase1-wisig-uncertain-prior-source-s2026092701-m2-r01'
RECIPE = {'waveform': 0., 'weak_detail': .20}
SEED = 2026092701
BATCH = 64
STEPS = 1200
CAL_COUNT = 4096


def modes_for(recipe):
    return ('direct', 'r69', *(f'{arm}_{mode}' for arm in recipe for mode in ('coarse', 'joint', 'gated')))


def source_iq(cache):
    if cache.parent.name != baseline.CACHE_RUN:
        raise ValueError('Only the fixed source cache may be passed to fit')
    contract = read(cache / 'complete.json')
    if contract['train_count'] != 63000 or contract['U_count'] != 56700 or contract['TX_labels_read'] or contract['target_reads']:
        raise ValueError('Fixed TX-blind source cache required')
    root = cache / 'train'
    x = np.load(root / 'clean.npy', mmap_mode='r', allow_pickle=False)
    roles = np.load(root / 'roles.npy', allow_pickle=False)
    if x.shape != (63000, 2, 256) or x.dtype != np.float32 or roles.shape != (len(x),):
        raise ValueError('Source training IQ or role shape differs')
    indices = np.random.default_rng(SEED).permutation(np.flatnonzero(roles == 'U_s'))
    if len(indices) != 56700:
        raise ValueError('Unexpected hidden-label U count')
    return x, indices[CAL_COUNT:], indices[:CAL_COUNT]


def proof(arm, recipe):
    return dict(status='SOURCE_FROZEN', steps=STEPS, seed=SEED, weak_weight=recipe[arm],
                target_reads=0, TX_truth_reads=0, F_reads=0, V_reads=0)


def fit(rt, arm, *, smoke=False, recipe=RECIPE, observe=None):
    if set(rt) != {'run_root', 'source_root'} or arm not in recipe:
        raise ValueError('Fit accepts only a source root, output root and registered recipe')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    source, fit_ids, calibration_ids = source_iq(Path(rt['source_root']))
    device = 'cuda:0'
    torch.manual_seed(SEED)
    model = UncertaintyPrior().to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    updates = 1 if smoke else STEPS
    lr = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=STEPS, eta_min=3e-5)
    generator = torch.Generator(device=device).manual_seed(SEED + 1)
    rng = np.random.default_rng(SEED + 2)
    started = perf_counter()
    for step in range(1, updates + 1):
        indices = rng.choice(fit_ids, BATCH, replace=False)
        x = as_complex(tensor(source[indices], device))
        optimizer.zero_grad(set_to_none=True)
        value, logs = fit_loss(model, x, generator, recipe[arm], observe=observe)
        if not torch.isfinite(value):
            raise FloatingPointError('Nonfinite source waveform objective')
        value.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        if step == 1 and (model.spectral[-1].weight.grad.norm() <= 0 or not any(
                p.grad is not None and p.grad.norm() > 0 for p in model.detail.parameters())):
            raise ValueError('Both coarse and fine networks require nonzero source gradient')
        optimizer.step()
        lr.step()
        if step == 1 or step % 50 == 0:
            print('WISIG_UNCERTAIN_FIT', arm, step, value.item(), logs.tolist(), norm.item(), flush=True)
    torch.cuda.synchronize()
    seconds = perf_counter() - started
    root = Path(rt['run_root'])
    if smoke:
        write(root / f'smoke-{arm}.json', dict(status='PASS', real_source_U_batch=BATCH, target_reads=0, F_reads=0))
        return
    evidence = dict(proof(arm, recipe), optimizer_seconds=seconds,
                    params=sum(p.numel() for p in model.parameters()), batch=BATCH)
    with torch.no_grad():
        measures = [received_features(as_complex(tensor(source[chunk], device)))[0]
                    for chunk in np.array_split(calibration_ids, CAL_COUNT // BATCH)]
        evidence['clean_threshold'] = torch.cat(measures).quantile(.999).item()
    with (root / f'{arm}.pt').open('xb') as stream:
        torch.save(dict(state=model.state_dict(), evidence=evidence), stream)
    write(root / f'{arm}-complete.json', evidence)


def load_models(root, recipe):
    models = {}
    for arm in recipe:
        checkpoint = torch.load(root / f'{arm}.pt', map_location='cuda:0', weights_only=True)
        evidence = read(root / f'{arm}-complete.json')
        if (checkpoint['evidence'] != evidence or any(evidence[k] != v for k, v in proof(arm, recipe).items())
                or not 0 <= evidence['clean_threshold'] <= 1):
            raise ValueError('Fixed source-only candidate required')
        model = UncertaintyPrior().to('cuda:0').eval().requires_grad_(False)
        model.load_state_dict(checkpoint['state'], strict=True)
        models[arm] = (model, evidence['clean_threshold'])
    return models


@torch.no_grad()
def evaluate(rt, recipe=RECIPE, prepare=None, *, model_root=None, reader_factory=None):
    cache, _, source, reader, ids, _, _ = baseline.handles(rt)
    f = baseline.load_reader(reader, source, 'cuda:0') if reader_factory is None else reader_factory(rt)
    f.eval().requires_grad_(False)
    classify = f if reader_factory is None else lambda y: f.classify_features(f.features(y))
    root = Path(rt['run_root'])
    old = Path(rt['baseline_root']) / 'predictions'
    done = read(old / 'complete.json')
    if (done['status'], done['count'], done['target_reads'], done['source_truth_reads']) != ('FROZEN', len(ids), 0, 0):
        raise ValueError('Expected fixed same-panel truth-blind direct/R69 evidence')
    models = load_models(root if model_root is None else model_root, recipe)
    modes = modes_for(recipe)
    activity = {}
    seconds = {}
    for route, folder in baseline.ROUTES.items():
        for scene in baseline.VIEWS:
            key = f'{route}_{scene}'
            started = perf_counter()
            with np.load(old / f'{key}.npz', allow_pickle=False) as ref:
                if (not np.array_equal(ref['ids'], ids) or ref['decision'].shape != (len(done['modes']), len(ids))
                        or ref['decision'].dtype != np.uint8 or (ref['decision'] > 5).any()):
                    raise ValueError('Baseline physical IDs or decision matrix differ')
                decisions = ref['decision'][[done['modes'].index(mode) for mode in ('direct', 'r69')]]
            observed = np.load(cache / folder / f'{scene}.npy', mmap_mode='r', allow_pickle=False)
            if observed.shape != (len(ids), 2, 256) or observed.dtype != np.float32:
                raise ValueError('Expected matching source observations')
            activity[key] = {arm: 0 for arm in models}
            rows = []
            for start in range(0, len(ids), BATCH):
                x = as_complex(tensor(observed[start:start + BATCH], 'cuda:0'))
                variants = []
                for arm, (model, threshold) in models.items():
                    y, shifted = (x, None) if prepare is None else prepare(x, arm)
                    coarse, joint, power = model(y)
                    active = power >= threshold
                    if shifted is not None:
                        active = active | shifted
                    activity[key][arm] += int(active.sum())
                    variants.extend((coarse, joint, torch.where(active[:, None], joint, x)))
                logits = torch.stack([classify(as_iq(value)) for value in variants])
                if logits.shape != (len(modes) - 2, len(x), 6) or not torch.isfinite(logits).all():
                    raise ValueError('Incomplete source predictions')
                rows.append(logits.argmax(dim=-1).tolist())
            out = root / 'predictions' / f'{key}.npz'
            out.parent.mkdir(exist_ok=True)
            with out.open('xb') as stream:
                np.savez(stream, ids=ids, decision=np.concatenate((decisions, np.concatenate(rows, axis=1))).astype(np.uint8))
            seconds[key] = perf_counter() - started
            print('WISIG_UNCERTAIN_FROZEN', key, len(ids), flush=True)
    write(root / 'predictions' / 'complete.json', dict(status='FROZEN', modes=list(modes), views=list(baseline.VIEWS),
          count=len(ids), source_truth_reads=0, target_reads=0, gate_activity=activity, readouts_reused=['direct', 'r69'],
          evaluation_seconds=seconds))


def train_worker(rt, action, arm, gpu):
    root = Path(rt['run_root'])
    config = root / 'configs' / f'{action}-{arm}.json'
    write(config, {key: rt[key] for key in ('run_root', 'source_root')})
    path = root / 'logs' / f'{action}-{arm}.log'
    path.parent.mkdir(exist_ok=True)
    cmd = [sys.executable, '-u', '-m', rt['runner'], action, str(config), arm]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1')
    with path.open('x', encoding='utf-8') as log:
        proc = subprocess.Popen(cmd, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
    print('WISIG_TRAIN_LAUNCHED', action, arm, proc.pid, gpu, flush=True)
    return proc


def dispatch(rt, recipe=RECIPE, timing=None):
    root = Path(rt['run_root'])
    root.mkdir(exist_ok=False)
    status = root / 'state.json'
    try:
        if timing is not None:
            write(root / 'timing.json', timing)
        for action in ('smoke', 'fit'):
            write(status, dict(status=action.upper()), exclusive=False)
            gpus = [baseline.select_gpu()]
            gpus.append(baseline.select_gpu(exclude=gpus))
            procs = [train_worker(rt, action, arm, gpu) for arm, gpu in zip(recipe, gpus, strict=True)]
            codes = [p.wait() for p in procs]
            if any(codes):
                raise RuntimeError(f'WiSig {action} failed with codes {codes}')
        for action in ('evaluate', 'score'):
            gpu = None if action == 'score' else baseline.select_gpu()
            write(status, dict(status=action.upper(), gpu=gpu), exclusive=False)
            if baseline.start(rt, action, 'uncertain', gpu, module=rt['runner'], extra_keys=('baseline_root',)).wait():
                raise RuntimeError(f'WiSig {action} failed')
        write(status, dict(status='SOURCE_SCORED'), exclusive=False)
    except Exception as exc:
        write(status, dict(status='FAILED', error=str(exc)), exclusive=False)
        raise


def publish(repo, output, context, ssh_config):
    config = read(context)
    runtime = {key: config[key] for key in ('project', 'python', 'frozen_sources')}
    project = PurePosixPath(runtime['project'])
    runtime.update(run_id=RUN, runner=__spec__.name, code_commit=release_commit(repo),
                   release_root=str(project / 'releases' / f'{RUN}-release'), run_root=str(project / 'runs' / RUN),
                   source_root=str(project / 'runs' / baseline.CACHE_RUN / 'cache'),
                   parent_root=str(project / 'runs' / baseline.R17_RUN), baseline_root=str(project / 'runs' / baseline.RUN))
    release(repo, output, runtime, ssh_config, archive_extras=('tools/r17_activation_factorization.py',))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest='action', required=True)
    pub = actions.add_parser('publish')
    for name in ('repo', 'output', 'context', 'ssh-config'):
        pub.add_argument('--' + name, type=Path, required=True)
    for action in ('dispatch', 'smoke', 'fit', 'evaluate', 'score'):
        worker = actions.add_parser(action)
        worker.add_argument('runtime', type=Path)
        worker.add_argument('name', nargs='?')
    cmd = parser.parse_args()
    if cmd.action == 'publish':
        publish(cmd.repo, cmd.output, cmd.context, cmd.ssh_config)
    elif cmd.action == 'dispatch':
        dispatch(read(cmd.runtime))
    elif cmd.action in ('smoke', 'fit'):
        fit(read(cmd.runtime), cmd.name, smoke=cmd.action == 'smoke')
    elif cmd.action == 'score':
        baseline.score(read(cmd.runtime), modes_for(RECIPE))
    else:
        rt = read(cmd.runtime)
        activate(rt)
        evaluate(rt)


if __name__ == '__main__':
    main()
