"""Evaluate a separately frozen source gate on matched residual DURA."""

import argparse
from functools import partial
from json import JSONDecodeError
from pathlib import Path, PurePosixPath
import time

import numpy as np
import torch

from experiments.cric_dura_comparison.g_lineage import load as load_g
from experiments.cric_dura_comparison.io import read, tensor, write
from .data import SOURCE, TARGET, VIEWS
from .full_adaptive import SET_METHOD, SET_RUN, case, complete_f
from .full_dura_observed_shape import ObservedShape
from .full_packet_data import LIMIT, PROTOCOL
from .full_packet_publish import DATA_RUN
from .full_packet_views import source_contract, source_role
from .full_supervised import construct as recognizer
from .packet_physics import chirp_windows, residual_view
from .packet_runner import schedule
from .publish import release, release_commit
from .refine_runner import start
from .residual_dura import ROUTE, RUN as PARENT_RUN, SEED, train_source
from .residual_dura_gate import ISOLATION, fit, load
from .runner import activate


RUN = '20261123-phase1-lora-dura-residual-gate-m1-r01'
MODES = ('direct', 'dura_half', 'dura_gated_half')
SEED_RUNS = {
    2026092701: '20261123-phase1-lora-dura-residual-gate-s2026092701-r01',
    SEED: RUN,
    2026092703: '20261123-phase1-lora-dura-residual-gate-s2026092703-r01',
}


def wait_for_parent(rt, artifact):
    parent = Path(rt['parent_root'])
    while True:
        if (parent / artifact).is_file():
            return
        try:
            status = read(parent / 'state.json')
        except (FileNotFoundError, JSONDecodeError):
            status = {}
        if status.get('status') in ('FAILED', 'PREDICTIONS_FROZEN'):
            raise RuntimeError(f'Residual parent ended without {artifact}: {status}')
        try:
            cwd = Path(f"/proc/{rt['parent_pid']}/cwd").resolve(strict=True)
        except FileNotFoundError:
            if (parent / artifact).is_file():
                return
            raise RuntimeError(f'Residual parent controller exited without {artifact}') from None
        if cwd.name != f'{PARENT_RUN}-release':
            raise RuntimeError('Residual parent controller working directory differs')
        time.sleep(30)


def frozen_models(rt, device):
    source = source_contract(Path(rt['source_root']))
    root = Path(rt['run_root'])
    parent = Path(rt['parent_root'])
    seed = rt.get('model_seed', SEED)
    residual = read(parent / 'source_pairs' / 'contract.json')
    lineage = (parent if seed == SEED else root) / 'G' / 'lineage'
    ready = read(lineage / 'complete.json')
    checkpoint = lineage / 'r69' / 'final.pt'
    if (residual != dict(source, practical=ROUTE) or ready['status'] != 'SOURCE_FROZEN'
            or ready['updates'] != 39520 or ready['checkpoint'] != str(checkpoint)
            or any(ready[key] != value for key, value in ISOLATION.items())):
        raise ValueError('Complete matched residual DURA lineage required')
    model, proof = load_g(checkpoint, residual, 'LU', device)
    if proof['stage'] != 'r69' or proof['model_seed'] != seed:
        raise ValueError('Wrong residual R69 checkpoint')
    gate = load(Path(rt.get('gate_path', root / 'gate.json')), source, device)
    return model.requires_grad_(False), gate, source


def fit_source(rt):
    fit(rt['source_root'], Path(rt['parent_root']) / 'source_pairs',
        Path(rt['run_root']) / 'gate.json', 'cuda:0')
    print('DISTURBANCE_GATE_SOURCE_FROZEN', flush=True)


def train_replica(rt):
    seed = rt['model_seed']
    if seed not in SEED_RUNS or seed == SEED:
        raise ValueError('Expected a registered residual replica seed')
    source = source_contract(Path(rt['source_root']))
    pairs = Path(rt['parent_root']) / 'source_pairs'
    if read(pairs / 'contract.json') != dict(source, practical=ROUTE):
        raise ValueError('Mismatched residual pair bank for replication')
    train_source(rt, source_pairs=pairs, model_seed=seed)


@torch.no_grad()
def smoke(rt):
    model, gate, contract = frozen_models(rt, 'cuda:0')
    projector = ObservedShape().to('cuda:0').eval()
    iq, receiver, pid, _ = source_role(Path(rt['source_root']), contract, 'L_s')
    rows = [int(np.flatnonzero(receiver == rx)[0]) for rx in SOURCE]
    clean = [chirp_windows(iq[i]) for i in rows]
    disturbed = [residual_view(iq[i], 'practical_low_urban', receiver[i], pid[i], 'source') for i in rows]
    batch = tensor(np.stack(clean + disturbed), 'cuda:0')
    restored, active = gate.restore(batch, model, projector)
    if not active.any() or active.all() or not torch.isfinite(restored).all():
        raise ValueError('Gate smoke requires finite outputs and both decision branches')
    selected = batch[active]
    flat = selected.flatten(0, 1)
    projected = projector.restore(flat, model(flat)).reshape_as(selected)
    if not torch.equal(restored[~active], batch[~active]) or not torch.allclose(
            restored[active], selected + .5 * (projected - selected), atol=1e-6, rtol=1e-5):
        raise ValueError('Disturbance gate differs from frozen half-dose contract')
    write(Path(rt['run_root']) / 'smoke.json', dict(status='PASS', seed=rt.get('model_seed', SEED), stage='r69',
          active=int(active.sum()), samples=len(batch), exact_inactive_bypass=True,
          active_matches_half_dose=True, **ISOLATION))
    print('DISTURBANCE_GATE_R69_SOURCE_SMOKE_PASS', flush=True)


@torch.no_grad()
def predict(rt):
    root = Path(rt['run_root'])
    seed = rt.get('model_seed', SEED)
    check = read(root / 'smoke.json')
    if check['status'] != 'PASS' or check['stage'] != 'r69' or check['seed'] != seed:
        raise ValueError('Frozen R69 source smoke required')
    target = Path(rt['target_views'])
    if read(target / 'complete.json') != dict(status='TARGET_READY', protocol=PROTOCOL,
            target_truth_read=False, receivers=list(TARGET), per_receiver=LIMIT,
            views=list(VIEWS), processing_route='residual'):
        raise ValueError('Matched residual blind target IQ required')
    device = 'cuda:0'
    g, gate, source = frozen_models(rt, device)
    f = recognizer(device).eval().requires_grad_(False)
    weight = complete_f(root.parent / SET_RUN, source, case(seed, SET_METHOD)) / 'model.pt'
    f.load_state_dict(torch.load(weight, map_location=device, weights_only=True), strict=True)
    projector = ObservedShape().to(device).eval()
    output = root / 'predictions'
    output.mkdir(exist_ok=False)
    activity = {}
    for rx in TARGET:
        iq = np.load(target / f'{rx}.npy', allow_pickle=False, mmap_mode='r')
        if iq.shape != (len(VIEWS), LIMIT, 6, 2, 256) or iq.dtype != np.float32:
            raise ValueError('Invalid residual blind IQ array')
        activity[rx] = {}
        for j, scene in enumerate(VIEWS):
            labels = [[], [], []]
            active_count = 0
            for batch in tensor(iq[j], device).split(48):
                flat = batch.flatten(0, 1)
                half = batch + .5 * (projector.restore(flat, g(flat)).reshape_as(batch) - batch)
                score = gate.score(batch)
                if not torch.isfinite(score).all():
                    raise ValueError('Nonfinite disturbance gate score')
                active = score > gate.threshold
                active_count += int(active.sum())
                gated = torch.where(active[:, None, None, None], half, batch)
                for values, wave in zip(labels, (batch, half, gated), strict=True):
                    logits = f(wave)[0]
                    if not torch.isfinite(logits).all():
                        raise ValueError('Nonfinite LoRa gate prediction logits')
                    values.extend(logits.argmax(1).tolist())
            activity[rx][scene] = active_count
            for mode, values in zip(MODES, labels, strict=True):
                np.save(output / f'{rx}__{scene}__{mode}.npy', np.asarray(values, dtype=np.int16))
        print('RESIDUAL_DURA_GATE_PREDICTED', seed, rx, flush=True)
    write(root / 'activity.json', activity)
    write(output / 'complete.json', dict(status='PREDICTIONS_FROZEN', seed=seed,
          receivers=list(TARGET), per_receiver=LIMIT, views=list(VIEWS), modes=list(MODES),
          practical=ROUTE, target_truth_read=False))


def dispatch(rt):
    root = Path(rt['run_root'])
    root.mkdir(exist_ok=False)
    (root / 'configs').mkdir()
    state = root / 'state.json'
    seed = rt.get('model_seed', SEED)
    try:
        if seed == SEED:
            launch = partial(start, module=__spec__.name)
            write(state, dict(status='FITTING_SOURCE_GATE'))
            schedule(rt, [('fit', 'source')], launch)
            write(state, dict(status='WAITING_FOR_RESIDUAL_R69'), exclusive=False)
            wait_for_parent(rt, 'G/lineage/complete.json')
        else:
            write(state, dict(status='WAITING_FOR_RESIDUAL_SOURCE_PAIRS'))
            wait_for_parent(rt, 'source_pairs/complete.json')
            write(state, dict(status='TRAINING_RESIDUAL_DURA'), exclusive=False)
            source_worker = partial(start, module=__spec__.name, extra_keys=('model_seed', 'code_commit'))
            schedule(rt, [('train', str(seed))], source_worker)
            launch = partial(start, module=__spec__.name, extra_keys=('model_seed', 'gate_path'))
        write(state, dict(status='SOURCE_R69_SMOKE'), exclusive=False)
        schedule(rt, [('smoke', 'r69')], launch)
        write(state, dict(status='WAITING_FOR_BLIND_VIEWS'), exclusive=False)
        wait_for_parent(rt, 'target_views/complete.json')
        write(state, dict(status='PREDICTING'), exclusive=False)
        schedule(rt, [('predict', str(seed))], launch)
        write(state, dict(status='PREDICTIONS_FROZEN', target_truth_read=False), exclusive=False)
    except Exception as exc:
        write(state, dict(status='FAILED', error=str(exc)), exclusive=False)
        raise


def publish(repo, output, context, ssh_config, parent_submit, model_seed=SEED):
    ctx = read(context)
    rt = {key: ctx[key] for key in ('project', 'python', 'frozen_sources')}
    project = PurePosixPath(rt['project'])
    parent = project / 'runs' / PARENT_RUN
    receipt = read(parent_submit)
    if (receipt['archive_transfer'] != 'VERIFIED' or PurePosixPath(receipt['run_root']) != parent
            or PurePosixPath(receipt['cwd']) != project / 'releases' / f'{PARENT_RUN}-release'):
        raise ValueError('Invalid residual DURA parent release receipt')
    run = SEED_RUNS[model_seed]
    rt.update(run_id=run, runner=__spec__.name, code_commit=release_commit(repo), model_seed=model_seed,
              run_root=str(project / 'runs' / run), release_root=str(project / 'releases' / f'{run}-release'),
              source_root=str(project / 'releases' / f'{DATA_RUN}-release' / 'data' / 'source'),
              parent_root=str(parent), parent_pid=int(receipt['pid']),
              target_views=str(parent / 'target_views'), allowed_gpus=[4, 2, 5, 1, 6, 0, 3],
              max_workers=1, max_compute_processes=4)
    if model_seed != SEED:
        rt['gate_path'] = str(project / 'runs' / RUN / 'gate.json')
        if model_seed == 2026092703:
            rt['allowed_gpus'] = [5, 6, 1, 4, 2, 0, 3]
    return release(repo, output, rt, ssh_config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='kind', required=True)
    pub = commands.add_parser('publish')
    for key in ('repo', 'output', 'context', 'ssh-config', 'parent-submit'):
        pub.add_argument('--' + key, type=Path, required=True)
    pub.add_argument('--model-seed', type=int, choices=tuple(SEED_RUNS), default=SEED)
    for key in ('dispatch', 'fit', 'train', 'smoke', 'predict'):
        stage = commands.add_parser(key)
        stage.add_argument('runtime', type=Path)
        stage.add_argument('name', nargs='?')
    args = vars(parser.parse_args())
    kind = args.pop('kind')
    if kind == 'publish':
        publish(**args)
    else:
        runtime = read(args['runtime'])
        if kind == 'dispatch':
            dispatch(runtime)
        else:
            activate(runtime)
            {'fit': fit_source, 'train': train_replica, 'smoke': smoke, 'predict': predict}[kind](runtime)
