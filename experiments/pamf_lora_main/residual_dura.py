"""Retrain the LoRa DURA lineage on native-packet residual practical snapshots."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path, PurePosixPath

import numpy as np
import torch

from experiments.cric_dura_comparison.g_lineage import construct, load, objective, trainable
from experiments.cric_dura_comparison.g_train import train
from experiments.cric_dura_comparison.io import read, tensor, write
from .data import FS_HZ, SOURCE, TARGET, VIEWS
from .full_adaptive import SET_METHOD, SET_RUN, case, complete_f
from .full_dura_lineage import prepare_model
from .full_dura_lineage_data import in_observed_phase
from .full_dura_observed_shape import CAP, TEACHER_WEIGHT, ObservedShape
from .full_packet_data import COUNTS, LIMIT, PROTOCOL
from .full_packet_publish import DATA_RUN
from .full_packet_views import build_target, source_contract, source_role
from .full_supervised import construct as recognizer
from .packet_physics import RESIDUAL_LATENCY, chirp_windows, residual_view
from .packet_runner import schedule
from .publish import release, release_commit
from .refine_runner import start
from .runner import activate


RUN = '20261123-phase1-lora-dura-residual-practical-m1-r01'
SEED = 2026092702
ROLES = ('L_s', 'U_s')
MODES = ('direct', 'dura_native', 'dura_projected', 'dura_half')
L_COUNT = COUNTS['L_s']
COUNT = sum(COUNTS[role] for role in ROLES)
ROUTE = dict(processing_route='residual', snapshot_fs=FS_HZ, snapshot_samples=256,
             delay_samples=RESIDUAL_LATENCY, power_reference='packet_rms',
             geometry='shared_within_packet', realization='independent_per_chirp',
             equalization=False)


def pair(item):
    raw, rx, pid, index, role = item
    ground = chirp_windows(raw)
    clean = ground[:, 0] + 1j * ground[:, 1]
    namespace = 'source-U' if role == 'U_s' else 'source'
    records = []
    smooth = np.ones((1, 17), np.float64) / 17
    from scipy.signal import convolve
    for draw in range(2):
        scene = VIEWS[1 + (int(index) + draw) % 6]
        received, signal, noise = residual_view(raw, scene, rx, pid, namespace, components=True,
                                                realization=None if draw == 0 else draw)
        y = received[:, 0] + 1j * received[:, 1]
        if not np.allclose(y, signal + noise, atol=8e-7, rtol=2e-6):
            raise ValueError('Residual training targets must reproduce the registered observed IQ')
        # A local complex gain is a diagonal approximation to the exact
        # noiseless residual. Its variance uses only the injected noise trace.
        cross = convolve(signal * clean.conj(), smooth, mode='same')
        energy = convolve(np.abs(clean)**2, smooth, mode='same')
        gain = cross / np.maximum(energy, 1e-6)
        main = np.stack((gain.real, gain.imag), axis=1).astype(np.float32)
        variance = np.clip(.5 * np.mean(np.abs(noise)**2, axis=1), 1e-8, 1.).astype(np.float32)
        if not np.isfinite(main).all() or not np.isfinite(variance).all():
            raise ValueError('Nonfinite residual source action')
        records.append((received, main, variance))
    return ground, records


def build_pairs(rt):
    source = Path(rt['source_root'])
    contract = source_contract(source)
    root = Path(rt['run_root']) / 'source_pairs'
    root.mkdir(exist_ok=False)
    shapes = dict(clean=(COUNT, 6, 2, 256), received=(2, COUNT, 6, 2, 256),
                  main=(2, COUNT, 6, 2, 256), variance=(2, COUNT, 6))
    arrays = {key: np.lib.format.open_memmap(root / f'{key}.npy', mode='w+',
              dtype=np.float32, shape=shape) for key, shape in shapes.items()}
    offset = 0
    with ProcessPoolExecutor(max_workers=rt['workers']) as pool:
        for role in ROLES:
            iq, rx, ids, indices = source_role(source, contract, role)
            jobs = ((z, r, pid, i, role) for z, r, pid, i in zip(iq, rx, ids, indices, strict=True))
            for j, (ground, records) in enumerate(pool.map(pair, jobs, chunksize=12), start=1):
                position = offset + j - 1
                arrays['clean'][position] = ground
                for draw, record in enumerate(records):
                    for name, value in zip(('received', 'main', 'variance'), record, strict=True):
                        arrays[name][draw, position] = value
                if j % 500 == 0 or j == len(ids):
                    print('RESIDUAL_DURA_SOURCE_PAIRS', role, j, len(ids), flush=True)
            offset += len(ids)
    for array in arrays.values():
        array.flush()
    write(root / 'contract.json', dict(contract, practical=ROUTE))
    write(root / 'complete.json', dict(status='SOURCE_READY', draws=2, roles=list(ROLES),
                                      n=offset, F_reads=0, TX_reads=0, target_reads=0))


class ResidualView:
    def __init__(self, path):
        if read(path / 'complete.json') != dict(status='SOURCE_READY', draws=2, roles=list(ROLES),
                                               n=COUNT, F_reads=0, TX_reads=0, target_reads=0):
            raise ValueError('Complete residual source pairs required')
        self.clean, self.received, self.main, self.variance = (
            np.load(path / f'{k}.npy', allow_pickle=False, mmap_mode='r')
            for k in ('clean', 'received', 'main', 'variance'))
        self.is_L = np.tile(np.arange(COUNT) < L_COUNT, 3)
        self.quantile = np.random.default_rng(2026112219).random((20, len(self)), dtype=np.float32)

    def __len__(self):
        return 3 * COUNT

    def ready(self):
        return (self.clean.shape == (COUNT, 6, 2, 256)
                and self.received.shape == self.main.shape == (2, COUNT, 6, 2, 256)
                and self.variance.shape == (2, COUNT, 6)
                and all(a.dtype == np.float32 for a in (self.clean, self.received,
                                                        self.main, self.variance)))

    def batch(self, indices, epoch, coarse=False, action=False):
        packet = indices % COUNT
        chirp = (packet + 2 * (indices // COUNT)) % 6
        draw = (epoch + 1) % 2
        identity = self.quantile[epoch-1, indices] < .15
        clean = self.clean[packet, chirp]
        received = self.received[draw, packet, chirp]
        received[identity] = clean[identity]
        if not action:
            return {'clean': in_observed_phase(clean, received),
                    'coarse' if coarse else 'received': received}
        kernel = np.zeros((len(indices), 2, 64), np.float32)
        kernel[:, 0, 0] = 1.
        coef = np.zeros((len(indices), 4, 256), np.float32)
        coef[:, :2] = self.main[draw, packet, chirp]
        coef[identity] = 0.
        coef[identity, 0] = 1.
        variance = self.variance[draw, packet, chirp]
        variance[identity] = 1e-8
        clean, coef = in_observed_phase(clean, received, coef)
        return dict(clean=clean, received=received, kernel=kernel,
                    coefficients=coef, variance=variance)


def smoke(rt):
    source = Path(rt['source_root'])
    iq, rx, ids, indices = source_role(source, source_contract(source), 'L_s')
    ground, draws = pair((iq[0], rx[0], ids[0], indices[0], 'L_s'))
    actual, meta = residual_view(iq[0], VIEWS[1+int(indices[0])%6], rx[0], ids[0],
                                 'source', trace=True)
    if (not np.array_equal(actual, draws[0][0]) or np.array_equal(actual, ground)
            or np.array_equal(draws[1][0], actual) or len(meta['chirps']) != 6):
        raise ValueError('Real residual source-pair smoke failed')
    main = draws[0][1]
    coef = np.concatenate((main, np.zeros_like(main)), axis=1)
    phase_clean, coef = in_observed_phase(ground, actual, coef)
    clean = tensor(phase_clean, 'cpu')
    received = tensor(actual, 'cpu')
    kernel = np.zeros((6, 2, 64), dtype=np.float32)
    kernel[:, 0, 0] = 1.
    actions = tuple(tensor(a, 'cpu') for a in (kernel, coef, draws[0][2]))
    gradients = {}
    parent = None
    for name, arrays in (('coarse', None), ('action', actions), ('r17', None)):
        model = construct(name, parent)
        prepare_model(model, name)
        loss = objective(model, None, name, received, clean, 1, action_arrays=arrays,
                         reconstruction=ObservedShape())
        loss.mean().backward()
        norm = torch.nn.utils.clip_grad_norm_(trainable(model, name), 1., error_if_nonfinite=True)
        if not torch.isfinite(loss).all() or norm <= 0:
            raise ValueError(f'Residual LoRa source loss does not update {name}')
        gradients[name] = float(norm)
        parent = model
    write(Path(rt['run_root']) / 'smoke.json', dict(status='PASS', practical=ROUTE,
          paired_source=True, source_gradients=gradients, F_reads=0, TX_reads=0, target_reads=0))


def train_source(rt, *, source_pairs=None, model_seed=SEED):
    root = Path(rt['run_root'])
    source = root / 'source_pairs' if source_pairs is None else Path(source_pairs)
    c = dict(output_root=str(root / 'G' / 'lineage'), model_seed=model_seed, code_commit=rt['code_commit'],
             cache=dict(source_contract=str(source / 'contract.json'), roles='LU'),
             practical=ROUTE, observed_phase=True, cap=CAP, teacher_weight=TEACHER_WEIGHT)
    train(c, view=ResidualView(source), prepare_model=prepare_model, reconstruction=ObservedShape())


@torch.no_grad()
def predict(rt):
    root = Path(rt['run_root'])
    target = root / 'target_views'
    if read(target / 'complete.json') != dict(status='TARGET_READY', protocol=PROTOCOL,
            target_truth_read=False, receivers=list(TARGET), per_receiver=LIMIT,
            views=list(VIEWS), processing_route='residual'):
        raise ValueError('Frozen residual target data required')
    contract = source_contract(Path(rt['source_root']))
    residual_contract = read(root / 'source_pairs' / 'contract.json')
    lineage = root / 'G' / 'lineage'
    proof = read(lineage / 'complete.json')
    if (residual_contract != dict(contract, practical=ROUTE) or proof['status'] != 'SOURCE_FROZEN'
            or proof['updates'] != 39520 or any(proof[k] for k in ('F_reads','TX_labels_read','target_reads'))):
        raise ValueError('Residual DURA lineage must freeze before prediction')
    device = 'cuda:0'
    g, item = load(lineage / 'r69' / 'final.pt', residual_contract, 'LU', device)
    if item['model_seed'] != SEED or item['stage'] != 'r69':
        raise ValueError('Wrong residual DURA checkpoint')
    g.eval().requires_grad_(False)
    f = recognizer(device).eval().requires_grad_(False)
    weight = complete_f(root.parent / SET_RUN, contract, case(SEED, SET_METHOD)) / 'model.pt'
    f.load_state_dict(torch.load(weight, map_location=device, weights_only=True), strict=True)
    projector = ObservedShape().to(device).eval()
    output = root / 'predictions'
    output.mkdir(exist_ok=False)
    for rx in TARGET:
        iq = np.load(target / f'{rx}.npy', allow_pickle=False, mmap_mode='r')
        if iq.shape != (len(VIEWS), LIMIT, 6, 2, 256) or iq.dtype != np.float32:
            raise ValueError('Unexpected residual target array')
        for j, scene in enumerate(VIEWS):
            labels = [[] for _ in MODES]
            for batch in tensor(iq[j], device).split(48):
                flat = batch.flatten(0, 1)
                native = g(flat)
                projected = projector.restore(flat, native).reshape_as(batch)
                paths = (batch, native.reshape_as(batch), projected, batch + .5 * (projected - batch))
                for values, path in zip(labels, paths, strict=True):
                    logits = f(path)[0]
                    if not torch.isfinite(logits).all():
                        raise ValueError('Nonfinite residual DURA logits')
                    values.extend(logits.argmax(1).tolist())
            for mode, values in zip(MODES, labels, strict=True):
                np.save(output / f'{rx}__{scene}__{mode}.npy', np.asarray(values, dtype=np.int16))
        print('RESIDUAL_DURA_PREDICTED', SEED, rx, flush=True)
    write(output / 'complete.json', dict(status='PREDICTIONS_FROZEN', seed=SEED,
          receivers=list(TARGET), per_receiver=LIMIT, views=list(VIEWS), modes=list(MODES),
          practical=ROUTE, target_truth_read=False))


def dispatch(rt):
    root = Path(rt['run_root'])
    root.mkdir(exist_ok=False)
    (root / 'configs').mkdir()
    state = root / 'state.json'
    launch = partial(start, module=__spec__.name, extra_keys=('code_commit', 'workers'))
    try:
        for kind, status in (('smoke', 'SOURCE_SMOKE'), ('pairs', 'BUILDING_RESIDUAL_SOURCE')):
            write(state, dict(status=status), exclusive=False)
            if launch(rt, kind, 'source').wait():
                raise RuntimeError(f'Residual DURA {kind} failed')
        write(state, dict(status='TRAINING_RESIDUAL_DURA'), exclusive=False)
        schedule(rt, [('train', str(SEED))], launch)
        write(state, dict(status='BUILDING_RESIDUAL_TARGET'), exclusive=False)
        target_launch = partial(start, module=__spec__.name, extra_keys=('target_root', 'workers'))
        if target_launch(rt, 'target', 'blind').wait():
            raise RuntimeError('Residual blind target generation failed')
        write(state, dict(status='PREDICTING'), exclusive=False)
        schedule(rt, [('predict', str(SEED))], launch)
        write(state, dict(status='PREDICTIONS_FROZEN', target_truth_read=False), exclusive=False)
    except Exception as error:
        write(state, dict(status='FAILED', error=str(error)), exclusive=False)
        raise


def publish(repo, output, context, ssh_config, data_submit, set_submit):
    ctx = read(context)
    rt = {k: ctx[k] for k in ('project', 'python', 'frozen_sources')}
    project = PurePosixPath(rt['project'])
    for path, name in ((data_submit, DATA_RUN), (set_submit, SET_RUN)):
        proof = read(path)
        if (proof['archive_transfer'] != 'VERIFIED'
                or PurePosixPath(proof['run_root']) != project / 'runs' / name
                or PurePosixPath(proof['cwd']) != project / 'releases' / f'{name}-release'):
            raise ValueError('Residual DURA requires the registered LoRa dependencies')
    inputs = project / 'releases' / f'{DATA_RUN}-release' / 'data'
    run = project / 'runs' / RUN
    rt.update(run_id=RUN, runner=__spec__.name, code_commit=release_commit(repo),
              run_root=str(run), release_root=str(project / 'releases' / f'{RUN}-release'),
              source_root=str(inputs / 'source'), target_root=str(inputs / 'target'),
              parent_root=str(project / 'runs' / DATA_RUN), target_views=str(run / 'target_views'),
              workers=16, allowed_gpus=[3, 0, 1, 4, 5, 6, 2], max_workers=1, max_compute_processes=4)
    return release(repo, output, rt, ssh_config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='kind', required=True)
    pub = sub.add_parser('publish')
    for k in ('repo', 'output', 'context', 'ssh-config', 'data-submit', 'set-submit'):
        pub.add_argument('--'+k, type=Path, required=True)
    for kind in ('dispatch', 'smoke', 'pairs', 'train', 'target', 'predict'):
        stage = sub.add_parser(kind)
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
            if kind == 'target':
                build_target(Path(runtime['target_root']), Path(runtime['run_root']) / 'target_views',
                             processing_route='residual', workers=runtime['workers'])
            else:
                dict(smoke=smoke, pairs=build_pairs, train=train_source, predict=predict)[kind](runtime)
