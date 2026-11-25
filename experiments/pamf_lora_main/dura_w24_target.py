"""Single-seed, truth-blind test of W24 input/output around frozen LoRa DURA."""

import argparse
import importlib
from json import JSONDecodeError
import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from multiprocessing import get_context
from pathlib import Path
import subprocess
import sys
from time import perf_counter, sleep
from types import ModuleType

import numpy as np
import torch

from experiments.cric_dura_comparison.io import read, tensor, write
from . import dura_capacity_target
from .channel_blind import corruption_score
from .data import TARGET, VIEWS
from .dura_capacity_source import RECIPES, recover
from .dura_repair import PacketRestorer
from .dura_w24 import apertures, readouts
from .dura_w24_model import WideTracker
from .dura_w24_train import RECIPE as WIDE_RECIPE, RUN as WIDE_RUN
from .full_packet_data import LIMIT
from .full_packet_views import native, source_contract, source_role, target_contract
from .joint_unroll_source import coordinates
from .packet_physics import CONTEXT, channel, hardware
from .publish import release, release_commit
from .runner import activate
from .spectral_block_source import load_models
from .weak_guard_target import DATA


RUN = '20261125-phase1-lora-dura-w24-io-s2026092702-r01'
F_RUN = '20261125-phase1-lora-pamf-tri-clean-nativewide98-m2-r01'
F_CODE = 'e6dde7d40f2cc3dd7b21d83daa4bcbabac52d3ae'
F_METHOD = 'phase_native24'
SEED = 2026092702
N = 200
SHARDS = 4
MODES = ('direct_w24', 'dura_6_lift', 'dura_8_lift', 'dura_w24_retrained')
DEVICE = 'cuda:0'


def archived_runtime(project, run, code):
    archive = project / 'releases' / f'{run}-release'
    rt = read(archive / 'runtime.json')
    if (rt['run_id'], rt['code_commit']) != (run, code):
        raise ValueError(f'Unexpected archived runtime for {run}')
    return archive, rt


def load_frozen_wide(rt, source, contract):
    _, wide_rt = archived_runtime(Path(rt['project']), WIDE_RUN, rt['training_commit'])
    wide_root = Path(wide_rt['run_root'])
    if (Path(wide_rt['source_root']) != source or read(wide_root/'state.json')
            != dict(status='SOURCE_FROZEN', steps=WIDE_RECIPE.steps)):
        raise ValueError('The independent W24 40,000-step fit must finish before prediction')
    models, threshold = load_models(wide_root, contract, DEVICE, {'w24': WIDE_RECIPE})
    return models[0], threshold


def frozen_models(rt):
    project = Path(rt['project'])
    source = Path(rt['source_root'])
    contract = source_contract(source)
    activate(rt)
    _, parent = archived_runtime(project, dura_capacity_target.ORIGIN, dura_capacity_target.ORIGIN_CODE)
    if Path(parent['source_root']) != source:
        raise ValueError('DURA source roles differ')
    room = Path(parent['run_root'])
    fit = dura_capacity_target.frozen_fit(room, RECIPES['expanded'].steps)
    models, threshold = load_models(room, contract, DEVICE, {'expanded': RECIPES['expanded']})
    if threshold != fit['threshold']:
        raise ValueError('DURA source threshold differs')

    wide_model, wide_threshold = load_frozen_wide(rt, source, contract)
    wide_tracker = WideTracker().to(DEVICE).eval()

    archive, reader_rt = archived_runtime(project, F_RUN, F_CODE)
    package = ModuleType('lora_w24_archive')
    package.__path__ = [str(archive / 'experiments' / 'pamf_lora_main')]
    sys.modules[package.__name__] = package
    module = importlib.import_module(f'{package.__name__}.tri_clean95')
    _, _, reader_contract = module.source_handles(reader_rt)
    f_run = Path(reader_rt['run_root'])
    f_path = f_run / 'F' / F_METHOD
    proof = read(f_path / 'complete.json')
    if (reader_contract != contract or F_METHOD not in read(f_run / 'selection.json')['names']
            or (proof['status'], proof['seed'], proof['method'], proof['frozen']['epoch'])
            != ('SOURCE_FROZEN', SEED, F_METHOD, 108)
            or proof['target_reads'] or proof['U_truth_reads']):
        raise ValueError('Expected the legal W24 source-frozen reader')
    row = module.NATIVE_WIDE_ROWS[F_METHOD]
    reader = module.clean_model(reader_rt, f_run, reader_contract, row, DEVICE).eval().requires_grad_(False)
    reader.load_state_dict(torch.load(f_path / 'best.pt', map_location=DEVICE, weights_only=True), strict=True)
    tracker = PacketRestorer('consensus').to(DEVICE).eval()

    def restored(x):
        wave, _, _ = recover(x, tracker, models, -float('inf'))
        return wave[3]

    def predict(narrow, wide):
        # Both arms share the original six-window activation decision.
        aligned = coordinates(tracker, narrow[:, 1:7])[0]
        active = corruption_score(aligned) > threshold
        six, eight = readouts(wide, narrow, restored, active)
        trained, trained_active = wide_model.restore(wide_tracker, wide, wide_threshold)
        direct = reader(wide)[0]
        logits = [direct, *(direct if wave is wide else reader(wave)[0] for wave in (six, eight, trained))]
        if not all(value.shape == (len(wide), 10) and torch.isfinite(value).all() for value in logits):
            raise ValueError('Invalid W24 frozen recognition output')
        values = np.asarray(torch.stack(logits).argmax(-1).tolist(), dtype=np.int16)
        return values, np.array([int(active.sum()), int(trained_active.sum())])

    return predict, contract, reader_rt, (models[0], wide_model)


def receive(scene, rx, item, *, namespace='w24-target-full'):
    """Generate the full received preamble, keeping channel metadata off the reader path."""
    index, packet = item
    from leo_practical.channel import ChannelStream, normalize_input, stable_seed

    z, _ = normalize_input(packet)
    cfg = channel(scene)
    # The recording begins at the preamble. Leave the out-of-recording guard
    # silent and retain the known causal latency in the simulation buffer.
    latency = cfg.fractional_delay_half_length
    rx_hw = hardware(scene, f'{namespace}/{rx}')
    seed = stable_seed(2026112207, namespace, scene, rx, str(index))
    output, meta = ChannelStream(cfg, seed, rx_hw).process(
        np.pad(z, (0, latency)), preceding_context=np.zeros(CONTEXT, dtype=z.dtype))
    if (meta['boundary'] != 'provided_context' or meta['channel_equalization_applied']
            or meta['numerical_filter_latency_samples'] != latency):
        raise ValueError('Full practical receive-side alignment differs')
    received = output[latency:latency+len(z)].astype(np.complex64)
    return apertures(received)


@torch.no_grad()
def worker(rt, shard):
    started = perf_counter()
    output = Path(rt['run_root'])
    predict, contract, f_rt, models = frozen_models(rt)

    def batch(narrow, wide):
        return predict(tensor(narrow, DEVICE), tensor(wide, DEVICE))

    if shard == 'smoke':
        iq, _, _, _ = source_role(Path(rt['source_root']), contract, 'U_s')
        ng, wg = apertures(iq[:12])
        raw, quiet = batch(ng, wg)
        rng = np.random.default_rng(2026112581)
        native_iq = iq[:12]
        noise = (rng.standard_normal(native_iq.shape) + 1j*rng.standard_normal(native_iq.shape))
        rms = np.sqrt(np.mean(np.abs(native_iq)**2, axis=-1, keepdims=True))
        noisy = (native_iq + .18 * rms * noise / 2**.5).astype(np.complex64)
        _, changed = batch(*apertures(noisy))
        if quiet.any() or not changed.all() or not np.array_equal(raw, np.repeat(raw[0:1], len(MODES), axis=0)):
            raise ValueError('Expected source clean bypass and active receiver-only recovery')
        write(output / 'smoke.json', dict(status='PASS', clean_active=quiet.tolist(), disturbed_active=changed.tolist(),
                                         parameters=[sum(p.numel() for p in m.parameters()) for m in models], target_reads=0))
        return

    target = Path(f_rt['native_target_root'])
    target_contract(target)
    reference = Path(f_rt['run_root']) / 'predictions' / F_METHOD
    clean_frozen = read(reference / 'complete.json')
    if (clean_frozen['status'] != 'PREDICTIONS_FROZEN' or clean_frozen['views'] != ['clean']
            or clean_frozen['seed'] != SEED or clean_frozen['per_receiver'] != LIMIT
            or clean_frozen['receivers'] != list(TARGET) or clean_frozen['target_truth_read']):
        raise ValueError('W24 clean comparison does not match the blind physical archive')
    replay_shard(rt, shard, target, batch, reference, count=N, namespace='w24-target-full',
                 check_views=('clean',), active_modes=2, batch_size=32, started=started)


def replay_shard(rt, shard, target, batch, reference, *, count, namespace, check_views,
                 active_modes, batch_size, started):
    output = Path(rt['run_root'])
    with ProcessPoolExecutor(max_workers=3, mp_context=get_context('spawn'),
                             initializer=activate, initargs=(rt,)) as pool:
        for rx in TARGET[int(shard)::SHARDS]:
            packets = native(target, rx, LIMIT)
            for view in VIEWS:
                rows, active = [], np.zeros(active_modes, dtype=np.int64)
                if view == 'clean':
                    for start in range(0, LIMIT, 64):
                        values, batch_active = batch(*apertures(packets[start:start+64]))
                        rows.append(values)
                        active += batch_active
                else:
                    rendered = pool.map(partial(receive, view, rx, namespace=namespace),
                                        enumerate(packets[:count]), chunksize=8)
                    narrow, wide = [], []
                    for x, y in rendered:
                        narrow.append(x)
                        wide.append(y)
                        if len(narrow) == batch_size:
                            values, batch_active = batch(np.stack(narrow), np.stack(wide))
                            rows.append(values)
                            active += batch_active
                            narrow.clear()
                            wide.clear()
                    if narrow:
                        values, batch_active = batch(np.stack(narrow), np.stack(wide))
                        rows.append(values)
                        active += batch_active
                decisions = np.concatenate(rows, axis=1)
                if view in check_views:
                    prior = np.load(reference / f'{rx}__{view}__direct.npy', allow_pickle=False)
                    if not np.array_equal(prior, decisions[0]):
                        raise ValueError(f'W24 direct decisions differ from the archive for {rx}/{view}')
                with (output / 'predictions' / f'{rx}__{view}.npy').open('xb') as stream:
                    np.save(stream, decisions, allow_pickle=False)
                print('W24_VIEW', rx, view, decisions.shape[1], active, flush=True)
    write(output / f'shard-{shard}.json', dict(status='COMPLETE', elapsed_seconds=perf_counter()-started,
                                               truth_reads=0))


def dispatch(rt, runtime, *, freeze=None):
    root = Path(rt['run_root'])
    root.mkdir(exist_ok=False)
    state = root / 'state.json'
    (root / 'predictions').mkdir()
    (root / 'logs').mkdir()

    def launch(shard, gpu):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='1',
                   MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
        command = [rt['python'], '-u', '-m', rt['runner'], 'worker', str(runtime), str(shard)]
        with (root / 'logs' / f'{shard}.log').open('x', encoding='utf-8') as log:
            proc = subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        print('W24_LAUNCHED', shard, proc.pid, gpu, flush=True)
        return proc

    try:
        waiting = dict(status='WAITING_FOR_FROZEN_TRAINING')
        write(state, waiting)
        parent = Path(rt['project']) / 'runs' / WIDE_RUN / 'state.json'
        while True:
            try:
                current = read(parent)['status']
            except (FileNotFoundError, JSONDecodeError):
                current = None
            if current == 'SOURCE_FROZEN':
                break
            if current == 'FAILED':
                raise RuntimeError('The W24 source run failed before freezing')
            cmdline = Path(f"/proc/{rt['training_dispatcher']}/cmdline")
            try:
                binding = cmdline.read_bytes()
            except FileNotFoundError:
                binding = b''
            if f'{WIDE_RUN}-release'.encode() not in binding:
                if read(parent)['status'] == 'SOURCE_FROZEN':
                    break
                raise RuntimeError('The W24 source dispatcher exited without a frozen model')
            sleep(20)
        raw = subprocess.check_output(['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.free',
                                       '--format=csv,noheader,nounits'], text=True)
        inventory = (tuple(map(int, line.split(','))) for line in raw.splitlines())
        available = [gpu for _, _, gpu in sorted((load, -free, gpu) for gpu, load, free in inventory if free >= 8192)]
        if not available:
            raise RuntimeError('No GPU has the required free memory')
        write(state, dict(status='SMOKE'), exclusive=False)
        if launch('smoke', available[0]).wait():
            raise RuntimeError('Real-checkpoint no-target W24 smoke failed')
        for start in range(0, SHARDS, len(available)):
            slots = dict(zip(range(start, SHARDS), available[:SHARDS], strict=False))
            write(state, dict(status='PREDICT', gpus=slots), exclusive=False)
            jobs = [launch(shard, gpu) for shard, gpu in slots.items()]
            if any([job.wait() for job in jobs]):
                raise RuntimeError('A W24 replay shard failed')
        if {p.name for p in (root / 'predictions').glob('*.npy')} != prediction_names():
            raise ValueError('W24 blind prediction matrix is incomplete')
        write(root / 'predictions' / 'complete.json', (freeze or completion)())
        write(state, dict(status='PREDICTIONS_FROZEN'), exclusive=False)
    except Exception as error:
        write(state, dict(status='FAILED', error=str(error)), exclusive=False)
        raise


def prediction_names():
    return {f'{rx}__{view}.npy' for rx in TARGET for view in VIEWS}


def completion():
    return dict(status='PREDICTIONS_FROZEN', run=RUN, reader=F_RUN, reader_commit=F_CODE,
                seed=SEED, source=dura_capacity_target.ORIGIN, retrained_source=WIDE_RUN,
                receivers=list(TARGET), views=list(VIEWS),
                modes=list(MODES), practical_per_rx=N, clean_per_rx=LIMIT, target_truth_read=False)


def publish(repo, output, context, ssh_config, training_receipt):
    cfg = read(context)
    project = cfg['project']
    training = read(training_receipt)
    if Path(training['run_root']).name != WIDE_RUN or training['archive_transfer'] != 'VERIFIED':
        raise ValueError('Expected the verified W24 training receipt')
    rt = {key: cfg[key] for key in ('project', 'python', 'frozen_sources')}
    rt.update(run_id=RUN, runner=__spec__.name, code_commit=release_commit(repo),
              release_root=f'{project}/releases/{RUN}-release', run_root=f'{project}/runs/{RUN}',
              source_root=f'{project}/releases/{DATA}-release/data/source',
              training_commit=training['code_commit'], training_dispatcher=training['pid'],
              external_releases=[f'{project}/releases/{F_RUN}-release',
                                 f'{project}/releases/{dura_capacity_target.ORIGIN}-release',
                                 f'{project}/releases/{WIDE_RUN}-release'])
    release(repo, output, rt, ssh_config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    publisher = commands.add_parser('publish')
    for arg in ('repo', 'output', 'context', 'ssh-config', 'training-receipt'):
        publisher.add_argument('--' + arg, type=Path, required=True)
    commands.add_parser('dispatch').add_argument('runtime', type=Path)
    item = commands.add_parser('worker')
    item.add_argument('runtime', type=Path)
    item.add_argument('shard')
    args = vars(parser.parse_args())
    tool = args.pop('command')
    if tool == 'publish':
        publish(**args)
    else:
        rt = read(args['runtime'])
        if rt['run_id'] != RUN:
            raise ValueError('Expected the dedicated immutable W24 study root')
        if tool == 'dispatch':
            dispatch(rt, args['runtime'])
        else:
            worker(rt, args['shard'])
