"""Truth-blind native Full replay with the fixed satellite-augmented PAMF W24."""

import argparse
import importlib
from pathlib import Path
import sys
from time import perf_counter
from types import ModuleType

import numpy as np
import torch

from experiments.cric_dura_comparison.io import read, tensor, write
from .data import TARGET, VIEWS
from .dura_w24 import apertures
from .dura_w24_model import WideTracker
from .dura_w24_target import (
    DEVICE, F_CODE, F_METHOD, F_RUN, SEED, WIDE_RUN, archived_runtime, dispatch,
    load_frozen_wide, replay_shard,
)
from .full_packet_data import LIMIT
from .full_packet_views import native, source_contract, target_contract
from .publish import release, release_commit
from .runner import activate


RUN = '20261125-phase1-lora-dura-w24-sat-aug-full-s2026092702-r01'
READER = '20261125-phase1-lora-pamf-w24-sat-aug-s2026092702-r01'
READER_CODE = '502629361a6aafa564a94947f3116e82f58c3692'
DATA = '20261122-phase1-lora-full-aligned-data-r01'
MODES = ('direct', 'dura_w24_retrained')
REFERENCE_MODES = ('direct', 'complex_response_half_center')


def reference_completion():
    return dict(status='PREDICTIONS_FROZEN', W24_commit=F_CODE, seed=SEED,
                modes=list(REFERENCE_MODES), receivers=list(TARGET), views=list(VIEWS),
                per_receiver=LIMIT, processing_route='full_native', target_truth_read=False,
                classifier_run=READER)


def completion():
    return reference_completion() | dict(run=RUN, modes=list(MODES), reader_commit=READER_CODE,
                                         retrained_source=WIDE_RUN, receive_namespace='target',
                                         exact_archived_direct=True)


def load_reader(rt, source, contract):
    project = Path(rt['project'])
    release_root, reference = archived_runtime(project, READER, READER_CODE)
    original, _ = archived_runtime(project, F_RUN, F_CODE)
    root = Path(reference['run_root'])
    folder = root / 'F'
    proof = read(folder / 'complete.json')
    if (Path(reference['source_root']) != source or Path(reference['w24_release_root']) != original
            or (proof['status'], proof['method'], proof['contract'], proof['seed'], proof['epochs'],
                proof['steps'], proof['frozen']['epoch'], proof['frozen']['correct'], proof['frozen']['n'])
            != ('SOURCE_FROZEN', READER, contract, SEED, 112, 9856, 104, 11803, 12000)
            or (proof['w24_recipe_commit'], proof['satellite_source_role'], proof['satellite_batches'],
                proof['satellite_weight'], proof['initialization']) != (F_CODE, 'L_s', 9856, .5, 'random_seed')
            or proof['clean_only'] or proof['target_reads'] or proof['U_truth_reads']):
        raise ValueError('The registered satellite-augmented W24 checkpoint is required')
    if read(release_root / 'submit.json')['code_commit'] != READER_CODE:
        raise ValueError('The archived augmented reader release differs')

    package = ModuleType('lora_w24_archive')
    package.__path__ = [str(original / 'experiments' / 'pamf_lora_main')]
    sys.modules[package.__name__] = package
    module = importlib.import_module(f'{package.__name__}.tri_clean95')
    reader = module.clean_model(dict(scratch=True), root, contract, module.NATIVE_WIDE_ROWS[F_METHOD], DEVICE)
    reader.load_state_dict(torch.load(folder / 'best.pt', map_location=DEVICE, weights_only=True), strict=True)
    return reader.eval().requires_grad_(False), module, reference


@torch.inference_mode()
def worker(rt, shard):
    started = perf_counter()
    activate(rt)
    root, source = Path(rt['run_root']), Path(rt['source_root'])
    contract = source_contract(source)
    model, threshold = load_frozen_wide(rt, source, contract)
    reader, module, reference = load_reader(rt, source, contract)
    tracker = WideTracker().to(DEVICE).eval()

    def predict(wave):
        scores = reader(wave)[0]
        if scores.shape != (len(wave), 10) or not torch.isfinite(scores).all():
            raise ValueError('Expected a finite, ten-class frozen W24 readout')
        return np.asarray(scores.argmax(1).tolist(), dtype=np.int16)

    def batch(_narrow, wide):
        iq = tensor(wide, DEVICE)
        restored, active = model.restore(tracker, iq, threshold)
        direct = predict(iq)
        changed = direct.copy()
        if active.any():
            changed[np.asarray(active.tolist())] = predict(restored[active])
        return np.stack((direct, changed)), np.array([int(active.sum())])

    if shard == 'smoke':
        packets = native(source, 'U_s', contract['counts']['U_s'])[:12]
        center = apertures(packets)[1]
        registered = module.clean_iq(packets, 384, module.ALL_CHIRPS, native_edges=True, edge_points=24)
        if not np.array_equal(center, registered):
            raise ValueError('W24 receive aperture differs from the registered augmented reader')
        decisions, clean = batch(None, center)
        rng = np.random.default_rng(2026112581)
        noise = rng.standard_normal(packets.shape) + 1j * rng.standard_normal(packets.shape)
        rms = np.sqrt(np.mean(np.abs(packets)**2, axis=-1, keepdims=True))
        noisy = (packets + .18 * rms * noise / 2**.5).astype(np.complex64)
        _, practical = batch(None, apertures(noisy)[1])
        if clean.any() or not practical.any() or not np.array_equal(*decisions):
            raise ValueError('Source clean bypass or receiver-only recovery smoke failed')
        write(root / 'smoke.json', dict(status='PASS', target_reads=0, source_U_packets=12,
                                       clean_active=int(clean[0]), perturbed_active=int(practical[0]),
                                       parameters=sum(p.numel() for p in model.parameters())))
        return

    target = Path(reference['native_target_root'])
    if target != source.parent / 'target':
        raise ValueError('The registered augmented W24 source/target dataset differs')
    target_contract(target)
    archived = Path(reference['run_root']) / 'predictions'
    if read(archived / 'complete.json') != reference_completion():
        raise ValueError('Missing or incompatible complete augmented W24 archive')
    replay_shard(rt, shard, target, batch, archived, count=LIMIT, namespace='target',
                 check_views=VIEWS, active_modes=1, batch_size=64, started=started)


def publish(repo, output, context, ssh_config, training_receipt):
    cfg, training = read(context), read(training_receipt)
    if Path(training['run_root']).name != WIDE_RUN or training['archive_transfer'] != 'VERIFIED':
        raise ValueError('Expected the verified W24 training receipt')
    project = cfg['project']
    rt = {key: cfg[key] for key in ('project', 'python', 'frozen_sources')}
    rt.update(run_id=RUN, runner=__spec__.name, code_commit=release_commit(repo),
              release_root=f'{project}/releases/{RUN}-release', run_root=f'{project}/runs/{RUN}',
              source_root=f'{project}/releases/{DATA}-release/data/source',
              training_commit=training['code_commit'], training_dispatcher=training['pid'],
              external_releases=[f'{project}/releases/{run}-release' for run in (READER, F_RUN, WIDE_RUN, DATA)])
    release(repo, output, rt, ssh_config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    publisher = commands.add_parser('publish')
    for field in ('repo', 'output', 'context', 'ssh-config', 'training-receipt'):
        publisher.add_argument(f'--{field}', required=True, type=Path)
    commands.add_parser('dispatch').add_argument('runtime', type=Path)
    child = commands.add_parser('worker')
    child.add_argument('runtime', type=Path)
    child.add_argument('shard')
    args = vars(parser.parse_args())
    command = args.pop('command')
    if command == 'publish':
        publish(**args)
    else:
        rt = read(args['runtime'])
        if rt['run_id'] != RUN:
            raise ValueError('The replay requires its own immutable run')
        if command == 'dispatch':
            dispatch(rt, args['runtime'], freeze=completion)
        else:
            worker(rt, args['shard'])
