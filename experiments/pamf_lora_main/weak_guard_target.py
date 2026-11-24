"""Replay one source-frozen LoRa guard-half candidate on both target routes."""

import argparse
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
from time import perf_counter


RUN = '20261124-phase1-lora-guard-half-target-m8-r02'
ORIGIN = '20261124-phase1-lora-rank-guard-extension-source-m2-r01'
ORIGIN_CODE = '0789ee3329648db6ee348c7fea56adf3a2171237'
DATA = '20261122-phase1-lora-full-aligned-data-r01'
RESIDUAL = '20261123-phase1-lora-dura-residual-practical-m1-r01'
ROUTES = ('full', 'residual')
SEEDS = (2026092701, 2026092702, 2026092703)
MODES = ('direct', 'guard_half')
SHARDS = 4


def matrix():
    from experiments.pamf_lora_main.data import TARGET
    return {f'{route}-{index}': (route, TARGET[index::SHARDS])
            for route in ROUTES for index in range(SHARDS)}


def expected_files(modes=MODES):
    from experiments.pamf_lora_main.data import TARGET, VIEWS
    return {f'{route}/{seed}/{receiver}__{view}__{mode}.npy'
            for route in ROUTES for seed in SEEDS for receiver in TARGET for view in VIEWS for mode in modes}


def completion(*, origin=ORIGIN, origin_code=ORIGIN_CODE, modes=MODES):
    from experiments.pamf_lora_main.data import TARGET, VIEWS
    from experiments.pamf_lora_main.full_packet_data import LIMIT
    return dict(status='PREDICTIONS_FROZEN', candidate_run=origin, origin_commit=origin_code,
                routes=list(ROUTES), seeds=list(SEEDS), modes=list(modes), receivers=list(TARGET),
                per_receiver=LIMIT, views=list(VIEWS), target_truth_read=False)


def source_inference(rt, device):
    import torch
    from experiments.cric_dura_comparison.io import read
    from experiments.pamf_lora_main.dura_repair import PacketRestorer
    from experiments.pamf_lora_main.full_adaptive import SET_METHOD, SET_RUN, case, complete_f
    from experiments.pamf_lora_main.full_packet_views import source_contract
    from experiments.pamf_lora_main.residual_dura import recognizer
    from experiments.pamf_lora_main.runner import activate
    from experiments.pamf_lora_main.weak_guard_source import MODES as source_modes
    from experiments.pamf_lora_main.weak_guard_source import frozen_guard, parent_model, recover

    # The original entry point imports its restoration dependencies before activation
    # makes the historical F and channel packages visible.
    activate(rt)
    runs = Path(rt['project']) / 'runs'
    contract = source_contract(Path(rt['source_root']))
    candidate = runs / ORIGIN
    if read(candidate / 'state.json')['status'] != 'SOURCE_SCORED':
        raise ValueError('Source candidate was not fully scored and frozen')
    model, threshold = parent_model(candidate, contract, device)
    gate, old_threshold = frozen_guard(candidate, contract, device)
    if threshold != old_threshold:
        raise ValueError('Source checkpoint and guard threshold disagree')
    tracker = PacketRestorer('consensus').to(device).eval()
    readers = {}
    for seed in SEEDS:
        weights = complete_f(runs / SET_RUN, contract, case(seed, SET_METHOD)) / 'model.pt'
        reader = recognizer(device).eval().requires_grad_(False)
        reader.load_state_dict(torch.load(weights, map_location=device, weights_only=True), strict=True)
        readers[seed] = reader
    slot = source_modes.index('guard_half') - 2

    def restore(iq):
        arms, active = recover(iq, tracker, model, gate, old_threshold)
        return arms[slot], active

    return readers, restore, contract


def blind_cache(rt, route):
    from experiments.cric_dura_comparison.io import read
    from experiments.pamf_lora_main.data import TARGET, VIEWS
    from experiments.pamf_lora_main.full_packet_data import LIMIT, PROTOCOL
    runs = Path(rt['project']) / 'runs'
    expected = dict(status='TARGET_READY', protocol=PROTOCOL, target_truth_read=False,
                    receivers=list(TARGET), per_receiver=LIMIT, views=list(VIEWS))
    if route == 'residual':
        expected['processing_route'] = 'residual'
    cache = runs / (RESIDUAL if route == 'residual' else DATA) / 'target_views'
    if read(cache / 'complete.json') != expected:
        raise ValueError(f'{route} target view differs from the existing frozen matrix')
    return cache


def smoke_disturbance(iq, rng):
    import torch

    options = dict(device=iq.device, generator=rng)
    draw = torch.complex(torch.randn(iq.shape, **options), torch.randn(iq.shape, **options))
    frequency = torch.fft.fftfreq(iq.shape[-1], device=iq.device)
    shape = (len(iq), 1, 1)
    angle = 2 * torch.pi * (frequency - torch.rand(shape, **options))
    depth = .95 * torch.rand(shape, **options)
    flat = torch.rand(shape, **options) < .40
    profile = (1 + torch.where(flat, 0., depth) * angle.cos()).sqrt()
    colored = torch.fft.ifft(torch.fft.fft(draw) * profile)
    return iq + .08 * colored / 2**.5


def run_worker(rt, stage, shard, *, bind=source_inference, modes=MODES, events='LORA_GUARD_TARGET'):
    import numpy as np
    import torch
    from experiments.cric_dura_comparison.io import tensor, write
    from experiments.pamf_lora_main.data import CLASSES, VIEWS
    from experiments.pamf_lora_main.full_packet_data import LIMIT
    from experiments.pamf_lora_main.full_packet_views import source_role
    from experiments.pamf_lora_main.packet_physics import chirp_windows
    from experiments.pamf_lora_main.dura_repair import complex_iq, real_iq

    device = 'cuda:0'
    root = Path(rt['run_root'])
    with torch.no_grad():
        readers, restore, contract = bind(rt, device)

        def classify(batch):
            recovered, active = restore(batch)
            result = {}
            for seed, reader in readers.items():
                decisions = []
                for sample in (batch, recovered) if active else (batch,):
                    logits = reader(sample)[0]
                    if logits.shape != (len(batch), len(CLASSES)) or not torch.isfinite(logits).all():
                        raise ValueError('Invalid frozen Tri-set prediction')
                    decisions.append(logits.argmax(1).tolist())
                result[seed] = (decisions[0], decisions[-1])
            return result, active

        if stage == 'smoke':
            iq, _, _, _ = source_role(Path(rt['source_root']), contract, 'U_s')
            quiet = tensor(np.stack([chirp_windows(iq[i]) for i in range(12)]), device)
            rng = torch.Generator(device=device).manual_seed(2026112481)
            disturbed = real_iq(smoke_disturbance(complex_iq(quiet), rng))
            raw_output, bypass = restore(quiet)
            _, active = classify(disturbed)
            classify(quiet)
            if not active or (bypass == 0 and not torch.equal(raw_output, quiet)):
                raise ValueError('Real checkpoint smoke did not exercise the selected readout')
            write(root / 'smoke.json', dict(status='PASS', source_U=len(quiet), active=active,
                                          clean_active=bypass, seeds=list(SEEDS), target_reads=0))
            return

        route, receivers = matrix()[shard]
        cache = blind_cache(rt, route)
        activation = {}
        started = perf_counter()
        for receiver in receivers:
            iq = np.load(cache / f'{receiver}.npy', allow_pickle=False, mmap_mode='r')
            if iq.shape != (len(VIEWS), LIMIT, 6, 2, 256) or iq.dtype != np.float32:
                raise ValueError('Target IQ shape differs from the published physical matrix')
            for view, sample in zip(VIEWS, iq, strict=True):
                rows = {seed: [[], []] for seed in SEEDS}
                count = 0
                for batch in tensor(sample, device).split(48):
                    result, active = classify(batch)
                    count += active
                    for seed, variants in result.items():
                        for row, labels in zip(rows[seed], variants, strict=True):
                            row.extend(labels)
                for seed, variants in rows.items():
                    for mode, labels in zip(modes, variants, strict=True):
                        path = root / 'predictions' / route / str(seed) / f'{receiver}__{view}__{mode}.npy'
                        with path.open('xb') as stream:
                            np.save(stream, np.asarray(labels, dtype=np.int16), allow_pickle=False)
                activation[f'{receiver}/{view}'] = count
                print(events + '_VIEW', route, receiver, view, count, flush=True)
        write(root / f'{shard}-complete.json', dict(status='COMPLETE', activation=activation,
                                                  elapsed_seconds=perf_counter()-started, truth_reads=0))


def dispatch(rt, runtime_path, *, entrypoint=Path(__file__).resolve(), modes=MODES,
             complete=completion, events='LORA_GUARD_TARGET'):
    from experiments.cric_dura_comparison.io import write
    from experiments.pamf_lora_main.refine_runner import select_gpu
    root = Path(rt['run_root'])
    root.mkdir(exist_ok=False)
    state = root / 'state.json'
    logs = root / 'logs'
    logs.mkdir()

    def launch(stage, row, gpu):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='1',
                   MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
        args = [rt['python'], '-u', str(entrypoint), stage, str(runtime_path), row]
        with (logs / f'{stage}-{row}.log').open('x', encoding='utf-8') as log:
            child = subprocess.Popen(args, env=env, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        print(events + '_LAUNCHED', stage, row, child.pid, gpu, flush=True)
        return child

    try:
        write(state, dict(status='SMOKE'))
        if launch('smoke', 'source', select_gpu()).wait():
            raise RuntimeError('No-query LoRa replay smoke failed')
        for route in ROUTES:
            for seed in SEEDS:
                (root / 'predictions' / route / str(seed)).mkdir(parents=True)
        command = ('nvidia-smi', '--query-gpu=index,utilization.gpu,memory.free', '--format=csv,noheader,nounits')
        inventory = (tuple(map(int, line.split(','))) for line in subprocess.check_output(command, text=True).splitlines())
        available = sorted((load, -free, gpu) for gpu, load, free in inventory if free >= 8192)
        slots = [gpu for _, _, gpu in available]
        if not slots:
            raise RuntimeError('No GPU with enough capacity for the LoRa target inference')
        rows = list(matrix())
        for offset in range(0, len(rows), len(slots)):
            batch = dict(zip(rows[offset:offset+len(slots)], slots))
            write(state, dict(status='PREDICT', gpus=batch), exclusive=False)
            workers = [launch('evaluate', row, gpu) for row, gpu in batch.items()]
            if any([job.wait() for job in workers]):
                raise RuntimeError('A LoRa target replay shard failed')
        files = {p.relative_to(root / 'predictions').as_posix() for p in (root / 'predictions').rglob('*.npy')}
        if files != expected_files(modes):
            raise ValueError('Missing or unexpected target predictions')
        write(root / 'predictions' / 'complete.json', complete())
        write(state, dict(status='PREDICTIONS_FROZEN', target_truth_read=False), exclusive=False)
    except Exception as error:
        write(state, dict(status='FAILED', error=str(error)), exclusive=False)
        raise


def publish(repo, output, context, ssh_config, *, run=RUN, runner=None):
    from experiments.cric_dura_comparison.io import read
    from experiments.pamf_lora_main.publish import release, release_commit
    cfg = read(context)
    project = PurePosixPath(cfg['project'])
    rt = {key: cfg[key] for key in ('project', 'python', 'frozen_sources')}
    rt.update(run_id=run, runner=runner or __spec__.name, code_commit=release_commit(repo),
              release_root=str(project / 'releases' / f'{run}-release'), run_root=str(project / 'runs' / run),
              source_root=str(project / 'releases' / f'{DATA}-release' / 'data' / 'source'))
    release(repo, output, rt, ssh_config)


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    actions = cli.add_subparsers(dest='action', required=True)
    pub = actions.add_parser('publish')
    for flag in ('repo', 'output', 'context', 'ssh-config'):
        pub.add_argument('--' + flag, type=Path, required=True)
    for action in ('dispatch', 'smoke', 'evaluate'):
        entry = actions.add_parser(action)
        entry.add_argument('runtime', type=Path)
        if action != 'dispatch':
            entry.add_argument('shard')
    args = vars(cli.parse_args())
    action = args.pop('action')
    if action == 'publish':
        publish(**args)
    else:
        runtime = args['runtime'].resolve()
        rt = json.loads(runtime.read_text(encoding='utf-8'))
        if action == 'dispatch':
            dispatch(rt, runtime)
        else:
            original = Path(rt['project']) / 'releases' / f'{ORIGIN}-release'
            provenance = json.loads((original / 'runtime.json').read_bytes())
            if provenance['code_commit'] != ORIGIN_CODE or provenance['run_id'] != ORIGIN:
                raise ValueError('Source inference release differs from its recorded execution')
            sys.path.insert(0, str(original))
            run_worker(rt, action, args['shard'])
