"""Source-only native LoRa reference with DURA v1.0's two-stage structure."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter

import numpy as np
import torch

from experiments.cric_dura_comparison.bootstrap import numerical_policy
from experiments.cric_dura_comparison.io import read, tensor, write
from experiments.pamf_lora_main import gain_dura_screen as screen
from experiments.pamf_lora_main.w24_native_dura import (
    NATIVE_LENGTH, action_loss, capture_scale, coarse_loss, iq_error, source_pair,
)
from experiments.pamf_r65_four_route.stage_polyphase import StageDistilled, _stages
from experiments.pamf_restoration.models import complex_iq, pair
from experiments.pamf_uncertainty_restore.models import weighted_map


SEED, BATCH, DEVICE = 2026113057, 6, 'cuda:0'
RUN = '20261128-phase1-lora-dura-stage2-reference-s2026113057-r01'
PREDICT_RETRY = RUN.removesuffix('r01') + 'r02'
PHASES = (('coarse', 3000, 2e-4), ('action', 3000, 2e-4), ('joint', 12000, 1.5e-4))
MODES = (*screen.MODES[:3], 'dura_stage2')
COUNT, VIEWS = screen.FACTOR_COUNT, screen.SCREEN_VIEWS


class LoRaStage2(StageDistilled):
    def __init__(self):
        super().__init__()
        self.frozen.estimator.input_points = NATIVE_LENGTH
        with torch.no_grad():
            self.frozen.estimator.initial_kernel.zero_()
            self.frozen.estimator.initial_kernel[0] = 1

    def stage_outputs(self, received, context=None, *, solve_map=weighted_map):
        if context is None:
            context = self.estimate_context(received)
        return _stages(self.prior, received, context, solve_map, points=NATIVE_LENGTH)

    def recover(self, raw):
        scale = capture_scale(raw)
        return complex_iq(self(pair(raw / scale))) * scale


def lora_config(rt):
    config = read(Path(rt['lora_ref']) / 'runtime.json')
    if (config['run_id'], config['code_commit']) != (screen.LO_RA, screen.LO_RA_COMMIT):
        raise ValueError('Expected the registered full-packet LoRa source and frozen W24')
    return config


def fit(rt):
    from experiments.pamf_lora_main.full_packet_views import source_contract, source_role

    root = Path(rt['run_root'])
    source = Path(lora_config(rt)['source_root'])
    contract = source_contract(source)
    data = source_role(source, contract, 'U_s')[0]
    torch.manual_seed(SEED)
    model = LoRaStage2().to(DEVICE)
    rng = torch.Generator(device=DEVICE).manual_seed(SEED + 1)
    sampler = np.random.default_rng(SEED)
    torch.cuda.reset_peak_memory_stats()
    started = perf_counter()
    phase_cost = {}
    for phase, steps, rate in PHASES:
        model.requires_grad_(False).eval()
        if phase == 'joint':
            active = model
        else:
            active = {'coarse': model.frozen.coarse, 'action': model.frozen.estimator}[phase]
        active.requires_grad_(True).train()
        params = tuple(p for p in active.parameters() if p.requires_grad)
        optimizer = torch.optim.AdamW(params, lr=rate, weight_decay=1e-4)
        decayer = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps, eta_min=rate / 10)
        phase_started = perf_counter()
        for step in range(1, steps + 1):
            indices = sampler.choice(len(data), BATCH, replace=False)
            raw = tensor(np.ascontiguousarray(data[indices]).view(np.float32), DEVICE).view(torch.complex64)
            clean, observed, action_target, variance_target = source_pair(raw, rng)
            received = pair(observed)
            if phase == 'coarse':
                loss = coarse_loss(complex_iq(active(received)), clean)
            elif phase == 'action':
                base = model.frozen.coarse(received)
                action, variance = active(received, base)
                loss = action_loss(action, variance, action_target, variance_target, observed, rng)
            else:
                outs, _, _, _ = model.stage_outputs(received)
                loss = coarse_loss(complex_iq(outs[-1]), clean) + .15 * iq_error(complex_iq(outs[0]), clean)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(params, 1, error_if_nonfinite=True)
            if step == 1 and not bool(grad > 0):
                raise ValueError(f'No source gradient in {phase}')
            optimizer.step()
            decayer.step()
            if step == 1 or step % 100 == 0:
                print('LORA_STAGE2', phase, step, float(loss.detach()), flush=True)
        torch.cuda.synchronize()
        phase_cost[phase] = perf_counter() - phase_started

    with (root / 'dura_stage2.pt').open('xb') as stream:
        torch.save(model.state_dict(), stream)
    write(root / 'fit.json', dict(status='SOURCE_FROZEN', seed=SEED, batch=BATCH, phases=PHASES,
          source_role='U_s', panel_count=len(data), source_contract=contract['protocol'],
          initial='scratch', parameters=sum(p.numel() for p in model.parameters()),
          TX_reads=0, F_reads=0, V_reads=0, target_reads=0,
          seconds=perf_counter()-started, phase_seconds=phase_cost,
          peak_cuda_bytes=torch.cuda.max_memory_allocated()))


def previous_panel(rt):
    folder = Path(rt['factor_panel']) / 'lora'
    if read(folder / 'complete.json') != screen.panel_receipt('lora', COUNT, screen.FACTOR_MODES):
        raise ValueError('Expected the previously frozen blind 160-packet panel')
    with np.load(folder / 'predictions.npz', allow_pickle=False) as original:
        return original['ids'].copy(), {v: original[v][:3].copy() for v in VIEWS}


def load_frozen_stage2(root, device):
    proof = read(root / 'fit.json')
    if (proof['status'] != 'SOURCE_FROZEN' or proof['seed'] != SEED
            or proof['phases'] != [list(p) for p in PHASES] or proof['source_role'] != 'U_s'
            or proof['panel_count'] != 25200 or proof['source_contract'] != 'lora_full_packets_v1'
            or any(proof[k] for k in ('V_reads', 'TX_reads', 'F_reads', 'target_reads'))):
        raise ValueError('Expected final scheduled source-only LoRa checkpoint')
    g = LoRaStage2().to(device).eval().requires_grad_(False)
    g.load_state_dict(torch.load(root / 'dura_stage2.pt', map_location=device, weights_only=True), strict=True)
    return g


@torch.no_grad()
def predict(rt):
    root = Path(rt['run_root'])
    g = load_frozen_stage2(Path(rt.get('fit_from', root)), DEVICE)
    config = lora_config(rt)
    w24 = Path(config['w24_release_root'])
    sys.path.insert(0, str(w24))
    screen.archive_replay(rt).archive_imports(config, extra=(w24,))
    from experiments.pamf_lora_main import w24_joint_pamf as joint

    source = Path(config['source_root'])
    contract, reader, _, _ = joint.boot(config, DEVICE)

    def raw_tensor(packets):
        return tensor(np.ascontiguousarray(packets).view(np.float32), DEVICE).view(torch.complex64)

    def classify(z):
        return screen.classify_lora(joint, reader, z)

    sample = joint.source_role(source, contract, 'U_s')[0][:2]
    raw = raw_tensor(sample)
    result = g.recover(raw)
    if result.shape != raw.shape or not bool(torch.isfinite(result).all()) or classify(result).shape != (2,):
        raise ValueError('Real frozen LoRa checkpoint + E104 no-query smoke failed')
    write(root / 'smoke.json', dict(status='PASS', source_count=2, source_V_reads=0, target_reads=0))

    def measure(operation):
        for _ in range(8):
            operation()
        torch.cuda.synchronize()
        measurements = []
        for _ in range(20):
            start = perf_counter()
            operation()
            torch.cuda.synchronize()
            measurements.append(1000 * (perf_counter() - start))
        return dict(median_ms=float(np.median(measurements)), p95_ms=float(np.quantile(measurements, .95)))

    result = raw[:1]
    write(root / 'timing.json', dict(batch=1, cuda=torch.cuda.get_device_name(), precision='FP32',
          G=measure(lambda: g.recover(result)), GF=measure(lambda: classify(g.recover(result))),
          host_to_device_included=False, context='native8192 single source packet; eager solver'))
    rows, ids = screen.lora_panel(dict(rt, factor_from=rt['first_panel']), joint, source, contract)
    old_ids, previous = previous_panel(rt)
    if not np.array_equal(ids, old_ids):
        raise ValueError('Previously frozen paired LoRa comparison changed physical IDs')
    x = joint.native(source, 'V', contract['counts']['V'])[rows]
    jobs = list(zip(rows.tolist(), [joint.SOURCE[screen.RX['lora']]] * COUNT, x, strict=True))
    decisions = {}
    with ThreadPoolExecutor(max_workers=3) as pool:
        for view in VIEWS:
            packets = np.stack(list(pool.map(partial(joint.source_receive, view), jobs)))
            selected = np.concatenate([classify(g.recover(raw_tensor(packets[i:i+8])))
                                       for i in range(0, COUNT, 8)])
            decisions[view] = np.vstack((previous[view], selected))
    screen.freeze(root, 'lora', ids, decisions, count=COUNT, modes=MODES)


def independent_score(rt):
    from experiments.pamf_lora_main.full_packet_views import source_contract

    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Independent CPU scorer required')
    root = Path(rt['run_root'])
    if read(root / 'complete.json') != screen.panel_receipt('lora', COUNT, MODES):
        raise ValueError('Expected all paired predictions before connecting truth')
    with np.load(root / 'predictions.npz', allow_pickle=False) as recorded:
        ids = recorded['ids'].copy()
        matrix = np.stack([recorded[v] for v in VIEWS])
    old_ids, previous = previous_panel(rt)
    if (not np.array_equal(ids, old_ids) or len(set(ids.tolist())) != COUNT
            or matrix.shape != (2, len(MODES), COUNT) or matrix.dtype.kind not in 'iu'
            or not ((0 <= matrix) & (matrix < 10)).all()
            or any(not np.array_equal(matrix[i, :3], previous[v]) for i, v in enumerate(VIEWS))):
        raise ValueError('Incomplete frozen predictions or unmatched controls')
    source = Path(lora_config(rt)['source_root'])
    contract = source_contract(source)
    with np.load(source / 'V_metadata.npz', allow_pickle=False) as metadata:
        physical = metadata['ids'].tolist()
        if physical != contract['role_ids']['V']:
            raise ValueError('Score source role order changed')
        by_id = {name: i for i, name in enumerate(physical)}
        rows = np.array([by_id[i] for i in ids.tolist()])
        if not (metadata['rx_i'][rows] == screen.RX['lora']).all():
            raise ValueError('Score panel receiver changed')
        labels = metadata['y'][rows]
    hit = matrix == labels
    counts = hit.sum(-1).tolist()
    draws = np.random.default_rng(screen.FACTOR_SEED+2).integers(0, COUNT, size=(4000, COUNT))
    new_hit = hit[:, -1].astype(np.int8)
    pairs = {mode: screen.paired_panel(new_hit - hit[:, i], draws) for i, mode in enumerate(MODES[:-1])}
    write(root / 'scores.json', dict(status='ANALYZED', truth_last=True, target_reads=0,
          count=COUNT, correct={v: dict(zip(MODES, row, strict=True)) for v, row in zip(VIEWS, counts, strict=True)},
          paired=pairs))


def dispatch(runtime, rt):
    from experiments.pamf_lora_main.runner import available

    root = Path(rt['run_root'])
    root.mkdir()
    state = root / 'state.json'
    try:
        cards = available(rt['allowed_gpus'], max_compute_processes=8)
        card = next((idx for idx in rt['allowed_gpus'] if idx in cards), None)
        if card is None:
            raise RuntimeError('No preregistered GPU slot with 12 GiB free')
        actions = ('predict', 'score') if rt.get('fit_from') else ('fit', 'predict', 'score')
        for action in actions:
            gpu = '' if action == 'score' else str(card)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1')
            with (root / f'{action}.log').open('x') as log:
                child = subprocess.Popen([sys.executable, '-u', '-m', __spec__.name, action, str(runtime)],
                                         stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, env=env)
                write(state, dict(status='RUNNING', phase=action, pid=child.pid, gpu=gpu), exclusive=False)
                if child.wait():
                    raise RuntimeError(f'{action} worker failed, see {action}.log')
        write(state, dict(status='ANALYZED', target_reads=0), exclusive=False)
    except Exception as error:
        write(state, dict(status='FAILED', error=str(error)), exclusive=False)
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('publish', 'dispatch', 'fit', 'predict', 'score'))
    parser.add_argument('runtime', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--ssh-config', type=Path)
    parser.add_argument('--reuse-frozen', action='store_true')
    arg = parser.parse_args()
    rt = read(arg.runtime)
    if arg.action == 'publish':
        from experiments.pamf_lora_main.publish import release, release_commit

        rt = {key: rt[key] for key in ('project', 'python', 'frozen_sources')}
        project = Path(rt['project'])
        releases = project / 'releases'
        runs = project / 'runs'
        run = PREDICT_RETRY if arg.reuse_frozen else RUN
        rt.update(run_id=run, code_commit=release_commit(Path.cwd()), runner=__spec__.name,
                  run_root=f'{runs.as_posix()}/{run}', release_root=f'{releases.as_posix()}/{run}-release',
                  lora_ref=f'{releases.as_posix()}/{screen.LO_RA}-release',
                  archive_root=f'{releases.as_posix()}/{screen.ARCHIVE}-release',
                  first_panel=f'{runs.as_posix()}/{screen.RUN}', factor_panel=f'{runs.as_posix()}/{screen.FACTOR_RUN}',
                  allowed_gpus=[0, 7, 2, 5, 6, 3])
        rt['external_releases'] = [rt[k] for k in ('lora_ref', 'archive_root', 'first_panel', 'factor_panel')]
        if arg.reuse_frozen:
            rt['fit_from'] = f'{runs.as_posix()}/{RUN}'
            rt['external_releases'].append(rt['fit_from'])
        release(Path.cwd(), arg.output, rt, arg.ssh_config)
    elif arg.action == 'dispatch':
        dispatch(arg.runtime, rt)
    elif arg.action == 'score':
        independent_score(rt)
    else:
        numerical_policy()
        (fit if arg.action == 'fit' else predict)(rt)


if __name__ == '__main__':
    main()
