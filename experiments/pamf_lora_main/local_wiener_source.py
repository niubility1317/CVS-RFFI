"""Freeze and assess packet-local complex Wiener recovery on LoRa source V."""

import argparse
from functools import partial
import os
from pathlib import Path, PurePosixPath
from time import perf_counter

import numpy as np
import torch

from experiments.cric_dura_comparison.io import read, write

from .channel_blind import generic_observation
from .channel_blind_source import CAL, panel, receive_only, setup
from .complex_vector_source import load_reader, score
from .data import SOURCE, VIEWS
from .denoise_source import EVAL_PER_RX, summarize
from .dura_repair import PacketRestorer, complex_iq, real_iq
from .full_packet_publish import DATA_RUN
from .full_packet_views import source_contract
from .joint_unroll_source import coordinates, source_control
from .local_wiener import LocalWiener
from .publish import release, release_commit
from .refine_runner import select_gpu, start
from .residual_dura import RUN as RESIDUAL_RUN
from .runner import activate
from .spectral_unroll import ROUTES


RUN = '20261123-phase1-lora-local-wiener-source-m2-r01'
MODES = ('direct', 'reference', 'diagonal_half', 'covariance_half', 'covariance_full')
CAL_MARGIN = 1.05


@torch.no_grad()
def calibrate(rt):
    device = 'cuda:0'
    tracker = PacketRestorer('consensus').to(device).eval()
    model = LocalWiener().to(device).eval()
    begun = perf_counter()
    (_, u, _), _ = setup(rt, device)
    ratios = []
    for iq in u.split(32):
        y = coordinates(tracker, iq)[0]
        _, _, ratio = model.analyze(y)
        ratios.append(ratio)
    scores = torch.cat(ratios)
    if not torch.isfinite(scores).all() or len(scores) != CAL * len(SOURCE):
        raise ValueError('Invalid fixed U-only noise calibration')
    write(Path(rt['run_root'])/'threshold.json', dict(threshold=float(CAL_MARGIN*scores.max()),
          rows=len(scores), wall_seconds=perf_counter()-begun, optimizer_updates=0,
          parameters=0, F_reads=0, V_reads=0, TX_reads=0, target_reads=0))


def threshold(root):
    record = read(root/'threshold.json')
    value = record['threshold']
    if (record['rows'] != CAL * len(SOURCE) or not np.isfinite(value) or value < 0
            or any(record[k] for k in ('F_reads', 'V_reads', 'TX_reads', 'target_reads'))):
        raise ValueError('Expected a finite source U-only threshold')
    return value


def recover(model, tracker, iq, gate):
    y, rotation, _, rms = coordinates(tracker, iq)
    spectrum, noise, relative = model.analyze(y)
    active = relative > gate
    outputs = [iq.clone() for _ in MODES[2:]]
    if bool(active.any()):
        z, s, n = y[active], spectrum[active], noise[active]
        base = rotation[active]*rms[active]
        diag = real_iq(model.restore(z, s, n, correlated=False)*base)
        cov = real_iq(model.restore(z, s, n, correlated=True)*base)
        for result, delta in zip(outputs, (.5*diag, .5*cov, cov), strict=True):
            result[active] = iq[active]+delta
    return outputs, int(active.sum())


def render(iq, _unused, baseline, tracker, models, *, gate):
    reference, _ = baseline(iq)
    values, active = recover(models[0], tracker, iq, gate)
    return torch.stack((iq, reference, *values)), active


@torch.no_grad()
def smoke(rt, gate):
    device = 'cuda:0'
    root = Path(rt['run_root'])
    (_, u, _), contract = setup(rt, device, smoke=True)
    reader = load_reader(root, contract, device)
    _, _, reference_contract, baseline, tracker = source_control(rt, 'full', device)
    if contract != reference_contract:
        raise ValueError('Expected the same source LoRa contract')
    model = LocalWiener().to(device).eval()
    random = torch.Generator(device=device).manual_seed(2026112304)
    noisy = real_iq(generic_observation(complex_iq(u), random))
    z = coordinates(tracker, noisy)[0]
    _, _, relative = model.analyze(z)
    disturbed = noisy[relative.argmax().view(1)]
    waves, active = render(torch.cat((u[:1], disturbed)), None, baseline, tracker, (model,), gate=gate)
    logits = reader(waves.flatten(0, 1))[0]
    if logits.shape != (2*len(MODES), 10) or not torch.isfinite(logits).all() or not active:
        raise ValueError('Real LoRa reader, baseline or active recovery smoke failed')
    results = {}
    for name, x in (('source_clean', u[:1]), ('generic_disturbance', disturbed)):
        times = []
        for i in range(28):
            torch.cuda.synchronize()
            begun = perf_counter()
            _, enabled = recover(model, tracker, x, gate)
            torch.cuda.synchronize()
            if i >= 8:
                times.append((perf_counter()-begun)*1000)
        results[name] = dict(median_ms=float(np.median(times)), active=bool(enabled))
    write(root/'smoke.json', dict(status='PASS', rows=2, V_reads=0, TX_reads=0, target_reads=0,
          b1_three_readouts=results, device=torch.cuda.get_device_name()))


def freeze(root, route, _source, chosen, ids, predictions, metrics, modes):
    write(root/f'{route}-predictions.json', dict(ids=ids[chosen].tolist(), modes=list(modes),
          rows={key: value.tolist() for key, value in predictions.items()}, metrics=metrics))


def predict(rt, route, gate):
    model = LocalWiener().to('cuda:0').eval()
    score(rt, route, panel=panel, modes=MODES, render=partial(render, gate=gate),
          frozen_models=(model,), replay_fn=receive_only, require_clean_bypass=False, finalize=freeze)


def read_frozen(root, route):
    row = read(root/f'{route}-predictions.json')
    n = EVAL_PER_RX*len(SOURCE)
    ids = row['ids']
    result = {key: np.asarray(value) for key, value in row['rows'].items()}
    if (len(ids) != n or len(set(ids)) != n or row['modes'] != list(MODES)
            or set(result) != set(VIEWS) or set(row['metrics']) != set(VIEWS)
            or any(x.shape != (len(MODES), n) or x.dtype.kind not in 'iu'
                   or (x<0).any() or (x>=10).any() for x in result.values())):
        raise ValueError('Both complete source V route matrices are required')
    return ids, result, row['metrics']


def score_frozen(rt):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('An independent CPU scorer is required')
    root = Path(rt['run_root'])
    source = Path(rt['source_root'])
    frozen = {route: read_frozen(root, route) for route in ROUTES}
    ids = np.asarray(source_contract(source)['role_ids']['V'])
    position = {pid: i for i, pid in enumerate(ids)}
    full_ids = frozen['full'][0]
    if full_ids != frozen['residual'][0] or not set(full_ids) <= set(position):
        raise ValueError('Source V physical IDs must match before scoring')
    chosen = np.asarray([position[key] for key in full_ids])
    with np.load(source/'V_metadata.npz', allow_pickle=False) as meta:
        if not np.array_equal(panel(np.asarray(SOURCE)[meta['rx_i']]), chosen):
            raise ValueError('Expected the fixed balanced source panel')
    for route, (_, predictions, metrics) in frozen.items():
        summarize(root, route, source, chosen, ids, predictions, metrics, MODES,
                  predictions_written=True,
                  comparisons=(('covariance_half', 'diagonal_half'), ('covariance_half', 'direct')))
    print('LORA_LOCAL_WIENER_SOURCE_SCORED', flush=True)


def dispatch(rt):
    root = Path(rt['run_root'])
    root.mkdir(exist_ok=False)
    status = root/'state.json'
    launch = partial(start, rt, module=rt['runner'])
    try:
        for action in ('calibrate', 'smoke'):
            gpu = select_gpu()
            write(status, dict(status=action.upper(), gpu=gpu), exclusive=False)
            if launch(action, 'source', gpu).wait():
                raise RuntimeError(f'LoRa local Wiener {action} failed')
        gpus = [select_gpu()]
        gpus.append(select_gpu(exclude=gpus))
        write(status, dict(status='PREDICT', gpus=dict(zip(ROUTES, gpus, strict=True))), exclusive=False)
        workers = [launch('evaluate', route, gpu) for route, gpu in zip(ROUTES, gpus, strict=True)]
        if any([worker.wait() for worker in workers]):
            raise RuntimeError('LoRa local Wiener source predictions failed')
        write(status, dict(status='SCORE'), exclusive=False)
        if launch('score', 'source').wait():
            raise RuntimeError('LoRa local Wiener truth-last scoring failed')
        write(status, dict(status='SOURCE_SCORED'), exclusive=False)
    except Exception as error:
        write(status, dict(status='FAILED', error=str(error)), exclusive=False)
        raise


def publish(repo, output, context, ssh_config):
    cfg = read(context)
    rt = {key: cfg[key] for key in ('project', 'python', 'frozen_sources')}
    project = PurePosixPath(rt['project'])
    rt.update(run_id=RUN, runner=__spec__.name, code_commit=release_commit(repo),
              run_root=str(project/'runs'/RUN), release_root=str(project/'releases'/f'{RUN}-release'),
              source_root=str(project/'releases'/f'{DATA_RUN}-release'/'data'/'source'),
              parent_root=str(project/'runs'/RESIDUAL_RUN))
    release(repo, output, rt, ssh_config)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest='action', required=True)
    pub = actions.add_parser('publish')
    for key in ('repo', 'output', 'context', 'ssh-config'):
        pub.add_argument('--'+key, type=Path, required=True)
    for action in ('dispatch', 'calibrate', 'smoke', 'evaluate', 'score'):
        cmd = actions.add_parser(action)
        cmd.add_argument('runtime', type=Path)
        if action != 'dispatch':
            cmd.add_argument('name', choices=ROUTES if action == 'evaluate' else ('source',))
    args = vars(parser.parse_args())
    action = args.pop('action')
    if action == 'publish':
        publish(**args)
        return
    rt = read(args['runtime'])
    if action == 'dispatch':
        dispatch(rt)
    elif action == 'score':
        score_frozen(rt)
    else:
        activate(rt)
        if action == 'calibrate':
            calibrate(rt)
        else:
            gate = threshold(Path(rt['run_root']))
            if action == 'smoke':
                smoke(rt, gate)
            else:
                predict(rt, args['name'], gate)


if __name__ == '__main__':
    main()
