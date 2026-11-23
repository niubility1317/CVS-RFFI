"""Run fixed WiSig receiver engineering controls on paired, labelled-after-prediction source V."""

import argparse
from pathlib import Path
import os
import subprocess
import time

import numpy as np

from experiments.cric_dura_comparison.io import read, tensor, write
from .dura_baseline import R69_RUN, SEEDS, VIEWS, historical, model_pair
from .plan import RUN as READER_RUN
from .receiver_pilot import READOUTS

RUN = '20261123-phase1-wisig-preamble-receiver-source-m2-r02'
TIMING_RUN = '20261123-phase1-wisig-joint-preamble-timing-source-m2-r01'
SOURCE_RUN = '20261021-phase1-CRIC-DURA-comparison-m153-r02'
PAIR_RUN = '20261019-phase1-reception-restoration-selfsup-manysig-m6-r02'
N = 1800
SOURCE_RECEIVERS = (1, 3, 4, 6, 8)
ROUTES = ('full', 'residual')
SEED = 2026112307


def source_root(rt):
    return Path(rt['project']) / 'runs' / SOURCE_RUN / 'data' / 'F0'


def panels(rt):
    reference = Path(rt['project']) / 'runs' / PAIR_RUN / 'cache'
    return dict(full=reference / 'validation_full', residual=reference / 'validation')


def iq_array(path, n):
    a = np.load(path, mmap_mode='r', allow_pickle=False)
    if a.shape != (n, 2, 256) or a.dtype != np.float32:
        raise ValueError('Unexpected source-only observation')
    return a


def activate(rt):
    from .runner import activate as activate_packages

    activate_packages(rt)


def readouts(rt):
    if rt.get('timing'):
        from .receiver_timing import READOUTS as timing_readouts

        return timing_readouts
    return READOUTS


def calibrate(rt):
    """Freeze one clean disturbance floor before the recognizer or V are opened."""
    import torch
    from .receiver_pilot import estimate

    train_u = iq_array(source_root(rt) / 'U_s_x.npy', 56700)
    ix = np.sort(np.random.default_rng(SEED).choice(len(train_u), 4096, replace=False))
    x = tensor(np.ascontiguousarray(train_u[ix]), 'cpu')
    with torch.no_grad():
        if rt.get('timing'):
            from .receiver_timing import QUANTILE, Timing

            timing = Timing('cpu')
            margins = torch.cat([timing.margins(torch.complex(b[:, 0], b[:, 1])) for b in x.split(64)])
            floors = torch.quantile(margins, QUANTILE, dim=0).clamp_min(0)
            if not torch.isfinite(margins).all():
                raise ValueError('Nonfinite preamble timing estimate')
            thresholds = dict(zip(('boundary', 'joint'), floors.tolist()))
            write(Path(rt['run_root']) / 'timing_thresholds.json',
                  dict(clean_u_only=True, seed=SEED, n=len(ix), thresholds=thresholds, quantile=QUANTILE))
            print('TIMING_SOURCE_THRESHOLDS', thresholds, flush=True)
            return
        q = estimate(torch.complex(x[:, 0], x[:, 1])).disturbance
    floor = torch.quantile(q, .95).item()
    if not np.isfinite(floor) or floor <= 0:
        raise ValueError('Invalid frozen source U disturbance floor')
    write(Path(rt['run_root']) / 'floor.json', dict(clean_u_only=True, seed=SEED, n=len(ix), floor=floor))
    print('PILOT_SOURCE_FLOOR', floor, flush=True)


def clean_floor(rt):
    meta = read(Path(rt['run_root']) / 'floor.json')
    if not meta['clean_u_only'] or meta['n'] != 4096 or meta['seed'] != SEED:
        raise ValueError('Expected frozen source U floor')
    return meta['floor']


def restorer(rt, device):
    if rt.get('timing'):
        from .receiver_timing import Timing

        meta = read(Path(rt['run_root']) / 'timing_thresholds.json')
        if not meta['clean_u_only'] or meta['n'] != 4096 or meta['seed'] != SEED:
            raise ValueError('Expected frozen source U timing margins')
        timing = Timing(device)
        return lambda iq: timing.restore(iq, meta['thresholds'])
    from .receiver_pilot import restore

    floor = clean_floor(rt)
    return lambda iq: restore(iq, clean_floor=floor)


def smoke(rt):
    import torch

    modes = ('raw', 'R69', *readouts(rt))
    restore = restorer(rt, 'cuda:0')
    cfg = historical(rt, 'predict', SEEDS[0])
    reader, r69 = model_pair(rt, cfg, 'cuda:0')
    clean = iq_array(source_root(rt) / 'U_s_x.npy', 56700)
    x = tensor(np.ascontiguousarray(clean[:4]), 'cuda:0')
    with torch.inference_mode():
        variants, diagnostics = restore(torch.complex(x[:, 0], x[:, 1]))
        logits, _ = read_logits(reader, r69, x, variants)
    if set(logits) != set(modes) or any(v.shape != (4, 6) or not torch.isfinite(v).all() for v in logits.values()):
        raise ValueError('Real-checkpoint source smoke failed')
    if not all(torch.isfinite(v).all() for v in diagnostics.values()):
        raise ValueError('Nonfinite source pilot estimate')
    print('PILOT_REAL_SOURCE_SMOKE_PASS', flush=True)


def as_iq(iq):
    import torch

    return torch.stack((iq.real, iq.imag), dim=1)


def read_logits(reader, r69, x, variants):
    def classify(value):
        return reader.classify_features(reader.features(value))

    recovered = r69(x)
    decisions = dict(raw=classify(x), R69=recovered[3], **{
        key: classify(as_iq(value)) for key, value in variants.items()})
    return decisions, recovered[0]


def percentile(values):
    a = np.concatenate(values)
    return [float(x) for x in np.quantile(a, (.1, .5, .9))]


def predict(rt, route):
    import torch

    root = Path(rt['run_root'])
    alternatives = readouts(rt)
    modes = ('raw', 'R69', *alternatives)
    restore = restorer(rt, 'cuda:0')
    ref = panels(rt)[route]
    ids = np.load(ref / 'physical_ids.npy', allow_pickle=False)
    if len(ids) != N or len(np.unique(ids)) != N or set(np.load(ref / 'roles.npy', allow_pickle=False)) != {'V'}:
        raise ValueError('Expected existing fixed 1800-record source V panel')
    clean = iq_array(ref / 'clean.npy', N)
    row = root / route
    row.mkdir()
    cfg = historical(rt, 'predict', SEEDS[0])
    reader, r69 = model_pair(rt, cfg, 'cuda:0')
    elapsed = time.perf_counter()
    evidence = {}
    predictions = {}
    for view in VIEWS:
        observations = iq_array(ref / (view + '.npy'), N)
        logits = {m: [] for m in modes}
        telemetry = {}
        errors = {m: [] for m in modes}
        changes = {m: [] for m in alternatives}
        for at in range(0, N, 128):
            x = tensor(np.ascontiguousarray(observations[at:at + 128]), 'cuda:0')
            target = tensor(np.ascontiguousarray(clean[at:at + 128]), 'cuda:0')
            with torch.inference_mode():
                variants, diag = restore(torch.complex(x[:, 0], x[:, 1]))
                ds, recovered = read_logits(reader, r69, x, variants)
                for mode, value in ds.items():
                    if value.shape != (len(x), 6) or not torch.isfinite(value).all():
                        raise ValueError('Invalid frozen default-head prediction')
                    logits[mode].append(np.asarray(value.argmax(1).tolist(), dtype=np.int64))
                wave = dict(raw=x, R69=recovered, **{m: as_iq(v) for m, v in variants.items()})
                scale = target.square().sum(dim=(1, 2)).clamp_min(1e-8)
                for mode, z in wave.items():
                    error = (z - target).square().sum(dim=(1, 2)) / scale
                    errors[mode].append(np.asarray(error.tolist(), dtype=np.float32))
                for mode in alternatives:
                    changes[mode].append(np.asarray((wave[mode] != x).flatten(1).any(dim=1).tolist(), dtype=bool))
                for key, vector in diag.items():
                    telemetry.setdefault(key, []).append(np.asarray(vector.tolist()))
        for mode in modes:
            predictions[f'{mode}__{view}'] = np.concatenate(logits[mode])
        evidence[view] = dict(estimate_quantiles={k: percentile(v) for k, v in telemetry.items()},
                              nmse_quantiles={k: percentile(v) for k, v in errors.items()},
                              changed={k: int(np.concatenate(v).sum()) for k, v in changes.items()})
        print('PILOT_SOURCE_VIEW', route, view, N, flush=True)
    np.savez(row / 'predictions.npz', ids=ids, **predictions)
    write(row / 'complete.json', dict(status='PREDICTIONS_COMPLETE', n=N,
          route=route, seed=SEEDS[0], modes=modes, views=VIEWS, telemetry=evidence,
          predictor_reads_truth=False, elapsed_seconds=time.perf_counter()-elapsed))


def load_predictions(rt, route):
    modes = ('raw', 'R69', *readouts(rt))
    output = Path(rt['run_root']) / route
    done = read(output / 'complete.json')
    if (done['status'], done['route'], done['seed'], done['n'], done['views'], done['modes']) != (
            'PREDICTIONS_COMPLETE', route, SEEDS[0], N, list(VIEWS), list(modes)):
        raise ValueError('Incomplete frozen source prediction row')
    with np.load(output / 'predictions.npz', allow_pickle=False) as data:
        if set(data.files) != {'ids', *(f'{mode}__{view}' for mode in modes for view in VIEWS)}:
            raise ValueError('Unexpected source prediction keys')
        ids = data['ids']
        decisions = {key: data[key] for key in data.files if key != 'ids'}
    if any(a.shape != (N,) or a.dtype.kind not in 'iu' or (a < 0).any() or (a >= 6).any()
           for a in decisions.values()):
        raise ValueError('Invalid frozen source predictions')
    return ids, decisions, done


def cpu_score(rt):
    from experiments.cric_dura_comparison.evaluation import metrics

    modes = ('raw', 'R69', *readouts(rt))
    loaded = {r: load_predictions(rt, r) for r in ROUTES}
    ids, _, _ = loaded['full']
    source_panels = panels(rt)
    rx = np.load(source_panels['full'] / 'rx.npy', allow_pickle=False)
    day = np.load(source_panels['full'] / 'day.npy', allow_pickle=False)
    for route in ROUTES:
        panel = source_panels[route]
        if (not np.array_equal(ids, loaded[route][0])
                or not np.array_equal(ids, np.load(panel / 'physical_ids.npy', allow_pickle=False))
                or not np.array_equal(rx, np.load(panel / 'rx.npy', allow_pickle=False))
                or not np.array_equal(day, np.load(panel / 'day.npy', allow_pickle=False))):
            raise ValueError('Physical V predictions differ between routes')
    if set(rx.tolist()) != set(SOURCE_RECEIVERS) or set(day.tolist()) != {1, 2, 3}:
        raise ValueError('Unexpected source sessions')
    # Only this scorer opens V TX truth, after all modes and routes validate.
    with np.load(source_root(rt) / 'V_metadata.npz', allow_pickle=False) as v:
        index = {key: i for i, key in enumerate(v['ids'])}
        pos = [index[key] for key in ids]
        if not np.array_equal(rx, v['rx_i'][pos]) or not np.array_equal(day, v['day_i'][pos]):
            raise ValueError('Recorded V sample metadata differ')
        truth = v['y'][pos].astype(np.int64)
    if truth.shape != (N,) or set(truth) != set(range(6)):
        raise ValueError('Unexpected source V classes')

    rng = np.random.default_rng(SEED)
    strata = [np.flatnonzero((rx == site) & (day == date)) for site in SOURCE_RECEIVERS for date in (1, 2, 3)]
    draw = np.concatenate([rng.choice(ix, size=(2000, len(ix))) for ix in strata], axis=1)
    report = {}
    for route in ROUTES:
        _, predictions, proof = loaded[route]
        metrics_by_view = {}
        control = {v: predictions[f'raw__{v}'] == truth for v in VIEWS}
        for view in VIEWS:
            metrics_by_view[view] = {}
            for mode in modes:
                pred = predictions[f'{mode}__{view}']
                hit = pred == truth
                item = metrics(truth, pred, 6)
                item.update(correct=int(hit.sum()), repairs=int((hit & ~control[view]).sum()),
                            harms=int((~hit & control[view]).sum()),
                            rx_net={str(i): int((hit[rx == i].astype(int) - control[view][rx == i]).sum())
                                    for i in SOURCE_RECEIVERS})
                metrics_by_view[view][mode] = item
        six = {}
        for mode in modes:
            hit = sum((predictions[f'{mode}__{v}'] == truth).astype(np.int16) for v in VIEWS[1:])
            inner = {}
            for name in ('raw', 'R69'):
                baseline = sum((predictions[f'{name}__{v}'] == truth).astype(np.int16) for v in VIEWS[1:])
                difference = hit - baseline
                inner[name] = dict(net=int(difference.sum()),
                    ci=[float(t) for t in np.quantile(difference[draw].sum(axis=1), (.025, .975))])
            six[mode] = dict(correct=int(hit.sum()), accuracy=float(hit.mean()/6), pairs=inner)
        report[route] = dict(views=metrics_by_view, six=six, prediction=proof)
    write(Path(rt['run_root']) / 'source_scores.json', dict(status='SOURCE_SCORED',
          truth_after_complete_predictions=True, target_reads=0, receiver_control_trainable_parameters=0,
          n=N, frozen_recognizer_seed=SEEDS[0], routes=report))
    print('PILOT_SOURCE_TRUTH_LAST_DONE', flush=True)


def launch(rt, kind, name, gpu=None):
    root = Path(rt['run_root'])
    log = root / 'logs' / f'{kind}-{name}.log'
    cmd = [rt['python'], '-u', '-m', __spec__.name, kind, str(Path(rt['release_root']) / 'runtime.json'), name]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='' if gpu is None else str(gpu),
               OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    with log.open('x', encoding='utf-8') as f:
        process = subprocess.Popen(cmd, cwd=rt['release_root'], env=env,
                                   stdin=subprocess.DEVNULL, stdout=f, stderr=subprocess.STDOUT)
    print('PILOT_SOURCE_STARTED', kind, name, process.pid, gpu, flush=True)
    return process


def dispatch(rt):
    from experiments.pamf_lora_main.packet_runner import schedule

    root = Path(rt['run_root'])
    root.mkdir()
    (root / 'logs').mkdir()
    try:
        if launch(rt, 'calibrate', 'source_u').wait():
            raise RuntimeError('Source U calibration failed')
        schedule(rt, [('smoke', 'source')], launch)
        schedule(rt, [('predict', route) for route in ROUTES], launch)
        if launch(rt, 'cpu_score', 'source').wait():
            raise RuntimeError('Source truth-last scoring failed')
        write(root / 'state.json', dict(status='SOURCE_SCORED'))
    except Exception as error:
        write(root / 'state.json', dict(status='FAILED', error=str(error)))
        raise


def publish(repo, output, local_context, ssh_config, timing=False):
    from experiments.pamf_lora_main.publish import release, release_commit

    old = read(local_context)
    project = old['project']
    rt = {k: old[k] for k in ('project', 'python', 'frozen_sources')}
    run = TIMING_RUN if timing else RUN
    rt.update(run_id=run, run_root=f'{project}/runs/{run}',
              release_root=f'{project}/releases/{run}-release', runner=__spec__.name, timing=timing,
              code_commit=release_commit(repo), allowed_gpus=[3, 6, 2, 7, 0, 1, 4, 5],
              max_workers=2, max_compute_processes=4,
              recognizer_run=f'{project}/runs/{READER_RUN}',
              r69_checkpoint=f'{project}/runs/{R69_RUN}/stage2/native_map_prior/final.pt')
    release(repo, output, rt, ssh_config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='action', required=True)
    pub = commands.add_parser('publish')
    for key in ('repo', 'output', 'local-context', 'ssh-config'):
        pub.add_argument('--' + key, required=True, type=Path)
    pub.add_argument('--timing', action='store_true')
    for kind in ('dispatch', 'calibrate', 'smoke', 'predict', 'cpu_score'):
        entry = commands.add_parser(kind)
        entry.add_argument('runtime', type=Path)
        entry.add_argument('name', nargs='?')
    args = vars(parser.parse_args())
    action = args.pop('action')
    if action == 'publish':
        publish(**args)
    else:
        rt = read(args['runtime'])
        if action in ('smoke', 'predict'):
            activate(rt)
        if action == 'predict':
            if args['name'] not in ROUTES:
                raise ValueError('Expected registered source route')
            predict(rt, args['name'])
        else:
            globals()[action](rt)
