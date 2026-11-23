"""Isolated LoRa source attribution with the registered additive-noise trace."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import os
from pathlib import Path, PurePosixPath
from time import perf_counter

import numpy as np
import torch

from experiments.cric_dura_comparison.io import read, tensor, write

from . import local_wiener_source as observed
from .channel_blind_source import setup
from .complex_vector_source import load_reader
from .data import SOURCE, VIEWS
from .denoise_source import replay, summarize
from .dura_repair import PacketRestorer, complex_iq, no_tone, real_iq
from .full_packet_publish import DATA_RUN
from .full_packet_views import source_contract, source_role
from .joint_unroll_source import coordinates
from .local_wiener import LocalWiener
from .packet_physics import aligned_view
from .publish import release, release_commit
from .refine_runner import select_gpu, start
from .runner import activate


RUN = '20261123-diagnostic-lora-noise-floor-source-m2-r01'
PARENT_COMMIT = 'ffbfb47e2392c3b5c447ead6c92a98ff63754c1b'
MODES = (*observed.MODES, 'exact_variance', 'exact_psd', 'exact_subtraction_projected', 'exact_subtraction')
ORACLES = MODES[len(observed.MODES):]
ROUTES = observed.ROUTES
BANDS = ((1, 8), (9, 32), (33, 64), (65, 128))
BATCH = 18


def parent(rt):
    project = Path(rt['run_root']).parent.parent
    path = project/'runs'/observed.RUN
    bound = read(project/'releases'/f'{observed.RUN}-release'/'runtime.json')
    if (bound['run_id'], bound['code_commit'], bound['run_root'], bound['source_root']) != (
            observed.RUN, PARENT_COMMIT, str(path), rt['source_root']) or read(path/'state.json')['status'] != 'SOURCE_SCORED':
        raise ValueError('Expected the completed same-source receiver-only comparison')
    return path


def whitened_solve(model, y, spectrum, psd):
    """Replace the scalar white floor with measured-from-truth frequency power."""
    scale = psd.clamp_min(1e-8).sqrt()[:, None, None]
    whitened = spectrum / scale
    common = whitened.mean(dim=1, keepdim=True)
    distinct = whitened - common
    unit = y.real.new_ones(len(y))
    count = y.shape[1]
    common = model.filter(common, model.covariance(common), unit / count, correlated=True)
    distinct = model.filter(distinct, model.covariance(distinct), unit * (1 - 1/count), correlated=True)
    return no_tone(model.synthesize((common+distinct)*scale, y)-y)


def signal_metrics(y, corrections, noise):
    """Compare against noiseless received IQ in the same tracked coordinates."""
    truth = y-noise
    x = torch.fft.fft(torch.stack([y, *(y+delta for delta in corrections)]), norm='ortho')
    clean = torch.fft.fft(truth, norm='ortho')
    k = torch.fft.fftfreq(y.shape[-1], device=y.device).abs()*y.shape[-1]
    rows = []
    for lo, hi in BANDS:
        cols = (k >= lo) & (k <= hi)
        z, s = x[..., cols], clean[..., cols]
        denom = s.abs().square().sum(dim=(-2, -1)).clamp_min(1e-12)
        error = (z-s).abs().square().sum(dim=(-2, -1))/denom
        amplitude = (z.abs()-s.abs()).square().sum(dim=(-2, -1))/denom
        dot = (z * s.conj()).sum(dim=(-2, -1)).real
        coherence = dot/(z.abs().square().sum(dim=(-2, -1))*denom).clamp_min(1e-20).sqrt()
        rows.append(torch.stack((error, amplitude, coherence)))
    return np.asarray(torch.stack(rows).tolist(), dtype=np.float32)


@torch.no_grad()
def readouts(model, tracker, iq, epsilon, gate, *, clean=False):
    y, rotation, _, rms = coordinates(tracker, iq)
    basis = rotation*rms
    spectrum, measured, ratio = model.analyze(y)
    exact = complex_iq(epsilon)/basis
    active = ratio > gate
    zero = torch.zeros_like(y)
    if clean:
        if torch.count_nonzero(epsilon):
            raise ValueError('Registered clean view must have zero simulated noise')
        return [iq]*len(ORACLES), int(active.sum()), None
    noise_spectrum, _, _ = model.analyze(exact)
    psd = noise_spectrum.abs().square().mean(dim=(1, 2))
    scalar = psd.mean(dim=(1, 2))
    actual = scalar.clamp_min(1e-12)
    deltas = [zero.clone() for _ in ORACLES]
    estimated_delta = zero.clone()
    if active.any():
        x, frequencies, level = y[active], spectrum[active], measured[active]
        estimated_delta[active] = model.restore(x, frequencies, level, correlated=True)
        deltas[0][active] = model.restore(x, frequencies, scalar[active], correlated=True)
        deltas[1][active] = whitened_solve(model, x, frequencies, psd[active])
        deltas[2][active] = no_tone(-exact[active])
        deltas[3][active] = -exact[active]
    telemetry = dict(estimate_ratio=np.asarray((measured/actual).tolist(), dtype=np.float32),
                     true_outer_to_total=np.asarray((psd.flatten(1)[:, model.outer].mean(1)/actual).tolist(),
                                                    dtype=np.float32),
                     band_metrics=signal_metrics(y, [estimated_delta, *deltas], exact))
    return [iq+real_iq(delta*basis) for delta in deltas], int(active.sum()), telemetry


def telemetry_summary(parts):
    output = {k: np.quantile(np.concatenate([p[k] for p in parts]), (.1, .5, .9)).tolist()
              for k in ('estimate_ratio', 'true_outer_to_total')}
    values = np.concatenate([p['band_metrics'] for p in parts], axis=-1)
    output['band_metrics'] = np.median(values, axis=-1).tolist()
    return output


@torch.no_grad()
def smoke(rt):
    gate = observed.threshold(parent(rt))
    device = 'cuda:0'
    (_, u, _), contract = setup(rt, device, smoke=True)
    reader = load_reader(Path(rt['run_root']), contract, device)
    model = LocalWiener().to(device).eval()
    tracker = PacketRestorer('consensus').to(device).eval()
    rng = torch.Generator(device=device).manual_seed(2026112319)
    raw = u[:2]
    noise = .5*torch.randn(raw.shape, device=device, generator=rng)
    noisy = raw+noise
    waves, active, data = readouts(model, tracker, noisy, noise, gate)
    logits = reader(torch.cat(waves))[0]
    if logits.shape != (len(ORACLES)*len(raw), 10) or not torch.isfinite(logits).all() or active != len(raw):
        raise ValueError('No-query frozen reader or diagnostic smoke failed')
    torch.testing.assert_close(waves[-1], raw, rtol=1e-5, atol=3e-7)
    if not np.isfinite(data['band_metrics']).all():
        raise ValueError('Nonfinite diagnostic bands')
    write(Path(rt['run_root'])/'smoke.json', dict(status='PASS', rows=len(raw), diagnostic_only=True,
                                                V_reads=0, TX_reads=0, target_reads=0))


@torch.no_grad()
def predict(rt, route):
    device = 'cuda:0'
    root, source = Path(rt['run_root']), Path(rt['source_root'])
    old = parent(rt)
    gate = observed.threshold(old)
    bound_ids, paired, metrics = observed.read_frozen(old, route)
    contract = source_contract(source)
    raw, rx, ids, _ = source_role(source, contract, 'V')
    chosen = observed.panel(rx)
    if ids[chosen].tolist() != bound_ids:
        raise ValueError('LoRa diagnostic rows differ from the observed-only comparison')
    rows = list(zip(raw[chosen], rx[chosen], ids[chosen], strict=True))
    tracker = PacketRestorer('consensus').to(device).eval()
    model = LocalWiener().to(device).eval()
    f = load_reader(root, contract, device)
    official = np.load(old.parent/DATA_RUN/'source_views'/'V.npy', mmap_mode='r', allow_pickle=False) if route=='full' else None
    predictions, telemetry = {}, {}
    with ProcessPoolExecutor(max_workers=8) as pool:
        for i, view in enumerate(VIEWS):
            begun = perf_counter()
            capture, noise = (np.stack(v) for v in zip(*pool.map(partial(replay, route, view), rows,
                                                                 chunksize=8), strict=True))
            if official is not None and not np.array_equal(capture, official[i, chosen]):
                raise ValueError('Full trace replay differs from registered source V IQ')
            if route == 'residual' and view != 'clean':
                for j in range(0, len(rows), observed.EVAL_PER_RX):
                    iq, receiver, pid = rows[j]
                    ref = aligned_view(iq, view, receiver, pid, 'source-V', processing_route=route)
                    if not np.array_equal(ref, capture[j]):
                        raise ValueError('Residual trace alters registered receive IQ')
            labels, measured, active = [], [], 0
            for y, eps in zip(tensor(capture, device).split(BATCH), tensor(noise, device).split(BATCH), strict=True):
                iq, count, detail = readouts(model, tracker, y, eps, gate, clean=(view=='clean'))
                logits = f(torch.cat(iq))[0]
                if logits.shape != (len(ORACLES)*len(y), 10) or not torch.isfinite(logits).all():
                    raise ValueError('Wrong frozen diagnostic classifier output')
                active += count
                labels.append(np.asarray(logits.argmax(1).reshape(len(ORACLES), len(y)).tolist(), dtype=np.int64))
                if detail is not None:
                    measured.append(detail)
            if active != metrics[view]['active']:
                raise ValueError('Diagnostic changed the historical observed gate')
            predictions[view] = np.concatenate((paired[view], np.concatenate(labels, axis=1)))
            if view == 'clean' and not np.all(predictions[view][-len(ORACLES):] == paired[view][0]):
                raise ValueError('Clean diagnostic readouts differ from fixed direct decisions')
            if measured:
                telemetry[view] = telemetry_summary(measured)
            print('LORA_NOISE_FLOOR_DIAGNOSTIC', route, view, active, perf_counter()-begun, flush=True)
    observed.freeze(root, route, source, chosen, ids, predictions, metrics, MODES)
    write(root/f'{route}-isolated-estimates.json', dict(diagnostic_only=True, route=route, n=len(chosen),
          band_khz=BANDS, metrics=('normalized_complex_error', 'normalized_amplitude_error', 'real_coherence'),
          modes=['direct','covariance_full',*ORACLES], estimates=telemetry, target_reads=0))


def score(rt):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('An independent CPU scorer is required')
    root, source = Path(rt['run_root']), Path(rt['source_root'])
    old = parent(rt)
    frozen = {}
    all_ids = np.asarray(source_contract(source)['role_ids']['V'])
    positions = {key:i for i,key in enumerate(all_ids)}
    for route in ROUTES:
        bound_ids, paired, _ = observed.read_frozen(old, route)
        row = read(root/f'{route}-predictions.json')
        measures = read(root/f'{route}-isolated-estimates.json')
        decisions = {view:np.asarray(values) for view,values in row['rows'].items()}
        if (row['ids'] != bound_ids or not set(bound_ids) <= set(positions) or row['modes'] != list(MODES)
                or not measures['diagnostic_only'] or set(measures['estimates']) != set(VIEWS[1:])
                or set(decisions) != set(VIEWS) or set(row['metrics']) != set(VIEWS)
                or any(x.shape != (len(MODES), len(bound_ids)) or x.dtype.kind not in 'iu'
                       or (x<0).any() or (x>9).any() or not np.array_equal(x[:len(observed.MODES)],paired[v])
                       for v,x in decisions.items())):
            raise ValueError('Incomplete isolated diagnostic matrix or changed observed panel')
        frozen[route] = (bound_ids, decisions, row['metrics'])
    if frozen['full'][0] != frozen['residual'][0]:
        raise ValueError('Mismatched diagnostic physical IDs across routes')
    chosen = np.asarray([positions[key] for key in frozen['full'][0]])
    comparisons = (('exact_variance','covariance_full'), ('exact_psd','exact_variance'),
                   ('exact_subtraction_projected','covariance_full'), ('exact_subtraction','exact_subtraction_projected'))
    for route, (_, decisions, metrics) in frozen.items():
        summarize(root, route, source, chosen, all_ids, decisions, metrics, MODES,
                  predictions_written=True, comparisons=comparisons)
    print('LORA_NOISE_FLOOR_DIAGNOSTIC_SCORED', flush=True)


def dispatch(rt):
    root = Path(rt['run_root'])
    root.mkdir(exist_ok=False)
    status = root/'state.json'
    launch = partial(start, rt, module=rt['runner'])
    try:
        gpu = select_gpu()
        write(status, dict(status='SMOKE', gpu=gpu))
        if launch('smoke', 'source', gpu).wait():
            raise RuntimeError('Isolated noise trace smoke failed')
        gpus = [select_gpu()]
        gpus.append(select_gpu(exclude=gpus))
        write(status, dict(status='PREDICT', gpus=dict(zip(ROUTES,gpus,strict=True))), exclusive=False)
        jobs = [launch('evaluate', route, gpu) for route,gpu in zip(ROUTES,gpus,strict=True)]
        if any([p.wait() for p in jobs]):
            raise RuntimeError('Isolated source prediction failed')
        write(status, dict(status='SCORE'), exclusive=False)
        if launch('score', 'source').wait():
            raise RuntimeError('Independent diagnostic scoring failed')
        write(status, dict(status='SOURCE_SCORED', diagnostic_only=True), exclusive=False)
    except Exception as exc:
        write(status, dict(status='FAILED', error=str(exc)), exclusive=False)
        raise


def publish(repo, output, context, ssh_config):
    cfg = read(context)
    rt = {k:cfg[k] for k in ('project','python','frozen_sources')}
    project = PurePosixPath(rt['project'])
    rt.update(run_id=RUN, runner=__spec__.name, code_commit=release_commit(repo),
              run_root=str(project/'runs'/RUN), release_root=str(project/'releases'/f'{RUN}-release'),
              source_root=str(project/'releases'/f'{DATA_RUN}-release'/'data'/'source'),
              parent_root=str(project/'runs'/observed.RESIDUAL_RUN))
    release(repo, output, rt, ssh_config)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    commands = cli.add_subparsers(dest='action',required=True)
    pub = commands.add_parser('publish')
    for field in ('repo','output','context','ssh-config'):
        pub.add_argument('--'+field, type=Path, required=True)
    for action in ('dispatch','smoke','evaluate','score'):
        p = commands.add_parser(action)
        p.add_argument('runtime',type=Path)
        if action!='dispatch':
            p.add_argument('name',choices=ROUTES if action=='evaluate' else ('source',))
    args = vars(cli.parse_args())
    action = args.pop('action')
    if action=='publish':
        publish(**args)
        return
    rt = read(args['runtime'])
    if action=='dispatch':
        dispatch(rt)
    elif action=='score':
        score(rt)
    else:
        activate(rt)
        if action=='smoke':
            smoke(rt)
        else:
            predict(rt,args['name'])


if __name__=='__main__':
    main()
