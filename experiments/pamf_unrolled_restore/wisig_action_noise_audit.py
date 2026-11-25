"""Source-only forward-action and noise audit with a separate simulator scorer."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from itertools import product
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys

import numpy as np

from experiments.cric_dura_comparison.io import read, write
from experiments.pamf_repl_view_shuffle.plan import VIEWS


RUN = '20261125-diagnostic-wisig-action-noise-s2026112501-r01'
FROZEN = '20261125-phase1-wisig-blind-capacity-stress-m2-r02'
SOURCE = '20261019-phase1-reception-restoration-selfsup-manysig-m6-r02'
PROFILES = ('default', 'strong')
ROUTES = ('full', 'residual')
CELLS = tuple(product(PROFILES, ROUTES, VIEWS[1:]))
SEED = 2026112501
PER_SESSION = 24
SOURCE_RX = (1, 3, 4, 6, 8)
SOURCE_DAYS = (1, 2, 3)
PACKETS = PER_SESSION * len(SOURCE_RX) * len(SOURCE_DAYS)


def source_panel(source):
    folder = source / 'validation'
    contract = read(source / 'complete.json')
    if (contract['status'], contract['source_V_count'], contract['target_reads']) != (
            'RECEPTION_SOURCE_CACHE_COMPLETE', 1800, 0):
        raise ValueError('Expected the registered source-only V panel')
    clean, ids, rx, day, roles = (
        np.load(folder / f'{name}.npy', allow_pickle=False)
        for name in ('clean', 'physical_ids', 'rx', 'day', 'roles'))
    if clean.shape != (1800, 2, 256) or set(roles) != {'V'} or len(set(ids)) != 1800:
        raise ValueError('Unexpected source validation panel')
    rng = np.random.default_rng(SEED)
    cols = [rng.choice(np.flatnonzero((rx == r) & (day == d)), PER_SESSION, replace=False)
            for r, d in product(SOURCE_RX, SOURCE_DAYS)]
    selected = np.concatenate(cols)
    return clean[selected], ids[selected], [f'rx{rx[i]}:day{day[i]}:eq1' for i in selected], selected


def snapshot(job):
    from leo_practical import ChannelStream, ResidualChannel, receiver_for_session, stable_seed

    cfg, clean, ident, session = job
    seed = stable_seed(2026101901, 'restoration/validation/draw0', cfg.scenario, str(ident))
    receiver = receiver_for_session(cfg, 2026101902, session)
    cls = ChannelStream if cfg.processing_route == 'full' else ResidualChannel
    y, meta, trace = cls(cfg, seed, receiver).process(clean, return_traces=True, return_operator=True)
    agc = 10.0**(meta['agc_gain_db']/20)
    signal, noise = (agc*trace[name] for name in ('pre_agc_signal', 'pre_agc_noise'))
    np.testing.assert_allclose(y, signal+noise, rtol=2e-6, atol=2e-7)
    # The registered no-equalizer routes use only rotation and widely linear IQ
    # processing on the injected white noise, followed by the observed AGC.
    total_gain = np.abs(trace['operator_main'])**2 + np.abs(trace['operator_image'])**2
    variance = meta['added_noise_variance_reference'] * agc**2 * total_gain.mean()
    return y, signal, noise, variance


def generate(root, source):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'code'))
    from leo_practical import Config
    from .wisig_dura_stress_data import harder

    clean, ids, sessions, selected = source_panel(source)
    waveform = clean[:, 0] + 1j*clean[:, 1]
    observed, signal, noise, sigma = [], [], [], []
    with ProcessPoolExecutor(max_workers=8) as pool:
        for profile, route, scene in CELLS:
            cfg = Config(fs_hz=25e6, fc_hz=2462e6, scenario=scene, processing_route=route,
                         mode='post_sync', equalization_enabled=False, boundary='reflect')
            if profile == 'strong':
                cfg = harder(cfg)
            rows = list(pool.map(snapshot, ((cfg, x, i, s) for x, i, s in zip(waveform, ids, sessions)),
                                 chunksize=24))
            y, exact, injected, variance = map(np.asarray, zip(*rows))
            if profile == 'default':
                folder = 'validation_full' if route == 'full' else 'validation'
                saved = np.load(source / folder / f'{scene}.npy', mmap_mode='r', allow_pickle=False)
                units = np.stack((y.real, y.imag), 1)
                np.testing.assert_allclose(units, saved[selected], rtol=2e-5, atol=2e-6)
            observed.append(np.stack((y.real, y.imag), 1))
            signal.append(exact)
            noise.append(injected)
            sigma.append(variance)
            print('SIM_IQ_WRITTEN', profile, route, scene, PACKETS, flush=True)
    np.save(root/'received.npy', np.stack(observed), allow_pickle=False)
    np.savez(root/'scorer_reference.npz', clean=waveform, signal=signal, noise=noise, variance=sigma)


def predict(iq_path, output, frozen):
    import torch
    from experiments.cric_dura_comparison.io import tensor
    from .wisig_dura_stress_fit import frozen_fit, stages
    from .wisig_receiver_estimates import receive_coordinates

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    observed = np.load(iq_path, mmap_mode='r', allow_pickle=False)
    if observed.shape != (len(CELLS), PACKETS, 2, 256):
        raise ValueError('Expected the registered IQ-only input')
    result = {}
    with torch.no_grad():
        for profile in PROFILES:
            model, gate = frozen_fit(frozen, profile)
            rows = {key: [] for key in ('main', 'image', 'echo', 'variance', 'repeat')}
            for index, cell in enumerate(observed):
                for part in np.array_split(cell, 4):
                    iq = tensor(part, 'cuda:0')
                    _, _, action, variance = stages(model, iq)
                    repeat = receive_coordinates(iq)[-1]
                    fields = (action.main[:, 0], action.image[:, 0], action.echo[:, 0],
                              variance.mean(dim=(1,2)), repeat.flatten())
                    if not all(torch.isfinite(value).all() for value in fields):
                        raise ValueError('Nonfinite blind estimator output')
                    for key, value in zip(rows, fields):
                        dtype = np.complex64 if value.is_complex() else np.float32
                        rows[key].append(np.asarray(value.cpu().tolist(), dtype=dtype))
                if index == 0:
                    print('SOURCE_ONLY_REAL_CHECKPOINT_SMOKE_PASS', profile, flush=True)
            result.update({f'{profile}_{key}': np.concatenate(value)
                           for key, value in rows.items()})
            result[f'{profile}_gate'] = gate
    np.savez(output, **result)


def quantities(clean, obs, expected, injected, variance, coeffs):
    """Compare outputs in the same packet-wide gain frame used by training."""
    scale = np.sqrt(np.mean(np.abs(obs)**2, axis=-1))
    gain = np.sum(clean.conj()*obs, axis=-1) / np.sum(np.abs(clean)**2, axis=-1)
    ref = clean * (gain/scale)[:, None]
    exact = expected / scale[:, None]
    main, image, echo, calibrated, repeat = coeffs
    delayed = np.pad(ref[:, :-1], ((0,0), (1,0)))
    prediction = main*ref + image*ref.conj() + echo*delayed
    denom = np.mean(np.abs(exact)**2, axis=-1)
    baseline = np.mean(np.abs(ref-exact)**2, axis=-1) / denom
    error = np.mean(np.abs(prediction-exact)**2, axis=-1) / denom
    true_variance = variance / scale**2
    empirical_variance = np.mean(np.abs(injected)**2, axis=-1) / scale**2
    angle = np.angle(np.sum(obs[:, 68:148]*obs[:, 48:128].conj(), axis=-1))
    rotation = np.exp(1j*angle)[:, None]
    repeated_signal = .5*np.mean(np.abs(expected[:, 68:148]-rotation*expected[:, 48:128])**2, axis=-1)
    repeated_noise = .5*np.mean(np.abs(injected[:, 68:148]-rotation*injected[:, 48:128])**2, axis=-1)
    return dict(identity_nmse=baseline, action_nmse=error, repeat=repeat, learned=calibrated,
                oracle=true_variance, empirical=empirical_variance,
                repeat_signal_to_noise=repeated_signal / repeated_noise)


def describe(rows, gate):
    r = {key: np.concatenate([v[key] for v in rows]) for key in rows[0]}
    quantiles = lambda v: np.quantile(v, (.1,.5,.9)).tolist()
    noise = {}
    for key in ('repeat', 'learned', 'empirical'):
        ratio = r[key] / r['oracle']
        db = 10*np.log10(ratio)
        noise[key] = dict(ratio_p10_med_p90=quantiles(ratio), db_p10_med_p90=quantiles(db),
                          med_abs_db=float(np.median(np.abs(db))),
                          within_3db=float(np.mean(np.abs(db) <= 3)))
    return dict(records=len(r['oracle']), identity_nmse=quantiles(r['identity_nmse']),
                action_nmse=quantiles(r['action_nmse']),
                action_to_identity=quantiles(r['action_nmse']/r['identity_nmse']),
                action_better_fraction=float(np.mean(r['action_nmse'] < r['identity_nmse'])),
                oracle_variance=quantiles(r['oracle']), noise=noise, gate_pass=int((r['repeat'] > gate).sum()),
                repeat_signal_to_noise=quantiles(r['repeat_signal_to_noise']))


def score(root):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Run the oracle scorer separately on CPU')
    with np.load(root/'estimates.npz', allow_pickle=False) as est:
        if any(est[f'{p}_main'].shape != (len(CELLS)*PACKETS, 256) for p in PROFILES):
            raise ValueError('All blind estimates must be written before oracle scoring')
        observed = np.load(root/'received.npy', mmap_mode='r', allow_pickle=False)
        with np.load(root/'scorer_reference.npz', allow_pickle=False) as oracle:
            clean, signal, injected, variance = (oracle[name] for name in ('clean','signal','noise','variance'))
        result = {}
        for model in PROFILES:
            gate = float(est[f'{model}_gate'])
            names = ('main','image','echo','variance','repeat')
            measurements = {}
            for index, cell in enumerate(CELLS):
                selection = slice(index*PACKETS, (index+1)*PACKETS)
                iq = observed[index]
                fields = [est[f'{model}_{key}'][selection] for key in names]
                measurements[cell] = quantities(clean, iq[:,0]+1j*iq[:,1], signal[index],
                                                 injected[index], variance[index], fields)
            result[model] = {profile: {route: {view: describe(
                [row for cell,row in measurements.items()
                 if cell[:2] == (profile,route) and (view=='combined' or cell[2]==view)], gate)
                for view in ('combined', *VIEWS[1:])} for route in ROUTES} for profile in PROFILES}
    write(root/'scores.json', dict(status='SCORED', oracle_last=True, target_reads=0,
                                   source_packets=PACKETS, sample_seed=SEED, results=result))


def launch(rt, action, name, gpu=None):
    root = Path(rt['run_root'])
    # The estimator receives exactly an IQ path, output path and frozen weights.
    if action == 'blind':
        args = [str(root/'received.npy'), str(root/'estimates.npz'), rt['frozen_root']]
    elif action == 'generate':
        args = [str(root), rt['source_root']]
    else:
        args = [str(root)]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='' if gpu is None else str(gpu),
               OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    with (root/f'{action}.log').open('x') as log:
        process = subprocess.Popen([rt['python'], '-u', '-m', rt['runner'], action, *args],
                                   cwd=rt['release_root'], env=env, stdout=log, stderr=subprocess.STDOUT)
    print('AUDIT_WORKER', action, process.pid, gpu, flush=True)
    return process


def dispatch(rt):
    from experiments.pamf_lora_main.packet_runner import schedule

    root = Path(rt['run_root'])
    root.mkdir()
    state = root/'state.json'
    try:
        write(state, dict(status='SOURCE_IQ_GENERATION'))
        if launch(rt,'generate','source_v').wait():
            raise RuntimeError('Source IQ generator failed')
        write(state, dict(status='BLIND_ESTIMATION'), exclusive=False)
        inventory = subprocess.check_output(
            ['nvidia-smi','--query-gpu=index,utilization.gpu,memory.free','--format=csv,noheader,nounits'],
            text=True).splitlines()
        priority = sorted(((load, -free, idx) for idx,load,free in
                           (tuple(map(int,row.split(','))) for row in inventory)))
        rt = dict(rt, allowed_gpus=[i for _,_,i in priority], max_workers=1, max_compute_processes=None)
        schedule(rt,[('blind','both')],launch)
        write(state, dict(status='SCORING'), exclusive=False)
        if launch(rt,'score','source_v').wait():
            raise RuntimeError('Oracle scorer failed')
        write(state, dict(status='ANALYZED'), exclusive=False)
    except Exception as exc:
        write(state, dict(status='FAILED', error=str(exc)), exclusive=False)
        raise


def publish(repo, output, context, ssh_config):
    from experiments.pamf_lora_main import publish as transport

    ctx = read(context)
    base = PurePosixPath(ctx['project'])
    runs = base/'runs'
    rt = dict(ctx, run_id=RUN, runner=__spec__.name, code_commit=transport.release_commit(repo),
              release_root=str(base/'releases'/f'{RUN}-release'), run_root=str(runs/RUN),
              source_root=str(runs/SOURCE/'cache'), frozen_root=str(runs/FROZEN),
              external_releases=[str(runs/FROZEN)])
    transport.release(repo, output, rt, ssh_config, archive_extras=('tools/r17_activation_factorization.py',))


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='action', required=True)
    publish_cmd = sub.add_parser('publish')
    for field in ('repo','output','context','ssh-config'):
        publish_cmd.add_argument('--'+field, type=Path, required=True)
    for action, fields in (('dispatch',('runtime',)), ('generate',('root','source')),
                           ('blind',('iq_path','output','frozen')), ('score',('root',))):
        command = sub.add_parser(action)
        for field in fields:
            command.add_argument(field, type=Path)
    args = vars(parser.parse_args())
    action = args.pop('action')
    if action == 'dispatch':
        dispatch(read(args['runtime']))
    else:
        {'publish':publish, 'generate':generate, 'blind':predict, 'score':score}[action](**args)


if __name__ == '__main__':
    main()
