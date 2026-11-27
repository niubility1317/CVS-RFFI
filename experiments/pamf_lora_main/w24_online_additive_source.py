"""Replay independent additive noise with the rest of a source Full draw held fixed."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import os
from pathlib import Path, PurePosixPath
import shutil

import numpy as np
import torch

from experiments.cric_dura_comparison.io import read, tensor, write
from experiments.pamf_lora_main.full_packet_views import source_contract, source_role
from experiments.pamf_lora_main.runner import activate
from experiments.pamf_lora_main.w24_native_full import CONTEXT, DATA_RUN, PACKET_LEN, SOURCE, VIEWS, channel, hardware
from experiments.pamf_lora_main.w24_practical_supervision import NAMESPACE, SEED as BANK_SEED
from experiments.pamf_lora_main.w24_third_prior import PRACTICAL_CACHE_RUN


RUN = '20261128-data-lora-w24-online-additive-source-s2026112991-r01'
ISOLATION = dict(F_reads=0, TX_reads=0, V_reads=0, target_reads=0)


def replay_source_operator(packet, scene, receiver, physical_id):
    """Return one source signal and its per-sample additive-noise operator before AGC."""
    from leo_practical.channel import ChannelStream, normalize_input, stable_seed

    normalized, _ = normalize_input(packet)
    cfg = channel(scene)
    if cfg.mode != 'post_sync' or cfg.equalization_enabled:
        raise ValueError('The registered source Full redraw requires post-sync/no-EQ')
    delay = cfg.fractional_delay_half_length
    seed = stable_seed(2026112207, NAMESPACE, scene, receiver, str(physical_id))
    receiver_params = hardware(scene, f'{NAMESPACE}/{receiver}')
    padded = np.pad(normalized, (0, delay))
    preceding = np.zeros(CONTEXT, dtype=normalized.dtype)
    stream = ChannelStream(cfg, seed, receiver_params)
    y, meta, trace = stream.process(padded, preceding_context=preceding, return_traces=True, return_operator=True)
    if (meta['boundary'] != 'provided_context' or meta['channel_equalization_applied']
            or meta['numerical_filter_latency_samples'] != delay):
        raise ValueError('The source operator differs from the registered Full replay')

    signal = trace['pre_agc_signal']
    noise_std = np.sqrt(meta['added_noise_variance_reference'])
    main = trace['operator_main'] * noise_std
    image = trace['operator_image'] * noise_std
    limit = cfg.agc_limit_db if cfg.agc_enabled else 0.
    return (signal, main, image), (y, trace['pre_agc_noise'], seed, delay, limit)


def sample_additive(signal, main, image, delay, agc_limit_db, *, generator=None):
    """Draw pre-AGC noise, then apply the same block AGC to observation and source target."""
    standard = torch.randn(signal.shape, dtype=signal.dtype, device=signal.device, generator=generator)
    disturbed = signal + main * standard + image * standard.conj()
    power = disturbed.abs().square().mean(-1, keepdim=True).clamp_min(1e-24)
    limit = agc_limit_db[..., None]
    gain_db = torch.minimum(torch.maximum(-10. * power.log10(), -limit), limit)
    gain = torch.pow(10., gain_db / 20.)
    crop = slice(delay, delay + PACKET_LEN)
    return disturbed[..., crop] * gain, signal[..., crop] * gain


def assignments(receivers):
    rng = np.random.default_rng(BANK_SEED)
    scenes = np.full(len(receivers), -1, dtype=np.int8)
    for receiver in SOURCE:
        rows = np.flatnonzero(receivers == receiver)
        scenes[rows] = rng.permutation(np.arange(len(rows)) % len(VIEWS[1:]))
    if np.any(scenes < 0):
        raise ValueError('Unknown receiver in the registered source U role')
    return np.asarray(VIEWS[1:])[scenes]


def probe(iq, receivers, ids, scenes, previous):
    from leo_practical.channel import stable_seed

    torch.set_num_threads(2)
    generator = torch.Generator().manual_seed(2026112897)
    correlations, relative_errors = [], []
    receivers_seen, scenes_seen = set(), set()
    for rx in SOURCE:
        for scene in VIEWS[1:]:
            index = np.flatnonzero((receivers == rx) & (scenes == scene))[0]
            operators, (canonical, exact_noise, seed, delay, limit) = replay_source_operator(
                iq[index], scene, rx, ids[index])
            signal, main, image = operators
            nr = np.random.default_rng(stable_seed(seed, 'noise')).normal(size=(len(signal), 2))
            noise = (nr[:, 0] + 1j * nr[:, 1]) / np.sqrt(2.)
            reconstructed = main * noise + image * noise.conj()
            rel_error = np.linalg.norm(reconstructed-exact_noise) / np.linalg.norm(exact_noise)
            if rel_error >= 1e-10 or not np.array_equal(canonical[delay:delay+PACKET_LEN], previous[index, 0]):
                raise ValueError('Source operator did not reproduce its canonical source-bank trace')
            relative_errors.append(float(rel_error))
            arrays = [tensor(np.ascontiguousarray(x[None], dtype=np.complex64).view(np.float32), 'cpu')
                      .view(torch.complex64) for x in operators]
            pair = [sample_additive(*arrays, delay, torch.tensor([limit]), generator=generator) for _ in range(2)]
            a, b = [(y-s).flatten() for y,s in pair]
            if not torch.isfinite(a).all() or not torch.isfinite(b).all() or torch.equal(a, b):
                raise ValueError('Source additive redraw produced invalid or duplicated noise')
            correlations.append(float(torch.vdot(a,b).abs() / (a.norm()*b.norm())))
            receivers_seen.add(rx)
            scenes_seen.add(scene)
    return dict(status='PASS', source_role='U_s', packets=len(relative_errors), source_receivers=len(receivers_seen),
                practical_scenes=len(scenes_seen), max_noise_operator_relative_error=max(relative_errors),
                cross_draw_abs_coherence_median=float(np.median(correlations)),
                cross_draw_abs_coherence_max=max(correlations), original_source_replay_exact=True, **ISOLATION)


def cache_row(args):
    operator, (_, _, _, delay, limit) = replay_source_operator(*args)
    return np.stack(operator).astype(np.complex64, copy=False), delay, limit


def build(runtime):
    from experiments.pamf_lora_main.physical_inverse_source import source_trace_implementation

    rt = read(runtime)
    activate(rt)
    source_trace_implementation()
    os.nice(10)
    project = Path(rt['project'])
    root, source, original = (Path(rt[k]) for k in ('run_root', 'source_root', 'practical_cache'))
    data = project/'releases'/f'{DATA_RUN}-release'/'data'/'source'
    if (root != project/'runs'/RUN or source != data
            or original != project/'runs'/PRACTICAL_CACHE_RUN/'source_practical'):
        raise ValueError('Expected the registered source-only operator bank')
    root.mkdir()
    status = root/'state.json'
    try:
        iq, receivers, ids, _ = source_role(source, source_contract(source), 'U_s')
        stamp = read(original/'complete.json')
        if (stamp['status'] != 'SOURCE_TRACE_FROZEN' or stamp['count'] != len(iq)
                or stamp['data_role'] != 'U_s' or stamp['source_root'] != str(source)
                or stamp['namespace'] != NAMESPACE or stamp['seed'] != BANK_SEED or any(stamp[k] for k in ISOLATION)):
            raise ValueError('Expected the corresponding fixed source-only Practical bank')
        previous = np.load(original/'pairs.npy', allow_pickle=False, mmap_mode='r')
        scenes = assignments(receivers)
        delays = {channel(s).fractional_delay_half_length for s in VIEWS[1:]}
        if len(delays) != 1:
            raise ValueError('Full source profiles must share the native capture padding')
        (delay,) = delays
        size = PACKET_LEN + delay
        nbytes = len(iq)*3*size*np.dtype(np.complex64).itemsize
        if shutil.disk_usage(root).free < nbytes + 128*1024**2:
            raise RuntimeError('Insufficient disk space for the registered source operator bank')
        proof = probe(iq, receivers, ids, scenes, previous)
        write(root/'source-smoke.json', proof)
        print('W24_ONLINE_ADDITIVE_SOURCE_PROBE', proof, flush=True)
        write(status, dict(status='BUILDING_SOURCE_BANK', count=len(iq), **ISOLATION))
        bank = np.lib.format.open_memmap(root/'operators.npy', mode='w+', dtype=np.complex64, shape=(len(iq),3,size))
        limits = np.empty(len(iq), dtype=np.float32)
        jobs = zip(iq, scenes.tolist(), list(map(str, receivers)), list(map(str, ids)), strict=True)
        with ProcessPoolExecutor(max_workers=rt['workers']) as pool:
            for i,(rows, row_delay, limit) in enumerate(pool.map(cache_row, jobs, chunksize=12)):
                if rows.shape != (3,size) or row_delay != delay or not np.isfinite(rows).all():
                    raise ValueError(f'Unexpected source operator shape or value at row {i}')
                bank[i], limits[i] = rows, limit
                if (i+1) % 1000 == 0:
                    print('W24_SOURCE_ONLINE_OPERATORS', i+1, len(iq), flush=True)
        bank.flush()
        with (root/'agc-limits.npy').open('xb') as stream:
            np.save(stream, limits, allow_pickle=False)
        result = dict(status='SOURCE_OPERATOR_BANK_FROZEN', data_role='U_s', count=len(iq), source_root=str(source),
                      source_namespace=NAMESPACE, source_seed=BANK_SEED, padding=delay,
                      operator_roles=['signal_pre_agc', 'noise_main', 'noise_image'], **ISOLATION)
        write(root/'complete.json', result)
        write(status, result, exclusive=False)
    except Exception as error:
        write(status, dict(status='FAILED', error=str(error)), exclusive=False)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    methods = parser.add_subparsers(dest='method', required=True)
    methods.add_parser('dispatch').add_argument('runtime', type=Path)
    publish = methods.add_parser('publish')
    for name in ('repo', 'output', 'context', 'ssh-config'):
        publish.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    if args.method == 'dispatch':
        build(args.runtime)
    else:
        from experiments.pamf_lora_main.publish import release, release_commit

        context = read(args.context)
        rt = {k:context[k] for k in ('project', 'python', 'frozen_sources')}
        project = PurePosixPath(rt['project'])
        run_root, release_root = project/'runs'/RUN, project/'releases'/f'{RUN}-release'
        source = project/'releases'/f'{DATA_RUN}-release'/'data'/'source'
        original = project/'runs'/PRACTICAL_CACHE_RUN/'source_practical'
        rt.update(run_id=RUN, runner=__spec__.name, code_commit=release_commit(args.repo),
                  run_root=str(run_root), release_root=str(release_root), source_root=str(source),
                  practical_cache=str(original), workers=4, external_releases=[str(source),str(original)])
        release(args.repo, args.output, rt, args.ssh_config)


if __name__ == '__main__':
    main()
