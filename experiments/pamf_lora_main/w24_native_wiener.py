"""Zero-parameter, full-aperture receive-only Wiener control for frozen W24."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from math import log
import os
from pathlib import Path, PurePosixPath
import subprocess
import time
import zipfile

import numpy as np
import torch

if __package__ is None:
    from experiments import pamf_lora_main

    pamf_lora_main.__path__.append(str(Path(__file__).resolve().parent))

from experiments.cric_dura_comparison.io import read, tensor, write
from experiments.pamf_lora_main.data import label_ids, packet_indices, recording
from experiments.pamf_lora_main.full_packet_views import native, source_contract, target_contract
from experiments.pamf_lora_main.runner import activate, available
from experiments.pamf_lora_main.w24_factor_audit import AUG_RUN, load_classifier
from experiments.pamf_lora_main.w24_native_full import (
    DATA_RUN, LIMIT, SOURCE, TARGET, VIEWS, W24_COMMIT, W24_RUN, channel_job, classify, validate_predictions,
)
from experiments.pamf_lora_main.w24_sat_aug import MODES as REFERENCE_MODES


RUN = '20261125-phase1-lora-w24-native-wiener-pilot-s2026092702-r01'
PILOT_SEED = 2026112561
PER_RECEIVER = 200
MODES = ('direct', 'wiener_half', 'wiener_full')
STRIDE = 1024
PAD = 128


def noise_profile(packets):
    if packets.ndim != 2 or packets.shape[-1] != 8 * STRIDE or not packets.is_complex():
        raise ValueError('Expected a batch of eight complex native LoRa chirps')
    windows = packets.reshape(len(packets), 8, STRIDE)
    hann = torch.hann_window(STRIDE, dtype=packets.real.dtype, device=packets.device)
    normalized = hann / hann.square().mean().sqrt()
    power = torch.fft.fft(windows * normalized, norm='ortho').abs().square()
    far = torch.fft.fftfreq(STRIDE, device=packets.device).abs() > .35
    variance = power[..., far].flatten(1).median(1).values / log(2)
    profile = power.mean(1)
    # Pool nearby bins for variance reduction without estimating complex phase.
    profile = torch.stack([torch.roll(profile, k, -1) for k in range(-4, 5)]).mean(0)
    gain = (1 - variance[:, None] / profile.clamp_min(1e-12)).clamp(0., 1.)
    ratio = variance / packets.abs().square().mean(1).clamp_min(1e-12)
    return ratio, gain


def denoise(packets, threshold):
    ratio, gain = noise_profile(packets)
    active = ratio > threshold
    result = packets.clone()
    if active.any():
        x = packets[active]
        extended = torch.cat((x[:, 1:PAD + 1].flip((-1,)), x, x[:, -PAD - 1:-1].flip((-1,))), dim=1)
        f = (torch.fft.fftfreq(extended.shape[-1], device=x.device) * STRIDE).remainder(STRIDE)
        i = f.floor().long()
        response = torch.lerp(gain[active][:, i], gain[active][:, (i + 1) % STRIDE], f - i)
        filtered = torch.fft.ifft(torch.fft.fft(extended) * response)
        result[active] = filtered[:, PAD:-PAD]
    return result, active


def pilot_rows(receiver_index, seed=PILOT_SEED):
    rng = np.random.default_rng([seed, receiver_index])
    return np.sort(rng.choice(LIMIT, PER_RECEIVER, replace=False))


def torch_iq(raw, device):
    return torch.complex(tensor(raw.real, device), tensor(raw.imag, device))


def fit_gate(runtime):
    rt = read(runtime)
    activate(rt)
    root, source, device = Path(rt['run_root']), Path(rt['source_root']), 'cuda:0'
    contract = source_contract(source)
    with np.load(source / 'U_s_metadata.npz', allow_pickle=False) as meta:
        if 'y' in meta.files or meta['ids'].tolist() != contract['role_ids']['U_s']:
            raise ValueError('Wiener calibration requires registered TX-blind source U')
        receivers = meta['rx_i'].copy()
    rng = np.random.default_rng(PILOT_SEED)
    chosen = np.concatenate([rng.choice(np.flatnonzero(receivers == r), 500, replace=False)
                             for r in range(len(SOURCE))])
    raw = native(source, 'U_s', contract['counts']['U_s'])
    t0 = time.perf_counter()
    ratios = torch.cat([noise_profile(torch_iq(raw[ids], device))[0] for ids in np.array_split(chosen, 50)])
    threshold = float(torch.quantile(ratios, .999))
    if not np.isfinite(threshold) or not 0. < threshold < 1.:
        raise ValueError('Invalid source-only Wiener gate')
    write(root / 'calibration.json', dict(status='SOURCE_FROZEN', n=len(chosen), quantile=.999,
          threshold=threshold, seed=PILOT_SEED, source_protocol=contract['protocol'],
          parameters=0, F_reads=0, TX_reads=0, V_reads=0, target_reads=0,
          seconds=time.perf_counter() - t0))


def receipt(modes=MODES, seed=PILOT_SEED):
    return dict(status='PREDICTIONS_FROZEN', classifier_run=AUG_RUN, modes=list(modes), views=list(VIEWS),
                receivers=list(TARGET), sample_rows=[pilot_rows(i, seed).tolist() for i in range(len(TARGET))],
                target_truth_reads=0)


def evaluate(runtime):
    from experiments.pamf_lora_main.w24_native_full import native_full_view

    rt = read(runtime)
    activate(rt)
    root, source, device = Path(rt['run_root']), Path(rt['source_root']), 'cuda:0'
    contract = source_contract(source)
    c = read(root / 'calibration.json')
    if c['status'] != 'SOURCE_FROZEN' or c['n'] != 2500 or c['source_protocol'] != contract['protocol']:
        raise ValueError('Matching source-frozen calibration required')
    f = load_classifier(rt, contract, device)
    u = native(source, 'U_s', contract['counts']['U_s'])
    simulated = native_full_view(u[0], VIEWS[-1], SOURCE[0], 0, RUN)
    for q in (u[0], simulated):
        x = torch_iq(q[None], device)
        y, active = denoise(x, c['threshold'])
        if not active.any() and not torch.equal(x, y):
            raise ValueError('Wiener inactive path was modified')
        classify(f, np.from_dlpack(y.cpu().contiguous()), device)
    write(root / 'smoke.json', dict(status='PASS', source_packets=2, target_reads=0))

    def apply(wave):
        restored, active = denoise(wave, c['threshold'])
        return (torch.lerp(wave, restored, .5), restored), active

    predict_pilot(rt, f, device, apply, modes=MODES, seed=PILOT_SEED, log_prefix='W24_WIENER_PILOT')


def predict_pilot(rt, classifier, device, apply, *, modes, seed, log_prefix):
    root = Path(rt['run_root'])
    target = Path(rt['native_target_root'])
    target_contract(target)
    output = root / 'predictions'
    output.mkdir()
    activity = {}
    with ProcessPoolExecutor(max_workers=rt['cpu_workers']) as pool:
        for index, rx in enumerate(TARGET):
            rows = pilot_rows(index, seed)
            recorded = native(target, rx, LIMIT)[rows]
            pred = {}
            for scene in VIEWS:
                if scene == 'clean':
                    frames = recorded
                else:
                    frames = np.stack(list(pool.map(partial(channel_job, scene, rx),
                                           [(int(i), a) for i, a in zip(rows, recorded, strict=True)], chunksize=8)))
                batches, n_active = [], 0
                for start in range(0, len(frames), 32):
                    raw = frames[start:start+32]
                    wave = torch_iq(raw, device)
                    variants, active = apply(wave)
                    if len(variants) != len(modes) - 1:
                        raise ValueError('Predictor does not match the registered pilot modes')
                    base = classify(classifier, raw, device)
                    labels = [base.copy() for _ in variants]
                    chosen = np.asarray(active.tolist(), dtype=bool)
                    if chosen.any():
                        for arm, restored in enumerate(variants):
                            edit = restored[active]
                            labels[arm][chosen] = classify(classifier, np.from_dlpack(edit.cpu().contiguous()), device)
                        n_active += int(chosen.sum())
                    batches.append(np.stack((base, *labels)))
                pred[scene] = np.concatenate(batches, axis=1)
                activity[f'{rx}/{scene}'] = n_active
                print(log_prefix, rx, scene, n_active, flush=True)
            np.savez(output / f'{rx}.npz', **pred)
    write(output / 'activity.json', activity)
    write(output / 'complete.json', receipt(modes, seed))


def score(predictions, prior, dataset, output, *, modes=MODES, seed=PILOT_SEED, contrast=None):
    if read(predictions / 'complete.json') != receipt(modes, seed):
        raise ValueError('Complete frozen pilot matrix required')
    ref, _ = validate_predictions(prior, REFERENCE_MODES, classifier_run=AUG_RUN)
    activity = read(predictions / 'activity.json')
    if set(activity) != {f'{r}/{v}' for r in TARGET for v in VIEWS}:
        raise ValueError('Incomplete pilot activity record')
    work = []
    disagreements = 0
    for idx, rx in enumerate(TARGET):
        with np.load(predictions / f'{rx}.npz', allow_pickle=False) as data:
            if set(data.files) != set(VIEWS):
                raise ValueError('Pilot view coverage differs')
            p = np.stack([data[v] for v in VIEWS])
        if p.shape != (len(VIEWS), len(modes), PER_RECEIVER) or p.dtype.kind not in 'iu' or not ((p >= 0) & (p < 10)).all():
            raise ValueError('Malformed pilot predictions')
        rows = pilot_rows(idx, seed)
        baseline = np.stack([ref[rx, view, 'direct'][rows] for view in VIEWS])
        disagreements += int(np.count_nonzero(baseline != p[:, 0]))
        for i, view in enumerate(VIEWS):
            count = activity[f'{rx}/{view}']
            if not 0 <= count <= PER_RECEIVER or (p[i, 1:] != p[i, :1]).sum(axis=1).max() > count:
                raise ValueError('Pilot predictions differ outside the active count')
        dura = np.stack([ref[rx, view, 'complex_response_half_center'][rows] for view in VIEWS])
        work.append(np.concatenate((p[:, :1], dura[:, None], p[:, 1:]), axis=1))
    output.mkdir(parents=True, exist_ok=False)
    truth = []
    with zipfile.ZipFile(dataset / 'multiple_receiver_test.zip') as archive:
        for i, rx in enumerate(TARGET):
            with recording(archive, f'{rx}_test.h5', output) as h5:
                y = label_ids(h5, LIMIT)[packet_indices(LIMIT, i)]
                truth.append(y[pilot_rows(i, seed)])
    hit = np.stack(work) == np.array(truth)[:, None, None, :]
    rng = np.random.default_rng(seed + 1)
    draws = rng.integers(PER_RECEIVER, size=(len(TARGET), 2000, PER_RECEIVER))
    audit = {}
    paired_modes = ('direct', 'prior_complex_response_half', *modes[1:])

    def six_interval(delta):
        net = delta[:, 1:].sum(1)
        drawn = np.take_along_axis(net[:, None, :], draws, axis=2).sum((0, 2))
        return np.quantile(drawn, [.025, .975]).tolist()

    for idx, name in enumerate(paired_modes):
        correct = hit[:, :, idx]
        against_raw = correct.astype(np.int16) - hit[:, :, 0]
        against_dura = correct.astype(np.int16) - hit[:, :, 1]
        ci = {key: six_interval(delta) for key, delta in (('raw', against_raw), ('dura', against_dura))}
        audit[name] = dict(correct_by_scene=dict(zip(VIEWS, correct.sum((0, 2)).tolist(), strict=True)),
            net_six_vs_raw=int(against_raw[:, 1:].sum()), net_six_vs_dura=int(against_dura[:, 1:].sum()),
            net_low_vs_raw=int(against_raw[:, -1].sum()),
            repairs_six=int((against_raw[:, 1:] > 0).sum()),
            harms_six=int((against_raw[:, 1:] < 0).sum()),
            CI95_six=ci, net_six_by_rx=dict(zip(TARGET, against_raw[:, 1:].sum((1, 2)).tolist(), strict=True)))
    result = dict(status='ANALYZED', scope='single_seed_3000_pilot', truth_last=True,
                  n_physical=len(TARGET)*PER_RECEIVER, n_six=len(TARGET)*PER_RECEIVER*6,
                  direct_disagreements_vs_prior=disagreements, results=audit, activity=activity)
    if contrast is not None:
        a, b = (paired_modes.index(mode) for mode in contrast)
        delta = hit[:, :, a].astype(np.int16) - hit[:, :, b]
        result['ablation'] = dict(candidate=contrast[0], control=contrast[1],
                                  net_by_scene=dict(zip(VIEWS, delta.sum((0, 2)).tolist(), strict=True)),
                                  net_six=int(delta[:, 1:].sum()), CI95_net_six=six_interval(delta))
    write(output / 'scores.json', result)


def dispatch(runtime):
    rt = read(runtime)
    root, reference = Path(rt['run_root']), Path(rt['w24_release_root'])
    root.mkdir()
    if read(reference / 'submit.json')['code_commit'] != W24_COMMIT:
        raise ValueError('Exact W24 source release required')
    source_runtime = root / 'source-runtime.json'
    write(source_runtime, {k: rt[k] for k in ('run_root', 'source_root', 'frozen_sources')})
    write(root / 'state.json', dict(status='WAIT_GPU'))
    deadline = time.monotonic() + 1800
    while not (cards := available(rt['allowed_gpus'], max_compute_processes=8)):
        if time.monotonic() >= deadline:
            write(root / 'state.json', dict(status='FAILED', error='GPU wait expired'), exclusive=False)
            return
        time.sleep(20)
    gpu = next(i for i in rt['allowed_gpus'] if i in cards)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1',
               PYTHONPATH=str(reference))
    script = str(Path(__file__).resolve())
    for phase, config in (('fit-gate', source_runtime), ('evaluate', runtime)):
        with (root / (phase + '.log')).open('x', encoding='utf-8') as log_file:
            proc = subprocess.Popen([rt['python'], '-u', script, phase, str(config)], cwd=reference,
                                    env=env, stdout=log_file, stderr=subprocess.STDOUT)
            write(root / 'state.json', dict(status='RUNNING', phase=phase, pid=proc.pid, gpu=gpu), exclusive=False)
            if proc.wait():
                write(root / 'state.json', dict(status='FAILED', error=phase), exclusive=False)
                return
    if read(root / 'predictions' / 'complete.json') != receipt():
        raise ValueError('Frozen Wiener matrix differs')
    write(root / 'state.json', dict(status='PREDICTIONS_FROZEN'), exclusive=False)


def main():
    from experiments.pamf_lora_main.publish import release, release_commit

    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='action', required=True)
    for phase in ('dispatch', 'fit-gate', 'evaluate'):
        sub.add_parser(phase).add_argument('runtime', type=Path)
    pub = sub.add_parser('publish')
    for key in ('repo', 'context', 'ssh-config', 'output'):
        pub.add_argument('--' + key, type=Path, required=True)
    scoring = sub.add_parser('score')
    for key in ('predictions', 'prior', 'dataset', 'output'):
        scoring.add_argument('--' + key, type=Path, required=True)
    args = parser.parse_args()
    if args.action == 'publish':
        ctx = read(args.context)
        rt = {k: ctx[k] for k in ('project', 'python', 'frozen_sources')}
        project = PurePosixPath(rt['project'])
        data = project / 'releases' / f'{DATA_RUN}-release' / 'data'
        rt.update(run_id=RUN, code_commit=release_commit(args.repo), runner=__spec__.name,
                  release_root=str(project / 'releases' / f'{RUN}-release'),
                  run_root=str(project / 'runs' / RUN),
                  source_root=str(data / 'source'), native_target_root=str(data / 'target'),
                  classifier_root=str(project / 'runs' / AUG_RUN),
                  w24_release_root=str(project / 'releases' / f'{W24_RUN}-release'),
                  allowed_gpus=[2, 7, 4], cpu_workers=8)
        release(args.repo, args.output, rt, args.ssh_config)
    elif args.action == 'score':
        score(args.predictions, args.prior, args.dataset, args.output)
    else:
        {'dispatch': dispatch, 'fit-gate': fit_gate, 'evaluate': evaluate}[args.action](args.runtime)


if __name__ == '__main__':
    main()
