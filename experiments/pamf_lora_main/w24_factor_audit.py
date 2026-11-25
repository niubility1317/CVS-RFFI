"""Source-only decomposition of native Full LoRa distortion under frozen W24.

Simulator traces and paired clean signals are used only in this audit.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import os
from pathlib import Path, PurePosixPath
import subprocess
import time

import numpy as np
import torch

if __package__ is None:
    from experiments import pamf_lora_main

    pamf_lora_main.__path__.append(str(Path(__file__).resolve().parent))

from experiments.cric_dura_comparison.io import read, write
from experiments.pamf_lora_main.full_packet_views import native, source_contract
from experiments.pamf_lora_main.runner import activate, available
from experiments.pamf_lora_main.w24_native_full import (
    CONTEXT, DATA_RUN, METHOD, PACKET_LEN, SEED, SOURCE, STRIDE, VIEWS, W24_COMMIT, W24_RUN,
    channel, classify, hardware, load_restorers, native_full_view, recover,
)
from experiments.pamf_lora_main.w24_sat_aug import RUN as AUG_RUN


RUN = '20261125-diagnostic-lora-w24-distortion-source-s2026112559-r01'
PANEL_SEED = 2026112559
PER_RX = 60
VIEWS_PRACTICAL = VIEWS[1:]
MODES = ('received', 'existing_dura', 'no_added_noise', 'paired_affine_phase',
         'noise_and_affine_phase', 'noise_phase_and_chirp_gain', 'source_clean')
FS = 1e6
MAX_CFO = 4000.


def affine_phase(signal, reference):
    """Estimate packet-wide affine phase using paired known-clean IQ."""
    cross = signal * reference.conj()
    fft_size = 4 * len(cross)
    frequencies = np.fft.fftfreq(fft_size, 1 / FS)
    spectrum = np.abs(np.fft.fft(cross, n=fft_size))
    search = np.flatnonzero(np.abs(frequencies) <= MAX_CFO)
    peak = search[np.argmax(spectrum[search])]
    offsets = np.arange(-1, 2) + peak
    left, mid, right = spectrum[offsets % fft_size]
    curvature = left - 2 * mid + right
    fraction = 0. if curvature >= 0. else np.clip(.5 * (left - right) / curvature, -.5, .5)
    frequency = np.clip(frequencies[peak] + fraction * FS / fft_size, -MAX_CFO, MAX_CFO)
    ramp = np.exp(2j * np.pi * (frequency / FS) * np.arange(len(cross)))
    constant = np.angle(np.vdot(ramp, cross))
    return ramp * np.exp(1j * constant), float(frequency)


def fit_chirp_gains(signal, reference):
    y, x = (a.reshape(-1, STRIDE) for a in (signal, reference))
    signal_energy = (np.abs(x) ** 2).sum(-1)
    gain = (x.conj() * y).sum(-1) / np.maximum(signal_energy, 1e-12)
    return np.where((np.abs(gain) >= 1e-6) & (signal_energy > 1e-12), gain, 1.)


def trace_components(packet, scene, receiver, physical_id, namespace):
    """Keep the physical replay identical to the registered native Full view."""
    from leo_practical.channel import ChannelStream, normalize_input, stable_seed

    normalized, _ = normalize_input(packet)
    cfg = channel(scene)
    delay = cfg.fractional_delay_half_length
    seed = stable_seed(2026112207, namespace, scene, receiver, str(physical_id))
    stream = ChannelStream(cfg, seed, hardware(scene, f'{namespace}/{receiver}'))
    y, meta, traces = stream.process(np.pad(normalized, (0, delay)),
                                     preceding_context=np.zeros(CONTEXT, dtype=normalized.dtype),
                                     return_traces=True)
    if (meta['boundary'] != 'provided_context' or meta['channel_equalization_applied']
            or meta['numerical_filter_latency_samples'] != delay):
        raise ValueError('Trace replay differs from registered Full native view')
    window = slice(delay, delay + PACKET_LEN)
    scale = 10 ** (meta['agc_gain_db'] / 20)
    received = y[window]
    signal = traces['pre_agc_signal'][window] * scale
    noise = traces['pre_agc_noise'][window] * scale
    if not np.allclose(received, signal + noise, rtol=2e-6, atol=1e-7):
        raise ValueError('Noiseless and additive components fail to close the received waveform')
    return received, signal, noise


def waveforms(scene, namespace, item, *, verify=False):
    physical_id, rx, source = item
    observed, signal, noise = trace_components(source, scene, rx, physical_id, namespace)
    if verify:
        expected = native_full_view(source, scene, rx, physical_id, namespace)
        if not np.array_equal(observed, expected):
            raise ValueError('Trace replay does not exactly match registered raw input')
    phase, cfo = affine_phase(signal, source)
    aligned = signal * phase.conj()
    gains = fit_chirp_gains(aligned, source)
    complete = aligned / np.repeat(gains, STRIDE)
    residual = np.linalg.norm(complete - source) ** 2 / np.linalg.norm(source) ** 2
    energy_ratio = np.linalg.norm(signal) ** 2 / max(np.linalg.norm(noise) ** 2, 1e-30)
    modes = (observed, signal, observed * phase.conj(), aligned, complete, source)
    return np.stack(modes).astype(np.complex64), (10 * np.log10(max(energy_ratio, 1e-30)), cfo, residual)


def panel(source, contract):
    with np.load(source / 'V_metadata.npz', allow_pickle=False) as metadata:
        ids = metadata['ids'].tolist()
        if ids != contract['role_ids']['V']:
            raise ValueError('Source V physical IDs differ from registered order')
        rx = metadata['rx_i'].copy()
    rng = np.random.default_rng(PANEL_SEED)
    rows = np.concatenate([np.sort(rng.choice(np.flatnonzero(rx == i), PER_RX, replace=False))
                           for i in range(len(SOURCE))])
    return rows, rx[rows], [ids[i] for i in rows]


def load_classifier(rt, contract, device):
    from experiments.pamf_lora_main.tri_clean95 import NATIVE_WIDE_ROWS, clean_model

    frozen = Path(rt['classifier_root']) / 'F'
    artifact = read(frozen / 'complete.json')
    if (artifact['status'] != 'SOURCE_FROZEN' or artifact['method'] != AUG_RUN
            or artifact['contract'] != contract or artifact['w24_recipe_commit'] != W24_COMMIT
            or artifact['seed'] != SEED or artifact['frozen']['epoch'] != 104
            or artifact['satellite_batches'] != 9856 or artifact['target_reads']):
        raise ValueError('Augmented E104 checkpoint differs from the frozen comparison')
    model = clean_model(dict(scratch=True), frozen, contract, NATIVE_WIDE_ROWS[METHOD], device)
    model.load_state_dict(torch.load(frozen / 'best.pt', map_location=device, weights_only=True), strict=True)
    return model.eval().requires_grad_(False)


def predict_source(runtime):
    rt = read(runtime)
    activate(rt)
    root, source, device = Path(rt['run_root']), Path(rt['source_root']), 'cuda:0'
    contract = source_contract(source)
    f = load_classifier(rt, contract, device)
    restorers = load_restorers(contract, root.parent, device)
    u = native(source, 'U_s', contract['counts']['U_s'])
    for scene in VIEWS_PRACTICAL:
        simulated, _ = waveforms(scene, RUN, (0, SOURCE[0], u[0]), verify=True)
        for wave in simulated:
            classify(f, wave[None, :], device)
    write(root / 'smoke.json', dict(status='PASS', exact_full_replay=True, target_reads=0, source_truth_reads=0))
    rows, rx, ids = panel(source, contract)
    raw = native(source, 'V', contract['counts']['V'])[rows]
    clean_predictions = classify(f, raw, device)
    output = root / 'predictions.npz'
    predictions = {'clean': np.repeat(clean_predictions[None, :], len(MODES), axis=0)}
    for offset in range(0, len(raw), 32):
        packets = raw[offset:offset + 32]
        _, _, dura = recover(packets, restorers, device)
        predictions['clean'][1, offset:offset + len(dura)] = classify(f, dura, device)
    telemetry = {}
    jobs = [(int(i), SOURCE[int(r)], pkt) for i, r, pkt in zip(rows, rx, raw, strict=True)]
    with ProcessPoolExecutor(max_workers=rt['cpu_workers']) as pool:
        for scene in VIEWS_PRACTICAL:
            result = pool.map(partial(waveforms, scene, RUN), jobs, chunksize=8)
            view_predictions, stats = [], []
            for offset in range(0, len(jobs), 32):
                items = [next(result) for _ in range(min(32, len(jobs) - offset))]
                waves = np.stack([item[0] for item in items], axis=1)
                stats.extend(item[1] for item in items)
                _, _, dura = recover(waves[0], restorers, device)
                variants = (waves[0], dura, *waves[1:])
                view_predictions.append(np.stack([classify(f, q, device) for q in variants]))
            predictions[scene] = np.concatenate(view_predictions, axis=1)
            fields = np.array(stats)
            telemetry[scene] = dict(
                added_SNR_db_median=float(np.median(fields[:, 0])),
                paired_abs_CFO_hz_median=float(np.median(abs(fields[:, 1]))),
                noiseless_after_affine_and_chirp_gain_NMSE_median=float(np.median(fields[:, 2])))
            print('W24_FACTOR_PREDICTED', scene, len(jobs), flush=True)
    np.savez(output, **predictions)
    write(root / 'source-predictions.json', dict(status='PREDICTIONS_FROZEN', modes=list(MODES),
          views=list(VIEWS), physical_ids=ids, target_reads=0, source_truth_reads=0, waveform=telemetry))


def score_source(runtime):
    rt = read(runtime)
    root, source = Path(rt['run_root']), Path(rt['source_root'])
    rows, rx, ids = panel(source, source_contract(source))
    frozen = read(root / 'source-predictions.json')
    if (frozen['status'] != 'PREDICTIONS_FROZEN' or frozen['physical_ids'] != ids
            or frozen['modes'] != list(MODES) or frozen['views'] != list(VIEWS)
            or frozen['target_reads'] or frozen['source_truth_reads']):
        raise ValueError('Source prediction matrix is not frozen')
    with np.load(root / 'predictions.npz', allow_pickle=False) as file:
        if set(file.files) != set(VIEWS):
            raise ValueError('Source view coverage differs')
        preds = np.stack([file[v] for v in VIEWS])
    if (preds.shape != (len(VIEWS), len(MODES), len(rows))
            or preds.dtype.kind not in 'iu' or preds.min() < 0 or preds.max() >= 10):
        raise ValueError('Source predictions have invalid shape or class IDs')
    with np.load(source / 'V_metadata.npz', allow_pickle=False) as metadata:
        truth = metadata['y'][rows]
    hits = preds == truth
    delta = hits.astype(np.int16) - hits[:, :1].astype(np.int16)
    rng = np.random.default_rng(PANEL_SEED + 1)
    samples = np.concatenate([rng.choice(np.flatnonzero(rx == i), (2000, PER_RX))
                              for i in range(len(SOURCE))], axis=1)
    readout = {}
    for i, mode in enumerate(MODES):
        change = delta[1:, i].sum(0)
        low = delta[-1, i]
        readout[mode] = dict(by_scene=dict(zip(VIEWS, hits[:, i].sum(1).tolist(), strict=True)),
            six_correct=int(hits[1:, i].sum()), low_urban_correct=int(hits[-1, i].sum()),
            six_corr=int((delta[1:, i] > 0).sum()), six_harm=int((delta[1:, i] < 0).sum()),
            net_six=int(change.sum()), net_low=int(low.sum()),
            CI95_net_six=np.quantile(change[samples].sum(1), [.025, .975]).tolist(),
            CI95_net_low=np.quantile(low[samples].sum(1), [.025, .975]).tolist(),
            net_six_by_rx={r: int(change[rx == j].sum()) for j, r in enumerate(SOURCE)})
    write(root / 'source-scores.json', dict(status='ANALYZED', n_packets=len(rows),
          n_practical=len(rows) * len(VIEWS_PRACTICAL), source_truth_last=True,
          target_reads=0, scores=readout, waveform=frozen['waveform']))


def dispatch(runtime):
    rt = read(runtime)
    root, reference = Path(rt['run_root']), Path(rt['w24_release_root'])
    root.mkdir()
    if read(reference / 'submit.json')['code_commit'] != W24_COMMIT:
        raise ValueError('Exact W24 source release required')
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
    for phase in ('predict-source', 'score-source'):
        with (root / (phase + '.log')).open('x', encoding='utf-8') as log:
            if phase == 'score-source':
                env['CUDA_VISIBLE_DEVICES'] = ''
            p = subprocess.Popen([rt['python'], '-u', script, phase, str(runtime)], cwd=reference,
                                 env=env, stdout=log, stderr=subprocess.STDOUT)
            write(root / 'state.json', dict(status='RUNNING', phase=phase, pid=p.pid, gpu=gpu), exclusive=False)
            if p.wait():
                write(root / 'state.json', dict(status='FAILED', error=phase), exclusive=False)
                return
    write(root / 'state.json', dict(status='SOURCE_SCORED'), exclusive=False)


def main():
    from experiments.pamf_lora_main.publish import release, release_commit

    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='action', required=True)
    for phase in ('dispatch', 'predict-source', 'score-source'):
        sub.add_parser(phase).add_argument('runtime', type=Path)
    publish = sub.add_parser('publish')
    for key in ('repo', 'context', 'ssh-config', 'output'):
        publish.add_argument('--' + key, type=Path, required=True)
    args = parser.parse_args()
    if args.action == 'publish':
        ctx = read(args.context)
        rt = {k: ctx[k] for k in ('project', 'python', 'frozen_sources')}
        project = PurePosixPath(rt['project'])
        rt.update(run_id=RUN, code_commit=release_commit(args.repo), runner=__spec__.name,
                  release_root=str(project / 'releases' / f'{RUN}-release'),
                  run_root=str(project / 'runs' / RUN),
                  source_root=str(project / 'releases' / f'{DATA_RUN}-release' / 'data/source'),
                  classifier_root=str(project / 'runs' / AUG_RUN),
                  w24_release_root=str(project / 'releases' / f'{W24_RUN}-release'),
                  allowed_gpus=[2, 7, 4], cpu_workers=8)
        release(args.repo, args.output, rt, args.ssh_config)
    else:
        {'dispatch': dispatch, 'predict-source': predict_source, 'score-source': score_source}[args.action](
            args.runtime)


if __name__ == '__main__':
    main()
