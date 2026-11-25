"""Isolate a native-edge edit with W24's real ungated center restoration fixed."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path, PurePosixPath

import numpy as np

from experiments.cric_dura_comparison.io import read, write


RUN = '20261125-diagnostic-lora-w24-edge-denoise-source-s2026112559-r01'
BLIND = ('direct', 'all_half', 'edge_only', 'half_plus_edge')
MODES = (*BLIND, 'half_plus_exact_edge')
CONTRASTS = (('half_plus_edge', 'all_half'), ('half_plus_exact_edge', 'all_half'),
             ('half_plus_exact_edge', 'half_plus_edge'), ('edge_only', 'direct'))


def adaptive_edges(edges, noise):
    """Shrink the native short-aperture residual continuously in each edge."""
    from scipy.signal import savgol_filter

    pairs = edges.reshape(len(edges), 8, 2, 24)
    fitted = savgol_filter(pairs.real, 7, 2) + 1j*savgol_filter(pairs.imag, 7, 2)
    residual = pairs-fitted
    power = np.mean(np.abs(residual)**2, axis=-1, keepdims=True)
    variance = noise.reshape(-1, 1, 1, 1)
    dose = .5*variance / np.maximum(variance + power, 1e-12)
    return (pairs-dose*residual).reshape(edges.shape).astype(edges.dtype, copy=False)


def noise_power(raw, rms):
    block = raw.reshape(len(raw), 8, 1024)
    hann = np.hanning(1024)
    power = np.abs(np.fft.fft(block*hann/np.sqrt(np.mean(hann**2)), norm='ortho'))**2
    far = np.abs(np.fft.fftfreq(1024)) > .35
    return np.median(power[..., far], axis=(1, 2)) / (np.log(2)*rms.reshape(-1)**2)


def classify_rows(f, middle, edge):
    from experiments.cric_dura_comparison.io import tensor
    from experiments.pamf_lora_main.w24_native_full import classify_aperture

    wave = np.concatenate((middle, edge), axis=-1)
    iq = np.stack((wave.real, wave.imag), axis=2).astype(np.float32, copy=False)
    return classify_aperture(f, tensor(iq, 'cuda:0'))


def blind_variants(f, raw, received_center, half):
    from experiments.pamf_lora_main.w24_aperture_noise_audit import center_rms, native_aperture

    aperture = native_aperture(raw)
    rms = center_rms(aperture)
    source = (aperture/rms).astype(np.complex64)
    np.testing.assert_array_equal(source[..., :384], received_center)
    edge = source[..., 384:]
    adjusted = adaptive_edges(edge, noise_power(raw, rms))
    return np.stack([classify_rows(f, mid, boundary) for mid, boundary in
                     ((received_center, edge), (half, edge), (received_center, adjusted), (half, adjusted))])


def predict(rt):
    import torch
    from experiments.pamf_lora_main.full_packet_views import native, source_contract
    from experiments.pamf_lora_main.runner import activate
    from experiments.pamf_lora_main.w24_factor_audit import load_classifier
    from experiments.pamf_lora_main.w24_native_residual_source import ROUTES, SOURCE, SWEEP_RUN, VIEWS, paired_iq
    from experiments.pamf_lora_main.w24_waveform_audit import RUN as WAVEFORMS, restore, source_rows
    from experiments.pamf_lora_main.w24_wide_transfer import TRAIN_RUN, load_restorer, training_runtime

    activate(rt)
    root, source = Path(rt['run_root']), Path(rt['source_root'])
    contract = source_contract(source)
    classifier = load_classifier(rt, contract, 'cuda:0')
    g, tracker, _, info = load_restorer(*training_runtime(rt), contract, 'cuda:0')
    packet = native(source, 'U_s', contract['counts']['U_s'])[0]
    with torch.no_grad():
        for view in ('clean', VIEWS[-1]):
            for iq in paired_iq(view, (0, SOURCE[0], packet)):
                direct, half, _ = restore(g, tracker, iq[None])
                values = blind_variants(classifier, iq[None], direct, half)
                if values.shape != (len(BLIND), 1):
                    raise ValueError('Expected real-source W24 edge smoke predictions')
    write(root/'smoke.json', dict(status='PASS', source_count=1, input_gate=False, target_reads=0, reference_reads=0))

    _, jobs, ids = source_rows(rt)
    frozen_root = root.parent/WAVEFORMS
    receipt = read(frozen_root/'waveforms-frozen.json')
    if (receipt['status'],receipt['ids'],receipt['source_frozen_G'],receipt['steps'],receipt['simulator_reference_reads'],
        receipt['source_truth_reads'],receipt['target_reads']) != ('FROZEN', ids, TRAIN_RUN, info['steps'], 0, 0, 0):
        raise ValueError('Expected the prior unfactored, ungated half waveforms')
    output = {route: {} for route in ROUTES}
    with ProcessPoolExecutor(max_workers=rt['cpu_workers']) as pool, torch.no_grad():
        for view in VIEWS:
            packets = list(pool.map(partial(paired_iq, view), jobs, chunksize=6))
            for route, rows in zip(ROUTES, zip(*packets, strict=True), strict=True):
                raw = np.stack(rows)
                with np.load(frozen_root/route/f'{view}.npz', allow_pickle=False) as data:
                    direct, half = data['received'], data['half']
                result = [blind_variants(classifier, raw[i:i+24], direct[i:i+24], half[i:i+24])
                          for i in range(0, len(raw), 24)]
                output[route][view] = np.concatenate(result, axis=1)
            print('W24_EDGE_BLIND_PREDICTED', view, len(jobs), flush=True)
    for route in ROUTES:
        original = root.parent/SWEEP_RUN/route
        proof = read(original/'source-predictions.json')
        if (proof['status'], proof['physical_ids'], proof['views'], proof['target_reads']) != (
                'PREDICTIONS_FROZEN', ids, list(VIEWS), 0):
            raise ValueError('Prior recognition controls do not match source rows')
        with np.load(original/'predictions.npz', allow_pickle=False) as old:
            indices = [proof['modes'].index(name) for name in ('direct', 'wide_ungated_050')]
            if any(not np.array_equal(output[route][v][:2], old[v][indices]) for v in VIEWS):
                raise ValueError('Direct or ungated center-half predictions differ from their archive')
        with (root/f'{route}-blind.npz').open('xb') as stream:
            np.savez(stream, **output[route])
    write(root/'blind-frozen.json', dict(status='FROZEN', physical_ids=ids, modes=list(BLIND), routes=list(ROUTES),
                                        views=list(VIEWS), target_reads=0, source_truth_reads=0, reference_reads=0))


def oracle(rt):
    import torch
    from experiments.pamf_lora_main.runner import activate
    from experiments.pamf_lora_main.w24_aperture_noise_audit import center_rms, native_aperture, packet_components
    from experiments.pamf_lora_main.w24_factor_audit import load_classifier
    from experiments.pamf_lora_main.w24_native_residual_source import ROUTES, VIEWS
    from experiments.pamf_lora_main.w24_waveform_audit import RUN as WAVEFORMS, source_rows

    activate(rt)
    root = Path(rt['run_root'])
    contract, jobs, ids = source_rows(rt)
    frozen = read(root/'blind-frozen.json')
    if frozen != dict(status='FROZEN', physical_ids=ids, modes=list(BLIND), routes=list(ROUTES), views=list(VIEWS),
                      target_reads=0, source_truth_reads=0, reference_reads=0):
        raise ValueError('Complete receive-only predictions required before opening reference waveforms')
    output = {route: {} for route in ROUTES}
    for route in ROUTES:
        with np.load(root/f'{route}-blind.npz', allow_pickle=False) as saved:
            if set(saved.files) != set(VIEWS) or any(saved[v].shape != (len(BLIND), len(ids)) for v in VIEWS):
                raise ValueError('Incomplete receive-only frozen source predictions')
            output[route].update({v:saved[v] for v in VIEWS})
    f = load_classifier(rt, contract, 'cuda:0')
    with ProcessPoolExecutor(max_workers=rt['cpu_workers']) as pool, torch.no_grad():
        for view in VIEWS:
            traces = list(pool.map(partial(packet_components, view, verify=True), jobs, chunksize=6))
            for route, waveforms in zip(ROUTES, zip(*traces, strict=True), strict=True):
                observed, reference = (native_aperture(np.stack(x)) for x in zip(*waveforms, strict=True))
                rms = center_rms(observed)
                with np.load(root.parent/WAVEFORMS/route/f'{view}.npz', allow_pickle=False) as data:
                    np.testing.assert_array_equal((observed[..., :384]/rms).astype(np.complex64), data['received'])
                    half = data['half']
                reference_edge = (reference[..., 384:]/rms).astype(np.complex64)
                values = np.concatenate([classify_rows(f, half[i:i+24], reference_edge[i:i+24])
                                         for i in range(0, len(half), 24)])
                output[route][view] = np.concatenate((output[route][view], values[None]), axis=0)
            print('W24_EDGE_ORACLE_PREDICTED', view, len(jobs), flush=True)
    for route in ROUTES:
        dest = root/route
        dest.mkdir()
        with (dest/'predictions.npz').open('xb') as stream:
            np.savez(stream, **output[route])
        write(dest/'source-predictions.json', dict(status='PREDICTIONS_FROZEN', modes=list(MODES), views=list(VIEWS),
              physical_ids=ids, source_truth_reads=0, target_reads=0, waveform=dict(ungated=True,
              actual_half_centers=True, oracle_isolated_after_blind=True)))


def score(runtime):
    from experiments.pamf_lora_main.w24_factor_audit import frozen_source_panel, score_source
    from experiments.pamf_lora_main.w24_native_residual_source import ROUTES

    rt = read(runtime)
    root, source = Path(rt['run_root']), Path(rt['source_root'])
    for route in ROUTES:
        frozen_source_panel(source, root/route, MODES)
    for route in ROUTES:
        score_source(runtime, modes=MODES, output=root/route, contrasts=CONTRASTS)


def main():
    if __package__ is None:
        from experiments import pamf_lora_main

        pamf_lora_main.__path__.append(str(Path(__file__).resolve().parent))
    from experiments.pamf_lora_main.publish import release, release_commit
    from experiments.pamf_lora_main.w24_native_residual_source import AUG_RUN, DATA_RUN, SWEEP_RUN, TRAIN_RUN, W24_RUN, dispatch
    from experiments.pamf_lora_main.w24_waveform_audit import RUN as WAVEFORMS

    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='action',required=True)
    for name in ('dispatch','predict-source','oracle-source','score-source'):
        sub.add_parser(name).add_argument('runtime',type=Path)
    publish = sub.add_parser('publish')
    for name in ('repo','context','ssh-config','output'):
        publish.add_argument('--'+name,type=Path,required=True)
    args = parser.parse_args()
    if args.action == 'dispatch':
        dispatch(args.runtime, entry=__file__, phases=('predict-source','oracle-source','score-source'),
                 gpu_phases=('predict-source','oracle-source'))
    elif args.action == 'predict-source':
        predict(read(args.runtime))
    elif args.action == 'oracle-source':
        oracle(read(args.runtime))
    elif args.action == 'score-source':
        score(args.runtime)
    else:
        context = read(args.context)
        rt = {key:context[key] for key in ('project','python','frozen_sources')}
        root = PurePosixPath(rt['project'])
        runs,releases = root/'runs',root/'releases'
        rt.update(run_id=RUN, runner=__spec__.name, code_commit=release_commit(args.repo),
                  run_root=str(runs/RUN), release_root=str(releases/f'{RUN}-release'),
                  source_root=str(releases/f'{DATA_RUN}-release'/'data/source'), classifier_root=str(runs/AUG_RUN),
                  w24_release_root=str(releases/f'{W24_RUN}-release'), training_release_root=str(releases/f'{TRAIN_RUN}-release'),
                  allowed_gpus=[3, 7, 4, 1, 2, 6, 0, 5], cpu_workers=8)
        rt['external_releases'] = [rt[key] for key in ('source_root','classifier_root','w24_release_root','training_release_root')]
        rt['external_releases'].extend(str(runs/name) for name in (WAVEFORMS,SWEEP_RUN))
        release(args.repo, args.output, rt, args.ssh_config)


if __name__ == '__main__':
    main()
