"""Freeze W24 IQ outputs before comparing them with isolated source references."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path, PurePosixPath

import numpy as np


RUN = '20261125-diagnostic-lora-w24-waveform-source-s2026112559-r01'
BANDS = (('0_to_100_khz', 0, 100), ('100_to_160_khz', 100, 160), ('160_to_192_khz', 160, 193))
MODES = ('received', 'half', 'full')


def spectral_error(wave, reference):
    """Packet-level complex, magnitude and phase errors in the original W24 centre frame."""
    if wave.shape != reference.shape or wave.ndim != 3 or wave.shape[1:] != (8, 384):
        raise ValueError('Expected matching packet, chirp and 384-point W24 centre axes')
    bins = np.abs(np.fft.fftfreq(384, d=1/384))
    estimate, target = (np.fft.fft(value, norm='ortho') for value in (wave, reference))
    result = {}
    for name, low, high in BANDS:
        band = (bins >= low) & (bins < high)
        z, x = estimate[..., band], target[..., band]
        zabs, xabs = np.abs(z), np.abs(x)
        energy = (xabs**2).sum(axis=(1, 2))
        scale = np.maximum(energy, 1e-12)
        cosine = (z * x.conj()).real / np.maximum(zabs * xabs, 1e-20)
        result[name] = dict(reference_energy=energy,
                            complex_nmse=(np.abs(z-x)**2).sum(axis=(1, 2)) / scale,
                            amplitude_nmse=((zabs-xabs)**2).sum(axis=(1, 2)) / scale,
                            phase_chord_loss=((1-np.clip(cosine, -1, 1))*xabs**2).sum(axis=(1, 2)) / scale)
    return result


def source_rows(rt):
    from experiments.pamf_lora_main.full_packet_views import native, source_contract
    from experiments.pamf_lora_main.w24_native_residual_source import SOURCE, panel

    source = Path(rt['source_root'])
    contract = source_contract(source)
    rows, rx, ids = panel(source, contract)
    raw = native(source, 'V', contract['counts']['V'])[rows]
    jobs = [(int(i), SOURCE[int(r)], packet) for i, r, packet in zip(rows, rx, raw, strict=True)]
    return contract, jobs, ids


def restore(model, tracker, packets):
    from experiments.pamf_lora_main.w24_formal import wide_delta
    from experiments.pamf_lora_main.w24_wide_transfer import received_aperture

    aperture = received_aperture(packets, 'cuda:0')
    center = aperture[..., :384]
    delta = wide_delta(model, tracker, aperture)
    values = (center, center + .5*delta, center + delta)
    rows = (np.from_dlpack(value.cpu().contiguous()) for value in values)
    return tuple(row[:, :, 0] + 1j*row[:, :, 1] for row in rows)


def predict(rt):
    import torch
    from experiments.cric_dura_comparison.io import write
    from experiments.pamf_lora_main.full_packet_views import native, source_contract
    from experiments.pamf_lora_main.runner import activate
    from experiments.pamf_lora_main.w24_native_residual_source import ROUTES, SOURCE, VIEWS, paired_iq
    from experiments.pamf_lora_main.w24_wide_transfer import TRAIN_RUN, load_restorer, training_runtime

    activate(rt)
    root, source = Path(rt['run_root']), Path(rt['source_root'])
    contract = source_contract(source)
    g, tracker, _, info = load_restorer(*training_runtime(rt), contract, 'cuda:0')
    packet = native(source, 'U_s', contract['counts']['U_s'])[0]
    with torch.no_grad():
        for scene in ('clean', VIEWS[-1]):
            for iq in paired_iq(scene, (0, SOURCE[0], packet)):
                wave = restore(g, tracker, iq[None])
                if any(x.shape != (1, 8, 384) or not np.isfinite(x).all() for x in wave):
                    raise ValueError('Real W24 source U waveform smoke failed')
    write(root/'smoke.json', dict(status='PASS', source_packets=1, input_gate=False,
                                target_reads=0, source_truth_reads=0, parameters=info['parameters']))

    _, jobs, ids = source_rows(rt)
    for route in ROUTES:
        (root/route).mkdir()
    batch = 24
    with ProcessPoolExecutor(max_workers=rt['cpu_workers']) as pool, torch.no_grad():
        for scene in VIEWS:
            stream = pool.map(partial(paired_iq, scene), jobs, chunksize=6)
            results = {route: [] for route in ROUTES}
            for offset in range(0, len(jobs), batch):
                chunk = [next(stream) for _ in range(min(batch, len(jobs)-offset))]
                for route, iq in zip(ROUTES, zip(*chunk, strict=True), strict=True):
                    results[route].append(restore(g, tracker, np.stack(iq)))
            for route, rows in results.items():
                matrix = {mode: np.concatenate(values) for mode, values in zip(MODES, zip(*rows), strict=True)}
                if any(not np.isfinite(value).all() for value in matrix.values()):
                    raise ValueError(f'Nonfinite {route}/{scene} W24 prediction')
                with (root/route/f'{scene}.npz').open('xb') as file:
                    np.savez(file, **matrix)
            print('W24_WAVEFORM_FROZEN', scene, len(jobs), flush=True)
    write(root/'waveforms-frozen.json', dict(status='FROZEN', routes=list(ROUTES), views=list(VIEWS), modes=list(MODES),
                                           ids=ids, source_frozen_G=TRAIN_RUN, steps=info['steps'],
                                           target_reads=0, source_truth_reads=0, simulator_reference_reads=0))


def summarize(modes, reference):
    def packet_private(z):
        return z-z.mean(axis=1, keepdims=True)

    groups = {}
    for name in ('all', 'chirp_private'):
        estimates = modes if name == 'all' else [packet_private(x) for x in modes]
        target = reference if name == 'all' else packet_private(reference)
        values = [spectral_error(x, target) for x in estimates]
        bands = {}
        for band in values[0]:
            keep = values[0][band]['reference_energy'] > 1e-12
            fields = {}
            for metric in ('complex_nmse', 'amplitude_nmse', 'phase_chord_loss'):
                columns = [entry[band][metric][keep] for entry in values]
                fields[metric] = dict(median={k: float(np.median(v)) if keep.any() else None
                                              for k, v in zip(MODES, columns, strict=True)},
                                      paired_delta={k: float(np.median(v-columns[0])) if keep.any() else None
                                                    for k, v in zip(MODES[1:], columns[1:], strict=True)})
            bands[band] = dict(supported_packets=int(keep.sum()), **fields)
        groups[name] = bands
    return groups


def score(rt):
    from experiments.cric_dura_comparison.io import read, write
    from experiments.pamf_lora_main.runner import activate
    from experiments.pamf_lora_main.w24_aperture_noise_audit import center_rms, native_aperture, packet_components
    from experiments.pamf_lora_main.w24_native_residual_source import ROUTES, SWEEP_RUN, VIEWS

    activate(rt)
    root = Path(rt['run_root'])
    _, jobs, ids = source_rows(rt)
    frozen = read(root/'waveforms-frozen.json')
    if (frozen['status'], frozen['routes'], frozen['views'], frozen['modes'], frozen['ids'],
        frozen['target_reads'], frozen['source_truth_reads'], frozen['simulator_reference_reads']) != (
            'FROZEN', list(ROUTES), list(VIEWS), list(MODES), ids, 0, 0, 0):
        raise ValueError('Expected complete waveform freeze before opening source references')
    for route in ROUTES:
        paired = read(root.parent/SWEEP_RUN/route/'source-predictions.json')
        if paired['physical_ids'] != ids or paired['views'] != list(VIEWS) or paired['status'] != 'PREDICTIONS_FROZEN':
            raise ValueError('Prior frozen source recognition panel does not match these waveforms')
        if any(not (root/route/f'{scene}.npz').is_file() for scene in VIEWS):
            raise ValueError('Incomplete frozen W24 waveform panel')

    results = {route: {} for route in ROUTES}
    with ProcessPoolExecutor(max_workers=rt['cpu_workers']) as pool:
        for scene in VIEWS:
            traces = list(pool.map(partial(packet_components, scene, verify=True), jobs, chunksize=6))
            for route, iq in zip(ROUTES, zip(*traces, strict=True), strict=True):
                observed, signal = (native_aperture(np.stack(waves)) for waves in zip(*iq, strict=True))
                rms = center_rms(observed)
                with np.load(root/route/f'{scene}.npz', allow_pickle=False) as saved:
                    waves = [saved[mode] for mode in MODES]
                np.testing.assert_array_equal(waves[0], observed[..., :384]/rms)
                results[route][scene] = summarize(waves, signal[..., :384]/rms)
            print('W24_WAVEFORM_SCORED', scene, len(jobs), flush=True)
    write(root/'waveform-scores.json', dict(status='SOURCE_SCORED', packets=len(jobs), target_reads=0,
          source_truth_reads=0, reference_after_freeze=True, recognition_comparison=SWEEP_RUN,
          matching_received_iq=True, results=results))


def main():
    if __package__ is None:
        from experiments import pamf_lora_main

        pamf_lora_main.__path__.append(str(Path(__file__).resolve().parent))
    from experiments.cric_dura_comparison.io import read
    from experiments.pamf_lora_main.publish import release, release_commit
    from experiments.pamf_lora_main.w24_native_residual_source import DATA_RUN, SWEEP_RUN, TRAIN_RUN, W24_RUN, dispatch

    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='action', required=True)
    for name in ('dispatch', 'predict-source', 'score-source'):
        sub.add_parser(name).add_argument('runtime', type=Path)
    publish = sub.add_parser('publish')
    for name in ('repo', 'context', 'ssh-config', 'output'):
        publish.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    if args.action == 'dispatch':
        dispatch(args.runtime, entry=__file__)
    elif args.action == 'predict-source':
        predict(read(args.runtime))
    elif args.action == 'score-source':
        score(read(args.runtime))
    else:
        context = read(args.context)
        rt = {key: context[key] for key in ('project', 'python', 'frozen_sources')}
        project = PurePosixPath(rt['project'])
        releases, runs = project/'releases', project/'runs'
        rt.update(run_id=RUN, runner=__spec__.name, code_commit=release_commit(args.repo),
                  release_root=str(releases/f'{RUN}-release'), run_root=str(runs/RUN),
                  source_root=str(releases/f'{DATA_RUN}-release'/'data/source'),
                  w24_release_root=str(releases/f'{W24_RUN}-release'),
                  training_release_root=str(releases/f'{TRAIN_RUN}-release'),
                  allowed_gpus=[3, 2, 7, 4, 1, 6, 0, 5], cpu_workers=8)
        rt['external_releases'] = [rt[key] for key in ('source_root', 'w24_release_root', 'training_release_root')]
        rt['external_releases'].append(str(runs/SWEEP_RUN))
        release(args.repo, args.output, rt, args.ssh_config)


if __name__ == '__main__':
    main()
