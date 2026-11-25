"""Source-only attribution of added noise across the W24 input coordinates."""

from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np

from experiments.cric_dura_comparison.io import read, tensor, write
from experiments.pamf_lora_main.full_packet_views import native, source_contract
from experiments.pamf_lora_main.runner import activate
from experiments.pamf_lora_main.w24_factor_audit import (
    APERTURE_REFERENCE, SOURCE, VIEWS, frozen_source_panel, load_classifier, panel, score_source, trace_components,
)
from experiments.pamf_lora_main.w24_native_full import EDGE, classify_aperture, native_full_view
from experiments.pamf_lora_main.w24_native_residual import ROUTE, native_residual
from experiments.pamf_lora_main.w24_native_residual_source import WAVEFORM_NAMESPACE


ROUTES = ('full_native', ROUTE)
CENTER = 384
MODES = ('direct', 'center_only', 'edges_only', 'outer_chirps_only', 'inner_chirps_only',
         'all_fixed_rms', 'all_own_rms')
CONTRASTS = (('all_fixed_rms', 'center_only'), ('all_fixed_rms', 'edges_only'),
             ('all_own_rms', 'all_fixed_rms'), ('all_fixed_rms', 'inner_chirps_only'))


def packet_components(scene, item, *, verify=False):
    row, rx, packet = item
    args = scene, rx, row, WAVEFORM_NAMESPACE
    if scene == 'clean':
        return (packet, packet), (packet, packet)
    full, full_signal, _ = trace_components(packet, *args)
    residual, residual_signal, _ = native_residual(packet, *args, component_trace=True)
    if verify:
        if (not np.array_equal(full, native_full_view(packet, *args))
                or not np.array_equal(residual, native_residual(packet, *args))):
            raise ValueError('The component trace changed the registered received IQ')
    return (full, full_signal.astype(np.complex64)), (residual, residual_signal.astype(np.complex64))


def native_aperture(packets):
    from scipy.signal import resample_poly
    from experiments.pamf_lora_main.clean_aperture import ALL_CHIRPS, CROPS, native_edge_positions

    padding = CENTER // 8
    window = resample_poly(packets[:, CROPS[ALL_CHIRPS]], up=padding, down=125, axis=-1)
    edges = np.asarray(ALL_CHIRPS)[:, None] * 1024 + np.asarray(native_edge_positions(EDGE))
    return np.concatenate((window[..., padding:padding + CENTER], packets[:, edges]), axis=-1)


def center_rms(aperture):
    rms = np.sqrt(np.mean(np.abs(aperture[..., :CENTER])**2, axis=(1, 2), keepdims=True))
    if not np.isfinite(rms).all() or (rms <= 1e-12).any():
        raise ValueError('Nonfinite or empty W24 aperture')
    return rms


def model_iq(aperture, rms):
    value = aperture / rms
    return np.stack((value.real, value.imag), axis=2).astype(np.float32, copy=False)


def hybrid_apertures(received, signal):
    observed, exact = native_aperture(received), native_aperture(signal)
    rms = center_rms(observed)
    yield model_iq(observed, rms)
    centers = np.arange(observed.shape[-1]) < CENTER
    outer = np.isin(np.arange(observed.shape[-2]), (0, 7))[:, None]
    for mask in (centers, ~centers, outer, ~outer):
        yield model_iq(np.where(mask, exact, observed), rms)
    yield model_iq(exact, rms)
    yield model_iq(exact, center_rms(exact))


def predict(rt):
    from experiments.pamf_lora_main.clean_aperture import ALL_CHIRPS, clean_iq

    activate(rt)
    root, source, device = Path(rt['run_root']), Path(rt['source_root']), 'cuda:0'
    contract = source_contract(source)
    model = load_classifier(rt, contract, device)
    u = native(source, 'U_s', contract['counts']['U_s'])[:1]

    def decisions(aperture):
        return classify_aperture(model, tensor(aperture, device))

    for scene in ('clean', VIEWS[1], VIEWS[-1]):
        for received, signal in packet_components(scene, (0, SOURCE[0], u[0]), verify=True):
            variants = list(hybrid_apertures(received[None], signal[None]))
            for implemented, reference in ((variants[0], received), (variants[-1], signal)):
                np.testing.assert_array_equal(implemented, clean_iq(
                    reference[None], CENTER, ALL_CHIRPS, native_edges=True, edge_points=EDGE))
            for variant in variants:
                decisions(variant)
    write(root / 'smoke.json', dict(status='PASS', routes=list(ROUTES), target_reads=0,
          source_truth_reads=0, exact_observation_replay=True, exact_input_aperture=True))

    rows, rx, ids = panel(source, contract)
    reference = root.parent / APERTURE_REFERENCE
    for route in ROUTES:
        old = read(reference / route / 'source-predictions.json')
        if (old['status'], old['physical_ids'], old['views'], old['target_reads']) != (
                'PREDICTIONS_FROZEN', ids, list(VIEWS), 0) or old['modes'][0] != 'direct':
            raise ValueError('Expected the registered two-route blind source panel')
    raw = native(source, 'V', contract['counts']['V'])[rows]
    clean = decisions(next(hybrid_apertures(raw, raw)))
    shared_clean = np.repeat(clean[None], len(MODES), axis=0)
    predictions = {route: {'clean': shared_clean} for route in ROUTES}
    jobs = [(int(row), SOURCE[int(receiver)], packet) for row, receiver, packet in zip(rows, rx, raw, strict=True)]
    with ProcessPoolExecutor(max_workers=rt['cpu_workers']) as pool:
        for scene in VIEWS[1:]:
            waveforms = pool.map(partial(packet_components, scene), jobs, chunksize=6)
            batches = {route: [] for route in ROUTES}
            for offset in range(0, len(jobs), 24):
                chunks = [next(waveforms) for _ in range(min(24, len(jobs)-offset))]
                for route, paired in zip(ROUTES, zip(*chunks, strict=True), strict=True):
                    obs, sig = (np.stack(values) for values in zip(*paired, strict=True))
                    batches[route].append(np.stack([decisions(x) for x in hybrid_apertures(obs, sig)]))
            for route in ROUTES:
                predictions[route][scene] = np.concatenate(batches[route], axis=1)
            print('W24_APERTURE_NOISE_PREDICTED', scene, len(jobs), flush=True)
    for route, matrix in predictions.items():
        with np.load(reference / route / 'predictions.npz', allow_pickle=False) as old:
            if any(not np.array_equal(matrix[view][0], old[view][0]) for view in VIEWS):
                raise ValueError(f'Direct predictions differ from the paired reference for {route}')
        out = root / route
        out.mkdir()
        np.savez(out / 'predictions.npz', **matrix)
        write(out / 'source-predictions.json', dict(status='PREDICTIONS_FROZEN', modes=list(MODES),
              views=list(VIEWS), physical_ids=ids, target_reads=0, source_truth_reads=0,
              waveform=dict(route=route, source_oracle_only=True, direct_parity=True,
                            normalization='observed_center_rms_except_all_own_rms')))


def score(runtime):
    rt = read(runtime)
    root = Path(rt['run_root'])
    for route in ROUTES:
        frozen_source_panel(Path(rt['source_root']), root / route, MODES)
    for route in ROUTES:
        score_source(runtime, modes=MODES, output=root / route, contrasts=CONTRASTS)
