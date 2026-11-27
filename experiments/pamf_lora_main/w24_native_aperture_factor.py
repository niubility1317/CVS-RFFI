"""Isolate learned center/edge edits and equal-energy noise directions after native DURA."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path, PurePosixPath

import numpy as np
import torch

if __package__ is None:
    from experiments import pamf_lora_main

    pamf_lora_main.__path__.append(str(Path(__file__).resolve().parent))

from experiments.cric_dura_comparison.io import read, tensor, write
from experiments.pamf_lora_main.full_packet_views import native, source_contract
from experiments.pamf_lora_main.runner import activate
from experiments.pamf_lora_main.w24_aperture_noise_audit import CENTER, center_rms, model_iq, native_aperture
from experiments.pamf_lora_main.w24_edit_budget_audit import full_components
from experiments.pamf_lora_main.w24_factor_audit import AUG_RUN, frozen_source_panel, load_classifier, panel, score_source
from experiments.pamf_lora_main.w24_native_full import DATA_RUN, SOURCE, VIEWS, W24_RUN, classify_aperture
from experiments.pamf_lora_main.w24_native_residual_source import BATCH, dispatch, mode_list
from experiments.pamf_lora_main.w24_third_prior import NATIVE_SPECTRAL_ARMS, NATIVE_SPECTRAL_RUN, load_candidate
from experiments.pamf_lora_main.w24_wide_transfer import TRAIN_RUN, load_restorer, received_aperture, training_runtime


RUN = '20261127-diagnostic-lora-w24-native-aperture-factor-s2026112559-r01'
FULL = 'full_native'
MODES = ('direct', 'control_half', 'band_half', 'band_center', 'band_edges',
         'equal_center', 'equal_edges', 'equal_both', 'exact_center', 'exact_edges', 'exact_both')
CONTRASTS = (('band_half', 'control_half'), ('band_center', 'control_half'), ('band_edges', 'control_half'),
             ('band_half', 'band_center'), ('band_half', 'band_edges'),
             ('equal_center', 'control_half'), ('equal_edges', 'control_half'), ('equal_both', 'control_half'),
             ('exact_center', 'equal_center'), ('exact_edges', 'equal_edges'), ('exact_both', 'equal_both'))
REGIONS = (slice(0, CENTER), slice(CENTER, None))


def power(iq):
    return (np.abs(iq)**2).sum(axis=(1, 2), keepdims=True)


def equal_direction(observed, learned, exact):
    chosen, ideal = learned - observed, exact - observed
    p, q = power(chosen), power(ideal)
    scale = np.sqrt(np.divide(p, q, out=np.zeros_like(p), where=q > 1e-15))
    return np.where(q > 1e-15, observed + ideal * scale, learned)


def hybrids(observed, control, band, ideal):
    def combine(centers, edges):
        return np.concatenate((centers[..., :CENTER], edges[..., CENTER:]), axis=-1)

    equal = np.concatenate([equal_direction(observed[..., r], control[..., r], ideal[..., r])
                            for r in REGIONS], axis=-1)
    exact = np.where((power(ideal - observed) > 1e-15), ideal, control)
    variants = (observed, control, band, combine(band, control), combine(control, band),
                combine(equal, control), combine(control, equal), equal,
                combine(exact, control), combine(control, exact), exact)
    # Every center substitution uses the deployed center-based packet scale.
    return [model_iq(iq, center_rms(iq)) for iq in variants]


def noise_measures(observed, control, band, ideal):
    rows = []
    for region in REGIONS:
        y, a, b, x = (iq[..., region] for iq in (observed, control, band, ideal))
        noise = power(x - y)
        for wave in (a, b):
            edit = wave - y
            edit_energy = power(edit)
            scale = np.sqrt(np.maximum(edit_energy * noise, 1e-20))
            cosine = ((edit.conj() * (x - y)).real.sum(axis=(1, 2), keepdims=True) / scale)
            residue = np.divide(power(wave - x), noise, out=np.zeros_like(noise), where=noise > 1e-15)
            rows.extend((np.sqrt(edit_energy / np.maximum(power(y), 1e-12)), cosine, residue))
    return np.concatenate(rows, axis=-1)[:, 0]


@torch.inference_mode()
def render(f, models, received, signal, device):
    raw = tensor(received.view(np.float32), device).view(torch.complex64)
    restored = [np.from_dlpack(torch.lerp(raw, g.restore_native(raw), .5).cpu().contiguous())
                for g in models]
    aperture = [native_aperture(iq) for iq in (received, *restored, signal)]
    variants = hybrids(*aperture)
    return np.stack([classify_aperture(f, tensor(iq, device)) for iq in variants]), noise_measures(*aperture)


def predict(rt):
    activate(rt)
    root, source = (Path(rt[n]) for n in ('run_root', 'source_root'))
    parent = root.parent / NATIVE_SPECTRAL_RUN
    if read(parent / 'state.json') != dict(status='SOURCE_SCORED', promote=False,
                                         arms=list(NATIVE_SPECTRAL_ARMS), steps=24000):
        raise ValueError('Expected both previously frozen native source models')
    device = 'cuda:0'
    contract = source_contract(source)
    f = load_classifier(rt, contract, device)
    load_restorer(*training_runtime(rt), contract, device)
    models = tuple(load_candidate(parent, contract, device, name)[0] for name in NATIVE_SPECTRAL_ARMS)
    smoke_source = native(source, 'U_s', contract['counts']['U_s'])[0]
    for scene in ('clean', VIEWS[-1]):
        y, x = full_components(scene, (0, SOURCE[0], smoke_source), verify=True)
        obs = native_aperture(y[None])
        expected = received_aperture(y[None], 'cpu')
        np.testing.assert_array_equal(model_iq(obs, center_rms(obs)), np.from_dlpack(expected))
        preds, detail = render(f, models, y[None], x[None], device)
        if preds.shape != (len(MODES), 1) or detail.shape != (1, 12) or not np.isfinite(detail).all():
            raise ValueError('Wrong real-checkpoint source-only aperture smoke')
        if scene == 'clean' and not np.all(preds[5:] == preds[1]):
            raise ValueError('Noise-free oracle edits must retain the matched control')
    write(root / 'smoke.json', dict(status='PASS', source_u_count=1, target_reads=0,
                                   source_truth_reads=0, native_capture_samples=8192, exact_aperture=True))

    rows, rx, ids = panel(source, contract)
    raw = native(source, 'V', contract['counts']['V'])[rows]
    jobs = [(int(row), SOURCE[int(receiver)], iq) for row, receiver, iq in zip(rows, rx, raw, strict=True)]
    output, detail = {}, {}
    with ProcessPoolExecutor(max_workers=rt['cpu_workers']) as pool:
        for scene in VIEWS:
            stream = pool.map(partial(full_components, scene), jobs, chunksize=6)
            batches = []
            for offset in range(0, len(jobs), BATCH):
                chunk = [next(stream) for _ in range(min(BATCH, len(jobs)-offset))]
                y, x = (np.stack(a) for a in zip(*chunk, strict=True))
                batches.append(render(f, models, y, x, device))
            preds, measurements = zip(*batches, strict=True)
            output[scene] = np.concatenate(preds, axis=1)
            detail[scene] = np.concatenate(measurements)
            print('NATIVE_APERTURE_FACTOR_PREDICTED', scene, len(jobs), flush=True)
    previous = parent / FULL
    old_modes = mode_list({'third_prior': 'native-band-risk'})
    _, _, old, _ = frozen_source_panel(source, previous, old_modes)
    ref = [old_modes.index(name) for name in ('direct', *(f'{n}_half' for n in NATIVE_SPECTRAL_ARMS))]
    if not np.array_equal(np.stack([output[v][:3] for v in VIEWS]), old[:, ref]):
        raise ValueError('Native frozen direct and two learned endpoints must match exactly')
    out = root / FULL
    out.mkdir()
    np.savez(out / 'predictions.npz', **output)
    np.savez(out / 'aperture-measurements.npz', **detail)
    write(out / 'source-predictions.json', dict(status='PREDICTIONS_FROZEN', modes=list(MODES),
          views=list(VIEWS), physical_ids=ids, target_reads=0, source_truth_reads=0,
          waveform=dict(route=FULL, oracle_only=True, parent=NATIVE_SPECTRAL_RUN,
                        equal_region_energy=True, own_center_scale=True,
                        measurement_order=['center_control', 'center_band', 'edge_control', 'edge_band'],
                        per_measurement=['edit_over_received_rms', 'cosine', 'remaining_noise_power'])))


def main():
    from experiments.pamf_lora_main.publish import release, release_commit

    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest='action', required=True)
    for phase in ('dispatch', 'predict-source', 'score-source'):
        actions.add_parser(phase).add_argument('runtime', type=Path)
    pub = actions.add_parser('publish')
    for name in ('repo', 'context', 'ssh-config', 'output'):
        pub.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    if args.action == 'dispatch':
        dispatch(args.runtime, entry=__file__)
    elif args.action == 'predict-source':
        predict(read(args.runtime))
    elif args.action == 'score-source':
        out = Path(read(args.runtime)['run_root']) / FULL
        score_source(args.runtime, modes=MODES, output=out, contrasts=CONTRASTS)
    else:
        ctx = read(args.context)
        rt = {key: ctx[key] for key in ('project', 'python', 'frozen_sources')}
        project = PurePosixPath(rt['project'])
        runs, releases = project / 'runs', project / 'releases'
        rt.update(run_id=RUN, runner=__spec__.name, code_commit=release_commit(args.repo),
                  release_root=str(releases / f'{RUN}-release'), run_root=str(runs / RUN),
                  source_root=str(releases / f'{DATA_RUN}-release' / 'data/source'),
                  classifier_root=str(runs / AUG_RUN), w24_release_root=str(releases / f'{W24_RUN}-release'),
                  training_release_root=str(releases / f'{TRAIN_RUN}-release'), allowed_gpus=[0, 3, 5, 6], cpu_workers=8)
        rt['external_releases'] = [rt[k] for k in ('source_root', 'classifier_root', 'w24_release_root',
                                                   'training_release_root')] + [str(runs / NATIVE_SPECTRAL_RUN)]
        release(args.repo, args.output, rt, args.ssh_config)


if __name__ == '__main__':
    main()
