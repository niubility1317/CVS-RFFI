"""Attribute frozen WiSig IQ outputs on an existing source-only repeat panel."""

import argparse
from itertools import product
from pathlib import Path, PurePosixPath

import numpy as np

from experiments.cric_dura_comparison.io import read, write


RUN = '20261125-diagnostic-wisig-waveform-source-s2026112501-r01'
PANEL = '20261125-diagnostic-wisig-action-noise-s2026112501-r01'
PANEL_CHECK = '20261125-diagnostic-wisig-action-noise-s2026112501-r02'
ROUTES = {'full': 'validation_full', 'residual': 'validation'}
MODES = ('direct', 'r69', 'centered_coarse', 'centered_map_1', 'centered_final')
BANDS = (('0_to_4_mhz', 0., 4.), ('4_to_8_mhz', 4., 8.), ('8_to_12p5_mhz', 8., 13.))


def waveform_errors(wave, reference):
    if wave.shape != reference.shape or wave.ndim != 2 or wave.shape[1] != 256:
        raise ValueError('Expected matching packet and native 256-point time axes')
    frequency = np.abs(np.fft.fftfreq(256, d=1/25))
    estimate, expected = (np.fft.fft(x, norm='ortho') for x in (wave, reference))
    results = {}
    for band, low, high in BANDS:
        keep = (frequency >= low) & (frequency < high)
        z, x = estimate[:, keep], expected[:, keep]
        za, xa = np.abs(z), np.abs(x)
        energy = (xa**2).sum(axis=-1)
        scale = np.maximum(energy, 1e-12)
        cosine = (z * x.conj()).real / np.maximum(za*xa, 1e-20)
        results[band] = dict(reference_energy=energy, complex_nmse=(np.abs(z-x)**2).sum(axis=-1)/scale,
                             amplitude_nmse=((za-xa)**2).sum(axis=-1)/scale,
                             phase_chord_loss=((1-np.clip(cosine, -1, 1))*xa**2).sum(axis=-1)/scale)
    return results


def panel(rt, parent):
    root = Path(parent['source_root'])
    audit = Path(rt['panel_root'])
    contract = read(root/'complete.json')
    if (contract['status'], contract['source_V_count'], contract['target_reads'], contract['TX_labels_read']) != (
            'RECEPTION_SOURCE_CACHE_COMPLETE', 1800, 0, 0):
        raise ValueError('Expected the original TX-blind WiSig source validation cache')
    if read(audit.parent/PANEL_CHECK/'state.json')['status'] != 'ANALYZED':
        raise ValueError('Original isolated channel replay requires verified completion')
    residual = root/ROUTES['residual']
    ids, rx, day, roles = (np.load(residual/f'{name}.npy', allow_pickle=False)
                          for name in ('physical_ids', 'rx', 'day', 'roles'))
    full = np.load(root/ROUTES['full']/'physical_ids.npy', allow_pickle=False)
    if len(ids) != 1800 or len(set(ids)) != 1800 or set(roles) != {'V'} or not np.array_equal(full, ids):
        raise ValueError('Expected the same 1,800 source V physical IDs on both routes')
    rng = np.random.default_rng(2026112501)
    strata = product((1, 3, 4, 6, 8), (1, 2, 3))
    chosen = np.concatenate([rng.choice(np.flatnonzero((rx==r) & (day==d)), 24, replace=False) for r,d in strata])
    return root, ids, chosen, rx[chosen]


def predict(rt, parent):
    import torch
    from experiments.cric_dura_comparison.io import tensor
    from experiments.pamf_unrolled_restore.wisig_anchored_window_source import classify, core, tri
    from experiments.pamf_unrolled_restore.wisig_native_fir_source import MODES as ORIGINAL_MODES
    from experiments.pamf_unrolled_restore.wisig_native_fir_anchor_probe import models_from
    from experiments.pamf_unrolled_restore.wisig_uncertain_prior import as_complex

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    root = Path(rt['run_root'])
    cache, ids, chosen, _ = panel(rt, parent)
    engine = core.baseline.original_r69(Path(parent['parent_root']))
    (model, _), _ = models_from(rt)
    reader = tri.reader_factory(parent).eval().requires_grad_(False)
    modes = list(MODES)
    source_predictions = Path(parent['run_root'])/'predictions'
    receipt = read(source_predictions/'complete.json')
    if (receipt['status'], receipt['count'], receipt['modes'], receipt['source_truth_reads'], receipt['target_reads']) != (
            'FROZEN', len(ids), list(ORIGINAL_MODES), 0, 0):
        raise ValueError('Expected matching archived source decisions')

    def output(received):
        (r69,) = engine(received)
        signal = as_complex(received)
        stages, _ = model.stages(model.analyze(signal))
        return signal, as_complex(r69), stages[0], stages[2], stages[4]

    with torch.no_grad():
        source, _, indices = core.source_iq(cache)
        u = tensor(source[indices[:4]], 'cuda:0')
        for wave in output(u):
            classify(reader, wave)
        write(root/'smoke.json', dict(status='PASS', source_packets=4, source_V_reads=0, target_reads=0,
                                    model_calls_per_packet=2, input_gate=False))
        for route, view in product(ROUTES, core.baseline.VIEWS):
            cached = np.load(cache/ROUTES[route]/f'{view}.npy', mmap_mode='r', allow_pickle=False)
            if cached.shape != (1800, 2, 256) or cached.dtype != np.float32:
                raise ValueError(f'Source IQ shape differs for {route}/{view}')
            chunks, decisions = [], []
            for offset in range(0, len(chosen), 64):
                samples = tensor(cached[chosen[offset:offset+64]], 'cuda:0')
                readouts = output(samples)
                chunks.append(np.stack([np.from_dlpack(z.cpu().contiguous()) for z in readouts]))
                decisions.append([classify(reader, z).argmax(-1).tolist() for z in readouts])
            wave = np.concatenate(chunks, axis=1)
            pred = np.concatenate(decisions, axis=1).astype(np.uint8)
            with np.load(source_predictions/f'{route}_{view}.npz', allow_pickle=False) as archived:
                old = archived['decision'][[ORIGINAL_MODES.index(name) for name in modes]][:, chosen]
                if not np.array_equal(archived['ids'], ids) or not np.array_equal(old, pred):
                    raise ValueError(f'The selected {route}/{view} decisions do not reproduce the archived panel')
            if wave.shape != (len(MODES), len(chosen), 256) or not np.isfinite(wave).all():
                raise ValueError('Incomplete or nonfinite frozen WiSig waveform')
            with (root/f'{route}_{view}.npz').open('xb') as stream:
                np.savez(stream, iq=wave, decisions=pred, ids=ids[chosen])
            print('WISIG_WAVEFORM_FROZEN', route, view, len(chosen), flush=True)
    write(root/'waveforms-frozen.json', dict(status='FROZEN', modes=modes, routes=list(ROUTES),
          views=list(core.baseline.VIEWS), packets=len(chosen), source_truth_reads=0, target_reads=0,
          simulator_reference_reads=0, all_archive_decisions_match=True))


def score(rt, parent):
    from experiments.pamf_unrolled_restore.wisig_anchored_window_source import core

    cache, ids, chosen, rx = panel(rt, parent)
    root, audit = Path(rt['run_root']), Path(rt['panel_root'])
    views = core.baseline.VIEWS
    receipt = read(root/'waveforms-frozen.json')
    expected = dict(status='FROZEN', modes=list(MODES), routes=list(ROUTES), views=list(views),
                    packets=len(chosen), source_truth_reads=0, target_reads=0, simulator_reference_reads=0,
                    all_archive_decisions_match=True)
    if receipt != expected or not all((root/f'{r}_{v}.npz').is_file() for r,v in product(ROUTES, views)):
        raise ValueError('Both complete frozen waveform routes are required before reading references')
    _, _, label_root, _, bound_ids, _, positions = core.baseline.handles(parent)
    if not np.array_equal(bound_ids, ids):
        raise ValueError('Source truth contract has different physical identities')
    truth = np.load(label_root/'V'/'labels.npy', allow_pickle=False)[positions[chosen]]
    recorded = np.load(audit/'received.npy', mmap_mode='r', allow_pickle=False)
    results = {route: {} for route in ROUTES}
    with np.load(audit/'scorer_reference.npz', allow_pickle=False) as isolation:
        clean = isolation['clean']
        signal = isolation['signal']
        if recorded.shape != (24, len(chosen), 2, 256) or signal.shape != (24, len(chosen), 256):
            raise ValueError('Isolation reference does not match the registered four-profile panel')
        for route_no, (route, folder) in enumerate(ROUTES.items()):
            for i, view in enumerate(views):
                with np.load(root/f'{route}_{view}.npz', allow_pickle=False) as bundle:
                    wave, pred = bundle['iq'], bundle['decisions']
                    if not np.array_equal(bundle['ids'], ids[chosen]):
                        raise ValueError(f'Frozen source array IDs differ for {route}/{view}')
                if i==0:
                    np.testing.assert_array_equal(wave[0], clean)
                    ref = clean
                else:
                    cell = route_no*(len(views)-1)+i-1
                    iq = recorded[cell]
                    np.testing.assert_allclose(wave[0], iq[:, 0]+1j*iq[:,1], rtol=2e-5, atol=3e-6)
                    ref = signal[cell]
                values = [waveform_errors(z, ref) for z in wave]
                bands = {}
                for name, _, _ in BANDS:
                    valid = values[0][name]['reference_energy'] > 1e-12
                    metrics = {field: dict(zip(MODES, [float(np.median(a[name][field][valid]))
                                                       if valid.any() else None for a in values], strict=True))
                               for field in ('complex_nmse', 'amplitude_nmse', 'phase_chord_loss')}
                    bands[name] = dict(supported=int(valid.sum()), **metrics)
                hit = pred==truth
                groups = {str(r): hit[:, rx==r].sum(axis=-1).tolist() for r in np.unique(rx)}
                results[route][view] = dict(bands=bands, correct=hit.sum(axis=-1).tolist(), correct_by_rx=groups,
                                           repaired=(hit & ~hit[:1]).sum(axis=-1).tolist(),
                                           damaged=(~hit & hit[:1]).sum(axis=-1).tolist())
                print('WISIG_WAVEFORM_SCORED', route, view, len(chosen), flush=True)
    write(root/'waveform-scores.json', dict(status='SOURCE_SCORED', modes=list(MODES), packets=len(chosen),
          source_truth_last=True, oracle_last=True, target_reads=0, reference='no_added_noise_receive_frame',
          results=results))


def main():
    if __package__ is None:
        from experiments import pamf_unrolled_restore

        pamf_unrolled_restore.__path__.append(str(Path(__file__).resolve().parent))
    from experiments.pamf_lora_main.publish import release, release_commit
    from experiments.pamf_unrolled_restore.wisig_native_fir_anchor_probe import PARENT, dispatch, reference

    parser = argparse.ArgumentParser()
    actions = parser.add_subparsers(dest='action', required=True)
    for action in ('dispatch','predict','score'):
        actions.add_parser(action).add_argument('runtime', type=Path)
    publish = actions.add_parser('publish')
    for field in ('repo','output','context','ssh-config'):
        publish.add_argument('--'+field, type=Path, required=True)
    args = parser.parse_args()
    if args.action == 'dispatch':
        dispatch(args.runtime, entry=__file__)
    elif args.action == 'publish':
        context = read(args.context)
        rt = {key: context[key] for key in ('project','python','frozen_sources')}
        project = PurePosixPath(rt['project'])
        releases, runs = project/'releases', project/'runs'
        parent, panel_root = releases/f'{PARENT}-release', runs/PANEL
        rt.update(run_id=RUN, runner=__spec__.name, code_commit=release_commit(args.repo), reference=str(parent),
                  release_root=str(releases/f'{RUN}-release'), run_root=str(runs/RUN), panel_root=str(panel_root),
                  external_releases=list(map(str, (parent, runs/PARENT, panel_root, runs/PANEL_CHECK))),
                  max_compute_processes=8)
        release(args.repo, args.output, rt, args.ssh_config)
    else:
        from experiments.pamf_lora_main.runner import activate

        rt = read(args.runtime)
        parent = reference(rt)
        activate(parent)
        (predict if args.action == 'predict' else score)(rt, parent)


if __name__ == '__main__':
    main()
