"""Compare shared and chirp-specific directions of frozen W24 edits at matched energy."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path, PurePosixPath

import numpy as np

from experiments.cric_dura_comparison.io import read, write


RUN = '20261125-diagnostic-lora-w24-repeat-partition-s2026112559-r01'
MODES = ('direct', 'all_half', 'shared_only', 'distinct_only', 'favor_shared', 'favor_distinct')
CONTRASTS = (('favor_shared', 'all_half'), ('favor_distinct', 'all_half'),
             ('favor_shared', 'favor_distinct'), ('shared_only', 'distinct_only'))


def energy(x):
    return np.sum(np.abs(x)**2, axis=(1, 2), keepdims=True)


def allocate(delta, phase):
    """Project in a unit-phase frame, then compare two allocations at the original dose."""
    aligned = delta * phase.conj()
    shared = aligned.mean(axis=1, keepdims=True) * phase
    distinct = delta - shared
    dose = energy(delta)

    def match(value):
        current = energy(value)
        fraction = np.divide(dose, current, out=np.zeros_like(dose), where=current > 0)
        return value * np.sqrt(fraction)

    return shared, distinct, match(1.25*shared + .75*distinct), match(.75*shared + 1.25*distinct)


def complex_channels(value):
    return value[:, :, 0] + 1j*value[:, :, 1]


def variants(classifier, tracker, packets, direct, half):
    from experiments.cric_dura_comparison.io import tensor
    from experiments.pamf_lora_main.w24_formal import coordinates
    from experiments.pamf_lora_main.w24_native_full import classify_aperture
    from experiments.pamf_lora_main.w24_wide_transfer import received_aperture

    observed = received_aperture(packets, 'cuda:0')
    center = np.from_dlpack(observed[..., :384].cpu().contiguous())
    np.testing.assert_array_equal(complex_channels(center), direct)
    _, phase, _, _ = coordinates(tracker, observed[..., :384])
    rotations = np.from_dlpack(phase.cpu().contiguous())
    np.testing.assert_allclose(np.abs(rotations), 1., rtol=1e-5, atol=1e-6)
    unit = rotations / np.abs(rotations)
    delta = half - direct
    shared, distinct, *matched = allocate(delta, unit)
    for edit in matched:
        np.testing.assert_allclose(energy(edit), energy(delta), rtol=1e-5, atol=1e-10)
    values = (direct, half, *(direct+e for e in (shared, distinct, *matched)))
    outputs = []
    for value in values:
        w24 = observed.clone()
        iq = np.stack((value.real, value.imag), axis=2)
        w24[..., :384] = tensor(iq, 'cuda:0')
        outputs.append(classify_aperture(classifier, w24))
    total = energy(delta).reshape(-1)
    fraction = np.divide(energy(shared).reshape(-1), total, out=np.zeros_like(total), where=total > 0)
    return np.stack(outputs), fraction


def predict(rt):
    import torch
    from experiments.pamf_lora_main.full_packet_views import native, source_contract
    from experiments.pamf_lora_main.runner import activate
    from experiments.pamf_lora_main.w24_factor_audit import load_classifier
    from experiments.pamf_lora_main.w24_native_residual_source import ROUTES, SOURCE, SWEEP_RUN, VIEWS, paired_iq
    from experiments.pamf_lora_main.w24_waveform_audit import RUN as WAVEFORM_RUN, restore, source_rows
    from experiments.pamf_lora_main.w24_wide_transfer import TRAIN_RUN, load_restorer, training_runtime

    activate(rt)
    root, source = Path(rt['run_root']), Path(rt['source_root'])
    contract = source_contract(source)
    classifier = load_classifier(rt, contract, 'cuda:0')
    model, tracker, _, info = load_restorer(*training_runtime(rt), contract, 'cuda:0')
    packet = native(source, 'U_s', contract['counts']['U_s'])[0]
    with torch.no_grad():
        for scene in ('clean', VIEWS[-1]):
            for iq in paired_iq(scene, (0, SOURCE[0], packet)):
                direct, half, _ = restore(model, tracker, iq[None])
                predictions, _ = variants(classifier, tracker, iq[None], direct, half)
                if predictions.shape != (len(MODES), 1):
                    raise ValueError('Exact W24 source U forward smoke failed')
    write(root/'smoke.json', dict(status='PASS', source_packets=1, input_gate=False,
                                  target_reads=0, source_truth_reads=0, parameters=info['parameters']))

    _, jobs, ids = source_rows(rt)
    frozen_root = root.parent / WAVEFORM_RUN
    frozen = read(frozen_root/'waveforms-frozen.json')
    if (frozen['status'], frozen['ids'], frozen['routes'], frozen['views'], frozen['source_frozen_G'], frozen['steps'],
        frozen['simulator_reference_reads'], frozen['source_truth_reads'], frozen['target_reads']) != (
            'FROZEN', ids, list(ROUTES), list(VIEWS), TRAIN_RUN, info['steps'], 0, 0, 0):
        raise ValueError('Expected the frozen prior all-packet W24 IQ outputs')
    output = {route: {} for route in ROUTES}
    energy_share = {route: {} for route in ROUTES}
    batch = 24
    with ProcessPoolExecutor(max_workers=rt['cpu_workers']) as pool, torch.no_grad():
        for scene in VIEWS:
            stream = pool.map(partial(paired_iq, scene), jobs, chunksize=6)
            chunks = list(stream)
            for route, arrays in zip(ROUTES, zip(*chunks, strict=True), strict=True):
                raw = np.stack(arrays)
                with np.load(frozen_root/route/f'{scene}.npz', allow_pickle=False) as arr:
                    direct, half = arr['received'], arr['half']
                results = [variants(classifier, tracker, raw[i:i+batch], direct[i:i+batch], half[i:i+batch])
                           for i in range(0, len(raw), batch)]
                decisions, shares = zip(*results, strict=True)
                output[route][scene] = np.concatenate(decisions, axis=1)
                energy_share[route][scene] = float(np.median(np.concatenate(shares)))
            print('W24_REPEAT_PARTITION_PREDICTED', scene, len(jobs), flush=True)

    for route, decisions in output.items():
        parent = root.parent/SWEEP_RUN/route
        proof = read(parent/'source-predictions.json')
        if (proof['status'], proof['physical_ids'], proof['views'], proof['source_truth_reads'], proof['target_reads']) != (
                'PREDICTIONS_FROZEN', ids, list(VIEWS), 0, 0):
            raise ValueError('Fixed recognition control physical IDs differ')
        indices = [proof['modes'].index(name) for name in ('direct', 'wide_ungated_050')]
        with np.load(parent/'predictions.npz', allow_pickle=False) as controls:
            if any(not np.array_equal(decisions[v][:2], controls[v][indices]) for v in VIEWS):
                raise ValueError(f'{route} direct or half predictions differ from original same-packet controls')
        dest = root/route
        dest.mkdir()
        with (dest/'predictions.npz').open('xb') as out:
            np.savez(out, **decisions)
        write(dest/'source-predictions.json', dict(status='PREDICTIONS_FROZEN', modes=list(MODES), views=list(VIEWS),
              physical_ids=ids, source_truth_reads=0, target_reads=0, waveform=dict(frozen_parent=WAVEFORM_RUN,
              ungated=True, same_dose_modes=list(MODES[-2:]), shared_edit_energy_median=energy_share[route])))


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
    from experiments.pamf_lora_main.w24_waveform_audit import RUN as WAVEFORM_RUN

    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='action', required=True)
    for name in ('dispatch', 'predict-source', 'score-source'):
        sub.add_parser(name).add_argument('runtime', type=Path)
    pub = sub.add_parser('publish')
    for name in ('repo', 'context', 'ssh-config', 'output'):
        pub.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    if args.action == 'dispatch':
        dispatch(args.runtime, entry=__file__)
    elif args.action == 'predict-source':
        predict(read(args.runtime))
    elif args.action == 'score-source':
        score(args.runtime)
    else:
        context = read(args.context)
        rt = {key: context[key] for key in ('project', 'python', 'frozen_sources')}
        project = PurePosixPath(rt['project'])
        runs, releases = project/'runs', project/'releases'
        rt.update(run_id=RUN, runner=__spec__.name, code_commit=release_commit(args.repo),
                  release_root=str(releases/f'{RUN}-release'), run_root=str(runs/RUN),
                  source_root=str(releases/f'{DATA_RUN}-release'/'data/source'), classifier_root=str(runs/AUG_RUN),
                  w24_release_root=str(releases/f'{W24_RUN}-release'),
                  training_release_root=str(releases/f'{TRAIN_RUN}-release'),
                  allowed_gpus=[3, 7, 2, 4, 1, 6, 0, 5], cpu_workers=8)
        rt['external_releases'] = [rt[k] for k in ('source_root','classifier_root','w24_release_root','training_release_root')]
        rt['external_releases'].extend(str(runs/name) for name in (SWEEP_RUN, WAVEFORM_RUN))
        release(args.repo, args.output, rt, args.ssh_config)


if __name__ == '__main__':
    main()
