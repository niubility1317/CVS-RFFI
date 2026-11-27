"""Source-only separation of the magnitude and direction of one frozen DURA edit."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import os
from pathlib import Path, PurePosixPath
import subprocess

import numpy as np
import torch

if __package__ is None:
    from experiments import pamf_lora_main

    pamf_lora_main.__path__.append(str(Path(__file__).resolve().parent))

from experiments.cric_dura_comparison.io import read, tensor, write
from experiments.pamf_lora_main.full_packet_views import native, source_contract
from experiments.pamf_lora_main.joint_unroll_source import coordinates
from experiments.pamf_lora_main.runner import activate
from experiments.pamf_lora_main.w24_aperture_noise_audit import CENTER, center_rms, model_iq, native_aperture
from experiments.pamf_lora_main.w24_edit_budget_audit import full_components
from experiments.pamf_lora_main.w24_factor_audit import AUG_RUN, load_classifier, panel, score_source
from experiments.pamf_lora_main.w24_native_full import DATA_RUN, SOURCE, VIEWS, W24_COMMIT, W24_RUN, classify_aperture
from experiments.pamf_lora_main.w24_third_prior import PRACTICAL_ARMS, PRACTICAL_RUN, load_candidate
from experiments.pamf_lora_main.w24_wide_transfer import TRAIN_RUN, load_restorer, received_aperture, training_runtime


RUN = '20261127-diagnostic-lora-w24-practical-direction-factor-s2026112559-r01'
ALLOCATION_RUN = '20261128-diagnostic-lora-w24-component-allocation-s2026112559-r02'
REFERENCE = '20261127-diagnostic-lora-w24-native-aperture-factor-s2026112559-r02'
ARM = PRACTICAL_ARMS[1]
MODES = ('direct', 'frozen_half', 'true_amplitude', 'true_direction', 'true_both', 'projected_amplitude')
CONTRASTS = (('true_amplitude', 'frozen_half'), ('true_direction', 'frozen_half'),
             ('true_both', 'frozen_half'), ('true_both', 'true_direction'),
             ('true_both', 'true_amplitude'), ('projected_amplitude', 'frozen_half'))
ALLOCATION_MODES = ('direct', 'frozen_half', 'component_norms', 'component_directions',
                    'component_both', 'exact_center')
ALLOCATION_CONTRASTS = (*((row, 'frozen_half') for row in ALLOCATION_MODES[2:]),
                        ('component_both', 'component_norms'), ('component_both', 'component_directions'),
                        ('exact_center', 'component_both'))
BATCH = 16


def interventions(received, frozen, signal):
    """Each region uses its own L2 norm. Zero-noise regions keep the learned edit."""
    variants = [[], [], [], []]
    for region in (slice(None, CENTER), slice(CENTER, None)):
        y, g, s = (v[..., region] for v in (received, frozen, signal))
        edit, exact = g - y, s - y
        model_power, true_power = ((abs(v)**2).sum((1, 2), keepdims=True) for v in (edit, exact))
        active = (true_power > 1e-15) & (model_power > 1e-15)
        rho = ((edit.conj()*exact).real).sum((1, 2), keepdims=True)
        true_amp = y + edit * np.sqrt(true_power / np.maximum(model_power, 1e-15))
        true_dir = y + exact * np.sqrt(model_power / np.maximum(true_power, 1e-15))
        best_amp = y + edit * np.maximum(rho, 0) / np.maximum(model_power, 1e-15)
        for columns, output in zip(variants, (true_amp, true_dir, s, best_amp), strict=True):
            valid = true_power > 1e-15 if columns is variants[2] else active
            columns.append(np.where(valid, output, g))
    return received, frozen, *(np.concatenate(v, axis=-1) for v in variants)


@torch.no_grad()
def render(head, recovery, y, s):
    received = tensor(np.ascontiguousarray(y).view(np.float32), 'cpu').view(torch.complex64)
    g = np.from_dlpack(torch.lerp(received, recovery.restore_native(received), .5).contiguous())
    ablations = interventions(*(native_aperture(v) for v in (y, g, s)))
    # Use each readout's actual W24 center RMS, including for raw-edge edits.
    return np.stack([classify_aperture(head, tensor(model_iq(v, center_rms(v)), 'cpu')) for v in ablations])


@torch.no_grad()
def render_allocations(head, recovery, tracker, y, s):
    from experiments.pamf_lora_main.w24_component_allocation import counterfactual_edits

    packets = tensor(np.ascontiguousarray(y).view(np.float32), 'cpu').view(torch.complex64)
    restored = np.from_dlpack(torch.lerp(packets, recovery.restore_native(packets), .5).contiguous())
    received, frozen, signal = (native_aperture(value) for value in (y, restored, s))
    original_iq = tensor(model_iq(received, center_rms(received)), 'cpu')
    _, phase, _, _ = coordinates(tracker, original_iq[..., :CENTER])
    original = received[..., :CENTER]
    estimated, exact = (tensor(np.ascontiguousarray(wave[..., :CENTER] - original,
                                                   dtype=np.complex64).view(np.float32), 'cpu').view(torch.complex64)
                        for wave in (frozen, signal))
    edits = counterfactual_edits(estimated, exact, phase)
    active = np.any(signal[..., :CENTER] != original, axis=(1, 2))
    rows = [received, frozen]
    for edit in edits:
        candidate = frozen.copy()
        candidate[active, :, :CENTER] = original[active] + np.from_dlpack(edit.contiguous())[active]
        rows.append(candidate)
    return np.stack([classify_aperture(head, tensor(model_iq(v, center_rms(v)), 'cpu')) for v in rows])


def predict(rt):
    activate(rt)
    torch.set_num_threads(2)
    root, source, candidate = (Path(rt[k]) for k in ('run_root', 'source_root', 'candidate_root'))
    if candidate != root.parent/PRACTICAL_RUN or read(candidate/'state.json')['status'] != 'SOURCE_FROZEN':
        raise ValueError('Expected verified frozen r02 source weights')
    contract = source_contract(source)
    head = load_classifier(rt, contract, 'cpu')
    _, tracker, _, _ = load_restorer(*training_runtime(rt), contract, 'cpu')
    model, evidence = load_candidate(candidate, contract, 'cpu', ARM)
    allocation = rt['run_id'] == ALLOCATION_RUN
    modes = ALLOCATION_MODES if allocation else MODES
    classification = partial(render_allocations, head, model, tracker) if allocation else partial(render, head, model)
    clean = native(source, 'U_s', contract['counts']['U_s'])[0]
    for scene in ('clean', VIEWS[-1]):
        y, s = full_components(scene, (0, SOURCE[0], clean), verify=True)
        obs = native_aperture(y[None])
        np.testing.assert_array_equal(model_iq(obs, center_rms(obs)), np.from_dlpack(received_aperture(y[None], 'cpu')))
        decisions = classification(y[None], s[None])
        if decisions.shape != (len(modes), 1) or (scene == 'clean' and not np.all(decisions[1:] == decisions[1])):
            raise ValueError('Unmatched real source U checkpoint smoke')
    write(root/'smoke.json', dict(status='PASS', classifier_epoch=104, source_role='U_s', source_packets=1,
                                 candidate_steps=evidence['steps'], target_reads=0, source_truth_reads=0))
    rows, rx, ids = panel(source, contract)
    reference = root.parent/RUN if allocation else root.parent/REFERENCE/'full_native'
    prior = read(reference/'source-predictions.json')
    if (prior['status'], prior['physical_ids'], prior['views'], prior['target_reads']) != (
            'PREDICTIONS_FROZEN', ids, list(VIEWS), 0):
        raise ValueError('Expected the registered paired source V panel')
    packets = native(source, 'V', contract['counts']['V'])[rows]
    jobs = [(int(row), SOURCE[int(receiver)], iq) for row, receiver, iq in zip(rows, rx, packets, strict=True)]
    predictions, parity = {}, {}
    with (ProcessPoolExecutor(max_workers=2) as pool,
          np.load(reference/'predictions.npz', allow_pickle=False) as archived):
        for scene in VIEWS:
            stream = pool.map(partial(full_components, scene), jobs, chunksize=6)
            results = []
            for i in range(0, len(jobs), BATCH):
                y, s = (np.stack(v) for v in zip(*(next(stream) for _ in jobs[i:i+BATCH]), strict=True))
                results.append(classification(y, s))
                if i % 80 == 0:
                    print('W24_PRACTICAL_DIRECTION_PROGRESS', scene, min(i+BATCH,len(jobs)), flush=True)
            predictions[scene] = np.concatenate(results, axis=1)
            if allocation:
                idx = [prior['modes'].index(k) for k in ('direct', 'frozen_half')]
                if not np.array_equal(predictions[scene][:2], archived[scene][idx]):
                    raise ValueError(f'Frozen direct and half readouts changed in {scene}')
            else:
                indices = [prior['modes'].index(k) for k in ('direct', 'exact_both')]
                parity[scene] = dict(direct=int((predictions[scene][0] != archived[scene][indices[0]]).sum()),
                    exact=None if scene == 'clean' else int((predictions[scene][4] != archived[scene][indices[1]]).sum()))
    with (root/'predictions.npz').open('xb') as f:
        np.savez(f, **predictions)
    fields = (dict(matched_panel=RUN, fixed_center_edit_energy=True, frozen_native_edges=True,
                   common_private_abs_bin_cutoff=32) if allocation else
              dict(matched_panel=REFERENCE, cpu_gpu_decision_disagreements=parity))
    write(root/'source-predictions.json', dict(status='PREDICTIONS_FROZEN', modes=list(modes), views=list(VIEWS),
          physical_ids=ids, target_reads=0, source_truth_reads=0,
          waveform=dict(oracle_only=True, candidate=ARM, frozen_run=PRACTICAL_RUN,
                        device='cpu', regional_edits=True, **fields)))


def dispatch(runtime):
    rt = read(runtime)
    root, reference = (Path(rt[k]) for k in ('run_root', 'w24_release_root'))
    root.mkdir()
    if read(reference/'submit.json')['code_commit'] != W24_COMMIT:
        raise ValueError('Expected the registered W24 evaluator archive')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONPATH=str(reference), OMP_NUM_THREADS='2',
               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    state = root/'state.json'
    for phase in ('predict-source', 'score-source'):
        with (root/f'{phase}.log').open('x', encoding='utf-8') as log:
            proc = subprocess.Popen([rt['python'], '-u', str(Path(__file__).resolve()), phase, str(runtime)],
                                    cwd=reference, env=env, stdout=log, stderr=subprocess.STDOUT)
        write(state, dict(status='RUNNING', phase=phase, pid=proc.pid), exclusive=False)
        if proc.wait():
            write(state, dict(status='FAILED', phase=phase), exclusive=False)
            raise RuntimeError(phase)
    write(state, dict(status='SOURCE_SCORED'), exclusive=False)


def main():
    from experiments.pamf_lora_main.publish import release, release_commit

    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    for phase in ('dispatch', 'predict-source', 'score-source'):
        sub.add_parser(phase).add_argument('runtime', type=Path)
    publish = sub.add_parser('publish')
    publish.add_argument('--component-allocation', action='store_true')
    for key in ('repo', 'output', 'context', 'ssh-config'):
        publish.add_argument('--'+key, type=Path, required=True)
    args = parser.parse_args()
    if args.action == 'publish':
        context = read(args.context)
        rt = {k:context[k] for k in ('project','python','frozen_sources')}
        project = PurePosixPath(rt['project'])
        releases, runs = project/'releases', project/'runs'
        run = ALLOCATION_RUN if args.component_allocation else RUN
        rt.update(run_id=run, runner=__spec__.name, code_commit=release_commit(args.repo),
                  release_root=str(releases/f'{run}-release'), run_root=str(runs/run),
                  source_root=str(releases/f'{DATA_RUN}-release'/'data/source'), classifier_root=str(runs/AUG_RUN),
                  w24_release_root=str(releases/f'{W24_RUN}-release'), candidate_root=str(runs/PRACTICAL_RUN),
                  training_release_root=str(releases/f'{TRAIN_RUN}-release'))
        rt['external_releases'] = [rt[k] for k in ('source_root', 'classifier_root', 'w24_release_root',
                                    'candidate_root', 'training_release_root')] + [str(runs/(RUN if args.component_allocation else REFERENCE))]
        release(args.repo, args.output, rt, args.ssh_config)
    elif args.action == 'dispatch':
        dispatch(args.runtime)
    elif args.action == 'predict-source':
        predict(read(args.runtime))
    else:
        allocation = read(args.runtime)['run_id'] == ALLOCATION_RUN
        score_source(args.runtime, modes=ALLOCATION_MODES if allocation else MODES,
                     contrasts=ALLOCATION_CONTRASTS if allocation else CONTRASTS)


if __name__ == '__main__':
    main()
