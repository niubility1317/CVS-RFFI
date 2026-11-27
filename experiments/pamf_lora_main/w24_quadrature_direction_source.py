"""Score receiver-only I/Q symmetry of a frozen DURA direction on source Full views."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import os
from pathlib import Path, PurePosixPath
import subprocess
from time import perf_counter

import numpy as np
import torch

if __package__ is None:
    from experiments import pamf_lora_main

    pamf_lora_main.__path__.append(str(Path(__file__).resolve().parent))

from experiments.cric_dura_comparison.io import read, tensor, write
from experiments.pamf_lora_main.full_packet_views import native, source_contract
from experiments.pamf_lora_main.runner import activate
from experiments.pamf_lora_main.w24_aperture_noise_audit import center_rms, model_iq, native_aperture
from experiments.pamf_lora_main.w24_edit_budget_audit import full_components
from experiments.pamf_lora_main.w24_factor_audit import AUG_RUN, load_classifier, panel, score_source
from experiments.pamf_lora_main.w24_fixed_dose_direction import (
    BATCH, EVAL_RUN as REFERENCE, RUN as FIT, load_head, readouts as original_readouts, source_parent,
)
from experiments.pamf_lora_main.w24_fixed_dose_direction_model import matched_aperture_edit
from experiments.pamf_lora_main.w24_native_full import DATA_RUN, SOURCE, VIEWS, W24_COMMIT, W24_RUN, classify_aperture
from experiments.pamf_lora_main.w24_quadrature_direction import rotated_proposals
from experiments.pamf_lora_main.w24_regional_dose import regional_power, regional_product
from experiments.pamf_lora_main.w24_regional_dose_source import complex_tensor
from experiments.pamf_lora_main.w24_third_prior import PRACTICAL_RUN
from experiments.pamf_lora_main.w24_wide_transfer import TRAIN_RUN


RUN = '20261127-phase1-lora-w24-quadrature-direction-s2026112991-r01'
MODES = ('direct', 'frozen_half', 'fixed_dose_direction', 'sign_average', 'quadrature_average')
CONTRASTS = ((MODES[3], MODES[2]), (MODES[4], MODES[2]), (MODES[4], MODES[3]))


@torch.no_grad()
def receive_readouts(parent, head, raw):
    wave = complex_tensor(raw)
    frozen = torch.lerp(wave, parent.restore_native(wave), .5)
    observed, half = (native_aperture(np.from_dlpack(x.contiguous())) for x in (wave, frozen))
    base = complex_tensor(half - observed)
    proposals = rotated_proposals(head, wave, frozen-wave)
    edits = tuple(matched_aperture_edit(base, complex_tensor(native_aperture(
        np.from_dlpack((wave + proposal).contiguous())) - observed)) for proposal in proposals)
    variants = (observed, half, *(observed + np.from_dlpack(edit.contiguous()) for edit in edits))
    prepared = tuple(tensor(model_iq(x, center_rms(x)), 'cpu') for x in variants)
    return prepared, observed, edits, base


def measure_geometry(observed, edits, base, signal):
    """Measure the frozen receive-only outputs against isolated source references."""
    want = complex_tensor(native_aperture(signal)-observed)
    anchor = regional_power(base)
    noise = regional_power(want)
    rows = []
    for edit in edits:
        power = regional_power(edit)
        dose_error = (power-anchor).abs() / anchor.clamp_min(1e-24)
        cosine = regional_product(edit, want) / (power.sqrt()*noise.sqrt()).clamp_min(1e-24)
        remaining = torch.where(noise > 1e-20, regional_power(edit-want) / noise.clamp_min(1e-20), 0.)
        drift = regional_power(edit-edits[0]) / anchor.clamp_min(1e-24)
        rows.append(torch.stack((dose_error, cosine, remaining, drift), dim=-1))
    result = torch.stack(rows, dim=1)
    if not torch.isfinite(result).all() or torch.max(result[..., 0]) > 1e-4:
        raise ValueError('Frozen regional energy or source geometry differs')
    return np.from_dlpack(result)


def eligible(scores):
    candidate, reference = (scores['scores'][m] for m in (MODES[-1], MODES[2]))
    paired = scores['paired_contrasts'][f'{MODES[-1]}_vs_{MODES[2]}']
    return (candidate['six_correct'] >= reference['six_correct'] + 18
            and candidate['low_urban_correct'] >= reference['low_urban_correct'] + 6
            and candidate['by_scene']['clean'] >= scores['scores']['direct']['by_scene']['clean']
            and paired['CI95_net_six'][0] > 0.)


def predict_source(rt):
    activate(rt)
    torch.set_num_threads(2)
    root, source, project = (Path(rt[k]) for k in ('run_root', 'source_root', 'project'))
    if (root != project/'runs'/RUN or source != project/'releases'/f'{DATA_RUN}-release'/'data'/'source'
            or any(key.startswith('target') for key in rt)):
        raise ValueError('Expected only the registered source evaluation handles')
    parent = source_parent(rt, 'cpu')
    head, _ = load_head(project/'runs'/FIT)
    contract = source_contract(source)
    f = load_classifier(rt, contract, 'cpu')
    u = native(source, 'U_s', contract['counts']['U_s'])[:1]
    y, s = full_components(VIEWS[-1], (0, SOURCE[0], u[0]), verify=True)
    prepared, observed, edited, base = receive_readouts(parent, head, y[None])
    for got, original in zip(prepared[:3], original_readouts(parent, head, y[None]), strict=True):
        torch.testing.assert_close(got, original, rtol=0, atol=0)
    measure_geometry(observed, edited, base, s[None])
    for readout in prepared:
        if classify_aperture(f, readout).shape != (1,):
            raise ValueError('Real frozen-source smoke failed')
    write(root/'smoke.json', dict(status='PASS', classifier_epoch=104, source_u=1,
                                 target_reads=0, source_truth_reads=0))

    timing = []
    for _ in range(10):
        start = perf_counter()
        receive_readouts(parent, head, y[None])
        timing.append(1000*(perf_counter()-start))
    write(root/'source-b1.json', dict(count=len(timing), threads=2, median_ms=float(np.median(timing)),
                                      p95_ms=float(np.quantile(timing, .95))))

    rows, rx, ids = panel(source, contract)
    prior = project/'runs'/REFERENCE
    marker = read(prior/'source-predictions.json')
    if (marker['status'], marker['physical_ids'], marker['views'], marker['modes'], marker['target_reads']) != (
        'PREDICTIONS_FROZEN', ids, list(VIEWS), list(MODES[:3]), 0):
        raise ValueError('Expected original matched source readouts')
    packets = native(source, 'V', contract['counts']['V'])[rows]
    jobs = [(int(row), SOURCE[int(r)], packet) for row, r, packet in zip(rows, rx, packets, strict=True)]
    output, details = {}, {}
    with (ProcessPoolExecutor(max_workers=2) as pool,
          np.load(prior/'predictions.npz', allow_pickle=False) as archived):
        for scene in VIEWS:
            stream = pool.map(partial(full_components, scene), jobs, chunksize=6)
            predictions, measured = [], []
            for start in range(0, len(jobs), BATCH):
                batch = [next(stream) for _ in jobs[start:start+BATCH]]
                y, s = (np.stack(v) for v in zip(*batch, strict=True))
                readout, observed, edited, base = receive_readouts(parent, head, y)
                predictions.append(np.stack([classify_aperture(f, view) for view in readout]))
                measured.append(measure_geometry(observed, edited, base, s))
            output[scene] = np.concatenate(predictions, axis=1)
            if not np.array_equal(output[scene][:3], archived[scene]):
                raise ValueError(f'Original source decision changed in {scene}')
            details[scene] = np.concatenate(measured)
            print('W24_QUADRATURE_DIRECTION_PREDICTED', scene, len(jobs), flush=True)
    with (root/'predictions.npz').open('xb') as file:
        np.savez(file, **output)
    with (root/'direction-geometry.npz').open('xb') as file:
        np.savez(file, **details)
    write(root/'source-predictions.json', dict(status='PREDICTIONS_FROZEN', physical_ids=ids, views=list(VIEWS),
          modes=list(MODES), target_reads=0, source_truth_reads=0, waveform=dict(parent=FIT, base=PRACTICAL_RUN,
               received_only=True, source_oracle_usage='measurement_only', geometry_modes=list(MODES[2:]),
               geometry_regions=['center', 'edges'],
               geometry_fields=['dose_relative_error', 'cosine', 'remaining_noise_ratio', 'direction_change_ratio'])))


def dispatch(runtime):
    rt = read(runtime)
    root, archive = (Path(rt[k]) for k in ('run_root', 'w24_release_root'))
    root.mkdir()
    state = root/'state.json'
    if read(archive/'submit.json')['code_commit'] != W24_COMMIT:
        raise ValueError('Expected pinned original W24')
    script = Path(__file__).resolve()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONPATH=str(archive),
               OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    try:
        for phase in ('predict-source', 'score-source'):
            with (root/f'{phase}.log').open('x') as log:
                proc = subprocess.Popen([rt['python'], '-u', str(script), phase, str(runtime)],
                                        cwd=archive, env=env, stdout=log, stderr=subprocess.STDOUT)
            write(state, dict(status='RUNNING', phase=phase, pid=proc.pid), exclusive=False)
            if proc.wait():
                raise RuntimeError(f'{phase} failed')
        write(state, dict(status='SOURCE_SCORED', promote=eligible(read(root/'source-scores.json')),
                         target_reads=0), exclusive=False)
    except Exception as exc:
        write(state, dict(status='FAILED', error=str(exc)), exclusive=False)
        raise


def publish(args):
    from experiments.pamf_lora_main.publish import release, release_commit

    ctx = read(args.context)
    rt = {key: ctx[key] for key in ('project', 'python', 'frozen_sources')}
    project = PurePosixPath(rt['project'])
    releases, runs = project/'releases', project/'runs'
    rt.update(run_id=RUN, runner=__spec__.name, code_commit=release_commit(args.repo),
              release_root=str(releases/f'{RUN}-release'), run_root=str(runs/RUN),
              source_root=str(releases/f'{DATA_RUN}-release'/'data'/'source'),
              classifier_root=str(runs/AUG_RUN), w24_release_root=str(releases/f'{W24_RUN}-release'),
              training_release_root=str(releases/f'{TRAIN_RUN}-release'))
    rt['external_releases'] = [rt[k] for k in ('source_root', 'classifier_root', 'w24_release_root',
                                             'training_release_root')]
    rt['external_releases'] += [str(runs/key) for key in (FIT, PRACTICAL_RUN, REFERENCE)]
    release(args.repo, args.output, rt, args.ssh_config)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest='action', required=True)
    for action in ('dispatch', 'predict-source', 'score-source'):
        subs.add_parser(action).add_argument('runtime', type=Path)
    pub = subs.add_parser('publish')
    for name in ('repo', 'output', 'context', 'ssh-config'):
        pub.add_argument('--'+name, required=True, type=Path)
    args = parser.parse_args()
    if args.action == 'publish':
        publish(args)
    elif args.action == 'dispatch':
        dispatch(args.runtime)
    elif args.action == 'predict-source':
        predict_source(read(args.runtime))
    else:
        score_source(args.runtime, modes=MODES, contrasts=CONTRASTS)


if __name__ == '__main__':
    main()
