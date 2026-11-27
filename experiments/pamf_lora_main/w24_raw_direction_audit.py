"""Source-only separation of learned direction and dose for frozen native LoRa recovery."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import multiprocessing
from pathlib import Path, PurePosixPath
import sys

import numpy as np
import torch

if __name__ == '__mp_main__':
    sys.path.insert(0, str(Path.cwd()))

if not __package__:
    from experiments import pamf_lora_main
    pamf_lora_main.__path__.append(str(Path(__file__).resolve().parent))

from experiments.cric_dura_comparison.io import read, write
from experiments.pamf_lora_main import w24_joint_pamf as joint
from experiments.pamf_lora_main import w24_native_training as native
from experiments.pamf_lora_main.w24_aperture_noise_audit import CENTER, native_aperture
from experiments.pamf_lora_main.w24_edit_budget_audit import full_components
from experiments.pamf_lora_main.w24_factor_audit import frozen_source_panel, score_source
from experiments.pamf_lora_main.w24_native_residual_source import dispatch


RUN = '20261127-diagnostic-lora-raw8192-edit-direction-s2026112559-r01'
MODES = ('direct', native.TASK, 'learned_best_gain', 'noise_same_dose', 'noise_exact', 'noise_exact_keep_orthogonal')
CONTRASTS = (
    ('learned_best_gain', native.TASK), ('noise_same_dose', native.TASK),
    ('noise_exact', 'noise_same_dose'), ('noise_exact_keep_orthogonal', 'noise_exact'),
    ('noise_exact_keep_orthogonal', native.TASK),
)


def edits(observed, recovered, signal):
    """Match the actual linear W24 center norm before its input RMS normalization."""
    received = observed.astype(np.complex128)
    exact = signal.astype(np.complex128)
    learned = recovered - received
    noise_direction = exact - received

    def center(x):
        return native_aperture(x)[..., :CENTER]

    def rms(x):
        return np.sqrt(np.mean(np.abs(x)**2, axis=(1, 2)))

    def quotient(a, b):
        return np.divide(a, b, out=np.zeros_like(a), where=b > 1e-12)

    received_center, learned_center, direction_center = map(center, (received, learned, noise_direction))
    dose, noise, observed_rms = map(rms, (learned_center, direction_center, received_center))
    dot = np.mean((learned_center.conj() * direction_center).real, axis=(1, 2))
    same_scale = quotient(dose, noise)
    best_gain = np.maximum(0., quotient(dot, dose**2))
    parallel = quotient(dot, noise**2)
    repaired = (
        received + best_gain[:, None]*learned,
        received + same_scale[:, None]*noise_direction,
        exact,
        exact + learned - parallel[:, None]*noise_direction,
    )
    valid = noise > 1e-12
    repaired = tuple(np.where(valid[:, None], x, received).astype(np.complex64) for x in repaired)
    matching_error = np.abs(rms(center(repaired[1] - received))-dose)
    if np.any(matching_error[valid] > 2e-6*np.maximum(observed_rms[valid], 1.)):
        raise ValueError('Same-dose oracle did not match the measured W24 center edit')
    metrics = dict(learned_dose=quotient(dose, observed_rms), exact_dose=quotient(noise, observed_rms),
                   same_dose_exact_fraction=same_scale, direction_cosine=quotient(dot, dose*noise),
                   learned_best_nonnegative_gain=best_gain, actual_parallel_fraction=parallel,
                   learned_remaining=quotient(rms(direction_center-learned_center),noise),
                   gain_only_remaining=quotient(rms(direction_center-learned_center*best_gain[:, None, None]),noise),
                   dose_match_error=matching_error)
    if any(not np.isfinite(v).all() for v in (*metrics.values(), *repaired)):
        raise ValueError('Nonfinite source direction diagnostic')
    return repaired, metrics


@torch.no_grad()
def render(reader, restorer, observed, noiseless, device):
    recovered = native.restored_raw(restorer, observed, native.TASK, device)
    alternatives, metrics = edits(observed, recovered, noiseless)
    decisions = [joint.classify_aperture(reader, joint.received_aperture(x, device))
                 for x in (observed, recovered, *alternatives)]
    return np.stack(decisions), metrics


def summarize(batches):
    result = {}
    for name in batches[0]:
        values = np.concatenate([part[name] for part in batches])
        result[name] = dict(p10_p50_p90=np.quantile(values, [.1, .5, .9]).tolist(),
                            maximum=float(values.max()))
    fraction = np.concatenate([part['same_dose_exact_fraction'] for part in batches])
    result['same_dose_over_exact_count'] = int((fraction > 1).sum())
    return result


def predict(rt):
    root, source, device = Path(rt['run_root']), Path(rt['source_root']), 'cuda:0'
    parent = root.parent / native.TASK_RUN
    boot = dict(rt, run_id=native.TASK_RUN, run_root=str(parent))
    classifier, restorer, _, contract = native.load_readers(joint, boot, device)
    u = joint.native(source, 'U_s', contract['counts']['U_s'])[0]
    for view in ('clean', joint.VIEWS[-1]):
        received, signal = full_components(view, (0, joint.SOURCE[0], u), verify=True)
        decisions, metrics = render(classifier, restorer, received[None], signal[None], device)
        if decisions.shape != (len(MODES), 1) or any(x.shape != (1,) for x in metrics.values()):
            raise ValueError('Real source checkpoint smoke produced malformed diagnostics')
        if view == 'clean' and not np.all(decisions[2:] == decisions[0]):
            raise ValueError('Oracle changed clean source IQ')
    write(root/'smoke.json', dict(status='PASS', source_u_count=1, native_parent=native.TASK_RUN,
                                 target_reads=0, source_truth_reads=0))

    rows, rx, ids = joint.panel(source, contract)
    raw = joint.native(source, 'V', contract['counts']['V'])[rows]
    jobs = [(int(i), joint.SOURCE[int(r)], x) for i, r, x in zip(rows, rx, raw, strict=True)]
    output, metrics = {}, {}
    with ProcessPoolExecutor(max_workers=rt['cpu_workers'], mp_context=multiprocessing.get_context('spawn')) as pool:
        for view in joint.VIEWS:
            traced = pool.map(partial(full_components, view), jobs, chunksize=8)
            guesses, measurements = [], []
            for offset in range(0, len(jobs), joint.PRED_BATCH):
                chunk = [next(traced) for _ in range(min(joint.PRED_BATCH, len(jobs)-offset))]
                observed, signal = (np.stack(values) for values in zip(*chunk, strict=True))
                decisions, statistics = render(classifier, restorer, observed, signal, device)
                guesses.append(decisions)
                measurements.append(statistics)
            output[view] = np.concatenate(guesses, axis=1)
            metrics[view] = summarize(measurements)
            print('NATIVE_DIRECTION_PREDICTED', view, len(jobs), flush=True)

    reference = parent/joint.FULL
    stamp = read(reference/'source-predictions.json')
    if (stamp['status']!='PREDICTIONS_FROZEN' or stamp['physical_ids']!=ids
            or stamp['views']!=list(joint.VIEWS) or stamp['target_reads'] or stamp['source_truth_reads']):
        raise ValueError('Native frozen reference differs from the registered source panel')
    indices = [stamp['modes'].index(name) for name in MODES[:2]]
    with np.load(reference/'predictions.npz', allow_pickle=False) as prior:
        if any(not np.array_equal(prior[v][indices], output[v][:2]) for v in joint.VIEWS):
            raise ValueError('Direct or frozen native decisions changed on the matched source input')
    out = root/joint.FULL
    out.mkdir()
    np.savez(out/'predictions.npz', **output)
    write(out/'source-predictions.json', dict(
        status='PREDICTIONS_FROZEN', physical_ids=ids, views=list(joint.VIEWS), modes=list(MODES),
        source_truth_reads=0, target_reads=0, waveform=dict(diagnostics_only=True, learned_parent=native.TASK_RUN,
        route=joint.FULL, direct_and_parent_parity=True, dose_by_scene=metrics),
    ))


def main():
    from experiments.pamf_lora_main.publish import release, release_commit
    from experiments.pamf_lora_main.w24_wide_transfer import TRAIN_RUN

    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    for name in ('dispatch', 'predict-source', 'score-source'):
        sub.add_parser(name).add_argument('runtime', type=Path)
    publish = sub.add_parser('publish')
    for field in ('repo', 'context', 'ssh-config', 'output'):
        publish.add_argument('--'+field, required=True, type=Path)
    args = parser.parse_args()
    if args.action == 'dispatch':
        dispatch(args.runtime, entry=__file__)
    elif args.action == 'predict-source':
        predict(read(args.runtime))
    elif args.action == 'score-source':
        rt = read(args.runtime)
        out = Path(rt['run_root']) / joint.FULL
        frozen_source_panel(Path(rt['source_root']), out, MODES)
        score_source(args.runtime, modes=MODES, contrasts=CONTRASTS, output=out)
    else:
        ctx = read(args.context)
        rt = {key: ctx[key] for key in ('project', 'python', 'frozen_sources')}
        project = PurePosixPath(rt['project'])
        runs, releases = project/'runs', project/'releases'
        rt.update(run_id=RUN, runner=__spec__.name, code_commit=release_commit(args.repo),
                  release_root=str(releases/f'{RUN}-release'), run_root=str(runs/RUN),
                  source_root=str(releases/f'{joint.DATA_RUN}-release'/'data/source'),
                  w24_release_root=str(releases/f'{joint.W24_RUN}-release'),
                  training_release_root=str(releases/f'{TRAIN_RUN}-release'),
                  classifier_root=str(runs/joint.AUG_RUN), unet_parent=str(runs/joint.UNET_FIDELITY_RUN),
                  allowed_gpus=[7, 2, 0, 6], cpu_workers=6)
        rt['external_releases'] = [rt[key] for key in (
            'source_root', 'w24_release_root', 'training_release_root', 'classifier_root', 'unet_parent'
        )] + [str(runs/native.TASK_RUN)]
        release(args.repo, args.output, rt, args.ssh_config)


if __name__ == '__main__':
    main()
