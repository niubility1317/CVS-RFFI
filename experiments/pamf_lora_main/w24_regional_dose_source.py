"""Fit receiver-only LoRa regional gains, then score fixed source ablations."""

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
from experiments.pamf_lora_main.channel_blind_source import partition
from experiments.pamf_lora_main.full_packet_views import native, source_contract, source_role
from experiments.pamf_lora_main.runner import activate, available
from experiments.pamf_lora_main.w24_aperture_noise_audit import center_rms, model_iq, native_aperture
from experiments.pamf_lora_main.w24_edit_budget_audit import full_components
from experiments.pamf_lora_main.w24_factor_audit import AUG_RUN, load_classifier, panel, score_source
from experiments.pamf_lora_main.w24_fixed_dose_direction import (
    ARM, BATCH, EVAL_RUN as REFERENCE, FIT_PER_RX, ISOLATION, RUN as DIRECTION_RUN,
    aperture_components, dtype_policy, load_head, source_parent,
)
from experiments.pamf_lora_main.w24_fixed_dose_direction_model import ApertureDirection
from experiments.pamf_lora_main.w24_native_full import DATA_RUN, SOURCE, VIEWS, W24_COMMIT, W24_RUN, classify_aperture
from experiments.pamf_lora_main.w24_practical_supervision import NAMESPACE, SEED as TRACE_SEED
from experiments.pamf_lora_main.w24_regional_dose import (
    apply_gain, fit_gain, gain_features, optimal_gain, predict_gain, regional_power,
)
from experiments.pamf_lora_main.w24_third_prior import PRACTICAL_CACHE_RUN, PRACTICAL_RUN
from experiments.pamf_lora_main.w24_wide_transfer import TRAIN_RUN, received_aperture


RUN = '20261127-phase1-lora-w24-regional-dose-s2026112991-r01'
CAL_PER_RX = 800
MODES = ('direct', 'frozen_half', 'fixed_dose_direction', 'constant_gain', 'adaptive_gain',
         'optimal_amplitude_on_learned_direction')
CONTRASTS = tuple((m, 'fixed_dose_direction') for m in MODES[3:]) + (('adaptive_gain', 'constant_gain'),)
FIT_FIELDS = ('project', 'run_root', 'source_root', 'practical_cache', 'training_release_root')


def fit_config(rt):
    project = Path(rt['project'])
    if (set(rt) != set(FIT_FIELDS) or Path(rt['run_root']) != project/'runs'/RUN
            or Path(rt['practical_cache']) != project/'runs'/PRACTICAL_CACHE_RUN/'source_practical'
            or Path(rt['source_root']) != project/'releases'/f'{DATA_RUN}-release'/'data'/'source'
            or Path(rt['training_release_root']) != project/'releases'/f'{TRAIN_RUN}-release'):
        raise ValueError('Fit accepts only the registered source bank and checkpoint paths')
    return project


def source_arrays(rt):
    root, source = Path(rt['practical_cache']), Path(rt['source_root'])
    contract = source_contract(source)
    iq, rx, _, _ = source_role(source, contract, 'U_s')
    proof = read(root/'complete.json')
    if (proof['status'] != 'SOURCE_TRACE_FROZEN' or proof['data_role'] != 'U_s'
            or proof['namespace'] != NAMESPACE or proof['seed'] != TRACE_SEED
            or proof['count'] != len(iq) or proof['source_root'] != str(source)
            or any(proof[k] for k in ISOLATION)):
        raise ValueError('Expected registered source U only')
    pairs = np.load(root/'pairs.npy', allow_pickle=False, mmap_mode='r')
    if pairs.shape != (len(iq), 2, 8192) or pairs.dtype != np.complex64:
        raise ValueError('Expected the original native Practical bank')
    parent_train, _, held = partition(rx, fit=FIT_PER_RX)
    selected = parent_train.reshape(len(SOURCE), FIT_PER_RX)[:, :CAL_PER_RX].reshape(-1)
    return iq, pairs, selected, held


def complex_tensor(value, device='cpu'):
    iq = np.ascontiguousarray(value, dtype=np.complex64).view(np.float32)
    return tensor(iq, device).view(torch.complex64)


@torch.no_grad()
def fit_source(rt):
    project = fit_config(rt)
    root = Path(rt['run_root'])
    dtype_policy()
    torch.set_num_threads(2)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    parent = source_parent(rt, device)
    head, _ = load_head(project/'runs'/DIRECTION_RUN)
    head, projection = head.to(device), ApertureDirection().to(device)
    iq, pairs, fit, held = source_arrays(rt)
    started = perf_counter()

    def values(y, s):
        base = torch.lerp(y, parent.restore_native(y), .5) - y
        _, edit = projection.edits(base, head(y, base))
        observed = projection.project_regions(y)
        wanted = projection.project_regions(s - y)
        return gain_features(y, observed, edit), edit, wanted, observed

    y, s = complex_tensor(pairs[fit[:1]], device).unbind(1)
    if not all(torch.isfinite(value).all() for value in values(y, s)):
        raise ValueError('Real frozen source U amplitude smoke failed')
    write(root/'fit-smoke.json', dict(status='PASS', source_role='U_s', device=device, **ISOLATION))

    result = []
    for start in range(0, len(fit), BATCH):
        rows = fit[start:start+BATCH]
        clean = complex_tensor(iq[rows], device)
        y, s = complex_tensor(pairs[rows], device).unbind(1)
        feature, edit, wanted, observed = values(torch.cat((y, clean)), torch.cat((s, clean)))
        weight = regional_power(edit) / (regional_power(wanted) + .002 * regional_power(observed)).clamp_min(1e-24)
        result.append((feature.cpu(), optimal_gain(edit, wanted).cpu(), weight.clamp_max(4.).cpu()))
        if start % 512 == 0:
            print('W24_REGIONAL_DOSE_FEATURES', start + len(rows), flush=True)
    parameters = fit_gain(*(torch.cat(items) for items in zip(*result, strict=True)))
    gain_gpu = {k: v.to(device) for k, v in parameters.items()}

    errors, diagnostic_gains = [], []
    for start in range(0, len(held), BATCH):
        y, s = complex_tensor(pairs[held[start:start+BATCH]], device).unbind(1)
        feature, edit, wanted, _ = values(y, s)
        gains = (torch.ones_like(feature[..., 0]), predict_gain(gain_gpu, feature, adaptive=False),
                 predict_gain(gain_gpu, feature), optimal_gain(edit, wanted))
        power = regional_power(wanted).clamp_min(1e-24)
        errors.append(torch.stack([regional_power(apply_gain(edit, g)-wanted) / power for g in gains]).cpu())
        diagnostic_gains.append(gains[2].cpu())
    error = torch.cat(errors, dim=1)
    score = dict(n=len(held), modes=list(MODES[2:]), median_residual=error.median(1).values.tolist(),
                 mean_residual=error.mean(1).tolist(),
                 median_adaptive_gain=torch.cat(diagnostic_gains).median(0).values.tolist())
    proof = dict(status='SOURCE_FROZEN', parent=DIRECTION_RUN, base=PRACTICAL_RUN, base_arm=ARM,
                 source_fit=len(fit), source_clean=len(fit), source_held=len(held),
                 seconds=perf_counter()-started, device=device,
                 peak_cuda=torch.cuda.max_memory_allocated() if device == 'cuda:0' else 0,
                 held_waveform=score, **ISOLATION)
    if not all(torch.isfinite(value).all() for value in (*parameters.values(), error)):
        raise ValueError('Nonfinite source amplitude calibration')
    with (root/'gain.pt').open('xb') as file:
        torch.save(parameters, file)
    write(root/'fit-complete.json', proof)


def load_gain(root):
    proof = read(root/'fit-complete.json')
    if (proof['status'] != 'SOURCE_FROZEN' or proof['parent'] != DIRECTION_RUN
            or proof['base'] != PRACTICAL_RUN or proof['source_fit'] != CAL_PER_RX*len(SOURCE)
            or proof['source_held'] != 500 or any(proof[k] for k in ISOLATION)):
        raise ValueError('Expected an isolated frozen source amplitude model')
    return torch.load(root/'gain.pt', map_location='cpu', weights_only=True)


@torch.no_grad()
def render(f, parent, head, gain, y, s):
    wave, observed, half, edit = aperture_components(parent, head, y)
    feature = gain_features(wave, complex_tensor(observed), edit)
    constant = predict_gain(gain, feature, adaptive=False)
    adaptive = predict_gain(gain, feature)
    # Source diagnostic only. Neither deployable gain receives this signal.
    wanted = complex_tensor(native_aperture(s) - observed)
    oracle = torch.where(regional_power(wanted) > 1e-16, optimal_gain(edit, wanted), 1.)
    variants = [observed, half, observed + np.from_dlpack(edit)]
    variants.extend(observed + np.from_dlpack(apply_gain(edit, g)) for g in (constant, adaptive, oracle))
    return np.stack([classify_aperture(f, tensor(model_iq(value, center_rms(value)), 'cpu'))
                     for value in variants])


def predict_source(rt):
    activate(rt)
    torch.set_num_threads(2)
    root, source, project = (Path(rt[k]) for k in ('run_root', 'source_root', 'project'))
    gain = load_gain(root)
    parent = source_parent(rt, 'cpu')
    head, _ = load_head(project/'runs'/DIRECTION_RUN)
    contract = source_contract(source)
    f = load_classifier(rt, contract, 'cpu')
    u = native(source, 'U_s', contract['counts']['U_s'])[0]
    y, s = full_components(VIEWS[-1], (0, SOURCE[0], u), verify=True)
    real = render(f, parent, head, gain, y[None], s[None])
    direct = classify_aperture(f, received_aperture(y[None], 'cpu'))
    if real.shape != (len(MODES), 1) or not np.array_equal(real[0], direct):
        raise ValueError('Real weight and source U prediction smoke failed')
    write(root/'smoke.json', dict(status='PASS', classifier_epoch=104, source_u=1,
                                 target_reads=0, source_truth_reads=0))

    rows, rx, ids = panel(source, contract)
    prior = project/'runs'/REFERENCE
    marker = read(prior/'source-predictions.json')
    if (marker['status'], marker['physical_ids'], marker['views'], marker['modes'], marker['target_reads']) != (
        'PREDICTIONS_FROZEN', ids, list(VIEWS), list(MODES[:3]), 0):
        raise ValueError('Expected matched source predictions from the frozen direction evaluation')
    packets = native(source, 'V', contract['counts']['V'])[rows]
    jobs = [(int(row), SOURCE[int(r)], packet) for row, r, packet in zip(rows, rx, packets, strict=True)]
    output = {}
    with (ProcessPoolExecutor(max_workers=2) as pool,
          np.load(prior/'predictions.npz', allow_pickle=False) as archived):
        for scene in VIEWS:
            stream = pool.map(partial(full_components, scene), jobs, chunksize=6)
            results = []
            for start in range(0, len(jobs), BATCH):
                batch = [next(stream) for _ in jobs[start:start+BATCH]]
                y, s = (np.stack(v) for v in zip(*batch, strict=True))
                results.append(render(f, parent, head, gain, y, s))
            output[scene] = np.concatenate(results, axis=1)
            if not np.array_equal(output[scene][:3], archived[scene]):
                raise ValueError(f'Fixed source comparator predictions changed for {scene}')
            print('W24_REGIONAL_DOSE_PREDICTED', scene, len(jobs), flush=True)
    with (root/'predictions.npz').open('xb') as file:
        np.savez(file, **output)
    write(root/'source-predictions.json', dict(status='PREDICTIONS_FROZEN', physical_ids=ids,
          views=list(VIEWS), modes=list(MODES), target_reads=0, source_truth_reads=0,
          waveform=dict(direction_parent=DIRECTION_RUN, base=PRACTICAL_RUN, source_gain=RUN,
                        oracle_only_mode=MODES[-1], device='cpu')))


def eligible(scores):
    ref, fixed, new = (scores['scores'][m] for m in MODES[2:5])
    return (new['six_correct'] >= ref['six_correct']+18 and new['six_correct'] > fixed['six_correct']
            and new['low_urban_correct'] >= ref['low_urban_correct']+6
            and new['by_scene']['clean'] >= scores['scores']['direct']['by_scene']['clean']
            and scores['paired_contrasts']['adaptive_gain_vs_fixed_dose_direction']['CI95_net_six'][0] > 0.)


def dispatch(runtime):
    rt = read(runtime)
    root, archive = (Path(rt[k]) for k in ('run_root', 'w24_release_root'))
    root.mkdir()
    state = root/'state.json'
    if read(archive/'submit.json')['code_commit'] != W24_COMMIT:
        raise ValueError('Expected the frozen original W24 archive')
    script = Path(__file__).resolve()

    def stage(phase, config, cwd, gpu=None):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='' if gpu is None else str(gpu), PYTHONPATH=str(cwd),
                   OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
        with (root/f'{phase}.log').open('x') as log:
            job = subprocess.Popen([rt['python'], '-u', str(script), phase, str(config)], cwd=cwd, env=env,
                                   stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        write(state, dict(status='RUNNING', phase=phase, pid=job.pid, gpu=gpu), exclusive=False)
        if job.wait():
            raise RuntimeError(f'{phase} failed')

    try:
        path = root/'fit-inputs.json'
        write(path, {k: rt[k] for k in FIT_FIELDS})
        cards = available(rt['allowed_gpus'], 2, training_cap=True)
        selected = next((card for card in rt['allowed_gpus'] if card in cards), None)
        stage('fit-source', path, script.parents[2], selected)
        stage('predict-source', runtime, archive)
        stage('score-source', runtime, archive)
        write(state, dict(status='SOURCE_SCORED', target_reads=0,
                         promote=eligible(read(root/'source-scores.json'))), exclusive=False)
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
              training_release_root=str(releases/f'{TRAIN_RUN}-release'),
              practical_cache=str(runs/PRACTICAL_CACHE_RUN/'source_practical'), allowed_gpus=[7, 0, 3, 6, 5])
    rt['external_releases'] = [rt[k] for k in ('source_root', 'classifier_root', 'w24_release_root',
                                             'training_release_root', 'practical_cache')]
    rt['external_releases'] += [str(runs/key) for key in (DIRECTION_RUN, PRACTICAL_RUN, REFERENCE)]
    release(args.repo, args.output, rt, args.ssh_config)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest='action', required=True)
    for action in ('dispatch', 'fit-source', 'predict-source', 'score-source'):
        subs.add_parser(action).add_argument('runtime', type=Path)
    release_args = subs.add_parser('publish')
    for name in ('repo', 'output', 'context', 'ssh-config'):
        release_args.add_argument('--'+name, required=True, type=Path)
    args = parser.parse_args()
    if args.action == 'publish':
        publish(args)
    elif args.action == 'dispatch':
        dispatch(args.runtime)
    elif args.action == 'fit-source':
        fit_source(read(args.runtime))
    elif args.action == 'predict-source':
        predict_source(read(args.runtime))
    else:
        score_source(args.runtime, modes=MODES, contrasts=CONTRASTS)


if __name__ == '__main__':
    main()
