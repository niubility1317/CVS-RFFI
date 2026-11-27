"""Train and independently score native peer spectral correction on legal LoRa source IQ."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import os
from pathlib import Path, PurePosixPath
import subprocess
from time import perf_counter, sleep

import numpy as np
import torch

if __package__ is None:
    from experiments import pamf_lora_main

    pamf_lora_main.__path__.append(str(Path(__file__).resolve().parent))

from experiments.cric_dura_comparison.io import read, tensor, write
from experiments.pamf_lora_main.full_packet_views import native, source_contract
from experiments.pamf_lora_main.joint_unroll_source import coordinates
from experiments.pamf_lora_main.runner import activate, available
from experiments.pamf_lora_main.w24_aperture_noise_audit import CENTER, center_rms, model_iq, native_aperture
from experiments.pamf_lora_main.w24_component_directions import component_angle, matched_component_edit
from experiments.pamf_lora_main.w24_factor_audit import AUG_RUN, load_classifier, panel, score_source
from experiments.pamf_lora_main.w24_fixed_dose_direction import (
    BATCH, EVAL_RUN as REFERENCE, FIT_FIELDS, FIT_PER_RX, ISOLATION, RUN as PRIOR, SEED,
    dtype_policy, load_head, readouts as prior_readouts, source_observation, source_parent_and_tracker,
)
from experiments.pamf_lora_main.w24_fixed_dose_direction_model import (
    ApertureDirection, direction_objective, matched_aperture_edit,
)
from experiments.pamf_lora_main.w24_native_full import DATA_RUN, SOURCE, VIEWS, W24_COMMIT, W24_RUN, classify_aperture
from experiments.pamf_lora_main.w24_peer_spectral_direction import PeerSpectralDirection
from experiments.pamf_lora_main.w24_post_peer_refinement_source import (
    CONTRASTS as REFINEMENT_CONTRASTS, MODES as REFINEMENT_MODES, RUN as REFINEMENT_RUN,
)
from experiments.pamf_lora_main.w24_practical_supervision import training_sets
from experiments.pamf_lora_main.w24_regional_dose_source import complex_tensor
from experiments.pamf_lora_main.w24_third_prior import PRACTICAL_CACHE_RUN, PRACTICAL_RUN
from experiments.pamf_lora_main.w24_wide_transfer import TRAIN_RUN


RUN = '20261127-phase1-lora-w24-peer-spectral-direction-s2026112991-r01'
COMPONENT_RUN = '20261128-phase1-lora-w24-component-directions-s2026112991-r01'
STEPS = 48000
MODES = ('direct', 'frozen_half', 'fixed_dose_direction', 'within_chirp', 'cross_chirp')
CONTRASTS = ((MODES[3], MODES[2]), (MODES[4], MODES[2]), (MODES[4], MODES[3]))
COMPONENT_MODES = ('direct', 'frozen_half', 'peer_grouped', 'trained_grouped')
COMPONENT_CONTRASTS = (('peer_grouped', 'frozen_half'), ('trained_grouped', 'frozen_half'),
                       ('trained_grouped', 'peer_grouped'))
COMPONENT_WEIGHT = .50
COMPONENTS = ('common_near', 'common_far', 'private_near', 'private_far')
BRANCH_WEIGHTS = (.35, .65)


def fit_config(rt):
    project = Path(rt['project'])
    roots = {project/'runs'/name for name in (RUN, COMPONENT_RUN, REFINEMENT_RUN)}
    if (set(rt) != set(FIT_FIELDS) or Path(rt['run_root']) not in roots
            or Path(rt['source_root']) != project/'releases'/f'{DATA_RUN}-release'/'data'/'source'
            or Path(rt['practical_cache']) != project/'runs'/PRACTICAL_CACHE_RUN/'source_practical'
            or Path(rt['training_release_root']) != project/'releases'/f'{TRAIN_RUN}-release'):
        raise ValueError('Fit must receive registered source-only handles')


def fit_source(rt):
    fit_config(rt)
    root, device = Path(rt['run_root']), 'cuda:0'
    dtype_policy()
    grouped = root.name == COMPONENT_RUN
    parent, tracker = source_parent_and_tracker(rt, device)
    torch.manual_seed(SEED)
    head = PeerSpectralDirection().to(device)
    aperture = ApertureDirection().to(device)
    optim = torch.optim.AdamW(head.parameters(), lr=2e-4, weight_decay=1e-4)
    lr = torch.optim.lr_scheduler.CosineAnnealingLR(optim, STEPS, eta_min=2e-5)

    def evaluate(batch, *, clean_identity=False):
        clean, y, s = batch.unbind(1)
        mask = None
        if clean_identity:
            y, s = (torch.cat((w[:-2], clean[-2:])) for w in (y, s))
            mask = torch.arange(len(y), device=device) >= len(y)-2
        with torch.no_grad():
            base = torch.lerp(y, parent.restore_native(y), .5) - y
        within, cross = head(y, base)
        parts = [direction_objective(aperture, y, base, estimate, s, clean_mask=mask)
                 for estimate in (within, cross)]
        loss = sum(w*row[0] for w, row in zip(BRANCH_WEIGHTS, parts, strict=True))
        if grouped:
            with torch.no_grad():
                _, phase, _, _ = coordinates(tracker, aperture.normalize(aperture.project(y)))
                true_center = aperture.project(s-y)
            angle, cosines = component_angle(aperture.project(cross), true_center, phase)
            loss = loss + COMPONENT_WEIGHT * angle
            parts[1][1].update(dict(zip(COMPONENTS, cosines, strict=True)))
        return loss, tuple(row[1] for row in parts)

    (smoke, _, _), _ = training_sets(rt, device, smoke=True, fit_per_rx=FIT_PER_RX, bank_per_rx=0)
    first, _ = evaluate(smoke[:BATCH], clean_identity=True)
    if not torch.isfinite(first):
        raise ValueError('Real source U startup loss is nonfinite')
    first.backward()
    torch.nn.utils.clip_grad_norm_(head.parameters(), 1., error_if_nonfinite=True)
    optim.zero_grad(set_to_none=True)
    write(root/'fit-smoke.json', dict(status='PASS', source_role='U_s', model='peer_spectral', **ISOLATION))
    del first, smoke

    (train, _, held), _ = training_sets(rt, device, smoke=False, fit_per_rx=FIT_PER_RX, bank_per_rx=0)
    if (len(train), len(held)) != (20000, 500):
        raise ValueError('Unexpected source fit and held physical counts')
    rng = np.random.default_rng(SEED+1)
    started = perf_counter()
    trajectory = {}
    for step in range(1, STEPS+1):
        indices = tensor(rng.choice(len(train), BATCH, replace=False), device)
        optim.zero_grad(set_to_none=True)
        loss, phases = evaluate(train[indices], clean_identity=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1., error_if_nonfinite=True)
        optim.step()
        lr.step()
        if step == 1 or step % 200 == 0:
            print('W24_PEER_SPECTRAL_FIT', step, float(loss.detach()),
                  [{key: float(value) for key, value in row.items()} for row in phases], flush=True)
        if step in (2000, 6000, 12000, 24000, 36000, STEPS):
            head.eval()
            with torch.no_grad():
                chunks = [(len(group), evaluate(group)[1]) for group in held.split(BATCH)]
            trajectory[str(step)] = [
                {k: sum(n*float(m[i][k]) for n, m in chunks)/len(held) for k in chunks[0][1][i]}
                for i in range(len(BRANCH_WEIGHTS))]
            head.train()
            print('W24_PEER_SPECTRAL_HELD', step, trajectory[str(step)], flush=True)
    metadata = dict(status='SOURCE_FROZEN', base=PRACTICAL_RUN, steps=STEPS, model_seed=SEED, batch=BATCH,
                    branch_weights=list(BRANCH_WEIGHTS), parameters=sum(p.numel() for p in head.parameters()),
                    fit=len(train), held=len(held), seconds=perf_counter()-started,
                    peak_cuda=torch.cuda.max_memory_allocated(), held_trajectory=trajectory, **ISOLATION)
    if grouped:
        metadata.update(component_angle_weight=COMPONENT_WEIGHT, component_groups=list(COMPONENTS))
    with (root/'direction.pt').open('xb') as out:
        torch.save(dict(weights=head.state_dict(), metadata=metadata), out)
    write(root/'fit-complete.json', metadata)


def load_trained(root):
    metadata = read(root/'fit-complete.json')
    state = torch.load(root/'direction.pt', map_location='cpu', weights_only=True)
    if (state['metadata'] != metadata or metadata['status'] != 'SOURCE_FROZEN'
            or metadata['steps'] != STEPS or metadata['base'] != PRACTICAL_RUN
            or metadata['fit'] != 20000 or metadata['held'] != 500
            or metadata.get('component_angle_weight') != (COMPONENT_WEIGHT if root.name == COMPONENT_RUN else None)
            or metadata.get('component_groups') != (list(COMPONENTS) if root.name == COMPONENT_RUN else None)
            or any(metadata[k] for k in ISOLATION)):
        raise ValueError('Expected the terminal source-only checkpoint')
    head = PeerSpectralDirection().eval().requires_grad_(False)
    head.load_state_dict(state['weights'], strict=True)
    return head


@torch.no_grad()
def receive_readouts(parent, old, new, raw):
    wave = complex_tensor(raw)
    frozen = torch.lerp(wave, parent.restore_native(wave), .5)
    observed, half = (native_aperture(np.from_dlpack(z.contiguous())) for z in (wave, frozen))
    base = complex_tensor(half-observed)
    proposal = (old(wave, frozen-wave), *new(wave, frozen-wave))
    edits = [matched_aperture_edit(base, complex_tensor(native_aperture(
        np.from_dlpack((wave+z).contiguous()))-observed)) for z in proposal]
    variants = (observed, half, *(observed+np.from_dlpack(delta.contiguous()) for delta in edits))
    return tuple(tensor(model_iq(value, center_rms(value)), 'cpu') for value in variants)


@torch.no_grad()
def receive_grouped_readouts(parent, tracker, peer, new, raw):
    wave = complex_tensor(raw)
    frozen = torch.lerp(wave, parent.restore_native(wave), .5)
    observed, half = (native_aperture(np.from_dlpack(z.contiguous())) for z in (wave, frozen))
    direct = tensor(model_iq(observed, center_rms(observed)), 'cpu')
    _, phase, _, _ = coordinates(tracker, direct[..., :CENTER])
    observed_center = observed[..., :CENTER]
    base = complex_tensor(half[..., :CENTER] - observed_center)
    views = [observed, half]
    for head in (peer, new):
        cross = head(wave, frozen-wave)[1]
        proposed_center = native_aperture(np.from_dlpack((wave+cross).contiguous()))[..., :CENTER]
        edit = matched_component_edit(base, complex_tensor(proposed_center-observed_center), phase)
        adjusted = half.copy()
        adjusted[..., :CENTER] = observed_center + np.from_dlpack(edit.contiguous())
        views.append(adjusted)
    return tuple(tensor(model_iq(value, center_rms(value)), 'cpu') for value in views)


def predict_source(rt):
    activate(rt)
    torch.set_num_threads(2)
    root, source, project = (Path(rt[k]) for k in ('run_root', 'source_root', 'project'))
    grouped = root.name == COMPONENT_RUN
    refined = root.name == REFINEMENT_RUN
    old, _ = load_head(project/'runs'/PRIOR)
    if refined:
        from experiments.pamf_lora_main.w24_post_peer_refinement_source import frozen_refiner_inputs, load_arms, receive

        inputs = frozen_refiner_inputs(rt, 'cpu')
        parent = inputs[1]
        render = partial(receive, inputs, load_arms(root))
        modes, reference, reference_modes, controls = REFINEMENT_MODES, COMPONENT_RUN, COMPONENT_MODES, 2
    else:
        new = load_trained(root)
        parent, tracker = source_parent_and_tracker(rt, 'cpu')
        if grouped:
            peer = load_trained(project/'runs'/RUN)
            render = partial(receive_grouped_readouts, parent, tracker, peer, new)
            modes, reference, reference_modes, controls = COMPONENT_MODES, RUN, MODES, 2
        else:
            render = partial(receive_readouts, parent, old, new)
            modes, reference, reference_modes, controls = MODES, REFERENCE, MODES[:3], 3
    contract = source_contract(source)
    f = load_classifier(rt, contract, 'cpu')
    u = native(source, 'U_s', contract['counts']['U_s'])[0]
    y = source_observation(VIEWS[-1], (0, SOURCE[0], u))[None]
    got = render(y)
    for actual, expected in zip(got[:controls], prior_readouts(parent, old, y)[:controls], strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    if any(classify_aperture(f, view).shape != (1,) for view in got):
        raise ValueError('Real weights and source U evaluation smoke failed')
    write(root/'smoke.json', dict(status='PASS', source_role='U_s', classifier_epoch=104,
                                 target_reads=0, source_truth_reads=0))
    elapsed = []
    for _ in range(10):
        start = perf_counter()
        render(y)
        elapsed.append(1000*(perf_counter()-start))
    write(root/'source-b1.json', dict(n=len(elapsed), threads=2, median_ms=float(np.median(elapsed)),
                                      p95_ms=float(np.quantile(elapsed, .95))))

    rows, rx, ids = panel(source, contract)
    ref = project/'runs'/reference
    marker = read(ref/'source-predictions.json')
    if (marker['status'], marker['physical_ids'], marker['views'], marker['modes'], marker['target_reads']) != (
        'PREDICTIONS_FROZEN', ids, list(VIEWS), list(reference_modes), 0):
        raise ValueError('Expected the previous exact source panel')
    packets = native(source, 'V', contract['counts']['V'])[rows]
    jobs = [(int(row), SOURCE[int(r)], x) for row, r, x in zip(rows, rx, packets, strict=True)]
    output = {}
    with (ProcessPoolExecutor(max_workers=2) as pool,
          np.load(ref/'predictions.npz', allow_pickle=False) as archived):
        for scene in VIEWS:
            stream = pool.map(partial(source_observation, scene), jobs, chunksize=6)
            parts = []
            for start in range(0, len(jobs), BATCH):
                y = np.stack([next(stream) for _ in jobs[start:start+BATCH]])
                parts.append(np.stack([classify_aperture(f, view)
                                       for view in render(y)]))
            output[scene] = np.concatenate(parts, axis=1)
            if not np.array_equal(output[scene][:controls], archived[scene][:controls]):
                raise ValueError(f'Previous source controls changed in {scene}')
            print('W24_PEER_SPECTRAL_PREDICTED', scene, len(jobs), flush=True)
    with (root/'predictions.npz').open('xb') as file:
        np.savez(file, **output)
    write(root/'source-predictions.json', dict(status='PREDICTIONS_FROZEN', views=list(VIEWS),
        physical_ids=ids, modes=list(modes), target_reads=0, source_truth_reads=0,
        waveform=dict(variant=root.name, base=PRACTICAL_RUN, received_only=True, matched_center_edge_energy=True,
                      **(dict(matched_component_energy=True, frozen_native_edges=True) if grouped or refined else {}))))


def eligible(scores):
    original, within, cross = (scores['scores'][m] for m in MODES[2:])
    paired = scores['paired_contrasts'][f'{MODES[-1]}_vs_{MODES[2]}']
    mech = scores['paired_contrasts'][f'{MODES[-1]}_vs_{MODES[3]}']
    return (cross['six_correct'] >= original['six_correct']+18 and paired['CI95_net_six'][0] > 0.
            and cross['low_urban_correct'] >= max(original['low_urban_correct']+6, within['low_urban_correct'])
            and cross['by_scene']['clean'] >= scores['scores']['direct']['by_scene']['clean']
            and mech['net_six'] > 0)


def dispatch(runtime):
    rt = read(runtime)
    root, archive = (Path(rt[k]) for k in ('run_root', 'w24_release_root'))
    root.mkdir()
    state = root/'state.json'
    if read(archive/'submit.json')['code_commit'] != W24_COMMIT:
        raise ValueError('Expected the pinned evaluation archive')
    script = Path(__file__).resolve()

    def stage(phase, config, cwd, gpu=None):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='' if gpu is None else str(gpu), PYTHONPATH=str(cwd),
                   OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
        with (root/f'{phase}.log').open('x') as log:
            proc = subprocess.Popen([rt['python'], '-u', str(script), phase, str(config)], cwd=cwd,
                                    env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            write(state, dict(status='RUNNING', phase=phase, pid=proc.pid, gpu=gpu), exclusive=False)
            if proc.wait():
                raise RuntimeError(f'{phase} failed')

    try:
        config = root/'fit-inputs.json'
        write(config, {key: rt[key] for key in FIT_FIELDS})
        write(state, dict(status='WAIT_GPU_TRAINING'))
        while not (cards := available(rt['allowed_gpus'], 2, training_cap=True)):
            sleep(20)
        gpu = next(card for card in rt['allowed_gpus'] if card in cards)
        stage('fit-source', config, script.parents[2], gpu)
        stage('predict-source', runtime, archive)
        stage('score-source', runtime, archive)
        fields = dict(promote=eligible(read(root/'source-scores.json'))) if root.name == RUN else {}
        write(state, dict(status='SOURCE_SCORED', target_reads=0, **fields), exclusive=False)
    except Exception as exc:
        write(state, dict(status='FAILED', error=str(exc)), exclusive=False)
        raise


def publish(args):
    from experiments.pamf_lora_main.publish import release, release_commit

    ctx = read(args.context)
    rt = {key: ctx[key] for key in ('project', 'python', 'frozen_sources')}
    project = PurePosixPath(rt['project'])
    releases, runs = project/'releases', project/'runs'
    run = (REFINEMENT_RUN if args.post_peer_refinement else COMPONENT_RUN if args.component_directions else RUN)
    rt.update(run_id=run, runner=__spec__.name, code_commit=release_commit(args.repo),
              release_root=str(releases/f'{run}-release'), run_root=str(runs/run),
              source_root=str(releases/f'{DATA_RUN}-release'/'data'/'source'),
              classifier_root=str(runs/AUG_RUN), w24_release_root=str(releases/f'{W24_RUN}-release'),
              training_release_root=str(releases/f'{TRAIN_RUN}-release'),
              practical_cache=str(runs/PRACTICAL_CACHE_RUN/'source_practical'),
              allowed_gpus=[7, 0, 3, 2, 6, 5])
    rt['external_releases'] = [rt[k] for k in ('source_root', 'classifier_root', 'w24_release_root',
                                             'training_release_root', 'practical_cache')]
    rt['external_releases'] += [str(runs/key) for key in (PRIOR, PRACTICAL_RUN, REFERENCE)]
    if args.component_directions:
        rt['external_releases'].append(str(runs/RUN))
    if args.post_peer_refinement:
        rt['external_releases'].append(str(runs/COMPONENT_RUN))
    release(args.repo, args.output, rt, args.ssh_config)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest='action', required=True)
    for action in ('dispatch', 'fit-source', 'predict-source', 'score-source'):
        subs.add_parser(action).add_argument('runtime', type=Path)
    pub = subs.add_parser('publish')
    variants = pub.add_mutually_exclusive_group()
    variants.add_argument('--component-directions', action='store_true')
    variants.add_argument('--post-peer-refinement', action='store_true')
    for name in ('repo', 'output', 'context', 'ssh-config'):
        pub.add_argument('--'+name, required=True, type=Path)
    args = parser.parse_args()
    if args.action == 'publish':
        publish(args)
    elif args.action == 'dispatch':
        dispatch(args.runtime)
    elif args.action == 'fit-source':
        config = read(args.runtime)
        if Path(config['run_root']).name == REFINEMENT_RUN:
            from experiments.pamf_lora_main.w24_post_peer_refinement_source import fit

            fit(config)
        else:
            fit_source(config)
    elif args.action == 'predict-source':
        predict_source(read(args.runtime))
    else:
        run = read(args.runtime)['run_id']
        matrices = {COMPONENT_RUN: (COMPONENT_MODES, COMPONENT_CONTRASTS),
                    REFINEMENT_RUN: (REFINEMENT_MODES, REFINEMENT_CONTRASTS)}
        modes, contrasts = matrices.get(run, (MODES, CONTRASTS))
        score_source(args.runtime, modes=modes, contrasts=contrasts)


if __name__ == '__main__':
    main()
