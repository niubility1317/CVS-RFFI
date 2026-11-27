"""Blind three-seed, 30,000-packet LoRa Full and native Residual evaluation."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
from time import monotonic, sleep

import numpy as np
import torch

if __package__ is None:
    from experiments import pamf_lora_main

    pamf_lora_main.__path__.append(str(Path(__file__).resolve().parent))

from experiments.cric_dura_comparison.io import read, tensor, write
from experiments.pamf_lora_main.dura_repair import complex_iq, real_iq
from experiments.pamf_lora_main.full_packet_views import native, source_contract, target_contract
from experiments.pamf_lora_main.joint_unroll_source import coordinates
from experiments.pamf_lora_main.packet_runner import schedule
from experiments.pamf_lora_main.runner import activate, available
from experiments.pamf_lora_main.w24_native_full import (
    BATCH, DATA_RUN, LIMIT, METHOD, SEED, SOURCE, TARGET, VIEWS, W24_COMMIT, W24_RUN,
    channel_job, classify_aperture, load_restorers, recover,
)
from experiments.pamf_lora_main.w24_residual_pilot import (
    R69_GATE, R69_TRAIN, ROUTE, r69_baseline, receive, restore_r69,
)
from experiments.pamf_lora_main.w24_sat_aug import AUG_WEIGHT, ELIGIBLE, EPOCHS, RUNS
from experiments.pamf_lora_main.w24_wide_transfer import (
    TRAIN_RUN, load_restorer, received_aperture, training_runtime,
)


RUN = '20261125-phase1-lora-w24-formal-both-routes-m3-r01'
RUN_UNGATED = '20261125-phase1-lora-w24-formal-ungated-s2026092702-r01'
RUN_OUTER = '20261125-phase1-lora-w24-formal-outer-objective-s2026092702-r01'
RUN_NATIVE_CENTER = '20261127-phase1-lora-w24-formal-native-center-s2026092702-r01'
RUN_PRACTICAL_CPU = '20261127-phase1-lora-w24-formal-practical-cpu-s2026092702-r01'
RUN_PEER_SPECTRAL_CPU = '20261127-phase1-lora-w24-formal-peer-spectral-cpu-s2026092702-r01'
RUN_PEER_SPECTRAL_FULL = '20261127-phase1-lora-w24-formal-peer-spectral-cpu-s2026092702-r02'
PEER_CPU_RUNS = (RUN_PEER_SPECTRAL_CPU, RUN_PEER_SPECTRAL_FULL)
NATIVE_CENTER_KIND = 'native-center-direction'
PEER_SPECTRAL_KIND = 'peer-spectral-direction'
UNGATED_COMMIT = 'aedf98271cf9473d11ee0d12a48d93e6c03bd24f'
SEEDS = (2026092701, SEED, 2026092703)
FULL = 'full_native'
ROUTES = (FULL, ROUTE)
WIDE = 'wide_dura_half'
WIDE_UNGATED = 'wide_dura_ungated_half'
OUTER_HALF = 'outer_objective_half'
MODES = {
    FULL: ('direct', 'dura_half', 'complex_response_half', WIDE),
    ROUTE: ('direct', 'r69_gated_half', WIDE),
}
RX_PER_JOB = 5
SHARDS = range((len(TARGET) + RX_PER_JOB - 1) // RX_PER_JOB)
ORIGINAL_AUG_COMMIT = '502629361a6aafa564a94947f3116e82f58c3692'
REPLICATION_COMMIT = 'abd0f4332aba04ac4cbca42cb5a3449193a9bbf2'


def cpu_practical(rt):
    return rt.get('run_id') == RUN_PRACTICAL_CPU


def cpu_formal(rt):
    return cpu_practical(rt) or rt.get('run_id') in PEER_CPU_RUNS


def evaluation_routes(third_prior=False):
    if third_prior == PEER_SPECTRAL_KIND:
        return (FULL,)
    if third_prior:
        from experiments.pamf_lora_main.w24_third_prior import trial

        return trial(third_prior).routes
    return ROUTES


def cases(third_prior=False):
    # Interleave routes to make progress on both when only a few cards are free.
    return {f'{route}-{i}': route for i in SHARDS for route in evaluation_routes(third_prior)}


def evaluation_seeds(ungated):
    return (SEED,) if ungated else SEEDS


def evaluation_modes(route, ungated, outer_objective=False, third_prior=False):
    names = MODES[route]
    if third_prior == PEER_SPECTRAL_KIND:
        from experiments.pamf_lora_main.w24_peer_spectral_direction_source import MODES as source_modes

        if route != FULL:
            raise ValueError('Peer spectral direction is registered for native Full only')
        return (*names[:-1], WIDE_UNGATED, *source_modes[2:])
    if third_prior:
        from experiments.pamf_lora_main.w24_third_prior import trial

        return (*names[:-1], WIDE_UNGATED, *trial(third_prior).source_modes[2:4])
    if outer_objective:
        if not ungated:
            raise ValueError('The outer-objective model requires all-packet restoration')
        return (*names[:-1], WIDE_UNGATED, OUTER_HALF)
    return (*names[:-1], WIDE_UNGATED) if ungated else names


def receipt(case, ungated=False, outer_objective=False, third_prior=False):
    route = cases(third_prior)[case]
    i = int(case.rsplit('-', 1)[1])
    proof = dict(status='PREDICTIONS_FROZEN', processing_route=route,
                 seeds=list(evaluation_seeds(ungated)), views=list(VIEWS),
                 modes=list(evaluation_modes(route, ungated, outer_objective, third_prior)), per_receiver=LIMIT,
                 receivers=list(TARGET[i*RX_PER_JOB:(i+1)*RX_PER_JOB]), target_truth_read=False)
    if ungated:
        proof['restoration_input_policy'] = 'all_packets'
    if outer_objective or third_prior:
        proof['reference_run'] = RUN_UNGATED
    return proof


def qualified_outer(rt):
    from experiments.pamf_lora_main.w24_outer_objective_source import MODES as SOURCE_MODES, RUN as SOURCE_RUN, STEPS

    root = Path(rt['project']) / 'runs' / SOURCE_RUN
    if read(root / 'state.json')['status'] != 'SOURCE_SCORED':
        raise ValueError('The matched outer-objective source evaluation is still pending')
    physical = None
    for route in ROUTES:
        mark = read(root / route / 'source-predictions.json')
        score = read(root / route / 'source-scores.json')
        wave = mark['waveform']
        ids = mark['physical_ids']
        if (mark['status'] != 'PREDICTIONS_FROZEN' or mark['modes'] != list(SOURCE_MODES)
                or mark['views'] != list(VIEWS) or len(set(ids)) != 300 or len(ids) != 300
                or mark['target_reads'] or mark['source_truth_reads'] or wave['candidate_run'] != SOURCE_RUN
                or wave['candidate_steps'] != STEPS or wave['ungated'] is not True
                or wave['route'] != route or set(wave['activation']) != set(VIEWS)
                or any(count != [300, 300] for count in wave['activation'].values())
                or score['status'] != 'ANALYZED' or score['source_truth_last'] is not True
                or score['target_reads'] or score['n_packets'] != 300 or score['n_practical'] != 1800
                or score['waveform'] != wave or (physical is not None and physical != ids)):
            raise ValueError(f'Incomplete registered same-packet source evidence for {route}')
        physical = ids
        results = score['scores']
        candidate = results[OUTER_HALF]
        if (candidate['six_correct'] <= results['all_half']['six_correct']
                or candidate['by_scene']['clean'] < results['direct']['by_scene']['clean']):
            raise ValueError(f'The source candidate did not pass the registered six-scene and clean rule for {route}')
    return root


def blind_reference(rt):
    project = Path(rt['project'])
    root = project / 'runs' / RUN_UNGATED
    old = read(project / 'releases' / f'{RUN_UNGATED}-release' / 'runtime.json')
    if (old['run_id'] != RUN_UNGATED or old['code_commit'] != UNGATED_COMMIT
            or old['ungated'] is not True or Path(old['run_root']) != root
            or any(old[key] != rt[key] for key in ('source_root', 'native_target_root', 'w24_release_root'))
            or read(root / 'state.json') != dict(status='PREDICTIONS_FROZEN', cases=list(cases()), target_truth_read=False)):
        raise ValueError('Expected the matching frozen same-route blind LoRa reference')
    return root / 'predictions'


def verify_matched_source(rt):
    if rt['third_prior'] == PEER_SPECTRAL_KIND:
        if rt['run_id'] == RUN_PEER_SPECTRAL_FULL:
            if peer_formal_handoff(rt) != 'PREDICT':
                raise ValueError('Expected the closed no-prediction predecessor before the Full handoff')
            return
        if peer_source_ready(rt) is not True:
            raise ValueError('Peer spectral source qualification is required before formal prediction')
        return
    from experiments.pamf_lora_main.w24_third_prior import PRACTICAL_KIND, frozen_source_results, qualified_source, trial

    root, kind = Path(rt['candidate_root']), rt['third_prior']
    if kind == PRACTICAL_KIND:
        spec = trial(kind)
        runs = Path(rt['project']) / 'runs'
        if cpu_practical(rt):
            if root != runs/spec.run or Path(rt['run_root']) != runs/RUN_PRACTICAL_CPU or spec.routes != (FULL,):
                raise ValueError('Expected the registered CPU Full Practical matrix and frozen parent')
            return
        if (rt.get('run_id') != spec.run or root != runs/spec.run or Path(rt['run_root']) != root/'formal'
                or spec.routes != (FULL,)):
            raise ValueError('Expected the registered fixed Full Practical supervision matrix')
        frozen_source_results(root, kind)
        return
    if rt.get('run_id') != RUN_NATIVE_CENTER:
        qualified_source(root, kind)
        return
    spec = trial(NATIVE_CENTER_KIND)
    runs = Path(rt['project']) / 'runs'
    if (kind != NATIVE_CENTER_KIND or root != runs / spec.run or Path(rt['run_root']) != runs / RUN_NATIVE_CENTER
            or spec.routes != (FULL,)
            or read(root / 'state.json') != dict(status='SOURCE_SCORED', promote=False,
                                                 arms=list(spec.arms), steps=spec.steps)):
        raise ValueError('Unexpected frozen parent for the requested native center Full evaluation')
    frozen_source_results(root, kind)


def peer_source_ready(rt):
    from experiments.pamf_lora_main.w24_peer_spectral_direction_source import MODES as source_modes, RUN as source_run, eligible

    runs = Path(rt['project'])/'runs'
    root = runs/source_run
    run = rt.get('run_id')
    if (run not in PEER_CPU_RUNS or Path(rt['run_root']) != runs/run
            or Path(rt['candidate_root']) != root or rt.get('third_prior') != PEER_SPECTRAL_KIND):
        raise ValueError('Unexpected source binding for the peer spectral formal matrix')
    try:
        state = read(root/'state.json')
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    if state['status'] == 'FAILED':
        raise ValueError('Peer spectral source run failed')
    if state['status'] != 'SOURCE_SCORED':
        return None
    if state['target_reads'] or not isinstance(state['promote'], bool):
        raise ValueError('Invalid frozen source qualification state')
    if not state['promote']:
        return False
    marker = read(root/'source-predictions.json')
    score = read(root/'source-scores.json')
    wave = marker['waveform']
    if (marker['status'] != 'PREDICTIONS_FROZEN' or marker['target_reads'] or marker['source_truth_reads']
            or marker['views'] != list(VIEWS) or marker['modes'] != list(source_modes)
            or len(marker['physical_ids']) != 300 or len(set(marker['physical_ids'])) != 300
            or wave['variant'] != source_run or wave['received_only'] is not True
            or wave['matched_center_edge_energy'] is not True
            or score['status'] != 'ANALYZED' or score['source_truth_last'] is not True or score['target_reads']
            or score['n_packets'] != 300 or score['n_practical'] != 1800 or score['waveform'] != wave
            or not eligible(score)):
        raise ValueError('Invalid peer spectral source qualification evidence')
    return True


def peer_formal_handoff(rt):
    """Complete the fixed three-arm matrix once, irrespective of source scores."""
    if rt.get('run_id') != RUN_PEER_SPECTRAL_FULL:
        raise ValueError('Expected the registered peer spectral Full handoff')
    old = Path(rt['project'])/'runs'/RUN_PEER_SPECTRAL_CPU
    try:
        state = read(old/'state.json')
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    if state['status'] == 'FAILED':
        raise ValueError(f'Peer spectral predecessor failed: {state["error"]}')
    if state['status'] == 'SOURCE_NOT_QUALIFIED':
        if (state != dict(status='SOURCE_NOT_QUALIFIED', target_truth_read=False)
                or any((old/'predictions').iterdir()) or peer_source_ready(rt) is not False):
            raise ValueError('Inconsistent peer spectral predecessor no-prediction state')
        return 'PREDICT'
    if state['status'] == 'PREDICTIONS_FROZEN':
        matrix = cases(PEER_SPECTRAL_KIND)
        if (state != dict(status='PREDICTIONS_FROZEN', cases=list(matrix), target_truth_read=False)
                or any(read(old/'predictions'/case/'complete.json') !=
                       receipt(case, True, third_prior=PEER_SPECTRAL_KIND) for case in matrix)):
            raise ValueError('Incomplete previously executed peer spectral Full matrix')
        return 'REUSE'
    return None


def load_heads(rt, contract, device):
    from experiments.pamf_lora_main.tri_clean95 import NATIVE_WIDE_ROWS, clean_model

    project = Path(rt['project'])
    heads, epochs = [], {}
    for seed in evaluation_seeds(rt.get('ungated', False)):
        name = RUNS[seed]
        run = project / 'runs' / name
        archived = read(project / 'releases' / f'{name}-release' / 'runtime.json')
        expected = ORIGINAL_AUG_COMMIT if seed == SEED else REPLICATION_COMMIT
        if (archived['run_id'] != name or archived['source_root'] != rt['source_root']
                or archived['code_commit'] != expected or Path(archived['run_root']) != run):
            raise ValueError(f'Mismatched augmented W24 source release for {seed}')
        proof = read(run / 'F' / 'complete.json')
        epoch = proof['frozen']['epoch']
        if (proof['status'] != 'SOURCE_FROZEN' or proof['method'] != name or proof['contract'] != contract
                or proof['seed'] != seed or proof['w24_recipe_commit'] != W24_COMMIT
                or proof['epochs'] != EPOCHS or proof['steps'] != 9856 or proof['satellite_batches'] != 9856
                or proof['initialization'] != 'random_seed' or proof['clean_only']
                or proof['satellite_source_role'] != 'L_s' or proof['satellite_weight'] != AUG_WEIGHT
                or epoch not in ELIGIBLE or (seed == SEED and epoch != 104)
                or proof['target_reads'] or proof['U_truth_reads']):
            raise ValueError(f'Mismatched frozen augmented W24 checkpoint for {seed}')
        f = clean_model(dict(scratch=True), run, contract, NATIVE_WIDE_ROWS[METHOD], device)
        f.load_state_dict(torch.load(run / 'F' / 'best.pt', map_location=device, weights_only=True), strict=True)
        heads.append(f.eval().requires_grad_(False))
        epochs[str(seed)] = epoch
    return heads, epochs


def full_restore(baseline, device, packets):
    return recover(packets, baseline, device)[1:]


def residual_restore(baseline, device, packets):
    return (restore_r69(packets, *baseline, device)[0],)


def bindings(rt, routes, device):
    activate(rt)
    contract = source_contract(Path(rt['source_root']))
    heads, epochs = load_heads(rt, contract, device)
    if rt.get('third_prior') == PEER_SPECTRAL_KIND:
        from experiments.pamf_lora_main.w24_fixed_dose_direction import RUN as previous, load_head, source_parent
        from experiments.pamf_lora_main.w24_peer_spectral_direction_source import STEPS, load_trained

        root = Path(rt['candidate_root'])
        parent = source_parent(rt, device)
        old, _ = load_head(root.parent/previous)
        new = load_trained(root)
        return heads, (parent, old, new), None, None, {}, dict(
            epochs=epochs, source_frozen_G=root.name, steps=STEPS,
            parameters=sum(p.numel() for p in parent.parameters()) + sum(p.numel() for p in new.parameters()))
    archive, parent = training_runtime(rt)
    wide, tracker, threshold, frozen = load_restorer(archive, parent, contract, device)
    if rt.get('third_prior'):
        from experiments.pamf_lora_main.w24_third_prior import NATIVE_PACKET_ARMS, load_candidate, trial

        source = Path(rt['candidate_root'])
        names = trial(rt['third_prior']).arms
        if names == NATIVE_PACKET_ARMS:
            arms = [load_candidate(source, contract, device, names[-1])] * 2
        else:
            arms = [load_candidate(source, contract, device, arm) for arm in names]
        info = dict(epochs=epochs, source_frozen_G=source.name, steps=arms[0][1]['steps'],
                    parameters={arm: summary['parameters'] for arm, (_, summary) in zip(names, arms, strict=True)})
        return heads, [model for model, _ in arms], tracker, None, {}, info
    if rt.get('outer_objective'):
        from experiments.pamf_lora_main.w24_outer_objective_source import RUN as SOURCE_RUN, load_candidate

        model, summary = load_candidate(Path(rt['project']) / 'runs' / SOURCE_RUN, contract, device)
        return heads, model, tracker, None, {}, dict(epochs=epochs, source_frozen_G=SOURCE_RUN,
                 steps=summary['steps'], parameters=summary['parameters'])
    # Keep the original response and R69 loader imports after the archived G.
    readers = {}
    if FULL in routes:
        base = load_restorers(contract, Path(rt['run_root']).parent, device)
        readers[FULL] = partial(full_restore, base, device)
    if ROUTE in routes:
        base = r69_baseline(rt, contract, device)
        readers[ROUTE] = partial(residual_restore, base, device)
    gate_key = 'archived_source_threshold' if rt.get('ungated', False) else 'threshold'
    return heads, wide, tracker, threshold, readers, dict(epochs=epochs, **{gate_key: threshold},
             source_frozen_G=TRAIN_RUN, steps=frozen['steps'], parameters=frozen['parameters'])


def wide_delta(g, tracker, observed, *, packets=None):
    y, phase, quality, rms = coordinates(tracker, observed[..., :384])
    edges = complex_iq(observed[..., 384:])
    native_proposal = getattr(g, 'propose_native', None)
    if native_proposal is None:
        stages, *_ = g.propose(y, quality, rms, edges)
    else:
        if packets is None or packets.shape != (len(observed), 8192) or packets.dtype != np.complex64:
            raise ValueError('A native restoration proposal needs the same full-rate received IQ')
        native = tensor(packets.view(np.float32), observed.device).view(torch.complex64)
        stages, *_ = native_proposal(tracker, y, phase, quality, rms, edges, native)
    return real_iq((stages[-1] - y) * phase * rms)


def ungated_decisions(heads, g, tracker, original, *, packets=None):
    wide = original.clone()
    wide[..., :384] += .5 * wide_delta(g, tracker, original, packets=packets)
    return np.stack([classify_aperture(f, wide) for f in heads])


def native_aperture(model, packets, original, fraction):
    if packets.shape != (len(original), 8192) or packets.dtype != np.complex64:
        raise ValueError('Full-packet DURA requires the raw 8,192 complex received samples')
    raw = tensor(packets.view(np.float32), original.device).view(torch.complex64)
    restored = torch.lerp(raw, model.restore_native(raw), fraction)
    return received_aperture(np.from_dlpack(restored.cpu().contiguous()), original.device)


def native_readouts(model, packets, original, fraction):
    whole = native_aperture(model, packets, original, fraction)
    center = torch.cat((whole[..., :384], original[..., 384:]), dim=-1)
    return center, whole


def native_alternatives(models, packets, original, fraction):
    control, candidate = models
    if control is candidate:
        return native_readouts(candidate, packets, original, fraction)
    return tuple(native_aperture(model, packets, original, fraction) for model in models)


@torch.no_grad()
def candidate_labels(heads, g, tracker, packets, device):
    original = received_aperture(packets, device)
    direct = np.stack([classify_aperture(f, original) for f in heads])
    return np.stack((direct, ungated_decisions(heads, g, tracker, original)), axis=1), len(original)


@torch.no_grad()
def prior_labels(heads, arms, tracker, packets, device):
    original = received_aperture(packets, device)
    direct = np.stack([classify_aperture(f, original) for f in heads])
    if hasattr(arms[-1], 'restore_native'):
        alternatives = native_alternatives(arms, packets, original, .5)
        return np.stack((direct, *(np.stack([classify_aperture(f, view) for f in heads])
                                   for view in alternatives)), axis=1), len(original)
    return np.stack((direct, *(ungated_decisions(heads, arm, tracker, original, packets=packets) for arm in arms)),
                    axis=1), len(original)


@torch.no_grad()
def peer_spectral_labels(heads, models, packets):
    from experiments.pamf_lora_main.w24_peer_spectral_direction_source import receive_readouts

    direct, _, original, within, cross = receive_readouts(*models, packets)
    labels = np.stack([np.stack([classify_aperture(f, view) for view in (direct, original, within, cross)])
                       for f in heads])
    return labels, len(packets)


def append_blind_prediction(reference, predicted, old_activity, count, *, require_direct_parity=True):
    if (reference.ndim != 3 or reference.shape[0] != 1 or reference.shape[1] < 2 or reference.shape[-1] != LIMIT
            or predicted.ndim != 3 or predicted.shape[0] != 1 or predicted.shape[1] < 2
            or predicted.shape[-1] != LIMIT or reference.dtype != np.int16 or predicted.dtype != np.int16
            or (require_direct_parity and not np.array_equal(reference[:, 0], predicted[:, 0]))):
        raise ValueError('Direct decisions must match the same-packet blind reference exactly')
    old_count = np.asarray(old_activity)
    if (count != LIMIT or old_count.shape != (reference.shape[1]-1,)
            or old_count.dtype.kind not in 'iu' or not ((0 <= old_count) & (old_count <= LIMIT)).all()
            or old_count[-1] != LIMIT
            or ((reference[:, 1:] != reference[:, :1]).sum(-1) > old_count).any()):
        raise ValueError('Invalid archived or candidate restoration activity')
    return np.concatenate((reference, predicted[:, 1:]), axis=1), [*old_activity, *([count]*(predicted.shape[1]-1))]


@torch.no_grad()
def batch_labels(heads, g, tracker, threshold, baseline, packets, device, *, ungated=False):
    original = received_aperture(packets, device)
    direct = np.stack([classify_aperture(f, original) for f in heads])
    variants, activity = [], []
    for restored in baseline(packets):
        changed = np.any(restored != packets, axis=1)
        values = direct.copy()
        if changed.any():
            aperture = received_aperture(restored[changed], device)
            values[:, changed] = np.stack([classify_aperture(f, aperture) for f in heads])
        variants.append(values)
        activity.append(int(changed.sum()))
    if ungated:
        values = ungated_decisions(heads, g, tracker, original)
        count = len(original)
    else:
        wide, active = g.restore(tracker, original, threshold)
        changed = np.asarray(active.tolist(), dtype=bool)
        values = direct.copy()
        if changed.any():
            values[:, changed] = np.stack([classify_aperture(f, wide[active]) for f in heads])
        count = int(changed.sum())
    return np.stack((direct, *variants, values), axis=1), np.asarray([*activity, count], dtype=int)


@torch.no_grad()
def smoke(rt):
    device = 'cpu' if cpu_formal(rt) else 'cuda:0'
    ungated = rt.get('ungated', False)
    outer = rt.get('outer_objective', False)
    third = rt.get('third_prior', False)
    peer = third == PEER_SPECTRAL_KIND
    if peer or rt.get('run_id') in PEER_CPU_RUNS:
        if not peer or not ungated or outer:
            raise ValueError('Peer spectral direction requires its own ungated Full matrix')
        verify_matched_source(rt)
    append = outer or third
    seeds = evaluation_seeds(ungated)
    routes = evaluation_routes(third)
    heads, g, track, gate, baseline, proof = bindings(rt, routes, device)
    source = Path(rt['source_root'])
    u = native(source, 'U_s', source_contract(source)['counts']['U_s'])[:4]
    clean = []
    for route in routes:
        receiver = channel_job if route == FULL else receive
        if peer:
            batch = partial(peer_spectral_labels, heads, g)
        elif append:
            batch = partial(prior_labels if third else candidate_labels, heads, g, track, device=device)
        else:
            batch = partial(batch_labels, heads, g, track, gate, baseline[route], device=device, ungated=ungated)
        c, clean_activity = batch(u)
        y = np.stack([receiver(VIEWS[-1], SOURCE[0], pair) for pair in enumerate(u)])
        disturbed, active = batch(y)
        if append:
            coverage = clean_activity == active == len(u)
            modes = 4 if peer else 3 if third else 2
        else:
            coverage = (clean_activity[-1] == active[-1] == len(u) if ungated
                        else clean_activity[-1] < len(u) and active[-1] > 0)
            modes = len(evaluation_modes(route, ungated))
        if disturbed.shape != (len(seeds), modes, len(u)) or not coverage:
            raise ValueError(f'Real frozen checkpoint source smoke failed for {route}')
        clean.append(c if append else c[:, 0])
    if len(clean) == 2 and not np.array_equal(*clean):
        raise ValueError('Shared clean received IQ must give identical direct outputs')
    write(Path(rt['run_root']) / 'smoke.json', dict(status='PASS', target_reads=0,
          seeds=list(seeds), ungated=ungated, **proof))
    print('W24_FORMAL_SOURCE_SMOKE', proof['epochs'], flush=True)


def predict(rt, case):
    cpu = cpu_formal(rt)
    device = 'cpu' if cpu else 'cuda:0'
    ungated = rt.get('ungated', False)
    outer = rt.get('outer_objective', False)
    third = rt.get('third_prior', False)
    peer = third == PEER_SPECTRAL_KIND
    if peer or rt.get('run_id') in PEER_CPU_RUNS:
        if not peer or not ungated or outer:
            raise ValueError('Peer spectral direction requires its own ungated Full matrix')
        verify_matched_source(rt)
    append = outer or third
    root = Path(rt['run_root'])
    if read(root / 'smoke.json')['status'] != 'PASS':
        raise ValueError('Source-only W24 smoke required before target')
    expected = receipt(case, ungated, outer, third)
    route = expected['processing_route']
    heads, g, track, gate, baseline, _ = bindings(rt, (route,), device)
    if append:
        batch = (partial(peer_spectral_labels, heads, g) if peer else
                 partial(prior_labels if third else candidate_labels, heads, g, track, device=device))
        prior = blind_reference(rt) / case
        prior_receipt = receipt(case, True)
        if read(prior / 'complete.json') != prior_receipt:
            raise ValueError(f'Invalid archived blind reference shard {case}')
        prior_activity = read(prior / 'activity.json')
    else:
        batch = partial(batch_labels, heads, g, track, gate, baseline[route], device=device, ungated=ungated)
    receiver = channel_job if route == FULL else receive
    data = Path(rt['native_target_root'])
    target_contract(data)
    out = root / 'predictions' / case
    out.mkdir()
    activity = {}
    cpu_direct_differences = {}
    with ProcessPoolExecutor(max_workers=rt['cpu_workers']) as pool:
        for rx in expected['receivers']:
            raw = native(data, rx, LIMIT)
            predictions = {}
            if append:
                with np.load(prior / f'{rx}.npz', allow_pickle=False) as archive:
                    if set(archive.files) != set(VIEWS):
                        raise ValueError(f'Incomplete archived blind predictions for {rx}')
                    earlier = {v: archive[v] for v in VIEWS}
            for scene in VIEWS:
                if scene == 'clean':
                    frames = raw
                else:
                    received = pool.map(partial(receiver, scene, rx), enumerate(raw), chunksize=16)
                    frames = np.stack(list(received))
                labels = []
                active = 0 if append else np.zeros(len(expected['modes']) - 1, dtype=int)
                for i in range(0, LIMIT, BATCH):
                    prediction, changed = batch(frames[i:i+BATCH])
                    labels.append(prediction)
                    active += changed
                predicted = np.concatenate(labels, axis=2)
                key = f'{rx}/{scene}'
                if append:
                    reference = earlier[scene]
                    if reference.shape != (1, len(prior_receipt['modes']), LIMIT):
                        raise ValueError(f'Invalid archived readout dimensions for {key}')
                    predictions[scene], activity[key] = append_blind_prediction(
                        reference, predicted, prior_activity[key], active, require_direct_parity=not cpu)
                    if cpu:
                        cpu_direct_differences[key] = int(np.count_nonzero(reference[:, 0] != predicted[:, 0]))
                else:
                    predictions[scene], activity[key] = predicted, active.tolist()
                print('W24_FORMAL_PREDICTED', case, rx, scene, activity[key], flush=True)
            np.savez(out / f'{rx}.npz', **predictions)
    write(out / 'activity.json', activity)
    if cpu:
        write(out / 'cpu-direct-parity.json', cpu_direct_differences)
    write(out / 'complete.json', expected)


def wait_practical_cpu_source(rt):
    from experiments.pamf_lora_main.w24_third_prior import trial

    root = Path(rt['candidate_root'])
    spec = trial(rt['third_prior'])
    artifacts = [root/arm/f'w24_{arm}-{suffix}' for arm in spec.fit_arms
                 for suffix in ('complete.json', 'model.pt')]
    while True:
        try:
            state = read(root/'state.json')['status']
        except (FileNotFoundError, json.JSONDecodeError):
            sleep(10)
            continue
        if state in ('SOURCE_FROZEN', 'RUNNING', 'SOURCE_SCORED', 'FORMAL_PREDICTING', 'PREDICTIONS_FROZEN', 'FAILED'):
            if all(path.is_file() for path in artifacts):
                # The ordinary no-query smoke reads and strictly validates both checkpoint proofs.
                return
            raise ValueError('Expected both terminal source-frozen candidate checkpoints')
        sleep(10)


def dispatch(runtime):
    rt = read(runtime)
    root, reference = Path(rt['run_root']), Path(rt['w24_release_root'])
    root.mkdir()
    (root / 'logs').mkdir()
    (root / 'predictions').mkdir()
    state = root / 'state.json'

    def status(**fields):
        write(state, fields, exclusive=False)

    try:
        if read(reference / 'submit.json')['code_commit'] != W24_COMMIT:
            raise ValueError('Original frozen W24 model implementation required')
        outer = rt.get('outer_objective', False)
        ungated = rt.get('ungated', False)
        third = rt.get('third_prior', False)
        if third == PEER_SPECTRAL_KIND or rt.get('run_id') in PEER_CPU_RUNS:
            if third != PEER_SPECTRAL_KIND or not ungated or outer:
                raise ValueError('Peer spectral direction requires its own ungated Full matrix')
            if rt['run_id'] == RUN_PEER_SPECTRAL_FULL:
                status(status='WAITING_FOR_PREDECESSOR')
                while (handoff := peer_formal_handoff(rt)) is None:
                    sleep(20)
                if handoff == 'REUSE':
                    status(status='MATCHED_FULL_EXISTS', previous_run=RUN_PEER_SPECTRAL_CPU, target_truth_read=False)
                    return
            else:
                status(status='WAITING_FOR_PEER_SOURCE_SCORE')
                while (ready := peer_source_ready(rt)) is None:
                    sleep(20)
                if not ready:
                    status(status='SOURCE_NOT_QUALIFIED', target_truth_read=False)
                    return
        if third:
            if not ungated or outer:
                raise ValueError('Matched formal prediction requires its own ungated matrix')
            if third != PEER_SPECTRAL_KIND:
                verify_matched_source(rt)
            blind_reference(rt)
            if cpu_practical(rt):
                status(status='WAITING_FOR_FROZEN_CANDIDATES', device='cpu')
                wait_practical_cpu_source(rt)
        if outer:
            if not ungated:
                raise ValueError('The outer-objective model requires all-packet restoration')
            qualified_outer(rt)
            blind_reference(rt)
        runs = Path(rt['project']) / 'runs'
        seeds = evaluation_seeds(ungated)
        status(status='WAITING_FOR_FROZEN_READERS', seeds=list(seeds))
        deadline = monotonic() + 7200
        while True:
            missing = [runs / RUNS[s] for s in seeds if not (runs / RUNS[s] / 'F' / 'complete.json').exists()]
            if not missing:
                break
            if monotonic() >= deadline or any(read(p / 'state.json')['status'] == 'FAILED' for p in missing):
                raise ValueError('Augmented W24 readers did not reach all source-frozen checkpoints')
            sleep(30)

        cpu = cpu_formal(rt)
        script = str(Path(__file__).resolve())

        def start(_plan, phase, name, gpu):
            env = dict(os.environ, CUDA_VISIBLE_DEVICES='' if gpu is None else str(gpu), OMP_NUM_THREADS='2',
                       OPENBLAS_NUM_THREADS='1', PYTHONPATH=str(reference))
            with (root / 'logs' / f'{name}.log').open('x', encoding='utf-8') as log:
                job = subprocess.Popen([rt['python'], '-u', script, phase, str(runtime), name],
                                       cwd=reference, env=env, stdout=log, stderr=subprocess.STDOUT)
            print('W24_FORMAL_STARTED', phase, name, job.pid, gpu, flush=True)
            return job

        gpu = None
        if not cpu:
            status(status='WAIT_GPU')
            deadline = monotonic() + 1800
            allowed = rt['allowed_gpus']
            policy = {'training_cap': True} if rt.get('training_cap') else {}
            while not (cards := available(allowed, max_compute_processes=rt['max_compute_processes'], **policy)):
                if not policy and monotonic() >= deadline:
                    raise ValueError('GPU wait expired')
                sleep(20)
            gpu = next(i for i in allowed if i in cards)
        status(status='SOURCE_SMOKE')
        if start(rt, 'smoke', 'source-smoke', gpu).wait():
            raise ValueError('Formal real checkpoint source smoke failed')
        status(status='TARGET_PREDICTING')
        jobs = cases(third)
        if cpu:
            for case in jobs:
                if start(rt, 'predict', case, None).wait():
                    raise RuntimeError(f'CPU Formal prediction failed for {case}')
        else:
            schedule(rt, [('predict', case) for case in jobs], start)
        for case in jobs:
            if read(root / 'predictions' / case / 'complete.json') != receipt(case, ungated, outer, third):
                raise ValueError(f'Incomplete frozen prediction shard {case}')
        status(status='PREDICTIONS_FROZEN', cases=list(jobs), target_truth_read=False)
    except Exception as exc:
        status(status='FAILED', error=str(exc))
        raise


def main():
    from experiments.pamf_lora_main.publish import release, release_commit

    parser = argparse.ArgumentParser()
    actions = parser.add_subparsers(dest='action', required=True)
    actions.add_parser('dispatch').add_argument('runtime', type=Path)
    for phase in ('smoke', 'predict'):
        cmd = actions.add_parser(phase)
        cmd.add_argument('runtime', type=Path)
        cmd.add_argument('case')
    publish = actions.add_parser('publish')
    choice = publish.add_mutually_exclusive_group()
    choice.add_argument('--outer-objective', action='store_true')
    choice.add_argument('--native-center-direction', action='store_true')
    choice.add_argument('--native-practical-cpu', action='store_true')
    choice.add_argument('--peer-spectral-cpu', action='store_true')
    choice.add_argument('--peer-spectral-full', action='store_true')
    for key in ('repo', 'context', 'ssh-config', 'output'):
        publish.add_argument('--' + key, type=Path, required=True)
    args = parser.parse_args()
    if args.action == 'publish':
        ctx = read(args.context)
        rt = {k: ctx[k] for k in ('project', 'python', 'frozen_sources')}
        project = PurePosixPath(rt['project'])
        runs, releases = project / 'runs', project / 'releases'
        data = releases / f'{DATA_RUN}-release' / 'data'
        run = (RUN_PEER_SPECTRAL_FULL if args.peer_spectral_full else
               RUN_PEER_SPECTRAL_CPU if args.peer_spectral_cpu else RUN_PRACTICAL_CPU if args.native_practical_cpu
               else RUN_NATIVE_CENTER if args.native_center_direction
               else RUN_OUTER if args.outer_objective else RUN_UNGATED)
        rt.update(run_id=run, code_commit=release_commit(args.repo), runner=__spec__.name, ungated=True,
                  outer_objective=args.outer_objective,
                  release_root=str(releases / f'{run}-release'), run_root=str(runs / run),
                  source_root=str(data / 'source'), native_target_root=str(data / 'target'),
                  w24_release_root=str(releases / f'{W24_RUN}-release'),
                  training_release_root=str(releases / f'{TRAIN_RUN}-release'),
                  allowed_gpus=[0, 5, 3, 4, 2, 7, 1, 6], cpu_workers=8, max_workers=2, max_compute_processes=8)
        rt['external_releases'] = [rt[k] for k in ('source_root', 'native_target_root', 'w24_release_root',
                                                  'training_release_root')]
        if run in PEER_CPU_RUNS:
            from experiments.pamf_lora_main.w24_fixed_dose_direction import RUN as prior_direction
            from experiments.pamf_lora_main.w24_peer_spectral_direction_source import PRACTICAL_RUN, RUN as source_run

            parent = runs/source_run
            rt.update(third_prior=PEER_SPECTRAL_KIND, candidate_root=str(parent), allowed_gpus=[],
                      cpu_workers=2, max_workers=1)
            rt['external_releases'] += [str(runs/x) for x in (RUNS[SEED], RUN_UNGATED, source_run,
                                                            PRACTICAL_RUN, prior_direction)]
            if args.peer_spectral_full:
                rt['external_releases'].append(str(runs/RUN_PEER_SPECTRAL_CPU))
        elif args.native_practical_cpu:
            from experiments.pamf_lora_main.w24_third_prior import PRACTICAL_KIND, trial

            parent = runs / trial(PRACTICAL_KIND).run
            rt.update(third_prior=PRACTICAL_KIND, candidate_root=str(parent), allowed_gpus=[],
                      cpu_workers=2, max_workers=1)
            rt['external_releases'] += [str(p) for p in (runs / RUNS[SEED], parent, runs / RUN_UNGATED)]
        elif args.native_center_direction:
            from experiments.pamf_lora_main.w24_third_prior import trial

            parent = runs / trial(NATIVE_CENTER_KIND).run
            rt.update(third_prior=NATIVE_CENTER_KIND, candidate_root=str(parent), allowed_gpus=[0, 3, 7, 2, 6, 5])
            rt['external_releases'] += [str(p) for p in (runs / RUNS[SEED], parent, runs / RUN_UNGATED)]
        elif args.outer_objective:
            from experiments.pamf_lora_main.w24_outer_objective_source import RUN as SOURCE_RUN

            rt['external_releases'] += [str(runs / x) for x in (RUNS[SEED], SOURCE_RUN, RUN_UNGATED)]
        else:
            rt['external_releases'] += [str(runs / x) for x in (RUNS[SEED], R69_TRAIN, R69_GATE)]
        release(args.repo, args.output, rt, args.ssh_config)
    elif args.action == 'dispatch':
        dispatch(args.runtime)
    elif args.action == 'smoke':
        smoke(read(args.runtime))
    else:
        predict(read(args.runtime), args.case)


if __name__ == '__main__':
    main()
