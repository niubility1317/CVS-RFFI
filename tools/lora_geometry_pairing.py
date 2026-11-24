"""Reconcile frozen LoRa predictions with source truth by scene and receiver."""

import json
import os
from pathlib import Path
import sys

import numpy as np

from experiments.cric_dura_comparison.io import read
from experiments.pamf_lora_main import held_chirp_precision_source, lapped_posterior_source
from experiments.pamf_lora_main import coherent_interval_source
from experiments.pamf_lora_main import joint_spectrum_source
from experiments.pamf_lora_main import noise_split_diagnostic
from experiments.pamf_lora_main import packet_relation_source, reference_cancel_source, split_shrink_source
from experiments.pamf_lora_main import received_update_source
from experiments.pamf_lora_main import phase_observation_diagnostic
from experiments.pamf_lora_main import phase_dose_source
from experiments.pamf_lora_main import variance_transport_source
from experiments.pamf_lora_main.channel_blind_source import panel
from experiments.pamf_lora_main.complex_geometry_source import COMPARISONS, MODES, RUN
from experiments.pamf_lora_main.data import SOURCE, VIEWS
from experiments.pamf_lora_main.denoise_source import KEY
from experiments.pamf_lora_main.full_packet_views import source_contract
from experiments.pamf_lora_main.local_wiener_source import ROUTES, read_frozen


ALL_VIEWS = (*VIEWS,'six_scene')
REPEATS = 1600
PAIRS = (*COMPARISONS,('direct','reference'))
NOISE_INTERACTIONS = {
    'common_private': ('exact_noncarrier','direct','exact_common','exact_private'),
    'phase_amplitude': ('exact_noncarrier','direct','noncarrier_phase','noncarrier_amplitude'),
}
LEARNED_INTERACTIONS = {
    'common_private': ('exchange_full','direct','common_only','private_only'),
    'phase_amplitude': ('exchange_full','common_only','common_with_phase','common_with_magnitude'),
}
LEARNED_RUN = '20261124-diagnostic-lora-learned-split-source-m2-r01'
ANCHOR_RUN = '20261124-phase1-lora-common-anchor-source-m4-r01'
BURST_DOSE_RUN = '20261124-phase1-lora-phase-dose-bursts-source-m4-r01'
RADIUS_RUN = '20261124-diagnostic-lora-phase-radius-source-m2-r01'
NEIGHBOR_PHASE_RUN = '20261124-phase1-lora-neighbor-phase-source-m4-r01'
RADIUS_INTERACTIONS = {
    'power_phase': ('power_joint', 'direct', 'power_radius', 'local_full'),
    'fold_phase': ('fold_joint', 'direct', 'fold_radius', 'local_full'),
    'isolated_equal_phase': ('exact_joint_equal', 'direct', 'exact_radius_equal', 'local_full'),
    'isolated_all_phase': ('exact_joint_all', 'direct', 'exact_radius_all', 'local_full'),
}


def with_six(value):
    return np.concatenate((value,value[:,1:].sum(axis=1,keepdims=True)),axis=1).astype(np.int16)


def compare(hits, pairs, groups, modes=MODES):
    indices = [(modes.index(a),modes.index(b)) for a,b in pairs]
    candidate, base = (hits[list(col)] for col in zip(*indices,strict=True))
    before, after = with_six(base),with_six(candidate)
    repairs, harms = with_six(candidate & ~base),with_six(base & ~candidate)
    net = repairs-harms
    if not np.array_equal(net,after-before):
        raise ValueError('The paired correct, repair, and harm counts must reconcile')
    rng = np.random.default_rng(KEY+3)
    table = {}
    pooled_draws = np.zeros((*net.shape[:2],REPEATS),dtype=np.int64)

    def summarize(selection, bootstrap):
        counts = [x[...,selection].sum(-1) for x in (before,after,repairs,harms,net)]
        lo,hi = np.quantile(bootstrap,(.025,.975),axis=-1)
        n = [len(selection)]*len(VIEWS)+[len(selection)*6]
        return {f'{a}_vs_{b}':{view:dict(n=n[v],base_correct=int(counts[0][p,v]),correct=int(counts[1][p,v]),
                repairs=int(counts[2][p,v]),harms=int(counts[3][p,v]),net=int(counts[4][p,v]),
                paired_ci=[float(lo[p,v]),float(hi[p,v])]) for v,view in enumerate(ALL_VIEWS)}
                for p,(a,b) in enumerate(pairs)}

    for receiver, choice in groups.items():
        sampled = choice[rng.integers(0,len(choice),(REPEATS,len(choice)))]
        bootstrap = net[...,sampled].sum(-1)
        table[receiver] = summarize(choice,bootstrap)
        pooled_draws += bootstrap
    table['all'] = summarize(np.concatenate(tuple(groups.values())),pooled_draws)
    return table


def factor_interactions(hits, groups, modes, contrasts):
    rng = np.random.default_rng(KEY+3)
    samples = {name:indices[rng.integers(0,len(indices),(REPEATS,len(indices)))]
               for name,indices in groups.items()}
    result = {}
    for name, contrast in contrasts.items():
        joint, raw, first, second = (hits[modes.index(mode)].astype(np.int16) for mode in contrast)
        value = joint + raw - first - second
        outcomes = dict(zip(ALL_VIEWS, (*value,value[1:].sum(0)), strict=True))
        result[name] = {}
        for view, row in outcomes.items():
            simulated = sum(row[draws].sum(-1) for draws in samples.values())
            result[name][view] = dict(net=int(row.sum()),paired_ci=np.quantile(simulated,(.025,.975)).tolist(),
                by_rx={rx:int(row[indices].sum()) for rx,indices in groups.items()})
    return result


def collect(rt):
    root, source = Path(rt['run_root']),Path(rt['source_root'])
    if read(root/'state.json')['status']!='SOURCE_SCORED':
        raise ValueError('Expected the complete dual-route source score')
    if rt.get('run_id')==RUN:
        modes, pairs = MODES, PAIRS
    elif rt.get('run_id')==split_shrink_source.RUN:
        modes = split_shrink_source.MODES
        pairs = (*split_shrink_source.COMPARISONS, ('direct','reference'))
    elif rt.get('run_id')==lapped_posterior_source.RUN:
        modes = lapped_posterior_source.MODES
        pairs = (*lapped_posterior_source.COMPARISONS, ('direct','reference'))
    elif rt.get('run_id')==held_chirp_precision_source.RUN:
        modes = held_chirp_precision_source.MODES
        pairs = (*held_chirp_precision_source.COMPARISONS, ('direct','reference'))
    elif rt.get('run_id')==packet_relation_source.RUN:
        modes = packet_relation_source.MODES
        pairs = (*packet_relation_source.COMPARISONS, ('direct','reference'))
    elif rt.get('run_id')==reference_cancel_source.RUN:
        modes = reference_cancel_source.MODES
        pairs = (*reference_cancel_source.COMPARISONS, ('direct','reference'))
    elif rt.get('run_id')==variance_transport_source.RUN:
        modes = variance_transport_source.MODES
        pairs = (*variance_transport_source.COMPARISONS, ('direct','reference'))
    elif rt.get('run_id')==coherent_interval_source.RUN:
        modes = coherent_interval_source.MODES
        pairs = (*coherent_interval_source.COMPARISONS, ('direct','reference'))
    elif rt.get('run_id')==noise_split_diagnostic.RUN:
        modes = noise_split_diagnostic.MODES
        pairs = (*noise_split_diagnostic.COMPARISONS, ('direct','reference'))
    elif rt.get('run_id')==joint_spectrum_source.RUN:
        modes = joint_spectrum_source.MODES
        pairs = (*joint_spectrum_source.COMPARISONS, ('direct','reference'))
    elif rt.get('run_id')==LEARNED_RUN:
        from experiments.pamf_lora_main import learned_split_diagnostic
        modes = learned_split_diagnostic.MODES
        pairs = (*learned_split_diagnostic.COMPARISONS, ('direct','reference'),('exchange_full','reference'),
                 ('exchange_half','reference'),('common_only','private_only'))
    elif rt.get('run_id')==ANCHOR_RUN:
        from experiments.pamf_lora_main import common_anchor_source
        modes = common_anchor_source.MODES
        pairs = (*common_anchor_source.COMPARISONS, ('direct','reference'))
    elif rt.get('run_id')==received_update_source.RUN:
        modes = received_update_source.MODES
        pairs = (*received_update_source.COMPARISONS, ('direct','reference'))
    elif rt.get('run_id')==phase_observation_diagnostic.RUN:
        modes = phase_observation_diagnostic.MODES
        pairs = (*phase_observation_diagnostic.COMPARISONS, ('direct','reference'))
    elif rt.get('run_id') in (phase_dose_source.RUN, BURST_DOSE_RUN):
        modes = phase_dose_source.MODES
        pairs = (*phase_dose_source.COMPARISONS, ('direct','reference'))
    elif rt.get('run_id')==RADIUS_RUN:
        from experiments.pamf_lora_main import phase_radius_diagnostic
        modes = phase_radius_diagnostic.MODES
        pairs = (*phase_radius_diagnostic.COMPARISONS, ('direct', 'reference'))
    elif rt.get('run_id')==NEIGHBOR_PHASE_RUN:
        from experiments.pamf_lora_main import neighbor_phase_source
        modes = neighbor_phase_source.MODES
        pairs = (*neighbor_phase_source.COMPARISONS, ('direct', 'reference'))
    else:
        from experiments.pamf_lora_main import empirical_tangent_source as tangent
        if rt.get('run_id')!=tangent.RUN:
            raise ValueError('Expected a registered LoRa recovery study')
        modes, pairs = tangent.MODES, (*tangent.COMPARISONS,('direct','reference'))
    pieces = {route:read_frozen(root,route,modes) for route in ROUTES}
    ids = pieces[ROUTES[0]][0]
    if any(item[0]!=ids for item in pieces.values()):
        raise ValueError('Frozen source physical rows differ across routes')
    available = source_contract(source)['role_ids']['V']
    index = {key:i for i,key in enumerate(available)}
    selected = np.asarray([index[key] for key in ids])
    with np.load(source/'V_metadata.npz',allow_pickle=False) as meta:
        receivers = np.asarray(SOURCE)[meta['rx_i']]
        if not np.array_equal(selected,panel(receivers)) or meta['ids'][selected].tolist()!=ids:
            raise ValueError('The registered source V order changed')
        truth = meta['y'][selected]
    groups = {name:np.flatnonzero(receivers[selected]==name) for name in SOURCE}
    result, interactions = {}, {}
    for route, (_, predictions, _) in pieces.items():
        hits = np.stack([predictions[view]==truth for view in VIEWS],axis=1)
        official = read(root/f'{route}-source-scores.json')
        if any(hits[:,i].sum(-1).tolist()!=official['metrics'][view]['correct'] for i,view in enumerate(VIEWS)):
            raise ValueError('Reconciled counts differ from the independent frozen source score')
        result[route] = compare(hits,pairs,groups,modes)
        if rt.get('run_id')==noise_split_diagnostic.RUN:
            interactions[route] = factor_interactions(hits,groups,modes,NOISE_INTERACTIONS)
        elif rt.get('run_id')==LEARNED_RUN:
            interactions[route] = factor_interactions(hits,groups,modes,LEARNED_INTERACTIONS)
        elif rt.get('run_id')==RADIUS_RUN:
            interactions[route] = factor_interactions(hits, groups, modes, RADIUS_INTERACTIONS)
    artifact = dict(reader_seed=2026092702,bootstrap_replicates=REPEATS,target_reads=0,routes=result)
    if interactions:
        key = {LEARNED_RUN: 'learned_interactions', RADIUS_RUN: 'radius_interactions'}.get(
            rt.get('run_id'), 'noise_interactions')
        artifact[key] = interactions
    return artifact


def phase_dose_interaction(rt):
    """Compare perturbation recipes on the same frozen source panel."""
    parent = Path(rt['run_root']).parent
    source = Path(rt['source_root'])
    modes = phase_dose_source.MODES
    roots = {'standard': parent / phase_dose_source.RUN, 'bursts': parent / BURST_DOSE_RUN}
    if any(read(run / 'state.json')['status'] != 'SOURCE_SCORED' for run in roots.values()):
        raise ValueError('Both phase-dose studies must already be truth-last scored')
    frozen = {name: {route: read_frozen(run, route, modes) for route in ROUTES}
              for name, run in roots.items()}
    ids = frozen['standard'][ROUTES[0]][0]
    if any(data[0] != ids for values in frozen.values() for data in values.values()):
        raise ValueError('The two trained phase-dose studies must use exactly the same physical IDs')
    available = source_contract(source)['role_ids']['V']
    index = {key: i for i, key in enumerate(available)}
    chosen = np.asarray([index[key] for key in ids])
    with np.load(source / 'V_metadata.npz', allow_pickle=False) as metadata:
        receivers = np.asarray(SOURCE)[metadata['rx_i']]
        if not np.array_equal(chosen, panel(receivers)) or metadata['ids'][chosen].tolist() != ids:
            raise ValueError('Unexpected source panel order')
        truth = metadata['y'][chosen]
    groups = {rx: np.flatnonzero(receivers[chosen] == rx) for rx in SOURCE}
    aliases = tuple(f'{arm}_{mode}' for arm in roots for mode in modes)
    pairs = tuple((f'bursts_{mode}', f'standard_{mode}')
                  for mode in ('pooled_full', 'local_full', 'local_equal', 'pooled_equal'))
    factors = {
        'trained_local_effect': ('bursts_local_full', 'standard_pooled_full',
                                 'bursts_pooled_full', 'standard_local_full'),
        'fixed_edit_local_effect': ('bursts_local_equal', 'standard_pooled_equal',
                                   'bursts_pooled_equal', 'standard_local_equal'),
    }
    output = {}
    for route in ROUTES:
        matrices = []
        previous = None
        for name, root in roots.items():
            guesses = np.stack([frozen[name][route][1][scene] for scene in VIEWS], axis=1)
            scored = read(root / f'{route}-source-scores.json')
            correct = guesses == truth
            if any(correct[:, i].sum(-1).tolist() != scored['metrics'][scene]['correct']
                   for i, scene in enumerate(VIEWS)):
                raise ValueError('Independent reconstruction differs from the frozen source score')
            if previous is not None and not np.array_equal(previous, guesses[:2]):
                raise ValueError('Direct or DURA predictions differ across training recipes')
            previous = guesses[:2]
            matrices.append(correct)
        hits = np.concatenate(matrices)
        output[route] = dict(paired=compare(hits, pairs, groups, aliases),
                             effects=factor_interactions(hits, groups, aliases, factors))
    return dict(reader_seed=2026092702, bootstrap_replicates=REPEATS, target_reads=0, routes=output)


if __name__=='__main__':
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':
        raise ValueError('This diagnostic scores on an independent CPU process')
    method = phase_dose_interaction if '--dose-interaction' in sys.argv else collect
    print(json.dumps(method(read(Path(sys.argv[1]))),allow_nan=False))
