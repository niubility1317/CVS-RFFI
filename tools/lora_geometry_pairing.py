"""Reconcile frozen LoRa predictions with source truth by scene and receiver."""

import json
import os
from pathlib import Path
import sys

import numpy as np

from experiments.cric_dura_comparison.io import read
from experiments.pamf_lora_main import held_chirp_precision_source, lapped_posterior_source, split_shrink_source
from experiments.pamf_lora_main.channel_blind_source import panel
from experiments.pamf_lora_main.complex_geometry_source import COMPARISONS, MODES, RUN
from experiments.pamf_lora_main.data import SOURCE, VIEWS
from experiments.pamf_lora_main.denoise_source import KEY
from experiments.pamf_lora_main.full_packet_views import source_contract
from experiments.pamf_lora_main.local_wiener_source import ROUTES, read_frozen


ALL_VIEWS = (*VIEWS,'six_scene')
REPEATS = 1600
PAIRS = (*COMPARISONS,('direct','reference'))


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
    result = {}
    for route, (_, predictions, _) in pieces.items():
        hits = np.stack([predictions[view]==truth for view in VIEWS],axis=1)
        official = read(root/f'{route}-source-scores.json')
        if any(hits[:,i].sum(-1).tolist()!=official['metrics'][view]['correct'] for i,view in enumerate(VIEWS)):
            raise ValueError('Reconciled counts differ from the independent frozen source score')
        result[route] = compare(hits,pairs,groups,modes)
    return dict(reader_seed=2026092702,bootstrap_replicates=REPEATS,target_reads=0,routes=result)


if __name__=='__main__':
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':
        raise ValueError('This diagnostic scores on an independent CPU process')
    print(json.dumps(collect(read(Path(sys.argv[1]))),allow_nan=False))
