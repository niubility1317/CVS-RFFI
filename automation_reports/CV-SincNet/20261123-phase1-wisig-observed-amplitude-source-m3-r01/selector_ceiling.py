"""Truth-last coverage bound for frozen WiSig source-only proposals."""
import json
from pathlib import Path
import sys

import numpy as np

PROJECT = Path('/home/szu2070436088/2510044040/CV-SincNet')
ARCHIVED = (
    '20261123-phase1-wisig-preamble-receiver-source-m2-r02',
    '20261123-phase1-wisig-joint-preamble-timing-source-m2-r01',
    '20261123-phase1-wisig-observable-action-source-m4-r01',
    '20261123-phase1-wisig-phase-magnitude-source-m4-r01',
    '20261123-phase1-wisig-repeat-block-source-m3-r01',
)
ADDED = '20261123-phase1-wisig-observed-amplitude-source-m3-r01'
VIEWS = ('clean','practical_high','practical_mid','practical_low_suburban',
         'practical_high_urban','practical_mid_urban','practical_low_urban')
COUNT = 1800
ROUTES = ('full','residual')
REFERENCE = '20261019-phase1-reception-restoration-selfsup-manysig-m6-r02'
SOURCE = '20261021-phase1-CRIC-DURA-comparison-m153-r02'


def load(root, route):
    if json.loads((root/'state.json').read_text())['status'] != 'SOURCE_SCORED':
        raise ValueError('Expected frozen scored archive')
    if 'repeat-block' in root.name:
        predictions = root/'predictions'
        proof=json.loads((predictions/'complete.json').read_text())
        assert proof['status']=='FROZEN' and proof['target_reads']==proof['source_truth_reads']==0
        ids=None
        modes=['raw' if k=='direct' else 'R69' if k=='r69' else k for k in proof['modes']]
        raw={
            k: [] for k in modes
        }
        for view in VIEWS:
            with np.load(predictions/f'{route}_{view}.npz',allow_pickle=False) as row:
                if ids is None:
                    ids=row['ids']
                assert np.array_equal(ids,row['ids'])
                for k, values in zip(modes,row['decision'],strict=True):
                    raw[k].append(values)
        decisions={k:np.stack(v) for k,v in raw.items()}
    else:
        path=root/route
        proof=json.loads((path/'complete.json').read_text())
        assert proof['status']=='PREDICTIONS_COMPLETE' and not proof['predictor_reads_truth']
        assert not proof.get('diagnostic_reference_input',False)
        with np.load(path/'predictions.npz',allow_pickle=False) as row:
            ids=row['ids']
            decisions={k:np.stack([row[f'{k}__{v}'] for v in VIEWS]) for k in proof['modes']}
    assert ids.shape==(COUNT,) and len(np.unique(ids))==COUNT
    assert all(p.shape==(7,COUNT) and p.dtype.kind in 'iu' and (p>=0).all() and (p<6).all()
               for p in decisions.values())
    return ids, decisions


matrices={}
for route in ROUTES:
    root=PROJECT/'runs'/REFERENCE/'cache'/('validation_full' if route=='full' else 'validation')
    expected=np.load(root/'physical_ids.npy',allow_pickle=False)
    rx=np.load(root/'rx.npy',allow_pickle=False)
    day=np.load(root/'day.npy',allow_pickle=False)
    decisions={}
    raw=None
    reference=None
    for item in (*ARCHIVED,ADDED):
        ids,data=load(PROJECT/'runs'/item,route)
        assert np.array_equal(ids,expected)
        raw=data['raw'] if raw is None else raw
        reference=data['R69'] if reference is None else reference
        assert np.array_equal(data['raw'],raw) and np.array_equal(data['R69'],reference)
        decisions[item]={k:v for k,v in data.items() if k not in ('raw','R69')}
    action_parent=ARCHIVED[2]
    for k in ('fitted','fitted_guarded'):
        assert np.array_equal(decisions[action_parent][k],decisions[ADDED][k])
    matrices[route]=(expected,rx,day,raw,reference,decisions)
assert np.array_equal(matrices['full'][0], matrices['residual'][0])

# Only after all frozen proposal matrices and physical bindings validate.
root=PROJECT/'runs'/SOURCE/'data'/'F0'
with np.load(root/'V_metadata.npz',allow_pickle=False) as data:
    index={key:i for i,key in enumerate(data['ids'])}
    pos=[index[i] for i in matrices['full'][0]]
    truth=data['y'][pos]
    assert all(np.array_equal(v[1],data['rx_i'][pos]) and np.array_equal(v[2],data['day_i'][pos])
               for v in matrices.values())

rng=np.random.default_rng(2026112339)
_,rx,day,*_=matrices['full']
strata=[np.flatnonzero((rx==r)&(day==d)) for r in np.unique(rx) for d in np.unique(day)]
draws=np.concatenate([rng.choice(ix,size=(2000,len(ix))) for ix in strata if len(ix)],axis=1)
by_route={}
for route,(_,_,_,raw,r69,decisions) in matrices.items():
    raw_hits=raw==truth
    ref_hits=r69==truth
    archive_union=raw_hits.copy()
    for k in ARCHIVED:
        for item in decisions[k].values():
            archive_union |= item==truth
    legal_union=archive_union.copy()
    for item in decisions[ADDED].values():
        legal_union |= item==truth
    record={}
    for name,hits in (('raw',raw_hits),('R69',ref_hits),('archived_union',archive_union),('with_new_union',legal_union)):
        delta=(hits[1:].astype(int)-ref_hits[1:].astype(int)).sum(0)
        record[name]=dict(six=int(hits[1:].sum()), per_view=dict(zip(VIEWS,map(int,hits.sum(axis=1)))),
                          vs_R69=int(delta.sum()), ci_vs_R69=np.quantile(delta[draws].sum(1),(.025,.975)).tolist())
    by_route[route]=record
print(json.dumps(dict(each_route_records=COUNT, views=list(VIEWS), archived_runs=list(ARCHIVED),
                      added_run=ADDED, target_reads=0, diagnostic_only=True, routes=by_route),indent=2))
