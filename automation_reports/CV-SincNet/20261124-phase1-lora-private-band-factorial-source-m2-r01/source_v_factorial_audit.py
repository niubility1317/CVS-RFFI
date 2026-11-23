from itertools import combinations
from pathlib import Path
import json
import numpy as np

from experiments.pamf_lora_main.chirp_spectrum_source import read_frozen
from experiments.pamf_lora_main.data import SOURCE,VIEWS
from experiments.pamf_lora_main.full_packet_views import source_contract
from experiments.pamf_lora_main.private_transport_bands import SUBSETS
from experiments.pamf_lora_main.private_transport_bands_source import MODES,RUN
from experiments.pamf_lora_main.private_transport_source import MODES as PARENT_MODES,RUN as PARENT
from experiments.pamf_lora_main.channel_blind_source import panel
from experiments.cric_dura_comparison.io import read

base=Path('/home/szu2070436088/2510044040/CV-SincNet')
root=base/'runs'/RUN
rt=read(base/'releases'/f'{RUN}-release'/'runtime.json')
if read(root/'state.json')['status']!='SOURCE_SCORED':
    raise ValueError('All source predictions and the original independent scorer must already be complete')
coverage=read(root/'clean-complete-scores.json')
assert coverage['summary']['n']==12000 and coverage['truth_after_predictions'] and not coverage['target_reads']
parent=root.parent/PARENT
mapping=(('direct','direct'),('reference','reference'),('rank_half','rank_half'),
          ('bands_000','wiener_half'),('bands_111','wiener_full'))
pred={r:read_frozen(root,r,MODES) for r in ('full','residual')}
assert pred['full'][0]==pred['residual'][0]
exist={r:read_frozen(parent,r,PARENT_MODES) for r in pred}
sanity={}
for r,(ids,rows,_) in pred.items():
    old_ids,old,_=exist[r]
    assert ids==old_ids
    sanity[r]={v:{l:int((rows[v][MODES.index(l)]!=old[v][PARENT_MODES.index(r1)]).sum())
                  for l,r1 in mapping} for v in VIEWS}
assert not any(value for rows in sanity.values() for d in rows.values() for value in d.values())
source=Path(rt['source_root'])
keys=np.asarray(source_contract(source)['role_ids']['V'])
positions={key:i for i,key in enumerate(keys)}
chosen=np.asarray([positions[k] for k in pred['full'][0]])
with np.load(source/'V_metadata.npz',allow_pickle=False) as truth:
    receiver=np.asarray(SOURCE)[truth['rx_i']]
    assert np.array_equal(panel(receiver),chosen)
    assert np.array_equal(keys[chosen],truth['ids'][chosen])
    labels=truth['y'][chosen]
rng=np.random.default_rng(2026113752)
sets=[np.flatnonzero(receiver[chosen]==one) for one in SOURCE]
draws=np.concatenate([rng.choice(pool,(1600,len(pool)),replace=True) for pool in sets],axis=1)
sign=np.array([[1 if c=='1' else -1 for c in key] for key in SUBSETS],dtype=np.float64)
factors=('1_to_8','9_to_32','33_to_64')
report={}
for r,(_,rows,_) in pred.items():
    hits=np.stack([rows[view][3:]==labels for view in VIEWS[1:]],axis=1).astype(np.int8)
    report[r]={}
    for order in (1,2,3):
        for indices in combinations(range(3),order):
            multiplier=sign[:,indices].prod(axis=1)/2**(3-order)
            effects=np.tensordot(multiplier,hits,axes=1)
            aggregate=effects.sum(0)
            ci=np.quantile(aggregate[draws].sum(1),(.025,.975)).tolist()
            low=effects[-1]
            ci_low=np.quantile(low[draws].sum(1),(.025,.975)).tolist()
            key='_x_'.join(factors[i] for i in indices)
            report[r][key]=dict(six=float(aggregate.sum()),ci=ci,low=float(low.sum()),ci_low=ci_low,
                                by_view=effects.sum(1).tolist())
print(json.dumps(dict(n=len(chosen),target_reads=0,source_V_truth_read_after_complete_predictions=True,
                      source_V_sample_bootstrap=1600,F_reads=0,endpoint_mismatches=sanity,factorial=report)),flush=True)
