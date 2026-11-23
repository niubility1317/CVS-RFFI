from concurrent.futures import ProcessPoolExecutor
from functools import partial
import json
from pathlib import Path

import numpy as np
import torch

from experiments.cric_dura_comparison.io import read, tensor
from experiments.pamf_lora_main.channel_blind import corruption_score
from experiments.pamf_lora_main.channel_blind_source import receive_only
from experiments.pamf_lora_main.chirp_spectrum_source import read_frozen
from experiments.pamf_lora_main.data import SOURCE,VIEWS
from experiments.pamf_lora_main.dura_repair import PacketRestorer
from experiments.pamf_lora_main.full_packet_publish import DATA_RUN
from experiments.pamf_lora_main.full_packet_views import source_contract,source_role
from experiments.pamf_lora_main.joint_unroll_source import coordinates
from experiments.pamf_lora_main.clean_pilot_response_source import MODES,RUN
from experiments.pamf_lora_main.runner import activate

project=Path('/home/szu2070436088/2510044040/CV-SincNet')
parent=project/'runs'/RUN
if read(parent/'state.json')['status']!='SOURCE_SCORED':
    raise ValueError('Need already-scored protected pilot source outputs')
rt=read(project/'releases'/f'{RUN}-release'/'runtime.json')
activate(rt)
torch.set_num_threads(2)
source=Path(rt['source_root'])
contract=source_contract(source)
routes=('full','residual')
frozen={r:read_frozen(parent,r,MODES) for r in routes}
ids=frozen['full'][0]
assert ids==frozen['residual'][0]
raw,rx,all_ids,_=source_role(source,contract,'V')
index={key:i for i,key in enumerate(all_ids.tolist())}
chosen=np.array([index[key] for key in ids])
rows=[(raw[i],rx[i],all_ids[i]) for i in chosen]
limits=[read(parent/'anchored-complete.json')['threshold'],.15,.40]
labels=('bypass','active_to_0.15','active_0.15_to_0.40','active_above_0.40')
assert limits[0]<limits[1]<limits[2]
strata={}
tracker=PacketRestorer('consensus').eval()
cache=np.load(project/'runs'/DATA_RUN/'source_views'/'V.npy',mmap_mode='r',allow_pickle=False)
for route in routes:
    severity=[]
    with ProcessPoolExecutor(max_workers=6) as pool:
        for scene in VIEWS[1:]:
            if route=='full':
                wave=cache[VIEWS.index(scene),chosen]
            else:
                captured=pool.map(partial(receive_only,route,scene),rows,chunksize=10)
                wave=np.stack([result[0] for result in captured])
            chunks=[]
            for portion in tensor(wave,'cpu').split(32):
                base=coordinates(tracker,portion)[0]
                chunks.append(corruption_score(base).numpy())
            noise=np.concatenate(chunks)
            bucket=np.digitize(noise,limits,right=True)
            assert int((bucket>0).sum())==frozen[route][2][scene]['active'],(route,scene)
            severity.append(bucket)
    strata[route]=np.stack(severity)
with np.load(source/'V_metadata.npz',allow_pickle=False) as data:
    assert np.array_equal(data['ids'][chosen],np.array(ids))
    truth=data['y'][chosen]
assert len(ids)==900 and len(SOURCE)==5
output = {}
for route, (_, predictions, metadata) in frozen.items():
    decision = np.stack([predictions[v] for v in VIEWS[1:]])
    correct = decision == truth
    groups = strata[route]
    pairs = (('protected_engineering','engineering'), ('protected_engineering','direct'),
             ('protected_half','crossfit_half'), ('anchored_half','protected_half'),
             ('protected_half','reference'), ('anchored_half','reference'))
    for mode in ('engineering','protected_engineering','protected_half','anchored_half'):
        assert np.array_equal(decision[:,0][groups==0],decision[:,MODES.index(mode)][groups==0])
    def counts(mask):
        compared = {}
        for variant,control in pairs:
            a,b = (correct[:,MODES.index(name)] for name in (variant,control))
            compared[variant+'__vs__'+control] = dict(
                better=int((a & ~b & mask).sum()),
                worse=int((~a & b & mask).sum()),
                variant_correct=int((a & mask).sum()),
                control_correct=int((b & mask).sum()))
        return dict(exposures=int(mask.sum()),comparisons=compared)
    def by_strata(select):
        return {name:counts((groups==i) & select) for i,name in enumerate(labels)}
    output[route] = dict(pooled=by_strata(np.ones_like(groups,dtype=bool)),
                         by_scene={v:by_strata(np.arange(len(VIEWS)-1)[:,None]==k)
                                   for k,v in enumerate(VIEWS[1:])})
print(json.dumps(dict(parent=RUN,n_packets=len(ids),observed_noise_rms_thresholds=limits,
                      scopes=labels,uses_actual_added_noise=False,target_reads=0,by_route=output)),flush=True)
