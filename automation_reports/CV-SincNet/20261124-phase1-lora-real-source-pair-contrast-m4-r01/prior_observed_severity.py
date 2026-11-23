
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import json
import sys
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
from experiments.pamf_lora_main.runner import activate

use_pair = '--pair' in sys.argv
if use_pair:
    from experiments.pamf_lora_main.real_pair_source import MODES, RUN
    candidate_mode, training_file = 'pair_half', 'pair-complete.json'
else:
    from experiments.pamf_lora_main.packet_phase_source import MODES, RUN
    candidate_mode, training_file = 'trained_joint_half', 'packet_phase-complete.json'

project=Path('/home/szu2070436088/2510044040/CV-SincNet')
parent=project/'runs'/RUN
if read(parent/'state.json')['status']!='SOURCE_SCORED':
    raise ValueError('Need already-scored phase tracker source outputs')
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
limits=[read(parent/training_file)['threshold'],.15,.40]
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
output={}
for route, (_,predictions, _) in frozen.items():
    decision=np.stack([predictions[v] for v in VIEWS[1:]])
    direct,dura,restored=(decision[:,MODES.index(mode)]==truth
                          for mode in ('direct','reference',candidate_mode))
    point = decision[:,MODES.index('point_half')]==truth if use_pair else None
    group=strata[route]
    assert np.array_equal(decision[:,0][group==0],decision[:,MODES.index(candidate_mode)][group==0])
    def counts(mask):
        result=dict(exposures=int(mask.sum()),
                    direct_correct=int((direct&mask).sum()),
                    dura_correct=int((dura&mask).sum()),
                    candidate_correct=int((restored&mask).sum()),
                    correction_vs_dura=int((restored&~dura&mask).sum()),
                    harm_vs_dura=int((~restored&dura&mask).sum()),
                    both_correct=int((dura&restored&mask).sum()),
                    both_wrong=int((~dura&~restored&mask).sum()))
        if point is not None:
            result.update(point_correct=int((point&mask).sum()),
                          correction_vs_point=int((restored&~point&mask).sum()),
                          harm_vs_point=int((~restored&point&mask).sum()))
        return result
    def by_strata(index):
        return {name:counts((group==i) & index) for i,name in enumerate(labels)}
    output[route]=dict(pooled=by_strata(np.ones_like(group,dtype=bool)),
                       by_scene={v:by_strata(np.arange(len(VIEWS)-1)[:,None]==k)
                                 for k,v in enumerate(VIEWS[1:])})
result = dict(n_packets=len(ids), observed_noise_rms_thresholds=limits, scopes=labels,
              uses_actual_added_noise=False,target_reads=0,by_route=output)
result['evaluation' if use_pair else 'parent'] = RUN
print(json.dumps(result),flush=True)
