
import json
from pathlib import Path

import numpy as np

from experiments.cric_dura_comparison.io import read
from experiments.pamf_lora_main.chirp_spectrum_source import read_frozen
from experiments.pamf_lora_main.data import SOURCE,VIEWS
from experiments.pamf_lora_main.full_packet_views import source_contract
from experiments.pamf_lora_main.tail_pilot_source import MIRROR_MODES,MIRROR_RUN

base=Path('/home/szu2070436088/2510044040/CV-SincNet')
root=base/'runs'/MIRROR_RUN
assert read(root/'state.json')['status']=='SOURCE_SCORED'
rt=read(base/'releases'/f'{MIRROR_RUN}-release'/'runtime.json')
source=Path(rt['source_root'])
frozen={route:read_frozen(root,route,MIRROR_MODES) for route in ('full','residual')}
ids=frozen['full'][0]
assert frozen['residual'][0]==ids
positions={name:i for i,name in enumerate(source_contract(source)['role_ids']['V'])}
chosen=np.array([positions[i] for i in ids])
with np.load(source/'V_metadata.npz',allow_pickle=False) as data:
    assert np.array_equal(data['ids'][chosen],np.array(ids))
    truth=data['y'][chosen]
arms=('adjacent_coarse','mirror_coarse','adjacent','mirror')
report={}
for route,(_,predictions,_) in frozen.items():
    def analyze(selected):
        x=np.stack([predictions[key] for key in selected])
        a,b,c,d=(x[:,MIRROR_MODES.index(name),:]==truth for name in arms)
        initial=b.astype(np.int8)-a
        final=d.astype(np.int8)-c
        sums={}
        for value,key in ((1,'mirror_only_correct'),(-1,'adjacent_only_correct'),(0,'coarse_equal')):
            mask=initial==value
            sums[key]=dict(coarse_count=int(mask.sum()),
                           adj_only_final=int((mask & c & ~d).sum()),
                           mirror_only_final=int((mask & d & ~c).sum()),
                           both_final_correct=int((mask & c & d).sum()),
                           both_final_wrong=int((mask & ~c & ~d).sum()),
                           swing=int((final[mask]-initial[mask]).sum()))
        return dict(net_coarse=int(initial.sum()),net_fine=int(final.sum()),strata=sums)
    report[route]=dict(total=analyze(VIEWS[1:]),by_scene={v:analyze((v,)) for v in VIEWS[1:]})
print(json.dumps(dict(parent=MIRROR_RUN,traces_read=0,target_reads=0,n_packets=len(ids),
                      evidence='Per-packet transitions between previously frozen, source-scored checkpoints',
                      routes=report)))
