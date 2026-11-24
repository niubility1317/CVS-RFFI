from pathlib import Path
import json
import numpy as np

from experiments.pamf_lora_main.chirp_spectrum_source import read_frozen
from experiments.pamf_lora_main.data import SOURCE,VIEWS
from experiments.pamf_lora_main.full_packet_views import source_contract
from experiments.pamf_lora_main.cross_chirp_hankel_source import MODES,RUN
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
mapping=(('direct','direct'),('reference','reference'),('rank_half','rank_half'))
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

print(json.dumps(dict(status='PASS',n=len(pred['full'][0]),target_reads=0,F_reads=0,TX_reads=0,
                      endpoint_mismatches=sanity)),flush=True)
