
import json
from pathlib import Path
import numpy as np
from experiments.pamf_lora_main.private_transport_source import MODES,RUN
from experiments.pamf_lora_main.spectral_action_source import MODES as OLD_MODES, RUN as PREDECESSOR
from experiments.pamf_lora_main.chirp_spectrum_source import read_frozen
from experiments.pamf_lora_main.data import VIEWS

top=Path('/home/szu2070436088/2510044040/CV-SincNet/runs')
assert json.loads((top/RUN/'state.json').read_text())['status']=='SOURCE_SCORED'
result={}
for route in ('full','residual'):
    ids,rows,_=read_frozen(top/RUN,route,MODES)
    keys,prior,_=read_frozen(top/PREDECESSOR,route,OLD_MODES)
    if ids!=keys:
        raise ValueError('Physical packet mismatch')
    result[route]={view:{new:int((rows[view][MODES.index(new)]!=prior[view][OLD_MODES.index(old)]).sum())
        for new,old in (('direct','direct'),('reference','reference'),('rank_half','rank_old'))} for view in VIEWS}
print(json.dumps(dict(source_only=True,n=900,truth_reads=0,F_reads=0,target_reads=0,by_route=result)),flush=True)
