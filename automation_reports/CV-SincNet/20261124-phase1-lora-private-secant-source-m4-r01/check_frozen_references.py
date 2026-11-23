
from pathlib import Path
import json
import numpy as np
from experiments.pamf_lora_main.private_secant_source import MODES,RUN
from experiments.pamf_lora_main.spectral_action_source import MODES as OLD_MODES,RUN as PREDECESSOR
from experiments.pamf_lora_main.chirp_spectrum_source import read_frozen
from experiments.pamf_lora_main.data import VIEWS

top = Path('/home/szu2070436088/2510044040/CV-SincNet/runs')
assert json.loads((top/RUN/'state.json').read_text())['status']=='SOURCE_SCORED'
compare={}
for route in ('full','residual'):
    ids,pred,_=read_frozen(top/RUN,route,MODES)
    prior_ids,prior,_=read_frozen(top/PREDECESSOR,route,OLD_MODES)
    if ids!=prior_ids:
        raise ValueError('New and parent panels differ in physical packet coverage')
    compare[route]={view:{name:int((pred[view][MODES.index(name)]!=prior[view][OLD_MODES.index(name)]).sum())
                         for name in ('direct','reference')} for view in VIEWS}
print(json.dumps(dict(source_only=True,n=len(ids),target_reads=0,truth_reads=0,compare=compare)),flush=True)
