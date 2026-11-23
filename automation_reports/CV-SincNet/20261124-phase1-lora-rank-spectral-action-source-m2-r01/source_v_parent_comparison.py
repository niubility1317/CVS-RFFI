import json
from pathlib import Path
import numpy as np
from experiments.pamf_lora_main.spectral_action_source import MODES,RUN
from experiments.pamf_lora_main.action_support_source import MODES as OLD_MODES,RUN as PREDECESSOR
from experiments.pamf_lora_main.chirp_spectrum_source import read_frozen
from experiments.pamf_lora_main.data import VIEWS

top=Path('/home/szu2070436088/2510044040/CV-SincNet/runs/20261124-phase1-lora-rank-spectral-action-source-m2-r01').parent
assert json.loads((top/RUN/'state.json').read_text())['status']=='SOURCE_SCORED'
comp={}
for route in ('full','residual'):
    ids,pred,_=read_frozen(top/RUN,route,MODES)
    prior_ids,baseline,_=read_frozen(top/PREDECESSOR,route,OLD_MODES)
    assert ids==prior_ids
    total={}
    for view in VIEWS:
        stable={mode:int((pred[view][i]!=baseline[view][OLD_MODES.index(mode)]).sum())
                for i,mode in enumerate(MODES[:3])}
        changes={mode:int(np.count_nonzero(pred[view][i]!=pred[view][2])) for i,mode in enumerate(MODES[3:],3)}
        total[view]=dict(parent_mismatches=stable,changed_vs_rank=changes)
    comp[route]=total
print(json.dumps(dict(packets=900,source_only=True,truth_reads=0,F_reads=0,target_reads=0,by_route=comp)),flush=True)
