import json
from pathlib import Path
import numpy as np
from experiments.pamf_lora_main.action_support_source import MODES,RANK_RUN,RUN
from experiments.pamf_lora_main.chirp_spectrum_source import read_frozen
from experiments.pamf_lora_main.data import VIEWS
from experiments.pamf_lora_main.spectral_covariance_source import MODES as RANK_MODES
from experiments.pamf_lora_main.weak_guard_source import MODES as GUARD_MODES,RUN as GUARD_RUN

base=Path('/home/szu2070436088/2510044040/CV-SincNet/runs/20261124-phase1-lora-rank-split-action-source-m2-r01').parent
assert json.loads((base/RUN/'state.json').read_text())['status']=='SOURCE_SCORED'
results={}
for route in ('full','residual'):
    ids,pred,active=read_frozen(base/RUN,route,MODES)
    ancestor,rank,_=read_frozen(base/RANK_RUN,route,RANK_MODES)
    initial,guard,_=read_frozen(base/GUARD_RUN,route,GUARD_MODES)
    assert ids==ancestor==initial
    rows={}
    for view in VIEWS:
        sample=pred[view]
        checks={'direct':(rank[view],'direct',RANK_MODES),'reference':(rank[view],'reference',RANK_MODES),
                'rank_old':(rank[view],'rank_half',RANK_MODES),
                'open_quarter':(guard[view],'guard_quarter',GUARD_MODES),'open_half':(guard[view],'guard_half',GUARD_MODES)}
        rows[view]={k:int(np.count_nonzero(sample[MODES.index(k)]!=p[methods.index(alias)]))
                    for k,(p,alias,methods) in checks.items()}
    results[route]=rows
print(json.dumps(dict(packets=900,truth_reads=0,target_reads=0,F_reads=0,by_route=results)),flush=True)
