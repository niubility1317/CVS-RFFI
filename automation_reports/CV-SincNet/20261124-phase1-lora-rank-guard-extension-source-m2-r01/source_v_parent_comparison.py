import json
from pathlib import Path

import numpy as np
from experiments.pamf_lora_main.chirp_spectrum_source import read_frozen
from experiments.pamf_lora_main.data import VIEWS
from experiments.pamf_lora_main.spectral_covariance_source import MODES as OLD_MODES
from experiments.pamf_lora_main.weak_guard_source import MODES, RANK_RUN, RUN

base=Path('/home/szu2070436088/2510044040/CV-SincNet/runs/20261124-phase1-lora-rank-guard-extension-source-m2-r01').parent
assert json.loads((base/RUN/'state.json').read_text())['status']=='SOURCE_SCORED'
out={}
for route in ('full','residual'):
    ids, outputs, metrics=read_frozen(base/RUN,route,MODES)
    parent_ids, previous, pmetrics=read_frozen(base/RANK_RUN,route,OLD_MODES)
    if ids!=parent_ids:
        raise ValueError('Different source V packet IDs across the gate extension and parent')
    comparison={}
    for view in VIEWS:
        current,old=outputs[view],previous[view]
        unchanged={k:int(np.count_nonzero(current[MODES.index(k)]!=old[OLD_MODES.index(p)]))
                   for k,p in (('direct','direct'),('reference','reference'),('rank_old','rank_half'))}
        edits={k:int(np.count_nonzero(current[MODES.index(k)]!=current[MODES.index('rank_old')]))
               for k in MODES[3:]}
        extra=metrics[view]['active']-pmetrics[view]['active']
        assert extra>=0
        assert all(n<=extra for n in edits.values()),(route,view,edits,extra)
        comparison[view]=dict(old_active=pmetrics[view]['active'],additional_trigger=extra,
                              parent_top1_mismatches=unchanged,changed_vs_own_old=edits)
    out[route]=comparison
print(json.dumps(dict(packets=900,F_reads=0,target_reads=0,truth_reads=0,by_route=out)),flush=True)
