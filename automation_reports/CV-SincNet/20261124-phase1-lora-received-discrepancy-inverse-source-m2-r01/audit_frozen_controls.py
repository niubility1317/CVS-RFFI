import json
from pathlib import Path
import numpy as np
from experiments.cric_dura_comparison.io import read
from experiments.pamf_lora_main.chirp_spectrum_source import read_frozen
from experiments.pamf_lora_main.data import VIEWS
from experiments.pamf_lora_main.discrepancy_inverse_source import MODES,PER_RX
from experiments.pamf_lora_main.learned_inverse_source import MODES as PARENT_MODES,RUN as PARENT
root=Path(read(Path('runtime.json'))['run_root'])
if read(root/'state.json')['status']!='SOURCE_SCORED':
    raise ValueError('Need completed independent scoring')
pairs=(('direct','direct'),('reference','reference'),('exact80_direct','known_80'))
report={}
for route in ('full','residual'):
    ids, predictions, _=read_frozen(root,route,MODES,per_rx=PER_RX)
    baseline=read(root.parent/PARENT/f'{route}-predictions.json')
    assert len(ids)==300 and ids==baseline['ids']
    report[route]={view:{k:int((predictions[view][MODES.index(k)]!=np.asarray(baseline['rows'][view][PARENT_MODES.index(j)])).sum())
                        for k,j in pairs} for view in VIEWS}
assert not any(n for r in report.values() for v in r.values() for n in v.values())
print(json.dumps(dict(status='PASS',n=300,target_reads=0,TX_reads=0,mismatches=report)))
