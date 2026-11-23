import json
from pathlib import Path
import numpy as np
from experiments.pamf_lora_main.chirp_spectrum_source import read_frozen
from experiments.pamf_lora_main.data import VIEWS
from experiments.pamf_lora_main.spectral_covariance_source import MODES, RUN

root=Path('/home/szu2070436088/2510044040/CV-SincNet/runs')/RUN
assert json.loads((root/'state.json').read_text())['status']=='SOURCE_SCORED'
i=MODES.index('rank_half')
j=MODES.index('rank_diagonal_half')
results={}
for route in ('full','residual'):
    ids, rows, metrics=read_frozen(root,route,MODES)
    results[route]={k:int(np.count_nonzero(matrix[i]!=matrix[j])) for k,matrix in rows.items()}
    assert len(ids)==900 and results[route]['clean']==0
print(json.dumps(dict(packets=900,targets_read=0,F_reads=0,truth_reads=0,changed_predictions=results)),flush=True)
