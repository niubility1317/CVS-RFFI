"""Same-physical-row contrasts against the frozen preexisting flat and integer arms."""
import json
from pathlib import Path
import sys

root=Path(sys.argv[1])
release=root.parent.parent/'releases'/(root.name+'-release')
sys.path.insert(0,str(release))

import numpy as np

from experiments.cric_dura_comparison.io import read
from experiments.pamf_lora_main.chirp_spectrum_source import read_frozen
from experiments.pamf_lora_main.data import SOURCE, VIEWS
from experiments.pamf_lora_main.denoise_source import EVAL_PER_RX, KEY
from experiments.pamf_lora_main.full_packet_views import source_contract
from experiments.pamf_lora_main.rf_noise_transport_source import CONFIDENCE_MODES, MODES, CONFIDENCE_RUN, RUN

assert root.name==CONFIDENCE_RUN
prior=root.parent/RUN
assert all(read(p/'state.json')['status']=='SOURCE_SCORED' for p in (root,prior))
rt=read(release/'runtime.json')
paths=[(root,CONFIDENCE_MODES),(prior,MODES)]
frozen={route:[read_frozen(p,route,modes) for p,modes in paths] for route in ('full','residual')}
ids=frozen['full'][0][0]
for pair in frozen.values():
    for other,_,_ in pair:
        assert other==ids
    _,a,_=pair[0]
    _,b,_=pair[1]
    assert all(np.array_equal(a[v][:2],b[v][:2]) for v in VIEWS)
source=Path(rt['source_root'])
contract=source_contract(source)
positions={identity:i for i,identity in enumerate(contract['role_ids']['V'])}
ix=[positions[identity] for identity in ids]
with np.load(source/'V_metadata.npz',allow_pickle=False) as z:
    assert z['ids'][ix].tolist()==ids
    truth=z['y'][ix]
    assert np.array_equal(z['rx_i'][ix],np.repeat(np.arange(len(SOURCE)),EVAL_PER_RX))
rng=np.random.default_rng(KEY+3)
draw=np.concatenate([rng.integers(i*EVAL_PER_RX,(i+1)*EVAL_PER_RX,(1600,EVAL_PER_RX))
                     for i in range(len(SOURCE))],axis=-1)
report={}
for route,pair in frozen.items():
    letter={}
    for arm,prior_arm in [('kernel','transport'),('confidence','flat'),
                          ('confidence_half','flat_half'),('confidence','transport'),
                          ('confidence_half','transport_half')]:
        history=pair[1][1]
        current=pair[0][1]
        x=CONFIDENCE_MODES.index(arm)
        y=MODES.index(prior_arm)
        differences=[(current[v][x]==truth).astype(int) - (history[v][y]==truth).astype(int)
                     for v in VIEWS[1:]]
        six=np.sum(differences,axis=0)
        values={'six':six,'low':differences[-1]}
        letter[arm+'_vs_'+prior_arm]={kind:dict(net=int(diff.sum()),
                                                interval=np.quantile(diff[draw].sum(1),(.025,.975)).tolist())
                                      for kind,diff in values.items()}
    report[route]=letter
print(json.dumps(dict(status='VERIFIED',physical_count=len(ids), routes=report,
                      independent_truth_last=True,target_reads=0),indent=2))
