"""Paired amplitude-by-detail interaction in the completed LoRa source factorial."""
import json
from pathlib import Path

import numpy as np

from experiments.cric_dura_comparison.io import read
from experiments.pamf_lora_main.carrier_factorial_source import ARM_NAMES, LEVELS, MODES
from experiments.pamf_lora_main.chirp_spectrum_source import read_frozen
from experiments.pamf_lora_main.data import SOURCE, VIEWS
from experiments.pamf_lora_main.full_packet_views import source_contract


def main():
    runtime = read(Path('runtime.json'))
    root = Path(runtime['run_root'])
    source = Path(runtime['source_root'])
    if read(root/'state.json')['status']!='SOURCE_SCORED':
        raise ValueError('Carrier attribution requires completed source scores')
    frozen = {r:read_frozen(root,r,MODES) for r in ('full','residual')}
    ids = frozen['full'][0]
    if ids != frozen['residual'][0]:
        raise ValueError('Carrier interaction requires identical route packets')
    book={v:i for i,v in enumerate(source_contract(source)['role_ids']['V'])}
    chosen = [book[x] for x in ids]
    with np.load(source/'V_metadata.npz',allow_pickle=False) as m:
        if m['ids'][chosen].tolist()!=ids:
            raise ValueError('Incorrect source truth order')
        labels = m['y'][chosen]
    rng = np.random.default_rng(2026113797)
    per_rx=len(ids)//len(SOURCE)
    sample=np.concatenate([rng.integers(j*per_rx,(j+1)*per_rx,(4000,per_rx))
                           for j in range(len(SOURCE))],axis=-1)
    contrast=np.asarray([1 if a==d else -1 for _,a,d in LEVELS],dtype=float)[:,None,None]*.5
    results={}
    for route,(_,predicted,_) in frozen.items():
        slices = [predicted[key][[MODES.index(a) for a in ARM_NAMES]] == labels for key in VIEWS[1:]]
        hits = np.stack(slices,axis=1)
        interaction=(hits*contrast).sum(axis=0)
        results[route]={
            name:dict(net=float(v.sum()),interval=np.quantile(v[sample].sum(1),(.025,.975)).tolist())
            for name,v in (('six',interaction.sum(0)),('low',interaction[-1]))
        }
    print(json.dumps(dict(status='VERIFIED',target_reads=0,uses_completed_source_truth=True,
                          bootstrap_draws=4000,amplitude_detail_interaction=results)))


if __name__=='__main__':
    main()
