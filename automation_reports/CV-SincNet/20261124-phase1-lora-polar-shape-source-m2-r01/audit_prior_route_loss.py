"""Trace remaining LoRa source errors through already scored readouts."""
import json
from pathlib import Path

import numpy as np

from experiments.cric_dura_comparison.io import read
from experiments.pamf_lora_main.chirp_spectrum_source import read_frozen
from experiments.pamf_lora_main.common_failure_source import RUN as ORACLE, MODES as ORACLE_MODES
from experiments.pamf_lora_main.data import VIEWS
from experiments.pamf_lora_main.full_packet_views import source_contract
from experiments.pamf_lora_main.independent_shape_source import RUN as CANDIDATE, MODES as CANDIDATE_MODES


def main():
    rt=read(Path('runtime.json'))
    source=Path(rt['source_root'])
    runs=Path(rt['run_root']).parent
    croot,oroot=runs/CANDIDATE,runs/ORACLE
    if any(read(r/'state.json')['status']!='SOURCE_SCORED' for r in (croot,oroot)):
        raise ValueError('Source predictions must already be complete')
    c={r:read_frozen(croot,r,CANDIDATE_MODES) for r in ('full','residual')}
    o={r:read_frozen(oroot,r,ORACLE_MODES) for r in c}
    ids=c['full'][0]
    if any(x[0]!=ids for x in (*c.values(),*o.values())):
        raise ValueError('Readouts must retain the same source physical IDs and order')
    for r in c:
        for s in VIEWS:
            for k in ('direct','reference'):
                if not np.array_equal(c[r][1][s][CANDIDATE_MODES.index(k)],o[r][1][s][ORACLE_MODES.index(k)]):
                    raise ValueError('Frozen data or classifier mismatch')
    lookup={v:i for i,v in enumerate(source_contract(source)['role_ids']['V'])}
    chosen=[lookup[i] for i in ids]
    with np.load(source/'V_metadata.npz',allow_pickle=False) as labels:
        if labels['ids'][chosen].tolist()!=ids:
            raise ValueError('Source label order mismatch')
        actual=labels['y'][chosen]
    results={}
    for r in c:
        results[r]={}
        for s in VIEWS[1:]:
            current={k:c[r][1][s][CANDIDATE_MODES.index(k)]==actual
                     for k in ('direct','reference','own_shape_half','rank_parent_half')}
            known={k:o[r][1][s][ORACLE_MODES.index(k)]==actual for k in ('noise80','noise100','noise80_action','noise100_action')}
            baseline=current['reference']; candidate=current['own_shape_half']; direct=current['direct']
            missed=baseline & ~candidate
            extra=candidate & ~baseline
            results[r][s]=dict(dura_only=int(missed.sum()),own_only=int(extra.sum()),
              dura_only_raw_correct=int((missed&direct).sum()),own_only_raw_correct=int((extra&direct).sum()),
              candidate_damage_vs_raw=int((direct&~candidate).sum()),dura_damage_vs_raw=int((direct&~baseline).sum()),
              candidate_repair_vs_raw=int((~direct&candidate).sum()),dura_repair_vs_raw=int((~direct&baseline).sum()),
              dura_only_diagnostic={k:int((missed&v).sum()) for k,v in known.items()},
              known_correct={k:int(v.sum()) for k,v in known.items()})
    print(json.dumps(dict(status='VERIFIED',n=len(ids),target_reads=0,truth_after_predictions=True,
                          sources=(CANDIDATE,ORACLE),results=results)))


if __name__=='__main__':
    main()
