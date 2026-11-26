"""Truth-last source coverage for the exact bank in the waveform dose audit."""
import json
from pathlib import Path

import numpy as np

project=Path.cwd()
name='20261126-diagnostic-lora-w24-waveform-dose-source-s2026112559-r01'
root=project/'runs'/name
folder=root/'full_native'
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
assert read(root/'state.json')['status']=='SOURCE_SCORED'
meta=read(folder/'source-predictions.json')
scores=read(folder/'source-scores.json')
assert meta['status']=='PREDICTIONS_FROZEN' and meta['source_truth_reads']==meta['target_reads']==0
assert scores['status']=='ANALYZED' and scores['source_truth_last'] and scores['target_reads']==0
modes=('direct','outer_half','standard_half','standard_full','complex_half','complex_full',
       'minimum_mse','minimum_balanced','continuous_mse')
assert meta['modes']==list(modes)
with np.load(folder/'predictions.npz',allow_pickle=False) as f:
    assert set(f.files)==set(meta['views'])
    predictions=np.stack([f[v] for v in meta['views']])
assert predictions.shape==(7,9,300)
source=Path(read(project/'releases'/f'{name}-release'/'runtime.json')['source_root'])
contract=read(source/'contract.json')
assert contract['source_only'] and contract['protocol']=='lora_full_packets_v1'
with np.load(source/'V_metadata.npz',allow_pickle=False) as v:
    ids=v['ids'].tolist()
    assert ids==contract['role_ids']['V'] and len(set(meta['physical_ids']))==300
    locations={x:i for i,x in enumerate(ids)}
    truth=v['y'][[locations[x] for x in meta['physical_ids']]]
hits=predictions==truth
assert all(int(hits[1:,i].sum())==scores['scores'][name]['six_correct'] for i,name in enumerate(modes))

def union(columns):
    a=hits[:,columns].any(1)
    return dict(n_routes=len(columns),six=int(a[1:].sum()),low=int(a[-1].sum()),clean=int(a[0].sum()))

start=max((root/'predict-source.log').stat().st_mtime-(root/'smoke.json').stat().st_mtime,0)
print(json.dumps(dict(status='ANALYZED',source_truth_last=True,target_reads=0,source_packets=300,
      scan_seconds_after_smoke=start,half_only=union([0,1,2,4]),fixed_bank=union(range(6)),
      all_diagnostics=union(range(9))),indent=2))
