"""Source-only complementary error audit. Oracle union is diagnostic, not a restore."""
import json
from pathlib import Path

import numpy as np

PROJECT = Path.cwd()
RUNS = PROJECT/'runs'
KEY = '20261126-phase1-lora-w24-'
SUFFIX = '-s2026112991-'
REFERENCE = 'packet-unet-fidelity'
READOUTS = {
    REFERENCE: ('r01', {
        'direct':'direct', 'old_half':'all_half', 'outer':'outer_parent_half',
        'unet_standard_half':'unet_standard_long_half','unet_complex_half':'unet_complex_long_half',
        'unet_standard_full':'unet_standard_long_full','unet_complex_full':'unet_complex_long_full'}),
    'native-innovation': ('r01', {
        'local_aperture_half':'aperture_half','local_native_half':'native_half',
        'local_aperture_full':'aperture_full','local_native_full':'native_full'}),
    'packet-noise-calibrated': ('r03', {
        'calibrated_aperture_half':'aperture_half','calibrated_native_half':'native_half',
        'calibrated_aperture_full':'aperture_full','calibrated_native_full':'native_full'}),
    'packet-flow': ('r01', {
        'flow_one_half':'one_half','flow_four_half':'four_half',
        'flow_one_full':'one_full','flow_four_full':'four_full'}),
}
RXS = ('b200_1','b200_mini_1','b210_1','pluto_1','rtl_1')

def run(key, suffix):
    return KEY+key+SUFFIX+suffix

def read(p):
    return json.loads(p.read_text(encoding='utf-8'))

blinded = {}
controls = None
same_ids = None
same_views = None
for name,(suffix,modes) in READOUTS.items():
    base=RUNS/run(name,suffix)
    p=base/'full_native'
    statement=read(p/'source-predictions.json')
    score=read(p/'source-scores.json')
    assert read(base/'state.json')['status']=='SOURCE_SCORED'
    assert statement['status']=='PREDICTIONS_FROZEN'
    assert statement['source_truth_reads']==statement['target_reads']==0
    assert score['status']=='ANALYZED' and score['source_truth_last'] and not score['target_reads']
    assert len(statement['physical_ids'])==len(set(statement['physical_ids']))==score['n_packets']==300
    assert len(statement['views'])==7 and statement['views'][0]=='clean'
    if same_ids is None:
        same_ids,same_views=statement['physical_ids'],statement['views']
    assert same_ids==statement['physical_ids'] and same_views==statement['views']
    registered=statement['modes']
    with np.load(p/'predictions.npz',allow_pickle=False) as frozen:
        assert set(frozen.files)==set(same_views)
        matrix=np.stack([frozen[view] for view in same_views])
        assert matrix.shape==(7,len(registered),300) and matrix.dtype.kind in 'iu'
        direct=matrix[:,registered.index('direct')]
        if controls is None:
            controls=direct
        assert np.array_equal(direct,controls)
        for label,mode in modes.items():
            blinded[label]=matrix[:,registered.index(mode)]
    for label,mode in modes.items():
        assert mode in score['scores']

ref=run(REFERENCE,'r01')
rt=read(PROJECT/'releases'/f'{ref}-release'/'runtime.json')
source=Path(rt['source_root'])
contract=read(source/'contract.json')
assert contract['source_only'] and contract['protocol']=='lora_full_packets_v1'
with np.load(source/'V_metadata.npz',allow_pickle=False) as source_v:
    ordered=source_v['ids'].tolist()
    assert ordered==contract['role_ids']['V']
    indices={s:i for i,s in enumerate(ordered)}
    rows=[indices[i] for i in same_ids]
    truth=source_v['y'][rows]
    rx=np.asarray(source_v['rx_i'])[rows]

hits={k:v==truth for k,v in blinded.items()}
base=hits['outer']
halves={k:v for k,v in hits.items() if not k.endswith('_full')}

def summarize(rows):
    union=np.stack(tuple(rows.values())).any(0)
    gains=(union & ~base)
    assert np.all(union>=base)
    return dict(n_routes=len(rows),correct_by_scene=dict(zip(same_views,union.sum(1).tolist(),strict=True)),
                six_correct=int(union[1:].sum()),low_urban_correct=int(union[-1].sum()),
                low_gain_by_rx={n:int(gains[-1,rx==i].sum()) for i,n in enumerate(RXS)})

paired={}
for k,v in halves.items():
    corr=v & ~base
    harm=base & ~v
    paired[k]=dict(six_gain=int(corr[1:].sum()),six_harm=int(harm[1:].sum()),
                   low_gain=int(corr[-1].sum()),low_harm=int(harm[-1].sum()))
print(json.dumps(dict(status='ANALYZED',n_source=300,n_practical=1800,target_reads=0,source_truth_last=True,
      scope='oracle union of named frozen archive decisions only',baseline={'six':int(base[1:].sum()),'low':int(base[-1].sum())},
      frozen_runs=[run(k,v[0]) for k,v in READOUTS.items()],half_and_direct=summarize(halves),
      including_full=summarize(hits),pair_with_outer=paired),indent=2))
