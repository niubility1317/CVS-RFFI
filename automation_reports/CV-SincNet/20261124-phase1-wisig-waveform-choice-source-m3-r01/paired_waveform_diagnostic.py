"""Compare paired clean waveform routing and label routing after the WiSig source run freezes."""
import json
from pathlib import Path

import numpy as np
import torch

from experiments.cric_dura_comparison.io import read
from experiments.pamf_repl_view_shuffle import observable_action_source as action
from experiments.pamf_repl_view_shuffle import receiver_pilot_source as panel
from experiments.pamf_repl_view_shuffle.waveform_choice import PROPOSALS, waveform_risk
from experiments.pamf_repl_view_shuffle.waveform_choice_source import MODES, RUN, model_at

rt = read(Path('runtime.json'))
assert rt['run_id'] == RUN
assert read(Path(rt['run_root'])/'state.json')['status']=='SOURCE_SCORED'
frozen = {route: panel.load_predictions(rt, route, ('raw', 'R69', *MODES)) for route in panel.ROUTES}
assert np.array_equal(frozen['full'][0],frozen['residual'][0])
model = model_at(rt)

routed = {}
with torch.no_grad():
    for route, source in panel.panels(rt).items():
        ids, predicted, _ = frozen[route]
        assert np.array_equal(ids, np.load(source/'physical_ids.npy',allow_pickle=False))
        clean = panel.iq_array(source/'clean.npy',panel.N)
        summaries={}
        for view in panel.VIEWS:
            observed = panel.iq_array(source/f'{view}.npy',panel.N)
            choices=[]
            cost=[]
            for start in range(0,panel.N,128):
                address = slice(start,start+128)
                candidate, _ = model(action.received(observed[address],'cuda:0'))
                # The network has already produced the distinct proposals at
                # this point. Use paired clean only to order those fixed waves.
                risk = waveform_risk(candidate,action.received(clean[address],'cuda:0'))
                assert torch.isfinite(risk).all()
                choices.extend(risk.argmin(1).tolist())
                cost.extend(risk.min(1).values.tolist())
            matrix=np.stack([predicted[f'{mode}__{view}'] for mode in PROPOSALS])
            selected=np.asarray(choices,dtype=np.int64)
            summaries[view]=(matrix[selected,np.arange(panel.N)],matrix,selected,float(np.mean(cost)))
        routed[route] = summaries

# Truth is only used after both routes' fixed proposals and waveform selections.
with np.load(panel.source_root(rt)/'V_metadata.npz',allow_pickle=False) as source:
    locate={key:i for i,key in enumerate(source['ids'])}
    at=[locate[key] for key in frozen['full'][0]]
    y,rx,day=(source[k][at] for k in ('y','rx_i','day_i'))
rng=np.random.default_rng(2026112431)
strata=[np.flatnonzero((rx==r)&(day==d)) for r in np.unique(rx) for d in np.unique(day)]
draws=np.concatenate([rng.choice(ix,size=(2000,len(ix))) for ix in strata if len(ix)],axis=1)
report={}
for route, views in routed.items():
    _, decisions, _ = frozen[route]
    hit_counts, proposal_counts, waveform_cost = {}, {}, {}
    for view,(best_iq,bank,index,cost) in views.items():
        candidates = dict(waveform_choice=best_iq==y, label_coverage=(bank==y).any(0),
                          **{k:decisions[f'{k}__{view}']==y for k in ('raw','R69','static','routed','routed_guarded')})
        hit_counts[view]=candidates
        proposal_counts[view]=dict(zip(PROPOSALS,np.bincount(index,minlength=len(PROPOSALS)).tolist()))
        waveform_cost[view]=cost
    keys=hit_counts['clean']
    six={k:np.stack([hit_counts[v][k] for v in panel.VIEWS[1:]]).sum(0) for k in keys}
    def delta_pair(hit, ref):
        d=hit-ref
        return dict(net=int(d.sum()),ci=np.quantile(d[draws].sum(1),(.025,.975)).tolist())
    report[route]=dict(six={k:int(v.sum()) for k,v in six.items()},
        waveform_vs_raw=delta_pair(six['waveform_choice'],six['raw']),
        waveform_vs_R69=delta_pair(six['waveform_choice'],six['R69']),
        per_view={v:{k:int(b.sum()) for k,b in h.items()} for v,h in hit_counts.items()},
        waveform_choice_counts=proposal_counts, mean_selected_waveform_risk=waveform_cost)
print(json.dumps(dict(target_reads=0, n=panel.N, diagnostic_only=True, routes=report),
                 indent=2, allow_nan=False))
