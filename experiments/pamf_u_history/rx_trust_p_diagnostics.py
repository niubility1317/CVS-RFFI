"""Counterfactual singleton/partial H targets from saved source V predictions."""
import inspect
import itertools
import json
from pathlib import Path
import statistics
import numpy as np
from .source_confidence_metrics import probabilities
from .rx_mean_source import RUN,METHODS,SEEDS
from .rx_mean_reference_diagnostics import SCENES


def partial_statistics(raw,H,y,*,role):
    if role!='V':raise ValueError('Visible source V proxy only; no U TX truth')
    p=probabilities(raw);q=probabilities(H);y=np.asarray(y);n=len(p)
    if q.shape!=p.shape or y.shape!=(n,) or y.dtype.kind not in 'iu' or y.min()<0 or y.max()>=6:raise ValueError('Aligned six-class source predictions/labels required')
    anchor=p.argmax(1);other=q.argmax(1);accepted=p.max(1)>=.95;mask=np.eye(6,dtype=bool)[anchor]|np.eye(6,dtype=bool)[other]
    count=int(accepted.sum());single=accepted&(anchor==other);pair=accepted&(anchor!=other);safe=mask[np.arange(n),y]
    mass=(q*mask).sum(1);conditional=q*mask/mass[:,None];hard=q-np.eye(6)[anchor];partial=q-conditional
    return dict(count=n,accepted=count,singleton=int(single.sum()),pair=int(pair.sum()),candidate_contains_truth=int((accepted&safe).sum()),
        raw_correct=int((accepted&(anchor==y)).sum()),coverage_pct=100*accepted.mean(),
        raw_precision_pct=100*(anchor[accepted]==y[accepted]).mean() if count else None,
        candidate_inclusion_pct=100*safe[accepted].mean() if count else None,
        partial_logit_gradient_norm_mean=float(np.linalg.norm(partial[accepted],axis=1).mean()) if count else None,
        hard_raw_label_logit_gradient_norm_mean=float(np.linalg.norm(hard[accepted],axis=1).mean()) if count else None,
        partial_H_loss_mean=float(-np.log(mass[accepted].clip(1e-300)).mean()) if count else None,
        hard_H_loss_mean=float(-np.log(q[np.arange(n),anchor][accepted].clip(1e-300)).mean()) if count else None)


def collect(runtime,output):
    from experiments.cric_dura_comparison.publish import ssh
    rt=json.loads(Path(runtime).read_text(encoding='utf-8'))
    if rt['run_id']!=RUN:raise ValueError('Completed six-model source artifacts only')
    out=Path(output)
    if out.exists():raise FileExistsError('Keep existing source candidate diagnostic')
    script='import numpy as np\n'+inspect.getsource(probabilities)+'\n'+inspect.getsource(partial_statistics)+r'''
import json
from pathlib import Path
rt=RUNTIME;root=Path(rt['run_root']);methods=METHODS;seeds=SEEDS;scenes=SCENES
d=json.loads((root/'complete.json').read_text())
if d['status']!='RX_MEAN_THREE_SEED_COMPLETE' or d['models']!=6:raise ValueError('Complete original six-model matrix required')
rows=[];metadata=None
for method in methods:
 for seed in seeds:
  folder=root/'F'/('F0-'+method+'-s'+str(seed));source=json.loads((folder/'source/complete.json').read_text());v=json.loads((folder/'source_validation/complete.json').read_text())
  if (source['status']!='SOURCE_FROZEN' or source['epoch']!=50 or source['updates']!=11100 or source['config']['paper_method']!=method or source['config']['model_seed']!=seed
   or source['target_access'] or source['U_truth_read'] or v['status']!='MULTISEED_SOURCE_EVALUATED' or v['method']!=method or v['model_seed']!=seed or v['V_seen']!=27000 or v['G_reads'] or v['target_reads']):raise ValueError('Matched fixed source-only E50 outputs required')
  with np.load(folder/'source_validation/source_V_predictions.npz',allow_pickle=False) as a:
   if a['logits'].shape!=(4,27000,6) or tuple(a['views'])!=('raw',*scenes):raise ValueError('Original same-packet raw and H views required')
   current={k:a[k].copy() for k in ('ids','y','rx','day')}
   if any(x.shape!=(27000,) for x in current.values()) or len(np.unique(current['ids']))!=27000:raise ValueError('Unique complete V population required')
   if metadata is None:metadata=current
   elif any(not np.array_equal(x,metadata[k]) for k,x in current.items()):raise ValueError('Physical source pairing differs')
   for j,scene in enumerate(scenes,1):
    for rx in [None,*np.unique(a['rx']).tolist()]:
     chosen=np.ones(27000,bool) if rx is None else a['rx']==rx
     rows.append(dict(method=method,model_seed=seed,scene=scene,RX='ALL' if rx is None else str(rx),**partial_statistics(a['logits'][0,chosen],a['logits'][j,chosen],a['y'][chosen],role='V')))
print(json.dumps(dict(status='SOURCE_PARTIAL_H_CANDIDATES_COLLECTED',rows=rows,models=6,source_V_per_model=27000,new_F_forwards=0,new_training_updates=0,U_TX_truth_reads=0,target_reads=0,G_forwards=0)))
'''
    for key,value in [('RUNTIME',rt),('METHODS',METHODS),('SEEDS',SEEDS),('SCENES',SCENES)]:script=script.replace(key,repr(value))
    compile(script,'source_partial_H_diagnostics','exec');r=ssh(script);summary=[]
    if len(r['rows'])!=108 or {(v['method'],v['model_seed']) for v in r['rows']}!=set(itertools.product(METHODS,SEEDS)):raise ValueError('All paired three-seed methods/receivers required')
    for method,scene in itertools.product(METHODS,SCENES):
        selected=[v for v in r['rows'] if v['method']==method and v['scene']==scene and v['RX']=='ALL']
        for metric in ('raw_precision_pct','candidate_inclusion_pct','partial_logit_gradient_norm_mean','hard_raw_label_logit_gradient_norm_mean','partial_H_loss_mean','hard_H_loss_mean'):
            values=[v[metric] for v in selected];summary.append(dict(method=method,scene=scene,metric=metric,models=3,
                mean=statistics.fmean(values),sample_SD=statistics.stdev(values)))
    r.update(summary=summary,minimum_source_RX_candidate_inclusion_pct=min(v['candidate_inclusion_pct'] for v in r['rows'] if v['RX']!='ALL'),
        fixed_E50_source_proxy_only=True,new_method_training_started=False,goal_complete=False)
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(r,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=r['status'],source_rows=108,minimum_source_RX_candidate_inclusion_pct=r['minimum_source_RX_candidate_inclusion_pct'],summary=summary)))


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--runtime',required=True);p.add_argument('--output',required=True);a=p.parse_args();collect(a.runtime,a.output)
