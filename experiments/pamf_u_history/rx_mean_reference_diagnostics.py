"""Same-packet raw/H reference errors from existing frozen source V logits only."""
import argparse
import inspect
import itertools
import json
from pathlib import Path
import statistics
import numpy as np
from .source_confidence_metrics import probabilities
from .rx_mean_source import RUN,METHODS,SEEDS
from .fonly_results import save

SCENES=('practical_low_suburban','practical_mid_urban','practical_low_urban')
METRICS=('raw_accuracy_pct','H_accuracy_pct','raw_H_disagreement_pct','raw_correct_H_wrong_pct','raw_wrong_H_correct_pct',
    'H_confident_wrong_pct','H_confident_precision_pct','raw_H_KL_mean','clean_soft_target_logit_gradient_norm_mean')


def paired_statistics(raw,H,y,*,role):
    if role!='V':raise ValueError('Visible source V diagnostics only; no U truth')
    p=probabilities(raw);q=probabilities(H);y=np.asarray(y)
    if q.shape!=p.shape or y.shape!=(len(p),) or y.dtype.kind not in 'iu' or y.min()<0 or y.max()>=6:raise ValueError('Aligned source V raw/H/visible labels required')
    a=p.argmax(1);b=q.argmax(1);ca=a==y;cb=b==y;high=q.max(1)>=.95;n=len(y)
    return dict(count=n,raw_correct=int(ca.sum()),H_correct=int(cb.sum()),raw_H_disagreements=int((a!=b).sum()),
        raw_correct_H_wrong=int((ca&~cb).sum()),raw_wrong_H_correct=int((~ca&cb).sum()),H_confident=int(high.sum()),H_confident_wrong=int((high&~cb).sum()),
        raw_accuracy_pct=100*ca.mean(),H_accuracy_pct=100*cb.mean(),raw_H_disagreement_pct=100*(a!=b).mean(),
        raw_correct_H_wrong_pct=100*(ca&~cb).mean(),raw_wrong_H_correct_pct=100*(~ca&cb).mean(),H_confident_wrong_pct=100*(high&~cb).mean(),
        H_confident_precision_pct=100*cb[high].mean() if high.any() else None,
        raw_H_KL_mean=float((p*(np.log(p.clip(1e-300))-np.log(q.clip(1e-300)))).sum(1).mean()),
        clean_soft_target_logit_gradient_norm_mean=float(np.linalg.norm(q-p,axis=1).mean()))


def diagnose(logits,y,rx,day,views):
    logits=np.asarray(logits);y=np.asarray(y);rx=np.asarray(rx);day=np.asarray(day)
    if tuple(views)!=('raw',*SCENES) or logits.shape!=(4,len(y),6) or y.shape!=rx.shape or y.shape!=day.shape:raise ValueError('Four original same-packet source views required')
    accepted=probabilities(logits[0]).max(1)>=.95;rows=[]
    masks=[('overall','ALL',np.ones(len(y),bool))]
    for name,a in [('TX',y),('RX',rx),('day',day)]:masks.extend((name,str(v),a==v) for v in np.unique(a))
    masks.extend(('RX_day',str(r)+'/'+str(d),(rx==r)&(day==d)) for r in np.unique(rx) for d in np.unique(day))
    for dimension,stratum,mask in masks:
        for scope,chosen in [('all',mask),('raw_accepted',mask&accepted),('raw_rejected',mask&~accepted)]:
            if not chosen.any():continue
            for j,scene in enumerate(SCENES,1):rows.append(dict(dimension=dimension,stratum=stratum,scope=scope,scene=scene,
                **paired_statistics(logits[0,chosen],logits[j,chosen],y[chosen],role='V')))
    return rows


def collection_script(rt):
    if rt['run_id']!=RUN:raise ValueError('Completed six-model RXMean runtime required')
    pieces=['import numpy as np',inspect.getsource(probabilities),inspect.getsource(paired_statistics),inspect.getsource(diagnose)]
    script=r'''
import json
from pathlib import Path
rt=RUNTIME;methods=METHODS;seeds=SEEDS
root=Path(rt['run_root']);done=json.loads((root/'complete.json').read_text())
if done['status']!='RX_MEAN_THREE_SEED_COMPLETE' or done['models']!=6 or done['methods']!=list(methods) or done['model_seeds']!=list(seeds):raise ValueError('Complete preregistered six-model matrix required')
rows=[];metadata=None;checks=[]
for method in methods:
 for seed in seeds:
  source=root/'F'/('F0-'+method+'-s'+str(seed));d=json.loads((source/'source/complete.json').read_text());v=json.loads((source/'source_validation/complete.json').read_text())
  if (d['status']!='SOURCE_FROZEN' or d['epoch']!=50 or d['updates']!=11100 or d['target_access'] or d['U_truth_read'] or d['head_fit_updates']
   or d['config']['paper_method']!=method or d['config']['model_seed']!=seed or v['status']!='MULTISEED_SOURCE_EVALUATED'
   or v['method']!=method or v['model_seed']!=seed or v['V_seen']!=27000 or v['G_reads'] or v['target_reads']):raise ValueError('Actual fixed E50 source-only saved predictions required')
  with np.load(source/'source_validation/source_V_predictions.npz',allow_pickle=False) as a:
   if set(a.files)!={'logits','y','ids','rx','day','views'} or a['logits'].shape!=(4,27000,6) or not np.isfinite(a['logits']).all():raise ValueError('Four complete finite saved source views required')
   current={k:a[k].copy() for k in ('ids','y','rx','day')}
   if any(v.shape!=(27000,) for v in current.values()) or len(np.unique(current['ids']))!=27000:raise ValueError('Original unique full V population required')
   if metadata is None:metadata=current
   elif any(not np.array_equal(x,metadata[k]) for k,x in current.items()):raise ValueError('Exact physical ID/label/RX/day pairing differs')
   summary=diagnose(a['logits'],a['y'],a['rx'],a['day'],a['views'].tolist())
  rows.extend(dict(method=method,model_seed=seed,**r) for r in summary);checks.append(dict(method=method,model_seed=seed,source_V=27000,exact_ID_label_RX_day_order=True))
print(json.dumps(dict(status='PAIRED_SOURCE_REFERENCE_DIAGNOSTICS_COLLECTED',run_id=rt['run_id'],rows=rows,metadata_checks=checks,
 existing_logit_packets_analyzed=648000,new_F_forwards=0,new_training_updates=0,U_TX_truth_reads=0,target_reads=0,G_forwards=0)))
'''
    for key,value in [('RUNTIME',rt),('METHODS',METHODS),('SEEDS',SEEDS)]:script=script.replace(key,repr(value))
    return '\n\n'.join(pieces)+'\nSCENES='+repr(SCENES)+'\n'+script


def summarize(bundle):
    if bundle['run_id']!=RUN or bundle['status']!='PAIRED_SOURCE_REFERENCE_DIAGNOSTICS_COLLECTED':raise ValueError('Exact completed source-only diagnostic required')
    if any(bundle[k] for k in ('new_F_forwards','new_training_updates','U_TX_truth_reads','target_reads','G_forwards')):raise ValueError('Existing source outputs only')
    expected=set(itertools.product(METHODS,SEEDS));checks=bundle['metadata_checks']
    if len(checks)!=6 or {(c['method'],c['model_seed']) for c in checks}!=expected or any(c['source_V']!=27000 or not c['exact_ID_label_RX_day_order'] for c in checks):raise ValueError('Both methods and all three paired seeds required')
    overall=[r for r in bundle['rows'] if r['dimension']=='overall'];summary=[]
    for method,scope,scene in itertools.product(METHODS,('all','raw_accepted','raw_rejected'),SCENES):
        selected=[r for r in overall if r['method']==method and r['scope']==scope and r['scene']==scene]
        if len(selected)!=3 or {r['model_seed'] for r in selected}!=set(SEEDS):raise ValueError('Every overall scope/scene requires all three seeds')
        if scope=='all' and any(r['count']!=27000 for r in selected):raise ValueError('Complete same-packet source V required')
        for metric in METRICS:
            values=[r[metric] for r in selected];defined=all(v is not None for v in values)
            summary.append(dict(method=method,scope=scope,scene=scene,metric=metric,models=3,mean=statistics.fmean(values) if defined else None,
                sample_SD=statistics.stdev(values) if defined else None,count_min=min(r['count'] for r in selected),count_max=max(r['count'] for r in selected)))
    return dict(status='ANALYZED_RX_MEAN_SOURCE_REFERENCES',run_id=RUN,models=6,model_seeds=list(SEEDS),source_V_per_model=27000,
        existing_logit_packets_analyzed=648000,new_F_forwards=0,new_training_updates=0,U_TX_truth_reads=0,target_reads=0,G_forwards=0,
        source_only=True,causal_generalization_claim=False,goal_complete=False,summary=summary)


def collect(runtime,output):
    from experiments.cric_dura_comparison.publish import ssh
    out=Path(output)
    if out.exists():raise FileExistsError('Keep existing paired source bundle')
    script=collection_script(json.loads(Path(runtime).read_text(encoding='utf-8')));compile(script,'source_reference_read','exec');bundle=ssh(script);summarize(bundle)
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(bundle,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=bundle['status'],rows=len(bundle['rows']),new_F_forwards=0)))


def analyze(bundle,output):
    bundle=json.loads(Path(bundle).read_text(encoding='utf-8'));result=summarize(bundle);out=Path(output);out.mkdir(parents=True,exist_ok=True)
    (out/'analysis.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');save(out/'three_seed_summary.csv',result['summary']);save(out/'per_seed_strata.csv',bundle['rows'])
    print(json.dumps(dict(status=result['status'],models=6,new_F_forwards=0,goal_complete=False)));return result


if __name__=='__main__':
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='action',required=True)
    c=sub.add_parser('collect');c.add_argument('--runtime',required=True);c.add_argument('--output',required=True)
    a=sub.add_parser('analyze');a.add_argument('--bundle',required=True);a.add_argument('--output',required=True)
    args=p.parse_args();collect(args.runtime,args.output) if args.action=='collect' else analyze(args.bundle,args.output)
