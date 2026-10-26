"""Read all twelve frozen CPU V rows; keep source confidence separate from target accuracy."""
import argparse
import itertools
import json
import math
from pathlib import Path
import statistics
from .source_confidence_readout import RUN,METHODS,SEEDS
from .fonly_results import save


REFERENCES=('SIG1-PL','SIG1-Sup')
METRICS=('accuracy_pct','coverage_pct','accepted_precision_pct','rejected_accuracy_pct','rejected_share_of_all_errors_pct',
    'accepted_pseudo_CE_logit_gradient_norm_mean','rejected_pseudo_CE_logit_gradient_norm_mean',
    'accepted_LS001_opposes_hard_pseudo_CE_pct','mean_confidence','NLL','Brier','ECE15_pct')


def collection_script(rt):
    if rt['run_id']!=RUN or not rt.get('cpu_only'):raise ValueError('Actual CPU source-confidence runtime required')
    script=r'''
import json
from pathlib import Path
import numpy as np
rt=RUNTIME;methods=METHODS;seeds=SEEDS
root=Path(rt['run_root']);done=json.loads((root/'complete.json').read_text());state=json.loads((root/'pipeline_state.json').read_text())
if (done['status']!='SOURCE_CONFIDENCE_ALL_ROWS_COMPLETE' or state['status']!=done['status'] or done['methods']!=list(methods)
 or done['model_seeds']!=list(seeds) or done['models']!=12 or len(state['rows'])!=12 or any(r['status']!='COMPLETE' for r in state['rows'].values())
 or done['source_forward_packets']!=324024 or done['new_training_updates'] or done['target_reads'] or done['U_TX_truth_reads'] or done['G_forwards']):raise ValueError('All twelve immutable source rows required')
rows=[];metadata=None;checks=[]
for method in methods:
 for seed in seeds:
  folder=root/'rows'/('F0-'+method+'-s'+str(seed));row=json.loads((folder/'complete.json').read_text());smoke=json.loads((folder/'smoke.json').read_text())
  if (row['method']!=method or row['model_seed']!=seed or row['status']!='SOURCE_CONFIDENCE_ROW_COMPLETE' or row['source_role']!='V'
   or row['V_seen']!=27000 or row['source_L_smoke_seen']!=2 or not row['parameters_and_buffers_unchanged'] or row['device']!='cpu' or not row['FP32']
   or row['optimizer_updates'] or row['U_TX_truth_reads'] or row['target_reads'] or row['G_weight_reads'] or row['G_forwards']
   or smoke['status']!='PASS' or smoke['signature_views']!=1 or smoke['signature_parameters']!=18976 or smoke['default_head']!=[6,160]):raise ValueError('Matched immutable source readout/smoke differs')
  with np.load(folder/'source_V_logits.npz',allow_pickle=False) as a:
   if set(a.files)!={'logits','y','ids','rx','day'} or a['logits'].shape!=(27000,6) or not np.isfinite(a['logits']).all():raise ValueError('Complete finite six-class source logits required')
   current={k:a[k].copy() for k in ('ids','y','rx','day')}
  if any(v.shape!=(27000,) for v in current.values()) or len(np.unique(current['ids']))!=27000:raise ValueError('Unique full source validation population required')
  if metadata is None:metadata=current
  elif any(not np.array_equal(v,metadata[k]) for k,v in current.items()):raise ValueError('Source physical IDs/labels/RX/day differ')
  checks.append(dict(method=method,model_seed=seed,count=27000,exact_ID_label_RX_day_order=True))
  rows.append(row)
print(json.dumps(dict(status='ALL_TWELVE_CPU_SOURCE_ROWS_COLLECTED',run_id=rt['run_id'],rows=rows,metadata_checks=checks,
 source_forward_packets=324024,new_training_updates=0,U_truth_reads=0,target_reads=0,G_forwards=0)))
'''
    for key,value in [('RUNTIME',rt),('METHODS',METHODS),('SEEDS',SEEDS)]:script=script.replace(key,repr(value))
    compile(script,'source_confidence_collection','exec');return script


def summarize(bundle):
    if bundle.get('status')!='ALL_TWELVE_CPU_SOURCE_ROWS_COLLECTED' or bundle.get('run_id')!=RUN:raise ValueError('One fully completed source diagnostic required')
    expected=set(itertools.product(METHODS,SEEDS));rows=bundle['rows'];actual=[(r['method'],r['model_seed']) for r in rows]
    if len(actual)!=12 or set(actual)!=expected:raise ValueError('All four methods and all three seeds exactly once')
    checks=bundle['metadata_checks']
    if len(checks)!=12 or {(r['method'],r['model_seed']) for r in checks}!=expected or any(r['count']!=27000 or not r['exact_ID_label_RX_day_order'] for r in checks):raise ValueError('Exact source metadata pairing required')
    flat=[];lookup={};strata=None
    for row in rows:
        if (row['source_role']!='V' or row['V_seen']!=27000 or row['source_L_smoke_seen']!=2 or row['device']!='cpu' or not row['FP32']
            or not row['parameters_and_buffers_unchanged'] or any(row[k] for k in ('optimizer_updates','U_TX_truth_reads','target_reads','G_weight_reads','G_forwards'))):raise ValueError('Frozen source V only')
        groups=row['rows'];keys={(v['dimension'],v['stratum']) for v in groups}
        if len(keys)!=len(groups) or len(keys)!=30 or ('overall','ALL') not in keys or (strata is not None and keys!=strata):raise ValueError('All thirty source strata must match')
        strata=keys
        for original in groups:
            g=dict(original)
            n,a,c,ac,re=g['count'],g['accepted'],g['correct'],g['accepted_correct'],g['rejected_errors']
            if not (0<=ac<=a<=n and ac<=c<=n and n-c==a-ac+re and g['rejected']==n-a):raise ValueError('Source confidence counts inconsistent')
            if g['dimension']=='overall' and n!=27000:raise ValueError('Full V population required')
            g['rejected_accuracy_pct']=100*(c-ac)/(n-a) if n>a else None
            for metric in METRICS:
                if g[metric] is not None and not math.isfinite(g[metric]):raise ValueError('Finite or explicitly undefined source metrics required')
            r=dict(method=row['method'],model_seed=row['model_seed'],**g);flat.append(r);lookup[row['method'],row['model_seed'],g['dimension'],g['stratum']]=r
    def stats(values):
        # Undefined precision/error fractions stay undefined; never silently drop a seed.
        return (None,None) if any(v is None for v in values) else (statistics.fmean(values),statistics.stdev(values))
    summary=[];paired=[]
    for method in METHODS:
        for dimension,stratum in sorted(strata):
            for metric in METRICS:
                values=[lookup[method,s,dimension,stratum][metric] for s in SEEDS];mean,sd=stats(values)
                summary.append(dict(method=method,dimension=dimension,stratum=stratum,metric=metric,seeds=3,defined_seeds=sum(v is not None for v in values),mean=mean,sample_SD=sd))
                for ref in REFERENCES:
                    if method==ref:continue
                    controls=[lookup[ref,s,dimension,stratum][metric] for s in SEEDS]
                    deltas=[v-c if v is not None and c is not None else None for v,c in zip(values,controls)];dm,ds=stats(deltas)
                    paired.append(dict(method=method,reference=ref,dimension=dimension,stratum=stratum,metric=metric,seeds=3,
                        defined_pairs=sum(v is not None for v in deltas),mean_difference=dm,sample_SD_difference=ds))
    result=dict(status='ANALYZED_THREE_SEED_SOURCE_CONFIDENCE',run_id=RUN,models=12,source_V_per_model=27000,source_forward_packets=324024,
        model_seeds=list(SEEDS),new_training_updates=0,target_reads=0,U_TX_truth_reads=0,G_forwards=0,
        source_V_is_proxy_for_U=True,logit_gradients_are_parameter_gradients=False,causal_generalization_claim=False,goal_complete=False,
        overall=[r for r in summary if r['dimension']=='overall'],paired_overall=[r for r in paired if r['dimension']=='overall'])
    return result,dict(per_seed_strata=flat,three_seed_summary=summary,paired_differences=paired)


def collect(runtime,output):
    from experiments.cric_dura_comparison.publish import ssh
    out=Path(output)
    if out.exists():raise FileExistsError('Preserve existing source diagnostic bundle')
    r=ssh(collection_script(json.loads(Path(runtime).read_text(encoding='utf-8'))));summarize(r)
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(r,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=r['status'],models=len(r['rows']),source_forward_packets=r['source_forward_packets'])))


def analyze(bundle,output):
    result,tables=summarize(json.loads(Path(bundle).read_text(encoding='utf-8')));out=Path(output);out.mkdir(parents=True,exist_ok=True)
    (out/'analysis.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    for name,rows in tables.items():save(out/(name+'.csv'),rows)
    print(json.dumps(dict(status=result['status'],overall=result['overall'],goal_complete=False)));return result


if __name__=='__main__':
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='action',required=True)
    c=sub.add_parser('collect');c.add_argument('--runtime',required=True);c.add_argument('--output',required=True)
    a=sub.add_parser('analyze');a.add_argument('--bundle',required=True);a.add_argument('--output',required=True)
    args=p.parse_args();collect(args.runtime,args.output) if args.action=='collect' else analyze(args.bundle,args.output)
