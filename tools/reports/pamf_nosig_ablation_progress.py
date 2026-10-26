"""Read-only replay of completed five-arm scores, including partial seed sets."""
import argparse
from collections import defaultdict
import csv
from datetime import datetime,timezone,timedelta
import gzip
from itertools import product
import json
from pathlib import Path
import statistics

from experiments.cric_dura_comparison.analyze_main import metrics
from experiments.pamf_no_signature_ablation.method import ARMS,PARAMETERS,READOUTS,TASKS

VIEWS=('clean','practical_high','practical_mid','practical_low_suburban',
       'practical_high_urban','practical_mid_urban','practical_low_urban')
NAMES=dict(BASE='去 SIG-TIME 基底',NO_PA='再去功放支路',NO_REFERENCE='再去参考响应支路',
           NO_PA_REFERENCE='同时去功放和参考响应支路',WRONG_REFERENCE='打乱参考模板相位')
COUNTS=dict(cross_date=30000,cross_rx=126000,cross_both=42000,pooled=198000)


def moments(values):
    values=list(values)
    return dict(n=len(values),mean=statistics.mean(values),sd=statistics.stdev(values) if len(values)>1 else None)


def save(path,rows):
    with path.open('x',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def recompute(snapshot):
    values={};status=[];counts={};CM=0;nets=0
    for entry in snapshot['records']:
        row=entry['row'];arm,seed=row['arm'],row['model_seed']
        status.append(dict(arm=arm,name=NAMES[arm],seed=seed,reused=row['reuse'],source_frozen=entry['source_frozen'],
                           predictions_complete=entry['predictions_complete'],scored=entry['scored'],epoch=entry['epoch']))
        if not entry['scored']:continue
        score=entry['score']
        if not score['truth_last'] or not score['target_feedback_forbidden'] or score['model_seed']!=seed:raise ValueError('Score independence/identity differs')
        lookup={}
        for r in score['results']:
            key=tuple(r[k] for k in ('test','readout','view','dimension','stratum'))
            if key in lookup:raise ValueError('Duplicate CM')
            lookup[key]=r;metrics(r,6);CM+=1
        for t,rd,v in product(TASKS,READOUTS,VIEWS):
            if lookup[t,rd,v,'overall','ALL']['count']!=COUNTS[t]:raise ValueError('Incomplete test population')
        for rd,v in product(READOUTS,VIEWS):
            a=[lookup[t,rd,v,'overall','ALL'] for t in TASKS]
            cm=[[sum(r['confusion'][i][j] for r in a) for j in range(6)] for i in range(6)]
            n=sum(map(sum,cm));f1=[]
            for i in range(6):
                den=sum(cm[i])+sum(r[i] for r in cm);f1.append(2*cm[i][i]/den if den else 0.)
            r=dict(test='pooled',readout=rd,view=v,dimension='overall',stratum='ALL',count=n,confusion=cm,
                   accuracy=sum(cm[i][i] for i in range(6))/n,macro_f1=statistics.mean(f1))
            lookup['pooled',rd,v,'overall','ALL']=r
            if n!=COUNTS['pooled']:raise ValueError('Pooled population count differs')
        for k,r in lookup.items():
            for m,value in metrics(r,6).items():values[(arm,seed)+k+(m,)]=100*value
        for p in score['paired_raw']:
            k=tuple(p[n] for n in ('test','readout','view','dimension','stratum'));raw=(k[0],'raw',*k[2:])
            correct=lambda row:sum(row['confusion'][i][i] for i in range(6))
            if p['net']!=p['repairs']-p['damage'] or p['net']!=correct(lookup[k])-correct(lookup[raw]):raise ValueError('Net correction differs from CM')
            nets+=1
        bases={(t,rd,dim,st,m) for a,s,t,rd,v,dim,st,m in values if a==arm and s==seed}
        for t,rd,dim,st,m in bases:
            values[arm,seed,t,rd,'six_practical_mean',dim,st,m]=statistics.mean(values[arm,seed,t,rd,v,dim,st,m] for v in VIEWS if v!='clean')
    groups=defaultdict(dict)
    for (arm,seed,*key),v in values.items():groups[(arm,)+tuple(key)][seed]=v
    summary=[];paired=[]
    for (arm,*key),a in sorted(groups.items()):
        t,rd,v,dim,st,m=key;common=dict(arm=arm,name=NAMES[arm],test=t,readout=rd,view=v,dimension=dim,stratum=st,metric=m)
        summary.append(dict(common,**moments(a.values()),seeds=';'.join(map(str,sorted(a)))))
        if arm!='BASE':
            base=groups[('BASE',)+tuple(key)]
            if not set(a)<=set(base):raise ValueError('Missing same-seed baseline')
            seeds=sorted(a);diff=[a[s]-base[s] for s in seeds]
            paired.append(dict(common,**moments(diff),seeds=';'.join(map(str,seeds)),seed_deltas=';'.join(map(str,diff)),
                               matched_baseline_mean=statistics.mean(base[s] for s in seeds),matched_baseline_sd=statistics.stdev(base[s] for s in seeds) if len(seeds)>1 else None))
    verify=dict(status='VERIFIED_COMPLETED_SCORE_REPLAY',CM_recomputed=CM,net_correction_rows_checked=nets,
        source_models_complete=sum(r['source_frozen'] for r in status),new_models_complete=sum(r['source_frozen'] and not r['reused'] for r in status),
        source_models_training=sum(not r['source_frozen'] for r in status),scored_models=sum(r['scored'] for r in status),
        scored_new_models=sum(r['scored'] and not r['reused'] for r in status),three_seed_complete_arms=[a for a in ARMS if sum(r['scored'] and r['arm']==a for r in status)==3],
        partial_arms=[a for a in ARMS if 0<sum(r['scored'] and r['arm']==a for r in status)<3],
        pooled_count_per_view=COUNTS['pooled'],incomplete_seeds_excluded=True,pairing='Exact completed model-seed intersection with BASE',
        target_truth_new_reads=0,no_training_or_queue_mutation=True)
    return status,summary,paired,verify


def run(source,output):
    source,output=Path(source),Path(output)
    with gzip.open(source,'rt',encoding='utf-8') as f:snapshot=json.load(f)
    output.mkdir(parents=True,exist_ok=False)
    status,summary,paired,verify=recompute(snapshot)
    observed=datetime.fromtimestamp(snapshot['observed_at'],timezone(timedelta(hours=8))).isoformat()
    verify.update(observed_at=observed,run_id=snapshot['runtime']['run_id'],controller_status=snapshot['state']['status'],
                  stage_counts=snapshot['state']['counts'],failures=snapshot['state']['failures'],timing_complete=snapshot['timing'] is not None)
    for name,rows in [('progress',status),('summary_overall',[r for r in summary if r['dimension']=='overall']),
                     ('paired_vs_baseline_overall',[r for r in paired if r['dimension']=='overall'])]:save(output/(name+'.csv'),rows)
    with gzip.open(output/'summary_all_strata.csv.gz','xt',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
    with gzip.open(output/'paired_all_strata.csv.gz','xt',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(paired[0]));w.writeheader();w.writerows(paired)
    (output/'verification.json').write_text(json.dumps(verify,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    headline=[]
    for arm,t,rd in product(ARMS,('cross_both','pooled'),READOUTS):
        out=dict(arm=arm,name=NAMES[arm],test=t,readout=rd)
        for v,m in product(('clean','six_practical_mean'),('accuracy','macro_f1')):
            s=next((r for r in summary if r['arm']==arm and r['test']==t and r['readout']==rd and r['view']==v and r['metric']==m and r['dimension']=='overall'),None)
            p=next((r for r in paired if r['arm']==arm and r['test']==t and r['readout']==rd and r['view']==v and r['metric']==m and r['dimension']=='overall'),None)
            out[v+'_'+m]=None if s is None else dict(n=s['n'],mean=s['mean'],sd=s['sd'],delta=None if p is None else p['mean'],delta_sd=None if p is None else p['sd'])
        headline.append(out)
    (output/'headline.json').write_text(json.dumps(headline,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(verification=verify,headline=headline),ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True);a=p.parse_args();run(a.source,a.output)
