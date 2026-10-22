"""CPU-only checks and three-F-seed paired summaries, conditional on fixed G."""
from collections import defaultdict
import csv
import math
from pathlib import Path
import statistics
from .io import read,write
from .external_dura import READOUTS,VIEWS


def checked_metric(r):
    cm=r['confusion'];n=sum(sum(row) for row in cm);k=len(cm)
    if n!=r['count'] or any(len(row)!=k or any(not isinstance(v,int) or v<0 for v in row) for row in cm):
        raise ValueError('Malformed external confusion matrix')
    correct=sum(cm[i][i] for i in range(k));f1=0.
    for i in range(k):
        denom=sum(cm[i])+sum(row[i] for row in cm)
        f1+=2*cm[i][i]/denom if denom else 0.
    if abs(correct/n-r['accuracy'])>1e-12 or abs(f1/k-r['macro_f1'])>1e-12:
        raise ValueError('CM metric reconstruction differs')


def run(c):
    stats=defaultdict(dict);paired=defaultdict(dict);methods=defaultdict(dict);pl=defaultdict(dict);groles=defaultdict(dict)
    metrics_count=pairs_count=0
    for item in c['rows']:
        s=read(item['scores'])
        if s['status']!='SCORED' or s.get('truth_last') is not True or (s['run_id'],s['row_id'],s['model_seed'])!=(item['run_id'],item['row_id'],item['model_seed']):
            raise ValueError('Frozen external score identity differs')
        seed=item['model_seed'];pid=item['protocol_id'];family=item['family'];variant=item['variant'];regime=item['training_regime']
        overall={(r['test'],r['readout'],r['view']) for r in s['results'] if r['dimension']=='overall'}
        tests={'main','source_unseen_date'} if pid=='chrono' else {'main'}
        if overall!={(t,n,v) for t in tests for n in READOUTS for v in VIEWS}:raise ValueError('Incomplete external tasks/views/readouts')
        correct={}
        for r in s['results']:
            checked_metric(r);metrics_count+=1
            tail=(r['test'],r['readout'],r['view'],r['dimension'],r['stratum'])
            correct[tail]=(r['count'],sum(r['confusion'][i][i] for i in range(len(r['confusion']))))
            key=(pid,family,variant,regime)+tail
            if seed in stats[key]:raise ValueError('Duplicate external model seed')
            stats[key][seed]=(r['accuracy'],r['macro_f1'])
            methods[(pid,family,regime)+tail][variant,seed]=(r['accuracy'],r['macro_f1'])
            pl[(pid,family,variant)+tail][regime,seed]=(r['accuracy'],r['macro_f1'])
            groles[(pid,family,variant,regime,r['test'],r['view'],r['dimension'],r['stratum'])][r['readout'],seed]=(r['accuracy'],r['macro_f1'])
        for r in s['paired_raw']:
            raw=correct[r['test'],'raw',r['view'],r['dimension'],r['stratum']]
            restored=correct[r['test'],r['readout'],r['view'],r['dimension'],r['stratum']]
            if (r['net']!=r['repairs']-r['damage'] or r['net']!=restored[1]-raw[1]
                    or raw[0]!=restored[0] or r['count']!=raw[0]):raise ValueError('Invalid net corrections')
            pairs_count+=1
            key=(pid,family,variant,regime,r['test'],r['readout'],r['view'],r['dimension'],r['stratum'])
            paired[key][seed]=(r['net']/r['count'],r['repairs']/r['count'],r['damage']/r['count'])
    out=Path(c['output_root']);out.mkdir(parents=True,exist_ok=False)
    def moments(v):return [statistics.mean(v),statistics.stdev(v) if len(v)>1 else 0.]
    def dump(name,keys,groups,value_names,require_three=True):
        path=out/name;count=0
        with path.open('x',newline='',encoding='utf-8') as f:
            writer=csv.writer(f);writer.writerow(list(keys)+['F_seeds']+[n+s for n in value_names for s in ('_mean','_sd')])
            for key,values in sorted(groups.items()):
                if require_three and set(values)!={2026092701,2026092702,2026092703}:raise ValueError('Incomplete three-seed group')
                seq=list(values.values());row=[]
                for i in range(len(value_names)):row+=moments([x[i] for x in seq])
                writer.writerow(list(key)+[len(seq)]+row);count+=1
        return count
    keys=('protocol','family','variant','regime','task','readout','view','dimension','stratum')
    n=dump('external_DURA_summary.csv',keys,stats,('accuracy','macro_f1'))
    dump('external_DURA_vs_raw.csv',keys,paired,('net','repairs','damage'))
    def contrasts(groups,lo,hi):
        result={}
        for key,values in groups.items():
            if not any((hi,seed) in values for seed in (2026092701,2026092702,2026092703)):continue
            if not any((lo,seed) in values for seed in (2026092701,2026092702,2026092703)):continue
            result[key]={seed:tuple(a-b for a,b in zip(values[hi,seed],values[lo,seed])) for seed in (2026092701,2026092702,2026092703)}
        return result
    dump('external_DURA_Full_minus_CE.csv',('protocol','family','regime')+keys[4:],contrasts(methods,'ce','full'),('accuracy_delta','macro_f1_delta'))
    dump('external_DURA_PL_minus_Sup.csv',('protocol','family','variant')+keys[4:],contrasts(pl,'supervised','pl'),('accuracy_delta','macro_f1_delta'))
    dump('external_DURA_LU_minus_L.csv',keys[:5]+keys[6:],contrasts(groles,'DURA-L','DURA-LU'),('accuracy_delta','macro_f1_delta'))
    # Keep native budgets/source/G ancestor cost separate; readout cost is not retraining.
    costs=[]
    for item in c['rows']:
        f=read(Path(item['source_output'])/'freeze.json')
        costs.append(dict(row_id=item['row_id'],run_id=item['run_id'],source_seconds=f.get('seconds'),
            source_stages=f.get('stages'),new_F_training_seconds=0,new_G_training_seconds=0))
    write(out/'source_cost_reuse.json',dict(rows=costs,note='Original method-native source cost reused, not incurred again. Common G ancestry and one-time IQ cache cost reported separately; fixed G is not three G seeds.'))
    write(out/'verification.json',dict(status='PASS',models=516,metrics_recomputed=metrics_count,net_corrections_checked=pairs_count,
        summary_groups=n,F_seeds=3,conditional_on_fixed_G=True,new_F=0,new_G=0,negative_effects_retained=True))
    write(out/'complete.json',dict(status='EXTERNAL_DURA_ANALYZED',run_id=c['run_id'],models=516,raw_reused=True,
        readouts=list(READOUTS),truth_last=True,target_feedback_forbidden=True))
