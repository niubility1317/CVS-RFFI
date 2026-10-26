"""Complete balanced three-seed analysis of frozen raw decisions and saved scores."""
import argparse
import itertools
import json
from pathlib import Path,PurePosixPath
import numpy as np
from .covariance_three_seed import RUN,OLD_CONFIRM,BASELINE,SEEDS,TASKS
from .covariance_objective import METHODS
from .fonly_results import index,metrics,save,VIEWS

REFERENCES=('SIG1-PL','SIG1-Sup')
ALL_METHODS=(*METHODS,*REFERENCES)
COUNTS=dict(cross_rx=126000,cross_both=42000)
POPULATIONS=(*TASKS,'overall_168000')


def expected_run(method,seed):
    if method not in ALL_METHODS or seed not in SEEDS:raise ValueError('Fixed four candidates and two completed controls,three seeds')
    if method in REFERENCES:return BASELINE
    return OLD_CONFIRM if seed==SEEDS[0] and method!='CovFull-U' else RUN


def pairing_keys(methods=METHODS):return set(itertools.product(methods,SEEDS,REFERENCES,TASKS))


def locations(rt):
    p=PurePosixPath(rt['project'])/'runs';result=[]
    for method,seed in itertools.product(ALL_METHODS,SEEDS):
        rid='F0-'+method+'-s'+str(seed);run=expected_run(method,seed)
        result.append(dict(method=method,model_seed=seed,run_id=run,row_id=rid,root=str(p/run/'readouts'/rid)))
    return result


def collect(runtime,output):
    from experiments.cric_dura_comparison.publish import ssh
    rt=json.loads(Path(runtime).read_text(encoding='utf-8'))
    if rt['run_id']!=RUN:raise ValueError('One complete12model raw matrix required')
    script=r'''
import json
from pathlib import Path
import numpy as np
rt=RUNTIME;rows=ROWS;views=VIEWS;counts=COUNTS;methods=METHODS;references=REFERENCES;tasks=TASKS;seeds=SEEDS
done=json.loads((Path(rt['run_root'])/'complete.json').read_text())
if (done['status']!='THREE_SEED_COVARIANCE_COMPLETE' or done['logical_models']!=12 or done['new_training']!=8
 or done['reused_seed01_models']!=4 or done['reused_seed01_readouts']!=3 or done['new_prediction_rows']!=9
 or done['model_seeds']!=list(seeds) or done['methods']!=list(methods) or not done['raw_only'] or not done['truth_last']):raise ValueError('All fixed models and independent scores must finish first')
scores=[];ids={};decisions=0
for row in rows:
 root=Path(row['root']);pred=json.loads((root/'predictions/complete.json').read_text());scored=json.loads((root/'scores/complete.json').read_text())
 if pred['status']!='ALL_PREDICTIONS_COMPLETE' or scored['status']!='SCORED_COMPLETE':raise ValueError('Prediction and independent score artifacts incomplete')
 score=json.loads((root/'scores/scores.json').read_text());scores.append(dict(method=row['method'],model_seed=row['model_seed'],score=score))
 for task in tasks:
  with np.load(root/'predictions'/task/'predictions.npz',allow_pickle=False) as a:
   n=counts[task];physical=a['ids'].copy()
   if physical.shape!=(n,) or len(np.unique(physical))!=n:raise ValueError('Full unique physical population required')
   ids[row['method'],row['model_seed'],task]=physical
   if row['method'] in methods:
    if set(a.files)!={'ids',*('raw__'+v for v in views)}:raise ValueError('Candidate must have only declared direct-F views')
    for view in views:
     value=a['raw__'+view]
     if value.shape!=(n,) or value.dtype.kind not in 'iu' or value.min()<0 or value.max()>=6:raise ValueError('Finite sixclass decisions required')
     decisions+=n
checks=[]
for method in methods:
 for seed in seeds:
  for reference in references:
   for task in tasks:
    if not np.array_equal(ids[method,seed,task],ids[reference,seed,task]):raise ValueError('Same-seed physical order differs')
    checks.append(dict(method=method,model_seed=seed,reference=reference,task=task,count=counts[task],exact_ID_order=True))
print(json.dumps(dict(status='THREE_SEED_FROZEN_RESULTS_COLLECTED',scores=scores,pairing_checks=checks,
 logical_candidate_raw_decisions=decisions,truth_reads=0,IQ_reads=0,weight_reads=0)))
'''
    for name,value in [('RUNTIME',rt),('ROWS',locations(rt)),('VIEWS',VIEWS),('COUNTS',COUNTS),('METHODS',METHODS),('REFERENCES',REFERENCES),('TASKS',TASKS),('SEEDS',SEEDS)]:script=script.replace(name,repr(value))
    result=ssh(script);out=Path(output);out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists():raise FileExistsError('Preserve the earlier frozen result bundle')
    out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=result['status'],score_rows=len(result['scores']),physical_pairings=len(result['pairing_checks']),candidate_raw_decisions=result['logical_candidate_raw_decisions'],new_truth_reads=0)))


def mean_sd(values):
    if len(values)!=3:raise ValueError('All three model seeds are required')
    return float(np.mean(values)),float(np.std(values,ddof=1))


def summarize(bundle,*,methods=METHODS,run_id=RUN,row_run=expected_run,status='ANALYZED_COVARIANCE_THREE_SEED_FONLY'):
    methods=tuple(methods);all_methods=(*methods,*REFERENCES)
    if not methods or len(set(all_methods))!=len(all_methods):raise ValueError('Unique fixed candidates and two distinct completed references required')
    expected_decisions=len(methods)*len(SEEDS)*sum(COUNTS.values())*len(VIEWS)
    if (bundle['status']!='THREE_SEED_FROZEN_RESULTS_COLLECTED' or bundle['logical_candidate_raw_decisions']!=expected_decisions
        or any(bundle[k]!=0 for k in ('truth_reads','IQ_reads','weight_reads'))):raise ValueError('Only completed frozen decision/score readback required')
    pairs={}
    for row in bundle['pairing_checks']:
        key=tuple(row[k] for k in ('method','model_seed','reference','task'))
        if key in pairs or not row['exact_ID_order'] or row['count']!=COUNTS.get(row['task']):raise ValueError('Exact full physical pairing required')
        pairs[key]=row
    if set(pairs)!=pairing_keys(methods):raise ValueError('Complete same-seed candidate/control pairing required')
    values={}
    for item in bundle['scores']:
        method,seed=item['method'],item['model_seed'];key=(method,seed);d=item['score']
        if key in values or d['run_id']!=row_run(method,seed) or d['row_id']!='F0-'+method+'-s'+str(seed):raise ValueError('Unique actual completed row identity required')
        values[key]=index(d,seed)
        for task,view in itertools.product(TASKS,VIEWS):
            if values[key][task,view,'overall','ALL']['count']!=COUNTS[task]:raise ValueError('Original physical population count required')
    if set(values)!=set(itertools.product(all_methods,SEEDS)):raise ValueError('Every fixed candidate/control model seed is required')
    if any(set(v)!=set(next(iter(values.values()))) for v in values.values()):raise ValueError('All model seeds must report the same declared strata')
    per_seed=[];scenes=[];paired=[];grouped=[];comparisons=[];strata_mean=[];scene_mean=[];scene_paired=[]
    for (method,seed),data in values.items():
        for population in POPULATIONS:
            cells=[]
            for view in VIEWS:
                cm=sum(np.asarray(data[t,view,'overall','ALL']['confusion']) for t in TASKS if population=='overall_168000' or t==population)
                m=metrics(cm);expected=168000 if population=='overall_168000' else COUNTS[population]
                if m['count']!=expected:raise ValueError('Count-weighted primary population required')
                row=dict(method=method,model_seed=seed,population=population,view=view,**m);cells.append(row);scenes.append(row)
            per_seed.append(dict(method=method,model_seed=seed,population=population,count=expected,raw_clean=cells[0]['accuracy'],raw_six_mean=float(np.mean([r['accuracy'] for r in cells[1:]])),
                clean_macro_f1=cells[0]['macro_f1'],six_macro_f1_mean=float(np.mean([r['macro_f1'] for r in cells[1:]]))))
    for method,population in itertools.product(all_methods,POPULATIONS):
        rows=[r for r in per_seed if r['method']==method and r['population']==population];row=dict(method=method,population=population,model_seeds=list(SEEDS),models=3,count_per_seed=rows[0]['count'])
        for metric in ('raw_clean','raw_six_mean','clean_macro_f1','six_macro_f1_mean'):
            average,sd=mean_sd([r[metric] for r in rows]);name='raw_six' if metric=='raw_six_mean' else metric
            row[name+'_mean']=average;row[name+'_sample_sd']=sd
        grouped.append(row)
    scene_lookup={(r['method'],r['model_seed'],r['population'],r['view']):r for r in scenes}
    for method,population,view in itertools.product(all_methods,POPULATIONS,VIEWS):
        row=dict(method=method,population=population,view=view,models=3,count_per_seed=168000 if population=='overall_168000' else COUNTS[population])
        for metric in ('accuracy','macro_f1'):
            average,sd=mean_sd([scene_lookup[method,s,population,view][metric] for s in SEEDS]);row[metric+'_mean']=average;row[metric+'_sample_sd']=sd
        scene_mean.append(row)
    for method,reference,population,view in itertools.product(methods,REFERENCES,POPULATIONS,VIEWS):
        row=dict(method=method,reference=reference,population=population,view=view,models=3)
        for metric in ('accuracy','macro_f1'):
            delta=[scene_lookup[method,s,population,view][metric]-scene_lookup[reference,s,population,view][metric] for s in SEEDS]
            average,sd=mean_sd(delta);row[metric+'_delta_mean_pp']=average;row[metric+'_delta_sample_sd_pp']=sd;row[metric+'_worst_seed_delta_pp']=min(delta)
        scene_paired.append(row)
    for method,reference,seed in itertools.product(methods,REFERENCES,SEEDS):
        a,b=values[method,seed],values[reference,seed]
        for key,now in a.items():
            old=b[key]
            if now['count']!=old['count']:raise ValueError('Same physical stratum required')
            paired.append(dict(method=method,reference=reference,model_seed=seed,**dict(zip(('test','view','dimension','stratum'),key)),count=now['count'],net_correct_gain=now['correct']-old['correct'],
                accuracy_delta_pp=100*(now['accuracy']-old['accuracy']),macro_f1_delta_pp=100*(now['macro_f1']-old['macro_f1'])))
    by_stratum={}
    for row in paired:
        key=tuple(row[k] for k in ('method','reference','test','view','dimension','stratum'));by_stratum.setdefault(key,[]).append(row)
    for key,rows in by_stratum.items():
        row=dict(zip(('method','reference','test','view','dimension','stratum'),key));row.update(models=3,count_per_seed=rows[0]['count'])
        for metric in ('accuracy_delta_pp','macro_f1_delta_pp'):
            average,sd=mean_sd([r[metric] for r in rows]);row[metric+'_mean']=average;row[metric+'_sample_sd']=sd
        strata_mean.append(row)
    lookup={(r['method'],r['model_seed'],r['population']):r for r in per_seed}
    for method,reference,population in itertools.product(methods,REFERENCES,POPULATIONS):
        row=dict(method=method,reference=reference,population=population,models=3)
        for metric in ('raw_clean','raw_six_mean','clean_macro_f1','six_macro_f1_mean'):
            delta=[lookup[method,s,population][metric]-lookup[reference,s,population][metric] for s in SEEDS]
            average,sd=mean_sd(delta);name='raw_six' if metric=='raw_six_mean' else metric
            row[name+'_delta_mean_pp']=average;row[name+'_delta_sample_sd_pp']=sd;row[name+'_worst_seed_delta_pp']=min(delta)
        means=[r for r in grouped if r['method']==method and r['population']==population][0]
        row.update(raw_clean_floor_met=means['raw_clean_mean']>=85.98174603174603,raw_six_floor_met=means['raw_six_mean']>72.0961970899471)
        relevant=[r for r in strata_mean if r['method']==method and r['reference']==reference and (population=='overall_168000' or r['test']==population)]
        row['min_declared_stratum_accuracy_delta_mean_pp']=min(r['accuracy_delta_pp_mean'] for r in relevant)
        row['min_declared_stratum_macro_f1_delta_mean_pp']=min(r['macro_f1_delta_pp_mean'] for r in relevant)
        practical=[r for r in scene_paired if r['method']==method and r['reference']==reference and r['population']==population and r['view']!='clean']
        row['min_practical_accuracy_delta_mean_pp']=min(r['accuracy_delta_mean_pp'] for r in practical)
        comparisons.append(row)
    result=dict(status=status,run_id=run_id,model_seeds=list(SEEDS),methods=list(methods),logical_models=len(methods)*len(SEEDS),
        formal_population_per_seed=168000,raw_only=True,physical_pairings=len(pairs),full_goal_complete=False,summary=grouped,comparisons=comparisons,
        new_truth_reads=0,new_IQ_reads=0,new_weight_reads=0,limitations=['Same physical split;model-seed variation only','Sample SD over3 models isnot a significance test',
        'Confusion-derived differences are net correct gains;no paired error significance inferred','Other original declared scope remains required;do not select variants orrerun usingthese scores'])
    return result,dict(per_seed=per_seed,per_scene=scenes,per_scene_mean=scene_mean,paired_scene_mean=scene_paired,summary=grouped,paired_strata=paired,paired_strata_mean=strata_mean,paired_summary=comparisons)


def analyze(bundle,output):
    result,tables=summarize(json.loads(Path(bundle).read_text(encoding='utf-8')));out=Path(output);out.mkdir(parents=True,exist_ok=True)
    (out/'analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    for name,rows in tables.items():save(out/(name+'.csv'),rows)
    print(json.dumps(dict(status=result['status'],model_seeds=result['model_seeds'],overall=[r for r in result['summary'] if r['population']=='overall_168000'],full_goal_complete=False)))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();s=p.add_subparsers(dest='action',required=True)
    c=s.add_parser('collect');c.add_argument('--runtime',required=True);c.add_argument('--output',required=True)
    a=s.add_parser('analyze');a.add_argument('--bundle',required=True);a.add_argument('--output',required=True)
    args=p.parse_args();collect(args.runtime,args.output) if args.action=='collect' else analyze(args.bundle,args.output)
