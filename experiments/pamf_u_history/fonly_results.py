"""Read frozen predictions/scores; compare direct F without reopening truth or IQ."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import numpy as np
from .fonly_confirm import RUN,BASELINE_RUN,METHODS,TASKS,SEED,VIEWS


def collect(runtime,context,output):
    from experiments.cric_dura_comparison.publish import ssh,CONNECTION
    rt=json.loads(Path(runtime).read_text());ctx=json.loads(Path(context).read_text());out=Path(output);out.mkdir(parents=True,exist_ok=True)
    if rt['run_id']!=RUN:raise ValueError('One fixed F-only confirmation required')
    script=r'''
import json
from pathlib import Path
import numpy as np
rt=RUNTIME;ctx=CONTEXT;methods=METHODS;tasks=TASKS;views=VIEWS
root=Path(rt['run_root']);done=json.loads((root/'complete.json').read_text())
if done['status']!='FONLY_CONFIRMATION_COMPLETE' or done['readouts']!=['raw'] or not done['truth_last']:raise ValueError('All frozen rows/scores mustfinish first')
checks=[];decisions=0
for method in methods:
 rid='F0-'+method+'-s2026092701'
 for task in tasks:
  with np.load(root/'readouts'/rid/'predictions'/task/'predictions.npz',allow_pickle=False) as a:
   n={'cross_rx':126000,'cross_both':42000}[task]
   if len(a['ids'])!=n or set(a.files)!={'ids',*('raw__'+v for v in views)}:raise ValueError('Fixed raw prediction coverage differs')
   for view in views:
    v=a['raw__'+view]
    if v.shape!=(n,) or v.dtype.kind not in 'iu' or v.min()<0 or v.max()>=6:raise ValueError('Invalidclassdecisions')
    decisions+=len(v)
   for reference,folder in ctx['reference_predictions'].items():
    with np.load(Path(folder)/task/'predictions.npz',allow_pickle=False) as b:
     if not np.array_equal(a['ids'],b['ids']):raise ValueError('Candidate/reference physical order differs')
    checks.append(dict(method=method,reference=reference,task=task,count=n,exact_ID_order=True))
print(json.dumps(dict(status='PHYSICAL_PAIRING_VERIFIED',new_raw_decisions=decisions,checks=checks,truth_reads=0,IQ_reads=0,F_G_weight_reads=0)))
'''.replace('RUNTIME',repr(rt)).replace('CONTEXT',repr(ctx)).replace('METHODS',repr(METHODS)).replace('TASKS',repr(TASKS)).replace('VIEWS',repr(VIEWS))
    pairing=ssh(script);(out/'physical_pairing.json').write_text(json.dumps(pairing,indent=2)+'\n',encoding='utf-8')
    for method,remote in ctx['reference_scores'].items():
        p=out/'references'/method/'scores.json';p.parent.mkdir(parents=True,exist_ok=True)
        subprocess.run(['scp',*CONNECTION,'N607:'+remote,str(p)],check=True,timeout=60,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
    print(json.dumps(dict(status='COLLECTED_FROZEN_REFERENCES',paired_checks=len(pairing['checks']),new_raw_decisions=pairing['new_raw_decisions'],new_truth_reads=0)))


def metrics(cm):
    n=int(cm.sum());den=cm.sum(0)+cm.sum(1)
    if n<=0:raise ValueError('Nonempty evaluated population required')
    return dict(count=n,correct=int(cm.trace()),accuracy=100*float(cm.trace()/n),macro_f1=100*float(np.divide(2*cm.diagonal(),den,out=np.zeros(6),where=den!=0).mean()))


def index(data):
    if data['status']!='SCORED' or not data['truth_last'] or not data['target_feedback_forbidden'] or data['model_seed']!=SEED:raise ValueError('Complete independent same-seed scores required')
    result={}
    for row in data['results']:
        if row['readout']!='raw':continue
        key=tuple(row[k] for k in ('test','view','dimension','stratum'));cm=np.asarray(row['confusion'])
        if cm.shape!=(6,6) or cm.dtype.kind not in 'iu' or (cm<0).any() or int(cm.sum())!=row['count'] or key in result:raise ValueError('Unique six-class confusion counts required')
        m=metrics(cm)
        if abs(m['accuracy']/100-row['accuracy'])>1e-12 or abs(m['macro_f1']/100-row['macro_f1'])>1e-12:raise ValueError('Saved score differs from its confusion')
        result[key]=dict(row,correct=m['correct'])
    if not {(t,v,'overall','ALL') for t in TASKS for v in VIEWS}.issubset(result):raise ValueError('All populations/seven raw views required')
    return result


def save(path,rows):
    if not rows:return
    with Path(path).open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def analyze(evidence,references,output):
    root=Path(evidence);refs=Path(references);out=Path(output);out.mkdir(parents=True,exist_ok=True)
    load=lambda p:json.loads(Path(p).read_text())
    pairing=load(refs/'physical_pairing.json')
    if pairing['status']!='PHYSICAL_PAIRING_VERIFIED' or pairing['new_raw_decisions']!=3528000 or len(pairing['checks'])!=12:raise ValueError('Complete exact physical pairing required')
    data={};summaries=[];scenes=[];paired=[]
    for method in METHODS:
        d=load(root/'readouts'/('F0-'+method+'-s'+str(SEED))/'scores/scores.json')
        if d['run_id']!=RUN or d['row_id']!='F0-'+method+'-s'+str(SEED):raise ValueError('Wrong candidate row')
        data[method]=index(d)
    for reference in ('SIG1-PL','SIG1-Sup'):
        d=load(refs/'references'/reference/'scores.json')
        if d['run_id']!=BASELINE_RUN or d['row_id']!='F0-'+reference+'-s'+str(SEED):raise ValueError('Exact completed matched single-view control required')
        data[reference]=index(d)
    for method,values in data.items():
        for population in (*TASKS,'overall_168000'):
            cells=[]
            for view in VIEWS:
                selected=[np.asarray(values[t,view,'overall','ALL']['confusion']) for t in TASKS if population=='overall_168000' or t==population]
                m=metrics(sum(selected));expected=168000 if population=='overall_168000' else {'cross_rx':126000,'cross_both':42000}[population]
                if m['count']!=expected:raise ValueError('Weighted formal count differs')
                row=dict(method=method,population=population,view=view,**m);cells.append(row);scenes.append(row)
            summaries.append(dict(method=method,population=population,model_seed=SEED,count=cells[0]['count'],raw_clean=cells[0]['accuracy'],raw_six_mean=float(np.mean([r['accuracy'] for r in cells[1:]])),
                clean_macro_f1=cells[0]['macro_f1'],six_macro_f1_mean=float(np.mean([r['macro_f1'] for r in cells[1:]]))))
    for method in METHODS:
        a=data[method]
        for reference in ('SIG1-PL','SIG1-Sup'):
            b=data[reference]
            if set(a)!=set(b):raise ValueError('Exact score strata differ')
            for key,row in a.items():
                old=b[key]
                if row['count']!=old['count']:raise ValueError('Compared physical stratum counts differ')
                paired.append(dict(method=method,reference=reference,**dict(zip(('test','view','dimension','stratum'),key)),count=row['count'],
                    net_correct_gain=row['correct']-old['correct'],accuracy_delta_pp=100*(row['accuracy']-old['accuracy']),macro_f1_delta_pp=100*(row['macro_f1']-old['macro_f1'])))
    comparisons=[]
    for method in METHODS:
        a=next(r for r in summaries if r['method']==method and r['population']=='overall_168000')
        for reference in ('SIG1-PL','SIG1-Sup'):
            b=next(r for r in summaries if r['method']==reference and r['population']=='overall_168000')
            av={r['view']:r['accuracy'] for r in scenes if r['method']==method and r['population']=='overall_168000'}
            bv={r['view']:r['accuracy'] for r in scenes if r['method']==reference and r['population']=='overall_168000'}
            comparisons.append(dict(method=method,reference=reference,raw_clean_delta_pp=a['raw_clean']-b['raw_clean'],raw_six_delta_pp=a['raw_six_mean']-b['raw_six_mean'],
                min_practical_scene_delta_pp=min(av[v]-bv[v] for v in VIEWS[1:]),raw_clean_floor_met=a['raw_clean']>=85.98174603174603,raw_six_floor_met=a['raw_six_mean']>72.0961970899471,
                all_stratum_accuracy_nondecreasing=all(r['accuracy_delta_pp']>=0 for r in paired if r['method']==method and r['reference']==reference)))
    result=dict(status='ANALYZED_FONLY_SINGLE_SEED',run_id=RUN,model_seed=SEED,summary=summaries,comparisons=comparisons,
        all_IDs_exactly_paired=True,raw_only=True,new_training=0,new_truth_reads=0,new_IQ_reads=0,G_weight_reads=0,full_goal_complete=False,
        limitations=['single model seed;original3seed andother declared scope incomplete','no paired significance claim from confusion-only comparisons','no target feedback or candidate reselection'])
    (out/'analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    save(out/'summary.csv',summaries);save(out/'per_scene.csv',scenes);save(out/'paired_strata.csv',paired);save(out/'comparisons.csv',comparisons)
    print(json.dumps(dict(status=result['status'],overall=[r for r in summaries if r['population']=='overall_168000'],comparisons=comparisons,full_goal_complete=False)));return result


if __name__=='__main__':
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='action',required=True)
    a=sub.add_parser('collect');a.add_argument('--runtime',required=True);a.add_argument('--context',required=True);a.add_argument('--output',required=True)
    a=sub.add_parser('analyze');a.add_argument('--evidence',required=True);a.add_argument('--references',required=True);a.add_argument('--output',required=True)
    c=p.parse_args();collect(c.runtime,c.context,c.output) if c.action=='collect' else analyze(c.evidence,c.references,c.output)
