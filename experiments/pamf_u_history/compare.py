"""Validate fixed physical pairing and compare every independent score stratum."""
import argparse
import json
from pathlib import Path
import numpy as np
from experiments.practical_full_2x2.plan import TASKS,VIEWS
from .confirm import RUN,ROW
from .source import SEED
from .analyze import dump_csv


def index_scores(d, *, candidate):
    if (d.get('status')!='SCORED' or d.get('truth_last') is not True or d.get('model_seed')!=SEED
            or (candidate and (d.get('run_id')!=RUN or d.get('row_id')!=ROW))):
        raise ValueError('Only matching completed independent truth-last scores are accepted')
    rows={}
    for row in d['results']:
        if row['readout'] not in ('raw','DURA-LU'):continue
        key=tuple(row[k] for k in ('test','readout','view','dimension','stratum'))
        cm=np.asarray(row['confusion'])
        if (cm.shape!=(6,6) or cm.dtype.kind not in 'iu' or bool((cm<0).any()) or cm.sum()!=row['count']
                or abs(float(cm.trace())/row['count']-row['accuracy'])>1e-12 or key in rows):
            raise ValueError('Unique complete six-class score counts must agree with reported accuracy')
        rows[key]=dict(row,correct=int(cm.trace()))
    expected={(t,r,v,'overall','ALL') for t in TASKS for r in ('raw','DURA-LU') for v in VIEWS}
    if not expected.issubset(rows):raise ValueError('Incomplete populations/readouts/seven-view scores')
    return rows


def compare(evidence,output):
    root=Path(evidence);out=Path(output);out.mkdir(parents=True,exist_ok=True)
    read=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
    actual=read(root/'candidate/complete.json')
    if actual['status']!='UHISTORY_CONFIRMATION_COMPLETE':raise ValueError('Whole confirmation must finish first')
    candidate=index_scores(read(root/'candidate/scores.json'),candidate=True);rows=[];summary=[]
    for method in ('ordinary_PL','supervised','REPL'):
        reference=index_scores(read(root/'references'/method/'scores.json'),candidate=False)
        for task in TASKS:
            with np.load(root/'candidate/predictions'/task/'predictions.npz',allow_pickle=False) as a,np.load(root/'references'/method/'predictions'/task/'predictions.npz',allow_pickle=False) as b:
                if not np.array_equal(a['ids'],b['ids']):raise ValueError('Candidate/reference physical sample IDs differ')
                n=len(a['ids'])
                for route in ('raw','DURA-LU'):
                    for view in VIEWS:
                        key=task,route,view,'overall','ALL'
                        if candidate[key]['count']!=n or reference[key]['count']!=n:raise ValueError('Physical prediction count differs from score count')
        if set(candidate)!=set(reference):raise ValueError('Candidate/reference score strata differ')
        for key,a in candidate.items():
            b=reference[key]
            if a['count']!=b['count']:raise ValueError('Compared score strata must use same physical count')
            net=a['correct']-b['correct']
            rows.append(dict(reference=method,**{k:a[k] for k in ('test','readout','view','dimension','stratum','count')},
                candidate_accuracy=100*a['accuracy'],reference_accuracy=100*b['accuracy'],delta_pp=100*net/a['count'],
                candidate_macro_f1=100*a['macro_f1'],reference_macro_f1=100*b['macro_f1'],macro_f1_delta_pp=100*(a['macro_f1']-b['macro_f1']),
                net_correct_gain=net,architecture='candidateSIG1/referenceX3;not a pure SSL-loss ablation'))
        for task in TASKS:
            group=[r for r in rows if r['reference']==method and r['test']==task and r['dimension']=='overall']
            scores={(r['readout'],r['view']):r for r in group};raw=scores['raw','clean'];g=scores['DURA-LU','clean']
            raw_six=[scores['raw',v] for v in VIEWS[1:]];g_six=[scores['DURA-LU',v] for v in VIEWS[1:]]
            item=dict(reference=method,test=task,seed=SEED,count=raw['count'],raw_clean=raw['candidate_accuracy'],raw_clean_delta_pp=raw['delta_pp'],
                raw_six_mean=sum(r['candidate_accuracy'] for r in raw_six)/6,raw_six_delta_pp=sum(r['delta_pp'] for r in raw_six)/6,
                G_clean=g['candidate_accuracy'],G_clean_delta_pp=g['delta_pp'],G_six_mean=sum(r['candidate_accuracy'] for r in g_six)/6,
                G_six_delta_pp=sum(r['delta_pp'] for r in g_six)/6,minimum_raw_scene_delta_pp=min(r['delta_pp'] for r in raw_six),
                minimum_G_scene_delta_pp=min(r['delta_pp'] for r in g_six),all_macro_f1_nondecreasing=all(r['macro_f1_delta_pp']>=0 for r in group))
            item['one_seed_goals_met']=(100*raw['net_correct_gain']>=raw['count'] and 100*sum(r['net_correct_gain'] for r in raw_six)>9*raw['count']
                and item['minimum_raw_scene_delta_pp']>0 and item['minimum_G_scene_delta_pp']>0 and g['net_correct_gain']>0 and item['all_macro_f1_nondecreasing'])
            summary.append(item)
    primary=[r for r in summary if r['reference'] in ('ordinary_PL','supervised')]
    result=dict(status='ANALYZED_SINGLE_SEED_CONFIRMATION',candidate='PL-History',model_seeds=[SEED],summary=summary,
        all_one_seed_goals_met=all(r['one_seed_goals_met'] for r in primary),full_goal_complete=False,
        goal_limitations=['only one seed;original multi-seed scope incomplete','X3 historical PL/Sup controls differ from new single-view architecture'],
        all_physical_IDs_paired=True,truth_last=True,new_training=0,method_reselected=False)
    (out/'comparison.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    dump_csv(out/'target_paired_metrics.csv',rows);dump_csv(out/'target_summary.csv',summary)
    print(json.dumps(result,ensure_ascii=False))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--evidence',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();compare(a.evidence,a.output)
