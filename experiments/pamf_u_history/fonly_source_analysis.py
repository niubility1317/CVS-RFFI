"""User-directed standalone F source analysis; no G criteria or new IQ/model reads."""
import argparse
import json
from pathlib import Path
from experiments.cric_dura_comparison.io import read,write
from .covariance_source import RUN as SOURCE_RUN,METHODS,SEED

RUN='20261026-analysis-PAMF-SIG1-Covariance-Fonly-m4-r01'
SCENES=('practical_low_suburban','practical_mid_urban','practical_low_urban')
RAW_VIEWS=('raw',*SCENES)


def counts(header,method):
    if (header['status']!='COVARIANCE_SOURCE_EVALUATED' or header['method']!=method or header['model_seed']!=SEED
        or header['source_V_seen']!=27000 or header['target_reads'] or header['U_truth_read']
        or not header['parameters_and_registered_buffers_unchanged'] or header['optimizer_updates']):
        raise ValueError('Only complete frozen same-seed source readout headers required')
    result={}
    for row in header['results']:
        if row['view'] not in RAW_VIEWS:continue
        if row['view'] in result or row['n']!=27000 or type(row['correct']) is not int or not 0<=row['correct']<=27000:
            raise ValueError('Unique complete raw V27000 integer counts required')
        result[row['view']]=row['correct']
    if set(result)!=set(RAW_VIEWS):raise ValueError('Complete four raw source views required')
    return result


def rule(candidate,control,state):
    if set(candidate)!=set(RAW_VIEWS) or set(control)!=set(RAW_VIEWS):raise ValueError('Only four direct-F raw source views enter new criteria')
    delta={v:candidate[v]-control[v] for v in RAW_VIEWS}
    conditions=dict(full_new_budget=state['calls']==6660 and state['seen']==1701000,
        all_U=state['unique_seen']==56700 and state['min_visits']==30 and state['max_visits']==30 and state['covariance_U_seen']==state['seen'],
        clean_floor=delta['raw']>=-27,raw_three_nondecreasing=sum(delta[v] for v in SCENES)>=0,
        legal_updates=not state['U_truth_read'] and not state['target_reads'] and not state['model_EMA_updates'] and state['new_inference_parameters']==0)
    return dict(source_necessary_pass=all(conditions.values()),conditions=conditions,raw_correct_count_delta=delta,
        raw_clean_delta_pp=100*delta['raw']/27000,raw_three_delta_pp=100*sum(delta[v] for v in SCENES)/81000,
        scope='new user-directed standalone F scope;old G-inclusive run verdict unchanged;not formal performance or goal success')


def analyze(evidence,output):
    root=Path(evidence)/'pilot';control=counts(read(root/'baseline_source/complete.json'),'SIG1-PL');rows=[]
    for method in METHODS:
        p=root/'F'/('F0-'+method+'-s'+str(SEED));header=p/'source_validation/complete.json'
        if not header.exists():rows.append(dict(method=method,status='WAITING_SOURCE_FINAL'));continue
        candidate=counts(read(header),method);state=read(p/'source/covariance_state.json')
        rows.append(dict(method=method,status='FROZEN_SOURCE_ANALYZED',candidate_correct=candidate,**rule(candidate,control,state)))
    result=dict(status='FONLY_SOURCE_ANALYZED' if all(r['status']=='FROZEN_SOURCE_ANALYZED' for r in rows) else 'PARTIAL_FONLY_SOURCE_ANALYSIS',
        run_id=RUN,source_run_id=SOURCE_RUN,model_seed=SEED,baseline='completed matched SIG1-PL',baseline_correct=control,rows=rows,
        raw_only=True,G_criteria_used=False,new_training=0,new_IQ_reads=0,new_F_G_weight_reads=0,new_target_reads=0,
        change_authorization='user20261026 explicitly removed DURA-cooperation route;cleanIQ and satellite derivatives only',
        old_G_inclusive_rule_relabelled=False,goal_complete=False)
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    for row in rows:write(out/row['method']/'analysis.json',row,exclusive=False)
    write(out/'analysis.json',result,exclusive=False)
    print(json.dumps(result,ensure_ascii=False));return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--evidence',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();analyze(a.evidence,a.output)
