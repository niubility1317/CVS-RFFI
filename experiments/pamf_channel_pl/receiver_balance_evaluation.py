"""Raw-only target evaluation, closed until both registered RB source rows pass selection."""
from pathlib import Path
import time
import numpy as np
from experiments.cric_dura_comparison.io import read,write
from experiments.cric_dura_comparison.evaluation import reject_truth,VIEWS
from . import receiver_balance_source as source
from . import clean_h_evaluation as shared

TASK_COUNTS=shared.TASK_COUNTS;SEEDS=shared.SEEDS
BASELINE_RUN=shared.BASELINE_RUN
CONFIRM_RUN='20261027-phase1-PAMF-SIG1-REPL-RxBalance-confirm-m3-r01'


def selected_method(c):
    if c.get('purpose') not in ('source_selection','selected_method'):raise ValueError('No baseline prediction or undeclared selection purpose')
    path=c['selection_source'];s=read(path)
    if (Path(path).parent.name!=source.RUN or s.get('status')!='RECEIVER_BALANCE_SOURCE_COMPLETE' or s.get('run_id')!=source.RUN
        or s.get('model_seed')!=2026092701 or s.get('target_reads')!=0 or s.get('U_truth_reads')!=0
        or set(s.get('source_scores',{}))!=set(source.METHODS) or s.get('reference')!=dict(raw_correct=26552,H3_correct=62148)):
        raise ValueError('Target closed until both source-only variants complete')
    decisions=[]
    for method in source.METHODS:
        score=s['source_scores'][method]
        if set(score)!={'raw_correct','H3_correct'} or any(type(score[k]) is not int or not 0<=score[k]<=n for k,n in [('raw_correct',27000),('H3_correct',81000)]):
            raise ValueError('Complete source physical counts required')
        decisions.append(source.source_decision(method,score,s['reference']))
    valid=[r for r in decisions if r['source_pass']]
    winner=max(valid,key=lambda r:(r['raw_correct'],r['H3_correct'],-source.METHODS.index(r['method'])))['method'] if valid else None
    if winner is None or s.get('source_selected')!=winner or s.get('decisions')!=decisions:raise ValueError('Target closed: unchanged fixed source gate/ranking must pass')
    if c['purpose']=='selected_method' and c.get('method')!=winner:raise ValueError('Only source-selected method may access target')
    return winner


def validate(c,scoring=False):
    expected=shared.COMMON|({'prediction_output'} if scoring else shared.PREDICT_ONLY)
    if (set(c)!=expected or c['schema']!='pamf_clean_h_raw_eval_v1' or c['run_id']!=CONFIRM_RUN or c['protocol_id']!='F0'
        or c['model_seed'] not in SEEDS or c['source_days']!=[1,2,3] or c['readouts']!=['raw'] or len(c['classes'])!=6
        or c['purpose']!='selected_method' or c['method'] not in source.METHODS or c['row_id']!='F0-'+c['method']+'-s'+str(c['model_seed'])
        or set(c['tests'])!=set(TASK_COUNTS)):raise ValueError('Three-seed selected-method raw-only complete F0 evaluation required')
    for task in c['tests'].values():
        if set(task)!=({'views_root','truth','kind'} if scoring else {'views_root'}):raise ValueError('Prediction and independent truth-side handles must remain separate')
    if not scoring:
        reject_truth(c)
        expected_run=source.RUN if c['model_seed']==SEEDS[0] else CONFIRM_RUN
        if (set(c['frozen_sources'])!={'F','AF','BASE','FOLD'} or c['selection_source']!=c['project']+'/runs/'+source.RUN+'/complete.json'
            or c['source_run_id']!=expected_run or c['source_row_id']!=c['row_id']
            or Path(c['source_output'])!=Path(c['project'])/'runs'/expected_run/'F'/c['row_id']/'source'):
            raise ValueError('Completed seed01 or own matched seed02/03 only; no G or baseline jobs')


def predict(c,device='cuda:0'):
    validate(c);winner=selected_method(c) # Must run before activation/model/any target input.
    source.base.activate(c)
    from experiments.pamf_repl_completion.source import load_reader
    from experiments.cric_dura_comparison.evaluation import handle
    contract=read(c['source_contract']);anchor=read(Path(c['selection_source']).parent/'configs'/(winner+'.json'))
    if contract!=read(anchor['source_contract']):raise ValueError('Own checkpoint contract must match selected F0 source')
    src=read(Path(c['source_output'])/'complete.json');shared.source_metadata(src,c,contract)
    reader,_=load_reader(c['source_output'],contract,device);pool=reader.sequence_pool
    if len(pool.views)!=1 or pool.arm!='SIG_TIME' or pool.modes!=('SIG_TIME',) or tuple(reader.core.id_backbone.cls_head.weight.shape)!=(6,160):
        raise ValueError('Actual complete original single-view PAMF/default head required')
    before=source.base.snapshot(reader);out=Path(c['output_root']);out.mkdir(parents=True,exist_ok=False);counts={};started=time.monotonic()
    for task in c['tests']:
        manifest,ids,positions=handle(c['tests'][task]['views_root'],c['classes'])
        if len(ids)!=TASK_COUNTS[task]:raise ValueError('Original physical population differs')
        values={}
        for view in VIEWS:
            path=manifest['clean_ref'] if view=='clean' else str(Path(manifest.get('observations_root',c['tests'][task]['views_root']))/(view+'.npy'))
            array=np.load(path,mmap_mode='r',allow_pickle=False)
            values['raw__'+view]=shared.array_predictions(reader,array,positions,len(ids),6,device)
            write(out/'progress.json',dict(status='PREDICTING',task=task,view=view,physical_packets=len(ids),truth_read=False),exclusive=False)
        folder=out/task;folder.mkdir();np.savez(folder/'predictions.npz',ids=ids,**values)
        counts[task]=dict(count=len(ids),views=list(VIEWS),readouts=['raw'])
    if not source.base.unchanged(reader,before):raise ValueError('Frozen target predictions must not change F parameters/buffers')
    write(out/'complete.json',dict(status='ALL_PREDICTIONS_COMPLETE',run_id=c['run_id'],row_id=c['row_id'],tests=counts,views=list(VIEWS),readouts=['raw'],
        truth_read=False,query_fit=False,parameters_and_buffers_unchanged=True,source_frozen_verified=True,source_selected=winner,seconds=time.monotonic()-started))


def score(c):
    validate(c,scoring=True);end=read(Path(c['prediction_output'])/'complete.json')
    if (end.get('parameters_and_buffers_unchanged') is not True or end.get('source_frozen_verified') is not True or end.get('source_selected')!=c['method']):
        raise ValueError('Truth closed until selected frozen-F prediction completes')
    from experiments.cric_dura_comparison import evaluation as scorer
    previous=scorer.tasks;scorer.tasks=lambda _:list(c['tests'])
    try:scorer.score(c) # Preflight all registered IDs/views before first truth read; no F activation.
    finally:scorer.tasks=previous
