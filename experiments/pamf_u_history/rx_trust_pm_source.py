"""Same RXTrust-P loss, with a raw/H mode-normalized coefficient from existing source gradients."""
import argparse
from contextlib import contextmanager
from copy import deepcopy
import os
from pathlib import Path
import statistics
import time
from experiments.cric_dura_comparison.io import read,write
from . import rx_mean_source as mean
from .rx_trust_p_objective import RECIPE as P_RECIPE,Objective
from .rx_trust_p_source import RUN as P_RUN,validate_train as validate_P_reference
import math

METHODS=('RXTrust-PM',)
RECIPE=deepcopy(P_RECIPE)
RECIPE['coefficient_rule']='min(1,.1/max_base_median(aux_norm/primary_norm));reuse_originalP12train_batches6raw6H,noV'
from .rx_mean_max_source import RUN as MAX_RUN,check_exposure as check_max_exposure

RUN='20261027-phase1-PAMF-RXTrustPM-SIG1-threeSeed-m3-r01'
RELEASE='20261027-phase1-PAMF-RXTrustPM-SIG1-threeSeed-release-r01'
BASELINE=mean.BASELINE
PREDECESSOR=P_RUN
MEAN_BASE=mean.RUN
SEEDS=mean.SEEDS
TASKS=mean.TASKS
_validate_mean=mean.validate_train


def validate_context(c):
    if c.get('predecessor_complete')!=c['project']+'/runs/'+PREDECESSOR+'/complete.json':raise ValueError('Fixed current-run resource dependency only')
    mean.validate_context({k:v for k,v in c.items() if k!='predecessor_complete'})
    return c


def config(ctx,commit,seed,alpha,calibration=False):
    validate_context(ctx);clean={k:v for k,v in ctx.items() if k!='predecessor_complete'}
    c=mean.config(clean,commit,'RXMean-U',seed,alpha,calibration);rid='F0-'+METHODS[0]+'-s'+str(seed)
    c.update(run_id=RUN,row_id=rid,paper_method=METHODS[0],algorithm_id='pamf_sig1_rx_trust_partial_modecap_v1',rx_mean_recipe=deepcopy(RECIPE),
        output_root=ctx['project']+'/runs/'+RUN+'/F/'+rid+'/source')
    c['repl_row'].update(row_id=rid,method=METHODS[0],group='rx_trust_partial_modecap')
    return validate_train(c)


def validate_train(c):
    seed=c.get('model_seed');rid='F0-'+METHODS[0]+'-s'+str(seed);root=c['project']+'/runs/'+RUN+'/F/'+rid+'/source'
    if (seed not in SEEDS or c.get('run_id')!=RUN or c.get('row_id')!=rid or c.get('paper_method')!=METHODS[0]
        or c.get('algorithm_id')!='pamf_sig1_rx_trust_partial_modecap_v1' or c.get('rx_mean_recipe')!=RECIPE
        or c.get('repl_row',{}).get('group')!='rx_trust_partial_modecap'):raise ValueError('One fixed max-KL sourceRX selection method, all three seeds')
    b=deepcopy(c);bid='F0-RXMean-U-s'+str(seed);base=c['project']+'/runs/'+MEAN_BASE+'/F/'+bid+'/source'
    b.update(run_id=MEAN_BASE,row_id=bid,paper_method='RXMean-U',algorithm_id='pamf_sig1_rx_mean_training_v1',rx_mean_recipe=mean.RECIPE)
    b['repl_row'].update(row_id=bid,method='RXMean-U',group='rx_mean_three_seed')
    if 'segment_epochs' in c:
        offset=c.get('epoch_offset');length={20:10,30:20}.get(offset);parent=c['shared_SIG1_E20'] if offset==20 else root+'/segment30'
        if length is None or c['output_root']!=root+'/segment'+str(offset+length) or c.get('parent_source')!=parent:raise ValueError('Own immutable segment paths and E20 parents only')
        b['output_root']=base+'/segment'+str(offset+length)
        b['parent_source']=c['shared_SIG1_E20'] if offset==20 else base+'/segment30'
    else:
        if c['output_root']!=root:raise ValueError('One exclusive row output root')
        b['output_root']=base
    _validate_mean(b)
    return c


@contextmanager
def trajectory():
    # Only this worker's process-local bindings change. Original releases and
    # currently healthy workers keep their committed code/configuration.
    old_validate,old_objective=mean.validate_train,mean.Objective
    mean.validate_train=validate_train;mean.Objective=Objective
    try:yield mean
    finally:mean.validate_train,mean.Objective=old_validate,old_objective


def check_exposure(h):
    check_max_exposure(h)
    if (h['P_target_rule']!=RECIPE['H_target'] or h['P_U_TX_truth_read'] or h['P_extra_F_forwards']
        or h['P_H_singleton']+h['P_H_pair']+h['P_H_soft_fallback']!=850500):raise ValueError('Every H packet must have one exact singleton/partial/local route')
    if set(h['P_H_by_source_RX'])!=set(map(str,(1,3,4,6,8))) or set(h['P_H_by_raw_pseudo_class'])!=set(map(str,range(6))):raise ValueError('Original RX and pseudo-class accounting required')
    for value in h['P_H_by_source_RX'].values():
        if set(value)!={'singleton','pair','fallback'} or any(v<0 for v in value.values()) or sum(value.values())!=170100:raise ValueError('All original H realizations per source RX required')
    for key,name in [('P_H_singleton','singleton'),('P_H_pair','pair'),('P_H_soft_fallback','fallback')]:
        if sum(v[name] for v in h['P_H_by_source_RX'].values())!=h[key]:raise ValueError('RX route counts do not match exposure')
    if sum(h['P_H_by_raw_pseudo_class'].values())!=h['P_H_singleton']+h['P_H_pair']:raise ValueError('Only raw-accepted H anchors enter singleton/partial identity sets')


def mode_coefficient(steps):
    if len(steps)!=12 or [r['step'] for r in steps]!=list(range(1,13)):raise ValueError('The original twelve complete source gradient measurements are required')
    medians={}
    for base in ('raw','H'):
        selected=[r for r in steps if r['base']==base]
        if len(selected)!=6:raise ValueError('Six raw and six H source batches required')
        ratios=[]
        for r in selected:
            a,b=r['gradient']['norm'],r['gradient']['base_norm']
            if not math.isfinite(a) or not math.isfinite(b) or a<0 or b<=0:raise ValueError('Finite source auxiliary/primary gradient norms required')
            ratios.append(a/b)
        medians[base]=statistics.median(ratios)
    largest=max(medians.values())
    if largest<=0:raise ValueError('Nonzero source auxiliary gradient evidence required')
    return min(1.,.1/largest),medians


def calibrate(rt):
    # No new model forward or optimizer: these statistics were produced from
    # the same original E20 states, with exact parameters/buffers unchanged.
    ctx=validate_context(read(rt['context']));seed=rt['model_seed'];root=Path(rt['output_root']);root.mkdir(parents=True,exist_ok=False)
    ref=Path(ctx['project'])/'runs'/PREDECESSOR;folder=ref/'calibration'/('s'+str(seed))
    reference=read(folder/'calibration.json');smoke=read(folder/'smoke.json');cfg=validate_P_reference(read(ref/'configs'/('train__'+str(seed)+'.json')))
    own=config(ctx,rt['code_commit'],seed,1.)
    if (reference['status']!='RX_TRUST_P_GRADIENT_CALIBRATED' or reference['formula']!=P_RECIPE['coefficient_rule'] or reference['model_seed']!=seed or reference['optimizer_updates'] or reference['V_reads'] or reference['target_reads'] or not reference['parameters_unchanged']
        or reference['L_seen']!=1536 or reference['U_seen']!=3072 or reference['raw_U']!=1536 or reference['H_U']!=1536
        or cfg['run_id']!=PREDECESSOR or cfg['paper_method']!='RXTrust-P' or cfg['model_seed']!=seed or cfg['rx_mean_recipe']!=P_RECIPE
        or cfg['source_cache']!=ctx['source_cache'] or read(cfg['source_contract'])!=read(ctx['source_contract'])
        or cfg['shared_SIG1_E20']!=own['shared_SIG1_E20'] or cfg['frozen_sources']!=ctx['frozen_sources']
        or smoke['status']!='PASS' or smoke['actual_parent']!=own['shared_SIG1_E20'] or smoke['actual_model_parameters']!=863730
        or smoke['optimizer_updates'] or smoke['U_truth_read'] or smoke['target_reads'] or not smoke['L_scope_isolated']):raise ValueError('Verified unchanged-objective same-E20/source P gradient reference required')
    alpha,medians=mode_coefficient(reference['steps'])
    write(Path(rt['run_root'])/'configs'/('train__'+str(seed)+'.json'),config(ctx,rt['code_commit'],seed,alpha))
    write(root/'calibration.json',dict(status='RX_TRUST_PM_CONFIGURED_FROM_SOURCE_P',model_seed=seed,coefficient=alpha,formula=RECIPE['coefficient_rule'],
        source_reference_run=PREDECESSOR,source_reference_model_seed=seed,source_reference_parent=own['shared_SIG1_E20'],source_batches_reused=12,
        unweighted_norm_ratio_medians=medians,weighted_norm_ratio_medians={k:alpha*v for k,v in medians.items()},
        new_F_forwards=0,optimizer_updates=0,V_reads=0,target_reads=0,U_TX_truth_reads=0,model_state_loaded=False,
        actual_new_checkpoint_smoke='first_training_segment_before_updates',reference_parameters_unchanged=True))
    write(root/'complete.json',dict(status='RX_TRUST_PM_CALIBRATED',model_seed=seed,coefficient=alpha,configs_frozen_before_training=True,new_F_forwards=0,target_reads=0))



def validate_evaluation(c,scoring=False):
    from experiments.cric_dura_comparison.evaluation import reject_truth
    if (c.get('run_id')!=RUN or c.get('model_seed') not in SEEDS or c.get('method')!=METHODS[0] or c.get('readouts')!=['raw'] or c.get('protocol_id')!='F0'
        or c.get('source_days')!=[1,2,3] or c.get('row_id')!='F0-'+METHODS[0]+'-s'+str(c['model_seed']) or set(c['tests'])!=set(TASKS) or len(c['classes'])!=6
        or any(k in c for k in ('G','source_G','restored_cache','restored_caches','teacher_checkpoint','mean_bank'))):raise ValueError('One original raw-F inference rule only')
    handles={'source_output','source_contract','source_cache','frozen_sources'}
    if scoring:
        if handles&set(c):raise ValueError('Independent scorer rejects model/source handles')
    else:
        reject_truth(c)
        if not handles.issubset(c) or set(c['frozen_sources'])!={'F','AF','BASE','FOLD'}:raise ValueError('Frozen direct F handles only')
    return c


def check_frozen(c):
    done=read(Path(c['source_output'])/'complete.json');cfg=validate_train(done['config']);check_exposure(read(Path(c['source_output'])/'rx_mean_state.json'))
    if (done['status']!='SOURCE_FROZEN' or done['epoch']!=50 or done['updates']!=11100 or done['target_access'] or done['U_truth_read'] or done['head_fit_updates']
        or cfg['row_id']!=c['row_id'] or cfg['model_seed']!=c['model_seed'] or cfg['source_cache']!=c['source_cache'] or read(cfg['source_contract'])!=read(c['source_contract'])):raise ValueError('Own fixed finalE50 source model required')
    return cfg


def prepare(rt):
    ctx=validate_context(read(rt['context']));evaluation=read(rt['evaluation_context']);root=Path(rt['run_root']);jobs=[]
    def add(key,kind,path,completion,status,deps):
        jobs.append(dict(id=key,kind=kind,config=str(path),completion=str(completion),successful_statuses=[status],dependencies=[str(v) for v in deps],
            execution_release=rt['release_root'],execution_python=rt['python'],entrypoint='experiments.pamf_u_history.rx_trust_pm_source'))
    clean={k:v for k,v in rt.items() if k!='evaluation_context'}
    for seed in SEEDS:
        path=root/'configs'/('calibrate__'+str(seed)+'.json');calroot=root/'calibration'/('s'+str(seed));write(path,dict(clean,model_seed=seed,output_root=str(calroot)))
        add('calibrate__'+str(seed),'calibrate',path,calroot/'complete.json','RX_TRUST_PM_CALIBRATED',[ctx['predecessor_complete']])
        rid='F0-'+METHODS[0]+'-s'+str(seed);source=root/'F'/rid/'source';path=root/'configs'/('train__'+str(seed)+'.json')
        add('train__'+str(seed),'train',path,source.parent/'source_validation/complete.json','MULTISEED_SOURCE_EVALUATED',[calroot/'complete.json'])
        out=root/'readouts'/rid
        p=dict(run_id=RUN,row_id=rid,method=METHODS[0],model_seed=seed,protocol_id='F0',source_days=[1,2,3],classes=evaluation['classes'],readouts=['raw'],
            tests={t:dict(views_root=evaluation['tests'][t]['views_root']) for t in TASKS},source_output=str(source),source_contract=ctx['source_contract'],source_cache=ctx['source_cache'],
            frozen_sources=ctx['frozen_sources'],code_commit=rt['code_commit'],output_root=str(out/'predictions'))
        validate_evaluation(p);path=root/'configs'/('predict__'+str(seed)+'.json');write(path,p)
        add('predict__'+str(seed),'predict',path,out/'predictions/complete.json','ALL_PREDICTIONS_COMPLETE',[source.parent/'source_validation/complete.json'])
        s=deepcopy(p);s.update(prediction_output=p['output_root'],output_root=str(out/'scores'),tests={t:evaluation['tests'][t] for t in TASKS})
        for k in ('source_output','source_contract','source_cache','frozen_sources'):s.pop(k)
        validate_evaluation(s,True);path=root/'configs'/('score__'+str(seed)+'.json');write(path,s)
        add('score__'+str(seed),'score',path,out/'scores/complete.json','SCORED_COMPLETE',[out/'predictions/complete.json'])
    write(root/'reused_controls.json',dict(runs=[BASELINE,PREDECESSOR],methods=['SIG1-PL','SIG1-Sup','RXTrust-P'],model_seeds=list(SEEDS),new_control_training=0,new_control_predictions=0))
    write(root/'jobs.json',jobs);return jobs


def worker(c,kind):
    if kind=='calibrate':return calibrate(c)
    if kind=='score':
        validate_evaluation(c,True)
        from experiments.cric_dura_comparison import evaluation
        evaluation.tasks=lambda _:list(TASKS);return evaluation.score(c)
    if kind=='predict':
        from .fonly_confirm import activate_raw,predict
        activate_raw(c);return predict(c,validator=validate_evaluation,source_validator=check_frozen)
    if kind!='train':raise ValueError('Only fixed source training and frozen independent tests')
    validate_train(c);mean.activate(dict(frozen_sources=c['frozen_sources'],channel_code=c['channel_code']))
    with trajectory() as original:original.train(c)
    check_exposure(read(Path(c['output_root'])/'rx_mean_state.json'))
    from .covariance_three_seed import evaluate_source
    evaluate_source(c)


def dispatch(rt):
    from experiments.pamf_repl_completion.runner import capacity,processes,foreign_CPU_score
    from experiments.pamf_peer_supervised.runner import launch
    from experiments.cric_dura_comparison.pipeline import done,ready
    from experiments.practical_full_2x2.cache import atomic
    from .covariance_queue import admissible,POLICY
    if rt['run_id']!=RUN or not rt.get('cpu_only') or rt.get('queue_policy')!=POLICY:raise ValueError('One frozen three-model matrix')
    root=Path(rt['run_root']);root.mkdir(parents=True,exist_ok=False);jobs=prepare(rt);by={j['id']:j for j in jobs};active={};children={};stop=False
    state=dict(status='WAITING_FOR_PREDECESSOR',pid=os.getpid(),code_commit=rt['code_commit'],jobs={j['id']:dict(status='QUEUED') for j in jobs});ticks=0
    while True:
        for key,item in list(active.items()):
            code=children[key].poll()
            if code is None:continue
            good=code==0 and done(by[key]);state['jobs'][key]=dict(item,status='COMPLETE' if good else 'FAILED',exit_code=code);stop|=not good;del active[key];del children[key]
        pending=processes()
        for job in sorted(jobs,key=lambda j:dict(score=0,predict=1,calibrate=2,train=3)[j['kind']]):
            key=job['id']
            if stop or state['jobs'][key]['status']!='QUEUED' or not ready(job):continue
            gpu=None
            if job['kind']=='score':
                if any(by[k]['kind']=='score' for k in active) or foreign_CPU_score({v['pid'] for v in active.values()}):continue
            else:
                choices=[]
                for card in rt['allowed_gpus']:
                    pids,memory=capacity(card)
                    if admissible(card,pids,pending,active,memory):choices.append((-memory,card))
                if not choices:continue
                gpu=min(choices)[1]
            child,item=launch(rt,job,gpu);active[key]=item;children[key]=child;state['jobs'][key]=item;state['status']='RUNNING'
            if gpu is not None:pending[child.pid]=gpu
        state.update(updated_at=time.time(),systemic_stop_new_launches=stop);atomic(root/'pipeline_state.json',state);ticks+=1
        if ticks%4==1:print('RX_TRUST_PM_STATUS',state['status'],'complete',sum(v['status']=='COMPLETE' for v in state['jobs'].values()),'active',len(active),flush=True)
        if not active and (stop or all(v['status']=='COMPLETE' for v in state['jobs'].values())):break
        time.sleep(12)
    state['status']='STOPPED_FOR_TECHNICAL_FAILURE' if stop else 'RX_TRUST_PM_THREE_SEED_COMPLETE';atomic(root/'pipeline_state.json',state)
    if not stop:write(root/'complete.json',dict(status='RX_TRUST_PM_THREE_SEED_COMPLETE',methods=list(METHODS),model_seeds=list(SEEDS),models=3,raw_only=True,truth_last=True,goal_complete=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['dispatch','worker']);p.add_argument('--kind');p.add_argument('--config',required=True)
    a=p.parse_args();c=read(a.config);dispatch(c) if a.action=='dispatch' else worker(c,a.kind)
