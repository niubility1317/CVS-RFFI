"""Eight missing covariance F models;all12 fixed rows,three seeds,raw-only tests."""
import argparse
from contextlib import contextmanager
from copy import deepcopy
import os
from pathlib import Path
import time
from experiments.cric_dura_comparison.io import read,write
from .covariance_objective import METHODS,RECIPE,Objective
from .covariance_seed_masks import SEEDS,NEW_SEEDS,RUN as MASK_RUN,activate,validate_context as validate_source_context

RUN='20261027-phase1-PAMF-Covariance-SIG1-threeSeed-m12-r01'
RELEASE='20261027-phase1-PAMF-Covariance-SIG1-threeSeed-release-r01'
OLD_SOURCE='20261026-phase1-PAMF-SIG1-Covariance-source-m4-r01'
OLD_CONFIRM='20261026-phase1-PAMF-SIG1-Covariance-Fonly-confirm-m3-r03'
BASELINE='20261026-phase1-PAMF-SIG1-matched-baselines-m6-r01'
TASKS=('cross_rx','cross_both')


def validate_context(ctx):
    validate_source_context(ctx)
    return ctx


def config(ctx,commit,method,seed,alpha,calibration=False):
    validate_context(ctx)
    if method not in METHODS or seed not in NEW_SEEDS:raise ValueError('Only eight missing method/seed models;01 is reused')
    c=deepcopy(next(r['baseline_config'] for r in ctx['cases'] if r['model_seed']==seed));rid='F0-'+method+'-s'+str(seed)
    c.update(run_id=RUN,row_id=rid,paper_method=method,algorithm_id='pamf_sig1_covariance_three_seed_v1',code_commit=commit,
        covariance_recipe=deepcopy(RECIPE),covariance_masks=ctx['project']+'/runs/'+MASK_RUN+'/s'+str(seed)+'/masks.pt',
        covariance_coefficient=alpha,covariance_calibration=calibration,channel_code=ctx['channel_code'],
        matched_baseline_source=ctx['project']+'/runs/'+BASELINE+'/F/F0-SIG1-PL-s'+str(seed)+'/source',
        output_root=ctx['project']+'/runs/'+RUN+'/F/'+rid+'/source')
    c['repl_row'].update(row_id=rid,method=method,group='covariance_three_seed')
    return validate_train(c)


def validate_train(c):
    seed=c.get('model_seed');method=c.get('paper_method');cal=c.get('covariance_calibration');alpha=c.get('covariance_coefficient')
    rid='F0-'+str(method)+'-s'+str(seed);row=c.get('repl_row',{})
    from experiments.pamf_channel_pl.sig1_matched_baselines import prefix
    if (seed not in NEW_SEEDS or method not in METHODS or c.get('run_id')!=RUN or c.get('row_id')!=rid
        or c.get('algorithm_id')!='pamf_sig1_covariance_three_seed_v1' or c.get('covariance_recipe')!=RECIPE
        or type(cal) is not bool or type(alpha) not in (int,float) or not (alpha==0 if cal else 0<alpha<=1)
        or c.get('covariance_masks')!=c['project']+'/runs/'+MASK_RUN+'/s'+str(seed)+'/masks.pt'
        or c.get('shared_SIG1_E20')!=prefix(c['project'],seed) or c.get('matched_baseline_source')!=c['project']+'/runs/'+BASELINE+'/F/F0-SIG1-PL-s'+str(seed)+'/source'
        or c.get('method_id')!='pamf_t1_e50' or c.get('epochs')!=50 or c.get('optimizer_updates')!=11100 or c.get('trajectory')!=[20,10,20]
        or c.get('num_classes')!=6 or c.get('source_rxs')!=[1,3,4,6,8] or c.get('source_days')!=[1,2,3]
        or c.get('sig_time_views')!=1 or row.get('model_seed')!=seed or row.get('structure')!='SIG_TIME' or any(row.get('blocks',{}).values()) or not all(row.get(k) for k in ('L_A','L_B','U'))
        or c.get('regime')!='matched_sat_plain_pl' or c.get('ablation')!='F21' or c.get('plain_PL_threshold')!=.95 or c.get('plain_PL_weight')!=1.
        or not c.get('U_use') or c.get('U_start_epoch')!=14 or c.get('labeled_batch')!=128 or c.get('unlabeled_batch')!=256 or not c.get('default_head_joint')
        or set(c.get('frozen_sources',{}))!={'F','AF','BASE','FOLD'} or any(c.get(k) for k in ('own_EMA','native_U_gate','native_unlabeled_loop','satellite_concat','satellite_CE_weight','satellite_consistency_weight','posthoc_head_fit'))):
        raise ValueError('Fixed fullPAMF SIG1/ordinaryPL matched02/03 covariance trajectory required')
    if any(k in c for k in ('source_G','tests','truth','target_inputs','target_truth','views_root','dataset_path','checkpoint','teacher_checkpoint','core_mechanisms','supervised_architecture')):
        raise ValueError('Training rejects G/target/foreign loss or head inputs')
    root=c['project']+'/runs/'+RUN+'/F/'+rid+'/source'
    if 'segment_epochs' in c:
        offset=c.get('epoch_offset');length={20:10,30:20}.get(offset);parent=c['shared_SIG1_E20'] if offset==20 else root+'/segment30'
        if cal or length is None or c['segment_epochs']!=length or c.get('step_offset')!=offset*222 or c.get('parent_source')!=parent or c['output_root']!=root+'/segment'+str(offset+length):raise ValueError('Own10+20 continuation required')
    elif c['output_root']!=root or any(k in c for k in ('epoch_offset','step_offset','parent_source')):raise ValueError('One exclusive source row root required')
    return c


def load_prefix(c):
    validate_train(c)
    from experiments.pamf_channel_pl.sig1_matched_baselines import load_prefix as original
    base=read(Path(c['matched_baseline_source'])/'complete.json')['config']
    if base['model_seed']!=c['model_seed'] or base['source_cache']!=c['source_cache'] or read(base['source_contract'])!=read(c['source_contract']):raise ValueError('Same-seed actual source handles differ')
    return original(base)


def load_masks(c):
    import torch
    p=torch.load(c['covariance_masks'],map_location='cpu',weights_only=False)
    if (p['schema']!='pamf_sig1_selective_covariance_masks_v2_natural_support' or p['recipe']['model_seed']!=c['model_seed']
        or p['source_prefix']!=c['shared_SIG1_E20'] or p['source_contract']!=read(c['source_contract']) or p['fit_role']!='L_s'
        or p['U_truth_read'] or p['V_used'] or p['target_used']):raise ValueError('Own seed/basis/contract L-only masks required')
    for level,channels in [('time_down',48),('t2.pool',96)]:
        total=channels*(channels-1)//2;n=p['diagnostics'][level]['selected_edges']
        if not 0<n<=int(total*.1):raise ValueError('Original natural support rule mustremain nonempty/capped')
        for key,count in [('selective',n),('random',n),('full',total)]:
            m=p['masks'][level][key]
            if m.dtype!=torch.bool or m.shape!=(channels,channels) or torch.tril(m).any() or int(m.sum())!=count:raise ValueError('Matched selective/random/full mask shape/count differs')
    return p['masks']


@contextmanager
def trajectory():
    # Reuse the reviewed original trajectory and objective. These bindings are
    # process-local;each launcher worker owns exactly one sequential row.
    from . import covariance_source as original
    old_validate,old_prefix,old_installed=original.validate_config,original.load_prefix,original.installed
    original.validate_config=validate_train;original.load_prefix=load_prefix
    original.installed=lambda c,holder:old_installed(c,holder,objective_factory=lambda cfg:Objective(cfg,masks=load_masks(cfg)))
    try:yield original
    finally:original.validate_config,original.load_prefix,original.installed=old_validate,old_prefix,old_installed


def calibrate(c):
    ctx=validate_context(read(c['context']));activate(ctx);seed=c['model_seed'];root=Path(c['output_root']);root.mkdir(parents=True,exist_ok=False)
    cfg=config(ctx,c['code_commit'],'CovSel-U',seed,0.,True)
    with trajectory() as original:original.calibration(dict(c,run_root=str(root)),prepared_config=cfg,prepare_masks=False)
    alpha=read(root/'calibration.json')['coefficient']
    for method in METHODS:write(Path(c['run_root'])/'configs'/('train__'+method+'-s'+str(seed)+'.json'),config(ctx,c['code_commit'],method,seed,alpha))
    write(root/'complete.json',dict(status='MULTISEED_CALIBRATED',model_seed=seed,coefficient=alpha,configs_frozen_before_training=True,optimizer_updates=0,target_reads=0))


def evaluate_source(c):
    import numpy as np
    import torch
    from experiments.pamf_repl_completion.source import load_reader
    from experiments.cric_dura_comparison.f_data import SourceDataset
    from experiments.pamf_channel_pl.restored_source import source_channel,snapshot,unchanged,SCENES
    contract=read(c['source_contract']);reader,_=load_reader(c['output_root'],contract,'cuda:0');data=SourceDataset(c['source_cache'],'V',contract)
    if len(data)!=27000:raise ValueError('Full fixed sourceV required')
    out=Path(c['output_root']).parent/'source_validation';out.mkdir(parents=True,exist_ok=False);before=snapshot(reader);values=np.empty((4,27000,6),np.float32);seen=0
    with torch.inference_mode():
        for batch in torch.utils.data.DataLoader(data,batch_size=256,shuffle=False,num_workers=0):
            x=batch[0].to('cuda:0');views=[x,*[source_channel(x,batch[3],s,contract) for s in SCENES]]
            for k,v in enumerate(views):
                p=reader(v)
                if p.shape!=(len(x),6) or not torch.isfinite(p).all():raise ValueError('Finite frozen defaulthead required')
                values[k,seen:seen+len(x)]=np.asarray(p.cpu().tolist(),np.float32)
            seen+=len(x)
    if seen!=27000 or not unchanged(reader,before):raise ValueError('Source F changed or V incomplete')
    views=['raw',*SCENES];np.savez(out/'source_V_predictions.npz',logits=values,y=data.metadata['y'],ids=data.metadata['ids'],rx=data.metadata['rx_i'],day=data.metadata['day_i'],views=np.asarray(views))
    write(out/'complete.json',dict(status='MULTISEED_SOURCE_EVALUATED',model_seed=c['model_seed'],method=c['paper_method'],V_seen=27000,raw_only=True,G_reads=0,target_reads=0,
        results=[dict(view=v,correct=int((values[j].argmax(1)==data.metadata['y']).sum()),n=27000) for j,v in enumerate(views)]))


def validate_evaluation(c,scoring=False):
    from experiments.cric_dura_comparison.evaluation import reject_truth
    if (c.get('run_id')!=RUN or c.get('model_seed') not in SEEDS or c.get('method') not in METHODS or c.get('readouts')!=['raw'] or c.get('protocol_id')!='F0' or c.get('source_days')!=[1,2,3]
        or c.get('row_id')!='F0-'+c['method']+'-s'+str(c['model_seed']) or set(c['tests'])!=set(TASKS) or len(c['classes'])!=6
        or any(k in c for k in ('G','source_G','restored_cache','restored_caches','teacher_checkpoint'))):raise ValueError('Fixed raw-only fourmethod/three-seed evaluation required')
    handles={'source_output','source_contract','source_cache','frozen_sources'}
    if scoring:
        if handles&set(c):raise ValueError('Independent scorer musthave no model/source handles')
    else:
        reject_truth(c)
        if not handles.issubset(c) or set(c['frozen_sources'])!={'F','AF','BASE','FOLD'}:raise ValueError('Direct F handles only')
    return c


def check_frozen(c):
    done=read(Path(c['source_output'])/'complete.json');cfg=done['config']
    if c['model_seed']==SEEDS[0]:
        from .covariance_source import validate_config
        validate_config(cfg)
    else:validate_train(cfg)
    state=read(Path(c['source_output'])/'covariance_state.json')
    if (done['status']!='SOURCE_FROZEN' or done['epoch']!=50 or done['updates']!=11100 or done['target_access'] or done['U_truth_read'] or done['head_fit_updates']
        or cfg['model_seed']!=c['model_seed'] or cfg['row_id']!=c['row_id'] or cfg['source_cache']!=c['source_cache'] or read(cfg['source_contract'])!=read(c['source_contract'])
        or state['calls']!=6660 or state['seen']!=1701000 or state['unique_seen']!=56700 or state['min_visits']!=30 or state['max_visits']!=30):raise ValueError('Actual full-budget own seed F required')
    return cfg


def prepare(rt):
    ctx=validate_context(read(rt['context']));evaluation=read(rt['evaluation_context']);root=Path(rt['run_root']);jobs=[]
    def add(key,kind,path,completion,status,deps):
        jobs.append(dict(id=key,kind=kind,config=str(path),completion=str(completion),successful_statuses=[status],dependencies=[str(v) for v in deps],execution_release=rt['release_root'],execution_python=rt['python'],entrypoint='experiments.pamf_u_history.covariance_three_seed'))
    reused=[]
    for seed in NEW_SEEDS:
        path=root/'configs'/('calibrate-s'+str(seed)+'.json');calroot=root/'calibration'/('s'+str(seed))
        clean={k:v for k,v in rt.items() if k!='evaluation_context'};write(path,dict(clean,model_seed=seed,output_root=str(calroot)))
        add('calibrate-s'+str(seed),'calibrate',path,calroot/'complete.json','MULTISEED_CALIBRATED',[])
        for method in METHODS:
            rid='F0-'+method+'-s'+str(seed);source=root/'F'/rid/'source';path=root/'configs'/('train__'+method+'-s'+str(seed)+'.json')
            add('train__'+method+'-s'+str(seed),'train',path,source.parent/'source_validation/complete.json','MULTISEED_SOURCE_EVALUATED',[calroot/'complete.json'])
    for seed in SEEDS:
        for method in METHODS:
            rid='F0-'+method+'-s'+str(seed)
            if seed==SEEDS[0] and method!='CovFull-U':
                reused.append(dict(method=method,model_seed=seed,readout_root=ctx['project']+'/runs/'+OLD_CONFIRM+'/readouts/'+rid));continue
            source=ctx['project']+'/runs/'+(OLD_SOURCE if seed==SEEDS[0] else RUN)+'/F/'+rid+'/source';out=root/'readouts'/rid
            p=dict(run_id=RUN,row_id=rid,method=method,model_seed=seed,protocol_id='F0',source_days=[1,2,3],classes=evaluation['classes'],readouts=['raw'],
                tests={t:dict(views_root=evaluation['tests'][t]['views_root']) for t in TASKS},source_output=source,source_contract=ctx['source_contract'],source_cache=ctx['source_cache'],
                frozen_sources=ctx['frozen_sources'],code_commit=rt['code_commit'],output_root=str(out/'predictions'))
            validate_evaluation(p);path=root/'configs'/('predict__'+rid+'.json');write(path,p)
            deps=[] if seed==SEEDS[0] else [Path(source).parent/'source_validation/complete.json']
            add('predict__'+rid,'predict',path,out/'predictions/complete.json','ALL_PREDICTIONS_COMPLETE',deps)
            s=deepcopy(p);s.update(prediction_output=p['output_root'],output_root=str(out/'scores'),tests={t:evaluation['tests'][t] for t in TASKS})
            for k in ('source_output','source_contract','source_cache','frozen_sources'):s.pop(k)
            validate_evaluation(s,True);path=root/'configs'/('score__'+rid+'.json');write(path,s)
            add('score__'+rid,'score',path,out/'scores/complete.json','SCORED_COMPLETE',[out/'predictions/complete.json'])
    write(root/'reused_rows.json',dict(model_seeds=list(SEEDS),methods=list(METHODS),existing_seed01_rows=reused,new_training=8,new_prediction_rows=9,source_score_seed_filtering=False))
    write(root/'jobs.json',jobs);return jobs


def worker(c,kind):
    if kind=='score':
        validate_evaluation(c,True)
        from experiments.cric_dura_comparison import evaluation
        evaluation.tasks=lambda _:list(TASKS);return evaluation.score(c)
    if kind=='predict':
        from .fonly_confirm import activate_raw,predict
        activate_raw(c);return predict(c,validator=validate_evaluation,source_validator=check_frozen)
    if kind=='calibrate':return calibrate(c)
    if kind!='train':raise ValueError('Only fixed calibration/train/raw prediction/independent score')
    validate_train(c);activate(dict(frozen_sources=c['frozen_sources'],channel_code=c['channel_code']))
    with trajectory() as original:original.train(c)
    evaluate_source(c)


def dispatch(rt):
    from experiments.pamf_repl_completion.runner import capacity,processes,foreign_CPU_score
    from experiments.pamf_peer_supervised.runner import launch
    from experiments.cric_dura_comparison.pipeline import done,ready
    from experiments.practical_full_2x2.cache import atomic
    if rt['run_id']!=RUN:raise ValueError('One fixed12model/three-seed matrix required')
    root=Path(rt['run_root']);root.mkdir(parents=True,exist_ok=False);jobs=prepare(rt);by={j['id']:j for j in jobs};active={};stop=False
    state=dict(status='RUNNING',pid=os.getpid(),code_commit=rt['code_commit'],jobs={j['id']:dict(status='QUEUED') for j in jobs})
    while True:
        for key,(child,item) in list(active.items()):
            code=child.poll()
            if code is None:continue
            good=code==0 and done(by[key]);state['jobs'][key]=dict(item,status='COMPLETE' if good else 'FAILED',exit_code=code);stop|=not good;del active[key]
        pending=processes()
        for job in sorted(jobs,key=lambda j:dict(score=0,predict=1,calibrate=2,train=3)[j['kind']]):
            key=job['id']
            if stop or state['jobs'][key]['status']!='QUEUED' or not ready(job):continue
            gpu=None
            if job['kind']=='score':
                if any(by[k]['kind']=='score' for k in active) or foreign_CPU_score({v[0].pid for v in active.values()}):continue
            else:
                free=[]
                for card in rt['allowed_gpus']:
                    pids,memory=capacity(card)
                    if not (pids|{pid for pid,g in pending.items() if g==card}) and memory>=10000:free.append((-memory,card))
                if not free:continue
                gpu=min(free)[1]
            child,item=launch(rt,job,gpu);active[key]=(child,item);state['jobs'][key]=item
            if gpu is not None:pending[child.pid]=gpu
        state.update(updated_at=time.time(),systemic_stop_new_launches=stop);atomic(root/'pipeline_state.json',state)
        if not active and (stop or all(v['status']=='COMPLETE' for v in state['jobs'].values())):break
        time.sleep(12)
    state['status']='STOPPED_FOR_TECHNICAL_FAILURE' if stop else 'THREE_SEED_COVARIANCE_COMPLETE';atomic(root/'pipeline_state.json',state)
    if not stop:write(root/'complete.json',dict(status='THREE_SEED_COVARIANCE_COMPLETE',model_seeds=list(SEEDS),methods=list(METHODS),logical_models=12,new_training=8,reused_seed01_models=4,reused_seed01_readouts=3,new_prediction_rows=9,raw_only=True,truth_last=True,goal_complete=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['dispatch','worker']);p.add_argument('--kind');p.add_argument('--config',required=True)
    a=p.parse_args();c=read(a.config);dispatch(c) if a.action=='dispatch' else worker(c,a.kind)
