"""Frozen direct-F clean/six-scene predictions; independent truth-last scoring."""
import argparse
from copy import deepcopy
import os
from pathlib import Path
import time
from experiments.cric_dura_comparison.io import read,write
from experiments.practical_full_2x2.plan import VIEWS

RUN='20261026-phase1-PAMF-SIG1-Covariance-Fonly-confirm-m3-r03'
RELEASE='20261026-phase1-PAMF-SIG1-Covariance-Fonly-confirm-release-r03'
SOURCE_RUN='20261026-phase1-PAMF-SIG1-Covariance-source-m4-r01'
BASELINE_RUN='20261026-phase1-PAMF-SIG1-matched-baselines-m6-r01'
METHODS=('CovSel-U','CovRandom-U','CovSel-LU')
TASKS=('cross_rx','cross_both')
SEED=2026092701


def validate_context(c):
    required={'project','python','frozen_sources','source_run_root','source_contract','source_cache','classes','tests','reference_predictions','reference_scores'}
    if (set(c)!=required or c['source_run_root']!=c['project']+'/runs/'+SOURCE_RUN or set(c['tests'])!=set(TASKS)
        or len(c['classes'])!=6 or 'G' in c['frozen_sources'] or set(c['reference_predictions'])!={'SIG1-PL','SIG1-Sup'}
        or set(c['reference_scores'])!={'SIG1-PL','SIG1-Sup'}):raise ValueError('Direct F only;fixed three source candidates and completed matched references required')
    return c


def config(c,commit,method,scoring=False):
    validate_context(c)
    if method not in METHODS:raise ValueError('Only the three independently passing direct-F candidates')
    rid='F0-'+method+'-s'+str(SEED);base=c['project']+'/runs/'+RUN+'/readouts/'+rid
    p=dict(run_id=RUN,row_id=rid,model_seed=SEED,method=method,protocol_id='F0',source_days=[1,2,3],classes=c['classes'],readouts=['raw'],
        tests={t:dict(views_root=c['tests'][t]['views_root']) for t in TASKS},source_output=c['source_run_root']+'/F/'+rid+'/source',
        source_contract=c['source_contract'],source_cache=c['source_cache'],frozen_sources=c['frozen_sources'],
        source_pilot=c['source_run_root'],output_root=base+'/predictions',code_commit=commit)
    if scoring:
        p.update(prediction_output=p['output_root'],output_root=base+'/scores',tests=deepcopy(c['tests']))
        for k in ('source_output','source_contract','source_cache','frozen_sources','source_pilot'):p.pop(k)
    return validate_config(p,scoring)


def validate_config(c,scoring=False):
    from experiments.cric_dura_comparison.evaluation import reject_truth
    if (c.get('run_id')!=RUN or c.get('method') not in METHODS or c.get('model_seed')!=SEED or c.get('protocol_id')!='F0'
        or c.get('row_id')!='F0-'+c['method']+'-s'+str(SEED) or c.get('readouts')!=['raw'] or set(c['tests'])!=set(TASKS)
        or len(c['classes'])!=6 or c['source_days']!=[1,2,3]
        or any(k in c for k in ('restored_caches','restored_cache','same_G','G','source_G','teacher_checkpoint'))):
        raise ValueError('Fixed raw-only two-population/seven-view confirmation required')
    source_handles={'source_output','source_contract','source_cache','frozen_sources','source_pilot'}
    if scoring:
        if source_handles&set(c):raise ValueError('Independent scorer cannot receive model/source handles')
    else:
        reject_truth(c)
        if not source_handles.issubset(c) or 'G' in c['frozen_sources']:raise ValueError('F-only source handles without G required')
    return c


def source_check(c):
    from .covariance_source import validate_config as validate_source
    from .fonly_source_analysis import counts,rule
    p=Path(c['source_output']);done=read(p/'complete.json');cfg=validate_source(done['config'])
    if (done['status']!='SOURCE_FROZEN' or done['epoch']!=50 or done['updates']!=11100 or done['target_access'] or done['U_truth_read']
        or done['head_fit_updates'] or cfg['row_id']!=c['row_id'] or cfg['paper_method']!=c['method']
        or cfg['source_cache']!=c['source_cache'] or read(cfg['source_contract'])!=read(c['source_contract'])):
        raise ValueError('Actual frozen same-contract SIG1 E50 required')
    a=counts(read(p.parent/'source_validation/complete.json'),c['method'])
    b=counts(read(Path(c['source_pilot'])/'baseline_source/complete.json'),'SIG1-PL')
    result=rule(a,b,read(p/'covariance_state.json'))
    if not result['source_necessary_pass']:raise ValueError('Fixed standalone-F source necessary conditions failed')
    return cfg


def predict(c,validator=None,source_validator=None):
    import numpy as np
    import torch
    from experiments.cric_dura_comparison.io import tensor
    from experiments.cric_dura_comparison.evaluation import handle
    from experiments.pamf_repl_completion.source import load_reader
    from experiments.cric_dura_comparison.f_data import SourceDataset
    from experiments.pamf_channel_pl.restored_source import snapshot,unchanged
    (validator or validate_config)(c);cfg=(source_validator or source_check)(c);reader,_=load_reader(c['source_output'],read(c['source_contract']),'cuda:0')
    labeled=SourceDataset(c['source_cache'],'L_s',read(c['source_contract']))
    with torch.inference_mode():p=reader(labeled.x[:2].to('cuda:0'))
    if p.shape!=(2,6) or not torch.isfinite(p).all() or len(reader.sequence_pool.views)!=1:raise ValueError('Actual single-view defaulthead no-query smoke failed')
    out=Path(c['output_root']);out.mkdir(parents=True,exist_ok=False)
    write(out.parent/'source_checkpoint_smoke.json',dict(status='PASS',source_samples=2,query_reads=0,optimizer_updates=0,signature_views=1,readouts=['raw']))
    del labeled;before=snapshot(reader);started=time.monotonic();counts_by_task={}
    for task in TASKS:
        manifest,ids,positions=handle(c['tests'][task]['views_root'],c['classes'])
        if len(ids)!={'cross_rx':126000,'cross_both':42000}[task]:raise ValueError('Original F0 physical population differs')
        values={}
        for view in VIEWS:
            path=manifest['clean_ref'] if view=='clean' else str(Path(manifest.get('observations_root',c['tests'][task]['views_root']))/(view+'.npy'))
            array=np.load(path,mmap_mode='r',allow_pickle=False);pred=[]
            with torch.inference_mode():
                for begin in range(0,len(ids),128):
                    index=positions[begin:begin+128] if positions is not None else slice(begin,begin+128)
                    x=tensor(np.ascontiguousarray(array[index],np.float32),'cuda:0');logits=reader.classify_features(reader.features(x))
                    if logits.shape!=(len(x),6) or not torch.isfinite(logits).all():raise ValueError('Finite frozen original default-head logits required')
                    pred.extend(logits.argmax(1).cpu().tolist())
            values['raw__'+view]=np.asarray(pred,np.int64)
            write(out/'progress.json',dict(task=task,view=view,count=len(pred),readouts=['raw'],truth_read=False),exclusive=False)
            print('FONLY_PREDICTION',c['row_id'],task,view,len(pred),flush=True)
        folder=out/task;folder.mkdir();np.savez(folder/'predictions.npz',ids=ids,**values)
        counts_by_task[task]=dict(count=len(ids),views=list(VIEWS),readouts=['raw'])
    if not unchanged(reader,before):raise ValueError('Frozen direct F state changed')
    write(out/'complete.json',dict(status='ALL_PREDICTIONS_COMPLETE',run_id=c['run_id'],row_id=c['row_id'],tests=counts_by_task,
        views=list(VIEWS),readouts=['raw'],truth_read=False,query_fit=False,parameters_and_buffers_unchanged=True,seconds=time.monotonic()-started))


def worker(c,kind):
    if kind=='score':
        validate_config(c,True)
        from experiments.cric_dura_comparison import evaluation
        evaluation.tasks=lambda _:list(TASKS)
        return evaluation.score(c)
    if kind!='predict':raise ValueError('Only direct-F prediction and independent CPU truth')
    activate_raw(c)
    return predict(c)


def activate_raw(c):
    # F's historical modules are sufficient for frozen classification. Do not
    # use the combined bootstrap, which also requires G for source simulation.
    import importlib
    import sys
    roots={k:Path(v) for k,v in c['frozen_sources'].items()}
    if set(roots)!={'F','AF','BASE','FOLD'} or any(not p.is_dir() for p in roots.values()):raise ValueError('Four immutable F dependencies required;G excluded')
    package=importlib.import_module('experiments');current=Path(__file__).resolve().parents[2]
    package.__path__=[str(roots[k]/'experiments') for k in ('F','AF','FOLD','BASE')]+[str(current/'experiments')]
    sys.path[:0]=[str(current/'code'),str(roots['BASE']),str(roots['BASE']/'code')]
    # Original trusted checkpoint metadata (SatViewStage); no channel is run.
    importlib.import_module('baseline_origin_sat_view')
    from experiments.practical_full_2x2.compat import numerical_policy
    numerical_policy()


def dispatch(rt):
    from experiments.pamf_repl_completion.runner import capacity,processes,foreign_CPU_score
    from experiments.pamf_peer_supervised.runner import launch
    from experiments.cric_dura_comparison.pipeline import done,ready
    from experiments.practical_full_2x2.cache import atomic
    if rt['run_id']!=RUN or rt['allowed_gpus']!=list(range(8)):raise ValueError('One fixed direct-F run required')
    ctx=validate_context(read(rt['context']));root=Path(rt['run_root']);root.mkdir(parents=True,exist_ok=False);jobs=[]
    for method in METHODS:
        for kind in ('predict','score'):
            c=config(ctx,rt['code_commit'],method,kind=='score');path=root/'configs'/(kind+'__'+method+'.json');write(path,c)
            jobs.append(dict(id=kind+'__'+method,kind=kind,config=str(path),completion=c['output_root']+'/complete.json',
                successful_statuses=['ALL_PREDICTIONS_COMPLETE' if kind=='predict' else 'SCORED_COMPLETE'],dependencies=[] if kind=='predict' else [c['prediction_output']+'/complete.json'],
                execution_release=rt['release_root'],execution_python=rt['python'],entrypoint='experiments.pamf_u_history.fonly_confirm'))
    write(root/'jobs.json',jobs);by={j['id']:j for j in jobs};state=dict(status='RUNNING',pid=os.getpid(),code_commit=rt['code_commit'],jobs={j['id']:dict(status='QUEUED') for j in jobs});active={};stop=False
    while True:
        for key,(child,item) in list(active.items()):
            code=child.poll()
            if code is None:continue
            good=code==0 and done(by[key]);state['jobs'][key]=dict(item,status='COMPLETE' if good else 'FAILED',exit_code=code);stop|=not good;del active[key]
        pending=processes()
        for job in sorted(jobs,key=lambda j:j['kind']!='score'):
            key=job['id']
            if stop or state['jobs'][key]['status']!='QUEUED' or not ready(job):continue
            gpu=None
            if job['kind']=='score':
                if any(by[k]['kind']=='score' for k in active) or foreign_CPU_score({v[0].pid for v in active.values()}):continue
            else:
                available=[]
                for g in rt['allowed_gpus']:
                    pids,free=capacity(g)
                    if not (pids|{pid for pid,card in pending.items() if card==g}) and free>=10000:available.append((-free,g))
                if not available:continue
                gpu=min(available)[1]
            child,item=launch(rt,job,gpu);active[key]=(child,item);state['jobs'][key]=item
            if gpu is not None:pending[child.pid]=gpu
        state.update(updated_at=time.time(),systemic_stop_new_launches=stop);atomic(root/'pipeline_state.json',state)
        if not active and (stop or all(v['status']=='COMPLETE' for v in state['jobs'].values())):break
        time.sleep(12)
    state['status']='STOPPED_FOR_TECHNICAL_FAILURE' if stop else 'FONLY_CONFIRMATION_COMPLETE';atomic(root/'pipeline_state.json',state)
    if not stop:write(root/'complete.json',dict(status='FONLY_CONFIRMATION_COMPLETE',methods=list(METHODS),model_seeds=[SEED],readouts=['raw'],populations=list(TASKS),views=list(VIEWS),truth_last=True,new_training=0,G_reads=0,goal_complete=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['dispatch','worker']);p.add_argument('--kind');p.add_argument('--config',required=True)
    a=p.parse_args();c=read(a.config);dispatch(c) if a.action=='dispatch' else worker(c,a.kind)
