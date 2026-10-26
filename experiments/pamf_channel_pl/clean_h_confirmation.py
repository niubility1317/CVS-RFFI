"""One source-selected clean/H method, three F seeds, raw-only truth-last tests."""
import argparse,csv,os,statistics,time
from collections import Counter
from copy import deepcopy
from pathlib import Path
from experiments.cric_dura_comparison.io import read,write
from . import clean_h_source as source
from . import clean_h_evaluation as evaluation
from .sig1_matched_baselines import prefix

RUN='20261027-phase1-PAMF-SIG1-REPL-CleanH-confirm-m3-r01'
RELEASE='20261027-phase1-PAMF-SIG1-REPL-CleanH-confirm-release-r01'
SEEDS=evaluation.SEEDS


def selection(project):
    path=project+'/runs/'+source.RUN+'/complete.json'
    winner=evaluation.selected_method(dict(selection_source=path,purpose='baseline_cross_date'))
    return winner,path


def template(project,commit,seed):
    if seed not in SEEDS[1:]:raise ValueError('Only missing seeds02/03 train; reuse completed seed01')
    method,path=selection(project);c=deepcopy(read(Path(path).parent/'configs'/(method+'.json')))
    source.validate_config(c)
    rid='F0-'+method+'-s'+str(seed)
    c.update(run_id=RUN,row_id=rid,model_seed=seed,code_commit=commit,shared_SIG1_E20=prefix(project,seed),
             output_root=project+'/runs/'+RUN+'/F/'+rid+'/source',clean_H_selection=path)
    c['repl_row'].update(row_id=rid,model_seed=seed)
    return c


def validate_config(c):
    expected=template(c['project'],c['code_commit'],c['model_seed']);base=expected['output_root']
    if 'segment_epochs' in c:
        offset=c.get('epoch_offset');length={20:10,30:20}.get(offset)
        if length is None:raise ValueError('Only ownE20+10+20 continuation')
        expected.update(segment_epochs=length,epoch_offset=offset,step_offset=offset*222,
                        parent_source=expected['shared_SIG1_E20'] if offset==20 else base+'/segment30',
                        output_root=base+'/segment'+str(offset+length))
        if c.get('candidate_smoke') is True:expected['candidate_smoke']=True
    if c!=expected:raise ValueError('Exact source-selected method, own seed parent and original budget required')
    return c


def validate_context(c):
    if set(c)!={'project','python','frozen_sources','classes','tests'} or set(c['frozen_sources'])!={'F','AF','BASE','FOLD'} or set(c['tests'])!=set(evaluation.TASK_COUNTS) or len(c['classes'])!=6:raise ValueError('F-only complete original F0 evaluation context required')
    for task in c['tests'].values():
        if set(task)!={'views_root','truth','kind'}:raise ValueError('Declared received-IQ and independent scorer handles required')


def validate_runtime(rt):
    if (set(rt)!={'run_id','project','python','release_root','run_root','context','runtime_file','code_commit','frozen_sources','allowed_gpus','launch_owner'}
        or rt['run_id']!=RUN or rt['run_root']!=rt['project']+'/runs/'+RUN or rt['release_root']!=rt['project']+'/releases/'+RELEASE
        or rt['context']!=rt['release_root']+'/context.json' or rt['runtime_file']!=rt['release_root']+'/runtime.json'
        or rt['allowed_gpus']!=list(range(8)) or rt['launch_owner']!='codex-pamf-clean-h-confirm-20261027'
        or set(rt['frozen_sources'])!={'F','AF','BASE','FOLD'}):raise ValueError('One immutable raw-only three-seed confirmation run')


def smoke(c):
    import numpy as np
    from experiments.pamf_repl_completion.source import load_reader
    rt=c['runtime'];cfg=read(c['train_config']);validate_config(cfg)
    checks=Path(rt['run_root'])/'checks';checks.mkdir()
    source.smoke(dict(rt,run_root=str(checks)),[cfg],validate_config)
    method,path=selection(rt['project']);old=read(Path(path).parent/'configs'/(method+'.json'))
    reader,_=load_reader(old['output_root'],read(old['source_contract']),'cuda:0');before=source.snapshot(reader)
    x=np.load(Path(old['source_cache'])/'L_s_x.npy',mmap_mode='r',allow_pickle=False)[:2]
    values=evaluation.array_predictions(reader,x,None,2,6,'cuda:0')
    if values.shape!=(2,) or not source.unchanged(reader,before):raise ValueError('Actual frozen source L2 raw evaluation smoke failed')
    write(Path(rt['run_root'])/'smoke.json',dict(status='PASS',source_selected=method,actual_joint_E20_backward=True,
        actual_frozen_E50_original_head_raw_L2=True,optimizer_updates=0,target_reads=0,U_truth_reads=0,restoration_weight_reads=0))


def prepare(rt):
    validate_runtime(rt);ctx=read(rt['context']);validate_context(ctx);method,path=selection(rt['project'])
    root=Path(rt['run_root']);(root/'configs').mkdir();jobs=[];train_configs=[]
    def add(kind,rid,c,end,status,deps):
        file=root/'configs'/(kind+'-'+rid+'.json');write(file,c)
        jobs.append(dict(id=kind+'::'+rid,kind=kind,config=str(file),completion=str(end),successful_statuses=[status],dependencies=deps,
            gpu_pool=list(range(8)),execution_release=rt['release_root'],execution_python=rt['python'],entrypoint='experiments.pamf_channel_pl.clean_h_confirmation'))
        return str(file)
    for seed in SEEDS[1:]:
        c=template(rt['project'],rt['code_commit'],seed);validate_config(c)
        train_configs.append(add('train',c['row_id'],c,Path(c['output_root'])/'complete.json','SOURCE_FROZEN',[str(root/'smoke.json')]))
    for purpose,methods,seeds in [('selected_method',(method,),SEEDS),('baseline_cross_date',('SIG1-Sup','SIG1-PL'),SEEDS)]:
        for m in methods:
            for seed in seeds:
                rid='F0-'+m+'-s'+str(seed)
                source_run=(source.RUN if seed==SEEDS[0] else RUN) if purpose=='selected_method' else evaluation.BASELINE_RUN
                src=Path(rt['project'])/'runs'/source_run/'F'/rid/'source'
                cfg=read(Path(path).parent/'configs'/(method+'.json')) if purpose=='selected_method' and seed==SEEDS[0] else template(rt['project'],rt['code_commit'],seed) if purpose=='selected_method' else read(src/'complete.json')['config']
                tests=ctx['tests'] if purpose=='selected_method' else {'cross_date':ctx['tests']['cross_date']};out=root/'evaluations'/rid
                p=dict(schema='pamf_clean_h_raw_eval_v1',run_id=RUN,row_id=rid,protocol_id='F0',purpose=purpose,method=m,model_seed=seed,
                    source_days=[1,2,3],classes=ctx['classes'],tests={t:dict(views_root=v['views_root']) for t,v in tests.items()},readouts=['raw'],output_root=str(out/'predictions'),
                    project=rt['project'],selection_source=path,source_run_id=source_run,source_row_id=rid,source_output=str(src),source_contract=cfg['source_contract'],frozen_sources=rt['frozen_sources'])
                evaluation.validate(p);add('predict',rid,p,out/'predictions/complete.json','ALL_PREDICTIONS_COMPLETE',[str(root/'smoke.json'),str(src/'complete.json')])
                s={k:v for k,v in p.items() if k not in evaluation.PREDICT_ONLY};s.update(prediction_output=p['output_root'],output_root=str(out/'scores'),tests=tests)
                evaluation.validate(s,scoring=True);add('score',rid,s,out/'scores/complete.json','SCORED_COMPLETE',[str(out/'predictions/complete.json')])
    add('smoke','first',dict(runtime=rt,train_config=train_configs[0]),root/'smoke.json','PASS',[])
    write(root/'jobs.json',jobs);write(root/'selection.json',dict(method=method,source_complete=path,source_only=True,seed01_reused=True,target_feedback_used=False))
    return jobs


def analyze(rt,jobs):
    rows=[];chosen,_=selection(rt['project']);populations=(*evaluation.TASK_COUNTS,'overall_168000')
    def collect(method,seed,data,tasks):
        if not data['truth_last'] or not data['target_feedback_forbidden']:raise ValueError('Independent truth-last evidence required')
        cells=[v for v in data['results'] if v['readout']=='raw' and v['dimension']=='overall']
        for pop in tasks:
            selected=[v for v in cells if v['test'] in ('cross_rx','cross_both')] if pop=='overall_168000' else [v for v in cells if v['test']==pop]
            for metric in ('clean','six_practical'):
                values=[v for v in selected if (v['view']=='clean')==(metric=='clean')];count=sum(v['count'] for v in values)
                expected=(168000 if pop=='overall_168000' else evaluation.TASK_COUNTS[pop])*(1 if metric=='clean' else 6)
                if count!=expected:raise ValueError('Complete physical populations/views required')
                correct=sum(sum(v['confusion'][i][i] for i in range(6)) for v in values)
                rows.append(dict(method=method,model_seed=seed,population=pop,metric=metric,correct=correct,count=count,accuracy=100*correct/count))
    for j in jobs:
        if j['kind']=='score':
            c=read(j['config']);collect(c['method'],c['model_seed'],read(Path(c['output_root'])/'scores.json'),populations if c['purpose']=='selected_method' else ('cross_date',))
    for method in ('SIG1-Sup','SIG1-PL'):
        for seed in SEEDS:
            rid='F0-'+method+'-s'+str(seed);old=Path(rt['project'])/'runs'/evaluation.BASELINE_RUN/'readouts'/rid/'scores/scores.json'
            collect(method,seed,read(old),('cross_rx','cross_both','overall_168000'))
    methods=(chosen,'SIG1-Sup','SIG1-PL');summary=[];paired=[]
    for method in methods:
        for pop in populations:
            for metric in ('clean','six_practical'):
                values=[r['accuracy'] for r in rows if (r['method'],r['population'],r['metric'])==(method,pop,metric)]
                if len(values)!=3:raise ValueError('Each same method requires three complete F seeds')
                summary.append(dict(method=method,population=pop,metric=metric,seeds=3,mean=statistics.mean(values),sample_sd=statistics.stdev(values)))
    for ref in methods[1:]:
        for pop in populations:
            for metric in ('clean','six_practical'):
                deltas=[]
                for seed in SEEDS:
                    lookup={r['method']:r['accuracy'] for r in rows if (r['model_seed'],r['population'],r['metric'])==(seed,pop,metric)}
                    deltas.append(lookup[chosen]-lookup[ref])
                paired.append(dict(reference=ref,population=pop,metric=metric,seeds=3,mean_delta_pp=statistics.mean(deltas),sample_sd_delta_pp=statistics.stdev(deltas),per_seed_delta_pp=deltas))
    primary={r['metric']:r['mean'] for r in summary if r['method']==chosen and r['population']=='overall_168000'}
    out=Path(rt['run_root'])/'analysis';out.mkdir()
    for name,values in [('per_seed.csv',rows),('summary.csv',summary),('paired_summary.csv',paired)]:
        with (out/name).open('x',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(values[0]));w.writeheader();w.writerows(values)
    matched=all(r['mean_delta_pp']>=(1. if r['metric']=='clean' else 1.5) and (r['metric']=='clean' or r['mean_delta_pp']>1.5) for r in paired if r['population']=='overall_168000')
    floors=primary['clean']>=85.98174603174603 and primary['six_practical']>72.0961970899471
    write(out/'complete.json',dict(status='CLEAN_H_CONFIRM_ANALYZED',run_id=RUN,method=chosen,seeds=list(SEEDS),results=summary,paired=paired,
        primary_population=168000,cross_date_diagnostic=30000,new_training_models=2,reused_selected_models=1,retrained_reference_models=0,
        target_clean_floor=85.98174603174603,target_six_floor_strict=72.0961970899471,
        target_floors_pass=floors,matched_baseline_goal_pass=matched,target_goals_pass=floors and matched,
        source_selected_before_target=True,target_feedback_used=False,restoration_used=False,readouts=['raw']))


def dispatch(rt):
    from experiments.practical_full_2x2.cache import atomic
    from experiments.cric_dura_comparison.pipeline import done,ready
    from experiments.pamf_repl_completion.runner import capacity,processes,foreign_CPU_score
    from experiments.pamf_peer_supervised.runner import launch
    validate_runtime(rt);root=Path(rt['run_root']);root.mkdir(parents=True,exist_ok=False);jobs=prepare(rt);by={j['id']:j for j in jobs}
    state=dict(status='RUNNING',pid=os.getpid(),code_commit=rt['code_commit'],jobs={j['id']:dict(status='QUEUED') for j in jobs});active={};stop=False
    while True:
        for key,(child,item) in list(active.items()):
            code=child.poll()
            if code is None:continue
            good=code==0 and done(by[key]);state['jobs'][key]=dict(item,status='COMPLETE' if good else 'FAILED',exit_code=code);stop=stop or not good;del active[key]
        pending=processes()
        for job in sorted(jobs,key=lambda j:dict(smoke=0,score=1,train=2,predict=3)[j['kind']]):
            key=job['id']
            if stop or state['jobs'][key]['status']!='QUEUED' or not ready(job):continue
            if job['kind']=='score':
                if any(by[k]['kind']=='score' for k in active) or foreign_CPU_score({v[0].pid for v in active.values()}):continue
                gpu=None
            else:
                available=[]
                for gpu in job['gpu_pool']:
                    pids,free=capacity(gpu);load=len(pids|{pid for pid,g in pending.items() if g==gpu})
                    if load<2 and free>=10000:available.append((load,-free,gpu))
                if not available:continue
                gpu=min(available)[2]
            child,item=launch(rt,job,gpu);active[key]=(child,item);state['jobs'][key]=item
            if gpu is not None:pending[child.pid]=gpu
        state.update(updated_at=time.time(),counts=dict(Counter(v['status'] for v in state['jobs'].values())),systemic_stop_new_launches=stop);atomic(root/'pipeline_state.json',state)
        if not active and (stop or all(v['status']=='COMPLETE' for v in state['jobs'].values())):break
        time.sleep(12)
    if not stop:analyze(rt,jobs);write(root/'complete.json',read(root/'analysis/complete.json'))
    state['status']='STOPPED_FOR_TECHNICAL_FAILURE' if stop else 'CLEAN_H_CONFIRM_ANALYZED';atomic(root/'pipeline_state.json',state)


def worker(c,kind):
    if kind=='score':return evaluation.score(c)
    if kind=='predict':return evaluation.predict(c)
    source.activate(c.get('runtime',c))
    import torch
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if kind=='smoke':return smoke(c)
    if kind=='train':return source.train(c,validate_config)
    raise ValueError('One selected method training/raw prediction/independent scoring only')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['dispatch','worker']);p.add_argument('--kind');p.add_argument('--config',required=True)
    a=p.parse_args();c=read(a.config);dispatch(c) if a.action=='dispatch' else worker(c,a.kind)
