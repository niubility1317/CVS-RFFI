"""Three source-only H-CE pressure falsifications, same original E20 and raw95 labels."""
import argparse,os,time
from collections import Counter
from copy import deepcopy
from pathlib import Path
from experiments.cric_dura_comparison.io import read,write
from . import clean_h_source as base,no_pair_source as reference
from .h_quality_objective import METHODS as METHOD_MODES,recipe,HQualityObjective
from .receiver_balance_objective import recipe as rb_recipe

RUN='20261027-phase1-PAMF-SIG1-REPL-HQuality-source-m3-r01'
RELEASE='20261027-phase1-PAMF-SIG1-REPL-HQuality-source-release-r01'
OWNER='codex-pamf-H-quality-source-20261027'
METHODS=tuple(METHOD_MODES);BASE_METHOD=reference.METHOD;SEED=2026092701
RAW_MIN=26568;H3_MIN=62109


def validate_context(c):
    if (set(c)!={'project','python','frozen_sources','template','reference','expected_source_contract'} or set(c['frozen_sources'])!={'F','AF','BASE','FOLD'}
        or c['template']!=c['project']+'/runs/'+reference.RUN+'/configs/'+BASE_METHOD+'.json'
        or c['reference']!=c['project']+'/runs/'+reference.RUN+'/evaluations/'+BASE_METHOD+'/complete.json'):
        raise ValueError('Completed same-contract NoPair metadata only; no G/target handles')


def template(ctx,commit,method):
    validate_context(ctx)
    if method not in METHODS:raise ValueError('Three fixed physical H CE modes only')
    c=deepcopy(read(ctx['template']));reference.validate_config(c)
    if c['expected_source_contract']!=ctx['expected_source_contract'] or c['frozen_sources']!=ctx['frozen_sources']:raise ValueError('Original source roles and four F roots must match')
    rid='F0-'+method+'-s'+str(SEED)
    c.update(run_id=RUN,row_id=rid,paper_method=method,algorithm_id='pamf_clean_h_quality_curriculum_v1',code_commit=commit,
        output_root=ctx['project']+'/runs/'+RUN+'/F/'+rid+'/source',receiver_balance=rb_recipe('REPL_RB_RXTX'),H_quality=recipe(method))
    c['repl_row'].update(row_id=rid,method=method)
    return c


def validate_config(c):
    ctx=dict(project=c['project'],python='unused',frozen_sources=c['frozen_sources'],template=c['project']+'/runs/'+reference.RUN+'/configs/'+BASE_METHOD+'.json',
        reference=c['project']+'/runs/'+reference.RUN+'/evaluations/'+BASE_METHOD+'/complete.json',expected_source_contract=c['expected_source_contract'])
    expected=template(ctx,c['code_commit'],c.get('paper_method'));root=expected['output_root']
    if 'segment_epochs' in c:
        offset=c.get('epoch_offset');length={20:10,30:20}.get(offset)
        if length is None:raise ValueError('Own original E20+10+20 only')
        expected.update(segment_epochs=length,epoch_offset=offset,step_offset=offset*222,parent_source=expected['shared_SIG1_E20'] if offset==20 else root+'/segment30',output_root=root+'/segment'+str(offset+length))
        if c.get('candidate_smoke') is True:expected['candidate_smoke']=True
    if c!=expected:raise ValueError('Exact bounded H-CE-only intervention on completed RX_TX recipe; no changed gate/parent/head/budget/data')
    return c


def validate_runtime(rt):
    if (set(rt)!={'run_id','project','python','release_root','run_root','context','runtime_file','code_commit','frozen_sources','allowed_gpus','launch_owner'}
        or rt['run_id']!=RUN or rt['run_root']!=rt['project']+'/runs/'+RUN or rt['release_root']!=rt['project']+'/releases/'+RELEASE
        or rt['context']!=rt['release_root']+'/context.json' or rt['runtime_file']!=rt['release_root']+'/runtime.json'
        or rt['allowed_gpus']!=list(range(8)) or rt['launch_owner']!=OWNER or set(rt['frozen_sources'])!={'F','AF','BASE','FOLD'}):
        raise ValueError('One immutable source-only H quality run')


def source_decision(method,score,ref):
    if method not in METHODS or ref!=dict(raw_correct=26552,H3_correct=62148):raise ValueError('Reuse the completed NoPair source reference')
    return dict(method=method,raw_correct=score['raw_correct'],H3_correct=score['H3_correct'],source_pass=score['raw_correct']>=RAW_MIN and score['H3_correct']>=H3_MIN,
        raw_min=RAW_MIN,H3_min=H3_MIN,delta_raw_pp=100*(score['raw_correct']-ref['raw_correct'])/27000,delta_H3_pp=100*(score['H3_correct']-ref['H3_correct'])/81000)


def prepare(rt):
    validate_runtime(rt);ctx=read(rt['context']);validate_context(ctx);done=read(ctx['reference'])
    if done.get('method')!=BASE_METHOD or done.get('signature_views')!=1:raise ValueError('Completed matching single-view NoPair only')
    score=base.source_score(done);source_decision(METHODS[0],score,score)
    completed=read(Path(done['source_output'])/'complete.json')
    if completed['target_access'] or completed['U_truth_read'] or completed['head_fit_updates'] or read(completed['source_contract'])!=read(ctx['expected_source_contract']):raise ValueError('Reference must retain original source-only contract')
    root=Path(rt['run_root']);(root/'configs').mkdir();jobs=[];configs=[]
    common=dict(gpu_pool=list(range(8)),execution_release=rt['release_root'],execution_python=rt['python'],entrypoint='experiments.pamf_channel_pl.h_quality_source')
    for method in METHODS:
        c=template(ctx,rt['code_commit'],method);validate_config(c);file=root/'configs'/(method+'.json');write(file,c);configs.append(str(file))
        jobs.append(dict(common,id=method,kind='train_eval',config=str(file),completion=str(root/'evaluations'/method/'complete.json'),successful_statuses=['CLEAN_H_SOURCE_EVALUATED'],dependencies=[str(root/'smoke.json')]))
    smoke=root/'configs/smoke.json';write(smoke,dict(runtime=rt,train_configs=configs))
    jobs=[dict(common,id='smoke',kind='smoke',config=str(smoke),completion=str(root/'smoke.json'),successful_statuses=['PASS'],dependencies=[])]+jobs
    write(root/'reference_summary.json',dict(method=BASE_METHOD,score=score,source=ctx['reference'],weights_loaded=False));write(root/'jobs.json',jobs);return jobs


def analyze(rt):
    root=Path(rt['run_root']);ref=read(root/'reference_summary.json')['score'];scores={m:base.source_score(read(root/'evaluations'/m/'complete.json')) for m in METHODS}
    decisions=[source_decision(m,scores[m],ref) for m in METHODS];valid=[r for r in decisions if r['source_pass']]
    chosen=max(valid,key=lambda r:(r['raw_correct'],r['H3_correct'],-METHODS.index(r['method'])))['method'] if valid else None
    write(root/'complete.json',dict(status='HQUALITY_SOURCE_COMPLETE',run_id=RUN,code_commit=rt['code_commit'],model_seed=SEED,
        source_scores=scores,reference=ref,decisions=decisions,source_selected=chosen,U_truth_reads=0,target_reads=0,
        scope='Three variants of one model seed,not three-seed/target gains; source pass only permits later registered confirmation'))


def dispatch(rt):
    from experiments.practical_full_2x2.cache import atomic
    from experiments.cric_dura_comparison.pipeline import done,ready
    from experiments.pamf_repl_completion.runner import capacity,processes
    from experiments.pamf_peer_supervised.runner import launch
    validate_runtime(rt);root=Path(rt['run_root']);root.mkdir(parents=True,exist_ok=False);jobs=prepare(rt);by={j['id']:j for j in jobs}
    state=dict(status='RUNNING',pid=os.getpid(),code_commit=rt['code_commit'],jobs={j['id']:dict(status='QUEUED') for j in jobs});active={};stop=False
    while True:
        for key,(child,item) in list(active.items()):
            code=child.poll()
            if code is None:continue
            good=code==0 and done(by[key]);state['jobs'][key]=dict(item,status='COMPLETE' if good else 'FAILED',exit_code=code);stop=stop or not good;del active[key]
        pending=processes()
        for job in jobs:
            key=job['id']
            if stop or state['jobs'][key]['status']!='QUEUED' or not ready(job):continue
            available=[]
            for gpu in job['gpu_pool']:
                pids,free=capacity(gpu);load=len(pids|{pid for pid,g in pending.items() if g==gpu})
                if load<2 and free>=10000:available.append((load,-free,gpu))
            if not available:continue
            gpu=min(available)[2];child,item=launch(rt,job,gpu);active[key]=(child,item);state['jobs'][key]=item;pending[child.pid]=gpu
        state.update(updated_at=time.time(),counts=dict(Counter(v['status'] for v in state['jobs'].values())),systemic_stop_new_launches=stop);atomic(root/'pipeline_state.json',state)
        if not active and (stop or all(v['status']=='COMPLETE' for v in state['jobs'].values())):break
        time.sleep(12)
    if not stop:analyze(rt)
    state['status']='STOPPED_FOR_TECHNICAL_FAILURE' if stop else 'HQUALITY_SOURCE_COMPLETE';atomic(root/'pipeline_state.json',state)


def worker(c,kind):
    base.activate(c.get('runtime',c))
    import torch
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if kind=='smoke':return base.smoke(c['runtime'],[read(p) for p in c['train_configs']],validate_config,HQualityObjective)
    if kind=='train_eval':
        base.train(c,validate_config,HQualityObjective)
        return base.evaluate(dict(method=c['paper_method'],context=c['source_context'],source_output=c['output_root'],signature_views=1,output_root=c['project']+'/runs/'+RUN+'/evaluations/'+c['paper_method'],frozen_sources=c['frozen_sources']))
    raise ValueError('Actual source smoke then three source-only variants, no target jobs')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['dispatch','worker']);p.add_argument('--kind');p.add_argument('--config',required=True)
    a=p.parse_args();c=read(a.config);dispatch(c) if a.action=='dispatch' else worker(c,a.kind)
