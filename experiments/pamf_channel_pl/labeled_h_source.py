"""Single-seed source falsification of true-L anchored high-confidence U/H."""
import argparse,os,time
from collections import Counter
from copy import deepcopy
from pathlib import Path
from experiments.cric_dura_comparison.io import read,write
from . import clean_h_source as base
from .labeled_h_anchor import RECIPE,LabeledHAnchor

RUN='20261027-phase1-PAMF-SIG1-REPL-LAnchorH-source-m1-r01'
RELEASE='20261027-phase1-PAMF-SIG1-REPL-LAnchorH-source-release-r02'
METHOD='REPL_H2_LU_LAnchorH';BASE_METHOD='REPL_H2_LU_Pair';SEED=2026092701
RAW_MIN=26568;H3_MIN=62109


def validate_context(c):
    if (set(c)!={'project','python','frozen_sources','template','reference','expected_source_contract'} or set(c['frozen_sources'])!={'F','AF','BASE','FOLD'}
        or c['template']!=c['project']+'/runs/'+base.RUN+'/configs/'+BASE_METHOD+'.json'
        or c['reference']!=c['project']+'/runs/'+base.RUN+'/evaluations/'+BASE_METHOD+'/complete.json'):
        raise ValueError('Known completed clean/H recipe/reference metadata only; F four roots, no target/G')


def template(ctx,commit):
    validate_context(ctx);c=deepcopy(read(ctx['template']));base.validate_config(c)
    if c['expected_source_contract']!=ctx['expected_source_contract'] or c['frozen_sources']!=ctx['frozen_sources']:raise ValueError('Original F0 contract and four F roots must match')
    rid='F0-'+METHOD+'-s'+str(SEED)
    c.update(run_id=RUN,row_id=rid,paper_method=METHOD,algorithm_id='pamf_clean_H_L_anchor_v1',code_commit=commit,
        output_root=ctx['project']+'/runs/'+RUN+'/F/'+rid+'/source',labeled_H_anchors=deepcopy(RECIPE))
    c['repl_row'].update(row_id=rid,method=METHOD)
    return c


def validate_config(c):
    ctx=dict(project=c['project'],python='unused',frozen_sources=c['frozen_sources'],template=c['project']+'/runs/'+base.RUN+'/configs/'+BASE_METHOD+'.json',
        reference=c['project']+'/runs/'+base.RUN+'/evaluations/'+BASE_METHOD+'/complete.json',expected_source_contract=c['expected_source_contract'])
    expected=template(ctx,c['code_commit']);root=expected['output_root']
    if 'segment_epochs' in c:
        offset=c.get('epoch_offset');length={20:10,30:20}.get(offset)
        if length is None:raise ValueError('Own original E20+10+20 trajectory only')
        expected.update(segment_epochs=length,epoch_offset=offset,step_offset=offset*222,parent_source=expected['shared_SIG1_E20'] if offset==20 else root+'/segment30',output_root=root+'/segment'+str(offset+length))
        if c.get('candidate_smoke') is True:expected['candidate_smoke']=True
    if c!=expected:raise ValueError('One exact high-confidence L-anchor-H intervention, no extra labels/heads/budget/handles')
    return c


def validate_runtime(rt):
    if (set(rt)!={'run_id','project','python','release_root','run_root','context','runtime_file','code_commit','frozen_sources','allowed_gpus','launch_owner'}
        or rt['run_id']!=RUN or rt['run_root']!=rt['project']+'/runs/'+RUN or rt['release_root']!=rt['project']+'/releases/'+RELEASE
        or rt['context']!=rt['release_root']+'/context.json' or rt['runtime_file']!=rt['release_root']+'/runtime.json'
        or rt['allowed_gpus']!=list(range(8)) or rt['launch_owner']!='codex-pamf-L-anchor-H-source-20261027'
        or set(rt['frozen_sources'])!={'F','AF','BASE','FOLD'}):raise ValueError('One immutable F-only source run')


def source_decision(score,reference):
    if reference!=dict(raw_correct=26554,H3_correct=62190):raise ValueError('Reuse complete predeclared same-view source reference')
    return dict(method=METHOD,raw_correct=score['raw_correct'],H3_correct=score['H3_correct'],source_pass=score['raw_correct']>=RAW_MIN and score['H3_correct']>=H3_MIN,
        raw_min=RAW_MIN,H3_min=H3_MIN,delta_raw_pp=100*(score['raw_correct']-reference['raw_correct'])/27000,delta_H3_pp=100*(score['H3_correct']-reference['H3_correct'])/81000)


def prepare(rt):
    validate_runtime(rt);ctx=read(rt['context']);validate_context(ctx)
    reference=base.source_score(read(ctx['reference']));source_decision(reference,reference)
    root=Path(rt['run_root']);(root/'configs').mkdir();c=template(ctx,rt['code_commit']);validate_config(c);file=root/'configs'/(METHOD+'.json');write(file,c)
    smoke_file=root/'configs/smoke.json';write(smoke_file,dict(runtime=rt,train_configs=[str(file)]))
    common=dict(gpu_pool=list(range(8)),execution_release=rt['release_root'],execution_python=rt['python'],entrypoint='experiments.pamf_channel_pl.labeled_h_source')
    jobs=[dict(common,id='smoke',kind='smoke',config=str(smoke_file),completion=str(root/'smoke.json'),successful_statuses=['PASS'],dependencies=[]),
        dict(common,id=METHOD,kind='train_eval',config=str(file),completion=str(root/'evaluations'/METHOD/'complete.json'),successful_statuses=['CLEAN_H_SOURCE_EVALUATED'],dependencies=[str(root/'smoke.json')])]
    write(root/'reference_summary.json',dict(method=BASE_METHOD,score=reference,source=ctx['reference'],weights_loaded=False));write(root/'jobs.json',jobs);return jobs


def analyze(rt):
    root=Path(rt['run_root']);reference=read(root/'reference_summary.json')['score'];score=base.source_score(read(root/'evaluations'/METHOD/'complete.json'));decision=source_decision(score,reference)
    write(root/'complete.json',dict(status='L_ANCHOR_H_SOURCE_COMPLETE',run_id=RUN,code_commit=rt['code_commit'],model_seed=SEED,source_score=score,reference=reference,decision=decision,
        target_reads=0,U_truth_reads=0,scope='One source student; no new target or three-seed performance evidence; existing confirmation recipe/processes unchanged'))


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
    state['status']='STOPPED_FOR_TECHNICAL_FAILURE' if stop else 'L_ANCHOR_H_SOURCE_COMPLETE';atomic(root/'pipeline_state.json',state)


def worker(c,kind):
    base.activate(c.get('runtime',c))
    import torch
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if kind=='smoke':return base.smoke(c['runtime'],[read(p) for p in c['train_configs']],validate_config,LabeledHAnchor)
    if kind=='train_eval':
        base.train(c,validate_config,LabeledHAnchor)
        return base.evaluate(dict(method=METHOD,context=c['source_context'],source_output=c['output_root'],signature_views=1,output_root=c['project']+'/runs/'+RUN+'/evaluations/'+METHOD,frozen_sources=c['frozen_sources']))
    raise ValueError('Source joint smoke then one candidate training/frozen source evaluation only')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['dispatch','worker']);p.add_argument('--kind');p.add_argument('--config',required=True)
    a=p.parse_args();c=read(a.config);dispatch(c) if a.action=='dispatch' else worker(c,a.kind)
