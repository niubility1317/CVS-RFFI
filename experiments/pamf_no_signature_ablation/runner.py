import argparse
from collections import Counter
from copy import deepcopy
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from experiments.cric_dura_comparison.io import read,write
from .method import RUN,rows,configuration,TASKS,READOUTS,ARMS,PARAMETERS


def activate(rt):
    from experiments.pamf_repl_completion.runner import activate as start
    start(rt)


def smoke_arguments(factory,c,device):
    # The frozen reader already sets this field; direct native smoke builds must
    # do the same before entering Phase1IdentityAdapter.
    a=factory(c,device);a.input_len=256
    return a


def prepare(rt):
    context=read(rt['context']);root=Path(rt['run_root']);jobs=[];sources={};cfgroot=root/'configs';cfgroot.mkdir()
    def add(kind,row,c,completion,status,deps=(),**extra):
        path=cfgroot/(kind+'__'+row['row_id']+'.json');write(path,c)
        jobs.append(dict(id=kind+'__'+row['row_id'],kind=kind,config=str(path),completion=str(completion),
                         successful_statuses=[status],dependencies=list(deps),**extra))
    for row in rows():
        rid=row['row_id'];c=configuration(context,row,rt['project'],rt['code_commit']);c['frozen_sources']=rt['frozen_sources']
        if row['reuse']:
            previous=context['baselines'][str(row['model_seed'])]
            sources[rid]=dict(row=row,source_output=previous['source'],scores=previous['scores'],predictions=previous['predictions'])
            if read(Path(previous['scores'])/'complete.json')['status']!='SCORED_COMPLETE':raise ValueError('Baseline scores missing')
            continue
        source=Path(c['output_root']);add('F',row,c,source/'complete.json','SOURCE_FROZEN')
        out=root/'readouts'/rid;h=context['handle']
        p=dict(row,run_id=RUN,protocol_id='F0',source_output=str(source),source_contract=h['source_contract'],classes=h['classes'],
               tests={t:dict(views_root=h['tests'][t]['views_root']) for t in TASKS},source_days=[1,2,3],
               restored_cache=context['caches']['L']['path'],readouts=READOUTS,output_root=str(out/'predictions'),frozen_sources=rt['frozen_sources'])
        add('predict',row,p,out/'predictions/complete.json','ALL_PREDICTIONS_COMPLETE',[str(source/'complete.json')])
        s=deepcopy(p);s.update(prediction_output=p['output_root'],output_root=str(out/'scores'),tests=h['tests'])
        for k in ('source_output','restored_cache','frozen_sources'):s.pop(k,None)
        add('score',row,s,out/'scores/complete.json','SCORED_COMPLETE',[str(out/'predictions/complete.json')])
        sources[rid]=dict(row=row,source_output=str(source),scores=str(out/'scores'),predictions=str(out/'predictions'))
    timing_sources={f'F0-CE_{a}-s2026092701':sources[f'{a}-s2026092701'] for a in ARMS}
    write(root/'timing_sources.json',timing_sources)
    tc=dict(run_id=RUN,panel='strict_structure',output_root=str(root/'timing'),context=rt['context'],
            sources=str(root/'timing_sources.json'),source_contract=context['handle']['source_contract'],
            source_cache=context['handle']['source_cache'],frozen_sources=rt['frozen_sources'],case_rounds=250)
    add('timing',dict(row_id='five_structures'),tc,root/'timing/complete.json','TIMING_COMPLETE',
        [str(Path(sources[f'{a}-s2026092701']['source_output'])/'complete.json') for a in ARMS],exclusive=True)
    write(root/'sources.json',sources);write(root/'jobs.json',jobs)
    return jobs


def smoke(rt):
    import torch
    import numpy as np
    from experiments.cric_dura_comparison.io import tensor
    from experiments.pamf_repl_completion.source import load_reader as existing
    from experiments.cric_dura_comparison.f_train import arguments,original_config,class_dimension
    from experiments.cvs_phase1_stack.runtime import installed as native_scope
    from experiments.pamf_repl_completion.objectives import installed
    from experiments.cric_dura_comparison import t1_model
    from .method import architecture_scope
    context=read(rt['context']);contract=read(context['handle']['source_contract'])
    # Validate the complete real-checkpoint ancestry with source IQ only.
    baseline=context['baselines']['2026092701']
    reader,_=existing(baseline['source'],contract,'cuda:0',require_no_unlabeled=True)
    x=tensor(np.ascontiguousarray(np.load(Path(context['handle']['source_cache'])/'L_s_x.npy',mmap_mode='r')[:2],np.float32),'cuda:0')
    with torch.no_grad():
        z=reader.classify_features(reader.features(x))
        if z.shape!=(2,6) or not bool(torch.isfinite(z).all()):raise ValueError('Actual checkpoint no-query smoke failed')
    del reader
    checks=[]
    for arm in ARMS:
        row=next(r for r in rows() if r['arm']==arm and r['model_seed']==2026092701)
        c=configuration(context,row,rt['project'],rt['code_commit'])
        torch.manual_seed(c['model_seed']);torch.cuda.manual_seed_all(c['model_seed'])
        with architecture_scope(),installed(c,{},False),class_dimension(c),native_scope(original_config(c['model_seed'])) as native:
            a=smoke_arguments(arguments,c,'cuda:0');net=t1_model.install(native.build_baseline_model(a,'cuda:0'),c).to('cuda:0')
        ref=net.id_backbone.encoder
        count=sum(p.numel() for p in ref.parameters())
        if count!=PARAMETERS[arm]:raise ValueError(('Architecture parameter mismatch',arm,count))
        path=Path(rt['run_root'])/('smoke_'+arm+'.pt');torch.save(net.state_dict(),path)
        net.load_state_dict(torch.load(path,map_location='cuda:0',weights_only=False),strict=True)
        net.eval();logits=net(x)
        loss=torch.nn.functional.cross_entropy(logits,torch.tensor([0,1],device='cuda:0'));loss.backward()
        if not torch.isfinite(loss) or not all(bool(torch.isfinite(p.grad).all()) for p in net.parameters() if p.grad is not None):raise FloatingPointError('Architecture backward failed')
        checks.append(dict(arm=arm,parameters=count,finite_backward=True,strict_checkpoint_roundtrip=True))
        del net,ref,logits,loss
    write(Path(rt['run_root'])/'smoke.json',dict(status='PASS',real_checkpoint=baseline['source'],query_reads=0,U_reads=0,architectures=checks))


def worker(rt,kind):
    if kind=='score':
        from .evaluation import score
        return score(rt)
    activate(rt)
    if kind=='smoke':return smoke(rt)
    if kind=='F':
        from .method import train
        return train(rt)
    if kind=='predict':
        from .evaluation import predict
        return predict(rt)
    if kind=='timing':
        from experiments.pamf_repl_completion import timing,plan
        from .method import load_reader
        context=read(rt['context'])
        # Adapt the existing source-only timing function to this five-arm panel.
        context['caches']={'F0':context['caches']}
        patched=Path(rt['output_root']).parent/'timing_context.json';write(patched,context)
        rt=dict(rt,context=str(patched));plan.STRUCTURES=ARMS;timing.load_reader=lambda folder,contract,device:load_reader(folder,contract,device)
        return timing.run(rt)
    raise ValueError('Unknown worker stage')


def dispatch(rt):
    from experiments.pamf_repl_completion.runner import processes,capacity,foreign_CPU_score
    from experiments.cric_dura_comparison.pipeline import done,ready,block_failed_dependencies
    from experiments.practical_full_2x2.cache import atomic
    root=Path(rt['run_root']);root.mkdir(parents=True,exist_ok=False);(root/'logs').mkdir()
    jobs=prepare(rt);by={j['id']:j for j in jobs};active={};failures=Counter()
    state=dict(status='SMOKE_PENDING',pid=os.getpid(),cwd=os.getcwd(),code_commit=rt['code_commit'],
               owner=rt['launch_owner'],jobs={j['id']:dict(status='QUEUED') for j in jobs})
    atomic(root/'pipeline_state.json',state)
    def launch(job,gpu):
        path=root/'logs'/(job['id']+'.log');cmd=[rt['python'],'-u','-m','experiments.pamf_no_signature_ablation.runner','worker','--kind',job['kind'],'--config',job['config']]
        env=dict(os.environ,PYTHONPATH=rt['release_root'],CUDA_VISIBLE_DEVICES='' if gpu is None else str(gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
        with path.open('x') as log:child=subprocess.Popen(cmd,cwd=rt['release_root'],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        item=dict(status='RUNNING',pid=child.pid,gpu=gpu,cwd=rt['release_root'],argv=cmd,log=str(path),started_at=time.time())
        return child,item
    # One source-only real-checkpoint smoke is the first GPU work; PASS continues.
    while True:
        p=processes();gpu=next((g for g in rt['allowed_gpus'] if len(capacity(g)[0]|{pid for pid,v in p.items() if v==g})<2 and capacity(g)[1]>=12000),None)
        if gpu is not None:break
        time.sleep(15)
    sm=dict(id='smoke',kind='smoke',config=rt['runtime_file']);child,item=launch(sm,gpu)
    state['smoke']=item;atomic(root/'pipeline_state.json',state)
    status=child.wait()
    state['smoke'].update(status='FAILED' if status else 'PASS',exit_code=status,ended_at=time.time())
    if status or read(root/'smoke.json')['status']!='PASS':
        state['status']='FAILED_NO_QUERY_SMOKE';atomic(root/'pipeline_state.json',state);return
    state['status']='RUNNING'
    while True:
        for key,(child,item) in list(active.items()):
            if child.poll() is None:continue
            ok=done(by[key]);state['jobs'][key]=dict(item,status='COMPLETE' if ok else 'FAILED',ended_at=time.time())
            if not ok:
                lines=Path(item['log']).read_text(errors='replace').splitlines();reason=lines[-1] if lines else 'missing completion'
                reason=re.sub(r's202609270[123]','sSEED',reason);failures[reason]+=1;state['jobs'][key]['failure']=reason
            del active[key]
        block_failed_dependencies(jobs,state,[])
        systemic=any(n>=2 or any(token in reason for token in ('CONTRACT_MISMATCH','PROVENANCE','Truth closed','U truth','Output collision')) for reason,n in failures.items())
        pending=[j for j in jobs if state['jobs'][j['id']]['status']=='QUEUED']
        candidates=[j for j in pending if ready(j)] if not systemic else []
        score=next((j for j in candidates if j['kind']=='score'),None)
        if score and not any(by[k]['kind']=='score' for k in active) and not foreign_CPU_score({v[0].pid for v in active.values()}):
            child,item=launch(score,None);active[score['id']]=(child,item);state['jobs'][score['id']]=item;candidates.remove(score)
        procs=processes()
        for gpu in rt['allowed_gpus']:
            if any(v[1]['gpu']==gpu for v in active.values()):continue
            pids,free=capacity(gpu);pids|={pid for pid,g in procs.items() if g==gpu}
            if len(pids)>=2 or free<12000:continue
            candidate=next((j for j in candidates if j['kind']!='score' and not (j.get('exclusive') and pids)),None)
            if candidate is None:continue
            child,item=launch(candidate,gpu);active[candidate['id']]=(child,item);state['jobs'][candidate['id']]=item;candidates.remove(candidate)
            procs[child.pid]=gpu
        state.update(updated_at=time.time(),counts=dict(Counter(v['status'] for v in state['jobs'].values())),failures=dict(failures),systemic_stop_new_launches=systemic)
        atomic(root/'pipeline_state.json',state)
        if not active and (not pending or systemic):break
        time.sleep(15)
    if all(v['status']=='COMPLETE' for v in state['jobs'].values()):
        from .analyze import run
        run(rt);state['status']='ANALYZED'
    else:state['status']='PARTIAL_TECHNICAL_FAILURE'
    state['ended_at']=time.time();atomic(root/'pipeline_state.json',state)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['dispatch','worker']);p.add_argument('--config',required=True);p.add_argument('--kind');a=p.parse_args()
    c=read(a.config)
    if a.action=='dispatch':dispatch(c)
    else:worker(c,a.kind)
