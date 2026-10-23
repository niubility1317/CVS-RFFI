"""Unique immutable owner; source jobs never receive target/truth handles."""
import argparse
from collections import Counter
import os,subprocess,sys,time
from pathlib import Path
from experiments.cric_dura_comparison.io import read,write

def activate(c):
    from experiments.cric_dura_comparison.bootstrap import activate as base,numerical_policy
    base(c)
    from experiments.cric_dura_comparison.external_dura import activate_external
    activate_external(c)
    numerical_policy()

def worker(c):
    activate(c)
    if c['stage']=='F':
        if c['kind']=='external':
            from .source import train
        else:from .pamf_train import train
    elif c['stage']=='G':
        from experiments.cric_dura_comparison.g_train import train
    elif c['stage']=='predict':
        from .evaluation import predict as train
    elif c['stage']=='cache':
        from .evaluation import cache as train
    elif c['stage']=='score':
        from .evaluation import score as train
    else:raise ValueError('Unknown registered stage')
    train(c)

def command(rt,mode,config,python=None):
    return [python or rt['python'],'-u','-m','experiments.practical_full_2x2.runner',mode,'--config',str(config)]

def launch(rt,job,gpu):
    log=Path(rt['run_root'])/'logs'/(job['log_name']+'.log');log.parent.mkdir(exist_ok=True)
    env=dict(os.environ,PYTHONPATH=rt['release_root'],CUDA_VISIBLE_DEVICES='' if gpu is None else str(gpu),
        OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONUNBUFFERED='1')
    with log.open('x') as stream:
        child=subprocess.Popen(command(rt,'worker',job['config'],job['python']),cwd=rt['release_root'],env=env,
            stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    return child,dict(status='RUNNING',pid=child.pid,gpu=gpu,log=str(log),started_at=time.time())

def dispatch(rt):
    # Historical Full packages are supplied by the immutable external releases.
    # Activate those namespace paths BEFORE importing plan/cache dependencies.
    activate(rt)
    from .plan import validate
    from .prepare import prepare
    from .smoke import run as smoke
    from .cache import Reader,atomic
    from experiments.cric_dura_comparison.pipeline import alive,done,ready,block_failed_dependencies
    from experiments.cric_dura_comparison.resource_balance import LoadPolicy
    from experiments.cric_dura_comparison.source_queue import capacity
    root=Path(rt['run_root']);root.mkdir(parents=True,exist_ok=False)
    p=read(rt['matrix'])['matrix'];validate(p)
    if rt['allowed_gpus']!=p['allowed_gpus'] or set(rt['allowed_gpus'])&set(p['reserved_gpus']):raise ValueError('User-reserved GPUs must remain untouched')
    state=dict(status='PREPARING_DATA',pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,code_commit=rt['code_commit'],
        owner=rt['launch_owner'],started_at=time.time(),jobs={},new_F=600,new_G=4)
    atomic(root/'pipeline_state.json',state)
    jobs=prepare(rt,p);by={j['id']:j for j in jobs}
    # One real-checkpoint, source-only smoke; no target is read before training.
    smoke(rt,jobs,root/'source_smoke.json')
    env=dict(os.environ,PYTHONPATH=rt['release_root'],CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    with (root/'augmentation_producer.log').open('x') as log:
        producer=subprocess.Popen(command(rt,'cache-service',rt['runtime_file']),cwd=rt['release_root'],env=env,
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    atomic(root/'augmentation_producer.json',dict(pid=producer.pid,cwd=rt['release_root'],argv=command(rt,'cache-service',rt['runtime_file'])))
    # Prime one shared epoch while the initial no-channel Full models train.
    first=next(read(j['config']) for j in jobs if j['kind']=='F' and read(j['config']).get('cell')=='C')
    Reader(first).request(1,1,.3)
    state.update(status='RUNNING',jobs={j['id']:dict(status='QUEUED') for j in jobs})
    active={};children={};failures=Counter();policy=LoadPolicy(dict(boost_gpus=[0,6],max_gpu_workers=26))
    turn=0
    while True:
        for k,item in list(active.items()):
            if alive(item['pid']):continue
            job=by[k];ok=done(job);state['jobs'][k]=dict(item,status='COMPLETE' if ok else 'FAILED',ended_at=time.time())
            if not ok:
                tail=Path(item['log']).read_text(errors='replace')[-5000:].splitlines()
                reason=tail[-1] if tail else 'missing_completion_without_log';failures[reason]+=1;state['jobs'][k]['failure']=reason
            children.pop(k).poll();del active[k]
        block_failed_dependencies(jobs,state,[])
        systemic=bool(failures and max(failures.values())>=2)
        pending=[j for j in jobs if state['jobs'][j['id']]['status']=='QUEUED']
        candidates=[j for j in pending if ready(j)] if not systemic else []
        score=next((j for j in candidates if j['kind']=='score'),None)
        if score and not any(by[k]['kind']=='score' for k in active):
            child,item=launch(rt,score,None);active[score['id']]=item;children[score['id']]=child;state['jobs'][score['id']]=item;candidates.remove(score)
        policy.poll(by)
        for gpu in policy.order(rt['allowed_gpus']):
            owned=[by[k] for k,v in active.items() if v['gpu']==gpu]
            if len(owned)>=3:continue  # Baseline24 slots; no unmeasured fourth-slot claim.
            if not policy.host_ok():continue
            apps,free=capacity(gpu)
            # One launch/card/iteration; owned pending CUDA PIDs still reserve slots.
            available=[j for j in candidates if j['kind']!='score' and not(j['kind'] in ('G','cache') and any(x['kind'] in ('G','cache') for x in owned))]
            if not available:continue
            kinds=('F','F','G','predict','cache');wanted=kinds[turn%len(kinds)];turn+=1
            job=next((j for j in available if j['kind']==wanted),available[0])
            admission={k:dict(v,kind='G' if v['kind']=='cache' else v['kind']) for k,v in by.items()}
            if not policy.admits(gpu,admission[job['id']],active,admission,apps,free,False):continue
            child,item=launch(rt,job,gpu);active[job['id']]=item;children[job['id']]=child;state['jobs'][job['id']]=item;candidates.remove(job)
        state.update(updated_at=time.time(),counts=dict(Counter(v['status'] for v in state['jobs'].values())),
            failures=dict(failures),systemic_stop_new_launches=systemic,augmentation_producer_alive=alive(producer.pid),
            gpu_workers=sum(v['gpu'] is not None for v in active.values()),host=policy.host,telemetry_error=policy.last_error)
        atomic(root/'pipeline_state.json',state)
        if not active and (not pending or systemic):break
        time.sleep(15)
    state['status']='ARTIFACTS_COMPLETE' if all(v['status']=='COMPLETE' for v in state['jobs'].values()) else 'PARTIAL_TECHNICAL_FAILURE'
    (Path(rt['augmentation_cache'])/'stop').touch()
    atomic(root/'pipeline_state.json',state)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['dispatch','worker','cache-service']);p.add_argument('--config',required=True)
    a=p.parse_args();c=read(a.config)
    if a.mode=='dispatch':dispatch(c)
    elif a.mode=='worker':worker(c)
    else:
        activate(c)
        from .cache import service
        service(c)
