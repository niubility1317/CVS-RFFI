"""One owner, immutable outputs, no retries, bounded CPU/GPU dispatch."""
import argparse
from collections import Counter
import os
from pathlib import Path
import subprocess
import sys
import time
from .io import read,write,atomic,require

def occupancy():
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,index,memory.free','--format=csv,noheader,nounits'],text=True)
    result={};mapping={}
    for line in raw.splitlines():
        uid,gpu,free=[v.strip() for v in line.split(',')];mapping[uid]=int(gpu);result[int(gpu)]=dict(pids=set(),free=int(free),blocked=False)
    for line in subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True).splitlines():
        uid,pid=[v.strip() for v in line.split(',')];pid=int(pid);r=result[mapping[uid]];r['pids'].add(pid)
        try:owner=Path('/proc',str(pid)).stat().st_uid
        except FileNotFoundError:owner=None
        if owner!=os.getuid():r['blocked']=True
    # Count admitted workers which have not initialized CUDA yet, including other owners.
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:
            if p.stat().st_uid!=os.getuid():continue
            env=dict(v.split(b'=',1) for v in (p/'environ').read_bytes().split(b'\0') if b'=' in v)
            gpu=env.get(b'CUDA_VISIBLE_DEVICES',b'').decode()
            cmd=(p/'cmdline').read_bytes()
            if gpu.isdigit() and int(gpu) in result and b'python' in cmd:result[int(gpu)]['pids'].add(int(p.name))
        except (FileNotFoundError,PermissionError,ProcessLookupError):pass
    return result

def eligible(device,own,slot,limit=3):
    return not device['blocked'] and own<slot and len(device['pids'])<limit and device['free']>=4000

def make_jobs(rt):
    root=Path(rt['run_root']);jobs=[]
    def add(key,kind,output,status,deps=(),gpu=False,**args):
        jobs.append(dict(id=key,kind=kind,output=str(output),completion=str(Path(output)/'complete.json') if kind!='smoke' else str(output),
                         successful_status=status,dependencies=list(deps),gpu=gpu,args=args))
    add('smoke','smoke',root/'smoke.json','PASS',gpu=True)
    if rt.get('preparation_only'):return jobs
    add('data','data',Path(rt['data_root'])/'capsule','VALIDATED_ONCE',('smoke',))
    jobs[-1]['completion']=str(Path(rt['data_root'])/'capsule/manifest.json');jobs[-1]['status_field']='phase2_data_status'
    add('DURA','restore',Path(rt['data_root'])/'DURA','RESTORED_COMPLETE',('data',),gpu=True)
    for seed in rt['model_seeds']:
        for path in ('P0','P1'):
            key=f'features-s{seed}-{path}'
            add(key,'features',root/'features'/f's{seed}-{path}','FEATURES_COMPLETE',('data',) if path=='P0' else ('DURA',),gpu=True,seed=seed,input_path=path)
    tech=[]
    for path in ('P0','P1'):
        key='technical-'+path;tech.append(key)
        add(key,'predict',root/'technical'/path,'PREDICTIONS_COMPLETE',(f"features-s{rt['model_seeds'][0]}-{path}",),seed=rt['model_seeds'][0],input_path=path,arm='local_ridge',technical_only=True)
    for seed in rt['model_seeds']:
        for path in ('P0','P1'):
            for arm in rt['arms']:
                key=f'predict-s{seed}-{path}-{arm}';output=root/'predictions'/f's{seed}-{path}-{arm}'
                add(key,'predict',output,'PREDICTIONS_COMPLETE',(f'features-s{seed}-{path}',*tech),seed=seed,input_path=path,arm=arm)
                add('score-'+key,'score',root/'scores'/f's{seed}-{path}-{arm}','SCORED_COMPLETE',(key,),prediction=str(output))
    return jobs

def completed(job):
    p=Path(job['completion'])
    return p.is_file() and read(p).get(job.get('status_field','status'))==job['successful_status']

def worker(rt,job):
    kind=job['kind'];args=job['args'];output=job['output']
    if kind=='score':
        from .score import run
        return run(Path(rt['data_root'])/'capsule',Path(rt['data_root'])/'score_only/truth.json',args['prediction'],output,rt['old_classes'])
    if kind=='smoke':
        from .features import source_smoke
        return source_smoke(rt,output)
    if kind=='data':
        from .data import build
        return build(rt)
    if kind in ('features','restore'):
        from .features import export,restore
        return restore(rt,output) if kind=='restore' else export(rt,args['seed'],args['input_path'],output)
    if kind=='predict':
        from .predict import run
        return run(rt,args['seed'],args['input_path'],args['arm'],output,args.get('technical_only',False))
    raise ValueError('Unknown job')

def dispatch(rt):
    root=Path(rt['run_root']);root.mkdir(parents=True,exist_ok=False);(root/'logs').mkdir()
    jobs=make_jobs(rt);write(root/'jobs.json',jobs);by={j['id']:j for j in jobs};active={};failed=[]
    state=dict(status='RUNNING',pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,commit=rt['code_commit'],launch_owner=rt['launch_owner'],jobs={j['id']:dict(status='QUEUED') for j in jobs})
    atomic(root/'pipeline_state.json',state)
    def launch(j,gpu):
        cmd=[rt['python'],'-u','-m','experiments.pamf_phase2_practical.runner','worker','--runtime',rt['runtime_file'],'--job',j['id']]
        env=dict(os.environ,PYTHONPATH=rt['release_root'],CUDA_VISIBLE_DEVICES='' if gpu is None else str(gpu),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
        log=root/'logs'/(j['id']+'.log')
        with log.open('x') as f:p=subprocess.Popen(cmd,cwd=rt['release_root'],env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        item=dict(status='RUNNING',pid=p.pid,gpu=gpu,cwd=rt['release_root'],argv=cmd,log=str(log),started_at=time.time())
        active[j['id']]=(p,item);state['jobs'][j['id']]=item
        print('LAUNCH',j['id'],p.pid,gpu,flush=True)
    while True:
        for key,(p,item) in list(active.items()):
            if p.poll() is None:continue
            ok=p.returncode==0 and completed(by[key]);state['jobs'][key]=dict(item,status='COMPLETE' if ok else 'FAILED',exit_code=p.returncode,ended_at=time.time())
            if not ok:failed.append(key)
            del active[key]
        pending=[j for j in jobs if state['jobs'][j['id']]['status']=='QUEUED']
        candidates=[j for j in pending if all(state['jobs'][k]['status']=='COMPLETE' for k in j['dependencies'])] if not failed else []
        for j in list(candidates):
            if j['gpu']:continue
            kind=j['kind'];cpu=[by[k]['kind'] for k,(p,i) in active.items() if i['gpu'] is None]
            if kind=='score' and 'score' in cpu:continue
            if len(cpu)>=rt.get('max_cpu_workers',4):continue
            launch(j,None);candidates.remove(j)
        res=occupancy()
        for slot in (1,2,3):
            for gpu in rt['allowed_gpus']:
                own=sum(item['gpu']==gpu for p,item in active.values())
                if not eligible(res[gpu],own,slot,rt['max_total_per_GPU']):continue
                j=next((j for j in candidates if j['gpu']),None)
                if j is None:continue
                launch(j,gpu);candidates.remove(j);res[gpu]['pids'].add(active[j['id']][0].pid);res[gpu]['free']-=4000
        state['counts']=dict(Counter(v['status'] for v in state['jobs'].values()));state['updated_at']=time.time()
        if failed:state['status']='STOP_NEW_DISPATCH_TECHNICAL_FAILURE';state['failed_jobs']=failed
        elif not pending and not active:state['status']='COMPLETE'
        atomic(root/'pipeline_state.json',state)
        if not active and (failed or not pending):break
        time.sleep(10)

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=('dispatch','worker'));p.add_argument('--runtime',required=True);p.add_argument('--job');a=p.parse_args();rt=read(a.runtime)
    if a.action=='dispatch':return dispatch(rt)
    j=next(j for j in make_jobs(rt) if j['id']==a.job);return worker(rt,j)

if __name__=='__main__':main()
