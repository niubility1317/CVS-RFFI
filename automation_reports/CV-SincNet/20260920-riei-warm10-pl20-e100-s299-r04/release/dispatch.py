"""One exclusive finite queue, inventorying all owners' GPU processes."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent


def save(state):
    state['time']=time.time()
    p=ROOT/'state.tmp'
    p.write_text(json.dumps(state,indent=2),encoding='utf-8')
    p.replace(ROOT/'state.json')


def available(running):
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
    observed={}
    for line in apps.splitlines():
        uuid,pid=line.split(',');observed.setdefault(uuid.strip(),set()).add(int(pid))
    gpus=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],text=True)
    choices=[]
    for line in gpus.splitlines():
        index,uuid,free=[v.strip() for v in line.split(',')]
        pids=observed.get(uuid,set())
        pending={p.pid for p,r in running.values() if r['gpu_uuid']==uuid and p.poll() is None}-pids
        if len(pids|pending)<3 and int(free)-len(pending)*6500>=6500:
            choices.append((len(pids|pending),-int(free),int(index),uuid))
    return sorted(choices)


def main():
    with (ROOT/'launch.lock').open('x') as f:f.write(str(os.getpid()))
    m=json.loads((ROOT/'manifest.json').read_text(encoding='utf-8'))
    state=dict(owner_pid=os.getpid(),status='RUNNING',rows=[dict(r,status='QUEUED') for r in m['rows']])
    (ROOT/'logs').mkdir(exist_ok=True)
    running={};save(state)
    while True:
        for key,(p,r) in list(running.items()):
            rc=p.poll()
            if rc is not None:
                done=Path(r['output_root'])/'COMPLETE.json'
                valid=done.exists() and json.loads(done.read_text()).get('epoch')==100
                r.update(status='COMPLETE' if rc==0 and valid else 'FAILED',exit_code=rc)
                del running[key];save(state)
        if any(r['status']=='FAILED' for r in state['rows']):
            for r in state['rows']:
                if r['status']=='QUEUED':r['status']='NOT_LAUNCHED_AFTER_FAILURE'
        queued=[r for r in state['rows'] if r['status']=='QUEUED']
        if not queued and not running:
            state['status']='COMPLETE' if all(r['status']=='COMPLETE' for r in state['rows']) else 'FINISHED_WITH_FAILURES'
            save(state);return
        if queued:
            slots=available(running)
            if slots:
                _,_,gpu,uuid=slots[0];r=queued[0]
                command=[sys.executable,'-X','utf8','-u',str(ROOT/'train.py'),'--method',r['method'],
                         '--data-root',m['data_root'],'--output',r['output_root'],'--use-pl',str(r['use_pl'])]
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONUNBUFFERED='1')
                log=ROOT/'logs'/(r['run_id']+'.log')
                with log.open('xb') as f:
                    p=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT)
                r.update(status='RUNNING',pid=p.pid,gpu=gpu,gpu_uuid=uuid,command=command,log=str(log))
                running[r['run_id']]=(p,r);save(state)
        time.sleep(3)


if __name__=='__main__':main()
