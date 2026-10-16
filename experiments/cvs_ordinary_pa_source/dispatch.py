import os
import subprocess
import sys
import time
from pathlib import Path
from . import design as d
from experiments.cvs_receiver_residual_capacity4.control import lock
from experiments.cvs_multi_state_action.dispatch import capacity

def dispatch():
    with lock(d.ROOT/'ordinary_PA_source_owner.lock',nonblocking=True):
        if d.BASE.exists():raise FileExistsError('Reconcile/preserve existing immutable PA run')
        spec=d.read(d.ROOT/'experiments/cvs_ordinary_pa_source/registration.json')
        if [r['config'] for r in spec['rows']]!=[d.config(r) for r in d.rows()]:raise ValueError('PA matrix differs')
        prior=d.read(Path(d.PROJECT)/'runs'/d.REST/'source_decision.json')
        if prior['status']!='FAIL' or prior['target_access']:raise ValueError('Fixed completed source question origin differs')
        d.BASE.mkdir();d.write(d.BASE/'dispatcher.json',dict(pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,owner=d.OWNER))
        while True:
            with lock(Path(d.PROJECT)/'runs/receiver_residual_capacity4.lock'):
                caps=capacity();ready=[g for g,v in caps.items() if len(v['pids'])<3 and v['free_mb']>=12000]
                if ready:
                    gpu=min(ready,key=lambda g:(len(caps[g]['pids']),-caps[g]['free_mb'],g))
                    command=[sys.executable,'-u',str(d.ROOT/'experiments/cvs_ordinary_pa_source/runner.py'),'--worker'];log=d.BASE/'worker.log'
                    with log.open('x') as f:child=subprocess.Popen(command,cwd=d.ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu)),
                        stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
                    d.write(d.BASE/'launch.json',dict(pid=child.pid,gpu=gpu,cwd=str(d.ROOT),argv=command,log=str(log),owner=d.OWNER))
                    break
            d.write(d.BASE/'waiting.json',dict(status='WAITING_CAPACITY',at=time.time()));time.sleep(5)
        code=child.wait()
        if code or not (d.BASE/'completion.json').is_file():
            d.write(d.BASE/'failure_dispatch.json',dict(status='FAILED',exit_code=code,no_automatic_retry=True))
            raise RuntimeError('Own PA diagnostic failure preserved')
