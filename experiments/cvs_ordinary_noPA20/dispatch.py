import gzip
from itertools import zip_longest
import os
from pathlib import Path
import subprocess
import sys
import time
from . import design as d
from experiments.cvs_receiver_residual_capacity4.control import lock
from experiments.cvs_multi_state_action.dispatch import capacity

def dispatch():
    with lock(d.ROOT/'ordinary_noPA_owner.lock',nonblocking=True):
        if d.BASE.exists():raise FileExistsError('Preserve/reconcile existing immutable N2 source run')
        spec=d.read(d.ROOT/'experiments/cvs_ordinary_noPA20/registration.json')
        if [v['config'] for v in spec['rows']]!=[d.config(r) for r in d.rows()]:raise ValueError('N2 matrix differs')
        first=d.prerequisite();d.BASE.mkdir();d.write(d.BASE/'reused_controls.json',first)
        d.write(d.BASE/'dispatcher.json',dict(pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,owner=d.OWNER));start=time.time()
        while True:
            with lock(Path(d.PROJECT)/'runs/receiver_residual_capacity4.lock'):
                caps=capacity();ready=[g for g,v in caps.items() if len(v['pids'])<3 and v['free_mb']>=12000]
                if ready:
                    gpu=min(ready,key=lambda g:(len(caps[g]['pids']),-caps[g]['free_mb'],g));command=[sys.executable,'-u',str(d.ROOT/'experiments/cvs_ordinary_noPA20/runner.py'),'--worker']
                    log=d.BASE/'worker.log'
                    with log.open('x') as f:child=subprocess.Popen(command,cwd=d.ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu)),
                        stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
                    d.write(d.BASE/'launch.json',dict(pid=child.pid,gpu=gpu,cwd=str(d.ROOT),argv=command,log=str(log),owner=d.OWNER));break
            time.sleep(5)
        code=child.wait();c=d.config(d.rows()[0]);source=Path(c['output_root'])
        if code or not (source/'PA_removal_proof.json').is_file():
            d.write(d.BASE/'failure.json',dict(status='FAILED',exit_code=code,no_automatic_retry=True));raise RuntimeError('Own N2 technical failure retained')
        try:
            count=0
            paths=[source/'training_sequence.jsonl.gz']+[d.old.BASE/f'{arm}-RX1-s{d.SEED}'/'source/training_sequence.jsonl.gz' for arm in ('N0','N1')]
            with gzip.open(paths[0],'rt') as a,gzip.open(paths[1],'rt') as b,gzip.open(paths[2],'rt') as z:
                for x,y,w in zip_longest(a,b,z):
                    if x!=y or x!=w:raise ValueError('Complete N2/N0/N1 L/U physical sequences differ')
                    count+=1
            d.write(d.BASE/'matched_sequences.json',dict(status='VERIFIED',actual_L_U_batch_records=count,compared=['N2','N0','N1']))
            result=d.read(source/'result.json');q=d.qualify(result,first['results']);d.write(d.BASE/'source_decision.json',q)
            d.write(d.BASE/'completion.json',dict(status='ARTIFACTS_COMPLETE',source_decision=q['status'],source_only=True,
                target_access=False,automatic_extension=False,wall_seconds=time.time()-start,new_models=1,new_model_epochs=20,new_updates=4440,reused_control_model_epochs=40,goal_completed=False))
        except Exception as error:
            d.write(d.BASE/'failure.json',dict(status='FAILED',phase='source closure',error=repr(error),no_automatic_retry=True));raise
