import argparse,gzip,os,subprocess,sys,time
from itertools import zip_longest
from pathlib import Path
from . import design as d
from experiments.cvs_receiver_residual_capacity4.control import lock
from experiments.cvs_multi_state_action.dispatch import capacity

def worker():
    row=d.rows()[0]
    with lock(d.BASE/'worker.owner.lock',nonblocking=True):
        gpu=int(os.environ['CUDA_VISIBLE_DEVICES'])
        with lock(Path(d.PROJECT)/'runs/receiver_residual_capacity4.lock'):
            if len(capacity()[gpu]['pids']|{os.getpid()})>3:raise ValueError('All-owner capacity exceeded')
        from .source import train
        train(d.config(row))
def dispatch():
    with lock(d.ROOT/'white_observation_owner.lock',nonblocking=True):
        if d.BASE.exists():raise FileExistsError('Preserve existing immutable white observation output')
        spec=d.read(d.ROOT/'experiments/cvs_white_observation20/registration.json')
        c=d.config(d.rows()[0]);rid=c['row_id']
        if [v['config'] for v in spec['rows']]!=[c]:raise ValueError('Fixed white matrix differs')
        control=d.prerequisite();d.BASE.mkdir();(d.BASE/'logs').mkdir();started=time.time()
        d.write(d.BASE/'dispatcher.json',dict(pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,owner=d.OWNER))
        child=None
        while child is None:
            with lock(Path(d.PROJECT)/'runs/receiver_residual_capacity4.lock'):
                caps=capacity();choices=[g for g,v in caps.items() if len(v['pids'])<3 and v['free_mb']>=12000]
                if choices:
                    gpu=min(choices,key=lambda g:(len(caps[g]['pids']),-caps[g]['free_mb'],g))
                    command=[sys.executable,'-u',str(d.ROOT/'experiments/cvs_white_observation20/runner.py'),'--worker']
                    log=d.BASE/'logs'/(rid+'.log')
                    with log.open('x') as f:child=subprocess.Popen(command,cwd=d.ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu)),stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
                    d.write(d.BASE/'launch.json',dict(rows=[dict(row_id=rid,pid=child.pid,gpu=gpu,cwd=str(d.ROOT),argv=command,log=str(log))]))
            if child is None:time.sleep(3)
        code=child.wait()
        if code or not (d.BASE/rid/'source/white_observation_proof.json').is_file():
            d.write(d.BASE/'failure.json',dict(status='FAILED',row_id=rid,exit_code=code,no_automatic_retry=True));raise RuntimeError('Own white observation failure preserved')
        try:
            count=0
            with gzip.open(d.CONTROL/'training_sequence.jsonl.gz','rt') as x,gzip.open(d.BASE/rid/'source/training_sequence.jsonl.gz','rt') as y:
                for a,b in zip_longest(x,y):
                    if a!=b:raise ValueError('All actual white/raw L/U physical sequences must match')
                    count+=1
            d.write(d.BASE/'matched_sequences.json',dict(status='VERIFIED',actual_L_U_batch_records=count,control=d.CONTROL.as_posix()))
            result=d.read(d.BASE/rid/'source/result.json');q=d.qualify(result,control);d.write(d.BASE/'source_decision.json',q)
            d.write(d.BASE/'completion.json',dict(status='ARTIFACTS_COMPLETE',source_decision=q['status'],source_only=True,target_access=False,
                automatic_extension=False,new_models=1,new_model_epochs=20,new_updates=4440,reused_control_model_epochs=20,reused_control_updates=4440,
                wall_seconds=time.time()-started,goal_completed=False))
        except Exception as error:
            d.write(d.BASE/'failure.json',dict(status='FAILED',phase='source closure',error=repr(error),no_automatic_retry=True));raise
def main():
    p=argparse.ArgumentParser();p.add_argument('--worker',action='store_true');a=p.parse_args()
    worker() if a.worker else dispatch()
