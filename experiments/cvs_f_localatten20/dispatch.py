import argparse,gzip,os,subprocess,sys,time
from itertools import zip_longest
from pathlib import Path
from . import design as d
from experiments.cvs_receiver_residual_capacity4.control import lock
from experiments.cvs_multi_state_action.dispatch import capacity

def worker(rid):
    row=next((r for r in d.rows() if r['row_id']==rid),None)
    if row is None:raise ValueError('One registered localatten row required')
    with lock(d.BASE/(rid+'.owner.lock'),nonblocking=True):
        gpu=int(os.environ['CUDA_VISIBLE_DEVICES'])
        with lock(Path(d.PROJECT)/'runs/receiver_residual_capacity4.lock'):
            if len(capacity()[gpu]['pids']|{os.getpid()})>3:raise ValueError('All-owner capacity exceeded')
        from .source import train
        train(d.config(row))

def dispatch():
    with lock(d.ROOT/'FcrossRX_localatten_owner.lock',nonblocking=True):
        if d.BASE.exists():raise FileExistsError('Reconcile existing immutable localatten output')
        spec=d.read(d.ROOT/'experiments/cvs_f_localatten20/registration.json')
        if [r['config'] for r in spec['rows']]!=[d.config(r) for r in d.rows()]:raise ValueError('Fixed source config differs')
        d.reference_result();d.BASE.mkdir();(d.BASE/'logs').mkdir();start=time.time()
        d.write(d.BASE/'dispatcher.json',dict(pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,owner=d.OWNER))
        row=d.rows()[0];rid=row['row_id'];child=None
        while child is None:
            with lock(Path(d.PROJECT)/'runs/receiver_residual_capacity4.lock'):
                caps=capacity();choices=[g for g,v in caps.items() if len(v['pids'])<3 and v['free_mb']>=12000]
                if choices:
                    gpu=min(choices,key=lambda g:(len(caps[g]['pids']),-caps[g]['free_mb'],g))
                    command=[sys.executable,'-u',str(d.ROOT/'experiments/cvs_f_localatten20/runner.py'),'--worker','--row',rid]
                    log=d.BASE/'logs'/(rid+'.log')
                    with log.open('x') as stream:child=subprocess.Popen(command,cwd=d.ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu)),stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                    d.write(d.BASE/'launch.json',dict(rows=[dict(row_id=rid,pid=child.pid,gpu=gpu,cwd=str(d.ROOT),argv=command,log=str(log))]))
            if child is None:time.sleep(3)
        while child.poll() is None:time.sleep(3)
        try:
            if child.returncode or not (d.BASE/rid/'source/localatten_proof.json').is_file():raise RuntimeError(('localatten row failed',child.returncode))
            count=0
            with gzip.open(d.BASE/rid/'source/training_sequence.jsonl.gz','rt') as x,gzip.open(d.REFERENCE/'training_sequence.jsonl.gz','rt') as y:
                for a,b in zip_longest(x,y):
                    if a!=b:raise ValueError('Actual primary L/U differs from C0')
                    count+=1
            if count!=5994:raise ValueError('Whole original L/U trace required')
            d.write(d.BASE/'matched_sequences.json',dict(status='VERIFIED',actual_L_U_batch_records=count,matching_C0_reference=True))
            decision=d.qualify(d.read(d.BASE/rid/'source/result.json'));d.write(d.BASE/'source_decision.json',decision)
            d.write(d.BASE/'completion.json',dict(status='ARTIFACTS_COMPLETE',source_only=True,target_access=False,source_decision=decision['status'],new_models=1,new_model_epochs=20,new_updates=4440,wall_seconds=time.time()-start,automatic_extension=False,goal_completed=False))
        except Exception as error:
            d.write(d.BASE/'failure.json',dict(status='FAILED',error=repr(error),preserve_outputs=True,no_automatic_retry=True));raise

def main():
    p=argparse.ArgumentParser();p.add_argument('--worker',action='store_true');p.add_argument('--row');a=p.parse_args()
    if a.worker:worker(a.row)
    elif a.row is not None:raise ValueError('Row only applies to registered worker')
    else:dispatch()
