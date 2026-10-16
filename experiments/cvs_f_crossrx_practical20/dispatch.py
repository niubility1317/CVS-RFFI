import argparse,gzip,os,subprocess,sys,time
from itertools import zip_longest
from pathlib import Path
from . import design as d
from experiments.cvs_receiver_residual_capacity4.control import lock
from experiments.cvs_multi_state_action.dispatch import capacity

def worker(rid):
    row=next((r for r in d.rows() if r['row_id']==rid),None)
    if row is None:raise ValueError('One registered source row required')
    with lock(d.BASE/(rid+'.owner.lock'),nonblocking=True):
        gpu=int(os.environ['CUDA_VISIBLE_DEVICES'])
        with lock(Path(d.PROJECT)/'runs/receiver_residual_capacity4.lock'):
            if len(capacity()[gpu]['pids']|{os.getpid()})>3:raise ValueError('All-owner capacity exceeded')
        from .source import train
        train(d.config(row))

def dispatch():
    with lock(d.ROOT/'FcrossRX_practical_owner.lock',nonblocking=True):
        if d.BASE.exists():raise FileExistsError('Reconcile existing immutable FcrossRX output')
        spec=d.read(d.ROOT/'experiments/cvs_f_crossrx_practical20/registration.json')
        if [r['config'] for r in spec['rows']]!=[d.config(r) for r in d.rows()]:raise ValueError('Fixed source configurations differ')
        d.BASE.mkdir();(d.BASE/'logs').mkdir();start=time.time()
        d.write(d.BASE/'dispatcher.json',dict(pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,owner=d.OWNER))
        pending=d.rows();running=[];launches=[];failures=[]
        while pending or running:
            for item in running[:]:
                code=item['child'].poll()
                if code is None:continue
                running.remove(item);rid=item['row']['row_id']
                if code or not (d.BASE/rid/'source/FcrossRX_pair_proof.json').is_file():
                    failures.append(dict(row_id=rid,exit_code=code));d.write(d.BASE/'row_failures.json',dict(failures=failures,no_automatic_retry=True))
            if pending:
                with lock(Path(d.PROJECT)/'runs/receiver_residual_capacity4.lock'):
                    caps=capacity()
                    assigned={g:v['pids']|{i['child'].pid for i in running if i['gpu']==g} for g,v in caps.items()}
                    choices=[g for g,v in caps.items() if len(assigned[g])<3 and v['free_mb']>=12000]
                    if choices:
                        gpu=min(choices,key=lambda g:(len(assigned[g]),-caps[g]['free_mb'],g));row=pending.pop(0);rid=row['row_id']
                        command=[sys.executable,'-u',str(d.ROOT/'experiments/cvs_f_crossrx_practical20/runner.py'),'--worker','--row',rid]
                        log=d.BASE/'logs'/(rid+'.log')
                        with log.open('x') as stream:child=subprocess.Popen(command,cwd=d.ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu)),
                            stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                        running.append(dict(child=child,row=row,gpu=gpu))
                        launches.append(dict(row_id=rid,pid=child.pid,gpu=gpu,cwd=str(d.ROOT),argv=command,log=str(log)))
                        d.write(d.BASE/'launch.json',dict(rows=launches))
            if pending or running:time.sleep(3)
        if failures:
            d.write(d.BASE/'failure.json',dict(status='FAILED',failures=failures,preserve_outputs=True,no_automatic_retry=True));raise RuntimeError('Source matrix incomplete')
        try:
            paths=[d.BASE/r['row_id']/'source/training_sequence.jsonl.gz' for r in d.rows()];count=0
            with gzip.open(paths[0],'rt') as x,gzip.open(paths[1],'rt') as y:
                for a,b in zip_longest(x,y):
                    if a!=b:raise ValueError('Actual whole L/U sequences differ')
                    count+=1
            if count!=5994:raise ValueError('Expected full E20 L/U exposure trace')
            d.write(d.BASE/'matched_sequences.json',dict(status='VERIFIED',actual_L_U_batch_records=count))
            results={r['arm']:d.read(d.BASE/r['row_id']/'source/result.json') for r in d.rows()}
            decision=d.qualify(results);d.write(d.BASE/'source_decision.json',decision)
            d.write(d.BASE/'completion.json',dict(status='ARTIFACTS_COMPLETE',source_only=True,target_access=False,source_decision=decision['status'],
                new_models=2,new_model_epochs=40,new_updates=8880,wall_seconds=time.time()-start,automatic_extension=False,goal_completed=False))
        except Exception as error:
            d.write(d.BASE/'failure.json',dict(status='FAILED',phase='source closure',error=repr(error),preserve_outputs=True,no_automatic_retry=True));raise

def main():
    p=argparse.ArgumentParser();p.add_argument('--worker',action='store_true');p.add_argument('--row');a=p.parse_args()
    if a.worker:worker(a.row)
    elif a.row is not None:raise ValueError('Row only applies to registered worker')
    else:dispatch()
