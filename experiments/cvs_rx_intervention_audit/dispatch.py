"""Three source-only diagnostic workers; no target evaluation or identity training."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
from . import design as d
from experiments.cvs_receiver_residual_capacity4.control import lock,process
from experiments.cvs_multi_state_action.dispatch import capacity

def dispatch():
    with lock(d.ROOT/'owner.lock',nonblocking=True):
        d.BASE.mkdir(parents=True,exist_ok=False);(d.BASE/'logs').mkdir()
        spec=d.read(d.ROOT/'registration/experiment.json')
        if spec['run_id']!=d.RUN or [r['config'] for r in spec['rows']]!=[d.config(r) for r in d.rows()]:raise ValueError('Preregistration differs')
        spec.update(status='RUNNING',execution_commit=(d.ROOT/'release_commit.txt').read_text().strip());d.write(d.BASE/'experiment.json',spec)
        d.write(d.BASE/'dispatcher.json',dict(pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,owner=d.OWNER))
        pending=d.rows();active={};receipts=[];completed=[];failures=[]
        while pending or active:
            for rowid,child in list(active.items()):
                code=child.poll()
                if code is None:continue
                p=d.BASE/rowid/'completion.json';value=d.read(p) if p.is_file() else {}
                if code or value.get('status')!='SOURCE_DIAGNOSTIC_COMPLETE' or value.get('identity_updates')!=0 or value.get('target_access'):
                    failures.append(dict(row_id=rowid,exit_code=code,completion=value))
                else:completed.append(rowid)
                del active[rowid]
            if pending and not failures:
                with lock(Path(d.PROJECT)/'runs/receiver_residual_capacity4.lock'):
                    states=capacity();reserved={item['pid']:item['gpu'] for item in receipts if item['row_id'] in active}
                    ready=[]
                    for gpu,state in states.items():
                        pids=state['pids']|{pid for pid,g in reserved.items() if g==gpu}
                        if len(pids)<3 and state['free_mb']>=12000:ready.append((len(pids),-state['free_mb'],gpu))
                    if ready:
                        gpu=min(ready)[2];row=pending.pop(0);cmd=[sys.executable,'-u','-m','experiments.cvs_rx_intervention_audit.dispatch','--worker',row['row_id']]
                        log=d.BASE/'logs'/f"{row['row_id']}.log";env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu))
                        with log.open('x') as f:child=subprocess.Popen(cmd,cwd=d.ROOT,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
                        active[row['row_id']]=child;receipts.append(dict(row_id=row['row_id'],pid=child.pid,gpu=gpu,cwd=str(d.ROOT),argv=cmd,log=str(log)))
                        d.write(d.BASE/'launch.json',dict(rows=receipts))
            d.write(d.BASE/'queue_state.json',dict(active=list(active),pending=pending,completed=completed,failures=failures,max_total_processes_per_GPU=3,updated_at=time.time()))
            if failures and not active:break
            if pending or active:time.sleep(3)
        if failures:
            d.write(d.BASE/'failure.json',dict(status='FAILED',rows=failures,no_automatic_retry=True));raise RuntimeError('Source audit failed; artifacts retained')
        if len(completed)!=3:raise ValueError('Incomplete three-row diagnostic')
        d.write(d.BASE/'completion.json',dict(status='SOURCE_DIAGNOSTIC_COMPLETE',rows=completed,identity_updates=0,target_access=False))

def worker(row):
    # Reservation stays in the single worker process, including its startup wait.
    gpu=int(os.environ['CUDA_VISIBLE_DEVICES'])
    with lock(d.BASE/(row+'.owner.lock'),nonblocking=True):
        while True:
            with lock(Path(d.PROJECT)/'runs/receiver_residual_capacity4.lock'):
                state=capacity()[gpu]
                if len(state['pids']|{os.getpid()})<=3 and state['free_mb']>=12000:break
            time.sleep(3)
        from .run import run
        run(row)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--worker');a=p.parse_args()
    if a.worker:worker(a.worker)
    else:dispatch()
