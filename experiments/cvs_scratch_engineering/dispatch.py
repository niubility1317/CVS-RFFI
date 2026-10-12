"""One PID per row across train -> predict -> independent score via exec."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
from . import design as d
from experiments.cvs_receiver_residual_capacity4.control import lock,process
from experiments.cvs_multi_state_action.dispatch import capacity


def require_scored(value,models):
    expected=dict(status='SCORED_COMPLETE',models=models,views=1,query_count_per_view=168000,prediction_count=models*168000,metric_records=models*14,day_metric_records=models*4,truth_last=True,independent_recount='VERIFIED')
    if any(value.get(k)!=v for k,v in expected.items()):raise ValueError('Incomplete row or matrix scoring')


def replace(mode,row):
    os.execv(sys.executable,[sys.executable,'-u','-m','experiments.cvs_scratch_engineering.dispatch',mode,row])


def worker(row):
    gpu=int(os.environ['CUDA_VISIBLE_DEVICES'])
    with lock(d.BASE/(row+'.owner.lock'),nonblocking=True):
        while True:
            with lock(Path(d.PROJECT)/'runs/receiver_residual_capacity4.lock'):
                state=capacity()[gpu]
                if len(state['pids']|{os.getpid()})<=3 and state['free_mb']>=12000:break
            time.sleep(3)
        from .source import train
        from .evaluate import source_provenance
        c=d.config(next(r for r in d.rows() if r['row_id']==row));train(c);source=source_provenance(c)
        frozen=d.read(d.BASE/row/'source_frozen.json')
        if frozen['config']!=c or frozen['optimizer_steps']!=4440 or frozen['target_access']:raise ValueError('Source freeze differs')
    replace('--predict-row',row)


def prepare_teachers():
    while True:
        child=None
        with lock(Path(d.PROJECT)/'runs/receiver_residual_capacity4.lock'):
            ready=[(len(s['pids']),-s['free_mb'],gpu) for gpu,s in capacity().items() if len(s['pids'])<3 and s['free_mb']>=12000]
            if ready:
                gpu=min(ready)[2]
                cmd=[sys.executable,'-u','-m','experiments.cvs_scratch_engineering.dispatch','--prepare-worker']
                path=d.BASE/'logs/peer_cache.log'
                with path.open('x') as stream:
                    child=subprocess.Popen(cmd,cwd=d.ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu)),stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                d.write(d.BASE/'cache_launch.json',dict(pid=child.pid,gpu=gpu,cwd=str(d.ROOT),argv=cmd,log=str(path)))
        if child is not None:break
        time.sleep(3)
    d.write(d.BASE/'queue_state.json',dict(active=[],pending=d.rows(),completed=[],failures=[],phase='PREPARING_TEACHERS',cache_pid=child.pid,updated_at=time.time()))
    code=child.wait()
    p=d.BASE/'peer_cache/completion.json'
    value=d.read(p) if p.exists() else {}
    if code or value.get('status')!='COMPLETE' or value.get('count')!=56700 or value.get('seeds')!=list(d.SEEDS) or value.get('target_access') is not False or value.get('U_truth') is not False:
        d.write(d.BASE/'failure.json',dict(status='FAILED',stage='teacher_preparation',exit_code=code,partial_retained=True))
        raise RuntimeError('Teacher cache incomplete; no training launched')


def dispatch():
    with lock(d.ROOT/'owner.lock',nonblocking=True):
        d.BASE.mkdir(parents=True,exist_ok=False);(d.BASE/'logs').mkdir()
        spec=d.read(d.ROOT/'registration/experiment.json')
        if spec['run_id']!=d.RUN or [r['config'] for r in spec['rows']]!=[d.config(r) for r in d.rows()]:raise ValueError('Preregistration differs')
        spec.update(status='RUNNING',execution_commit=(d.ROOT/'release_commit.txt').read_text().strip());d.write(d.BASE/'experiment.json',spec)
        d.write(d.BASE/'dispatcher.json',dict(pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,owner=d.OWNER))
        prepare_teachers()
        pending=d.rows();active={};receipts=[];completed=[];failures=[]
        while pending or active:
            for rowid,child in list(active.items()):
                code=child.poll()
                if code is None:continue
                p=d.BASE/rowid/'completion.json';value=d.read(p) if p.is_file() else {}
                try:
                    if code or value.get('status')!='TRAINED_AND_SCORED':raise ValueError('Row subprocess not complete')
                    require_scored(d.read(d.BASE/'row_scoring'/rowid/'scoring_complete.json'),1)
                    completed.append(rowid)
                except Exception as exc:failures.append(dict(row_id=rowid,exit_code=code,error=repr(exc)))
                del active[rowid]
            if pending and not failures:
                with lock(Path(d.PROJECT)/'runs/receiver_residual_capacity4.lock'):
                    states=capacity();reserved={item['pid']:item['gpu'] for item in receipts if item['row_id'] in active};ready=[]
                    for gpu,state in states.items():
                        pids=state['pids']|{pid for pid,g in reserved.items() if g==gpu}
                        if len(pids)<3 and state['free_mb']>=12000:ready.append((len(pids),-state['free_mb'],gpu))
                    if ready:
                        gpu=min(ready)[2];row=pending.pop(0);cmd=[sys.executable,'-u','-m','experiments.cvs_scratch_engineering.dispatch','--worker',row['row_id']]
                        log=d.BASE/'logs'/f"{row['row_id']}.log"
                        with log.open('x') as f:child=subprocess.Popen(cmd,cwd=d.ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu)),stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
                        active[row['row_id']]=child;receipts.append(dict(row_id=row['row_id'],pid=child.pid,gpu=gpu,cwd=str(d.ROOT),argv=cmd,log=str(log)))
                        d.write(d.BASE/'launch.json',dict(rows=receipts))
            d.write(d.BASE/'queue_state.json',dict(active=list(active),pending=pending,completed=completed,failures=failures,max_total_processes_per_GPU=3,updated_at=time.time()))
            if failures and not active:break
            if pending or active:time.sleep(3)
        if failures:
            d.write(d.BASE/'failure.json',dict(status='FAILED',rows=failures,no_automatic_retry=True));raise RuntimeError('Failures retained; no repeat launch')
        if len(completed)!=len(d.rows()):raise ValueError('Incomplete E20 budget Fishr matrix')
        cmd=[sys.executable,'-u','-m','experiments.cvs_scratch_engineering.evaluate','--mode','score']
        with (d.BASE/'logs/aggregate.log').open('x') as f:subprocess.run(cmd,cwd=d.ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdout=f,stderr=subprocess.STDOUT,check=True)
        require_scored(d.read(d.BASE/'scoring_complete.json'),len(d.rows()))
        d.write(d.BASE/'completion.json',dict(status='EXPERIMENT_COMPLETE',trained_models=len(completed),tested_models=len(completed),rows=completed))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--worker');p.add_argument('--predict-row');p.add_argument('--score-row');p.add_argument('--prepare-worker',action='store_true');a=p.parse_args()
    if a.prepare_worker:
        gpu=int(os.environ['CUDA_VISIBLE_DEVICES'])
        while True:
            with lock(Path(d.PROJECT)/'runs/receiver_residual_capacity4.lock'):
                s=capacity()[gpu]
                if len(s['pids']|{os.getpid()})<=3 and s['free_mb']>=12000:break
            time.sleep(3)
        from .prepare import prepare
        prepare()
    elif a.worker:worker(a.worker)
    elif a.predict_row:
        from .evaluate import predict
        predict(a.predict_row);replace('--score-row',a.predict_row)
    elif a.score_row:
        from .evaluate import score
        score(a.score_row);require_scored(d.read(d.BASE/'row_scoring'/a.score_row/'scoring_complete.json'),1)
        d.write(d.BASE/a.score_row/'completion.json',dict(status='TRAINED_AND_SCORED',row_id=a.score_row))
    else:dispatch()
