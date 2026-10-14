"""Unique row owner; scratch30 -> fixed seven-view prediction -> truth-side exec."""
import argparse,json,os,subprocess,sys,time,traceback
from pathlib import Path
from . import design as d
from experiments.cvs_receiver_residual_capacity4.control import lock,process
from experiments.cvs_multi_state_action.dispatch import capacity

def execute(mode,row):
    os.execv(sys.executable,[sys.executable,'-u','-m','experiments.cvs_urban_sigtime.dispatch','--mode',mode,'--row',row])

def worker(row):
    c=d.config(next(r for r in d.rows() if r['row_id']==row))
    with lock(d.BASE/(row+'.owner.lock'),nonblocking=True):
        gpu=int(os.environ['CUDA_VISIBLE_DEVICES'])
        with lock(Path(d.PROJECT)/'runs/receiver_residual_capacity4.lock'):
            state=capacity()[gpu]
            if len(state['pids']|{os.getpid()})>3:raise ValueError('GPU experiment capacity exceeded')
        import torch
        torch.cuda.init()
        from .source import train
        from . import model as api
        from .observer import Observer
        obs=Observer(c);train(c,design=d,model_api=api,observer=obs);obs.finish()
    execute('predict',row)

def source_qualification():
    """Reads source evidence only. Target scores never decide expansion."""
    first={r['arm']:r for r in d.rows() if r['model_seed']==d.SEEDS[0] and r['arm']!='C4'}
    evidence={};baseline=d.read(d.BASE/first['B0']['row_id']/'source/source_learning_progress.json')['records'][-1]
    for arm,row in first.items():
        root=d.BASE/row['row_id']/'source';done=d.read(root/'completion.json');progress=d.read(root/'source_learning_progress.json')['records']
        mech=d.read(root/'training_mechanism_complete.json')
        if done['optimizer_steps']!=6660 or done['epoch']!=30 or done['target_access'] or mech['status']!='VERIFIED':raise ValueError('Source eligibility is incomplete')
        early=sum(v['hard_margin'] for v in progress[:3])/3;last=progress[-1]
        passes=arm in ('C0','B0') or (last['clean_accuracy']>=baseline['clean_accuracy']-.02 and last['hard_margin']>=early-.01 and done['final_training_L_CE']<2.)
        evidence[arm]=dict(passed=passes,clean=last['clean_accuracy'],hard_margin=last['hard_margin'],early_hard_margin=early,
            exact_update_budget=True,coverage_verified=True if arm!='C0' else None)
    from .model import own_model
    from .panel import load
    import torch
    model=own_model(d.config(first['C3']),'cpu');panel=load()
    with torch.no_grad():p=model.forward_identity_only(panel['x'])['tx_logits'].softmax(1)
    conf,pseudo=p.max(1);mask=conf>=.95;precision=float((pseudo[mask]==panel['y'][mask]).float().mean()) if bool(mask.any()) else 0.
    coverage=float(mask.float().mean())
    ssl=bool(evidence['C3']['passed'] and precision>=.95 and coverage>=.1)
    value=dict(status='SOURCE_QUALIFICATION_FROZEN',source_only=True,target_scores_read=False,
        rules=dict(clean_drop_from_B0_max=.02,hard_margin_progress_floor=-.01,final_L_CE_max=2.,
            fixed_steps=6660,SSL_L_proxy_confidence=.95,SSL_precision_min=.95,SSL_coverage_min=.1),
        arms=evidence,C4_launch=ssl,SSL_source_proxy=dict(precision=precision,coverage=coverage,
            provider='final source student clean L proxy; does not reveal U precision or prove EMA precision'),
        qualified=[a for a in first if evidence[a]['passed']]+(['C4'] if ssl else []),recorded_at=time.time())
    d.write(d.BASE/'source_qualification.json',value);return value

def dispatch():
    with lock(d.ROOT/'urban_owner.lock',nonblocking=True):
        if d.BASE.exists():raise FileExistsError('Run already exists; reconcile independently before retry')
        d.BASE.mkdir();(d.BASE/'logs').mkdir()
        spec=d.read(d.ROOT/'experiments/cvs_urban_sigtime/registration.json')
        if [r['config'] for r in spec['rows']]!=[d.config(r) for r in d.rows()]:raise ValueError('Registered matrix/config differs')
        execution_commit=(d.ROOT/'release_commit.txt').read_text().strip()
        spec['code']['commit']=execution_commit
        d.write(d.BASE/'experiment.json',dict(spec,execution_commit=execution_commit))
        d.write(d.BASE/'dispatcher.json',dict(pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,owner=d.OWNER))
        # One real-checkpoint no-query smoke is the first execution step inside this source study.
        from .diagnose import run
        run()
        pending=[r for r in d.rows() if r['model_seed']==d.SEEDS[0] and r['arm']!='C4']
        active={};receipts=[];completed=[];failures=[];expanded=False;qualified=None
        while pending or active or not expanded:
            for row,child in list(active.items()):
                code=child.poll()
                if code is None:continue
                path=d.BASE/row/'complete.json';done=d.read(path) if path.is_file() else {}
                if code or done.get('status')!='TRAINED_AND_SCORED':failures.append(dict(row_id=row,exit_code=code,error=done))
                else:completed.append(row)
                del active[row]
            first=[r for r in d.rows() if r['model_seed']==d.SEEDS[0] and r['arm']!='C4']
            ready=all((d.BASE/r['row_id']/'source/training_mechanism_complete.json').is_file() for r in first)
            if not expanded and ready and not failures:
                qualified=source_qualification();expanded=True
                if qualified['C4_launch']:pending.append(next(r for r in d.rows() if r['model_seed']==d.SEEDS[0] and r['arm']=='C4'))
                pending.extend(r for r in d.rows() if r['model_seed']!=d.SEEDS[0] and r['arm'] in qualified['qualified'])
            if pending and not failures:
                with lock(Path(d.PROJECT)/'runs/receiver_residual_capacity4.lock'):
                    states=capacity();reserved={r['pid']:r['gpu'] for r in receipts if r['row_id'] in active};ready_gpu=[]
                    for gpu,state in states.items():
                        pids=state['pids']|{pid for pid,g in reserved.items() if g==gpu}
                        if len(pids)<3 and state['free_mb']>=12000:ready_gpu.append((len(pids),-state['free_mb'],gpu))
                    if ready_gpu:
                        gpu=min(ready_gpu)[2];row=pending.pop(0);cmd=[sys.executable,'-u','-m','experiments.cvs_urban_sigtime.dispatch','--mode','worker','--row',row['row_id']]
                        logfile=d.BASE/'logs'/(row['row_id']+'.log')
                        with logfile.open('x',encoding='utf-8') as f:child=subprocess.Popen(cmd,cwd=d.ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu)),stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
                        active[row['row_id']]=child;receipts.append(dict(row_id=row['row_id'],pid=child.pid,gpu=gpu,cwd=str(d.ROOT),argv=cmd,log=str(logfile)))
                        d.write(d.BASE/'launch.json',dict(rows=receipts))
            d.write(d.BASE/'queue_state.json',dict(active=list(active),pending=pending,completed=completed,failures=failures,
                expanded=expanded,max_total_experiments_per_GPU=3,updated_at=time.time()))
            if failures and not active:break
            if pending or active or not expanded:time.sleep(3)
        if failures:
            d.write(d.BASE/'failure.json',dict(status='FAILED',rows=failures,remaining_preserved=pending,no_automatic_retry=True));raise RuntimeError('Failures retained; healthy rows were allowed to finish')
        from .report import collect
        result=collect(completed)
        d.write(d.BASE/'complete.json',dict(status='ANALYZED',rows=completed,unlaunched=[r['row_id'] for r in d.rows() if r['row_id'] not in completed],
            actual_model_count=len(completed),all_completed_rows_seven_views=True,source_qualification=qualified,
            performance_status=result['performance_status'],result_claim_scope='fixed practical proxy, exposed benchmark'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['worker','predict','score']);p.add_argument('--row');a=p.parse_args()
    if a.mode=='worker':worker(a.row)
    elif a.mode=='predict':
        from .evaluate import predict
        predict(a.row);execute('score',a.row)
    elif a.mode=='score':
        from .evaluate import score
        score(a.row);d.write(d.BASE/a.row/'complete.json',dict(status='TRAINED_AND_SCORED',row_id=a.row))
    else:dispatch()
