"""Two independent scratch E50 rows; automatic seven-view truth-last evaluation."""
import argparse,os,subprocess,sys,time
from pathlib import Path
from . import design as d
from experiments.cvs_receiver_residual_capacity4.control import lock
from experiments.cvs_multi_state_action.dispatch import capacity


def execute(mode,row):
    os.execv(sys.executable,[sys.executable,'-u','-m','experiments.cvs_urban_identity50.dispatch','--mode',mode,'--row',row])


def worker(row):
    c=d.config(next(r for r in d.rows() if r['row_id']==row))
    with lock(d.BASE/(row+'.owner.lock'),nonblocking=True):
        gpu=int(os.environ['CUDA_VISIBLE_DEVICES'])
        with lock(Path(d.PROJECT)/'runs/receiver_residual_capacity4.lock'):
            if len(capacity()[gpu]['pids']|{os.getpid()})>3:raise ValueError('Total GPU process capacity exceeded')
        import torch
        torch.cuda.init()
        from .source import train
        train(c)
    execute('predict',row)


def smoke():
    import torch
    from experiments.cvs_urban_sigtime50 import design as reference
    from experiments.cvs_urban_sigtime50.model import own_model
    from .features import satellite_forward
    from .channel import audit
    recipe=d.recipe()
    calibration=d.read(Path(d.PROJECT)/'runs'/recipe['run_id']/'results.json')
    if not calibration['source_only'] or calibration['query_read'] or calibration['added_training_updates']:
        raise ValueError('Source-only calibration not complete')
    if any(calibration[k]!=recipe[k] for k in ('coefficient','common_training_eta','source_checkpoint','commit')):
        raise ValueError('Static source recipe differs from completed calibration')
    model=own_model(reference.config(next(r for r in reference.rows() if r['arm']=='C3')),'cpu')
    with torch.no_grad():out,z=satellite_forward(model,torch.zeros(2,2,256));output=out['tx_logits']
    if output.shape!=(2,6) or not torch.isfinite(output).all():raise ValueError('Real source-only checkpoint smoke failed')
    d.write(d.BASE/'source_smoke.json',dict(status='PASS',query_read=False,checkpoint_initialization_used=False,channel=audit()))
    print('SMOKE PASS real reference checkpoint; query_read=false; scratch training follows',flush=True)


def dispatch():
    with lock(d.ROOT/'joint160_e50_owner.lock',nonblocking=True):
        if d.BASE.exists():raise FileExistsError('Existing run; reconcile without resubmission')
        spec=d.read(d.ROOT/'experiments/cvs_urban_identity50/registration.json')
        if [r['config'] for r in spec['rows']]!=[d.config(r) for r in d.rows()]:raise ValueError('Registered E50 matrix differs')
        d.BASE.mkdir();(d.BASE/'logs').mkdir()
        commit=(d.ROOT/'release_commit.txt').read_text().strip();spec['code']['commit']=commit
        d.write(d.BASE/'experiment.json',spec)
        d.write(d.BASE/'dispatcher.json',dict(pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,owner=d.OWNER))
        smoke()
        pending=d.rows();active={};launched=[];completed=[];failures=[]
        while pending or active:
            for row,child in list(active.items()):
                code=child.poll()
                if code is None:continue
                path=d.BASE/row/'complete.json';done=d.read(path) if path.is_file() else {}
                if code or done.get('status')!='TRAINED_AND_SCORED':failures.append(dict(row_id=row,exit_code=code,completion=done))
                else:completed.append(row)
                del active[row]
            if pending and not failures:
                with lock(Path(d.PROJECT)/'runs/receiver_residual_capacity4.lock'):
                    reserved={v['pid']:v['gpu'] for v in launched if v['row_id'] in active};available=[]
                    for gpu,state in capacity().items():
                        ids=state['pids']|{p for p,g in reserved.items() if g==gpu}
                        if len(ids)<3 and state['free_mb']>=12000:available.append((len(ids),-state['free_mb'],gpu))
                    if available:
                        gpu=min(available)[2];row=pending.pop(0)
                        command=[sys.executable,'-u','-m','experiments.cvs_urban_identity50.dispatch','--mode','worker','--row',row['row_id']]
                        log=d.BASE/'logs'/(row['row_id']+'.log')
                        with log.open('x') as f:
                            child=subprocess.Popen(command,cwd=d.ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu)),
                                stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
                        active[row['row_id']]=child
                        launched.append(dict(row_id=row['row_id'],pid=child.pid,gpu=gpu,cwd=str(d.ROOT),argv=command,log=str(log)))
                        d.write(d.BASE/'launch.json',dict(rows=launched))
            d.write(d.BASE/'queue_state.json',dict(active=list(active),pending=pending,completed=completed,failures=failures,
                max_total_experiments_per_GPU=3,updated_at=time.time()))
            if failures and not active:break
            if active or pending:time.sleep(3)
        if failures:
            d.write(d.BASE/'failure.json',dict(status='FAILED',rows=failures,pending_preserved=pending,no_automatic_retry=True))
            raise RuntimeError('Systemic technical stop; outputs preserved and healthy rows finished')
        from .report import collect
        result=collect(completed)
        d.write(d.BASE/'complete.json',dict(status='ANALYZED',rows=completed,actual_model_count=len(completed),
            all_completed_rows_seven_views=True,unlaunched=[],performance_status=result['performance_status'],
            source_only_design=True,target_feedback_forbidden=True,seed_count=1,cross_seed_confirmation='NOT_RUN'))


def score_and_report(row):
    from .evaluate import score
    score(row)
    values=d.read(d.BASE/row/'scoring/scores.json')['results']
    lines=['# '+row,'','Fixed E50/default head; all seven predictions fixed before independent truth-last scoring.','',
        '| view | accuracy (%) | Macro-F1 (%) |','|---|---:|---:|']
    for r in values:
        if r['dimension']=='overall':lines.append(f"| {r['view']} | {100*r['accuracy']:.4f} | {100*r['macro_f1']:.4f} |")
    (d.BASE/row/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    d.write(d.BASE/row/'complete.json',dict(status='TRAINED_AND_SCORED',row_id=row,target_feedback_forbidden=True))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['worker','predict','score']);p.add_argument('--row');a=p.parse_args()
    if a.mode=='worker':worker(a.row)
    elif a.mode=='predict':
        from .evaluate import predict
        predict(a.row);execute('score',a.row)
    elif a.mode=='score':score_and_report(a.row)
    else:dispatch()
