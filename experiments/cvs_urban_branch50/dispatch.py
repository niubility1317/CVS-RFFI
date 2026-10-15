"""Eight scratch rows; one common source calibration and per-row automatic truth-last."""
import argparse,os,subprocess,sys,time
from pathlib import Path
from . import design as d
from experiments.cvs_receiver_residual_capacity4.control import lock
from experiments.cvs_multi_state_action.dispatch import capacity


def execute(mode,row):
    os.execv(sys.executable,[sys.executable,'-u','-m','experiments.cvs_urban_branch50.dispatch','--mode',mode,'--row',row])


def worker(row):
    c=d.config(next(r for r in d.rows() if r['row_id']==row))
    with lock(d.BASE/(row+'.owner.lock'),nonblocking=True):
        gpu=int(os.environ['CUDA_VISIBLE_DEVICES'])
        with lock(Path(d.PROJECT)/'runs/receiver_residual_capacity4.lock'):
            if len(capacity()[gpu]['pids']|{os.getpid()})>3:raise ValueError('Total GPU experiment capacity exceeded')
        import torch
        torch.cuda.init()
        from .source import train
        train(c)
    execute('predict',row)


def smoke():
    import torch
    from experiments.cvs_urban_sigtime50 import design as old
    from experiments.cvs_urban_sigtime50.model import own_model
    from . import branches as b
    from .channel import audit
    # Old full-L checkpoint is an operational no-query smoke only. It is never
    # used as an initializer, EMA target or internal-fold selection/calibration.
    reference=own_model(old.config(next(r for r in old.rows() if r['arm']=='B0')),'cpu')
    with torch.no_grad():out,values=b.forward(reference,torch.zeros(2,2,256))
    if out['tx_logits'].shape!=(2,6) or not torch.isfinite(out['tx_logits']).all():raise ValueError('Real checkpoint smoke failed')
    d.write(d.BASE/'source_smoke.json',dict(status='PASS',query_read=False,
        checkpoint_initialization_used=False,scope='frozen full-L B0 operational smoke only',channel=audit()))
    print('SMOKE PASS real default checkpoint; query_read=false; all eight new models scratch/internal-fold',flush=True)


def ready(pending):
    # A0 first, then all other positions once the epoch5 global coefficients
    # exist. No source/target performance test filters individual pilot arms.
    if not pending:return False
    if pending[0]['arm']=='A0':return True
    if not d.calibration_path().is_file():return False
    d.recipe();return True


def dispatch():
    with lock(d.ROOT/'branch50_owner.lock',nonblocking=True):
        if d.BASE.exists():raise FileExistsError('Existing run: reconcile without resubmitting')
        spec=d.read(d.ROOT/'experiments/cvs_urban_branch50/registration.json')
        if [r['config'] for r in spec['rows']]!=[d.config(r) for r in d.rows()]:raise ValueError('Registered eight-row matrix differs')
        d.BASE.mkdir();(d.BASE/'logs').mkdir()
        spec['code']['commit']=(d.ROOT/'release_commit.txt').read_text().strip();d.write(d.BASE/'experiment.json',spec)
        d.write(d.BASE/'dispatcher.json',dict(pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,owner=d.OWNER))
        smoke();pending=d.rows();active={};launched=[];completed=[];failures=[]
        while pending or active:
            for row,child in list(active.items()):
                code=child.poll()
                if code is None:continue
                p=d.BASE/row/'complete.json';done=d.read(p) if p.is_file() else {}
                if code or done.get('status')!='TRAINED_AND_SCORED':failures.append(dict(row_id=row,exit_code=code,completion=done))
                else:completed.append(row)
                del active[row]
            if pending and not failures and ready(pending):
                with lock(Path(d.PROJECT)/'runs/receiver_residual_capacity4.lock'):
                    reserved={v['pid']:v['gpu'] for v in launched if v['row_id'] in active};available=[]
                    for gpu,state in capacity().items():
                        ids=state['pids']|{p for p,g in reserved.items() if g==gpu}
                        if len(ids)<3 and state['free_mb']>=12000:available.append((len(ids),-state['free_mb'],gpu))
                    if available:
                        gpu=min(available)[2];row=pending.pop(0)
                        command=[sys.executable,'-u','-m','experiments.cvs_urban_branch50.dispatch','--mode','worker','--row',row['row_id']]
                        log=d.BASE/'logs'/(row['row_id']+'.log')
                        with log.open('x') as stream:
                            child=subprocess.Popen(command,cwd=d.ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu)),
                                stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                        active[row['row_id']]=child
                        launched.append(dict(row_id=row['row_id'],pid=child.pid,gpu=gpu,cwd=str(d.ROOT),argv=command,log=str(log)))
                        d.write(d.BASE/'launch.json',dict(rows=launched))
            d.write(d.BASE/'queue_state.json',dict(active=list(active),pending=pending,completed=completed,failures=failures,
                waiting_for_source_epoch5_calibration=bool(pending and not d.calibration_path().is_file()),
                max_total_experiments_per_GPU=3,updated_at=time.time()))
            if failures and not active:break
            if active or pending:time.sleep(3)
        if failures:
            d.write(d.BASE/'failure.json',dict(status='FAILED',rows=failures,pending_preserved=pending,no_automatic_retry=True))
            raise RuntimeError('Owned technical stop; preserve outputs and allow healthy rows to finish')
        from .selection import from_sources
        d.write(d.BASE/'source_family_decision.json',from_sources())
        from .report import collect
        result=collect(completed)
        d.write(d.BASE/'complete.json',dict(status='ANALYZED',rows=completed,actual_model_count=len(completed),
            all_completed_rows_seven_views=True,unlaunched=[],performance_status=result['performance_status'],
            source_only_design=True,target_feedback_forbidden=True,seed_count=1,cross_seed_confirmation='NOT_RUN'))


def score_and_report(row):
    from .evaluate import score
    score(row);values=d.read(d.BASE/row/'scoring/scores.json')['results']
    lines=['# '+row,'','Fixed E50/default head, internal L fold. All seven predictions fixed before independent truth-last scoring.','',
        '| view | accuracy (%) | Macro-F1 (%) |','|---|---:|---:|']
    for r in values:
        if r['dimension']=='overall':lines.append(f"| {r['view']} | {100*r['accuracy']:.4f} | {100*r['macro_f1']:.4f} |")
    (d.BASE/row/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    d.write(d.BASE/row/'complete.json',dict(status='TRAINED_AND_SCORED',row_id=row,target_feedback_forbidden=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['worker','predict','score']);parser.add_argument('--row');args=parser.parse_args()
    if args.mode=='worker':worker(args.row)
    elif args.mode=='predict':
        from .evaluate import predict
        predict(args.row);execute('score',args.row)
    elif args.mode=='score':score_and_report(args.row)
    else:dispatch()
