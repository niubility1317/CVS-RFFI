"""Three new E50 rows; two archived controls; per-row automatic truth-last."""
import argparse,os,subprocess,sys,time
from pathlib import Path
from . import design as d
from experiments.cvs_receiver_residual_capacity4.control import lock
from experiments.cvs_multi_state_action.dispatch import capacity


def execute(mode,row):
    env=dict(os.environ)
    if mode=='score':env['CUDA_VISIBLE_DEVICES']=''
    os.execve(sys.executable,[sys.executable,'-u','-m','experiments.cvs_urban_semantic50.dispatch','--mode',mode,'--row',row],env)


def worker(row):
    c=d.config(next(r for r in d.fresh_rows() if r['row_id']==row))
    with lock(d.BASE/(row+'.owner.lock'),nonblocking=True):
        gpu=int(os.environ['CUDA_VISIBLE_DEVICES'])
        with lock(Path(d.PROJECT)/'runs/receiver_residual_capacity4.lock'):
            if len(capacity()[gpu]['pids']|{os.getpid()})>3:raise ValueError('Shared total GPU process limit exceeded')
        import torch
        torch.cuda.init()
        from .source import train
        train(c)
    execute('predict',row)


def smoke_and_references():
    import torch
    from .model import own_model
    records=[]
    for arm in ('D1','D2'):
        c=d.config(next(r for r in d.rows() if r['arm']==arm));source=Path(c['output_root']);root=source.parent
        # Completion handles only; no target metrics are read here or used to
        # select an initializer, calibration, candidate or source qualification.
        pred=d.read(root/'prediction/complete.json');score=d.read(root/'scoring/complete.json')
        if pred['status']!='PREDICTIONS_COMPLETE' or pred['views']!=list(d.VIEWS) or pred['truth_read'] or score['status']!='SCORED_COMPLETE':
            raise ValueError('Archived exact control predictions/scores incomplete')
        model=own_model(c,'cpu')
        with torch.no_grad():out=model.forward_identity_only(torch.zeros(2,2,256))['tx_logits']
        if out.shape!=(2,6) or not torch.isfinite(out).all():raise ValueError('Real archived default checkpoint smoke failed')
        records.append(dict(logical_arm=arm,original_config=c['reuse_source_config'],source=str(source),original_predictions=str(root/'prediction'),original_scores=str(root/'scoring'),head_fit_updates=0,query_IQ_read=False))
        del model
    d.write(d.BASE/'source_smoke.json',dict(status='PASS',source_only=True,query_read=False,forward_packets=4,
        checkpoint_initialization_used=False,scope='Two read-only exact default-head controls; all fresh rows scratch',references=records))
    print('SMOKE PASS real fixed controls/default heads; query_read=false; three new rows scratch',flush=True)


def dispatch():
    with lock(d.ROOT/'semantic50_owner.lock',nonblocking=True):
        if d.BASE.exists():raise FileExistsError('Existing run must be reconciled, never overwritten/resubmitted')
        spec=d.read(d.ROOT/'experiments/cvs_urban_semantic50/registration.json')
        if [r['config'] for r in spec['rows']]!=[d.config(r) for r in d.rows()]:raise ValueError('Registered five-row matrix differs')
        d.BASE.mkdir();(d.BASE/'logs').mkdir();spec['code']['commit']=(d.ROOT/'release_commit.txt').read_text().strip()
        d.write(d.BASE/'experiment.json',spec);d.write(d.BASE/'dispatcher.json',dict(pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,owner=d.OWNER))
        try:
            smoke_and_references()
            from .calibrate import run
            run()
        except Exception as exc:
            d.write(d.BASE/'failure.json',dict(status='FAILED',phase='owned source smoke/calibration',error=str(exc),no_automatic_retry=True))
            raise
        pending=d.fresh_rows();active={};launched=[];complete=[];failures=[]
        while pending or active:
            for rid,child in list(active.items()):
                code=child.poll()
                if code is None:continue
                path=d.BASE/rid/'complete.json';done=d.read(path) if path.is_file() else {}
                if code or done.get('status')!='TRAINED_AND_SCORED':failures.append(dict(row_id=rid,exit_code=code,completion=done))
                else:complete.append(rid)
                del active[rid]
            if pending and not failures:
                with lock(Path(d.PROJECT)/'runs/receiver_residual_capacity4.lock'):
                    caps=capacity();choices=[g for g,v in caps.items() if len(v['pids'])<3 and v['free_mb']>=12000]
                    if choices:
                        gpu=min(choices,key=lambda g:(len(caps[g]['pids']),-caps[g]['free_mb'],g));row=pending.pop(0)
                        command=[sys.executable,'-u','-m','experiments.cvs_urban_semantic50.dispatch','--mode','worker','--row',row['row_id']]
                        log=d.BASE/'logs'/(row['row_id']+'.log')
                        with log.open('x') as f:
                            child=subprocess.Popen(command,cwd=d.ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu)),
                                stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
                        active[row['row_id']]=child
                        launched.append(dict(row_id=row['row_id'],pid=child.pid,gpu=gpu,cwd=str(d.ROOT),argv=command,log=str(log),
                            owner=d.OWNER,launched_at=time.time(),capacity_before={str(k):dict(pids=sorted(v['pids']),free_mb=v['free_mb']) for k,v in caps.items()}))
                        d.write(d.BASE/'launch.json',dict(rows=launched))
            d.write(d.BASE/'queue_state.json',dict(active=list(active),pending=pending,completed=complete,failures=failures,
                reused_controls=['D1=A0','D2=A-J'],shared_lock='receiver_residual_capacity4.lock',all_owner_preCUDA_count=True,
                max_total_experiments_per_GPU=3,policy='spread on idle cards, borrow all unused capacity, no science expansion to fill GPU',updated_at=time.time()))
            if failures and not active:break
            if pending or active:time.sleep(3)
        if failures:
            d.write(d.BASE/'failure.json',dict(status='FAILED',rows=failures,pending_preserved=pending,no_automatic_retry=True))
            raise RuntimeError('Owned technical stop; preserve artifacts, healthy rows finish')
        from .report import source_decision,collect
        d.write(d.BASE/'source_family_decision.json',source_decision())
        collect()
        d.write(d.BASE/'complete.json',dict(status='ANALYZED',actual_model_count=3,reused_model_count=2,rows=complete,
            logical_row_count=5,source_only_design=True,target_feedback_forbidden=True,seed_count=1,cross_seed_confirmation='NOT_RUN'))


def score_and_report(row):
    from .evaluate import score
    score(row);values=d.read(d.BASE/row/'scoring/scores.json')['results'];lines=['# '+row,'',
        'Fixed E50/internal source fold/default head; all seven predictions before independent truth-last.','',
        '|view|accuracy %|Macro-F1 %|','|---|---:|---:|']
    for v in values:
        if v['dimension']=='overall':lines.append(f"|{v['view']}|{100*v['accuracy']:.4f}|{100*v['macro_f1']:.4f}|")
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
