"""One launch owner: train each new row, freeze, predict seven views, truth-last."""
import argparse,os,subprocess,sys,time
from pathlib import Path
from . import design as d
from experiments.cvs_receiver_residual_capacity4.control import lock
from experiments.cvs_multi_state_action.dispatch import capacity


def execute(mode,row):
    env=dict(os.environ)
    if mode=='score':env['CUDA_VISIBLE_DEVICES']=''
    os.execve(sys.executable,[sys.executable,'-u','-m','experiments.cvs_urban_bridge50.dispatch','--mode',mode,'--row',row],env)


def worker(row):
    c=d.config(next(r for r in d.rows() if r['row_id']==row))
    with lock(d.BASE/(row+'.owner.lock'),nonblocking=True):
        gpu=int(os.environ['CUDA_VISIBLE_DEVICES'])
        with lock(Path(d.PROJECT)/'runs/receiver_residual_capacity4.lock'):
            if len(capacity()[gpu]['pids']|{os.getpid()})>3:raise ValueError('Shared all-owner GPU process limit exceeded')
        import torch
        torch.cuda.init()
        from .source import train
        train(c)
    execute('predict',row)


def smoke_and_references():
    import torch
    from experiments.cvs_urban_branch50.model import own_model
    references=d.references();records={};fold=None
    for name,c in references.items():
        source=Path(c['output_root']);root=source.parent
        done=d.read(source/'completion.json');frozen=d.read(source/'source_frozen.json');inner=d.read(source/'inner_source_contract.json')
        pred=d.read(root/'prediction/complete.json');score=d.read(root/'scoring/complete.json')
        if (done['config']!=c or frozen['config']!=c or done['epoch']!=50 or done['optimizer_steps']!=11100
                or done['target_access'] or done['head_fit_updates'] or done['initialization']!='scratch'
                or done['checkpoint_sources'] or done['ancestors'] or frozen['selection']!='final_only' or frozen['target_access']
                or pred['status']!='PREDICTIONS_COMPLETE' or pred['views']!=list(d.VIEWS) or pred['truth_read']
                or score['status']!='SCORED_COMPLETE' or not score['truth_last']):
            raise ValueError('Exact predefined source/default-head control provenance differs: '+name)
        if fold is None:fold=inner
        if inner!=fold:raise ValueError('Original reference physical source folds differ')
        records[name]=dict(source=str(source),config=c,completed=True,head_fit_updates=0,
            original_prediction=str(root/'prediction'),original_scoring=str(root/'scoring'),new_forward_packets=0)
    model=own_model(references['NOMINAL'],'cpu')
    with torch.no_grad():out=model.forward_identity_only(torch.zeros(2,2,256))['tx_logits']
    if out.shape!=(2,6) or not torch.isfinite(out).all():raise ValueError('Real source/default checkpoint smoke failed')
    d.write(d.BASE/'source_smoke.json',dict(status='PASS',query_read=False,source_only=True,
        forward_packets=2,checkpoint_initialization_used=False,references=records,
        scope='Read-only real A0 default checkpoint smoke; both new students and own EMAs start scratch'))
    print('SMOKE PASS actual A0/default head; query_read=false; scratch rows continue',flush=True)


def dispatch():
    with lock(d.ROOT/'bridge50_owner.lock',nonblocking=True):
        if d.BASE.exists():raise FileExistsError('Existing run must be reconciled, never resubmitted')
        spec=d.read(d.ROOT/'experiments/cvs_urban_bridge50/registration.json')
        if [r['config'] for r in spec['rows']]!=[d.config(r) for r in d.rows()]:raise ValueError('Registered two-row bridge matrix differs')
        d.BASE.mkdir();(d.BASE/'logs').mkdir();spec['code']['commit']=(d.ROOT/'release_commit.txt').read_text().strip()
        d.write(d.BASE/'experiment.json',spec);d.write(d.BASE/'dispatcher.json',dict(pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,owner=d.OWNER))
        try:smoke_and_references()
        except Exception as exc:
            d.write(d.BASE/'failure.json',dict(status='FAILED',phase='owned real-CP source smoke',error=str(exc),no_automatic_retry=True));raise
        pending=d.rows();active={};launched=[];complete=[];failures=[]
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
                        command=[sys.executable,'-u','-m','experiments.cvs_urban_bridge50.dispatch','--mode','worker','--row',row['row_id']]
                        log=d.BASE/'logs'/(row['row_id']+'.log')
                        with log.open('x') as f:child=subprocess.Popen(command,cwd=d.ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu)),
                            stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
                        active[row['row_id']]=child
                        launched.append(dict(row_id=row['row_id'],pid=child.pid,gpu=gpu,cwd=str(d.ROOT),argv=command,log=str(log),
                            owner=d.OWNER,launched_at=time.time(),capacity_before={str(k):dict(pids=sorted(v['pids']),free_mb=v['free_mb']) for k,v in caps.items()}))
                        d.write(d.BASE/'launch.json',dict(rows=launched))
            d.write(d.BASE/'queue_state.json',dict(active=list(active),pending=pending,completed=complete,failures=failures,
                shared_lock='receiver_residual_capacity4.lock',all_owner_preCUDA_count=True,max_total_experiments_per_GPU=3,
                policy='spread across available cards; no science expansion to fill GPU; never stop healthy foreign jobs',updated_at=time.time()))
            if failures and not active:break
            if pending or active:time.sleep(3)
        if failures:
            d.write(d.BASE/'failure.json',dict(status='FAILED',rows=failures,pending_preserved=pending,no_automatic_retry=True))
            raise RuntimeError('Owned technical stop; artifacts retained, healthy rows finished')
        from .report import source_decision,collect
        d.write(d.BASE/'source_family_decision.json',source_decision());collect()
        d.write(d.BASE/'complete.json',dict(status='ANALYZED',actual_model_count=2,reused_model_count=3,rows=complete,
            source_only_design=True,target_feedback_forbidden=True,seed_count=1,cross_seed_confirmation='NOT_RUN',
            frozen_local_mechanism_attribution='separate source analysis still required'))


def score_and_report(row):
    from .evaluate import score
    score(row);values=d.read(d.BASE/row/'scoring/scores.json')['results']
    lines=['# '+row,'','Fixed E50/default head; all seven predictions before independent truth-last.','',
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
