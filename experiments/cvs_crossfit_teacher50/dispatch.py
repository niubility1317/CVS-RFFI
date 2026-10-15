"""Two source-only scratch teachers, automatic frozen source evaluations."""
import argparse,os,subprocess,sys,time
from pathlib import Path
from . import design as d
from experiments.cvs_receiver_residual_capacity4.control import lock
from experiments.cvs_multi_state_action.dispatch import capacity


def execute(mode,row):
    os.execve(sys.executable,[sys.executable,'-u','-m','experiments.cvs_crossfit_teacher50.dispatch','--mode',mode,'--row',row],
        dict(os.environ,CUDA_VISIBLE_DEVICES=''))


def worker(row):
    c=d.config(next(r for r in d.rows() if r['row_id']==row));gpu=int(os.environ['CUDA_VISIBLE_DEVICES'])
    with lock(d.BASE/(row+'.owner.lock'),nonblocking=True):
        with lock(Path(d.PROJECT)/'runs/receiver_residual_capacity4.lock'):
            if len(capacity()[gpu]['pids']|{os.getpid()})>3:raise ValueError('All-owner GPU process limit exceeded')
        import torch
        torch.cuda.init()
        from .source import train
        train(c)
    execute('predict',row)


def smoke():
    import torch
    from experiments.cvs_urban_branch50.model import own_model
    model=own_model(d.operational_parent(),'cpu')
    with torch.no_grad():value=model.forward_identity_only(torch.zeros(2,2,256))['tx_logits']
    if value.shape!=(2,6) or not torch.isfinite(value).all():raise ValueError('Real default checkpoint smoke failed')
    d.write(d.BASE/'source_smoke.json',dict(status='PASS',source_only=True,query_read=False,forward_packets=2,
        source=d.operational_parent()['output_root'],used_as_initializer=False,
        scope='Read-only original A0 gate reference; both teachers start from scratch without its weights'))
    print('SMOKE PASS original default-head source checkpoint; query_read=false; two scratch teachers continue',flush=True)


def dispatch():
    with lock(d.ROOT/'crossfit_teacher50_owner.lock',nonblocking=True):
        if d.BASE.exists():raise FileExistsError('Existing run must be reconciled, never overwritten')
        spec=d.read(d.ROOT/'experiments/cvs_crossfit_teacher50/registration.json')
        if [r['config'] for r in spec['rows']]!=[d.config(r) for r in d.rows()]:raise ValueError('Registered source two-teacher matrix differs')
        d.BASE.mkdir();(d.BASE/'logs').mkdir();spec['code']['commit']=(d.ROOT/'release_commit.txt').read_text().strip()
        d.write(d.BASE/'experiment.json',spec);d.write(d.BASE/'dispatcher.json',dict(pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,owner=d.OWNER))
        try:smoke()
        except Exception as exc:
            d.write(d.BASE/'failure.json',dict(status='FAILED',phase='real source smoke',error=str(exc),no_automatic_retry=True));raise
        pending=d.rows();active={};launched=[];completed=[];failures=[]
        while pending or active:
            for row,child in list(active.items()):
                code=child.poll()
                if code is None:continue
                path=d.BASE/row/'complete.json';done=d.read(path) if path.is_file() else {}
                if code or done.get('status')!='SOURCE_TRAINED_AND_EVALUATED':failures.append(dict(row_id=row,exit_code=code,completion=done))
                else:completed.append(row)
                del active[row]
            if pending and not failures:
                with lock(Path(d.PROJECT)/'runs/receiver_residual_capacity4.lock'):
                    caps=capacity();choices=[g for g,v in caps.items() if len(v['pids'])<3 and v['free_mb']>=12000]
                    if choices:
                        gpu=min(choices,key=lambda g:(len(caps[g]['pids']),-caps[g]['free_mb'],g));row=pending.pop(0)
                        command=[sys.executable,'-u','-m','experiments.cvs_crossfit_teacher50.dispatch','--mode','worker','--row',row['row_id']]
                        log=d.BASE/'logs'/(row['row_id']+'.log')
                        with log.open('x') as stream:child=subprocess.Popen(command,cwd=d.ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu)),
                            stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                        active[row['row_id']]=child
                        launched.append(dict(row_id=row['row_id'],pid=child.pid,gpu=gpu,cwd=str(d.ROOT),argv=command,log=str(log),
                            owner=d.OWNER,launched_at=time.time(),capacity_before={str(k):dict(pids=sorted(v['pids']),free_mb=v['free_mb']) for k,v in caps.items()}))
                        d.write(d.BASE/'launch.json',dict(rows=launched))
            d.write(d.BASE/'queue_state.json',dict(active=list(active),pending=pending,completed=completed,failures=failures,
                source_only=True,max_total_processes_per_GPU=3,all_owners_preCUDA=True,updated_at=time.time()))
            if failures and not active:break
            if pending or active:time.sleep(3)
        if failures:
            d.write(d.BASE/'failure.json',dict(status='FAILED',rows=failures,pending_preserved=pending,no_automatic_retry=True))
            raise RuntimeError('Owned source failure retained; healthy jobs unchanged')
        from .evaluate import compare
        from .report import collect
        compare();collect()
        d.write(d.BASE/'complete.json',dict(status='ANALYZED',source_only=True,query_read=False,actual_model_count=2,
            student_models_trained=0,seed_count=1,automatic_bank_expansion=False,target_predictions=0))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['worker','predict']);p.add_argument('--row');a=p.parse_args()
    if a.mode=='worker':worker(a.row)
    elif a.mode=='predict':
        from .evaluate import predict
        predict(a.row);d.write(d.BASE/a.row/'complete.json',dict(status='SOURCE_TRAINED_AND_EVALUATED',source_only=True,query_read=False))
    else:dispatch()
