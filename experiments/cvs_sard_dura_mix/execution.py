import argparse,os,subprocess,sys,time,traceback
from pathlib import Path
from . import design as d
from experiments.cvs_receiver_residual_capacity4.control import lock
from experiments.cvs_multi_state_action.dispatch import capacity

def configure():
    from experiments.cvs_sard_dura_eval.execution import configure as inherited
    inherited()

def worker(c):
    from experiments.cvs_equivariant_identity.precision import numerical_context
    from .preflight import smoke
    from .train import train
    from .evaluate import predict
    with lock(d.BASE/(c['row_id']+'.lock'),nonblocking=True):
        gpu = int(os.environ['CUDA_VISIBLE_DEVICES'])
        with lock(d.PROJECT/'runs/receiver_residual_capacity4.lock'):
            if gpu not in range(8) or len(capacity()[gpu]['pids']|{os.getpid()})>3:
                raise ValueError('All-owner GPU capacity exceeded')
        root = d.row_root(c['row_id'])
        root.mkdir()
        with numerical_context(d.FULL_FP32_POLICY):
            smoke(c)
            train(c)
            predict(c)
    os.execve(sys.executable,[sys.executable,'-u',str(d.ROOT/'experiments/cvs_sard_dura_mix/runner.py'),
        '--mode','score','--row',c['row_id']],dict(os.environ,CUDA_VISIBLE_DEVICES=''))

def collect():
    result=[]
    for row in d.rows():
        root=d.row_root(row['row_id'])
        if d.read(root/'complete.json')['status']!='ANALYZED':
            raise ValueError('Both matched rows must finish before group closure')
        result.append(dict(row=row,training=d.read(root/'source/result.json'),scores=d.read(root/'scoring/scores.json')))
    d.write(d.BASE/'results.json',dict(rows=result,model_seed_count=1,views=list(d.VIEWS),
        target_feedback_forbidden=True,calendar='scratch50=13L+37U',matched_calendar=True))
    d.write(d.BASE/'complete.json',dict(status='ANALYZED',rows=d.rows(),model_seed_count=1,views=list(d.VIEWS),truth_last=True))

def dispatch():
    if d.BASE.exists():
        raise FileExistsError('Immutable run exists; reconcile,never duplicate')
    d.BASE.mkdir()
    (d.BASE/'logs').mkdir()
    d.write(d.BASE/'launch.json',dict(pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,owner=d.OWNER,
        commit=(d.ROOT/'release_commit.txt').read_text().strip(),rows=d.rows()))
    pending,active,launched,completed = d.rows(),{ },[],[]
    try:
        while pending or active:
            for rid,info in list(active.items()):
                code=info['child'].poll()
                if code is None:continue
                done=d.row_root(rid)/'complete.json'
                if code or not done.exists() or d.read(done)['status']!='ANALYZED':
                    raise RuntimeError('Own technical failure;preserve healthy siblings/outputs;no retry: '+rid)
                completed.append(rid)
                del active[rid]
            if pending and len(active)<2:
                with lock(d.PROJECT/'runs/receiver_residual_capacity4.lock'):
                    states=capacity()
                    used={v['gpu'] for v in active.values()}
                    slots=sorted((len(v['pids']),-v['free_mb'],g) for g,v in states.items()
                        if g in range(8) and g not in used and len(v['pids'])<3 and v['free_mb']>=12000)
                    while slots and pending and len(active)<2:
                        gpu=slots.pop(0)[2]
                        row=pending.pop(0)
                        cmd=[sys.executable,'-u',str(d.ROOT/'experiments/cvs_sard_dura_mix/runner.py'),'--mode','worker','--row',row['row_id']]
                        log=d.BASE/'logs'/(row['row_id']+'.log')
                        with log.open('x') as f:
                            child=subprocess.Popen(cmd,cwd=d.ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu)),
                                stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
                        active[row['row_id']]=dict(child=child,gpu=gpu)
                        launched.append(dict(row=row,pid=child.pid,gpu=gpu,cwd=str(d.ROOT),argv=cmd,log=str(log)))
                        d.write(d.BASE/'workers.json',dict(workers=launched))
            d.write(d.BASE/'queue.json',dict(active=list(active),pending=pending,completed=completed,updated_at=time.time()))
            if active or pending:time.sleep(5)
        collect()
    except BaseException as error:
        d.write(d.BASE/'failure.json',dict(status='FAILED_TECHNICAL',error=repr(error),traceback=traceback.format_exc(),
            preserve_outputs=True,no_automatic_retry=True,healthy_siblings_not_stopped=True))
        raise

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--mode',choices=['worker','score'])
    parser.add_argument('--row')
    args=parser.parse_args()
    configure()
    if args.mode is None:return dispatch()
    row=next((r for r in d.rows() if r['row_id']==args.row),None)
    if row is None:raise ValueError('Unregistered row')
    c=d.config(row)
    if args.mode=='worker':return worker(c)
    from .evaluate import score
    with lock(d.BASE/'CPU_scorer.lock'):
        score(c)
