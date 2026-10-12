import argparse,os,sys,time,subprocess
from pathlib import Path
from . import design as d
from experiments.cvs_receiver_residual_capacity4.control import lock,process
from experiments.cvs_multi_state_action.dispatch import capacity

def stage(name):
    c=d.config(d.rows()[0]);row=c['row_id']
    if name=='train':
        from .source import train
        from .evaluate import configure
        train(c);source=configure().source_provenance(c)
        d.write(d.BASE/row/'source_frozen.json',dict(status='SOURCE_FROZEN',config=c,target_access=False,selection='final_only',epoch=30,optimizer_steps=16890,checkpoint=str(source/'final_ssdg.pth')))
    elif name=='predict':
        from .evaluate import predict
        predict()
    elif name=='score':
        from .evaluate import score
        score()
    else:raise ValueError(name)

def dispatch():
    with lock(d.ROOT/'owner.lock',nonblocking=True):
        d.BASE.mkdir(parents=True,exist_ok=False);(d.BASE/'logs').mkdir()
        spec=d.read(d.ROOT/'registration/experiment.json')
        c=d.config(d.rows()[0]);row=c['row_id']
        if spec['run_id']!=d.RUN or [r['config'] for r in spec['rows']]!=[c]:raise ValueError('Preregistration differs')
        d.write(d.BASE/'experiment.json',dict(spec,status='RUNNING',execution_commit=(d.ROOT/'release_commit.txt').read_text().strip()))
        d.write(d.BASE/'dispatcher.json',dict(pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,owner=d.OWNER))
        try:
            for name in ('train','predict','score'):
                gpu=None
                while True:
                    with lock(Path(d.PROJECT)/'runs/receiver_residual_capacity4.lock'):
                        if name in ('train','predict'):
                            ready=[(len(s['pids']),-s['free_mb'],g) for g,s in capacity().items() if len(s['pids'])<3 and s['free_mb']>=12000]
                            if ready:gpu=min(ready)[2]
                        if name not in ('train','predict') or gpu is not None:
                            command=[sys.executable,'-u','-m','experiments.cvs_oracle80_sat.dispatch','--stage',name]
                            log=d.BASE/'logs'/(name+'.log')
                            with log.open('x') as stream:
                                child=subprocess.Popen(command,cwd=d.ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu) if gpu is not None else ''),stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                            info=dict(stage=name,row_id=row,pid=child.pid,gpu=gpu,cwd=str(d.ROOT),argv=command,log=str(log))
                            d.write(d.BASE/'launch.json',info);d.write(d.BASE/'queue_state.json',dict(stage=name,status='RUNNING',pid=child.pid,gpu=gpu,updated_at=time.time()))
                            break
                    time.sleep(5)
                code=child.wait()
                if code:raise RuntimeError(name+' subprocess failed: '+str(code))
            done=d.read(d.BASE/'scores/scoring_complete.json')
            if done['status']!='SCORED_COMPLETE' or done['prediction_count']!=d.COUNT*len(d.VIEWS) or done['views']!=list(d.VIEWS) or not done['truth_last']:raise ValueError('Incomplete clean scoring')
            d.write(d.BASE/row/'completion.json',dict(status='TRAINED_AND_SCORED',row_id=row))
            d.write(d.BASE/'completion.json',dict(status='EXPERIMENT_COMPLETE',trained_models=1,tested_models=1,seed=2026092701,overall=done['overall']))
            d.write(d.BASE/'queue_state.json',dict(stage='complete',status='EXPERIMENT_COMPLETE',updated_at=time.time()))
        except Exception as exc:
            d.write(d.BASE/'failure.json',dict(status='FAILED',error=repr(exc),no_automatic_retry=True,preserve_artifacts=True));raise

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['train','predict','score']);a=parser.parse_args()
    if a.stage:stage(a.stage)
    else:dispatch()
