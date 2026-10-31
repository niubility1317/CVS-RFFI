"""One owner: matched scratch30 rows, frozen tests, then independent truth-side."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from .cost_probe import gpu_choice,write

RUNS={
    'IQ':'20261101-phase3-P3Restore-TF-IQ-E30-s2026092701-r01',
    'F0':'20261101-phase3-P3Restore-TF-F0-E30-s2026092701-r01',
}
MODULE='experiments.pamf_phase3_restoration.'


def commands(args):
    common=['--root',str(args.root),'--arm',args.arm,'--commit',args.commit]
    return [
        ('train',[sys.executable,'-u','-m',MODULE+'train',*common,'--cache',str(args.cache),'--inputs',str(args.inputs)]),
        ('phase3_predict',[sys.executable,'-u','-m',MODULE+'phase3_eval','predict','--config',str(args.root/'runs'/RUNS[args.arm]/'phase3_config.json')]),
        ('historical_predict',[sys.executable,'-u','-m',MODULE+'historical_eval','predict',*common]),
        ('phase3_score',[sys.executable,'-u','-m',MODULE+'phase3_eval','score','--prediction-root',str(args.root/'runs'/RUNS[args.arm]/'phase3_prediction'),'--output',str(args.root/'runs'/RUNS[args.arm]/'phase3_score')]),
        ('historical_score',[sys.executable,'-u','-m',MODULE+'historical_eval','score',*common])]


def row(args):
    out=args.root/'runs'/RUNS[args.arm]
    write(out/'phase3_config.json',dict(run_id=RUNS[args.arm],arm=args.arm,code_commit=args.commit,
        reference_runtime=str(args.reference),candidate_checkpoint=str(out/'training/final.pt'),
        output_root=str(out/'phase3_prediction')))
    stages=commands(args);started=time.perf_counter();records=[]
    for stage,command in stages:
        env=dict(os.environ)
        if stage=='historical_predict':
            base=args.root/'releases/pamf_stage2_inference_20261021_r01'
            env['PYTHONPATH']=':'.join(map(str,[Path.cwd(),base,base/'code']))
        if stage.endswith('_score'):env['CUDA_VISIBLE_DEVICES']=''
        else:
            gpu=int(os.environ['CUDA_VISIBLE_DEVICES'])
            while not any(a['GPU']==gpu for a in gpu_choice()):
                print('WAIT_ALLOWED_GPU_FOR_STAGE',args.arm,stage,gpu,flush=True);time.sleep(20)
        begin=time.perf_counter()
        # Both prediction processes must terminate successfully before either
        # independent scorer is started. Failed output roots are never reused.
        with (out/'logs'/(stage+'.log')).open('xb') as log:
            if stage.endswith('_score'):
                import fcntl
                with (args.root/'runs'/RUNS['IQ']/'score.lock').open('a') as lock:
                    fcntl.flock(lock,fcntl.LOCK_EX)
                    subprocess.run(command,cwd=Path.cwd(),env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,check=True)
            else:subprocess.run(command,cwd=Path.cwd(),env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,check=True)
        records.append(dict(stage=stage,seconds=time.perf_counter()-begin))
        print('P3_TF_PIPELINE_STAGE_COMPLETE',args.arm,stage,records[-1]['seconds'],flush=True)
    write(out/'complete.json',dict(status='ANALYZED',run_id=RUNS[args.arm],code_commit=args.commit,stages=records,
        seconds=time.perf_counter()-started,all_registered_predictions_before_truth=True,
        default_DURA_changed=False,goal_complete=False,target_feedback=False))


def controller(args):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CUDA-empty launch owner required')
    for arm,run in RUNS.items():
        out=args.root/'runs'/run;out.mkdir();(out/'logs').mkdir()
        write(out/'launch.json',dict(controller_PID=os.getpid(),CWD=os.getcwd(),argv=sys.argv,code_commit=args.commit))
    pending=list(RUNS);active={};failed=[];last=0.
    while pending or active:
        for arm,(process,gpu) in list(active.items()):
            code=process.poll()
            if code is not None:
                write(args.root/'runs'/RUNS[arm]/'controller_complete.json',
                    dict(status='ROW_FINISHED' if code==0 else 'FAILED_PRESERVED',exit_code=code,pid=process.pid,GPU=gpu))
                if code:failed.append(arm)
                del active[arm]
        choices=[a for a in gpu_choice() if a['GPU'] not in [gpu for _,gpu in active.values()]]
        while pending and choices and len(active)<2:
            arm=pending.pop(0);admission=choices.pop(0);out=args.root/'runs'/RUNS[arm]
            cmd=[sys.executable,'-u','-m',MODULE+'pipeline','row','--root',str(args.root),'--commit',args.commit,
                 '--arm',arm,'--inputs',str(args.inputs),'--cache',str(args.cache),'--reference',str(args.reference)]
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(admission['GPU']))
            with (out/'logs/pipeline.log').open('xb') as stream:
                process=subprocess.Popen(cmd,cwd=Path.cwd(),env=env,stdin=subprocess.DEVNULL,stdout=stream,
                                         stderr=subprocess.STDOUT,start_new_session=True)
            active[arm]=(process,admission['GPU'])
            write(out/'submitted.json',dict(pid=process.pid,command=cmd,CWD=os.getcwd(),admission=admission))
        if time.monotonic()-last>30:
            print('P3_TF_QUEUE',json.dumps(dict(pending=pending,active={a:dict(pid=v[0].pid,GPU=v[1]) for a,v in active.items()},failed=failed)),flush=True)
            last=time.monotonic()
        if pending or active:time.sleep(5)
    if failed:raise RuntimeError('Failed rows preserved: '+str(failed))


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['start','row'])
    p.add_argument('--root',type=Path,required=True);p.add_argument('--inputs',type=Path,required=True)
    p.add_argument('--cache',type=Path,required=True);p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--commit',required=True);p.add_argument('--arm',choices=tuple(RUNS));args=p.parse_args()
    if (Path.cwd()/'release_commit.txt').read_text().strip()!=args.commit:raise ValueError('Actual execution Git commit required')
    if args.mode=='start':controller(args)
    else:
        if args.arm is None:p.error('--arm required')
        try:row(args)
        except Exception as e:
            write(args.root/'runs'/RUNS[args.arm]/'pipeline_failure.json',dict(status='FAILED_PRESERVED',type=type(e).__name__,error=str(e)))
            raise


if __name__=='__main__':main()
