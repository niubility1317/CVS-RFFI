import argparse,json,os,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT/'code') not in sys.path:sys.path.append(str(ROOT/'code'))
import torch
from . import design as d
from .models import load_model,validate_members
from .evaluation import predict,score
from experiments.pamf_feature_refine.target import readers
from experiments.pamf_restoration.data import IQSource,write_json
from experiments.pamf_restoration.runner import array_tensor,code_commit
from experiments.pamf_restoration.dispatcher import capacity


@torch.no_grad()
def source_smoke(args):
    x=array_tensor(IQSource(args.source_cache,'L_s').iq[:2],'cpu');frozen=readers(args,'cpu')
    for arm in d.ARMS:
        model=load_model(args.run_root,arm,'cpu');value=model(x)
        if (not torch.equal(value,.5*model.base(x)+.5*model.partner(x)) or
            any(p.requires_grad for p in model.parameters()) or model.training):
            raise ValueError('Exact fixed frozen IQ composition required')
        for reader in frozen.values():
            scores=reader(value)
            if scores.shape!=(2,6) or not torch.isfinite(scores).all():raise ValueError('Actual composed original-head smoke failed')
        del model
    write_json(args.output/'smoke.json',dict(status='PASS',real_source_samples=2,arms=list(d.ARMS),
        actual_frozen_member_checkpoints=True,exact_equal_IQ_mean=True,F_seeds=list(d.F_SEEDS),original_default_heads=True,
        TX_labels_read=0,V_reads=0,target_reads=0,optimizer_updates=0,head_fit_updates=0))


def acquire(args):
    import fcntl
    while True:
        with (args.output.parent/'receiver_residual_capacity4.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX);state=capacity()[d.GPUS[args.arm]]
            if state['free_mb']>=12000 and len(state['pids'])<2:
                model=load_model(args.run_root,args.arm,'cuda:0');torch.cuda.synchronize();return model
        time.sleep(5)


def worker(args):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=str(d.GPUS[args.arm]):raise ValueError('Registered physical GPU required')
    row=args.output/args.arm;row.mkdir();write_json(row/'composition.json',d.configuration(args.arm))
    try:
        started=time.perf_counter();model=acquire(args);torch.cuda.reset_peak_memory_stats()
        write_json(row/'gpu_ready.json',dict(pid=os.getpid(),cwd=str(ROOT),argv=sys.argv,physical_GPU=d.GPUS[args.arm]))
        print(json.dumps(dict(event='SOURCE_IQMEAN_READY',arm=args.arm,pid=os.getpid(),GPU=d.GPUS[args.arm])),flush=True)
        predict(args,model)
        write_json(row/'inference_complete.json',dict(status='SOURCE_INFERENCE_COMPLETE',count=1800,
            seconds=time.perf_counter()-started,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            model_parameters=sum(p.numel() for p in model.parameters()),new_training_epochs=0,parameter_updates=0,
            F_optimizer_updates=0,head_fit_updates=0,TX_labels_read=0,target_reads=0,state_updates=0))
        del model;torch.cuda.empty_cache()
        import fcntl
        with (args.output/'source_score.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            command=[sys.executable,'-X','utf8','-m','experiments.pamf_iqmean_restore.runner','score']
            for key in ('output','parent_cache','source_cache','arm'):command.extend(['--'+key.replace('_','-'),str(getattr(args,key))])
            subprocess.run(command,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),check=True)
        write_json(row/'complete.json',dict(status='ANALYZED',source_only=True,target_reads=0,new_training_epochs=0,parameter_updates=0))
    except BaseException as error:
        write_json(row/'failure.json',dict(status='FAILED_TECHNICAL',error=repr(error),traceback=traceback.format_exc(),automatic_retry=False));raise


def run(args):
    if args.output.exists():raise FileExistsError('Immutable new output required')
    if code_commit()!=args.expected_commit:raise ValueError('Pinned code/config commit required')
    members=validate_members(args.run_root);args.output.mkdir(parents=True)
    write_json(args.output/'launch.json',dict(pid=os.getpid(),cwd=str(ROOT),argv=sys.argv,commit=args.expected_commit,
        owner='codex/root/iqmean',source_members=members,source_only=True,new_training_epochs=0))
    source_smoke(args)
    logs=args.output/'logs';logs.mkdir();children={};launches=[]
    for arm in d.ARMS:
        command=[sys.executable,'-u','-m','experiments.pamf_iqmean_restore.runner','worker','--arm',arm]
        for key in ('output','parent_cache','source_cache','run_root','parent_restoration','reader_folder'):
            command.extend(['--'+key.replace('_','-'),str(getattr(args,key))])
        with (logs/(arm+'.log')).open('xb') as log:
            process=subprocess.Popen(command,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(d.GPUS[arm])),
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        children[arm]=process;launches.append(dict(arm=arm,pid=process.pid,argv=command,GPU=d.GPUS[arm]))
    write_json(args.output/'workers.json',dict(workers=launches))
    codes={arm:p.wait() for arm,p in children.items()}
    if any(codes.values()):
        write_json(args.output/'failure.json',dict(status='FAILED_TECHNICAL',exit_codes=codes,healthy_rows_preserved=True,automatic_retry=False));return
    current={arm:json.loads((args.output/arm/'summary.json').read_text()) for arm in d.ARMS}
    control=json.loads((args.run_root/d.BASE_RUN/'posterior03/summary.json').read_text())
    decision=d.source_rule(current,control)
    write_json(args.output/'source_decision.json',dict(status='SOURCE_COMPARISON_FROZEN',**decision,target_reads=0))
    write_json(args.output/'source_complete.json',dict(status='ANALYZED',source_only=True,arms=list(d.ARMS),target_reads=0,new_training_epochs=0))
    if decision['qualified']:
        arm=decision['selected_arm']
        command=[sys.executable,'-u','-m','experiments.pamf_iqmean_restore.target','run',
            '--source-run',str(args.output),'--output',str(args.output/'target'),'--arm',arm]
        for key in ('source_cache','formal_residual','reader_root','target_truth','expected_commit'):
            command.extend(['--'+key.replace('_','-'),str(getattr(args,key))])
        with (logs/'target.log').open('xb') as log:
            process=subprocess.Popen(command,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(d.GPUS[arm])),
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        write_json(args.output/'target_launch.json',dict(pid=process.pid,argv=command,GPU=d.GPUS[arm]))
        if process.wait()!=0:
            write_json(args.output/'failure.json',dict(status='FAILED_TECHNICAL',stage='target',automatic_retry=False));return
    write_json(args.output/'complete.json',dict(status='ANALYZED',source_only=not decision['qualified'],arms=list(d.ARMS),
        source_qualified=decision['qualified'],target_evaluation='COMPLETE' if decision['qualified'] else 'NOT_QUALIFIED_NOT_ACCESSED',new_training_epochs=0))


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('run','worker','score'))
    for key in ('output','parent-cache','source-cache'):p.add_argument('--'+key,type=Path,required=True)
    for key in ('run-root','parent-restoration','reader-folder','reader-root','formal-residual','target-truth'):p.add_argument('--'+key,type=Path)
    p.add_argument('--arm',choices=d.ARMS);p.add_argument('--expected-commit');args=p.parse_args()
    for key,value in vars(args).items():
        if isinstance(value,Path):setattr(args,key,value.resolve())
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if args.mode=='score':
        if args.arm is None:p.error('Registered source arm required')
        score(args);return
    if any(getattr(args,k) is None for k in ('run_root','parent_restoration','reader_folder')):p.error('Actual source members/reference/head required')
    if args.mode=='worker':
        if args.arm is None:p.error('Registered arm required')
        worker(args)
    else:
        if any(getattr(args,k) is None for k in ('reader_root','formal_residual','target_truth','expected_commit')):p.error('Automatic target inputs required')
        run(args)


if __name__=='__main__':main()
