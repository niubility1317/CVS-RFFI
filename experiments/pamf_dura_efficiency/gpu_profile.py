"""Source-only GPU step attribution; no optimizer or deployable new weights."""
import argparse
import gc
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

import numpy as np
import torch

from experiments.pamf_dura_direct_train import fast_lr
from experiments.pamf_dura_direct_train import runner as training
from experiments.pamf_dura_diagnostics.runner import source_indices,tensor
from .model import direct_objective,supervised_action

RUN='20261030-diagnostic-DURA-training-profile-s2026092701-r01'
MODULE='experiments.pamf_dura_efficiency.gpu_profile'
SOURCE='20261030-phase1-DURA-direct-fastlr-s2026092701-r01'
SOURCE_COMMIT='a0acfbe0c237e38d0460a9f1ea29c46b22cf84a6'
CASES=('individual','buffered')
SWITCH_FACTORY=None
METRIC_MODE=None


def source(root,device):
    fast_lr.configure()
    p=training.paths(root)
    if p['output'].name!=SOURCE:raise ValueError('Predesignated source model required')
    done=training.read_json(p['output']/'scratch_joint/training_complete.json')
    expected=dict(status='SOURCE_TRAINED',epoch=20,updates=4940,L_exposures=126000,U_exposures=1134000,
        training_TX_labels_read=0,training_F_access=False,V_reads=0,target_reads=0,
        learned_teacher_inputs=0,initialization_checkpoint_loads=0)
    if any(done.get(k)!=v for k,v in expected.items()):raise ValueError('Complete legal source model required')
    model=training.load_model(p['output']/'scratch_joint/final.pt',device,arm='scratch_joint',commit=SOURCE_COMMIT)
    clean,roles=training.source_training(p)
    return model,clean,roles,p


def benchmark(args):
    model,clean,roles,p=source(args.root,'cuda:0')
    model.train().requires_grad_(True)
    indices=source_indices(roles)[:256]
    folder=p['cache']/'epochs/20'
    arrays={key:np.load(folder/(key+'.npy'),mmap_mode='r',allow_pickle=False)
            for key in ('received','kernel','coefficients','variance')}
    before={key:value.detach().clone() for key,value in model.state_dict().items()}
    parameters=list(model.parameters())
    switch=SWITCH_FACTORY(model) if SWITCH_FACTORY is not None else None
    training.write_json(args.output/'worker.json',dict(pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,
        physical_GPU=os.environ['CUDA_VISIBLE_DEVICES'],code_commit=args.commit))

    def step(mode,copy_gradients=False):
        if switch is not None:switch(mode)
        torch.cuda.synchronize();started=time.perf_counter()
        for parameter in parameters:parameter.grad=None
        # Reset the probe RNG so the two logging paths see identical inputs.
        torch.manual_seed(2026103006)
        with torch.profiler.record_function('dura_source_prepare'):
            target=tensor(clean[indices],'cuda:0')
            values={key:tensor(value[indices],'cuda:0') for key,value in arrays.items()}
            action=supervised_action(values['kernel'],values['coefficients'])
            probes=[torch.complex(torch.randn(256,256,device='cuda:0'),torch.randn(256,256,device='cuda:0')) for _ in range(2)]
        with torch.profiler.record_function('dura_forward_objective'):
            loss,terms=direct_objective(model,values['received'],target,action,values['variance'],probes)
            if not torch.isfinite(loss).all():raise FloatingPointError('Nonfinite source objective')
        with torch.profiler.record_function('dura_backward_clip'):
            loss.mean().backward()
            norm=torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True)
            clipped=int(norm>1)
        with torch.profiler.record_function('dura_metric_copy'):
            sums=[value.detach().sum() for value in dict(loss=loss,**terms).values()]
            metrics=([float(value) for value in sums] if (METRIC_MODE or mode)=='individual' else torch.stack(sums).cpu().tolist())
        torch.cuda.synchronize();elapsed=time.perf_counter()-started
        gradients=[None if v.grad is None else v.grad.detach().clone() for v in parameters] if copy_gradients else None
        return elapsed,metrics,clipped,gradients

    # Allow the small repeatability error of existing CUDA reductions.
    _,first,clip_a,reference=step(CASES[0],True)
    _,second,clip_b,other=step(CASES[1],True)
    if clip_a!=clip_b:raise ValueError('Logging path changed clipping state')
    torch.testing.assert_close(torch.tensor(first),torch.tensor(second),rtol=2e-5,atol=2e-6)
    for a,b in zip(reference,other):
        if (a is None)!=(b is None):raise ValueError('Logging path changed gradient participation')
        if a is not None:torch.testing.assert_close(a,b,rtol=2e-5,atol=2e-6)
    del reference,other
    timings={key:[] for key in CASES};rounds=[]
    for round_index in range(3):
        order=CASES if round_index%2==0 else CASES[::-1]
        record={}
        for mode in order:
            samples=[step(mode)[0] for _ in range(3)]
            timings[mode]+=samples;record[mode]=samples
        rounds.append(record)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],
                                record_shapes=False,with_stack=False,profile_memory=False) as profiler:
        for _ in range(4):step(CASES[0])
    events=[dict(name=event.key,count=event.count,self_cpu_us=event.self_cpu_time_total,
                 cpu_total_us=event.cpu_time_total,self_cuda_us=event.self_cuda_time_total,
                 cuda_total_us=event.cuda_time_total) for event in profiler.key_averages()]
    for parameter in parameters:parameter.grad=None
    if any(not torch.equal(value,before[key]) for key,value in model.state_dict().items()):
        raise ValueError('Diagnostic changed model state')
    summaries={key:dict(samples=len(values),median_seconds=statistics.median(values),
                       min_seconds=min(values),max_seconds=max(values)) for key,values in timings.items()}
    training.write_json(args.output/'results.json',dict(status='ANALYZED_SOURCE_ONLY',source_run=SOURCE,
        source_commit=SOURCE_COMMIT,epoch=20,source_examples=256,batch=256,
        L_s=int((roles[indices]=='L_s').sum()),U_s=int((roles[indices]=='U_s').sum()),
        repeats=timings,rounds=rounds,summaries=summaries,
        cases=list(CASES),metric_copy_mode=METRIC_MODE or 'case-specific',
        relative_reduction_fraction=1-summaries[CASES[1]]['median_seconds']/summaries[CASES[0]]['median_seconds'],
        values_and_clipped_gradients_close=True,gradient_rtol=2e-5,gradient_atol=2e-6,
        model_unchanged=True,optimizer_updates=0,TX_labels_read=0,
        F_reads=0,V_reads=0,target_reads=0,CPU_threads=2,physical_GPU=os.environ['CUDA_VISIBLE_DEVICES'],
        profile_steps=4,profile_events=events,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        scope='Shared-card same-source no-update steps; full objective/backward/clip/data gather and copy/logging; excludes optimizer step and epoch-cache construction; not end-to-end training speed or quality'))
    training.write_json(args.output/'complete.json',dict(status='ANALYZED_SOURCE_ONLY',optimizer_updates=0,predictions=0,goal_complete=False))


def launch(args):
    import fcntl
    if args.output.name!=RUN or args.output.exists() or os.environ.get('CUDA_VISIBLE_DEVICES')!='':
        raise ValueError('New immutable output and CPU-only controller required')
    args.output.mkdir()
    training.write_json(args.output/'launch.json',dict(pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,commit=args.commit))
    model,clean,roles,_=source(args.root,'cpu')
    with torch.no_grad():
        value=model(tensor(clean[source_indices(roles)[:2]],'cpu'))
        if value.shape!=(2,2,256) or not torch.isfinite(value).all():raise ValueError('Source checkpoint noquery smoke failed')
    training.write_json(args.output/'smoke.json',dict(status='PASS',real_Ls=2,F_reads=0,TX_labels_read=0,target_reads=0))
    del model,clean,roles;gc.collect()
    print(json.dumps(dict(event='SOURCE_CHECKPOINT_NOQUERY_SMOKE_PASS')),flush=True)
    while True:
        with (args.root/'runs/receiver_residual_capacity4.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            state=training.capacity()
            commands=subprocess.check_output(['ps','-eo','pid=,args='],text=True).splitlines()
            tokens=('pamf_dura_prior_refine.runner worker','pamf_dura_prior_recovery.runner worker',
                'pamf_dura_direct_train.runner worker','pamf_dura_direct_train.fast_lr worker',
                'pamf_dura_direct_train.identity_init worker','pamf_dura_direct_train.no_variance worker',
                'pamf_dura_direct_train.fast_interp worker','pamf_dura_direct_train.strong_action worker',MODULE+' worker')
            own=sum(any(token in command for token in tokens) for command in commands)
            used=set()
            for command in commands:
                if not any(token in command for token in tokens):continue
                pid=command.split(None,1)[0]
                try:environment=(Path('/proc')/pid/'environ').read_bytes().split(b'\0')
                except FileNotFoundError:continue
                for item in environment:
                    if item.startswith(b'CUDA_VISIBLE_DEVICES='):
                        used.update(int(v) for v in item.split(b'=',1)[1].split(b',') if v.isdigit())
            free=[gpu for gpu,row in state.items() if gpu in range(8) and gpu not in used and row['free_mb']>=12000 and len(row['pids'])<2]
            free.sort(key=lambda gpu:-state[gpu]['free_mb'])
            if own<2 and free:
                gpu=free[0]
                command=[sys.executable,'-u','-m',MODULE,'worker','--root',str(args.root),'--output',str(args.output),'--commit',args.commit]
                with (args.output/'gpu.log').open('xb') as stream:
                    process=subprocess.Popen(command,cwd=Path.cwd(),env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu)),
                        stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                training.write_json(args.output/'submitted.json',dict(pid=process.pid,GPU=gpu,command=command))
                print(json.dumps(dict(event='SUBMITTED',pid=process.pid,GPU=gpu)),flush=True)
                break
        print(json.dumps(dict(event='WAITING_CAPACITY',own_GPU_workers=own)),flush=True);time.sleep(30)
    while process.poll() is None:time.sleep(5)
    if process.returncode or not (args.output/'complete.json').exists():
        raise RuntimeError('Source profiler failed; preserve outputs without retry')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('run','worker'))
    for key in ('root','output'):parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--commit',required=True);args=parser.parse_args()
    args.root=args.root.resolve();args.output=args.output.resolve()
    if Path('release_commit.txt').read_text().strip()!=args.commit:raise ValueError('Pinned release required')
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if args.mode=='run':launch(args)
    else:benchmark(args)


if __name__=='__main__':main()
