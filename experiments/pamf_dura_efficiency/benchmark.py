"""CPU source-only training-step timing; zero optimizer updates or new weights."""
import argparse
import copy
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

from .model import scratch_model,supervised_action,direct_objective
from experiments.pamf_dura_diagnostics.runner import tensor
from experiments.pamf_dura_prior_refine.model import stage_details,objective
from experiments.pamf_dura_prior_refine.runner import paths,parent_model,source_training,write_json

RUN='20261030-diagnostic-DURA-single-stage-cost-s2026092701-r01'


def run(args):
    if args.output.name!=RUN or args.output.exists() or os.environ.get('CUDA_VISIBLE_DEVICES')!='':
        raise ValueError('Immutable output root and CPU-only cost diagnosis required')
    p=paths(args.root)
    clean,roles=source_training(p)
    take=np.flatnonzero(roles=='L_s')[:256]
    folder=p['cache']/'epochs/01'
    received=np.load(folder/'received.npy',mmap_mode='r',allow_pickle=False)
    values={key:np.load(folder/(key+'.npy'),mmap_mode='r',allow_pickle=False) for key in ('kernel','coefficients','variance')}
    trained,_=parent_model(p,'cpu')
    teacher=copy.deepcopy(trained).eval().requires_grad_(False)
    random=scratch_model(2026092701)
    assert sum(v.numel() for v in random.parameters())==266951
    args.output.mkdir(parents=True)
    write_json(args.output/'launch.json',dict(pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,commit=args.commit,
        CPU_threads=2,GPU_workers=0,owner='codex/root/dura-single-stage-cost'))
    # Actual inherited checkpoint and random graph both consume two real L_s.
    x=tensor(clean[take[:2]],'cpu');y=tensor(received[take[:2]],'cpu')
    with torch.no_grad():
        if not torch.isfinite(trained(y)).all() or not torch.isfinite(random(y)).all():
            raise ValueError('Actual source no-query smoke failed')
    print(json.dumps(dict(event='SOURCE_NO_QUERY_SMOKE_PASS',real_Ls=2,F_reads=0,target_reads=0)),flush=True)
    models=dict(online_teacher=copy.deepcopy(trained).train().requires_grad_(True),
                cached_teacher=copy.deepcopy(trained).train().requires_grad_(True),
                scratch_joint=random)
    before={name:{k:v.detach().clone() for k,v in model.state_dict().items()} for name,model in models.items()}
    results={}
    for batch in (32,256):
        x=tensor(clean[take[:batch]],'cpu');y=tensor(received[take[:batch]],'cpu')
        true=supervised_action(tensor(values['kernel'][take[:batch]],'cpu'),tensor(values['coefficients'][take[:batch]],'cpu'))
        noise=tensor(values['variance'][take[:batch]],'cpu')
        generator=torch.Generator().manual_seed(2026103003)
        probes=[torch.complex(torch.randn(batch,256,generator=generator),torch.randn(batch,256,generator=generator)) for _ in range(2)]
        with torch.no_grad():cached=teacher.stage_outputs(y)[0]
        for name,model in models.items():
            elapsed=[];losses=[]
            for iteration in range(8):
                model.zero_grad(set_to_none=True)
                started=time.perf_counter()
                if name=='scratch_joint':loss,terms=direct_objective(model,y,x,true,noise,probes)
                else:
                    if name=='online_teacher':
                        with torch.no_grad():desired=teacher.stage_outputs(y)[0]
                    else:desired=cached
                    stages,priors=stage_details(model,y)
                    loss,terms=objective(stages,priors,desired,x,1,'continuation')
                if not torch.isfinite(loss).all():raise FloatingPointError('Nonfinite source train-step objective')
                loss.mean().backward()
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                seconds=time.perf_counter()-started
                if iteration>=2:elapsed.append(seconds);losses.append(float(loss.detach().mean()))
            results[f'{name}_B{batch}']=dict(median_seconds=float(np.median(elapsed)),p95_seconds=float(np.quantile(elapsed,.95)),
                samples_per_second=float(batch/np.mean(elapsed)),timed_iterations=6,warmup=2,
                example_loss=losses[-1],trainable_parameters=sum(v.numel() for v in model.parameters()))
            print(json.dumps(dict(event='TRAIN_STEP_TIMED',case=f'{name}_B{batch}',**results[f'{name}_B{batch}'])),flush=True)
    for name,model in models.items():
        if any(not torch.equal(v,before[name][k]) for k,v in model.state_dict().items()):
            raise ValueError('Timing-only diagnostic changed model state')
        model.zero_grad(set_to_none=True)
    write_json(args.output/'results.json',dict(status='ANALYZED_SOURCE_ONLY',cases=results,
        source_records=256,training_updates=0,F_reads=0,TX_labels_read=0,V_reads=0,target_reads=0,
        models_unchanged=True,scratch_initialization=random.initialization_report,
        scope='CPU2 forward/objective/backward/gradientclip;excludesoptimizerstep,IO,transfer,init,teacher-cacheproduction;notGPUthroughputorquality',
        candidate='Single joint source training from random full DURA graph;pairedclean plus source simulator action/noise supervision;0 learnedteacher/ancestorweights',
        cached_teacher_cost='Cache was computed outside timing for this same fixed diagnostic batch;not a totaltrainingcostclaim'))
    write_json(args.output/'complete.json',dict(status='ANALYZED_SOURCE_ONLY',optimizer_updates=0,predictions=0,goal_complete=False))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('root','output'):parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--commit',required=True);args=parser.parse_args()
    if Path('release_commit.txt').read_text(encoding='ascii').strip()!=args.commit:raise ValueError('Pinned release required')
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    run(args)


if __name__=='__main__':main()
