"""Synthetic matched full Phase3 fixed-system timing, after frozen scores."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from . import source_train as candidate
from . import phase3_eval as evaluation
from experiments.pamf_phase3_open_world import fresh_dura_run as reference


def summarize(times,batch):
    values=np.asarray(times,dtype=np.float64)
    if values.ndim!=1 or not len(values) or not np.isfinite(values).all() or np.any(values<=0) or batch<1:
        raise ValueError('Positive finite matched timing samples required')
    return dict(iterations=len(values),median_ms=float(np.median(values)),p95_ms=float(np.quantile(values,.95)),
        throughput_samples_s=float(batch*1000/values.mean()))


def check_exclusive_gpu():
    physical=os.environ.get('CUDA_VISIBLE_DEVICES','')
    if not physical.isdigit():raise ValueError('A single physical GPU is required')
    GPUs=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader,nounits'],text=True).splitlines()
    lookup={v[0].strip():v[1].strip() for line in GPUs if len(v:=line.split(','))==2}
    uuid=lookup[physical]
    processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True).splitlines()
    pids={int(v[1].strip()) for line in processes if len(v:=line.split(','))==2 and v[0].strip()==uuid}
    if pids-{os.getpid()}:raise RuntimeError('Another compute process contaminated the matched GPU timing')
    return sorted(pids)


def validate_prediction(existing,run,runtime,commit):
    expected=dict(status='IDENTITY_FLOW_PHASE3_PREDICTIONS_COMPLETE',run_id=candidate.RUN,
        candidate_code_commit=commit,private_truth_reads=0,
        counts=dict(known=27000,novel=3150,clean=10238,star=10238))
    if (any(existing.get(k)!=v for k,v in expected.items())
            or Path(existing['config']['reference_runtime']).resolve()!=runtime.resolve()
            or Path(existing['config']['candidate_checkpoint']).resolve()!=(run/candidate.ARM/'final.pt').resolve()):
        raise ValueError('The exact frozen candidate and complete predictions are required')


def profile(root,runtime,commit,profile_commit,output,device):
    run=root/'runs'/candidate.RUN;done=evaluation._read(run/'complete.json')
    if done.get('status')!='ANALYZED':raise ValueError('Complete fixed training and scoring required before cost profiling')
    existing=evaluation._read(run/'phase3_predict/complete.json')
    validate_prediction(existing,run,runtime,commit)
    if output.exists():raise FileExistsError('New immutable profile root required')
    cfg=reference.validate_config(evaluation._read(runtime)['eval_config'])
    parent,states,tau,_=reference._reference(cfg)
    backend=reference._backend(cfg,device);pcd,pcd_identity=reference._load_model(cfg,device)
    models=dict(E0=backend['model'],PCD=pcd)
    if (backend['model_identity']!=parent['model_identities']['E0'] or pcd_identity!=parent['model_identities']['TARGET_PCD']
            or backend['reader_identity']!=parent['parent_F0_identity'] or backend['dura_identity']!=parent['loaded_dura_identity']
            or existing['parent_F0_identity']!=parent['parent_F0_identity']
            or existing['model_identities']!=dict(E0=parent['model_identities']['E0'],PCD=parent['model_identities']['TARGET_PCD'])):
        raise ValueError('Matched fixed Phase3 model identities required')
    flow=candidate.load(run/candidate.ARM/'final.pt',device,commit)
    modules=[backend['reader'],backend['dura'][0],*models.values(),flow]
    snapshots=[(m,reference.fresh.best._snapshot_model(m)) for m in modules]
    frozen_states=deepcopy(states);frozen_tau=tau.copy()
    from experiments.pamf_phase3_open_world.star_open_feature_export import _synthetic
    original=np.asarray(_synthetic('cpu').detach().cpu().tolist(),dtype=np.float32)
    if original.shape!=(2,2,256) or not np.isfinite(original).all():raise ValueError('Fixed synthetic G10 signals required')
    class Dura:
        def __call__(self,x):return backend['dura'][1](x)[0].detach().clone()
    def system(restorer,values):
        result=evaluation._current(backend,models,restorer,values,device)
        raw=reference._raw(states,result['signal_features'],result['closed'])
        scores=raw-tau[:,result['closed']].T
        if scores.shape[0]!=len(values) or not np.isfinite(scores).all():raise ValueError('Finite full-system scores required')
        return scores
    duration={};restorers=[('DURA',Dura()),('identity_flow',flow)]
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.cuda.reset_peak_memory_stats()
    output.mkdir(parents=True)
    with torch.no_grad():
        check_exclusive_gpu()
        for _,restorer in restorers:system(restorer,original)
        torch.cuda.synchronize()
        evaluation._write(output/'smoke.json',dict(status='PASS',actual_candidate_F0_E0_PCD_and_DURA=True,
            synthetic_G10_inputs=2,real_IQ_reads=0,truth_reads=0,optimizer_updates=0))
        for batch in (1,128):
            values=np.tile(original,((batch+1)//2,1,1))[:batch]
            samples={name:[] for name,_ in restorers}
            for repeat in range(3):
                sequence=restorers if repeat%2==0 else list(reversed(restorers))
                for name,restorer in sequence:
                    check_exclusive_gpu()
                    for _ in range(2):system(restorer,values)
                    torch.cuda.synchronize()
                    for _ in range(8):
                        before=time.perf_counter();system(restorer,values);torch.cuda.synchronize()
                        samples[name].append(1000*(time.perf_counter()-before))
                    check_exclusive_gpu()
            duration['batch_'+str(batch)]={k:summarize(v,batch) for k,v in samples.items()}
    for model,snapshot in snapshots:reference.fresh.best._check_model(model,snapshot)
    if not reference._same(states,frozen_states) or not np.array_equal(tau,frozen_tau):raise ValueError('Frozen states changed')
    evaluation._write(output/'complete.json',dict(status='MATCHED_PHASE3_PROFILE_COMPLETE',candidate_run=candidate.RUN,
        candidate_code_commit=commit,profile_code_commit=profile_commit,reference_runtime=str(runtime.resolve()),
        physical_GPU=os.environ.get('CUDA_VISIBLE_DEVICES'),cases=duration,
        input='two fixed synthetic G10 signals repeated; not real packets or additional observations',
        timing_scope='Same GPU alternating three rounds, two warmups and eight timed iterations per round; DURA uses existing CUDA graph, flow uses eager; full G + F0 + E0 + PCD + source density/threshold scores, input CPU array and output CPU scores included; loading, graph setup and scheduling excluded',
        GPU_occupancy_check='Exclusive GPU at admission and sampled before/after each timing block; cannot exclude brief unobserved transient processes',
        candidate_inference=candidate.config()['inference'],precision='FP32, TF32 disabled',
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),memory_scope='All shared benchmark models resident; not standalone deployment memory',
        F0=parent['parent_F0_identity'],fixed_PCD_has_target_U_ancestor=True,real_IQ_reads=0,truth_reads=0,optimizer_updates=0))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--reference-runtime',type=Path,required=True);parser.add_argument('--candidate-commit',required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--profile-commit',required=True)
    args=parser.parse_args()
    if Path('release_commit.txt').read_text(encoding='ascii').strip()!=args.profile_commit:raise ValueError('Pinned profile release required')
    if not os.environ.get('CUDA_VISIBLE_DEVICES'):raise ValueError('Explicit assigned GPU required')
    torch.set_num_threads(2);profile(args.root,args.reference_runtime,args.candidate_commit,args.profile_commit,args.output,'cuda:0')


if __name__=='__main__':main()
