"""Source-only bounded replay of the interrupted response training epoch."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

from . import response_joint_prior_trial as science
from . import multiview_trial as trial
from .cache import check_F0_state, load_F0, tensor
from .coupled_covariance import normalize_modes
from .covariance_detail import clip_branch_gradients
from .covariance_model import source_action_loss
from .cost_probe import write
from .mismatched_prior_training import choose_prior, load_training_priors
from .objective import PairedObjective
from .signed_covariance import source_modes
from .source_pair import read, validate_source
from .train import learning_rate
from .variance_trial import load_scale, recovery_gpu_choice

RUN='20261108-diagnostic-Restore-ResponseE10Replay-source-s2026092701-r01'
MODULE='experiments.pamf_phase3_restoration.response_epoch10_diagnostic'
ORIGINAL_COMMIT='ee7f69470c15bd053777cd6b6115c46b39737b9c'
LIMIT=1e-8


def guard_flags(scalar, residual):
    flags=[]
    if not torch.isfinite(scalar).all():flags.append('NONFINITE_LOSS')
    if not torch.isfinite(residual).all():flags.append('NONFINITE_RESIDUAL')
    elif residual.max()>LIMIT:flags.append('POSTERIOR_RESIDUAL_EXCEEDS_LIMIT')
    return flags


def finite_number(value):
    value=float(value)
    return value if np.isfinite(value) else None


def replay(args,out):
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    contract=validate_source(read(args.inputs))
    original=args.root/'runs'/science.RUN
    training=original/'rows'/science.ARM/'training'
    terminal=read(original/('terminal-'+science.ARM+'-train.json'))
    if terminal['exit_code']!=1 or (training/'complete.json').exists():raise ValueError('Exact incomplete failed source run required')
    if any(original.glob('submitted-'+science.ARM+'-cross*.json')):raise ValueError('Original target prediction must not have begun')
    checkpoint=torch.load(training/'recovery/E09.pt',map_location='cpu',weights_only=False)
    if (checkpoint['code_commit'],checkpoint['epoch'],checkpoint['updates'],checkpoint['selection_use'])!=(
            ORIGINAL_COMMIT,9,2223,'technical_recovery_only/not_source_or_target_model_selection'):
        raise ValueError('Exact source-only E09 technical checkpoint required')
    if checkpoint['config']!=science.configuration(science.ARM):raise ValueError('Original scientific configuration required')
    smoke=read(training/'smoke.json')
    if smoke['status']!='PASS' or any(smoke[k] for k in ('TX_labels_read','V_reads','OE_reads','target_inputs')):
        raise ValueError('Original source-only provenance and smoke required')
    with np.load(training/'data_schedule.npz',allow_pickle=False) as schedule:
        order=schedule['orders'][9].copy();draw=int(schedule['draws'][9])
    if draw!=10 or order.shape!=(63000,):raise ValueError('Original epoch10 source order required')
    old,_=trial.source(args)
    # Construct the original G before loading F0, which imports its own modules.
    model=science.factory(science.ARM)
    model.load_state_dict(checkpoint['state'],strict=True);model=model.to('cuda:0').train()
    reader,identity=load_F0(contract,'cuda:0');trial.main_eval.valid_identity(identity)
    frozen={k:v.detach().clone() for k,v in reader.state_dict().items()}
    scale=load_scale(args.root).to('cuda:0');objective=PairedObjective().to('cuda:0')
    pair,assignment=load_training_priors(args.root,'cpu')
    modes=torch.stack([normalize_modes(covariance,source_modes(covariance)) for covariance in pair[1]])
    priors=tuple(v.to('cuda:0') for v in (*pair,modes));assignment=assignment.to('cuda:0')
    def load_array(path):
        array=np.load(path,mmap_mode='r',allow_pickle=False)
        try:return tensor(array,'cuda:0')
        finally:array._mmap.close()
    clean=load_array(old/'train/clean.npy');folder=old/'epochs'/'10'
    if (read(folder/'complete.json')['count'],clean.shape)!=(63000,(63000,2,256)):
        raise ValueError('Original source draw10 and IQ dimensions required')
    values={k:load_array(folder/(k+'.npy')) for k in ('received','kernel','coefficients','variance')}
    with torch.no_grad():targets=torch.cat([reader.features(clean[i:i+256]).detach() for i in range(0,63000,256)])
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=1e-4)
    optimizer.load_state_dict(checkpoint['optimizer'])
    if {float(x['step']) for x in optimizer.state.values() if 'step' in x}!={2223.}:
        raise ValueError('Exact original optimizer boundary required')
    def readout(indices):
        trace={k:v[indices] for k,v in values.items()}
        return science.response_readout(model,reader,values['received'][indices],clean[indices],targets[indices],scale,objective,
            source_indices=indices,epoch=10,priors=priors,assignment=assignment,source_trace=trace)
    first=torch.tensor(order[:2].tolist(),device='cuda:0',dtype=torch.long)
    with torch.no_grad():
        restored,aux,detail,_=readout(first)
        physical,_=source_action_loss(aux,values['kernel'][first],values['coefficients'][first],values['variance'][first])
        if guard_flags((physical+detail).mean(),aux['relative_solve_residual']):
            raise ValueError('Real source checkpoint smoke guard did not pass; preserve diagnostic')
    check_F0_state(reader,frozen)
    write(out/'smoke.json',dict(status='PASS',source_rows=2,actual_original_E09=True,actual_original_F0=True,
        optimizer_updates=0,TX_labels_read=0,V_reads=0,OE_reads=0,target_inputs=0))
    print('RESPONSE_E09_REAL_SOURCE_SMOKE_PASS',flush=True)
    torch.set_rng_state(checkpoint['torch_rng_cpu'])
    torch.cuda.set_rng_state(checkpoint['torch_rng_cuda'])
    update=2223;anomaly=None;started=time.perf_counter()
    for offset in range(0,63000,256):
        take=order[offset:offset+256];optimizer.zero_grad(set_to_none=True)
        for group in optimizer.param_groups:group['lr']=learning_rate(update+1)
        for start in range(0,len(take),32):
            indices=torch.tensor(take[start:start+32].tolist(),device='cuda:0',dtype=torch.long)
            try:
                restored,aux,detail,terms=readout(indices)
                physical,physical_terms=source_action_loss(aux,values['kernel'][indices],values['coefficients'][indices],values['variance'][indices])
                scalar=(physical+detail).sum()/len(take)
                flags=guard_flags(scalar,aux['relative_solve_residual'])
                if flags:
                    anomaly=dict(flags=flags,epoch=10,logical_offset=offset,micro_offset=start,completed_updates=update,
                        microbatch=len(indices),loss=finite_number(scalar.detach()),
                        residual_max=finite_number(aux['relative_solve_residual'].detach().max()),
                        nonfinite_detail=int((~torch.isfinite(detail)).sum()),nonfinite_physical=int((~torch.isfinite(physical)).sum()),
                        nonfinite_response=int((~torch.isfinite(terms['source_response_relative_error'])).sum()))
                    break
                scalar.backward()
            except Exception as error:
                anomaly=dict(flags=['FORWARD_OR_BACKWARD_EXCEPTION'],epoch=10,logical_offset=offset,micro_offset=start,
                    completed_updates=update,exception_type=type(error).__name__,exception_message=str(error)[:500])
                break
        if anomaly is not None:break
        clip_branch_gradients(model);optimizer.step();update+=1
        if (update-2223)%10==0:print('RESPONSE_E10_REPLAY_UPDATES',update-2223,flush=True)
    torch.cuda.synchronize();check_F0_state(reader,frozen)
    write(out/'results.json',dict(status='DIAGNOSTIC_COMPLETE',run_id=RUN,code_commit=args.commit,
        original_run=science.RUN,original_code_commit=ORIGINAL_COMMIT,original_recovery_epoch=9,
        original_recovery_updates=2223,replay_epoch=10,replay_draw=10,replay_optimizer_steps=update-2223,
        outcome='GUARD_OR_EXCEPTION_OBSERVED' if anomaly is not None else 'NO_FAILURE_IN_BOUNDED_REPLAY',anomaly=anomaly,
        elapsed_seconds=time.perf_counter()-started,restored_original_rng=True,original_checkpoint_unchanged=True,
        diagnostic_checkpoint_written=False,TX_labels_read=0,V_reads=0,OE_reads=0,target_inputs=0,original_run_modified=False,
        interpretation='Replay can diverge numerically; observed guard is evidence, no failure does not prove the original cause.'))
    print('RESPONSE_E10_DIAGNOSTIC_COMPLETE',flush=True)


def controller(args):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CUDA-empty controller required')
    out=args.root/'runs'/RUN;out.mkdir();write(out/'launch.json',dict(pid=os.getpid(),CWD=str(Path.cwd()),code_commit=args.commit))
    while True:
        available=recovery_gpu_choice(allowed_gpus=(6,),maximum_processes=5)
        if available:break
        print('WAITING_FOR_APPROVED_GPU6_CAP5',flush=True);time.sleep(30)
    admission=available[0]
    command=[sys.executable,'-u','-m',MODULE,'replay','--root',str(args.root),'--inputs',str(args.inputs),'--cache',str(args.cache),'--commit',args.commit]
    with (out/'replay.log').open('xb') as log:
        child=subprocess.Popen(command,cwd=Path.cwd(),env=dict(os.environ,CUDA_VISIBLE_DEVICES=admission['CUDA_ID']),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        write(out/'submitted.json',dict(pid=child.pid,command=command,admission=admission));rc=child.wait()
    write(out/'terminal.json',dict(pid=child.pid,exit_code=rc))
    if rc:raise RuntimeError('Diagnostic failed; preserve original and diagnostic outputs')
    if read(out/'results.json')['status']!='DIAGNOSTIC_COMPLETE':raise ValueError('Missing diagnostic result')
    write(out/'complete.json',dict(status='ARTIFACTS_COMPLETE',code_commit=args.commit,target_reads=0))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('run','replay'))
    for name in ('root','inputs','cache'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--commit',required=True);args=parser.parse_args()
    if (Path.cwd()/'release_commit.txt').read_text().strip()!=args.commit:raise ValueError('Actual diagnostic code identity required')
    if args.stage=='run':controller(args)
    else:replay(args,args.root/'runs'/RUN)


if __name__=='__main__':main()
