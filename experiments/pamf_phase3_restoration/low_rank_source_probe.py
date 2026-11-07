"""Source-only numerical and shared-GPU timing probe; no optimizer or target use."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

from . import coupled_covariance_trial as reference
from .cache import tensor,load_F0,check_F0_state
from .coupled_covariance import coupled_covariance,mode_factor,normalize_modes,scratch_coupled_covariance
from .covariance_detail import detail_objective
from .covariance_model import source_action_loss
from .cost_probe import write
from .low_rank_posterior import conditional_prior_posterior
from .mismatched_prior_training import load_training_priors
from .objective import PairedObjective
from .projected_prior_detail import fixed_posterior_frame
from .signed_covariance import source_modes
from .source_inverse import R07
from .source_pair import read,validate_source
from .variance_trial import load_scale,recovery_gpu_choice

RUN='20261107-diagnostic-Restore-CoupledLowRank-sourceLU-s2026092701-r02'
MODULE='experiments.pamf_phase3_restoration.low_rank_source_probe'
REFERENCE_COMMIT='6d3dda3c88765c38e1813ecb733948234bb63190'
SCENES=('clean','practical_high','practical_mid','practical_low_suburban','practical_high_urban','practical_mid_urban','practical_low_urban')
PER_SCENE=32
RECEIVERS=(1,3,4,6,8)


def select_rows(scene,roles,receivers):
    selected=[]
    for name in SCENES:
        cell=[]
        for rx in RECEIVERS:
            for role,count in (('L_s',1),('U_s',5)):
                ids=np.flatnonzero((scene==name)&(receivers==rx)&(roles==role))[:count]
                if len(ids)!=count:raise ValueError('Every source scene/RX/role cell needs its fixed quota')
                cell.extend(ids.tolist())
        extra=[int(i) for i in np.flatnonzero((scene==name)&(roles=='U_s')) if int(i) not in cell][:2]
        if len(extra)!=2:raise ValueError('Every preregistered source scene needs32 distinct rows')
        selected.extend(cell+extra)
    return np.asarray(selected,dtype=np.int64)


def validate_reference(checkpoint,smoke):
    if (checkpoint['code_commit'],checkpoint['epoch'],checkpoint['updates'])!=(REFERENCE_COMMIT,20,4940):
        raise ValueError('Only the fixed original source Coupled E20 diagnostic checkpoint is allowed')
    if checkpoint['config']!=reference.configuration(reference.ARM):raise ValueError('Original source checkpoint configuration mismatch')
    if smoke['status']!='PASS' or any(smoke[k] for k in ('TX_labels_read','V_reads','OE_reads','target_inputs')):
        raise ValueError('Source checkpoint provenance/smoke mismatch')


def posterior_forward(model,received,prior,solver,mean_shift):
    aux=model.physical.estimate(received)
    action,variance=aux['action_matrix'].detach(),aux['variance'].detach()
    mean,covariance,modes=((model.physical.prior_mean,model.physical.prior_covariance,model.physical.prior_modes)
                          if prior is None else prior)
    batch_mean=mean.expand(len(received),-1)
    diagonal=covariance.diagonal(dim1=-2,dim2=-1).expand(len(received),-1)
    innovation=received.flatten(1).double()-(action@batch_mean.double()[:,:,None]).squeeze(-1)
    raw=model.detail(received,batch_mean.float().reshape_as(received),diagonal.float().reshape_as(received),
                     innovation.float().reshape_as(received),variance)
    factor=mode_factor(raw)
    used_mean=batch_mean.double()+mean_shift.double()
    if solver=='dense':
        changed,_=coupled_covariance(covariance.double(),modes.double(),factor)
        restored,_,_,residual=fixed_posterior_frame(action,received.flatten(1).double(),used_mean,changed,variance.double())
    elif solver=='low_rank':
        restored,residual=conditional_prior_posterior(action,received.flatten(1).double(),used_mean,covariance,modes,factor,variance)
    else:raise ValueError('Only the two fixed numerical implementations are allowed')
    return restored.float().reshape_as(received),dict(aux,relative_solve_residual=residual)


def relative_error(actual,reference_value):
    difference=(actual-reference_value).double()
    return dict(max_abs=float(difference.abs().max()),relative_l2=float(difference.norm()/reference_value.double().norm().clamp_min(1e-12)))


def save_cell(out,records):
    # Each scientific cell is immutable; the shared writer intentionally uses x.
    write(out/f'cell-{len(records):02d}.json',records[-1])


def probe(args,out):
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    contract=validate_source(read(args.inputs))
    trained=args.root/'runs'/reference.RUN/'rows'/reference.ARM/'training'
    checkpoint=torch.load(trained/'recovery/E20.pt',map_location='cpu',weights_only=False)
    smoke=read(trained/'smoke.json')
    validate_reference(checkpoint,smoke)
    # Construct G before the original F0 namespace bootstrap.
    model=scratch_coupled_covariance(2026092701)
    model.load_state_dict(checkpoint['state'],strict=True);model=model.to('cuda:0')
    original_state={k:v.detach().clone() for k,v in model.state_dict().items()}
    reader,identity=load_F0(contract,'cuda:0')
    reference.trial.main_eval.valid_identity(identity)
    reader_state={k:v.detach().clone() for k,v in reader.state_dict().items()}
    scale=load_scale(args.root).to('cuda:0');objective=PairedObjective().to('cuda:0')
    cache=args.root/'runs'/R07/'cache'
    cache_meta=read(cache/'complete.json')
    if (cache_meta['status'],cache_meta['train_count'],cache_meta['epochs'],cache_meta['TX_labels_read'],cache_meta['target_reads'])!=('ACTION_SOURCE_CACHE_COMPLETE',63000,20,0,0):
        raise ValueError('Original source-only R07 cache required')
    draw=cache/'epochs/01'
    roles=np.load(cache/'train/roles.npy',allow_pickle=False)
    receivers=np.load(cache/'train/rx.npy',allow_pickle=False)
    with np.load(draw/'groups.npz',allow_pickle=False) as groups:indices=select_rows(groups['scene'],roles,receivers)
    if len(set(indices.tolist()))!=224:raise ValueError('Distinct224 source rows required')
    def subset(path):
        source=np.load(path,mmap_mode='r',allow_pickle=False)
        try:value=source[indices].copy()
        finally:source._mmap.close()
        return tensor(value,'cuda:0')
    data={name:subset(draw/(name+'.npy')) for name in ('received','kernel','coefficients','variance')}
    clean=subset(cache/'train/clean.npy')
    if clean.shape!=(224,2,256) or data['received'].shape!=clean.shape:raise ValueError('Fixed224 source IQ rows required')
    pair,assignment=load_training_priors(args.root,'cpu')
    modes=torch.stack([normalize_modes(covariance,source_modes(covariance)) for covariance in pair[1]])
    bank=tuple(value.to('cuda:0') for value in (*pair,modes))
    selected_assignment=assignment[torch.tensor(indices.tolist(),dtype=torch.long)].to('cuda:0')
    with torch.no_grad():targets=reader.features(clean).detach()
    model.eval()
    with torch.no_grad():
        observed=data['received'][32:34]
        actual=model(observed)
        duplicate,_=posterior_forward(model,observed,None,'dense',torch.zeros(2,512,device='cuda:0'))
        torch.testing.assert_close(actual,duplicate,rtol=0,atol=0)
        logits=reader.classify_features(reader.features(actual))
        if logits.shape!=(2,6) or not torch.isfinite(logits).all():raise ValueError('Actual original source G/F0 smoke failed')
    write(out/'smoke.json',dict(status='PASS',actual_original_F0=True,actual_original_source_E20_G=True,
        original_dense_source_forward_matches=True,source_rows=2,TX_labels_read=0,V_reads=0,OE_reads=0,target_inputs=0,optimizer_updates=0))
    print('LOW_RANK_REAL_SOURCE_CHECKPOINT_SMOKE_PASS',flush=True)
    model.train()
    parameters=list(model.parameters())
    source_forward_rows=0
    def step(batch,prior,solver,seed,capture=False):
        nonlocal source_forward_rows
        model.zero_grad(set_to_none=True)
        torch.manual_seed(seed)
        mean_shift=torch.zeros(len(batch),512,device='cuda:0',requires_grad=True)
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        start_memory=torch.cuda.memory_allocated();started=time.perf_counter()
        restored,aux=posterior_forward(model,data['received'][batch],prior,solver,mean_shift)
        paired,_=detail_objective(reader,restored,clean[batch],targets[batch],scale,objective)
        physical,_=source_action_loss(aux,data['kernel'][batch],data['coefficients'][batch],data['variance'][batch])
        loss=(paired+physical).mean()
        torch.cuda.synchronize();forward_end=time.perf_counter()
        loss.backward();torch.cuda.synchronize();ended=time.perf_counter()
        source_forward_rows+=len(batch)
        error=float(aux['relative_solve_residual'].max())
        if not torch.isfinite(loss) or error>1e-8:raise ValueError('Nonfinite objective or posterior residual above the original1e-8 limit')
        result=dict(loss=float(loss.detach()),solve_residual_max=error,forward_seconds=forward_end-started,
            backward_seconds=ended-forward_end,total_seconds=ended-started,start_allocated_bytes=start_memory,
            peak_allocated_bytes=torch.cuda.max_memory_allocated())
        if capture:
            gradients=[]
            for parameter in parameters:
                if parameter.grad is None or not torch.isfinite(parameter.grad).all():raise ValueError('All original trainable G parameters require finite gradients')
                gradients.append(parameter.grad.detach().flatten().cpu())
            result.update(IQ=restored.detach().cpu(),gradients=torch.cat(gradients),mean_gradient=mean_shift.grad.detach().cpu())
        return result
    records=[]
    for prior_name in ('global','other_RX'):
        for scene_index,scene in enumerate(SCENES):
            batch=torch.arange(scene_index*32,(scene_index+1)*32,device='cuda:0')
            prior=None if prior_name=='global' else tuple(value[selected_assignment[batch]] for value in bank)
            cell=len(records);seed=2026110707+cell
            dense=step(batch,prior,'dense',seed,True);low=step(batch,prior,'low_rank',seed,True)
            iq=relative_error(low.pop('IQ'),dense.pop('IQ'))
            gradients=relative_error(low.pop('gradients'),dense.pop('gradients'))
            mean_gradient=relative_error(low.pop('mean_gradient'),dense.pop('mean_gradient'))
            loss_relative=abs(low['loss']-dense['loss'])/max(abs(dense['loss']),1e-12)
            if iq['max_abs']>1e-5 or gradients['relative_l2']>1e-4 or mean_gradient['relative_l2']>1e-4 or loss_relative>1e-5:
                write(out/'failed_numeric_cell.json',dict(prior=prior_name,scene=scene,IQ=iq,gradients=gradients,mean_gradient=mean_gradient,loss_relative=loss_relative))
                raise ValueError('Preregistered full-model source numerical/gradient tolerance failed')
            # First pair above is also the warmup. Timing orders alternate.
            timing={'dense':[],'low_rank':[]}
            for repeat in range(3):
                order=('dense','low_rank') if repeat%2==0 else ('low_rank','dense')
                for solver in order:timing[solver].append(step(batch,prior,solver,seed+repeat+1))
            record=dict(prior=prior_name,scene=scene,count=32,IQ_error=iq,G_gradient_error=gradients,
                mean_gradient_error=mean_gradient,relative_loss_error=loss_relative,dense_numerics=dense,low_rank_numerics=low,timing=timing)
            records.append(record);save_cell(out,records)
            print('LOW_RANK_SOURCE_CELL_COMPLETE',prior_name,scene,len(records),flush=True)
    check_F0_state(reader,reader_state)
    if any(not torch.equal(value,model.state_dict()[key]) for key,value in original_state.items()):raise ValueError('Read-only source probe changed G state')
    write(out/'results.json',dict(status='SOURCE_LOW_RANK_PROBE_COMPLETE',run_id=RUN,code_commit=args.commit,
        source_G_run=reference.RUN,source_G_commit=REFERENCE_COMMIT,source_G_epoch=20,source_G_updates=4940,F0_identity=identity,
        source_draw=1,source_unique_rows=224,source_scenes=list(SCENES),rows_per_cell=32,cells=14,
        source_L_s_rows=int((roles[indices]=='L_s').sum()),source_U_s_rows=int((roles[indices]=='U_s').sum()),source_receivers=list(RECEIVERS),
        original_dense_smoke_rows=2,F0_clean_target_rows=224,source_gradient_forward_rows=source_forward_rows,
        optimizer_updates=0,G_state_updates=0,F0_state_updates=0,TX_labels_read=0,V_reads=0,OE_reads=0,target_inputs=0,
        precision='Physical/F0 FP32 with TF32off; all posterior arithmetic FP64',
        timing_scope='SharedGPU3 source microbatch physical+prior+F0 objective forward/backward. Excludes optimizer step, data loading and epoch bookkeeping; no full-training or B1-latency claim.',
        thresholds=dict(IQ_max_abs=1e-5,G_gradient_relative_l2=1e-4,mean_gradient_relative_l2=1e-4,loss_relative=1e-5,solve_residual_max=1e-8),
        records=records,actual_GPU=torch.cuda.get_device_name(0),Main_performance_claim=False,active_training_changed=False))


def controller(args):
    out=args.root/'runs'/RUN;out.mkdir()
    write(out/'launch.json',dict(pid=os.getpid(),CWD=str(Path.cwd()),code_commit=args.commit,CUDA=os.environ.get('CUDA_VISIBLE_DEVICES')))
    while True:
        available=recovery_gpu_choice(allowed_gpus=(3,),maximum_processes=3)
        if available:break
        print('WAITING_FOR_NORMAL_GPU3_CAP3',flush=True);time.sleep(30)
    admission=available[0]
    command=[sys.executable,'-u','-m',MODULE,'probe','--root',str(args.root),'--inputs',str(args.inputs),'--commit',args.commit]
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=admission['CUDA_ID'])
    with (out/'probe.log').open('xb') as stream:
        child=subprocess.Popen(command,cwd=Path.cwd(),env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
        write(out/'submitted.json',dict(pid=child.pid,command=command,admission=admission,code_commit=args.commit))
        rc=child.wait()
    write(out/'terminal.json',dict(pid=child.pid,exit_code=rc))
    if rc:raise SystemExit(rc)
    if read(out/'results.json')['status']!='SOURCE_LOW_RANK_PROBE_COMPLETE':raise ValueError('Missing complete numerical probe')
    write(out/'complete.json',dict(status='ARTIFACTS_COMPLETE',code_commit=args.commit,source_cells=14,target_reads=0))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('run','probe'))
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--inputs',type=Path,required=True);parser.add_argument('--commit',required=True)
    args=parser.parse_args()
    if args.stage=='run':controller(args)
    else:probe(args,args.root/'runs'/RUN)


if __name__=='__main__':main()
