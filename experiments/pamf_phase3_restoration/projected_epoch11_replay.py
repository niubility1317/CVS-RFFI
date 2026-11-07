"""Bounded source-only replay to localize a projected-prior training spike."""
import argparse
import importlib
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

RUN='20261108-diagnostic-Restore-ProjectedE11Replay-sourceLU-s2026092701-r01'
MODULE='experiments.pamf_phase3_restoration.projected_epoch11_replay'
PARENT='20261107-phase1-Restore-ProjectedPriorDetail-E30-s2026092701-r01'
ARM='TF_ProjectedPriorDetail'
PARENT_COMMIT='8cf3669c4a5d98f6c4930117952203afd56f2f6e'
THRESHOLD=100.


def spike_indices(values):
    if values.ndim!=1 or not torch.isfinite(values).all() or (values<0).any():
        raise ValueError('Finite nonnegative projected source error required')
    return torch.nonzero(values>THRESHOLD,as_tuple=False).flatten()


def original_modules(root):
    package=importlib.import_module('experiments.pamf_phase3_restoration')
    release=root/'releases'/(PARENT+'-release')/'experiments/pamf_phase3_restoration'
    if not (release/'projected_prior_detail.py').is_file():raise ValueError('Original source release required')
    package.__path__.insert(0,str(release))
    model=importlib.import_module('.projected_prior_detail',package.__name__)
    if Path(model.__file__).resolve()!=release.joinpath('projected_prior_detail.py').resolve():
        raise ValueError('Exact original source model implementation required')
    names=['projected_prior_detail_trial','multiview_trial','cache','source_pair','objective','train','covariance_detail','covariance_model','variance_trial','cost_probe']
    return model,{name:importlib.import_module('.'+name,package.__name__) for name in names}


def replay(args,out):
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model_api,m=original_modules(args.root);science=m['projected_prior_detail_trial'];trial=m['multiview_trial'];cache=m['cache'];io=m['source_pair'];write=m['cost_probe'].write
    source=io.validate_source(io.read(args.inputs));old,_=trial.source(args)
    training=args.root/'runs'/PARENT/'rows'/ARM/'training';smoke=io.read(training/'smoke.json')
    if smoke['status']!='PASS' or any(smoke[k] for k in ('TX_labels_read','V_reads','OE_reads','target_inputs')):
        raise ValueError('Original source-only parent provenance required')
    checkpoint=torch.load(training/'recovery/E10.pt',map_location='cpu',weights_only=False)
    if (checkpoint['code_commit'],checkpoint['epoch'],checkpoint['updates'],checkpoint['config'],checkpoint['selection_use'])!=(
            PARENT_COMMIT,10,2470,science.configuration(ARM),'technical_recovery_only/not_source_or_target_model_selection'):
        raise ValueError('Exact original source E10 technical checkpoint required')
    with np.load(training/'data_schedule.npz',allow_pickle=False) as schedule:
        order=schedule['orders'][10].copy();draw=int(schedule['draws'][10])
    if order.shape!=(63000,) or draw!=11:raise ValueError('Original epoch11 source schedule required')
    model=science.factory(ARM);model.load_state_dict(checkpoint['state'],strict=True);model=model.to('cuda:0').train()
    reader,identity=cache.load_F0(source,'cuda:0');trial.main_eval.valid_identity(identity)
    frozen={k:v.detach().clone() for k,v in reader.state_dict().items()}
    scale=m['variance_trial'].load_scale(args.root).to('cuda:0');objective=m['objective'].PairedObjective().to('cuda:0')
    def load(path):
        a=np.load(path,mmap_mode='r',allow_pickle=False)
        try:return cache.tensor(a,'cuda:0')
        finally:a._mmap.close()
    clean=load(old/'train/clean.npy');folder=old/'epochs/11';done=io.read(folder/'complete.json')
    if (done['count'],done['TX_labels_read'],done['target_reads'],clean.shape)!=(63000,0,0,(63000,2,256)):
        raise ValueError('Original source draw11 and clean shape required')
    values={key:load(folder/(key+'.npy')) for key in ('received','kernel','coefficients','variance')}
    with np.load(folder/'groups.npz',allow_pickle=False) as groups:scenes=groups['scene'].copy()
    with torch.no_grad():targets=torch.cat([reader.features(clean[i:i+256]).detach() for i in range(0,63000,256)])
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=1e-4);optimizer.load_state_dict(checkpoint['optimizer'])
    if {float(x['step']) for x in optimizer.state.values() if 'step' in x}!={2470.}:
        raise ValueError('Exact original optimizer boundary required')
    def readout(indices):
        return model_api.shrinkage_training_readout(model,reader,values['received'][indices],clean[indices],targets[indices],scale,objective)
    first=torch.tensor(order[:2].tolist(),device='cuda:0',dtype=torch.long)
    with torch.no_grad():
        restored,aux,detail,terms=readout(first)
        physical,_=m['covariance_model'].source_action_loss(aux,values['kernel'][first],values['coefficients'][first],values['variance'][first])
        if not torch.isfinite((physical+detail).mean()) or aux['relative_solve_residual'].max()>1e-8:
            raise ValueError('Actual original E10/F0 source smoke failed')
    cache.check_F0_state(reader,frozen)
    write(out/'smoke.json',dict(status='PASS',actual_original_E10=True,actual_original_F0=True,source_rows=2,
        optimizer_updates=0,TX_labels_read=0,V_reads=0,OE_reads=0,target_inputs=0))
    print('PROJECTED_E10_REAL_SOURCE_SMOKE_PASS',flush=True)
    torch.set_rng_state(checkpoint['torch_rng_cpu']);torch.cuda.set_rng_state(checkpoint['torch_rng_cuda'])
    captured=[];hook=model.detail.register_forward_hook(lambda module,inputs,value:captured.append(value.detach()))
    update=2470;anomaly=None;started=time.perf_counter();maximum=0.
    try:
        for offset in range(0,63000,256):
            take=order[offset:offset+256];optimizer.zero_grad(set_to_none=True)
            for group in optimizer.param_groups:group['lr']=m['train'].learning_rate(update+1)
            for start in range(0,len(take),32):
                indices=torch.tensor(take[start:start+32].tolist(),device='cuda:0',dtype=torch.long);captured.clear()
                restored,aux,detail,terms=readout(indices)
                physical,_=m['covariance_model'].source_action_loss(aux,values['kernel'][indices],values['coefficients'][indices],values['variance'][indices])
                scalar=(physical+detail).sum()/len(take)
                if not torch.isfinite(scalar) or aux['relative_solve_residual'].max()>1e-8:
                    anomaly=dict(kind='ORIGINAL_GUARD',logical_offset=offset,micro_offset=start,completed_updates=update);break
                correction=terms['projected_prior_update_nmse'];flagged=spike_indices(correction);maximum=max(maximum,float(correction.detach().max()))
                if len(flagged):
                    if len(captured)!=1:raise ValueError('One original detail forward required')
                    i=int(flagged[correction[flagged].argmax()]);proposal=captured[0][i]-aux['base_iq'][i];projected=aux['projected_prior_update'][i]
                    scalar.backward()
                    gradients=[p.grad.detach().double().norm() for p in model.detail.parameters() if p.grad is not None]
                    grad=float(torch.stack(gradients).norm()) if gradients else 0.
                    with torch.no_grad():
                        action=aux['action_matrix'][i:i+1].detach().double();noise=aux['variance'][i:i+1].detach().double()
                        mean=model.physical.prior_mean.detach().double();covariance=model.physical.prior_covariance.detach().double()
                        _,ac,factor,residual=model_api.fixed_posterior_frame(action,values['received'][indices[i:i+1]].flatten(1).double(),mean,covariance,noise)
                        projection=torch.eye(512,device='cuda:0',dtype=torch.float64)[None]-ac.transpose(-1,-2)@torch.cholesky_solve(action,factor)
                        reconstructed=(projection@proposal.double().flatten()[None,:,None]).squeeze()
                        torch.testing.assert_close(reconstructed,projected.double().flatten(),rtol=2e-5,atol=2e-6)
                        spectral=float(torch.linalg.svdvals(projection)[0,0])
                    anomaly=dict(kind='PROJECTED_UPDATE_SPIKE',epoch=11,logical_offset=offset,micro_offset=start,completed_updates=update,
                        flagged_count=len(flagged),source_scene=str(scenes[int(indices[i])]),projected_nmse=float(correction[i]),
                        noiseless_iq_nmse=float(terms['noiseless_iq_nmse'][i]),deployed_iq_nmse=float(terms['deployed_iq_nmse'][i]),
                        proposal_norm=float(proposal.double().norm()),projected_norm=float(projected.double().norm()),
                        detail_gradient_norm=grad,predicted_noise=float(noise[0]),original_residual=float(residual[0]),
                        euclidean_projection_operator_norm=spectral)
                    break
                scalar.backward()
            if anomaly is not None:break
            m['covariance_detail'].clip_branch_gradients(model);optimizer.step();update+=1
            if (update-2470)%10==0:print('PROJECTED_E11_REPLAY_UPDATES',update-2470,flush=True)
    finally:hook.remove()
    torch.cuda.synchronize();cache.check_F0_state(reader,frozen)
    write(out/'results.json',dict(status='SOURCE_PROJECTED_REPLAY_COMPLETE',run_id=RUN,code_commit=args.commit,
        parent_run=PARENT,parent_commit=PARENT_COMMIT,source_epoch=11,source_draw=11,technical_checkpoint_epoch=10,
        replay_optimizer_steps=update-2470,trigger_threshold_projected_nmse=THRESHOLD,maximum_observed_projected_nmse=maximum,anomaly=anomaly,
        outcome='SPIKE_OR_GUARD_OBSERVED' if anomaly else 'NO_SPIKE_IN_BOUNDED_REPLAY',elapsed_seconds=time.perf_counter()-started,
        original_run_modified=False,diagnostic_checkpoint_written=False,TX_labels_read=0,V_reads=0,OE_reads=0,target_inputs=0,
        limitations='Shared-GPU numerical trajectories may diverge from original; this source replay never supplies a deployment model or a target accuracy result.'))
    print('PROJECTED_E11_SOURCE_REPLAY_COMPLETE',flush=True)


def controller(args):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CUDA-empty controller required')
    from .cost_probe import write
    from .variance_trial import recovery_gpu_choice
    out=args.root/'runs'/RUN;out.mkdir();write(out/'launch.json',dict(pid=os.getpid(),CWD=str(Path.cwd()),code_commit=args.commit))
    while True:
        choices=recovery_gpu_choice(allowed_gpus=(2,3,4),maximum_processes=3)
        if choices:break
        print('WAITING_FOR_ADMITTED_GPU_CAP3',flush=True);time.sleep(30)
    admission=choices[0];command=[sys.executable,'-u','-m',MODULE,'replay','--root',str(args.root),'--inputs',str(args.inputs),'--cache',str(args.cache),'--commit',args.commit]
    with (out/'replay.log').open('xb') as log:
        child=subprocess.Popen(command,cwd=Path.cwd(),env=dict(os.environ,CUDA_VISIBLE_DEVICES=admission['CUDA_ID']),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        write(out/'submitted.json',dict(pid=child.pid,command=command,admission=admission));rc=child.wait()
    write(out/'terminal.json',dict(pid=child.pid,exit_code=rc))
    if rc:raise RuntimeError('Preserved source replay failure; no automatic restart')
    write(out/'complete.json',dict(status='ARTIFACTS_COMPLETE',code_commit=args.commit,target_reads=0))


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('run','replay'))
    for name in ('root','inputs','cache'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--commit',required=True);args=p.parse_args()
    if (Path.cwd()/'release_commit.txt').read_text().strip()!=args.commit:raise ValueError('Actual diagnostic code identity required')
    if args.stage=='run':controller(args)
    else:replay(args,args.root/'runs'/RUN)


if __name__=='__main__':main()
