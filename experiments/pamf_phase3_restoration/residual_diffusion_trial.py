"""Scratch single-stage conditional residual diffusion; fixed final30 then original Main7."""
import argparse
import copy
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from . import multiview_trial as trial
from .full_feature_trial import configuration as full_configuration,full_feature_guide
from .variance_trial import load_scale,recovery_gpu_choice
from .residual_diffusion import scratch_residual_diffusion,diffusion_loss
from .cache import tensor,load_F0,check_F0_state
from .source_pair import read,validate_source
from .objective import PairedObjective
from .train import learning_rate
from .cost_probe import write

RUN='20261102-phase1-Restore-TFResidualDiffusion-E30-s2026092701-r01'
MODULE='experiments.pamf_phase3_restoration.residual_diffusion_trial'
ARM='TF_ResidualDiffusion'
TEMPLATE=full_configuration('MSAT_FullFGrad')
WAIT_PARENT='20261101-phase1-Restore-SoftPower-E30-s2026092701-r01'


def configuration(arm):
    if arm!=ARM:raise ValueError('Only fixed source conditional residual diffusion allowed')
    c=copy.deepcopy(TEMPLATE)
    c.update(run_id=RUN,arm=ARM,architecture='joint-time-frequency-patch4-128tokens-width128-6Transformer-4heads-MLP512',
        objective='EDMweighted-residual-MSE+.05STFT+.1(F0cos+source-variance-featureMSE)',
        diffusion_sigma_data=.5,diffusion_log_sigma_mean=-1.2,diffusion_log_sigma_std=1.2,
        diffusion_noise_seed=2026110203,sampling_fixed_latent_seed=2026110204,
        sampling_steps=8,sampling_denoiser_calls=15,sampling_solver='Heun/Eulerlast/FP32/no-churn',
        sampling_sigma_min=.002,sampling_sigma_max=80.,sampling_rho=7.,
        source_loss_denoiser_calls=1,output_complex_power=None,unit_power_projection=False,
        latent_feature_head=False,raw_power_anchor_weight=0.,
        matched_control='Frozen DURA and FullFGrad original Main2pop7;system comparison,not single-factor',
        scientific_change='Replace MSAT/direct-clean regression with dual-domain Transformer conditional residual diffusion;one-stage scratch30,one denoiser per training pair,fixed15-call sampler. No pretrained G/diffusion or IQ teacher.')
    return c


def factory(arm):
    configuration(arm);return scratch_residual_diffusion(2026092701)


def wait_for_parent(root):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CUDA-empty queue controller required')
    parent=root/'runs'/WAIT_PARENT
    print('WAITING_FOR_FIXED_SOFTPOWER_PARENT_MAIN_TECHNICAL_COMPLETION',flush=True)
    while True:
        if any(read(p)['exit_code'] for p in parent.glob('terminal-*.json')):
            raise RuntimeError('Parent technical failure;preserve queued run without automatic retry')
        marker=parent/'complete.json'
        if marker.is_file() and read(marker)['status']=='ARTIFACTS_COMPLETE':return
        time.sleep(5)


def train(args,out):
    started=time.perf_counter();c=validate_source(read(args.inputs));old,roles=trial.source(args);device='cuda:0'
    G=factory(args.arm).to(device)
    F,identity=load_F0(c,device);trial.main_eval.valid_identity(identity)
    frozen={k:v.detach().clone() for k,v in F.state_dict().items()}
    a=np.load(old/'train/clean.npy',mmap_mode='r',allow_pickle=False)
    try:
        if a.shape!=(63000,2,256) or a.dtype!=np.float32:raise ValueError('Original source FP32 clean required')
        clean=tensor(a,device)
    finally:a._mmap.close()
    if not torch.isfinite(clean).all():raise ValueError('Finite original source clean required')
    out.mkdir();write(out/'config.json',dict(configuration(args.arm),code_commit=args.commit,F0_identity=identity))
    objective=PairedObjective().to(device)
    a=np.load(old/'epochs/01/received.npy',mmap_mode='r',allow_pickle=False)
    try:source2=tensor(a[:2],device)
    finally:a._mmap.close()
    before={k:v.detach().clone() for k,v in G.state_dict().items()}
    with torch.no_grad():target=F.features(clean[:2]).detach();sampled=G(source2)
    if sampled.shape!=source2.shape or not torch.isfinite(sampled).all():raise ValueError('Finite actual source2 full sampler required')
    sigma=torch.full((2,),.5,device=device);noise=G.fixed_latent.expand_as(source2)
    restored,terms=diffusion_loss(G,source2,clean[:2],sigma,noise)
    _,physical=objective(restored,clean[:2]);guide,_=trial.feature_guide(F,restored,clean[:2],target)
    (terms['diffusion_mse']+.05*physical['spectral_error']+guide).mean().backward()
    for name in ['core.time_out.weight','core.freq_out.weight']:
        gradient=dict(G.named_parameters())[name].grad
        if gradient is None or not torch.isfinite(gradient).all() or not gradient.abs().sum()>0:
            raise ValueError('Real paired source objective output head gradient required')
    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in G.parameters()):raise ValueError('Finite source smoke gradients required')
    G.zero_grad(set_to_none=True);check_F0_state(F,frozen)
    if any(not torch.equal(v,G.state_dict()[k]) for k,v in before.items()):raise ValueError('No-update smoke changed G')
    trial.save(out/'smoke_E0.pt',G,args.arm,args.commit,0);probe=trial.load(out/'smoke_E0.pt',args.arm,args.commit,device,epoch=0)
    with torch.no_grad():torch.testing.assert_close(probe(source2),sampled,atol=2e-5,rtol=2e-5)
    del probe,restored,terms,physical,guide,before,source2,sampled,target,noise,sigma
    write(out/'smoke.json',dict(status='PASS',source_Ls=2,original_F0=True,full_sampler_calls=15,
        finite_E0_Gaussian_residual_prior=True,identity_initial_output=False,unit_power_projection=False,
        output_head_gradients=True,zero_output_heads_block_body_gradient_at_E0_expected=True,
        strict_E0_reload=True,clean_features_loss_only=True,optimizer_updates=0,TX_labels_read=0,V_reads=0,OE_reads=0,target_inputs=0))
    print('RESIDUAL_DIFFUSION_REAL_SOURCE_SMOKE_PASS',flush=True)
    cache_start=time.perf_counter()
    with torch.no_grad():features=torch.cat([F.features(clean[i:i+256]).detach() for i in range(0,63000,256)])
    torch.cuda.synchronize();cache_seconds=time.perf_counter()-cache_start
    G.train();optimizer=torch.optim.AdamW(G.parameters(),lr=.001,weight_decay=1e-4)
    rng=np.random.default_rng(2026110102);orders=np.stack([rng.permutation(63000) for _ in range(30)]).astype(np.int32)
    with (out/'data_schedule.npz').open('xb') as f:np.savez(f,orders=orders,draws=configuration(args.arm)['source_draws'])
    torch.manual_seed(2026110203);torch.cuda.manual_seed_all(2026110203)
    update=0;train_start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    with (out/'training.jsonl').open('x',encoding='utf-8',buffering=1) as log:
        for epoch,draw in enumerate(configuration(args.arm)['source_draws'],1):
            begin=time.perf_counter();folder=old/'epochs'/f'{draw:02d}';done=read(folder/'complete.json')
            if (done['count'],done['TX_labels_read'],done['target_reads'])!=(63000,0,0):raise ValueError('Complete original source draw required')
            a=np.load(folder/'received.npy',mmap_mode='r',allow_pickle=False)
            try:
                if a.shape!=(63000,2,256) or a.dtype!=np.float32:raise ValueError('Original source received FP32 required')
                observed=tensor(a,device)
            finally:a._mmap.close()
            if not torch.isfinite(observed).all():raise ValueError('Finite source received required')
            sums={};hist=torch.zeros(5,device=device,dtype=torch.long);sigma_min=float('inf');sigma_max=0.
            for offset in range(0,63000,256):
                take=torch.tensor(orders[epoch-1,offset:offset+256].tolist(),device=device,dtype=torch.long)
                optimizer.zero_grad(set_to_none=True)
                for group in optimizer.param_groups:group['lr']=learning_rate(update+1)
                sigma=(torch.randn(len(take),device=device)*1.2-1.2).exp();noise=torch.randn_like(observed[take])
                restored,terms=diffusion_loss(G,observed[take],clean[take],sigma,noise)
                _,physical=objective(restored,clean[take]);guide,info=trial.feature_guide(F,restored,clean[take],features[take])
                loss=terms['diffusion_mse']+.05*physical['spectral_error']+guide
                if not torch.isfinite(loss).all():raise ValueError('Nonfinite source loss;preserve run')
                loss.mean().backward();torch.nn.utils.clip_grad_norm_(G.parameters(),1.,error_if_nonfinite=True)
                optimizer.step();update+=1
                metrics=dict(loss=loss,**terms,**physical,feature_error=info['feature_error'],
                    feature_standardized_error=info['feature_standardized_error'],guide_weight=info['weight'],
                    output_complex_power=restored.detach().square().sum(1).mean(1))
                for key,value in metrics.items():sums[key]=sums.get(key,0.)+float(value.detach().sum())
                hist+=torch.bincount(torch.bucketize(sigma,torch.tensor([.01,.1,1.,10.],device=device)),minlength=5)
                sigma_min=min(sigma_min,float(sigma.min()));sigma_max=max(sigma_max,float(sigma.max()))
            check_F0_state(F,frozen);torch.cuda.synchronize();trial.save_recovery(out/'recovery',G,optimizer,args.arm,args.commit,epoch,update)
            row=dict(epoch=epoch,draw=draw,updates=update,steps=247,count=63000,L_s=6300,U_s=56700,seconds=time.perf_counter()-begin,
                last_learning_rate=learning_rate(update),sigma_histogram=hist.tolist(),sigma_min=sigma_min,sigma_max=sigma_max,
                train_denoiser_calls_per_record=1,sampler_denoiser_calls_per_record=15,metrics={k:v/63000 for k,v in sums.items()})
            log.write(json.dumps(row,allow_nan=False)+'\n');print('RESIDUAL_DIFFUSION_EPOCH',args.arm,json.dumps(row),flush=True)
            del observed
    if update!=7410:raise ValueError('Fixed full scratch30 budget required')
    trial.save(out/'final.pt',G,args.arm,args.commit,30)
    write(out/'complete.json',dict(status='SOURCE_TRAINED',code_commit=args.commit,arm=args.arm,epoch=30,updates=7410,
        source_Ls_exposures=189000,source_Us_exposures=1701000,online_F0_feature_rows=1890000,F0_clean_target_rows=63000,
        source_F0_feature_cache_seconds=cache_seconds,F0_identity=identity,G_parameters=sum(p.numel() for p in G.parameters()),
        training_seconds=time.perf_counter()-train_start,worker_seconds=time.perf_counter()-started,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        G_initialization_checkpoint_loads=0,TX_labels_read=0,V_reads=0,OE_reads=0,target_training_reads=0,model_state_F0_updates=0,
        checkpoint_selection='final30_only',training_stages=1,source_training_denoiser_calls_per_record=1,
        sampling_denoiser_calls_per_record=15,output_complex_power=None,unit_power_projection=False))


def install(scale):
    by_device={}
    def guide(reader,restored,clean,target):
        if restored.device not in by_device:by_device[restored.device]=scale.to(restored.device)
        return full_feature_guide(reader,restored,clean,target,by_device[restored.device])
    trial.RUN=RUN;trial.MODULE=MODULE;trial.ARMS=(ARM,);trial.configuration=configuration
    trial.factory=factory;trial.train=train;trial.feature_guide=guide;trial.gpu_choice=recovery_gpu_choice


if __name__=='__main__':
    p=argparse.ArgumentParser(add_help=False);p.add_argument('--root',type=Path,required=True);p.add_argument('--commit',required=True);a,_=p.parse_known_args()
    if (Path.cwd()/'release_commit.txt').read_text().strip()!=a.commit:raise ValueError('Actual Git code/CWD required')
    if len(os.sys.argv)>1 and os.sys.argv[1]=='run':
        try:wait_for_parent(a.root)
        except Exception as error:
            write(a.root/'logs'/RUN/'queue_terminal.json',dict(status='QUEUE_FAILED',exception=type(error).__name__,reason=str(error),model_loads=0,GPU_workers=0))
            raise
    install(load_scale(a.root));trial.main()
