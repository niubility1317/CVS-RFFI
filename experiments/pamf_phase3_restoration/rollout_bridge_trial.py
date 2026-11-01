"""Source-only actual four-step training; scratch single30 and automatic original Main."""
import argparse
import copy
import json
from pathlib import Path
import time
import numpy as np
import torch
from . import multiview_trial as base
from .state_bridge_trial import configuration as state_configuration
from .state_bridge import scratch_state_bridge
from .power_anchor_trial import power_anchor_loss
from .full_feature_trial import full_feature_guide
from .variance_trial import load_scale,recovery_gpu_choice
from .feature_conditioned import unit_power,latent_feature_loss
from .cache import tensor,load_F0,check_F0_state
from .source_pair import read,validate_source
from .objective import PairedObjective
from .train import learning_rate
from .cost_probe import write

RUN='20261101-phase1-Restore-RolloutBridge-E30-s2026092701-r01'
MODULE='experiments.pamf_phase3_restoration.rollout_bridge_trial'
ARM='MSAT_RolloutBridge'
TEMPLATE=state_configuration('MSAT_StateBridge')


def configuration(arm):
    if arm!=ARM:raise ValueError('Only fixed actual-rollout source training allowed')
    c=copy.deepcopy(TEMPLATE)
    c.update(run_id=RUN,arm=ARM,training_state='Actual differentiable four-step network rollout from receivedIQ;clean only final loss reference',
        bridge_time_probabilities=None,bridge_time_random_draws=0,training_clean_state_inputs=0,
        training_core_calls_per_record=4,training_loss_at='fourth predicted-clean only/all four state transitions differentiable/no detach',
        matched_control='20261101-phase1-Restore-StateBridge-E30-s2026092701-r01/MSAT_StateBridge',
        scientific_change='Replace sampled clean-mixture one-core training with actual four-core differentiable rollout/final loss;same architecture,E0,source order,optimizer,30epochs and deployed function;training compute differs')
    return c


def factory(arm):
    configuration(arm);return scratch_state_bridge(2026092701)


def rollout(model,received,*,trace=False):
    # Exactly the deployed iterate_four arithmetic, with the last loss auxiliaries.
    state=received;path=[]
    for remaining in (4,3,2,1):
        t=torch.full((len(received),),remaining/4,device=received.device,dtype=received.dtype)
        predicted,aux=model.estimate(state,received,t,return_aux=True)
        if trace:path.append(predicted)
        state=predicted if remaining==1 else state+(predicted-state)/remaining
    return (state,aux,path) if trace else (state,aux)


def train(args,out):
    started=time.perf_counter();c=validate_source(read(args.inputs));old,roles=base.source(args);device='cuda:0'
    # Own factory is imported before the original F0 reader's namespace bootstrap.
    G=factory(args.arm).to(device)
    F,identity=load_F0(c,device);base.main_eval.valid_identity(identity)
    frozen={k:v.detach().clone() for k,v in F.state_dict().items()}
    a=np.load(old/'train/clean.npy',mmap_mode='r',allow_pickle=False)
    try:
        if a.shape!=(63000,2,256) or a.dtype!=np.float32:raise ValueError('Original source FP32 clean required')
        clean=tensor(a,device)
    finally:a._mmap.close()
    if not torch.isfinite(clean).all():raise ValueError('Finite original source clean required')
    scale=load_scale(args.root).to(device)
    out.mkdir();write(out/'config.json',dict(configuration(args.arm),code_commit=args.commit,F0_identity=identity))
    objective=PairedObjective().to(device)
    a=np.load(old/'epochs/01/received.npy',mmap_mode='r',allow_pickle=False)
    try:source2=tensor(a[:2],device)
    finally:a._mmap.close()
    before={k:v.detach().clone() for k,v in G.state_dict().items()}
    with torch.no_grad():
        target=F.features(clean[:2]).detach();deployed=G(source2)
        torch.testing.assert_close(deployed,unit_power(source2),atol=1e-6,rtol=2e-6)
    restored,aux,path=rollout(G,source2,trace=True)
    if not torch.equal(restored,deployed):raise ValueError('Training rollout and actual deployed four steps differ')
    for value in path:value.retain_grad()
    iq,_=objective(restored,clean[:2]);guide,_=base.feature_guide(F,restored,clean[:2],target)
    latent,_=latent_feature_loss(aux['predicted_features'],target,scale);anchor,_=power_anchor_loss(aux,clean[:2])
    (iq+guide+latent+anchor).mean().backward()
    if any(value.grad is None or not torch.isfinite(value.grad).all() or not value.grad.abs().sum()>0 for value in path):
        raise ValueError('Final loss must backpropagate through all four actual rollout states')
    for name in ['output.weight','stem.weight','feature_regressor.3.weight']:
        p=dict(G.named_parameters())[name]
        if p.grad is None or not torch.isfinite(p.grad).all() or not p.grad.abs().sum()>0:raise ValueError('Real source output/encoder/feature-regressor gradient required')
    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in G.parameters()):raise ValueError('Finite source bridge smoke gradients required')
    G.zero_grad(set_to_none=True);check_F0_state(F,frozen)
    if any(not torch.equal(q,G.state_dict()[k]) for k,q in before.items()):raise ValueError('No-update bridge smoke changed G')
    base.save(out/'smoke_E0.pt',G,args.arm,args.commit,0);probe=base.load(out/'smoke_E0.pt',args.arm,args.commit,device,epoch=0)
    with torch.no_grad():torch.testing.assert_close(probe(source2),deployed,atol=2e-5,rtol=2e-5)
    del probe,restored,aux,iq,guide,latent,anchor,before,source2,deployed,path,value
    write(out/'smoke.json',dict(status='PASS',source_Ls=2,original_F0=True,four_step_projected_initial_output=True,
        output_encoder_feature_regressor_gradients=True,strict_E0_reload=True,clean_features_loss_only=True,
        optimizer_updates=0,TX_labels_read=0,V_reads=0,OE_reads=0,target_inputs=0,inference_core_calls=4,training_core_calls=4,actual_rollout_equals_deployment=True,all_four_step_gradients=True,clean_only_loss_reference=True))
    print('ROLLOUT_BRIDGE_REAL_SOURCE_SMOKE_PASS',flush=True)
    cache_start=time.perf_counter()
    with torch.no_grad():features=torch.cat([F.features(clean[i:i+256]).detach() for i in range(0,63000,256)])
    torch.cuda.synchronize();cache_seconds=time.perf_counter()-cache_start
    G.train();optimizer=torch.optim.AdamW(G.parameters(),lr=.001,weight_decay=1e-4)
    rng=np.random.default_rng(2026110102);orders=np.stack([rng.permutation(63000) for _ in range(30)]).astype(np.int32)
    with (out/'data_schedule.npz').open('xb') as f:np.savez(f,orders=orders,draws=configuration(args.arm)['source_draws'])
    torch.cuda.manual_seed(2026110103)
    update=0;train_start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    with (out/'training.jsonl').open('x',encoding='utf-8',buffering=1) as log:
        for epoch,draw in enumerate(configuration(args.arm)['source_draws'],1):
            begin=time.perf_counter();folder=old/'epochs'/f'{draw:02d}';done=read(folder/'complete.json')
            if (done['count'],done['TX_labels_read'],done['target_reads'])!=(63000,0,0):raise ValueError('Complete original source training draw required')
            a=np.load(folder/'received.npy',mmap_mode='r',allow_pickle=False)
            try:
                if a.shape!=(63000,2,256) or a.dtype!=np.float32:raise ValueError('Original source received FP32 required')
                observed=tensor(a,device)
            finally:a._mmap.close()
            if not torch.isfinite(observed).all():raise ValueError('Finite source received required')
            sums={}
            for offset in range(0,63000,256):
                take=torch.tensor(orders[epoch-1,offset:offset+256].tolist(),device=device,dtype=torch.long)
                optimizer.zero_grad(set_to_none=True)
                for group in optimizer.param_groups:group['lr']=learning_rate(update+1)
                restored,aux=rollout(G,observed[take])
                loss,terms=objective(restored,clean[take]);guide,info=base.feature_guide(F,restored,clean[take],features[take])
                latent,latent_terms=latent_feature_loss(aux['predicted_features'],features[take],scale)
                anchor,anchor_terms=power_anchor_loss(aux,clean[take]);loss=loss+guide+latent+anchor
                if not torch.isfinite(loss).all():raise ValueError('Nonfinite source bridge loss; preserve run')
                loss.mean().backward();torch.nn.utils.clip_grad_norm_(G.parameters(),1.,error_if_nonfinite=True)
                optimizer.step();update+=1
                power_error=(restored.detach().square().sum(1).mean(1)-1).abs()
                metrics=dict(loss=loss,**terms,**latent_terms,feature_error=info['feature_error'],feature_standardized_error=info['feature_standardized_error'],
                    guide_weight=info['weight'],output_power_abs_error=power_error,**anchor_terms)
                for key,value in metrics.items():sums[key]=sums.get(key,0.)+float(value.detach().sum())
            check_F0_state(F,frozen);torch.cuda.synchronize();base.save_recovery(out/'recovery',G,optimizer,args.arm,args.commit,epoch,update)
            row=dict(epoch=epoch,draw=draw,updates=update,steps=247,count=63000,L_s=6300,U_s=56700,seconds=time.perf_counter()-begin,
                rollout_core_calls_per_record=4,final_loss_calls_per_record=1,all_steps_differentiable=True,
                last_learning_rate=learning_rate(update),metrics={k:v/63000 for k,v in sums.items()})
            log.write(json.dumps(row,allow_nan=False)+'\n');print('ROLLOUT_BRIDGE_EPOCH',args.arm,json.dumps(row),flush=True)
            del observed
    if update!=7410:raise ValueError('Fixed full scratch30 budget required')
    base.save(out/'final.pt',G,args.arm,args.commit,30)
    write(out/'complete.json',dict(status='SOURCE_TRAINED',code_commit=args.commit,arm=args.arm,epoch=30,updates=7410,
        source_Ls_exposures=189000,source_Us_exposures=1701000,online_F0_feature_rows=1890000,F0_clean_target_rows=63000,
        source_F0_feature_cache_seconds=cache_seconds,F0_identity=identity,G_parameters=sum(p.numel() for p in G.parameters()),
        training_seconds=time.perf_counter()-train_start,worker_seconds=time.perf_counter()-started,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        G_initialization_checkpoint_loads=0,TX_labels_read=0,V_reads=0,OE_reads=0,target_training_reads=0,model_state_F0_updates=0,
        checkpoint_selection='final30_only',training_stages=1,latent_teacher_features_are_loss_only=True,output_complex_power=1.,
        inference_core_calls=4,training_core_calls_per_record=4,bridge_state_rng_seed=2026110103,
        training_G_core_rows=7560000,training_clean_state_inputs=0,bridge_time_random_draws=0,all_steps_differentiable=True))


def install(scale):
    by_device={}
    def guide(reader,restored,clean,target):
        if restored.device not in by_device:by_device[restored.device]=scale.to(restored.device)
        return full_feature_guide(reader,restored,clean,target,by_device[restored.device])
    base.RUN=RUN;base.MODULE=MODULE;base.ARMS=(ARM,);base.configuration=configuration
    base.factory=factory;base.train=train;base.feature_guide=guide;base.gpu_choice=recovery_gpu_choice


if __name__=='__main__':
    p=argparse.ArgumentParser(add_help=False);p.add_argument('--root',type=Path,required=True);a,_=p.parse_known_args()
    install(load_scale(a.root));base.main()
