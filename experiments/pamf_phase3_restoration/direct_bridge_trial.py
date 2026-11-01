"""Source-only scratch single30 direct IQ bridge; fixed four-step Main evaluation."""
import argparse
import copy
import json
from pathlib import Path
import time
import numpy as np
import torch
from . import multiview_trial as base
from .power_anchor_trial import configuration as anchor_configuration,power_anchor_loss
from .full_feature_trial import full_feature_guide
from .variance_trial import load_scale,recovery_gpu_choice
from .feature_conditioned import unit_power,latent_feature_loss
from .direct_bridge import scratch_direct_bridge,training_state,training_times
from .cache import tensor,load_F0,check_F0_state
from .source_pair import read,validate_source
from .objective import PairedObjective
from .train import learning_rate
from .cost_probe import write

RUN='20261101-phase1-Restore-DirectBridge-E30-s2026092701-r01'
MODULE='experiments.pamf_phase3_restoration.direct_bridge_trial'
ARM='MSAT_DirectBridge'
TEMPLATE=anchor_configuration('MSAT_PowerAnchor')


def configuration(arm):
    if arm!=ARM:raise ValueError('Only fixed four-step direct IQ bridge allowed')
    c=copy.deepcopy(TEMPLATE)
    c.update(run_id=RUN,arm=ARM,
        architecture='shared-MSAT paired-IQ/time encoder+continuous160+four-feature/time-FiLM+four-direct-iterations',
        training_state='(1-t)*detached sourceclean+t*same-record sourcereceived/no extra noise',
        bridge_times=[1.,.75,.5,.25],bridge_time_probabilities=[.5,1/6,1/6,1/6],bridge_state_rng_seed=2026110103,
        inference_core_calls=4,training_core_calls_per_record=1,
        inference_update='z_next=z+(predicted_clean-z)/remaining;remaining=4,3,2,1;last exactly predictedclean',
        inference_condition='same current receivedIQ in every step/no new observations/no randomness',
        decoder_condition='Predictedcontinuous160 plus current bridge time;no clean/teacherF0features in forward',
        training_stages=1,matched_control='20261101-phase1-Restore-PowerAnchor-E30-s2026092701-r01/MSAT_PowerAnchor',
        scientific_change='Paired intermediate-state training plus observedIQ/time conditioning and four shared direct iterations;system comparison/not single-factor')
    return c


def factory(arm):
    configuration(arm);return scratch_direct_bridge(2026092701)


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
    t=torch.tensor([1.,.5],device=device)
    z=training_state(clean[:2],source2,t);restored,aux=G.estimate(z,source2,t,return_aux=True)
    iq,_=objective(restored,clean[:2]);guide,_=base.feature_guide(F,restored,clean[:2],target)
    latent,_=latent_feature_loss(aux['predicted_features'],target,scale);anchor,_=power_anchor_loss(aux,clean[:2])
    (iq+guide+latent+anchor).mean().backward()
    for name in ['output.weight','stem.weight','feature_regressor.3.weight']:
        p=dict(G.named_parameters())[name]
        if p.grad is None or not torch.isfinite(p.grad).all() or not p.grad.abs().sum()>0:raise ValueError('Real source output/encoder/feature-regressor gradient required')
    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in G.parameters()):raise ValueError('Finite source bridge smoke gradients required')
    G.zero_grad(set_to_none=True);check_F0_state(F,frozen)
    if any(not torch.equal(q,G.state_dict()[k]) for k,q in before.items()):raise ValueError('No-update bridge smoke changed G')
    base.save(out/'smoke_E0.pt',G,args.arm,args.commit,0);probe=base.load(out/'smoke_E0.pt',args.arm,args.commit,device,epoch=0)
    with torch.no_grad():torch.testing.assert_close(probe(source2),deployed,atol=2e-5,rtol=2e-5)
    del probe,restored,aux,iq,guide,latent,anchor,before,source2,deployed,z,t
    write(out/'smoke.json',dict(status='PASS',source_Ls=2,original_F0=True,four_step_projected_initial_output=True,
        output_encoder_feature_regressor_gradients=True,strict_E0_reload=True,clean_features_loss_only=True,
        optimizer_updates=0,TX_labels_read=0,V_reads=0,OE_reads=0,target_inputs=0,inference_core_calls=4,training_core_calls=1))
    print('DIRECT_BRIDGE_REAL_SOURCE_SMOKE_PASS',flush=True)
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
            sums={};histogram=np.zeros(4,dtype=np.int64);time_sums=None;metric_keys=None
            for offset in range(0,63000,256):
                take=torch.tensor(orders[epoch-1,offset:offset+256].tolist(),device=device,dtype=torch.long)
                optimizer.zero_grad(set_to_none=True)
                for group in optimizer.param_groups:group['lr']=learning_rate(update+1)
                t=training_times(len(take),device);state=training_state(clean[take],observed[take],t)
                restored,aux=G.estimate(state,observed[take],t,return_aux=True)
                loss,terms=objective(restored,clean[take]);guide,info=base.feature_guide(F,restored,clean[take],features[take])
                latent,latent_terms=latent_feature_loss(aux['predicted_features'],features[take],scale)
                anchor,anchor_terms=power_anchor_loss(aux,clean[take]);loss=loss+guide+latent+anchor
                if not torch.isfinite(loss).all():raise ValueError('Nonfinite source bridge loss; preserve run')
                loss.mean().backward();torch.nn.utils.clip_grad_norm_(G.parameters(),1.,error_if_nonfinite=True)
                optimizer.step();update+=1
                power_error=(restored.detach().square().sum(1).mean(1)-1).abs()
                metrics=dict(loss=loss,**terms,**latent_terms,feature_error=info['feature_error'],feature_standardized_error=info['feature_standardized_error'],
                    guide_weight=info['weight'],output_power_abs_error=power_error,bridge_time=t,bridge_endpoint=(t==1).float(),**anchor_terms)
                for key,value in metrics.items():sums[key]=sums.get(key,0.)+float(value.detach().sum())
                index=(t*4).long()-1
                histogram+=np.asarray(torch.bincount(index,minlength=4).cpu().tolist(),dtype=np.int64)
                if metric_keys is None:
                    metric_keys=list(metrics);time_sums=torch.zeros(4,len(metric_keys),device=device,dtype=torch.float64)
                masks=(torch.arange(4,device=device)[:,None]==index[None,:]).double()
                time_sums+=masks@torch.stack([metrics[k].detach() for k in metric_keys],1).double()
            check_F0_state(F,frozen);torch.cuda.synchronize();base.save_recovery(out/'recovery',G,optimizer,args.arm,args.commit,epoch,update)
            time_totals=np.asarray(time_sums.cpu().tolist(),dtype=np.float64)
            time_metrics={label:dict(zip(metric_keys,(time_totals[i]/histogram[i]).tolist())) for i,label in enumerate(['0.25','0.50','0.75','1.00'])}
            row=dict(epoch=epoch,draw=draw,updates=update,steps=247,count=63000,L_s=6300,U_s=56700,seconds=time.perf_counter()-begin,
                bridge_time_counts=dict(zip(['0.25','0.50','0.75','1.00'],histogram.tolist())),
                bridge_time_metrics=time_metrics,
                last_learning_rate=learning_rate(update),metrics={k:v/63000 for k,v in sums.items()})
            log.write(json.dumps(row,allow_nan=False)+'\n');print('DIRECT_BRIDGE_EPOCH',args.arm,json.dumps(row),flush=True)
            del observed
    if update!=7410:raise ValueError('Fixed full scratch30 budget required')
    base.save(out/'final.pt',G,args.arm,args.commit,30)
    write(out/'complete.json',dict(status='SOURCE_TRAINED',code_commit=args.commit,arm=args.arm,epoch=30,updates=7410,
        source_Ls_exposures=189000,source_Us_exposures=1701000,online_F0_feature_rows=1890000,F0_clean_target_rows=63000,
        source_F0_feature_cache_seconds=cache_seconds,F0_identity=identity,G_parameters=sum(p.numel() for p in G.parameters()),
        training_seconds=time.perf_counter()-train_start,worker_seconds=time.perf_counter()-started,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        G_initialization_checkpoint_loads=0,TX_labels_read=0,V_reads=0,OE_reads=0,target_training_reads=0,model_state_F0_updates=0,
        checkpoint_selection='final30_only',training_stages=1,latent_teacher_features_are_loss_only=True,output_complex_power=1.,
        inference_core_calls=4,training_core_calls_per_record=1,bridge_state_rng_seed=2026110103))


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
