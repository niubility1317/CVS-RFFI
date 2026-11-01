"""One source-only feature-conditioned scratch30 candidate and full Main evaluation."""
import argparse
import copy
import json
from pathlib import Path
import time
import numpy as np
import torch
from . import multiview_trial as trial
from .full_feature_trial import configuration as full_configuration,full_feature_guide
from .variance_trial import load_scale,recovery_gpu_choice
from .feature_conditioned import scratch_feature_conditioned,unit_power,latent_feature_loss
from .cache import tensor,load_F0,check_F0_state
from .source_pair import read,validate_source
from .objective import PairedObjective
from .train import learning_rate
from .cost_probe import write

RUN='20261101-phase1-Restore-FeatureCond-E30-s2026092701-r01'
MODULE='experiments.pamf_phase3_restoration.feature_conditioned_trial'
ARM='MSAT_FeatureCond'
TEMPLATE=full_configuration('MSAT_FullFGrad')


def configuration(arm):
    if arm!=ARM:raise ValueError('Only fixed scratch feature-conditioned candidate allowed')
    c=copy.deepcopy(TEMPLATE)
    c.update(run_id=RUN,arm=ARM,architecture='shared-MSAT-encoder+continuous-fused160-regression+four-FiLM-stages+unit-power-residual-IQ',
        objective='IQNMSE+.05STFT+.1(F0outputcos+sourcevarMSE)+.1(predicted-continuous-feature-cos+sourcevarMSE)',
        continuous_feature_dimension=160,condition_widths=[256,192,128,64],output_complex_power=1.,
        latent_teacher='Same frozen originalF0 sourceclean fused160/detached/loss-only',
        decoder_condition='PredictedfromcurrentIQ only;no clean/teacherF0features in forward',
        deployment_inputs='current_received_IQ_only/no_F0_inside_G',training_stages=1,
        matched_control='20261101-phase1-Restore-MSAT-FullFGrad-E30-s2026092701-r01/MSAT_FullFGrad',
        scientific_change='Continuous feature condition+auxiliary feature regression+unitpower constraint;system comparison/not single-factor')
    return c


def factory(arm):
    configuration(arm);return scratch_feature_conditioned(2026092701)


def train(args,out):
    started=time.perf_counter();c=validate_source(read(args.inputs));old,roles=trial.source(args);device='cuda:0'
    F,identity=load_F0(c,device);trial.main_eval.valid_identity(identity)
    frozen={k:v.detach().clone() for k,v in F.state_dict().items()}
    a=np.load(old/'train/clean.npy',mmap_mode='r',allow_pickle=False)
    try:
        if a.shape!=(63000,2,256) or a.dtype!=np.float32:raise ValueError('Original source FP32 clean required')
        clean=tensor(a,device)
    finally:a._mmap.close()
    if not torch.isfinite(clean).all():raise ValueError('Finite original source clean required')
    scale=load_scale(args.root).to(device)
    out.mkdir();write(out/'config.json',dict(configuration(args.arm),code_commit=args.commit,F0_identity=identity))
    G=factory(args.arm).to(device);objective=PairedObjective().to(device)
    a=np.load(old/'epochs/01/received.npy',mmap_mode='r',allow_pickle=False)
    try:source2=tensor(a[:2],device)
    finally:a._mmap.close()
    before={k:v.detach().clone() for k,v in G.state_dict().items()}
    with torch.no_grad():target=F.features(clean[:2]).detach()
    restored,aux=G(source2,return_aux=True)
    torch.testing.assert_close(restored,unit_power(source2),atol=0,rtol=0)
    iq,_=objective(restored,clean[:2]);guide,_=trial.feature_guide(F,restored,clean[:2],target)
    latent,_=latent_feature_loss(aux['predicted_features'],target,scale)
    (iq+guide+latent).mean().backward()
    for name in ['output.weight','stem.weight','feature_regressor.3.weight']:
        p=dict(G.named_parameters())[name]
        if p.grad is None or not torch.isfinite(p.grad).all() or not p.grad.abs().sum()>0:raise ValueError('Real source output/encoder/feature-regressor gradient required')
    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in G.parameters()):raise ValueError('Finite source smoke gradients required')
    G.zero_grad(set_to_none=True);check_F0_state(F,frozen)
    if any(not torch.equal(q,G.state_dict()[k]) for k,q in before.items()):raise ValueError('No-update smoke changed G')
    trial.save(out/'smoke_E0.pt',G,args.arm,args.commit,0);probe=trial.load(out/'smoke_E0.pt',args.arm,args.commit,device,epoch=0)
    with torch.no_grad():torch.testing.assert_close(probe(source2),restored,atol=2e-5,rtol=2e-5)
    del probe,restored,aux,iq,guide,latent,before,source2
    write(out/'smoke.json',dict(status='PASS',source_Ls=2,original_F0=True,unit_power_initial_output=True,
        output_encoder_feature_regressor_gradients=True,strict_E0_reload=True,clean_features_loss_only=True,
        optimizer_updates=0,TX_labels_read=0,V_reads=0,OE_reads=0,target_inputs=0))
    print('FEATURECOND_REAL_SOURCE_SMOKE_PASS',flush=True)
    cache_start=time.perf_counter()
    with torch.no_grad():features=torch.cat([F.features(clean[i:i+256]).detach() for i in range(0,63000,256)])
    torch.cuda.synchronize();cache_seconds=time.perf_counter()-cache_start
    G.train();optimizer=torch.optim.AdamW(G.parameters(),lr=.001,weight_decay=1e-4)
    rng=np.random.default_rng(2026110102);orders=np.stack([rng.permutation(63000) for _ in range(30)]).astype(np.int32)
    with (out/'data_schedule.npz').open('xb') as f:np.savez(f,orders=orders,draws=configuration(args.arm)['source_draws'])
    update=0;train_start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    with (out/'training.jsonl').open('x',encoding='utf-8',buffering=1) as log:
        for epoch,draw in enumerate(configuration(args.arm)['source_draws'],1):
            begin=time.perf_counter();folder=old/'epochs'/f'{draw:02d}';done=read(folder/'complete.json')
            if (done['count'],done['TX_labels_read'],done['target_reads'])!=(63000,0,0):raise ValueError('Complete original source training draw required')
            a=np.load(folder/'received.npy',mmap_mode='r',allow_pickle=False)
            try:
                if a.shape!=(63000,2,256) or a.dtype!=np.float32:raise ValueError('Original source received required')
                observed=tensor(a,device)
            finally:a._mmap.close()
            if not torch.isfinite(observed).all():raise ValueError('Finite source received required')
            sums={}
            for offset in range(0,63000,256):
                take=torch.tensor(orders[epoch-1,offset:offset+256].tolist(),device=device,dtype=torch.long)
                optimizer.zero_grad(set_to_none=True)
                for group in optimizer.param_groups:group['lr']=learning_rate(update+1)
                restored,aux=G(observed[take],return_aux=True)
                loss,terms=objective(restored,clean[take]);guide,info=trial.feature_guide(F,restored,clean[take],features[take])
                latent,latent_terms=latent_feature_loss(aux['predicted_features'],features[take],scale);loss=loss+guide+latent
                if not torch.isfinite(loss).all():raise ValueError('Nonfinite source loss; preserve run')
                loss.mean().backward();torch.nn.utils.clip_grad_norm_(G.parameters(),1.,error_if_nonfinite=True)
                optimizer.step();update+=1
                power_error=(restored.detach().square().sum(1).mean(1)-1).abs()
                metrics=dict(loss=loss,**terms,**latent_terms,feature_error=info['feature_error'],
                    feature_standardized_error=info['feature_standardized_error'],guide_weight=info['weight'],output_power_abs_error=power_error)
                for key,value in metrics.items():sums[key]=sums.get(key,0.)+float(value.detach().sum())
            check_F0_state(F,frozen);torch.cuda.synchronize();trial.save_recovery(out/'recovery',G,optimizer,args.arm,args.commit,epoch,update)
            row=dict(epoch=epoch,draw=draw,updates=update,steps=247,count=63000,L_s=6300,U_s=56700,seconds=time.perf_counter()-begin,
                last_learning_rate=learning_rate(update),metrics={k:v/63000 for k,v in sums.items()})
            log.write(json.dumps(row,allow_nan=False)+'\n');print('FEATURECOND_EPOCH',args.arm,json.dumps(row),flush=True)
            del observed
    if update!=7410:raise ValueError('Fixed full scratch30 budget required')
    trial.save(out/'final.pt',G,args.arm,args.commit,30)
    write(out/'complete.json',dict(status='SOURCE_TRAINED',code_commit=args.commit,arm=args.arm,epoch=30,updates=7410,
        source_Ls_exposures=189000,source_Us_exposures=1701000,online_F0_feature_rows=1890000,F0_clean_target_rows=63000,
        source_F0_feature_cache_seconds=cache_seconds,F0_identity=identity,G_parameters=sum(p.numel() for p in G.parameters()),
        training_seconds=time.perf_counter()-train_start,worker_seconds=time.perf_counter()-started,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        G_initialization_checkpoint_loads=0,TX_labels_read=0,V_reads=0,OE_reads=0,target_training_reads=0,model_state_F0_updates=0,
        checkpoint_selection='final30_only',training_stages=1,latent_teacher_features_are_loss_only=True,output_complex_power=1.))


def install(scale):
    by_device={}
    def guide(reader,restored,clean,target):
        if restored.device not in by_device:by_device[restored.device]=scale.to(restored.device)
        return full_feature_guide(reader,restored,clean,target,by_device[restored.device])
    trial.RUN=RUN;trial.MODULE=MODULE;trial.ARMS=(ARM,);trial.configuration=configuration
    trial.factory=factory;trial.train=train;trial.feature_guide=guide;trial.gpu_choice=recovery_gpu_choice


if __name__=='__main__':
    p=argparse.ArgumentParser(add_help=False);p.add_argument('--root',type=Path,required=True);a,_=p.parse_known_args()
    install(load_scale(a.root));trial.main()
