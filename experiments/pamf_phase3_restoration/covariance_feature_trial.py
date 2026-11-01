"""Scratch30 physical operator plus sampled source F0 guidance through actual IQ."""
import copy
import json
from pathlib import Path
import time
import numpy as np
import torch
from . import multiview_trial as trial
from . import covariance_trial as control
from .covariance_model import scratch_covariance,source_action_loss
from .covariance_inverse import gaussian_posterior
from .variance_trial import load_scale,recovery_gpu_choice,GEOMETRY_RUN,GEOMETRY_COMMIT
from .cache import tensor,load_F0,check_F0_state
from .source_pair import read,validate_source
from .train import learning_rate
from .cost_probe import write

RUN='20261102-phase1-Restore-CovarianceFeature-E30-s2026092701-r01'
MODULE='experiments.pamf_phase3_restoration.covariance_feature_trial'
ARM='TF_CovarianceFeature'
FEATURE_WEIGHT=.01
FEATURE_ROWS_PER_EPOCH=7896
TEMPLATE=control.configuration(control.ARM)


def configuration(arm):
    if arm!=ARM:raise ValueError('Only fixed posterior-feature-guided covariance candidate allowed')
    c=copy.deepcopy(TEMPLATE)
    c.update(run_id=RUN,arm=arm,
        objective='.1source-Action-matrix-relativeMSE+.001noise-logMSE+.01actual-posterior-F0(cos+sourcevarMSE)',
        training_path='Allrecords G.estimate/physical loss;firstmicro ofeachlogicalbatch also actualFP64posterior andfrozenF0features;onejointoptimizer/no stages',
        source_F0_knowledge='Original3ff8F0/fused160/clean63000cache+sampledactualIQfeaturegradients;defaultheadsmoke-evaluationonly',
        F0_guidance_for_G_updates=True,feature_weight=FEATURE_WEIGHT,feature_loss_input='actualcurrentreceived posterior IQ,notnoisedclean orlatentregressor',
        feature_sample_rule='Firstmin(32,n)records ofeachuniformlyshuffledlogicalbatch;lastpartial24all;noTX-ID-qualityselection',
        feature_rows_per_epoch=FEATURE_ROWS_PER_EPOCH,training_F0_forward_rows=FEATURE_ROWS_PER_EPOCH*30,
        training_posterior_calls=7410,training_posterior_rows=FEATURE_ROWS_PER_EPOCH*30,
        feature_scale_source=GEOMETRY_RUN,feature_scale_code=GEOMETRY_COMMIT,
        feature_scale='max(sourceF0variance,.01meanvariance,1e-10)/sameoriginal63000/160dims',
        matched_control=control.RUN+'/'+control.ARM,
        scientific_change='AddsourcecontinuousF0guidance onactualphysicalposterior/MCfirstmicro;samearchitecture/actualE0/prior/sourceorders-draws/LR/30/fullMain. No pretrainedG initialization.')
    return c


def factory(arm):
    configuration(arm);return scratch_covariance(2026092701)


def feature_window(logical_count):
    if not 1<=logical_count<=256:raise ValueError('Actual logical source batch count in1..256 required')
    return slice(0,min(32,logical_count))


def micro_objective(physical,guide,logical_count):
    value=physical.sum()/logical_count
    return value if guide is None else value+guide.mean()


def posterior_feature_guide(reader,aux,received,clean,target_features,scale,model):
    if target_features.shape!=(len(received),160) or scale.shape!=(160,):
        raise ValueError('Matched source fused160 target and source variance required')
    restored,residual=gaussian_posterior(aux['action_matrix'],received.flatten(1).double(),model.prior_mean.double(),model.prior_covariance.double(),aux['variance'].double())
    restored=restored.float().reshape(-1,2,256)
    features=reader.features(restored);target=target_features.detach()
    cosine=1-torch.nn.functional.cosine_similarity(features,target,dim=1,eps=1e-8)
    standardized=((features-target).square()/scale).mean(1)
    energy=clean.detach().square().mean((1,2)).clamp_min(1e-4)
    return FEATURE_WEIGHT*(cosine+standardized),dict(feature_error=cosine,feature_standardized_error=standardized,
        iq_nmse=(restored-clean.detach()).square().mean((1,2))/energy,
        output_complex_power=restored.square().sum(1).mean(1),relative_solve_residual=residual)


def train(args,out):
    start=time.perf_counter();c=validate_source(read(args.inputs));old,roles=trial.source(args)
    G=factory(args.arm);prior_identity=control.bind_prior(args.root,G);G=G.to('cuda:0')
    scale=load_scale(args.root).to('cuda:0');F,identity=load_F0(c,'cuda:0');trial.main_eval.valid_identity(identity)
    if any(p.requires_grad for p in F.parameters()):raise ValueError('Original F0 must remain frozen')
    frozen={k:v.detach().clone() for k,v in F.state_dict().items()}
    a=np.load(old/'train/clean.npy',mmap_mode='r',allow_pickle=False)
    try:clean=tensor(a,'cuda:0')
    finally:a._mmap.close()
    if clean.shape!=(63000,2,256) or not torch.isfinite(clean).all():raise ValueError('Original full source clean required')
    out.mkdir();write(out/'config.json',dict(configuration(args.arm),code_commit=args.commit,F0_identity=identity,source_prior_identity=prior_identity))
    def source_draw(draw):
        folder=old/'epochs'/f'{draw:02d}';done=read(folder/'complete.json')
        if (done.get('count'),done.get('TX_labels_read'),done.get('target_reads'))!=(63000,0,0):raise ValueError('Original source simulator traces required')
        shapes={'received':(63000,2,256),'kernel':(63000,2,64),'coefficients':(63000,4,256),'variance':(63000,)};values={}
        for key,shape in shapes.items():
            a=np.load(folder/(key+'.npy'),mmap_mode='r',allow_pickle=False)
            try:
                if a.shape!=shape or a.dtype!=np.float32:raise ValueError('Original source action/received shape differs')
                values[key]=tensor(a,'cuda:0')
            finally:a._mmap.close()
        if any(not torch.isfinite(x).all() for x in values.values()) or (values['variance']<=0).any():raise ValueError('Finite positive-noise source trace required')
        return values
    first=source_draw(1);before={k:v.detach().clone() for k,v in G.state_dict().items()}
    with np.load(old/'epochs/01/groups.npz',allow_pickle=False) as groups:smoke_indices=np.flatnonzero(groups['scene']!='clean')[:2]
    if len(smoke_indices)!=2:raise ValueError('Two fixed non-clean source cases required')
    take=torch.tensor(smoke_indices.tolist(),device='cuda:0',dtype=torch.long)
    with torch.no_grad():
        target2=F.features(clean[take]);restored,aux=G(first['received'][take],return_aux=True)
        actual=F.classify_features(F.features(restored))
        if actual.shape!=(2,6) or not torch.isfinite(actual).all() or aux['relative_solve_residual'].max()>1e-8:raise ValueError('Actual full G/F0 source smoke failed')
    aux=G.estimate(first['received'][take]);physical,_=source_action_loss(aux,first['kernel'][take],first['coefficients'][take],first['variance'][take])
    guide,info=posterior_feature_guide(F,aux,first['received'][take],clean[take],target2,scale,G)
    if info['relative_solve_residual'].max()>1e-8:raise ValueError('Source feature posterior solve failed')
    micro_objective(physical,guide,2).backward()
    for head in (G.kernel_head,G.coefficient_head,G.noise_head):
        if head.weight.grad is None or not torch.isfinite(head.weight.grad).all() or not head.weight.grad.abs().sum()>0:raise ValueError('All actual action/noise heads require gradients')
    if any(p.grad is not None for p in F.parameters()):raise ValueError('Frozen F0 received gradients')
    G.zero_grad(set_to_none=True)
    if any(not torch.equal(v,G.state_dict()[k]) for k,v in before.items()):raise ValueError('Source smoke changed G state')
    check_F0_state(F,frozen);trial.save(out/'smoke_E0.pt',G,args.arm,args.commit,0)
    probe=trial.load(out/'smoke_E0.pt',args.arm,args.commit,'cuda:0',epoch=0)
    with torch.no_grad():torch.testing.assert_close(probe(first['received'][take]),restored,rtol=0,atol=0)
    del probe,restored,aux,physical,guide,info,before,first,target2
    write(out/'smoke.json',dict(status='PASS',source_rows=2,actual_original_F0=True,actual_full_posterior=True,FP64_Cholesky=True,
        source_operator_and_feature_gradients=True,all_action_heads_gradients=True,strict_E0_reload=True,source_prior_bound=True,
        F0_parameters_frozen=True,F0_state_unchanged=True,source_indices=smoke_indices.tolist(),optimizer_updates=0,
        TX_labels_read=0,V_reads=0,OE_reads=0,target_inputs=0,feature_weight=FEATURE_WEIGHT))
    print('COVARIANCE_FEATURE_REAL_SOURCE_SMOKE_PASS',flush=True)
    cache_start=time.perf_counter()
    with torch.no_grad():features=torch.cat([F.features(clean[i:i+256]).detach() for i in range(0,63000,256)])
    torch.cuda.synchronize();cache_seconds=time.perf_counter()-cache_start
    G.train();optimizer=torch.optim.AdamW(G.parameters(),lr=.001,weight_decay=1e-4)
    rng=np.random.default_rng(2026110102);orders=np.stack([rng.permutation(63000) for _ in range(30)]).astype(np.int32)
    with (out/'data_schedule.npz').open('xb') as f:np.savez(f,orders=orders,draws=configuration(args.arm)['source_draws'])
    update=0;feature_total=0;feature_L_total=0;feature_U_total=0;train_start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    fixed_prior={k:G.state_dict()[k].detach().clone() for k in ['prior_mean','prior_covariance']}
    with (out/'training.jsonl').open('x',encoding='utf-8',buffering=1) as log:
        for epoch,draw in enumerate(configuration(args.arm)['source_draws'],1):
            begin=time.perf_counter();v=source_draw(draw);sums={};feature_sums={};step_sums={};fcount=0;fl=0;fu=0
            for offset in range(0,63000,256):
                take=orders[epoch-1,offset:offset+256];optimizer.zero_grad(set_to_none=True)
                for group in optimizer.param_groups:group['lr']=learning_rate(update+1)
                step=dict(loss=0.,step_action_matrix_relative_mse=0.,step_noise_log_mse=0.)
                selected=feature_window(len(take))
                for begin_micro in range(0,len(take),32):
                    part=take[begin_micro:begin_micro+32];indices=torch.tensor(part.tolist(),device='cuda:0',dtype=torch.long)
                    aux=G.estimate(v['received'][indices]);physical,terms=source_action_loss(aux,v['kernel'][indices],v['coefficients'][indices],v['variance'][indices]);guide=None
                    if begin_micro==selected.start:
                        guide,info=posterior_feature_guide(F,aux,v['received'][indices],clean[indices],features[indices],scale,G)
                        if len(indices)!=selected.stop:raise ValueError('Fixed first-micro sample changed')
                        fcount+=len(indices);fl+=int((roles[part]=='L_s').sum());fu+=int((roles[part]=='U_s').sum())
                        for key,value in info.items():feature_sums[key]=feature_sums.get(key,0.)+float(value.detach().sum())
                        step['step_feature_error']=float(info['feature_error'].detach().mean());step['step_feature_standardized_error']=float(info['feature_standardized_error'].detach().mean())
                    scalar=micro_objective(physical,guide,len(take))
                    if not torch.isfinite(scalar):raise ValueError('Nonfinite joint source loss;preserve run')
                    scalar.backward();step['loss']+=float(scalar.detach())
                    step['step_action_matrix_relative_mse']+=float(terms['action_matrix_relative_mse'].detach().sum())/len(take)
                    step['step_noise_log_mse']+=float(terms['noise_log_mse'].detach().sum())/len(take)
                    for key,value in dict(physical_loss=physical,**terms,predicted_noise=aux['variance']).items():sums[key]=sums.get(key,0.)+float(value.detach().sum())
                torch.nn.utils.clip_grad_norm_(G.parameters(),1.,error_if_nonfinite=True);optimizer.step();update+=1
                for key,value in step.items():step_sums[key]=step_sums.get(key,0.)+value
            if fcount!=FEATURE_ROWS_PER_EPOCH or fl+fu!=fcount:raise ValueError('Actual sampled feature exposure differs')
            feature_total+=fcount;feature_L_total+=fl;feature_U_total+=fu
            if any(not torch.equal(value,G.state_dict()[key]) for key,value in fixed_prior.items()):raise ValueError('Fixed source prior changed')
            check_F0_state(F,frozen);torch.cuda.synchronize();trial.save_recovery(out/'recovery',G,optimizer,args.arm,args.commit,epoch,update)
            metrics={key:value/63000 for key,value in sums.items()};metrics.update({key:value/247 for key,value in step_sums.items()})
            row=dict(epoch=epoch,draw=draw,updates=update,steps=247,count=63000,L_s=6300,U_s=56700,seconds=time.perf_counter()-begin,
                last_learning_rate=learning_rate(update),microbatch=32,training_posterior_calls=247,training_posterior_rows=fcount,
                F0_training_forward_rows=fcount,feature_L_s=fl,feature_U_s=fu,feature_metrics={key:value/fcount for key,value in feature_sums.items()},metrics=metrics)
            log.write(json.dumps(row,allow_nan=False)+'\n');print('COVARIANCE_FEATURE_EPOCH',args.arm,json.dumps(row),flush=True);del v
    if update!=7410 or feature_total!=FEATURE_ROWS_PER_EPOCH*30:raise ValueError('Full single30/update/feature exposure budget required')
    trial.save(out/'final.pt',G,args.arm,args.commit,30)
    write(out/'complete.json',dict(status='SOURCE_TRAINED',code_commit=args.commit,arm=args.arm,epoch=30,updates=7410,
        source_Ls_exposures=189000,source_Us_exposures=1701000,online_F0_feature_rows=feature_total,F0_clean_target_rows=63000,
        F0_feature_L_s_exposures=feature_L_total,F0_feature_U_s_exposures=feature_U_total,source_F0_feature_cache_seconds=cache_seconds,
        F0_identity=identity,source_prior_identity=prior_identity,G_parameters=sum(p.numel() for p in G.parameters()),
        training_seconds=time.perf_counter()-train_start,worker_seconds=time.perf_counter()-start,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        G_initialization_checkpoint_loads=0,TX_labels_read=0,V_reads=0,OE_reads=0,target_training_reads=0,
        checkpoint_selection='final30_only',model_state_F0_updates=0,training_stages=1,training_posterior_calls=7410,training_posterior_rows=feature_total,
        source_operator_full_matrix_supervision=True,source_prior_buffers_unchanged=True,source_actual_IQ_F0_guidance=True,feature_weight=FEATURE_WEIGHT))


def install():
    trial.RUN=RUN;trial.MODULE=MODULE;trial.ARMS=(ARM,)
    trial.configuration=configuration;trial.factory=factory;trial.train=train;trial.gpu_choice=recovery_gpu_choice


if __name__=='__main__':
    install();trial.main()
