"""Scratch single30 with independent physical and full-exposure detail gradients."""
import copy
import json
import time
import numpy as np
import torch
from . import multiview_trial as trial
from . import covariance_trial as control
from .covariance_detail import scratch_covariance_detail,detail_objective,clip_branch_gradients
from .covariance_model import source_action_loss
from .objective import PairedObjective
from .variance_trial import load_scale,recovery_gpu_choice
from .source_pair import read,validate_source
from .cache import tensor,load_F0,check_F0_state
from .train import learning_rate
from .cost_probe import write

RUN='20261102-phase1-Restore-CovarianceDetail-E30-s2026092701-r01'
MODULE='experiments.pamf_phase3_restoration.covariance_detail_trial'
ARM='TF_CovarianceDetail'


def configuration(arm):
    if arm!=ARM:raise ValueError('Only fixed split-gradient posterior/detail candidate allowed')
    c=copy.deepcopy(control.configuration(control.ARM))
    c.update(run_id=RUN,arm=arm,architecture='CovarianceMAP TF128x6 physical estimator/full512realGaussianposterior + independent MSAT nonlinear IQdetail/context receivedIQ+lognoise',
        objective='physical:.1fullActionRelativeMSE+.001noiseLogMSE;detail:IQnmse+.05multiSTFT+.01F0(cos+sourcevarMSE)',
        training_path='Everyrecord physicalestimate+detachedFP64posterior+nonlinearIQdetail+frozenF0continuousfeatures;two independent gradient branches,onejointoptimizer/all30',
        source_F0_knowledge='Original3ff8F0 continuous fused160/clean63000cache+allrestoredsourceIQ;defaultheadsmoke-evaluationonly',
        F0_guidance_for_G_updates=True,feature_weight=.01,training_F0_forward_rows=1890000,F0_clean_target_rows=63000,
        training_posterior_calls=59070,training_posterior_rows=1890000,training_stages=1,
        gradient_routing='Physical onlysimulatorAction/noise;posterior andlognoisecontext detached;IQ+F0feature gradients detailonly',
        gradient_accumulation='Eachmicro objective sum/actuallogicalN;independentnorm1 clipping perbranch thenoneAdamWstep',
        learned_IQ_decoder=True,learned_detail='MSAT64-128-192-256/3down-up/4globalattentionblocks/context3/zeroinitoutput/additiveposteriorIQ',
        detail_model_seed=2026092702,detail_full_source_exposure=True,source_prior_buffers='physical.prior_mean/physical.prior_covariance fixed',
        deployment_inputs='currentreceivedIQonly,sourcepriorbuffers,learnedphysical+detail;0F0-clean-trueAction-noise',
        scientific_change='Compositearchitectureandfullsource-detailfeatureexposure;notisolatedablationofrouting. AllGscratchsingle30;no oldGweights.',
        matched_control=control.RUN+'/'+control.ARM,unit_power_projection=False)
    return c


def factory(arm):configuration(arm);return scratch_covariance_detail(2026092701)


def train(args,out,*,build=factory,config_for=configuration,bind_prior=control.bind_prior,smoke_check=None,
          training_readout=None):
    started=time.perf_counter();c=validate_source(read(args.inputs));old,roles=trial.source(args)
    cfg=config_for(args.arm);G=build(args.arm);prior_identity=bind_prior(args.root,G.physical);G=G.to('cuda:0')
    F,identity=load_F0(c,'cuda:0');trial.main_eval.valid_identity(identity);scale=load_scale(args.root).to('cuda:0')
    frozen={k:v.detach().clone() for k,v in F.state_dict().items()};objective=PairedObjective().to('cuda:0')
    def detail_pass(received,clean,target):
        if training_readout is not None:
            return training_readout(G,F,received,clean,target,scale,objective)
        restored,aux=G(received,return_aux=True)
        loss,terms=detail_objective(F,restored,clean,target,scale,objective)
        return restored,aux,loss,terms
    a=np.load(old/'train/clean.npy',mmap_mode='r',allow_pickle=False)
    try:clean=tensor(a,'cuda:0')
    finally:a._mmap.close()
    if clean.shape!=(63000,2,256) or not torch.isfinite(clean).all():raise ValueError('Original finite63000clean required')
    out.mkdir();write(out/'config.json',dict(cfg,code_commit=args.commit,F0_identity=identity,source_prior_identity=prior_identity))
    def draw(number):
        folder=old/'epochs'/f'{number:02d}';done=read(folder/'complete.json')
        if (done['count'],done['TX_labels_read'],done['target_reads'])!=(63000,0,0):raise ValueError('Original completed R07 source draw required')
        shapes={'received':(63000,2,256),'kernel':(63000,2,64),'coefficients':(63000,4,256),'variance':(63000,)};values={}
        for key,shape in shapes.items():
            a=np.load(folder/(key+'.npy'),mmap_mode='r',allow_pickle=False)
            try:
                if a.shape!=shape or a.dtype!=np.float32:raise ValueError('Original source array shape/dtype required')
                values[key]=tensor(a,'cuda:0')
            finally:a._mmap.close()
        if any(not torch.isfinite(v).all() for v in values.values()) or (values['variance']<=0).any():raise ValueError('Finite positive-noise trace required')
        return values
    first=draw(1)
    with np.load(old/'epochs/01/groups.npz',allow_pickle=False) as f:smoke_indices=np.flatnonzero(f['scene']!='clean')[:2]
    if len(smoke_indices)!=2:raise ValueError('Two nonclean source smoke packets required')
    take=torch.tensor(smoke_indices.tolist(),device='cuda:0',dtype=torch.long);before={k:v.detach().clone() for k,v in G.state_dict().items()}
    with torch.no_grad():
        restored,aux=G(first['received'][take],return_aux=True);logits=F.classify_features(F.features(restored));target2=F.features(clean[take])
        if logits.shape!=(2,6) or not torch.isfinite(logits).all() or not torch.isfinite(restored).all() or aux['relative_solve_residual'].max()>1e-8:raise ValueError('Actual full scratchG/sourceF0 smoke failed')
        torch.testing.assert_close(restored,aux['base_iq'],rtol=0,atol=0)
        extra_smoke=smoke_check(G,first['received'][take],restored) if smoke_check else {}
    restored,aux,detail,unused=detail_pass(first['received'][take],clean[take],target2);detail.mean().backward()
    if any(p.grad is not None for p in G.physical.parameters()) or not G.detail.output.weight.grad.abs().sum()>0:raise ValueError('Detail loss must update detail only')
    G.zero_grad(set_to_none=True)
    physical,unused=source_action_loss(aux,first['kernel'][take],first['coefficients'][take],first['variance'][take]);physical.mean().backward()
    if any(p.grad is not None for p in G.detail.parameters()):raise ValueError('Physical loss reached detail branch')
    for head in [G.physical.kernel_head,G.physical.coefficient_head,G.physical.noise_head]:
        if head.weight.grad is None or not torch.isfinite(head.weight.grad).all() or not head.weight.grad.abs().sum()>0:raise ValueError('Actual physical heads require gradients')
    G.zero_grad(set_to_none=True);check_F0_state(F,frozen)
    if any(not torch.equal(v,G.state_dict()[k]) for k,v in before.items()):raise ValueError('No-update source smoke changedG')
    trial.save(out/'smoke_E0.pt',G,args.arm,args.commit,0);probe=trial.load(out/'smoke_E0.pt',args.arm,args.commit,'cuda:0',epoch=0)
    with torch.no_grad():torch.testing.assert_close(probe(first['received'][take]),restored,rtol=0,atol=0)
    del probe,first,before,restored,aux,detail,physical,target2
    write(out/'smoke.json',dict(status='PASS',source_rows=2,actual_original_F0=True,actual_full_posterior=True,detail_initial_zero=True,
        detail_and_physical_gradient_isolation=True,strict_E0_reload=True,F0_frozen=True,optimizer_updates=0,TX_labels_read=0,V_reads=0,OE_reads=0,target_inputs=0,**extra_smoke))
    print('COVARIANCE_DETAIL_REAL_SOURCE_SMOKE_PASS',flush=True)
    tick=time.perf_counter()
    with torch.no_grad():targets=torch.cat([F.features(clean[i:i+256]).detach() for i in range(0,63000,256)])
    torch.cuda.synchronize();cache_seconds=time.perf_counter()-tick
    optimizer=torch.optim.AdamW(G.parameters(),lr=.001,weight_decay=1e-4);G.train()
    rng=np.random.default_rng(2026110102);orders=np.stack([rng.permutation(63000) for _ in range(30)]).astype(np.int32)
    draws=cfg['source_draws']
    with (out/'data_schedule.npz').open('xb') as f:np.savez(f,orders=orders,draws=draws)
    prior={k:G.physical.state_dict()[k].clone() for k in ['prior_mean','prior_covariance']};update=0;train_start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    with (out/'training.jsonl').open('x',encoding='utf-8',buffering=1) as log:
        for epoch,number in enumerate(draws,1):
            begin=time.perf_counter();v=draw(number);sums={};step_sum=0.;norm_sums={'physical':0.,'detail':0.}
            for offset in range(0,63000,256):
                take=orders[epoch-1,offset:offset+256];optimizer.zero_grad(set_to_none=True)
                for group in optimizer.param_groups:group['lr']=learning_rate(update+1)
                step=0.
                for start in range(0,len(take),32):
                    indices=torch.tensor(take[start:start+32].tolist(),device='cuda:0',dtype=torch.long)
                    restored,aux,detail,detail_terms=detail_pass(v['received'][indices],clean[indices],targets[indices])
                    physical,physical_terms=source_action_loss(aux,v['kernel'][indices],v['coefficients'][indices],v['variance'][indices])
                    scalar=(physical+detail).sum()/len(take)
                    if not torch.isfinite(scalar) or aux['relative_solve_residual'].max()>1e-8:raise ValueError('Nonfinite source objective or posterior failure;preserve')
                    scalar.backward();step+=float(scalar.detach())
                    for key,value in dict(loss=physical+detail,physical_loss=physical,detail_loss=detail,**physical_terms,**detail_terms,
                            base_iq_nmse=(aux['base_iq']-clean[indices]).square().mean((1,2))/clean[indices].square().mean((1,2)).clamp_min(1e-4),
                            predicted_noise=aux['variance'],relative_solve_residual=aux['relative_solve_residual']).items():sums[key]=sums.get(key,0.)+float(value.detach().sum())
                norms=clip_branch_gradients(G)
                for key,value in norms.items():norm_sums[key]+=float(value)
                optimizer.step();update+=1;step_sum+=step
            if any(not torch.equal(value,G.physical.state_dict()[key]) for key,value in prior.items()):raise ValueError('Fixed source prior changed')
            check_F0_state(F,frozen);torch.cuda.synchronize();trial.save_recovery(out/'recovery',G,optimizer,args.arm,args.commit,epoch,update)
            row=dict(epoch=epoch,draw=number,updates=update,steps=247,count=63000,L_s=6300,U_s=56700,seconds=time.perf_counter()-begin,
                last_learning_rate=learning_rate(update),microbatch=32,training_posterior_calls=1969,training_posterior_rows=63000,F0_training_forward_rows=63000,
                metrics={key:value/63000 for key,value in sums.items()},step_objective=step_sum/247,preclip_branch_norms={k:x/247 for k,x in norm_sums.items()})
            log.write(json.dumps(row,allow_nan=False)+'\n');print('COVARIANCE_DETAIL_EPOCH',args.arm,json.dumps(row),flush=True);del v
    if update!=7410:raise ValueError('Full single30 update budget required')
    trial.save(out/'final.pt',G,args.arm,args.commit,30)
    write(out/'complete.json',dict(status='SOURCE_TRAINED',code_commit=args.commit,arm=args.arm,epoch=30,updates=7410,
        source_Ls_exposures=189000,source_Us_exposures=1701000,online_F0_feature_rows=1890000,F0_clean_target_rows=63000,
        F0_feature_L_s_exposures=189000,F0_feature_U_s_exposures=1701000,source_F0_feature_cache_seconds=cache_seconds,
        F0_identity=identity,source_prior_identity=prior_identity,G_parameters=sum(p.numel() for p in G.parameters()),
        physical_parameters=sum(p.numel() for p in G.physical.parameters()),detail_parameters=sum(p.numel() for p in G.detail.parameters()),
        training_seconds=time.perf_counter()-train_start,worker_seconds=time.perf_counter()-started,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        G_initialization_checkpoint_loads=0,TX_labels_read=0,V_reads=0,OE_reads=0,target_training_reads=0,
        checkpoint_selection='final30_only',model_state_F0_updates=0,training_stages=1,training_posterior_calls=59070,training_posterior_rows=1890000,
        source_operator_full_matrix_supervision=True,source_prior_buffers_unchanged=True,source_actual_IQ_F0_guidance=cfg.get('source_actual_IQ_F0_guidance',True),feature_weight=.01,
        physical_detail_gradient_isolation=True,independent_branch_clipping=True))


def install():
    trial.RUN=RUN;trial.MODULE=MODULE;trial.ARMS=(ARM,)
    trial.configuration=configuration;trial.factory=factory;trial.train=train;trial.gpu_choice=recovery_gpu_choice


if __name__=='__main__':install();trial.main()
