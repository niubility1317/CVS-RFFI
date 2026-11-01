"""Scratch30 source-operator training and frozen full-covariance deployment/Main."""
import argparse
import copy
from pathlib import Path
import time
import numpy as np
import torch
from . import multiview_trial as trial
from .covariance_model import scratch_covariance,source_action_loss
from .covariance_inverse import RUN as PRIOR_RUN
from .variance_trial import recovery_gpu_choice
from .cache import tensor,load_F0,check_F0_state
from .source_pair import read,validate_source
from .train import learning_rate
from .cost_probe import write

RUN='20261102-phase1-Restore-CovarianceMAP-E30-s2026092701-r01'
MODULE='experiments.pamf_phase3_restoration.covariance_trial'
ARM='TF_CovarianceMAP'
TEMPLATE=trial.configuration('MSAT_M20')
PRIOR_COMMIT='a3a90b55a73702954c453eec9c1ad17fec24d87f'


def configuration(arm):
    if arm!=ARM:raise ValueError('Only fixed source-operator/covariance candidate allowed')
    c=copy.deepcopy(TEMPLATE)
    c.update(run_id=RUN,arm=arm,architecture='time-frequency-patch4/128tokens/width128/6Transformer4heads/learnedFIR64-main-image-noise/full512realGaussianposterior',
        objective='.1full-source-Action-matrix-relative-MSE+.001real-source-noise-log-MSE',
        operator_loss='All512realbasisdirections via full Frobenius relative error;notonlyAction(clean). True sourceAction usedonlyloss.',
        training_path='G.estimate only/no posterior-Cholesky/F0/IQdecoder/featurecache/teacher in updates',
        logical_batch=256,microbatch=32,gradient_accumulation='sum each microbatch loss divided by actual logical batch count;clip once per update',
        source_action_supervision='OriginalR07simulatorFIR/main-image/realnoisevariance;0TXlabels',
        source_prior_run=PRIOR_RUN,source_prior_code=PRIOR_COMMIT,source_prior_fit_packets=63000,
        source_prior='Globalreal512mean/populationC/.001shrinkage-to-sametrace-tauI;all512dimensions/noTX-RXgrouping',
        source_prior_storage='FP32persistentbuffers/fromFP64sourcefit;exactFP64denseposteriorarithmetic',
        source_F0_knowledge='Original3ff8F0onlysource2no-querysmokeandafterfreezingMain;0F0forwardsinGupdates',
        F0_guidance_for_G_updates=False,new_G_initialization_checkpoints=0,learned_IQ_decoder=False,unit_power_projection=False,
        posterior='mu+C Ahat^T(Ahat C Ahat^T+estimated_real_noise I)^-1(y-Ahat mu)/FP64Cholesky/FP32IQ',
        predicted_noise_range=[1e-7,1.],deployment_inputs='currentreceivedIQonly/savedsourcepriorbuffers;0trueAction-noise-F0-clean',
        training_stages=1,training_F0_forward_rows=0,training_posterior_calls=0,default_DURA_changed=False)
    return c


def factory(arm):
    configuration(arm);return scratch_covariance(2026092701)


def bind_prior(root,model):
    folder=root/'runs'/PRIOR_RUN;m=read(folder/'results.json')
    if (read(folder/'complete.json')['status'],m['status'],m['code_commit'],m['source_prior_fit_packets'],m['source_prior_shrinkage'])!=(
            'ARTIFACTS_COMPLETE','SOURCE_COVARIANCE_INVERSE_COMPLETE',PRIOR_COMMIT,63000,.001):
        raise ValueError('Original completed source-only full covariance statistics required')
    if m['source_prior_roles']!={'L_s':6300,'U_s':56700} or any(m[k] for k in ['TX_labels_read','V_reads','OE_reads','novel_reads','target_inputs','optimizer_updates']):
        raise ValueError('Original source-only prior contract required')
    with np.load(folder/'source_prior.npz',allow_pickle=False) as f:mean=f['mean'].copy();cov=f['covariance'].copy()
    model.bind_prior(tensor(mean,'cpu'),tensor(cov,'cpu'))
    return dict(source_run=PRIOR_RUN,source_prior_code=PRIOR_COMMIT,source_fit_packets=63000,
        source_mean_FP32_rounding_max=float(np.max(np.abs(mean.astype(np.float32).astype(np.float64)-mean))),
        source_covariance_FP32_rounding_max=float(np.max(np.abs(cov.astype(np.float32).astype(np.float64)-cov))))


def train(args,out):
    start=time.perf_counter();c=validate_source(read(args.inputs));old,roles=trial.source(args)
    G=factory(args.arm);prior_identity=bind_prior(args.root,G);G=G.to('cuda:0')
    F,identity=load_F0(c,'cuda:0');trial.main_eval.valid_identity(identity)
    frozen={k:v.detach().clone() for k,v in F.state_dict().items()}
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
    # Two deterministic non-clean source cases exercise every action output head.
    with np.load(old/'epochs/01/groups.npz',allow_pickle=False) as groups:
        smoke_indices=np.flatnonzero(groups['scene']!='clean')[:2]
    if len(smoke_indices)!=2:raise ValueError('Two original non-clean source smoke cases required')
    take=torch.tensor(smoke_indices.tolist(),device='cuda:0',dtype=torch.long)
    # Actual full deployment and original F0 first, followed by the cheap operator training path.
    with torch.no_grad():
        restored,aux=G(first['received'][take],return_aux=True)
        actual=F.classify_features(F.features(restored))
        if actual.shape!=(2,6) or not torch.isfinite(actual).all() or aux['relative_solve_residual'].max()>1e-8:
            raise ValueError('Actual covariance deployment/source2 F0 smoke failed')
    aux=G.estimate(first['received'][take]);loss,_=source_action_loss(aux,first['kernel'][take],first['coefficients'][take],first['variance'][take]);loss.mean().backward()
    for head in (G.kernel_head,G.coefficient_head,G.noise_head):
        if head.weight.grad is None or not torch.isfinite(head.weight.grad).all() or not head.weight.grad.abs().sum()>0:
            raise ValueError('All action/noise heads require actual source trace gradients')
    G.zero_grad(set_to_none=True)
    if any(not torch.equal(v,G.state_dict()[k]) for k,v in before.items()):raise ValueError('No-update source smoke changed G')
    check_F0_state(F,frozen);trial.save(out/'smoke_E0.pt',G,args.arm,args.commit,0)
    probe=trial.load(out/'smoke_E0.pt',args.arm,args.commit,'cuda:0',epoch=0)
    with torch.no_grad():torch.testing.assert_close(probe(first['received'][take]),restored,rtol=0,atol=0)
    del probe,restored,aux,loss,before,first,F,frozen
    write(out/'smoke.json',dict(status='PASS',source_rows=2,actual_original_F0=True,actual_full_posterior=True,FP64_Cholesky=True,
        source_operator_gradients_all_heads=True,strict_E0_reload=True,source_prior_bound=True,source_indices=smoke_indices.tolist(),optimizer_updates=0,
        TX_labels_read=0,V_reads=0,OE_reads=0,target_inputs=0,F0_gradient_guidance=False))
    print('COVARIANCE_MAP_REAL_SOURCE_SMOKE_PASS',flush=True)
    G.train();optimizer=torch.optim.AdamW(G.parameters(),lr=.001,weight_decay=1e-4)
    rng=np.random.default_rng(2026110102);orders=np.stack([rng.permutation(63000) for _ in range(30)]).astype(np.int32)
    with (out/'data_schedule.npz').open('xb') as f:np.savez(f,orders=orders,draws=configuration(args.arm)['source_draws'])
    update=0;train_start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    fixed_prior={k:G.state_dict()[k].detach().clone() for k in ['prior_mean','prior_covariance']}
    with (out/'training.jsonl').open('x',encoding='utf-8',buffering=1) as log:
        for epoch,draw in enumerate(configuration(args.arm)['source_draws'],1):
            begin=time.perf_counter();v=source_draw(draw);sums={}
            for offset in range(0,63000,256):
                take=orders[epoch-1,offset:offset+256];optimizer.zero_grad(set_to_none=True)
                for group in optimizer.param_groups:group['lr']=learning_rate(update+1)
                for begin_micro in range(0,len(take),32):
                    indices=torch.tensor(take[begin_micro:begin_micro+32].tolist(),device='cuda:0',dtype=torch.long)
                    aux=G.estimate(v['received'][indices]);loss,terms=source_action_loss(aux,v['kernel'][indices],v['coefficients'][indices],v['variance'][indices])
                    if not torch.isfinite(loss).all():raise ValueError('Nonfinite operator loss;preserve run')
                    (loss.sum()/len(take)).backward()
                    metrics=dict(loss=loss,**terms,predicted_noise=aux['variance'])
                    for key,value in metrics.items():sums[key]=sums.get(key,0.)+float(value.detach().sum())
                torch.nn.utils.clip_grad_norm_(G.parameters(),1.,error_if_nonfinite=True);optimizer.step();update+=1
            if any(not torch.equal(value,G.state_dict()[key]) for key,value in fixed_prior.items()):raise ValueError('Fixed source prior changed')
            torch.cuda.synchronize();trial.save_recovery(out/'recovery',G,optimizer,args.arm,args.commit,epoch,update)
            row=dict(epoch=epoch,draw=draw,updates=update,steps=247,count=63000,L_s=6300,U_s=56700,
                seconds=time.perf_counter()-begin,last_learning_rate=learning_rate(update),microbatch=32,training_posterior_calls=0,F0_training_forward_rows=0,
                metrics={key:value/63000 for key,value in sums.items()})
            log.write(__import__('json').dumps(row,allow_nan=False)+'\n');print('COVARIANCE_MAP_EPOCH',args.arm,__import__('json').dumps(row),flush=True)
            del v
    if update!=7410:raise ValueError('Full scratch single30 budget required')
    trial.save(out/'final.pt',G,args.arm,args.commit,30)
    write(out/'complete.json',dict(status='SOURCE_TRAINED',code_commit=args.commit,arm=args.arm,epoch=30,updates=7410,
        source_Ls_exposures=189000,source_Us_exposures=1701000,online_F0_feature_rows=0,F0_clean_target_rows=0,
        F0_identity=identity,source_prior_identity=prior_identity,G_parameters=sum(p.numel() for p in G.parameters()),
        training_seconds=time.perf_counter()-train_start,worker_seconds=time.perf_counter()-start,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        G_initialization_checkpoint_loads=0,TX_labels_read=0,V_reads=0,OE_reads=0,target_training_reads=0,
        checkpoint_selection='final30_only',model_state_F0_updates=0,training_stages=1,training_posterior_calls=0,
        source_operator_full_matrix_supervision=True,source_prior_buffers_unchanged=True))


def install():
    trial.RUN=RUN;trial.MODULE=MODULE;trial.ARMS=(ARM,)
    trial.configuration=configuration;trial.factory=factory;trial.train=train;trial.gpu_choice=recovery_gpu_choice


if __name__=='__main__':
    install();trial.main()
