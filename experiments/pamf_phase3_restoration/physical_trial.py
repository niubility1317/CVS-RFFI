"""Single scratch30 PHY-MSAT candidate; existing full Main truth-last runner."""
import argparse
import copy
import json
from pathlib import Path
import time
import numpy as np
import torch
from . import multiview_trial as trial
from .physical import scratch_physical,action_supervision,PRIOR_COUNT,CG_STEPS,PRIOR_VARIANCE,MIN_NOISE,TOTAL_CG,COMPLEX_FFT_COUNT
from .variance_trial import load_scale,variance_feature_guide,recovery_gpu_choice
from .cache import tensor,load_F0,check_F0_state
from .objective import PairedObjective
from .source_pair import read,validate_source
from .train import learning_rate
from .cost_probe import write

RUN='20261101-phase1-Restore-PHYMSAT-E30-s2026092701-r01'
MODULE='experiments.pamf_phase3_restoration.physical_trial'
ARM='PHY_MSat'
TEMPLATE=trial.configuration('MSAT_M20')


def configuration(arm):
    if arm!=ARM:raise ValueError('Only fixed jointly trained PHY-MSAT candidate allowed')
    c=copy.deepcopy(TEMPLATE)
    c.update(run_id=RUN,arm=arm,architecture='source-action-estimator+two-unshared-contextual-MSAT-priors+real-linear-reflect-MAP',
        prior_count=PRIOR_COUNT,CG_steps_each=CG_STEPS,prior_real_variance=PRIOR_VARIANCE,
        objective='IQNMSE+.05STFT+.1(F0cos+source-var-featureMSE)/IQgradcap.5+.1source-action-forwardNMSE+.001source-noise-logMSE',
        source_action_supervision='R07train/kernel+main-image-coefficients+real-noise-variance;simulator-traces-not-TX-labels',
        source_action_loss_weights={'forward_nmse':.1,'noise_log_mse':.001},minimum_positive_noise=MIN_NOISE,
        feature_geometry_source='20261101-diagnostic-Restore-learned-prior-physics-s2026092701-r01',
        feature_variance_floor='max(var,.01mean(var),1e-10)',deployment_inputs='current_received_IQ_only',
        deployment_complex_FFT_IFFT_count=COMPLEX_FFT_COUNT,deployment_CG_iterations=TOTAL_CG,training_stages=1)
    return c


def factory(arm):
    configuration(arm);return scratch_physical(2026092701)


def choices():return [g for g in recovery_gpu_choice() if g['free_memory_MB']>=16000]


def train(args,out):
    start=time.perf_counter();c=validate_source(read(args.inputs));old,roles=trial.source(args)
    F,identity=load_F0(c,'cuda:0');trial.main_eval.valid_identity(identity)
    frozen={k:v.detach().clone() for k,v in F.state_dict().items()}
    a=np.load(old/'train/clean.npy',mmap_mode='r',allow_pickle=False)
    if a.shape!=(63000,2,256) or a.dtype!=np.float32:raise ValueError('Original source clean FP32 required')
    try:y=tensor(a,'cuda:0')
    finally:a._mmap.close()
    if not torch.isfinite(y).all():raise ValueError('Finite source clean required')
    out.mkdir();write(out/'config.json',dict(configuration(args.arm),code_commit=args.commit,F0_identity=identity))
    G=factory(args.arm).to('cuda:0');objective=PairedObjective().to('cuda:0')
    def source_draw(draw):
        folder=old/'epochs'/f'{draw:02d}';done=read(folder/'complete.json')
        if (done.get('count'),done.get('TX_labels_read'),done.get('target_reads'))!=(63000,0,0):raise ValueError('Original source training trace required')
        shapes={'received':(63000,2,256),'kernel':(63000,2,64),'coefficients':(63000,4,256),'variance':(63000,)}
        values={}
        for key,shape in shapes.items():
            a=np.load(folder/(key+'.npy'),mmap_mode='r',allow_pickle=False)
            try:
                if a.shape!=shape or a.dtype!=np.float32:raise ValueError('Original source action/received shape differs')
                values[key]=tensor(a,'cuda:0')
            finally:a._mmap.close()
        if any(not torch.isfinite(x).all() for x in values.values()) or (values['variance']<0).any():raise ValueError('Finite source simulator trace/noise required')
        return values
    # The first real neural work is source2, including all new physics gradients.
    first=source_draw(1);before={k:v.detach().clone() for k,v in G.state_dict().items()}
    with torch.no_grad():target=F.features(y[:2]).detach()
    restored,aux=G(first['received'][:2],return_aux=True)
    torch.testing.assert_close(restored,first['received'][:2],rtol=2e-5,atol=2e-5)
    physical,_=objective(restored,y[:2]);guide,_=trial.feature_guide(F,restored,y[:2],target)
    trace,_=action_supervision(aux,y[:2],first['kernel'][:2],first['coefficients'][:2],first['variance'][:2])
    (physical+guide+trace).mean().backward()
    if any(p.output.weight.grad is None or not torch.isfinite(p.output.weight.grad).all() or not p.output.weight.grad.abs().sum()>0 for p in G.priors):
        raise ValueError('Both new prior output heads need real source gradients')
    G.zero_grad(set_to_none=True)
    if any(not torch.equal(v,G.state_dict()[k]) for k,v in before.items()):raise ValueError('No-update source smoke changed G')
    check_F0_state(F,frozen);trial.save(out/'smoke_E0.pt',G,args.arm,args.commit,0)
    probe=trial.load(out/'smoke_E0.pt',args.arm,args.commit,'cuda:0',epoch=0)
    with torch.no_grad():torch.testing.assert_close(probe(first['received'][:2]),restored,rtol=2e-5,atol=2e-5)
    del probe,first,restored,aux,physical,guide,trace
    write(out/'smoke.json',dict(status='PASS',source_rows=2,actual_original_F0=True,new_physics_and_both_prior_gradients=True,
        near_identity=True,strict_E0_reload=True,optimizer_updates=0,TX_labels_read=0,V_reads=0,OE_reads=0,target_inputs=0))
    print('PHYMSAT_REAL_SOURCE_SMOKE_PASS',flush=True)
    cache_start=time.perf_counter()
    with torch.no_grad():features=torch.cat([F.features(y[i:i+256]).detach() for i in range(0,63000,256)])
    torch.cuda.synchronize();cache_seconds=time.perf_counter()-cache_start
    G.train();optimizer=torch.optim.AdamW(G.parameters(),lr=.001,weight_decay=1e-4)
    rng=np.random.default_rng(2026110102);orders=np.stack([rng.permutation(63000) for _ in range(30)]).astype(np.int32)
    with (out/'data_schedule.npz').open('xb') as f:np.savez(f,orders=orders,draws=configuration(args.arm)['source_draws'])
    update=0;train_start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    with (out/'training.jsonl').open('x',encoding='utf-8',buffering=1) as log:
        for epoch,draw in enumerate(configuration(args.arm)['source_draws'],1):
            begin=time.perf_counter();v=source_draw(draw);sums={}
            for offset in range(0,63000,256):
                take=torch.tensor(orders[epoch-1,offset:offset+256].tolist(),device='cuda:0',dtype=torch.long)
                optimizer.zero_grad(set_to_none=True)
                for group in optimizer.param_groups:group['lr']=learning_rate(update+1)
                restored,aux=G(v['received'][take],return_aux=True)
                loss,terms=objective(restored,y[take]);guide,info=trial.feature_guide(F,restored,y[take],features[take])
                trace,trace_terms=action_supervision(aux,y[take],v['kernel'][take],v['coefficients'][take],v['variance'][take])
                loss=loss+guide+trace
                if not torch.isfinite(loss).all():raise ValueError('Nonfinite joint source loss; preserve run')
                loss.mean().backward();torch.nn.utils.clip_grad_norm_(G.parameters(),1.,error_if_nonfinite=True)
                optimizer.step();update+=1
                metrics=dict(loss=loss,**terms,**trace_terms,feature_error=info['feature_error'],
                    feature_standardized_error=info['feature_standardized_error'],guide_weight=info['weight'])
                for key,value in metrics.items():sums[key]=sums.get(key,0.)+float(value.detach().sum())
            check_F0_state(F,frozen);torch.cuda.synchronize();trial.save_recovery(out/'recovery',G,optimizer,args.arm,args.commit,epoch,update)
            row=dict(epoch=epoch,draw=draw,updates=update,steps=247,count=63000,L_s=6300,U_s=56700,
                seconds=time.perf_counter()-begin,last_learning_rate=learning_rate(update),metrics={k:x/63000 for k,x in sums.items()})
            log.write(json.dumps(row,allow_nan=False)+'\n');print('PHYMSAT_EPOCH',args.arm,json.dumps(row),flush=True)
            del v
    if update!=7410:raise ValueError('Full scratch single30 budget required')
    trial.save(out/'final.pt',G,args.arm,args.commit,30)
    write(out/'complete.json',dict(status='SOURCE_TRAINED',code_commit=args.commit,arm=args.arm,epoch=30,updates=7410,
        source_Ls_exposures=189000,source_Us_exposures=1701000,online_F0_feature_rows=1890000,F0_clean_target_rows=63000,
        source_F0_feature_cache_seconds=cache_seconds,F0_identity=identity,G_parameters=sum(p.numel() for p in G.parameters()),
        training_seconds=time.perf_counter()-train_start,worker_seconds=time.perf_counter()-start,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        G_initialization_checkpoint_loads=0,TX_labels_read=0,V_reads=0,OE_reads=0,target_training_reads=0,
        checkpoint_selection='final30_only',model_state_F0_updates=0,training_stages=1,
        source_simulator_action_supervision=True,CG_iterations=TOTAL_CG,complex_FFT_IFFT_count=COMPLEX_FFT_COUNT))


def install(scale):
    by_device={}
    def guide(reader,restored,clean,target):
        if restored.device not in by_device:by_device[restored.device]=scale.to(restored.device)
        return variance_feature_guide(reader,restored,clean,target,by_device[restored.device])
    trial.RUN=RUN;trial.MODULE=MODULE;trial.ARMS=(ARM,)
    trial.configuration=configuration;trial.factory=factory;trial.train=train;trial.feature_guide=guide;trial.gpu_choice=choices


if __name__=='__main__':
    p=argparse.ArgumentParser(add_help=False);p.add_argument('--root',type=Path,required=True);args,_=p.parse_known_args()
    install(load_scale(args.root));trial.main()
