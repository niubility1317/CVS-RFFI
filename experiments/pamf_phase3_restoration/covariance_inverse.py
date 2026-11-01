"""Frozen source Gaussian oracle with full real IQ covariance; never deployable."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import torch
from . import multiview_trial as base
from .source_inverse import panel,action_values
from .source_pair import read,validate_source
from .cache import tensor,load_F0,check_F0_state
from .variance_trial import recovery_gpu_choice
from .cost_probe import write

RUN='20261102-diagnostic-Restore-source-covariance-inverse-s2026092701-r01'
MODULE='experiments.pamf_phase3_restoration.covariance_inverse'
CONTROL='20261101-diagnostic-Restore-source-inverse-s2026092701-r01'
METHODS=('isotropic','full_covariance')
METRICS=('iq_nmse','feature_cosine_distance','clean_decision_agreement','relative_solve_residual')
SHRINKAGE=.001


def dense_action(kernel,main,image):
    """Real [Re(x),Im(x)] matrix of the saved causal-reflect widely linear action."""
    if (kernel.ndim!=2 or main.ndim!=2 or main.shape!=image.shape or len(kernel)!=len(main)
            or not 1<=kernel.shape[1]<=main.shape[1]):
        raise ValueError('Matched complex FIR/main/image source action required')
    if not all(x.is_complex() and torch.isfinite(x).all() for x in (kernel,main,image)):
        raise ValueError('Finite complex source action required')
    b,n=main.shape;k=kernel.shape[1]
    index=(torch.arange(n,device=main.device)[:,None]-torch.arange(k,device=main.device)[None]).abs()
    fir=torch.zeros((b,n,n),dtype=kernel.dtype,device=kernel.device).scatter_add(
        2,index[None].expand(b,-1,-1),kernel[:,None].expand(-1,n,-1))
    p,q,r,s=[x[:,:,None] for x in (main.real,main.imag,image.real,image.imag)]
    a,b0=fir.real,fir.imag
    return torch.cat((torch.cat(((p+r)*a+(-q+s)*b0,-(p+r)*b0+(-q+s)*a),2),
        torch.cat(((q+s)*a+(p-r)*b0,-(q+s)*b0+(p-r)*a),2)),1)


def gaussian_posterior(action,observed,mean,covariance,variance):
    """Exact linear-Gaussian mean via FP64 Cholesky, without an iteration budget."""
    b,d=observed.shape
    if (action.shape!=(b,d,d) or mean.shape!=(d,) or covariance.shape!=(d,d)
            or variance.shape!=(b,) or not torch.isfinite(variance).all() or (variance<=0).any()):
        raise ValueError('Matched dense action, Gaussian prior and positive real noise required')
    ac=action@covariance;normal=ac@action.transpose(1,2)+torch.diag_embed(variance[:,None].expand(-1,d))
    rhs=observed-action@mean
    solution=torch.cholesky_solve(rhs[:,:,None],torch.linalg.cholesky(normal)).squeeze(-1)
    restored=mean+(ac.transpose(1,2)@solution[:,:,None]).squeeze(-1)
    error=(normal@solution[:,:,None]).squeeze(-1)-rhs
    relative=error.norm(dim=1)/rhs.norm(dim=1).clamp_min(1e-30)
    if not torch.isfinite(restored).all() or not torch.isfinite(relative).all():
        raise ValueError('Nonfinite Gaussian posterior;preserve without fallback')
    return restored,relative


def source_prior(clean):
    if clean.shape!=(63000,2,256) or clean.dtype!=np.float32:
        raise ValueError('Original source63000 FP32 clean IQ required')
    mean=np.asarray(clean,dtype=np.float64).reshape(63000,512).mean(0)
    covariance=np.zeros((512,512),np.float64)
    for offset in range(0,63000,2048):
        residual=np.asarray(clean[offset:offset+2048],dtype=np.float64).reshape(-1,512)-mean
        covariance+=residual.T@residual
    covariance/=63000;tau=float(np.trace(covariance)/512)
    covariance=(1-SHRINKAGE)*covariance+SHRINKAGE*tau*np.eye(512)
    eigenvalues=np.linalg.eigvalsh(covariance)
    if not np.isfinite(covariance).all() or eigenvalues[0]<=0 or tau<=0:
        raise ValueError('Finite positive source covariance required')
    return mean,covariance,tau,eigenvalues


def summarize(a):
    if len(a['epoch'])!=10240 or set(a['epoch'])!=set(range(1,21)):
        raise ValueError('All fixed source512x20 cases required')
    result=[]
    masks=[('overall','ALL',np.ones(10240,bool))]
    for key in ('scene','lock'):masks.extend((key,str(v),a[key]==v) for v in np.unique(a[key]))
    for method in METHODS:
        after=a[method+'__clean_decision_agreement'].astype(bool)
        for dimension,stratum,mask in masks:
            pairs={}
            for name in ('isotropic','DURA_v1','received'):
                before=a[name+'__clean_decision_agreement'].astype(bool)
                repairs=int((~before&after&mask).sum());damage=int((before&~after&mask).sum())
                pairs[name]=dict(repairs=repairs,damage=damage,net=repairs-damage)
            result.append(dict(method=method,dimension=dimension,stratum=stratum,count=int(mask.sum()),
                metrics={k:float(a[method+'__'+k][mask].mean()) for k in METRICS},pairs=pairs))
    return result


def worker(args):
    started=time.perf_counter();torch.set_num_threads(2);device='cuda:0'
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    old,roles=base.source(args);selected=panel(roles)
    previous=args.root/'runs'/CONTROL;done=read(previous/'results.json')
    if (done['status'],done['count'],done['unique_source_packets'],done['existing_draws'])!=(
            'SOURCE_INVERSE_DIAGNOSTIC_COMPLETE',10240,512,20):raise ValueError('Completed source control required')
    with np.load(previous/'source_records.npz',allow_pickle=False) as f:control={k:f[k].copy() for k in f.files}
    a=np.load(old/'train/clean.npy',mmap_mode='r',allow_pickle=False)
    try:mu,covariance,tau,eigenvalues=source_prior(a);clean=tensor(a[selected],device)
    finally:a._mmap.close()
    # The established IQ helper intentionally casts to FP32; retain FP64 statistics here.
    mean=torch.frombuffer(memoryview(mu.copy()),dtype=torch.float64).to(device)
    cov=torch.frombuffer(memoryview(covariance.copy()),dtype=torch.float64).reshape(512,512).to(device)
    priors={'isotropic':tau*torch.eye(512,dtype=torch.float64,device=device),'full_covariance':cov}
    c=validate_source(read(args.inputs));F,identity=load_F0(c,device);base.main_eval.valid_identity(identity)
    if done['F0_identity']!=identity:raise ValueError('Exact frozen original F0 source control required')
    before={k:v.detach().clone() for k,v in F.state_dict().items()}
    out=args.root/'runs'/RUN
    records={k:[] for k in ['epoch','source_index','scene','lock','Fclean_prediction',
        *[m+'__'+key for m in METHODS for key in (*METRICS,'output_prediction')]]}
    with torch.inference_mode():
        for epoch in range(1,21):
            folder=old/'epochs'/f'{epoch:02d}';m=read(folder/'complete.json')
            if (m['count'],m['TX_labels_read'],m['target_reads'])!=(63000,0,0):raise ValueError('Original source draw required')
            with np.load(folder/'groups.npz',allow_pickle=False) as f:groups={k:f[k][selected] for k in ['scene','lock']}
            arrays={k:np.load(folder/(k+'.npy'),mmap_mode='r',allow_pickle=False) for k in ['received','kernel','coefficients','variance']}
            try:
                for offset in range(0,512,32):
                    take=selected[offset:offset+32];n=len(take);v={k:tensor(a[take],device) for k,a in arrays.items()}
                    h=torch.complex(v['kernel'][:,0],v['kernel'][:,1]).to(torch.complex128)
                    c0=torch.complex(v['coefficients'][:,0],v['coefficients'][:,1]).to(torch.complex128)
                    c1=torch.complex(v['coefficients'][:,2],v['coefficients'][:,3]).to(torch.complex128)
                    matrix=dense_action(h,c0,c1);reference=clean[offset:offset+n];energy=reference.square().mean((1,2))
                    if epoch==1 and offset==0:
                        # Launcher's first neural output: actual F0 on two oracle source outputs.
                        action=action_values(v['kernel'][:2],v['coefficients'][:2])
                        actual=action(torch.complex(reference[:2,0],reference[:2,1]))
                        dense=(matrix[:2]@reference[:2].flatten(1).double()[:,:,None]).squeeze(-1).reshape(2,2,256)
                        torch.testing.assert_close(dense.float(),torch.stack((actual.real,actual.imag),1),rtol=2e-5,atol=2e-5)
                        smoke,residual=gaussian_posterior(matrix[:2],v['received'][:2].flatten(1).double(),mean,cov,v['variance'][:2].double())
                        z=F.classify_features(F.features(smoke.float().reshape(2,2,256)))
                        if z.shape!=(2,6) or not torch.isfinite(z).all() or residual.max()>1e-8:raise ValueError('Actual source2 covariance/F0 smoke failed')
                        write(out/'smoke.json',dict(status='PASS',source_rows=2,dense_reflect_matches_actual_action=True,
                            actual_original_F0=True,FP64_Cholesky=True,TX_labels_read=0,V_reads=0,OE_reads=0,target_inputs=0,optimizer_updates=0))
                        print('SOURCE_COVARIANCE_REAL_SMOKE_PASS',flush=True)
                        target=F.features(clean);prediction=F.classify_features(target).argmax(1)
                    for method,prior in priors.items():
                        restored,residual=gaussian_posterior(matrix,v['received'].flatten(1).double(),mean,prior,v['variance'].double())
                        if residual.max()>1e-8:raise ValueError('Dense posterior solve residual too large;no fallback')
                        restored=restored.float().reshape(n,2,256);features=F.features(restored);p=F.classify_features(features).argmax(1)
                        metrics=dict(iq_nmse=(restored-reference).square().mean((1,2))/energy,
                            feature_cosine_distance=1-torch.nn.functional.cosine_similarity(features,target[offset:offset+n],dim=1),
                            clean_decision_agreement=p==prediction[offset:offset+n],relative_solve_residual=residual)
                        if any(not torch.isfinite(x).all() for x in metrics.values()):raise ValueError('Nonfinite complete source output')
                        for key,value in metrics.items():records[method+'__'+key].extend(value.cpu().tolist())
                        records[method+'__output_prediction'].extend(p.cpu().tolist())
                    for key,value in [('epoch',[epoch]*n),('source_index',take.tolist()),('scene',groups['scene'][offset:offset+n].tolist()),
                            ('lock',groups['lock'][offset:offset+n].tolist()),('Fclean_prediction',prediction[offset:offset+n].cpu().tolist())]:records[key].extend(value)
            finally:
                for a in arrays.values():a._mmap.close()
            print('SOURCE_COVARIANCE_DRAW_COMPLETE',epoch,512,flush=True)
    records={k:np.asarray(v) for k,v in records.items()}
    for key in ('epoch','source_index','scene','lock'):
        if not np.array_equal(records[key],control[key]):raise ValueError('Exact existing control case alignment required')
    for method in ('DURA_v1','received'):
        records[method+'__clean_decision_agreement']=control[method+'__clean_decision_agreement']
        records[method+'__iq_nmse']=control[method+'__nmse'];records[method+'__feature_cosine_distance']=control[method+'__feature_cosine_distance']
    check_F0_state(F,before)
    with (out/'source_records.npz').open('xb') as f:np.savez(f,**records)
    with (out/'source_prior.npz').open('xb') as f:np.savez(f,mean=mu,covariance=covariance,eigenvalues=eigenvalues)
    write(out/'results.json',dict(status='SOURCE_COVARIANCE_INVERSE_COMPLETE',run_id=RUN,code_commit=args.commit,F0_identity=identity,
        source_prior_fit_packets=63000,source_prior_roles={'L_s':6300,'U_s':56700},source_prior_shrinkage=SHRINKAGE,
        source_real_prior_variance=tau,source_covariance_min_eigenvalue=float(eigenvalues[0]),
        source_covariance_max_eigenvalue=float(eigenvalues[-1]),source_covariance_trace=float(eigenvalues.sum()),
        covariance_top_energy_fraction={str(k):float(eigenvalues[-k:].sum()/eigenvalues.sum()) for k in (16,32,64,128,256)},
        source_packets=512,source_draws=20,source_cases=10240,methods=list(METHODS),summary=summarize(records),
        F0_clean_feature_rows=512,F0_output_feature_rows=20480,smoke_extra_F0_feature_rows=2,
        new_G_forwards=0,TX_labels_read=0,V_reads=0,OE_reads=0,novel_reads=0,target_inputs=0,optimizer_updates=0,model_state_updates=0,
        source_in_sample=True,physical_oracle_deployable=False,agreement_is_TX_accuracy=False,theoretical_ceiling=False,
        default_DURA_changed=False,seconds=time.perf_counter()-started,goal_complete=False))


def controller(args):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CUDA-empty diagnostic controller required')
    out=args.root/'runs'/RUN;out.mkdir();(out/'logs').mkdir()
    write(out/'launch.json',dict(pid=os.getpid(),command=sys.argv,CWD=os.getcwd(),code_commit=args.commit))
    # The prior SigmaMix task is already complete; this diagnostic cannot overlap it.
    prior=args.root/'runs/20261102-phase1-Restore-DiffusionSigmaMix-E30-s2026092701-r01'
    if read(prior/'complete.json')['status']!='ARTIFACTS_COMPLETE':raise ValueError('Previous fixed task must finish first')
    choices=[g for g in recovery_gpu_choice() if g['GPU']==6]
    while not choices:time.sleep(5);choices=[g for g in recovery_gpu_choice() if g['GPU']==6]
    admission=choices[0];command=[sys.executable,'-u','-m',MODULE,'worker','--root',str(args.root),'--commit',args.commit,
        '--inputs',str(args.inputs),'--cache',str(args.cache)]
    with (out/'logs/worker.log').open('xb') as log:
        p=subprocess.Popen(command,cwd=Path.cwd(),env=dict(os.environ,CUDA_VISIBLE_DEVICES=admission['CUDA_ID']),
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    write(out/'submitted.json',dict(pid=p.pid,command=command,CWD=os.getcwd(),admission=admission))
    code=p.wait();write(out/'terminal.json',dict(pid=p.pid,exit_code=code))
    if code:raise RuntimeError('Source covariance diagnostic failed;preserve/no automatic retry')
    if read(out/'results.json')['status']!='SOURCE_COVARIANCE_INVERSE_COMPLETE':raise ValueError('All source cases must finish')
    write(out/'complete.json',dict(status='ARTIFACTS_COMPLETE',optimizer_updates=0,target_inputs=0,goal_complete=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['run','worker']);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--commit',required=True);p.add_argument('--inputs',type=Path,required=True);p.add_argument('--cache',type=Path,required=True);a=p.parse_args()
    if (Path.cwd()/'release_commit.txt').read_text().strip()!=a.commit:raise ValueError('Actual Git execution identity required')
    if a.mode=='run':controller(a)
    else:worker(a)
