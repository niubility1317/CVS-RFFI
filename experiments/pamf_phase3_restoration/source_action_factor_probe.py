"""Frozen source attribution of FIR shape and timewise receiver modulation."""
import argparse
import os
from pathlib import Path
import time

import numpy as np
import torch

from .cache import check_F0_state, load_F0, tensor
from .cost_probe import write
from .covariance_guidance_source import source_metrics
from .covariance_inverse import dense_action, gaussian_posterior
from .covariance_parameter_source import G_COMMIT, load_model
from .factorized_action_loss import aligned_factors
from .main_split_eval import valid_identity
from .phase_mixture_source import REFERENCE_COMMIT, REFERENCE_RUN, read_trace
from .source_coordinate_probe import select_indices
from .source_inverse import R07, SCENES
from .source_pair import read, validate_source
from .structured_noise_source import known_parameters

RUN = '20261107-diagnostic-Restore-action-factor-sourceV-s2026092701-r01'
MODULE = 'experiments.pamf_phase3_restoration.source_action_factor_probe'
METHODS = ('blind','true_fir','true_modulation','true_action','true_action_noise')


def readouts(G,trace,*,verify_gauge=False):
    received=trace['received']
    estimate=G.estimate(received,dense=False)
    native=[estimate[k].to(torch.complex128) for k in ('kernel','main','image')]
    h,m,i,noise=known_parameters(trace)
    aligned,known,correlation=aligned_factors(native,(h,m,i))
    a=dense_action(*aligned)
    true_action=dense_action(*known)
    if verify_gauge:
        torch.testing.assert_close(a,dense_action(*native),rtol=1e-11,atol=1e-12)
        torch.testing.assert_close(true_action,dense_action(h,m,i),rtol=1e-11,atol=1e-12)
    actions=(a,dense_action(known[0],*aligned[1:]),dense_action(aligned[0],*known[1:]),true_action,true_action)
    variances=[estimate['variance'].double()]*4+[noise]
    mean,covariance=G.prior_mean.double(),G.prior_covariance.double()
    observed=received.flatten(1).double()
    energy=true_action.square().sum((1,2)).clamp_min(1e-8)
    outputs={}
    for method,operator,var in zip(METHODS,actions,variances):
        iq,residual=gaussian_posterior(operator,observed,mean,covariance,var)
        if residual.max()>1e-8:
            raise ValueError('Source posterior linear residual exceeds 1e-8')
        outputs[method]=dict(iq=iq.float().reshape_as(received),
            action_relative_mse=(operator-true_action).square().sum((1,2))/energy)
    return outputs,correlation


def references(root,cache,identity):
    folder=root/'runs'/REFERENCE_RUN
    done=read(folder/'predictions_complete.json')
    if (done['status'],done['code_commit'],done['source_V_packets'],done['TX_labels_read'],done['target_inputs'])!=(
            'SOURCE_V_PREDICTIONS_COMPLETE',REFERENCE_COMMIT,1800,0,0) or done['F0_identity']!=identity:
        raise ValueError('Exact cached original source V reference and F0 required')
    with np.load(folder/'references.npz',allow_pickle=False) as f:
        take=select_indices(f['rx'],f['day'])
        if not np.array_equal(f['ids'],np.load(cache/'validation/physical_ids.npy',allow_pickle=False)):
            raise ValueError('Original V physical order differs')
        ref={key:f[key][take] for key in ['ids','rx','day','clean','features','logits']}
        ref['scale']=f['scale']
    return ref,take


def worker(args,out):
    start=time.perf_counter()
    G=load_model(args.root,'cpu')
    F,identity=load_F0(validate_source(read(args.inputs)),'cpu')
    valid_identity(identity)
    models=(G,F)
    snapshots=[{k:v.detach().clone() for k,v in model.state_dict().items()} for model in models]
    cache=args.root/'runs'/R07/'cache'
    take=np.flatnonzero(np.load(cache/'train/roles.npy',allow_pickle=False)=='L_s')[:2]
    if len(take)!=2:
        raise ValueError('Two original L_s smoke records required')
    with torch.inference_mode():
        source=read_trace(cache/'epochs/01',take,63000)
        values,_=readouts(G,source,verify_gauge=True)
        torch.testing.assert_close(values['blind']['iq'],G(source['received']),rtol=2e-6,atol=2e-7)
        for view in values.values():
            logits=F.classify_features(F.features(view['iq']))
            if logits.shape!=(2,6) or not torch.isfinite(logits).all():
                raise ValueError('Original actual BF/F0 source smoke failed')
    write(out/'smoke.json',dict(status='PASS',actual_Ls=2,actual_BF=True,actual_F0=True,
        blind_matches_original=True,gauge_preserves_native_operators=True,F0_rows=10,TX_labels_read=0,target_inputs=0))
    print('SOURCE_ACTION_FACTOR_REAL_SMOKE_PASS',flush=True)
    ref,take=references(args.root,cache,identity)
    with (out/'reference.npz').open('xb') as f:
        np.savez(f,ids=ref['ids'],rx=ref['rx'],day=ref['day'],logits=ref['logits'],source_index=take)
    clean,features,logits,scale=[tensor(ref[key],'cpu') for key in ('clean','features','logits','scale')]
    rows=[]
    with torch.inference_mode():
        for scene in SCENES:
            trace=read_trace(cache/'validation_actions'/scene,take,1800)
            records={}
            for offset in range(0,len(take),16):
                index=slice(offset,offset+16)
                views,corr=readouts(G,{key:value[index] for key,value in trace.items()})
                values=dict(FIR_gauge_correlation=corr)
                for method,view in views.items():
                    z=F.features(view['iq'])
                    p=F.classify_features(z)
                    metrics=source_metrics(view['iq'],clean[index],z,p,features[index],logits[index],scale)
                    values.update({method+'__'+key:value for key,value in dict(metrics,logits=p,
                        action_relative_mse=view['action_relative_mse']).items()})
                if not all(torch.isfinite(value).all() for value in values.values()):
                    raise ValueError('Nonfinite frozen source output')
                for key,value in values.items():
                    records.setdefault(key,[]).extend(value.tolist())
            arrays={key:np.asarray(value) for key,value in records.items()}
            with (out/(scene+'.npz')).open('xb') as f:
                np.savez(f,**arrays)
            base=arrays['blind__clean_decision_agreement'].astype(bool)
            for method in METHODS:
                after=arrays[method+'__clean_decision_agreement'].astype(bool)
                rows.append(dict(scene=scene,method=method,count=len(take),
                    IQ_NMSE=float(arrays[method+'__iq_nmse'].mean()),
                    action_relative_mse=float(arrays[method+'__action_relative_mse'].mean()),
                    feature_cosine_distance=float(arrays[method+'__feature_cosine_distance'].mean()),
                    Fclean_agreement=float(after.mean()),
                    repairs_vs_blind=int((~base&after).sum()),damage_vs_blind=int((base&~after).sum()),
                    FIR_gauge_min_correlation=float(arrays['FIR_gauge_correlation'].min()),
                    FIR_gauge_mean_correlation=float(arrays['FIR_gauge_correlation'].mean())))
            print('SOURCE_ACTION_FACTOR_VIEW_COMPLETE',scene,len(take),flush=True)
    for model,before in zip(models,snapshots):
        check_F0_state(model,before)
    write(out/'results.json',dict(status='SOURCE_ACTION_FACTOR_COMPLETE',run_id=RUN,code_commit=args.commit,
        frozen_BF_code=G_COMMIT,reference_code=REFERENCE_COMMIT,methods=list(METHODS),views=list(SCENES),
        source_V_records=480,source_V_views=3360,F0_new_rows=16800,F0_smoke_rows=10,F0_clean_rows_reused=480,
        estimator_rows=3360,estimator_smoke_rows=4,records=rows,CPU_threads=2,
        TX_labels_read=0,OE_reads=0,novel_reads=0,target_inputs=0,optimizer_updates=0,model_state_updates=0,
        new_deployment_candidate=False,theoretical_ceiling=False,seconds=time.perf_counter()-start))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--inputs',type=Path,required=True)
    p.add_argument('--commit',required=True)
    args=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or (Path.cwd()/'release_commit.txt').read_text().strip()!=args.commit:
        raise ValueError('CUDA-empty source diagnostic with actual code identity required')
    torch.set_num_threads(2)
    out=args.root/'runs'/RUN
    out.mkdir()
    write(out/'launch.json',dict(pid=os.getpid(),code_commit=args.commit,CPU_threads=2,CUDA_empty=True))
    worker(args,out)
    write(out/'complete.json',dict(status='ARTIFACTS_COMPLETE',TX_labels_read=0,target_inputs=0,optimizer_updates=0))


if __name__=='__main__':
    main()
