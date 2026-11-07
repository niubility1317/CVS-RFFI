"""Frozen source V decomposition of a fixed Gaussian IQ posterior."""

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
from .main_split_eval import valid_identity
from .phase_mixture_source import REFERENCE_COMMIT, read_trace
from .posterior_bias_noise import posterior_bias_noise
from .source_action_factor_probe import RUN as CONTROL_RUN, references
from .source_inverse import R07, SCENES
from .source_pair import read, validate_source
from .structured_noise_source import known_parameters

RUN = '20261107-diagnostic-Restore-posterior-bias-noise-sourceV-s2026092701-r01'
MODULE = 'experiments.pamf_phase3_restoration.posterior_bias_noise_source'
CONTROL_COMMIT = '1065b7fd515402862a8b47bca8bddf597d823b54'
METHODS = ('known_gaussian','noise_removed','shrinkage_removed')


def load_control(folder,ref,take):
    d = read(folder/'results.json')
    if ((d['status'],d['code_commit'],d['source_V_records'],d['TX_labels_read'],d['target_inputs'])
            != ('SOURCE_ACTION_FACTOR_COMPLETE',CONTROL_COMMIT,480,0,0)
            or read(folder/'complete.json')['status']!='ARTIFACTS_COMPLETE'):
        raise ValueError('Original complete known-operator source controls required')
    with np.load(folder/'reference.npz',allow_pickle=False) as f:
        if not np.array_equal(f['source_index'],take) or not all(np.array_equal(f[k],ref[k]) for k in ('ids','rx','day','logits')):
            raise ValueError('Original known-operator panel differs')


def inputs(trace):
    h,m,i,variance = known_parameters(trace)
    return dense_action(h,m,i),trace['received'].flatten(1).double(),variance


def readouts(action,observed,variance,clean,mean,covariance):
    flat = clean.flatten(1).double()
    original,noiseless,unbiased = posterior_bias_noise(action,observed,flat,mean,covariance,variance)
    shrinkage = flat-noiseless
    noise = unbiased-flat
    power = flat.square().mean(1).clamp_min(1e-4)
    clean_obs = (action@flat[:,:,None]).squeeze(-1)
    evidence = dict(shrinkage_nmse=shrinkage.square().mean(1)/power,noise_leakage_nmse=noise.square().mean(1)/power,
        twice_noise_shrinkage_inner=2*(noise*shrinkage).mean(1)/power,linear_decomposition_max_error=(unbiased+noiseless-original-flat).abs().amax(1),
        observed_signal_power=clean_obs.square().mean(1),observed_noise_power=(observed-clean_obs).square().mean(1))
    outputs = {method:value.float().reshape_as(clean) for method,value in zip(METHODS,(original,noiseless,unbiased))}
    return outputs,evidence


def worker(args,out):
    start = time.perf_counter()
    G = load_model(args.root,'cpu')
    F,identity = load_F0(validate_source(read(args.inputs)),'cpu')
    valid_identity(identity)
    snapshots = [{key:value.detach().clone() for key,value in m.state_dict().items()} for m in (G,F)]
    mean,covariance = G.prior_mean.double(),G.prior_covariance.double()
    cache = args.root/'runs'/R07/'cache'
    source2 = np.flatnonzero(np.load(cache/'train/roles.npy',allow_pickle=False)=='L_s')[:2]
    if len(source2)!=2:
        raise ValueError('Two original L_s smoke records required')
    a = np.load(cache/'train/clean.npy',mmap_mode='r',allow_pickle=False)
    try:
        if a.shape!=(63000,2,256) or a.dtype!=np.float32:
            raise ValueError('Original L_s/U_s clean IQ layout required')
        clean2 = tensor(a[source2],'cpu')
    finally:
        a._mmap.close()
    with torch.inference_mode():
        action,observed,var = inputs(read_trace(cache/'epochs/01',source2,63000))
        values,evidence = readouts(action,observed,var,clean2,mean,covariance)
        expected,_ = gaussian_posterior(action,observed,mean,covariance,var)
        torch.testing.assert_close(values['known_gaussian'],expected.float().reshape_as(clean2),rtol=2e-6,atol=2e-7)
        if evidence['linear_decomposition_max_error'].max()>1e-10:
            raise ValueError('Exact source IQ decomposition required')
        for iq in values.values():
            logits = F.classify_features(F.features(iq))
            if logits.shape!=(2,6) or not torch.isfinite(logits).all():
                raise ValueError('Actual original F0 source smoke failed')
    write(out/'smoke.json',dict(status='PASS',actual_Ls=2,actual_original_BF=True,actual_original_F0=True,
        known_matches_original_Gaussian=True,exact_bias_noise_split=True,F0_rows=6,CPU_only=True,TX_labels_read=0,target_inputs=0))
    print('SOURCE_POSTERIOR_BIAS_NOISE_REAL_SMOKE_PASS',flush=True)

    ref,take = references(args.root,cache,identity)
    control = args.root/'runs'/CONTROL_RUN
    load_control(control,ref,take)
    with (out/'reference.npz').open('xb') as file:
        np.savez(file,**ref,source_index=take)
    clean,features,clean_logits,scale = [tensor(ref[k],'cpu') for k in ('clean','features','logits','scale')]
    summaries = []
    with torch.inference_mode():
        for scene in SCENES:
            trace = read_trace(cache/'validation_actions'/scene,take,1800)
            records = {}
            for offset in range(0,len(take),16):
                view = slice(offset,offset+16)
                action,observed,var = inputs({key:value[view] for key,value in trace.items()})
                outputs,values = readouts(action,observed,var,clean[view],mean,covariance)
                for name,iq in outputs.items():
                    z = F.features(iq)
                    logits = F.classify_features(z)
                    metrics = source_metrics(iq,clean[view],z,logits,features[view],clean_logits[view],scale)
                    values.update({name+'__'+key:value for key,value in dict(metrics,logits=logits).items()})
                if not all(torch.isfinite(value).all() for value in values.values()):
                    raise ValueError('Nonfinite frozen source bias/noise output')
                for key,value in values.items():
                    records.setdefault(key,[]).extend(value.tolist())
            arrays = {key:np.asarray(value) for key,value in records.items()}
            with (out/(scene+'.npz')).open('xb') as file:
                np.savez(file,**arrays)
            correct = {name:arrays[name+'__prediction']==ref['logits'].argmax(1) for name in METHODS}
            before = correct[METHODS[0]]
            for name,after in correct.items():
                summaries.append(dict(scene=scene,method=name,count=len(take),Fclean_correct=int(after.sum()),
                    repairs=int((~before&after).sum()),damage=int((before&~after).sum()),
                    iq_nmse=float(arrays[name+'__iq_nmse'].mean()),
                    feature_cosine_distance=float(arrays[name+'__feature_cosine_distance'].mean())))
            print('SOURCE_POSTERIOR_BIAS_NOISE_VIEW_COMPLETE',scene,len(take),flush=True)
    for model,before in zip((G,F),snapshots):
        check_F0_state(model,before)
    write(out/'results.json',dict(status='SOURCE_POSTERIOR_BIAS_NOISE_COMPLETE',run_id=RUN,code_commit=args.commit,
        F0_identity=identity,frozen_BF_code=G_COMMIT,reference_code=REFERENCE_COMMIT,control_code=CONTROL_COMMIT,
        methods=list(METHODS),views=list(SCENES),source_V_records=480,source_V_views=3360,F0_new_rows=10080,
        F0_smoke_rows=6,F0_clean_rows_reused=480,records=summaries,known_simulator_and_clean_IQ_diagnostic=True,
        TX_labels_read=0,OE_reads=0,novel_reads=0,target_inputs=0,optimizer_updates=0,model_state_updates=0,
        G_estimator_rows=0,CPU_threads=2,new_deployment_candidate=False,theoretical_ceiling=False,
        seconds=time.perf_counter()-start))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--inputs',type=Path,required=True)
    p.add_argument('--commit',required=True)
    args = p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or (Path.cwd()/'release_commit.txt').read_text().strip()!=args.commit:
        raise ValueError('CPU-only source diagnostic with exact release identity required')
    os.nice(10)
    torch.set_num_threads(2)
    out = args.root/'runs'/RUN
    out.mkdir()
    write(out/'launch.json',dict(pid=os.getpid(),code_commit=args.commit,CPU_threads=2,CUDA_empty=True))
    worker(args,out)
    write(out/'complete.json',dict(status='ARTIFACTS_COMPLETE',TX_labels_read=0,target_inputs=0,optimizer_updates=0))


if __name__=='__main__':
    main()
