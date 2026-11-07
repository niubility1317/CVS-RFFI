"""Frozen source V diagnostic of observation-space channel phase calibration."""
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
from .phase_ramp_posterior import calibrated_action, fixed_grid, ramp_posteriors
from .source_inverse import R07, SCENES
from .source_pair import read, validate_source
from .structured_noise_source import known_parameters, load_reference

RUN = '20261107-diagnostic-Restore-observed-phase-ramp-sourceV-s2026092701-r01'
MODULE = 'experiments.pamf_phase3_restoration.phase_ramp_source'
METHODS = ('blind_scalar','phase_only','phase_and_drift','known_scalar')


def readouts(G,trace,grid):
    received = trace['received']
    observed = received.flatten(1).double()
    mean,covariance = G.prior_mean.double(),G.prior_covariance.double()
    estimate = G.estimate(received)
    action = estimate['action_matrix']
    outputs,gamma = ramp_posteriors(action,observed,mean,covariance,estimate['variance'].double(),*grid)
    # The known simulator operator enters only the separate control and the frozen error metric.
    kernel,main,image,variance = known_parameters(trace)
    known = dense_action(kernel,main,image)
    oracle,residual = gaussian_posterior(known,observed,mean,covariance,variance)
    if not torch.isfinite(outputs).all() or not torch.isfinite(gamma).all() or residual.max()>1e-8:
        raise ValueError('Invalid phase posterior or known scalar control')
    energy = known.square().sum((1,2)).clamp_min(1e-8)
    operators = (action,calibrated_action(action,gamma[:,0]),calibrated_action(action,gamma[:,1]),known)
    signal = (*outputs.unbind(1),oracle)
    views = {method:dict(iq=iq.float().reshape_as(received),
        action_relative_mse=(matrix-known).square().sum((1,2))/energy)
        for method,iq,matrix in zip(METHODS,signal,operators)}
    return views,gamma


def worker(args,out):
    start = time.perf_counter()
    G = load_model(args.root,'cpu')
    F,identity = load_F0(validate_source(read(args.inputs)),'cpu')
    valid_identity(identity)
    models = (G,F)
    snapshots = [{name:value.detach().clone() for name,value in model.state_dict().items()} for model in models]
    cache = args.root/'runs'/R07/'cache'
    source2 = np.flatnonzero(np.load(cache/'train/roles.npy',allow_pickle=False)=='L_s')[:2]
    if len(source2)!=2:
        raise ValueError('Original source L_s smoke required')
    grid = fixed_grid('cpu')
    with torch.inference_mode():
        trace = read_trace(cache/'epochs/01',source2,63000)
        values,_ = readouts(G,trace,grid)
        torch.testing.assert_close(values['blind_scalar']['iq'],G(trace['received']),rtol=2e-6,atol=2e-7)
        for fields in values.values():
            logits = F.classify_features(F.features(fields['iq']))
            if logits.shape!=(2,6) or not torch.isfinite(logits).all():
                raise ValueError('Actual original F0 CPU smoke failed')
    write(out/'smoke.json',dict(status='PASS',actual_Ls=2,actual_F0=True,actual_original_BF=True,
        scalar_matches_original_BF=True,shared_factor_with_fixed_quadrature=True,methods=list(METHODS),
        F0_rows=8,estimator_calls=2,CPU_only=True,TX_labels_read=0,target_inputs=0))
    print('SOURCE_PHASE_RAMP_REAL_SMOKE_PASS',flush=True)
    ref,take = load_reference(args.root,identity,cache)
    with (out/'reference.npz').open('xb') as f:
        np.savez(f,**ref,source_index=take)
    clean,target,clean_logits,scale = [tensor(ref[k],'cpu') for k in ['clean','features','logits','scale']]
    summary = []
    with torch.inference_mode():
        for scene in SCENES:
            trace = read_trace(cache/'validation_actions'/scene,take,1800)
            records = {}
            for offset in range(0,len(take),16):
                view = slice(offset,offset+16)
                outputs,gamma = readouts(G,{key:value[view] for key,value in trace.items()},grid)
                values = dict(calibration_real=gamma.real,calibration_imag=gamma.imag)
                for name,fields in outputs.items():
                    iq = fields['iq']
                    features = F.features(iq)
                    logits = F.classify_features(features)
                    metrics = source_metrics(iq,clean[view],features,logits,target[view],clean_logits[view],scale)
                    values.update({name+'__'+key:value for key,value in dict(fields,**metrics,features=features,logits=logits).items()})
                if not all(torch.isfinite(value).all() for value in values.values()):
                    raise ValueError('Nonfinite source posterior or F0 metrics')
                for key,value in values.items():
                    records.setdefault(key,[]).extend(value.tolist())
            arrays = {key:np.asarray(value) for key,value in records.items()}
            with (out/(scene+'.npz')).open('xb') as f:
                np.savez(f,**arrays)
            before = arrays['blind_scalar__clean_decision_agreement'].astype(bool)
            for name in METHODS:
                after = arrays[name+'__clean_decision_agreement'].astype(bool)
                summary.append(dict(scene=scene,method=name,count=len(take),Fclean_agreement=float(after.mean()),
                    vs_blind_repairs=int((~before & after).sum()),vs_blind_damage=int((before & ~after).sum()),
                    iq_nmse=float(arrays[name+'__iq_nmse'].mean()),feature_cosine_distance=float(arrays[name+'__feature_cosine_distance'].mean()),
                    action_relative_mse=float(arrays[name+'__action_relative_mse'].mean())))
            print('SOURCE_PHASE_RAMP_VIEW_COMPLETE',scene,len(take),flush=True)
    for model,before in zip(models,snapshots):
        check_F0_state(model,before)
    write(out/'results.json',dict(status='SOURCE_PHASE_RAMP_COMPLETE',run_id=RUN,code_commit=args.commit,
        frozen_BF_code=G_COMMIT,reference_code=REFERENCE_COMMIT,methods=list(METHODS),views=list(SCENES),
        phase_points=256,drift_points=9,drift_total_range_degrees=[-45,45],prior_phase_std_degrees=15,prior_drift_std_degrees=15,
        source_V_records=120,source_V_views=840,F0_new_rows=3360,F0_smoke_rows=8,F0_clean_rows_reused=120,
        estimator_rows=840,estimator_smoke_rows=4,records=summary,CPU_threads=2,
        TX_labels_read=0,OE_reads=0,novel_reads=0,target_inputs=0,optimizer_updates=0,model_state_updates=0,
        new_deployment_candidate=False,theoretical_ceiling=False,seconds=time.perf_counter()-start))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--inputs',type=Path,required=True)
    parser.add_argument('--commit',required=True)
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or (Path.cwd()/'release_commit.txt').read_text().strip()!=args.commit:
        raise ValueError('CUDA-empty source diagnostic with actual code identity required')
    torch.set_num_threads(2)
    out = args.root/'runs'/RUN
    out.mkdir()
    write(out/'launch.json',dict(pid=os.getpid(),code_commit=args.commit,CPU_threads=2,CUDA_empty=True))
    worker(args,out)
    write(out/'complete.json',dict(status='ARTIFACTS_COMPLETE',TX_labels_read=0,target_inputs=0,optimizer_updates=0))


if __name__=='__main__':
    main()
