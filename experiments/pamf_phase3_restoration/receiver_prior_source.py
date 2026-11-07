"""Fixed source receiver-prior diagnostic with known and estimated channels."""

import argparse
import os
from pathlib import Path
import time

import numpy as np
import torch

from .cache import check_F0_state, load_F0, tensor
from .cost_probe import write
from .covariance_inverse import dense_action, gaussian_posterior
from .covariance_parameter_source import load_model
from .main_split_eval import valid_identity
from .phase_mixture_source import REFERENCE_RUN, read_trace
from .preamble_component_source import logits
from .posterior_bias_noise_source import load_control
from .source_action_factor_probe import RUN as KNOWN_RUN, references
from .source_inverse import R07, SCENES
from .source_pair import read, validate_source
from .source_receiver_priors import fit_prior, select_receivers
from .structured_noise_source import known_parameters


RUN = '20261107-diagnostic-Restore-receiver-prior-sourceV-s2026092701-r01'
METHODS = ('full_blind', 'same_blind', 'other_blind', 'full_known', 'same_known', 'other_known')


def recover(action, variance, observed, mean, covariance):
    value, residual = gaussian_posterior(action, observed.flatten(1).double(), mean.double(), covariance.double(), variance.double())
    if residual.max() > 1e-8:
        raise ValueError('Source posterior solve exceeded the numerical tolerance')
    return value.float().reshape_as(observed)


def worker(args, out):
    started = time.perf_counter()
    model = load_model(args.root, 'cpu')
    reader, identity = load_F0(validate_source(read(args.inputs)), 'cpu')
    valid_identity(identity)
    frozen = [{k:v.detach().clone() for k,v in m.state_dict().items()} for m in (model, reader)]
    cache = args.root/'runs'/R07/'cache'
    source_rx, source_day, roles = [np.load(cache/'train'/(key+'.npy'),allow_pickle=False) for key in ('rx','day','roles')]
    fitting = select_receivers(source_rx, source_day, roles)
    a = np.load(cache/'train/clean.npy',mmap_mode='r',allow_pickle=False)
    if a.shape != (63000,2,256) or a.dtype != np.float32:
        raise ValueError('Original 63000 source training IQ required')
    priors = {}
    for rx, selected in fitting.items():
        priors[rx] = [tuple(tensor(value,'cpu') for value in fit_prior(a,index)) for index in selected]
    prior_data = {f'{rx}__{name}__{key}':np.asarray(value.tolist(),np.float32) for rx, pairs in priors.items()
                  for name, pair in zip(('same','other'),pairs) for key,value in zip(('mean','covariance'),pair)}
    with (out/'fitted_priors.npz').open('xb') as file:
        np.savez(file,**prior_data)
    source = np.flatnonzero(roles=='L_s')[:2]
    sample = np.load(cache/'epochs/01/received.npy',mmap_mode='r',allow_pickle=False)
    with torch.inference_mode():
        observed = tensor(sample[source],'cpu')
        logits(reader, model(observed))
        estimate = model.estimate(observed)
        for rx in priors:
            for mean,covariance in priors[rx]:
                logits(reader,recover(estimate['action_matrix'],estimate['variance'],observed,mean,covariance))
    write(out/'smoke.json',dict(status='PASS',real_L_s=2,all_ten_fitted_priors=True,actual_BF=True,actual_F0=True,target_inputs=0))
    print('RECEIVER_PRIOR_REAL_SMOKE_PASS',flush=True)
    ref,take = references(args.root,cache,identity)
    known_folder = args.root/'runs'/KNOWN_RUN
    load_control(known_folder,ref,take)
    clean = tensor(ref['clean'],'cpu')
    with (out/'reference.npz').open('xb') as file:
        np.savez(file,**{key:ref[key] for key in ('ids','rx','day','logits')},source_index=take)
    old = args.root/'runs'/REFERENCE_RUN
    with torch.inference_mode():
        for scene in SCENES:
            trace = read_trace(cache/'validation_actions'/scene,take,1800)
            with np.load(old/(scene+'.npz'),allow_pickle=False) as physical, np.load(known_folder/(scene+'.npz'),allow_pickle=False) as known:
                if not np.array_equal(physical['ids'][take],ref['ids']):
                    raise ValueError('Exact prior comparison reference required')
                result = dict(ids=ref['ids'],full_blind__prediction=physical['physical__prediction'][take],
                    full_blind__nmse=physical['physical__iq_nmse'][take],
                    full_known__prediction=known['true_action_noise__prediction'],
                    full_known__nmse=known['true_action_noise__iq_nmse'])
            for method in ('same_blind','other_blind','same_known','other_known'):
                result[method+'__prediction'] = np.empty(len(take),np.int64)
                result[method+'__nmse'] = np.empty(len(take),np.float64)
            for rx in sorted(priors):
                indices = np.flatnonzero(ref['rx']==rx)
                for start in range(0,len(indices),32):
                    index = indices[start:start+32]
                    v = {key:value[index] for key,value in trace.items()}
                    estimate = model.estimate(v['received'])
                    h,m,i,noise = known_parameters(v)
                    true_action = dense_action(h,m,i)
                    energy = clean[index].square().mean((1,2)).clamp_min(1e-4)
                    for name,(mean,covariance) in zip(('same','other'),priors[rx]):
                        for condition,action,var in (('blind',estimate['action_matrix'],estimate['variance']),('known',true_action,noise)):
                            output = recover(action,var,v['received'],mean,covariance)
                            method = name+'_'+condition
                            result[method+'__prediction'][index] = logits(reader,output).argmax(1).tolist()
                            result[method+'__nmse'][index] = ((output-clean[index]).square().mean((1,2))/energy).tolist()
            with (out/(scene+'.npz')).open('xb') as file:
                np.savez(file,**result)
            print('RECEIVER_PRIOR_VIEW_COMPLETE',scene,len(take),flush=True)
    for m,before in zip((model,reader),frozen):
        check_F0_state(m,before)
    write(out/'results.json',dict(status='SOURCE_RECEIVER_PRIOR_PREDICTIONS_COMPLETE',code_commit=args.commit,F0_identity=identity,
        source_V_records=480,views=list(SCENES),methods=list(METHODS),prior_samples=12600,prior_L_s=1260,prior_U_s=11340,
        prior_fit_roles=['L_s','U_s'],prior_fit_seed=2026110707,known_receiver_used_for_diagnostic_prior=True,
        IQ_posterior_new_rows=13440,F0_new_rows=13440,TX_labels_read=0,V_fit_rows=0,target_inputs=0,
        optimizer_updates=0,new_deployment_candidate=False,seconds=time.perf_counter()-started))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--inputs',type=Path,required=True)
    parser.add_argument('--commit',required=True)
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or (Path.cwd()/'release_commit.txt').read_text().strip()!=args.commit:
        raise ValueError('CPU source diagnostic and exact release required')
    os.nice(10)
    torch.set_num_threads(2)
    out = args.root/'runs'/RUN
    out.mkdir()
    write(out/'launch.json',dict(pid=os.getpid(),code_commit=args.commit,CPU_threads=2,CUDA_empty=True))
    worker(args,out)
    write(out/'complete.json',dict(status='ARTIFACTS_COMPLETE',TX_labels_read=0,target_inputs=0))


if __name__=='__main__':
    main()
