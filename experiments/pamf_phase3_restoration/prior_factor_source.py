"""Cross source receiver-prior means and covariances with fixed channels."""

import argparse
import os
from pathlib import Path
import time

import numpy as np
import torch

from .cache import check_F0_state, load_F0, tensor
from .cost_probe import write
from .covariance_inverse import dense_action
from .covariance_parameter_source import load_model
from .main_split_eval import valid_identity
from .mismatched_prior_training import RECEIVERS, STATISTICS_CODE, STATISTICS_RUN
from .phase_mixture_source import read_trace
from .preamble_component_source import logits
from .receiver_prior_source import recover
from .source_action_factor_probe import references
from .source_inverse import R07, SCENES
from .source_pair import read, validate_source
from .structured_noise_source import known_parameters


RUN = '20261107-diagnostic-Restore-prior-factor-sourceV-s2026092701-r01'
FACTORS = ('same','mean_other','covariance_other','other')
METHODS = tuple(f'{factor}_{condition}' for condition in ('blind','known') for factor in FACTORS)


def cross_priors(same,other):
    mean_s,cov_s = same
    mean_o,cov_o = other
    return {'same':(mean_s,cov_s),'mean_other':(mean_o,cov_s),
            'covariance_other':(mean_s,cov_o),'other':(mean_o,cov_o)}


def load_controls(root,identity):
    folder = root/'runs'/STATISTICS_RUN
    meta = read(folder/'results.json')
    if (read(folder/'complete.json')['status']!='ARTIFACTS_COMPLETE'
            or meta['status']!='SOURCE_RECEIVER_PRIOR_PREDICTIONS_COMPLETE'
            or meta['code_commit']!=STATISTICS_CODE or meta['F0_identity']!=identity
            or meta['prior_fit_roles']!=['L_s','U_s'] or meta['V_fit_rows']!=0
            or meta['TX_labels_read']!=0 or meta['target_inputs']!=0):
        raise ValueError('Original complete source receiver prior matrix required')
    with np.load(folder/'fitted_priors.npz',allow_pickle=False) as file:
        priors = {rx:cross_priors(*[tuple(tensor(file[f'{rx}__{arm}__{key}'],'cpu') for key in ('mean','covariance'))
                                   for arm in ('same','other')]) for rx in RECEIVERS}
    return folder,priors


def worker(args,out):
    started = time.perf_counter()
    model = load_model(args.root,'cpu')
    reader,identity = load_F0(validate_source(read(args.inputs)),'cpu')
    valid_identity(identity)
    folder,priors = load_controls(args.root,identity)
    snapshots = [{key:value.detach().clone() for key,value in m.state_dict().items()} for m in (model,reader)]
    cache = args.root/'runs'/R07/'cache'
    source = np.flatnonzero(np.load(cache/'train/roles.npy',allow_pickle=False)=='L_s')[:2]
    if len(source)!=2:
        raise ValueError('Original L_s real-source check required')
    array = np.load(cache/'epochs/01/received.npy',mmap_mode='r',allow_pickle=False)
    with torch.inference_mode():
        sample = tensor(array[source],'cpu')
        logits(reader,model(sample))
        estimate = model.estimate(sample)
        for factor in ('mean_other','covariance_other'):
            logits(reader,recover(estimate['action_matrix'],estimate['variance'],sample,*priors[RECEIVERS[0]][factor]))
    write(out/'smoke.json',dict(status='PASS',real_L_s=2,original_BF=True,original_F0=True,
        both_crossed_priors=True,target_inputs=0))
    print('PRIOR_FACTOR_REAL_SMOKE_PASS',flush=True)
    ref,take = references(args.root,cache,identity)
    with np.load(folder/'reference.npz',allow_pickle=False) as file:
        if (not np.array_equal(file['source_index'],take)
                or not all(np.array_equal(file[key],ref[key]) for key in ('ids','rx','day','logits'))):
            raise ValueError('Original source prior panel differs')
    clean = tensor(ref['clean'],'cpu')
    with (out/'reference.npz').open('xb') as file:
        np.savez(file,**{key:ref[key] for key in ('ids','rx','day','logits')},source_index=take)
    with torch.inference_mode():
        for scene in SCENES:
            trace = read_trace(cache/'validation_actions'/scene,take,1800)
            with np.load(folder/(scene+'.npz'),allow_pickle=False) as old:
                if not np.array_equal(old['ids'],ref['ids']):
                    raise ValueError('Original prior endpoint IDs differ')
                result = dict(ids=ref['ids'])
                for condition in ('blind','known'):
                    for factor in ('same','other'):
                        for key in ('prediction','nmse'):
                            result[f'{factor}_{condition}__{key}'] = old[f'{factor}_{condition}__{key}']
            for condition in ('blind','known'):
                for factor in ('mean_other','covariance_other'):
                    result[f'{factor}_{condition}__prediction'] = np.empty(len(take),np.int64)
                    result[f'{factor}_{condition}__nmse'] = np.empty(len(take),np.float64)
            for rx in RECEIVERS:
                indices = np.flatnonzero(ref['rx']==rx).tolist()
                for start in range(0,len(indices),32):
                    index = indices[start:start+32]
                    values = {key:value[index] for key,value in trace.items()}
                    estimate = model.estimate(values['received'])
                    h,m,i,noise = known_parameters(values)
                    true_action = dense_action(h,m,i)
                    energy = clean[index].square().mean((1,2)).clamp_min(1e-4)
                    for factor in ('mean_other','covariance_other'):
                        mean,covariance = priors[rx][factor]
                        for condition,action,var in (('blind',estimate['action_matrix'],estimate['variance']),('known',true_action,noise)):
                            output = recover(action,var,values['received'],mean,covariance)
                            method = factor+'_'+condition
                            result[method+'__prediction'][index] = logits(reader,output).argmax(1).tolist()
                            result[method+'__nmse'][index] = ((output-clean[index]).square().mean((1,2))/energy).tolist()
            with (out/(scene+'.npz')).open('xb') as file:
                np.savez(file,**result)
            print('PRIOR_FACTOR_VIEW_COMPLETE',scene,len(take),flush=True)
    for m,before in zip((model,reader),snapshots):
        check_F0_state(m,before)
    write(out/'results.json',dict(status='SOURCE_PRIOR_FACTOR_PREDICTIONS_COMPLETE',code_commit=args.commit,
        F0_identity=identity,methods=list(METHODS),views=list(SCENES),source_V_records=480,
        same_fitted_source_moments=STATISTICS_RUN,known_source_RX_for_diagnostic=True,
        F0_new_rows=13440,reused_endpoint_rows=13440,TX_labels_read=0,target_inputs=0,
        optimizer_updates=0,new_deployment_candidate=False,seconds=time.perf_counter()-started))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--inputs',type=Path,required=True)
    parser.add_argument('--commit',required=True)
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or (Path.cwd()/'release_commit.txt').read_text().strip()!=args.commit:
        raise ValueError('CPU source diagnostic and exact code commit required')
    os.nice(10)
    torch.set_num_threads(2)
    out = args.root/'runs'/RUN
    out.mkdir()
    write(out/'launch.json',dict(pid=os.getpid(),code_commit=args.commit,CPU_threads=2,CUDA_empty=True))
    worker(args,out)
    write(out/'complete.json',dict(status='ARTIFACTS_COMPLETE',TX_labels_read=0,target_inputs=0))


if __name__=='__main__':
    main()
