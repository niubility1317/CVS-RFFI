"""Source-only factorial check of joint phase and waveform prior inference."""

import argparse
import os
from pathlib import Path
import time

import numpy as np
import torch

from .cache import check_F0_state, load_F0, tensor
from .cost_probe import write
from .covariance_guidance_source import source_metrics
from .covariance_parameter_source import G_COMMIT, load_model
from .joint_phase_prior import joint_phase_mean
from .main_split_eval import valid_identity
from .phase_mixture_source import REFERENCE_COMMIT, read_trace
from .phase_ramp_posterior import fixed_grid
from .phase_ramp_source import RUN as PHASE_RUN
from .source_inverse import R07, SCENES
from .source_pair import read, validate_source
from .structured_noise_source import load_reference
from .waveform_mixture_prior import mixture_posterior
from .waveform_mixture_source import RUN as WAVEFORM_RUN

RUN = '20261107-diagnostic-Restore-joint-phase-waveform-sourceV-s2026092701-r01'
MODULE = 'experiments.pamf_phase3_restoration.joint_phase_source'
WAVEFORM_COMMIT = 'f4dab0140b598bbc3cd212ab4259138e468e1a20'
PHASE_COMMIT = 'e97058e837ea494b2efeb2fd0da5a237032e7736'
METHODS = ('blind_gaussian', 'blind_mixture', 'phase_only', 'joint_phase_mixture')


def prior_path(folder):
    result = read(folder/'results.json')
    if (read(folder/'complete.json')['status'] != 'ARTIFACTS_COMPLETE'
            or result['status'] != 'SOURCE_WAVEFORM_MIXTURE_COMPLETE' or result['code_commit'] != WAVEFORM_COMMIT
            or any(result[key] for key in ('TX_labels_read','target_inputs','OE_reads','novel_reads','optimizer_updates'))
            or (result['source_prior']['source_Ls'],result['source_prior']['source_Us']) != (6300,56700)):
        raise ValueError('Original IQ-only moment-matched prior is required')
    return folder/'waveform_prior.npz'


def load_prior(folder, G):
    with np.load(prior_path(folder), allow_pickle=False) as f:
        within, offsets, weights = (f[key].copy() for key in ('within','offsets','weights'))
    if (within.shape, offsets.shape, weights.shape) != ((512,512),(4,512),(4,)):
        raise ValueError('Exact four-mode IQ prior required')
    if not all(np.isfinite(a).all() for a in (within,offsets,weights)) or (weights <= 0).any():
        raise ValueError('Finite source prior with positive weights required')
    within, offsets, weights = (torch.frombuffer(memoryview(a),dtype=torch.float64).reshape(a.shape)
                                for a in (within,offsets,weights))
    torch.testing.assert_close(weights.sum(),torch.tensor(1.,dtype=torch.float64),atol=1e-12,rtol=0)
    torch.testing.assert_close(weights@offsets,torch.zeros(512,dtype=torch.float64),atol=1e-12,rtol=0)
    torch.testing.assert_close(within + (offsets.T*weights)@offsets,G.prior_covariance.double(),atol=1e-12,rtol=0)
    return within,offsets,weights


def prior_controls(root, identity, ref, take):
    wave, phase = (root/'runs'/name for name in (WAVEFORM_RUN, PHASE_RUN))
    prior_path(wave)
    for folder, commit, status in [(wave,WAVEFORM_COMMIT,'SOURCE_WAVEFORM_MIXTURE_COMPLETE'),
                                   (phase,PHASE_COMMIT,'SOURCE_PHASE_RAMP_COMPLETE')]:
        done = read(folder/'results.json')
        if ((done['status'],done['code_commit'],done['source_V_records'],done['TX_labels_read'],done['target_inputs'])
                != (status,commit,120,0,0) or read(folder/'complete.json')['status'] != 'ARTIFACTS_COMPLETE'):
            raise ValueError('Completed original frozen source predictions required')
        if 'F0_identity' in done and done['F0_identity'] != identity:
            raise ValueError('Exact original F0 control required')
        with np.load(folder/'reference.npz',allow_pickle=False) as saved:
            if not np.array_equal(saved['source_index'],take) or not all(np.array_equal(saved[k],v) for k,v in ref.items()):
                raise ValueError('Source control physical order, waveform or clean F0 differs')
    return wave,phase


def readout(G, received, prior, phase, log_prior):
    params = G.estimate(received)
    values = joint_phase_mean(params['action_matrix'],received.flatten(1).double(),G.prior_mean.double(),
        *prior,params['variance'].double(),phase,log_prior)
    iq,mode_prob,moments = values
    if not all(torch.isfinite(value).all() for value in values):
        raise ValueError('Nonfinite joint phase and waveform posterior')
    torch.testing.assert_close(mode_prob.sum(1),torch.ones_like(mode_prob[:,0]),rtol=0,atol=1e-12)
    return iq.float().reshape_as(received), mode_prob, moments


def worker(args,out):
    start = time.perf_counter()
    G = load_model(args.root,'cpu')
    prior = load_prior(args.root/'runs'/WAVEFORM_RUN,G)
    F,identity = load_F0(validate_source(read(args.inputs)),'cpu')
    valid_identity(identity)
    models = G,F
    snapshots = [{k:v.detach().clone() for k,v in model.state_dict().items()} for model in models]
    cache = args.root/'runs'/R07/'cache'
    source2 = np.flatnonzero(np.load(cache/'train/roles.npy',allow_pickle=False)=='L_s')[:2]
    if len(source2) != 2:
        raise ValueError('Original source L_s2 smoke required')
    phase,_,log_prior = fixed_grid('cpu')
    with torch.inference_mode():
        received = read_trace(cache/'epochs/01',source2,63000)['received']
        restored,_,_ = readout(G,received,prior,phase,log_prior[4])
        params = G.estimate(received)
        inputs = params['action_matrix'],received.flatten(1).double(),G.prior_mean.double(),*prior,params['variance'].double()
        zero = phase[128:129]
        degenerate,probabilities,_ = joint_phase_mean(*inputs,zero,zero)
        expected,weights = mixture_posterior(*inputs)
        torch.testing.assert_close(degenerate,expected,atol=1e-10,rtol=1e-9)
        torch.testing.assert_close(probabilities,weights,atol=1e-10,rtol=1e-9)
        logits = F.classify_features(F.features(restored))
        if logits.shape != (2,6) or not torch.isfinite(logits).all():
            raise ValueError('Actual source L_s2 original F0 smoke failed')
    write(out/'smoke.json',dict(status='PASS',actual_Ls=2,actual_F0=True,actual_original_BF=True,
        zero_phase_equals_original_mixture=True,source_F0_rows=2,CPU_only=True,TX_labels_read=0,target_inputs=0))
    print('SOURCE_JOINT_PHASE_REAL_SMOKE_PASS',flush=True)

    ref,take = load_reference(args.root,identity,cache)
    wave,phase_parent = prior_controls(args.root,identity,ref,take)
    with (out/'reference.npz').open('xb') as file:
        np.savez(file,**ref,source_index=take)
    clean,target,clean_logits,scale = [tensor(ref[k],'cpu') for k in ('clean','features','logits','scale')]
    records = []
    with torch.inference_mode():
        for scene in SCENES:
            received = read_trace(cache/'validation_actions'/scene,take,1800)['received']
            new = {}
            for start_row in range(0,len(take),16):
                view = slice(start_row,start_row+16)
                iq,prob,moments = readout(G,received[view],prior,phase,log_prior[4])
                features = F.features(iq)
                logits = F.classify_features(features)
                values = source_metrics(iq,clean[view],features,logits,target[view],clean_logits[view],scale)
                for key,value in dict(values,iq=iq,logits=logits,probabilities=prob,phase_moments=moments).items():
                    if not torch.isfinite(value).all():
                        raise ValueError('Nonfinite source joint phase readout')
                    new.setdefault('joint_phase_mixture__'+key,[]).extend(value.tolist())
            arrays = {key:np.asarray(value) for key,value in new.items()}
            with np.load(wave/(scene+'.npz'),allow_pickle=False) as wave_view, np.load(phase_parent/(scene+'.npz'),allow_pickle=False) as phase_view:
                if not np.array_equal(wave_view['blind_gaussian__prediction'],phase_view['blind_scalar__prediction']):
                    raise ValueError('Original scalar baseline source decisions differ')
                for name,parent in [('blind_gaussian',wave_view),('blind_mixture',wave_view),('phase_only',phase_view)]:
                    for key in ('logits','prediction','iq_nmse','feature_cosine_distance'):
                        field = name+'__'+key
                        arrays[field] = parent[field]
            with (out/(scene+'.npz')).open('xb') as file:
                np.savez(file,**arrays)
            correct = {name:arrays[name+'__prediction']==ref['logits'].argmax(1) for name in METHODS}
            before = correct[METHODS[0]]
            for name in METHODS:
                after = correct[name]
                records.append(dict(scene=scene,method=name,count=len(take),Fclean_correct=int(after.sum()),
                    repairs=int((~before&after).sum()),damage=int((before&~after).sum()),
                    iq_nmse=float(arrays[name+'__iq_nmse'].mean()),
                    feature_cosine_distance=float(arrays[name+'__feature_cosine_distance'].mean())))
            print('SOURCE_JOINT_PHASE_VIEW_COMPLETE',scene,len(take),flush=True)
    for model,before in zip(models,snapshots):
        check_F0_state(model,before)
    write(out/'results.json',dict(status='SOURCE_JOINT_PHASE_COMPLETE',run_id=RUN,code_commit=args.commit,
        F0_identity=identity,frozen_BF_code=G_COMMIT,reference_code=REFERENCE_COMMIT,
        waveform_code=WAVEFORM_COMMIT,phase_code=PHASE_COMMIT,methods=list(METHODS),views=list(SCENES),
        phase_points=256,prior_phase_std_degrees=15,waveform_modes=4,records=records,
        source_V_records=120,source_V_views=840,F0_new_rows=840,F0_smoke_rows=2,F0_control_rows_reused=2520,
        source_prior_refit=False,TX_labels_read=0,OE_reads=0,novel_reads=0,target_inputs=0,
        optimizer_updates=0,model_state_updates=0,CPU_threads=2,new_deployment_candidate=False,
        theoretical_ceiling=False,seconds=time.perf_counter()-start))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--inputs',type=Path,required=True)
    parser.add_argument('--commit',required=True)
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or (Path.cwd()/'release_commit.txt').read_text().strip() != args.commit:
        raise ValueError('CUDA-empty source diagnostic with actual code identity required')
    torch.set_num_threads(2)
    out = args.root/'runs'/RUN
    out.mkdir()
    write(out/'launch.json',dict(pid=os.getpid(),code_commit=args.commit,CPU_threads=2,CUDA_empty=True))
    worker(args,out)
    write(out/'complete.json',dict(status='ARTIFACTS_COMPLETE',TX_labels_read=0,target_inputs=0,optimizer_updates=0))


if __name__ == '__main__':
    main()
