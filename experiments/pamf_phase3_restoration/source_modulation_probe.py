"""Isolate phase, magnitude and image terms in frozen source modulation recovery."""

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
from .phase_mixture_source import REFERENCE_COMMIT, read_trace
from .source_action_factor_probe import RUN as CONTROL_RUN, references
from .source_inverse import R07, SCENES
from .source_pair import read, validate_source
from .structured_noise_source import known_parameters

RUN = '20261107-diagnostic-Restore-modulation-components-sourceV-s2026092701-r01'
MODULE = 'experiments.pamf_phase3_restoration.source_modulation_probe'
CONTROL_COMMIT = '1065b7fd515402862a8b47bca8bddf597d823b54'
NEW_METHODS = ('main_phase','main_complex_gain','true_main','true_image')
METHODS = ('blind',*NEW_METHODS,'true_modulation')


def modulation_controls(aligned, known):
    h, main, image = aligned
    _, true_main, true_image = known
    energy = main.abs().square().sum(1)
    if (energy <= 0).any():
        raise ValueError('Estimated main path must carry energy')
    gain = (main.conj()*true_main).sum(1)/energy
    phase = torch.polar(torch.ones_like(gain.real),gain.angle())
    return dict(main_phase=(h,main*phase[:,None],image),
        main_complex_gain=(h,main*gain[:,None],image), true_main=(h,true_main,image),
        true_image=(h,main,true_image)), gain


def load_controls(folder, ref, take):
    result = read(folder/'results.json')
    if ((result['status'],result['code_commit'],result['source_V_records'],result['TX_labels_read'],result['target_inputs'])
            != ('SOURCE_ACTION_FACTOR_COMPLETE',CONTROL_COMMIT,480,0,0)
            or read(folder/'complete.json')['status'] != 'ARTIFACTS_COMPLETE'):
        raise ValueError('Exact frozen source factor controls required')
    with np.load(folder/'reference.npz',allow_pickle=False) as file:
        if not np.array_equal(file['source_index'],take) or not all(
                np.array_equal(file[key],ref[key]) for key in ('ids','rx','day','logits')):
            raise ValueError('Original source controls have different records or F0 outputs')


def readouts(G, trace, verify_gauge=False):
    received = trace['received']
    params = G.estimate(received,dense=False)
    native = tuple(params[k].to(torch.complex128) for k in ('kernel','main','image'))
    kernel, main, image, _ = known_parameters(trace)
    aligned, known, correlation = aligned_factors(native,(kernel,main,image))
    controls, gain = modulation_controls(aligned,known)
    known_action = dense_action(*known)
    if verify_gauge:
        torch.testing.assert_close(known_action,dense_action(kernel,main,image),rtol=1e-11,atol=1e-12)
        torch.testing.assert_close(dense_action(*aligned),dense_action(*native),rtol=1e-11,atol=1e-12)
    known_energy = known_action.square().sum((1,2)).clamp_min(1e-8)
    true_main_energy = known[1].abs().square().sum(1).clamp_min(1e-8)
    evidence = dict(main_phase_offset_radians=gain.angle(),main_gain_magnitude=gain.abs(),
        main_relative_fit_residual=(controls['main_complex_gain'][1]-known[1]).abs().square().sum(1)/true_main_energy,
        true_image_relative_energy=known[2].abs().square().sum(1)/true_main_energy,FIR_gauge_correlation=correlation)
    mean,covariance = G.prior_mean.double(),G.prior_covariance.double()
    observed,variance = received.flatten(1).double(),params['variance'].double()
    outputs = {}
    for name, factors in controls.items():
        action = dense_action(*factors)
        iq,residual = gaussian_posterior(action,observed,mean,covariance,variance)
        if residual.max() > 1e-8:
            raise ValueError('Source posterior solve residual exceeds 1e-8')
        outputs[name] = dict(iq=iq.float().reshape_as(received),
            action_relative_mse=(action-known_action).square().sum((1,2))/known_energy)
    return outputs, evidence


def worker(args,out):
    start = time.perf_counter()
    G = load_model(args.root,'cpu')
    F, identity = load_F0(validate_source(read(args.inputs)),'cpu')
    valid_identity(identity)
    models = G,F
    snapshots = [{k:v.detach().clone() for k,v in model.state_dict().items()} for model in models]
    cache = args.root/'runs'/R07/'cache'
    source2 = np.flatnonzero(np.load(cache/'train/roles.npy',allow_pickle=False)=='L_s')[:2]
    if len(source2) != 2:
        raise ValueError('Original source L_s2 smoke required')
    with torch.inference_mode():
        trace = read_trace(cache/'epochs/01',source2,63000)
        outputs,evidence = readouts(G,trace,verify_gauge=True)
        for fields in outputs.values():
            logits = F.classify_features(F.features(fields['iq']))
            if logits.shape != (2,6) or not torch.isfinite(logits).all():
                raise ValueError('Actual source L_s2 F0 smoke failed')
        if not all(torch.isfinite(x).all() for x in evidence.values()):
            raise ValueError('Nonfinite source modulation evidence')
    write(out/'smoke.json',dict(status='PASS',actual_Ls=2,actual_F0=True,actual_original_BF=True,
        gauge_preserves_native_operators=True,F0_rows=8,CPU_only=True,TX_labels_read=0,target_inputs=0))
    print('SOURCE_MODULATION_COMPONENTS_REAL_SMOKE_PASS',flush=True)

    ref,take = references(args.root,cache,identity)
    control = args.root/'runs'/CONTROL_RUN
    load_controls(control,ref,take)
    with (out/'reference.npz').open('xb') as file:
        np.savez(file,**ref,source_index=take)
    clean,target,clean_logits,scale = [tensor(ref[k],'cpu') for k in ('clean','features','logits','scale')]
    summaries = []
    with torch.inference_mode():
        for scene in SCENES:
            trace = read_trace(cache/'validation_actions'/scene,take,1800)
            records = {}
            for start_row in range(0,len(take),16):
                view = slice(start_row,start_row+16)
                outputs,values = readouts(G,{key:value[view] for key,value in trace.items()})
                for name,fields in outputs.items():
                    iq = fields['iq']
                    features = F.features(iq)
                    logits = F.classify_features(features)
                    metrics = source_metrics(iq,clean[view],features,logits,target[view],clean_logits[view],scale)
                    values.update({name+'__'+key:value for key,value in dict(metrics,logits=logits,
                        action_relative_mse=fields['action_relative_mse']).items()})
                if not all(torch.isfinite(value).all() for value in values.values()):
                    raise ValueError('Nonfinite source modulation readout')
                for key,value in values.items():
                    records.setdefault(key,[]).extend(value.tolist())
            arrays = {key:np.asarray(value) for key,value in records.items()}
            with np.load(control/(scene+'.npz'),allow_pickle=False) as file:
                np.testing.assert_allclose(arrays['FIR_gauge_correlation'],file['FIR_gauge_correlation'],rtol=1e-11,atol=1e-12)
                for name in ('blind','true_modulation'):
                    for key in ('logits','prediction','iq_nmse','feature_cosine_distance','action_relative_mse'):
                        field = name+'__'+key
                        arrays[field] = file[field]
            with (out/(scene+'.npz')).open('xb') as file:
                np.savez(file,**arrays)
            correct = {name:arrays[name+'__prediction']==ref['logits'].argmax(1) for name in METHODS}
            before = correct['blind']
            for name in METHODS:
                after = correct[name]
                summaries.append(dict(scene=scene,method=name,count=len(take),Fclean_correct=int(after.sum()),
                    repairs=int((~before&after).sum()),damage=int((before&~after).sum()),
                    iq_nmse=float(arrays[name+'__iq_nmse'].mean()),
                    feature_cosine_distance=float(arrays[name+'__feature_cosine_distance'].mean()),
                    action_relative_mse=float(arrays[name+'__action_relative_mse'].mean())))
            print('SOURCE_MODULATION_COMPONENTS_VIEW_COMPLETE',scene,len(take),flush=True)
    for model,before in zip(models,snapshots):
        check_F0_state(model,before)
    write(out/'results.json',dict(status='SOURCE_MODULATION_COMPONENTS_COMPLETE',run_id=RUN,code_commit=args.commit,
        F0_identity=identity,frozen_BF_code=G_COMMIT,reference_code=REFERENCE_COMMIT,control_code=CONTROL_COMMIT,
        methods=list(METHODS),views=list(SCENES),source_V_records=480,source_V_views=3360,
        F0_new_rows=13440,F0_smoke_rows=8,F0_control_rows_reused=6720,records=summaries,
        TX_labels_read=0,OE_reads=0,novel_reads=0,target_inputs=0,optimizer_updates=0,model_state_updates=0,
        CPU_threads=2,new_deployment_candidate=False,theoretical_ceiling=False,seconds=time.perf_counter()-start))


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
