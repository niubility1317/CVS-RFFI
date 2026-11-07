"""Test blind main-only phase estimates on fixed source observations."""

import argparse
import os
from pathlib import Path
import time

import numpy as np
import torch

from .cache import check_F0_state, load_F0, tensor
from .coherent_main_phase import coherent_phase
from .cost_probe import write
from .covariance_guidance_source import source_metrics
from .covariance_inverse import dense_action, gaussian_posterior
from .covariance_parameter_source import G_COMMIT, load_model
from .main_split_eval import valid_identity
from .phase_mixture_source import REFERENCE_COMMIT
from .phase_ramp_posterior import calibrated_action, fixed_grid, ramp_posteriors
from .source_action_factor_probe import references
from .source_inverse import R07, SCENES
from .source_modulation_probe import RUN as CONTROL_RUN
from .source_pair import read, validate_source

RUN = '20261107-diagnostic-Restore-coherent-main-phase-sourceV-s2026092701-r01'
MODULE = 'experiments.pamf_phase3_restoration.source_coherent_main_phase'
CONTROL_COMMIT = '53901dd11e69bf90041f67bf5a6668ee89963d55'
NEW_METHODS = ('whole_action_phase','main_mean_scalar','main_mean_full')
METHODS = ('blind',*NEW_METHODS,'main_phase')


def received_only(folder, take, count):
    done = read(folder/'complete.json')
    if (done['count'],done['TX_labels_read'],done['target_reads']) != (count,0,0):
        raise ValueError('Original complete source-only observations required')
    values = np.load(folder/'received.npy',mmap_mode='r',allow_pickle=False)
    try:
        if values.shape != (count,2,256) or values.dtype != np.float32:
            raise ValueError('Exact original source IQ layout required')
        received = tensor(values[take],'cpu')
    finally:
        values._mmap.close()
    if not torch.isfinite(received).all():
        raise ValueError('Finite source observations required')
    return received


def load_control(folder, ref, take):
    result = read(folder/'results.json')
    if ((result['status'],result['code_commit'],result['source_V_records'],result['TX_labels_read'],result['target_inputs'])
            != ('SOURCE_MODULATION_COMPONENTS_COMPLETE',CONTROL_COMMIT,480,0,0)
            or read(folder/'complete.json')['status'] != 'ARTIFACTS_COMPLETE'):
        raise ValueError('Exact frozen source modulation comparison required')
    with np.load(folder/'reference.npz',allow_pickle=False) as file:
        if not np.array_equal(file['source_index'],take) or not all(np.array_equal(file[key],value) for key,value in ref.items()):
            raise ValueError('Original source controls differ in records or clean reference')


def readouts(G,received,grid):
    est = G.estimate(received,dense=False)
    h,main,image = (est[k].to(torch.complex128) for k in ('kernel','main','image'))
    action = dense_action(h,main,image)
    main_action = dense_action(h,main,torch.zeros_like(image))
    image_action = action-main_action
    observed,variance = received.flatten(1).double(),est['variance'].double()
    mean,covariance = G.prior_mean.double(),G.prior_covariance.double()
    phase,drift,log_prior = grid
    scalar,full = coherent_phase(action,main_action,observed,mean,covariance,variance,phase,log_prior[0])
    original,gamma = ramp_posteriors(action,observed,mean,covariance,variance,*grid)
    values = {'whole_action_phase':dict(iq=original[:,1].float().reshape_as(received),phase=gamma[:,0,0].angle())}
    for name,direction in [('main_mean_scalar',scalar),('main_mean_full',full)]:
        theta = direction.angle()
        rotation = torch.polar(torch.ones_like(theta),theta)
        calibrated = calibrated_action(main_action,rotation[:,None].expand_as(main))+image_action
        restored,residual = gaussian_posterior(calibrated,observed,mean,covariance,variance)
        if residual.max() > 1e-8:
            raise ValueError('Main-only posterior residual exceeds 1e-8')
        values[name] = dict(iq=restored.float().reshape_as(received),phase=theta)
    if not all(torch.isfinite(v).all() for fields in values.values() for v in fields.values()):
        raise ValueError('Nonfinite blind main-phase readout')
    return values, original[:,0].float().reshape_as(received)


def worker(args,out):
    started = time.perf_counter()
    G = load_model(args.root,'cpu')
    F, identity = load_F0(validate_source(read(args.inputs)),'cpu')
    valid_identity(identity)
    models = G,F
    snapshots = [{k:v.detach().clone() for k,v in model.state_dict().items()} for model in models]
    cache = args.root/'runs'/R07/'cache'
    source2 = np.flatnonzero(np.load(cache/'train/roles.npy',allow_pickle=False)=='L_s')[:2]
    if len(source2) != 2:
        raise ValueError('Original source L_s2 smoke required')
    phase,drift,prior = fixed_grid('cpu')
    grid = phase,drift[4:5],prior[4:5]
    with torch.inference_mode():
        x = received_only(cache/'epochs/01',source2,63000)
        outputs,blind = readouts(G,x,grid)
        torch.testing.assert_close(blind,G(x),rtol=2e-6,atol=2e-7)
        for fields in outputs.values():
            logits = F.classify_features(F.features(fields['iq']))
            if logits.shape != (2,6) or not torch.isfinite(logits).all():
                raise ValueError('Actual source L_s2 F0 smoke failed')
    write(out/'smoke.json',dict(status='PASS',actual_Ls=2,actual_F0=True,actual_original_BF=True,
        blind_matches_original_BF=True,F0_rows=6,CPU_only=True,simulator_factors_read=0,TX_labels_read=0,target_inputs=0))
    print('SOURCE_COHERENT_MAIN_REAL_SMOKE_PASS',flush=True)

    ref,take = references(args.root,cache,identity)
    control = args.root/'runs'/CONTROL_RUN
    load_control(control,ref,take)
    with (out/'reference.npz').open('xb') as file:
        np.savez(file,**ref,source_index=take)
    clean,target,clean_logits,scale = [tensor(ref[k],'cpu') for k in ('clean','features','logits','scale')]
    summaries = []
    with torch.inference_mode():
        for scene in SCENES:
            received = received_only(cache/'validation_actions'/scene,take,1800)
            records = {}
            for offset in range(0,len(take),16):
                view = slice(offset,offset+16)
                outputs,_ = readouts(G,received[view],grid)
                for name,fields in outputs.items():
                    iq = fields['iq']
                    features = F.features(iq)
                    logits = F.classify_features(features)
                    metrics = source_metrics(iq,clean[view],features,logits,target[view],clean_logits[view],scale)
                    for key,value in dict(metrics,logits=logits,phase_radians=fields['phase']).items():
                        if not torch.isfinite(value).all():
                            raise ValueError('Nonfinite source phase readout')
                        records.setdefault(name+'__'+key,[]).extend(value.tolist())
            arrays = {key:np.asarray(value) for key,value in records.items()}
            # Simulator-dependent comparison enters only after the scene's blind outputs.
            with np.load(control/(scene+'.npz'),allow_pickle=False) as file:
                for name in ('blind','main_phase'):
                    for key in ('logits','prediction','iq_nmse','feature_cosine_distance'):
                        field = name+'__'+key
                        arrays[field] = file[field]
                arrays['reference_main_phase_radians'] = file['main_phase_offset_radians']
            with (out/(scene+'.npz')).open('xb') as file:
                np.savez(file,**arrays)
            correct = {name:arrays[name+'__prediction']==ref['logits'].argmax(1) for name in METHODS}
            before = correct['blind']
            for name in METHODS:
                after = correct[name]
                row = dict(scene=scene,method=name,count=len(take),Fclean_correct=int(after.sum()),
                    repairs=int((~before&after).sum()),damage=int((before&~after).sum()),
                    iq_nmse=float(arrays[name+'__iq_nmse'].mean()),
                    feature_cosine_distance=float(arrays[name+'__feature_cosine_distance'].mean()))
                if name != 'main_phase':
                    estimate = arrays[name+'__phase_radians'] if name in NEW_METHODS else 0.
                    error = np.abs(np.angle(np.exp(1j*(estimate-arrays['reference_main_phase_radians']))))
                    row['mean_main_phase_error_degrees'] = float(np.degrees(error).mean())
                summaries.append(row)
            print('SOURCE_COHERENT_MAIN_VIEW_COMPLETE',scene,len(take),flush=True)
    for model,before in zip(models,snapshots):
        check_F0_state(model,before)
    write(out/'results.json',dict(status='SOURCE_COHERENT_MAIN_COMPLETE',run_id=RUN,code_commit=args.commit,
        F0_identity=identity,frozen_BF_code=G_COMMIT,reference_code=REFERENCE_COMMIT,control_code=CONTROL_COMMIT,
        methods=list(METHODS),views=list(SCENES),source_V_records=480,source_V_views=3360,
        F0_new_rows=10080,F0_smoke_rows=6,F0_control_rows_reused=6720,records=summaries,phase_points=256,
        phase_prior_std_degrees=15,simulator_factors_read_in_blind_readouts=0,TX_labels_read=0,OE_reads=0,novel_reads=0,
        target_inputs=0,optimizer_updates=0,model_state_updates=0,CPU_threads=2,new_deployment_candidate=False,
        theoretical_ceiling=False,seconds=time.perf_counter()-started))


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
