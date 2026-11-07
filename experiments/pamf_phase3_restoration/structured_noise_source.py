"""Frozen source V check of structured receiver noise in a Gaussian IQ inverse."""
import argparse
import os
from pathlib import Path
import time

import numpy as np
import torch

from .cache import check_F0_state, load_F0, tensor
from .cost_probe import write
from .covariance_guidance_source import source_metrics
from .covariance_inverse import dense_action
from .covariance_parameter_source import G_COMMIT, load_model
from .main_split_eval import valid_identity
from .phase_mixture_source import REFERENCE_COMMIT, REFERENCE_RUN, read_trace, source_indices
from .source_inverse import R07, SCENES
from .source_pair import read, validate_source
from .structured_noise_posterior import iq_noise_blocks, posterior

RUN = '20261107-diagnostic-Restore-structured-IQ-noise-sourceV-s2026092701-r01'
MODULE = 'experiments.pamf_phase3_restoration.structured_noise_source'
METHODS = ('blind_scalar','blind_structured','known_scalar','known_structured')


def known_parameters(trace):
    h = torch.complex(*trace['kernel'].double().unbind(1))
    coeff = trace['coefficients'].double()
    main = torch.complex(coeff[:,0],coeff[:,1])
    image = torch.complex(coeff[:,2],coeff[:,3])
    return h,main,image,trace['variance'].double()


def readouts(G, trace):
    observed = trace['received'].flatten(1).double()
    mean,covariance = G.prior_mean.double(),G.prior_covariance.double()
    estimated = G.estimate(trace['received'],dense=False)
    blind = [estimated[name].to(torch.complex128) for name in ('kernel','main','image')]
    blind.append(estimated['variance'].double())
    outputs = {}
    for prefix, values in [('blind',blind),('known',known_parameters(trace))]:
        kernel,main,image,variance = values
        action = dense_action(kernel,main,image)
        structure = iq_noise_blocks(main,image,variance)
        for suffix,blocks in [('scalar',None),('structured',structure)]:
            iq,residual,risk = posterior(action,observed,mean,covariance,variance,
                blocks=blocks,risk=(prefix=='known'))
            if residual.max()>1e-8:
                raise ValueError('Gaussian solve residual exceeds 1e-8')
            outputs[prefix+'_'+suffix] = dict(iq=iq.float().reshape_as(trace['received']),
                relative_solve_residual=residual)
            if risk is not None:
                outputs[prefix+'_'+suffix]['posterior_variance_fraction'] = risk/covariance.trace()
        if prefix=='known':
            noise_model = action,variance,structure
    return outputs,noise_model


def noise_evidence(observed,clean,action,variance,blocks):
    # Compare both Gaussian proxy noise models on the same realized source residual.
    i,q = (observed.flatten(1).double() - (action @ clean.flatten(1).double()[:,:,None]).squeeze(-1)).chunk(2,dim=1)
    vi,vq,cross = blocks
    v = variance[:,None]
    determinant = vi*vq-cross.square()
    structured = (vq*i.square()+vi*q.square()-2*cross*i*q)/determinant
    scalar = (i.square()+q.square())/v
    shape_distance = (((vi-v).square()+(vq-v).square()+2*cross.square()).mean(1)/(2*variance.square())).sqrt()
    return dict(known_noise_relative_shape_difference=shape_distance,
        known_structured_minus_scalar_NLL_per_real=.25*(structured-scalar+determinant.log()-2*v.log()).mean(1),
        known_scalar_whitened_energy_per_real=.5*scalar.mean(1),
        known_structured_whitened_energy_per_real=.5*structured.mean(1))


def load_reference(root, identity, cache):
    folder = root/'runs'/REFERENCE_RUN
    done = read(folder/'predictions_complete.json')
    if (done['status'],done['code_commit'],done['source_V_packets'],done['TX_labels_read'],done['target_inputs']) != (
        'SOURCE_V_PREDICTIONS_COMPLETE',REFERENCE_COMMIT,1800,0,0) or done['F0_identity'] != identity:
        raise ValueError('Exact original source V prediction and F0 identity required')
    with np.load(folder/'references.npz',allow_pickle=False) as f:
        take = source_indices(f['rx'],f['day'])
        if not np.array_equal(f['ids'],np.load(cache/'validation/physical_ids.npy',allow_pickle=False)):
            raise ValueError('Original source V physical record order differs')
        ref = {key:f[key][take] for key in ['ids','rx','day','clean','features','logits']}
        ref['scale'] = f['scale']
    return ref,take


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
        raise ValueError('Original source L_s smoke records required')
    # First actual model forward is a source-only smoke of all readout paths.
    with torch.inference_mode():
        trace = read_trace(cache/'epochs/01',source2,63000)
        outputs,noise_model = readouts(G,trace)
        torch.testing.assert_close(outputs['blind_scalar']['iq'],G(trace['received']),rtol=0,atol=0)
        _,variance,structure = noise_model
        torch.testing.assert_close(.5*(structure[0]+structure[1]).mean(1),variance,rtol=1e-12,atol=1e-12)
        for values in outputs.values():
            logits = F.classify_features(F.features(values['iq']))
            if logits.shape!=(2,6) or not torch.isfinite(logits).all():
                raise ValueError('Original actual F0 source CPU smoke failed')
    write(out/'smoke.json',dict(status='PASS',actual_Ls=2,actual_F0=True,actual_original_BF=True,
        scalar_exactly_matches_original_BF=True,structured_trace_matches_scalar=True,methods=list(METHODS),
        F0_rows=8,estimator_calls=2,CPU_only=True,TX_labels_read=0,target_inputs=0))
    print('SOURCE_STRUCTURED_NOISE_REAL_SMOKE_PASS',flush=True)
    ref,take = load_reference(args.root,identity,cache)
    with (out/'reference.npz').open('xb') as f:
        np.savez(f,**ref,source_index=take)
    clean,target,clean_logits,scale = [tensor(ref[k],'cpu') for k in ['clean','features','logits','scale']]
    summaries = []
    noise_summaries = []
    with torch.inference_mode():
        for scene in SCENES:
            trace = read_trace(cache/'validation_actions'/scene,take,1800)
            records = {}
            for offset in range(0,len(take),16):
                view = slice(offset,offset+16)
                batch = {key:value[view] for key,value in trace.items()}
                outputs,noise_model = readouts(G,batch)
                values = noise_evidence(batch['received'],clean[view],*noise_model)
                for name,fields in outputs.items():
                    iq = fields['iq']
                    features = F.features(iq)
                    logits = F.classify_features(features)
                    metrics = source_metrics(iq,clean[view],features,logits,target[view],clean_logits[view],scale)
                    values.update({name+'__'+key:value for key,value in dict(fields,**metrics,features=features,logits=logits).items()})
                if not all(torch.isfinite(x).all() for x in values.values()):
                    raise ValueError('Nonfinite source output or noise evidence')
                for key,value in values.items():
                    records.setdefault(key,[]).extend(value.tolist())
            arrays = {key:np.asarray(value) for key,value in records.items()}
            with (out/(scene+'.npz')).open('xb') as f:
                np.savez(f,**arrays)
            for name in METHODS:
                before = arrays[name.split('_')[0]+'_scalar__clean_decision_agreement'].astype(bool)
                after = arrays[name+'__clean_decision_agreement'].astype(bool)
                row = dict(scene=scene,method=name,count=len(take),Fclean_agreement=float(after.mean()),
                    vs_scalar_repairs=int((~before & after).sum()),vs_scalar_damage=int((before & ~after).sum()),
                    iq_nmse=float(arrays[name+'__iq_nmse'].mean()),feature_cosine_distance=float(arrays[name+'__feature_cosine_distance'].mean()))
                key = name+'__posterior_variance_fraction'
                if key in arrays:
                    row['posterior_variance_fraction']=float(arrays[key].mean())
                summaries.append(row)
            noise_summaries.append(dict(scene=scene,**{key:float(value.mean()) for key,value in arrays.items() if '__' not in key}))
            print('SOURCE_STRUCTURED_NOISE_VIEW_COMPLETE',scene,len(take),flush=True)
    for model,before in zip(models,snapshots):
        check_F0_state(model,before)
    write(out/'results.json',dict(status='SOURCE_STRUCTURED_NOISE_COMPLETE',run_id=RUN,code_commit=args.commit,
        frozen_BF_code=G_COMMIT,reference_code=REFERENCE_COMMIT,methods=list(METHODS),views=list(SCENES),
        source_V_records=120,source_V_views=840,F0_new_rows=3360,F0_smoke_rows=8,F0_clean_rows_reused=120,
        estimator_rows=840,estimator_smoke_rows=4,records=summaries,noise_evidence=noise_summaries,
        known_proxy_noise_risk=True,conditioned_on_cached_simulator_Action=True,TX_labels_read=0,OE_reads=0,novel_reads=0,
        target_inputs=0,optimizer_updates=0,model_state_updates=0,CPU_threads=2,new_deployment_candidate=False,
        theoretical_ceiling=False,seconds=time.perf_counter()-start))


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
