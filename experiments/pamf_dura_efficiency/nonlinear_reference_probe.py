"""Source-only nonlinear IQ reachability and real-G exact-gradient smoke."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
import torch
from experiments.pamf_dura_direct_train import runner as base,f_guided
from experiments.pamf_dura_diagnostics.runner import source_indices,tensor,numpy_values
from .head_features import head_feature_basis
from .model import direct_objective,supervised_action
from .exact_reference import feature_phi,exact_guide
from .soft_reference_tangent import minimum_shift,ALPHA

RUN='20261031-diagnostic-DURA-source-nonlinearReference-s2026092701-r01'
REFERENCE='20261031-diagnostic-DURA-source-softReference10IQ-s2026092701-r01'
STEPS=8
HALF_STEP=.5
RULE=('Samequalifiedsource10featuretarget/beta.25/originalF50,source512128L384U. EightfixedGaussNewtonrelinearizations '
 'withhalfstep.5,FP64Gram eigencut1e-10/FP32representedIQ;noearlystop/line/dimension/alpha/beta search. '
 'FinalactualFrelativeoffseterrormedian<=.5/p95<=1,IQshiftNMSEmean<=.001/p95<=.005,512originalcleanchoicespreserved. '
 'ExactguideusesoneFforward/inputreverse,detachedperpacketIQgradientcap1/alphaunchanged;real2LsGgradient/stateFfreeze smoke. '
 'Nooptimizer/Gtraining/deployableGNoracle/target/TXtruth. PassonlysupportsseparateGPUcostmeasurement,notautomatic30epochtraining.')


def source(args,p):
    clean,roles=base.source_training(p);ids=f_guided.source_ids(p,roles);indices=source_indices(roles)
    folder=args.root/'runs'/REFERENCE;m=base.read_json(folder/'results.json')
    if (m['F_teacher']!=f_guided.teacher_reference() or not m['F_unchanged'] or not m['source_extension_geometry']['eligible']
        or m['reference_beta']!=.25 or m['alpha']!=ALPHA or m['source_records']!=512
        or any(m[k]!=0 for k in ['new_G_inputs','optimizer_updates','TX_labels_read','V_reads','target_reads'])):
        raise ValueError('Legal source10 reference geometry and frozen original F required')
    keys=['source_positions','physical_ids','roles','rx','day','clean','basis','original_basis','target_phi','clean_phi','reference_logits']
    with np.load(folder/'feasibility.npz',allow_pickle=False) as f:a={key:f[key] for key in keys}
    rx=np.load(p['cache']/'train/rx.npy',allow_pickle=False)[indices];day=np.load(p['cache']/'train/day.npy',allow_pickle=False)[indices]
    for key,value in dict(source_positions=indices,physical_ids=ids[indices],roles=roles[indices],rx=rx,day=day,clean=clean[indices]).items():
        if not np.array_equal(a[key],value):raise ValueError('Original source512 physical roles/order required')
    if (a['basis'].shape!=(160,10) or a['target_phi'].shape!=(512,10) or not np.isfinite(a['basis']).all()
        or not np.isfinite(a['target_phi']).all() or not np.array_equal(a['basis'][:,:5],a['original_basis'])
        or not np.allclose(a['basis'].T@a['basis'],np.eye(10),atol=3e-6,rtol=3e-6)):
        raise ValueError('Fixed source10 basis and target required')
    return a,p,dict(source_run=REFERENCE,F_teacher=m['F_teacher'],reference_beta=.25,source_contract='original7/63/30',
        original_reference_geometry_eligible=True,failed_tangent_surrogate_used=False,old_G_error_or_J_arrays_read=False)


def smoke(reader,a,p,output):
    if not np.all(a['roles'][:2]=='L_s'):raise ValueError('First real smoke inputs must be L_s')
    basis=tensor(a['basis'],'cpu');target=tensor(a['target_phi'][:2],'cpu');clean=tensor(a['clean'][:2],'cpu')
    with torch.no_grad():
        features,phi=feature_phi(reader,clean,basis);scores=reader.classify_features(features)
    if (scores.shape!=(2,6) or not torch.isfinite(scores).all()
        or float((phi-tensor(a['clean_phi'][:2],'cpu')).abs().max())>3e-5):raise ValueError('Actual F/source10 cache smoke failed')
    student=base.scratch_model(2026092701);before={k:v.clone() for k,v in student.state_dict().items()}
    take=a['source_positions'][:2];folder=p['cache']/'epochs/01'
    values={key:tensor(np.load(folder/(key+'.npy'),mmap_mode='r',allow_pickle=False)[take],'cpu') for key in ['received','kernel','coefficients','variance']}
    rng=torch.Generator().manual_seed(2026103004)
    probes=[torch.complex(torch.randn(2,256,generator=rng),torch.randn(2,256,generator=rng)) for _ in range(2)]
    _,terms,restored=direct_objective(student,values['received'],clean,supervised_action(values['kernel'],values['coefficients']),values['variance'],probes,return_restored=True)
    guide,info=exact_guide(reader,restored,clean,target,basis)
    actual=torch.autograd.grad(guide.sum(),restored,retain_graph=True)[0]
    torch.testing.assert_close(actual,info['weighted_gradient'],rtol=2e-5,atol=2e-6)
    if not actual.abs().sum()>0 or (actual.flatten(1).norm(dim=1)>info['IQ_gradient_norm']*(1+2e-5)).any():raise ValueError('Actual exact source guide cap failed')
    physical=terms['iq_nmse']+.1*terms['prior_nmse']+.05*terms['spectral_error']+.05*terms['action_probe_nmse']
    (physical+guide).mean().backward()
    for prefix in ['frozen.coarse.','frozen.estimator.','prior.']:
        gradients=[v.grad for k,v in student.named_parameters() if k.startswith(prefix) and v.grad is not None]
        if not gradients or not all(torch.isfinite(v).all() for v in gradients) or not any(v.abs().sum()>0 for v in gradients):raise ValueError('Real joint G gradient missing: '+prefix)
    if any(not torch.equal(v,before[k]) for k,v in student.state_dict().items()):raise ValueError('No-update G smoke changed state')
    if any(v.grad is not None for v in reader.parameters()):raise ValueError('Frozen F accumulated parameter gradients')
    base.write_json(output/'smoke.json',dict(status='PASS',real_Ls=2,F_inputs=4,F_input_VJP_inputs=2,G_inputs=2,
        exact_guide_injection_verified=True,IQ_gradient_cap=True,G_coarse_action_prior_gradients=True,G_state_unchanged=True,
        optimizer_updates=0,initialization_checkpoint_loads=0,TX_labels_read=0,V_reads=0,target_reads=0))
    print(json.dumps(dict(event='REAL_EXACT_REFERENCE_G_SMOKE_PASS')),flush=True)


def run(args):
    if args.output!=args.root/'runs'/RUN or args.output.exists() or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('New immutable CPU-only source diagnostic required')
    a,p,provenance=source(args,base.paths(args.root))
    from experiments.pamf_fcrossrx_readout.frozen import load_reader
    reader=load_reader(p['reader'],p['source_cache'],'cpu').eval().requires_grad_(False)
    before={key:value.clone() for key,value in reader.state_dict().items()};basis=tensor(a['basis'],'cpu')
    real_basis=head_feature_basis(reader).double();old=tensor(a['original_basis'],'cpu').double()
    if float((old-real_basis@(real_basis.T@old)).abs().max())>2e-6:raise ValueError('Actual frozen F head span differs')
    args.output.mkdir();base.write_json(args.output/'launch.json',dict(pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,commit=args.commit,CPU_threads=2,GPU_workers=0,rule=RULE))
    smoke(reader,a,p,args.output);started=time.perf_counter()
    initial=tensor(a['clean'],'cpu');current=initial.clone();target=tensor(a['target_phi'],'cpu')
    requested=(target-tensor(a['clean_phi'],'cpu')).norm(dim=1).clamp_min(1e-12)
    power=initial.square().mean((1,2)).clamp_min(1e-4)
    path_iq=[];path_phi=[];path_J=[];path_delta=[];path_rank=[];path_linear_residual=[];records=[]
    for iteration in range(1,STEPS+1):
        phi_values=[];jacobians=[];path_iq.append(numpy_values(current))
        for start in range(0,512,32):
            value=current[start:start+32].detach().requires_grad_(True);_,phi=feature_phi(reader,value,basis)
            columns=[torch.autograd.grad(phi[:,k].sum(),value,retain_graph=k<9)[0] for k in range(10)]
            phi_values.append(numpy_values(phi.detach()));jacobians.append(numpy_values(torch.stack(columns,dim=1)))
        phi_np=np.concatenate(phi_values);J_np=np.concatenate(jacobians)
        if not np.isfinite(phi_np).all() or not np.isfinite(J_np).all():raise FloatingPointError('Nonfinite nonlinear F/J source state')
        phi=tensor(phi_np,'cpu');jac=tensor(J_np,'cpu');delta,info=minimum_shift(jac,target-phi,dimension=10)
        if not torch.isfinite(delta).all():raise FloatingPointError('Nonfinite relinearized IQ step')
        path_phi.append(phi_np);path_J.append(J_np);path_delta.append(numpy_values(delta));path_rank.append(numpy_values(info['rank']))
        path_linear_residual.append(numpy_values(info['relative_residual']))
        error=(phi-target).norm(dim=1)/requested
        current=(current+HALF_STEP*delta).detach();energy=(current-initial).square().mean((1,2))/power
        record=dict(iteration=iteration,input_relative_error_median=float(np.median(numpy_values(error))),input_relative_error_p95=float(torch.quantile(error,.95)),
            output_IQ_shift_NMSE_mean=float(energy.mean()),output_IQ_shift_NMSE_p95=float(torch.quantile(energy,.95)),
            minimum_numerical_rank=int(info['rank'].min()),seconds=time.perf_counter()-started)
        records.append(record);print(json.dumps(dict(event='SOURCE_NONLINEAR_GN_ITERATION',**record)),flush=True)
    phis=[];scores=[]
    with torch.no_grad():
        for start in range(0,512,32):
            features,phi=feature_phi(reader,current[start:start+32],basis)
            phis.append(numpy_values(phi));scores.append(numpy_values(reader.classify_features(features)))
    final_phi=np.concatenate(phis);logits=np.concatenate(scores);path_iq.append(numpy_values(current));path_phi.append(final_phi)
    final_error=np.linalg.norm(final_phi-a['target_phi'],axis=1)/numpy_values(requested)
    shift=current-initial;energy=numpy_values(shift.square().mean((1,2))/power);choices=logits.argmax(1)==a['reference_logits'].argmax(1)
    feasibility=dict(shift_NMSE_mean=float(energy.mean(dtype=np.float64)),shift_NMSE_p95=float(np.quantile(energy,.95)),
        nonlinear_relative_error_median=float(np.median(final_error)),nonlinear_relative_error_p95=float(np.quantile(final_error,.95)),choices_preserved=int(choices.sum()))
    feasible=(feasibility['shift_NMSE_mean']<=.001 and feasibility['shift_NMSE_p95']<=.005
        and feasibility['nonlinear_relative_error_median']<=.5 and feasibility['nonlinear_relative_error_p95']<=1 and feasibility['choices_preserved']==512)
    if any(not torch.equal(v,before[k]) for k,v in reader.state_dict().items()) or any(v.grad is not None for v in reader.parameters()):raise ValueError('Original frozen F changed')
    if not np.isfinite(final_phi).all() or not np.isfinite(energy).all() or not np.isfinite(final_error).all():raise FloatingPointError('Finite complete nonlinear source path required')
    np.savez(args.output/'nonlinear_path.npz',source_positions=a['source_positions'],physical_ids=a['physical_ids'],roles=a['roles'],rx=a['rx'],day=a['day'],
        clean=a['clean'],basis=a['basis'],clean_phi=a['clean_phi'],target_phi=a['target_phi'],reference_logits=a['reference_logits'],
        path_iq=np.stack(path_iq),path_phi=np.stack(path_phi),path_jacobian=np.stack(path_J),path_delta=np.stack(path_delta),
        path_rank=np.stack(path_rank),path_linear_residual=np.stack(path_linear_residual),final_phi=final_phi,final_logits=logits,
        final_shift=numpy_values(shift),final_IQ_shift_NMSE=energy,final_relative_error=final_error,final_choices_preserved=choices)
    decision=dict(eligible=bool(feasible),automatic_training=False,next_stage='SeparatelyregisteredGPUcostbenchmarkonly')
    base.write_json(args.output/'results.json',dict(status='ANALYZED_SOURCE_ONLY',rule=RULE,iterations=STEPS,half_step=HALF_STEP,dimension=10,
        provenance=provenance,feasibility=feasibility,records=records,decision=decision,F_teacher=f_guided.teacher_reference(),F_unchanged=True,
        source_records=512,source_L=128,source_U=384,F_inputs=4608,F_input_VJP_inputs=40960,F_smoke_inputs=4,F_smoke_VJP_inputs=2,
        G_smoke_inputs=2,G_state_unchanged=True,initialization_checkpoint_loads=0,optimizer_updates=0,TX_labels_read=0,V_reads=0,target_reads=0,
        no_full_source_J_cache=True,new_G_training=False,seconds=time.perf_counter()-started,
        scope='Pairedsourceclean nonlinearinverse diagnostic,notdeployableGN/trueTXhardware/targetefficacy;exactguideoneFreverse tested,GPUspeedunmeasured'))
    base.write_json(args.output/'complete.json',dict(status='ANALYZED_SOURCE_ONLY',decision=decision,optimizer_updates=0,formal_predictions=0,goal_complete=False))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ['root','output']:parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--commit',required=True);args=parser.parse_args()
    if Path('release_commit.txt').read_text().strip()!=args.commit:raise ValueError('Pinned release required')
    torch.set_num_threads(2);run(args)


if __name__=='__main__':main()
