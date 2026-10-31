"""Source10 calibration: retain rank5 head plus five source-stable orthogonal axes."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
import torch
from torch.nn import functional as F
from experiments.pamf_dura_direct_train import runner as base,f_guided
from experiments.pamf_dura_diagnostics.runner import source_indices,tensor,numpy_values
from .head_features import head_feature_basis
from .soft_rx_reference_probe import RUN as REFERENCE_RUN
from .soft_reference_tangent import ALPHA,minimum_shift,prototype_tangent_loss,calibration_decision
from .tangent_probe import geometry
from .extended_reference import extended_basis,auxiliary_decision
from .soft_rx_reference import head_geometry

RUN='20261031-diagnostic-DURA-source-softReference10IQ-s2026092701-r01'
J_RUN='20261031-phase1-DURA-cleanTangent-E30-s2026092701-r01'
ERROR_RUN='20261031-diagnostic-DURA-source-cleanTangent-s2026092701-r01'
RULE=('Fixedsource512128L384U/beta.25/alphaunchanged/cap1. OriginalrawJ5/headbasis unchanged; appendfive '
 'orthogonal axes fromcenteredsixsoftmeans averaged equallyover90fitRXdaycells excludingprobe. '
 'Geometry RXscatterratio<=.8/softspreadratio>=.9/512headchoicespreserved;observedexploratory46.65percentreferencecapture notnewblindgate. '
 'NewauxiliarycleanJonly512x5,notfullsourcecache. FP64rank10minimumnormGramcut1e-10/FP32actualF: '
 'linearrelerror max<=1e-3/shiftNMSEmean<=.001,p95<=.005/actualFrelativeoffseterrormedian<=.5,p95<=1/512cleanchoicespreserved. '
 'Both01/20combinedandadditionalfivealone:all512prototype/IQcosinesdefined/mean>=.5 and>=IQmean+.1/384nonnegative/eachRXmean>=0. '
 'Actualcombineddetached-stepIQgradientpressuremax<=1+2e-5,notGparameterbound/notguidevaluebound. '
 'PasssupportsseparatelyregisteredscratchjointE30/fullsevenformal;noautomatictraining/no beta,dimension,alpha orthresholdsearch.')


def feature_and_phi(reader,value,basis):
    features=reader.features(value)
    return features,F.normalize(features,dim=1,eps=1e-4)@basis*4


def summarize(values,rx):
    valid=values['prototype_defined'].astype(bool);iqvalid=values['IQ_defined'].astype(bool)
    mean=lambda key,mask:float(values[key][mask].mean(dtype=np.float64)) if mask.any() else None
    return dict(count=len(rx),defined=int(valid.sum()),defined_IQ=int(iqvalid.sum()),mean_cosine=mean('prototype_cosine',valid),
        mean_IQ_cosine=mean('IQ_cosine',iqvalid),nonnegative=int(((values['prototype_cosine']>=0)&valid).sum()),
        RX_mean_cosine={str(r):mean('prototype_cosine',valid&(rx==r)) for r in np.unique(rx)},
        maximum_balanced_pressure=float(values['actual_balanced_pressure'].max()),
        instance_to_prototype_exact_mean_cosine=mean('instance_cosine',values['instance_defined'].astype(bool)),
        mean_step_weight=float(values['step_weight'].mean(dtype=np.float64)),
        mean_weighted_guide_loss=float(values['weighted_guide_loss'].mean(dtype=np.float64)),
        guide_value_exceeds_IQ_count=int((values['weighted_guide_loss']>values['IQ_nmse']).sum()),
        median_linearization_relative_error=float(np.median(values['linearization_relative_error'])))


def summarize_auxiliary(values,rx):
    valid=values['auxiliary_defined'].astype(bool);iqvalid=values['auxiliary_IQ_defined'].astype(bool)
    mean=lambda key,mask:float(values[key][mask].mean(dtype=np.float64)) if mask.any() else None
    return dict(count=len(rx),defined=int(valid.sum()),defined_IQ=int(iqvalid.sum()),
        mean_cosine=mean('auxiliary_cosine',valid),mean_IQ_cosine=mean('auxiliary_IQ_cosine',iqvalid),
        nonnegative=int(((values['auxiliary_cosine']>=0)&valid).sum()),
        RX_mean_cosine={str(r):mean('auxiliary_cosine',valid&(rx==r)) for r in np.unique(rx)})


def run(args):
    if args.output!=args.root/'runs'/RUN or args.output.exists() or os.environ.get('CUDA_VISIBLE_DEVICES')!='':
        raise ValueError('New immutable CPU-only source IQ diagnostic required')
    p=base.paths(args.root);clean,roles=base.source_training(p);ids=f_guided.source_ids(p,roles);indices=source_indices(roles)
    rx=np.load(p['cache']/'train/rx.npy',allow_pickle=False)[indices];day=np.load(p['cache']/'train/day.npy',allow_pickle=False)[indices]
    reference=args.root/'runs'/REFERENCE_RUN;refmeta=base.read_json(reference/'results.json')
    if (not refmeta['source_reference_geometry_eligible'] or refmeta['beta']!=.25 or refmeta['F_teacher']!=f_guided.teacher_reference()
        or not refmeta['F_unchanged'] or any(refmeta[k]!=0 for k in ['G_checkpoint_reads','optimizer_updates','TX_labels_read','V_reads','target_reads'])):
        raise ValueError('Fixed qualified source soft reference required')
    with np.load(reference/'reference_geometry.npz',allow_pickle=False) as f:ref={key:f[key] for key in f.files}
    for key,value in dict(source_positions=indices,physical_ids=ids[indices],roles=roles[indices],rx=rx,day=day).items():
        if not np.array_equal(ref[key],value):raise ValueError('Identical source reference physical order required')
    cache=args.root/'runs'/J_RUN/'scratch_joint/source_clean_J_cache';meta=base.read_json(cache/'complete.json')
    expected=dict(status='SOURCE_F_CLEAN_J_COMPLETE',count=63000,shape=[63000,5,2,256],numeric_bytes=645120000,
        F_teacher=f_guided.teacher_reference(),F_inputs=63000,F_J_VJP_inputs=315000,F_updates=0,F_unchanged=True,TX_labels_read=0,V_reads=0,target_reads=0)
    if any(meta.get(key)!=value for key,value in expected.items()):raise ValueError('Original raw source F Jacobian cache required')
    if not np.array_equal(np.load(cache/'physical_ids.npy',allow_pickle=False),ids):raise ValueError('Raw source derivative physical order mismatch')
    allJ=np.load(cache/'jacobian.npy',mmap_mode='r',allow_pickle=False)
    if allJ.shape!=(63000,5,2,256) or allJ.dtype!=np.float32:raise ValueError('Complete original FP32 source derivative handle required')
    jacobians=allJ[indices];basis_np=np.load(cache/'basis.npy',allow_pickle=False)
    if (basis_np.shape!=(160,5) or not np.isfinite(jacobians).all() or not np.isfinite(basis_np).all()
        or not np.allclose(basis_np.T@basis_np,np.eye(5),atol=2e-6,rtol=2e-6)
        or np.max(abs(basis_np-ref['head_basis']@(ref['head_basis'].T@basis_np)))>2e-6):raise ValueError('Original head span/Jacobians required')
    old_basis_np=basis_np.copy();old_jacobians=jacobians.copy()
    basis_np,basis_info=extended_basis(ref['cell_means'],old_basis_np)
    probability=ref['full_probabilities'][indices]
    original_geometry=head_geometry(ref['probe_features'],probability,rx,basis_np)
    reference_geometry=head_geometry(ref['reference_features'],probability,rx,basis_np)
    extension_geometry=dict(original=original_geometry,reference=reference_geometry,
        RX_scatter_ratio=reference_geometry['RX_scatter']/original_geometry['RX_scatter'],
        class_spread_ratio=reference_geometry['class_spread']/original_geometry['class_spread'],
        original_head_choices_preserved=int((ref['original_logits'].argmax(1)==ref['reference_logits'].argmax(1)).sum()))
    extension_geometry['eligible']=bool(extension_geometry['RX_scatter_ratio']<=.8 and extension_geometry['class_spread_ratio']>=.9
        and extension_geometry['original_head_choices_preserved']==512)
    previous=args.root/'runs'/ERROR_RUN;error_meta=base.read_json(previous/'results.json')
    if (error_meta['source_checkpoint']!='20261031-phase1-DURA-repeatViews-E30-s2026092701-r01'
        or error_meta['source_commit']!='8cf8e22c9ffb54d271b22e621401de8e34c69068' or error_meta['F_teacher']!=f_guided.teacher_reference()
        or not error_meta['F_unchanged'] or any(error_meta[k]!=0 for k in ['new_G_inputs','optimizer_updates','TX_labels_read','V_reads','target_reads'])):
        raise ValueError('Fixed pureG source errors required')
    archives={}
    for view in ['1','20']:
        with np.load(previous/f'source-{view}.npz',allow_pickle=False) as f:archives[view]={key:f[key] for key in f.files}
        for key,value in dict(source_positions=indices,physical_ids=ids[indices],roles=roles[indices],rx=rx,clean=clean[indices]).items():
            if not np.array_equal(archives[view][key],value):raise ValueError('Identical source error physical order required')
        if archives[view]['restored'].shape!=(512,2,256) or not np.isfinite(archives[view]['restored']).all():raise ValueError('Complete fixed source restored IQ required')
    from experiments.pamf_fcrossrx_readout.frozen import load_reader
    reader=load_reader(p['reader'],p['source_cache'],'cpu').eval().requires_grad_(False)
    before={key:value.clone() for key,value in reader.state_dict().items()};basis=tensor(basis_np.astype(np.float32),'cpu')
    actual_basis=head_feature_basis(reader).double();old_basis=tensor(old_basis_np,'cpu').double()
    if float((old_basis-actual_basis@(actual_basis.T@old_basis)).abs().max())>2e-6:raise ValueError('Actual F head span differs')
    cleanphi=(4*ref['probe_features']@basis_np).astype(np.float32);targetphi=(4*ref['reference_features']@basis_np).astype(np.float32)
    offset=(targetphi-cleanphi).astype(np.float32)
    args.output.mkdir();base.write_json(args.output/'launch.json',dict(pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,commit=args.commit,
        CPU_threads=2,GPU_workers=0,rule=RULE))
    x=tensor(clean[indices[:2]],'cpu').requires_grad_(True);features,phi=feature_and_phi(reader,x,basis)
    loss=(phi-tensor(targetphi[:2],'cpu')).square().sum(1);gradient=torch.autograd.grad(loss.sum(),x)[0]
    with torch.no_grad():scores=reader.classify_features(features.detach())
    if (scores.shape!=(2,6) or not torch.isfinite(scores).all() or not torch.isfinite(gradient).all()
        or not gradient.abs().sum()>0 or float((phi.detach()-tensor(cleanphi[:2],'cpu')).abs().max())>3e-5):raise ValueError('Actual source2Ls F/J-reference smoke failed')
    base.write_json(args.output/'smoke.json',dict(status='PASS',real_Ls=2,F_inputs=2,F_input_VJP_inputs=2,TX_labels_read=0,target_reads=0))
    print(json.dumps(dict(event='SOURCE_EXTENDED_REFERENCE_IQ_SMOKE_PASS',real_Ls=2)),flush=True)
    started=time.perf_counter();auxiliary_jacobians=[];jac_started=time.perf_counter()
    for start in range(0,512,32):
        value=tensor(clean[indices[start:start+32]],'cpu').requires_grad_(True)
        _,auxiliary_phi=feature_and_phi(reader,value,basis[:,5:])
        columns=[torch.autograd.grad(auxiliary_phi[:,k].sum(),value,retain_graph=k<4)[0] for k in range(5)]
        auxiliary_jacobians.append(numpy_values(torch.stack(columns,dim=1)))
    auxiliary_jacobians=np.concatenate(auxiliary_jacobians);jacobians=np.concatenate([old_jacobians,auxiliary_jacobians],axis=1)
    if jacobians.shape!=(512,10,2,256) or not np.isfinite(jacobians).all():raise ValueError('Complete source10 derivatives required')
    auxiliary_J_seconds=time.perf_counter()-jac_started
    jac=tensor(jacobians,'cpu');d=tensor(offset,'cpu');x=tensor(clean[indices],'cpu')
    shift,info=minimum_shift(jac,d,dimension=10);power=x.square().mean((1,2)).clamp_min(1e-4)
    if not torch.isfinite(shift).all():raise FloatingPointError('Nonfinite source minimum IQ shift')
    actual_shift=x+shift-x;linear=(jac.double()*actual_shift.double()[:,None]).sum((2,3));dnorm=d.double().norm(dim=1)
    linear_relative=(linear-d.double()).norm(dim=1)/dnorm.clamp_min(1e-12)
    modified=x+shift;phis=[];scores=[]
    with torch.no_grad():
        for start in range(0,512,32):
            embedding,phi=feature_and_phi(reader,modified[start:start+32],basis)
            phis.append(numpy_values(phi));scores.append(numpy_values(reader.classify_features(embedding)))
    actual_phi=np.concatenate(phis);actual_logits=np.concatenate(scores)
    requested=np.linalg.norm(offset,axis=1)>1e-12
    nonlinear_error=np.linalg.norm(actual_phi-targetphi,axis=1)/np.maximum(np.linalg.norm(offset,axis=1),1e-12)
    # A zero requested offset has no relative-error interpretation; its absolute
    # residual is retained and it is excluded from relative quantiles explicitly.
    if not requested.any():raise ValueError('At least one nonzero prototype shift required')
    iqnmse=numpy_values(actual_shift.square().mean((1,2))/power)
    choices=actual_logits.argmax(1)==ref['original_logits'].argmax(1)
    feasibility=dict(linear_residual_max=float(linear_relative.max()),shift_NMSE_mean=float(iqnmse.mean(dtype=np.float64)),
        shift_NMSE_p95=float(np.quantile(iqnmse,.95)),nonlinear_relative_error_median=float(np.median(nonlinear_error[requested])),
        nonlinear_relative_error_p95=float(np.quantile(nonlinear_error[requested],.95)),choices_preserved=int(choices.sum()),
        requested_nonzero=int(requested.sum()),minimum_numerical_rank=int(info['rank'].min()))
    np.savez(args.output/'feasibility.npz',source_positions=indices,physical_ids=ids[indices],roles=roles[indices],rx=rx,day=day,
        clean=clean[indices],jacobian=jacobians,basis=basis_np,original_basis=old_basis_np,clean_phi=cleanphi,target_phi=targetphi,offset=offset,
        shift=numpy_values(shift),actual_shift=numpy_values(actual_shift),modified=numpy_values(modified),actual_phi=actual_phi,actual_logits=actual_logits,
        reference_logits=ref['original_logits'],linear_relative=numpy_values(linear_relative),rank=numpy_values(info['rank']),
        shift_NMSE=iqnmse,nonlinear_relative_error=nonlinear_error,requested_nonzero=requested,choices_preserved=choices)
    print(json.dumps(dict(event='SOURCE_EXTENDED_REFERENCE_FEASIBILITY_COMPLETE',summary=feasibility,seconds=time.perf_counter()-started)),flush=True)
    views={};groups={};auxiliary_views={}
    for view,a in archives.items():
        accumulated={};recorded_phi=[];gradients={key:[] for key in ['exact_gradient','prototype_gradient','balanced_gradient','IQ_gradient','instance_gradient','auxiliary_exact_gradient','auxiliary_prototype_gradient']}
        with np.load(p['cache']/'epochs'/f'{int(view):02d}'/'groups.npz',allow_pickle=False) as f:labels={key:f[key][indices] for key in ['scene','state','lock']}
        for start in range(0,512,32):
            sl=slice(start,start+32);x=tensor(clean[indices[sl]],'cpu');r=tensor(a['restored'][sl],'cpu').requires_grad_(True)
            j=tensor(jacobians[sl],'cpu');off=tensor(offset[sl],'cpu');target=tensor(targetphi[sl],'cpu')
            _,phi=feature_and_phi(reader,r,basis);exact=ALPHA*(phi-target).square().sum(1)
            exact_gradient=torch.autograd.grad(exact.sum(),r,retain_graph=True)[0]
            auxiliary_exact=ALPHA*(phi[:,5:]-target[:,5:]).square().sum(1)
            auxiliary_exact_gradient=torch.autograd.grad(auxiliary_exact.sum(),r)[0]
            recorded_phi.append(numpy_values(phi.detach()))
            delta=r.detach()-x;projected=(j*delta[:,None]).sum((2,3));residual=projected-off
            prototype_gradient=2*ALPHA*(residual[:,:,None,None]*j).sum(1)
            instance_gradient=2*ALPHA*(projected[:,:,None,None]*j).sum(1)
            auxiliary_prototype_gradient=2*ALPHA*(residual[:,5:,None,None]*j[:,5:]).sum(1)
            IQ_gradient=2*delta/(512*x.square().mean((1,2),keepdim=True).clamp_min(1e-4))
            value,stat=prototype_tangent_loss(r,x,j,off,ALPHA,dimension=10)
            balanced_gradient=torch.autograd.grad((ALPHA*value).sum(),r)[0]
            torch.testing.assert_close(balanced_gradient,prototype_gradient*stat['weight'][:,None,None],atol=2e-6,rtol=2e-5)
            actual_pressure=balanced_gradient.flatten(1).norm(dim=1)/IQ_gradient.flatten(1).norm(dim=1).clamp_min(1e-12)
            tg=geometry(prototype_gradient,exact_gradient);ig=geometry(IQ_gradient,exact_gradient);cg=geometry(instance_gradient,exact_gradient)
            ag=geometry(auxiliary_prototype_gradient,auxiliary_exact_gradient);aq=geometry(IQ_gradient,auxiliary_exact_gradient)
            local=dict(auxiliary_cosine=ag['cosine'],auxiliary_defined=ag['defined'].float(),auxiliary_IQ_cosine=aq['cosine'],auxiliary_IQ_defined=aq['defined'].float(),prototype_cosine=tg['cosine'],prototype_defined=tg['defined'].float(),IQ_cosine=ig['cosine'],IQ_defined=ig['defined'].float(),
                instance_cosine=cg['cosine'],instance_defined=cg['defined'].float(),raw_pressure=stat['raw_pressure'],pressure_defined=stat['pressure_defined'].float(),
                step_weight=stat['weight'],actual_balanced_pressure=actual_pressure,weighted_guide_loss=ALPHA*value.detach(),
                IQ_nmse=delta.square().mean((1,2))/x.square().mean((1,2)).clamp_min(1e-4),exact_prototype_loss=exact.detach(),
                linearization_relative_error=(residual-(phi.detach()-target)).norm(dim=1)/(phi.detach()-target).norm(dim=1).clamp_min(1e-8))
            for key,v in local.items():
                if v.shape!=(32,) or not torch.isfinite(v).all():raise FloatingPointError('Finite full source prototype geometry required')
                accumulated.setdefault(key,[]).append(numpy_values(v))
            for key in gradients:gradients[key].append(numpy_values(locals()[key]))
        values={key:np.concatenate(v) for key,v in accumulated.items()};gradient_values={key:np.concatenate(v) for key,v in gradients.items()}
        views[view]=summarize(values,rx);auxiliary_views[view]=summarize_auxiliary(values,rx);masks={'all':np.ones(512,bool)}
        for key,v in dict(role=roles[indices],rx=rx,day=day,**labels).items():
            for label in np.unique(v):masks[f'{key}:{label}']=v==label
        groups[view]={key:dict(count=int(mask.sum()),defined=int(values['prototype_defined'][mask].sum()),
            mean_cosine=float(values['prototype_cosine'][mask].mean(dtype=np.float64)),
            mean_step_weight=float(values['step_weight'][mask].mean(dtype=np.float64)),auxiliary_mean_cosine=float(values['auxiliary_cosine'][mask].mean(dtype=np.float64))) for key,mask in masks.items()}
        np.savez(args.output/f'source-{view}.npz',source_positions=indices,physical_ids=ids[indices],roles=roles[indices],rx=rx,day=day,
            **labels,restored=a['restored'],clean=clean[indices],restored_phi=np.concatenate(recorded_phi),**values,**gradient_values)
        print(json.dumps(dict(event='SOURCE_EXTENDED_REFERENCE_GRADIENT_COMPLETE',view=view,summary=views[view],auxiliary=auxiliary_views[view],seconds=time.perf_counter()-started)),flush=True)
    if any(not torch.equal(value,before[key]) for key,value in reader.state_dict().items()) or any(v.grad is not None for v in reader.parameters()):raise ValueError('Frozen F changed')
    total_decision=calibration_decision(feasibility,views);extra_decision=auxiliary_decision(auxiliary_views)
    decision=dict(eligible=bool(extension_geometry['eligible'] and total_decision['eligible'] and extra_decision['eligible']),
        geometry=extension_geometry['eligible'],combined=total_decision,auxiliary=extra_decision,automatic_training=False)
    base.write_json(args.output/'results.json',dict(status='ANALYZED_SOURCE_ONLY',rule=RULE,feasibility=feasibility,views=views,auxiliary_views=auxiliary_views,groups=groups,decision=decision,source_extension_geometry=extension_geometry,basis_info=basis_info,
        reference_run=REFERENCE_RUN,reference_beta=.25,raw_J_cache_run=J_RUN,source_error_run=ERROR_RUN,
        source_checkpoint=error_meta['source_checkpoint'],source_commit=error_meta['source_commit'],F_teacher=f_guided.teacher_reference(),F_unchanged=True,
        alpha=ALPHA,source_records=512,source_L=128,source_U=384,F_inputs=2048,F_smoke_inputs=2,F_input_VJP_inputs=4608,F_smoke_VJP_inputs=2,
        new_G_inputs=0,new_F_clean_J_inputs=512,new_F_clean_auxiliary_J_VJP_inputs=2560,new_full_source_J_cache=False,auxiliary_J_seconds=auxiliary_J_seconds,raw_J_cache_metadata=meta,optimizer_updates=0,TX_labels_read=0,V_reads=0,target_reads=0,
        seconds=time.perf_counter()-started,scope='Source-only min-changeIQ/featuretarget andcachedlocalgradient calibration;usespairedsourceclean,notdeployableoracle/trueTX/targetbenefit/globalexactloss;stepIQgradientcapnotlossvalueorparameterbound'))
    base.write_json(args.output/'complete.json',dict(status='ANALYZED_SOURCE_ONLY',decision=decision,optimizer_updates=0,predictions=0,goal_complete=False))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ['root','output']:parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--commit',required=True);args=parser.parse_args()
    if Path('release_commit.txt').read_text().strip()!=args.commit:raise ValueError('Pinned release required')
    torch.set_num_threads(2);run(args)


if __name__=='__main__':main()
