"""Frozen source diagnostic of posterior uncertainty along F0 decision margins."""
import argparse
import os
from pathlib import Path
import time

import numpy as np
import torch

from .cache import check_F0_state, load_F0, tensor
from .cost_probe import write
from .covariance_inverse import dense_action
from .covariance_parameter_source import G_COMMIT, load_model
from .main_split_eval import valid_identity
from .phase_mixture_source import read_trace
from .source_inverse import R07, SCENES
from .source_pair import read, validate_source
from .structured_noise_source import RUN as KNOWN_RUN, known_parameters

RUN = '20261107-diagnostic-Restore-posterior-margin-uncertainty-sourceV-s2026092701-r01'
MODULE = 'experiments.pamf_phase3_restoration.margin_uncertainty_source'
KNOWN_COMMIT = 'd46cffe092dd667dab9df04c69c0f31a7aec5326'


def margin_values(logits,order):
    ordered = logits.gather(1,order)
    return ordered[:,:1]-ordered[:,1:]


def input_margin_directions(reader,clean,order=None):
    x = clean.detach().requires_grad_(True)
    logits = reader.classify_features(reader.features(x))
    if order is None:
        order = logits.detach().argsort(dim=1,descending=True)
    margins = margin_values(logits,order)
    directions = []
    for competitor in range(margins.shape[1]):
        gradient, = torch.autograd.grad(margins[:,competitor].sum(),x,
                                        retain_graph=competitor+1<margins.shape[1])
        directions.append(gradient.flatten(1))
    return logits.detach(),torch.stack(directions,dim=-1)


def projected_variances(action,covariance,variance,directions):
    """Diagonal of G^T C G and G^T Cov[X | y,A] G for the supplied directions G."""
    ac = action @ covariance
    normal = ac @ action.transpose(-1,-2)
    normal.diagonal(dim1=-2,dim2=-1).add_(variance[:,None])
    factor = torch.linalg.cholesky(normal)
    prior = (directions*(covariance@directions)).sum(dim=1)
    whitened = torch.linalg.solve_triangular(factor,ac@directions,upper=False)
    posterior = prior-whitened.square().sum(dim=1)
    if (posterior < -1e-9*prior).any() or not torch.isfinite(posterior).all() or (prior<=0).any():
        raise ValueError('Invalid source margin variance in fixed Gaussian model')
    return prior,posterior.clamp_min(0)


def read_parent(root,cache):
    out=root/'runs'/KNOWN_RUN
    done=read(out/'results.json')
    if (read(out/'complete.json')['status'],done['status'],done['code_commit'],done['frozen_BF_code'],
        done['source_V_records'],done['source_V_views'])!=(
            'ARTIFACTS_COMPLETE','SOURCE_STRUCTURED_NOISE_COMPLETE',KNOWN_COMMIT,G_COMMIT,120,840):
        raise ValueError('Exact prior frozen known-Action source run required')
    if any(done[key] for key in ('TX_labels_read','target_inputs','model_state_updates','optimizer_updates')):
        raise ValueError('Source-only frozen reference required')
    with np.load(out/'reference.npz',allow_pickle=False) as f:
        ref={key:f[key] for key in ('ids','rx','day','source_index','clean','logits')}
    if (ref['source_index'].shape!=(120,) or len(set(ref['ids'].tolist()))!=120
            or not np.array_equal(np.load(cache/'validation/physical_ids.npy',allow_pickle=False)[ref['source_index']],ref['ids'])):
        raise ValueError('Original 120 source V order required')
    return out,ref


def summarize(arrays):
    prior,post=arrays['prior_variance'],arrays['posterior_variance']
    actual,linear=arrays['actual_margin_change'],arrays['linear_margin_change']
    clean=arrays['clean_margins']
    actual_miss=~arrays['cached_known_agrees'].astype(bool)
    local_cross=np.any(clean<=np.sqrt(post),axis=1)
    linear_flip=np.any(clean+linear<0,axis=1)
    return dict(count=len(clean),
        known_iq_remaining_fraction=float(arrays['known_iq_remaining_fraction'].mean()),
        nearest_margin_remaining_fraction=float(post[:,0].sum()/prior[:,0].sum()),
        nearest_margin_median_remaining_fraction=float(np.median(post[:,0]/prior[:,0])),
        all_margins_remaining_fraction=float(post.sum()/prior.sum()),
        actual_margin_delta_rms=float(np.sqrt(np.mean(actual**2))),
        linear_margin_delta_error_rms=float(np.sqrt(np.mean((linear-actual)**2))),
        nearest_actual_delta_rms=float(np.sqrt(np.mean(actual[:,0]**2))),
        nearest_linear_delta_error_rms=float(np.sqrt(np.mean((linear[:,0]-actual[:,0])**2))),
        exact_known_Fclean_misses=int(actual_miss.sum()),
        local_one_sigma_boundary_count=int(local_cross.sum()),
        actual_misses_in_one_sigma_group=int((local_cross&actual_miss).sum()),
        linearized_flip_count=int(linear_flip.sum()),linearized_and_actual_flip=int((linear_flip&actual_miss).sum()))


def worker(args,out):
    start=time.perf_counter()
    # Only the original frozen prior buffer is taken from BF, no G forward.
    g=load_model(args.root,'cpu')
    covariance=g.prior_covariance.detach().double().clone()
    del g
    reader,identity=load_F0(validate_source(read(args.inputs)),'cpu')
    valid_identity(identity)
    frozen={k:v.detach().clone() for k,v in reader.state_dict().items()}
    cache=args.root/'runs'/R07/'cache'
    indices=np.flatnonzero(np.load(cache/'train/roles.npy',allow_pickle=False)=='L_s')[:2]
    if len(indices)!=2:
        raise ValueError('Two original source L_s smoke records required')
    a=np.load(cache/'train/clean.npy',mmap_mode='r',allow_pickle=False)
    try:
        source=tensor(a[indices],'cpu')
    finally:
        a._mmap.close()
    z,j=input_margin_directions(reader,source)
    if z.shape!=(2,6) or j.shape!=(2,512,5) or not torch.isfinite(z).all() or not torch.isfinite(j).all() or not j.abs().sum()>0:
        raise ValueError('Actual original F0 input-gradient smoke failed')
    _,single=input_margin_directions(reader,source[:1],z[:1].argsort(dim=1,descending=True))
    torch.testing.assert_close(single,j[:1],rtol=2e-3,atol=2e-4)
    check_F0_state(reader,frozen)
    write(out/'smoke.json',dict(status='PASS',actual_original_F0=True,actual_Ls=2,input_only_margin_gradients=True,
        grouped_single_direction_match=True,model_updates=0,TX_labels_read=0,target_inputs=0))
    print('SOURCE_MARGIN_UNCERTAINTY_REAL_SMOKE_PASS',flush=True)
    parent,ref=read_parent(args.root,cache)
    take=ref['source_index']
    clean=tensor(ref['clean'],'cpu')
    order=tensor(ref['logits'],'cpu').argsort(dim=1,descending=True)
    logits,jacobian=[],[]
    for offset in range(0,len(take),16):
        view=slice(offset,offset+16)
        z,j=input_margin_directions(reader,clean[view],order[view])
        logits.append(z)
        jacobian.append(j.double())
    logits,jacobian=torch.cat(logits),torch.cat(jacobian)
    if not torch.equal(logits.argmax(1),order[:,0]) or not torch.isfinite(jacobian).all():
        raise ValueError('Actual frozen F0 source reference or input gradients differ')
    clean_margins=margin_values(logits,order).double()
    cpu_logit_difference=float((logits-tensor(ref['logits'],'cpu')).abs().max())
    with (out/'reference.npz').open('xb') as f:
        np.savez(f,ids=ref['ids'],rx=ref['rx'],day=ref['day'],source_index=take,
                 clean_logits=np.asarray(logits.tolist()),class_order=np.asarray(order.tolist()))
    rows=[]
    with torch.inference_mode():
        for scene in SCENES:
            trace=read_trace(cache/'validation_actions'/scene,take,1800)
            with np.load(parent/(scene+'.npz'),allow_pickle=False) as f:
                known_iq=tensor(f['known_scalar__iq'],'cpu')
                known_logits=tensor(f['known_scalar__logits'],'cpu')
                iq_risk=f['known_scalar__posterior_variance_fraction']
            prior_values,post_values=[],[]
            for offset in range(0,len(take),16):
                view=slice(offset,offset+16)
                h,m,i,var=known_parameters({key:value[view] for key,value in trace.items()})
                a=dense_action(h,m,i)
                prior,post=projected_variances(a,covariance,var,jacobian[view])
                prior_values.extend(prior.tolist())
                post_values.extend(post.tolist())
            actual=margin_values(known_logits,order).double()-clean_margins
            linear=((known_iq-clean).flatten(1).double()[:,:,None]*jacobian).sum(1)
            arrays=dict(prior_variance=np.asarray(prior_values),posterior_variance=np.asarray(post_values),
                clean_margins=np.asarray(clean_margins.tolist()),actual_margin_change=np.asarray(actual.tolist()),
                linear_margin_change=np.asarray(linear.tolist()),known_iq_remaining_fraction=iq_risk,
                cached_known_agrees=np.asarray((known_logits.argmax(1)==order[:,0]).tolist()))
            with (out/(scene+'.npz')).open('xb') as f:
                np.savez(f,**arrays)
            rows.append(dict(scene=scene,**summarize(arrays)))
            print('SOURCE_MARGIN_UNCERTAINTY_VIEW_COMPLETE',scene,len(take),flush=True)
    check_F0_state(reader,frozen)
    write(out/'results.json',dict(status='SOURCE_MARGIN_UNCERTAINTY_COMPLETE',run_id=RUN,code_commit=args.commit,
        parent_known_code=KNOWN_COMMIT,frozen_BF_prior_code=G_COMMIT,
        source_V_records=120,source_V_views=840,directions_per_record=5,source_V_input_VJPs=600,
        reused_known_iq_rows=840,reused_known_F0_rows=840,new_G_forwards=0,new_recovered_F0_rows=0,
        original_cpu_clean_forward_rows=120,source_smoke_clean_forward_rows=3,source_smoke_input_VJP_rows=15,
        cpu_vs_cached_clean_logit_max_abs=cpu_logit_difference,
        views=list(SCENES),records=rows,TX_labels_read=0,OE_reads=0,novel_reads=0,target_inputs=0,
        optimizer_updates=0,model_state_updates=0,source_V_training_rows=0,CPU_threads=2,
        linearized_input_margin_diagnostic=True,classification_error_bound=False,seconds=time.perf_counter()-start))


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
