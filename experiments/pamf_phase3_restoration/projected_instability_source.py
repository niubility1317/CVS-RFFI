"""Frozen source-only stress probe around the projected-prior E11 loss spike."""
import argparse
import importlib
import os
from pathlib import Path
import time

import numpy as np
import torch

RUN='20261108-diagnostic-Restore-ProjectedE11-sourceLU-s2026092701-r01'
PARENT='20261107-phase1-Restore-ProjectedPriorDetail-E30-s2026092701-r01'
ARM='TF_ProjectedPriorDetail'
PARENT_COMMIT='8cf3669c4a5d98f6c4930117952203afd56f2f6e'
R07='20261019-phase1-restoration-learned-action-source-s2026092701-r07'
GROUPS=('anchor','high_received_peak','high_coefficient_peak','low_noise','high_noise')


def select_groups(received_peak,coefficient_peak,variance,n=32):
    r,c,v=[np.asarray(x,dtype=np.float64) for x in (received_peak,coefficient_peak,variance)]
    if r.ndim!=1 or c.shape!=r.shape or v.shape!=r.shape or len(r)<n or n<1:
        raise ValueError('Matched source-only stress statistics required')
    if not all(np.isfinite(x).all() for x in (r,c,v)) or (v<=0).any():
        raise ValueError('Finite source inputs and positive variance required')
    groups=dict(anchor=np.linspace(0,len(r)-1,n,dtype=np.int64),
        high_received_peak=np.argsort(r,kind='stable')[-n:],high_coefficient_peak=np.argsort(c,kind='stable')[-n:],
        low_noise=np.argsort(v,kind='stable')[:n],high_noise=np.argsort(v,kind='stable')[-n:])
    return groups,np.unique(np.concatenate(list(groups.values())))


def summary(values):
    a=np.asarray(values,dtype=np.float64)
    if a.ndim!=1 or not len(a) or not np.isfinite(a).all() or (a<0).any():
        raise ValueError('Finite nonnegative source stress measurements required')
    q=np.quantile(a,[.5,.9,.99,1.])
    return dict(count=len(a),median=float(q[0]),p90=float(q[1]),p99=float(q[2]),maximum=float(q[3]))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--commit',required=True);args=parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or (Path.cwd()/'release_commit.txt').read_text().strip()!=args.commit:
        raise ValueError('CPU-only source diagnostic with actual code identity required')
    torch.set_num_threads(2);started=time.perf_counter();root=args.root
    # Import the exact original implementation, including its old posterior arithmetic.
    package=importlib.import_module('experiments.pamf_phase3_restoration')
    parent_release=root/'releases'/(PARENT+'-release')/'experiments/pamf_phase3_restoration'
    if not (parent_release/'projected_prior_detail.py').is_file():raise ValueError('Original source-only release required')
    package.__path__.insert(0,str(parent_release))
    science=importlib.import_module('.projected_prior_detail_trial',package.__name__)
    original=importlib.import_module('.projected_prior_detail',package.__name__)
    if Path(original.__file__).resolve()!=parent_release.joinpath('projected_prior_detail.py').resolve():
        raise ValueError('Exact original projected posterior implementation required')
    cache=importlib.import_module('.cache',package.__name__)
    io=importlib.import_module('.source_pair',package.__name__)
    writer=importlib.import_module('.cost_probe',package.__name__).write
    source=root/'runs'/R07/'cache';metadata=io.read(source/'complete.json')
    if (metadata['status'],metadata['train_count'],metadata['TX_labels_read'],metadata['target_reads'])!=('ACTION_SOURCE_CACHE_COMPLETE',63000,0,0):
        raise ValueError('Original valid source simulator cache required')
    training=root/'runs'/PARENT/'rows'/ARM/'training';smoke=io.read(training/'smoke.json')
    if smoke['status']!='PASS' or any(smoke[k] for k in ('TX_labels_read','V_reads','OE_reads','target_inputs')):
        raise ValueError('Original source-only parent provenance required')
    draw=source/'epochs/11';complete=io.read(draw/'complete.json')
    if (complete['count'],complete['TX_labels_read'],complete['target_reads'])!=(63000,0,0):raise ValueError('Original source draw11 required')
    roles=np.load(source/'train/roles.npy',allow_pickle=False)
    if roles.shape!=(63000,) or set(roles.tolist())!={'L_s','U_s'}:raise ValueError('Original source training roles required')
    received=np.load(draw/'received.npy',mmap_mode='r',allow_pickle=False)
    coefficients=np.load(draw/'coefficients.npy',mmap_mode='r',allow_pickle=False)
    variance=np.load(draw/'variance.npy',mmap_mode='r',allow_pickle=False)
    clean=np.load(source/'train/clean.npy',mmap_mode='r',allow_pickle=False)
    if (received.shape,coefficients.shape,variance.shape,clean.shape)!=((63000,2,256),(63000,4,256),(63000,),(63000,2,256)):
        raise ValueError('Original source draw layouts required')
    peaks=[];coefficient_peaks=[]
    for i in range(0,63000,1024):
        peaks.extend(np.abs(received[i:i+1024]).max((1,2)).tolist())
        coefficient_peaks.extend(np.abs(coefficients[i:i+1024]).max((1,2)).tolist())
    groups,selected=select_groups(peaks,coefficient_peaks,variance)
    x=cache.tensor(received[selected],'cpu');y=cache.tensor(clean[selected],'cpu')
    models={};snapshots={}
    for epoch in (10,11):
        checkpoint=torch.load(training/'recovery'/('E%02d.pt'%epoch),map_location='cpu',weights_only=False)
        if (checkpoint['code_commit'],checkpoint['epoch'],checkpoint['updates'],checkpoint['config'])!=(
                PARENT_COMMIT,epoch,247*epoch,science.configuration(ARM)):
            raise ValueError('Exact original E10/E11 source-only technical checkpoints required')
        model=science.factory(ARM);model.load_state_dict(checkpoint['state'],strict=True);model.eval().requires_grad_(False)
        models[epoch]=model;snapshots[epoch]={k:v.detach().clone() for k,v in model.state_dict().items()}
    out=root/'runs'/RUN;out.mkdir();writer(out/'launch.json',dict(pid=os.getpid(),CWD=str(Path.cwd()),code_commit=args.commit))
    def measure(model,input_iq,target):
        saved=[]
        hook=model.detail.register_forward_hook(lambda module,inputs,value:saved.append(value.detach()))
        try:
            with torch.inference_mode():restored,aux=model.source_readout(input_iq,target)
        finally:hook.remove()
        if len(saved)!=1:raise ValueError('Exactly one original detail forward required')
        base=aux['base_iq'];proposal=saved[0]-base;correction=aux['projected_prior_update'];energy=target.square().mean((1,2)).clamp_min(1e-4)
        p=proposal.square().mean((1,2));c=correction.square().mean((1,2));v=dict(
            proposal_nmse=p/energy,projected_update_nmse=c/energy,projected_to_proposal_norm=(c/p.clamp_min(1e-24)).sqrt(),
            deployed_iq_nmse=(restored-target).square().mean((1,2))/energy,
            base_iq_nmse=(base-target).square().mean((1,2))/energy,
            relative_solve_residual=aux['relative_solve_residual'])
        if any(not torch.isfinite(value).all() for value in v.values()):raise ValueError('Finite source stress outputs required')
        return {k:value.detach().cpu().tolist() for k,value in v.items()}
    for epoch,model in models.items():
        check=measure(model,x[:2],y[:2])
        if max(check['relative_solve_residual'])>1e-8:raise ValueError('Actual original source posterior smoke failed')
    writer(out/'smoke.json',dict(status='PASS',source_rows=2,original_E10_E11=True,original_release=True,
        TX_labels_read=0,V_reads=0,OE_reads=0,target_inputs=0,optimizer_updates=0))
    print('PROJECTED_E11_SOURCE_SMOKE_PASS',flush=True)
    rows=[]
    for epoch,model in models.items():
        collected={}
        for offset in range(0,len(selected),8):
            values=measure(model,x[offset:offset+8],y[offset:offset+8])
            for key,value in values.items():collected.setdefault(key,[]).extend(value)
        for name,before in snapshots[epoch].items():
            if not torch.equal(before,model.state_dict()[name]):raise ValueError('Frozen source model changed')
        for group,indices in [('unique_all',selected),*groups.items()]:
            positions=np.searchsorted(selected,indices)
            if not np.array_equal(selected[positions],indices):raise ValueError('Source stress group alignment failed')
            rows.append(dict(epoch=epoch,group=group,metrics={k:summary(np.asarray(v)[positions]) for k,v in collected.items()}))
        print('PROJECTED_E11_SOURCE_CHECKPOINT_COMPLETE',epoch,len(selected),flush=True)
    writer(out/'results.json',dict(status='SOURCE_PROJECTED_STRESS_COMPLETE',run_id=RUN,code_commit=args.commit,
        original_run=PARENT,original_code_commit=PARENT_COMMIT,epochs=[10,11],source_draw=11,unique_source_rows=len(selected),
        group_size=32,groups=list(GROUPS),rows=rows,source_group_selection='Fixed source input/noise stress strata; no model scores or TX.',
        source_only=True,TX_labels_read=0,V_reads=0,OE_reads=0,target_inputs=0,optimizer_updates=0,model_state_updates=0,
        limitations='Stress groups overlap and are nonrepresentative. Fixed E10/E11 checkpoints do not reconstruct intermediate minibatch states or prove the cause of the E11 mean spike.',
        seconds=time.perf_counter()-started))
    writer(out/'complete.json',dict(status='ARTIFACTS_COMPLETE',code_commit=args.commit,target_reads=0))
    print('PROJECTED_E11_SOURCE_STRESS_COMPLETE',flush=True)


if __name__=='__main__':main()
