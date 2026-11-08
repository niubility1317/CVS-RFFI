"""Source-only, label-free F0 feature error decomposition over three views."""
import argparse
from pathlib import Path
import time

import numpy as np
import torch

from . import physical_feature_gradient_probe as execution
from . import multiview_trial as trial
from .cache import check_F0_state, load_F0, tensor
from .cost_probe import write
from .source_pair import read, validate_source
from .variance_trial import load_scale

RUN = '20261108-diagnostic-Restore-FeatureViewDispersion-sourceLU-s2026092701-r01'
MODULE = 'experiments.pamf_phase3_restoration.source_feature_view_dispersion'


def select_rows(roles):
    if roles.shape!=(63000,):raise ValueError('Original 63000 source roles required')
    result=[]
    for role in ('L_s','U_s'):
        values=np.flatnonzero(roles==role)
        if len(values)<256:raise ValueError('At least 256 records per source role required')
        result.extend(values[np.linspace(0,len(values)-1,256,dtype=np.int64)].tolist())
    indices=np.asarray(result,dtype=np.int64)
    if len(np.unique(indices))!=512:raise ValueError('Unique source records required')
    return indices


def decompose(clean, views, scale):
    clean=np.asarray(clean,dtype=np.float64);views=np.asarray(views,dtype=np.float64);scale=np.asarray(scale,dtype=np.float64)
    if clean.ndim!=2 or views.shape!=(3,*clean.shape) or scale.shape!=(clean.shape[1],):
        raise ValueError('Three aligned feature views and per-feature scale required')
    if not all(np.isfinite(x).all() for x in (clean,views,scale)) or (scale<=0).any():
        raise ValueError('Finite features and positive scales required')
    errors=views-clean[None];mean=errors.mean(axis=0);centered=errors-mean[None]
    per_view=np.mean(errors**2/scale[None,None,:],axis=(0,2))
    bias=np.mean(mean**2/scale[None,:],axis=1)
    variation=np.mean(centered**2/scale[None,None,:],axis=(0,2))
    if not np.allclose(per_view,bias+variation,rtol=1e-11,atol=1e-13):
        raise ValueError('Source feature bias/dispersion identity failed')
    return per_view,bias,variation


def summarize(clean, raw, restored, scale, roles, scenes):
    if len(clean)!=512 or roles.shape!=(512,) or scenes.shape!=(3,512):
        raise ValueError('Fixed source panel required')
    masks=[('all',np.ones(512,dtype=bool)),('L_s',roles=='L_s'),('U_s',roles=='U_s'),
           ('at_least_one_low_urban',(scenes=='practical_low_urban').any(axis=0)),
           ('no_low_urban',~(scenes=='practical_low_urban').any(axis=0))]
    rows=[]
    for name,mask in masks:
        if not mask.any():continue
        row=dict(group=name,count=int(mask.sum()))
        for kind,values in (('received',raw),('restored',restored)):
            total,bias,variation=decompose(clean[mask],values[:,mask],scale)
            row[kind]=dict(mean_squared_feature_error=float(total.mean()),
                mean_feature_bias_squared=float(bias.mean()),mean_feature_variation=float(variation.mean()),
                variation_fraction=float(variation.sum()/total.sum()) if total.sum()>0 else None)
        rows.append(row)
    return rows


def worker(args):
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    device='cuda:0';started=time.perf_counter();config=validate_source(read(args.inputs));source,roles=trial.source(args)
    selected=select_rows(roles);selected_roles=roles[selected]
    reader,identity=load_F0(config,device);trial.main_eval.valid_identity(identity);reader.eval().requires_grad_(False)
    fstate={k:v.detach().clone() for k,v in reader.state_dict().items()}
    models,metadata=execution.load_models(args.root,device);model=models['fixed_source_E30'];del models
    model.eval().requires_grad_(False)
    if metadata['F0_identity']!=identity:raise ValueError('Same frozen original F0 required')
    state={k:v.detach().clone() for k,v in model.state_dict().items()}
    scale=np.asarray(load_scale(args.root).detach().cpu().tolist(),dtype=np.float32)
    def take(path):
        a=np.load(path,mmap_mode='r',allow_pickle=False)
        try:return tensor(a[selected].copy(),device)
        finally:a._mmap.close()
    clean=take(source/'train/clean.npy'); first=source/'epochs'/'01';m=read(first/'complete.json')
    if (m['count'],m['TX_labels_read'],m['target_reads'])!=(63000,0,0):raise ValueError('Original source draw required')
    first_received=take(first/'received.npy')
    with torch.no_grad():
        restored,aux=model(first_received[:2],return_aux=True);f0=reader.features(restored)
        if (f0.shape!=(2,160) or not torch.isfinite(f0).all() or aux['relative_solve_residual'].max()>1e-8):
            raise ValueError('Actual source G/F0 smoke failed')
    out=args.root/'runs'/RUN
    write(out/'smoke.json',dict(status='PASS',source_rows=2,actual_source_G=True,actual_original_F0=True,
        logits_called=False,TX_labels_read=0,V_reads=0,OE_reads=0,target_inputs=0,optimizer_updates=0))
    print('REAL_SOURCE_FEATURE_VIEW_SMOKE_PASS',flush=True)
    def features(signal):
        with torch.no_grad():
            output=torch.cat([reader.features(signal[i:i+32]) for i in range(0,len(signal),32)])
            return np.asarray(output.detach().cpu().tolist(),dtype=np.float32)
    clean_f=features(clean);raw_f=[];restored_f=[];scenes=[];max_residual=0.
    for draw in (1,2,3):
        folder=source/'epochs'/f'{draw:02d}';info=read(folder/'complete.json')
        if (info['count'],info['TX_labels_read'],info['target_reads'])!=(63000,0,0):raise ValueError('Completed original source draw required')
        with np.load(folder/'groups.npz',allow_pickle=False) as group:scenes.append(group['scene'][selected].copy())
        received=first_received if draw==1 else take(folder/'received.npy')
        raw_f.append(features(received));saved=[]
        with torch.no_grad():
            for i in range(0,512,32):
                output,aux=model(received[i:i+32],return_aux=True)
                if not torch.isfinite(output).all():raise ValueError('Finite restored source IQ required')
                max_residual=max(max_residual,float(aux['relative_solve_residual'].max()))
                if max_residual>1e-8:raise ValueError('Source posterior residual too large')
                saved.append(np.asarray(reader.features(output).detach().cpu().tolist(),dtype=np.float32))
        restored_f.append(np.concatenate(saved));print('SOURCE_FEATURE_VIEW_COMPLETE',draw,512,flush=True)
    raw_f=np.stack(raw_f);restored_f=np.stack(restored_f);scenes=np.stack(scenes)
    rows=summarize(clean_f,raw_f,restored_f,scale,selected_roles,scenes)
    if any(not torch.equal(v,model.state_dict()[k]) for k,v in state.items()):raise ValueError('Source G state changed')
    check_F0_state(reader,fstate)
    with (out/'features.npz').open('xb') as f:np.savez_compressed(f,clean=clean_f,received=raw_f,restored=restored_f,
        scale=scale,roles=selected_roles,scenes=scenes)
    write(out/'results.json',dict(status='SOURCE_FEATURE_VIEW_COMPLETE',source_records=512,draws=[1,2,3],
        roles={'L_s':256,'U_s':256},selection='evenly spaced within original role order; source training in-sample',
        scene_counts={str(k):int((scenes==k).sum()) for k in np.unique(scenes)},summary=rows,
        F0_identity=identity,source_G_run=execution.PARENT_RUN,source_G_commit=execution.PARENT_COMMIT,
        model_updates=0,optimizer_updates=0,logits_called=False,TX_labels_read=0,V_reads=0,OE_reads=0,
        target_inputs=0,feature_average_deployable=False,source_sample_representative=False,
        max_posterior_residual=max_residual,seconds=time.perf_counter()-started))
    write(out/'complete.json',dict(status='ARTIFACTS_COMPLETE',source_only=True,records=512,views=3))
    print('SOURCE_FEATURE_VIEW_DISPERSION_COMPLETE',flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['run','worker']);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--inputs',type=Path,required=True);p.add_argument('--cache',type=Path,required=True);p.add_argument('--commit',required=True)
    args=p.parse_args()
    if (Path.cwd()/'release_commit.txt').read_text().strip()!=args.commit:raise ValueError('Release identity mismatch')
    if args.mode=='run':
        execution.RUN,execution.MODULE=RUN,MODULE;execution.controller(args)
    else:worker(args)


if __name__=='__main__':main()
