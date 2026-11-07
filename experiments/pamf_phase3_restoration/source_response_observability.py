"""Relate fixed source directional responses to a recorded-action noise proxy."""
import argparse
import os
from pathlib import Path
import time

import numpy as np
import torch

from .cost_probe import write
from .covariance_inverse import dense_action
from .phase_mixture_source import read_trace
from .source_fingerprint_response import DIRECTIONS, SCENES
from .source_inverse import R07
from .source_pair import read
from .structured_noise_source import known_parameters

RUN='20261108-diagnostic-Restore-ResponseObservability-sourceLU-s2026092701-r01'
SOURCE='20261107-diagnostic-Restore-CovarianceResponse-sourceLU-s2026092701-r01'
SOURCE_COMMIT='4f1551ce4d81f4f3d1cade52bebaaeabe1b7316e'
METHODS=('nonnegative','signed','coupled')
BINS=((0.,1.),(1.,9.),(9.,25.),(25.,float('inf')))


def proxy(action,delta,variance):
    if action.ndim!=3 or action.shape[1:]!=(512,512) or delta.shape!=(len(action),4,2,256):
        raise ValueError('Matched source action and four IQ directions required')
    if variance.shape!=(len(action),) or not torch.isfinite(variance).all() or (variance<=0).any():
        raise ValueError('Finite positive scalar source noise proxy required')
    response=action.double()@delta.flatten(2).transpose(1,2).double()
    value=response.square().sum(1)/variance.double()[:,None]
    if not torch.isfinite(value).all() or (value<0).any():raise ValueError('Finite nonnegative action/noise proxy required')
    return value


def aggregate(proxy_values,iq_stats,feature_stats):
    values=np.asarray(proxy_values,dtype=np.float64)
    stats=[np.asarray(v,dtype=np.float64) for v in (iq_stats,feature_stats)]
    if values.ndim!=1 or any(v.shape!=(len(values),4) for v in stats):raise ValueError('Aligned scalar and response statistics required')
    if not np.isfinite(values).all() or (values<0).any() or not all(np.isfinite(v).all() for v in stats):raise ValueError('Finite source results required')
    result=[]
    for low,high in BINS:
        selected=(values>=low)&(values<high);row=dict(lower=low,upper=high if np.isfinite(high) else None,count=int(selected.sum()))
        for name,v in zip(('iq','feature'),stats):
            totals=v[selected].sum(0);ref=float(totals[0])
            row[name+'_projection_gain']=None if ref==0 else float(totals[2]/ref)
            row[name+'_relative_error']=None if ref==0 else float(totals[3]/ref)
        result.append(row)
    return result


def analyze(root,out,commit):
    start=time.perf_counter();source=root/'runs'/SOURCE;original=read(source/'results.json')
    if (original['status'],original['code_commit'],original['source_records'],original['TX_labels_read'],original['V_reads'],original['target_inputs'])!=(
            'SOURCE_DIRECTIONAL_RESPONSE_COMPLETE',SOURCE_COMMIT,224,0,0,0):raise ValueError('Original fixed source-only response study required')
    cache=root/'runs'/R07/'cache';meta=read(cache/'complete.json')
    if (meta['status'],meta['train_count'],meta['TX_labels_read'],meta['target_reads'])!=('ACTION_SOURCE_CACHE_COMPLETE',63000,0,0):
        raise ValueError('Original complete source simulator cache required')
    with np.load(source/'reference.npz',allow_pickle=False) as ref:
        indices=ref['source_index'].copy();ids=ref['ids'].copy();roles=ref['roles'].copy()
    if len(indices)!=224 or len(set(indices.tolist()))!=224 or set(roles.tolist())!={'L_s','U_s'}:
        raise ValueError('Original unique source L_s/U_s panel required')
    with np.load(cache/'epochs/01/groups.npz',allow_pickle=False) as groups:
        scenes=groups['scene'][indices].copy()
    traces=read_trace(cache/'epochs/01',indices,63000)
    records=[];raw=[]
    for position,scene in enumerate(SCENES):
        subset=slice(position*32,(position+1)*32)
        with np.load(source/(scene+'.npz'),allow_pickle=False) as file:
            if not np.array_equal(file['ids'],ids[subset]) or not np.all(scenes[subset]==scene):
                raise ValueError('Fixed scene, source records and response outputs must align')
            delta=torch.tensor(file['delta_IQ'].tolist(),dtype=torch.float64)
            stats={(m,kind):file[m+'__'+kind+'_statistics'].copy() for m in METHODS for kind in ('IQ','F0')}
        trace={key:value[subset] for key,value in traces.items()}
        kernel,main,image,variance=known_parameters(trace)
        action=dense_action(kernel,main,image)
        kappa=np.asarray(proxy(action,delta,variance).cpu().tolist(),dtype=np.float64)
        if scene==SCENES[0]:
            # Direct two-row computation uses the same stored simulator operator.
            direct=np.stack([np.asarray(((action[:2]@delta[:2,j].flatten(1).double().unsqueeze(-1)).squeeze(-1).square().sum(1)/variance[:2]).cpu().tolist(),dtype=np.float64) for j in range(4)],1)
            np.testing.assert_allclose(kappa[:2],direct,rtol=1e-12,atol=1e-12)
            write(out/'smoke.json',dict(status='PASS',source_rows=2,recorded_source_action=True,vectorized_vs_individual=True,
                neural_forwards=0,TX_labels_read=0,V_reads=0,OE_reads=0,target_inputs=0))
            print('SOURCE_RESPONSE_OBSERVABILITY_SMOKE_PASS',flush=True)
        for j,direction in enumerate(DIRECTIONS):
            for method in METHODS:
                a=stats[method,'IQ'][:,j];b=stats[method,'F0'][:,j]
                for row in aggregate(kappa[:,j],a,b):records.append(dict(scene=scene,direction=direction,method=method,**row))
                raw.append(dict(scene=scene,direction=direction,method=method,proxy=kappa[:,j],iq=a,feature=b))
        print('SOURCE_RESPONSE_OBSERVABILITY_SCENE',scene,flush=True)
    for direction in DIRECTIONS:
        for method in METHODS:
            selected=[q for q in raw if q['direction']==direction and q['method']==method]
            for row in aggregate(*(np.concatenate([q[k] for q in selected]) for k in ('proxy','iq','feature'))):
                records.append(dict(scene='all',direction=direction,method=method,**row))
    if len(records)!=8*4*3*4:raise ValueError('Complete fixed scene/direction/method/bin matrix required')
    write(out/'results.json',dict(status='SOURCE_RESPONSE_OBSERVABILITY_COMPLETE',run_id=RUN,code_commit=commit,
        source_run=SOURCE,source_commit=SOURCE_COMMIT,source_records=224,directions=4,methods=list(METHODS),records=records,
        proxy='||A_true delta||_2^2 / recorded_scalar_noise_variance',bins=[[0,1],[1,9],[9,25],[25,None]],
        source_only=True,neural_forwards=0,optimizer_updates=0,TX_labels_read=0,V_reads=0,OE_reads=0,target_inputs=0,
        limitations='White scalar noise is a proxy; true action/noise available only in source simulation. Source scenes use different fixed records. Bins are not identity or information-recovery guarantees.',
        seconds=time.perf_counter()-start))
    print('SOURCE_RESPONSE_OBSERVABILITY_COMPLETE',flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--commit',required=True);args=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or (Path.cwd()/'release_commit.txt').read_text().strip()!=args.commit:
        raise ValueError('CPU-only source diagnostic and actual code identity required')
    torch.set_num_threads(2);out=args.root/'runs'/RUN;out.mkdir()
    write(out/'launch.json',dict(pid=os.getpid(),CWD=str(Path.cwd()),code_commit=args.commit))
    analyze(args.root,out,args.commit)
    write(out/'complete.json',dict(status='ARTIFACTS_COMPLETE',target_reads=0,code_commit=args.commit))


if __name__=='__main__':main()
