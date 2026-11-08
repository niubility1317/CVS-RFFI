"""Source-only diagnostic: estimated versus simulated noise and F0 feature error."""
import argparse
from pathlib import Path
import time

import numpy as np
import torch

from . import physical_feature_gradient_probe as execution
from . import source_feature_view_dispersion as panel
from . import multiview_trial as trial
from .cache import tensor
from .cost_probe import write
from .source_pair import read, validate_source

RUN = '20261108-diagnostic-Restore-PredNoiseFeatureRank-sourceLU-s2026092701-r01'
MODULE = 'experiments.pamf_phase3_restoration.source_predicted_noise_rank'
PANEL = '20261108-diagnostic-Restore-FeatureViewDispersion-sourceLU-s2026092701-r01'


def ranks(values):
    x=np.asarray(values,dtype=np.float64)
    if x.ndim!=1 or not np.isfinite(x).all():raise ValueError('Finite 1D values required')
    order=np.argsort(x,kind='stable');sorted_x=x[order];out=np.empty(len(x),dtype=np.float64)
    start=0
    while start<len(x):
        end=start+1
        while end<len(x) and sorted_x[end]==sorted_x[start]:end+=1
        out[order[start:end]]=(start+end-1)/2
        start=end
    return out


def spearman(left,right):
    a,b=ranks(left),ranks(right)
    if a.shape!=b.shape or not len(a):raise ValueError('Equal nonempty score vectors required')
    a-=a.mean();b-=b.mean();denom=np.linalg.norm(a)*np.linalg.norm(b)
    return None if denom==0 else float((a@b)/denom)


def within_record(score,error):
    score=np.asarray(score,dtype=np.float64);error=np.asarray(error,dtype=np.float64)
    if score.ndim!=2 or score.shape[0]!=3 or error.shape!=score.shape:
        raise ValueError('Three aligned source views required')
    if not np.isfinite(score).all() or not np.isfinite(error).all():raise ValueError('Finite scores required')
    lo=np.argmin(score,axis=0);hi=np.argmax(score,axis=0);index=np.arange(score.shape[1])
    keep=score[hi,index]>score[lo,index];loerr=error[lo,index][keep];hierr=error[hi,index][keep];delta=hierr-loerr
    return dict(records=int(keep.sum()),high_error_mean=float(hierr.mean()) if len(delta) else None,
        low_error_mean=float(loerr.mean()) if len(delta) else None,
        high_greater=int((delta>0).sum()),equal=int((delta==0).sum()),high_lower=int((delta<0).sum()))


def summarize(predicted,truth,error,scenes):
    if predicted.shape!=truth.shape or truth.shape!=error.shape or scenes.shape!=truth.shape or truth.shape!=(3,512):
        raise ValueError('Fixed aligned source views required')
    flat=predicted.ravel();flat_error=error.ravel();flat_scenes=scenes.ravel();order=np.argsort(flat,kind='stable');bins=[]
    for n,sub in enumerate(np.array_split(order,4),1):
        bins.append(dict(quartile=n,views=len(sub),predicted_proxy_median=float(np.median(flat[sub])),
            feature_error_mean=float(flat_error[sub].mean()),feature_error_median=float(np.median(flat_error[sub])),
            clean_views=int((flat_scenes[sub]=='clean').sum()),low_urban_views=int((flat_scenes[sub]=='practical_low_urban').sum())))
    pred_hi=np.argmax(predicted,axis=0);true_hi=np.argmax(truth,axis=0);index=np.arange(512)
    valid=(np.max(predicted,axis=0)>np.min(predicted,axis=0))&(np.max(truth,axis=0)>np.min(truth,axis=0))
    return dict(spearman_predicted_vs_simulated_proxy=spearman(predicted.ravel(),truth.ravel()),
        spearman_predicted_proxy_vs_feature_error=spearman(predicted.ravel(),error.ravel()),
        spearman_simulated_proxy_vs_feature_error=spearman(truth.ravel(),error.ravel()),
        same_record_max_view_agreement=dict(records=int(valid.sum()),matches=int((pred_hi[valid]==true_hi[valid]).sum())),
        predicted_within_record=within_record(predicted,error),simulated_within_record=within_record(truth,error),
        predicted_proxy_quartiles=bins)


def worker(args):
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    device='cuda:0';started=time.perf_counter();validate_source(read(args.inputs));source,roles=trial.source(args);selected=panel.select_rows(roles)
    source_run=args.root/'runs'/PANEL;done=read(source_run/'complete.json');previous=read(source_run/'results.json')
    if (done['status'],done['records'],done['views'])!=('ARTIFACTS_COMPLETE',512,3):raise ValueError('Fixed completed source panel required')
    if any(previous[k] for k in ('model_updates','optimizer_updates','TX_labels_read','V_reads','OE_reads','target_inputs')) or previous['logits_called']:
        raise ValueError('Source feature panel legality required')
    if (previous['source_G_run'],previous['source_G_commit'])!=(execution.PARENT_RUN,execution.PARENT_COMMIT):raise ValueError('Fixed source G provenance required')
    with np.load(source_run/'features.npz',allow_pickle=False) as f:
        clean=f['clean'].astype(np.float64);restored=f['restored'].astype(np.float64);scale=f['scale'].astype(np.float64)
        prev_roles=f['roles'].copy();prev_scenes=f['scenes'].copy()
    if (clean.shape,restored.shape,scale.shape,prev_scenes.shape)!=((512,160),(3,512,160),(160,),(3,512)):
        raise ValueError('Exact frozen feature panel required')
    if not np.array_equal(roles[selected],prev_roles) or not np.isfinite(clean).all() or not np.isfinite(restored).all() or not np.isfinite(scale).all() or (scale<=0).any():
        raise ValueError('Finite aligned original source features required')
    models,metadata=execution.load_models(args.root,device);model=models['fixed_source_E30'];del models
    model.eval().requires_grad_(False)
    if metadata['F0_identity']!=previous['F0_identity']:raise ValueError('Same original F0 provenance required')
    state={k:v.detach().clone() for k,v in model.state_dict().items()}
    def read_draw(draw):
        folder=source/'epochs'/f'{draw:02d}';info=read(folder/'complete.json')
        if (info['count'],info['TX_labels_read'],info['target_reads'])!=(63000,0,0):raise ValueError('Original source draw required')
        with np.load(folder/'groups.npz',allow_pickle=False) as f:
            if not np.array_equal(f['scene'][selected],prev_scenes[draw-1]):raise ValueError('Original source scenes must align')
        x=np.load(folder/'received.npy',mmap_mode='r',allow_pickle=False);v=np.load(folder/'variance.npy',mmap_mode='r',allow_pickle=False)
        try:y=np.asarray(x[selected].copy(),dtype=np.float32);truth=np.asarray(v[selected].copy(),dtype=np.float64)
        finally:x._mmap.close();v._mmap.close()
        power=np.mean(y.astype(np.float64)**2,axis=(1,2))
        if not np.isfinite(y).all() or not np.isfinite(truth).all() or (truth<=0).any() or (power<=0).any():raise ValueError('Finite source noise and input power required')
        return y,truth,power
    y,truth,power=read_draw(1)
    with torch.no_grad():
        probe=model.physical.estimate(tensor(y[:2],device),dense=False)['variance']
        if not torch.isfinite(probe).all() or (probe<=0).any():raise ValueError('Actual source G noise head smoke failed')
    out=args.root/'runs'/RUN
    write(out/'smoke.json',dict(status='PASS',source_rows=2,actual_source_G=True,original_feature_panel=True,
        TX_labels_read=0,V_reads=0,OE_reads=0,target_inputs=0,optimizer_updates=0,F0_forwards=0))
    print('SOURCE_PREDICTED_NOISE_SMOKE_PASS',flush=True)
    predictions=[];truths=[];powers=[]
    for draw in (1,2,3):
        if draw!=1:y,truth,power=read_draw(draw)
        chunks=[]
        with torch.no_grad():
            for i in range(0,512,32):
                estimate=model.physical.estimate(tensor(y[i:i+32].copy(),device),dense=False)['variance']
                if not torch.isfinite(estimate).all() or (estimate<=0).any():raise ValueError('Finite positive estimated noise required')
                chunks.extend(estimate.detach().cpu().tolist())
        predictions.append(np.asarray(chunks,dtype=np.float64));truths.append(truth);powers.append(power)
        print('SOURCE_PREDICTED_NOISE_DRAW_COMPLETE',draw,512,flush=True)
    predictions=np.stack(predictions);truths=np.stack(truths);powers=np.stack(powers)
    predicted_proxy=predictions/powers;true_proxy=truths/powers;error=np.mean((restored-clean[None])**2/scale[None,None,:],axis=2)
    summary=summarize(predicted_proxy,true_proxy,error,prev_scenes)
    if any(not torch.equal(v,model.state_dict()[k]) for k,v in state.items()):raise ValueError('Frozen G state changed')
    with (out/'noise_fields.npz').open('xb') as f:np.savez_compressed(f,predicted_proxy=predicted_proxy,
        simulated_proxy=true_proxy,feature_error=error,scenes=prev_scenes,roles=prev_roles)
    write(out/'results.json',dict(status='SOURCE_PREDICTED_NOISE_RANK_COMPLETE',source_records=512,views=1536,
        roles={'L_s':256,'U_s':256},source_G_run=execution.PARENT_RUN,source_G_commit=execution.PARENT_COMMIT,
        F0_identity=previous['F0_identity'],F0_forwards=0,G_noise_head_rows=1536,simulation_variance_deployable=False,
        estimated_proxy='G predicted noise variance / mean current received real-IQ squared amplitude',
        simulator_proxy='source simulator variance / mean current received real-IQ squared amplitude',
        feature_error='mean squared restored-minus-clean F0 feature / source feature variance',
        source_in_sample=True,source_sample_representative=False,summary=summary,
        optimizer_updates=0,model_updates=0,TX_labels_read=0,V_reads=0,OE_reads=0,target_inputs=0,
        no_target_model_selection=True,seconds=time.perf_counter()-started))
    write(out/'complete.json',dict(status='ARTIFACTS_COMPLETE',source_only=True,records=512,views=3))
    print('SOURCE_PREDICTED_NOISE_RANK_COMPLETE',flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['run','worker']);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--inputs',type=Path,required=True);p.add_argument('--cache',type=Path,required=True);p.add_argument('--commit',required=True)
    args=p.parse_args()
    if (Path.cwd()/'release_commit.txt').read_text().strip()!=args.commit:raise ValueError('Release identity mismatch')
    if args.mode=='run':execution.RUN,execution.MODULE=RUN,MODULE;execution.controller(args)
    else:worker(args)


if __name__=='__main__':main()
