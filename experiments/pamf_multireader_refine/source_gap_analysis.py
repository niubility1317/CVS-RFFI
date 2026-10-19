"""Post-scoring source-only associations, without model fitting or target access."""
import argparse,json
from pathlib import Path
import numpy as np
from .evaluation import completed_predictions
from experiments.pamf_fcrossrx_readout.source_diagnostic import source_reference,VIEWS

REGIONS={'full256':slice(0,256),'prefix232':slice(0,232),'tail24':slice(232,256)}


def errors(estimate,clean):
    if estimate.shape!=clean.shape or clean.ndim!=3 or clean.shape[1:]!=(2,256):
        raise ValueError('Matched same-physical 256-point IQ required')
    if not np.isfinite(estimate).all() or not np.isfinite(clean).all():raise ValueError('Finite IQ required')
    values={}
    for name,region in REGIONS.items():
        target=np.asarray(clean[:,:,region],np.float64)
        delta=np.asarray(estimate[:,:,region],np.float64)-target
        values[name]=np.square(delta).mean((1,2))/np.square(target).mean((1,2)).clip(1e-4)
    return values


def summarize(values,labels,prediction,clean_prediction):
    labels=np.asarray(labels);prediction=np.asarray(prediction);clean_prediction=np.asarray(clean_prediction)
    n=len(labels)
    if any(v.shape!=(n,) for v in values.values()) or prediction.shape!=(n,) or clean_prediction.shape!=(n,):
        raise ValueError('Matched source labels and fixed predictions required')
    valid=clean_prediction==labels
    masks={'all':np.ones(n,bool),'clean_correct':valid,'restored_correct_given_clean_correct':valid&(prediction==labels),
        'restored_wrong_given_clean_correct':valid&(prediction!=labels)}
    groups={}
    for name,mask in masks.items():
        count=int(mask.sum());regions={}
        for region,value in values.items():
            selected=value[mask]
            regions[region]=None if not count else dict(mean=float(selected.mean()),median=float(np.median(selected)),p90=float(np.quantile(selected,.9)))
        groups[name]=dict(count=count,regions=regions)
    return groups


def analyze(cache,source_cache,control,rows):
    reference=cache/'validation';ids,_,index=source_reference(reference,source_cache)
    paths={'single_teacher':control/'joint_posterior',**rows}
    # Every original row already passed all-seven source prediction checks before V truth.
    packets={name:completed_predictions(path,ids) for name,path in paths.items()}
    labels=np.load(source_cache/'V/labels.npy',allow_pickle=False)[index]
    clean=np.load(reference/'clean.npy',allow_pickle=False)
    clean_prediction=packets['single_teacher']['clean']['identity']
    records=[];pooled={name:{region:[] for region in REGIONS} for name in paths};pooled_labels=[];pooled_clean=[];pooled_pred={name:[] for name in paths}
    for view in VIEWS:
        for name,path in paths.items():
            array=np.load(path/'prediction'/(view+'_refined.npy'),allow_pickle=False)
            values=errors(array,clean);prediction=packets[name][view]['refined']
            records.append(dict(view=view,row=name,groups=summarize(values,labels,prediction,clean_prediction)))
            if view!='clean':
                for region in REGIONS:pooled[name][region].append(values[region])
                pooled_pred[name].append(prediction)
        if view!='clean':pooled_labels.append(labels);pooled_clean.append(clean_prediction)
    combined={name:summarize({r:np.concatenate(v) for r,v in values.items()},np.concatenate(pooled_labels),
        np.concatenate(pooled_pred[name]),np.concatenate(pooled_clean)) for name,values in pooled.items()}
    return dict(status='ANALYZED',source_only=True,target_reads=0,new_model_forwards=0,parameter_updates=0,count=len(ids),
        records=records,pooled_six_views=combined,
        limitation='Post-hoc source association conditional on the raw clean F1 decision; no causal or target-performance claim')


def main():
    p=argparse.ArgumentParser()
    for key in ('cache','source-cache','control','source-run'):p.add_argument('--'+key,type=Path,required=True)
    args=p.parse_args();rows={name:args.source_run/name for name in ('teacher_mean','teacher_worst')}
    print(json.dumps(analyze(args.cache,args.source_cache,args.control,rows),allow_nan=False))


if __name__=='__main__':main()
