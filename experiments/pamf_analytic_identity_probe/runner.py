import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import numpy as np
import torch
from . import design as d
from .analysis import fit_moments,representations,class_similarities,contrasts,permuted_queries,null_control
from tools.analyze_pamf_source_cross_record import reference_pool,summarize
from experiments.pamf_analytic_unrolled_restore.descriptors import AnalyticViews
from experiments.pamf_fcrossrx_readout.source_diagnostic import source_reference
from experiments.pamf_restoration.data import write_json
from experiments.pamf_restoration.runner import array_tensor,tensor_array,code_commit,read_json

ROOT=Path(__file__).resolve().parents[2]
PATH_ARGS=('runs_root','parent_action','source_cache','output')


def command(args,mode):
    result=[sys.executable,'-u','-m','experiments.pamf_analytic_identity_probe.runner',mode]
    for key in (*PATH_ARGS,'expected_commit'):result+=['--'+key.replace('_','-'),str(getattr(args,key))]
    return result


def inputs(args):
    parent=args.runs_root/d.PARENT
    if read_json(parent/'launch.json').get('commit')!=d.PARENT_COMMIT:raise ValueError('Fixed original source-selected R17 required')
    row=parent/d.CANDIDATE;done=read_json(row/'complete.json')
    expected=dict(status='ANALYZED',candidate=d.CANDIDATE,role='source',all_predictions_before_truth=True,training_TX_F_access=False)
    if any(done.get(k)!=v for k,v in expected.items()):raise ValueError('Completed source-only R17 row required')
    if args.parent_action.name!=d.ACTION or read_json(args.parent_action/'launch.json').get('commit')!=d.ACTION_COMMIT:
        raise ValueError('Original source-only R07 cache required')
    panel=args.parent_action/'cache/validation';ids,source,index=source_reference(panel,args.source_cache)
    pred=row/'native_map_prior/prediction';meta=read_json(pred/'complete.json')
    required=dict(status='PREDICTIONS_COMPLETE',kind='native_map_prior',role='source',count=1800,views=list(d.VIEWS),
        readouts=['restored'],TX_truth_reads=0,state_updates=0,head_fit_updates=0)
    if any(meta.get(k)!=v for k,v in required.items()) or not np.array_equal(ids,np.load(pred/'physical_ids.npy',allow_pickle=False)):
        raise ValueError('All seven completed same-panel source IQ predictions required')
    arrays={}
    for view in d.VIEWS:
        for name,path in [('raw',panel/(view+'.npy')),('restored',pred/(view+'.restored.iq.npy'))]:
            value=np.load(path,mmap_mode='r',allow_pickle=False)
            if value.shape!=(1800,2,256) or not np.isfinite(value).all():raise ValueError('All finite source IQ required before labels')
            arrays[(view,name)]=value
    return ids,source,index,arrays


@torch.no_grad()
def extract(args):
    ids,_,_,arrays=inputs(args);cache=args.parent_action/'cache'
    info=read_json(cache/'complete.json')
    if info.get('status')!='ACTION_SOURCE_CACHE_COMPLETE' or info.get('train_count')!=63000 or info.get('TX_labels_read')!=0 or info.get('target_reads')!=0:
        raise ValueError('Original label-free LU cache required')
    clean=np.load(cache/'train/clean.npy',mmap_mode='r',allow_pickle=False);roles=np.load(cache/'train/roles.npy',allow_pickle=False)
    if clean.shape!=(63000,2,256) or roles.shape!=(63000,) or (roles=='L_s').sum()!=6300 or (roles=='U_s').sum()!=56700:
        raise ValueError('Original full LU training roles required')
    transform=AnalyticViews().eval()
    if list(transform.parameters()):raise ValueError('Only parameter-free analytic transformation allowed')
    def features(iq):return {k:tensor_array(v) for k,v in transform(array_tensor(iq,'cpu')).items()}
    calibration=fit_moments(features(clean[i:min(i+128,d.CALIBRATION_COUNT)]) for i in range(0,d.CALIBRATION_COUNT,128))
    calibration.update(role='source_LU_only',L_count=int((roles[:d.CALIBRATION_COUNT]=='L_s').sum()),
        U_count=int((roles[:d.CALIBRATION_COUNT]=='U_s').sum()),V_fit_updates=0,target_reads=0)
    if calibration['count']!=d.CALIBRATION_COUNT:raise ValueError('Complete fixed calibration required')
    write_json(args.output/'calibration.json',calibration);result={};started=time.perf_counter()
    for view in d.VIEWS:
        for name in ('raw','restored'):
            parts={k:[] for k in d.GROUPS}
            for i in range(0,len(ids),128):
                values=representations(features(arrays[(view,name)][i:i+128]),calibration['moments'])
                for key in parts:parts[key].append(values[key])
            for key in parts:result[view+'__'+name+'__'+key]=np.concatenate(parts[key])
        print(json.dumps(dict(event='ANALYTIC_VIEW_COMPLETE',view=view,count=len(ids),F_forward_calls=0)),flush=True)
    with (args.output/'analytic_features.npz').open('xb') as stream:np.savez(stream,**result)
    np.save(args.output/'physical_ids.npy',ids,allow_pickle=False)
    write_json(args.output/'feature_complete.json',dict(status='FEATURES_COMPLETE',count=1800,views=list(d.VIEWS),groups=d.GROUPS,
        readouts=['raw','restored'],TX_label_reads=0,F_weights_read=0,F_embeddings_logits_read=0,extra_restorer_predictions=0,
        analytic_transform_samples=d.CALIBRATION_COUNT+2*len(ids)*len(d.VIEWS),seconds=time.perf_counter()-started))


def load_features(args,ids):
    complete=read_json(args.output/'feature_complete.json')
    expected=dict(status='FEATURES_COMPLETE',count=1800,views=list(d.VIEWS),groups=d.GROUPS,readouts=['raw','restored'],
        TX_label_reads=0,F_weights_read=0,F_embeddings_logits_read=0,extra_restorer_predictions=0)
    if any(complete.get(k)!=v for k,v in expected.items()) or not np.array_equal(np.load(args.output/'physical_ids.npy',allow_pickle=False),ids):
        raise ValueError('All fixed analytic features required before source labels')
    keys={v+'__'+r+'__'+g for v in d.VIEWS for r in ('raw','restored') for g in d.GROUPS};values={}
    with np.load(args.output/'analytic_features.npz',allow_pickle=False) as archive:
        if set(archive.files)!=keys:raise ValueError('Every registered source analytic cell required before labels')
        for key in keys:
            value=archive[key]
            if value.shape!=(1800,d.GROUPS[key.split('__')[-1]]) or not np.isfinite(value).all():raise ValueError('Finite complete analytic cells required')
            values[key]=value
    return values


def score(args,*,label_loader=np.load):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Separate CPU source scorer required')
    ids,source,index,_=inputs(args);features=load_features(args,ids)
    # First TX label open:every analytic cell is fixed,finite and aligned above.
    labels=label_loader(args.source_cache/'V/labels.npy',allow_pickle=False)[index]
    rx,day=source.rx[index],source.day[index];permuted=permuted_queries(labels,rx,day)
    groups=[('overall','all',np.ones(len(ids),bool))]
    for dimension,values in [('TX',labels),('RX',rx),('day',day)]:
        groups.extend((dimension,str(v),values==v) for v in np.unique(values))
    records=[];negative=[];pools={}
    for group in d.GROUPS:
        clean=features['clean__raw__'+group];reference=reference_pool(clean,ids,labels,rx,day)
        pools[group]={k:reference[k] for k in ('cap','reference_count','empty_TX_cells','zero_reference_count')}
        clean_scores=class_similarities(clean,reference,rx,day);clean_contrast=contrasts(clean_scores,labels)
        for view in d.VIEWS:
            raw=features[view+'__raw__'+group];restored=features[view+'__restored__'+group]
            before_scores=class_similarities(raw,reference,rx,day);after_scores=class_similarities(restored,reference,rx,day)
            before,after=contrasts(before_scores,labels),contrasts(after_scores,labels)
            common=before['valid']&after['valid']&clean_contrast['valid']
            reducer=np.sum if group=='combined' else np.mean
            err_before=reducer((raw-clean)**2,axis=1);err_after=reducer((restored-clean)**2,axis=1)
            for dimension,value,mask in groups:
                record=dict(group=group,view=view,dimension=dimension,value=value,**summarize(before,after,clean_contrast,mask),
                    paired_descriptor_error=dict(count=int(mask.sum()),raw=float(err_before[mask].mean()),restored=float(err_after[mask].mean())))
                records.append(record)
            for name,values in [('raw',before_scores),('restored',after_scores),('clean',clean_scores)]:
                negative.append(dict(group=group,view=view,readout=name,**null_control(values,labels,permuted,common)))
    write_json(args.output/'summary.json',dict(status='ANALYZED',run=d.RUN,role='source_V',count=1800,views=list(d.VIEWS),
        candidate=d.CANDIDATE,groups=d.GROUPS,records=records,query_label_negative_controls=negative,reference_pools=pools,
        source_labels_after_all_features=True,calibration_source_LU_only=True,source_V_fit_updates=0,new_neural_training_updates=0,
        head_fit_updates=0,extra_restorer_predictions=0,F_weights_read=0,F_embeddings_logits_read=0,target_reads=0,
        runtime_decisions_changed=False,source_selection_changed=False,configuration=d.configuration(args.expected_commit),
        claim_scope=d.configuration(args.expected_commit)['claim_scope'],goal_complete=False))
    print(json.dumps(dict(event='ANALYTIC_IDENTITY_SCORED',records=len(records),negative_controls=len(negative),target_reads=0)),flush=True)


def run(args):
    if args.output.exists() or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('New immutable root and CPU execution required')
    args.output.mkdir();(args.output/'logs').mkdir()
    write_json(args.output/'launch.json',dict(pid=os.getpid(),cwd=str(ROOT),argv=sys.argv,commit=args.expected_commit,
        owner='codex/root/analytic-identity-probe',started=time.time()))
    write_json(args.output/'resolved_config.json',d.configuration(args.expected_commit))
    try:
        for mode in ('extract','score'):
            with (args.output/'logs'/(mode+'.log')).open('x',encoding='utf-8') as stream:
                process=subprocess.Popen(command(args,mode),cwd=ROOT,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES=''))
                write_json(args.output/(mode+'_launch.json'),dict(pid=process.pid,cwd=str(ROOT),argv=command(args,mode),started=time.time()))
                if process.wait():raise RuntimeError('CPU '+mode+' failed;preserve artifacts,no automatic retry')
        write_json(args.output/'complete.json',dict(status='ANALYZED',new_neural_training_updates=0,head_fit_updates=0,
            extra_restorer_predictions=0,F_weights_read=0,F_embeddings_logits_read=0,target_reads=0,goal_complete=False))
    except BaseException as error:
        write_json(args.output/'failure.json',dict(status='FAILED_TECHNICAL',error=repr(error),traceback=traceback.format_exc(),
            outputs_preserved=True,automatic_retry=False));raise


def main():
    parser=argparse.ArgumentParser(description='F-independent analytic source identity contrasts from existing positive U-Net IQ')
    parser.add_argument('mode',choices=('run','extract','score'))
    for key in PATH_ARGS:parser.add_argument('--'+key.replace('_','-'),type=Path,required=True)
    parser.add_argument('--expected-commit',required=True);args=parser.parse_args()
    for key in PATH_ARGS:setattr(args,key,getattr(args,key).resolve())
    if args.output.name!=d.RUN or code_commit()!=args.expected_commit:raise ValueError('Pinned source diagnostic run/code required')
    torch.set_num_threads(2)
    if args.mode=='run':run(args)
    elif args.mode=='extract':extract(args)
    else:score(args)


if __name__=='__main__':main()
