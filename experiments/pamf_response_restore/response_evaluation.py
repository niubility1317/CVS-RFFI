import json
import time
import numpy as np
import torch
from . import design as d
from .runtime import load_checkpoint
from .models import teacher_response
from experiments.pamf_rf_response.models import directions
from experiments.pamf_rf_response.evaluation import numpy_metrics,aggregate
from experiments.pamf_fcrossrx_readout.source_diagnostic import source_reference
from experiments.pamf_restoration.data import write_json
from experiments.pamf_restoration.models import pair
from experiments.pamf_restoration.runner import read_json,array_tensor,tensor_array
from experiments.pamf_action_restore.evaluation import teacher_action


def complete_contract(value,row):
    required=dict(status='SOURCE_RESPONSE_COMPLETE',run=d.RUN,candidate=row,count=1800,
        views=list(d.RESPONSE_VIEWS),directions=list(d.DIRECTIONS),amplitude=.01,cells=12,
        TX_truth_reads=0,F_access=0,target_reads=0,optimizer_updates=0,state_updates=0)
    if row not in d.ROWS or any(value.get(k)!=v for k,v in required.items()):
        raise ValueError('Every registered source response output must precede truth')


@torch.no_grad()
def predict(args):
    folder=args.parent_action/'cache/validation';ids,_,_=source_reference(folder,args.source_cache)
    if len(ids)!=1800:raise ValueError('Original complete source response panel required')
    clean=np.load(folder/'clean.npy',mmap_mode='r',allow_pickle=False)
    model,payload=load_checkpoint(args.output/args.row/d.KIND/'final.pt','cuda:0')
    if payload['candidate']!=args.row or (payload['epoch'],payload['updates'])!=(20,4940):raise ValueError('Actual fixed final response row required')
    model.requires_grad_(False).eval();root=args.output/args.row/'source_response';root.mkdir();timings={}
    for view in d.RESPONSE_VIEWS:
        received=np.load(folder/(view+'.npy'),mmap_mode='r',allow_pickle=False)
        known=args.parent_action/'cache/validation_actions'/view
        metadata=read_json(known/'complete.json')
        if (metadata['count']!=1800 or metadata['namespace']!='restoration/validation/draw0' or metadata['TX_labels_read']
                or metadata['F_outputs_read'] or metadata['target_reads'] or metadata['native_engine_modified']):
            raise ValueError('Original finite source-only action handle required')
        arrays={key:np.load(known/(key+'.npy'),mmap_mode='r',allow_pickle=False) for key in ('kernel','coefficients','variance')}
        out=root/view;out.mkdir();base=np.lib.format.open_memmap(out/'base.npy',mode='w+',dtype=np.float32,shape=(1800,2,256))
        outputs={}
        for name in d.DIRECTIONS:
            (out/name).mkdir();outputs[name]={key:np.lib.format.open_memmap(out/name/(key+'.npy'),mode='w+',dtype=np.float32,shape=shape)
                for key,shape in [('plus',(1800,2,256)),('minus',(1800,2,256)),('oracle',(1800,2,256)),
                                  ('observation_delta',(1800,2,256)),('CG_residual',(1800,))]}
        start=time.perf_counter()
        for begin in range(0,1800,128):
            index=np.arange(begin,min(begin+128,1800));r=array_tensor(received[index],'cuda:0');x=array_tensor(clean[index],'cuda:0')
            action=teacher_action(arrays,index,'cuda:0');variance=array_tensor(arrays['variance'][index],'cuda:0')
            value=model(r)
            if not torch.isfinite(value).all():raise FloatingPointError('Nonfinite source response base')
            base[index]=tensor_array(value);tangents=directions(x,index,.01)
            for name in d.DIRECTIONS:
                delta,valid=tangents[name];change,target,_,teacher=teacher_response(action,delta,variance,valid)
                plus,minus=model(r+pair(change)),model(r-pair(change))
                for key,value in [('plus',plus),('minus',minus),('oracle',pair(target)),('observation_delta',pair(change)),('CG_residual',teacher['CG_residual'])]:
                    if not torch.isfinite(value).all():raise FloatingPointError('Nonfinite source counterfactual response')
                    outputs[name][key][index]=tensor_array(value)
        base.flush()
        for values in outputs.values():
            for array in values.values():array.flush()
        torch.cuda.synchronize();timings[view]=time.perf_counter()-start
        print(json.dumps(dict(event='RESPONSE_SOURCE_VIEW_COMPLETE',candidate=args.row,view=view,count=1800,seconds=timings[view])),flush=True)
    np.save(root/'physical_ids.npy',ids,allow_pickle=False)
    value=dict(status='SOURCE_RESPONSE_COMPLETE',run=d.RUN,candidate=args.row,count=1800,views=list(d.RESPONSE_VIEWS),
        directions=list(d.DIRECTIONS),amplitude=.01,cells=12,TX_truth_reads=0,F_access=0,target_reads=0,optimizer_updates=0,state_updates=0,
        per_view_seconds=timings)
    complete_contract(value,args.row);write_json(root/'complete.json',value)


def load_predictions(root,ids,row):
    complete_contract(read_json(root/'complete.json'),row)
    if not np.array_equal(np.load(root/'physical_ids.npy',allow_pickle=False),ids):raise ValueError('Response IDs differ;truth stays closed')
    result={}
    for view in d.RESPONSE_VIEWS:
        value=np.load(root/view/'base.npy',mmap_mode='r',allow_pickle=False)
        if value.shape!=(1800,2,256) or not np.isfinite(value).all():raise ValueError('Missing source response;truth stays closed')
        result[view]=dict(base=value,rows={})
        for name in d.DIRECTIONS:
            result[view]['rows'][name]={}
            for key in ('plus','minus','oracle','observation_delta','CG_residual'):
                value=np.load(root/view/name/(key+'.npy'),mmap_mode='r',allow_pickle=False)
                shape=(1800,) if key=='CG_residual' else (1800,2,256)
                if value.shape!=shape or not np.isfinite(value).all():raise ValueError('Missing source counterfactual;truth stays closed')
                result[view]['rows'][name][key]=value
    return result


def score(args):
    folder=args.parent_action/'cache/validation';ids,source,index=source_reference(folder,args.source_cache)
    root=args.output/args.row/'source_response';predictions=load_predictions(root,ids,args.row)
    # All counterfactuals and standard seven-view predictions were checked by the caller.
    labels=np.load(args.source_cache/'V/labels.npy',allow_pickle=False)[index]
    clean=array_tensor(np.load(folder/'clean.npy',allow_pickle=False),'cpu')
    groups=[('overall','all',np.ones(1800,bool))]
    for dimension,values in [('TX',labels),('RX',source.rx[index]),('day',source.day[index])]:
        groups += [(dimension,str(value),values==value) for value in np.unique(values)]
    tangents=directions(clean,np.arange(1800),.01);records=[]
    for view in d.RESPONSE_VIEWS:
        variance=np.load(args.parent_action/'cache/validation_actions'/view/'variance.npy',allow_pickle=False)
        for name in d.DIRECTIONS:
            delta,valid=tangents[name];arrays=predictions[view]['rows'][name]
            target=tensor_array(pair(delta));values=numpy_metrics(arrays,predictions[view]['base'],target,tensor_array(valid).astype(bool),variance)
            central=(np.asarray(arrays['plus'],dtype=np.float64)-np.asarray(arrays['minus'],dtype=np.float64))/2
            error=((central-np.asarray(arrays['oracle'],dtype=np.float64))**2).sum((1,2))/np.maximum((target.astype(np.float64)**2).sum((1,2)),1e-20)
            np.savez(root/(view+'.'+name+'.metrics.npz'),**values,oracle_response_error=error)
            for dimension,value,mask in groups:
                observable=mask&values['observable'];n=int(observable.sum())
                records.append(dict(view=view,direction=name,dimension=dimension,value=value,
                    **aggregate(values,mask),observable_oracle_response_error_mean=float(error[observable].mean()) if n else None))
    write_json(root/'summary.json',dict(status='ANALYZED',run=d.RUN,candidate=args.row,count=1800,records=records,
        all_predictions_before_truth=True,training_TX_F_access=False,target_access=False,source_V_state_updates=0,goal_complete=False))
