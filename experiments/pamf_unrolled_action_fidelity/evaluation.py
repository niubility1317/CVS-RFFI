import os
import time
import numpy as np
import torch
from . import design as d
from .models import counterfactuals
from .runtime import source_inputs,load_model
from experiments.pamf_restoration.data import SCENES,write_json
from experiments.pamf_restoration.runner import array_tensor,tensor_array,read_json
from experiments.pamf_action_restore.evaluation import teacher_action
from experiments.pamf_fcrossrx_readout.frozen import load_reader,REFERENCE_RUN
from experiments.pamf_uncertainty_restore.evaluation import prediction_arrays,waveform
from experiments.pamf_restoration.frozen import classification_statistics
from experiments.pamf_restoration.feature_diagnostic import capped_pair_margin

VIEWS=('clean',*SCENES)


def complete_contract(value,count):
    required=dict(status='SOURCE_PREDICTIONS_COMPLETE',role='source',count=count,views=list(VIEWS),readouts=list(d.ROWS),
        TX_truth_reads=0,state_updates=0,head_fit_updates=0,target_reads=0,optimizer_updates=0,
        F_run=REFERENCE_RUN,F_seed=2026092701,original_default_head=True,oracle_not_deployable=True)
    if any(value.get(key)!=val for key,val in required.items()):
        raise ValueError('Every registered source prediction must precede truth')


@torch.no_grad()
def predict(args):
    folder,ids,_,_=source_inputs(args);model,_=load_model(args,'cuda:0')
    reader=load_reader(args.reader_folder,args.source_cache,'cuda:0').requires_grad_(False).eval()
    output=args.output/'prediction';output.mkdir()
    write_json(args.output/'gpu_ready.json',dict(pid=os.getpid(),cwd=os.getcwd(),argv=__import__('sys').argv,GPU=os.environ.get('CUDA_VISIBLE_DEVICES')))
    print('SOURCE_ORACLE_GPU_READY',flush=True);timings={};started_all=time.perf_counter()
    for view in VIEWS:
        observed=np.load(folder/(view+'.npy'),mmap_mode='r',allow_pickle=False)
        known=args.parent_action/'cache/validation_actions'/view
        metadata=read_json(known/'complete.json')
        if (metadata['count']!=len(ids) or metadata['namespace']!='restoration/validation/draw0'
                or metadata['TX_labels_read'] or metadata['F_outputs_read'] or metadata['target_reads'] or metadata['native_engine_modified']):
            raise ValueError('Matched original source-only action handle required')
        arrays={key:np.load(known/(key+'.npy'),mmap_mode='r',allow_pickle=False) for key in ('kernel','coefficients','variance')}
        if (observed.shape!=(len(ids),2,256) or arrays['kernel'].shape!=(len(ids),2,64)
                or arrays['coefficients'].shape!=(len(ids),4,256) or arrays['variance'].shape!=(len(ids),)):
            raise ValueError('Source action/observation coverage differs')
        outputs={name:{key:np.lib.format.open_memmap(output/(view+'.'+name+'.'+key+'.npy'),mode='w+',dtype=np.float32,shape=shape)
            for key,shape in (('iq',observed.shape),('features',(len(ids),160)),('logits',(len(ids),6)))} for name in d.ROWS}
        started=time.perf_counter()
        for begin in range(0,len(ids),128):
            index=np.arange(begin,min(begin+128,len(ids)))
            r=array_tensor(observed[index],'cuda:0');known_action=teacher_action(arrays,index,'cuda:0')
            values=counterfactuals(model,r,known_action,array_tensor(arrays['variance'][index],'cuda:0'))
            for name,value in values.items():
                feature=reader.features(value);logits=reader.classify_features(feature)
                if feature.shape!=(len(index),160) or logits.shape!=(len(index),6) or not torch.isfinite(feature).all() or not torch.isfinite(logits).all():
                    raise ValueError('Finite originalF/defaulthead source outputs required')
                for key,item in (('iq',value),('features',feature),('logits',logits)):outputs[name][key][index]=tensor_array(item)
        for row in outputs.values():
            for array in row.values():array.flush()
        torch.cuda.synchronize();timings[view]=time.perf_counter()-started
        print(__import__('json').dumps(dict(event='SOURCE_ACTION_VIEW_COMPLETE',view=view,count=len(ids),readouts=list(d.ROWS))),flush=True)
    np.save(output/'physical_ids.npy',ids,allow_pickle=False)
    complete=dict(status='SOURCE_PREDICTIONS_COMPLETE',role='source',count=len(ids),views=list(VIEWS),readouts=list(d.ROWS),
        TX_truth_reads=0,state_updates=0,head_fit_updates=0,target_reads=0,optimizer_updates=0,F_run=REFERENCE_RUN,F_seed=2026092701,
        original_default_head=True,oracle_not_deployable=True,per_view_seconds=timings,seconds=time.perf_counter()-started_all)
    complete_contract(complete,len(ids));write_json(output/'complete.json',complete)


def load_predictions(path,ids):
    complete_contract(read_json(path/'complete.json'),len(ids))
    if not np.array_equal(np.load(path/'physical_ids.npy',allow_pickle=False),ids):raise ValueError('Source prediction IDs differ')
    values={}
    for view in VIEWS:
        values[view]={}
        for name in d.ROWS:
            values[view][name]={}
            for key,shape in (('iq',(len(ids),2,256)),('features',(len(ids),160)),('logits',(len(ids),6))):
                array=np.load(path/(view+'.'+name+'.'+key+'.npy'),mmap_mode='r',allow_pickle=False)
                if array.shape!=shape or not np.isfinite(array).all():raise ValueError('Incomplete/nonfinite source prediction;truth staysclosed')
                values[view][name][key]=array
    return values


def interpret(summary):
    required=dict(status='ANALYZED',role='source',count=1800,target_access=False,all_predictions_before_truth=True,
        F_run=REFERENCE_RUN,F_seed=2026092701,original_default_head=True,optimizer_updates=0,head_fit_updates=0,
        source_V_state_updates=0,oracle_not_deployable=True,theoretical_upper_bound=False)
    if any(summary.get(key)!=value for key,value in required.items()) or list(summary['views'])!=list(VIEWS):
        raise ValueError('Complete source-only interpretation required')
    def acc(view,name,dimension='overall'):
        rows=[r for r in summary['records'] if r['view']==view and r['readout']==name and r['dimension']==dimension]
        result={r['value']:r['accuracy'] for r in rows}
        if len(result)!=len(rows) or not result or any(not np.isfinite(v) for v in result.values()):raise ValueError('Unique finite source groups required')
        return result
    means={name:sum(summary['views'][v][name]['waveform']['iq_nmse'] for v in VIEWS[1:])/6 for name in (*d.REFERENCES,*d.ROWS)}
    if any(not np.isfinite(v) or v<0 for v in means.values()) or means['native_estimated']<=0:raise ValueError('Finite nonnegative source errors required')
    rows={}
    for name in d.ROWS:
        low=acc('practical_low_urban',name)['all']-acc('practical_low_urban','native_estimated')['all']
        mid=acc('practical_mid_urban',name)['all']-acc('practical_mid_urban','native_estimated')['all']
        clean=acc('clean','identity')['all']-acc('clean',name)['all']
        tx=acc('practical_low_urban',name,'TX');old=acc('practical_low_urban','native_estimated','TX')
        if set(tx)!={str(i) for i in range(6)} or set(tx)!=set(old):raise ValueError('All six TX required')
        rows[name]=dict(mean6_NMSE=means[name],NMSE_ratio=means[name]/means['native_estimated'],lowurban_gain_pp=low,midurban_gain_pp=mid,
            clean_drop_vs_original_pp=clean,worst_lowurban_TX_delta_pp=min(tx[key]-old[key] for key in tx))
    primary=rows['known_action_simvar']
    supported=(primary['NMSE_ratio']<=.95 and primary['lowurban_gain_pp']>=1. and primary['midurban_gain_pp']>=.5
        and primary['clean_drop_vs_original_pp']<=.1 and primary['worst_lowurban_TX_delta_pp']>=-1.)
    return dict(status='SOURCE_DIAGNOSTIC_COMPLETE',supports_operator_priority=bool(supported),comparisons=rows,mean6_NMSE=means,
        thresholds=d.configuration()['source_rule'],oracle_not_deployable=True,theoretical_upper_bound=False,source_only=True,
        target_promotion=False,target_reads=0,target_feedback=False,goal_complete=False)


def score(args):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Independent CPU truth scorer required')
    folder,ids,source,index=source_inputs(args)
    new=load_predictions(args.output/'prediction',ids)
    refs=prediction_arrays(args.reference_run/'reference_controls/prediction',ids,'reference_controls','source')
    # First TX truth and paired clean open, after every registered output is complete.
    labels=np.load(args.source_cache/'V/labels.npy',allow_pickle=False)[index]
    clean=np.load(folder/'clean.npy',mmap_mode='r',allow_pickle=False);rx=source.rx[index];day=source.day[index]
    clean_feature=refs['clean']['identity']['features'];clean_unit=clean_feature/np.maximum(np.linalg.norm(clean_feature,axis=1,keepdims=True),1e-8)
    records=[];views={}
    for view in VIEWS:
        values=dict(identity=refs[view]['identity'],coarse=refs[view]['coarse'],**new[view])
        before=values['identity']['logits'].argmax(1);views[view]={}
        for name,arrays in values.items():
            metrics=waveform(arrays['iq'],clean);decision=arrays['logits'].argmax(1)
            groups=[('overall','all',np.ones(len(ids),bool))]
            for dimension,values_group in (('TX',labels),('RX',rx),('day',day)):
                groups += [(dimension,str(value),values_group==value) for value in np.unique(values_group)]
            for dimension,value,mask in groups:
                records.append(dict(view=view,readout=name,dimension=dimension,value=value,
                    **classification_statistics(labels[mask],decision[mask],before[mask]),iq_nmse=float(metrics['iq_nmse'][mask].mean())))
            feature=arrays['features'];unit=feature/np.maximum(np.linalg.norm(feature,axis=1,keepdims=True),1e-8)
            views[view][name]=dict(waveform={k:float(v.mean()) for k,v in metrics.items()},
                paired_clean_feature_cosine_distance=float((1-(unit*clean_unit).sum(1)).mean()),paired_clean_vs_other_TX=capped_pair_margin(unit,clean_unit,labels,512))
    summary=dict(status='ANALYZED',role='source',count=len(ids),views=views,records=records,F_run=REFERENCE_RUN,F_seed=2026092701,
        original_default_head=True,all_predictions_before_truth=True,target_access=False,optimizer_updates=0,head_fit_updates=0,
        source_V_state_updates=0,oracle_not_deployable=True,theoretical_upper_bound=False,goal_complete=False)
    write_json(args.output/'summary.json',summary);write_json(args.output/'interpretation.json',interpret(summary))
