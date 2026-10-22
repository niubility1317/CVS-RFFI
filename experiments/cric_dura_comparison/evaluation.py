"""All declared readouts/views first; independent CPU truth-last afterwards."""
import argparse
from pathlib import Path
import time
import numpy as np
import torch
from .io import read, write, tensor

VIEWS = ('clean','practical_high','practical_mid','practical_low_suburban',
         'practical_high_urban','practical_mid_urban','practical_low_urban')
DATES = ('2021_03_01','2021_03_08','2021_03_15','2021_03_23')


def reject_truth(value):
    if isinstance(value,dict):
        if set(value)&{'truth','target_truth','labels','label'}:raise ValueError('Predictor rejects truth')
        for v in value.values():reject_truth(v)
    elif isinstance(value,list):
        for v in value:reject_truth(v)


def handle(root, classes):
    root=Path(root);m=read(root/'manifest.json')
    if m['status']!='VALIDATED_ONCE' or m['classes']!=classes or m['scenes']!=list(VIEWS[1:]) or m['source_target_disjoint'] is not True:
        raise ValueError('Validated received-IQ contract differs')
    with np.load(root/'index.npz',allow_pickle=False) as index:
        ids=index['ids'].copy();positions=index['positions'].copy() if 'positions' in index.files else None
    if len(ids)!=m['count'] or len(set(ids.tolist()))!=len(ids) or not len(ids):raise ValueError('Prediction ID coverage differs')
    if positions is not None and (positions.shape!=ids.shape or positions.dtype.kind not in 'iu'):raise ValueError('Invalid wrapper positions')
    return m,ids,positions


def tasks(c):
    return ['main','source_unseen_date'] if c['protocol_id']=='chrono' else ['main']


def predict(c, device='cuda:0'):
    reject_truth(c)
    if sorted(c['tests'])!=sorted(tasks(c)) or not c['readouts'] or 'raw' not in c['readouts']:
        raise ValueError('All declared tasks and raw/G readouts required')
    from .bootstrap import activate,numerical_policy
    activate(c);numerical_policy()
    from .frozen_inference import load_F,load_original_F,load_G
    from experiments.pamf_stage2_inference.engine import InferenceEngine
    contract=read(c['source_contract'])
    if c.get('original_arm'):
        reader=load_original_F(c['model_seed'],contract,device,c['original_arm'])
    else:
        reader,source=load_F(c['source_output'],contract,device,require_no_unlabeled=c.get('strict_supervised_system',False))
        if (source['run_id'],source['row_id'],source['model_seed'])!=(c.get('source_run_id',c['run_id']),c.get('source_row_id',c['row_id']),c['model_seed']):raise ValueError('Different F source row')
        if c.get('strict_supervised_system'):
            from .supervised_extension import validate_supervised_source
            validate_supervised_source(source, c['readouts'])
    models={};calls={}
    for name,spec in c['readouts'].items():
        if name=='raw':continue
        models[name]=load_G(spec,contract,device)
        if not getattr(models[name],'_cd_eager_variant',False):
            calls[name]=InferenceEngine(models[name],reader,reuse=True,graph=True,native=False)
    out=Path(c['output_root']);out.mkdir(parents=True,exist_ok=False)
    if c.get('source_physics_diagnostics'):
        from .source_physics import run as source_physics
        source_physics(reader,c,out/'source_physics.json',device)
    if c.get('source_cache'):
        actual=np.load(Path(c['source_cache'])/'L_s_x.npy',mmap_mode='r',allow_pickle=False)[:2]
        sample=tensor(actual,device)
        with torch.inference_mode():
            for name,call in calls.items():
                optimized=call(sample)[0];reference=models[name](sample)
                if not torch.allclose(optimized,reference,atol=1e-5,rtol=1e-5):
                    raise ValueError('Frozen source graph/native IQ differ')
        write(out/'source_execution_check.json',dict(status='PASS',source_samples=2,target_reads=0,graph_native_tolerance=1e-5))
    completed={}
    for task in tasks(c):
        manifest,ids,positions=handle(c['tests'][task]['views_root'],contract['classes'])
        reused_raw=None
        if c.get('raw_predictions_from'):
            previous=Path(c['raw_predictions_from'])
            finished=read(previous/'complete.json')
            if (finished.get('status')!='ALL_PREDICTIONS_COMPLETE' or finished.get('truth_read') is not False
                    or finished.get('run_id')!=c['source_run_id'] or finished.get('row_id')!=c['source_row_id']):
                raise ValueError('Frozen raw reuse belongs to another F row')
            with np.load(previous/task/'predictions.npz',allow_pickle=False) as archive:
                if not np.array_equal(archive['ids'],ids):raise ValueError('Frozen raw physical IDs differ')
                reused_raw={view:archive['raw__'+view].copy() for view in VIEWS}
        if task=='source_unseen_date' and manifest.get('diagnostic_unseen_date') is not True:raise ValueError('Unseen-date diagnostic handle differs')
        values={};seconds={}
        for view in VIEWS:
            path=manifest['clean_ref'] if view=='clean' else str(Path(manifest.get('observations_root',c['tests'][task]['views_root']))/(view+'.npy'))
            array=np.load(path,mmap_mode='r',allow_pickle=False)
            if array.ndim!=3 or array.shape[1:]!=(2,256) or (positions is None and len(array)!=len(ids)):
                raise ValueError('Received-IQ shape differs')
            if positions is not None and (positions.min()<0 or positions.max()>=len(array)):raise ValueError('Invalid wrapper coverage')
            result={name:[] for name in c['readouts']}
            tic=time.perf_counter()
            with torch.inference_mode():
                for start in range(0,len(ids),128):
                    selected=slice(start,start+128) if positions is None else positions[start:start+128]
                    x=tensor(np.ascontiguousarray(array[selected],dtype=np.float32),device)
                    logits={} if reused_raw is not None else {'raw':reader.classify_features(reader.features(x))}
                    for name,call in calls.items():logits[name]=call(x)[3]
                    for name,model in models.items():
                        if name not in calls:
                            restored=model(x)
                            logits[name]=reader.classify_features(reader.features(restored))
                    for name,value in logits.items():
                        if value.shape!=(len(x),len(contract['classes'])) or not bool(torch.isfinite(value).all()):raise FloatingPointError('Invalid frozen default-head output')
                        result[name].extend(value.argmax(1).cpu().tolist())
            if reused_raw is not None:result['raw']=reused_raw[view].tolist()
            torch.cuda.synchronize()
            seconds[view]=time.perf_counter()-tic
            for name,v in result.items():values[name+'__'+view]=np.asarray(v,np.int64)
            print('CRIC_DURA_PREDICT',c['row_id'],task,view,len(ids),flush=True)
        folder=out/task;folder.mkdir()
        np.savez(folder/'predictions.npz',ids=ids,**values)
        write(folder/'complete.json',dict(status='PREDICTIONS_COMPLETE',count=len(ids),views=list(VIEWS),readouts=list(c['readouts']),truth_read=False,query_fit=False,seconds=seconds))
        completed[task]=dict(count=len(ids),views=list(VIEWS),readouts=list(c['readouts']))
    write(out/'complete.json',dict(status='ALL_PREDICTIONS_COMPLETE',run_id=c['run_id'],row_id=c['row_id'],
        tests=completed,views=list(VIEWS),readouts=list(c['readouts']),truth_read=False,query_fit=False))


def preflight(c):
    root=Path(c['prediction_output']);done=read(root/'complete.json')
    if done.get('status')!='ALL_PREDICTIONS_COMPLETE' or done.get('row_id')!=c['row_id'] or done.get('run_id')!=c['run_id'] or done.get('truth_read') is not False or done.get('query_fit') is not False:
        raise ValueError('Truth closed: incomplete prediction')
    if sorted(c['tests'])!=sorted(tasks(c)) or sorted(done['tests'])!=sorted(tasks(c)) or done['views']!=list(VIEWS) or done['readouts']!=c['readouts']:
        raise ValueError('Truth closed: incomplete readout/task/view matrix')
    prepared={};names={name+'__'+view for name in c['readouts'] for view in VIEWS}
    for task in tasks(c):
        manifest,ids,_=handle(c['tests'][task]['views_root'],c['classes'])
        with np.load(root/task/'predictions.npz',allow_pickle=False) as p:
            if set(p.files)!=names|{'ids'} or not np.array_equal(p['ids'],ids):raise ValueError('Truth closed: coverage/order differs')
            values={key:p[key].copy() for key in names}
        for value in values.values():
            if value.shape!=ids.shape or value.dtype.kind not in 'iu' or value.min()<0 or value.max()>=len(c['classes']):raise ValueError('Truth closed: invalid class decision')
        prepared[task]=(ids,values)
    return prepared


def metrics(labels,predicted,classes):
    cm=np.bincount(classes*labels+predicted,minlength=classes**2).reshape(classes,classes)
    denominator=cm.sum(0)+cm.sum(1)
    f1=np.divide(2*cm.diagonal(),denominator,out=np.zeros(classes),where=denominator!=0)
    return dict(count=len(labels),accuracy=float(cm.trace()/len(labels)),macro_f1=float(f1.mean()),confusion=cm.tolist())


def score(c):
    prepared=preflight(c)  # All tasks/readouts/views validated before truth IO.
    results=[];pairs=[];classes=len(c['classes'])
    for task,(ids,values) in prepared.items():
        truth=read(c['tests'][task]['truth'])
        y=np.asarray([truth[str(i)]['label'] for i in ids],np.int64)
        rx=np.asarray([str(truth[str(i)]['receiver']) for i in ids]);day=np.asarray([str(truth[str(i)]['day']) for i in ids])
        if y.min()<0 or y.max()>=classes:raise ValueError('Truth class map differs')
        visibility=np.asarray(['seen_dates' if d in {DATES[j] for j in c['source_days']} else 'unseen_dates' for d in day])
        masks=[('overall','ALL',np.ones(len(y),bool))]
        for dim,group in [('receiver',rx),('date',day),('date_visibility',visibility),('transmitter',y.astype(str))]:
            masks.extend((dim,str(k),group==k) for k in sorted(set(group)))
        masks.extend(('receiver_date',r+'|'+d,(rx==r)&(day==d)) for r in sorted(set(rx)) for d in sorted(set(day)) if ((rx==r)&(day==d)).any())
        for name in c['readouts']:
            for view in VIEWS:
                p=values[name+'__'+view]
                for dim,group,mask in masks:
                    results.append(dict(test=task,readout=name,view=view,dimension=dim,stratum=group,**metrics(y[mask],p[mask],classes)))
                if name!='raw':
                    raw=values['raw__'+view];a=raw==y;b=p==y
                    for dim,group,mask in masks:
                        repairs=int((~a&b&mask).sum());damage=int((a&~b&mask).sum())
                        pairs.append(dict(test=task,readout=name,view=view,dimension=dim,stratum=group,count=int(mask.sum()),repairs=repairs,damage=damage,net=repairs-damage))
    out=Path(c['output_root']);out.mkdir(parents=True,exist_ok=False)
    write(out/'scores.json',dict(status='SCORED',run_id=c['run_id'],row_id=c['row_id'],model_seed=c['model_seed'],results=results,paired_raw=pairs,truth_last=True,target_feedback_forbidden=True))
    write(out/'complete.json',dict(status='SCORED_COMPLETE',run_id=c['run_id'],row_id=c['row_id'],tests=tasks(c),views=list(VIEWS),readouts=c['readouts']))
    print('CRIC_DURA_SCORED',c['row_id'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['predict','score']);p.add_argument('--config',required=True)
    a=p.parse_args();globals()[a.action](read(a.config))
