"""Independent truth-last scoring of fixed stage-scoped episode predictions."""
from pathlib import Path
import numpy as np
from .io import read,write,require
from .episodic_io import query_complement


def metrics(labels,predicted,count):
    require(len(labels)==len(predicted) and len(labels)>0,'Invalid metric arrays')
    require((labels>=0).all() and (labels<count).all(),'Truth outside registered classes')
    require((predicted>=0).all() and (predicted<count).all(),'Prediction outside registered classes')
    cm=np.bincount(np.asarray(labels,dtype=np.int64)*count+predicted,minlength=count*count).reshape(count,count)
    den=cm.sum(1);recall=np.divide(cm.diagonal(),den,out=np.zeros(count),where=den>0)
    fden=den+cm.sum(0);f1=np.divide(2*cm.diagonal(),fden,out=np.zeros(count),where=fden>0)
    old=float(recall[:6][den[:6]>0].mean()) if (den[:6]>0).any() else None
    new=float(recall[6:][den[6:]>0].mean()) if count>6 and (den[6:]>0).any() else None
    return dict(count=len(labels),correct=int(cm.trace()),accuracy=float(cm.trace()/len(labels)),macro_accuracy=float(recall[den>0].mean()),
                macro_f1=float(f1[den>0].mean()),class_floor=float(recall[den>0].min()),old_macro_accuracy=old,new_macro_accuracy=new,
                H_macro=(2*old*new/(old+new) if old+new else 0.) if old is not None and new is not None else None,
                class_counts=den.tolist(),class_recall=recall.tolist(),confusion=cm.tolist())


def run(rt,job):
    prediction=Path(job['prediction']);root=Path(job['output']);complete=read(prediction/'complete.json')
    require(complete['status']=='PREDICTIONS_COMPLETE' and complete['truth_read'] is False,'Prediction must complete before truth')
    root.mkdir(parents=True,exist_ok=False);payload=read(prediction/'predictions.json');cap=Path(rt['data_root'])/payload['capsule'];m=read(cap/'manifest.json')
    require(m['capsule_id']==payload['capsule_id']==complete['capsule_id'],'Wrong data/prediction binding')
    specs={s['split_id']:s for s in (read(p) for p in (cap/'splits').glob('*.json'))}
    pool,e,r=m['physical_pool'],m['episode_id'],m['receiver']
    with np.load(Path(rt['data_root'])/'score_only'/pool/e/(r+'.npz')) as z:truth={k:z[k] for k in ('labels','days','scene_indices')}
    result=[];seen=set()
    for item in payload['rows']:
        s=specs[item['split_id']];key=item['split_id'],item['stage'],item['input_path'],item['arm'];require(key not in seen,'Duplicate prediction row');seen.add(key)
        reg=np.load(cap/'stage_rows'/f'N{s["new_count"]:02d}.npy');expected=query_complement(reg,np.asarray(s['support_indices']))
        with np.load(prediction/item['prediction']) as z:
            query=z['query_indices'];pred=z['predicted_indices'];restricted=z['old_restricted_indices'] if 'old_restricted_indices' in z else None
        require(np.array_equal(query,expected) and len(pred)==s['query_count'],'Wrong prediction query order/count')
        labels=truth['labels'][query];count=len(s['registered_classes']);require((labels<count).all(),'Future class in query')
        value=dict(item,physical_pool=pool,episode=e,receiver=r,model_seed=payload['model_seed'],metrics=metrics(labels,pred,count))
        value.pop('audit',None)
        value['by_day']={str(d):metrics(labels[truth['days'][query]==d],pred[truth['days'][query]==d],count) for d in np.unique(truth['days'][query])}
        value['by_scene']={str(d):metrics(labels[truth['scene_indices'][query]==d],pred[truth['scene_indices'][query]==d],count) for d in np.unique(truth['scene_indices'][query])}
        if restricted is not None:
            old=labels<6;value['old_restricted_macro']=metrics(labels[old],restricted[old],6)['macro_accuracy']
        result.append(value)
    require(len(result)==complete['rows'],'Incomplete score matrix')
    write(root/'metrics.json',dict(capsule_id=m['capsule_id'],model_seed=payload['model_seed'],rows=result,truth_last=True,target_feedback=False))
    write(root/'complete.json',dict(status='SCORED_COMPLETE',rows=len(result),truth_last=True,target_feedback=False))
