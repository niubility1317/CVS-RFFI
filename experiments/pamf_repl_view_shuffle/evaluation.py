from pathlib import Path
import numpy as np
from experiments.cric_dura_comparison.io import read,write,tensor
from experiments.cric_dura_comparison.evaluation import handle,reject_truth
from .plan import TASKS,VIEWS

def predict(c,device='cuda:0'):
    import torch
    from .source import load_reader
    reject_truth(c)
    if list(c['tests'])!=list(TASKS) or c['readouts']!=['raw']:raise ValueError('Direct recognition only; both target populations required')
    reader,source=load_reader(c['source_output'],read(c['source_contract']),device)
    if source['run_id']!=c['run_id'] or source['row_id']!=c['row_id'] or source['model_seed']!=c['model_seed']:raise ValueError('Different frozen source row')
    out=Path(c['output_root']);out.mkdir(parents=True,exist_ok=False);counts={}
    for task in TASKS:
        m,ids,pos=handle(c['tests'][task]['views_root'],c['classes']);values={}
        for view in VIEWS:
            path=m['clean_ref'] if view=='clean' else str(Path(m.get('observations_root',c['tests'][task]['views_root']))/(view+'.npy'))
            array=np.load(path,mmap_mode='r',allow_pickle=False);result=[]
            if array.ndim!=3 or array.shape[1:]!=(2,256):raise ValueError('Wrong received IQ shape')
            with torch.inference_mode():
                for start in range(0,len(ids),128):
                    ix=pos[start:start+128] if pos is not None else slice(start,start+128)
                    x=tensor(np.ascontiguousarray(array[ix],np.float32),device);logits=reader.classify_features(reader.features(x))
                    if logits.shape!=(len(x),6) or not bool(torch.isfinite(logits).all()):raise ValueError('Invalid default head')
                    result.extend(logits.argmax(1).cpu().tolist())
            values['raw__'+view]=np.asarray(result,np.int64)
            print('FROZEN_PREDICTION',c['row_id'],task,view,len(ids),flush=True)
        folder=out/task;folder.mkdir();np.savez(folder/'predictions.npz',ids=ids,**values);counts[task]=dict(count=len(ids),views=list(VIEWS),readouts=['raw'])
    write(out/'complete.json',dict(status='ALL_PREDICTIONS_COMPLETE',run_id=c['run_id'],row_id=c['row_id'],tests=counts,views=list(VIEWS),readouts=['raw'],truth_read=False,query_fit=False))

def score(c):
    from experiments.cric_dura_comparison import evaluation as old
    previous=old.tasks;old.tasks=lambda _:list(TASKS)
    try:return old.score(c)
    finally:old.tasks=previous
