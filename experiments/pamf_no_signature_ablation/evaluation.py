from pathlib import Path
import time
import numpy as np
from experiments.cric_dura_comparison.io import read,write,tensor
from experiments.cric_dura_comparison.evaluation import handle,reject_truth
from experiments.practical_full_2x2.plan import VIEWS
from .method import TASKS,READOUTS,load_reader


def predict(c,device='cuda:0'):
    import torch
    reject_truth(c)
    if list(c['tests'])!=list(TASKS) or c['readouts']!=READOUTS:raise ValueError('Wrong frozen evaluation matrix')
    contract=read(c['source_contract']);reader,_=load_reader(c['source_output'],contract,device)
    cache=Path(c['restored_cache']);done=read(cache/'complete.json')
    if (done['status']!='DURA_IQ_CACHE_COMPLETE' or done['protocol_id']!='F0'
            or done['readout']!='DURA-L' or done['truth_read'] or read(done['G']['source_contract'])!=contract):
        raise ValueError('DURA-L cache ancestry/contract differs')
    out=Path(c['output_root']);out.mkdir(parents=True,exist_ok=False);tic=time.monotonic();counts={}
    for task in TASKS:
        manifest,ids,pos=handle(c['tests'][task]['views_root'],c['classes']);values={}
        with np.load(cache/task/'index.npz',allow_pickle=False) as ix:
            if not np.array_equal(ix['ids'],ids):raise ValueError('Restored physical-ID order differs')
        for view in VIEWS:
            raw=manifest['clean_ref'] if view=='clean' else str(Path(manifest.get('observations_root',c['tests'][task]['views_root']))/(view+'.npy'))
            arrays=(np.load(raw,mmap_mode='r',allow_pickle=False),np.load(cache/task/(view+'.npy'),mmap_mode='r',allow_pickle=False))
            for route,array in zip(READOUTS,arrays):
                result=[]
                with torch.inference_mode():
                    for begin in range(0,len(ids),128):
                        ix=pos[begin:begin+128] if route=='raw' and pos is not None else slice(begin,begin+128)
                        x=tensor(np.ascontiguousarray(array[ix],np.float32),device)
                        logits=reader.classify_features(reader.features(x))
                        if logits.shape!=(len(x),6) or not bool(torch.isfinite(logits).all()):raise FloatingPointError('Invalid default-head logits')
                        result.extend(logits.argmax(1).cpu().tolist())
                values[route+'__'+view]=np.asarray(result,np.int64)
            print('FROZEN_PREDICTION',c['row_id'],task,view,len(ids),flush=True)
        folder=out/task;folder.mkdir();np.savez(folder/'predictions.npz',ids=ids,**values)
        counts[task]=dict(count=len(ids),views=list(VIEWS),readouts=READOUTS)
    write(out/'complete.json',dict(status='ALL_PREDICTIONS_COMPLETE',run_id=c['run_id'],row_id=c['row_id'],
          tests=counts,views=list(VIEWS),readouts=READOUTS,truth_read=False,query_fit=False,seconds=time.monotonic()-tic))


def score(c):
    from experiments.cric_dura_comparison import evaluation as old
    old.tasks=lambda _:list(TASKS)
    old.score(c)
