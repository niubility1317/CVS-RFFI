"""Reuse frozen decisions, fill missing cells, then independent truth-last score."""
from pathlib import Path
import time
import numpy as np
from experiments.cric_dura_comparison.io import read,write,tensor
from .plan import TASKS,READOUTS,VIEWS
from .model import load_reader

def predict(c,device='cuda:0'):
    import torch
    from experiments.cric_dura_comparison.evaluation import reject_truth,handle
    from experiments.pamf_channel_pl.restored_source import snapshot,unchanged
    reject_truth(c)
    if list(c['tests'])!=list(TASKS) or c['readouts']!=list(READOUTS):raise ValueError('Incomplete frozen row matrix')
    row=c['paper_row'];contract=read(c['source_contract']);reader,_=load_reader(c['source_output'],contract,row,device)
    out=Path(c['output_root']);out.mkdir(parents=True,exist_ok=False);cached={};counts={};started=time.monotonic();before=snapshot(reader)
    for role,folder in c['restored_caches'].items():
        d=read(Path(folder)/'complete.json')
        if (d['status']!='DURA_IQ_CACHE_COMPLETE' or d['protocol_id']!=c['protocol_id'] or d['readout']!='DURA-'+role
            or d['truth_read'] or read(d['G']['source_contract'])!=contract):raise ValueError('Matching frozen DURA ancestry required')
        cached[role]=Path(folder)
    reused=computed=0
    for task in TASKS:
        manifest,ids,pos=handle(c['tests'][task]['views_root'],c['classes']);values={};old={}
        if c.get('reuse_predictions'):
            prior=Path(c['reuse_predictions']);done=read(prior/'complete.json')
            if done['status']!='ALL_PREDICTIONS_COMPLETE' or done['truth_read']:raise ValueError('Only complete frozen decisions reusable')
            path=prior/task/'predictions.npz'
            if path.exists():
                with np.load(path,allow_pickle=False) as f:
                    if not np.array_equal(f['ids'],ids):raise ValueError('Reused decision physical IDs differ')
                    old={k:f[k].copy() for k in f.files if k!='ids'}
        for folder in cached.values():
            with np.load(folder/task/'index.npz',allow_pickle=False) as ix:
                if not np.array_equal(ix['ids'],ids):raise ValueError('Restored cache physical IDs differ')
        for view in c['views']:
            for route in READOUTS:
                key=route+'__'+view
                if key in old:
                    p=old[key]
                    if p.shape!=(len(ids),) or p.dtype.kind not in 'iu' or p.min()<0 or p.max()>=len(c['classes']):raise ValueError('Illegal reusable decisions')
                    values[key]=p;reused+=len(ids);continue
                path=(manifest['clean_ref'] if view=='clean' else str(Path(manifest.get('observations_root',c['tests'][task]['views_root']))/(view+'.npy'))) if route=='raw' else str(cached[route[5:]]/task/(view+'.npy'))
                array=np.load(path,mmap_mode='r',allow_pickle=False);result=[]
                with torch.inference_mode():
                    for begin in range(0,len(ids),128):
                        index=pos[begin:begin+128] if route=='raw' and pos is not None else slice(begin,begin+128)
                        x=tensor(np.ascontiguousarray(array[index],np.float32),device)
                        logits=reader.classify_features(reader.features(x))
                        if logits.shape!=(len(x),len(c['classes'])) or not bool(torch.isfinite(logits).all()):raise FloatingPointError('Invalid frozen default-head logits')
                        result.extend(logits.argmax(1).cpu().tolist())
                values[key]=np.asarray(result,np.int64);computed+=len(ids)
            print('FROZEN_PREDICTION',c['row_id'],task,view,len(ids),flush=True)
        folder=out/task;folder.mkdir();np.savez(folder/'predictions.npz',ids=ids,**values)
        counts[task]=dict(count=len(ids),views=c['views'],readouts=list(READOUTS))
    if not unchanged(reader,before):raise ValueError('Frozen model state changed')
    write(out/'complete.json',dict(status='ALL_PREDICTIONS_COMPLETE',run_id=c['run_id'],row_id=c['row_id'],tests=counts,
        views=c['views'],readouts=list(READOUTS),truth_read=False,query_fit=False,reused_decisions=reused,new_decisions=computed,seconds=time.monotonic()-started))

def score(c):
    from experiments.cric_dura_comparison import evaluation as old
    if any(k in c for k in ('source_output','source_cache','restored_caches','frozen_sources')):raise ValueError('Independent scorer only')
    # The input capsule always describes seven received views. A clean-only row
    # selects a subset to score; it does not change the capsule's scene contract.
    original_handle,original_views,original_tasks=old.handle,old.VIEWS,old.tasks
    selected=tuple(c['views'])
    def handle_full_capsule(*args,**kwargs):
        old.VIEWS=VIEWS
        try:return original_handle(*args,**kwargs)
        finally:old.VIEWS=selected
    old.handle=handle_full_capsule;old.tasks=lambda _:list(TASKS);old.VIEWS=selected
    try:old.score(c)
    finally:old.handle,old.VIEWS,old.tasks=original_handle,original_views,original_tasks
