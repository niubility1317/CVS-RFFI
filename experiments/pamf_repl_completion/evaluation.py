"""Three populations, three existing IQ routes; all decisions before independent truth."""
from pathlib import Path
import time
import numpy as np
from experiments.cric_dura_comparison.io import read,write,tensor
from experiments.cric_dura_comparison.evaluation import handle,reject_truth
from experiments.practical_full_2x2.plan import TASKS,VIEWS
from .source import load_reader,reader_structure

def predict(c,device='cuda:0'):
    import torch
    reject_truth(c);contract=read(c['source_contract'])
    if c.get('G_component'):
        from experiments.cric_dura_comparison import evaluation as old
        old.tasks=lambda _:list(TASKS)
        return old.predict(c,device)
    reader,_=load_reader(c['source_output'],contract,device,require_no_unlabeled=not c['U'])
    out=Path(c['output_root']);out.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    if c.get('source_physics_diagnostics'):
        from experiments.cric_dura_comparison.source_physics import run
        run(reader,c,out/'source_physics.json',device)
    if list(c['tests'])!=list(TASKS) or c['readouts']!=['raw','DURA-L','DURA-LU']:raise ValueError('Incomplete current readout matrix')
    cached={}
    for role,root in c['restored_caches'].items():
        d=read(Path(root)/'complete.json')
        if d['status']!='DURA_IQ_CACHE_COMPLETE' or d['protocol_id']!=c['protocol_id'] or d['readout']!='DURA-'+role or d['truth_read']:
            raise ValueError('Wrong frozen IQ cache')
        if read(d['G']['source_contract'])!=contract:raise ValueError('G ancestral contract differs')
        cached[role]=Path(root)
    counts={}
    for task in TASKS:
        manifest,ids,pos=handle(c['tests'][task]['views_root'],c['classes']);values={}
        for root in cached.values():
            with np.load(root/task/'index.npz',allow_pickle=False) as ix:
                if not np.array_equal(ix['ids'],ids):raise ValueError('Restored physical ID order differs')
        for view in VIEWS:
            raw=manifest['clean_ref'] if view=='clean' else str(Path(manifest.get('observations_root',c['tests'][task]['views_root']))/(view+'.npy'))
            arrays=[np.load(raw,mmap_mode='r',allow_pickle=False)]+[np.load(cached[r]/task/(view+'.npy'),mmap_mode='r',allow_pickle=False) for r in ('L','LU')]
            for route,array in zip(c['readouts'],arrays):
                result=[]
                with torch.inference_mode():
                    for begin in range(0,len(ids),128):
                        ix=pos[begin:begin+128] if route=='raw' and pos is not None else slice(begin,begin+128)
                        x=tensor(np.ascontiguousarray(array[ix],np.float32),device)
                        logits=reader.classify_features(reader.features(x))
                        if logits.shape!=(len(x),len(c['classes'])) or not bool(torch.isfinite(logits).all()):raise FloatingPointError('Frozen default head invalid')
                        result.extend(logits.argmax(1).cpu().tolist())
                values[route+'__'+view]=np.asarray(result,np.int64)
            print('FROZEN_PREDICTION',c['row_id'],task,view,len(ids),flush=True)
        folder=out/task;folder.mkdir();np.savez(folder/'predictions.npz',ids=ids,**values)
        counts[task]=dict(count=len(ids),views=list(VIEWS),readouts=c['readouts'])
    write(out/'complete.json',dict(status='ALL_PREDICTIONS_COMPLETE',run_id=c['run_id'],row_id=c['row_id'],tests=counts,
        views=list(VIEWS),readouts=c['readouts'],truth_read=False,query_fit=False,seconds=time.monotonic()-started))

def score(c):
    from experiments.cric_dura_comparison import evaluation as old
    old.tasks=lambda _:list(TASKS);old.score(c)
