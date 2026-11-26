"""Two frozen routes, three disjoint populations, seven views; CPU truth-last."""
from pathlib import Path
import time
import numpy as np
from experiments.cric_dura_comparison.io import read,write,tensor
from experiments.cric_dura_comparison.evaluation import handle,reject_truth
from .plan import TASKS,VIEWS,RUN

def cache(c):
    # Reuse the existing source-execution equivalence check and optimized G.
    from experiments.cric_dura_comparison import external_dura as old
    old.tasks=lambda _:list(TASKS);old.CACHE=RUN
    old.cache(c)

def load_reader(c,device):
    if c['kind']=='external':
        from .source import frozen_model
        from experiments.phase1_external_dg_aug.models import identity
        expected=read(c['external_source_contract'])
        physical=read(c['source_contract'])
        if any(expected[k]!=physical[k] for k in ('role_ids','counts','source_rxs','source_days','classes')):
            raise ValueError('External F and G contract mismatch')
        model,actual=frozen_model(c['source_output'],expected,device)
        if actual['row_id']!=c['row_id'] or actual['model_seed']!=c['model_seed']:raise ValueError('Different F row')
        return lambda x:identity(model,actual,x)
    from experiments.cric_dura_comparison.frozen_inference import load_F,load_original_T1
    contract=read(c['source_contract'])
    if c.get('original_arm'):reader=load_original_T1(c['model_seed'],contract,device)
    else:reader,_=load_F(c['source_output'],contract,device,require_no_unlabeled=c['G_roles']=='L')
    return lambda x:reader.classify_features(reader.features(x))

def predict(c,device='cuda:0',*,classify_view=None,task_names=TASKS,reuse_raw=None):
    import torch
    reject_truth(c)
    routes=c['readouts']
    if (tuple(c['tests'])!=tuple(task_names) or
            (routes!=['raw','DURA-'+str(c.get('G_roles'))] and not (classify_view is not None and routes==['raw']))):
        raise ValueError('Readout matrix differs')
    run_id=RUN if classify_view is None else c['run_id']
    if classify_view is None:
        classify=load_reader(c,device)
        classify_view=lambda x,view:classify(x)
    cached=None
    if len(routes)==2:
        cached=Path(c['restored_cache']);done=read(cached/'complete.json')
        if done['status']!='DURA_IQ_CACHE_COMPLETE' or done['protocol_id']!=c['protocol_id'] or done['readout']!=routes[1] or done['truth_read']:
            raise ValueError('Restored IQ cache role differs')
        if read(done['G']['source_contract'])!=read(c['source_contract']):raise ValueError('Restored cache contract mismatch')
    reuse_done=None
    if reuse_raw is not None:
        reuse_path=Path(reuse_raw['path'])
        reuse_done=read(reuse_path/'complete.json')
        if (reuse_done['status']!='ALL_PREDICTIONS_COMPLETE' or reuse_done['truth_read']
                or reuse_done['query_fit'] or reuse_done['views']!=list(VIEWS)
                or reuse_done['run_id']!=reuse_raw['run_id'] or reuse_done['row_id']!=reuse_raw['row_id']):
            raise ValueError('Historical raw predictions are not from the matching frozen row')
    out=Path(c['output_root']);out.mkdir(parents=True,exist_ok=False);tic=time.monotonic();counts={}
    for task in task_names:
        manifest,ids,pos=handle(c['tests'][task]['views_root'],c['classes'])
        if cached is not None:
            with np.load(cached/task/'index.npz',allow_pickle=False) as a:
                if not np.array_equal(a['ids'],ids):raise ValueError('Restored physical ID order differs')
        values={}
        if reuse_done is not None and task in reuse_done['tests']:
            with np.load(reuse_path/task/'predictions.npz',allow_pickle=False) as a:
                if not np.array_equal(a['ids'],ids):raise ValueError('Historical raw physical ID order differs')
                values.update({f'raw__{v}':a[f'raw__{v}'].copy() for v in VIEWS})
        for view in VIEWS:
            raw_path=manifest['clean_ref'] if view=='clean' else str(Path(manifest.get('observations_root',c['tests'][task]['views_root']))/(view+'.npy'))
            for route in routes:
                key=route+'__'+view
                if key in values:
                    continue
                path=raw_path if route=='raw' else cached/task/(view+'.npy')
                array=np.load(path,mmap_mode='r',allow_pickle=False)
                if array.ndim!=3 or array.shape[1:]!=(2,256) or array.dtype!=np.float32:
                    raise ValueError('IQ schema differs')
                if (route!='raw' or pos is None) and len(array)!=len(ids):raise ValueError('IQ record count differs')
                result=[]
                with torch.inference_mode():
                    for start in range(0,len(ids),128):
                        ix=pos[start:start+128] if route=='raw' and pos is not None else slice(start,start+128)
                        x=tensor(np.ascontiguousarray(array[ix],np.float32),device);logits=classify_view(x,view)
                        if logits.shape!=(len(x),len(c['classes'])) or not bool(torch.isfinite(logits).all()):raise FloatingPointError('Invalid frozen head')
                        result.extend(logits.argmax(1).cpu().tolist())
                values[key]=np.asarray(result,np.int64)
            print('FROZEN_PREDICTION',c['row_id'],task,view,len(ids),flush=True)
        folder=out/task;folder.mkdir();np.savez(folder/'predictions.npz',ids=ids,**values)
        counts[task]=dict(count=len(ids),views=list(VIEWS),readouts=routes)
    write(out/'complete.json',dict(status='ALL_PREDICTIONS_COMPLETE',run_id=run_id,row_id=c['row_id'],tests=counts,
        views=list(VIEWS),readouts=routes,truth_read=False,query_fit=False,seconds=time.monotonic()-tic))

def score(c, *, task_names=TASKS):
    from experiments.cric_dura_comparison import evaluation as old
    before=old.tasks
    old.tasks=lambda _:list(task_names)
    try:
        return old.score(c)
    finally:
        old.tasks=before
