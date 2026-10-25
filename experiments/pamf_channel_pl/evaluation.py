"""Raw F and the unchanged contract-matched G-LU, followed by independent truth."""
from pathlib import Path
import time
import numpy as np
from experiments.cric_dura_comparison.io import read,write,tensor
from experiments.cric_dura_comparison.evaluation import handle,reject_truth
from experiments.practical_full_2x2.plan import TASKS,VIEWS


def validate_prediction(c):
    reject_truth(c)
    if (set(c['tests'])!=set(TASKS) or c['readouts']!=['raw','DURA-LU']
            or set(c['restored_caches'])!={'LU'} or len(c['classes']) not in (6,10)):
        raise ValueError('All three declared populations and raw/same-G-LU required')


def predict(c,device='cuda:0'):
    import torch
    from experiments.pamf_repl_completion.source import load_reader
    validate_prediction(c);contract=read(c['source_contract'])
    reader,_=load_reader(c['source_output'],contract,device)
    cache=Path(c['restored_caches']['LU']);done=read(cache/'complete.json')
    if (done['status']!='DURA_IQ_CACHE_COMPLETE' or done['protocol_id']!=c['protocol_id']
        or done['readout']!='DURA-LU' or done['truth_read'] or read(done['G']['source_contract'])!=contract):
        raise ValueError('Same-contract frozen G-LU IQ cache required')
    out=Path(c['output_root']);out.mkdir(parents=True,exist_ok=False);start=time.monotonic();counts={}
    for task in TASKS:
        manifest,ids,positions=handle(c['tests'][task]['views_root'],c['classes']);values={}
        with np.load(cache/task/'index.npz',allow_pickle=False) as index:
            if not np.array_equal(index['ids'],ids):raise ValueError('G-LU physical ID order differs')
        for view in VIEWS:
            raw=manifest['clean_ref'] if view=='clean' else str(Path(manifest.get('observations_root',c['tests'][task]['views_root']))/(view+'.npy'))
            arrays=[np.load(raw,mmap_mode='r',allow_pickle=False),np.load(cache/task/(view+'.npy'),mmap_mode='r',allow_pickle=False)]
            for route,array in zip(c['readouts'],arrays):
                result=[]
                with torch.inference_mode():
                    for begin in range(0,len(ids),128):
                        ix=positions[begin:begin+128] if route=='raw' and positions is not None else slice(begin,begin+128)
                        x=tensor(np.ascontiguousarray(array[ix],np.float32),device)
                        logits=reader.classify_features(reader.features(x))
                        if logits.shape!=(len(x),len(c['classes'])) or not bool(torch.isfinite(logits).all()):
                            raise FloatingPointError('Frozen original default head invalid')
                        result.extend(logits.argmax(1).cpu().tolist())
                values[route+'__'+view]=np.asarray(result,np.int64)
            print('CHANNELPL_FROZEN_PREDICTION',c['row_id'],task,view,len(ids),flush=True)
        folder=out/task;folder.mkdir();np.savez(folder/'predictions.npz',ids=ids,**values)
        counts[task]=dict(count=len(ids),views=list(VIEWS),readouts=c['readouts'])
    write(out/'complete.json',dict(status='ALL_PREDICTIONS_COMPLETE',run_id=c['run_id'],row_id=c['row_id'],
        tests=counts,views=list(VIEWS),readouts=c['readouts'],truth_read=False,query_fit=False,seconds=time.monotonic()-start))


def score(c):
    # Runs in its own CPU process after every population/view is frozen. The
    # supplied score config contains no source-model, source-IQ or G handle.
    from experiments.cric_dura_comparison import evaluation as baseline
    baseline.tasks=lambda _:list(TASKS)
    baseline.score(c)
