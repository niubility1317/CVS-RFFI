"""The missing ordinary-PL F / labeled-only G cell; frozen inference only."""
from pathlib import Path
import time
from .io import read,write,tensor
from .external_dura import VIEWS,cache_preflight,fp32_copy,tasks,reject_truth,handle
from .completeness import GFACTOR,main_source,SEEDS

CONTROL='20261023-phase1-PAMF-PL-Gfactor-control-r01'
ANALYSIS='20261023-phase1-PAMF-PL-Gfactor-analysis-m51-r01'
READOUTS=('raw','DURA-L')


def validate_cells(cells):
    from .plan import make_plan
    expected={(pid,seed) for pid in make_plan()['protocols'] for seed in SEEDS}
    if len(cells)!=51 or {(c['protocol_id'],c['model_seed']) for c in cells}!=expected:
        raise ValueError('Exactly17 contracts x3 existing PL F seeds are required')
    for c in cells:
        source=main_source(c['protocol_id'],'matched_sat_plain_pl',c['model_seed'])
        if any(c['source']['source_'+k]!=v for k,v in source.items()):
            raise ValueError('Ordinary-PL source differs, including F0 binding replacement')
        if c['row_id']!=c['protocol_id']+'-PL-G-L-s'+str(c['model_seed']):
            raise ValueError('Unexpected logical Gfactor row')
    return cells


def config(cell):
    return dict(run_id=GFACTOR,row_id=cell['row_id'],protocol_id=cell['protocol_id'],
        model_seed=cell['model_seed'],source_contract=cell['source_contract'],source_cache=cell['source_cache'],
        source_days=cell['source_days'],classes=cell['classes'],**cell['source'],
        raw_predictions_from=cell['raw_predictions_from'],readouts=list(READOUTS),
        cache_configs={'DURA-L':cell['cache_config']},
        tests={k:{'views_root':v['views_root']} for k,v in cell['tests'].items()},
        pamf_gfactor_action='predict',new_F_training=False,new_G_training=False)


def load_reader(c,device):
    from .frozen_inference import load_F
    reader,source=load_F(c['source_output'],read(c['source_contract']),device)
    if (source['run_id'],source['row_id'],source['model_seed'])!=(c['source_run_id'],c['source_row_id'],c['model_seed']):
        raise ValueError('Frozen PL F identity differs')
    if source['regime']!='matched_sat_plain_pl' or source['own_EMA'] or source['native_U_gate'] or not source['U_use']:
        raise ValueError('This factorial requires the existing ordinary-PL recipe')
    return reader


def raw(c,task,ids):
    import numpy as np
    root=Path(c['raw_predictions_from']);done=read(root/'complete.json')
    if (done.get('status')!='ALL_PREDICTIONS_COMPLETE' or done.get('truth_read') is not False
            or done.get('query_fit') is not False or done.get('run_id')!=c['source_run_id']
            or done.get('row_id')!=c['source_row_id'] or set(done['tests'])!=set(tasks(c))):
        raise ValueError('Completed raw predictions belong to a different PL F')
    with np.load(root/task/'predictions.npz',allow_pickle=False) as p:
        if not np.array_equal(p['ids'],ids):raise ValueError('Frozen PL raw physical IDs differ')
        values={v:p['raw__'+v].copy() for v in VIEWS}
    for a in values.values():
        if a.shape!=ids.shape or a.dtype.kind not in 'iu' or a.min()<0 or a.max()>=len(c['classes']):
            raise ValueError('Invalid frozen raw class decisions')
    return values


def predict(c,device='cuda:0'):
    import numpy as np
    import torch
    reject_truth(c)
    if c['readouts']!=list(READOUTS) or sorted(c['tests'])!=sorted(tasks(c)):
        raise ValueError('Only the missing raw/G-L PL readout is authorized')
    reader=load_reader(c,device);out=Path(c['output_root']);out.mkdir(parents=True,exist_ok=False)
    tic=time.monotonic();completed={}
    for task in tasks(c):
        _,ids,_=handle(c['tests'][task]['views_root'],c['classes'])
        original=raw(c,task,ids);cache=cache_preflight(c,'DURA-L',ids,task)
        values={'raw__'+v:original[v] for v in VIEWS}
        for view in VIEWS:
            data=np.load(cache/(view+'.npy'),mmap_mode='r',allow_pickle=False)
            if data.dtype!=np.float32 or data.shape!=(len(ids),2,256):raise ValueError('Restored IQ schema differs')
            decisions=[]
            with torch.inference_mode():
                for start in range(0,len(ids),128):
                    x=tensor(np.ascontiguousarray(data[start:start+128]),device)
                    logits=reader.classify_features(reader.features(x))
                    if logits.shape!=(len(x),len(c['classes'])) or not bool(torch.isfinite(logits).all()):
                        raise FloatingPointError('Invalid frozen PL default-head logits')
                    decisions.extend(logits.argmax(1).cpu().tolist())
            values['DURA-L__'+view]=np.asarray(decisions,dtype=np.int64)
            print('PAMF_PL_G_L_PREDICT',c['row_id'],task,view,len(ids),flush=True)
        folder=out/task;folder.mkdir();np.savez(folder/'predictions.npz',ids=ids,**values)
        completed[task]=dict(count=len(ids),views=list(VIEWS),readouts=list(READOUTS))
    write(out/'complete.json',dict(status='ALL_PREDICTIONS_COMPLETE',run_id=c['run_id'],row_id=c['row_id'],
        tests=completed,views=list(VIEWS),readouts=list(READOUTS),truth_read=False,query_fit=False,
        raw_reused=True,new_F_training=False,new_G_training=False,head_fit_updates=0,
        online_latency_measurement=False,seconds=time.monotonic()-tic))


def smoke(rt,output):
    import torch
    from .f_data import SourceDataset
    from .frozen_inference import load_G
    from experiments.pamf_stage2_inference.engine import InferenceEngine
    cells=validate_cells(rt['pamf_gfactor_manifest']['cells'])
    cell=next(c for c in cells if c['protocol_id']=='F0' and c['model_seed']==SEEDS[0])
    c=config(cell);reader=load_reader(c,'cpu');contract=read(c['source_contract'])
    cache=read(cell['cache_config'])
    if cache['readout']!='DURA-L' or cache['G']['roles']!='L' or read(cache['source_contract'])!=contract:
        raise ValueError('PAMF F and G-L source contracts differ')
    g=load_G(cache['G'],contract,'cpu')
    x=SourceDataset(c['source_cache'],'L_s',contract).x[:3]
    with torch.inference_mode():
        restored=InferenceEngine(g,reuse=True,graph=False,native=False)(x)[0]
        eager=g(x)
        if not torch.allclose(restored,eager,atol=1e-5,rtol=1e-5):raise ValueError('G-L source routes differ')
        a=reader.classify_features(reader.features(restored))
        b=reader.classify_features(reader.features(tensor(fp32_copy(restored).copy(),'cpu')))
        if not torch.equal(a,b) or not bool(torch.isfinite(a).all()):raise ValueError('FP32 cache copy changed PL default logits')
    write(output,dict(status='PASS',actual_PL_F_checkpoint=True,actual_same_contract_G_L=True,
        source_L_samples=3,FP32_cache_copy_logits_exact=True,G_routes_close=True,
        target_reads=0,U_IQ_reads=0,optimizer_updates=0,head_fit_updates=0))
