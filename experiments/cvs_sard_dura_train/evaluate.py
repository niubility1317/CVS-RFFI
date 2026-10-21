import os, time
import numpy as np
from . import design as d
from .plan import require_complete_predictions

COUNT = 168000

def arrays():
    from experiments.cvs_sard_dura_eval.runtime import observations
    return observations()

def predict(c):
    import torch
    from .model import own_model
    from experiments.pamf_fcrossrx_readout.frozen import from_state
    from experiments.pamf_r46_multipoint_fast.tensor_io import array_tensor
    from experiments.cvs_sard_dura_eval.runtime import native_source_logits
    native = own_model(c, 'cuda:0')
    payload = torch.load(d.row_root(c['row_id'])/'source/final_ssdg.pth',map_location='cpu',weights_only=False)
    reader = from_state(payload['model'], c['model_seed']).to('cuda:0')
    x = torch.randn(2,2,256,device='cuda:0')
    with torch.inference_mode():
        torch.testing.assert_close(reader(x), native_source_logits(native,x),atol=1e-5,rtol=2e-5)
    del native, payload
    ids,raw,restored = arrays()
    out = d.row_root(c['row_id'])/'prediction'
    out.mkdir()
    decisions, times = {}, {}
    before = {k:v.clone() for k,v in reader.state_dict().items()}
    with torch.inference_mode():
        for mode, inputs in [('raw',raw),('dura',restored)]:
            for view in d.VIEWS:
                key = mode+'__'+view
                start = time.perf_counter()
                a = np.load(inputs[view],mmap_mode='r',allow_pickle=False)
                if a.shape!=(COUNT,2,256) or a.dtype!=np.float32:
                    raise ValueError('Same fixed formal IQ required')
                z = np.empty((COUNT,6),np.float32)
                for begin in range(0,COUNT,128):
                    result = reader(array_tensor(a[begin:begin+128],'cuda:0'))
                    if not torch.isfinite(result).all() or result.shape!=(min(128,COUNT-begin),6):
                        raise ValueError('Invalid frozen original head logits')
                    z[begin:begin+len(result)] = result.cpu().numpy()
                np.save(out/(key+'.npy'),z,allow_pickle=False)
                decisions[key] = z.argmax(1).astype(np.int64)
                times[key] = time.perf_counter()-start
                print(dict(event='VIEW_COMPLETE',row=c['row_id'],readout=mode,view=view,count=COUNT,seconds=times[key]),flush=True)
    require_complete_predictions(list(decisions))
    if any(not torch.equal(v,reader.state_dict()[k]) for k,v in before.items()):
        raise ValueError('Frozen F state changed')
    np.savez_compressed(out/'predictions.npz',ids=ids,**decisions)
    d.write(out/'complete.json',dict(status='PREDICTIONS_COMPLETE',config=c,count=COUNT,
        views=list(d.VIEWS),readouts=['raw','dura'],truth_read=False,state_updates=0,head_fit_updates=0,
        native_defaulthead_equivalence=True,F_NN=COUNT*14+4,G_new_predictions=0,
        DURA_original_IQ_reused=True,inference_seconds=times))

def prediction_preflight(c):
    out = d.row_root(c['row_id'])/'prediction'
    done = d.read(out/'complete.json')
    if (done['status']!='PREDICTIONS_COMPLETE' or done['config']!=c or done['count']!=COUNT
            or done['views']!=list(d.VIEWS) or done['truth_read'] or done['state_updates'] or done['head_fit_updates']):
        raise ValueError('Truth closed: incomplete/flexible prediction')
    ids,_,_ = arrays()
    with np.load(out/'predictions.npz',allow_pickle=False) as f:
        require_complete_predictions([k for k in f.files if k!='ids'])
        if not np.array_equal(f['ids'],ids):
            raise ValueError('Truth closed: physical IDs differ')
        values = {k:f[k].copy() for k in f.files if k!='ids'}
    for key,predicted in values.items():
        z = np.load(out/(key+'.npy'),allow_pickle=False)
        if (z.shape!=(COUNT,6) or z.dtype!=np.float32 or not np.isfinite(z).all()
                or predicted.shape!=(COUNT,) or not np.array_equal(z.argmax(1),predicted)):
            raise ValueError('Truth closed: incomplete or inconsistent logits')
    return ids,values

def score(c):
    from comparison_suite.score import metrics
    from experiments.cvs_early_geometry.evaluate import truth_arrays
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':
        raise ValueError('Separate CPU truth-side process required')
    ids,values = prediction_preflight(c)
    y,rx,days = truth_arrays(d.read(d.TRUTH),ids)
    records,paired = [],[]
    for view in d.VIEWS:
        raw,restored = values['raw__'+view],values['dura__'+view]
        for dim,groups in [('overall',['ALL']),('receiver',sorted(set(rx))),('day',sorted(set(days))),('transmitter',range(6))]:
            for group in groups:
                mask = np.ones(COUNT,bool) if dim=='overall' else rx==group if dim=='receiver' else days==group if dim=='day' else y==group
                a,b = raw[mask]==y[mask],restored[mask]==y[mask]
                paired.append(dict(view=view,dimension=dim,stratum=str(group),count=int(mask.sum()),
                    repaired=int((b&~a).sum()),damaged=int((a&~b).sum()),net=int(b.sum()-a.sum()),accuracy_delta_pp=100*float(b.mean()-a.mean())))
                for mode in ('raw','dura'):
                    pred = values[mode+'__'+view][mask]
                    result = metrics(y[mask],pred,6)
                    cm = np.bincount(y[mask]*6+pred,minlength=36).reshape(6,6)
                    den = cm.sum(0)+cm.sum(1)
                    f1 = np.divide(2*np.diag(cm),den,out=np.zeros(6),where=den>0).mean()
                    if abs(result['accuracy']-np.trace(cm)/cm.sum())>1e-12 or abs(result['macro_f1']-f1)>1e-12:
                        raise ValueError('Independent metric recount mismatch')
                    records.append(dict(row_id=c['row_id'],arm=c['arm'],model_seed=c['model_seed'],readout=mode,
                        view=view,dimension=dim,stratum=str(group),**result))
    out = d.row_root(c['row_id'])/'scoring'
    out.mkdir()
    d.write(out/'scores.json',dict(results=records,paired=paired,truth_last=True,target_feedback=False))
    d.write(out/'complete.json',dict(status='SCORED_COMPLETE',metric_records=len(records),paired_records=len(paired),views=list(d.VIEWS),truth_last=True))
    d.write(d.row_root(c['row_id'])/'complete.json',dict(status='ANALYZED',row_id=c['row_id'],truth_last=True,all14cells_before_truth=True))
    print(dict(status='SCORED_COMPLETE',row=c['row_id'],metric_records=len(records)),flush=True)
