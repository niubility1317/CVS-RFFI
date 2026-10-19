"""Unmodified fixed target arrays; all seven predictions before independent truth."""
from pathlib import Path
import json,time
from . import design as d
COUNT=168000

def observations():
    import numpy as np
    m=d.read(d.FORMAL_VIEWS/'manifest.json')
    expected=dict(status='VALIDATED_ONCE',count=COUNT,scenes=list(d.SCENES),classes=d.CLASSES,
        channel='residual/post_sync/noeq',truth_read=False,rows_share_observations=True)
    if any(m.get(k)!=v for k,v in expected.items()):raise ValueError('Fixed formal observation handles differ')
    if not m['source_target_disjoint']:raise ValueError('Source/target physical overlap')
    with np.load(d.FORMAL_VIEWS/'index.npz',allow_pickle=False) as f:ids=f['ids'].copy()
    if ids.shape!=(COUNT,) or len(set(ids.tolist()))!=COUNT:raise ValueError('Fixed target IDs differ')
    return ids,m

def predict(row):
    import numpy as np,torch
    from .model import own_model
    from .channel import tensor_iq
    from experiments.cvs_equivariant_identity.precision import numerical_context,actual_flags
    c=d.config(next(r for r in d.rows() if r['row_id']==row));out=d.row_root(row)/'prediction'
    if out.exists():raise FileExistsError('Reconcile existing predictions; do not overwrite')
    torch.set_num_threads(2);start=time.perf_counter()
    with numerical_context(d.FULL_FP32_POLICY):
        m=own_model(c,'cuda:0');before={k:v.clone() for k,v in m.state_dict().items()}
        with torch.no_grad():smoke=m.forward_identity_only(torch.zeros(2,2,256,device='cuda:0'))['tx_logits']
        if smoke.shape!=(2,6) or not torch.isfinite(smoke).all():raise ValueError('Final default checkpoint smoke failed')
        ids,manifest=observations();out.mkdir(parents=True,exist_ok=False)
        d.write(out/'resolved_config.json',dict(config=c,views=list(d.VIEWS),formal_manifest=str(d.FORMAL_VIEWS/'manifest.json'),
            backend_flags=actual_flags(),model_frozen=True,separate_head_fit=False,truth_read=False,query_fit=False,
            checkpoint=c['output_root']+'/final_ssdg.pth',commit=(d.ROOT/'release_commit.txt').read_text().strip()))
        predictions={};times={}
        with torch.no_grad():
            for view in d.VIEWS:
                p=Path(manifest['clean_ref']) if view=='clean' else d.FORMAL_VIEWS/(view+'.npy')
                array=np.load(p,mmap_mode='r',allow_pickle=False)
                if array.shape!=(COUNT,2,256) or array.dtype!=np.float32:raise ValueError('Formal IQ shape/dtype differs')
                scores=np.empty((COUNT,6),np.float32);tic=time.perf_counter()
                for begin in range(0,COUNT,256):
                    x=tensor_iq(array[begin:begin+256],'cuda:0');logits=m.forward_identity_only(x)['tx_logits']
                    if logits.shape!=(len(x),6) or not torch.isfinite(logits).all():raise ValueError('Invalid default head predictions')
                    scores[begin:begin+len(x)]=logits.cpu().tolist()
                predictions[view]=scores.argmax(1).astype(np.int64)
                np.savez_compressed(out/(view+'_scores.npz'),ids=ids,logits=scores)
                torch.cuda.synchronize();times[view]=time.perf_counter()-tic
                print(json.dumps(dict(view=view,count=COUNT,inference_seconds=times[view],truth_read=False)),flush=True)
        if any(not torch.equal(v,m.state_dict()[k]) for k,v in before.items()):raise ValueError('Target inference changed frozen state')
        if m.id_backbone.encoder._full_eval_packets!=COUNT*7+2:raise ValueError('Exact frozen prediction forward cost differs')
        np.savez_compressed(out/'predictions.npz',ids=ids,**predictions)
        d.write(out/'complete.json',dict(status='PREDICTIONS_COMPLETE',row_id=row,count=COUNT,views=list(d.VIEWS),
            config=c,truth_read=False,query_fit=False,all_parameters_frozen=True,head_fit_updates=0,
            inference_seconds=times,wall_seconds=time.perf_counter()-start,actual_NN=COUNT*7+2,synthetic_smoke_NN=2))

def preflight(row):
    import numpy as np
    c=d.config(next(r for r in d.rows() if r['row_id']==row));out=d.row_root(row)/'prediction'
    done=d.read(out/'complete.json');ids,m=observations()
    if done['status']!='PREDICTIONS_COMPLETE' or done['config']!=c or done['count']!=COUNT or done['views']!=list(d.VIEWS) or done['truth_read'] or done['query_fit'] or not done['all_parameters_frozen'] or done['head_fit_updates']:
        raise ValueError('Truth closed: incomplete or unfrozen row predictions')
    with np.load(out/'predictions.npz',allow_pickle=False) as p:
        if set(p.files)!={'ids',*d.VIEWS} or not np.array_equal(p['ids'],ids):raise ValueError('Truth closed: prediction IDs/views differ')
        for view in d.VIEWS:
            values=p[view]
            if values.shape!=(COUNT,) or values.dtype.kind not in 'iu' or values.min()<0 or values.max()>5:raise ValueError('Truth closed: invalid default decisions')
            with np.load(out/(view+'_scores.npz'),allow_pickle=False) as f:
                logits=f['logits']
                if not np.array_equal(f['ids'],ids) or logits.shape!=(COUNT,6) or not np.isfinite(logits).all() or not np.array_equal(logits.argmax(1),values):raise ValueError('Truth closed: logits/decisions differ')
    return c,ids

def score(row):
    import numpy as np
    from comparison_suite.score import metrics
    c,ids=preflight(row);out=d.row_root(row)/'scoring'
    if out.exists():raise FileExistsError('Existing independent scores must be reconciled, never overwritten')
    out.mkdir(parents=True,exist_ok=False)
    # This is a separate exec stage. No training object or teacher survives.
    truth=d.read(d.TRUTH)
    from experiments.cvs_early_geometry.evaluate import truth_arrays
    y,rx,days=truth_arrays(truth,ids);records=[]
    with np.load(d.row_root(row)/'prediction/predictions.npz',allow_pickle=False) as f:
        for view in d.VIEWS:
            predicted=f[view]
            for dim,values in [('overall',['ALL']),('receiver',sorted(set(rx))),('day',sorted(set(days))),('transmitter',range(6))]:
                for value in values:
                    mask=np.ones(COUNT,bool) if dim=='overall' else rx==value if dim=='receiver' else days==value if dim=='day' else y==value
                    result=metrics(y[mask],predicted[mask],6)
                    cm=np.bincount(y[mask]*6+predicted[mask],minlength=36).reshape(6,6)
                    denominator=cm.sum(0)+cm.sum(1);f1=np.divide(2*np.diag(cm),denominator,out=np.zeros(6),where=denominator>0).mean()
                    if abs(result['accuracy']-np.trace(cm)/cm.sum())>1e-12 or abs(result['macro_f1']-f1)>1e-12:raise ValueError('Independent metric recount differs')
                    records.append(dict(row_id=row,arm=c['arm'],model_seed=c['model_seed'],view=view,dimension=dim,stratum=str(value),**result))
    d.write(out/'scores.json',dict(results=records,truth_last=True,target_feedback_forbidden=True))
    d.write(out/'complete.json',dict(status='SCORED_COMPLETE',row_id=row,views=list(d.VIEWS),predictions=COUNT*7,
        metric_records=len(records),independent_recount='VERIFIED',truth_last=True))
    print(json.dumps(dict(status='SCORED_COMPLETE',row_id=row,metric_records=len(records))),flush=True)
