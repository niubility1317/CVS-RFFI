"""Reuse frozen decisions; never expose truth or training data to each other."""
from pathlib import Path
import numpy as np
from experiments.cric_dura_comparison.io import read,write

def predict(c):
    if c['paper_row']['family']=='architecture':
        from .model import ce_scope
        from experiments.standard_ce_baselines.predict import predict as native
        with ce_scope():return native(c)
    from experiments.pamf_single_view_paper import evaluation as native
    from .model import load_reader
    old=native.load_reader;native.load_reader=load_reader
    try:return native.predict(c)
    finally:native.load_reader=old

def score(c):
    if any(c.get(k) for k in ('source_output','source_cache','restored_caches','frozen_sources','dataset')):raise ValueError('Independent truth-side scoring only')
    if c['paper_row']['family']=='architecture':
        return score_ce(c)
    from experiments.pamf_single_view_paper.evaluation import score as native
    return native(c)

def score_ce(c):
    from experiments.standard_ce_baselines.score import preflight
    from experiments.cric_dura_comparison.evaluation import handle
    from comparison_suite.score import metrics,csvwrite
    ids,pred=preflight(c)
    _,both,_=handle(c['double_cross_views_root'],c['classes'])
    mask=np.isin(ids,both)
    if len(both)!=42000 or int(mask.sum())!=42000:raise ValueError('Double-cross physical population mismatch; truth remains closed')
    truth=read(c['truth']);y=np.asarray([truth[i]['label'] for i in ids]);rx=np.asarray([str(truth[i]['receiver']) for i in ids])
    if len(c['classes'])!=6 or y.min()<0 or y.max()>=6 or len(set(rx))!=7:raise ValueError('CE truth contract mismatch')
    results=[]
    for test,selection in (('combined_target',np.ones(len(ids),bool)),('cross_rx',~mask),('cross_both',mask)):
        for dimension,strata in (('overall',['ALL']),('receiver',sorted(set(rx))),('transmitter',c['classes'])):
            for stratum in strata:
                selected=selection & (True if dimension=='overall' else rx==stratum if dimension=='receiver' else y==c['classes'].index(stratum))
                if not selected.any():raise ValueError('Empty CE stratum')
                results.append(dict(row_id=c['row_id'],variant=c['variant'],model_seed=c['model_seed'],test=test,readout='raw',view='clean',dimension=dimension,stratum=stratum,**metrics(y[selected],pred[selected],6)))
    out=Path(c['output_root']);out.mkdir(parents=True,exist_ok=False)
    write(out/'scores.json',dict(status='SCORED',results=results,truth_last=True,target_feedback_forbidden=True))
    csvwrite(out/'scores.csv',[{k:v for k,v in r.items() if k!='confusion'} for r in results])
    write(out/'complete.json',dict(status='SCORED_COMPLETE',count=len(ids),double_cross_count=42000,views=['clean'],truth_last=True))
