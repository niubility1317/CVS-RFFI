"""Separate scorer validates all eight complete predictions before opening truth."""
import argparse
from pathlib import Path
import numpy as np
from comparison_suite.score import metrics,csvwrite
from experiments.cvs_sixscene_eval.common import VIEWS,SEEDS,read,write

def preflight(spec):
    if len(spec['rows'])!=8 or {(r['condition'],r['model_seed']) for r in spec['rows']}!={(c,s) for c in ('clean_train','mid_low_aug') for s in SEEDS}:raise ValueError('Incomplete matrix')
    with np.load(Path(spec['views_root'])/'index.npz',allow_pickle=False) as ix:ids=ix['ids'].copy()
    predictions=[]
    for row in spec['rows']:
        out=Path(row['output_root']);done=read(out/'complete.json')
        if done.get('status')!='PREDICTIONS_COMPLETE' or done.get('truth_read') is not False or done.get('query_fit') is not False or done.get('views')!=list(VIEWS) or done.get('count')!=len(ids):raise ValueError('Truth remains closed: incomplete prediction')
        with np.load(out/'predictions.npz',allow_pickle=False) as p:
            if set(p.files)!=set(VIEWS)|{'ids'} or not np.array_equal(ids,p['ids']):raise ValueError('Prediction identity mismatch')
            values={v:p[v].copy() for v in VIEWS}
        for value in values.values():
            if value.shape!=(len(ids),) or value.dtype.kind not in 'iu' or value.min()<0 or value.max()>=6:raise ValueError('Invalid predicted classes')
        predictions.append((row,values))
    return ids,predictions

def score(spec):
    ids,predictions=preflight(spec)
    truth=read(spec['truth']);y=np.asarray([truth[s]['label'] for s in ids]);rx=np.asarray([str(truth[s]['receiver']) for s in ids])
    if y.min()<0 or y.max()>=6 or len(set(rx))!=7:raise ValueError('Truth contract mismatch')
    manifest=read(Path(spec['views_root'])/'manifest.json');results=[]
    for row,values in predictions:
        for view in VIEWS:
            for dimension,strata in [('overall',['ALL']),('receiver',sorted(set(rx))),('transmitter',manifest['classes'])]:
                for stratum in strata:
                    mask=np.ones(len(ids),dtype=bool) if dimension=='overall' else rx==stratum if dimension=='receiver' else y==manifest['classes'].index(stratum)
                    if not mask.any():raise ValueError('Empty test stratum')
                    results.append(dict(row_id=row['row_id'],condition=row['condition'],model_seed=row['model_seed'],view=view,
                        dimension=dimension,stratum=stratum,**metrics(y[mask],values[view][mask],6)))
    root=Path(spec['runtime_root']);write(root/'scores.json',dict(status='SCORED',results=results,target_feedback_forbidden=True))
    csvwrite(root/'scores.csv',[{k:v for k,v in r.items() if k!='confusion'} for r in results])
    write(root/'scoring_complete.json',dict(status='SCORED_COMPLETE',models=8,views=7,query_count_per_view=len(ids),prediction_count=8*7*len(ids),metric_records=len(results),truth_last=True))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--spec',required=True);a=p.parse_args();score(read(a.spec))
