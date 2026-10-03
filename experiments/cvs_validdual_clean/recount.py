"""Independent NumPy bincount metric reconstruction after scoring is sealed."""
import argparse
from pathlib import Path
import numpy as np
from experiments.cvs_validdual_clean.contracts import read,write,CLASSES
from experiments.cvs_validdual_clean.score import preflight_predictions

def recount(spec):
    root=Path(spec['runtime_root']);destination=root/'independent_recount.json'
    if destination.exists():raise FileExistsError('Preserve independent recount')
    done=read(root/'scoring_clean_complete.json')
    if done.get('status')!='SCORED_COMPLETE' or done.get('rows')!=8:raise ValueError('Completed fixed scoring required')
    ids,predictions=preflight_predictions(spec);scored=read(root/'clean_scored_results.json')
    truth=read(spec['p1_truth']);y=np.asarray([truth[i]['label'] for i in ids.tolist()]);rx=np.asarray([str(truth[i]['receiver']) for i in ids.tolist()])
    groups=[('ALL',np.ones(len(y),dtype=bool))]+[('RX'+v,rx==v) for v in sorted(set(rx))]
    records=scored['results'];classes=scored['class_results']
    lookup={(r['row_id'],r['group']):r for r in records};clookup={(r['row_id'],r['tx']):r for r in classes}
    expected={(r['row_id'],g) for r,_ in predictions for g,_ in groups}
    cexpected={(r['row_id'],t) for r,_ in predictions for t in CLASSES}
    if len(lookup)!=len(records) or set(lookup)!=expected or len(clookup)!=len(classes) or set(clookup)!=cexpected:raise ValueError('Scored matrix incomplete or duplicate')
    errors=[]
    for row,pred in predictions:
        for group,mask in groups:
            actual=lookup[(row['row_id'],group)]
            cm=np.bincount(6*y[mask]+pred[mask],minlength=36).reshape(6,6)
            if not np.array_equal(cm,np.asarray(actual['confusion'])) or int(cm.sum())!=actual['query_count']:raise ValueError('Independent confusion/count mismatch')
            tp=cm.diagonal().astype(float);support=cm.sum(1);denom=cm.sum(0)+support
            f1=np.mean([2*tp[j]/denom[j] if denom[j] else 0. for j in range(6)])
            acc=float((y[mask]==pred[mask]).mean());macro=float(np.mean([tp[j]/support[j] for j in range(6) if support[j]]))
            errors.extend(abs(float(actual[k])-v) for k,v in [('accuracy',acc),('macro_f1',f1),('macro_accuracy',macro)])
        for label,tx in enumerate(CLASSES):
            mask=y==label;actual=clookup[(row['row_id'],tx)]
            if actual['count']!=int(mask.sum()):raise ValueError('Independent TX count mismatch')
            errors.append(abs(actual['accuracy']-float((y[mask]==pred[mask]).mean())))
    if not np.isfinite(errors).all() or max(errors)>1e-12:raise ValueError('Independent metric mismatch')
    result=dict(status='VERIFIED',rows=8,query_count=len(ids),decisions=8*len(ids),receiver_records=len(records),class_records=len(classes),
        max_metric_absolute_error=max(errors),method='Raw fixed predictions and physical IDs; NumPy bincount and direct per-class formulas; no scorer metrics function',target_feedback_forbidden=True)
    write(destination,result);return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--spec',required=True);a=p.parse_args();print(recount(read(a.spec)))
