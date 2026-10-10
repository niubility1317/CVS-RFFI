"""Independent row scorer: complete opaque-ID prediction precedes truth read."""
import argparse
from pathlib import Path
import numpy as np
from comparison_suite.score import metrics,csvwrite
from experiments.standard_ce_baselines.common import read,write,VIEWS


def preflight(c):
    folder=Path(c['prediction_output']);done=read(folder/'complete.json')
    with np.load(Path(c['capsule'])/'index.npz',allow_pickle=False) as ix:ids=ix['ids'].copy()
    if done.get('status')!='PREDICTIONS_COMPLETE' or done.get('truth_read') is not False or done.get('query_fit') is not False or done.get('views')!=list(VIEWS) or done.get('count')!=len(ids):raise ValueError('Truth closed: incomplete prediction')
    with np.load(folder/'predictions.npz',allow_pickle=False) as values:
        if set(values.files)!={'ids','clean'} or not np.array_equal(values['ids'],ids):raise ValueError('Truth closed: wrong physical IDs')
        prediction=values['clean'].copy()
    if prediction.shape!=(len(ids),) or prediction.dtype.kind not in 'iu' or prediction.min()<0 or prediction.max()>=6:raise ValueError('Truth closed: invalid class predictions')
    return ids,prediction


def score(c):
    ids,prediction=preflight(c)
    truth=read(c['truth']);y=np.asarray([truth[s]['label'] for s in ids]);rx=np.asarray([str(truth[s]['receiver']) for s in ids])
    classes=read(Path(c['capsule'])/'manifest.json')['classes']
    if len(classes)!=6 or y.min()<0 or y.max()>=6 or len(set(rx))!=7:raise ValueError('Truth contract mismatch')
    output=Path(c['output_root']);output.mkdir(parents=True,exist_ok=False);results=[]
    for dimension,strata in [('overall',['ALL']),('receiver',sorted(set(rx))),('transmitter',classes)]:
        for stratum in strata:
            mask=np.ones(len(ids),dtype=bool) if dimension=='overall' else rx==stratum if dimension=='receiver' else y==classes.index(stratum)
            if not mask.any():raise ValueError('Empty stratum')
            results.append(dict(row_id=c['row_id'],variant=c['variant'],model_seed=c['model_seed'],view='clean',dimension=dimension,stratum=stratum,**metrics(y[mask],prediction[mask],6)))
    write(output/'scores.json',dict(status='SCORED',results=results,truth_last=True,target_feedback_forbidden=True))
    csvwrite(output/'scores.csv',[{k:v for k,v in r.items() if k!='confusion'} for r in results])
    overall=results[0]
    (output/'report.md').write_text('# '+c['row_id']+'\n\nStatus: SCORED; clean only; truth-last; query_count='+str(len(ids))+'\n\n'+str(overall)+'\n',encoding='utf-8')
    write(output/'complete.json',dict(status='SCORED_COMPLETE',count=len(ids),views=list(VIEWS),truth_last=True))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args();score(read(a.config))
