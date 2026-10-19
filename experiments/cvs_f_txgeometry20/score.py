"""Independent CPU truth-last scoring, after every declared row prediction is fixed."""
import os
from pathlib import Path
import numpy as np
from . import design as d
COUNT=168000

def preflight(c,base):
 out=Path(base)/c['row_id']/'prediction';done=d.read(out/'complete.json')
 if any(done.get(k)!=v for k,v in dict(status='PREDICTIONS_COMPLETE',config=c,count=COUNT,views=list(d.VIEWS),truth_read=False,query_fit=False,head_fit_updates=0).items()):raise ValueError('Truth stays closed: incomplete row predictions')
 with np.load(out/'predictions.npz',allow_pickle=False) as p:
  if set(p.files)!={'ids',*d.VIEWS}:raise ValueError('Truth stays closed: prediction view matrix differs')
  ids=p['ids'].copy();values={k:p[k].copy() for k in d.VIEWS}
 if ids.shape!=(COUNT,) or len(set(ids.tolist()))!=COUNT or any(a.shape!=(COUNT,) or a.dtype.kind not in 'iu' or a.min()<0 or a.max()>=6 for a in values.values()):raise ValueError('Truth stays closed: prediction coverage differs')
 return ids,values,done

def metrics(pred,y,rx,days):
 cm=np.zeros((6,6),dtype=np.int64);np.add.at(cm,(y,pred),1);recall=cm.diagonal()/cm.sum(1);precision=cm.diagonal()/np.maximum(cm.sum(0),1);f1=2*precision*recall/np.maximum(precision+recall,1e-12)
 rxacc={str(k):float((pred[rx==k]==y[rx==k]).mean()) for k in np.unique(rx)};dayacc={str(k):float((pred[days==k]==y[days==k]).mean()) for k in np.unique(days)}
 return dict(count=len(y),accuracy=float((pred==y).mean()),macro_f1=float(f1.mean()),confusion=cm.tolist(),per_TX_recall=recall.tolist(),receiver_accuracy=rxacc,day_accuracy=dayacc,worst_RX=min(rxacc.values()),worst_day=min(dayacc.values()))

def score(c,profile,base):
 if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Separate CPU-only scorer required')
 d.validate(c);ids,values,done=preflight(c,base);out=Path(base)/c['row_id']/'scores'
 if out.exists():raise FileExistsError('Existing score preserved')
 # First and only truth connection follows complete validated predictions for all7views.
 from experiments.cvs_report_ssl.evaluate import truth_arrays
 truth=d.read(profile['dataset']['scoring_truth']);y,rx,days=truth_arrays(truth,ids);records={v:metrics(values[v],y,rx,days) for v in d.VIEWS}
 out.mkdir();np.savez_compressed(out/'correctness.npz',ids=ids,**{v:values[v]==y for v in d.VIEWS})
 d.write(out/'metrics.json',dict(status='MODEL_PREDICTED_SCORED',config=c,views=list(d.VIEWS),count=COUNT,records=records,truth_last=True,independent_CPU_score=True,prediction_commit=done['commit'],head_fit_updates=0))
 print('SCORED '+c['row_id'],flush=True)

if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--row',required=True);p.add_argument('--profile',required=True);p.add_argument('--base',required=True);a=p.parse_args();score(d.config(next(r for r in d.rows() if r['row_id']==a.row)),d.read(a.profile),a.base)
