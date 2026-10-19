"""Prediction-confidence counts only: not accuracy, precision or PL acceptance."""
import numpy as np
THRESHOLDS=(.92,.95,.97)
def describe(probabilities,rx,day,class_names):
 p=np.asarray(probabilities,dtype=np.float64);rx=np.asarray(rx);day=np.asarray(day);names=list(class_names)
 if p.ndim!=2 or p.shape[1]!=len(names) or not len(p) or rx.shape!=(len(p),) or day.shape!=(len(p),) or not np.isfinite(p).all() or (p<0).any() or (p>1).any() or not np.allclose(p.sum(1),1,atol=1e-5):raise ValueError('Finite normalized probabilities and source group metadata required')
 pred=p.argmax(1);conf=p.max(1);entropy=-(p*np.log(np.maximum(p,1e-12))).sum(1)
 def group(mask):
  n=int(mask.sum());records=[]
  for c,name in enumerate(names):
   chosen=mask&(pred==c);v=conf[chosen];records.append(dict(predicted_TX=name,predicted_count=int(chosen.sum()),prediction_fraction=float(chosen.sum()/n),mean_confidence=float(v.mean()) if len(v) else None,q10_confidence=float(np.quantile(v,.1)) if len(v) else None,confidence_tail_count={str(t):int((v>=t).sum()) for t in THRESHOLDS}))
  return dict(count=n,mean_confidence=float(conf[mask].mean()),mean_entropy=float(entropy[mask].mean()),per_predicted_TX=records,confidence_tail_count={str(t):int(((conf>=t)&mask).sum()) for t in THRESHOLDS})
 return dict(global_summary=group(np.ones(len(p),dtype=bool)),by_RX={str(r):group(rx==r) for r in np.unique(rx)},by_day={str(r):group(day==r) for r in np.unique(day)},U_truth_read=False,claim='Frozen sourceU prediction confidence only; no true class, precision/recall, historical gate acceptance or target-performance inference')
