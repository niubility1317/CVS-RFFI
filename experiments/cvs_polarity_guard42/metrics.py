"""Source labels only;fixed registered-class metrics and prospective guard."""
import numpy as np
RX=(1,3,4,6,8)
def arrays(y,raw,reflected,rx):
 y=np.asarray(y);raw=np.asarray(raw);reflected=np.asarray(reflected);rx=np.asarray(rx)
 if y.ndim!=1 or y.dtype.kind not in 'iu' or rx.shape!=y.shape or rx.dtype.kind not in 'iu' or raw.shape!=(len(y),6) or reflected.shape!=raw.shape or not len(y) or not np.isfinite(raw).all() or not np.isfinite(reflected).all() or ((y<0)|(y>=6)).any() or set(rx.tolist())!=set(RX):raise ValueError('Complete legal visible source labels/all5RX/finite default6logits required;no hidden U')
 return y,raw,reflected,rx

def decide(y,raw,reflected,rx):
 y,raw,reflected,rx=arrays(y,raw,reflected,rx);a=raw.argmax(1)==y;b=reflected.argmax(1)==y;delta=float(b.mean()-a.mean());per={str(r):float(b[rx==r].mean()-a[rx==r].mean()) for r in RX};retention=float(b[a].mean()) if a.any() else 0.
 checks=dict(global_accuracy_preserved=delta>=-.005-1e-12,every_RX_accuracy_preserved=all(v>=-.015-1e-12 for v in per.values()),fixed_raw_correct_retained=retention>=.99-1e-12)
 return dict(status='SOURCE_GUARD_PASS' if all(checks.values()) else 'SOURCE_GUARD_FAIL',candidate_admissible=all(checks.values()),checks=checks,accuracy_raw=float(a.mean()),accuracy_reflected=float(b.mean()),accuracy_delta_pp=100*delta,per_RX_delta_pp={r:100*v for r,v in per.items()},fixed_raw_correct_count=int(a.sum()),fixed_raw_correct_retention=retention,repairs=int((b&~a).sum()),new_errors=int((a&~b).sum()),source_only=True,Ag_proved=False,target_access=False,automatic_training=False,goal_complete=False)

def metric(y,logits):
 y=np.asarray(y);p=np.asarray(logits).argmax(1);cm=np.bincount(6*y+p,minlength=36).reshape(6,6);tp=np.diag(cm);den=cm.sum(0)+cm.sum(1);f1=np.divide(2*tp,den,out=np.zeros(6,float),where=den>0);other=np.array(logits,copy=True);other[np.arange(len(y)),y]=-np.inf;margin=(np.asarray(logits)[np.arange(len(y)),y]-other.max(1))/30
 return dict(count=len(y),correct=int(tp.sum()),accuracy=float(tp.sum()/len(y)),macro_F1_all6=float(f1.mean()),confusion=cm.tolist(),mean_true_margin_over30=float(margin.mean()),margin_quantiles_over30=np.quantile(margin,[.05,.25,.5,.75,.95]).tolist())

def stratified(y,raw,reflected,rx,day):
 y,raw,reflected,rx=arrays(y,raw,reflected,rx);day=np.asarray(day)
 if day.shape!=y.shape or set(day.tolist())!={1,2,3}:raise ValueError('Original3 source days required')
 masks=[('overall','ALL',np.ones(len(y),bool))]+[('RX',str(r),rx==r) for r in RX]+[('day',str(d),day==d) for d in (1,2,3)]+[('TX',str(t),y==t) for t in range(6)]
 return [dict(view=view,dimension=d,stratum=s,**metric(y[mask],logit[mask])) for view,logit in [('raw',raw),('conjugate',reflected)] for d,s,mask in masks]
