"""Fixed nonstationary amplitude observation;constant packet gains cancel."""
import numpy as np
RX=(1,3,4,6,8)
def basis():
 n=np.arange(16,dtype=np.float64);return np.sqrt(2./16)*np.cos(np.pi*(n[:,None]+.5)*np.asarray([1.,2.])[None,:]/16)
def observe(x):
 x=np.asarray(x,dtype=np.float64)
 if x.ndim!=3 or x.shape[1:]!=(2,256) or not np.isfinite(x).all():raise ValueError('Finite original IQ256 required')
 p=(x*x).sum(1).reshape(len(x),16,16).mean(2);total=p.mean(1)
 if (total<=0).any():raise ValueError('Positive packet power required;no deleting samples')
 log=np.log(np.maximum(p/total[:,None],1e-6));log-=log.mean(1,keepdims=True)
 return log,log@basis(),total

def fit(cells):
 a=np.asarray(cells,dtype=np.float64)
 if a.shape!=(6,3,5,2) or not np.isfinite(a).all():raise ValueError('All6TX/3day/5RX/2modes required')
 folds=[];per_rx=np.zeros((5,2),dtype=np.float64)
 for tx in range(6):
  for day in range(3):
   fitTX=[t for t in range(6) if t!=tx];fitDay=[z for z in range(3) if z!=day]
   predicted=a[np.ix_(fitTX,fitDay,range(5),range(2))].mean((0,1));predicted-=predicted.mean(0)
   held=a[tx,day]-a[tx,day].mean(0);wrong=np.roll(predicted,1,axis=0)
   err=((held-predicted)**2).sum(1);zero=(held**2).sum(1);control=((held-wrong)**2).sum(1)
   per_rx[:,0]+=err;per_rx[:,1]+=zero
   e,z,w=map(float,(err.sum(),zero.sum(),control.sum()))
   folds.append(dict(TX=tx,day=day+1,fit_TX=fitTX,fit_days=[z+1 for z in fitDay],fit_RX=list(RX),predicted=predicted.tolist(),held_centered=held.tolist(),prediction_SSE=e,zero_SSE=z,wrong_RX_SSE=w,relative_MSE=e/z if z>1e-24 else None,better=bool(z>1e-24 and e<z)))
 sums={k:sum(v[k] for v in folds) for k in ('prediction_SSE','zero_SSE','wrong_RX_SSE')}
 rel=sums['prediction_SSE']/sums['zero_SSE'] if sums['zero_SSE']>1e-24 else None
 each={str(r):float(per_rx[i,0]/per_rx[i,1]) if per_rx[i,1]>1e-24 else None for i,r in enumerate(RX)}
 better=sum(v['better'] for v in folds)
 conditions=dict(relative_MSE_at_most_0_9=bool(rel is not None and rel<=.9),at_least15_of18_folds_better=better>=15,better_than_wrong_RX=sums['prediction_SSE']<sums['wrong_RX_SSE'],each_RX_better_than_zero=all(v is not None and v<1 for v in each.values()))
 return dict(folds=folds,totals=sums,relative_MSE=rel,better_folds=better,per_RX_relative_MSE=each,conditions=conditions,decision='PASS_SOURCE_ENVELOPE_ASSOCIATION_ONLY' if all(conditions.values()) else 'FAIL_FIXED_TEMPORAL_ENVELOPE_BASIS_CLOSED')
