"""RX pattern of within-cell scale;fixed existing DCT1 observations only."""
import numpy as np
RX=(1,3,4,6,8)
def cells(coef,y,rx,day):
 c=np.asarray(coef,dtype=np.float64);y=np.asarray(y);rx=np.asarray(rx);day=np.asarray(day)
 if c.shape!=(6300,2) or any(v.shape!=(6300,) for v in (y,rx,day)) or not np.isfinite(c).all():raise ValueError('Complete finite original6300/two coefficients required')
 if set(y.tolist())!=set(range(6)) or set(rx.tolist())!=set(RX) or set(day.tolist())!={1,2,3}:raise ValueError('All original groups required')
 a=np.zeros((6,3,5));records=[];within=total=0.;global_mean=c[:,0].mean()
 for t in range(6):
  for d in (1,2,3):
   for r,rr in enumerate(RX):
    v=c[(y==t)&(day==d)&(rx==rr),0]
    if len(v)!=70:raise ValueError('All90 cells must have70;no filtering')
    med=float(np.median(v));mad=float(np.median(abs(v-med)));a[t,d-1,r]=mad
    within+=float(((v-v.mean())**2).sum());total+=float(((v-global_mean)**2).sum())
    records.append(dict(TX=t,day=d,RX=rr,count=70,mean=float(v.mean()),median=med,MAD=mad,variance=float(v.var())))
 return a,records,dict(within_cell_SSE=within,total_SSE=total,within_cell_fraction=within/total if total else None)
def fit(scales):
 s=np.asarray(scales,dtype=np.float64)
 if s.shape!=(6,3,5) or not np.isfinite(s).all() or (s<0).any():raise ValueError('6TX/3day/5RX finite nonnegative MAD required')
 a=np.log(np.maximum(s,1e-12));folds=[];sums=dict(RX_SSE=0.,zero_SSE=0.,wrong_RX_SSE=0.,TX_specific_SSE=0.);per=np.zeros((5,2));better=0
 for t in range(6):
  for d in range(3):
   ts=[v for v in range(6) if v!=t];ds=[v for v in range(3) if v!=d]
   p=a[np.ix_(ts,ds,range(5))].mean((0,1));p-=p.mean();h=a[t,d]-a[t,d].mean();q=a[t,ds].mean(0);q-=q.mean()
   e=(h-p)**2;z=h*h;v=dict(RX_SSE=float(e.sum()),zero_SSE=float(z.sum()),wrong_RX_SSE=float(((h-np.roll(p,1))**2).sum()),TX_specific_SSE=float(((h-q)**2).sum()))
   for k,x in v.items():sums[k]+=x
   per[:,0]+=e;per[:,1]+=z;better+=int(v['RX_SSE']<v['zero_SSE'])
   folds.append(dict(TX=t,day=d+1,fit_TX=ts,fit_days=[x+1 for x in ds],fit_RX=list(RX),predicted_RX_log_MAD=p.tolist(),held_centered_log_MAD=h.tolist(),TX_specific_prediction=q.tolist(),**v))
 ratios={str(r):float(per[i,0]/per[i,1]) if per[i,1]>1e-18 else (0. if per[i,0]<=1e-12 else None) for i,r in enumerate(RX)}
 rel=sums['RX_SSE']/sums['zero_SSE'] if sums['zero_SSE']>1e-18 else None
 cond=dict(RX_relative_MSE_at_most_point9=rel is not None and rel<=.9,at_least15_of18_folds_better=better>=15,each_RX_not_worse_than_zero=all(v is not None and v<=1+1e-12 for v in ratios.values()),better_than_wrong_RX=sums['RX_SSE']<sums['wrong_RX_SSE'],shared_RX_not_5percent_worse_than_TX_specific=sums['RX_SSE']<=1.05*sums['TX_specific_SSE']+1e-12)
 return dict(decision='PASS_SOURCE_RX_VARIABILITY_ONLY' if all(cond.values()) else 'FAIL_FIXED_WITHIN_CELL_ENVELOPE_VARIABILITY_CLOSED',conditions=cond,relative_MSE=rel,better_folds=better,per_RX_relative_MSE=ratios,totals=sums,folds=folds,MAD_median=float(np.median(s)),MAD_zero_cells=int((s==0).sum()),by_RX={str(r):dict(median=float(np.median(s[:,:,i])),q10=float(np.quantile(s[:,:,i],.1)),q90=float(np.quantile(s[:,:,i],.9))) for i,r in enumerate(RX)})
