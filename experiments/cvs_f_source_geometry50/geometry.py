"""Parameter-free class/RX/day geometry; never makes classifier decisions."""
import numpy as np
from statistics import mean,stdev
EPS=1e-12

def unit(v):
 n=np.linalg.norm(v,axis=-1,keepdims=True)
 return v/np.maximum(n,EPS),np.squeeze(n,axis=-1)
def summary(values):
 a=np.asarray(values,dtype=np.float64)
 return dict(count=int(a.size),mean=float(a.mean()) if a.size else None,q10=float(np.quantile(a,.1)) if a.size else None,median=float(np.median(a)) if a.size else None,q90=float(np.quantile(a,.9)) if a.size else None)
def spread(z,groups):
 centers=[];degenerate=0
 for group in np.unique(groups):
  c,n=unit(z[groups==group].mean(0));degenerate+=int(n<=EPS)
  if n>EPS:centers.append(c)
 if len(centers)<2:return dict(mean_one_minus_cosine=None,pairs=0,degenerate_centers=degenerate)
 vals=[1-float(np.clip(a@b,-1,1)) for i,a in enumerate(centers) for b in centers[i+1:]]
 return dict(mean_one_minus_cosine=float(np.mean(vals)),pairs=len(vals),degenerate_centers=degenerate)
def describe(features,y,rx,day,class_names):
 f=np.asarray(features,dtype=np.float64);y=np.asarray(y);rx=np.asarray(rx);day=np.asarray(day);names=list(class_names);K=len(names)
 if f.ndim!=2 or not len(f) or K<2 or any(v.shape!=(len(f),) for v in (y,rx,day)) or set(np.unique(y))!=set(range(K)) or not np.isfinite(f).all():raise ValueError('Complete finite labeled source features required')
 z,norms=unit(f);means=np.stack([z[y==c].mean(0) for c in range(K)]);centers,center_norm=unit(means);similarity=np.clip(centers@centers.T,-1,1);records=[]
 for c,name in enumerate(names):
  chosen=y==c;zc=z[chosen];rx_spread=spread(zc,rx[chosen]);day_spread=spread(zc,day[chosen]);valid_margins=[];by_rx=[]
  for r in np.unique(rx):
   group=chosen&(rx==r);other=rx!=r
   if not group.any():continue
   if not all(np.any((y==k)&other) for k in range(K)):
    by_rx.append(dict(RX=int(r),count=int(group.sum()),valid_margin_count=0,reason='missing other-RX class'));continue
   out_centers,out_norm=unit(np.stack([z[(y==k)&other].mean(0) for k in range(K)]))
   if (out_norm<=EPS).any():
    by_rx.append(dict(RX=int(r),count=int(group.sum()),valid_margin_count=0,reason='degenerate other-RX centroid'));continue
   # Continuous labeled geometry only: own-class cosine minus strongest
   # other-class cosine; no prototype predictions, accuracy or fitted head.
   cosine=z[group]@out_centers.T;margin=cosine[:,c]-np.max(cosine[:,np.arange(K)!=c],axis=1);valid_margins.extend(margin.tolist());by_rx.append(dict(RX=int(r),count=int(group.sum()),valid_margin_count=int(len(margin)),margin=summary(margin)))
  closest=float(max(similarity[c,k] for k in range(K) if k!=c)) if np.all(center_norm>EPS) else None
  denom=None if closest is None else max(1-closest,1e-6);dispersion=rx_spread['mean_one_minus_cosine']
  records.append(dict(class_id=c,TX=name,count=int(chosen.sum()),feature_norm=summary(norms[chosen]),zero_feature_count=int((norms[chosen]<=EPS).sum()),class_centroid_norm=float(center_norm[c]),within_TX_one_minus_cosine=None if center_norm[c]<=EPS else float(np.mean(1-zc@centers[c])),RX_centroid_spread=rx_spread,day_centroid_spread=day_spread,closest_other_TX_centroid_cosine=closest,RX_spread_to_inter_TX_gap=None if denom is None or dispersion is None else float(dispersion/denom),out_RX_margin=summary(valid_margins),by_RX=by_rx))
 return dict(count=len(f),dimensions=f.shape[1],class_names=names,source_receivers=[int(v) for v in np.unique(rx)],source_days=[int(v) for v in np.unique(day)],per_TX=records,TX_centroid_cosine_matrix=similarity.tolist(),head_fit_updates=0,prototype_classifier_decisions=0,definition='L2 feature angles; centroids are descriptive source class/group means, never classifier parameters or class decisions')
def aggregate(results):
 names=results[0]['geometry']['class_names'];fields=('within_TX_one_minus_cosine','RX_spread_to_inter_TX_gap','closest_other_TX_centroid_cosine');rows=[]
 def stats(values):
  v=[float(x) for x in values if x is not None]
  return dict(n=len(v),mean=mean(v) if v else None,sample_SD=stdev(v) if len(v)>1 else None)
 for c,name in enumerate(names):
  rr=[r['geometry']['per_TX'][c] for r in results];row=dict(TX=name,class_id=c)
  row.update({k:stats([r[k] for r in rr]) for k in fields});row['out_RX_margin_q10']=stats([r['out_RX_margin']['q10'] for r in rr]);row['RX_centroid_spread']=stats([r['RX_centroid_spread']['mean_one_minus_cosine'] for r in rr]);row['day_centroid_spread']=stats([r['day_centroid_spread']['mean_one_minus_cosine'] for r in rr]);rows.append(row)
 return dict(models=len(results),per_TX=rows,head_fit_updates=0,prototype_classifier_decisions=0)
