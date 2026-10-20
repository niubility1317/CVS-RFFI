"""Three disjoint source packet partitions; only scalar scatter statistics exported."""
import itertools,time
from pathlib import Path
import numpy as np
from . import design as d

def partition(y,rx,day):
 out=np.full(len(y),-1,dtype=np.int8);rng=np.random.default_rng(d.PARTITION_SEED)
 for c in range(6):
  for r in d.RX:
   for t in d.DAYS:
    ix=np.flatnonzero((y==c)&(rx==r)&(day==t))
    if len(ix)<3 or len(ix)%3:raise ValueError('Three equally sized distinct rows per source cell required')
    ix=rng.permutation(ix)
    for part,group in enumerate(np.split(ix,3)):out[group]=part
 if (out<0).any():raise ValueError('Unexpected source labels/RX/day')
 return out

def covariance(z,groups):
 z=np.asarray(z,dtype=np.float64);groups=np.asarray(groups)
 if z.ndim!=2 or len(z)!=len(groups) or not np.isfinite(z).all():raise ValueError('Finite cached source feature rows required')
 residual=z.copy();values=np.unique(groups)
 for key in values:residual[groups==key]-=z[groups==key].mean(0)
 if len(z)<=len(values):raise ValueError('Positive residual degrees of freedom required')
 return residual.T@residual/(len(z)-len(values))

def precision(S):
 if S.ndim!=2 or S.shape[0]!=S.shape[1] or not np.isfinite(S).all() or not np.allclose(S,S.T,atol=1e-12,rtol=1e-12):raise ValueError('Finite symmetric source covariance required')
 eig,U=np.linalg.eigh(S);tau=float(np.trace(S)/len(S))
 if eig.min()<-1e-10 or tau<0:raise ValueError('Non-PSD covariance')
 if tau<=1e-14:return np.eye(len(S)),dict(zero_trace_fallback=True,trace=float(np.trace(S)),dimension=len(S),weight_min=1.,weight_max=1.,weight_mean=1.)
 eig=np.maximum(eig,0.);inverse=1/((1-d.SHRINKAGE)*eig+d.SHRINKAGE*tau);lo=0.;hi=d.PRECISION_MAX/float(inverse.min())
 for _ in range(64):
  middle=(lo+hi)/2
  if np.clip(middle*inverse,d.PRECISION_MIN,d.PRECISION_MAX).mean()<1:lo=middle
  else:hi=middle
 w=np.clip((lo+hi)/2*inverse,d.PRECISION_MIN,d.PRECISION_MAX);P=(U*w)@U.T
 if not np.isfinite(P).all() or abs(w.mean()-1)>1e-10:raise ValueError('Bounded mean1 precision failed')
 return P,dict(zero_trace_fallback=False,trace=float(np.trace(S)),dimension=len(S),eigenvalue_min=float(eig.min()),eigenvalue_max=float(eig.max()),effective_rank=float(eig.sum()**2/np.square(eig).sum()),weight_min=float(w.min()),weight_max=float(w.max()),weight_mean=float(w.mean()))

def metrics(z,y,rx,day,parts,a,b,P):
 means={};variance=[]
 for part in (1,2):
  mask=(parts==part)&np.isin(y,[a,b]);variance.append(covariance(z[mask],y[mask]))
  for c in (a,b):
   means[part,c]=z[(parts==part)&(y==c)].mean(0)
   for r in d.RX:
    means[part,c,r]=z[(parts==part)&(y==c)&(rx==r)].mean(0)
    for t in d.DAYS:means[part,c,r,t]=z[(parts==part)&(y==c)&(rx==r)&(day==t)].mean(0)
 dot=lambda x,v:float(x@P@v)
 global_signal=dot(means[1,a]-means[1,b],means[2,a]-means[2,b])
 cross=np.mean([dot(means[1,a,r]-means[1,b,r],means[2,a,s]-means[2,b,s]) for r in d.RX for s in d.RX if r!=s])
 cells=np.mean([dot(means[1,a,r,t]-means[1,b,r,t],means[2,a,r,t]-means[2,b,r,t]) for r in d.RX for t in d.DAYS])
 spread=[float(np.mean([dot(means[1,c,r]-means[1,c],means[2,c,r]-means[2,c]) for r in d.RX])) for c in (a,b)]
 noise=float(np.trace(P@((variance[0]+variance[1])/2)))
 if noise<=0 or not np.isfinite([global_signal,cross,cells,noise,*spread]).all():raise ValueError('Nondegenerate finite held-partition scatter required')
 return dict(global_class_signal=global_signal,cross_RX_signal=float(cross),same_RX_day_signal=float(cells),within_TX_variance=noise,SNR=float(cross/noise),per_TX_RX_spread=spread,per_TX_RX_spread_to_signal=[float(v/cross) if cross>0 else None for v in spread])

def load(root,seed):
 p=Path(root)/('BASE-s'+str(seed));meta=d.read(p/'source_geometry.json')
 if meta['status']!='SOURCE_GEOMETRY_COMPLETE' or meta['model_seed']!=seed or meta['source_code_commit']!=d.SOURCE_COMMIT or meta['source_role']!='V' or meta['count']!=27000 or not meta['model_state_unchanged'] or any(meta[k] for k in ('head_fit_updates','optimizer_updates','prototype_classifier_decisions','target_reads','truth_reads','U_TX_reads')) or meta['geometry']['class_names']!=list(d.CLASSES):raise ValueError('Existing original BASE sourceV cache only')
 with np.load(p/'source_features.npz',allow_pickle=False) as f:z=f['features'].astype(np.float64);y=f['y'];rx=f['rx'];day=f['day']
 if z.shape!=(27000,160) or not np.isfinite(z).all() or y.shape!=rx.shape or y.shape!=day.shape or y.shape!=(27000,) or not np.issubdtype(y.dtype,np.integer) or set(np.unique(y))!=set(range(6)) or set(np.unique(rx))!=set(d.RX) or set(np.unique(day))!=set(d.DAYS):raise ValueError('Full finite actual joint160 sourceV required')
 norms=np.linalg.norm(z,axis=1,keepdims=True)
 if (norms<=1e-12).any():raise ValueError('Nonzero source features required')
 z/=norms
 for c in range(6):
  for r in d.RX:
   for t in d.DAYS:
    if np.sum((y==c)&(rx==r)&(day==t))!=300:raise ValueError('Existing balanced physical source V cell differs')
 return z,y,rx,day

def analyze(root,row_callback=None):
 root=Path(root)
 if root.name!=d.SOURCE_RUN or d.read(root/'completion.json')['status']!='ARTIFACTS_COMPLETE':raise ValueError('Fixed completed source diagnostic required')
 start=time.perf_counter();rows=[];reference=None
 for seed in d.SEEDS:
  z,y,rx,day=load(root,seed);labels=np.stack((y,rx,day),1)
  if reference is not None and not np.array_equal(reference,labels):raise ValueError('Same source cache order needed for shared packet partition')
  reference=labels.copy();parts=partition(y,rx,day);pairs=[]
  for a,b in itertools.combinations(range(6),2):
   A=(parts==0)&np.isin(y,[a,b]);groups={'TX':y[A],'TX_RX_DAY':y[A]*15+np.searchsorted(d.RX,rx[A])*3+day[A]-1};values={'RAW':metrics(z,y,rx,day,parts,a,b,np.eye(160))};evidence={}
   for mode,group in groups.items():
    P,evidence[mode]=precision(covariance(z[A],group));values[mode]=metrics(z,y,rx,day,parts,a,b,P)
   pairs.append(dict(indices=[a,b],TX=[d.CLASSES[a],d.CLASSES[b]],calibration_A_count=int(A.sum()),evaluation_B_count=int(((parts==1)&np.isin(y,[a,b])).sum()),evaluation_C_count=int(((parts==2)&np.isin(y,[a,b])).sum()),metrics=values,precision_summary=evidence))
  rows.append(dict(model_seed=seed,source_count=27000,dimension=160,partition_counts=[int((parts==v).sum()) for v in range(3)],pairs=pairs))
  if row_callback is not None:row_callback(rows[-1])
  print('SOURCE_SEED_COMPLETE '+str(seed),flush=True)
 return dict(status='ANALYZED',run_id=d.RUN,source_run=d.SOURCE_RUN,source_commit=d.SOURCE_COMMIT,rows=rows,decision=d.decision(rows),seconds=time.perf_counter()-start,source_feature_rows=81000,unique_source_packets=27000,model_forwards=0,IQ_reads=0,target_reads=0,U_truth_reads=0,optimizer_updates=0,head_fit_updates=0,classifier_decisions=0,precision_matrices_exported=0,limitations=['Read-only scatter statistics in seen source receivers; not fitted default-head or prototype predictions.','Three model seeds share the same source packets; A/B/C have disjoint cache rows, not independent acquisition conditions.','A estimates covariance; only B/C estimate signals and RX dispersion. No V-derived matrix enters training or deployment.','Fixed shrinkage and bounded precision are descriptive hypotheses, not a guaranteed Fisher-optimal model or target improvement.'])
