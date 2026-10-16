"""Scalar properness observation and cross-TX/day RX prediction; no identity fit."""
from itertools import combinations,permutations
import numpy as np
RX=(3,4,6,8)
EPS=1e-6

def circularity(w):
 centered=w-w.mean(axis=1,keepdims=True);power=np.mean(np.abs(centered)**2,axis=1)
 return np.abs(np.mean(centered**2,axis=1))/(power+EPS**2)

def observe(x):
 x=np.asarray(x,dtype=np.float64)
 if x.ndim!=3 or x.shape[1:]!=(2,256) or not np.isfinite(x).all():raise ValueError('Finite native IQ[N,2,256] required')
 w=x[:,0,80:160]+1j*x[:,1,80:160];cross=np.sum(w[:,20:]*w[:,:-20].conj(),axis=1)
 omega=np.arctan2(cross.imag,cross.real+EPS**2)/20
 corrected=w*np.exp(-1j*omega[:,None]*np.arange(80)[None,:])
 cycles=corrected.reshape(-1,4,20)
 coherence=np.abs(cross)/np.sqrt(np.sum(np.abs(w[:,20:])**2,axis=1)*np.sum(np.abs(w[:,:-20])**2,axis=1)+EPS**4)
 value=dict(primary=circularity(corrected),raw=circularity(w),odd=circularity(cycles[:,::2].reshape(-1,40)),even=circularity(cycles[:,1::2].reshape(-1,40)),coherence=coherence)
 if any(not np.isfinite(v).all() or (v<0).any() or (v>1+1e-12).any() for v in value.values()):raise ValueError('Finite bounded circularity/coherence required')
 return value

def cosine(a,b):
 n=float(np.linalg.norm(a)*np.linalg.norm(b))
 return float(np.dot(a.ravel(),b.ravel())/n) if n>1e-20 else 0.

def analyze_cells(cells,odd,even):
 cells,odd,even=(np.asarray(v,dtype=np.float64) for v in (cells,odd,even))
 if any(v.shape!=(4,3,6) or not np.isfinite(v).all() for v in (cells,odd,even)):raise ValueError('All72 scalar means required')
 wrong=[p for p in permutations(range(4)) if all(i!=v for i,v in enumerate(p))]
 folds=[];bs=[];ms=[];wr=[[] for _ in wrong]
 for hd in range(3):
  for ht in range(6):
   train=cells[:,[j for j in range(3) if j!=hd]][:,:,[t for t in range(6) if t!=ht]]
   mean=train.mean((1,2));pooled=float(mean.mean());actual=cells[:,hd,ht]
   base=(actual-pooled)**2;model=(actual-mean)**2
   gain=1-float(model.sum()/base.sum()) if base.mean()>1e-16 else None
   folds.append(dict(held_TX=ht,held_day=hd+1,risk_gain=gain,baseline_risk=float(base.mean()),RX_risk=float(model.mean())))
   bs.append(base);ms.append(model)
   for j,p in enumerate(wrong):wr[j].append((actual-mean[list(p)])**2)
 b=np.asarray(bs);m=np.asarray(ms);baseline=float(b.mean());risk=float(m.mean());gain=1-risk/baseline if baseline>1e-16 else None
 wrong_risks=[float(np.mean(v)) for v in wr];wrong_gains=[1-v/baseline if baseline>1e-16 else None for v in wrong_risks]
 per_rx={str(r):1-float(m[:,j].mean()/b[:,j].mean()) if b[:,j].mean()>1e-16 else None for j,r in enumerate(RX)}
 splits=[]
 for subset in combinations(range(6),3):
  if 0 not in subset:continue
  other=[t for t in range(6) if t not in subset];a=cells[:,:,subset].mean((1,2));a-=a.mean();v=cells[:,:,other].mean((1,2));v-=v.mean();splits.append(dict(TX_half=list(subset),other_TX=other,center_cosine=cosine(a,v)))
 minimum=min(v['center_cosine'] for v in splits);centers=cells.mean((1,2));centers-=centers.mean()
 oddc=odd.mean((1,2));oddc-=oddc.mean();evenc=even.mean((1,2));evenc-=evenc.mean()
 signal=float(np.mean(centers**2));difference=float(np.mean((oddc-evenc)**2));repeat=signal/max(difference,1e-30)
 positive=sum(v['risk_gain'] is not None and v['risk_gain']>0 for v in folds);over=gain-max(wrong_gains) if gain is not None else None
 conditions=dict(nonzero_baseline=baseline>1e-16,mean_risk_gain=gain is not None and gain>=.05,positive_folds=positive>=14,all_RX_nonnegative=all(v is not None and v>=0 for v in per_rx.values()),above_best_wrong_RX=over is not None and over>=.02,all_TX_split_center_cosine=minimum>=.8,RX_cycle_repeatability=repeat>=2.)
 return dict(decision='PASS' if all(conditions.values()) else 'FAIL',conditions=conditions,mean_risk_gain=gain,baseline_cell_risk=baseline,RX_cell_risk=risk,positive_folds=positive,total_folds=18,per_RX_risk_gain=per_rx,wrong_RX_derangements=[list(v) for v in wrong],wrong_RX_risks=wrong_risks,wrong_RX_risk_gains=wrong_gains,gain_over_best_wrong_RX=over,TX_split_controls=splits,min_TX_split_center_cosine=minimum,fit_RX_center_offsets=centers.tolist(),RX_center_power=signal,odd_even_RX_center_difference_power=difference,RX_center_repeatability_ratio=repeat,folds=folds)

def crossfit(values,tx,rx,day):
 tx,rx,day=(np.asarray(v) for v in (tx,rx,day))
 if any(v.shape!=(5040,) for v in (tx,rx,day)) or set(tx)!=set(range(6)) or set(rx)!=set(RX) or set(day)!={1,2,3}:raise ValueError('Complete visible L5040 fitRX roles required')
 if set(values)!={'primary','raw','odd','even','coherence'} or any(np.asarray(v).shape!=(5040,) or not np.isfinite(v).all() or (np.asarray(v)<0).any() or (np.asarray(v)>1+1e-12).any() for v in values.values()):raise ValueError('Complete bounded scalar observations required')
 cubes={k:np.zeros((4,3,6)) for k in values};cells=[]
 for ri,r in enumerate(RX):
  for di,d in enumerate((1,2,3)):
   for t in range(6):
    mask=(rx==r)&(day==d)&(tx==t)
    if mask.sum()!=70:raise ValueError('Every fit RX/day/TX cell requires original70 L packets')
    row=dict(RX=r,day=d,TX=t,count=70)
    for k,v in values.items():
     data=np.asarray(v)[mask];cubes[k][ri,di,t]=data.mean();row[k+'_mean']=float(data.mean());row[k+'_std']=float(data.std(ddof=1))
    cells.append(row)
 result=analyze_cells(cubes['primary'],cubes['odd'],cubes['even'])
 result.update(cells=cells,primary='CFOcompensated centered circularity magnitude',secondary_not_qualified=dict(raw_cell_range=[float(cubes['raw'].min()),float(cubes['raw'].max())],mean_coherence=float(np.mean(values['coherence']))),interpretation='RX-associated scalar predictability only;not identified TX/RX hardware or recognition gain')
 return result
