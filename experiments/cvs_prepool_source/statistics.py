"""Class-conditional cross-half scatter, with no fitted decision rule."""
import numpy as np

def halves(y,rx,day,ids,seed):
 y,rx,day=np.asarray(y),np.asarray(rx),np.asarray(day);rng=np.random.default_rng(seed);result=[]
 if len(ids)!=1800 or len(set(ids))!=1800 or set(y)!=set(range(6)):raise ValueError('Fixed labeled physical diagnostic panel required')
 for r in sorted(set(rx)):
  for t in sorted(set(day)):
   groups=[]
   for label in range(6):
    ix=np.flatnonzero((rx==r)&(day==t)&(y==label));ix=np.asarray(sorted(ix,key=lambda i:ids[i]))
    if len(ix)!=20:raise ValueError('Each TX/RX/day must have 20 unique source samples')
    shuffled=rng.permutation(ix);groups.append((shuffled[:10],shuffled[10:]))
   result.append((int(r),int(t),groups))
 if len(result)!=15:raise ValueError('All five source RX and three source days required')
 return result

def scatter(features,split):
 x=np.asarray(features,np.float64)
 if x.shape!=(1800,144) or not np.isfinite(x).all():raise ValueError('Fixed finite 144D nonparametric descriptors required')
 records=[]
 for rx,day,groups in split:
  a=np.asarray([x[i].mean(0) for i,j in groups]);b=np.asarray([x[j].mean(0) for i,j in groups]);a-=a.mean(0);b-=b.mean(0);s=float((a*b).sum()/6)
  ix=np.concatenate([np.concatenate((i,j)) for i,j in groups]);z=x[ix];total=float(np.square(z-z.mean(0)).sum()/len(z))
  records.append(dict(RX=rx,day=day,count=len(ix),cross_half_TX_scatter=s,total_scatter=total,ratio=s/total if total>0 else 0.))
 s=sum(v['cross_half_TX_scatter'] for v in records);t=sum(v['total_scatter'] for v in records)
 return dict(cross_half_TX_scatter_sum=s,total_scatter_sum=t,ratio=s/t if t>0 else 0.,positive_RX_day_count=sum(v['cross_half_TX_scatter']>0 for v in records),strata=records)
