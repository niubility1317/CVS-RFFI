"""Randomized conditional one-dimensional LHS;no changes to a single-packet prior."""
from statistics import NormalDist
import numpy as np
NORMAL=NormalDist()
def shadow_draws(mode,groups,seed):
 g=np.asarray(groups)
 if mode not in ('IID','LHS') or g.ndim!=1 or len(g)==0 or not np.issubdtype(g.dtype,np.integer) or np.any((g<0)|(g>8)):raise ValueError('IID/LHS and nonempty integer scene-state groups0..8 required')
 u=np.empty(len(g));ranks=np.full(len(g),-1,dtype=np.int64)
 for key in np.unique(g):
  ix=np.flatnonzero(g==key);n=len(ix);rng=np.random.default_rng(np.random.SeedSequence([int(seed),int(key)]));jitter=rng.random(n)
  while np.any(jitter==0):jitter[jitter==0]=rng.random(int(np.count_nonzero(jitter==0)))
  if mode=='LHS':
   rank=rng.permutation(n);value=(rank+jitter)/n
   while np.any(value>=1):
    bad=value>=1;jitter[bad]=rng.random(int(np.count_nonzero(bad)));value=(rank+jitter)/n
   u[ix]=value;ranks[ix]=rank
  else:u[ix]=jitter
 if not np.all((u>0)&(u<1)):raise ValueError('Open uniform variates required;no clipping')
 return u,ranks
def apply_shadow(mode,latents,groups,seed):
 if mode not in ('NATIVE','IID','LHS') or len(latents)!=len(groups):raise ValueError('Fixed native/IID/LHS aligned packet list required')
 old=[float(v.z) for v in latents]
 if mode=='NATIVE':u=np.full(len(groups),np.nan);rank=np.full(len(groups),-1,dtype=np.int64)
 else:u,rank=shadow_draws(mode,groups,seed)
 records=[]
 for i,v in enumerate(latents):
  key=int(groups[i]);n=sum(int(g)==key for g in groups);value=old[i] if mode=='NATIVE' else float(NORMAL.inv_cdf(float(u[i])))
  if not np.isfinite(value):raise ValueError('Finite original-prior shadow required')
  v.z=value;records.append(dict(mode=mode,group=key,group_size=n,uniform=None if mode=='NATIVE' else float(u[i]),stratum=None if mode!='LHS' else int(rank[i]),shadow_original=old[i],shadow_used=value,replacement_applied=mode!='NATIVE'))
 return records
