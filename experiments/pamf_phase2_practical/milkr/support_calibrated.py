"""Inductive kernel ridge with support-only conditional leave-one-out selection.

The kernel grid is fixed before target access. K=1 uses a source-selected rule;
K>1 selects on current support labels only, never on a query or its statistics.
"""
from dataclasses import dataclass
import numpy as np
from .semantic_residual import unit_features, NAMES

LAMBDAS=(.001,.01,.1,1.,10.)
CONFIGS=tuple((geometry,band,mix,scales) for geometry in ('historical','balanced','identity')
              for band in ('nearest','all') for mix in (0.,.5) for scales in ('single','multi'))


def similarity(a,b,geometry):
    z=a['z_id']@b['z_id'].T
    if geometry=='identity':return z
    fft=a['fft']@b['fft'].T
    aux=sum(a[n]@b[n].T for n in NAMES[2:])/3
    weight=16 if geometry=='historical' else 1
    bg=(z+weight*fft)/(1+weight)
    interaction=(bg+aux+bg*aux)/3
    return interaction if geometry=='historical' else (z+interaction)/2


def make_kernel(sim,tau,mix,scales):
    d=np.maximum(2-2*sim,0)
    scale=(1.,) if scales=='single' else (.25,1.,4.)
    local=sum(np.exp(-d/(tau*s)) for s in scale)/len(scale)
    return mix*sim+(1-mix)*local


def target_and_score(labels,pred,old_count):
    count=int(labels.max())+1
    rec=np.asarray([(pred[labels==c]==c).mean() for c in range(count)])
    old=float(rec[:old_count].mean())
    if count==old_count:return old
    new=float(rec[old_count:].mean())
    return 2*old*new/(old+new) if old+new else 0.


def decompose(K,y):
    mean=K.mean(0);grand=float(mean.mean())
    centered=K-mean[:,None]-mean[None,:]+grand
    eig,U=np.linalg.eigh(centered)
    if eig.min() < -1e-8 or not np.isfinite(eig).all():
        raise FloatingPointError('Kernel is not PSD')
    eig=np.maximum(eig,0)
    Y=np.eye(int(y.max())+1)[y];average=Y.mean(0);Yc=Y-average
    return eig,U,Y,average,Yc,mean


def coefficients(parts,lam):
    eig,U,Y,average,Yc,mean=parts;n=len(Y)
    alpha=U@((U.T@Yc)/(eig[:,None]+lam))
    alpha-=alpha.mean(0,keepdims=True)
    intercept=average-mean@alpha
    # M is the upper-left inverse of the kernel+unpenalized-intercept system.
    diag=(U*U)@(1/(eig+lam))-1/(lam*n)
    if (diag<=0).any() or not np.isfinite(diag).all():
        raise FloatingPointError('Invalid conditional LOO diagonal')
    loo=Y-alpha/diag[:,None]
    return alpha,intercept,loo


@dataclass(frozen=True)
class State:
    features:dict
    alpha:np.ndarray
    intercept:np.ndarray
    config:tuple
    tau:float
    ridge:float
    selection:dict
    normalizer:object=unit_features
    similarity_fn:object=similarity

    def score(self,features):
        v=self.normalizer(features);g,_,mix,scales=self.config
        K=make_kernel(self.similarity_fn(v,self.features,g),self.tau,mix,scales)
        return K@self.alpha+self.intercept


def fit(features,labels,ids,*,old_count=6,fixed=None,conditional_loo=True,
        normalizer=unit_features,similarity_fn=similarity,configs=CONFIGS):
    x=normalizer(features);y=np.asarray(labels,dtype=np.int64);ids=tuple(map(str,ids))
    if y.ndim!=1 or not len(y) or (y<0).any() or len(set(ids))!=len(y) or len(ids)!=len(y):
        raise ValueError('Invalid physical support')
    count=int(y.max())+1;counts=np.bincount(y,minlength=count)
    if count<old_count or (counts!=counts[0]).any() or counts[0]<1 or len(x['z_id'])!=len(y):
        raise ValueError('Positive equal K per registered class required')
    k=int(counts[0])
    if k==1 or not conditional_loo:
        if fixed is None:raise ValueError('K1 requires a frozen source rule')
        configs=[tuple(fixed['config'])];lambdas=[float(fixed['ridge'])]
    else:lambdas=LAMBDAS
    scope='old' if fixed is None else fixed.get('bandwidth_scope','old')
    if scope not in ('old','registered'):raise ValueError('Invalid frozen bandwidth scope')
    scale_mask=y<old_count if scope=='old' else np.ones(len(y),bool)
    best=None;records=[];sim_cache={}
    for config in configs:
        geometry,band,mix,scales=config
        if geometry not in sim_cache:sim_cache[geometry]=similarity_fn(x,x,geometry)
        sim=sim_cache[geometry];d=np.maximum(2-2*sim,0)
        # The scope is frozen before query access; every row is current support.
        od=d[np.ix_(scale_mask,scale_mask)];oy=y[scale_mask]
        if band=='nearest':tau=float(np.median(np.min(np.where(oy[:,None]!=oy[None,:],od,np.inf),axis=1)))
        else:tau=float(np.median(od[oy[:,None]!=oy[None,:]]))
        if not np.isfinite(tau) or tau<=0:raise FloatingPointError('No positive support bandwidth scale')
        K=make_kernel(sim,tau,mix,scales);parts=decompose(K,y)
        for ridge in lambdas:
            alpha,intercept,loo=coefficients(parts,ridge)
            objective=target_and_score(y,loo.argmax(1),old_count)
            loss=float(np.mean((loo-parts[2])**2))
            rank=(objective,-loss,ridge)
            records.append(dict(config=list(config),ridge=ridge,conditional_LOO=objective,LOO_loss=loss))
            if best is None or rank>best[0]:best=(rank,config,tau,ridge,alpha,intercept)
    _,config,tau,ridge,alpha,intercept=best
    alpha.flags.writeable=False;intercept.flags.writeable=False
    return State(x,alpha,intercept,config,tau,ridge,dict(rule='source_fixed' if k==1 or not conditional_loo else 'support_conditional_LOO',
        objective='old_macro_B_or_old_new_H_C_then_lower_squared_loss_then_larger_ridge',
        kernel_geometry_conditioned_on_full_support=True,not_unbiased_nested_cross_validation=True,
        physical_count=len(y),bandwidth_scope=scope,bandwidth_physical_count=int(scale_mask.sum()),
        candidate_count=len(records),query_fit=False,records=records),normalizer,similarity_fn)
