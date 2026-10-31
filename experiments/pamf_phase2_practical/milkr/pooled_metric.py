"""Kernel pooled-within-class metric; exact scatter/centroid LOO downdates.

Only the kernel geometry is conditioned on full support. Each LOO decision
removes that physical sample from both its class centroid and within scatter.
"""
from dataclasses import dataclass
import numpy as np
from .semantic_residual import unit_features
from .support_calibrated import CONFIGS,LAMBDAS,similarity,make_kernel,target_and_score


def metric_parts(K,y):
    n=len(y);c=int(y.max())+1;k=n//c
    means=np.eye(c)[y]/k
    H=np.eye(n)-means@np.eye(c)[y].T
    residual=H@K;scatter=residual@H
    eig,U=np.linalg.eigh((scatter+scatter.T)/2)
    if not np.isfinite(eig).all() or eig.min() < -1e-8:
        raise FloatingPointError('Invalid within-class scatter')
    return means,H,np.maximum(eig,0),U,K@H@U


def solve_metric(K,y,parts,ridge):
    means,H,eig,U,R=parts;n=len(y);c=means.shape[1];k=n//c
    P=K-(R/(eig[None,:]+ridge))@R.T
    P=(P+P.T)/2
    cross=P@means;centroid=means.T@cross
    alpha=2*(means-H@U@((R.T@means)/(eig[:,None]+ridge)))
    intercept=-np.diag(centroid)
    if k==1:return alpha,intercept,None
    # Removing i changes scatter by -k/(k-1) r_i r_i^T.
    diag=np.diag(P);own=cross[np.arange(n),y]
    rnorm=diag-2*own+np.diag(centroid)[y]
    a=k/(k-1);den=ridge-a*rnorm
    if (den<=0).any() or not np.isfinite(den).all():
        raise FloatingPointError('Invalid scatter LOO downdate')
    factor=a/den;query_residual=diag-own
    cr=cross-centroid[y]
    cr[np.arange(n),y]=(k*cr[np.arange(n),y]-query_residual)/(k-1)
    loo_cross=cross.copy();loo_norm=np.broadcast_to(np.diag(centroid),(n,c)).copy()
    loo_cross[np.arange(n),y]=(k*own-diag)/(k-1)
    loo_norm[np.arange(n),y]=(k*k*np.diag(centroid)[y]-2*k*own+diag)/(k-1)**2
    loo=2*(loo_cross+factor[:,None]*query_residual[:,None]*cr)-(loo_norm+factor[:,None]*cr*cr)
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

    def score(self,features):
        x=unit_features(features);g,_,mix,scales=self.config
        return make_kernel(similarity(x,self.features,g),self.tau,mix,scales)@self.alpha+self.intercept


def fit(features,labels,ids,*,old_count=6,fixed=None,conditional_loo=True):
    x=unit_features(features);y=np.asarray(labels,dtype=np.int64);ids=tuple(map(str,ids))
    if y.ndim!=1 or not len(y) or (y<0).any() or len(set(ids))!=len(y) or len(ids)!=len(y):
        raise ValueError('Invalid physical support')
    count=int(y.max())+1;counts=np.bincount(y,minlength=count)
    if count<old_count or (counts!=counts[0]).any() or counts[0]<1 or len(x['z_id'])!=len(y):
        raise ValueError('Positive equal K per class required')
    k=int(counts[0])
    if k==1 or not conditional_loo:
        if fixed is None:raise ValueError('K1 requires frozen source rule')
        configs=[tuple(fixed['config'])];lambdas=[float(fixed['ridge'])]
    else:configs=CONFIGS;lambdas=LAMBDAS
    cache={};old=y<old_count;best=None;records=[]
    for config in configs:
        g,band,mix,scales=config
        if g not in cache:cache[g]=similarity(x,x,g)
        sim=cache[g];d=np.maximum(2-2*sim,0);od=d[np.ix_(old,old)];oy=y[old]
        if band=='nearest':tau=float(np.median(np.min(np.where(oy[:,None]!=oy[None,:],od,np.inf),axis=1)))
        else:tau=float(np.median(od[oy[:,None]!=oy[None,:]]))
        if not np.isfinite(tau) or tau<=0:raise FloatingPointError('Invalid support bandwidth')
        K=make_kernel(sim,tau,mix,scales);parts=metric_parts(K,y)
        for ridge in lambdas:
            alpha,intercept,loo=solve_metric(K,y,parts,ridge)
            if loo is None:objective=loss=0.
            else:
                objective=target_and_score(y,loo.argmax(1),old_count)
                p=np.exp(loo-loo.max(1,keepdims=True));p/=p.sum(1,keepdims=True)
                loss=float(np.mean((p-np.eye(count)[y])**2))
            rank=(objective,-loss,ridge);records.append(dict(config=list(config),ridge=ridge,conditional_LOO=objective,loss=loss))
            if best is None or rank>best[0]:best=(rank,config,tau,ridge,alpha,intercept)
    _,config,tau,ridge,alpha,intercept=best;alpha.flags.writeable=False;intercept.flags.writeable=False
    return State(x,alpha,intercept,config,tau,ridge,dict(rule='source_fixed' if k==1 or not conditional_loo else 'support_scatter_and_centroid_LOO',
        geometry_conditioned_on_full_support=True,centroid_and_scatter_remove_held_sample=True,query_fit=False,
        physical_count=len(y),candidate_count=len(records),records=records))
