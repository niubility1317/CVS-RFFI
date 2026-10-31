"""MILKR on complete C4 distributions; one kernel row per physical record.

The kernel averages all sixteen phase-pair RBF/linear values. It keeps the
concentration of the orbit, without cross-query or self-diagonal calibration.
"""
from dataclasses import dataclass
import numpy as np
from . import support_calibrated as base,signature_kernel as sig
from .phase_encoder import flatten


def normalize(features):
    values=sig.normalize(flatten(features));n=len(features['z_id'])
    return {key:value.reshape(n,4,value.shape[1]) for key,value in values.items()}


def similarity(a,b,geometry):
    left={key:value.reshape(len(value)*4,value.shape[-1]) for key,value in a.items()}
    right={key:value.reshape(len(value)*4,value.shape[-1]) for key,value in b.items()}
    return sig.similarity(left,right,geometry).reshape(len(a['z_id']),4,len(b['z_id']),4)


def kernel(phase_similarity,tau,mix,scales):
    return base.make_kernel(phase_similarity,tau,mix,scales).mean(axis=(1,3))


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
        geometry,_,mix,scales=self.config
        K=kernel(similarity(normalize(features),self.features,geometry),self.tau,mix,scales)
        return K@self.alpha+self.intercept


def fit(features,labels,ids,*,old_count=6,fixed=None,conditional_loo=True):
    if fixed is None or fixed.get('ridge_reference_classes')!=6:raise ValueError('Frozen source six-class ridge reference required')
    x=normalize(features);y=np.asarray(labels,np.int64);ids=tuple(map(str,ids))
    if y.ndim!=1 or not len(y) or (y<0).any() or len(ids)!=len(y) or len(set(ids))!=len(y):raise ValueError('Invalid physical support')
    c=int(y.max())+1;counts=np.bincount(y,minlength=c)
    if c<old_count or (counts!=counts[0]).any() or counts[0]<1 or len(x['z_id'])!=len(y):raise ValueError('Positive equal physical K per class required')
    if tuple(fixed.get('config',())) not in sig.CONFIGS:raise ValueError('Wrong source phase-kernel rule')
    reference=float(fixed['ridge'])
    if not np.isfinite(reference) or reference<=0:raise ValueError('Positive source ridge required')
    k=int(counts[0]);configs=[tuple(fixed['config'])] if k==1 or not conditional_loo else sig.CONFIGS
    lambdas=[reference*c/6 if k==1 else reference] if k==1 or not conditional_loo else base.LAMBDAS
    best=None;records=[];sim_cache={};old=y<old_count
    for config in configs:
        geometry,band,mix,scales=config
        if geometry not in sim_cache:sim_cache[geometry]=similarity(x,x,geometry)
        sim=sim_cache[geometry];d=np.maximum(2-2*sim.mean(axis=(1,3)),0)
        od=d[np.ix_(old,old)];oy=y[old]
        if band=='nearest':tau=float(np.median(np.min(np.where(oy[:,None]!=oy[None,:],od,np.inf),axis=1)))
        else:tau=float(np.median(od[oy[:,None]!=oy[None,:]]))
        if not np.isfinite(tau) or tau<=0:raise FloatingPointError('No positive physical old-support bandwidth')
        K=kernel(sim,tau,mix,scales);parts=base.decompose(K,y)
        for ridge in lambdas:
            alpha,intercept,loo=base.coefficients(parts,ridge)
            objective=base.target_and_score(y,loo.argmax(1),old_count);loss=float(np.mean((loo-parts[2])**2));rank=(objective,-loss,ridge)
            records.append(dict(config=list(config),ridge=ridge,conditional_LOO=objective,LOO_loss=loss))
            if best is None or rank>best[0]:best=(rank,config,tau,ridge,alpha,intercept)
    _,config,tau,ridge,alpha,intercept=best;alpha.flags.writeable=False;intercept.flags.writeable=False
    return State(x,alpha,intercept,config,tau,ridge,dict(rule='source_fixed' if k==1 or not conditional_loo else 'physical_support_conditional_LOO',
        objective='old_macro_B_or_old_new_H_C_then_lower_squared_loss_then_larger_ridge',physical_count=len(y),phases_per_physical_record=4,
        physical_K=k,candidate_count=len(records),phase_pairs_per_kernel=16,query_fit=False,records=records,
        kernel_geometry_conditioned_on_full_support=True,not_unbiased_nested_cross_validation=True,
        K1_reference_ridge=reference,K1_current_class_scaling=bool(k==1),source_reference_classes=6))
