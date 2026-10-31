"""PSD local-RBF features anchored only in current physical support.

Centre scales and column norms are fitted on support. Each query uses its own
row normalization and a collapsed readout; no query batch statistics are used.
"""
from dataclasses import dataclass
import numpy as np
from . import signature_kernel as sig,support_calibrated as base
from .nested_fisher import folds

POLICY='column_norm_row_unit_v1'


def factors(scales):return (1.,) if scales=='single' else (.25,1.,4.)


def logs(sim,tau,scales):
    distance=np.maximum(2-2*sim,0)
    return np.concatenate([-distance/(tau[None,:]*scale) for scale in factors(scales)],axis=1)


def embedding(sim,tau,column_norm,scales):
    value=logs(sim,tau,scales)-np.log(column_norm)[None,:]
    # A common positive factor cancels in each unit row; avoid RBF underflow.
    value-=value.max(axis=1,keepdims=True);value=np.exp(value)
    return value/np.linalg.norm(value,axis=1,keepdims=True)


def basis(sim,labels,band,scales):
    y=np.asarray(labels,np.int64);distance=np.maximum(2-2*sim,0)
    if band=='nearest':tau=np.min(np.where(y[:,None]!=y[None,:],distance,np.inf),axis=1)
    else:tau=np.asarray([np.median(row[y!=label]) for row,label in zip(distance,y)])
    if not np.isfinite(tau).all() or (tau<=0).any():raise FloatingPointError('No positive support-centre bandwidth')
    raw=np.exp(logs(sim,tau,scales));column=np.linalg.norm(raw,axis=0)
    if not np.isfinite(column).all() or (column<=0).any():raise FloatingPointError('Invalid support column norm')
    return tau,column,embedding(sim,tau,column,scales)


def context(features,labels,config):
    geometry,band,mix,scales=config;sim=sig.similarity(features,features,geometry)
    tau,column,phi=basis(sim,labels,band,scales);K=mix*sim+(1-mix)*(phi@phi.T)
    return dict(features=features,tau=tau,column=column,phi=phi,parts=base.decompose(K,labels))


@dataclass(frozen=True)
class State:
    features:dict
    alpha:np.ndarray
    intercept:np.ndarray
    config:tuple
    tau:float
    ridge:float
    selection:dict
    tau_vector:np.ndarray
    column_norm:np.ndarray
    local_weights:np.ndarray

    def score(self,features):
        geometry,_,mix,scales=self.config;sim=sig.similarity(sig.normalize(features),self.features,geometry)
        phi=embedding(sim,self.tau_vector,self.column_norm,scales)
        return (1-mix)*(phi@self.local_weights)+mix*(sim@self.alpha)+self.intercept


def fit(features,labels,ids,*,old_count=6,fixed=None,conditional_loo=True):
    if fixed is None or fixed.get('local_anchor_policy')!=POLICY:raise ValueError('Frozen local-anchor policy required')
    if fixed.get('ridge_reference_classes')!=6 or tuple(fixed.get('config',())) not in sig.CONFIGS:raise ValueError('Frozen source six-class rule required')
    reference=float(fixed['ridge'])
    if not np.isfinite(reference) or reference<=0:raise ValueError('Positive source ridge required')
    x=sig.normalize(features);y=np.asarray(labels,np.int64);ids=tuple(map(str,ids))
    if y.ndim!=1 or not len(y) or (y<0).any() or len(ids)!=len(y) or len(set(ids))!=len(y):raise ValueError('Invalid physical support')
    count=int(y.max())+1;counts=np.bincount(y,minlength=count)
    if count<old_count or (counts!=counts[0]).any() or counts[0]<1 or len(x['z_id'])!=len(y):raise ValueError('Positive equal physical K per class required')
    k=int(counts[0]);records=[];audit=[]
    if k==1 or not conditional_loo:
        config=tuple(fixed['config']);ridge=reference*count/6 if k==1 else reference
    else:
        split=folds(y,ids);best=None;target=np.eye(count)[y]
        audit=[dict(train=len(train),held=len(held)) for train,held in split]
        for config in sig.CONFIGS:
            geometry,_,mix,scales=config;predictions={lam:np.zeros((len(y),count)) for lam in base.LAMBDAS}
            for train,held in split:
                tx={key:value[train] for key,value in x.items()};hx={key:value[held] for key,value in x.items()}
                ctx=context(tx,y[train],config);sim=sig.similarity(hx,tx,geometry);phi=embedding(sim,ctx['tau'],ctx['column'],scales)
                for lam in base.LAMBDAS:
                    alpha,intercept,_=base.coefficients(ctx['parts'],lam)
                    predictions[lam][held]=(1-mix)*(phi@(ctx['phi'].T@alpha))+mix*(sim@alpha)+intercept
            for lam,scores in predictions.items():
                objective=base.target_and_score(y,scores.argmax(1),old_count);loss=float(np.mean((scores-target)**2));rank=(objective,-loss,lam)
                records.append(dict(config=list(config),ridge=lam,physical_OOF_H=objective,OOF_loss=loss))
                if best is None or rank>best[0]:best=(rank,config,lam)
        _,config,ridge=best
    ctx=context(x,y,config);alpha,intercept,_=base.coefficients(ctx['parts'],ridge);weights=ctx['phi'].T@alpha
    for value in (alpha,intercept,weights,ctx['tau'],ctx['column']):value.flags.writeable=False
    selection=dict(rule='source_fixed' if not audit else 'three_physical_fold_OOF',physical_OOF=bool(audit),folds=audit,
        centre_count=len(y),physical_count=len(y),physical_K=k,feature_dimension=ctx['phi'].shape[1],query_fit=False,
        centres_bandwidth_column_norm_and_coefficients_exclude_held_physical_rows=bool(audit),
        local_anchor_policy=POLICY,bandwidth_scope='registered',bandwidth_physical_count=len(y),
        candidate_count=len(records) if audit else 1,records=records,K1_current_class_scaling=bool(k==1),
        extra_state_bytes=weights.nbytes+ctx['tau'].nbytes+ctx['column'].nbytes)
    return State(x,alpha,intercept,config,float(np.median(ctx['tau'])),float(ridge),selection,ctx['tau'],ctx['column'],weights)
