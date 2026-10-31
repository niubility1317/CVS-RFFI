"""A fixed frozen-head probability kernel mixed with continuous RF geometry."""
from dataclasses import dataclass
import numpy as np
from . import signature_kernel as sig,support_calibrated as base
from .nested_fisher import folds

WEIGHT=.25


def normalize(features):
    if set(features)!=set(base.NAMES)|{'sig_views','teacher_probs'}:raise ValueError('Exact teacher-kernel features required')
    out=sig.normalize({key:value for key,value in features.items() if key!='teacher_probs'});p=np.array(features['teacher_probs'],np.float64,copy=True)
    if p.ndim!=2 or len(p)!=len(out['z_id']) or not 2<=p.shape[1]<=6 or not np.isfinite(p).all() or (p<0).any() or not np.allclose(p.sum(1),1):raise ValueError('Invalid frozen-head probabilities')
    p/=np.linalg.norm(p,axis=1,keepdims=True);p.flags.writeable=False;out['teacher_probs']=p
    return out


def similarity(a,b,geometry):
    return (1-WEIGHT)*sig.similarity(a,b,geometry)+WEIGHT*(a['teacher_probs']@b['teacher_probs'].T)


def context(features,labels,config):
    geometry,band,mix,scales=config;sim=similarity(features,features,geometry);d=np.maximum(2-2*sim,0);y=np.asarray(labels,np.int64)
    if band=='nearest':tau=float(np.median(np.min(np.where(y[:,None]!=y[None,:],d,np.inf),axis=1)))
    else:tau=float(np.median(d[y[:,None]!=y[None,:]]))
    if not np.isfinite(tau) or tau<=0:raise FloatingPointError('No positive teacher-kernel support bandwidth')
    return dict(features=features,tau=tau,parts=base.decompose(base.make_kernel(sim,tau,mix,scales),y))


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
        x=normalize(features)
        if x['teacher_probs'].shape[1]!=self.features['teacher_probs'].shape[1]:raise ValueError('Old-head dimensions differ')
        geometry,_,mix,scales=self.config
        return base.make_kernel(similarity(x,self.features,geometry),self.tau,mix,scales)@self.alpha+self.intercept


def fit(features,labels,ids,*,old_count=6,fixed=None,conditional_loo=True):
    if fixed is None or fixed.get('teacher_weight')!=WEIGHT:raise ValueError('Frozen teacher weight .25 required')
    if fixed.get('ridge_reference_classes')!=6 or tuple(fixed.get('config',())) not in sig.CONFIGS:raise ValueError('Frozen source rule required')
    reference=float(fixed['ridge'])
    if not np.isfinite(reference) or reference<=0:raise ValueError('Positive source ridge required')
    x=normalize(features);y=np.asarray(labels,np.int64);ids=tuple(map(str,ids))
    if y.ndim!=1 or not len(y) or (y<0).any() or len(ids)!=len(y) or len(set(ids))!=len(y):raise ValueError('Invalid physical support')
    count=int(y.max())+1;counts=np.bincount(y,minlength=count)
    if count<old_count or (counts!=counts[0]).any() or counts[0]<1 or len(x['z_id'])!=len(y) or x['teacher_probs'].shape[1]!=old_count:raise ValueError('Positive physical K and matching old-head columns required')
    k=int(counts[0]);records=[];audit=[]
    if k==1 or not conditional_loo:
        config=tuple(fixed['config']);ridge=reference*count/6 if k==1 else reference
    else:
        split=folds(y,ids);best=None;target=np.eye(count)[y];audit=[dict(train=len(train),held=len(held)) for train,held in split]
        for config in sig.CONFIGS:
            geometry,_,mix,scales=config;predictions={lam:np.zeros((len(y),count)) for lam in base.LAMBDAS}
            for train,held in split:
                tx={key:value[train] for key,value in x.items()};hx={key:value[held] for key,value in x.items()};ctx=context(tx,y[train],config)
                cross=base.make_kernel(similarity(hx,tx,geometry),ctx['tau'],mix,scales)
                for lam in base.LAMBDAS:
                    alpha,intercept,_=base.coefficients(ctx['parts'],lam);predictions[lam][held]=cross@alpha+intercept
            for lam,scores in predictions.items():
                objective=base.target_and_score(y,scores.argmax(1),old_count);loss=float(np.mean((scores-target)**2));rank=(objective,-loss,lam)
                records.append(dict(config=list(config),ridge=lam,physical_OOF_H=objective,OOF_loss=loss))
                if best is None or rank>best[0]:best=(rank,config,lam)
        _,config,ridge=best
    ctx=context(x,y,config);alpha,intercept,_=base.coefficients(ctx['parts'],ridge);alpha.flags.writeable=False;intercept.flags.writeable=False
    return State(x,alpha,intercept,config,ctx['tau'],float(ridge),dict(rule='source_fixed' if not audit else 'three_physical_fold_OOF',
        teacher_weight=WEIGHT,old_head_columns=old_count,physical_OOF=bool(audit),folds=audit,physical_K=k,physical_count=len(y),
        candidate_count=len(records) if audit else 1,records=records,query_fit=False,bandwidth_scope='registered',bandwidth_physical_count=len(y),
        frozen_default_head_updates=0,extra_state_bytes=x['teacher_probs'].nbytes,K1_current_class_scaling=bool(k==1)))
