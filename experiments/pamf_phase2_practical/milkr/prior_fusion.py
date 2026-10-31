"""Frozen-head function prior and same-record gated raw/DURA score fusion."""
from dataclasses import dataclass
import numpy as np
from . import signature_kernel as sig,support_calibrated as base
from .nested_fisher import folds

PRIOR_WEIGHT=.5
FUSION_POLICY='rf_distance_margin_half_v1'


def head_columns(source_classes,registered_old):
    if len(set(source_classes))!=len(source_classes) or set(source_classes)!=set(registered_old) or len(source_classes)!=len(registered_old):raise ValueError('Old/source class tables differ')
    return [source_classes.index(name) for name in registered_old]


def normalize(features):
    if set(features)!=set(base.NAMES)|{'sig_views','teacher_probs'}:raise ValueError('Exact prior features required')
    out=sig.normalize({key:value for key,value in features.items() if key!='teacher_probs'});p=np.array(features['teacher_probs'],np.float64,copy=True)
    if p.ndim!=2 or len(p)!=len(out['z_id']) or not 2<=p.shape[1]<=6 or not np.isfinite(p).all() or (p<0).any() or not np.allclose(p.sum(1),1):raise ValueError('Invalid default-head probabilities')
    p.flags.writeable=False;out['teacher_probs']=p;return out


def prior(probability,count,gate=None):
    p=np.asarray(probability,np.float64);value=np.zeros((len(p),count),np.float64)
    weight=1. if gate is None else gate.values(p)[:,None]
    value[:,:p.shape[1]]=PRIOR_WEIGHT*weight*(p-1/p.shape[1]);return value


def context(features,labels,config,confidence_rule=None):
    geometry,band,mix,scales=config;y=np.asarray(labels,np.int64);sim=sig.similarity(features,features,geometry);d=np.maximum(2-2*sim,0)
    if band=='nearest':tau=float(np.median(np.min(np.where(y[:,None]!=y[None,:],d,np.inf),axis=1)))
    else:tau=float(np.median(d[y[:,None]!=y[None,:]]))
    if not np.isfinite(tau) or tau<=0:raise FloatingPointError('No positive prior-kernel support scale')
    K=base.make_kernel(sim,tau,mix,scales);eig,U,Y,_,_,mean=base.decompose(K,y)
    gate=None
    if confidence_rule is not None:
        from . import support_prior_gate,support_ridge_prior
        if confidence_rule==support_prior_gate.RULE:calibrator=support_prior_gate
        elif confidence_rule==support_ridge_prior.RULE:calibrator=support_ridge_prior
        else:raise ValueError('Wrong support-confidence rule')
        gate=calibrator.fit(features['teacher_probs'],y)
    target=Y-prior(features['teacher_probs'],Y.shape[1],gate);average=target.mean(0)
    return dict(tau=tau,parts=(eig,U,target,average,target-average,mean),gate=gate)


def fuse(raw,restored):
    score0,d0=raw;scoreG,dG=restored;score0=np.asarray(score0,np.float64);scoreG=np.asarray(scoreG,np.float64);d0=np.asarray(d0,np.float64);dG=np.asarray(dG,np.float64)
    if score0.ndim!=2 or score0.shape!=scoreG.shape or score0.shape[1]<2 or d0.shape!=dG.shape or d0.shape!=(len(score0),) or not all(np.isfinite(v).all() for v in (score0,scoreG,d0,dG)):raise ValueError('Invalid same-record fusion inputs')
    top0=np.partition(score0,-2,axis=1)[:,-2:];topG=np.partition(scoreG,-2,axis=1)[:,-2:]
    margin0=top0[:,1]-top0[:,0];marginG=topG[:,1]-topG[:,0]
    weight=.5*((dG<d0)&(marginG>margin0)).astype(np.float64)
    return score0+weight[:,None]*(scoreG-score0),weight


@dataclass(frozen=True)
class State:
    features:dict
    alpha:np.ndarray
    intercept:np.ndarray
    config:tuple
    tau:float
    ridge:float
    selection:dict
    prior_gate:object=None

    def evaluate(self,features):
        x=normalize(features)
        if x['teacher_probs'].shape[1]!=self.features['teacher_probs'].shape[1]:raise ValueError('Old-head dimensions differ')
        geometry,_,mix,scales=self.config;sim=sig.similarity(x,self.features,geometry)
        scores=prior(x['teacher_probs'],self.alpha.shape[1],self.prior_gate)+base.make_kernel(sim,self.tau,mix,scales)@self.alpha+self.intercept
        return scores,np.maximum(2-2*sim,0).min(1)

    def score(self,features):return self.evaluate(features)[0]


def fit(features,labels,ids,*,old_count=6,fixed=None,conditional_loo=True):
    if fixed is None or fixed.get('prior_weight')!=PRIOR_WEIGHT or fixed.get('fusion_policy')!=FUSION_POLICY:raise ValueError('Frozen function-prior/fusion rule required')
    if fixed.get('ridge_reference_classes')!=6 or tuple(fixed.get('config',())) not in sig.CONFIGS:raise ValueError('Frozen source rule required')
    confidence_rule=fixed.get('confidence_rule')
    if confidence_rule is not None:
        from .support_prior_gate import RULE
        from .support_ridge_prior import RULE as RIDGE_RULE
        if confidence_rule not in (RULE,RIDGE_RULE):raise ValueError('Wrong frozen confidence rule')
    def build(features,labels,config):
        return context(features,labels,config) if confidence_rule is None else context(features,labels,config,confidence_rule=confidence_rule)
    reference=float(fixed['ridge'])
    if not np.isfinite(reference) or reference<=0:raise ValueError('Positive source ridge required')
    x=normalize(features);y=np.asarray(labels,np.int64);ids=tuple(map(str,ids))
    if y.ndim!=1 or not len(y) or (y<0).any() or len(ids)!=len(y) or len(set(ids))!=len(y):raise ValueError('Invalid physical support')
    count=int(y.max())+1;counts=np.bincount(y,minlength=count)
    if count<old_count or (counts!=counts[0]).any() or counts[0]<1 or len(x['z_id'])!=len(y) or x['teacher_probs'].shape[1]!=old_count:raise ValueError('Positive physical K and mapped old-head columns required')
    k=int(counts[0]);records=[];audit=[]
    if k==1 or not conditional_loo:config=tuple(fixed['config']);ridge=reference*count/6 if k==1 else reference
    else:
        split=folds(y,ids);best=None;target=np.eye(count)[y];audit=[dict(train=len(train),held=len(held)) for train,held in split]
        for config in sig.CONFIGS:
            geometry,_,mix,scales=config;predictions={lam:np.zeros((len(y),count)) for lam in base.LAMBDAS}
            for train,held in split:
                tx={key:value[train] for key,value in x.items()};hx={key:value[held] for key,value in x.items()};ctx=build(tx,y[train],config)
                cross=base.make_kernel(sig.similarity(hx,tx,geometry),ctx['tau'],mix,scales);bias=prior(hx['teacher_probs'],count,ctx['gate'])
                for lam in base.LAMBDAS:
                    alpha,intercept,_=base.coefficients(ctx['parts'],lam);predictions[lam][held]=bias+cross@alpha+intercept
            for lam,scores in predictions.items():
                objective=base.target_and_score(y,scores.argmax(1),old_count);loss=float(np.mean((scores-target)**2));rank=(objective,-loss,lam)
                records.append(dict(config=list(config),ridge=lam,physical_OOF_H=objective,OOF_loss=loss))
                if best is None or rank>best[0]:best=(rank,config,lam)
        _,config,ridge=best
    ctx=build(x,y,config);alpha,intercept,_=base.coefficients(ctx['parts'],ridge);alpha.flags.writeable=False;intercept.flags.writeable=False
    return State(x,alpha,intercept,config,ctx['tau'],float(ridge),dict(rule='source_fixed' if not audit else 'three_physical_fold_OOF',
        prior_weight=PRIOR_WEIGHT,fusion_policy=FUSION_POLICY,old_head_columns=old_count,physical_OOF=bool(audit),folds=audit,physical_K=k,physical_count=len(y),
        candidate_count=len(records) if audit else 1,records=records,query_fit=False,bandwidth_scope='registered',bandwidth_physical_count=len(y),
        frozen_default_head_updates=0,extra_state_bytes=x['teacher_probs'].nbytes+(ctx['gate'].state_bytes if ctx['gate'] is not None else 0),K1_current_class_scaling=bool(k==1),
        prior_confidence_gate=ctx['gate'].audit() if ctx['gate'] is not None else None),ctx['gate'])
