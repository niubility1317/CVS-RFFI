"""Equal arithmetic mean of source C4/C8 kernels, with current-SUP-only fitting."""
from dataclasses import dataclass
import numpy as np
from . import prior_fusion as prior,signature_kernel as sig,support_calibrated as base
from .support_ridge_prior import fit as fit_gate,RULE
from .support_class_prior import fit as fit_class_gate,RULE as CLASS_RULE
from .nested_fisher import folds

WEIGHT=.5
RF=tuple(base.NAMES)+('sig_views',)
PREFIX='C4_'


def normalize(features):
    if set(features)!=set(RF)|{PREFIX+k for k in RF}|{'teacher_probs'}:raise ValueError('Exact C4/C8 ensemble features required')
    out=prior.normalize({k:features[k] for k in (*RF,'teacher_probs')})
    c4=prior.normalize(dict({k:features[PREFIX+k] for k in RF},teacher_probs=features['teacher_probs']))
    out.update({PREFIX+k:c4[k] for k in RF});return out


def view(features,orbit):
    return {key:features[(PREFIX if orbit==4 else '')+key] for key in RF}


def similarities(left,right,geometry):
    return tuple(sig.similarity(view(left,orbit),view(right,orbit),geometry) for orbit in (4,8))


def make_kernel(similarity,tau,mix,scales):
    return WEIGHT*base.make_kernel(similarity[0],tau[0],mix,scales)+(1-WEIGHT)*base.make_kernel(similarity[1],tau[1],mix,scales)


def context(features,labels,config,confidence_rule=RULE):
    geometry,band,mix,scales=config;y=np.asarray(labels,np.int64);similarity=similarities(features,features,geometry);taus=[]
    for sim in similarity:
        distance=np.maximum(2-2*sim,0)
        if band=='nearest':tau=float(np.median(np.min(np.where(y[:,None]!=y[None,:],distance,np.inf),axis=1)))
        else:tau=float(np.median(distance[y[:,None]!=y[None,:]]))
        if not np.isfinite(tau) or tau<=0:raise FloatingPointError('No positive ensemble SUP scale')
        taus.append(tau)
    if confidence_rule not in (RULE,CLASS_RULE):raise ValueError('Wrong ensemble prior calibration')
    K=make_kernel(similarity,taus,mix,scales);eig,U,Y,_,_,mean=base.decompose(K,y)
    gate=(fit_gate if confidence_rule==RULE else fit_class_gate)(features['teacher_probs'],y)
    target=Y-prior.prior(features['teacher_probs'],Y.shape[1],gate);average=target.mean(0)
    return dict(tau=tuple(taus),parts=(eig,U,target,average,target-average,mean),gate=gate)


@dataclass(frozen=True)
class State:
    features:dict
    alpha:np.ndarray
    intercept:np.ndarray
    config:tuple
    tau:tuple
    ridge:float
    selection:dict
    prior_gate:object

    def evaluate(self,features):
        x=normalize(features)
        if x['teacher_probs'].shape[1]!=self.features['teacher_probs'].shape[1]:raise ValueError('Old-head columns differ')
        geometry,_,mix,scales=self.config;sim=similarities(x,self.features,geometry)
        score=prior.prior(x['teacher_probs'],self.alpha.shape[1],self.prior_gate)+make_kernel(sim,self.tau,mix,scales)@self.alpha+self.intercept
        distance=WEIGHT*np.maximum(2-2*sim[0],0)+(1-WEIGHT)*np.maximum(2-2*sim[1],0)
        return score,distance.min(1)

    def score(self,features):return self.evaluate(features)[0]


def fit(features,labels,ids,*,old_count=6,fixed=None,conditional_loo=True):
    if fixed is None or fixed.get('orbit_ensemble_weight')!=WEIGHT or fixed.get('confidence_rule') not in (RULE,CLASS_RULE):raise ValueError('Frozen equal orbit ensemble/ridge gate required')
    if fixed.get('prior_weight')!=prior.PRIOR_WEIGHT or fixed.get('fusion_policy')!=prior.FUSION_POLICY or fixed.get('ridge_reference_classes')!=6 or tuple(fixed.get('config',())) not in sig.CONFIGS:raise ValueError('Wrong frozen ensemble source rule')
    if fixed.get('support_center_strength') is not None:raise ValueError('Ensemble has no support center')
    confidence_rule=fixed['confidence_rule']
    def build(features,labels,config):
        return context(features,labels,config) if confidence_rule==RULE else context(features,labels,config,confidence_rule=confidence_rule)
    reference=float(fixed['ridge'])
    if not np.isfinite(reference) or reference<=0:raise ValueError('Positive source ridge required')
    x=normalize(features);y=np.asarray(labels,np.int64);ids=tuple(map(str,ids))
    if y.ndim!=1 or not len(y) or (y<0).any() or len(ids)!=len(y) or len(set(ids))!=len(y):raise ValueError('Invalid physical support')
    count=int(y.max())+1;counts=np.bincount(y,minlength=count)
    if count<old_count or (counts!=counts[0]).any() or counts[0]<1 or len(x['z_id'])!=len(y) or x['teacher_probs'].shape[1]!=old_count:raise ValueError('Physical K and mapped old-head columns required')
    k=int(counts[0]);records=[];audit=[]
    if k==1 or not conditional_loo:config=tuple(fixed['config']);ridge=reference*count/6 if k==1 else reference
    else:
        split=folds(y,ids);target=np.eye(count)[y];best=None;audit=[dict(train=len(train),held=len(held)) for train,held in split]
        for config in sig.CONFIGS:
            geometry,_,mix,scales=config;predictions={lam:np.zeros((len(y),count)) for lam in base.LAMBDAS}
            for train,held in split:
                tx={key:value[train] for key,value in x.items()};hx={key:value[held] for key,value in x.items()};ctx=build(tx,y[train],config)
                cross=make_kernel(similarities(hx,tx,geometry),ctx['tau'],mix,scales);bias=prior.prior(hx['teacher_probs'],count,ctx['gate'])
                for lam in base.LAMBDAS:
                    alpha,intercept,_=base.coefficients(ctx['parts'],lam);predictions[lam][held]=bias+cross@alpha+intercept
            for lam,scores in predictions.items():
                objective=base.target_and_score(y,scores.argmax(1),old_count);loss=float(np.mean((scores-target)**2));rank=(objective,-loss,lam)
                records.append(dict(config=list(config),ridge=lam,physical_OOF_H=objective,OOF_loss=loss))
                if best is None or rank>best[0]:best=(rank,config,lam)
        _,config,ridge=best
    ctx=build(x,y,config);alpha,intercept,_=base.coefficients(ctx['parts'],ridge);alpha.flags.writeable=False;intercept.flags.writeable=False
    extra=sum(x[PREFIX+key].nbytes for key in RF)
    selection=dict(rule='source_fixed' if not audit else 'three_physical_fold_OOF',prior_weight=prior.PRIOR_WEIGHT,fusion_policy=prior.FUSION_POLICY,old_head_columns=old_count,
        physical_OOF=bool(audit),folds=audit,physical_K=k,physical_count=len(y),candidate_count=len(records) if audit else 1,records=records,query_fit=False,
        bandwidth_scope='registered',bandwidth_physical_count=len(y),frozen_default_head_updates=0,K1_current_class_scaling=bool(k==1),
        extra_state_bytes=x['teacher_probs'].nbytes+ctx['gate'].state_bytes+extra,prior_confidence_gate=ctx['gate'].audit(),support_center=None,
        orbit_ensemble=dict(orbits=[4,8],weights=[WEIGHT,1-WEIGHT],kernel='arithmetic_mean_after_separate_bandwidths',extra_feature_bytes=extra,query_fit=False))
    return State(x,alpha,intercept,config,ctx['tau'],float(ridge),selection,ctx['gate'])
