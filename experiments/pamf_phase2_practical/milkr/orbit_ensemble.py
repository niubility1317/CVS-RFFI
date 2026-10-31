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


def context(features,labels,config,confidence_rule=RULE,density_rule=None,rf_gate_rule=None,old_count=None,rf_class_rule=None):
    geometry,band,mix,scales=config;y=np.asarray(labels,np.int64);similarity=similarities(features,features,geometry);taus=[]
    for sim in similarity:
        distance=np.maximum(2-2*sim,0)
        if band=='nearest':tau=float(np.median(np.min(np.where(y[:,None]!=y[None,:],distance,np.inf),axis=1)))
        else:tau=float(np.median(distance[y[:,None]!=y[None,:]]))
        if not np.isfinite(tau) or tau<=0:raise FloatingPointError('No positive ensemble SUP scale')
        taus.append(tau)
    if confidence_rule not in (RULE,CLASS_RULE):raise ValueError('Wrong ensemble prior calibration')
    K=make_kernel(similarity,taus,mix,scales);density=None
    if density_rule is not None:
        from .kernel_density import Density,RULE as DENSITY_RULE
        if density_rule!=DENSITY_RULE:raise ValueError('Wrong support kernel density rule')
        affinity=make_kernel(similarity,taus,0.,scales);density=Density.fit(affinity);K=density.cross(K,affinity)
    eig,U,Y,_,_,mean=base.decompose(K,y)
    gate=(fit_gate if confidence_rule==RULE else fit_class_gate)(features['teacher_probs'],y)
    rf_class=None;calibrated=None
    if rf_class_rule is not None:
        from .support_rf_class_prior import Calibrator,RULE as RF_CLASS_RULE
        if rf_class_rule!=RF_CLASS_RULE or old_count!=features['teacher_probs'].shape[1]:raise ValueError('Wrong RF class prior rule')
        rf_class=Calibrator.fit(y,old_count);calibrated=rf_class.probabilities(features['teacher_probs'],make_kernel(similarity,taus,0.,scales))
    bias=prior.prior(features['teacher_probs'],Y.shape[1],gate,calibrated);rf_gate=None
    if rf_gate_rule is not None:
        from .support_kernel_gate import fit as fit_rf_gate,RULE as RF_RULE
        if rf_gate_rule!=RF_RULE or old_count!=features['teacher_probs'].shape[1]:raise ValueError('Wrong RF prior evidence rule')
        rf_gate=fit_rf_gate(K,y,old_count);bias*=rf_gate.factors(K)[:,None]
    target=Y-bias;average=target.mean(0)
    return dict(tau=tuple(taus),parts=(eig,U,target,average,target-average,mean),gate=gate,density=density,rf_gate=rf_gate,rf_class=rf_class)


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
    kernel_density:object=None
    rf_gate:object=None
    rf_class:object=None

    def evaluate(self,features):
        x=normalize(features)
        if x['teacher_probs'].shape[1]!=self.features['teacher_probs'].shape[1]:raise ValueError('Old-head columns differ')
        geometry,_,mix,scales=self.config;sim=similarities(x,self.features,geometry)
        kernel=make_kernel(sim,self.tau,mix,scales)
        if self.kernel_density is not None:kernel=self.kernel_density.cross(kernel,make_kernel(sim,self.tau,0.,scales))
        calibrated=None if self.rf_class is None else self.rf_class.probabilities(x['teacher_probs'],make_kernel(sim,self.tau,0.,scales))
        bias=prior.prior(x['teacher_probs'],self.alpha.shape[1],self.prior_gate,calibrated)
        if self.rf_gate is not None:bias*=self.rf_gate.factors(kernel)[:,None]
        score=bias+kernel@self.alpha+self.intercept
        distance=WEIGHT*np.maximum(2-2*sim[0],0)+(1-WEIGHT)*np.maximum(2-2*sim[1],0)
        return score,distance.min(1)

    def score(self,features):return self.evaluate(features)[0]


def fit(features,labels,ids,*,old_count=6,fixed=None,conditional_loo=True):
    if fixed is None or fixed.get('orbit_ensemble_weight')!=WEIGHT or fixed.get('confidence_rule') not in (RULE,CLASS_RULE):raise ValueError('Frozen equal orbit ensemble/ridge gate required')
    if fixed.get('prior_weight')!=prior.PRIOR_WEIGHT or fixed.get('fusion_policy')!=prior.FUSION_POLICY or fixed.get('ridge_reference_classes')!=6 or tuple(fixed.get('config',())) not in sig.CONFIGS:raise ValueError('Wrong frozen ensemble source rule')
    if fixed.get('support_center_strength') is not None:raise ValueError('Ensemble has no support center')
    confidence_rule=fixed['confidence_rule']
    density_rule=fixed.get('kernel_density_rule')
    rf_gate_rule=fixed.get('rf_prior_gate_rule')
    rf_class_rule=fixed.get('rf_class_prior_rule')
    if rf_class_rule is not None:
        from .support_rf_class_prior import RULE as RF_CLASS_RULE
        if rf_class_rule!=RF_CLASS_RULE or confidence_rule!=RULE:raise ValueError('Wrong frozen RF class prior rule')
    if rf_gate_rule is not None:
        from .support_kernel_gate import RULE as RF_RULE
        if rf_gate_rule!=RF_RULE:raise ValueError('Wrong frozen RF prior evidence rule')
    if density_rule is not None:
        from .kernel_density import RULE as DENSITY_RULE
        if density_rule!=DENSITY_RULE:raise ValueError('Wrong frozen kernel density rule')
    def build(features,labels,config):
        options={}
        if confidence_rule!=RULE:options['confidence_rule']=confidence_rule
        if density_rule is not None:options['density_rule']=density_rule
        if rf_gate_rule is not None:options.update(rf_gate_rule=rf_gate_rule,old_count=old_count)
        if rf_class_rule is not None:options.update(rf_class_rule=rf_class_rule,old_count=old_count)
        return context(features,labels,config,**options)
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
                sim=similarities(hx,tx,geometry);cross=make_kernel(sim,ctx['tau'],mix,scales)
                if ctx['density'] is not None:cross=ctx['density'].cross(cross,make_kernel(sim,ctx['tau'],0.,scales))
                calibrated=None if ctx['rf_class'] is None else ctx['rf_class'].probabilities(hx['teacher_probs'],make_kernel(sim,ctx['tau'],0.,scales))
                bias=prior.prior(hx['teacher_probs'],count,ctx['gate'],calibrated)
                if ctx['rf_gate'] is not None:bias*=ctx['rf_gate'].factors(cross)[:,None]
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
        extra_state_bytes=x['teacher_probs'].nbytes+ctx['gate'].state_bytes+extra+(ctx['density'].support_degree.nbytes+8 if ctx['density'] is not None else 0),prior_confidence_gate=ctx['gate'].audit(),support_center=None,
        orbit_ensemble=dict(orbits=[4,8],weights=[WEIGHT,1-WEIGHT],kernel='arithmetic_mean_after_separate_bandwidths',extra_feature_bytes=extra,query_fit=False))
    if ctx['density'] is not None:selection['kernel_density']=ctx['density'].audit()
    if ctx['rf_gate'] is not None:
        selection['rf_prior_gate']=ctx['rf_gate'].audit();selection['extra_state_bytes']+=ctx['rf_gate'].state_bytes
    if ctx['rf_class'] is not None:
        selection['rf_class_prior']=ctx['rf_class'].audit();selection['extra_state_bytes']+=ctx['rf_class'].labels.nbytes+ctx['rf_class'].old_counts.nbytes
    return State(x,alpha,intercept,config,ctx['tau'],float(ridge),selection,ctx['gate'],ctx['density'],ctx['rf_gate'],ctx['rf_class'])
