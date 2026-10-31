"""Current physical SUP chooses shrinkage of a frozen source residual encoder."""
from dataclasses import dataclass
import numpy as np
from . import orbit_ensemble as core

STRENGTHS=(0.,.25,.5,.75,1.)
RF=tuple(core.RF)+tuple(core.PREFIX+k for k in core.RF)
BASE='BASE_'


def blend(features,strength):
    if strength not in STRENGTHS:raise ValueError('Wrong frozen residual strength')
    if set(features)!=set(RF)|{BASE+k for k in RF}|{'teacher_probs'}:raise ValueError('Exact base/full residual feature schema required')
    if any(features[k].shape!=features[BASE+k].shape for k in RF):raise ValueError('Base/full physical alignment differs')
    if strength==1:return {k:features[k] for k in (*RF,'teacher_probs')}
    if strength==0:return dict({k:features[BASE+k] for k in RF},teacher_probs=features['teacher_probs'])
    # core.normalize performs the branch-wise unit normalization for both
    # fitting and query evaluation. Equal endpoints stay numerically equal.
    return dict({k:features[BASE+k]+strength*(features[k]-features[BASE+k]) for k in RF},teacher_probs=features['teacher_probs'])


@dataclass(frozen=True)
class State:
    model:object
    strength:float
    selection:dict

    @property
    def alpha(self):return self.model.alpha
    @property
    def config(self):return self.model.config
    @property
    def tau(self):return self.model.tau
    @property
    def ridge(self):return self.model.ridge

    def evaluate(self,features):return self.model.evaluate(blend(features,self.strength))
    def score(self,features):return self.evaluate(features)[0]


def fit(features,labels,ids,*,old_count=6,fixed=None,conditional_loo=True):
    if fixed is None or tuple(fixed.get('support_residual_strengths',()))!=STRENGTHS or fixed.get('K1_residual_strength')!=1.:raise ValueError('Frozen support residual policy required')
    y=np.asarray(labels,np.int64)
    if y.ndim!=1 or not len(y) or (y<0).any():raise ValueError('Invalid physical support labels')
    count=np.bincount(y)
    if (count<=0).any() or (count!=count[0]).any():raise ValueError('Equal positive physical K required')
    k=int(count[0]);records=[];choices=[];best=None
    if k>1 and not conditional_loo:raise ValueError('Multi-shot residual selection requires physical OOF')
    for strength in ((1.,) if k==1 else STRENGTHS):
        model=core.fit(blend(features,strength),y,ids,old_count=old_count,fixed=fixed,conditional_loo=conditional_loo)
        if k==1:selected=dict(config=list(model.config),ridge=model.ridge);rank=(0.,0.,model.ridge,-strength)
        else:
            selected=next(r for r in model.selection['records'] if r['config']==list(model.config) and r['ridge']==model.ridge)
            rank=(selected['physical_OOF_H'],-selected['OOF_loss'],model.ridge,-strength)
        choices.append(dict(strength=strength,**selected))
        records.extend(dict(strength=strength,**r) for r in model.selection['records'])
        if best is None or rank>best[0]:best=(rank,strength,model)
    _,strength,model=best
    selection=dict(model.selection,records=records,candidate_count=1 if k==1 else len(records),
        adaptive_residual=dict(strength=strength,strengths=list((1.,) if k==1 else STRENGTHS),choices=choices,
            rule='source_fixed_full' if k==1 else 'joint_physical_OOF',physical_K=k,physical_count=len(y),
            objective='old/new_H_then_lower_squared_loss_then_larger_ridge_then_weaker_residual',
            no_independent_generalization_estimate=True,registered_SUP_only=True,query_fit=False,
            extra_encoder_parameters=0,unselected_models_retained=0))
    return State(model,float(strength),selection)
