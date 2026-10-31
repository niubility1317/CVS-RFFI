"""Old/new H calibration using only physically held-out current support."""
from dataclasses import dataclass
import numpy as np
from . import nested_fisher as nested,fisher_signature as fisher,support_calibrated as base
from .group_calibrated import choose_offset

KAPPAS=(0.,5.,20.,'none')


def row_standardize(scores):
    scores=np.asarray(scores,dtype=np.float64);centered=scores-scores.mean(1,keepdims=True)
    return centered/np.sqrt(np.mean(centered**2,axis=1,keepdims=True)+np.finfo(np.float64).eps)


@dataclass(frozen=True)
class State:
    base_state:object
    beta:float
    old_count:int
    selection:dict

    def __getattr__(self,name):return getattr(self.base_state,name)

    def score(self,features):
        values=self.base_state.score(features)
        if self.beta==0.:return values
        values=row_standardize(values);values[:,:self.old_count]+=self.beta
        return values


def fit_grid(features,labels,ids,*,old_count=6,fixed=None,conditional_loo=True):
    if fixed is None or fixed.get('outer_folds')!=3 or tuple(fixed.get('outer_strengths',()))!=fisher.STRENGTHS:
        raise ValueError('Frozen physical support-fold rule required')
    y=np.asarray(labels,dtype=np.int64);count=int(y.max())+1;k=len(y)//count
    if k==1 or count==old_count:
        state=nested.fit(features,y,ids,old_count=old_count,fixed=fixed,conditional_loo=conditional_loo)
        return {kap:State(state,0.,old_count,dict(state.selection,H_calibration=dict(kappa=kap,beta=0.,rule='K1_or_B_exact_R14',query_fit=False))) for kap in KAPPAS}
    split=nested.folds(y,ids);rules={kap:[] for kap in KAPPAS}
    for strength in fisher.STRENGTHS:
        oof=np.empty((len(y),count),np.float64)
        for train,held in split:
            state=fisher.fit({name:value[train] for name,value in features.items()},y[train],[ids[i] for i in train],old_count=old_count,fixed=dict(fixed,fisher_strength=strength))
            oof[held]=state.score({name:value[held] for name,value in features.items()})
        standardized=row_standardize(oof);raw_beta,audit=choose_offset(standardized,y,old_count)
        for kappa in KAPPAS:
            beta=0. if kappa=='none' else float(raw_beta*k/(k+kappa))
            scores=oof if beta==0 else standardized.copy()
            if beta!=0:scores[:,:old_count]+=beta
            h=base.target_and_score(y,scores.argmax(1),old_count)
            rules[kappa].append(dict(strength=strength,beta=beta,raw_beta=float(raw_beta),H=h,thresholds=audit['thresholds']))
    final={};cache={}
    for kappa,rows in rules.items():
        chosen=max(rows,key=lambda r:(r['H'],-r['strength']));strength=chosen['strength']
        if strength not in cache:cache[strength]=fisher.fit(features,y,ids,old_count=old_count,fixed=dict(fixed,fisher_strength=strength),conditional_loo=conditional_loo)
        state=cache[strength]
        selection=dict(state.selection,H_calibration=dict(kappa=kappa,beta=chosen['beta'],raw_beta=chosen['raw_beta'],selected_strength=strength,
            outer_support_H=chosen['H'],strengths=rows,rule='physical_support_OOF_H_threshold_then_K_shrink',
            physical_held_excluded_from_feature_weights_kernel_and_coefficients=True,query_fit=False,query_role_read=False))
        final[kappa]=State(state,chosen['beta'],old_count,selection)
    return final


def fit(features,labels,ids,*,old_count=6,fixed=None,conditional_loo=True):
    if fixed is None or fixed.get('H_bias_kappa') not in KAPPAS:raise ValueError('Source-frozen H calibration shrinkage required')
    return fit_grid(features,labels,ids,old_count=old_count,fixed=fixed,conditional_loo=conditional_loo)[fixed['H_bias_kappa']]
