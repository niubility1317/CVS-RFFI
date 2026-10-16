import numpy as np
from . import design as d

def cells(z,tx,rx,day):
    z=np.asarray(z,dtype=np.float64);tx=np.asarray(tx);rx=np.asarray(rx);day=np.asarray(day)
    if z.shape!=(5040,2) or any(v.shape!=(5040,) for v in (tx,rx,day)) or not np.isfinite(z).all():raise ValueError('Finite source L5040 DC descriptors required')
    if set(tx.tolist())!=set(d.TX) or set(rx.tolist())!=set(d.FIT) or set(day.tolist())!=set(d.DAYS):raise ValueError('Only fixed labeled fit RX/TX/day metadata permitted')
    result=np.empty((6,4,3,2),dtype=np.float64)
    for t in d.TX:
        for ri,r in enumerate(d.FIT):
            for di,day_value in enumerate(d.DAYS):
                mask=(tx==t)&(rx==r)&(day==day_value)
                if int(mask.sum())!=70:raise ValueError('Exactly70 legal L packets in each TX/RX/day cell required')
                result[t,ri,di]=z[mask].mean(0)
    return result

def effect(cell_means):
    a=np.asarray(cell_means,dtype=np.float64)
    if a.shape!=(6,4,3,2) or not np.isfinite(a).all():raise ValueError('Complete finite6TX/4RX/3day complex DC cells required')
    return a-a.mean(axis=1,keepdims=True)

def prediction(e,held_TX,held_day_index):
    if held_TX not in d.TX or held_day_index not in range(3):raise ValueError('One registered TX/day holdout required')
    support=e[np.arange(6)!=held_TX][:,:,np.arange(3)!=held_day_index,:]
    if support.shape!=(5,4,2,2):raise ValueError('Strict leaveTX AND leaveDay support required')
    return support.mean(axis=(0,2))

def analyze(cell_means,old_DC_max):
    e=effect(cell_means)
    if not np.isfinite(old_DC_max) or old_DC_max<=0:raise ValueError('Actual active old ordinary DC component bound required')
    folds=[]
    for t in d.TX:
        for di,day in enumerate(d.DAYS):
            observed=e[t,:,di];pred=prediction(e,t,di)
            baseline=float(np.square(observed).sum());risk=float(np.square(observed-pred).sum())
            folds.append(dict(held_TX=t,held_day=day,training_TX=5,training_days=2,held_RX_count=4,zero_effect_risk=baseline,predicted_effect_risk=risk,
                risk_reduction=1.-risk/baseline if baseline>1e-18 else None))
    baseline=sum(r['zero_effect_risk'] for r in folds);risk=sum(r['predicted_effect_risk'] for r in folds)
    reduction=1.-risk/baseline if baseline>1e-18 else None
    profile=e.mean(axis=(0,2));deltas=[float(np.max(np.abs(a-b))) for i,a in enumerate(profile) for b in profile[i+1:]]
    envelope=float(np.quantile(deltas,.95));positive=sum(r['risk_reduction'] is not None and r['risk_reduction']>0 for r in folds)
    checks=dict(nonzero_effect=baseline>1e-18,risk=reduction is not None and reduction>=.1,positive_folds=positive>=12,outside_old_ordinary=envelope>old_DC_max)
    return dict(status='PASS' if all(checks.values()) else 'FAIL',checks=checks,aggregate_risk_reduction=reduction,positive_folds=positive,total_folds=18,
        zero_effect_risk=baseline,predicted_effect_risk=risk,RX_pair_DC_component_p95=envelope,old_ordinary_DC_component_max=float(old_DC_max),
        folds=folds,interpretation='Source RX-associated mean contrast repeatability only; no classifier, causal hardware identification or Ag efficacy claim',
        source_only=True,target_access=False,automatic_training=False,goal_completed=False),profile
