import numpy as np
from . import design as d
def descriptor(x):
    x=np.asarray(x)
    if x.ndim!=3 or x.shape[1:]!=(2,256) or not np.isfinite(x).all():raise ValueError('Finite packets x2x256 required')
    ac=x[:,0].astype(np.float64)+1j*x[:,1].astype(np.float64)
    ac-=ac.mean(1,keepdims=True);power=np.square(np.abs(ac)).mean(1)
    if np.any(power<=1e-18):raise ValueError('Positive AC power required')
    z=np.abs(np.square(ac).mean(1))/power
    if np.any(z>1+1e-12):raise ValueError('Invalid ellipticity')
    return z
def cells(z,tx,rx,day):
    z=np.asarray(z,dtype=np.float64);tx=np.asarray(tx);rx=np.asarray(rx);day=np.asarray(day)
    if z.shape!=(5040,) or any(v.shape!=(5040,) for v in (tx,rx,day)) or not np.isfinite(z).all() or np.any(z<0) or np.any(z>1+1e-12):raise ValueError('L5040 ellipticity and metadata required')
    if set(tx.tolist())!=set(d.TX) or set(rx.tolist())!=set(d.FIT) or set(day.tolist())!=set(d.DAYS):raise ValueError('Only fixed fit L source roles')
    result=np.empty((6,4,3))
    for t in d.TX:
        for ri,r in enumerate(d.FIT):
            for di,date in enumerate(d.DAYS):
                mask=(tx==t)&(rx==r)&(day==date)
                if int(mask.sum())!=70:raise ValueError('Exactly70 legal L per cell')
                result[t,ri,di]=z[mask].mean()
    return result
def effect(a):
    a=np.asarray(a,dtype=np.float64)
    if a.shape!=(6,4,3) or not np.isfinite(a).all():raise ValueError('Complete6TX x4RX x3day cells')
    return a-a.mean(axis=1,keepdims=True)
def prediction(e,t,di):
    if t not in d.TX or di not in range(3):raise ValueError('Registered TX/day holdout')
    return e[np.arange(6)!=t][:,:,np.arange(3)!=di].mean(axis=(0,2))
def analyze(a):
    e=effect(a);folds=[]
    for t in d.TX:
        for di,date in enumerate(d.DAYS):
            observed=e[t,:,di];pred=prediction(e,t,di)
            base=float(np.square(observed).sum());risk=float(np.square(observed-pred).sum())
            folds.append(dict(held_TX=t,held_day=date,training_TX=5,training_days=2,held_RX_count=4,zero_effect_risk=base,predicted_effect_risk=risk,risk_reduction=1-risk/base if base>1e-18 else None))
    base=sum(v['zero_effect_risk'] for v in folds);risk=sum(v['predicted_effect_risk'] for v in folds)
    reduction=1-risk/base if base>1e-18 else None
    profile=e.mean(axis=(0,2));pairs=[abs(float(a-b)) for i,a in enumerate(profile) for b in profile[i+1:]]
    p95=float(np.quantile(pairs,.95));positive=sum(v['risk_reduction'] is not None and v['risk_reduction']>0 for v in folds)
    checks=dict(nonzero_effect=base>1e-18,risk=reduction is not None and reduction>=.1,positive_folds=positive>=12,ellipticity_scale=p95>=.01)
    return dict(status='PASS' if all(checks.values()) else 'FAIL',checks=checks,aggregate_risk_reduction=reduction,positive_folds=positive,total_folds=18,
        zero_effect_risk=base,predicted_effect_risk=risk,RX_pair_ellipticity_p95=p95,RX_pair_ellipticity_p95_min=.01,folds=folds,
        interpretation='RX-associated full256 AC ellipticity magnitude repeatability only; no physical causal attribution or classifier efficacy',
        source_only=True,target_access=False,automatic_training=False,goal_completed=False),profile
