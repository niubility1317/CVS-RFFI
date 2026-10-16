"""Fixed IQ descriptor and non-fitted RX-contrast mean prediction."""
import numpy as np
from . import design as d

def describe(x):
    x=np.asarray(x,dtype=np.float64)
    if x.ndim!=3 or x.shape[1:]!=(2,256) or not np.isfinite(x).all():raise ValueError('Finite IQ256 packets only')
    z=x[:,0]+1j*x[:,1];a=z[:,80:160].reshape(-1,4,20)
    power=np.mean(np.abs(a)**2,axis=(1,2))
    if np.any(power<=0):raise ValueError('Positive fixed preamble power required')
    before=a[:,:-1];after=a[:,1:];cross=np.sum(after*np.conj(before),axis=(1,2))
    denom=np.sqrt(np.sum(np.abs(before)**2,axis=(1,2))*np.sum(np.abs(after)**2,axis=(1,2)))
    coherence=np.divide(np.abs(cross),denom,out=np.zeros_like(power),where=denom>0)
    phase=np.divide(cross,np.abs(cross),out=np.ones_like(cross),where=np.abs(cross)>0)
    residual=np.stack((a[:,1]-phase[:,None]*a[:,0],a[:,3]-phase[:,None]*a[:,2]),axis=1)/np.sqrt(2.)
    fraction=np.mean(np.abs(residual)**2,axis=(1,2))/power
    left,right=residual[:,:,:-1],residual[:,:,1:]
    rden=np.sqrt(np.sum(np.abs(left)**2,axis=(1,2))*np.sum(np.abs(right)**2,axis=(1,2)))
    valid=(fraction>1e-12)&(rden>0)
    rho=np.divide(np.sum(right*np.conj(left),axis=(1,2)),rden,out=np.zeros_like(cross),where=valid)
    if np.any(np.abs(rho)>1+1e-12) or not np.isfinite(rho).all():raise ValueError('Cauchy-bounded finite correlation required')
    return np.stack((rho.real,rho.imag),axis=1),dict(coherence=coherence,residual_ratio=fraction,rho_valid=valid)

def cells(z,tx,rx,day):
    z=np.asarray(z,dtype=np.float64);tx=np.asarray(tx);rx=np.asarray(rx);day=np.asarray(day)
    if z.shape!=(5040,2) or any(v.shape!=(5040,) for v in (tx,rx,day)) or not np.isfinite(z).all():raise ValueError('Full finite source L5040 correlation descriptors')
    if set(tx.tolist())!=set(d.TX) or set(rx.tolist())!=set(d.FIT) or set(day.tolist())!=set(d.DAYS):raise ValueError('Only legal fixed fitL metadata')
    a=np.empty((6,4,3,2))
    for t in d.TX:
        for i,r in enumerate(d.FIT):
            for j,dd in enumerate(d.DAYS):
                mask=(tx==t)&(rx==r)&(day==dd)
                if mask.sum()!=70:raise ValueError('Exactly70 packets in each source cell')
                a[t,i,j]=z[mask].mean(0)
    return a

def prediction(e,t,di):
    if t not in d.TX or di not in range(3):raise ValueError('Fixed TX/day holdout')
    return e[np.arange(6)!=t][:,:,np.arange(3)!=di,:].mean(axis=(0,2))

def analyze(a,quality):
    a=np.asarray(a,dtype=np.float64)
    if a.shape!=(6,4,3,2) or not np.isfinite(a).all() or np.any(np.linalg.norm(a,axis=-1)>1+1e-12):raise ValueError('Complete bounded correlation means')
    if set(quality)!=set(('valid_packets','coherence_p10','RX_median_residual_ratio')) or not 0<=quality['valid_packets']<=5040 or len(quality['RX_median_residual_ratio'])!=4 or not all(np.isfinite(v) for v in [quality['coherence_p10'],*quality['RX_median_residual_ratio']]):raise ValueError('Complete finite source quality census')
    e=a-a.mean(axis=1,keepdims=True);folds=[]
    for t in d.TX:
        for di,day in enumerate(d.DAYS):
            obs=e[t,:,di];p=prediction(e,t,di);null=np.square(obs).sum(axis=-1);risk=np.square(obs-p).sum(axis=-1)
            folds.append(dict(held_TX=t,held_day=day,zero_risk=float(null.sum()),predicted_risk=float(risk.sum()),risk_reduction=float(1-risk.sum()/null.sum()) if null.sum()>1e-18 else None,
                RX_zero_risk=null.tolist(),RX_predicted_risk=risk.tolist(),wrong_RX_risk=[float(np.square(obs-np.roll(p,k,axis=0)).sum()) for k in (1,2,3)]))
    zero=sum(r['zero_risk'] for r in folds);risk=sum(r['predicted_risk'] for r in folds);gain=1-risk/zero if zero>1e-18 else None
    per_null=np.sum([r['RX_zero_risk'] for r in folds],axis=0);per_risk=np.sum([r['RX_predicted_risk'] for r in folds],axis=0)
    per_RX=[float(1-b/a) if a>1e-18 else None for a,b in zip(per_null,per_risk)]
    wrong=np.sum([r['wrong_RX_risk'] for r in folds],axis=0);wrong_gain=[float(1-v/zero) if zero>1e-18 else None for v in wrong]
    over_wrong=gain-max(wrong_gain) if gain is not None else None
    p1=e[:3].mean(axis=(0,2)).reshape(-1);p2=e[3:].mean(axis=(0,2)).reshape(-1);norm=np.linalg.norm(p1)*np.linalg.norm(p2);cos=float(p1@p2/norm) if norm>1e-18 else None
    profile=a.mean(axis=(0,2));pairs=[float(np.linalg.norm(x-y)) for i,x in enumerate(profile) for y in profile[i+1:]]
    distance=float(np.quantile(pairs,.95));nonwhite=float(np.median(np.linalg.norm(profile,axis=-1)));positive=sum(r['risk_reduction'] is not None and r['risk_reduction']>0 for r in folds)
    checks=dict(all_valid_packets=quality['valid_packets']==5040,repeat_coherence=quality['coherence_p10']>=.8,small_residual=all(v<=.1 for v in quality['RX_median_residual_ratio']),
        nonzero_RX_effect=zero>1e-18,aggregate=gain is not None and gain>=.1,positive_folds=positive>=14,all_RX=all(v is not None and v>=0 for v in per_RX),
        wrong_RX_control=over_wrong is not None and over_wrong>=.02,TX_half_agreement=cos is not None and cos>=.8,nonwhite=nonwhite>=.05,RX_difference=distance>=.05)
    return dict(status='PASS' if all(checks.values()) else 'FAIL',checks=checks,quality=quality,aggregate_risk_reduction=gain,positive_folds=positive,zero_effect_risk=zero,predicted_effect_risk=risk,
        RX_risk_reduction=per_RX,wrong_RX_risk_reduction=wrong_gain,correct_over_best_wrong=over_wrong,TX_half_profile_cosine=cos,median_RX_profile_absolute_rho=nonwhite,RX_pair_profile_distance_p95=distance,folds=folds,
        source_only=True,target_access=False,new_models=0,model_forwards=0,optimizer_updates=0,checkpoint_loads=0,head_fit_updates=0,automatic_training=False,goal_completed=False,
        interpretation='Source RX-associated correlation repeatability only. Residual may contain TX/propagation/processing effects, not proven pure RX noise or identity-safe augmentation'),profile
