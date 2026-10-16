"""Source DC statistics and cross-TX/day transport; no identity classifier."""
import numpy as np

RX=(3,4,6,8)
WINDOW=(80,160)


def observe(x):
    x=np.asarray(x,dtype=np.float64)
    if x.ndim!=3 or x.shape[1:]!=(2,256) or not np.isfinite(x).all():raise ValueError('Finite native IQ[N,2,256] required')
    w=x[:,:,80:160];dc=w.mean(2);cycles=w.reshape(len(x),2,4,20).mean(3)
    difference=cycles[:,:,::2].mean(2)-cycles[:,:,1::2].mean(2)
    power=(w*w).sum(1).mean(1)
    return dict(dc=dc,odd_even_DC_difference=difference,window_power=power,
                dc_power_fraction=(dc*dc).sum(1)/np.maximum(power,1e-30))


def _cosine(a,b):
    norm=np.linalg.norm(a)*np.linalg.norm(b)
    return float(np.dot(a.ravel(),b.ravel())/norm) if norm>1e-20 else 0.


def crossfit(dc,tx,rx,day):
    dc=np.asarray(dc,dtype=np.float64);tx=np.asarray(tx);rx=np.asarray(rx);day=np.asarray(day)
    n=len(dc)
    if dc.shape!=(5040,2) or any(v.shape!=(n,) for v in (tx,rx,day)) or not np.isfinite(dc).all():raise ValueError('Only complete L5040 fit-RX DC observations required')
    if set(tx)!=set(range(6)) or set(rx)!=set(RX) or set(day)!={1,2,3}:raise ValueError('Six visible L TX, exact four fit RX and three source days required')
    cells=np.zeros((4,3,6,2));spread=np.zeros((4,3,6))
    for i,r in enumerate(RX):
        for j,d in enumerate((1,2,3)):
            for t in range(6):
                values=dc[(rx==r)&(day==d)&(tx==t)]
                if len(values)!=70:raise ValueError('Every fit RX/day/TX cell must contain its original70 L packets')
                cells[i,j,t]=values.mean(0);spread[i,j,t]=np.square(values-values.mean(0)).sum(1).mean()
    folds=[];base_errors=[];model_errors=[];wrong_errors=[[],[],[]]
    for held_day in range(3):
        for held_tx in range(6):
            # Neither held TX nor held day contributes to RX or pooled estimates.
            training=cells[:,[d for d in range(3) if d!=held_day]][:,:,[t for t in range(6) if t!=held_tx]]
            rx_mean=training.mean((1,2));pooled=rx_mean.mean(0)
            actual=cells[:,held_day,held_tx]
            base=np.square(actual-pooled).sum(1);model=np.square(actual-rx_mean).sum(1)
            wrong=[np.square(actual-np.roll(rx_mean,s,axis=0)).sum(1) for s in (1,2,3)]
            gain=1-float(model.sum())/float(base.sum()) if float(base.mean())>1e-16 else None
            folds.append(dict(held_TX=held_tx,held_day=held_day+1,risk_gain=gain,baseline_risk=float(base.mean()),RX_risk=float(model.mean())))
            base_errors.append(base);model_errors.append(model)
            for i,v in enumerate(wrong):wrong_errors[i].append(v)
    b=np.array(base_errors);m=np.array(model_errors)
    baseline=float(b.mean());risk=float(m.mean());gain=1-risk/baseline if baseline>1e-16 else None
    wrong_risks=[float(np.mean(v)) for v in wrong_errors]
    wrong_gains=[1-v/baseline if baseline>1e-16 else None for v in wrong_risks]
    per_RX={str(r):1-float(m[:,i].mean())/float(b[:,i].mean()) if float(b[:,i].mean())>1e-16 else None for i,r in enumerate(RX)}
    a=cells[:,:,[0,2,4]].mean((1,2));a-=a.mean(0)
    other=cells[:,:,[1,3,5]].mean((1,2));other-=other.mean(0)
    cosine=_cosine(a,other)
    positive=sum(v['risk_gain'] is not None and v['risk_gain']>0 for v in folds)
    over_wrong=gain-max(wrong_gains) if gain is not None else None
    conditions=dict(nonzero_baseline=baseline>1e-16,mean_risk_gain=gain is not None and gain>=.05,positive_folds=positive>=14,
                    all_RX_nonnegative=all(v is not None and v>=0 for v in per_RX.values()),above_best_wrong_RX=over_wrong is not None and over_wrong>=.02,
                    TX_half_center_cosine=cosine>=.8)
    centers=cells.mean((1,2));centers-=centers.mean(0)
    return dict(decision='PASS' if all(conditions.values()) else 'FAIL',conditions=conditions,
                mean_risk_gain=gain,positive_folds=positive,total_folds=18,per_RX_risk_gain=per_RX,
                baseline_cell_risk=baseline,RX_cell_risk=risk,wrong_RX_risks=wrong_risks,wrong_RX_risk_gains=wrong_gains,
                gain_over_best_wrong_RX=over_wrong,TX_half_center_cosine=cosine,folds=folds,
                fit_RX_center_offsets=centers.tolist(),fit_RX_center_max_norm=float(np.linalg.norm(centers,axis=1).max()),
                mean_within_cell_DC_variance=float(spread.mean()),cell_means=cells.tolist(),
                role='L_train5040, RX3/4/6/8 only; no heldRX1/U/V',interpretation='Cross-TX/day RX-associated DC predictability only; not identified hardware or identity-recognition benefit')
