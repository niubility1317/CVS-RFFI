"""Fixed cycle-power slope; statistical RX association is not identity efficacy."""
import numpy as np

RX=(3,4,6,8)
Q=np.array([-1.,-1/3,1/3,1.])


def observe(x):
    x=np.asarray(x,dtype=np.float64)
    if x.ndim!=3 or x.shape[1:]!=(2,256) or not np.isfinite(x).all():raise ValueError('Finite native IQ[N,2,256] required')
    power=np.square(x[:,:,80:160]).sum(1).reshape(len(x),4,20).mean(2)
    if np.any(power<=1e-20):raise ValueError('Nondegenerate fixed cycles required; preserve failure rather than silently drop packets')
    level=10*np.log10(power);center=level-level.mean(1,keepdims=True)
    slope=center@Q/float(Q@Q);residual=center-slope[:,None]*Q
    total=np.square(center).sum(1);error=np.square(residual).sum(1)
    r2=np.where(total>1e-16,1-error/np.maximum(total,1e-30),0.)
    # Whole-packet positive rescaling cancels exactly in the centered log powers.
    rms_x=x/np.sqrt(np.square(x).mean((1,2),keepdims=True))
    rms_power=np.square(rms_x[:,:,80:160]).sum(1).reshape(len(x),4,20).mean(2)
    rms_level=10*np.log10(rms_power);rms_slope=(rms_level-rms_level.mean(1,keepdims=True))@Q/float(Q@Q)
    return dict(slope_db=slope,cycle_levels_centered_db=center,linear_R2=r2,
                max_RMS_slope_difference=float(np.abs(slope-rms_slope).max()))


def crossfit(slope,tx,rx,day):
    slope=np.asarray(slope,dtype=np.float64);tx=np.asarray(tx);rx=np.asarray(rx);day=np.asarray(day)
    if slope.shape!=(5040,) or any(v.shape!=(5040,) for v in (tx,rx,day)) or not np.isfinite(slope).all():raise ValueError('Complete finite L5040 fit set required')
    if set(tx)!=set(range(6)) or set(rx)!=set(RX) or set(day)!={1,2,3}:raise ValueError('Exact visible L TX/RX/day roles required')
    cells=np.zeros((4,3,6));signed=np.zeros_like(cells)
    for i,r in enumerate(RX):
        for j,d in enumerate((1,2,3)):
            for t in range(6):
                v=slope[(rx==r)&(day==d)&(tx==t)]
                if len(v)!=70:raise ValueError('Original70 physical L packets per cell required')
                cells[i,j,t]=np.median(np.abs(v));signed[i,j,t]=np.median(v)
    folds=[];base_errors=[];model_errors=[];wrong_errors=[[],[],[]]
    for hd in range(3):
        for ht in range(6):
            training=cells[:,[d for d in range(3) if d!=hd]][:,:,[t for t in range(6) if t!=ht]]
            centers=training.mean((1,2));common=float(centers.mean());observed=cells[:,hd,ht]
            b=np.square(observed-common);m=np.square(observed-centers)
            baseline=float(b.mean());gain=1-float(m.mean())/baseline if baseline>1e-16 else None
            folds.append(dict(held_day=hd+1,held_TX=ht,baseline_risk=baseline,RX_risk=float(m.mean()),risk_gain=gain))
            base_errors.append(b);model_errors.append(m)
            for i,s in enumerate((1,2,3)):wrong_errors[i].append(np.square(observed-np.roll(centers,s)))
    b=np.array(base_errors);m=np.array(model_errors);baseline=float(b.mean());risk=float(m.mean());gain=1-risk/baseline if baseline>1e-16 else None
    wrong=[1-float(np.mean(v))/baseline if baseline>1e-16 else None for v in wrong_errors]
    per_RX={str(r):1-float(m[:,i].mean())/float(b[:,i].mean()) if float(b[:,i].mean())>1e-16 else None for i,r in enumerate(RX)}
    a=cells[:,:,[0,2,4]].mean((1,2));a-=a.mean();z=cells[:,:,[1,3,5]].mean((1,2));z-=z.mean()
    norm=float(np.linalg.norm(a)*np.linalg.norm(z));cosine=float(a@z/norm) if norm>1e-20 else 0.
    positive=sum(v['risk_gain'] is not None and v['risk_gain']>0 for v in folds);over_wrong=gain-max(wrong) if gain is not None else None
    conditions=dict(nonzero_baseline=baseline>1e-16,mean_risk_gain=gain is not None and gain>=.05,positive_folds=positive>=14,
                    all_RX_nonnegative=all(v is not None and v>=0 for v in per_RX.values()),above_best_wrong_RX=over_wrong is not None and over_wrong>=.02,TX_half_center_cosine=cosine>=.8)
    return dict(decision='PASS' if all(conditions.values()) else 'FAIL',conditions=conditions,mean_risk_gain=gain,positive_folds=positive,total_folds=18,
                per_RX_risk_gain=per_RX,baseline_cell_risk=baseline,RX_cell_risk=risk,wrong_RX_risk_gains=wrong,gain_over_best_wrong_RX=over_wrong,
                TX_half_center_cosine=cosine,folds=folds,median_abs_slope_cells=cells.tolist(),median_signed_slope_cells=signed.tolist(),
                interpretation='Only RX-associated cycle-power slope repeatability; not identified AGC hardware, label preservation, or identity benefit')
