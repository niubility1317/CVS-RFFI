"""Source soft-coordinate references balanced over other RX/day cells."""
import numpy as np

RX=(1,3,4,6,8)
DAYS=(1,2,3)
BETA=.25


def fit_references(features,probabilities,rx,day,fit_mask):
    n=len(rx)
    if (features.shape!=(n,160) or probabilities.shape!=(n,6) or day.shape!=(n,) or fit_mask.shape!=(n,)
        or fit_mask.dtype!=bool or not np.isfinite(features).all() or not np.isfinite(probabilities).all()
        or (probabilities<0).any() or not np.allclose(probabilities.sum(1),1,rtol=1e-6,atol=1e-6)
        or set(rx.tolist())!=set(RX) or set(day.tolist())!=set(DAYS)):
        raise ValueError('Aligned finite source160D, soft6D probabilities and original fiveRX/three days required')
    means=np.empty((5,3,6,160),np.float64);masses=np.empty((5,3,6),np.float64);effective=np.empty_like(masses)
    for i,r in enumerate(RX):
        for j,d in enumerate(DAYS):
            mask=fit_mask&(rx==r)&(day==d);weights=probabilities[mask];mass=weights.sum(0)
            if np.any(mass<=0):raise ValueError('Each source RX/day soft coordinate must have positive mass')
            means[i,j]=weights.T@features[mask]/mass[:,None];masses[i,j]=mass
            effective[i,j]=np.square(mass)/np.square(weights).sum(0)
    leave=np.stack([means[np.arange(5)!=i].mean((0,1)) for i in range(5)])
    return dict(cell_means=means,cell_mass=masses,cell_effective_n=effective,leave_RX_centroids=leave)


def make_reference(features,probabilities,rx,centroids,beta=BETA):
    if (features.shape!=(len(rx),160) or probabilities.shape!=(len(rx),6) or centroids.shape!=(5,6,160)
        or not np.isin(rx,RX).all() or not 0<=beta<=1):
        raise ValueError('Paired original features and fixed leave-RX soft centroids required')
    result=np.empty_like(features,dtype=np.float64)
    for i,r in enumerate(RX):
        mask=rx==r;target=probabilities[mask]@centroids[i]
        value=(1-beta)*features[mask]+beta*target
        norm=np.linalg.norm(value,axis=1,keepdims=True)
        if np.any(norm<=1e-12) or not np.isfinite(value).all():raise ValueError('Nonzero finite source feature reference required')
        result[mask]=value/norm
    return result


def head_geometry(features,probabilities,rx,basis):
    projected=features@basis;means=[]
    for r in RX:
        mask=rx==r;weights=probabilities[mask];mass=weights.sum(0)
        if np.any(mass<=0):raise ValueError('Defined fiveRX soft-coordinate geometry required')
        means.append(weights.T@projected[mask]/mass[:,None])
    means=np.stack(means);classes=means.mean(0)
    return dict(RX_scatter=float(np.square(means-classes[None]).sum(2).mean()),
        class_spread=float(np.square(classes-classes.mean(0)).sum(1).mean()))


def eligible(min_effective_n,choices_preserved,before,after):
    if before['RX_scatter']<=0 or before['class_spread']<=0:raise ValueError('Nondegenerate original head geometry required')
    return bool(min_effective_n>=32 and choices_preserved==512 and after['RX_scatter']<=.8*before['RX_scatter']
        and after['class_spread']>=.9*before['class_spread'])
