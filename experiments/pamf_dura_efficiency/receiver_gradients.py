"""Balanced SOURCE receiver gradient geometry, not optimizer-step causality."""
import numpy as np

RECEIVERS=(1,3,4,6,8)
SCENES=('clean','practical_high','practical_mid','practical_low_suburban',
        'practical_high_urban','practical_mid_urban','practical_low_urban')
PER_CELL=12


def balanced_indices(rx,scene,order):
    if (rx.ndim!=1 or scene.shape!=rx.shape or order.shape!=rx.shape
        or order.dtype.kind not in 'iu' or not np.array_equal(np.sort(order),np.arange(len(rx)))):
        raise ValueError('Matched source metadata and complete positional permutation required')
    rows=[]
    for receiver in RECEIVERS:
        cells=[]
        for name in SCENES:
            selected=order[(rx[order]==receiver)&(scene[order]==name)][:PER_CELL]
            if len(selected)!=PER_CELL:raise ValueError('Source RX/scene cell lacks twelve records')
            cells.append(selected)
        rows.append(np.concatenate(cells))
    return np.stack(rows)


def geometry(gradients):
    if (gradients.ndim!=2 or gradients.shape[0]!=5 or gradients.shape[1]<1
        or not np.isfinite(gradients).all()):
        raise ValueError('Five finite matched raw parameter-gradient vectors required')
    g=gradients.astype(np.float64);norm=np.linalg.norm(g,axis=1);mean=g.mean(0);mean_norm=float(np.linalg.norm(mean))
    pair_defined=(norm[:,None]*norm[None,:])>1e-24
    cosine=(g@g.T)/np.maximum(norm[:,None]*norm[None,:],1e-24)
    mean_defined=(norm*mean_norm)>1e-24
    mean_cosine=(g@mean)/np.maximum(norm*mean_norm,1e-24)
    upper=np.triu_indices(5,1)
    return dict(gradient_norms=norm.tolist(),mean_gradient_norm=mean_norm,pair_cosines=cosine.tolist(),
        pair_defined=pair_defined.tolist(),cosine_with_equal_RX_mean=mean_cosine.tolist(),mean_defined=mean_defined.tolist(),
        negative_pair_count=int(((cosine[upper]<-.05)&pair_defined[upper]).sum()),
        mean_conflict_count=int(((mean_cosine<-.05)&mean_defined).sum()),
        all_directions_defined=bool(pair_defined[upper].all() and mean_defined.all()))


def decision(views):
    if set(views)!=set(('1','20')):raise ValueError('Both fixed source views required')
    checks={view:dict(defined=r['all_directions_defined'],pair_conflicts=r['negative_pair_count']>=3,
                     mean_conflict=r['mean_conflict_count']>=1) for view,r in views.items()}
    return dict(supports_RX_conflict_hypothesis=all(all(c.values()) for c in checks.values()),
                checks=checks,automatic_training=False)
