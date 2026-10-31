"""Source-only CCA coordinates weighted by square-root canonical correlation."""
import numpy as np
from .orbit_cca import OrbitCCA,blocks
from .paired_cca import inverse_root

POWER=.5


def pair_projection(x,y,rank=64):
    mx=x.mean(0);my=y.mean(0);xc=x-mx;yc=y-my
    wx=inverse_root(xc.T@xc/len(x));wy=inverse_root(yc.T@yc/len(y))
    u,s,vt=np.linalg.svd(wx@(xc.T@yc/len(x))@wy,full_matrices=False)
    if not np.isfinite(s).all() or s[0]>1+1e-8:raise FloatingPointError('Invalid canonical correlations')
    correlation=np.clip(s[:rank],0,1);weight=correlation**POWER
    return (mx,my,(wx@u[:,:rank])*weight,(wy@vt.T[:,:rank])*weight),correlation


class CorrelationCCA(OrbitCCA):
    @classmethod
    def fit(cls,raw,restored):
        x,y=blocks(raw),blocks(restored)
        if len(x[0])<2 or any(len(v)!=len(x[0]) for v in x+y):raise ValueError('Source pair alignment differs')
        parameters=[];summary=[]
        for a,b in zip(x,y):
            p,c=pair_projection(a,b);parameters.append(p)
            summary.append(dict(min=float(c.min()),mean=float(c.mean()),max=float(c.max())))
        model=cls(parameters);model.correlation_summary=summary;return model
