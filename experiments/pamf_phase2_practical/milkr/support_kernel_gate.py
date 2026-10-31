"""Balanced current-support RF membership evidence for the old-head prior."""
from dataclasses import dataclass
import numpy as np

RULE='support_balanced_kernel_evidence_v1'
FRACTION=.1


@dataclass(frozen=True)
class Gate:
    alpha:np.ndarray
    intercept:float
    ridge:float
    old_rows:int
    new_rows:int

    def factors(self,cross):
        value=np.asarray(cross,np.float64)
        if value.ndim!=2 or not np.isfinite(value).all():raise ValueError('Invalid RF prior cross-kernel')
        if not self.new_rows:return np.ones(len(value))
        if value.shape[1]!=len(self.alpha):raise ValueError('RF gate support alignment differs')
        posterior=np.clip(value@self.alpha+self.intercept,0,1)
        # Under balanced membership, 0.5 means no RF evidence against old.
        return np.minimum(2*posterior,1.)

    @property
    def state_bytes(self):return self.alpha.nbytes+16

    def audit(self):
        return dict(rule=RULE,ridge_fraction=FRACTION,ridge=self.ridge,old_rows=self.old_rows,new_rows=self.new_rows,
            mode='old_only' if not self.new_rows else 'constant_unseparated' if self.ridge==0 else 'balanced_kernel_ridge',
            state_bytes=self.state_bytes,neutral_membership=.5,factor='min(2*clip(kernel_membership,0,1),1)',
            scope='current_physical_train_support',query_fit=False)


def fit(kernel,labels,old_count):
    k=np.asarray(kernel,np.float64);y=np.asarray(labels,np.int64);old=y<old_count
    if y.ndim!=1 or k.shape!=(len(y),len(y)) or not len(y) or not np.isfinite(k).all() or not np.allclose(k,k.T):raise ValueError('Invalid RF gate training kernel')
    nold=int(old.sum());nnew=len(y)-nold
    if not nold:raise ValueError('RF gate needs registered old support')
    if not nnew:
        alpha=np.empty(0,np.float64);alpha.flags.writeable=False;return Gate(alpha,1.,0.,nold,0)
    weights=np.where(old,.5/nold,.5/nnew);target=old.astype(np.float64)
    variance=float(weights@np.diag(k)-weights@k@weights)
    if variance<0 and abs(variance)>1e-12:raise FloatingPointError('Negative RF gate feature variance')
    ridge=FRACTION*max(variance,0.)
    if ridge==0:alpha=np.zeros(len(y));intercept=.5
    else:
        system=k+ridge*np.diag(1/weights)
        solved=np.linalg.solve(system,np.column_stack((target,np.ones(len(y)))))
        intercept=float(solved[:,0].sum()/solved[:,1].sum());alpha=solved[:,0]-intercept*solved[:,1]
        if not np.isfinite(alpha).all() or not np.isfinite(intercept):raise FloatingPointError('Invalid RF gate coefficients')
    alpha.flags.writeable=False;return Gate(alpha,intercept,ridge,nold,nnew)
