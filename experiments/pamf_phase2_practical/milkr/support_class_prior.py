"""Current-SUP closed-form class calibration; the neural default head stays frozen."""
from dataclasses import dataclass
import numpy as np
from .support_ridge_prior import features,RIDGE_FRACTION

RULE='support_class_ridge_v1'


def simplex(value):
    z=np.asarray(value,np.float64)
    if z.ndim!=2 or z.shape[1]<2 or not np.isfinite(z).all():raise ValueError('Invalid class calibration scores')
    ordered=np.sort(z,axis=1)[:,::-1];cumulative=np.cumsum(ordered,axis=1)-1
    active=ordered-cumulative/np.arange(1,z.shape[1]+1)>0;last=active.sum(1)-1
    if (last<0).any():raise FloatingPointError('Invalid simplex calibration threshold')
    threshold=cumulative[np.arange(len(z)),last]/(last+1);out=np.maximum(z-threshold[:,None],0);total=out.sum(1,keepdims=True)
    if not np.isfinite(total).all() or (total<=0).any():raise FloatingPointError('Invalid calibrated probability mass')
    return out/total


@dataclass(frozen=True)
class Gate:
    mode:str
    mean:np.ndarray
    coefficient:np.ndarray
    target_mean:np.ndarray
    old_count:int
    old_rows:int
    new_rows:int
    covariance_trace:float
    ridge:float

    @property
    def state_bytes(self):return self.mean.nbytes+self.coefficient.nbytes+self.target_mean.nbytes

    def latent(self,probability):
        p=np.asarray(probability,np.float64)
        if p.ndim!=2 or p.shape[1]!=self.old_count:raise ValueError('Different mapped old-head columns')
        return self.target_mean+(features(p)-self.mean)@self.coefficient

    def values(self,probability):
        if self.new_rows==0:return np.ones(len(probability),np.float64)
        return np.clip(self.latent(probability)[:,0],0,1)

    def calibrated_probabilities(self,probability):return simplex(self.latent(probability)[:,1:])

    def audit(self):
        return dict(rule=RULE,mode=self.mode,old_count=self.old_count,old_rows=self.old_rows,new_rows=self.new_rows,feature_dimension=len(self.mean),output_dimension=len(self.target_mean),
            ridge_fraction=RIDGE_FRACTION,covariance_trace=self.covariance_trace,ridge=self.ridge,coefficient_norm=float(np.linalg.norm(self.coefficient)),state_bytes=self.state_bytes,
            fit_scope='current_SUP_only',group_weights='old/new each .5 when both present; old-only total1',class_target='old onehot, new uniform over old columns',class_projection='Euclidean simplex',query_fit=False)


def fit(probability,labels):
    p=np.asarray(probability,np.float64);x=features(p);y=np.asarray(labels,np.int64)
    if y.shape!=(len(p),) or (y<0).any():raise ValueError('Invalid physical support labels')
    m=p.shape[1];old=y<m;no=int(old.sum());nn=int((~old).sum());d=x.shape[1]
    if not no:raise ValueError('Old physical support required')
    if nn:weight=np.where(old,.5/no,.5/nn)
    else:weight=np.full(len(p),1/no)
    classes=np.full((len(p),m),1/m);classes[old]=np.eye(m)[y[old]];target=np.column_stack((old.astype(np.float64),classes))
    mean=weight@x;target_mean=weight@target;centered=x-mean;covariance=centered.T@(weight[:,None]*centered);trace=float(np.trace(covariance))
    if trace<=np.finfo(np.float64).eps:
        coefficient=np.zeros((d,m+1));ridge=0.;mode='old_only_constant' if not nn else 'constant_unseparated'
    else:
        ridge=RIDGE_FRACTION*trace/d;coefficient=np.linalg.solve(covariance+ridge*np.eye(d),centered.T@(weight[:,None]*(target-target_mean)))
        if not np.isfinite(coefficient).all():raise FloatingPointError('Nonfinite class prior calibration')
        mode='old_only' if not nn else 'balanced_class_ridge'
    for value in (mean,coefficient,target_mean):value.flags.writeable=False
    return Gate(mode,mean,coefficient,target_mean,m,no,nn,trace,ridge)
