"""Balanced current-support calibration of old-head reliability in probability space."""
from dataclasses import dataclass
import numpy as np
from .support_prior_gate import confidence

RULE='support_balanced_ridge_v1'
RIDGE_FRACTION=.1


def features(probability):
    p=np.asarray(probability,np.float64)
    if p.ndim!=2 or p.shape[1]<2 or not np.isfinite(p).all() or (p<0).any() or not np.allclose(p.sum(1),1):raise ValueError('Invalid frozen-head probabilities')
    top=np.partition(p,-2,axis=1)[:,-2:];margin=top[:,1]-top[:,0]
    return np.column_stack((np.sqrt(p.shape[1])*(p-1/p.shape[1]),confidence(p),margin))


@dataclass(frozen=True)
class Gate:
    mode:str
    mean:np.ndarray
    coefficient:np.ndarray
    old_count:int
    old_rows:int
    new_rows:int
    covariance_trace:object
    ridge:object

    @property
    def state_bytes(self):return self.mean.nbytes+self.coefficient.nbytes

    def values(self,probability):
        p=np.asarray(probability,np.float64)
        if p.ndim!=2 or p.shape[1]!=self.old_count:raise ValueError('Different old-head columns')
        if self.mode=='old_only':return np.ones(len(p),np.float64)
        return np.clip(.5+(features(p)-self.mean)@self.coefficient,0,1)

    def audit(self):
        return dict(rule=RULE,mode=self.mode,old_count=self.old_count,old_rows=self.old_rows,new_rows=self.new_rows,feature_dimension=len(self.mean),
            ridge_fraction=RIDGE_FRACTION,covariance_trace=self.covariance_trace,ridge=self.ridge,coefficient_norm=float(np.linalg.norm(self.coefficient)),state_bytes=self.state_bytes,
            fit_scope='current_SUP_only',group_weights='old/new each total .5',query_fit=False)


def fit(probability,labels):
    p=np.asarray(probability,np.float64);x=features(p);y=np.asarray(labels,np.int64)
    if y.shape!=(len(x),) or (y<0).any():raise ValueError('Invalid physical support labels')
    old=y<p.shape[1];no=int(old.sum());nn=int((~old).sum());d=x.shape[1]
    if not no:raise ValueError('Old physical support required')
    if not nn:mode='old_only';mean=np.zeros(d);coefficient=np.zeros(d);trace=None;ridge=None
    else:
        weight=np.where(old,.5/no,.5/nn);mean=weight@x;centered=x-mean
        covariance=centered.T@(weight[:,None]*centered);trace=float(np.trace(covariance))
        if trace<=np.finfo(np.float64).eps:
            mode='constant_unseparated';coefficient=np.zeros(d);ridge=0.
        else:
            mode='balanced_ridge';ridge=RIDGE_FRACTION*trace/d
            coefficient=np.linalg.solve(covariance+ridge*np.eye(d),centered.T@(weight*(old.astype(np.float64)-.5)))
            if not np.isfinite(coefficient).all():raise FloatingPointError('Nonfinite support prior calibration')
    mean.flags.writeable=False;coefficient.flags.writeable=False
    return Gate(mode,mean,coefficient,p.shape[1],no,nn,trace,ridge)
