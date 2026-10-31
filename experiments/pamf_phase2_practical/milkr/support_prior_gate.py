"""Frozen support calibration of default-head confidence, not query groups."""
from dataclasses import dataclass,asdict
import numpy as np

RULE='support_entropy_means_v1'


def confidence(probability):
    p=np.asarray(probability,np.float64);logp=np.zeros_like(p);np.log(p,out=logp,where=p>0)
    return np.clip(1+(p*logp).sum(1)/np.log(p.shape[1]),0,1)


@dataclass(frozen=True)
class Gate:
    mode:str
    old_mean:float
    new_mean:object
    old_count:int
    old_rows:int
    new_rows:int

    @property
    def state_bytes(self):return 16

    def values(self,probability):
        if self.mode=='old_only':return np.ones(len(probability),np.float64)
        if self.mode=='no_positive_gap':return np.zeros(len(probability),np.float64)
        return np.clip((confidence(probability)-self.new_mean)/(self.old_mean-self.new_mean),0,1)

    def audit(self):return dict(rule=RULE,**asdict(self))


def fit(probability,labels):
    p=np.asarray(probability,np.float64);y=np.asarray(labels,np.int64);old=y<p.shape[1];c=confidence(p)
    a=float(c[old].mean());old_rows=int(old.sum());new_rows=int((~old).sum())
    if not new_rows:return Gate('old_only',a,None,p.shape[1],old_rows,0)
    b=float(c[~old].mean());mode='positive_gap' if a-b>np.finfo(np.float64).eps else 'no_positive_gap'
    return Gate(mode,a,b,p.shape[1],old_rows,new_rows)
