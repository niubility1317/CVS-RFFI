"""Positive diagonal kernel normalization against the current support measure."""
from dataclasses import dataclass
import numpy as np

RULE='support_rbf_degree_v1'


def degree(affinity):
    value=np.asarray(affinity,np.float64)
    if value.ndim!=2 or not value.shape[1] or not np.isfinite(value).all() or (value<0).any():
        raise ValueError('Invalid nonnegative support affinities')
    result=value.mean(1)
    if (result<=0).any():raise FloatingPointError('Nonpositive support kernel degree')
    return result


@dataclass(frozen=True)
class Density:
    support_degree:np.ndarray
    average:float

    @classmethod
    def fit(cls,affinity):
        if np.ndim(affinity)!=2 or affinity.shape[0]!=affinity.shape[1]:raise ValueError('Square training affinity required')
        values=degree(affinity);values.flags.writeable=False
        return cls(values,float(values.mean()))

    def cross(self,kernel,affinity):
        k=np.asarray(kernel,np.float64);a=np.asarray(affinity,np.float64)
        if k.shape!=a.shape or k.ndim!=2 or k.shape[1]!=len(self.support_degree) or not np.isfinite(k).all():
            raise ValueError('Density kernel/SUP alignment differs')
        query_degree=degree(a)
        return self.average*k/np.sqrt(query_degree[:,None]*self.support_degree[None,:])

    def audit(self):
        return dict(rule=RULE,physical_count=len(self.support_degree),include_self=True,
            affinity='equal_C4_C8_RBF_without_linear_component',normalization='mean_degree_times_K_over_sqrt_row_column_degrees',
            degree_min=float(self.support_degree.min()),degree_max=float(self.support_degree.max()),degree_mean=self.average,
            state_bytes=self.support_degree.nbytes+8,scope='current_physical_train_support',query_fit=False)
