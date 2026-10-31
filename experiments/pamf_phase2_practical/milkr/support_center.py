"""Half-shrink RF centering fitted on current physical support only."""
from dataclasses import dataclass
import numpy as np
from .support_calibrated import NAMES

STRENGTH=.5


@dataclass(frozen=True)
class Center:
    means:dict
    physical_count:int

    @property
    def state_bytes(self):return sum(value.nbytes for value in self.means.values())

    def apply(self,features):
        result={}
        for name in NAMES:
            value=features[name]-STRENGTH*self.means[name];norm=np.linalg.norm(value,axis=1,keepdims=True)
            if not np.isfinite(norm).all() or (norm<=1e-12).any():raise FloatingPointError('Invalid support-centered RF norm')
            value=value/norm;value.flags.writeable=False;result[name]=value
        value=features['sig_views'].reshape(-1,3,112)*np.sqrt(3)-STRENGTH*self.means['sig_views']
        norm=np.linalg.norm(value,axis=2,keepdims=True)
        if not np.isfinite(norm).all() or (norm<=1e-12).any():raise FloatingPointError('Invalid support-centered signature norm')
        value=(value/norm/np.sqrt(3)).reshape(-1,336);value.flags.writeable=False;result['sig_views']=value
        result['teacher_probs']=features['teacher_probs']
        return result

    def audit(self):
        norms=[float(np.linalg.norm(self.means[name])) for name in NAMES]+np.linalg.norm(self.means['sig_views'],axis=1).tolist()
        return dict(strength=STRENGTH,fit_scope='current_train_SUP_only',physical_count=self.physical_count,blocks=8,mean_norms=norms,state_bytes=self.state_bytes,query_fit=False)


def fit(features,strength):
    if strength!=STRENGTH:raise ValueError('Fixed half-shrink support center required')
    means={name:features[name].mean(0) for name in NAMES}
    means['sig_views']=(features['sig_views'].reshape(-1,3,112)*np.sqrt(3)).mean(0)
    for value in means.values():
        if not np.isfinite(value).all():raise FloatingPointError('Nonfinite support RF center')
        value.flags.writeable=False
    return Center(means,len(features['teacher_probs']))
