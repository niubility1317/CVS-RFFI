"""Frozen default-head probabilities; no new classification head is fitted."""
import numpy as np
from .milkr.orbit_cca import OrbitCCA
from .milkr.source_bridge import BLOCKS

PHASE_KEY='teacher_phase_logits'


def probabilities(logits,old_columns=None):
    value=np.asarray(logits,np.float64)
    if value.ndim!=3 or value.shape[1:]!=(4,6) or not np.isfinite(value).all():raise ValueError('Expected actual four-phase six-class default logits')
    columns=list(range(6)) if old_columns is None else list(old_columns)
    if len(columns)<2 or len(set(columns))!=len(columns) or any(c not in range(6) for c in columns):raise ValueError('Invalid old-head columns')
    # Remove held-new head columns BEFORE softmax in source old4/new2 proxies.
    selected=value[:,:,columns];selected=np.exp(selected-selected.max(2,keepdims=True))
    return (selected/selected.sum(2,keepdims=True)).mean(1)


def ordinary(features):
    if set(features)!=set(BLOCKS[:5])|{'sig_views',PHASE_KEY}:raise ValueError('Exact teacher feature payload required')
    return {key:value for key,value in features.items() if key!=PHASE_KEY}


def extract(reader,raw,device,engine=None,batch=128):
    from .phase_orbit import extract as orbit_extract
    features,logits=orbit_extract(reader,raw,device,engine,batch,return_phase_logits=True)
    return dict(features,teacher_phase_logits=logits),logits[:,0,:]


class TeacherBridge:
    encoder_type=OrbitCCA
    def __init__(self,cca):self.cca=cca;self.rank=64

    @classmethod
    def fit(cls,raw,restored):return cls(cls.encoder_type.fit(ordinary(raw),ordinary(restored)))

    def transform(self,features,*,restored,old_columns=None):
        out=self.cca.transform(ordinary(features),restored=restored)
        out['teacher_probs']=probabilities(features[PHASE_KEY],old_columns)
        return out

    def save(self,path):self.cca.save(path)

    @classmethod
    def load(cls,path):return cls(cls.encoder_type.load(path))


class NativeSkipBridge(TeacherBridge):
    from .milkr.native_skip import NativeSkip as encoder_type


def bridge_type(method):return NativeSkipBridge if method=='native_skip' else TeacherBridge
