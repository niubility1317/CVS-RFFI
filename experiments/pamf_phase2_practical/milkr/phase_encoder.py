"""Frozen shared per-phase paired CCA; physical records keep their C4 axis."""
import numpy as np
from .paired_cca import PairedCCA
from .source_bridge import BLOCKS,DIMS


def flatten(features):
    if set(features)!=set(BLOCKS[:5])|{'sig_views'}:raise ValueError('Exact C4 feature schema required')
    result={};n=len(features['z_id'])
    for key,d in list(zip(BLOCKS[:5],DIMS[:5]))+[('sig_views',336)]:
        value=np.asarray(features[key],np.float64)
        if value.shape!=(n,4,d) or not np.isfinite(value).all():raise ValueError('Wrong C4 feature dimensions')
        result[key]=value.reshape(n*4,d)
    return result


class PhaseEncoder:
    def __init__(self,encoder):
        if not isinstance(encoder,PairedCCA) or encoder.rank!=64:raise ValueError('Fixed per-phase CCA rank64 required')
        self.encoder=encoder;self.rank=64

    @classmethod
    def fit(cls,raw,restored):
        return cls(PairedCCA.fit(flatten(raw),flatten(restored),64))

    def transform(self,features,*,restored):
        values=self.encoder.transform(flatten(features),restored=restored)
        return {key:value.reshape(len(features['z_id']),4,value.shape[1]) for key,value in values.items()}

    def save(self,path):self.encoder.save(path)

    @classmethod
    def load(cls,path):return cls(PairedCCA.load(path))
