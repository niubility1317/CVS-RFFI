"""Strict source-only checkpoint forward before clean target tasks."""
import numpy as np
from .external_execution import infer
from .io import read, require


def source_samples(rt,count):
    from experiments.cric_dura_comparison.source_iq import aligned_source_iq
    contract=read(rt['source_contract'])
    x,_=aligned_source_iq(rt['source_cache'],'L_s',contract['role_ids']['L_s'])
    require(x.shape==(contract['counts']['L_s'],2,256) and 0<count<=len(x),'Source smoke input shape mismatch')
    return np.array(x[:count],dtype=np.float32,copy=True)


def check(rt,bundle):
    x=source_samples(rt,2)
    require(x.shape==(2,2,256),'Source smoke input shape mismatch')
    batch=1 if bundle.get('profile')=='tweak_m1' else 2
    pred,_=infer(bundle,x,6,batch=batch)
    require(pred.shape==(2,), 'Source-only forward failed')
