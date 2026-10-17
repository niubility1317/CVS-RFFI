"""Keep augmentation randomness separate from loaders, dropout and U recipe."""
from contextlib import contextmanager
import random
import numpy as np
import torch


@contextmanager
def stream(seed,call,device=None):
    py=random.getstate();np_state=np.random.get_state()
    devices=[device.index if device.index is not None else torch.cuda.current_device()] if device is not None and device.type=='cuda' else []
    # CPU generator seeding must not seed every initialized CUDA device.
    local_seed=int(np.random.SeedSequence([seed,call]).generate_state(1)[0])
    try:
        with torch.random.fork_rng(devices=devices):
            random.seed(local_seed);np.random.seed(local_seed);torch.random.default_generator.manual_seed(local_seed)
            for index in devices:torch.cuda.default_generators[index].manual_seed(local_seed)
            yield
    finally:random.setstate(py);np.random.set_state(np_state)


class PrivateAugment:
    def __init__(self,inner,seed):
        object.__setattr__(self,'_inner',inner);object.__setattr__(self,'_seed',seed)
        object.__setattr__(self,'_calls',0);object.__setattr__(self,'_packets',0)
        object.__setattr__(self,'_mse_sum',0.);object.__setattr__(self,'_max_abs',0.)
    def __getattr__(self,name):return getattr(self._inner,name)
    def __setattr__(self,name,value):
        if name.startswith('_'):object.__setattr__(self,name,value)
        else:setattr(self._inner,name,value)
    def __call__(self,x,*args,**kwargs):
        if x.shape!=(128,2,256):raise ValueError('Only original L128 enters ordinary augmentor')
        self._calls+=1;before=x.detach().clone()
        with stream(self._seed,self._calls,x.device):out=self._inner(x,*args,**kwargs)
        if out.shape!=x.shape or not bool(torch.isfinite(out).all()):raise ValueError('Ordinary IQ must remain finite')
        self._packets+=len(x);self._mse_sum+=float((out-before).square().sum())
        self._max_abs=max(self._max_abs,float(out.detach().abs().max()))
        return out
    def summary(self):
        return dict(calls=self._calls,packets=self._packets,waveform_MSE=self._mse_sum/max(1,self._packets*512),max_abs=self._max_abs,
            probabilities={k:float(getattr(self._inner,k)) for k in ('p_dac','p_pa','p_time_shift','p_amp_scale','p_phase_rot','p_cfo','p_phase_noise','p_awgn','p_multipath','p_dc_offset','p_bandedge_taper')})
