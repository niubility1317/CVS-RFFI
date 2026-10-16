import math
import torch
from experiments.cvs_ordinary_fold20.streams import stream

class WhiteObservation:
    def __init__(self,seed):
        self.seed=int(seed);self._calls=0;self._packets=0;self.zero_packets=0
        self.ratio_min=None;self.ratio_max=None;self.mse_sum=0.;self.max_abs=0.;self.schedules=[]
    @staticmethod
    def transform(x,z):
        if x.shape!=z.shape or x.ndim!=3 or x.shape[1:]!=(2,256) or x.dtype!=torch.float32 or not bool(torch.isfinite(x).all()) or not bool(torch.isfinite(z).all()):
            raise ValueError('Finite FP32 IQ2x256 and matching independent draw required')
        power=x.double().square().sum((1,2),keepdim=True);direction=z.double().square().sum((1,2),keepdim=True)
        if bool((direction<=0).any()):raise ValueError('Nonzero noise direction required')
        return x+z*(.001*power/direction).sqrt().to(x.dtype)
    def __call__(self,x,labels=None,no_pa=True):
        if x.shape!=(128,2,256) or not no_pa:raise ValueError('White view only applies to L128 with PA disabled')
        self._calls+=1
        with torch.no_grad(),stream(self.seed,self._calls,x.device):
            out=self.transform(x,torch.randn_like(x))
        signal=x.double().square().sum((1,2));added=(out-x).double().square().sum((1,2));positive=signal>0
        ratio=added[positive]/signal[positive]
        if ratio.numel():
            low=float(ratio.min());high=float(ratio.max())
            if low<.000999 or high>.001001:raise ValueError('Actual white noise energy outside fixed 30dB bound')
            self.ratio_min=low if self.ratio_min is None else min(self.ratio_min,low)
            self.ratio_max=high if self.ratio_max is None else max(self.ratio_max,high)
        if not bool(torch.isfinite(out).all()) or bool((added[~positive]!=0).any()):raise ValueError('White zero-power/finiteness policy failed')
        self.zero_packets+=int((~positive).sum());self._packets+=len(x);self.mse_sum+=float(added.sum())
        self.max_abs=max(self.max_abs,float(out.abs().max()));return out
    def summary(self):
        return dict(kind='WHITE_FIXED_ENERGY',calls=self._calls,packets=self._packets,zero_power_packets=self.zero_packets,
            signal_to_added_noise_db_min=-10*math.log10(self.ratio_max) if self.ratio_max else None,
            signal_to_added_noise_db_max=-10*math.log10(self.ratio_min) if self.ratio_min else None,
            energy_ratio_min=self.ratio_min,energy_ratio_max=self.ratio_max,waveform_MSE=self.mse_sum/max(1,self._packets*512),
            max_abs=self.max_abs,all_other_operators=False,labels_used=False,hardware_changes=False,total_SNR_claim=False)
