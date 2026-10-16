import torch
from experiments.cvs_ordinary_fold20.streams import stream

class OriginJitter:
    def __init__(self,seed):
        self.seed=int(seed);self._calls=0;self._packets=0;self.schedules=[];self.delay_counts=[0]*25
        self.mse_sum=0.;self.max_abs=0.;self.ratio_min=None;self.ratio_max=None
    @staticmethod
    def transform(x,delays):
        if (x.ndim!=3 or x.shape[1:]!=(2,256) or x.dtype!=torch.float32 or not bool(torch.isfinite(x).all())
            or delays.shape!=(len(x),) or delays.dtype!=torch.long or delays.device!=x.device
            or bool(((delays<0)|(delays>24)).any())):raise ValueError('FP32 IQ256 and integer observation origin0..24 required')
        indices=(torch.arange(256,device=x.device)[None,:]-delays[:,None]).abs()
        return x.gather(2,indices[:,None,:].expand(-1,2,-1))
    def __call__(self,x,labels=None,no_pa=True):
        if x.shape!=(128,2,256) or not no_pa:raise ValueError('OJ24 only applies to L128 with PA disabled')
        self._calls+=1
        with torch.no_grad(),stream(self.seed,self._calls,x.device):
            delays=torch.randint(0,25,(len(x),),device=x.device);out=self.transform(x,delays)
        for value in delays.cpu().tolist():self.delay_counts[value]+=1
        power=x.double().square().sum((1,2));after=out.double().square().sum((1,2));ratio=after[power>0]/power[power>0]
        if ratio.numel():
            low,high=float(ratio.min()),float(ratio.max());self.ratio_min=low if self.ratio_min is None else min(self.ratio_min,low);self.ratio_max=high if self.ratio_max is None else max(self.ratio_max,high)
        self._packets+=len(x);self.mse_sum+=float((out-x).double().square().sum());self.max_abs=max(self.max_abs,float(out.abs().max()))
        return out
    def summary(self):
        return dict(kind='CAUSAL_REFLECT_ORIGIN_JITTER',calls=self._calls,packets=self._packets,delay_counts=self.delay_counts,
            delay_min=0,delay_max=24,synthetic_prefix_max=24,real_observation_min=232,window_energy_ratio_min=self.ratio_min,
            window_energy_ratio_max=self.ratio_max,waveform_MSE=self.mse_sum/max(1,self._packets*512),max_abs=self.max_abs,
            circular_roll=False,post_normalization=False,all_other_operators=False,labels_used=False,hardware_changes=False)
