"""Public, parameter-free observation transforms; no trained F or class inputs."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as functional

DIMENSIONS=dict(mirror=128,spectral=4,reference=28)
FLOOR=.05
WEIGHT=.01


def require_iq(x):
    if x.ndim!=3 or x.shape[1:]!=(2,256) or x.dtype not in (torch.float32,torch.float64) or not len(x) or not torch.isfinite(x).all():
        raise ValueError('Finite nonempty current IQ[B,2,256], FP32/FP64 required')


class AnalyticViews(nn.Module):
    def __init__(self):
        super().__init__()
        tones=np.array([-24,-20,-16,-12,-8,-4,4,8,12,16,20,24])
        signs=np.array([1,-1,1,-1,-1,1,-1,-1,1,1,1,1])
        n=np.arange(20,dtype=np.float64)
        bank=np.stack([np.sum(signs*(1+1j)*np.exp(2j*np.pi*(n[:,None]-d/16)*tones/80),axis=-1)/np.sqrt(24) for d in range(320)])
        self.register_buffer('bank_real',torch.tensor(bank.real.tolist(),dtype=torch.float64),persistent=False)
        self.register_buffer('bank_imag',torch.tensor(bank.imag.tolist(),dtype=torch.float64),persistent=False)
        self.register_buffer('tone_bins',torch.tensor((tones//4%20).tolist()),persistent=False)

    def reference(self,z):
        epsilon=1e-6;w=z[:,80:160]
        cross=(w[:,20:]*w[:,:-20].conj()).sum(-1)
        omega=torch.atan2(cross.imag,cross.real+epsilon**2)/20
        n=torch.arange(80,device=z.device,dtype=z.real.dtype)
        corrected=w*torch.complex(torch.cos(-omega[:,None]*n),torch.sin(-omega[:,None]*n))
        mean=corrected.reshape(-1,4,20).mean(1);energy=mean.abs().square().mean(-1)
        unit=mean/(energy+epsilon**2).sqrt()[:,None]
        bank=torch.complex(self.bank_real.to(z.real.dtype),self.bank_imag.to(z.real.dtype))
        corr=unit@bank.conj().T/20;chosen=corr.abs().square().argmax(-1)
        gain=corr.gather(1,chosen[:,None]).squeeze(1)
        observed=torch.fft.fft(unit,dim=-1)[:,self.tone_bins]
        ideal=torch.fft.fft(bank[chosen],dim=-1)[:,self.tone_bins]
        relative=observed*ideal.conj()/ideal.abs().square()
        relative=relative*gain.conj()[:,None]/(gain.abs().square()+epsilon)[:,None]
        residual=relative-1;bounded=residual/torch.sqrt(1+residual.abs().square()/16)
        power=w.abs().square().mean(-1);packet_power=z.abs().square().mean(-1)
        norm=(w[:,20:].abs().square().sum(-1)*w[:,:-20].abs().square().sum(-1)+epsilon**4).sqrt()
        return torch.cat((bounded.real,bounded.imag,gain.abs().square()[:,None],(cross.abs()/norm)[:,None],
            (energy/(power+epsilon**2))[:,None],(power/(packet_power+epsilon**2))[:,None]),1)

    def forward(self,x):
        require_iq(x);epsilon=1e-8;z=torch.complex(x[:,0],x[:,1]);spec=torch.fft.fft(z,dim=-1)
        power=torch.nan_to_num(spec.abs().square(),nan=0.,posinf=1e6,neginf=0.).clamp_min(0.)
        pos=power[:,1:128];neg=power[:,129:].flip(-1)
        logp=torch.log1p(pos.clamp_max(1e6));logn=torch.log1p(neg.clamp_max(1e6))
        ratio=(torch.log(pos+epsilon)-torch.log(neg+epsilon)).clamp(-20,20)
        asym=(pos-neg).abs()/(pos+neg+epsilon)
        pooled=functional.adaptive_avg_pool1d(torch.stack((logp,logn,ratio,asym,pos,neg),1),32)
        total=torch.nan_to_num(pooled[:,4]+pooled[:,5],nan=0.,posinf=1e6,neginf=0.).clamp_min(epsilon).clamp_max(1e6)
        edge=total[:,-6:].sum(1,keepdim=True);center=total[:,9:22].sum(1,keepdim=True)+epsilon
        mean=total.mean(1,keepdim=True);variance=(total-mean).square().mean(1,keepdim=True)+epsilon
        kurtosis=((total-mean).clamp(-1e3,1e3).pow(4).mean(1,keepdim=True)/variance.square().clamp_min(epsilon)).clamp(0,1e4)
        circularity=(z.square().mean(-1).abs()/(z.abs().square().mean(-1)+epsilon))[:,None]
        stats=torch.cat((edge/(total.sum(1,keepdim=True)+epsilon),edge/center,kurtosis,circularity),1)
        return dict(mirror=pooled[:,:4].flatten(1),spectral=stats,reference=self.reference(z))


def validate_scales(scales):
    if scales is None or set(scales)!=set(DIMENSIONS):raise ValueError('All three fixed source scale groups required')
    for key,size in DIMENSIONS.items():
        value=scales[key]
        if not torch.is_tensor(value) or value.shape!=(size,) or not torch.isfinite(value).all() or (value<FLOOR-1e-8).any():
            raise ValueError('Finite fixed per-coordinate training std with floor .05 required')


@torch.no_grad()
def fit_scales(batches,extractor=None):
    sums={};squares={};count=0
    for x in batches:
        if extractor is None:extractor=AnalyticViews().to(x.device)
        for key,value in extractor(x).items():
            value=value.double();sums[key]=sums.get(key,0)+value.sum(0);squares[key]=squares.get(key,0)+value.square().sum(0)
        count+=len(x)
    if not count:raise ValueError('Nonempty source training clean required')
    scales={key:(squares[key]/count-(sums[key]/count).square()).clamp_min(0).sqrt().clamp_min(FLOOR) for key in DIMENSIONS}
    validate_scales(scales);return scales


def paired_loss(estimate,clean,scales,*,analytic,extractor=None):
    require_iq(estimate);require_iq(clean)
    if estimate.shape!=clean.shape:raise ValueError('Matching simulated clean target required')
    target=clean.detach();iq=(estimate-target).square().mean((1,2))/target.square().mean((1,2)).clamp_min(1e-4)
    if not analytic:return iq,dict(iq_nmse=iq)
    validate_scales(scales)
    if extractor is None:extractor=AnalyticViews().to(estimate.device)
    predicted=extractor(estimate)
    with torch.no_grad():truth=extractor(target)
    parts={}
    for key in DIMENSIONS:
        delta=(predicted[key]-truth[key])/scales[key].detach().to(device=estimate.device,dtype=estimate.dtype)
        parts[key+'_loss']=functional.smooth_l1_loss(delta,torch.zeros_like(delta),beta=1.,reduction='none').mean(1)
    total=torch.stack(list(parts.values()),1).mean(1)
    return iq+WEIGHT*total,dict(iq_nmse=iq,analytic_loss=total,**parts)
