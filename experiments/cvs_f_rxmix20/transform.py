from collections import Counter
import numpy as np
import torch
from . import design as d
KEYS=('eq_i','rx_i','day_i','sig_i','base_index')
def source_bank():
    from experiments.cvs_source_ellipticity import design as sd
    s=d.read(d.STATISTIC/'summary.json');done=d.read(d.STATISTIC.parent/'completion.json')
    if s['config']!=sd.config(sd.rows()[0]) or s['status']!='PASS' or not all(s['checks'].values()) or done['status']!='ARTIFACTS_COMPLETE':raise ValueError('Exact registered source statistic PASS required')
    if s['roles']['selected_L_packets']!=5040 or any(s['roles'][k] for k in ('held_RX1_IQ_values_used','U_values_or_truth_used','V_values_used','target_access')) or any(s[k] for k in ('checkpoint_loads','model_forwards','optimizer_updates','head_fit_updates')):raise ValueError('Statistic source roles differ')
    with np.load(d.STATISTIC/'source_ellipticity_statistics.npz',allow_pickle=False) as f:profile=f['RX_mean_profile']
    if profile.shape!=(4,) or not np.isfinite(profile).all():raise ValueError('Four finite fitRX ellipticity contrasts required')
    return np.asarray([min(.02,abs(float(a-b))/2) for i,a in enumerate(profile) for b in profile[i+1:]],dtype=np.float64)
def mix(x,b):
    if x.ndim!=3 or x.shape[1:]!=(2,256) or not x.is_floating_point() or not bool(torch.isfinite(x).all()):raise ValueError('Finite source IQ[batch,2,256]')
    if b.shape!=(len(x),2) or b.device!=x.device or b.dtype!=x.dtype or not bool(torch.isfinite(b).all()) or bool((b.square().sum(1)>.02**2+1e-10).any()):raise ValueError('One bounded complex b per packet')
    br,bi=b[:,0,None],b[:,1,None];scale=(1+br.square()+bi.square()).sqrt();i,q=x[:,0],x[:,1]
    return torch.stack(((1+br)*i+bi*q,bi*i+(1-br)*q),1)/scale[:,None,:]
def inverse(y,b):
    br,bi=b[:,0,None],b[:,1,None];z=y*(1+br.square()+bi.square()).sqrt()[:,None,:];i,q=z[:,0],z[:,1]
    return torch.stack(((1-br)*i-bi*q,-bi*i+(1+br)*q),1)/(1-br.square()-bi.square())[:,None,:]
class Sessions:
    def __init__(self,seed,bank,fit=(3,4,6,8)):
        self.seed=int(seed);self.bank=np.asarray(bank,dtype=np.float64);self.fit=tuple(fit)
        if self.bank.shape!=(6,) or not np.isfinite(self.bank).all() or np.any(self.bank<0) or np.any(self.bank>.02):raise ValueError('Six bounded source-derived amplitude entries')
        self.parameters={};self.visits=Counter();self.coverage=Counter();self.calls=0;self.packets=0;self.max_inverse_error=0.;self.max_power_bound_error=0.
    def parameter(self,epoch,rx,day):
        if epoch not in range(1,21) or rx not in self.fit or day not in (1,2,3):raise ValueError('Registered E20 source session only')
        key=(epoch,rx,day)
        if key not in self.parameters:
            offset=int(np.random.default_rng(np.random.SeedSequence([self.seed,0,rx,day])).integers(6));index=(offset+epoch-1)%6
            angle=float(np.random.default_rng(np.random.SeedSequence([self.seed,epoch,rx,day,2])).uniform(-np.pi,np.pi))
            self.parameters[key]=(float(self.bank[index]*np.cos(angle)),float(self.bank[index]*np.sin(angle)),index)
        return self.parameters[key]
    def apply(self,x,meta,epoch):
        if x.shape!=(128,2,256):raise ValueError('Only original L128')
        vals=[meta[k].tolist() for k in KEYS]
        if any(len(v)!=128 for v in vals):raise ValueError('One physical metadata record per source packet')
        ids=list(zip(*vals))
        if any(p[0]!=1 or p[1] not in self.fit or p[2] not in (1,2,3) or any(int(v)!=v or v<0 for v in p) for p in ids):raise ValueError('Legal original fitL metadata required')
        params=[self.parameter(epoch,int(p[1]),int(p[2])) for p in ids]
        b=torch.tensor([v[:2] for v in params],dtype=x.dtype,device=x.device);y=mix(x,b)
        error=float((x-inverse(y,b)).abs().max());limit=1e-5*max(1.,float(x.abs().max()))
        if error>limit:raise ValueError('Bounded IQ response lost complex information')
        energy=x.square().sum(1);new=y.square().sum(1);r=b.square().sum(1).sqrt()[:,None]
        lower=(1-r).square()/(1+r.square());upper=(1+r).square()/(1+r.square())
        bound_error=float(torch.maximum((lower*energy-new).clamp_min(0),(new-upper*energy).clamp_min(0)).max())
        if bound_error>1e-5*max(1.,float(energy.max())):raise ValueError('RXMIX violated declared pointwise power bounds')
        for p,v in zip(ids,params):self.visits[p]+=1;self.coverage[(epoch,int(p[1]),int(p[2]),v[2])]+=1
        self.calls+=1;self.packets+=len(x);self.max_inverse_error=max(self.max_inverse_error,error);self.max_power_bound_error=max(self.max_power_bound_error,bound_error)
        return y,ids,[v[2] for v in params]
    def summary(self):
        return dict(calls=self.calls,packets=self.packets,physical_IDs=len(self.visits),sessions=len(self.parameters),max_b_abs=float(self.bank.max()),
            bank_entries=6,max_inverse_error=self.max_inverse_error,max_pointwise_power_bound_error=self.max_power_bound_error,
            TX_used_for_selection=False,global_RNG_advance=False,per_packet_normalization=False,boundary_change=False,added_noise=False,unknown_TX_inversion=False)
