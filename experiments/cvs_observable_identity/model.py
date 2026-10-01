"""Whole-identity physical observables, with explicit nuisance/attribution limits."""
import math
import torch
from torch import nn
import torch.nn.functional as F

VARIANTS=('observable_phase','observable_affine')
LAGS=(1,2,5,20)
AFFINE_COLUMNS=(0,1,2,3,4,13,14,15,16,17,18,19,20)


def observable_contract(variant):
    if variant not in VARIANTS:raise ValueError('Unregistered observable variant')
    affine=variant=='observable_affine'
    return dict(physics_mode='amplitude_and_phase_curvature' if affine else 'amplitude_increment_and_curvature',
                lags=list(LAGS),epsilon=1e-6,observation_channels=13 if affine else 21,
                amplitude_channels=5,increment_channels=0 if affine else 8,closure_channels=8,
                global_phase_invariant=True,affine_phase_invariant=affine,raw_iq_bypass=False,
                repetition_window=[80,160],repetition_period_samples=20,repetition_cycles=4,
                temporal_samples=256,spectral_signal='real_physical_observable_time_variations',
                sample_rate_hz=25000000,input='received_equalized1_unit_rms_IQ',
                parameter_interpretation='identity_feature_weights_not_identified_TX_hardware')


def delay(x,d):
    if d>=x.shape[-1]:return torch.zeros_like(x)
    return F.pad(x[...,:-d],(d,0)) if d else x


class PhysicalObservables(nn.Module):
    """No fit/state update. Constant phase quotiented before ALL learned paths.

    A=log(1+|x|²), causal ΔA, regularized phase increments and second-order
    phase closures. For nonzero samples A plus lag1 increment preserves x up
    to constant phase. Closure-only additionally removes affine phase; zeros
    and disconnected segments limit such reconstruction statements.
    """
    def __init__(self,variant):
        super().__init__()
        if variant not in VARIANTS:raise ValueError('Unknown physical observables')
        self.variant=variant;self.epsilon=1e-6;self.lags=LAGS

    def forward(self,iq):
        if iq.ndim!=3 or iq.shape[1]!=2 or iq.shape[-1]!=256:
            raise ValueError('ExpectedreceivedIQ[B,2,256]')
        r,i=iq[:,0],iq[:,1];p=r.square()+i.square();a=torch.log1p(p)
        amplitude=[a];increments=[];closures=[]
        for d in self.lags:
            dr,di=delay(r,d),delay(i,d)
            denom=torch.sqrt((p+self.epsilon)*(delay(p,d)+self.epsilon))
            ur=(r*dr+i*di)/denom;ui=(i*dr-r*di)/denom
            # Missing history is zero, not a wrap around to future samples.
            du_r,du_i=delay(ur,d),delay(ui,d)
            cr=ur*du_r+ui*du_i;ci=ui*du_r-ur*du_i
            valid=(torch.arange(iq.shape[-1],device=iq.device)>=d).to(iq.dtype)
            amplitude.append((a-delay(a,d))*valid)
            increments += [ur,ui];closures += [cr,ci]
        return torch.stack(amplitude+([] if self.variant=='observable_affine' else increments)+closures,dim=1)


class TimeResidual(nn.Module):
    def __init__(self,dilation):
        super().__init__()
        self.path=nn.Sequential(nn.Conv1d(48,48,5,padding=2*dilation,dilation=dilation,bias=False),
                                nn.GroupNorm(8,48),nn.SiLU(),nn.Conv1d(48,48,5,padding=2,bias=False),nn.GroupNorm(8,48))
    def forward(self,x):return F.avg_pool1d(F.silu(x+self.path(x)),2)


def moments(x):
    return torch.cat([x.mean(-1),(x.var(-1,unbiased=False)+1e-6).sqrt()],dim=1)


class ObservableCVS(nn.Module):
    """CVS time/repetition/spectral-variation fusion; only identity CE trains it.

    Whole model is global-phase invariant. Only affine variant is invariant
    to a multiplicative exp(j(θ+ωn)) on provided IQ. No universal channel/RX
    invariance or unique TX/RX hardware separation is asserted.
    """
    def __init__(self,variant):
        super().__init__();self.observables=PhysicalObservables(variant);self.variant=variant
        self.repetition_window=(80,160);self.period=20;self.embedding_dim=160;self.scale=30.0
        # Build full21-channel scratch bank in both modes; prune afterwards.
        # Common first-layer columns and every later tensor match by pairedseed.
        self.time_stem=nn.Sequential(nn.Conv1d(21,48,7,padding=3,bias=False),nn.GroupNorm(8,48),nn.SiLU())
        self.time_blocks=nn.Sequential(TimeResidual(1),TimeResidual(3))
        self.time_projection=nn.Linear(96,160)
        self.repeat_stem=nn.Sequential(nn.Conv2d(21,32,(3,5),padding=(1,2),bias=False),nn.GroupNorm(8,32),nn.SiLU())
        self.repeat_block=nn.Sequential(nn.Conv2d(32,48,3,padding=1,bias=False),nn.GroupNorm(8,48),nn.SiLU())
        self.repeat_projection=nn.Linear(480,160)
        self.spectral_stem=nn.Sequential(nn.Conv1d(21,32,5,padding=2,bias=False),nn.GroupNorm(8,32),nn.SiLU())
        self.spectral_block=nn.Sequential(nn.Conv1d(32,48,3,padding=1,bias=False),nn.GroupNorm(8,48),nn.SiLU())
        self.spectral_projection=nn.Linear(96,160)
        self.norms=nn.ModuleList([nn.LayerNorm(160) for _ in range(3)])
        self.gains=nn.Parameter(torch.full((2,160),math.atanh(.25)))
        self.class_weight=nn.Parameter(torch.empty(6,160));nn.init.xavier_uniform_(self.class_weight)
        if variant=='observable_affine':
            for stem in (self.time_stem,self.repeat_stem,self.spectral_stem):
                conv=stem[0]
                conv.weight=nn.Parameter(conv.weight[:,AFFINE_COLUMNS].detach().clone())
                conv.in_channels=len(AFFINE_COLUMNS)

    def contract(self):
        d=observable_contract(self.variant)
        d.update(observation_channels=self.time_stem[0].in_channels,
                 repetition_window=list(self.repetition_window),repetition_period_samples=self.period,
                 epsilon=self.observables.epsilon,lags=list(self.observables.lags))
        if self.repeat_stem[0].in_channels!=d['observation_channels'] or self.spectral_stem[0].in_channels!=d['observation_channels']:
            raise ValueError('Branchobservablechannel mismatch')
        return d

    def features(self,iq):
        obs=self.observables(iq)
        time=self.time_projection(moments(self.time_blocks(self.time_stem(obs))))
        lo,hi=self.repetition_window
        grid=obs[...,lo:hi].reshape(len(iq),obs.shape[1],(hi-lo)//self.period,self.period)
        repeat=self.repeat_block(self.repeat_stem(grid)).mean(2)
        repeat=F.avg_pool1d(repeat,2).flatten(1)
        repeat=self.repeat_projection(repeat)
        spectrum=torch.fft.rfft(obs,dim=-1,norm='ortho')
        spectrum=torch.log1p(spectrum.real.square()+spectrum.imag.square())
        spectral=self.spectral_projection(moments(self.spectral_block(self.spectral_stem(spectrum))))
        time,repeat,spectral=[norm(x) for norm,x in zip(self.norms,(time,repeat,spectral))]
        return time+self.gains[0].tanh()*repeat+self.gains[1].tanh()*spectral

    def diagnostics(self):
        return dict(repetition_gain_abs_mean=float(self.gains[0].detach().tanh().abs().mean()),
                    spectral_gain_abs_mean=float(self.gains[1].detach().tanh().abs().mean()),
                    fusion_gain_gradient_norm=float(self.gains.grad.norm()) if self.gains.grad is not None else None)

    def forward(self,iq):
        return self.scale*F.linear(F.normalize(self.features(iq),dim=1,eps=1e-4),
                                   F.normalize(self.class_weight,dim=1,eps=1e-4))


def build(variant):
    if variant not in VARIANTS:raise ValueError('Unregisteredobservablevariant')
    return ObservableCVS(variant)


@torch.no_grad()
def frozen_synthetic_diagnostics(model):
    """Hand-set physical interventions after E200; never a training batch.

    No actual TX identity is attached to these signals. Distances measure
    sensitivity only, and explicitly include receiver/channel counterexamples.
    """
    model.eval();parameter=next(model.parameters());device=parameter.device;dtype=parameter.dtype
    n=torch.arange(256,device=device,dtype=dtype)
    amplitude=1+.3*torch.sin(.12*n);phase=.2*n+.03*torch.sin(.07*n)
    base=torch.stack([amplitude*phase.cos(),amplitude*phase.sin()],dim=0)[None]
    normalize=lambda x:x/(x.square().sum(1).mean(-1,keepdim=True)+1e-12).sqrt().unsqueeze(1)
    base=normalize(base)
    def rotate(x,p):
        r,i=x[:,0],x[:,1]
        return torch.stack([r*p.cos()-i*p.sin(),r*p.sin()+i*p.cos()],dim=1)
    constant=rotate(base,n.new_tensor(.731))
    affine=rotate(base,.4+.031*n)
    power=base.square().sum(1,keepdim=True)
    amam=normalize(base/(1+.12*power))
    ampm=rotate(base,.17*power[:,0])
    image=base.clone();image[:,0]*=1.08;image[:,1]*=.92;image=normalize(image)
    multipath=normalize(base+.2*F.pad(base[...,:-2],(2,0)))
    with torch.no_grad():
        baseline=model(base);emb=F.normalize(model.features(base),dim=1,eps=1e-4)
        distances={name:float((F.normalize(model.features(x),dim=1,eps=1e-4)-emb).norm())
                   for name,x in [('handset_am_am',amam),('handset_am_pm',ampm),('rx_iq_counterexample',image),('multipath_counterexample',multipath)]}
        constant_error=float((model(constant)-baseline).abs().max())
        affine_error=float((model(affine)-baseline).abs().max())
    return dict(phase_logit_max_abs_error=constant_error,affine_logit_max_abs_error=affine_error,
                affine_invariance_claimed=model.variant=='observable_affine',unit_embedding_distances=distances,
                synthetic_only=True,training_augmentation=False,target_access=False,
                physical_parameters=dict(positive_amp_compression=.12,amp_phase_beta=.17,rx_iq_scale=[1.08,.92],
                                         channel_delay_samples=2,channel_tap=.2,constant_phase_radians=.731,affine_phase_slope_radians_per_sample=.031),
                input_rms_normalized=True,claim='Frozen model transform verification and sensitivity only;not TX parameter identification or synthetic identity accuracy.')
