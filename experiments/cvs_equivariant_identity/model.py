"""All identity paths respect global complex phase; received RF behavior only."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from experiments.cvs_residual_identity.model import build as residual_build

VARIANTS=('equivariant_memory',)


def equivariant_contract(variant):
    if variant not in VARIANTS:raise ValueError('Unregistered equivariant variant')
    return dict(mode=variant,input='received_equalized1_unit_rms_IQ',sample_rate_hz=25000000,
        whole_identity_global_phase_invariant_real_arithmetic=True,global_phase_equivariant_until_readout=True,
        removes_linear_phase=False,arbitrary_channel_rx_invariant=False,hardware_parameter_estimation=False,
        time_complex_channels=[24,24,32,32],time_kernels=[5,5,3],time_strides=[2,2,1],
        behavior_orders=[1,3,5],behavior_delays=[0,1,2,3],radial_clip=4.,order_scale_base=2.,
        behavior_complex_channels=[12,24,32,32],behavior_kernels=[7,5,3],behavior_dilations=[1,2,4],behavior_causal_filters=True,
        per_packet_complex_rms_normalization=True,complex_bias=False,radial_gate='2sigmoid(a*log1p(power)+b)',
        invariant_readout='logpower mean/std,lag1/2 normalized complex correlation,4 positional logpower means',
        readout_channels_per_complex_channel=10,embedding_dimension=160,classifier_scale=30.,
        original_real_frequency_path_retained=True,raw_unconstrained_identity_bypass=False,
        interpretation='Shared complex received-response filters with invariant observations; not recovered TX PA/IQ coefficients')


def rotate_pair(z,phase):
    r,i=z.unbind(1);c=phase.cos();s=phase.sin()
    while c.ndim<r.ndim:c=c.unsqueeze(-1);s=s.unsqueeze(-1)
    return torch.stack([r*c-i*s,r*s+i*c],1)


class ComplexConv(nn.Module):
    def __init__(self,cin,cout,k,stride=1,dilation=1,causal=False):
        super().__init__();self.cin=cin;self.cout=cout;self.k=k;self.stride=stride;self.dilation=dilation;self.causal=causal
        self.weight_real=nn.Parameter(torch.randn(cout,cin,k)/math.sqrt(2*cin*k))
        self.weight_imag=nn.Parameter(torch.randn(cout,cin,k)/math.sqrt(2*cin*k))

    def forward(self,z):
        r,i=z.unbind(1)
        w=torch.cat([torch.cat([self.weight_real,-self.weight_imag],1),torch.cat([self.weight_imag,self.weight_real],1)],0)
        inputs=torch.cat([r,i],1)
        if self.causal:inputs=F.pad(inputs,(self.dilation*(self.k-1),0))
        y=F.conv1d(inputs,w,bias=None,stride=self.stride,padding=0 if self.causal else self.dilation*(self.k//2),dilation=self.dilation)
        return torch.stack(y.split(self.cout,1),1)


class RadialBlock(nn.Module):
    def __init__(self,cin,cout,k,stride=1,dilation=1,causal=False):
        super().__init__();self.conv=ComplexConv(cin,cout,k,stride,dilation,causal)
        self.scale=nn.Parameter(torch.ones(cout));self.gate_a=nn.Parameter(torch.zeros(cout));self.gate_b=nn.Parameter(torch.zeros(cout))
        self.epsilon=1e-6

    def forward(self,z):
        z=self.conv(z);power=z.square().sum(1)
        # One isotropic scale per packet/channel; no centering or RX/query state.
        z=z*torch.rsqrt(power.mean(-1,keepdim=True)+self.epsilon)[:,None]*self.scale[None,None,:,None]
        power=z.square().sum(1)
        gate=2*torch.sigmoid(self.gate_a[None,:,None]*torch.log1p(power)+self.gate_b[None,:,None])
        return z*gate[:,None]


def behavior_basis(x):
    power=x.square().sum(1,keepdim=True)
    z=x*(4*torch.rsqrt(power+1e-6)).clamp(max=1.)
    terms=[]
    for delay in range(4):
        v=z if delay==0 else F.pad(z[...,:-delay],(delay,0))
        p=v.square().sum(1)
        for order in (1,3,5):terms.append(v*(p/4).pow((order-1)//2)[:,None])
    return torch.stack(terms,2)


class InvariantReadout(nn.Module):
    def forward(self,z):
        r,i=z.unbind(1);power=r.square()+i.square();logpower=torch.log1p(power)
        parts=[logpower.mean(-1),torch.sqrt(logpower.var(-1,unbiased=False)+1e-6)]
        for lag in (1,2):
            ar,ai=r[...,lag:],i[...,lag:];br,bi=r[...,:-lag],i[...,:-lag]
            denominator=torch.sqrt((ar.square()+ai.square()).mean(-1)*(br.square()+bi.square()).mean(-1)+1e-6)
            parts += [(ar*br+ai*bi).mean(-1)/denominator,(ai*br-ar*bi).mean(-1)/denominator]
        parts.append(F.adaptive_avg_pool1d(logpower,4).flatten(1))
        return torch.cat(parts,1)


class EquivariantCVS(nn.Module):
    def __init__(self,variant):
        super().__init__();self.variant=variant;equivariant_contract(variant)
        # New scratch build, retaining the actual native frequency/fusion/head.
        b=residual_build('residual_fusion').id_backbone;self.id_backbone=b
        if b.sinc.out_channels!=24 or b.freq_feature_source!='raw_fft' or b.use_dac_path or b.nmfdu_gate is not None:
            raise ValueError('Native retained path differs from registered construction')
        for name in ('hf','time_fuse','time_down','t1','t2','t3','t_pool','pa_lift','pa_gate','pa_b1','pa_b2','pa_b3','pa_pool'):
            setattr(b,name,nn.Identity())
        b.t_proj=nn.Linear(320,160)
        pa_dropout=b.pa_proj[-1].p
        b.pa_proj=nn.Sequential(nn.Linear(320,160),nn.ReLU(),nn.Dropout(pa_dropout))
        self.time=nn.Sequential(RadialBlock(24,24,5,2),RadialBlock(24,32,5,2),RadialBlock(32,32,3))
        self.behavior=nn.Sequential(RadialBlock(12,24,7,2,causal=True),RadialBlock(24,32,5,2,2,causal=True),RadialBlock(32,32,3,1,4,causal=True))
        self.readout=InvariantReadout()

    def contract(self):
        d=equivariant_contract(self.variant)
        d.update(time_complex_channels=[self.time[0].conv.cin,*[m.conv.cout for m in self.time]],
            time_kernels=[m.conv.k for m in self.time],time_strides=[m.conv.stride for m in self.time],
            behavior_complex_channels=[self.behavior[0].conv.cin,*[m.conv.cout for m in self.behavior]],
            behavior_kernels=[m.conv.k for m in self.behavior],behavior_dilations=[m.conv.dilation for m in self.behavior],
            behavior_causal_filters=all(m.conv.causal for m in self.behavior),
            original_real_frequency_path_retained=self.id_backbone.freq_feature_source=='raw_fft',
            classifier_scale=self.id_backbone.cls_head.scale)
        return d

    def features(self,x):
        if x.ndim!=3 or x.shape[1:]!=(2,256):raise ValueError('Expected IQ[B,2,256]')
        b=self.id_backbone;sinc=b._sinc_on_iq(x).reshape(len(x),2,24,256)
        time=b.t_proj(self.readout(self.time(sinc)))
        physical=b.pa_proj(self.readout(self.behavior(behavior_basis(x))))
        spectral,rho,dac_stats,pa_stats=b._mirror_compressed_features(x,sinc_iq=None)
        f=b.f_pool(b.f3(b.f2(b.f1(b.freq_gate(spectral))))).squeeze(-1)
        frequency=b.f_proj(f)
        if b.use_stats_path and b.use_freq_stats and b.freq_stats_proj is not None:frequency=frequency+b.freq_stats_proj(dac_stats)
        if b.use_stats_path and b.pa_stats_proj is not None:physical=physical+.25*b.pa_stats_proj(pa_stats)
        parts=[time,frequency]
        if rho is not None:parts.append(rho if b.use_stats_path else torch.zeros_like(rho))
        base=b.fuse(torch.cat(parts,1))
        return b.cls_head.components(base,physical)[2]

    def classify_features(self,features):return self.id_backbone.cls_head.classify(features)
    def forward(self,x):return self.classify_features(self.features(x))

    @torch.no_grad()
    def diagnostics(self,x):
        return dict(time_weight_norm=float(torch.stack([m.conv.weight_real.norm()+m.conv.weight_imag.norm() for m in self.time]).norm()),
            behavior_weight_norm=float(torch.stack([m.conv.weight_real.norm()+m.conv.weight_imag.norm() for m in self.behavior]).norm()),
            radial_gate_a_abs_mean=float(torch.cat([m.gate_a for m in list(self.time)+list(self.behavior)]).abs().mean()),
            time_gradient_norm=float(torch.stack([p.grad.norm() for p in self.time.parameters() if p.grad is not None]).norm()) if any(p.grad is not None for p in self.time.parameters()) else None,
            behavior_gradient_norm=float(torch.stack([p.grad.norm() for p in self.behavior.parameters() if p.grad is not None]).norm()) if any(p.grad is not None for p in self.behavior.parameters()) else None,
            scope='Measured weights and last source batch gradient;not hardware parameter recovery')


def build(variant):return EquivariantCVS(variant)


@torch.no_grad()
def frozen_synthetic_diagnostics(model):
    from experiments.cvs_equivariant_identity.physics import controlled_diagnostics
    return controlled_diagnostics(model)
