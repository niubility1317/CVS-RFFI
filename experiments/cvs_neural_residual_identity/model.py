"""Learned convolutional residual features; no new loss or training strategy."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from experiments.cvs_adaptive_volterra_identity.model import AdaptiveVolterraCVS,adaptive_contract
from experiments.cvs_energy_identity.model import GlobalEnergyBlock
from experiments.cvs_equivariant_identity.model import ComplexConv

VARIANTS=('neural_residual_shallow','neural_residual_deep')

class SharedRadial(nn.Module):
    def __init__(self,channels):
        super().__init__()
        self.scale=nn.Parameter(torch.ones(channels))
        self.a=nn.Parameter(torch.zeros(channels));self.b=nn.Parameter(torch.zeros(channels))
    def forward(self,z):
        z=z*torch.rsqrt(z.square().sum(1).mean((1,2),keepdim=True)+1e-6)[:,None]
        z=z*self.scale[None,None,:,None]
        gate=2*torch.sigmoid(self.a[None,:,None]*torch.log1p(z.square().sum(1))+self.b[None,:,None])
        return z*gate[:,None]

class ComplexDepthwise(nn.Module):
    def __init__(self,channels,causal):
        super().__init__();self.channels=channels;self.causal=causal
        self.weight_real=nn.Parameter(torch.randn(channels,1,5)/math.sqrt(10))
        self.weight_imag=nn.Parameter(torch.randn(channels,1,5)/math.sqrt(10))
    def forward(self,z):
        # Each group holds one channel's real/imaginary pair.
        w=torch.stack([torch.cat([self.weight_real,-self.weight_imag],1),torch.cat([self.weight_imag,self.weight_real],1)],1).reshape(2*self.channels,2,5)
        x=z.permute(0,2,1,3).reshape(len(z),2*self.channels,z.shape[-1])
        x=F.pad(x,(4,0) if self.causal else (2,2))
        y=F.conv1d(x,w,groups=self.channels)
        return y.reshape(len(z),self.channels,2,-1).permute(0,2,1,3)

class LearnedResidual(nn.Module):
    def __init__(self,causal):
        super().__init__()
        self.expand=ComplexConv(32,64,1)
        self.first=SharedRadial(64)
        self.temporal=ComplexDepthwise(64,causal)
        self.second=SharedRadial(64)
        self.project=ComplexConv(64,32,1)
        with torch.no_grad():
            self.project.weight_real.zero_();self.project.weight_imag.zero_()
    def forward(self,z):
        delta=self.project(self.second(self.temporal(self.first(self.expand(z)))))
        return z+delta

class ResidualOutput(GlobalEnergyBlock):
    def __init__(self,block,depth,causal):
        super().__init__(block)
        self.neural=nn.Sequential(*(LearnedResidual(causal) for _ in range(depth)))
    def forward(self,z):return self.neural(super().forward(z))

def neural_contract(variant):
    if variant not in VARIANTS:raise ValueError('Unregistered neural residual architecture')
    depth=1 if variant==VARIANTS[0] else 2
    d=adaptive_contract('adaptive_volterra_lag4')
    d.update(mode=variant,identity_core='neural_residual_identity',new_trainable_parameters=18432*depth,
        neural_residual_active=True,neural_depth_per_path=depth,neural_paths=['time.2','behavior.2'],
        neural_channels=[32,64,32],neural_kernel=5,neural_time_padding='symmetric',neural_behavior_padding='causal_left',
        neural_rule='z+complex_pointwise_project(radial(depthwise_k5(radial(complex_pointwise_expand(z)))))',
        neural_exit_initialization='zero',neural_hidden_initialization='random complex fan-in',
        initial_function='exact own scratch adaptive_volterra_lag4; isolated new initialization RNG',
        neural_shared_gain='one RMS over complex channels and time per packet',
        neural_loss='unchanged single cross_entropy',new_training_strategy=False,
        feature_curvature_active=False,whole_affine_phase_invariant=False)
    return d

class NeuralResidualCVS(AdaptiveVolterraCVS):
    def __init__(self,variant):
        super().__init__('adaptive_volterra_lag4');self.neural_variant=variant
        depth=neural_contract(variant)['neural_depth_per_path']
        # New modules only; preserve the existing base initialization and RNG stream.
        with torch.random.fork_rng(devices=[]):
            for path in ('time','behavior'):
                blocks=getattr(self.core,path)
                blocks[2]=ResidualOutput(blocks[2],depth,path=='behavior')
    def residual_blocks(self):
        return [(path+'.2.neural.'+str(i),block) for path in ('time','behavior') for i,block in enumerate(getattr(self.core,path)[2].neural)]
    def residual_parameters(self):return [p for _,b in self.residual_blocks() for p in b.parameters()]
    def contract(self):
        d=neural_contract(self.neural_variant)
        d['neural_residual_active']=all(isinstance(getattr(self.core,path)[2],ResidualOutput) for path in ('time','behavior'))
        d['new_trainable_parameters']=sum(p.numel() for p in self.residual_parameters())
        return d
    @torch.no_grad()
    def neural_diagnostics(self,x):
        records=[];hooks=[]
        def capture(name):
            def hook(module,inputs,output):
                before=inputs[0];change=(output-before).square().sum((1,2,3)).sqrt()
                denom=before.square().sum((1,2,3)).sqrt().clamp_min(1e-12)
                records.append(dict(block=name,packets=len(output),relative_output_change_mean=float((change/denom).mean()),
                    projection_norm=float(torch.stack([module.project.weight_real.norm(),module.project.weight_imag.norm()]).norm())))
            return hook
        try:
            for name,b in self.residual_blocks():hooks.append(b.register_forward_hook(capture(name)))
            self.features(x)
        finally:
            for h in hooks:h.remove()
        return dict(active=True,records=records,scope='last source batch; no additional loss or updates')
    @torch.no_grad()
    def diagnostics(self,x):return dict(super().diagnostics(x),neural_residual=self.neural_diagnostics(x))

def build(variant):return NeuralResidualCVS(variant)
