"""Six bounded, zero-initialized delayed phase-curvature residuals.

These operate on received features after each complex FIR, before the shared
normalization. They are classification operators, not identified TX kernels.
"""
import torch
from torch import nn
from experiments.cvs_adaptive_volterra_identity.model import AdaptiveVolterraCVS, adaptive_contract
from experiments.cvs_equivariant_identity.model import ComplexConv
from experiments.cvs_coupled_identity.model import delay
from experiments.cvs_volterra_identity.model import multiply

VARIANTS=('phase_curvature_lag14','phase_curvature_lag24')
PATHS=tuple(path+'.'+str(i) for path in ('time','behavior') for i in range(3))
CLIP_RADIUS=2.

def lag_pair(variant):
    if variant not in VARIANTS:raise ValueError('Unregistered feature phase-curvature candidate')
    return (1,4) if variant==VARIANTS[0] else (2,4)

def clipped_complex(z):
    if z.ndim!=4 or z.shape[1]!=2:raise ValueError('Expected complex[B,2,C,T]')
    power=z.square().sum(1,keepdim=True)
    return z*(CLIP_RADIUS*torch.rsqrt(power+1e-6)).clamp(max=1.)

def curvature_delta(z,a,b):
    if (a,b) not in ((1,4),(2,4)):raise ValueError('Unregistered fixed mixed delays')
    v=clipped_complex(z)
    conjugate=delay(v,a+b)*v.new_tensor([1.,-1.]).view(1,2,1,1)
    delayed=multiply(multiply(delay(v,a),delay(v,b)),conjugate)
    instantaneous=v*v.square().sum(1,keepdim=True)
    valid=(torch.arange(z.shape[-1],device=z.device)>=a+b).view(1,1,1,-1)
    return (delayed-instantaneous)*valid/4

def curvature_transform(z,raw,a,b):
    if raw.shape!=():raise ValueError('Exactly one global scalar per FIR output')
    return z+raw.tanh()*curvature_delta(z,a,b)

class PhaseCurvatureConv(ComplexConv):
    def __init__(self,conv,a,b):
        # Reuse modules/parameters of this newly initialized model only.
        # No new random draw, checkpoint, wrapper state-key remapping or bias.
        nn.Module.__init__(self)
        for name in ('cin','cout','k','stride','dilation','causal'):
            setattr(self,name,getattr(conv,name))
        self.weight_real=conv.weight_real;self.weight_imag=conv.weight_imag
        self.memory_raw=nn.Parameter(conv.weight_real.new_zeros(()))
        self.delay_a=a;self.delay_b=b
    def forward(self,z):
        return curvature_transform(ComplexConv.forward(self,z),self.memory_raw,self.delay_a,self.delay_b)

def curvature_contract(variant):
    a,b=lag_pair(variant);contract=adaptive_contract('adaptive_volterra_lag4')
    contract.update(mode=variant,identity_core='phase_curvature_identity',feature_curvature_active=True,
        feature_curvature_paths=list(PATHS),feature_curvature_delays=[a,b,a+b],
        feature_curvature_rule='y+tanh(beta)*(v[t-a]*v[t-b]*conj(v[t-a-b])-v[t]*abs(v[t])^2)/4;prefix masked',
        feature_curvature_location='each of six complex FIR outputs, before existing shared RMS and radial gate',
        feature_curvature_clip_radius=CLIP_RADIUS,feature_curvature_delta_abs_bound=4.,
        feature_curvature_mask='t>=a+b;no wraparound or future access',
        feature_curvature_parameter_count=6,feature_curvature_initial_raw=[0.]*6,
        actual_feature_curvature_delays=[[a,b,a+b] for _ in PATHS],
        actual_feature_curvature_parameter_shapes=[[] for _ in PATHS],
        feature_curvature_parameter_scope='six global identity parameters learned by source CE only',
        feature_output_grid_strides_original_samples=[2,4,4,2,4,4],
        affine_phase_covariant_feature_operator=True,constant_amplitude_affine_phase_delta_zero=True,
        initial_function='exact own-scratch adaptive_volterra_lag4 under matching RNG',
        new_trainable_parameters=6,total_global_residual_gates=8,
        full_network_causal=False,hardware_parameter_recovery=False,whole_affine_phase_invariant=False)
    return contract

class PhaseCurvatureCVS(AdaptiveVolterraCVS):
    def __init__(self,variant):
        a,b=lag_pair(variant);super().__init__('adaptive_volterra_lag4')
        self.curvature_variant=variant
        for path in ('time','behavior'):
            for block in getattr(self.core,path):block.conv=PhaseCurvatureConv(block.conv,a,b)
    def memory_parameters(self):
        return [block.conv.memory_raw for path in ('time','behavior') for block in getattr(self.core,path)]
    def contract(self):
        contract=curvature_contract(self.curvature_variant)
        convs=[block.conv for path in ('time','behavior') for block in getattr(self.core,path)]
        contract.update(feature_curvature_active=all(isinstance(conv,PhaseCurvatureConv) for conv in convs),
            actual_feature_curvature_delays=[[conv.delay_a,conv.delay_b,conv.delay_a+conv.delay_b] for conv in convs],
            actual_feature_curvature_parameter_shapes=[list(conv.memory_raw.shape) for conv in convs])
        return contract
    @torch.no_grad()
    def curvature_diagnostics(self,x):
        records=[];hooks=[]
        def capture(name):
            def hook(conv,inputs,output):
                linear=ComplexConv.forward(conv,inputs[0])
                expected=curvature_transform(linear,conv.memory_raw,conv.delay_a,conv.delay_b)
                delta=curvature_delta(linear,conv.delay_a,conv.delay_b)
                change=(output-linear).square().sum((1,2,3)).sqrt()
                denominator=linear.square().sum((1,2,3)).sqrt().clamp_min(1e-12)
                prefix=output[...,:conv.delay_a+conv.delay_b]-linear[...,:conv.delay_a+conv.delay_b]
                records.append(dict(block=name,packets=len(output),complex_channels=conv.cout,grid_length=output.shape[-1],
                    actual_delays=[conv.delay_a,conv.delay_b,conv.delay_a+conv.delay_b],
                    raw_parameter=float(conv.memory_raw),coefficient=float(conv.memory_raw.tanh()),
                    input_formula_max_abs_error=float((output-expected).abs().max()),
                    prefix_correction_max_abs_error=float(prefix.abs().max()) if prefix.numel() else 0.,
                    delta_complex_abs_max=float(delta.square().sum(1).sqrt().max()),
                    actual_relative_output_change_mean=float((change/denominator).mean()),
                    eligible_grid_positions=max(0,output.shape[-1]-conv.delay_a-conv.delay_b)))
            return hook
        try:
            for path in ('time','behavior'):
                for i,block in enumerate(getattr(self.core,path)):
                    hooks.append(block.conv.register_forward_hook(capture(path+'.'+str(i))))
            self.features(x)
        finally:
            for hook in hooks:hook.remove()
        return dict(active=True,records=records,
            scope='Actual six FIR+curvature outputs before shared normalization;same packet;not TX hardware recovery')
    @torch.no_grad()
    def diagnostics(self,x):
        return dict(super().diagnostics(x),feature_curvature=self.curvature_diagnostics(x))

def build(variant):return PhaseCurvatureCVS(variant)
