"""Learned finite-memory correction with an unchanged raw identity path.

No missing-history padding enters the correction. Bounds below hold for a
fixed predicted kernel, not the whole input-conditioned nonlinear map.
"""
import torch
from torch import nn
from torch.nn import functional as F
from experiments.cvs_neural_residual_identity.model import NeuralResidualCVS,neural_contract
from experiments.cvs_channel_order_identity.model import _diagnostic_mode,_gradient_norm

VARIANTS=('validdual_static','validdual_dynamic')
TAPS=33
RHO=.5
BASE_PARAMETERS=220987
NEW_PARAMETERS={'validdual_static':226,'validdual_dynamic':8738}

def context(x):
    """Current-packet power plus normalized complex autocorrelation lags1..32."""
    r,i=x.unbind(1);power=r.square()+i.square();logpower=torch.log1p(power)
    values=[logpower.mean(-1),(logpower.var(-1,unbiased=False)+1e-6).sqrt()]
    for lag in range(1,33):
        ar,ai=r[:,lag:],i[:,lag:];br,bi=r[:,:-lag],i[:,:-lag]
        scale=((ar.square()+ai.square()).mean(-1)*(br.square()+bi.square()).mean(-1)+1e-12).sqrt()
        values.extend(((ar*br+ai*bi).mean(-1)/scale,(ai*br-ar*bi).mean(-1)/scale))
    return torch.stack(values,1)

class ValidHistoryFilter(nn.Module):
    def __init__(self,dynamic):
        super().__init__();self.dynamic=dynamic
        if dynamic:
            self.generator=nn.Sequential(nn.Linear(66,64),nn.GELU(),nn.Linear(64,66))
            nn.init.zeros_(self.generator[-1].weight);nn.init.zeros_(self.generator[-1].bias)
        else:self.coeff_raw=nn.Parameter(torch.zeros(2,TAPS))
        # All224 valid windows; first16 corrections smoothly enter the raw path.
        taper=torch.ones(224)
        taper[:16]=.5-.5*torch.cos(torch.pi*torch.arange(1,17)/16)
        self.register_buffer('taper',taper)

    def coefficients(self,x):
        raw=self.generator(context(x)).reshape(len(x),2,TAPS) if self.dynamic else self.coeff_raw[None].expand(len(x),-1,-1)
        return raw/torch.linalg.vector_norm(raw,dim=1).sum(-1).clamp_min(1.)[:,None,None]

    def kernel(self,coefficients):return coefficients

    def apply_kernel(self,x,coefficients):
        # Reverse each real observed window so tap k always means delay k.
        w=x.unfold(-1,TAPS,1).flip(-1);r,i=w.unbind(1);kr,ki=coefficients.unbind(1)
        d=torch.stack(((r*kr[:,None]-i*ki[:,None]).sum(-1),(r*ki[:,None]+i*kr[:,None]).sum(-1)),1)
        correction=F.pad(d*self.taper,(32,0))
        return x+RHO*correction

    def forward(self,x):
        if x.ndim!=3 or x.shape[1:]!=(2,256):raise ValueError('Expected IQ[B,2,256]')
        coefficients=self.coefficients(x)
        return self.apply_kernel(x,coefficients),coefficients

def dual_contract(variant):
    if variant not in VARIANTS:raise ValueError('Unregistered valid-history dual architecture')
    d=neural_contract('neural_residual_shallow')
    d.update(mode=variant,identity_core='validdual_identity',base_variant='neural_residual_shallow',
        base_trainable_parameters=BASE_PARAMETERS,new_trainable_parameters=NEW_PARAMETERS[variant],
        total_parameters=BASE_PARAMETERS+NEW_PARAMETERS[variant],total_trainable_parameters=BASE_PARAMETERS+NEW_PARAMETERS[variant],
        validdual_active=True,validdual_variant=variant,validdual_dynamic=variant==VARIANTS[1],
        validdual_taps=TAPS,validdual_rho=RHO,validdual_context_dimension=66,validdual_context_hidden=64,
        validdual_context='logpower mean/std and normalized complex autocorrelation lags1..32; current packet only',
        validdual_coefficients='33 complex taps in causal delay order; complex-modulus L1<=1',
        validdual_boundary='first32 samples unmodified; valid observed windows only; first16 valid corrections raised-cosine tapered',
        validdual_post_unit_rms=False,validdual_backbone_shared=True,validdual_original_input_bypass=True,
        validdual_fusion='z_raw+sigmoid(gate160)*(z_corrected-z_raw); one original classifier and one CE',
        validdual_gate_initialization='zero logits: coefficient0.5 per feature',validdual_filter_exit_initialization='zero',
        validdual_initial_function='own scratch shallow in eval; original dropout sampled independently in two training arms',
        validdual_fixed_kernel_delta_operator_bound=.5,validdual_dynamic_map_invertible_claim=False,
        validdual_cross_packet_state=False,validdual_inference_fit=False,validdual_loss='unchanged single cross_entropy',
        new_training_strategy=False,whole_model_channel_invariant=False,hardware_parameter_recovery=False,
        interpretation='discriminative correction plus raw fingerprint retention; not identified H inverse or TX/RX separation')
    return d

class ValidDualCVS(NeuralResidualCVS):
    def __init__(self,variant):
        dual_contract(variant);super().__init__('neural_residual_shallow');self.validdual_variant=variant
        with torch.random.fork_rng(devices=[]):
            self.validdual=ValidHistoryFilter(variant==VARIANTS[1]);self.fusion_logit=nn.Parameter(torch.zeros(160))

    def parts(self,x):
        corrected,coefficients=self.validdual(x)
        # Weight sharing, one combined call; two deterministic paths, no extra
        # samples or loss. Original backbone dropout is retained in training.
        raw,changed=super().features(torch.cat((x,corrected),0)).chunk(2,0)
        z=raw+torch.sigmoid(self.fusion_logit)*(changed-raw)
        return z,raw,changed,corrected,coefficients

    def features(self,x):return self.parts(x)[0]
    def validdual_parameters(self):return [*self.validdual.parameters(),self.fusion_logit]

    def contract(self):
        d=dual_contract(self.validdual_variant);new=self.validdual_parameters();ids={id(p) for p in new}
        d.update(new_trainable_parameters=sum(p.numel() for p in new if p.requires_grad),
            base_trainable_parameters=sum(p.numel() for p in self.parameters() if p.requires_grad and id(p) not in ids),
            total_parameters=sum(p.numel() for p in self.parameters()),total_trainable_parameters=sum(p.numel() for p in self.parameters() if p.requires_grad))
        if d['new_trainable_parameters']!=NEW_PARAMETERS[self.validdual_variant]:raise ValueError('Architecture parameter count differs')
        return d

    @torch.no_grad()
    def diagnostics(self,x):
        with _diagnostic_mode(self,x):
            z,raw,changed,gx,c=self.parts(x);g=torch.sigmoid(self.fusion_logit)
            delta=(gx-x).flatten(1).norm(dim=1)/x.flatten(1).norm(dim=1).clamp_min(1e-12)
            return dict(validdual=dict(active=True,packets=len(x),valid_windows=224,raw_prefix_unchanged=bool(torch.equal(gx[...,:32],x[...,:32])),
                input_change_mean=float(delta.mean()),input_change_max=float(delta.max()),kernel_l1_max=float(torch.linalg.vector_norm(c,dim=1).sum(-1).max()),
                coefficient_packet_variance=float(c.var(0,unbiased=False).mean()),
                feature_difference_mean=float((changed-raw).norm(dim=1).mean()),gate_mean=float(g.mean()),gate_min=float(g.min()),gate_max=float(g.max()),
                correction_gradient_norm=_gradient_norm(self.validdual.parameters()),fusion_gradient_norm=_gradient_norm([self.fusion_logit]),
                scope='one frozen current source-batch forward; no extra loss, fitting, or persistent state'))

def build(variant):return ValidDualCVS(variant)
