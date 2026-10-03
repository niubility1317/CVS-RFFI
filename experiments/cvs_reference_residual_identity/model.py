"""Differentiable reference-response solve inside the identity network.

The solve estimates an apparent periodic received response, not TX-free CSI.
The original raw CVS path is retained because linear TX/RX effects are not
identifiable from one known periodic excitation and an arbitrary channel.
"""
import math
import numpy as np
import torch
from torch import nn
from experiments.cvs_reference_identity.model import ExcitationResponse
from experiments.cvs_residual_identity.model import build as build_core

VARIANTS=('reference_residual_scalar','reference_residual_inverse')
LAGS=tuple(range(-4,5))
RIDGE_MIN=1e-4
RIDGE_SPAN=.1
RIDGE_INITIAL=.001
INVERSE_RIDGE=.01

def reference_residual_contract(variant):
    if variant not in VARIANTS:raise ValueError('Unregistered reference residual architecture')
    return dict(mode=variant,base_variant='residual_fusion',raw_path_retained=True,source_checkpoint_inheritance=False,
        window=[80,160],period=20,cycles=4,response_lags=list(LAGS),response_rank=9,
        response_mean='four CFO-corrected cycles; fixed public fractional-delay L-STF bank',
        learned_ridge_parameters=9,ridge_min=RIDGE_MIN,ridge_span=RIDGE_SPAN,ridge_initial=RIDGE_INITIAL,
        inverse_ridge=INVERSE_RIDGE,response_solve='(A^HA+diag(lambda)) h=A^H mean_fft; complex differentiable9x9',
        response_scope='apparent periodic response; lag-4 aliases delay16 on20period; not physical CIR',
        residual='cycle_fft-reference_fft*(D@h)',
        residual_canonicalizer='per-bin conjugate response' if variant==VARIANTS[1] else 'common conjugate scalar response',
        residual_shape=[2,80],soft_radius=2.,physical_operator_precision='float64/complex128; residual features cast to backbone input dtype',
        trainable_parameter_precision='float32',new_training_strategy=False,new_losses=[],
        initial_function='own scratch residual_fusion; zero exit projection; original RNG and dropout preserved',
        whole_model_channel_invariant=False,hardware_parameter_recovery=False,reference_equalization_itself_novel=False)

class ReferenceResidual(nn.Module):
    def __init__(self,variant):
        super().__init__();self.variant=variant;reference_residual_contract(variant)
        self.reference=ExcitationResponse('reference_response')
        bins=torch.arange(20,dtype=torch.float64)[:,None];lags=torch.tensor(LAGS,dtype=torch.float64)[None]
        phase=-2*torch.pi*bins*lags/20
        self.register_buffer('delay_real',phase.cos(),persistent=False);self.register_buffer('delay_imag',phase.sin(),persistent=False)
        p=(RIDGE_INITIAL-RIDGE_MIN)/RIDGE_SPAN
        self.ridge_raw=nn.Parameter(torch.full((9,),math.log(p/(1-p))))

    def ridge(self):return RIDGE_MIN+RIDGE_SPAN*self.ridge_raw.sigmoid()
    def contract(self):return reference_residual_contract(self.variant)

    def components(self,x):
        if x.ndim!=3 or x.shape[1:]!=(2,256):raise ValueError('Expected[B,2,256] received IQ')
        output_dtype=x.dtype
        # The small ill-conditioned response solve and spectral extrapolation
        # amplify FP32 roundoff. Keep this analytic branch in FP64 explicitly.
        x=x.to(torch.float64)
        z=torch.complex(x[:,0],x[:,1]);w=z[:,80:160]
        cross=(w[:,20:]*w[:,:-20].conj()).sum(-1)
        omega=torch.atan2(cross.imag,cross.real+1e-12)/20
        n=torch.arange(80,device=x.device,dtype=x.dtype)
        w=w*torch.complex(torch.cos(-omega[:,None]*n),torch.sin(-omega[:,None]*n))
        # A current-packet scale only; no fitted/persistent normalization state.
        w=w/(w.abs().square().mean(-1)+1e-12).sqrt()[:,None]
        cycles=w.reshape(-1,4,20);mean=cycles.mean(1)
        bank=torch.complex(self.reference.bank_real.to(x.dtype),self.reference.bank_imag.to(x.dtype))
        corr=mean@bank.conj().T/20;chosen=corr.abs().square().argmax(-1)
        s=torch.fft.fft(bank[chosen],dim=-1)/20
        y=torch.fft.fft(cycles,dim=-1)/20;m=y.mean(1)
        d=torch.complex(self.delay_real.to(x.dtype),self.delay_imag.to(x.dtype))
        a=s[:,:,None]*d[None]
        gram=a.conj().transpose(-1,-2)@a
        rhs=(a.conj().transpose(-1,-2)@m[:,:,None])
        matrix=gram+torch.diag_embed(self.ridge().to(x.dtype)).to(gram.dtype)
        h=torch.linalg.solve(matrix,rhs).squeeze(-1);response=h@d.T
        fitted=s*response;residual=y-fitted[:,None]
        power=response.abs().square().mean(-1,keepdim=True)
        gain=(s.conj()*m).sum(-1)/(s.abs().square().sum(-1)+1e-12)
        denominator=response if self.variant==VARIANTS[1] else gain[:,None].expand(-1,20)
        inverse=denominator.conj()/(denominator.abs().square()+INVERSE_RIDGE*power+1e-8)
        canonical=residual*inverse[:,None]
        time=torch.fft.ifft(canonical*20,dim=-1).reshape(-1,80)
        bounded=time/torch.sqrt(1+time.abs().square()/4)
        features=torch.stack((bounded.real,bounded.imag),1).to(output_dtype)
        return dict(features=features,cycle_spectra=y,reference_spectrum=s,response=response,coefficients=h,
            residual_spectra=residual,canonical_spectra=canonical,selected_delay=chosen.to(x.dtype)/16,
            fit_error=residual.abs().square().mean((1,2)),reference_quality=corr.abs().square().max(-1).values,
            inverse_abs_max=inverse.abs().max(-1).values,relative_cfo=omega*20/torch.pi)

    def forward(self,x):return self.components(x)['features']

class ReferenceResidualCVS(nn.Module):
    def __init__(self,variant):
        super().__init__();self.variant=variant
        # Existing original CVS architecture as a fixed implementation anchor,
        # never historical weights or a test-ranked checkpoint selection.
        self.core=build_core('residual_fusion')
        with torch.random.fork_rng(devices=[]):
            self.response=ReferenceResidual(variant)
            self.encoder=nn.Sequential(nn.Conv1d(2,16,5,padding=2),nn.GELU(),nn.Conv1d(16,32,3,padding=1),nn.GELU(),nn.Conv1d(32,32,3,padding=1),nn.GELU(),nn.AdaptiveAvgPool1d(4))
            self.projection=nn.Linear(128,160,bias=False);nn.init.zeros_(self.projection.weight)
    def features(self,x):return self.core.features(x)+torch.tanh(self.projection(self.encoder(self.response(x)).flatten(1)))
    def classify_features(self,z):return self.core.id_backbone.cls_head.classify(z)
    def forward(self,x):return self.classify_features(self.features(x))
    def contract(self):
        return dict(reference_residual_contract(self.variant),total_parameters=sum(p.numel() for p in self.parameters()),
            new_parameters=sum(p.numel() for module in (self.response,self.encoder,self.projection) for p in module.parameters()))
    @torch.no_grad()
    def diagnostics(self,x):
        c=self.response.components(x)
        return dict(ridge=self.response.ridge().tolist(),fit_error_mean=float(c['fit_error'].mean()),
            inverse_abs_max=float(c['inverse_abs_max'].max()),reference_quality_mean=float(c['reference_quality'].mean()),
            residual_feature_rms=float(c['features'].square().mean().sqrt()),projection_weight_norm=float(self.projection.weight.norm()),
            scope='Current source batch only; apparent response and residual, not recovered TX hardware')

def build(variant):return ReferenceResidualCVS(variant)
