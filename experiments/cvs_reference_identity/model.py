"""Known-excitation relative response augments CVS identity, not TX estimation."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from experiments.cvs_residual_identity.model import build as build_residual
from experiments.cvs_rff_physics.known_excitation import template,TONES,REFERENCE

VARIANTS=('reference_response',)


def response_contract(variant):
    if variant not in VARIANTS:raise ValueError('Unregistered reference variant')
    return dict(mode=variant,reference=REFERENCE,nominal_wifi_bandwidth_hz=20_000_000,
        sample_rate_hz=25_000_000,window=[80,160],period_samples=20,cycles=4,
        delay_grid_count=320,delay_grid_step_samples=1/16,epsilon=1e-6,residual_soft_radius=4.,
        response_features=29,relative_cfo_alias_period_hz=1_250_000,
        reference_branch_flat_gain_phase_quotient=True,whole_model_invariant=False,
        raw_identity_path_retained=True,rx_arbitrary_channel_invariant=False,
        iq_convention_identified=False,hardware_parameter_estimation=False,
        initial_residual_gain=0.,learned_frontend_parameters=0,
        interpretation='Relative received in-band transfer/quality/CFO; RX/channel/equalizer remain confounds')


class ExcitationResponse(nn.Module):
    """Finite fixed correlator bank; no update, optimization or cross-packet state.

    The public L-STF bank is evaluated independently per packet. The normalized
    cross spectrum removes one complex scalar; it cannot remove unknown FIRs.
    CFO remains an explicit relative (TX-RX) observable. Hard delay routing is
    a forward operation, not model/query adaptation or a fitted persistent state.
    """
    def __init__(self,variant):
        super().__init__();self.variant=variant;self.epsilon=1e-6;self.radius=4.
        self.window=(80,160);self.period=20
        bank=np.stack([template(np.arange(20),d/16) for d in range(320)])
        # Public constants regenerated from the contract, never sample-derived.
        self.register_buffer('bank_real',torch.tensor(bank.real.tolist(),dtype=torch.float64),persistent=False)
        self.register_buffer('bank_imag',torch.tensor(bank.imag.tolist(),dtype=torch.float64),persistent=False)
        self.register_buffer('tone_bins',torch.tensor((TONES//4%20).tolist()),persistent=False)

    def contract(self):
        d=response_contract(self.variant)
        d.update(epsilon=self.epsilon,residual_soft_radius=self.radius,window=list(self.window),period_samples=self.period,
            delay_grid_count=self.bank_real.shape[0])
        if self.bank_real.shape!=(320,20) or self.bank_imag.shape!=(320,20) or self.tone_bins.tolist()!=(TONES//4%20).tolist():
            raise ValueError('Public correlator bank shape/tone contract differs')
        return d

    def components(self,x):
        if x.ndim!=3 or x.shape[1:]!=(2,256):raise ValueError('Expected IQ[B,2,256]')
        z=torch.complex(x[:,0],x[:,1]);lo,hi=self.window;w=z[:,lo:hi]
        cross=(w[:,20:]*w[:,:-20].conj()).sum(-1)
        # Fixed epsilon regularization also defines the zero-history derivative.
        omega=torch.atan2(cross.imag,cross.real+self.epsilon**2)/20
        n=torch.arange(80,device=x.device,dtype=x.dtype)
        corrected=w*torch.complex(torch.cos(-omega[:,None]*n),torch.sin(-omega[:,None]*n))
        mean=corrected.reshape(-1,4,20).mean(1)
        energy=mean.abs().square().mean(-1)
        # Normalize before correlation, retaining stability for nearly zero IQ.
        unit=mean/torch.sqrt(energy+self.epsilon**2)[:,None]
        bank=torch.complex(self.bank_real.to(x.dtype),self.bank_imag.to(x.dtype))
        corr=unit@bank.conj().T/20
        chosen=corr.abs().square().argmax(-1)
        reference=bank[chosen]
        gain=corr.gather(1,chosen[:,None]).squeeze(1)
        observed=torch.fft.fft(unit,dim=-1)[:,self.tone_bins]
        ideal=torch.fft.fft(reference,dim=-1)[:,self.tone_bins]
        transfer=observed*ideal.conj()/ideal.abs().square()
        relative=transfer*gain.conj()[:,None]/(gain.abs().square()+self.epsilon)[:,None]
        residual=relative-1
        bounded=residual/torch.sqrt(1+residual.abs().square()/self.radius**2)
        power=w.abs().square().mean(-1);packet_power=z.abs().square().mean(-1)
        repeat_norm=torch.sqrt(w[:,20:].abs().square().sum(-1)*w[:,:-20].abs().square().sum(-1)+self.epsilon**4)
        coherence=cross.abs()/repeat_norm
        coherent_fraction=energy/(power+self.epsilon**2)
        quality=gain.abs().square()
        # +/-625 kHz maps to [-1,1]; does not recover a calibrated TX oscillator.
        cfo=omega*20/torch.pi
        window_power=power/(packet_power+self.epsilon**2)
        features=torch.cat([bounded.real,bounded.imag,quality[:,None],coherence[:,None],
            coherent_fraction[:,None],cfo[:,None],window_power[:,None]],dim=1)
        return dict(features=features,relative_transfer=relative,selected_delay_samples=chosen.to(x.dtype)/16,
            quality=quality,coherence=coherence,coherent_fraction=coherent_fraction,cfo=cfo)

    def forward(self,x):return self.components(x)['features']

    @torch.no_grad()
    def diagnostics(self,x):
        d=self.components(x)
        return dict(template_correlation_mean=float(d['quality'].sqrt().mean()),
            cycle_coherence_mean=float(d['coherence'].mean()),coherent_fraction_mean=float(d['coherent_fraction'].mean()),
            cfo_normalized_abs_mean=float(d['cfo'].abs().mean()),
            relative_transfer_deviation_rms=float((d['relative_transfer']-1).abs().square().mean().sqrt()),
            scope='Last source batch, fixed per-packet forward, not hardware recovery')


class ReferenceCVS(nn.Module):
    def __init__(self,variant):
        super().__init__();self.variant=variant
        # Preserve the original core RNG draw; never inherit its weights.
        self.core=build_residual('residual_fusion')
        self.response=ExcitationResponse(variant)
        self.response_projection=nn.Sequential(nn.Linear(29,64),nn.SiLU(),nn.Linear(64,160),nn.LayerNorm(160))
        self.response_gain=nn.Parameter(torch.zeros(160))

    def contract(self):return self.response.contract()

    def features(self,x):
        base=self.core.features(x)
        physical=self.response_projection(self.response(x))
        return base+self.response_gain.tanh()*physical

    def classify_features(self,features):return self.core.id_backbone.cls_head.classify(features)
    def forward(self,x):return self.classify_features(self.features(x))

    @torch.no_grad()
    def diagnostics(self,x):
        d=self.response.diagnostics(x)
        d.update(response_gain_abs_mean=float(self.response_gain.tanh().abs().mean()),
            response_gain_gradient_norm=float(self.response_gain.grad.norm()) if self.response_gain.grad is not None else None,
            response_projection_gradient_norm=float(torch.stack([p.grad.norm() for p in self.response_projection.parameters() if p.grad is not None]).norm())
            if any(p.grad is not None for p in self.response_projection.parameters()) else None)
        return d


def build(variant):
    if variant not in VARIANTS:raise ValueError('Unregistered reference model')
    return ReferenceCVS(variant)


@torch.no_grad()
def frozen_synthetic_diagnostics(model):
    from experiments.cvs_reference_identity.physics import controlled_cascade_diagnostics
    return controlled_cascade_diagnostics(model)
