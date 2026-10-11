"""29D projection compatibility with no explicit CFO output/estimator.

Public correlator banks are copied from the verified native frontend after the
native runtime builds the model; importing this module never bootstraps data.
"""
import torch
from torch import nn
from .physical import iq,aligned_cycles,repetition_quality


class NoExplicitCFOResponse(nn.Module):
    def __init__(self,original):
        super().__init__()
        self.variant=original.variant;self.epsilon=original.epsilon;self.radius=original.radius
        self.window=(80,160);self.period=20
        self.original_contract=original.contract()
        for key in ('bank_real','bank_imag','tone_bins'):
            self.register_buffer(key,getattr(original,key).detach().clone(),persistent=False)

    def contract(self):
        value={k:v for k,v in self.original_contract.items() if 'cfo' not in k.lower()}
        value.update(frontend_version='report_no_explicit_cfo_v1',response_features=29,
            zero_compatibility_slot=27,frequency_offset_estimator=False,
            alignment='per-cycle unit-modulus relative phase; alignment coefficients not exposed',
            raw_identity_path_retained=True,whole_model_frequency_shift_invariant=False,
            interpretation='Effective linear response/repetition quality; no calibrated hardware recovery')
        return value

    def components(self,x):
        z=iq(x);cycles=aligned_cycles(x);mean=cycles.mean(1)
        energy=mean.abs().square().mean(-1)
        unit=mean/torch.sqrt(energy+self.epsilon**2)[:,None]
        bank=torch.complex(self.bank_real.to(x.dtype),self.bank_imag.to(x.dtype))
        corr=unit@bank.conj().T/20;chosen=corr.abs().square().argmax(-1)
        reference=bank[chosen];gain=corr.gather(1,chosen[:,None]).squeeze(1)
        observed=torch.fft.fft(unit,dim=-1)[:,self.tone_bins]
        ideal=torch.fft.fft(reference,dim=-1)[:,self.tone_bins]
        transfer=observed*ideal.conj()/ideal.abs().square().clamp_min(self.epsilon**2)
        relative=transfer*gain.conj()[:,None]/(gain.abs().square()+self.epsilon)[:,None]
        residual=relative-1;bounded=residual/torch.sqrt(1+residual.abs().square()/self.radius**2)
        quality=gain.abs().square();q=repetition_quality(x)
        window_power=z[:,80:160].abs().square().mean(-1)/(z.abs().square().mean(-1)+self.epsilon**2)
        features=torch.cat((bounded.real,bounded.imag,quality[:,None],q[:,0:1],q[:,2:3],torch.zeros_like(quality[:,None]),window_power[:,None]),1)
        return dict(features=features,relative_transfer=relative,quality=quality,
                    coherence=q[:,0],coherent_fraction=q[:,2])

    def forward(self,x):return self.components(x)['features']

    @torch.no_grad()
    def diagnostics(self,x):
        c=self.components(x)
        return dict(template_correlation_mean=float(c['quality'].sqrt().mean()),
            cycle_coherence_mean=float(c['coherence'].mean()),coherent_fraction_mean=float(c['coherent_fraction'].mean()),
            relative_transfer_deviation_rms=float((c['relative_transfer']-1).abs().square().mean().sqrt()),
            frontend_version='report_no_explicit_cfo_v1',explicit_frequency_offset_output=False,
            scope='packet-local received response; not pure TX parameters or strict blindspot')


def install_reference(model):
    replacements=[]
    for name,module in list(model.named_modules()):
        if not name.startswith('id_backbone.') or not hasattr(module,'response_projection') or not hasattr(module,'response'):
            continue
        first=next(p for p in module.response_projection.modules() if isinstance(p,nn.Linear))
        if first.in_features!=29:raise ValueError('Unexpected parent reference projection')
        if isinstance(module.response,NoExplicitCFOResponse):raise ValueError('Reference already installed')
        module.response=NoExplicitCFOResponse(module.response);replacements.append(name)
    if len(replacements)!=1:raise ValueError('Expected exactly one identity reference frontend')
    return dict(version='report_no_explicit_cfo_v1',replacements=replacements,
        parent_weights_compatible=True,weight_values_unchanged=True,L_readaptation_required=True,
        old_reference_state_action_caches_compatible=False)
