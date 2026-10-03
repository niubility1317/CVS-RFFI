"""Public, fixed-input property checks; never augmentation or source selection."""
from contextlib import contextmanager
import torch
from torch.nn import functional as F


@contextmanager
def diagnostic_mode(model, device):
    flags=[(m,m.training) for m in model.modules()]
    try:
        with torch.random.fork_rng(devices=[device.index or 0] if device.type=='cuda' else []):
            model.eval()
            yield
    finally:
        for module,flag in flags:module.training=flag


def unit_rms(x):
    return x*torch.rsqrt(x.square().sum(1).mean(-1,keepdim=True).clamp_min(1e-12))[:,None]


def multiply_pair(x, real, imag):
    r,i=x.unbind(1)
    return torch.stack((r*real-i*imag,i*real+r*imag),1)


def public_inputs(device,dtype):
    generator=torch.Generator(device='cpu').manual_seed(2026100307)
    noise=torch.randn(6,2,256,generator=generator,dtype=torch.float64).to(device=device,dtype=dtype)
    n=torch.arange(256,device=device,dtype=dtype)[None]
    a=torch.arange(1,7,device=device,dtype=dtype)[:,None]
    phase=(.071+.021*a)*n+.13*a
    tone=torch.stack((phase.cos(),phase.sin()),1)
    envelope=1+.22*torch.cos(2*torch.pi*n/20)+.11*torch.sin(4*torch.pi*n/20+.1*a)
    periodic=torch.stack((envelope*(phase+.1*envelope).cos(),envelope*(phase+.1*envelope).sin()),1)
    dc=torch.stack((torch.ones_like(phase)*(.1*a).cos(),torch.ones_like(phase)*(.1*a).sin()),1)
    return torch.cat((unit_rms(noise),tone,unit_rms(periodic),dc,torch.zeros_like(noise)),0)


def causal_fir(x,taps):
    # tap tuples are (delay, real, imag); output keeps 256 observed positions.
    result=torch.zeros_like(x)
    for delay,real,imag in taps:
        shifted=x if delay==0 else F.pad(x[...,:-delay],(delay,0))
        result=result+multiply_pair(shifted,real,imag)
    return result


def relative_distance(a,b):
    return float(((a-b).flatten(1).norm(dim=1)/a.flatten(1).norm(dim=1).clamp_min(1e-12)).mean())


@torch.no_grad()
def frozen_synthetic_diagnostics(model):
    parameter=next(model.parameters());device,dtype=parameter.device,parameter.dtype
    with diagnostic_mode(model,device):
        x=public_inputs(device,dtype);block=model.core.spectral_relation
        generator=torch.Generator(device='cpu').manual_seed(2026100311)
        # Isolated algebraic-domain inputs: no claim that this is an exact STFT FIR.
        spectra=torch.randn(6,2,64,7,generator=generator,dtype=torch.float64).to(device=device,dtype=dtype)
        v=block.mix_spectra(spectra);q=block.relations(v)
        f=torch.arange(64,device=device,dtype=dtype)[None,:,None]
        amplitude=.6+1.1*(.5+.5*torch.sin(.13*f))
        phase=.37+.17*f
        changed=multiply_pair(spectra,amplitude*phase.cos(),amplitude*phase.sin())
        q_changed=block.relations(block.mix_spectra(changed))
        common=multiply_pair(spectra,1.7*torch.cos(f.new_tensor(.71)),1.7*torch.sin(f.new_tensor(.71)))
        q_common=block.relations(block.mix_spectra(common))
        phase_only=multiply_pair(spectra,phase.cos(),phase.sin())
        q_phase=block.relations(block.mix_spectra(phase_only))
        transformed=block.mix_spectra(changed)
        energy=v.square().sum(1).sum(-1)
        changed_energy=transformed.square().sum(1).sum(-1)
        eligible=(energy>=block.energy_floor(v)) & (changed_energy>=block.energy_floor(transformed))
        error_by_frequency=(q-q_changed).abs().amax(dim=(1,3,4))
        ideal=dict(per_frequency_normalization=block.per_frequency,
            varying_frequency_gain_max_error=float((q-q_changed).abs().max()),
            varying_frequency_gain_relative_distance=relative_distance(q,q_changed),
            common_complex_gain_max_error=float((q-q_common).abs().max()),
            frequency_phase_only_max_error=float((q-q_phase).abs().max()),
            minimum_original_frequency_energy=float(energy.min()),
            minimum_changed_frequency_energy=float(changed_energy.min()),
            original_floor_active_count=int(block.floor_active(v).sum()),
            changed_floor_active_count=int(block.floor_active(transformed).sum()),
            varying_gain_eligible_frequency_count=int(eligible.sum()),
            varying_gain_excluded_frequency_count=int((~eligible).sum()),
            varying_frequency_gain_eligible_max_error=float(error_by_frequency[eligible].max()) if eligible.any() else None,
            scope='Exact per-frequency complex multiplication on public spectral tensors; above-floor qualification is explicit')
        logits=model(x);rotated=multiply_pair(x,torch.cos(x.new_tensor(.731)),torch.sin(x.new_tensor(.731)))
        rotated_logits=model(rotated)
        phase_check=dict(ordinary_max_abs_error=float((logits[:18]-rotated_logits[:18]).abs().max()),
            dc_max_abs_error=float((logits[18:24]-rotated_logits[18:24]).abs().max()),
            zero_finite=bool(torch.isfinite(logits[24:]).all()),
            scope='Whole model finite-precision common-phase check; no frequency-gain invariance claim')
        # Full packet finite causal FIRs and nonlinear TX/RX examples deliberately
        # do not satisfy the ideal diagonal short-time spectral action in general.
        z=x[:18];original_q=block.statistics(z);original_emb=F.normalize(model.features(z),dim=1,eps=1e-4)
        firs={'delay2':[(0,1.,0.),(2,.23,.09)],
              'delay8':[(0,1.,0.),(8,.23,.09)],
              'delay16':[(0,1.,0.),(16,.23,.09)]}
        fir_records=[]
        for name,taps in firs.items():
            y=unit_rms(causal_fir(z,taps))
            fir_records.append(dict(name=name,taps=[list(t) for t in taps],
                relation_relative_distance=relative_distance(original_q,block.statistics(y)),
                mean_unit_embedding_distance=float((original_emb-F.normalize(model.features(y),dim=1,eps=1e-4)).norm(dim=1).mean())))
        power=z.square().sum(1)
        angle=.17*power
        interventions=dict(am_am=unit_rms(z/(1+.12*power)[:,None]),
            am_pm=unit_rms(multiply_pair(z,angle.cos(),angle.sin())),
            rx_iq=unit_rms(z*z.new_tensor([1.08,.92])[None,:,None]))
        sensitivities={}
        for name,y in interventions.items():
            sensitivities[name]=dict(relation_relative_distance=relative_distance(original_q,block.statistics(y)),
                mean_unit_embedding_distance=float((original_emb-F.normalize(model.features(y),dim=1,eps=1e-4)).norm(dim=1).mean()))
        return dict(schema='spectral_relation_public_physics_v1',synthetic_only=True,target_access=False,
            training_augmentation=False,optimizer_updates=0,public_packets=30,
            signal_groups=['6 fixed broadband noise','6 analytic tones','6 periodic envelope signals','6 DC','6 zero'],
            ideal_gain=ideal,whole_model_phase=phase_check,finite_fir_sensitivity=fir_records,
            tx_rx_sensitivity=sensitivities,spectral_relation=model.spectral_relation_diagnostics(x),
            physical_parameters=dict(am_am_coefficient=.12,am_pm_coefficient=.17,rx_iq_gains=[1.08,.92],constant_phase=.731),
            limits=['No threshold or accuracy inference for finite FIR/nonlinear sensitivity',
                    'The same above-floor ideal quotient erases a purely multiplicative TX frequency response',
                    'Absolute backbone is retained; whole-model channel invariance is not claimed'])
