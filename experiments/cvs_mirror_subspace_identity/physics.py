"""Fixed public-only mirror-mixing diagnostics; no training or model selection."""
import torch
from torch.nn import functional as F
from experiments.cvs_spectral_relation_identity.physics import (
    diagnostic_mode,public_inputs,unit_rms,multiply_pair,causal_fir,relative_distance)


def row_mix(v,m):
    z=torch.complex(v[:,0],v[:,1])
    y=m@z
    return torch.stack((y.real,y.imag),1)


def compare(block,v,w):
    q,r=block.relations(v),block.relations(w)
    c,d=block.components(v),block.components(w)
    energy=(~c['energy_floor_active'])&(~d['energy_floor_active'])
    rank=(~c['determinant_floor_active'])&(~d['determinant_floor_active'])
    eligible=energy&rank
    err=(q-r).abs().amax((1,3,4))
    return dict(total_pair_count=eligible.numel(),eligible_pair_count=int(eligible.sum()),
        excluded_pair_count=int((~eligible).sum()),energy_excluded_pair_count=int((~energy).sum()),
        determinant_excluded_pair_count=int((~rank).sum()),
        qualified_max_error=float(err[eligible].max()) if eligible.any() else None,
        excluded_max_error=float(err[~eligible].max()) if (~eligible).any() else None,
        full_max_error=float(err.max()),full_relative_distance=relative_distance(q,r),
        original_energy_floor_count=int(c['energy_floor_active'].sum()),
        changed_energy_floor_count=int(d['energy_floor_active'].sum()),
        original_determinant_floor_count=int(c['determinant_floor_active'].sum()),
        changed_determinant_floor_count=int(d['determinant_floor_active'].sum()))


@torch.no_grad()
def frozen_synthetic_diagnostics(model):
    p=next(model.parameters());device,dtype=p.device,p.dtype
    with diagnostic_mode(model,device):
        block=model.core.mirror_relation;x=public_inputs(device,dtype)
        gen=torch.Generator(device='cpu').manual_seed(2026100311)
        s=torch.randn(6,2,64,7,generator=gen,dtype=torch.float64).to(device=device,dtype=dtype)
        v=block.mix_spectra(s)
        # Fixed invertible nonunitary, frequency-varying left actions.
        f=torch.arange(31,device=device,dtype=dtype)
        m=torch.zeros(31,2,2,device=device,dtype=torch.complex128 if dtype==torch.float64 else torch.complex64)
        m[:,0,0]=1.2+.15*torch.sin(.2*f)+.12j
        m[:,0,1]=.18+.09j;m[:,1,0]=-.08+.05j;m[:,1,1]=.75+.13*torch.cos(.3*f)-.11j
        ideal=compare(block,v,row_mix(v,m))
        theta=v.new_tensor(.43)
        u=torch.complex(torch.stack((torch.stack((theta.cos(),-theta.sin())),
                                    torch.stack((theta.sin(),theta.cos())))),torch.zeros(2,2,device=device,dtype=dtype))
        ideal['unitary_mixing_max_error']=float((block.relations(v)-block.relations(row_mix(v,u))).abs().max())
        ideal['minimum_mixing_abs_determinant']=float(torch.linalg.det(m).abs().min())
        ideal['scope']='Public spectral tensors; qualification requires both energy and determinant floors inactive before and after mixing'
        # Exact memoryless receiver IQ action under real-window STFT.
        z=x[:18];conj=z*z.new_tensor([1.,-1.])[None,:,None]
        y=multiply_pair(z,1.04,.12)+multiply_pair(conj,.14,-.07)
        a=torch.complex(z.new_tensor(1.04),z.new_tensor(.12));b=torch.complex(z.new_tensor(.14),z.new_tensor(-.07))
        iqm=torch.stack((torch.stack((a,b)),torch.stack((b.conj(),a.conj()))))
        original=block.mix_spectra(block.spectral(z));changed=block.mix_spectra(block.spectral(y))
        iq=compare(block,original,changed)
        iq['commutation_max_error']=float((changed-row_mix(original,iqm)).abs().max())
        iq['receiver_a']=[1.04,.12];iq['receiver_b']=[.14,-.07]
        # Include all weak/rank-one/zero pairs, no eligible-only reporting.
        allv=block.mix_spectra(block.spectral(x))
        domain=compare(block,allv,row_mix(allv,iqm))
        logits=model(x);rotated=multiply_pair(x,torch.cos(x.new_tensor(.731)),torch.sin(x.new_tensor(.731)))
        phase=dict(ordinary_max_abs_error=float((logits[:18]-model(rotated)[:18]).abs().max()),
                   zero_finite=bool(torch.isfinite(logits[24:]).all()))
        original_q=block.statistics(z);embedding=F.normalize(model.features(z),dim=1,eps=1e-4)
        fir=[]
        for delay in (2,8,16):
            taps=[(0,1.,0.),(delay,.23,.09)];changed_iq=unit_rms(causal_fir(z,taps))
            fir.append(dict(name='delay'+str(delay),taps=[list(t) for t in taps],
                relation_relative_distance=relative_distance(original_q,block.statistics(changed_iq)),
                mean_unit_embedding_distance=float((embedding-F.normalize(model.features(changed_iq),dim=1,eps=1e-4)).norm(dim=1).mean())))
        power=z.square().sum(1);angle=.17*power
        interventions=dict(am_am=unit_rms(z/(1+.12*power)[:,None]),
            am_pm=unit_rms(multiply_pair(z,angle.cos(),angle.sin())),rx_iq=unit_rms(y))
        sensitivity={name:dict(relation_relative_distance=relative_distance(original_q,block.statistics(value)),
            mean_unit_embedding_distance=float((embedding-F.normalize(model.features(value),dim=1,eps=1e-4)).norm(dim=1).mean())) for name,value in interventions.items()}
        return dict(schema='mirror_relation_public_physics_v1',synthetic_only=True,target_access=False,
            training_augmentation=False,optimizer_updates=0,public_packets=30,
            signal_groups=['6 fixed broadband noise','6 analytic tones','6 periodic envelope signals','6 DC','6 zero'],
            ideal_mixing=ideal,time_iq=iq,all_public_domain=domain,whole_model_phase=phase,
            finite_fir_sensitivity=fir,tx_rx_sensitivity=sensitivity,
            mirror_relation=model.mirror_relation_diagnostics(x),
            limits=['Qualified subspace property only; finite FIR and nonlinear RX are outside exact model',
                    'The quotient can erase TX linear IQ distinctions too; absolute backbone retained',
                    'Rank-one subspace output tends to zero; energy/determinant excluded pairs reported',
                    'No whole-model receiver invariance or accuracy inference'])
