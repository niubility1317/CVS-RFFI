import math
import torch

from experiments.pamf_dura_efficiency.phase_oracle import align_phase, phase_component, phase_decision


def test_phase_only_oracle_preserves_amplitude_and_gives_exact_mse_decomposition():
    rng=torch.Generator().manual_seed(29)
    x=torch.randn(4,2,32,generator=rng)
    z=torch.complex(x[:,0],x[:,1]);angle=torch.tensor([.4,-1.,math.pi,.0])
    r=1.7*z*torch.polar(torch.ones_like(angle),angle)[:,None]
    restored=torch.stack((r.real,r.imag),1)
    aligned,m=align_phase(restored,x)
    torch.testing.assert_close(aligned,1.7*x,atol=1e-6,rtol=2e-6)
    torch.testing.assert_close(aligned.square().sum(1),restored.square().sum(1),atol=4e-6,rtol=2e-6)
    torch.testing.assert_close(m['native_nmse']-m['aligned_nmse'],phase_component(restored,x),atol=2e-6,rtol=2e-6)
    assert m['valid_phase'].sum()==4


def test_no_phase_information_is_explicit_and_not_an_amplitude_fit():
    x=torch.zeros(2,2,8);x[:,0,0]=1
    r=torch.zeros_like(x);r[1,0,1]=1
    aligned,m=align_phase(r,x)
    assert torch.equal(aligned,r) and m['valid_phase'].sum()==0
    assert torch.equal(m['phase_nmse_drop'],torch.zeros(2))
    assert m['aligned_nmse'].tolist()==[1.,2.]
    variable=torch.randn(3,2,8,requires_grad=True)
    phase_component(variable,torch.ones_like(variable)).mean().backward()
    assert torch.isfinite(variable.grad).all() and variable.grad.abs().sum()>0


def passed_summary():
    def group():
        return dict(count=512,valid_phases=512,phase_fraction=.2,
                    native=dict(teacher_correct=470),aligned=dict(teacher_correct=480))
    return dict(clean=dict(all=group()),**{str(e):dict(all=group(),**{f'scene:{s}':group() for s in range(7)}) for e in (1,20)})


def test_qualification_requires_both_views_choices_phase_fraction_and_clean():
    value=passed_summary();assert phase_decision(value)['phase_weight_study_eligible']
    value['20']['all']['phase_fraction']=.099
    assert not phase_decision(value)['phase_weight_study_eligible']
    value=passed_summary();value['1']['all']['aligned']['teacher_correct']=475
    assert not phase_decision(value)['phase_weight_study_eligible']
    value=passed_summary();value['20']['scene:6']['aligned']['teacher_correct']=469
    assert not phase_decision(value)['phase_weight_study_eligible']
    value=passed_summary();value['clean']['all']['aligned']['teacher_correct']=469
    assert not phase_decision(value)['phase_weight_study_eligible']
