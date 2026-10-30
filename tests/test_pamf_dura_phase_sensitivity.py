import numpy as np
import pytest
import torch

from experiments.pamf_dura_efficiency.phase_sensitivity import metrics, phase_decision, rotate


def test_rotation_preserves_pointwise_power_and_zero_is_exact():
    x=torch.randn(9,2,32,generator=torch.Generator().manual_seed(19))
    assert torch.equal(rotate(x,0.),x)
    torch.testing.assert_close(rotate(rotate(x,.7),-.7),x,atol=5e-7,rtol=2e-6)
    torch.testing.assert_close(rotate(x,1.3).square().sum(1),x.square().sum(1),atol=2e-6,rtol=2e-6)


def test_metrics_ignore_common_logit_shift_but_detect_head_changes():
    f=torch.eye(6); z=torch.eye(6)*3
    same=metrics(f,z+7,f,z)
    assert same['teacher_agreement'].sum()==6 and same['posterior_KL'].max()<1e-6
    changed=metrics(f,z.roll(1,1),f,z)
    assert changed['teacher_agreement'].sum()==0 and changed['posterior_KL'].min()>1
    assert same['feature_cosine_error'].max()==0
    with pytest.raises(ValueError):metrics(f,z[:,:5],f,z)


def passed_summary():
    def group(n):return dict(count=n,teacher_correct=n,metrics=dict(feature_cosine_error=dict(mean=0.),posterior_KL=dict(mean=0.)))
    return {str(a):dict(all=group(512),**{f'rx:{r}':group(n) for r,n in zip((1,3,4,6,8),(102,102,102,103,103))}) for a in range(8)}


def test_all_angles_and_all_receivers_must_pass_and_failures_are_not_averaged():
    summary=passed_summary();assert phase_decision(summary)['phase_quotient_study_eligible']
    summary['6']['all']['teacher_correct']=509
    assert not phase_decision(summary)['phase_quotient_study_eligible']
    summary=passed_summary();summary['2']['rx:8']['teacher_correct']=101
    assert not phase_decision(summary)['phase_quotient_study_eligible']
    summary=passed_summary();summary['5']['all']['metrics']['posterior_KL']['mean']=.00101
    assert not phase_decision(summary)['phase_quotient_study_eligible']
    summary=passed_summary();summary['3'].pop('rx:1')
    with pytest.raises(ValueError):phase_decision(summary)
