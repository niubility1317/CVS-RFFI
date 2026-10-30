import math
import numpy as np
import pytest
import torch

from experiments.cvs_rff_physics.known_excitation import template
from experiments.cvs_reference_identity.model import ExcitationResponse
from experiments.pamf_dura_efficiency.phase_anchor import public_gain, circular_summary, anchor_decision


def test_public_correlator_reports_known_phase_and_transports_global_rotation():
    bank=ExcitationResponse('reference_response')
    assert not list(bank.parameters())
    phase=torch.tensor([.4,-1.2,2.3]);base=torch.tensor(template(np.arange(256),4.375),dtype=torch.complex64)
    z=2*base[None]*torch.polar(torch.ones_like(phase),phase)[:,None]
    x=torch.stack((z.real,z.imag),1)
    gain=public_gain(x,bank)
    torch.testing.assert_close(gain/gain.abs(),torch.polar(torch.ones_like(phase),phase),atol=2e-5,rtol=2e-5)
    rotation=torch.polar(torch.ones(3),torch.tensor([.2,.5,-.3]))
    shifted=z*rotation[:,None]
    changed=public_gain(torch.stack((shifted.real,shifted.imag),1),bank)
    torch.testing.assert_close(changed,gain*rotation,atol=3e-6,rtol=2e-5)


def test_circular_statistics_handle_wrap_and_do_not_invent_phase_for_cancellation():
    gain=np.exp(1j*np.deg2rad([179.,-179.]))
    stats=circular_summary(gain)
    assert abs(abs(stats['mean_degrees'])-180)<1e-10 and stats['resultant']>.99
    assert stats['p95_absolute_error_degrees']==pytest.approx(1.)
    cancelled=circular_summary(np.exp(1j*np.asarray([0.,math.pi/2,math.pi,3*math.pi/2])))
    assert cancelled['mean_degrees'] is None
    empty=circular_summary(np.zeros(3,dtype=np.complex64))
    assert empty['valid_count']==0 and empty['mean_degrees'] is None


def passed_fixture():
    phase=dict(all=dict(valid_fraction=1.,resultant=.999,mean_degrees=0.),
               sessions={f'{r}:{d}':dict(valid_fraction=1.,resultant=.999,mean_error_to_global_degrees=0.)
                         for r in (1,3,4,6,8) for d in (1,2,3)})
    f=dict(all=dict(count=512,teacher_correct=512,feature_cosine_error=.0001,posterior_KL=.0001),
           receivers={str(r):dict(count=102,teacher_correct=102) for r in (1,3,4,6,8)})
    f['receivers']['8']=dict(count=104,teacher_correct=104)
    return phase,f


def test_global_anchor_requires_every_source_session_and_heldout_f_preservation():
    phase,f=passed_fixture();assert anchor_decision(phase,f)['source_anchor_study_eligible']
    phase['sessions']['8:3']['mean_error_to_global_degrees']=5.01
    assert not anchor_decision(phase,f)['source_anchor_study_eligible']
    phase,f=passed_fixture();f['receivers']['1']['teacher_correct']=100
    assert not anchor_decision(phase,f)['source_anchor_study_eligible']
    phase,f=passed_fixture();phase['all']['resultant']=.989
    assert not anchor_decision(phase,f)['source_anchor_study_eligible']
    phase,f=passed_fixture();phase['all']['mean_degrees']=None
    assert not anchor_decision(phase,f)['source_anchor_study_eligible']
    phase,f=passed_fixture();phase['sessions'].pop('1:1')
    with pytest.raises(ValueError):anchor_decision(phase,f)
