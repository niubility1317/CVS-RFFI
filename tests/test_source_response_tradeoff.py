import torch
from experiments.phase1_dura_orthogonal.source_response_tradeoff import apply,fit_weights,measurements


def test_zero_wave_error_leaves_identity_when_only_wave_is_fit():
    torch.manual_seed(19)
    clean=torch.complex(torch.randn(16,256),torch.randn(16,256))
    delta=torch.complex(torch.randn(16,4,256),torch.randn(16,4,256))*.01
    weights=fit_weights(clean,clean,delta,delta*.5,0.)
    torch.testing.assert_close(apply(clean,weights),clean)
    assert measurements(clean,clean,delta,delta*.5,weights).shape==(16,9)


def test_response_objective_can_correct_a_known_half_gain():
    torch.manual_seed(20)
    clean=torch.complex(torch.randn(32,256),torch.randn(32,256))
    delta=torch.complex(torch.randn(32,4,256),torch.randn(32,4,256))*.01
    weights=fit_weights(clean,clean,delta,delta*.5,10.)
    measured=measurements(clean,clean,delta,delta*.5,weights)
    assert measured[:,1:5].mean()<.25
    assert measured[:,5:9].mean()>.5
