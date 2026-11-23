import torch

from experiments.pamf_lora_main.spectral_action import band_confidence, band_doses, apply_bands


def test_band_validation_requires_both_groups_and_bounds_the_consensus_dose():
    rng = torch.Generator().manual_seed(75)
    a = torch.randn((3, 16), dtype=torch.complex64, generator=rng)
    b = torch.randn((3, 16), dtype=torch.complex64, generator=rng)
    bands = torch.stack((torch.arange(16)<8,torch.arange(16)>=8)).float()
    pa,pb=a+1.5*(b-a),b+2*(a-b)
    gain,optimum=band_confidence(a,b,pa,pb,bands)
    torch.testing.assert_close(gain,torch.full_like(gain,.9375),atol=5e-7,rtol=1e-6)
    torch.testing.assert_close(optimum,torch.full_like(optimum,.5))
    gain,dose=band_confidence(a,b,pa,b,bands)
    assert torch.equal(gain,torch.zeros_like(gain))
    assert torch.equal(dose,torch.zeros_like(dose))


def test_confirmed_doses_preserve_legacy_strength_and_restrict_each_band():
    observed=torch.tensor([True,False])
    gain=torch.tensor([[.95,.12,.12,.12],[.95,.12,.12,.12]])
    cut=torch.full((4,),.3)
    optimum=torch.full((2,4),.9)
    hard,taper,boost=band_doses(observed,gain,optimum,cut)
    baseline=torch.tensor([[.50,.50,.50,.50],[.50,0.,0.,0.]])
    torch.testing.assert_close(hard,baseline)
    torch.testing.assert_close(taper,baseline)
    torch.testing.assert_close(boost,torch.tensor([[.90,.50,.50,.50],[.50,0.,0.,0.]]))
    bands=torch.eye(4,5)
    bands[:,0]=0.
    edit=torch.ones((2,6,5),dtype=torch.complex64)
    projection=apply_bands(edit,boost,bands)
    assert torch.equal(projection[:,:,0],torch.zeros_like(projection[:,:,0]))
    torch.testing.assert_close(projection[1].abs().sum(),torch.tensor(0.))


def test_boost_only_changes_the_existing_action_relative_to_weak_taper():
    legacy=torch.tensor([True,False])
    gain=torch.full((2,4),.8)
    optimum=torch.tensor([[.95,.80,.35,.20],[.95,.80,.35,.20]])
    _,taper,boost=band_doses(legacy,gain,optimum,torch.full((4,),.3))
    torch.testing.assert_close(boost[1],taper[1])
    torch.testing.assert_close(boost[0],torch.tensor([.95,.80,.50,.50]))
