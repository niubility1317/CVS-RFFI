import torch

from experiments.pamf_lora_main import carrier_coupled as m
from experiments.pamf_lora_main.carrier_coupled_source import RECIPES
from experiments.pamf_lora_main.dura_repair import PacketRestorer, real_iq


def test_packet_frame_retains_relative_chirp_power_and_reconstructs_iq():
    local = torch.ones((1, 3, 8), dtype=torch.complex64)
    rotation = torch.polar(torch.ones_like(local.real), .2*local.real)
    quality = torch.ones((1,3,1))
    rms = torch.tensor([[[1.],[2.],[4.]]])
    observed, rot, q, scale = m.packet_frame((local,rotation,quality,rms))
    torch.testing.assert_close(observed*rot*scale,local*rotation*rms)
    torch.testing.assert_close(observed.abs().square().mean(),torch.tensor(1.))
    assert torch.equal(q,quality)
    torch.testing.assert_close(observed[0,:,0].real/observed[0,0,0].real,rms.flatten())


def test_carrier_coupling_transports_relative_complex_detail():
    clean = torch.tensor([[[1+.15j,1-.15j],[1.5+.2j,1.5-.2j]]])
    gain = torch.tensor([[[.6],[1.8]]])
    observed = clean*gain
    linked = m.synthesize(observed,observed,gain.reciprocal(),coupled=True)
    separate = m.synthesize(observed,observed,gain.reciprocal(),coupled=False)
    torch.testing.assert_close(linked,clean)
    torch.testing.assert_close(separate.mean(-1),clean.mean(-1))
    assert not torch.allclose(separate,clean)


def test_gain_perturbation_matches_the_weak_change_and_preserves_detached_identity():
    rng = torch.Generator().manual_seed(177)
    base = torch.ones((5,6,256),dtype=torch.complex64)
    weak = base*(1+.05j)
    a,b = m.amplitude_perturb(base,weak,generator=rng)
    torch.testing.assert_close(b,a*(1+.05j))
    assert (a.abs()-1).abs().max()>.02


def test_complete_source_objective_reaches_envelope_and_coarse_model():
    tracker = PacketRestorer('consensus').eval()
    rng = torch.Generator().manual_seed(784)
    drift = torch.complex(torch.randn((6,6,256),generator=rng),torch.randn((6,6,256),generator=rng))
    iq = real_iq(tracker.reference.chirp*(1+.08*drift))
    for linked in (False, True):
        name = 'coupled_gain' if linked else 'independent_gain'
        model = m.CarrierCoupledRecovery((4,8,8,8),coupled=linked)
        loss,diagnostics = m.training_objective(model,tracker,iq,rng,with_identity=True,recipe=RECIPES[name],step=1)
        loss.backward()
        assert diagnostics.isfinite().all()
        for network in (model.envelope,model.coarse):
            gradients = [v.grad for v in network.parameters() if v.grad is not None]
            assert gradients and all(v.isfinite().all() for v in gradients)
            assert any(v.abs().sum()>0 for v in gradients)
