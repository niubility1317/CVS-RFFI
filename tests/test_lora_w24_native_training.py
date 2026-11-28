import pytest
import torch

from experiments.pamf_lora_main import w24_joint_pamf as joint
from experiments.pamf_lora_main import w24_native_training as train
from experiments.pamf_lora_main.w24_clean_geometry import clean_geometry_risk
from experiments.pamf_lora_main.w24_native_dura import NativeDura, iq_error, source_pair
from experiments.pamf_lora_main.w24_native_edit_alignment import edit_parts
from experiments.pamf_lora_main.w24_native_spectral_innovation import SpectralInnovation
from experiments.pamf_lora_main.w24_native_direction import (
    DirectionRefiner, projected_pair_loss, remove_coarse_component,
)
from experiments.pamf_lora_main.w24_strong_source import StrongSourceMix
from experiments.pamf_lora_main.w24_native_stream import NativeStreamRefiner
from experiments.pamf_lora_main.w24_native_noise_pair import NoisePairRefiner, PairedNoiseViews
from experiments.pamf_lora_main.w24_aperture_reliability import InvisibleIQ, LocalizedWhite, VisibleIQ, reliability
from experiments.pamf_lora_main.w24_packet_unet import PacketUNet
from experiments.pamf_restoration.models import complex_iq, pair


def test_native_r02_r07_r17_preserve_aperture_length_and_unroll_gradients():
    iq = torch.randn(1,8192,dtype=torch.complex64,generator=torch.Generator().manual_seed(15))
    model = NativeDura()
    for stage in train.STAGES:
        with torch.no_grad():
            out = model(iq,stage=stage)
        assert out.shape == iq.shape and out.is_complex() and torch.isfinite(out).all()
    model.requires_grad_(False)
    for priors in model.priors:
        priors.out.requires_grad_(True)
    clean, observed, _, _ = source_pair(iq,torch.Generator().manual_seed(8))
    incoming = pair(observed)
    coarse, action, variance = model.estimate(incoming)
    loss = iq_error(complex_iq(model.unroll(incoming,coarse,action,variance)),clean)
    loss.backward()
    assert all(p.out.weight.grad is not None and torch.isfinite(p.out.weight.grad).all()
               for p in model.priors)
    assert all(p.grad is None for p in model.estimator.parameters())


def test_native_length_transfer_changes_only_positional_grid():
    model = PacketUNet(across_chirps=True)
    native = train.NativeTaskRestorer(torch.nn.Identity(),model)
    parent = model.state_dict()
    resized = native.core.state_dict()
    assert resized['temporal_position'].shape == (128,64)
    assert parent.keys() == resized.keys()
    assert all(torch.equal(value,resized[name]) for name,value in parent.items()
               if name != 'temporal_position')


def test_native_stream_only_crosses_real_chirp_seams_in_continuous_arm():
    raw = torch.zeros(1,8192,dtype=torch.complex64)
    raw[0,1024] = 1
    coarse = raw * (1+.1j)
    isolated = NativeStreamRefiner(continuous=False)
    continuous = NativeStreamRefiner(continuous=True)
    for model in (isolated, continuous):
        torch.testing.assert_close(model(raw,coarse),coarse,rtol=0.,atol=0.)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
            model.input.weight[0,0,4] = 1
            model.edit.weight[0,0,2] = 1
    independent = isolated(raw,coarse)
    adjacent = continuous(raw,coarse)
    assert independent[0,1023] == 0 and adjacent[0,1023].real > 0
    torch.testing.assert_close(independent[...,1024:],adjacent[...,1024:],rtol=0.,atol=0.)
    assert torch.count_nonzero(adjacent[0,:1023]) == 0
    assert (adjacent-coarse).abs().max() <= .25*raw.abs().square().mean().sqrt()


def test_paired_source_changes_only_the_independent_training_witness():
    clean = torch.randn(32,256,dtype=torch.complex64,generator=torch.Generator().manual_seed(19))
    original, shared = (PairedNoiseViews('cpu',shared_action=value) for value in (False,True))
    source = lambda: torch.Generator().manual_seed(271)
    left, left_witness = original(clean, source())
    right, right_witness = shared(clean, source())
    torch.testing.assert_close(left,right,rtol=0.,atol=0.)
    assert original.selected==shared.selected and original.acted==shared.acted
    mask = (right_witness-left_witness).abs().amax(-1)>1e-5
    assert mask.sum()==shared.acted and mask.any()
    assert ((right[mask]-right_witness[mask]).abs().amax(dim=-1)>1e-5).all()


def test_refinement_preserves_parent_at_start_and_cannot_update_first_stage():
    class Tracker(torch.nn.Module):
        def track(self, parts):
            iq = torch.complex(*parts.unbind(dim=2))
            return iq, torch.ones_like(iq), torch.ones_like(iq.real[...,0])

    model = train.NativeTaskRestorer(Tracker())
    iq = torch.randn(1,8192,dtype=torch.complex64,generator=torch.Generator().manual_seed(7))
    with torch.no_grad():
        parent = model(iq)
    model.core.refine = NoisePairRefiner()
    first, final = model(iq,return_intermediate=True)
    torch.testing.assert_close(first,parent,rtol=0.,atol=0.)
    torch.testing.assert_close(final,parent,rtol=0.,atol=0.)
    with torch.no_grad():
        model.core.refine.time.edit.weight.fill_(.01)
    _, final = model(iq,return_intermediate=True)
    fusion, restore = torch.autograd.grad(final.real.mean(),
        (model.core.time_edit.weight,model.core.refine.time.edit.weight),allow_unused=True)
    assert fusion is None and restore is not None and restore.abs().sum()>0

    def unexpected_stage_two(*args):
        raise AssertionError('First-stage readout must bypass refinement')

    with model.core.refine.register_forward_pre_hook(unexpected_stage_two), torch.no_grad():
        torch.testing.assert_close(model(iq,coarse_only=True),first,rtol=0.,atol=0.)


def test_raw_formal_receipt_preserves_w24_parent_and_source_control():
    for run,stages in train.RUNS.items():
        mode=stages[-1]
        parent=joint.source_parent(run)
        cases=joint.FULL_CASES
        for case in cases:
            proof=joint.formal_receipt(case,run)
            assert proof['reference_run']==parent
            expected=[joint.FORMAL_ARMS[parent],*joint.formal_arms(run)]
            assert proof['modes'][-len(expected):]==expected
        scores={name:dict(six_correct=1487,low_urban_correct=187,by_scene={'clean':297})
                for name in (*joint.parent_modes(parent),*stages)}
        assert not joint.qualifies(scores,mode)
        scores[mode].update(six_correct=1492,low_urban_correct=189)
        assert joint.qualifies(scores,mode)
        scores[mode]['by_scene']['clean']=296
        assert not joint.qualifies(scores,mode)


@pytest.mark.parametrize('run',(train.ALIGN_RUN,train.INNOVATION_RUN,train.PHASE_RUN,train.DIRECTION_RUN))
def test_fixed_native_arms_enter_formal_without_reading_source_scores(monkeypatch, tmp_path, run):
    def unexpected_read(*args):
        pytest.fail('Formal prediction tried to open a source score before the baseline')

    def baseline_reached(*args):
        raise RuntimeError('fixed formal baseline reached')

    monkeypatch.setattr(joint,'read',unexpected_read)
    monkeypatch.setattr(joint,'formal_reference',baseline_reached)
    runtime=dict(run_root=tmp_path,run_id=run)
    with pytest.raises(RuntimeError,match='fixed formal baseline reached'):
        joint.predict_formal(runtime,joint.FULL_CASES[0])


def test_alignment_separates_reversed_and_orthogonal_iq_with_selection():
    desired=torch.ones(3,2,4,dtype=torch.complex64)
    edit=torch.stack((-desired[0],desired[1]*(1+2j),desired[2]*(3+4j))).requires_grad_()
    selection=torch.tensor([True,True,False])
    scale=1.001
    torch.testing.assert_close(edit_parts(edit,desired,selection),torch.tensor([2.,2.])/scale)
    total=edit_parts(edit,desired,selection).sum()
    torch.testing.assert_close(total,(edit[:2]-desired[:2]).abs().square().mean()/scale)
    total.backward()
    assert torch.count_nonzero(edit.grad[2])==0
    torch.testing.assert_close(edit_parts(edit,desired,torch.zeros_like(selection)),torch.zeros(2))


def test_strong_source_selection_preserves_mixing_and_identifies_additive_pairs():
    clean=torch.ones(32,8,16,dtype=torch.complex64)
    mixes=[StrongSourceMix(lambda signal, rng: 2*signal,torch.Generator().manual_seed(19),
                         'cpu',enabled=True) for _ in range(2)]
    original=mixes[0](clean,torch.Generator().manual_seed(7))
    marked,selected=mixes[1](clean,torch.Generator().manual_seed(7),return_selection=True)
    torch.testing.assert_close(marked,original,rtol=0,atol=0)
    assert 0<selected.sum()<len(clean)
    assert torch.equal(selected,(marked!=2*clean).any(dim=(1,2)))


def test_spectral_innovation_starts_at_parent_and_retains_chirp_complex_phase():
    observed=torch.randn(2,8,1024,dtype=torch.complex64,generator=torch.Generator().manual_seed(17))
    coarse=(.7*observed).requires_grad_()
    model=SpectralInnovation()
    edit=model(observed,coarse)
    torch.testing.assert_close(edit,torch.zeros_like(observed),rtol=0,atol=0)
    loss=(coarse+edit-observed).abs().square().mean()
    gain_gradient,coarse_gradient=torch.autograd.grad(loss,(model.gain[-1].weight,coarse))
    assert gain_gradient.abs().sum()>0 and coarse_gradient.abs().sum()>0

    with torch.no_grad():
        model.gain[-1].bias.fill_(.7)
    phase=torch.exp(1j*torch.arange(8))[None,:,None]
    update=model(observed,coarse)
    torch.testing.assert_close(model(observed*phase,coarse*phase),update*phase,rtol=2e-5,atol=2e-6)
    torch.testing.assert_close(update.mean(-1),torch.zeros_like(update[...,0]),rtol=0,atol=2e-7)
    torch.testing.assert_close(model(observed,observed),torch.zeros_like(observed),rtol=0,atol=0)


def test_phase_aware_innovation_can_turn_a_frequency_residual_with_bounded_gain():
    control=SpectralInnovation(phase_aware=True)
    candidate=SpectralInnovation(phase_aware=True,quadrature=True)
    x=torch.zeros(1,8,1024,dtype=torch.complex64)
    x[...,9]=torch.exp(1j*torch.arange(8))
    residual=torch.fft.ifft(x,norm='ortho')
    y,coarse=1.5*residual,.5*residual
    torch.testing.assert_close(candidate(y,coarse),torch.zeros_like(y))
    assert torch.autograd.grad(candidate(y,coarse).imag[...,0].sum(),
                               candidate.gain[-1].bias)[0][1].abs()>0
    with torch.no_grad():
        control.gain[-1].bias.fill_(2.)
        candidate.gain[-1].bias.copy_(torch.tensor([0.,2.]))
    scale=1/(5**.5)
    torch.testing.assert_close(control(y,coarse),scale*residual)
    torch.testing.assert_close(candidate(y,coarse),1j*scale*residual)
    theta=torch.exp(1j*torch.arange(8))[None,:,None]
    torch.testing.assert_close(candidate(theta*y,theta*coarse),theta*candidate(y,coarse))
    dc=torch.ones_like(y)
    torch.testing.assert_close(candidate(dc,dc*.5),torch.zeros_like(y))
    assert torch.isfinite(candidate(y,y*0)).all()
    with pytest.raises(ValueError,match='phase'):
        SpectralInnovation(quadrature=True)


def test_direction_projection_removes_complex_coarse_component_in_deployed_center():
    rng=torch.Generator().manual_seed(402)
    coarse=torch.randn(3,8192,dtype=torch.complex64,generator=rng)
    coarse[1].zero_()
    proposal=(torch.randn(3,8192,dtype=torch.complex64,generator=rng)+(2+1j)*coarse).requires_grad_()

    class Frontend:
        def aperture(self, iq):
            chirps=iq.unflatten(-1,(8,1024))
            return chirps[..., :432]+.2*chirps.roll(1,dims=1)[...,24:456]

    frontend=Frontend()
    direction=remove_coarse_component(proposal,coarse,frontend)
    axis=frontend.aperture(coarse)[..., :384]
    selected=torch.tensor([True,False,False])
    inner=(frontend.aperture(direction)[..., :384]*axis.conj()).mean((1,2))
    torch.testing.assert_close(inner,torch.zeros_like(inner),rtol=0,atol=2e-6)
    torch.testing.assert_close(direction[1],proposal[1],rtol=0,atol=0)
    y=frontend.aperture(proposal.detach())[..., :384]
    target=frontend.aperture(-proposal.detach())[..., :384]
    loss=projected_pair_loss(frontend.aperture(direction)[..., :384],axis,target-y,selected)
    assert loss>0
    loss.backward()
    assert torch.count_nonzero(proposal.grad[1:])==0 and torch.count_nonzero(proposal.grad[0])>0

    head=DirectionRefiner(projected=True)
    framed=coarse.unflatten(-1,(8,1024))
    torch.testing.assert_close(head(framed,framed*.75),torch.zeros_like(framed))


def test_native_aperture_projection_annihilates_the_exact_visible_functionals():
    idx=torch.tensor([[[0,1],[2,3]]])
    taps=torch.tensor([[.5,.5],[.5,.5]])
    edges=torch.tensor([[4]])
    subtract=InvisibleIQ(idx,taps,edges,width=6)
    raw=torch.tensor([[1j,2j,3j,4j,5j,6j],[0j,2j,2j,2j,1j,0j]])
    invisible=subtract(raw)
    torch.testing.assert_close(invisible[:,[0,2]]+invisible[:,[1,3]],torch.zeros(2,2,dtype=raw.dtype))
    torch.testing.assert_close(invisible[:,4],torch.zeros(2,dtype=raw.dtype))
    torch.testing.assert_close(invisible[:,5],raw[:,5])
    torch.testing.assert_close(subtract(invisible),invisible)
    assert not subtract.state_dict()


def test_local_reliability_changes_position_only_and_starts_at_identical_parent():
    raw=torch.ones(1,8192,dtype=torch.complex64)
    raw[:,192:256]*=3
    rms=torch.ones(1,8,1)
    local,pooled=(reliability(raw,rms,localized=flag) for flag in (True,False))
    assert local.shape==pooled.shape==(1,8,1,1024)
    assert local[0,0,0,224]>pooled[0,0,0,224]
    assert torch.count_nonzero(pooled-pooled[...,:1])==0
    torch.testing.assert_close(local[:,1:],pooled[:,1:])
    parent=PacketUNet(across_chirps=True,center=1024)
    model=PacketUNet(across_chirps=True,center=1024,with_reliability=True)
    merged=parent.state_dict()
    merged['reliability_stem.weight']=model.reliability_stem.weight.detach()
    model.load_state_dict(merged,strict=True)
    iq=raw.unflatten(-1,(8,1024))
    observation=(iq,rms,rms,iq[...,:48])
    expected=parent(*observation)[-1]
    actual=model(*observation,reliability=local)[-1]
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    actual.real.square().mean().backward()
    assert model.reliability_stem.weight.grad.abs().sum()>0


def test_nonstationary_source_shape_preserves_energy_and_is_reproducible():
    noise=torch.randn(8,8,1024,dtype=torch.complex64,generator=torch.Generator().manual_seed(1))
    a,b=(LocalizedWhite(torch.Generator().manual_seed(3))(noise) for _ in (0,1))
    torch.testing.assert_close(a,b,rtol=0,atol=0)
    torch.testing.assert_close(a.abs().square().sum((1,2)),noise.abs().square().sum((1,2)))
    assert not torch.equal(a,noise)


def test_visible_wave_objective_projects_the_error_and_its_gradient():
    idx=torch.tensor([[[0,1],[2,3]]])
    taps=torch.tensor([[.5,.5],[.5,.5]])
    edges=torch.tensor([[4]])
    operator=InvisibleIQ(idx,taps,edges,width=6)
    x=torch.tensor([[1+1j,2j,3j,4j,5j,6j]],requires_grad=True)
    value=VisibleIQ.apply(x,operator)
    torch.testing.assert_close(value,x-operator(x))
    weight=torch.tensor([[1+0j,2j,-1j,1j,0j,9j]])
    g,=torch.autograd.grad((value*weight.conj()).real.sum(),x)
    torch.testing.assert_close(g,weight-operator(weight))
    assert g[0,-1]==0
    torch.testing.assert_close(operator(g),torch.zeros_like(g))


def test_same_anchor_distance_distinguishes_motion_towards_a_false_rival():
    target = torch.tensor([0])
    anchor = torch.tensor([[1.,0.]],requires_grad=True)
    table = torch.tensor([[1.,0.],[.95,.3122499],[-1.,0.]],requires_grad=True)
    toward = torch.tensor([[.99,.14106736]],requires_grad=True)
    away = torch.tensor([[.99,-.14106736]],requires_grad=True)
    positive = [clean_geometry_risk(z,anchor,target,table,relative=False) for z in (toward,away)]
    relative = [clean_geometry_risk(z,anchor,target,table,relative=True) for z in (toward,away)]
    torch.testing.assert_close(*positive)
    assert relative[0]>positive[0] and relative[1]==0
    relative[0].backward()
    assert toward.grad.abs().sum()>0
    assert anchor.grad is None and table.grad is None


def test_empty_negative_does_not_enter_when_clean_prototype_matches_true_label():
    labels=torch.tensor([1])
    clean=torch.tensor([[1.,0.]],requires_grad=True)
    table=torch.tensor([[.9,.4358899],[1.,0.],[-1.,0.]])
    identical=clean.detach().clone().requires_grad_()
    risk=clean_geometry_risk(identical,clean,labels,table,relative=True)
    risk.backward()
    assert risk==0 and torch.count_nonzero(identical.grad)==0
