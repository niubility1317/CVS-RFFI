import math

import torch

from experiments.pamf_phase3_restoration.covariance_inverse import dense_action
from experiments.pamf_phase3_restoration.phase_supervised_covariance import (
    canonical_factors, circular_supervision, phase_supervised_action_loss, reference_basis,
    scratch_phase_supervised_covariance,
)


def test_reference_canonicalization_preserves_action_and_removes_gauge():
    dtype = torch.complex64
    ref = torch.tensor([1+.3j,.8+.2j,1-.1j,.5+.4j,1.2+.1j],dtype=dtype)
    basis = reference_basis(ref,2)
    h = torch.tensor([[.8+.3j,.2-.15j]],dtype=dtype)
    m = torch.tensor([[.9+.3j,1-.1j,.9+.2j,.8+.3j,1+.1j]],dtype=dtype)
    i = torch.tensor([[.04+.01j,.03+.02j,.06-.02j,.04+.03j,.02+.02j]],dtype=dtype)
    a = canonical_factors(h,m,i,ref,basis)
    q = torch.tensor(1.2+.7j,dtype=dtype)
    b = canonical_factors(h*q,m/q,i/q.conj(),ref,basis)
    torch.testing.assert_close(dense_action(*a),dense_action(h,m,i))
    torch.testing.assert_close(dense_action(*b),dense_action(h,m,i))
    direction = ((a[0]@basis.T)*ref.conj()).sum(1)
    torch.testing.assert_close(direction.imag,torch.zeros_like(direction.imag),atol=1e-6,rtol=0)
    assert (direction.real>0).all()
    # Regauging also changes FIR scale, which is intentionally preserved.
    torch.testing.assert_close(a[0]/a[0].norm(),b[0]/b[0].norm())


def test_phase_loss_measures_known_phase_and_is_gauge_invariant():
    ref = torch.tensor([1+.3j,.8+.2j,1-.1j,.5+.4j,1.2+.1j],dtype=torch.complex64)
    basis = reference_basis(ref,2)
    h = torch.tensor([[1.+0j,0j]],dtype=torch.complex64)
    shape = torch.exp(1j*torch.linspace(-.2,.2,5))[None]
    angle = torch.tensor([.31])
    m = 1.4*shape*torch.exp(1j*angle[:,None])
    phase = torch.tensor([-.12],requires_grad=True)
    _, canonical_main, _ = canonical_factors(h,m,torch.zeros_like(m),ref,basis)
    loss,err = circular_supervision(phase,shape,canonical_main)
    torch.testing.assert_close(loss,1-(angle-phase).cos())
    torch.testing.assert_close(err,(angle-phase).abs()*180/math.pi)
    q = torch.tensor(-.4+.7j,dtype=h.dtype)
    _, regauged_main, _ = canonical_factors(h*q,m/q,torch.zeros_like(m),ref,basis)
    other,_ = circular_supervision(phase,shape,regauged_main)
    torch.testing.assert_close(loss,other)
    loss.sum().backward()
    assert phase.grad.isfinite().all() and phase.grad.abs().sum()>0


def test_scratch_model_uses_phase_for_operator_training_and_deployment():
    G = scratch_phase_supervised_covariance(917)
    F = scratch_phase_supervised_covariance(917)
    assert all(torch.equal(value,F.state_dict()[key]) for key,value in G.state_dict().items())
    assert not G.initialization_report['ancestor_checkpoints']
    t = torch.linspace(-1,1,256)
    x = torch.stack((.6+t*.1,.25+t.square()*.1)).repeat(2,1,1)
    G.bind_prior(x[0].flatten(),.06*torch.eye(512))
    h = torch.zeros(2,2,64)
    h[:,0,0] = 1
    h[:,1,1] = .2
    m = torch.exp(1j*(.3+t[None,:]*.2)).expand(2,-1)
    image = (.06+.03j)*m
    coefficients = torch.stack((m.real,m.imag,image.real,image.imag),dim=1)
    aux = G.estimate(x,dense=False)
    loss,terms = phase_supervised_action_loss(aux,h,coefficients,torch.full((2,),.003))
    loss.mean().backward()
    for weight in (G.phase_head.angle.weight,G.kernel_head.weight,G.coefficient_head.gains.weight,
                   G.coefficient_head.slopes.weight,G.coefficient_head.local_phase.weight,G.noise_head.weight):
        assert weight.grad is not None and weight.grad.isfinite().all() and weight.grad.abs().sum()>0
    assert (terms['main_phase_absolute_degrees']>0).all()
    with torch.no_grad():
        raw = G(x)
        G.phase_head.angle.bias.add_(.2)
        updated,debug = G(x,return_aux=True)
        assert debug['relative_solve_residual'].max()<1e-8
        assert not torch.allclose(raw,updated)


def test_train_entry_preserves_scratch_budget_and_truth_last_pipeline(tmp_path,monkeypatch):
    import json
    from experiments.pamf_phase3_restoration import phase_supervised_covariance_trial as trial

    cfg = trial.configuration(trial.ARM)
    assert cfg['training_stages']==1 and cfg['training_posterior_calls']==0 and cfg['training_F0_forward_rows']==0
    assert cfg['new_G_initialization_checkpoints']==0 and len(cfg['source_draws'])==30
    assert cfg['matched_control']==trial.PREVIOUS_RUN+'/'+trial.PREVIOUS_ARM
    observed = {}
    monkeypatch.setattr(trial.control,'train',lambda *args,**kwargs:observed.update(kwargs))
    trial.train(object(),tmp_path)
    assert observed['build'] is trial.factory and observed['objective'] is phase_supervised_action_loss
    assert observed['dense_supervision'] is False
    old = tmp_path/'runs'/trial.PREVIOUS_RUN
    old.mkdir(parents=True)
    assert not trial.previous_gpu_finished(tmp_path)
    (old/f'terminal-{trial.PREVIOUS_ARM}-train.json').write_text(json.dumps({'exit_code':0}),encoding='utf-8')
    assert not trial.previous_gpu_finished(tmp_path)
    (old/f'terminal-{trial.PREVIOUS_ARM}-cross_both.json').write_text(json.dumps({'exit_code':0}),encoding='utf-8')
    assert trial.previous_gpu_finished(tmp_path)


def test_reject_unregistered_arm():
    import pytest
    from experiments.pamf_phase3_restoration.phase_supervised_covariance_trial import configuration

    with pytest.raises(ValueError,match='fixed coherent phase candidate'):
        configuration('other')
