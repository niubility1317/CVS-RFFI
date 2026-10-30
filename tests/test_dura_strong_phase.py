import numpy as np
import torch
from types import SimpleNamespace
from experiments.pamf_dura_direct_train import strong_phase as trial
from experiments.pamf_dura_direct_train.strong_phase import strong_phase_component,learning_rate


def test_phase_component_is_zero_for_positive_gain_and_has_expected_rotation_cost():
    torch.manual_seed(8);x=torch.randn(3,2,256)
    zero,share=strong_phase_component(.7*x,x)
    torch.testing.assert_close(zero,torch.zeros_like(zero),atol=3e-7,rtol=0)
    rotated=torch.stack((-x[:,1],x[:,0]),1)
    phase,share=strong_phase_component(rotated,x)
    torch.testing.assert_close(phase,2*share,atol=3e-6,rtol=3e-6)


def test_phase_partition_matches_independent_numpy_and_is_bounded_by_iq_error():
    rng=np.random.default_rng(9);x=rng.normal(size=(4,2,256)).astype('float32');r=x+.3*rng.normal(size=x.shape).astype('float32')
    value,share=strong_phase_component(torch.tensor(r),torch.tensor(x))
    xf=np.fft.fft(x[:,0].astype(float)+1j*x[:,1].astype(float),norm='ortho')
    rf=np.fft.fft(r[:,0].astype(float)+1j*r[:,1].astype(float),norm='ortho')
    order=np.argsort(-abs(xf)**2,axis=1,kind='stable')[:,:64]
    c=rf*xf.conj();part=2*(abs(c)-c.real)
    expected=np.take_along_axis(part,order,axis=1).sum(1)/(abs(xf)**2).sum(1)
    np.testing.assert_allclose(value.detach().cpu().tolist(),expected,atol=5e-7,rtol=5e-6)
    error=np.square(r.astype(float)-x).sum((1,2))/np.square(x).sum((1,2))
    assert np.all(np.asarray(value.detach().cpu().tolist())<=error+1e-6)


def test_zero_output_has_finite_gradient_and_clean_target_is_detached():
    x=torch.randn(2,2,256,requires_grad=True);r=torch.zeros_like(x,requires_grad=True)
    phase,_=strong_phase_component(r,x);phase.sum().backward()
    assert x.grad is None and r.grad is not None and torch.isfinite(r.grad).all()
    assert r.grad.abs().sum()>0


def test_e30_learning_rate_budget():
    assert learning_rate(246)==.001 and learning_rate(7409)==.00002


def test_one_student_forward_and_phase_weight_preserve_the_base_objective(monkeypatch):
    q=torch.tensor(1.,requires_grad=True);x=torch.randn(2,2,256);restored=torch.stack((-x[:,1],x[:,0]),1)
    calls=[]
    terms=dict(iq_nmse=q.square(),prior_nmse=2*q.square(),spectral_error=3*q.square(),action_probe_nmse=4*q.square(),log_variance_mse=99*q.square())
    def forward(*args,**kwargs):
        calls.append(kwargs);return q*99,terms,restored
    monkeypatch.setattr(trial,'source_objective',forward)
    loss,logged=trial.objective(None,None,x,None,None,None)
    expected=1+.1*2+.05*3+.05*4+logged['strong_phase_nmse']
    torch.testing.assert_close(loss,expected)
    assert calls==[dict(return_restored=True)]
    loss.mean().backward();torch.testing.assert_close(q.grad,torch.tensor(3.1))


def test_fixed_budget_and_module_routes_without_training_teacher(monkeypatch):
    base=trial.base
    for key,value in vars(base).copy().items():monkeypatch.setattr(base,key,value)
    trial.configure();c=base.configuration()
    assert (c['epochs'],c['updates'],c['parameters'])==(30,7410,266951)
    assert c['strong_phase_extra_weight']==1 and c['strong_frequency_bin_count']==64
    assert c['additional_training_FFTs_per_student_forward']==2 and c['additional_inference_FFTs']==0
    assert not c['training_F_access'] and c['teacher_model'] is None and c['ancestor_checkpoints']==[]
    args=SimpleNamespace(root='root',commit='a'*40)
    assert all(base.command(args,m,'scratch_joint','target')[3]==trial.MODULE for m in ('worker','score'))
