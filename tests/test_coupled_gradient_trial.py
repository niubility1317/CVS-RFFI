import pytest
import torch

from experiments.pamf_phase3_restoration import coupled_gradient_trial as candidate


def test_scale_gradient_preserves_values_and_scales_only_backward():
    x=torch.tensor([2.,-3.],requires_grad=True)
    scaled=candidate.ScaleGradient.apply(x,.05)
    assert torch.equal(x,scaled)
    (scaled.square().sum()+x.square().sum()).backward()
    torch.testing.assert_close(x.grad,2*x.detach()*1.05)


def test_real_model_coupling_changes_only_physical_detail_gradient(monkeypatch):
    torch.set_num_threads(2);torch.manual_seed(12)
    model=candidate.factory(candidate.ARM).eval();x=torch.randn(1,2,256)
    head=model.physical.coefficient_head.weight;detail=model.detail.output.weight
    monkeypatch.setattr(candidate,'COUPLING',1.)
    output,aux=model(x,return_aux=True)
    full=torch.autograd.grad(output.square().mean(),(head,detail),retain_graph=True)
    supervised=torch.autograd.grad(aux['action_matrix'].square().mean(),head)[0]
    monkeypatch.setattr(candidate,'COUPLING',.05)
    reduced,aux2=model(x,return_aux=True)
    bounded=torch.autograd.grad(reduced.square().mean(),(head,detail),retain_graph=True)
    supervised2=torch.autograd.grad(aux2['action_matrix'].square().mean(),head)[0]
    torch.testing.assert_close(output,reduced,rtol=0,atol=0)
    torch.testing.assert_close(bounded[0],.05*full[0],rtol=2e-4,atol=1e-7)
    torch.testing.assert_close(bounded[1],full[1],rtol=1e-5,atol=1e-7)
    torch.testing.assert_close(supervised2,supervised,rtol=0,atol=0)
    assert full[0].norm()>0 and full[1].norm()>0 and supervised.norm()>0


def test_wrong_candidate_is_rejected():
    with pytest.raises(ValueError):candidate.factory('other')
