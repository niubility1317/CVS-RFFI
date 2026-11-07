import pytest
import torch

from experiments.pamf_phase3_restoration.bounded_projected_prior import BoundedProjectedPriorDetail,bound_correction
from experiments.pamf_phase3_restoration.projected_prior_detail import ProjectedPriorDetail
from experiments.pamf_phase3_restoration import bounded_projected_prior_trial as candidate
from experiments.pamf_phase3_restoration import projected_prior_detail_trial as control


def test_cap_preserves_small_correction_and_bounds_extreme_with_finite_gradient():
    received=torch.ones(2,2,256)
    correction=torch.stack((torch.ones(2,256),torch.full((2,256),10.))).requires_grad_()
    bounded,scale=bound_correction(received,correction)
    assert scale[0].item()==1.
    torch.testing.assert_close(bounded[0],correction[0],rtol=0,atol=0)
    torch.testing.assert_close(bounded.flatten(1).norm(dim=1)[1],2*received.flatten(1).norm(dim=1)[1])
    (bounded*torch.arange(512).reshape(1,2,256)).sum().backward()
    assert torch.isfinite(correction.grad).all() and correction.grad[1].abs().sum()>0
    zero,zero_scale=bound_correction(torch.zeros_like(received),torch.zeros_like(received))
    assert torch.equal(zero,torch.zeros_like(received)) and torch.all(zero_scale==1)
    huge,_=bound_correction(received,torch.full_like(received,1e20))
    assert torch.isfinite(huge).all() and torch.all(huge.flatten(1).double().norm(dim=1)<=2*received.flatten(1).double().norm(dim=1)+1e-5)
    with pytest.raises(ValueError):bound_correction(received,torch.full_like(received,float('nan')))


def test_candidate_keeps_original_scratch_initialization_and_budget():
    a=control.factory(control.ARM);b=candidate.factory(candidate.ARM)
    assert a.state_dict().keys()==b.state_dict().keys()
    assert all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items())
    x=control.configuration(control.ARM);y=candidate.configuration(candidate.ARM)
    for key in ('model_seed','epochs','updates','batch','source_draws','objective'):
        assert x[key]==y[key]
    assert y['original_parent_weights_loaded'] is False


def test_train_and_inference_use_the_same_bounded_correction(monkeypatch):
    received=torch.ones(2,2,256);clean=torch.ones_like(received);baseline=received*.5
    raw=torch.stack((torch.full((2,256),.1),torch.full((2,256),10.))).requires_grad_()
    shrinkage=torch.full_like(received,.2)
    def original(self,received,source_clean,*,prior=None):
        aux={'base_iq':baseline}
        if source_clean is not None:
            aux.update(projected_prior_update=raw,source_shrinkage=shrinkage,
                       source_noiseless_corrected_iq=source_clean-shrinkage+raw)
        return baseline+raw,aux
    monkeypatch.setattr(ProjectedPriorDetail,'_readout',original)
    model=BoundedProjectedPriorDetail(2026092701)
    trained,aux=model._readout(received,clean);inferred,_=model._readout(received,None)
    torch.testing.assert_close(trained,inferred)
    torch.testing.assert_close(aux['source_noiseless_corrected_iq'],clean-shrinkage+aux['projected_prior_update'])
    assert aux['correction_scale'][0]==1 and aux['correction_scale'][1]<1
    assert torch.isfinite(torch.autograd.grad(trained.square().sum(),raw)[0]).all()


def test_training_callback_accepts_and_forwards_source_context(monkeypatch):
    received=torch.ones(2,2,256);indices=torch.tensor([4,7]);seen={}
    def readout(model,reader,x,clean,features,scale,objective,*,source_indices,epoch):
        seen.update(indices=source_indices,epoch=epoch)
        return x,{'unbounded_projected_update':torch.zeros_like(x),'correction_scale':torch.ones(2)},torch.zeros(2),{}
    monkeypatch.setattr(candidate,'shrinkage_training_readout',readout)
    _,_,_,terms=candidate.training_readout(None,None,received,received,None,None,None,source_indices=indices,epoch=3)
    assert seen['indices'] is indices and seen['epoch']==3
    assert torch.equal(terms['correction_limited_fraction'],torch.zeros(2))
