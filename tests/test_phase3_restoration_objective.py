import pytest
import torch
from experiments.pamf_phase3_restoration.objective import PairedObjective, feature_guide, INPUT_GRADIENT_CAP


class Reader(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.projection=torch.nn.Linear(512,160)
        self.eval().requires_grad_(False)
    def features(self, x): return self.projection(x.flatten(1))


def test_paired_objective_detaches_clean_and_preserves_phase_sensitive_IQ_error():
    torch.set_num_threads(2); torch.manual_seed(1)
    clean=torch.randn(2,2,256,requires_grad=True); restored=clean.detach().clone().requires_grad_()
    objective=PairedObjective(); loss, terms=objective(restored,clean)
    assert torch.equal(loss,torch.zeros_like(loss))
    loss.mean().backward(); assert clean.grad is None and torch.isfinite(restored.grad).all()
    rotated=torch.stack((-clean[:,1],clean[:,0]),1).detach().requires_grad_()
    loss,terms=objective(rotated,clean)
    torch.testing.assert_close(terms['iq_nmse'],torch.full((2,),2.),atol=1e-6,rtol=1e-6)
    torch.testing.assert_close(terms['spectral_error'],torch.zeros(2),atol=1e-6,rtol=0)
    assert torch.isfinite(torch.autograd.grad(loss.sum(),rotated)[0]).all()


def test_feature_guide_detaches_reference_caps_actual_input_gradient_and_freezes_F():
    torch.set_num_threads(2); torch.manual_seed(2); reader=Reader()
    clean=torch.randn(3,2,256,requires_grad=True)
    restored=(clean.detach()+.0001*torch.randn_like(clean)).requires_grad_()
    target=torch.randn(3,160,requires_grad=True)
    before={k:v.clone() for k,v in reader.state_dict().items()}
    guide,info=feature_guide(reader,restored,clean,target)
    gradient=torch.autograd.grad(guide.sum(),restored,retain_graph=True)[0].flatten(1).norm(dim=1)
    assert (info['weight']<1).all()
    torch.testing.assert_close(gradient,INPUT_GRADIENT_CAP*info['IQ_gradient_norm'],atol=1e-10,rtol=2e-5)
    guide.sum().backward()
    assert target.grad is None and clean.grad is None
    assert all(p.grad is None for p in reader.parameters())
    assert all(torch.equal(v,before[k]) for k,v in reader.state_dict().items())


def test_objectives_reject_mismatched_source_pair_and_feature_target():
    with pytest.raises(ValueError,match='Matched source'):
        PairedObjective()(torch.zeros(2,2,256),torch.zeros(3,2,256))
    with pytest.raises(ValueError,match='Aligned source'):
        feature_guide(Reader(),torch.zeros(2,2,256,requires_grad=True),torch.zeros(2,2,256),torch.zeros(2,159))
