import copy
import pytest
import torch
from experiments.cvs_additive_identity.model import build,VARIANTS
from experiments.cvs_additive_identity.usage import MODES,counterfactual_logits,assert_state_unchanged

@pytest.mark.parametrize('variant',VARIANTS)
def test_frozen_same_forward_all_reference_modes_and_state(variant):
    torch.set_num_threads(2);torch.manual_seed(37);model=build(variant).eval();x=torch.randn(3,2,256)
    with torch.no_grad():
        model.conditioner.weight.normal_(0,.1);model.conditioner.bias.fill_(.1)
    before=copy.deepcopy(model.state_dict());r=counterfactual_logits(model,x)
    assert tuple(r)==MODES and torch.equal(r['trained'],model(x))
    aligned,_,_,d=model.coordinates(x);u=model.core.features(aligned)
    for mode in MODES:
        altered=d.clone()
        if mode in ('frequency_reference','coordinate_reference'):altered[:,:2]=0
        if mode in ('quality_reference','coordinate_reference'):altered[:,2]=0
        f=u if mode=='zero_delta' else u+.25*u.norm(dim=1,keepdim=True)*torch.tanh(model.conditioner(altered))
        assert torch.equal(r[mode],model.classify_features(f))
    # Reference keeps learned bias; zero_delta removes both input and bias.
    assert not torch.equal(r['coordinate_reference'],r['zero_delta'])
    assert all(not a.requires_grad and torch.isfinite(a).all() for a in r.values())
    assert_state_unchanged(model,before)

@pytest.mark.parametrize('variant',VARIANTS)
def test_zero_initialized_modes_exactly_identical_and_training_rejected(variant):
    torch.set_num_threads(2);m=build(variant);x=torch.zeros(2,2,256)
    with pytest.raises(ValueError,match='Frozen eval'):counterfactual_logits(m,x)
    m.eval();r=counterfactual_logits(m,x)
    assert all(torch.equal(r['trained'],a) for a in r.values())

def test_buffer_or_parameter_mutation_detected():
    m=build(VARIANTS[0]).eval();before=copy.deepcopy(m.state_dict())
    with torch.no_grad():m.conditioner.bias.add_(.001)
    with pytest.raises(ValueError,match='changed frozen'):assert_state_unchanged(m,before)
