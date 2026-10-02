import copy
import pytest
import torch
from experiments.cvs_coordinate_identity.model import build,VARIANTS
from experiments.cvs_coordinate_identity.usage import MODES,counterfactual_logits,assert_state_unchanged


@pytest.mark.parametrize('variant',VARIANTS)
def test_zero_initialized_counterfactuals_equal_actual_forward(variant):
    torch.set_num_threads(2);torch.manual_seed(9);m=build(variant).eval();x=torch.randn(3,2,256)
    before=copy.deepcopy(m.state_dict());d=counterfactual_logits(m,x)
    assert tuple(d)==MODES
    with torch.no_grad():actual=m(x)
    for logits in d.values():torch.testing.assert_close(logits,actual,rtol=0,atol=0)
    assert_state_unchanged(m,before)


@pytest.mark.parametrize('variant',VARIANTS)
def test_coordinate_reference_keeps_bias_unit_gain_removes_it(variant):
    torch.set_num_threads(2);torch.manual_seed(11);m=build(variant).eval();x=torch.randn(2,2,256)
    with torch.no_grad():
        m.conditioner.weight.copy_(torch.randn_like(m.conditioner.weight)*.2)
        m.conditioner.bias.copy_(torch.linspace(-.5,.5,160))
        aligned,_,_,d=m.coordinates(x);u=m.core.features(aligned)
        expected=m.classify_features(u*(1+.25*torch.tanh(m.conditioner.bias)))
        unit=m.classify_features(u);actual=m(x)
    before=copy.deepcopy(m.state_dict());out=counterfactual_logits(m,x)
    torch.testing.assert_close(out['coordinate_reference'],expected,rtol=0,atol=0)
    torch.testing.assert_close(out['unit_gain'],unit,rtol=0,atol=0)
    torch.testing.assert_close(out['trained'],actual,rtol=0,atol=0)
    assert_state_unchanged(m,before)
    assert not torch.equal(out['coordinate_reference'],out['unit_gain'])
    for mode in MODES:
        alone=counterfactual_logits(m,x[:1])[mode]
        torch.testing.assert_close(alone,out[mode][:1],rtol=2e-5,atol=2e-5)


def test_frozen_state_and_eval_guards():
    torch.set_num_threads(2);m=build(VARIANTS[0])
    with pytest.raises(ValueError,match='eval'):counterfactual_logits(m,torch.zeros(1,2,256))
    m.eval();before=copy.deepcopy(m.state_dict())
    out=counterfactual_logits(m,torch.zeros(2,2,256));assert all(torch.isfinite(v).all() for v in out.values())
    assert_state_unchanged(m,before)
    with torch.no_grad():m.conditioner.bias.add_(1)
    with pytest.raises(ValueError,match='state'):assert_state_unchanged(m,before)
