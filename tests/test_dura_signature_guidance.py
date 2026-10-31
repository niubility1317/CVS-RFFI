import pytest
import torch
from experiments.pamf_dura_efficiency.signature_iq import signature_features
from experiments.pamf_dura_efficiency.signature_guidance import signature_guide,CAP


class Reader(torch.nn.Module):
    def __init__(self):
        super().__init__();self.core=torch.nn.Module();self.core.id_backbone=torch.nn.Module()
        self.core.id_backbone.t2=torch.nn.Conv1d(2,4,3,padding=1)
        self.sequence_pool=torch.nn.Sequential(torch.nn.AdaptiveAvgPool1d(8),torch.nn.Flatten(),torch.nn.Linear(32,160))
        self.reverse_calls=0
        def observe(m,a,v):
            if v.requires_grad:v.register_hook(self.reverse)
        self.sequence_pool.register_forward_hook(observe)
    def reverse(self,g):self.reverse_calls+=1;return g
    def features(self,x):return self.sequence_pool(self.core.id_backbone.t2(x))


def test_sig_gradient_is_injected_once_with_half_iq_norm_ceiling():
    torch.manual_seed(45);reader=Reader().eval().requires_grad_(False)
    clean=torch.randn(3,2,256);restored=(clean+.1*torch.randn_like(clean)).requires_grad_(True)
    target=signature_features(reader,clean).detach();before={k:v.clone() for k,v in reader.state_dict().items()}
    guide,info=signature_guide(reader,restored,clean,target)
    assert reader.reverse_calls==1
    expected=info['weighted_gradient'];guide.sum().backward()
    torch.testing.assert_close(restored.grad,expected)
    assert reader.reverse_calls==1 and torch.all(info['bounded_pressure']<=CAP+1e-6)
    assert expected.abs().sum()>0 and all(p.grad is None for p in reader.parameters())
    assert all(torch.equal(v,before[k]) for k,v in reader.state_dict().items())


def test_perfect_iq_receives_zero_sig_gradient_even_with_inconsistent_reference():
    reader=Reader().eval().requires_grad_(False);clean=torch.randn(2,2,256)
    restored=clean.clone().requires_grad_(True);target=signature_features(reader,clean).detach()+.1
    guide,info=signature_guide(reader,restored,clean,target);guide.sum().backward()
    assert torch.count_nonzero(restored.grad)==0 and torch.count_nonzero(info['weight'])==0
    assert torch.isfinite(guide).all() and not info['pressure_defined'].any()
    with pytest.raises(ValueError):signature_guide(reader.train(),restored,clean,target)
