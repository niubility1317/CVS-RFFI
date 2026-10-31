import pytest
import torch
from experiments.pamf_dura_efficiency.branch_features import components,replacements,BRANCHES


class Head(torch.nn.Module):
    def __init__(self):
        super().__init__();self.base_norm=torch.nn.LayerNorm(160);self.pa_norm=torch.nn.LayerNorm(160);self.gain=torch.nn.Parameter(torch.linspace(-1,1,160))
class Reader(torch.nn.Module):
    def __init__(self):
        super().__init__();self.core=torch.nn.Module();self.core.id_backbone=torch.nn.Module()
        self.core.id_backbone.cls_head=Head();self.response_projection=torch.nn.Linear(160,160);self.sequence_pool=torch.nn.Linear(160,160)
        self.response_gain=torch.nn.Parameter(torch.linspace(1,-1,160))
    def features(self,x):
        head=self.core.id_backbone.cls_head
        base=head.base_norm(x);pa=head.gain.tanh()[None]*head.pa_norm(x.square())
        response=self.response_gain.tanh()[None]*self.response_projection(x)
        return base+pa+response+self.sequence_pool(x)


def test_components_reproduce_actual_weighted_sum_and_hook_cleanup():
    torch.manual_seed(41);reader=Reader().eval().requires_grad_(False);x=torch.randn(3,160)
    expected=reader.features(x);actual,parts=components(reader,x)
    assert set(parts)==set(BRANCHES);torch.testing.assert_close(actual,expected)
    head=reader.core.id_backbone.cls_head
    torch.testing.assert_close(parts['PA'],head.gain.tanh()[None]*head.pa_norm(x.square()))
    torch.testing.assert_close(parts['response'],reader.response_gain.tanh()[None]*reader.response_projection(x))
    assert all(not m._forward_hooks for m in reader.modules())
    torch.testing.assert_close(reader.features(x),expected)


def test_branch_replacements_change_exactly_one_weighted_component():
    reader=Reader().eval().requires_grad_(False);x=torch.randn(2,160);y=torch.randn_like(x)
    actual,p=components(reader,x);clean,c=components(reader,y);cases=replacements(actual,p,clean,c)
    for name in BRANCHES:torch.testing.assert_close(cases[name]-actual,c[name]-p[name],atol=5e-7,rtol=1e-5)
    assert cases['native'] is actual and cases['all_clean'] is clean


def test_frozen_state_required_and_hooks_removed_after_forward_failure():
    reader=Reader().eval().requires_grad_(False);x=torch.randn(2,160)
    with pytest.raises(ValueError):components(reader.train(),x)
    with pytest.raises(ValueError):components(reader.eval().requires_grad_(True),x)
    reader.requires_grad_(False)
    reader.sequence_pool.register_forward_pre_hook(lambda *args:(_ for _ in ()).throw(RuntimeError('test failure')))
    with pytest.raises(RuntimeError):components(reader,x)
    assert all(not m._forward_hooks for m in reader.modules())
