import pytest
import torch
from experiments.pamf_dura_efficiency.signature_iq import signature_features, signature_steps, project


class Reader(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.core = torch.nn.Module(); self.core.id_backbone = torch.nn.Module()
        self.core.id_backbone.t2 = torch.nn.Conv1d(2, 4, 3, padding=1)
        self.sequence_pool = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(4*8, 160))
        self.after = torch.nn.Linear(160, 160)
        self.after_calls = 0
    def features(self, x):
        local = {}
        hook = self.core.id_backbone.t2.register_forward_hook(lambda m,a,v: local.update(h=v))
        try:
            self.core.id_backbone.t2(x)
            self.after_calls += 1
            base = self.after(x.new_zeros(len(x), 160))
        finally:
            hook.remove()
        return base + self.sequence_pool(local['h'])


def test_actual_prefix_matches_full_signature_value_and_input_vjp_without_suffix():
    torch.manual_seed(11); reader = Reader().eval().requires_grad_(False)
    x = torch.randn(3, 2, 8, requires_grad=True); seen = {}
    hook = reader.sequence_pool.register_forward_hook(lambda m,a,v: seen.update(signature=v))
    reader.features(x); hook.remove(); expected = seen['signature']
    direction = torch.randn_like(expected)
    reference_grad = torch.autograd.grad((expected*direction).sum(), x)[0]
    calls = reader.after_calls
    actual = signature_features(reader, x)
    actual_grad = torch.autograd.grad((actual*direction).sum(), x)[0]
    torch.testing.assert_close(actual, expected); torch.testing.assert_close(actual_grad, reference_grad)
    assert reader.after_calls == calls
    assert all(not module._forward_hooks for module in reader.modules())


def test_prefix_failure_cleans_both_hooks_and_requires_frozen_state():
    reader = Reader().eval().requires_grad_(False); x = torch.randn(2,2,8)
    with pytest.raises(ValueError): signature_features(reader.train(), x)
    reader.eval(); bad = reader.core.id_backbone.t2.register_forward_pre_hook(
        lambda *args: (_ for _ in ()).throw(RuntimeError('native prefix failure')))
    with pytest.raises(RuntimeError, match='native prefix failure'): signature_features(reader, x)
    bad.remove(); assert all(not m._forward_hooks for m in reader.modules())


def test_fixed_steps_respect_per_record_ball_and_keep_reader_frozen():
    torch.manual_seed(12); reader = Reader().eval().requires_grad_(False)
    anchor = torch.randn(3,2,8); clean = anchor + .1*torch.randn_like(anchor)
    before = {k:v.clone() for k,v in reader.state_dict().items()}
    target = signature_features(reader, clean).detach()
    value, losses = signature_steps(reader, anchor, target)
    relative = (value-anchor).square().sum((1,2)) / anchor.square().sum((1,2))
    assert losses.shape == (4,3) and torch.all(relative <= .00010001)
    assert not value.requires_grad and all(p.grad is None for p in reader.parameters())
    assert all(torch.equal(v,before[k]) for k,v in reader.state_dict().items())
    torch.testing.assert_close(project(anchor, torch.zeros_like(anchor)), anchor)
    oracle = project(anchor, clean-anchor)
    torch.testing.assert_close((oracle-anchor).square().sum((1,2))/anchor.square().sum((1,2)), torch.full((3,),.0001))
