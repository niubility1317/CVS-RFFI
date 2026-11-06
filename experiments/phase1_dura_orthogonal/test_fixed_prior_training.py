import torch
from torch import nn
from experiments.phase1_dura_orthogonal.train_source import configure_internal_trainability


class TinyDura(nn.Module):
    def __init__(self):
        super().__init__()
        self.frozen = nn.Linear(2, 2)
        self.prior = nn.Linear(2, 2)
    def forward(self, x):
        return self.prior(self.frozen(x))


def test_fixed_prior_propagates_to_context_without_changing_prior():
    model = TinyDura()
    configure_internal_trainability(model, 'sim_action_fixed_prior')
    before = [p.detach().clone() for p in model.prior.parameters()]
    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=.01)
    model(torch.ones(2, 2)).square().sum().backward()
    assert any(p.grad is not None and torch.count_nonzero(p.grad) for p in model.frozen.parameters())
    assert all(not p.requires_grad and p.grad is None for p in model.prior.parameters())
    opt.step()
    assert all(torch.equal(x, p) for x, p in zip(before, model.prior.parameters()))


def test_existing_sim_action_keeps_both_trainable():
    model = TinyDura()
    configure_internal_trainability(model, 'sim_action_guard')
    assert all(p.requires_grad for p in model.parameters())
