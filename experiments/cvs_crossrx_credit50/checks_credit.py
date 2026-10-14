"""Actual AdamW and fixed-readout counterexamples on public synthetic IQ."""
import copy
import json
import torch
from .credit import assess


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.feature = torch.nn.Parameter(torch.tensor(.2, dtype=torch.float64))
        self.W = torch.nn.Parameter(torch.zeros(6, 160, dtype=torch.float64))
        with torch.no_grad():
            self.W[0, 0] = 1.
    def forward(self, x):
        z = torch.zeros(len(x), 160, dtype=x.dtype)
        z[:, 0] = self.feature * x[:, 0, 0]
        return z @ self.W.T


def check():
    m = Toy()
    opt = torch.optim.AdamW(m.parameters(), lr=.03, weight_decay=.01)
    for p in m.parameters():
        p.grad = torch.zeros_like(p)
    opt.step()
    opt.zero_grad(set_to_none=True)
    x = torch.ones(10, 2, 256, dtype=torch.float64)
    y = torch.zeros(10, dtype=torch.long)
    rx = torch.tensor([1, 3, 4, 6, 8] * 2)
    before = copy.deepcopy(m.state_dict())
    rng = torch.get_rng_state().clone()
    base = tuple(torch.zeros_like(p) for p in m.parameters())
    good = (torch.tensor(-1., dtype=torch.float64), None)
    bad = (torch.tensor(1., dtype=torch.float64), None)
    plus = assess(m, opt, base, good, x, y, rx, 1, default_weight_name='W')
    minus = assess(m, opt, base, bad, x, y, rx, 1, default_weight_name='W')
    assert plus['summary']['accepted'] and plus['summary']['mean_credit'] > 0
    assert not minus['summary']['accepted'] and minus['summary']['mean_credit'] < 0
    # A classifier-only proposal can change parameters, but has zero feature
    # credit when both temporary readouts use the exact same pre-step W.
    wgrad = torch.zeros_like(m.W)
    wgrad[0, 0] = -1.
    head_only = assess(m, opt, base, (None, wgrad), x, y, rx, 1, default_weight_name='W')
    assert head_only['summary']['mean_credit'] == 0
    assert not head_only['summary']['accepted']
    assert head_only['summary']['step']['U_increment_norm'] > 0
    zero = assess(m, opt, base, (None, None), x, y, rx, 1, default_weight_name='W')
    assert zero['summary']['mean_credit'] == 0
    assert zero['summary']['step']['U_increment_norm'] == 0
    for kwargs in (dict(role='U_s'), dict(role='V'), dict(role='target'), dict(tolerance=1e-5)):
        try:
            assess(m, opt, base, good, x, y, rx, 1, default_weight_name='W', **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid role/tolerance accepted')
    assert torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(value, m.state_dict()[name]) for name, value in before.items())
    assert all(p.grad is None for p in m.parameters())
    return dict(status='VERIFIED', beneficial_feature_accepted=True, harmful_feature_rejected=True,
                classifier_only_increment_not_credited=True, zero_proposal_exact=True,
                native_AdamW_initialized_moments_used=True, live_state_and_RNG_unchanged=True,
                extra_real_updates=0, positive_credit=plus['summary']['mean_credit'],
                negative_credit=minus['summary']['mean_credit'])


if __name__ == '__main__':
    print(json.dumps(check()))
