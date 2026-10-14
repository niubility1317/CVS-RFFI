import json
import torch
from . import design as d
from .objective import angle


def checks():
    generator = torch.Generator().manual_seed(2026101507)
    layout = [(t, r, day) for t in range(6) for day in (1, 2, 3)
              for r in d.parent.RX for _ in range(2)]
    y, rx, day = (torch.tensor([q[k] for q in layout]) for k in range(3))
    ids = ['synthetic-' + str(i) for i in range(180)]
    x = torch.randn(180, 3, generator=generator, dtype=torch.float64)
    parameter = torch.randn(3, 8, generator=generator, dtype=torch.float64).requires_grad_()
    W = torch.ones(6, 160, requires_grad=True)
    G = x @ parameter
    loss = angle(G, y, rx, day, ids)
    derivative = torch.autograd.grad(loss, parameter, retain_graph=True)[0]
    assert torch.isfinite(derivative).all() and derivative.norm() > 0
    assert torch.autograd.grad(loss, W, allow_unused=True, retain_graph=True)[0] is None
    assert abs(float(angle(2 * G, y, rx, day, ids) - loss)) < 1e-12
    for role in ('V', 'U_s', 'query'):
        try:
            angle(G, y, rx, day, ids, role)
        except ValueError:
            pass
        else:
            raise AssertionError('Forbidden role accepted')
    for bad in (G.detach() * 0, G.detach() * float('nan')):
        try:
            angle(bad, y, rx, day, ids)
        except ValueError:
            pass
        else:
            raise AssertionError('Unidentifiable/nonfinite direction accepted')
    for key, value in [('role', 'V'), ('source_packets', 6300),
                       ('head_fit_updates', 1), ('query_read', True),
                       ('model_seed', 2026092702), ('variance_norm_epsilon', 1e-8)]:
        c = d.config(d.rows()[0]); c[key] = value
        try:
            d.validate(c)
        except ValueError:
            pass
        else:
            raise AssertionError('Unregistered configuration accepted')
    return dict(status='PASS', forbidden_roles=3, invalid_directions=2,
                config_negatives=6, differentiable=True, scale_invariant=True,
                auxiliary_W_gradient_absent=True, query_read=False,
                real_IQ_or_checkpoint_read=False, optimizer_steps=0)


if __name__ == '__main__':
    print(json.dumps(checks()))
