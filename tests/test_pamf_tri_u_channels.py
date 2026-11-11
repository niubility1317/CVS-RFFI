from pathlib import Path
import pytest
import torch
from experiments.pamf_tri_u_channels import plan
from experiments.pamf_tri_u_channels.plan import ARMS, candidate_rows, configuration
from experiments.pamf_tri_u_channels.transform import channel_view


def test_cfo_and_echo_are_bounded_non_circular_and_differentiable():
    x = torch.zeros(2, 2, 256, requires_grad=True)
    with torch.no_grad():
        x[0, 0, 0] = 1
        x[1, 0, -1] = 1
    angles = torch.tensor([0.25, -0.25])
    for recipe in ARMS.values():
        y = channel_view(x, angles, **recipe)
        assert y.shape == x.shape and torch.isfinite(y).all()
        assert torch.allclose(y.square().sum((1,2)), x.square().sum((1,2)), atol=1e-6)
        assert y[1, :, :-1].count_nonzero() == 0  # no circular wrap of final-sample echo
        y.square().sum().backward(retain_graph=True)
    assert torch.isfinite(x.grad).all()
    cfo = channel_view(x, torch.zeros(2), **ARMS['tri-u-cfo'])
    assert torch.allclose(cfo, x)
    echo = channel_view(x, torch.zeros(2), **ARMS['tri-u-echo3'])
    assert echo[0, 0, 3] > 0 and echo[0, :, 1:3].count_nonzero() == 0
    with pytest.raises(ValueError):
        channel_view(x, angles, kind='echo', delay=7, echo_amplitude=.15)


def test_matrix_has_four_fixed_arms_three_seeds(monkeypatch):
    monkeypatch.setattr(plan.phase, 'configuration', lambda context, row, project, commit: {'model_seed':row['model_seed']})
    rows = candidate_rows()
    assert len(rows) == 12 and len({r['row_id'] for r in rows}) == 12
    for row in rows:
        c = configuration({}, row, Path('/test/project'), 'test')
        assert c['channel_perturbation'] == ARMS[row['arm']]
        assert c['model_seed'] == row['model_seed']
        assert not c.get('target_truth') and not c.get('tests')
