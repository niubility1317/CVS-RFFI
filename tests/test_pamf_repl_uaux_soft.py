import torch

from experiments.pamf_repl_uaux_soft.projection import auxiliary_loss
from experiments.pamf_repl_uaux_soft import plan


def run(alpha, beta, reference, other, labeled, unlabeled, p):
    value, _, _, stats = auxiliary_loss(labeled, unlabeled, reference, other,
                                        [('other.p', p)], alpha=alpha, beta=beta)
    value.backward()
    return p.grad, stats


def test_soft_projection_preserves_labeled_and_pseudo_ce():
    for alpha, expected in [(0., [-2., 6.]), (.5, [-1.5, 6.]), (1., [-1., 6.])]:
        p = torch.nn.Parameter(torch.zeros(2))
        r, h, ce, labeled = p[0], -p[0] + p[1], 3*p[1], -2*p[0] + 2*p[1]
        grad, stats = run(alpha, 0., r, h, labeled, r+h+ce, p)
        assert torch.allclose(grad, torch.tensor(expected))
        assert stats['other_capped'] == 0


def test_zero_reference_and_label_conflict():
    p = torch.nn.Parameter(torch.zeros(2))
    r, h, ce, labeled = 0*p[0], -p[0], 3*p[1], p[0]
    grad, stats = run(1., 1., r, h, labeled, r+h+ce, p)
    assert torch.allclose(grad, torch.tensor([1., 3.]))
    assert stats['other_zero_reference'] == 1
    assert stats['other_label_aux_dot_after'] == 0


def test_matrix():
    rows = plan.candidate_rows()
    assert len(rows) == 16 and len({row['row_id'] for row in rows}) == 16
    assert sum(row['setting'] == 'ten_class' for row in rows) == 8
    assert all(row['views'] == ['clean'] for row in rows)
