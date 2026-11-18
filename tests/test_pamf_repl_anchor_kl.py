import torch

from experiments.pamf_repl_anchor_kl import plan
from experiments.pamf_repl_anchor_kl.objective import anchor_kl, directional_kl


def test_directional_target_is_detached_and_denominator_keeps_rejected_rows():
    student = torch.tensor([[1., -1.], [-1., 1.]], requires_grad=True)
    teacher = torch.tensor([[2., -2.], [2., -2.]], requires_grad=True)
    mask = torch.tensor([True, False])
    recipe = plan.RECIPES['ten_eval_t2_to_ordinary_full']
    loss = anchor_kl((student, student), (student, student), teacher, mask, recipe)
    torch.testing.assert_close(loss, directional_kl(student[:1], teacher[:1], 2.).sum() / 2.)
    loss.backward()
    assert teacher.grad is None and torch.count_nonzero(student.grad[1]) == 0
    assert torch.count_nonzero(student.grad[0]) == 2


def test_symmetric_cross_has_gradients_on_both_views():
    ordinary = torch.tensor([[1., -1.]], requires_grad=True)
    raw = torch.tensor([[2., -2.]], requires_grad=True)
    loss = anchor_kl((ordinary, ordinary), (raw, raw), None, torch.tensor([True]),
                     plan.RECIPES['six_cross_symmetric_full'])
    a, b = torch.autograd.grad(loss, (ordinary, raw))
    assert a.abs().sum() > 0 and b.abs().sum() > 0


def test_registered_rows_and_gpu_policy():
    assert len({r['row_id'] for r in plan.candidate_rows()}) == 28
    assert plan.GPUS == tuple(range(7))
