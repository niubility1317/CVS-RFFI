import torch
from experiments.phase1_dura_orthogonal.physical_line_ground import apply_physical_line


def test_line_step_reduces_observed_residual_and_preserves_zero_direction():
    anchor = torch.zeros(3, 2, 256)
    proposal = anchor.clone()
    proposal[:2, 0] = 1.
    received = anchor.clone()
    received[0, 0] = .3
    received[1, 0] = -1.
    received[2, 0] = 5.
    out = apply_physical_line(received, anchor, proposal, lambda z: z)
    torch.testing.assert_close(out[0, 0], torch.full((256,), .3))
    torch.testing.assert_close(out[1:], anchor[1:])
    assert torch.all((received-out).square().sum((1,2)) <= (received-anchor).square().sum((1,2)))


def test_controller_hides_gpu_even_if_parent_exposes_it(monkeypatch):
    from experiments.phase1_dura_orthogonal.physical_line_controller import cpu_environment
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '3')
    env = cpu_environment()
    assert env['CUDA_VISIBLE_DEVICES'] == ''
    assert env['OMP_NUM_THREADS'] == '2'
