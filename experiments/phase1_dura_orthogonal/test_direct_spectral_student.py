import torch
from experiments.phase1_dura_orthogonal.direct_spectral_student import DirectSpectralStudent
from experiments.phase1_dura_orthogonal.train_source import source_loss


def test_student_is_initially_identity_and_ignores_external_context():
    torch.manual_seed(14)
    model = DirectSpectralStudent(width=8, taps=5)
    x = torch.randn(2, 2, 256)
    torch.testing.assert_close(model(x), x, rtol=0, atol=0)
    torch.nn.init.normal_(model.wave.weight, std=.1)
    y = model(x)
    unrelated = torch.randn_like(x)
    torch.testing.assert_close(model(x, unrelated, unrelated, torch.ones(2)), y, rtol=0, atol=0)
    y.square().mean().backward()
    assert torch.isfinite(model.wave.weight.grad).all()
    assert torch.isfinite(model.filter_head.weight.grad).all()


def test_distillation_loss_uses_source_clean_and_teacher_iq():
    clean = torch.ones(2, 2, 256)
    teacher = 2*clean
    output = 3*clean
    torch.testing.assert_close(source_loss(output, teacher, clean, objective='distill_iq'), torch.tensor(4.5))
