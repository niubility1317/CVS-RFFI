import torch

from experiments.pamf_phase1_flow.filter_student import FilterStudent, apply_filter_delta, project_delta
from experiments.pamf_phase1_flow.filter_student_experiment import step


def test_complex_and_conjugate_filter_application():
    torch.manual_seed(4)
    anchor = torch.randn(2, 2, 256)
    x = torch.complex(anchor[:, 0], anchor[:, 1])
    coeff = torch.zeros(2, 2, 3, 2)
    coeff[:, 0, 1, 0] = .2
    coeff[:, 1, 1, 1] = .3
    expected = .2 * x + .3j * x.conj()
    delta = apply_filter_delta(anchor, coeff)
    torch.testing.assert_close(delta[:, 0], expected.real)
    torch.testing.assert_close(delta[:, 1], expected.imag)


def test_identity_and_per_record_bound():
    torch.manual_seed(5)
    model = FilterStudent(feature_dim=8, width=16, layers=1, heads=4, taps=9, epsilon=.2)
    received = torch.randn(3, 2, 256)
    anchor = torch.randn_like(received)
    features, logits = torch.randn(3, 8), torch.randn(3, 6)
    assert torch.equal(model(received, anchor, features, logits), anchor)
    with torch.no_grad():
        model.output.weight.normal_(0, 10)
        model.output.bias.normal_(0, 10)
    result = model(received, anchor, features, logits)
    assert torch.all((result-anchor).flatten(1).norm(dim=1) <= .200001*anchor.flatten(1).norm(dim=1))
    assert torch.equal(project_delta(anchor, anchor, epsilon=.2), anchor)


def test_paired_source_loss_trains_filter_head():
    torch.manual_seed(3)
    model = FilterStudent(feature_dim=8, width=16, layers=1, heads=4, taps=9, epsilon=.2)
    x = torch.randn(3, 2, 256)
    anchor = torch.randn_like(x)
    out = model(x, anchor, torch.randn(3, 8), torch.randn(3, 6))
    (out-anchor-.05).square().mean().backward()
    assert model.output.weight.grad is not None and model.output.weight.grad.abs().sum() > 0


def test_real_training_step_uses_only_frozen_networks_and_paired_iq():
    class Teacher(torch.nn.Module):
        def features(self, x):
            return x.mean((1, 2))[:, None]

        def classify_features(self, features):
            return features.expand(-1, 6)

    class Dura(torch.nn.Module):
        def forward(self, x):
            return x

    torch.manual_seed(8)
    x = torch.randn(4, 2, 256)
    clean = x*.96
    teacher, dura = Teacher().eval().requires_grad_(False), Dura().eval().requires_grad_(False)
    model = FilterStudent(feature_dim=1, width=16, layers=1, heads=4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    stats = step(model, teacher, dura, x, clean, optimizer)
    assert stats['head_grad'] > 0 and stats['grad_norm'] > 0
    assert model.output.weight.detach().abs().sum() > 0
