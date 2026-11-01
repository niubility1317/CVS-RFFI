import torch

from experiments.pamf_phase1_flow.class_hypothesis import ClassHypothesis, select_verified, source_oracle
from experiments.pamf_phase1_flow.class_hypothesis_experiment import step


def test_zero_initialization_and_classwise_bound():
    torch.manual_seed(4)
    model = ClassHypothesis(feature_dim=8, width=16, layers=1, heads=4, epsilon=.05)
    x = torch.randn(3, 2, 256)
    anchor = torch.randn_like(x)
    features = torch.randn(3, 8)
    logits = torch.randn(3, 6)
    for label in range(6):
        q, candidate = model(x, anchor, features, logits, torch.full((3,), label))
        assert torch.equal(candidate, anchor)
        assert torch.equal(q, logits)
    with torch.no_grad():
        model.output.weight.normal_(0, 10)
        model.output.bias.normal_(0, 10)
    for label in range(6):
        _, candidate = model(x, anchor, features, logits, torch.full((3,), label))
        assert torch.all((candidate-anchor).flatten(1).norm(dim=1) <= .050001 * anchor.flatten(1).norm(dim=1))


def test_gate_requires_confidence_disagreement_and_verified_class():
    anchor = torch.zeros(4, 2, 256)
    candidate = torch.ones_like(anchor)
    posterior = torch.tensor([[.01, .96, .01, .01, .005, .005],
                              [.3, .4, .1, .1, .05, .05],
                              [.96, .01, .01, .01, .005, .005],
                              [.01, .96, .01, .01, .005, .005]])
    baseline = torch.tensor([0, 0, 0, 0])
    verified = torch.tensor([1, 1, 0, 2])
    result, accepted = select_verified(anchor, candidate, posterior, baseline, verified, threshold=.95)
    assert accepted.tolist() == [True, False, False, False]
    assert torch.equal(result[0], candidate[0])
    assert torch.equal(result[1:], anchor[1:])


def test_hypothesis_and_proposal_both_learn():
    model = ClassHypothesis(feature_dim=8, width=16, layers=1, heads=4, epsilon=.05)
    x = torch.randn(3, 2, 256)
    anchor = torch.randn_like(x)
    labels = torch.tensor([0, 2, 5])
    q, candidate = model(x, anchor, torch.randn(3, 8), torch.randn(3, 6), labels)
    loss = torch.nn.functional.cross_entropy(q, labels) + (candidate - anchor - .01).square().mean()
    loss.backward()
    assert model.posterior.weight.grad.abs().sum() > 0
    assert model.output.weight.grad.abs().sum() > 0


def test_source_oracle_only_changes_valid_mismatches():
    class Frozen(torch.nn.Module):
        def features(self, x):
            return x.mean((1, 2), keepdim=False)[:, None]

        def classify_features(self, features):
            return torch.cat((features, -features), dim=1)

    teacher = Frozen().eval().requires_grad_(False)
    anchor = torch.ones(3, 2, 256)
    target = torch.tensor([1, 1, 0])
    valid = torch.tensor([True, False, True])
    result, reached = source_oracle(teacher, anchor, target, valid, epsilon=.05, steps=2)
    assert not torch.equal(result[0], anchor[0])
    assert torch.equal(result[1:], anchor[1:])
    assert not reached[1:].any()


def test_real_training_step_updates_both_heads_with_frozen_models():
    class Teacher(torch.nn.Module):
        def features(self, x):
            return x.mean((1, 2))[:, None]

        def classify_features(self, features):
            return torch.cat((features * 10, -features * 10, features.expand(-1, 4) * 0 - 10), dim=1)

    class Dura(torch.nn.Module):
        def forward(self, x):
            return x

    torch.manual_seed(3)
    received = torch.randn(4, 2, 256)
    received = received - received.mean((1, 2), keepdim=True) + .01
    teacher = Teacher().eval().requires_grad_(False)
    dura = Dura().eval().requires_grad_(False)
    model = ClassHypothesis(feature_dim=1, width=16, layers=1, heads=4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    stats = step(model, teacher, dura, received, -received, torch.ones(4, dtype=torch.long), optimizer)
    assert stats['oracle_reached'] == 4
    assert stats['posterior_grad'] > 0 and stats['proposal_grad'] > 0
    assert model.posterior.weight.detach().abs().sum() > 0
    assert model.output.weight.detach().abs().sum() > 0
