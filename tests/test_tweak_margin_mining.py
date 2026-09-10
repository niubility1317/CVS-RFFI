"""Protect the opt-in reference-guided miner and keep strict reproduction unchanged."""
import pytest
import torch

from paper_reproduction.gaskin_tweak_2023 import triplet
from paper_reproduction.gaskin_tweak_2023 import official_lora_experiment as runner


def test_margin_mining_retains_semihard_pairs_and_rewards_separation():
    # Six violating triplets have losses .0475 four times and .0875 twice.
    # Two easy triplets must not dilute the mean; strict mining sees none.
    z = torch.tensor([[0.0], [.1], [.25], [.35]], dtype=torch.float64, requires_grad=True)
    labels = torch.tensor([0, 0, 1, 1])
    assert hasattr(triplet, 'all_margin_violating_triplet_loss')
    loss, active = triplet.all_margin_violating_triplet_loss(z, labels)
    assert active and float(loss.detach()) == pytest.approx(.365 / 6)
    gradient = torch.autograd.grad(loss, z)[0]
    assert float((gradient * z.detach()).sum()) < 0  # descent rewards expansion, not uniform contraction
    strict_loss, strict_active = triplet.all_strict_hard_triplet_loss(z, labels)
    assert not strict_active and float(strict_loss.detach()) == 0


def test_margin_mining_returns_inactive_when_every_negative_is_beyond_margin():
    assert hasattr(triplet, 'all_margin_violating_triplet_loss')
    z = torch.tensor([[0.0], [.01], [10.0], [10.01]], requires_grad=True)
    loss, active = triplet.all_margin_violating_triplet_loss(z, torch.tensor([0, 0, 1, 1]))
    assert not active and float(loss.detach()) == 0
    loss.backward()
    assert torch.equal(z.grad, torch.zeros_like(z))


def test_epoch_opt_in_changes_real_optimizer_update_but_default_stays_strict(monkeypatch):
    # If the runner ignores the option, the semi-hard-only batch makes no update.
    monkeypatch.setattr(runner, '_source_training_batches', lambda *a, **kw: iter([
        (torch.tensor([[0.0], [.1], [.25], [.35]]), torch.tensor([0, 0, 1, 1]))]))
    weights, reports = [], []
    for mode in ('strict_hard', 'margin_violating'):
        model = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(1)
        kwargs = {} if mode == 'strict_hard' else {'triplet_mining': mode}
        row = runner._run_training_epoch(model, torch.optim.SGD(model.parameters(), lr=.1), [], None,
                                         device=torch.device('cpu'), seed=1, max_batches_per_epoch=1, **kwargs)
        weights.append(float(model.weight.detach()))
        reports.append(row)
    assert weights[0] == 1 and weights[1] > 1
    assert reports[0]['active_batches'] == 0 and reports[1]['active_batches'] == 1
    assert reports[1]['first_batch_mining']['selected_triplets'] == 6
    assert reports[1]['first_batch_mining']['strict_hard_triplets'] == 0
    assert reports[1]['first_batch_mining']['semihard_triplets'] == 6


def test_reference_guided_variant_cannot_be_reported_as_strict_paper_parity():
    assert hasattr(runner, 'method_metadata_for_mining')
    variant = runner.method_metadata_for_mining('margin_violating')
    assert variant['parity_status'] == 'REFERENCE_GUIDED_MINING_DIAGNOSTIC_NOT_STRICT_REPRODUCTION'
    assert variant['unpublished_defaults']['triplet_mining']['value']['hardness_filter'] == 'positive_squared_distance-negative_squared_distance+0.1>0'
    original = runner.method_metadata_for_mining('strict_hard')
    assert original['parity_status'] == 'PAPER_METHOD_PARITY_WITH_UNPUBLISHED_DEFAULTS'
    assert original['unpublished_defaults']['triplet_mining']['value']['hardness_filter'] != variant['unpublished_defaults']['triplet_mining']['value']['hardness_filter']
    with pytest.raises(ValueError):
        runner.method_metadata_for_mining('typo')
