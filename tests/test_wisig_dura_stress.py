import json

import numpy as np
import pytest
import torch

from experiments.pamf_lora_main.dura_capacity import DuraCapacity
from experiments.pamf_unrolled_restore import wisig_dura_stress_fit as training
from experiments.pamf_unrolled_restore.wisig_dura_stress import CASES, COUNTS, SEEDS, VIEWS
from experiments.pamf_unrolled_restore.wisig_dura_stress_data import use_route
from experiments.pamf_unrolled_restore.wisig_dura_stress_score import closed_predictions


def test_one_window_matches_repeating_the_same_observation():
    torch.set_num_threads(2)
    torch.manual_seed(17)
    model = DuraCapacity('expanded').eval()
    iq = torch.randn(2, 2, 256)
    y, level, stages, _, _ = training.stages(model, iq)
    from experiments.pamf_unrolled_restore.wisig_receiver_estimates import receive_coordinates

    _, _, quality, variance = receive_coordinates(iq)
    repeated, *_ = model.stages(y.repeat(1, 6, 1), quality[:, None].repeat(1, 6, 1), level,
                                receiver_variance=variance[:, None].repeat(1, 6, 1))
    for single, six in zip(stages, repeated):
        torch.testing.assert_close(single[:, 0], six.mean(dim=1), rtol=3e-5, atol=4e-6)
    bypass, active = training.restore(model, float('inf'), iq, variance)
    assert not active.any()
    assert torch.equal(bypass, iq)


def test_matched_training_input_has_only_iq_and_identity_flag():
    clean = np.zeros((4, 2, 256), dtype=np.float32)
    full, residual = clean + 1, clean + 2
    q = np.array([.3, .02, .6, .8])
    assert not np.any(use_route(q, 'full') & use_route(q, 'residual'))
    observed, target, identity = training.examples(clean, (q, full, residual), np.arange(4))
    np.testing.assert_equal(observed[:, 0, 0], [2, 0, 1, 1])
    assert not target.any()
    np.testing.assert_equal(identity, [False, True, False, False])


def test_one_source_update_is_finite():
    torch.set_num_threads(2)
    torch.manual_seed(35)
    model = DuraCapacity('expanded').train()
    target = torch.randn(3, 2, 256)
    noisy = target + torch.tensor([0., .2, 1.4])[:, None, None] * torch.randn_like(target)
    loss = training.fit_loss(model, noisy, target, torch.tensor([True, False, False]))
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_independent_scorer_requires_all_views_and_exact_packet_ids(tmp_path):
    assert len(CASES) == 8 and sum(COUNTS.values()) == 168000
    closure = dict(status='PREDICTIONS_FROZEN', profile='strong', route='residual', split='cross_rx',
                   count=2, seeds=list(SEEDS), views=list(VIEWS), truth_read=False, active={})
    (tmp_path / 'complete.json').write_text(json.dumps(closure), encoding='utf-8')
    np.savez(tmp_path / 'decisions.npz', ids=['b', 'a'])
    with pytest.raises(ValueError, match='physical IDs'):
        closed_predictions(tmp_path, np.array(['a', 'b']), 'strong', 'residual', 'cross_rx')
    closure['truth_read'] = True
    (tmp_path / 'complete.json').write_text(json.dumps(closure), encoding='utf-8')
    with pytest.raises(ValueError, match='blind prediction'):
        closed_predictions(tmp_path, np.array(['a', 'b']), 'strong', 'residual', 'cross_rx')
