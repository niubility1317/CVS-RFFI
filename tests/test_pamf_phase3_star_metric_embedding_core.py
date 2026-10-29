import inspect
import math
from unittest.mock import patch

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from experiments.pamf_phase3_open_world import star_metric_embedding_core as core
from experiments.pamf_phase3_open_world import star_relative_core as relative


def units(n, seed=9):
    x = np.random.default_rng(seed).normal(size=(n, 160))
    return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)


def tensor(x):
    return torch.tensor(x.tolist(), dtype=torch.float64)


def loss_inputs():
    y = torch.arange(6).repeat_interleave(4)
    rx = torch.tensor([1, 1, 3, 3] * 6)
    z = tensor(units(24)).requires_grad_()
    oe = tensor(units(5, 4)).requires_grad_()
    original = tensor(units(5, 7)).requires_grad_()
    return z, y, rx, oe, original


def manual_contrast(z, y, rx):
    rows = z.detach().tolist()
    labels, receivers = y.tolist(), rx.tolist()
    result = []
    for i, a in enumerate(rows):
        positive = [j for j in range(len(rows)) if labels[j] == labels[i] and receivers[j] != receivers[i]]
        denominator = [j for j in range(len(rows)) if labels[j] != labels[i] or receivers[j] != receivers[i]]
        logits = {j: sum(u * v for u, v in zip(a, rows[j])) / .1 for j in denominator}
        lse = math.log(sum(math.exp(v) for v in logits.values()))
        result.append(sum(lse - logits[j] for j in positive) / len(positive))
    return sum(result) / len(result)


def test_residual_architecture_same_initialization_and_trainable_gradient():
    torch.manual_seed(2026092701)
    first = core.MetricNet()
    torch.manual_seed(2026092701)
    second = core.MetricNet()
    assert core.ARMS == ('identity', 'compact_oe', 'compact_oe_structure')
    assert core.METHODS == ('minimum_relative', 'predicted_identity_mahalanobis')
    assert (core.PRIMARY_ARM, core.PRIMARY_METHOD, core.TRAIN_STEPS, core.PARAMETERS) == (
        'compact_oe_structure', 'predicted_identity_mahalanobis', 1000, 20704)
    assert sum(p.numel() for p in first.parameters()) == 20704
    assert set(first.state_dict()) == {'fc1.weight', 'fc1.bias', 'fc2.weight', 'fc2.bias'}
    assert not first.fc2.weight.any() and not first.fc2.bias.any()
    for key in first.state_dict():
        torch.testing.assert_close(first.state_dict()[key], second.state_dict()[key], rtol=0, atol=0)
    x = torch.tensor(units(120).tolist())
    torch.testing.assert_close(first(x), F.normalize(x, dim=1, eps=1e-12), rtol=0, atol=0)
    y = torch.arange(6).repeat_interleave(20)
    rx = torch.tensor([r for _ in range(6) for r in (1, 3, 4, 6, 8) for _ in range(4)])
    oe = torch.tensor(units(128, 5).tolist())
    opt = torch.optim.Adam(first.parameters(), lr=1e-3, weight_decay=1e-4)
    out = core.losses(first(x), y, rx, first(oe), oe, 'compact_oe_structure')
    out['total'].backward()
    assert torch.isfinite(first.fc2.weight.grad).all() and first.fc2.weight.grad.abs().sum() > 0
    opt.step(); opt.zero_grad(set_to_none=True)
    core.losses(first(x), y, rx, first(oe), oe, 'compact_oe_structure')['total'].backward()
    assert torch.isfinite(first.fc1.weight.grad).all() and first.fc1.weight.grad.abs().sum() > 0
    torch.testing.assert_close(first(x).norm(dim=1), torch.ones(120), rtol=0, atol=2e-7)


def test_known_mask_and_exact_loss_terms():
    z, y, rx, oe, original = loss_inputs()
    out = core.losses(z, y, rx, oe, original, 'compact_oe_structure')
    assert set(out) == {'total', 'known_contrast', 'OE_repulsion', 'OE_structure'}
    assert out['known_contrast'].item() == pytest.approx(manual_contrast(z, y, rx), abs=2e-14)
    prototypes = F.normalize(torch.stack([z[y == c].mean(0) for c in range(6)]), dim=1)
    rep = (oe @ prototypes.T).max(dim=1).values.relu().square().mean()
    g = oe @ oe.T - F.normalize(original, dim=1) @ F.normalize(original, dim=1).T
    structure = g[~torch.eye(len(oe), dtype=torch.bool)].square().mean()
    torch.testing.assert_close(out['OE_repulsion'], rep, rtol=0, atol=2e-16)
    torch.testing.assert_close(out['OE_structure'], structure, rtol=0, atol=2e-16)
    torch.testing.assert_close(out['total'], out['known_contrast'] + 10 * rep + 10 * structure)
    out['total'].backward()
    assert z.grad.abs().sum() > 0 and oe.grad.abs().sum() > 0
    assert original.grad is None


def test_structure_weight_control_detach_and_offdiagonal_hand_example():
    eye = torch.eye(160, dtype=torch.float64)
    z = eye[:6].repeat_interleave(2, 0).requires_grad_()
    y = torch.arange(6).repeat_interleave(2)
    rx = torch.tensor([1, 3] * 6)
    oe = eye[:1].repeat(2, 1).requires_grad_()
    original = eye[:2].clone().requires_grad_()
    control = core.losses(z, y, rx, oe, original, 'compact_oe')
    main = core.losses(z, y, rx, oe, original, 'compact_oe_structure')
    assert control['known_contrast'].item() == pytest.approx(math.log1p(10 * math.exp(-10)), abs=2e-15)
    assert main['OE_repulsion'].item() == 1 and main['OE_structure'].item() == 1
    assert not control['OE_structure'].requires_grad
    torch.testing.assert_close(main['total'] - control['total'], torch.tensor(10., dtype=torch.float64))
    main['total'].backward()
    assert original.grad is None
    assert list(inspect.signature(core.losses).parameters) == ['z_L', 'y_L', 'rx_L', 'z_OE', 'x_OE', 'arm']


def test_sampler_balanced_without_replacement_same_rng_and_masks():
    y = np.repeat(np.arange(6), 30).astype(np.int64)
    rx = np.tile(np.repeat([1, 3, 4, 6, 8], 6), 6).astype(np.int64)
    originals = y.copy(), rx.copy()
    left, right = np.random.default_rng(2026103002), np.random.default_rng(2026103002)
    for _ in range(3):
        a, b = core.sample_indices(y, rx, 150, left)
        c, d = core.sample_indices(y, rx, 150, right)
        np.testing.assert_array_equal(a, c); np.testing.assert_array_equal(b, d)
        assert a.dtype == b.dtype == np.int64 and a.shape == (120,) and b.shape == (128,)
        assert len(set(a.tolist())) == 120 and len(set(b.tolist())) == 128
        assert 0 <= b.min() <= b.max() < 150
        for cls in range(6):
            for receiver in (1, 3, 4, 6, 8):
                assert np.count_nonzero((y[a] == cls) & (rx[a] == receiver)) == 4
        pos = (y[a, None] == y[a][None, :]) & (rx[a, None] != rx[a][None, :])
        neg = y[a, None] != y[a][None, :]
        assert (pos.sum(1) == 16).all() and (neg.sum(1) == 100).all()
    np.testing.assert_array_equal(y, originals[0]); np.testing.assert_array_equal(rx, originals[1])
    assert list(inspect.signature(core.sample_indices).parameters) == ['L_y', 'L_rx', 'OE_count', 'rng']


def test_l_only_four_aggregates_shrinkage_means_and_no_rows():
    x = units(30); y = np.repeat(np.arange(6), 5).astype(np.int64)
    saved = x.copy(), y.copy()
    with patch.object(relative, '_whitener', wraps=relative._whitener) as white:
        state = core.fit_metric(x, y)
    assert set(state) == set(core.METRIC_FIELDS) == {
        'class_means', 'background_mean', 'within_whitener', 'background_whitener'}
    assert [call.args[1] for call in white.call_args_list] == [24, 29]
    u = x.astype(np.float64); u /= np.linalg.norm(u, axis=1, keepdims=True)
    means = np.stack([u[y == c].mean(0) for c in range(6)])
    # The independent linalg norm and fixed einsum norm differ at FP64 roundoff.
    np.testing.assert_allclose(state['class_means'], means, rtol=0, atol=1e-16)
    assert (np.linalg.norm(state['class_means'], axis=1) < 1).all()
    assert all(a.dtype == np.float64 for a in state.values())
    assert sum(a.nbytes for a in state.values()) == 418560
    np.testing.assert_array_equal(x, saved[0]); np.testing.assert_array_equal(y, saved[1])
    permutation = np.random.default_rng(3).permutation(len(x))
    permuted = core.fit_metric(x[permutation], y[permutation])
    query = units(9, 13); closed = np.arange(9) % 6
    np.testing.assert_allclose(core.metric_scores(query, closed, state),
                               core.metric_scores(query, closed, permuted), rtol=0, atol=1e-10)


def test_score_hand_calculation_row_independence_closed_routing_and_immutable_state():
    state = dict(class_means=np.eye(160, dtype=np.float64)[:6],
                 background_mean=np.zeros(160, np.float64),
                 within_whitener=2 * np.eye(160, dtype=np.float64),
                 background_whitener=np.eye(160, dtype=np.float64))
    q = np.eye(160, dtype=np.float32)[:2]
    np.testing.assert_array_equal(core.metric_scores(q, np.array([1, 1]), state), [[1, -8], [1, 0]])
    query = units(259, 15); closed = np.arange(259) % 6
    original = {k: v.copy() for k, v in state.items()}
    out = core.metric_scores(query, closed, state)
    assert out.dtype == np.float64 and out.shape == (259, 2)
    rowwise = np.concatenate([core.metric_scores(query[i:i+1], closed[i:i+1], state) for i in range(len(query))])
    np.testing.assert_array_equal(out, rowwise)
    order = np.random.default_rng(11).permutation(len(query))
    np.testing.assert_array_equal(core.metric_scores(query[order], closed[order], state), out[order])
    for key in state: np.testing.assert_array_equal(state[key], original[key])


def test_invalid_boundaries(subtests):
    y = np.repeat(np.arange(6), 20).astype(np.int64)
    rx = np.tile(np.repeat([1, 3, 4, 6, 8], 4), 6).astype(np.int64)
    x = units(120); metric = core.fit_metric(x, y)
    bad = [lambda: core.sample_indices(y, rx, 127, np.random.default_rng(1)),
           lambda: core.sample_indices(y[:-1], rx[:-1], 128, np.random.default_rng(1)),
           lambda: core.sample_indices(y.astype(float), rx, 128, np.random.default_rng(1)),
           lambda: core.sample_indices(y, np.ones_like(rx), 128, np.random.default_rng(1)),
           lambda: core.fit_metric(x, y + 1),
           lambda: core.fit_metric(x.astype(np.float64), y),
           lambda: core.fit_metric(x * 2, y),
           lambda: core.fit_metric(np.full_like(x, np.nan), y),
           lambda: core.metric_scores(x, np.full(len(x), 6), metric),
           lambda: core.metric_scores(x, y.astype(float), metric),
           lambda: core.metric_scores(x[:, :159], y, metric),
           lambda: core.metric_scores(x, y, dict(metric, source_rows=x)),
           lambda: core.MetricNet()(torch.zeros(2, 159)),
           lambda: core.losses(*loss_inputs(), 'identity'),
           lambda: core.losses(*loss_inputs()[:3], loss_inputs()[3][:1], loss_inputs()[4][:1], 'compact_oe'),
           lambda: core.losses(loss_inputs()[0], loss_inputs()[1], torch.ones(24, dtype=torch.int64),
                               loss_inputs()[3], loss_inputs()[4], 'compact_oe')]
    for index, action in enumerate(bad):
        with subtests.test(index=index):
            with pytest.raises(ValueError): action()
