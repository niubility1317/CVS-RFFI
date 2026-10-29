import io

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from experiments.pamf_phase3_open_world import star_reciprocal_adapter as adapter


def raw(n=12, seed=5):
    return np.random.default_rng(seed).normal(size=(n, 160)).astype(np.float32)


class FrozenCosineHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.eye(160)[:6].clone(), requires_grad=False)
        self.calls = self.rows = 0

    def classify_features(self, x):
        self.calls += 1
        self.rows += len(x)
        return 10 * F.normalize(x, dim=1) @ self.weight.T


def test_identity_seed_rng_norm_bound_and_state_reload():
    torch.manual_seed(39)
    before = torch.get_rng_state().clone()
    first, second = adapter.Adapter(), adapter.Adapter()
    assert torch.equal(torch.get_rng_state(), before)
    assert adapter.PARAMETERS == sum(p.numel() for p in first.parameters()) == 10432
    assert adapter.STEPS == 200
    assert not first.fc2.weight.any() and not first.fc2.bias.any()
    for key, value in first.state_dict().items():
        torch.testing.assert_close(value, second.state_dict()[key], rtol=0, atol=0)
    x = torch.tensor(raw().tolist(), dtype=torch.float32)
    torch.testing.assert_close(first(x), x, rtol=2e-6, atol=2e-6)
    with torch.no_grad():
        first.fc2.weight.fill_(4)
        first.fc2.bias.fill_(7)
    residual = first.residual(x)
    assert torch.all(residual.norm(dim=1) <= .1000001)
    torch.testing.assert_close(first(x).norm(dim=1), x.norm(dim=1), rtol=2e-6, atol=2e-6)
    buffer = io.BytesIO()
    torch.save(first.state_dict(), buffer)
    buffer.seek(0)
    loaded = adapter.Adapter()
    loaded.load_state_dict(torch.load(buffer, map_location='cpu', weights_only=True), strict=True)
    torch.testing.assert_close(loaded(x), first(x), rtol=0, atol=0)


def test_gram_is_offdiagonal_teacher_detached_and_rotation_invariant():
    a = torch.eye(160)[:3].clone().requires_grad_()
    b = a.detach().clone().requires_grad_()
    with torch.no_grad():
        a[1] = F.normalize(torch.eye(160)[0] + torch.eye(160)[1], dim=0)
    value = adapter.gram_loss(a, b)
    assert value.item() == pytest.approx(1 / 6, abs=3e-8)
    value.backward()
    assert a.grad.abs().sum() > 0 and b.grad is None
    rotation = torch.eye(160).roll(3, dims=1)
    torch.testing.assert_close(adapter.gram_loss(a @ rotation, b @ rotation), value, rtol=0, atol=1e-7)
    one = torch.eye(160)[:1].requires_grad_()
    assert adapter.gram_loss(one, one).item() == 0


def test_fit_full_pool_200_steps_freezes_head_and_retains_input_gradient():
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        head = FrozenCosineHead().eval()
        before = {k: v.clone() for k, v in head.state_dict().items()}
        values = raw(9)
        labels = np.asarray([0, -1, 2, 3, -1, 5, 0, 1, -1], np.int64)
        model, trace, summary = adapter.fit(values, labels, head.classify_features, device='cpu')
        assert summary['status'] == 'COMPLETE'
        assert summary['steps'] == len(trace) == 200
        assert summary['selected_count'] == 6 and summary['U_count'] == 9
        assert summary['classifier_forward_calls'] == head.calls == 200
        assert summary['classifier_forward_rows'] == head.rows == 1800
        assert summary['training_adapter_forward_rows'] == 1800
        assert summary['diagnostic_adapter_forward_rows'] == 9
        assert summary['adapter_forward_rows'] == 1809
        assert summary['parameters_changed'] and summary['max_abs_feature_change'] > 0
        assert summary['finite'] and summary['max_pre_normalization_residual'] <= .1000001
        assert all(row['step'] == i + 1 and row['full_U_count'] == 9 for i, row in enumerate(trace))
        assert trace[0]['grad_norm'] > 0
        for row in trace:
            assert row['loss_total'] == pytest.approx(row['loss_ce'] + row['loss_gram'], abs=1e-6)
            assert all(np.isfinite(row[k]) for k in ('loss_total', 'loss_ce', 'loss_gram', 'grad_norm'))
        assert not model.training and all(not p.requires_grad for p in model.parameters())
        for key in before:
            torch.testing.assert_close(before[key], head.state_dict()[key], rtol=0, atol=0)
        assert head.weight.grad is None
        out, logits = adapter.predict(model, values, head.classify_features, device='cpu')
        assert out.dtype == logits.dtype == np.float32
        assert out.shape == (9, 160) and logits.shape == (9, 6)
        np.testing.assert_allclose(np.linalg.norm(out, axis=1), np.linalg.norm(values, axis=1), rtol=2e-6, atol=2e-6)
        one_out, one_logits = adapter.predict(model, values[2:3], head.classify_features, device='cpu')
        np.testing.assert_allclose(one_out[0], out[2], rtol=1e-5, atol=2e-6)
        np.testing.assert_allclose(one_logits[0], logits[2], rtol=1e-5, atol=2e-6)
    finally:
        torch.set_num_threads(previous_threads)


def test_no_pseudolabels_is_identity_zero_steps_and_no_classifier_calls():
    head = FrozenCosineHead().eval()
    values = raw(5)
    model, trace, summary = adapter.fit(values, np.full(5, -1, np.int64), head.classify_features, 'cpu')
    assert summary['status'] == 'SKIPPED_NO_PSEUDOLABELS'
    assert trace == [] and summary['steps'] == summary['selected_count'] == 0
    assert summary['classifier_forward_rows'] == summary['adapter_forward_rows'] == head.calls == 0
    assert not summary['parameters_changed'] and not model.training
    out, _ = adapter.predict(model, values, head.classify_features, 'cpu')
    np.testing.assert_allclose(out, values, rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize('bad', [np.zeros((3, 160), np.float32), np.ones((3, 160), np.float64),
    np.full((3, 160), np.nan, np.float32), np.ones((3, 159), np.float32), np.empty((0, 160), np.float32)])
def test_invalid_raw_is_rejected(bad):
    with pytest.raises(ValueError):
        adapter.fit(bad, np.zeros(len(bad), np.int64), FrozenCosineHead().classify_features, 'cpu')


@pytest.mark.parametrize('labels', [np.asarray([0, 6], np.int64), np.asarray([-2, 0], np.int64),
    np.asarray([0, 1], np.int32), np.asarray([0], np.int64)])
def test_invalid_pseudolabels_are_rejected(labels):
    with pytest.raises(ValueError):
        adapter.fit(raw(2), labels, FrozenCosineHead().classify_features, 'cpu')


def test_detached_classifier_output_fails_instead_of_fake_training():
    with pytest.raises(ValueError, match='gradient'):
        adapter.fit(raw(2), np.asarray([0, -1], np.int64), lambda x: torch.zeros((len(x), 6)), 'cpu')


def test_trainable_classifier_is_rejected_before_it_can_accumulate_gradients():
    head = FrozenCosineHead()
    head.weight.requires_grad_(True)
    with pytest.raises(ValueError, match='frozen'):
        adapter.fit(raw(2), np.asarray([0, -1], np.int64), head.classify_features, 'cpu')
    assert head.calls == 0 and head.weight.grad is None


def test_cpu_initialization_does_not_seed_cuda(monkeypatch):
    def forbidden_seed(*args, **kwargs):
        raise AssertionError('CPU-only initialization must not seed CUDA')
    monkeypatch.setattr(torch.cuda, 'manual_seed_all', forbidden_seed)
    adapter.Adapter()
