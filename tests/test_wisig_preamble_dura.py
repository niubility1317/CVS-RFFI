import numpy as np
import torch

from experiments.pamf_unrolled_restore.wisig_joint_preamble import waveform
from experiments.pamf_unrolled_restore.wisig_preamble_dura import PreambleDura, objective
from experiments.pamf_unrolled_restore.wisig_preamble_dura_source import MODES, proposals, trainable_heads
from experiments.pamf_unrolled_restore.wisig_smooth_preamble import SMOOTH, STARTS, TIME, choose, estimate
from experiments.pamf_unrolled_restore.wisig_timing import NOMINAL


def source_waveforms():
    shifts = np.asarray([-28, -12, 12, 28])
    return torch.as_tensor(waveform(TIME[None] - NOMINAL - shifts[:, None]).astype(np.complex64))


def test_receiver_fit_matches_frozen_engineering_reference():
    source = source_waveforms()
    model = PreambleDura(anchored=True)
    index, fit, confidence, margin = model.preamble_fit(source)
    scores, held, responses = estimate(source.numpy(), SMOOTH)
    expected, expected_margin = choose(scores)
    rows = np.arange(len(source))
    np.testing.assert_array_equal(STARTS[index.numpy()] - NOMINAL, expected)
    np.testing.assert_allclose(margin.numpy(), expected_margin, atol=3e-6)
    np.testing.assert_allclose(fit.numpy(), responses[rows, index.numpy()], atol=3e-6)
    np.testing.assert_allclose(confidence.numpy(), np.clip(held[rows, index.numpy()], 0., 1.), atol=3e-6)


def test_both_arms_are_end_to_end_and_clean_guard_is_exact():
    source = source_waveforms()
    parameters = []
    for active, count in ((False, 6), (True, 9)):
        model = PreambleDura(anchored=active)
        parameters.append(sum(parameter.numel() for parameter in model.parameters()))
        loss, terms = objective(model, source, torch.Generator().manual_seed(23), identity=True)
        assert len(terms) == 7 and torch.isfinite(terms).all()
        loss.backward()
        assert all(head.weight.grad is not None and torch.count_nonzero(head.weight.grad)
                   and torch.isfinite(head.weight.grad).all() for head in trainable_heads(model))
        values, touched, shift = proposals(model.eval(), (float('inf'),float('inf')), source)
        assert len(values) == count and not touched.any() and shift.shape == (len(source),)
        assert all(isinstance(value, torch.Tensor) and value.shape == source.shape and torch.isfinite(value).all()
                   for value in values)
        assert all(torch.equal(value, source) for value in values[5:])
    assert len(MODES) == 2 + 6 + 9 and parameters[0] == parameters[1]


def test_uncertain_zero_iq_leaves_nominal_coordinates():
    model = PreambleDura(anchored=True).eval()
    empty = torch.zeros(3, 256, dtype=torch.complex64)
    analysis = model.analyze(empty)
    assert analysis[3].all() and torch.all(analysis[5] == 1)
    assert torch.count_nonzero(analysis[-2]) == torch.count_nonzero(analysis[-1]) == 0


def test_model_constructor_does_not_require_numpy_to_torch_bridge(monkeypatch):
    tensor = torch.tensor

    def verified(data, *args, **kwargs):
        assert not isinstance(data, np.ndarray)
        return tensor(data, *args, **kwargs)

    def fail(*_args, **_kwargs):
        raise AssertionError('NumPy bridge unavailable')

    monkeypatch.setattr(torch, 'tensor', verified)
    monkeypatch.setattr(torch, 'as_tensor', fail)
    monkeypatch.setattr(torch, 'from_numpy', fail)
    model = PreambleDura(anchored=True)
    assert model.center.dtype == torch.bool and model.projection_0.dtype == torch.complex64
