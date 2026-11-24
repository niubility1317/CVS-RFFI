import numpy as np
import pytest
import torch

from experiments.pamf_lora_main import dura_observed_variance_target as target
from experiments.pamf_lora_main import fixed_mid_target_score as scorer


def test_frozen_doses_reuse_one_solution_and_bypass_inactive_packets(monkeypatch):
    iq = torch.zeros(2, 6, 2, 256)
    observed = torch.ones(2, 6, 256, dtype=torch.complex64)
    ones = torch.ones_like(observed)
    monkeypatch.setattr(target, 'coordinates', lambda *_: (observed, ones, ones, ones.real))
    monkeypatch.setattr(target, 'corruption_score', lambda _: torch.tensor([0., 1.]))
    monkeypatch.setattr(target, 'packet_frame', lambda frame: frame)
    monkeypatch.setattr(target, 'mix', lambda y, x, dose: y + dose * (x - y))
    calls = []

    def restore(y, *_):
        calls.append(y.shape)
        return y, y + 4

    (half, quarter), count = target.readouts(iq, None, restore, .5)
    assert count == 1 and calls == [torch.Size((1, 6, 256))]
    assert torch.equal(half[0], iq[0]) and torch.equal(quarter[0], iq[0])
    assert torch.all(half[1,:,0] == 2.) and torch.all(quarter[1,:,0] == 1.)


def test_dose_matched_width_controls_require_same_gate_activity_and_direct_predictions():
    original = {route: {mode: np.zeros((1, 2, 3), dtype=np.int16)
                        for mode in scorer.CONTROL_MODES[route]} for route in scorer.ROUTES}
    historical = {route: {**modes, 'wide_quarter': np.ones((1, 2, 3), dtype=np.int16)}
                  for route,modes in original.items()}
    activity = {route: {'rx': {'clean': 0}} for route in scorer.ROUTES}
    with pytest.raises(ValueError, match='alignment or activity'):
        scorer.add_previous(original, activity, historical, {**activity, 'full': {}}, ('wide_quarter',))
    scorer.add_previous(original, activity, historical, activity, ('wide_quarter',))
    assert all(np.all(modes['wide_quarter'] == 1) for modes in original.values())


def test_secondary_dose_has_its_own_paired_control(monkeypatch):
    monkeypatch.setattr(scorer, 'TARGET', ('rx',))
    monkeypatch.setattr(scorer, 'VIEWS', ('clean', 'practical'))
    rows = {k: np.array([[[0, 0], data]]) for k,data in
            {'direct': [1, 1], 'observed_half': [1, 1],
             'observed_quarter': [0, 0], 'wide_quarter': [1, 0]}.items()}
    truth = np.zeros((1, 2), dtype=np.int16)
    draws = np.array([[[0, 0], [1, 1]]])
    scores = scorer.summarize(rows, truth, draws, primary='observed_half',
                              additional_pairs={'observed_quarter': 'wide_quarter'})
    contrast = scores['dose_matched_pairs']['observed_quarter']
    assert contrast['reference'] == 'wide_quarter'
    assert contrast['six']['net'] == 1 and contrast['six']['repair'] == 1
