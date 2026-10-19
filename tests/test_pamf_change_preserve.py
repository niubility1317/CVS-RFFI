import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'code')]
from experiments.pamf_change_preserve.model import POLICY, PreserveSmallChanges, calibrated_threshold, preserve
from experiments.pamf_change_preserve import runner


def test_per_packet_route_preserves_exact_iq_without_labels_or_scene():
    received = torch.ones(3, 2, 256)
    restored = received.clone()
    restored[1] += .01
    restored[2] += .2
    output, mask, _ = preserve(received, restored, .001)
    assert mask.tolist() == [True, True, False]
    assert torch.equal(output[:2], received[:2])
    assert torch.equal(output[2], restored[2])
    for index in range(3):
        assert torch.equal(output[index:index+1], preserve(received[index:index+1], restored[index:index+1], .001)[0])
    permutation = torch.tensor([2, 0, 1])
    assert torch.equal(output[permutation], preserve(received[permutation], restored[permutation], .001)[0])


@pytest.mark.parametrize('threshold', [float('nan'), float('inf'), -.1, .021])
def test_reject_unfrozen_or_outside_policy_threshold(threshold):
    with pytest.raises(ValueError):
        PreserveSmallChanges(nn.Identity(), threshold)


def test_calibration_is_fixed_unlabeled_training_only_and_capped():
    indices = runner.calibration_selection([6300, 56700])
    assert len(indices) == 8192 and len(np.unique(indices)) == 8192
    assert np.array_equal(indices, runner.calibration_selection([6300, 56700]))
    changes = torch.linspace(0, .01, 8192)
    assert calibrated_threshold(changes) == float(changes.max())
    changes[-1] = .5
    assert calibrated_threshold(changes) == POLICY['threshold_cap']
    with pytest.raises(ValueError):
        runner.calibration_selection([6300, 56700, 27000])
    with pytest.raises(ValueError):
        calibrated_threshold(torch.zeros(1800))


def test_frozen_parent_stays_eval_and_zero_input_is_finite():
    model = PreserveSmallChanges(nn.Sequential(nn.Dropout(.9), nn.Identity()), 0.)
    model.train()
    assert not model.training and not model.restorer.training
    zero = torch.zeros(2, 2, 256)
    assert torch.equal(model(zero), zero)
    with pytest.raises(FloatingPointError):
        preserve(zero, torch.full_like(zero, float('nan')), .01)


def test_incomplete_predictions_cannot_open_source_labels(tmp_path, monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    ids = np.array(['source0', 'source1'])
    monkeypatch.setattr(runner, 'source_reference', lambda *_: (ids, None, np.array([0, 1])))
    saved = tmp_path/'prediction'
    saved.mkdir()
    (saved/'complete.json').write_text(json.dumps(dict(status='RUNNING')), encoding='utf-8')
    opened = []
    monkeypatch.setattr(np, 'load', lambda *a, **kw: opened.append(a[0]))
    with pytest.raises(ValueError, match='before labels'):
        runner.score(tmp_path, tmp_path/'forbidden', tmp_path)
    assert opened == []
