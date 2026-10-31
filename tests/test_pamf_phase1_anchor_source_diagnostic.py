import numpy as np
import pytest
import torch

from experiments.pamf_phase1_flow.anchor_source_diagnostic import masked_mean, paired_counts


def test_repairs_and_damages_count_against_same_clean_teacher():
    reference = np.array([0, 1, 1, 2, 0, 2])
    dura = np.array([0, 0, 1, 0, 1, 2])
    candidate = np.array([1, 1, 1, 0, 0, 2])
    mask = np.array([True, True, True, True, False, True])
    assert paired_counts(reference, dura, candidate, mask) == {
        'count': 5, 'dura_matches': 3, 'candidate_matches': 3,
        'repairs': 1, 'damages': 1,
    }


def test_rejects_misaligned_rows():
    with pytest.raises(ValueError):
        paired_counts(np.array([0, 1]), np.array([0]), np.array([0, 1]), np.array([True, True]))


def test_masked_mean_uses_boolean_selection_with_no_numpy_torch_bridge(monkeypatch):
    def unavailable(*args, **kwargs):
        raise TypeError('NumPy bridge unavailable')
    monkeypatch.setattr(torch, 'from_numpy', unavailable)
    assert masked_mean(torch.tensor([1., 5., 9.]), np.array([True, False, True])) == 5.
