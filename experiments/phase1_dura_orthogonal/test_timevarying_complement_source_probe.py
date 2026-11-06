import pytest
import torch
from experiments.phase1_dura_orthogonal.timevarying_complement_source_probe import summarize_line


def test_line_summary_respects_per_record_improvement_and_negative_counts():
    values = {'anchor_error': torch.tensor([.2, .3]), 'positive_error': torch.tensor([.1, .3]),
              'signed_error': torch.tensor([.1, .2]), 'optimal_alpha': torch.tensor([.2, -.3])}
    s = summarize_line(values)
    assert s['anchor_nmse'] == pytest.approx(.25)
    assert s['oracle_positive_nmse'] == pytest.approx(.2)
    assert s['oracle_signed_nmse'] == pytest.approx(.15)
    assert s['positive_improved'] == 1 and s['negative_optimal'] == 1
