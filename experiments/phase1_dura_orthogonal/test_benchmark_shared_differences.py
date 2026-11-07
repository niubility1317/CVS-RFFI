import pytest
import torch

from experiments.phase1_dura_orthogonal import benchmark_shared_differences as bench
from experiments.phase1_dura_orthogonal.parallel_source_benchmark import encode


def test_field_difference_and_decision_counts(monkeypatch):
    monkeypatch.setattr(bench, 'MODELS', (('first', 'run1', 'c'), ('second', 'run2', 'c')))
    base = [torch.zeros(2, 2) for _ in bench.FIELDS]
    changed = [v.clone() for v in base]
    changed[3][0, 1] = .25
    changed[6][0, 1] = 1.
    result = bench.compare_details([[encode(changed), encode(base)]],
                                   {'run1': [encode(base)], 'run2': [encode(base)]})
    assert result['restored'] == dict(compared=2, mismatched=1, max_abs=.25, class_changes=0)
    assert result['restored_logits'] == dict(compared=2, mismatched=1, max_abs=1., class_changes=1)
    assert result['anchor']['mismatched'] == 0


def test_shape_mismatch_and_nonfinite_rejected(monkeypatch):
    monkeypatch.setattr(bench, 'MODELS', (('first', 'run1', 'c'),))
    base = [torch.zeros(2, 2) for _ in bench.FIELDS]
    bad = [v.clone() for v in base]
    bad[0] = torch.zeros(3, 2)
    with pytest.raises(ValueError, match='shape and dtype'):
        bench.compare_details([[encode(bad)]], {'run1': [encode(base)]})
    bad[0] = torch.full((2, 2), float('nan'))
    with pytest.raises(FloatingPointError):
        bench.compare_details([[encode(bad)]], {'run1': [encode(base)]})
