import pytest

from experiments.phase1_dura_orthogonal.benchmark_pipelined_balanced import GROUPS, MODELS, validate_groups


def test_every_model_is_assigned_once_and_no_empty_worker():
    validate_groups(GROUPS, MODELS)
    with pytest.raises(ValueError, match='exactly once'):
        validate_groups(((0, 1), (1,)), MODELS)
    with pytest.raises(ValueError, match='exactly once'):
        validate_groups(((0, 1), ()), MODELS)
    with pytest.raises(ValueError, match='exactly once'):
        validate_groups(((0,), (1,)), MODELS)
