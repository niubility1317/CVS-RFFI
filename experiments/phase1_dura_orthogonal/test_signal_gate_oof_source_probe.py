import numpy as np
import pytest
from experiments.phase1_dura_orthogonal.signal_gate_oof_source_probe import validate_pilot_disjoint


def test_gate_fit_must_exclude_pilot():
    validate_pilot_disjoint(np.array([1,2]),np.array([3,4]))
    with pytest.raises(ValueError):
        validate_pilot_disjoint(np.array([1,2]),np.array([2,4]))
