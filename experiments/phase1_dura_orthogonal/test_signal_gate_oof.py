import numpy as np
import pytest
from experiments.phase1_dura_orthogonal.signal_gate_oof_train import select_oof_ids


def test_oof_selection_excludes_parent_pilot_and_protected_days():
    rx=np.array([1,3,4,4,6,8,1,1]); day=np.array([1,1,3,2,3,1,1,1])
    roles=np.array(['L_s','U_s','L_s','U_s','U_s','L_s','U_s','L_s'])
    chosen=select_oof_ids(rx,day,roles,np.array([0,1]),np.array([6]),expected_size=3)
    assert chosen.tolist()==[3,5,7]
    roles[5]='V'
    with pytest.raises(ValueError):
        select_oof_ids(rx,day,roles,np.array([0,1]),np.array([6]),expected_size=3)


def test_ground_rejects_parent_fit_overlap():
    from experiments.phase1_dura_orthogonal.signal_gate_oof_ground import validate_gate_fit
    fit=np.arange(17632)
    validate_gate_fit(fit,np.array([20000,20001]))
    with pytest.raises(ValueError):
        validate_gate_fit(fit,np.array([1,20001]))
