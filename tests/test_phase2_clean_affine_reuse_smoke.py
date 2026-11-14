import pytest

from experiments.pamf_phase2_practical.clean_mopc_affine_reuse_smoke import run


def test_reuse_smoke_rejects_wrong_profile_before_data_access():
    with pytest.raises(ValueError, match='ProtoNet'):
        run({}, {'profile':'dadda_sda'}, 'unused', 'unused', 'unused', 10, 20, 'cpu')
