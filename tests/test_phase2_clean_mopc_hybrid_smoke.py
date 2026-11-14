import pytest
from experiments.pamf_phase2_practical.clean_mopc_hybrid_diagnostic import run


def test_hybrid_smoke_rejects_unapproved_profile_before_data_access():
    with pytest.raises(ValueError,match='profile'):
        run({},'radionet_ada_knn','capsules/P01/MIX01/1-1',2026092702,5,20,'cpu')
