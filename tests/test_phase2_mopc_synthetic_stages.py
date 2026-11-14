import pytest

from experiments.pamf_phase2_practical.group2_mopc_synthetic_diagnostic import run_variant


def test_rejects_invalid_number_of_new_classes_before_gpu_allocation():
    with pytest.raises(ValueError, match='new classes'):
        run_variant(None, semantics='public_code', seed=1731, new_count=3)
