import pytest

from experiments.phase1_dura_orthogonal.gpu_capacity_queue import capacity


def test_capacity_counts_all_users_on_selected_gpu_only():
    gpu = '6, GPU-A, 5000\n7, GPU-B, 9999\n'
    assert capacity(gpu, 'GPU-A, 1\nGPU-B, 2\nGPU-A, 3\n', 6) == (True, 5000, 2)
    assert capacity(gpu, 'GPU-A, 1\nGPU-A, 2\nGPU-A, 3\n', 6) == (False, 5000, 3)
    assert capacity('6, GPU-A, 4095\n', '', 6) == (False, 4095, 0)
    with pytest.raises(ValueError):
        capacity(gpu, '', 4)
