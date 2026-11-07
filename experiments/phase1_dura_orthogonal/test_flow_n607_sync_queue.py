from experiments.phase1_dura_orthogonal.benchmark_flow_n607_sync import idle_uuid


def test_only_idle_selected_gpu_with_sufficient_memory_is_eligible():
    rows = '0, GPU-a, 20000\n1, GPU-b, 3000\n2, GPU-c, 20000\n'
    assert idle_uuid(rows, '', 0) == 'GPU-a'
    assert idle_uuid(rows, 'GPU-a, 123\n', 0) is None
    assert idle_uuid(rows, '', 1) is None
    assert idle_uuid(rows, 'GPU-a, 123\n', 2) == 'GPU-c'
