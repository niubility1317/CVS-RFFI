from experiments.phase1_dura_orthogonal.benchmark_flow_n607_sync import idle_uuid


def test_only_idle_selected_gpu_with_sufficient_memory_is_eligible():
    rows = '0, GPU-a, 20000\n1, GPU-b, 3000\n2, GPU-c, 20000\n'
    assert idle_uuid(rows, '', 0) == 'GPU-a'
    assert idle_uuid(rows, 'GPU-a, 123\n', 0) is None
    assert idle_uuid(rows, '', 1) is None
    assert idle_uuid(rows, 'GPU-a, 123\n', 2) == 'GPU-c'


def test_opt_in_shared_gpu_keeps_memory_and_three_total_process_limit():
    rows = '0, GPU-a, 20000\n1, GPU-b, 3000\n'
    assert idle_uuid(rows, 'GPU-a, 1\nGPU-a, 2\n', 0, allow_shared=True) == 'GPU-a'
    assert idle_uuid(rows, 'GPU-a, 1\nGPU-a, 2\nGPU-a, 3\n', 0, allow_shared=True) is None
    assert idle_uuid(rows, '', 1, allow_shared=True) is None
