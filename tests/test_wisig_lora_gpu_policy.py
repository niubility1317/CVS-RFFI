from experiments.pamf_unrolled_restore.wisig_lora_dispatch import available_gpus


def test_exclusive_policy_counts_other_compute_processes():
    devices = [(0, 'a', 16000), (1, 'b', 16000), (2, 'c', 3000)]
    assert available_gpus(devices, {'a'}, set(), False, 8000) == [1]


def test_authorized_sharing_still_reserves_memory_and_own_workers():
    devices = [(0, 'a', 16000), (1, 'b', 9000), (2, 'c', 3000)]
    assert available_gpus(devices, {'a', 'b'}, {0}, True, 8000) == [1]
