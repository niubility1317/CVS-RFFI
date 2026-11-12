from experiments.pamf_physics_tri.concurrency_takeover import LIMITS, may_launch


def test_limits_count_other_owners_and_gpu_seven_cap():
    assert LIMITS[0] == 6 and LIMITS[7] == 4
    assert may_launch({1, 2, 3, 4, 5}, set(), set(), 9000, 6000, LIMITS[0])
    assert not may_launch({1, 2, 3, 4, 5, 6}, set(), set(), 9000, 6000, LIMITS[0])
    assert not may_launch({1, 2, 3, 4}, set(), set(), 9000, 6000, LIMITS[7])


def test_initializing_worker_reserves_memory_and_counts_once():
    assert not may_launch({1}, {1, 2}, {2}, 10000, 6000, 6)
    assert may_launch({1}, {1, 2}, {2}, 12000, 6000, 6)
