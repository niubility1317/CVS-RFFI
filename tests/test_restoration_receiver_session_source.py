import numpy as np
from leo_practical.channel import receiver_for_session

from experiments.pamf_phase3_restoration.receiver_session_source import (
    ARMS,
    TRAINING_SEED,
    VALIDATION_SEED,
    profile_assignments,
    replay,
    scene_config,
)


def test_balanced_rotation_is_order_independent_and_never_retains_its_own_profile():
    rx = np.repeat([1, 3, 4, 6, 8], 6)
    day = np.tile(np.repeat([1, 2, 3], 2), 5)
    original, rotated = profile_assignments(rx, day)
    assert len(set(original)) == 15
    assert sorted(original) == sorted(rotated)
    assert all(left != right for left, right in zip(original, rotated))
    permutation = np.random.default_rng(29).permutation(len(rx))
    again = profile_assignments(rx[permutation], day[permutation])
    assert again == [[profiles[i] for i in permutation] for profiles in [original, rotated]]


def test_same_profile_replays_iq_and_swapped_hardware_preserves_latent():
    cfg = scene_config('practical_low_urban')
    original = receiver_for_session(cfg, VALIDATION_SEED, 'rx1:day1:eq1')
    other = receiver_for_session(cfg, TRAINING_SEED, 'rx1:day1:eq1')
    x = np.random.default_rng(61).normal(size=(2, 256)).astype(np.float32)
    result = replay(x, 'synthetic-unit-test-source-record', cfg, (original, other, original))
    assert tuple(result) == ARMS
    np.testing.assert_array_equal(result['original'][0], result['rotated_val_profiles'][0])
    assert result['original'][1:] == result['rotated_val_profiles'][1:]
    assert not np.array_equal(result['original'][0], result['train_profiles'][0])
