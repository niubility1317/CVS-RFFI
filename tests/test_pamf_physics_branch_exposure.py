import pytest
from experiments.pamf_physics_branch.exposure import fixed_epoch_exposure


@pytest.mark.parametrize('count', [1, 124, 128, 256, 257, 11340, 22680, 34020, 45360, 56700, 58828, 62370, 68040])
def test_fixed_epoch_exposure_matches_actual_cycling_batches(count):
    sizes = [min(256, count-i) for i in range(0, count, 256)]
    actual = [sizes[i % len(sizes)] for i in range(222)]
    out = fixed_epoch_exposure(count)
    assert out['seen'] == sum(actual)
    assert out['selected'] == sum(min(128, n) for n in actual)


def test_invalid_loader_counts_are_not_silently_accepted():
    with pytest.raises(ValueError):
        fixed_epoch_exposure(0)
    with pytest.raises(ValueError):
        fixed_epoch_exposure(True)
