import pytest
from experiments.pamf_physics_branch.exposure import fixed_epoch_exposure
from experiments.cric_dura_comparison.f_data import FixedEpochLoader


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


@pytest.mark.parametrize('count', [11340, 45360, 56700])
def test_exposure_matches_real_fixed_epoch_loader(count):
    class Dataset:
        def __len__(self):
            return count
        def __getitem__(self, index):
            return index, 0
    loader = FixedEpochLoader(Dataset(), 256, seed=42, steps=222, drop_last=False)
    sizes = [len(batch[0]) for batch in loader]
    expected = fixed_epoch_exposure(count)
    assert len(sizes) == expected['steps']
    assert sum(sizes) == expected['seen']
    assert sum(min(128, size) for size in sizes) == expected['selected']
