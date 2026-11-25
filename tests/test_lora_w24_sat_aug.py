import numpy as np

from experiments.pamf_lora_main.w24_sat_aug import scene_slots, eligible_checkpoint


def test_source_augmentation_has_balanced_reproducible_independent_draws():
    a = scene_slots(2800, 0)
    b = scene_slots(2800, 1)
    assert np.array_equal(a, scene_slots(2800, 0))
    assert not np.array_equal(a, b)
    for draw in (a, b):
        counts = np.bincount(draw, minlength=6)
        assert counts.max() - counts.min() == 1


def test_source_clean_selection_compares_only_late_complete_checkpoints():
    assert not eligible_checkpoint(96, dict(correct=100, nll=.1), None)
    assert not eligible_checkpoint(101, dict(correct=100, nll=.1), None)
    assert eligible_checkpoint(100, dict(correct=98, nll=.1), None)
    assert eligible_checkpoint(108, dict(correct=98, nll=.1), dict(correct=98, nll=.2))
    assert not eligible_checkpoint(112, dict(correct=97, nll=.01), dict(correct=98, nll=.2))
