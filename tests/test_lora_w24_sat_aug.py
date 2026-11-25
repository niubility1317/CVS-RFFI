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


def test_extra_seed_has_distinct_prediction_receipt():
    from experiments.pamf_lora_main.w24_sat_aug import RUNS, MODES
    from experiments.pamf_lora_main.w24_native_full import receipt

    for seed in (2026092701, 2026092702, 2026092703):
        record = receipt(MODES, classifier_run=RUNS[seed], seed=seed)
        assert record["seed"] == seed
        assert record["classifier_run"].endswith(f"-s{seed}-r01")
        assert record["per_receiver"] == 2000
        assert not record["target_truth_read"]
    assert len(set(RUNS.values())) == 3
