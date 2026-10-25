import pytest
from experiments.pamf_channel_pl.results import moments,combine_populations


def test_seed_mean_and_sample_sd_are_in_percentage_points():
    value=moments([90.,80.,85.])
    assert value['mean']==85. and value['sd']==5. and value['n']==3


def test_original_F0_target_population_is_weighted_RX_plus_both_only():
    value=combine_populations(dict(cross_rx=dict(count=126000,accuracy=.9),
        cross_both=dict(count=42000,accuracy=.8),cross_date=dict(count=30000,accuracy=1.)))
    assert value['count']==168000 and value['accuracy']==pytest.approx(.875)
