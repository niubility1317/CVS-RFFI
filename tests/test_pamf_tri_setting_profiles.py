import pytest

from experiments.pamf_repl_component_guard import plan as base
from experiments.pamf_repl_source_adapt import setting_profiles as profile


def test_approved_profiles_and_only_missing_seed_training():
    rows = list(profile.matrix())
    assert len({row['row_id'] for row in rows}) == 33
    assert [(row['setting'], row['model_seed']) for row in profile.candidate_rows()] == [
        ('ten_class', 2026092702), ('ten_class', 2026092703)]
    for setting, expected in {
        'receiver_split_2': ('fusion_rx_balance', 3.),
        'ten_class': (profile.TEN, .75),
        'source_rx_1': ('hybrid_fusion_branch', None),
        'source_rx_6': ('raw_kl_same_day_fusion', .75),
        'labeled_1pct': ('original_tri', 6.),
        'labeled_5pct': ('fusion_rx_balance', 6.),
    }.items():
        assert [(row['method'], row['weight']) for row in rows if row['setting'] == setting] == [expected] * 3
    before = base.RUN, base.candidate_rows
    with profile.scoped():
        assert base.RUN == profile.RUN and base.candidate_rows() == profile.candidate_rows()
        assert base.RECIPES[profile.TEN] == profile.adaptive.RECIPES[profile.TEN]
    assert (base.RUN, base.candidate_rows) == before


def test_reuse_rejects_wrong_row_and_wrong_recipe():
    row = next(row for row in profile.matrix() if row['setting'] == 'ten_class')
    config = dict(run_id=row['reuse_run'], row_id=row['reuse_row'],
                  tuning_recipe=profile.adaptive.RECIPES[profile.TEN])
    profile.verify_recipe(row, config)
    with pytest.raises(ValueError, match='Readout row'):
        profile.verify_recipe(row, dict(config, row_id='another_ten_class'))
    with pytest.raises(ValueError, match='Historical recipe'):
        profile.verify_recipe(row, dict(config, tuning_recipe=dict(config['tuning_recipe'], weight=1.5)))
