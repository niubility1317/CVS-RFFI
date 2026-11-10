import copy
import json

import numpy as np
import pytest

from experiments.pamf_trisig_dura_baseline import evaluation as e


def test_plan_is_three_seed_matched_seven_scenes_and_fixed_dura():
    plan = e.plan()
    assert plan['reuse_seed'] == 2026092701
    assert plan['new_seeds'] == [2026092702, 2026092703]
    assert plan['views'] == list(e.VIEWS)
    assert plan['tasks'] == ['cross_rx', 'cross_both']
    assert plan['restorer'] == e.DURA_CHECKPOINT


def test_rejects_truth_or_wrong_baseline_identity_in_predictor():
    c = e.predict_config('/project', 2026092702)
    e.validate_prediction_config(c)
    for change in ({'truth': '/target/truth.json'}, {'model_seed': 2026092701},
                   {'views': ['clean', 'leo_clear_weak']}, {'restorer': '/other.pt'}):
        altered = copy.deepcopy(c)
        altered.update(change)
        with pytest.raises(ValueError):
            e.validate_prediction_config(altered)


def test_cached_physical_samples_must_match_raw_predictions():
    ids = np.array([3, 5, 7])
    assert e.check_ids(ids, ids.copy(), ids.copy()) == 3
    with pytest.raises(ValueError):
        e.check_ids(ids, ids[::-1], ids)
    with pytest.raises(ValueError):
        e.check_ids(ids, ids, np.array([3, 5, 8]))


def test_no_truth_until_both_new_seeds_have_all_frozen_predictions(tmp_path):
    with pytest.raises(ValueError, match='Both'):
        e.check_prediction_barrier(tmp_path)
    for seed in (2026092702, 2026092703):
        path = tmp_path / str(seed)
        path.mkdir()
        (path / 'complete.json').write_text(json.dumps({
            'status': 'ALL_PREDICTIONS_COMPLETE', 'model_seed': seed,
            'truth_read': False, 'views': list(e.VIEWS), 'tests': list(e.TASKS),
            'readouts': ['raw', 'DURA_v1']}))
    e.check_prediction_barrier(tmp_path)
    (tmp_path / '2026092702' / 'complete.json').unlink()
    with pytest.raises(ValueError, match='Both'):
        e.check_prediction_barrier(tmp_path)
