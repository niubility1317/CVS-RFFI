import numpy as np
import pytest

from experiments.cric_dura_comparison.io import write
from experiments.pamf_lora_main import dura_capacity_target as target, weak_guard_target as guard
from experiments.pamf_lora_main.data import VIEWS
from tools import lora_weak_guard_target_score as scoring


@pytest.mark.parametrize('record', [None, dict(status='SOURCE_FROZEN', estimator='expanded', steps=39000),
                                    dict(status='SOURCE_FROZEN', estimator='base', steps=40000)])
def test_replay_requires_fixed_terminal_model(tmp_path, record):
    if record is not None:
        write(tmp_path / 'expanded-complete.json', record)
    with pytest.raises(ValueError, match='terminal'):
        target.frozen_fit(tmp_path, 40000)


def test_frozen_model_can_replay_without_source_accuracy(tmp_path):
    write(tmp_path / 'state.json', {'status': 'PREDICT'})
    record = dict(status='SOURCE_FROZEN', estimator='expanded', steps=40000, threshold=.123)
    write(tmp_path / 'expanded-complete.json', record)
    assert target.frozen_fit(tmp_path, 40000) == record
    for route in target.ROUTES:
        write(tmp_path / f'{route}-source-scores.json', {'accuracy': 0.})
    assert target.frozen_fit(tmp_path, 40000) == record


def test_target_matrix_keeps_same_routes_packets_and_readers():
    current, previous = target.completion(), guard.completion()
    for field in ('routes', 'seeds', 'receivers', 'per_receiver', 'views', 'target_truth_read'):
        assert current[field] == previous[field]
    assert current['modes'] == ['direct', 'expanded_half']
    assert len(target.expected_files()) == 1260
    assert target.expected_files() == {path.replace('__guard_half.npy', '__expanded_half.npy')
                                      for path in guard.expected_files()}


def test_paired_scoring_repair_and_harm_for_both_candidate_names(monkeypatch):
    monkeypatch.setattr(scoring, 'TARGET', ('receiver',))
    monkeypatch.setattr(scoring, 'SEEDS', (1,))
    monkeypatch.setattr(scoring, 'LIMIT', 4)
    monkeypatch.setattr(scoring, 'REPLICATES', 20)
    truth = np.asarray([[0, 1, 0, 1]])
    direct = np.broadcast_to(truth, (1, len(VIEWS), 1, 4)).copy()
    direct[..., 0] = 1
    candidate = direct.copy()
    candidate[:, 1:, :, 0] = 0
    candidate[:, -1, :, 1] = 0
    for mode in ('guard_half', 'expanded_half'):
        scores = scoring.score_arrays({'full': {'direct': direct, mode: candidate}}, truth, mode)
        aggregate = scores['aggregate']['full'][mode]
        paired = scores['paired']['full']['direct']['six_scene']
        assert aggregate == {'clean': 3, 'six_scene': 23, 'low_urban': 3}
        assert (paired['repairs'], paired['harms'], paired['net']) == (6, 1, 5)
        assert f'{mode}_vs' in scores['rows'][0]


def test_acceptance_uses_formal_thresholds_and_both_clean_counts():
    aggregate = {
        'full': {'dura_half': {'six_scene': 301599, 'clean': 64490},
                 'expanded_half': {'six_scene': 361800, 'low_urban': 47700, 'clean': 64490}},
        'residual': {'dura_half': {'six_scene': 289162, 'clean': 64307},
                     'expanded_half': {'six_scene': 289163, 'clean': 64307}},
    }
    condition = scoring.dura_capacity_acceptance(aggregate)
    assert condition['passed'] and not condition['full_clean_gap_within_3pp']

    aggregate['full']['expanded_half']['low_urban'] -= 1
    aggregate['residual']['expanded_half']['clean'] -= 1
    failed = scoring.dura_capacity_acceptance(aggregate)
    assert not failed['passed']
    assert not failed['full_minima']['low_urban']['passed']
    assert failed['versus_original_dura']['residual'] == {'six_gain': 1, 'clean_gain': -1, 'passed': False}
