import numpy as np
import pytest

from experiments.cric_dura_comparison.io import write
from experiments.pamf_lora_main import dura_capacity_target as target, weak_guard_target as guard
from experiments.pamf_lora_main.data import VIEWS
from tools import lora_weak_guard_target_score as scoring


SOURCE_MODES = ('direct', 'reference', 'base_half', 'expanded_half')


def write_source_scores(root, route, *, candidate_six=610, candidate_clean=600):
    counts = {view: {'correct': [590, 600, 606, candidate_six]} for view in VIEWS[1:]}
    counts['clean'] = {'correct': [600, 600, 600, candidate_clean]}
    write(root / f'{route}-source-scores.json', dict(n=900, truth_after_predictions=True,
          target_reads=0, modes=SOURCE_MODES, metrics=counts))


@pytest.mark.parametrize('route,accuracy,clean', [('full', 600, 600), ('residual', 610, 599)])
def test_admission_requires_gain_and_clean_on_both_routes(tmp_path, route, accuracy, clean):
    write(tmp_path / 'state.json', {'status': 'SOURCE_SCORED'})
    for name in target.ROUTES:
        write_source_scores(tmp_path, name, candidate_six=accuracy if name == route else 610,
                            candidate_clean=clean if name == route else 600)
    with pytest.raises(ValueError, match=f'source admission for {route}'):
        target.require_source_admission(tmp_path, SOURCE_MODES)


def test_admission_requires_terminal_scores(tmp_path):
    state = tmp_path / 'state.json'
    write(state, {'status': 'FIT'})
    with pytest.raises(ValueError, match='must be complete'):
        target.require_source_admission(tmp_path, SOURCE_MODES)
    write(state, {'status': 'SOURCE_SCORED'}, exclusive=False)
    for route in target.ROUTES:
        write_source_scores(tmp_path, route)
    target.require_source_admission(tmp_path, SOURCE_MODES)


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
