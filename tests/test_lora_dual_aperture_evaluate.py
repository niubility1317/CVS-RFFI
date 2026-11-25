import numpy as np
import pytest

from experiments.cric_dura_comparison.io import read, write
from experiments.pamf_lora_main import dura_balanced_transfer_target as balanced
from experiments.pamf_lora_main import dura_dual_aperture_evaluate as pilot, weak_guard_target as replay
from experiments.pamf_lora_main import dura_variance_profile_evaluate as variance
from experiments.pamf_lora_main.data import TARGET, VIEWS
from tools import lora_dual_aperture_pilot_score as scorer, lora_weak_guard_target_score as scoring


def test_pilot_matrix_does_not_change_archived_three_seed_contract():
    result = pilot.completion()
    assert result['routes'] == ['full', 'residual'] and result['seeds'] == [2026092702]
    assert result['per_receiver'] == 200 and not result['target_truth_read']
    assert len(pilot.expected_files()) == 2*15*7*2
    assert all('/2026092702/' in path for path in pilot.expected_files())
    assert replay.completion()['per_receiver'] == 2000
    assert len(replay.expected_files()) == 2*3*15*7*2


def test_must_reach_fixed_optimizer_endpoint_before_eval(tmp_path):
    terminal = dict(status='SOURCE_FROZEN', estimator='dual_aperture', steps=40000, threshold=.03)
    write(tmp_path/'dual_aperture-complete.json', terminal)
    write(tmp_path/'state.json', dict(status='SOURCE_TRAINING', steps=40000))
    with pytest.raises(ValueError, match='optimizer'):
        pilot.check_frozen(tmp_path, 'dual_aperture', 40000)
    write(tmp_path/'state.json', dict(status='SOURCE_FROZEN', steps=40000), exclusive=False)
    assert pilot.check_frozen(tmp_path, 'dual_aperture', 40000) == terminal


@pytest.mark.parametrize('experiment', (pilot, variance, balanced))
def test_align_pilot_contiguous_physical_prefix_and_controls_before_truth(tmp_path, monkeypatch, experiment):
    folder = tmp_path/'predictions'
    write(folder/'complete.json', experiment.completion())
    write(tmp_path/'smoke.json', dict(status='PASS', target_reads=0, seeds=[pilot.SEED]))
    if experiment is variance:
        variance.freeze_rule(tmp_path, None)
    expected = experiment.expected_files()
    for path in expected:
        file = folder/path
        file.parent.mkdir(parents=True, exist_ok=True)
        file.touch()
    controls = (tmp_path/'full', tmp_path/'res', {'expanded_half': tmp_path/'expanded'})
    monkeypatch.setattr(scorer, 'frozen_controls', lambda *args: controls)
    # Model and controls agree on the sealed physical prefix; archived suffix differs.
    prefix = np.zeros((len(VIEWS), len(TARGET), pilot.PER_RECEIVER), dtype=np.int16)
    original = np.concatenate((prefix, np.ones_like(prefix)), -1)

    def predictions(path, mode, count=2000):
        return prefix if count == pilot.PER_RECEIVER else original

    monkeypatch.setattr(scoring, 'predictions', predictions)
    result = scorer.frozen_rows(tmp_path, None, None, None, None, experiment=experiment)
    assert result['full'][experiment.MODE].shape == (1, 7, 15, 200)
    assert (result['residual']['dura_half'] == 0).all()

    def drift(path, mode, count=2000):
        value = predictions(path, mode, count)
        return value + int(mode == 'direct' and count == pilot.PER_RECEIVER)

    monkeypatch.setattr(scoring, 'predictions', drift)
    with pytest.raises(ValueError, match='direct decisions differ'):
        scorer.score(tmp_path, None, None, None, None, tmp_path/'absent_truth', experiment=experiment)


def test_balanced_transfer_wait_requires_closed_source_and_copies_negative_reference(tmp_path, monkeypatch):
    source_run = tmp_path/'source'
    pilot_run = tmp_path/'pilot'
    source_run.mkdir()
    pilot_run.mkdir()
    monkeypatch.setattr(balanced, 'source_release',
                        lambda rt: (dict(run_root=str(source_run)), source_run))
    mode = balanced.training.MODES
    predictions = dict(ids=list(range(900)), modes=list(mode),
                       rows={view: np.zeros((len(mode), 900), dtype=int).tolist() for view in VIEWS},
                       metrics={view: dict(active=0) for view in VIEWS})
    scores = dict(status='SOURCE_SCORED', target_reads=0, truth_after_predictions=True,
                  n=900, modes=list(mode), six=[15, 17, 18, 12],
                  metrics={'clean': {'correct': [6, 6, 6, 4]}})
    for route in balanced.ROUTES:
        write(source_run/f'{route}-predictions.json', predictions)
        write(source_run/f'{route}-source-scores.json', dict(scores, route=route))
    write(source_run/'state.json', dict(status='FIT'))
    with pytest.raises(RuntimeError, match='without a closed source matrix'):
        balanced.await_source(dict(training_dispatcher=-1), pilot_run, pilot_run/'unclosed.json')
    assert not (pilot_run/'source-promotion.json').exists()
    write(source_run/'state.json', dict(status='SOURCE_SCORED'), exclusive=False)
    balanced.await_source(dict(training_dispatcher=-1), pilot_run, pilot_run/'closed.json')
    assert read(pilot_run/'source-promotion.json') == dict(reference_only=True, target_reads=0,
        pairs={route: dict(six_net=-6, clean_net=-2) for route in balanced.ROUTES})
