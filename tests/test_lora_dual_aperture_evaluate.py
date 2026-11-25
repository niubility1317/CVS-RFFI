import numpy as np
import pytest

from experiments.cric_dura_comparison.io import write
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


@pytest.mark.parametrize('experiment', (pilot, variance))
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
