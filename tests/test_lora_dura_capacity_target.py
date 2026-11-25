import numpy as np
import pytest
import torch

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


def test_signed_covariance_ablation_uses_the_same_half_update_and_bypasses_exactly(monkeypatch):
    from experiments.pamf_lora_main import dura_signed_covariance_target as signed
    from experiments.pamf_lora_main.dura_repair import PacketRestorer, real_iq

    class ConstantEdit:
        def stages(self, y, quality, rms, **kwargs):
            return (y+.1, y+.2, y+.3), None, None, None, None

    tracker = PacketRestorer('consensus')
    iq = real_iq(torch.rand(2, 6, 256, dtype=torch.complex64))
    monkeypatch.setattr(signed, 'activation', lambda y: y.real.new_tensor([0., 2*signed.THRESHOLD]))
    original, candidate = signed.readouts([ConstantEdit()], tracker, -float('inf'))
    base, base_count = original(iq)
    new, count = candidate(iq)
    assert base_count == 2 and count == 1
    torch.testing.assert_close(new[1], base[1], rtol=0, atol=0)
    assert torch.equal(new[0], iq[0])


def test_signed_gate_rejects_a_coherent_one_sided_mode_and_preserves_pilot_contract():
    from experiments.pamf_lora_main import dura_dual_aperture_evaluate as pilot, dura_signed_covariance_target as signed
    from experiments.pamf_lora_main.dura_apertures import signed_covariance_rms

    vectors = torch.eye(6, 41, dtype=torch.complex64)[None]
    independent = torch.cat((vectors, torch.zeros_like(vectors)), dim=-1)
    repeated = independent[:, :1].expand(-1, 6, -1)
    assert float(signed_covariance_rms(repeated)) < 1e-4
    assert float(signed_covariance_rms(independent)) > signed.THRESHOLD
    matrix = signed.completion()
    assert matrix['seeds'] == [pilot.SEED] and matrix['per_receiver'] == 200
    assert matrix['routes'] == list(pilot.ROUTES) and matrix['target_truth_read'] is False
    assert len(signed.expected_files()) == 420


def test_shared_no_query_probe_has_no_legacy_training_dependency():
    iq = torch.zeros(12, 6, 256, dtype=torch.complex64)
    first = guard.smoke_disturbance(iq, torch.Generator().manual_seed(2026112481))
    again = guard.smoke_disturbance(iq, torch.Generator().manual_seed(2026112481))
    assert torch.equal(first, again) and torch.isfinite(first).all()
    assert first.shape == iq.shape and .07 < first.abs().square().mean().sqrt() < .09


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


def test_receiver_intervals_keep_seed_and_view_changes_on_one_packet(monkeypatch):
    monkeypatch.setattr(scoring, 'TARGET', ('rx_a', 'rx_b'))
    monkeypatch.setattr(scoring, 'SEEDS', (1, 2))
    monkeypatch.setattr(scoring, 'LIMIT', 4)
    monkeypatch.setattr(scoring, 'REPLICATES', 20)
    truth = np.zeros((2, 4), dtype=int)
    direct = np.zeros((2, len(VIEWS), *truth.shape), dtype=int)
    candidate = direct.copy()
    direct[0, 1:, 0, :] = 1
    candidate[1, 1:, 0, :] = 1
    outputs = {'full': {'direct': direct, 'expanded_half': candidate}}
    pairs = scoring.score_arrays(outputs, truth, 'expanded_half', detailed_pairing=True)[
        'paired_by_seed_and_receiver']['full']['direct']
    six = pairs['all_seeds']['six_practical']
    assert six['overall']['net'] == 0
    assert six['by_receiver']['rx_a']['repairs'] == six['by_receiver']['rx_a']['harms'] == 24
    assert six['by_receiver']['rx_a']['ci95_pp'] == [0., 0.]
    assert six['by_receiver']['rx_b']['net'] == 0
    assert pairs['1']['six_practical']['by_receiver']['rx_a']['ci95_pp'] == [100., 100.]
    assert pairs['2']['six_practical']['by_receiver']['rx_a']['ci95_pp'] == [-100., -100.]
    assert pairs['all_seeds']['clean']['overall']['candidate_correct'] == 16


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
