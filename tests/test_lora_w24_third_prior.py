import unittest
from types import SimpleNamespace

from experiments.pamf_lora_main.w24_third_prior import (
    fit_inputs, promotion, restorer,
)


class Prior:
    def __init__(self, number):
        self.number = number


class TwoStage:
    def __init__(self):
        self.priors = [Prior(1), Prior(2)]

    def propose(self):
        return ("coarse", *(p.number for p in self.priors)), "action", "variance"


class TestThirdPrior(unittest.TestCase):
    def test_extra_prior_uses_independent_state_and_preserves_output_contract(self):
        archive = SimpleNamespace(DuraW24=TwoStage)
        two, three = restorer(archive, "two"), restorer(archive, "three")
        self.assertEqual(two.propose(), (("coarse", 1, 2), "action", "variance"))
        self.assertEqual(len(three.priors), 3)
        self.assertIsNot(three.priors[1], three.priors[2])
        three.priors[2].number = 3
        self.assertEqual(three.propose(), (("coarse", 1, 3), "action", "variance"))

    def test_fit_never_receives_inference_or_classifier_paths(self):
        data = dict(source_root="/source", run_root="/run", classifier_root="/reader",
                    native_target_root="/target", target_truth="/truth")
        self.assertEqual(fit_inputs(data, "three"), dict(source_root="/source", run_root="/run/three"))
        with self.assertRaises(ValueError):
            fit_inputs(data, "../target")

    def test_promotion_requires_benefit_and_clean_in_both_routes(self):
        def arm(clean, six, low):
            return {"by_scene": {"clean": clean}, "six_correct": six, "low_urban_correct": low}
        route = dict(direct=arm(296, 1300, 150), two_half=arm(295, 1350, 160), all_half=arm(295, 1352, 161),
                     three_half=arm(296, 1360, 165))
        self.assertTrue(promotion([route, route]))
        self.assertFalse(promotion([route, dict(route, three_half=arm(295, 1360, 165))]))
        self.assertFalse(promotion([route, dict(route, three_half=arm(296, 1360, 159))]))
        self.assertFalse(promotion([route]))

def test_third_prior_formal_records_both_arms_and_current_thresholds():
    from experiments.pamf_lora_main import w24_formal as formal, w24_formal_score as scorer

    row = formal.receipt(f'{formal.FULL}-0', ungated=True, third_prior=True)
    assert row['seeds'] == [2026092702]
    assert row['modes'][-2:] == ['two_half', 'three_half']
    assert row['reference_run'] == formal.RUN_UNGATED
    assert row['target_truth_read'] is False
    entry = dict(six_correct=162000, correct_by_scene=dict(clean=28565, practical_low_urban=25500))
    result = {formal.FULL: {'three_half': {'pooled': entry}}}
    verdict = scorer.acceptance(result, third_prior=True)
    assert verdict['full_correct_count_gaps'] == {'six': 0, 'low_urban': 0, 'clean': 1}
    assert not verdict['all_numeric_checks_pass']


def test_matched_append_preserves_archive_and_rejects_decision_mismatch(monkeypatch):
    import numpy as np
    import pytest
    from experiments.pamf_lora_main import w24_formal as formal

    monkeypatch.setattr(formal, 'LIMIT', 3)
    archive = np.array([[[0, 1, 2], [1, 1, 2], [0, 0, 2]]], dtype=np.int16)
    current = np.array([[[0, 1, 2], [2, 1, 2], [0, 1, 0]]], dtype=np.int16)
    rows, activity = formal.append_blind_prediction(archive, current, [1, 3], 3)
    np.testing.assert_array_equal(rows[:, :3], archive)
    np.testing.assert_array_equal(rows[:, 3:], current[:, 1:])
    assert activity == [1, 3, 3, 3]
    with pytest.raises(ValueError, match='Invalid archived or candidate restoration activity'):
        formal.append_blind_prediction(archive, current, [1, 3], 2)
    current[:, 0] = 7
    with pytest.raises(ValueError, match='Direct decisions must match'):
        formal.append_blind_prediction(archive, current, [1, 3], 3)
