import torch

from experiments.pamf_lora_main.w24_first_direction import direction_risk
from experiments.pamf_lora_main.w24_third_prior import promotion, trial
from experiments.pamf_lora_main.w24_formal import FULL, cases, evaluation_modes


def test_direction_respects_complex_phase_and_leaves_clean_to_identity_loss():
    target = torch.tensor([[[1 + 1j, 2 - 1j]], [[0j, 0j]]])
    clean_edit = torch.tensor([[[0j, 0j]], [[100 + 90j, -30j]]])
    aligned = direction_risk(target + clean_edit, target)
    rotated = direction_risk(1j * target + clean_edit, target)
    reversed_edit = direction_risk(-target + clean_edit, target)
    assert 0 <= aligned < 0.01
    torch.testing.assert_close(rotated, torch.tensor(1.0))
    assert reversed_edit > 1.99
    assert direction_risk(clean_edit, torch.zeros_like(target)) == 0


def test_direction_has_finite_gradient_from_zero_initial_edit():
    target = torch.tensor([[[1 + 1j, 2 - 1j]], [[0j, 0j]]])
    correction = torch.zeros_like(target, requires_grad=True)
    direction_risk(correction, target).backward()
    assert torch.isfinite(correction.grad).all()
    assert correction.grad[0].abs().sum() > 0
    assert not correction.grad[1].any()


def test_direction_trial_has_its_own_source_and_formal_identities():
    spec = trial('direction')
    assert spec.arms == ('direction_control', 'direction_first')
    assert spec.source_modes[-2:] == ('direction_control_full', 'direction_first_full')
    assert evaluation_modes(FULL, True, third_prior='direction')[-2:] == spec.source_modes[2:4]
    assert trial(True).source_modes[-2:] == ('two_full', 'three_full')


def test_direction_runs_only_three_full_cases_and_retains_clean_source_gate():
    assert set(cases('direction')) == {f'{FULL}-{i}' for i in range(3)}
    assert len(cases(True)) == 6

    def score(clean, six, low):
        return dict(by_scene={'clean': clean}, six_correct=six, low_urban_correct=low)

    route = dict(direct=score(297, 1394, 150), all_half=score(293, 1434, 164),
                 direction_control_half=score(296, 1420, 160), direction_first_half=score(297, 1440, 164))
    assert promotion([route], 'direction')
    assert not promotion([route, route], 'direction')
    route['direction_first_half'] = score(296, 1440, 164)
    assert not promotion([route], 'direction')
