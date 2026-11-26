import torch

from experiments.pamf_lora_main.w24_observed_dose import readout_gain, source_qualifies


def test_weighted_noise_gain_is_packet_local_and_scale_invariant():
    variance = torch.tensor([[[0.], [0.]], [[.275**2], [.275**2]], [[4.], [4.]]])
    rms = torch.tensor([[[1.], [2.]], [[3.], [4.]], [[8.], [16.]]])
    result = readout_gain(variance, rms)
    torch.testing.assert_close(result, readout_gain(variance, rms * 8))
    torch.testing.assert_close(result[0], torch.tensor([[[.5]]]).squeeze(0))
    torch.testing.assert_close(result[1], torch.ones_like(result[1]))
    assert 1. < result[2].item() < 1.5


def test_full_source_rule_requires_material_six_low_and_clean():
    def arm(six, low, clean):
        return dict(six_correct=six, low_urban_correct=low, by_scene={'clean': clean})
    baseline = dict(direct=arm(1394,150,297), outer_half=arm(1441,177,297))
    assert source_qualifies(dict(baseline, observed_dose=arm(1459,187,297)))
    assert not source_qualifies(dict(baseline, observed_dose=arm(1458,187,297)))
    assert not source_qualifies(dict(baseline, observed_dose=arm(1459,186,297)))
    assert not source_qualifies(dict(baseline, observed_dose=arm(1459,187,296)))


def test_formal_plan_uses_all_full_receivers_and_no_residual():
    import pytest
    from experiments.pamf_lora_main import w24_observed_dose as dose
    from experiments.pamf_lora_main.w24_native_full import TARGET

    partials = [dose.formal_receipt(case) for case in dose.FULL_CASES]
    assert len(partials) == 3
    assert sum([p['receivers'] for p in partials], []) == list(TARGET)
    assert {p['processing_route'] for p in partials} == {'full_native'}
    assert all(p['modes'][-2:] == ['outer_objective_half', 'observed_dose'] for p in partials)
    with pytest.raises(ValueError, match='Only Full'):
        dose.formal_receipt('native_residual_w24_v1-0')
