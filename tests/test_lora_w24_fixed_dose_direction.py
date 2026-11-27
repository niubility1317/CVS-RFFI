import numpy as np
import pytest
import torch

from experiments.pamf_lora_main import w24_fixed_dose_direction as experiment


def test_fit_config_rejects_evaluation_handles_before_opening_inputs():
    with pytest.raises(ValueError, match='U-only'):
        experiment.fit_root({**dict.fromkeys(experiment.FIT_FIELDS), 'classifier_root': 'forbidden'})


def test_cpu_readout_keeps_the_deployed_shape_and_original_region_norm(monkeypatch):
    monkeypatch.setattr(experiment, 'native_aperture', lambda x: x.reshape(-1, 8, 1024)[..., :432])
    y = np.ones((1,8192),dtype=np.complex64)

    class Parent:
        def restore_native(self, received):
            return received + 2

    def proposal(received, base):
        return 2j*base

    direct, baseline, candidate = experiment.readouts(Parent(),proposal,y)
    assert direct.shape == baseline.shape == candidate.shape == (1, 8, 2, 432)
    assert torch.equal(direct,baseline)
    torch.testing.assert_close(candidate, torch.full_like(candidate, 2**-.5))
