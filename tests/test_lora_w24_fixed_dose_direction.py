import numpy as np
import pytest
import torch

from experiments.pamf_lora_main import w24_fixed_dose_direction as experiment


def test_fit_config_rejects_evaluation_handles_before_opening_inputs():
    with pytest.raises(ValueError, match='U-only'):
        experiment.fit_root({**dict.fromkeys(experiment.FIT_FIELDS), 'classifier_root': 'forbidden'})


@pytest.mark.parametrize('dtype',(np.complex64,np.complex128))
def test_cpu_readout_keeps_the_deployed_shape_and_original_region_norm(monkeypatch, dtype):
    monkeypatch.setattr(experiment, 'native_aperture', lambda x: x.reshape(-1, 8, 1024)[..., :432].astype(dtype))
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


def test_cpu_recovery_bound_to_original_fit(tmp_path):
    runs = tmp_path/'runs'
    expected = runs/experiment.RUN
    runtime = dict(project=str(tmp_path),run_id=experiment.EVAL_RUN,run_root=str(runs/experiment.EVAL_RUN),
                   fixed_checkpoint_root=str(expected))
    assert experiment.checkpoint_root(runtime)==expected
    with pytest.raises(ValueError, match='checkpoint binding'):
        experiment.checkpoint_root(dict(runtime,fixed_checkpoint_root=str(runs/'different')))
