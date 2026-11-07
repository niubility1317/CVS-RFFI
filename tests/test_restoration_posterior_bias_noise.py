import torch

from experiments.pamf_phase3_restoration.covariance_inverse import gaussian_posterior
from experiments.pamf_phase3_restoration.posterior_bias_noise import posterior_bias_noise


def test_split_matches_independent_observed_and_noiseless_solves():
    rng = torch.Generator().manual_seed(250)
    def rand(*shape):
        return torch.randn(*shape,generator=rng,dtype=torch.float64)
    action = rand(3,6,6)
    clean = rand(3,6)
    observed = (action@clean[:,:,None]).squeeze(-1) + .2*rand(3,6)
    mean = rand(6)
    shape = rand(6,6)
    covariance = shape@shape.T + .2*torch.eye(6,dtype=torch.float64)
    variance = torch.tensor([.03,.1,.7],dtype=torch.float64)
    actual,noiseless,unbiased = posterior_bias_noise(action,observed,clean,mean,covariance,variance)
    known = gaussian_posterior(action,observed,mean,covariance,variance)[0]
    without_noise = gaussian_posterior(action,(action@clean[:,:,None]).squeeze(-1),mean,covariance,variance)[0]
    torch.testing.assert_close(actual,known,rtol=1e-11,atol=1e-12)
    torch.testing.assert_close(noiseless,without_noise,rtol=1e-11,atol=1e-12)
    torch.testing.assert_close(unbiased,clean+known-without_noise,rtol=1e-11,atol=1e-12)
    torch.testing.assert_close(noiseless+unbiased-actual,clean,rtol=1e-11,atol=1e-12)


def test_zero_realized_noise_can_still_leave_regularization_bias():
    eye = torch.eye(6,dtype=torch.float64)
    clean = torch.tensor([[1.,2.,-3.,4.,-5.,6.]],dtype=torch.float64)
    original,noiseless,unbiased = posterior_bias_noise(eye[None],clean,clean,torch.zeros(6,dtype=torch.float64),
        eye,torch.ones(1,dtype=torch.float64))
    torch.testing.assert_close(original,clean/2)
    torch.testing.assert_close(noiseless,original)
    torch.testing.assert_close(unbiased,clean)


def test_reject_parent_with_target_access_before_loading_predictions(tmp_path):
    import json
    import pytest
    from experiments.pamf_phase3_restoration.posterior_bias_noise_source import CONTROL_COMMIT, load_control

    d=dict(status='SOURCE_ACTION_FACTOR_COMPLETE',code_commit=CONTROL_COMMIT,source_V_records=480,
        TX_labels_read=0,target_inputs=1)
    (tmp_path/'results.json').write_text(json.dumps(d),encoding='utf-8')
    (tmp_path/'complete.json').write_text('{"status":"ARTIFACTS_COMPLETE"}',encoding='utf-8')
    with pytest.raises(ValueError,match='Original complete known-operator'):
        load_control(tmp_path,None,None)
