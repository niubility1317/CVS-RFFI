import pytest
import torch

from experiments.pamf_phase3_restoration import response_joint_prior_dense_trial as dense
from experiments.pamf_phase3_restoration import response_joint_prior_trial as original


def test_dense_candidate_has_original_scratch_parameters_and_budget():
    old=original.factory(original.ARM);new=dense.factory(dense.ARM)
    assert old.state_dict().keys()==new.state_dict().keys()
    assert all(torch.equal(v,new.state_dict()[k]) for k,v in old.state_dict().items())
    before=original.configuration(original.ARM);after=dense.configuration(dense.ARM)
    for key in ('model_seed','epochs','updates','batch','source_draws','source_response_epsilon',
                'source_response_weight','source_response_directions','objective'):
        assert after[key]==before[key]
    assert after['training_solver'].startswith('Original dense FP64')
    assert after['prior_failed_run_loaded'] is False
    with pytest.raises(ValueError):dense.configuration('wrong')


def test_training_readout_uses_same_dense_method_as_deployment(monkeypatch):
    model=dense.factory(dense.ARM);received=torch.zeros(2,2,256);prior=object();result=object()
    monkeypatch.setattr(model,'_readout',lambda actual,actual_prior: result if actual is received and actual_prior is prior else None)
    assert model.source_readout(received,prior=prior) is result
