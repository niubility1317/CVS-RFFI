import numpy as np
import pytest
import torch
from experiments.pamf_phase3_restoration.physical import scratch_physical,MIN_NOISE
from experiments.pamf_phase3_restoration.physical_diagnostic import true_trace,factorial,restore_with,summarize,METHODS,METRICS,SCENES


def test_factorial_blind_replays_deployment_and_true_noise_uses_source_floor():
    torch.set_num_threads(2);model=scratch_physical(7).eval()
    with torch.no_grad():
        for name,p in model.named_parameters():
            if name.startswith(('estimator.coefficient_head.','estimator.kernel_head.','priors.0.output.','priors.1.output.')):p.normal_(std=.002)
        x=torch.randn(2,2,256);kernel=torch.zeros(2,2,64);kernel[:,0,0]=1
        c=torch.zeros(2,4,256);c[:,0]=1
        action,variance=true_trace(kernel,c,torch.tensor([1e-8,.003]))
        assert variance[0]==MIN_NOISE
        paths,estimated,noise=factorial(model,x,action,variance)
        torch.testing.assert_close(paths['blind'],model(x),atol=0,rtol=0)
        torch.testing.assert_close(paths['true_H'],restore_with(model,x,action,noise),atol=0,rtol=0)
        torch.testing.assert_close(paths['true_noise'],restore_with(model,x,estimated,variance),atol=0,rtol=0)
        assert all(torch.isfinite(v).all() for v in paths.values())


def test_source_trace_and_noise_reject_invalid_external_values():
    model=scratch_physical(1);kernel=torch.zeros(2,2,64);c=torch.zeros(2,4,256)
    with pytest.raises(ValueError,match='traces'):true_trace(kernel,c,torch.tensor([1.,-1.]))
    with pytest.raises(ValueError,match='traces'):true_trace(kernel,c[:1],torch.ones(2))
    kernel[0,0,0]=float('nan')
    with pytest.raises(ValueError,match='traces'):true_trace(kernel,c,torch.ones(2))
    with pytest.raises(ValueError,match='positive'):restore_with(model,torch.zeros(2,2,256),None,torch.zeros(2))


def test_partial_diagnostic_cannot_be_complete_and_pairs_match_counts():
    with pytest.raises(ValueError,match='All10240'):summarize({'scene':np.asarray(SCENES)})
    records={'scene':np.resize(np.asarray(SCENES),10240)}
    for method in METHODS:
        for metric in METRICS:records[method+'__'+metric]=np.ones(10240)
    for key in ('estimated_action_forward_nmse','estimated_noise_log_mse'):records[key]=np.zeros(10240)
    records['blind__clean_decision_agreement'][0]=0
    result=summarize(records)
    assert sum(result[s]['count'] for s in SCENES)==10240
    assert result['ALL']['paired_against_blind']['true_H']==dict(repairs=1,damage=0,net=1)
