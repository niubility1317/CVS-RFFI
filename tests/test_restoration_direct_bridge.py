import pytest
import torch
from experiments.pamf_phase3_restoration.direct_bridge import (
    training_state,training_times,iterate_four,scratch_direct_bridge)
from experiments.pamf_phase3_restoration.feature_conditioned import unit_power
from experiments.pamf_phase3_restoration import direct_bridge_trial as trial


def test_paired_bridge_has_exact_endpoints_and_detached_source_references():
    x=torch.randn(4,2,256,requires_grad=True);y=torch.randn_like(x,requires_grad=True)
    t=torch.tensor([1.,.75,.5,.25]);z=training_state(x,y,t)
    torch.testing.assert_close(z,(1-t[:,None,None])*x+t[:,None,None]*y)
    assert not z.requires_grad and torch.equal(z[0],y[0])
    with pytest.raises(ValueError):training_state(x,y,torch.zeros(4))
    with pytest.raises(ValueError):training_state(x,y[:1],t)


def test_update_uses_current_state_and_perfect_estimator_traces_the_paired_path():
    x=torch.randn(2,2,256);y=torch.randn_like(x);calls=[]
    def estimate(z,received,t):
        calls.append((z.clone(),received.clone(),t.clone()));return x
    actual=iterate_four(estimate,y)
    assert len(calls)==4 and torch.equal(actual,x)
    for i,(z,received,t) in enumerate(calls):
        expected_t=1-i/4
        assert torch.equal(received,y) and (t==expected_t).all()
        torch.testing.assert_close(z,(1-expected_t)*x+expected_t*y,atol=3e-7,rtol=1e-6)


def test_training_times_include_endpoint_half_and_each_intermediate_sixth():
    torch.manual_seed(91);t=training_times(60000,'cpu')
    assert set(t.tolist())=={1.,.75,.5,.25}
    assert abs(float((t==1).float().mean())-.5)<.01
    for v in [.75,.5,.25]:assert abs(float((t==v).float().mean())-1/6)<.01


def test_scratch_four_step_identity_unit_output_and_no_ancestors():
    torch.set_num_threads(2);G=scratch_direct_bridge(17);y=torch.randn(3,2,256)
    with torch.no_grad():actual=G(y)
    torch.testing.assert_close(actual,unit_power(y),atol=5e-7,rtol=2e-6)
    torch.testing.assert_close(actual.square().sum(1).mean(1),torch.ones(3))
    assert sum(p.numel() for p in G.parameters())==4677538
    assert G.initialization_report['ancestor_checkpoints']==[] and G.initialization_report['training_stages']==1


def activated():
    G=scratch_direct_bridge(18).eval()
    with torch.no_grad():
        G.output.weight.normal_(std=.003)
        for layer in [*G.conditions,*G.time_conditions]:layer.weight.normal_(std=.002)
    return G


def test_latent_and_all_time_conditions_receive_reconstruction_gradients():
    torch.set_num_threads(2);G=activated();y=torch.randn(2,2,256);z=torch.randn_like(y);t=torch.tensor([1.,.25])
    value,aux=G.estimate(z,y,t,return_aux=True)
    (value*torch.randn_like(value)).sum().backward()
    for layer in [*G.conditions,*G.time_conditions]:
        assert layer.weight.grad.abs().sum()>0 and torch.isfinite(layer.weight.grad).all()
    assert G.feature_regressor[-1].weight.grad.abs().sum()>0
    assert aux['predicted_features'].shape==(2,160)
    with pytest.raises(ValueError):G.estimate(z,y,torch.tensor([0.,1.]))


def test_full_deployment_is_deterministic_per_record_without_extra_observations():
    torch.set_num_threads(2);G=activated();y=torch.randn(3,2,256);p=torch.tensor([2,0,1]);before=torch.get_rng_state()
    with torch.no_grad():a=G(y);b=G(y[p]);single=G(y[:1])
    assert torch.equal(before,torch.get_rng_state())
    torch.testing.assert_close(b,a[p],atol=3e-5,rtol=3e-5);torch.testing.assert_close(single,a[:1],atol=3e-5,rtol=3e-5)


def test_strict_new_arm_budget_and_no_parent_checkpoint_loading(monkeypatch,tmp_path):
    c=trial.configuration(trial.ARM)
    for k in ['source_roles','source_draws','model_seed','data_seed','epochs','updates','batch','optimizer','lr','weight_decay']:
        assert c[k]==trial.TEMPLATE[k]
    assert c['inference_core_calls']==4 and c['training_core_calls_per_record']==1
    assert c['ancestor_checkpoints']==[] and c['bridge_state_rng_seed']==2026110103
    assert c['deployment_inputs']=='current_received_IQ_only/no_F0_inside_G'
    base=trial.base
    for k in ['RUN','MODULE','ARMS','configuration','factory','train','feature_guide','gpu_choice']:monkeypatch.setattr(base,k,getattr(base,k))
    trial.install(torch.ones(160));p=tmp_path/'old.pt'
    torch.save(dict(config=trial.TEMPLATE,code_commit='new',epoch=30,updates=7410,state={}),p)
    with pytest.raises(ValueError,match='Exact scratch30'):base.load(p,trial.ARM,'new','cpu')
