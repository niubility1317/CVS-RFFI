import pytest
import torch
from experiments.pamf_phase3_restoration.direct_bridge import training_state,iterate_four,scratch_direct_bridge
from experiments.pamf_phase3_restoration.state_bridge import scratch_state_bridge
from experiments.pamf_phase3_restoration import state_bridge_trial as trial


def test_observed_condition_has_an_exact_training_inverse_and_rollout_fixed_point():
    x=torch.randn(3,2,256);y=torch.randn_like(x);t=torch.tensor([.25,.5,.75]);z=training_state(x,y,t)
    recovered=(z-t[:,None,None]*y)/(1-t[:,None,None])
    torch.testing.assert_close(recovered,x,atol=2e-6,rtol=2e-6)
    first=torch.randn_like(x)
    def oracle(state,received,time):
        if (time==1).all():return first
        return (state-time[:,None,None]*received)/(1-time[:,None,None])
    torch.testing.assert_close(iterate_four(oracle,y),first,atol=5e-6,rtol=5e-6)


def activated():
    G=scratch_state_bridge(27).eval()
    with torch.no_grad():
        G.output.weight.normal_(std=.003)
        for layer in [*G.conditions,*G.time_conditions]:layer.weight.normal_(std=.002)
    return G


def test_estimator_cannot_access_original_observation_given_same_state_and_time():
    torch.set_num_threads(2);G=activated();state=torch.randn(2,2,256);t=torch.tensor([.5,.25])
    with torch.no_grad():
        a,aux=G.estimate(state,torch.randn_like(state),t,return_aux=True)
        b,bux=G.estimate(state,torch.randn_like(state)*100,t,return_aux=True)
    assert torch.equal(a,b) and torch.equal(aux['predicted_features'],bux['predicted_features'])


def test_scratch_matches_parent_initial_weights_except_removed_received_channels():
    torch.set_num_threads(2);G=scratch_state_bridge(2026092701);parent=scratch_direct_bridge(2026092701)
    assert sum(p.numel() for p in G.parameters())==4676898
    assert G.stem.in_channels==4 and G.initialization_report['ancestor_checkpoints']==[]
    for key,value in G.state_dict().items():
        expected=parent.state_dict()[key]
        if key=='stem.weight':expected=expected[:,[0,1,4,5]]
        assert torch.equal(value,expected)


def test_full_four_step_path_remains_deterministic_per_record_and_differentiable():
    torch.set_num_threads(2);G=activated();x=torch.randn(3,2,256);p=torch.tensor([2,0,1]);rng=torch.get_rng_state()
    with torch.no_grad():a=G(x);b=G(x[p]);single=G(x[:1])
    assert torch.equal(rng,torch.get_rng_state())
    torch.testing.assert_close(b,a[p],atol=3e-5,rtol=3e-5);torch.testing.assert_close(single,a[:1],atol=3e-5,rtol=3e-5)
    torch.testing.assert_close(a.square().sum(1).mean(1),torch.ones(3))
    value,_=G.estimate(x,x,torch.tensor([1.,.5,.25]),return_aux=True)
    (value*torch.randn_like(value)).sum().backward()
    assert G.stem.weight.grad.abs().sum()>0
    for layer in G.time_conditions:assert layer.weight.grad.abs().sum()>0


def test_exact_time_protocol_budget_and_source_only_routing(monkeypatch,tmp_path):
    c=trial.configuration(trial.ARM)
    for k in ['source_roles','source_draws','model_seed','data_seed','epochs','updates','batch','optimizer','lr','weight_decay','objective','bridge_times','bridge_time_probabilities','bridge_state_rng_seed','inference_core_calls']:
        assert c[k]==trial.TEMPLATE[k]
    assert c['estimator_inputs']=='current_intermediate_state_and_time_only/no_original_observed_condition'
    base=trial.parent.base
    for k in ['RUN','MODULE','ARMS','configuration','factory','train','feature_guide','gpu_choice']:monkeypatch.setattr(base,k,getattr(base,k))
    for k in ['RUN','MODULE','ARM','configuration','factory']:monkeypatch.setattr(trial.parent,k,getattr(trial.parent,k))
    trial.install(torch.ones(160));p=tmp_path/'old.pt'
    torch.save(dict(config=trial.TEMPLATE,code_commit='new',epoch=30,updates=7410,state={}),p)
    with pytest.raises(ValueError,match='Exact scratch30'):base.load(p,trial.ARM,'new','cpu')
