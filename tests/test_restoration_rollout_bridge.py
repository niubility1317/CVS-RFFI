import pytest
import torch
from experiments.pamf_phase3_restoration import rollout_bridge_trial as trial
from experiments.pamf_phase3_restoration.state_bridge import scratch_state_bridge

torch.set_num_threads(2)


def test_actual_rollout_is_deployed_function_and_independent_per_record():
    model=trial.factory(trial.ARM).eval().requires_grad_(False)
    with torch.no_grad():
        model.output.weight.normal_(0,.001)
        received=torch.randn(3,2,256)
        value,aux,path=trial.rollout(model,received,trace=True)
        assert torch.equal(value,model(received))
        torch.testing.assert_close(value,torch.cat([model(x[None]) for x in received]),atol=2e-6,rtol=2e-5)
        assert len(path)==4 and aux['predicted_features'].shape==(3,160)
        assert torch.isfinite(value).all()


def test_final_loss_reaches_every_step_without_clean_state_input():
    class Probe:
        def __init__(self):
            self.weights=[torch.tensor(float(i+1),requires_grad=True) for i in range(4)]
            self.seen=[]
        def estimate(self,state,unused,t,*,return_aux):
            i=len(self.seen);self.seen.append((state,t))
            return state+.01*self.weights[i],dict(predicted_features=state.mean(-1))
    model=Probe();received=torch.randn(2,2,256)
    result,_,path=trial.rollout(model,received,trace=True)
    for value in path:value.retain_grad()
    result.square().mean().backward()
    assert [float(t[0]) for _,t in model.seen]==[1.,.75,.5,.25]
    assert model.seen[0][0] is received
    assert all(w.grad is not None and w.grad.abs()>0 for w in model.weights)
    assert all(v.grad is not None and torch.isfinite(v.grad).all() and v.grad.abs().sum()>0 for v in path)


def test_scratch_initial_weights_exactly_match_state_control():
    model=trial.factory(trial.ARM);control=scratch_state_bridge(2026092701)
    assert sum(p.numel() for p in model.parameters())==4676898
    assert all(torch.equal(v,control.state_dict()[k]) for k,v in model.state_dict().items())
    config=trial.configuration(trial.ARM)
    assert (config['epochs'],config['updates'],config['training_stages'])==(30,7410,1)
    assert config['initialization']=='scratch' and config['ancestor_checkpoints']==[]
    assert config['training_core_calls_per_record']==config['inference_core_calls']==4
    assert config['training_clean_state_inputs']==config['bridge_time_random_draws']==0
    assert config['checkpoint_selection']=='final30_only'
    assert all(config[k]==0 for k in ['TX_labels_read','V_reads','OE_reads','target_training_reads'])
    with pytest.raises(ValueError):trial.configuration('MSAT_StateBridge')


def test_real_network_final_objective_keeps_all_step_gradients_finite():
    model=trial.factory(trial.ARM)
    received=torch.randn(2,2,256);target=torch.randn_like(received)
    restored,aux,path=trial.rollout(model,received,trace=True)
    for value in path:value.retain_grad()
    ((restored-target).square().mean()+aux['predicted_features'].square().mean()).backward()
    assert all(v.grad is not None and torch.isfinite(v.grad).all() and v.grad.abs().sum()>0 for v in path)
    assert model.output.weight.grad.abs().sum()>0
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
