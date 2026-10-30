from types import SimpleNamespace

import torch

from experiments.pamf_dura_direct_train import analytic_rf as trial


def test_auxiliary_reuses_one_restored_output_and_propagates_input_gradient(monkeypatch):
    signal=torch.tensor([2.],requires_grad=True);variance=torch.tensor([3.],requires_grad=True)
    restored=signal.reshape(1,1,1).expand(1,2,8);calls=[]
    terms={key:signal.square() for key in ('iq_nmse','prior_nmse','spectral_error','action_probe_nmse')}
    terms['log_variance_mse']=variance.square()
    def base_objective(*args,**kwargs):
        assert kwargs==dict(return_restored=True);calls.append(1)
        return torch.zeros(1),terms,restored
    monkeypatch.setattr(trial,'source_objective',base_objective)
    monkeypatch.setattr(trial,'extractor',lambda device:object())
    def auxiliary(value,clean,extractor):
        assert value is restored
        return {key:3*signal.square() for key in trial.BRANCHES}
    monkeypatch.setattr(trial,'analytic_terms',auxiliary)
    loss,logged=trial.objective(None,None,torch.ones(1,2,8),None,None,None)
    grad=torch.autograd.grad(loss,(signal,variance),allow_unused=True)
    assert calls==[1] and grad[1] is None
    torch.testing.assert_close(grad[0],(2.4+30*trial.ALPHA)*signal)
    assert set(trial.BRANCHES)==set(k.removeprefix('analytic_').removesuffix('_nmse') for k in logged if k.startswith('analytic_') and k!='analytic_RF_nmse')


def test_trial_preserves_scratch_budget_loss_identity_and_child_modes():
    base=trial.base
    original={key:getattr(base,key) for key in ('RUN','configuration','command','learning_rate','direct_objective','smoke','train')}
    try:
        trial.configure();config=base.configuration()
        assert config['epochs']==20 and config['updates']==4940 and config['training_stages']==1
        assert config['ancestor_checkpoints']==[] and not config['training_F_access']
        assert config['analytic_RF_weight']==.009527944261805803
        assert config['G_forwards_per_training_visit']==1 and len(config['extra_loss_metrics'])==6
        assert config['max_total_visible_GPU_processes']==3 and config['allowed_GPUs']==[4,6,7]
        args=SimpleNamespace(root='root',commit='a'*40)
        assert all(base.command(args,mode,'scratch_joint','target')[3]==trial.MODULE for mode in ('worker','score'))
    finally:
        for key,value in original.items():setattr(base,key,value)
