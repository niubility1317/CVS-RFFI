from types import SimpleNamespace
import torch
from experiments.pamf_dura_direct_train import strong_action


def test_only_operator_probe_gradient_weight_is_quadrupled(monkeypatch):
    signal=torch.tensor([2.],requires_grad=True)
    action=torch.tensor([3.],requires_grad=True)
    terms={key:signal.square() for key in ('iq_nmse','prior_nmse','spectral_error','log_variance_mse')}
    terms['action_probe_nmse']=action.square()
    original=terms['iq_nmse']+.1*terms['prior_nmse']+.05*terms['spectral_error']+.05*terms['action_probe_nmse']+.001*terms['log_variance_mse']
    monkeypatch.setattr(strong_action,'source_objective',lambda *a,**k:(original,terms))
    first=torch.autograd.grad(original,(signal,action),retain_graph=True)
    actual,logged=strong_action.objective()
    second=torch.autograd.grad(actual,(signal,action))
    torch.testing.assert_close(first[0],second[0])
    torch.testing.assert_close(4*first[1],second[1])
    assert logged is terms


def test_source_action_recipe_reaches_all_child_modes():
    base=strong_action.base
    original=(base.RUN,base.configuration,base.command,base.learning_rate,base.direct_objective)
    try:
        strong_action.configure()
        args=SimpleNamespace(root='root',commit='a'*40)
        for mode in ('worker','score'):
            assert base.command(args,mode,'scratch_joint','target')[3]==strong_action.MODULE
        value=base.configuration()
        assert value['action_aux_weight']==.2 and value['epochs']==20 and value['updates']==4940
        assert value['ancestor_checkpoints']==[] and value['teacher_model'] is None
    finally:
        base.RUN,base.configuration,base.command,base.learning_rate,base.direct_objective=original
