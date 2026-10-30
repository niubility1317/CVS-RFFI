from types import SimpleNamespace
import torch

from experiments.pamf_dura_direct_train import no_variance


def test_removed_variance_target_has_no_gradient_but_other_losses_remain(monkeypatch):
    signal=torch.tensor([2.],requires_grad=True)
    noise=torch.tensor([3.],requires_grad=True)
    terms={key:signal.square() for key in ('iq_nmse','prior_nmse','spectral_error','action_probe_nmse')}
    terms['log_variance_mse']=noise.square()
    monkeypatch.setattr(no_variance,'source_objective',lambda *a,**k:(sum(terms.values()),terms))
    loss,logged=no_variance.objective()
    gradient=torch.autograd.grad(loss,(signal,noise),allow_unused=True)
    assert gradient[0] is not None and gradient[0].abs()>0
    assert gradient[1] is None
    assert 'log_variance_mse' in logged


def test_removed_term_recipe_reaches_all_child_modes():
    base=no_variance.base
    original=(base.RUN,base.configuration,base.command,base.learning_rate,base.direct_objective)
    try:
        no_variance.configure()
        args=SimpleNamespace(root='root',commit='a'*40)
        for mode in ('worker','score'):
            assert base.command(args,mode,'scratch_joint','target')[3]==no_variance.MODULE
        assert base.configuration()['variance_aux_weight']==0.
        assert base.configuration()['ancestor_checkpoints']==[]
    finally:
        base.RUN,base.configuration,base.command,base.learning_rate,base.direct_objective=original
