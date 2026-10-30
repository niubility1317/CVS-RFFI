from types import SimpleNamespace

import torch
from torch import nn

from experiments.pamf_action_restore.operators import Action
from experiments.pamf_dura_direct_train.identity_init import initialize_identity
from experiments.pamf_dura_efficiency.model import direct_objective
from experiments.pamf_dura_prior_refine.model import stage_details


class Core(nn.Module):
    def __init__(self,incoming,outgoing):
        super().__init__();self.out=nn.Conv1d(incoming,outgoing,1)
    def forward(self,value):return self.out(value)


class ToyDURA(nn.Module):
    def __init__(self):
        super().__init__()
        self.frozen=nn.Module()
        self.frozen.coarse=nn.Module();self.frozen.coarse.temporal=Core(4,2)
        estimator=nn.Module();estimator.core=Core(4,4);estimator.global_head=nn.Linear(4,129)
        impulse=torch.zeros(64);impulse[24]=1;estimator.register_buffer('initial_kernel',impulse)
        self.frozen.estimator=estimator
        self.prior=nn.ModuleList([Core(9,2),Core(9,2)])
        self.initialization_report={}

    def estimate_context(self,received):
        base=received+self.frozen.coarse.temporal(torch.cat((received,received),1))
        estimator=self.frozen.estimator
        inputs=torch.cat((received,base),1)
        global_value=estimator.global_head(inputs.mean(-1))
        kernel=torch.complex(global_value[:,:64]+estimator.initial_kernel,global_value[:,64:128])
        kernel=kernel/kernel.abs().square().sum(1,keepdim=True).sqrt()
        coefficients=estimator.core(inputs)
        action=Action(kernel,1+torch.complex(coefficients[:,0],coefficients[:,1]),
                      torch.complex(coefficients[:,2],coefficients[:,3]))
        noise=torch.exp(-9.2103404+global_value[:,-1])
        return base,action,noise


def test_identity_initialization_preserves_iq_and_remains_trainable():
    torch.manual_seed(42)
    model=initialize_identity(ToyDURA())
    x=torch.randn(3,2,256)
    outputs,priors=stage_details(model,x)
    for value in outputs+priors:torch.testing.assert_close(value,x,atol=2e-6,rtol=2e-6)
    actual=Action(torch.ones(3,1,dtype=torch.complex64),torch.full((3,256),1.1,dtype=torch.complex64),torch.zeros(3,256,dtype=torch.complex64))
    probes=[torch.randn(3,256,dtype=torch.complex64) for _ in range(2)]
    loss,_=direct_objective(model,1.1*x,x,actual,torch.full((3,),.001),probes)
    loss.mean().backward()
    gradients=[p.grad for p in model.parameters() if p.grad is not None]
    assert all(torch.isfinite(v).all() for v in gradients)
    assert any(v.abs().sum()>0 for v in gradients)


def test_identity_recipe_propagates_to_training_and_score_children():
    from experiments.pamf_dura_direct_train import identity_init as recipe
    base=recipe.base
    original=(base.RUN,base.scratch_model,base.configuration,base.command,base.learning_rate)
    try:
        recipe.configure()
        args=SimpleNamespace(root='root',commit='a'*40)
        for mode in ('worker','score'):
            assert base.command(args,mode,'scratch_joint','target')[3]==recipe.MODULE
        config=base.configuration()
        assert config['run']==recipe.RUN and config['ancestor_checkpoints']==[]
        assert config['initialization']=='scratch_residual_identity'
    finally:
        base.RUN,base.scratch_model,base.configuration,base.command,base.learning_rate=original
