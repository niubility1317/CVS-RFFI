import torch
from torch import nn

from experiments.pamf_action_restore.operators import Action
from experiments.pamf_dura_efficiency.model import direct_objective


class SmallTwoStage(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale=nn.Parameter(torch.tensor(.8))
        self.lognoise=nn.Parameter(torch.tensor(-4.))
        self.prior=nn.ModuleList([nn.Conv1d(9,2,1),nn.Conv1d(9,2,1)])

    def estimate_context(self,received):
        count=len(received)
        kernel=torch.ones(count,1,dtype=torch.complex64)
        main=self.scale*torch.ones(count,256,dtype=torch.complex64)
        action=Action(kernel,main,torch.zeros_like(main))
        return .5*received,action,self.lognoise.exp().expand(count)


def test_single_joint_objective_gradients_do_not_enter_source_targets():
    torch.manual_seed(8)
    model=SmallTwoStage()
    received=torch.randn(3,2,256)
    clean=torch.randn(3,2,256,requires_grad=True)
    main=torch.ones(3,256,dtype=torch.complex64,requires_grad=True)
    actual=Action(torch.ones(3,1,dtype=torch.complex64),main,torch.zeros_like(main))
    variance=torch.full((3,),.02,requires_grad=True)
    probes=[torch.randn(3,256,dtype=torch.complex64) for _ in range(2)]
    loss,terms=direct_objective(model,received,clean,actual,variance,probes)
    assert loss.shape==(3,) and torch.isfinite(loss).all()
    loss.mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert model.scale.grad.abs()>0 and model.lognoise.grad.abs()>0
    assert clean.grad is None and main.grad is None and variance.grad is None
    assert set(terms)=={'iq_nmse','prior_nmse','spectral_error','action_probe_nmse','log_variance_mse'}
