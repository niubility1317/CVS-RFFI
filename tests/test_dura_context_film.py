import copy
import pytest
import torch
from torch import nn
from experiments.pamf_dura_efficiency.context_film import ContextPrior,descriptor,BOUND


class Block(nn.Module):
    def __init__(self,inputs,outputs):
        super().__init__();self.net=nn.Sequential(nn.Conv1d(inputs,outputs,1),nn.SiLU())
    def forward(self,x):return self.net(x)


class Core(nn.Module):
    def __init__(self):
        super().__init__();self.conditional=False;self.encoder=nn.Module();widths=(4,8,12,16)
        self.encoder.blocks=nn.ModuleList(nn.Sequential(Block(a,b)) for a,b in zip((9,*widths[:-1]),widths))
        self.decode=nn.ModuleList(Block(a+b,b) for a,b in zip(widths[:0:-1],widths[-2::-1]));self.out=nn.Conv1d(4,2,1)
    def forward(self,x):
        f=[]
        for i,b in enumerate(self.encoder.blocks):x=b(torch.nn.functional.avg_pool1d(x,2) if i else x);f.append(x)
        for b,skip in zip(self.decode,f[-2::-1]):x=b(torch.cat((torch.nn.functional.interpolate(x,size=skip.shape[-1],mode='linear',align_corners=False),skip),1))
        return self.out(x)


def test_zero_modulation_preserves_core_state_output_and_rng():
    torch.manual_seed(71);core=Core();old=copy.deepcopy(core);state=torch.get_rng_state().clone()
    prior=ContextPrior(core,123);assert torch.equal(state,torch.get_rng_state())
    for k,v in old.state_dict().items():assert torch.equal(v,prior.state_dict()[k])
    x=torch.randn(2,9,256);torch.testing.assert_close(prior(x),old(x),atol=0,rtol=0)
    # Changing another record cannot change the first record's modulation/output.
    y=x.clone();y[1]*=10;torch.testing.assert_close(prior(y)[0],prior(x)[0],atol=1e-7,rtol=1e-6)


def test_context_modulates_all_levels_and_has_input_and_parameter_gradients():
    torch.manual_seed(72);p=ContextPrior(Core(),124)
    with torch.no_grad():nn.init.normal_(p.conditioner[-1].weight,std=.1);p.conditioner[-1].bias.fill_(.2)
    x=torch.randn(2,9,256,requires_grad=True);p(x).square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all() and x.grad.abs().sum()>0
    for name,param in p.named_parameters():assert param.grad is not None and torch.isfinite(param.grad).all() and param.grad.abs().sum()>0,name
    params=p.conditioner(descriptor(x.detach()))
    assert (BOUND*params.tanh()).abs().max()<=BOUND


def test_descriptor_has_no_batch_statistics_and_rejects_wrong_input():
    x=torch.randn(2,9,256);d=descriptor(x);assert d.shape==(2,27)
    torch.testing.assert_close(d[0],descriptor(x[:1])[0],atol=0,rtol=0)
    with pytest.raises(ValueError):descriptor(torch.zeros(2,8,256))
