import copy
import pytest
import torch
from torch import nn
from torch.nn import functional as F
from experiments.pamf_dura_efficiency.signature_context_prior import SignatureContextPrior,signature_errors,BOUND


class Block(nn.Module):
    def __init__(self,a,b):super().__init__();self.net=nn.Sequential(nn.Conv1d(a,b,1),nn.SiLU())
    def forward(self,x):return self.net(x)


class Encoder(nn.Module):
    def __init__(self):
        super().__init__();widths=(12,16,28,40);self.blocks=nn.ModuleList(nn.Sequential(Block(a,b)) for a,b in zip((9,*widths[:-1]),widths))
    def forward(self,x):
        result=[]
        for i,b in enumerate(self.blocks):x=b(F.avg_pool1d(x,2) if i else x);result.append(x)
        return result


class Core(nn.Module):
    def __init__(self):
        super().__init__();self.conditional=False;self.encoder=Encoder();widths=(12,16,28,40)
        self.decode=nn.ModuleList(Block(a+b,b) for a,b in zip(widths[:0:-1],widths[-2::-1]));self.out=nn.Conv1d(12,2,1)
    def forward(self,x):
        features=self.encoder(x);x=features[-1]
        for b,skip in zip(self.decode,features[-2::-1]):x=b(torch.cat((F.interpolate(x,size=skip.shape[-1],mode='linear',align_corners=False),skip),1))
        return self.out(x)


def test_zero_conditioner_preserves_base_output_parameters_and_rng():
    torch.set_num_threads(2);torch.manual_seed(91);core=Core();original=copy.deepcopy(core);rng=torch.get_rng_state().clone()
    prior=SignatureContextPrior(core,123);assert torch.equal(rng,torch.get_rng_state())
    assert all(torch.equal(v,prior.state_dict()[k]) for k,v in original.state_dict().items())
    x=torch.randn(3,9,256);actual,sig=prior.forward_with_signature(x)
    assert sig.shape==(3,160);torch.testing.assert_close(actual,original(x),rtol=0,atol=0);torch.testing.assert_close(prior(x),actual,rtol=0,atol=0)
    assert sum(p.numel() for m in [prior.signature_projection,prior.signature_conditioner] for p in m.parameters())==38672


def test_learned_signature_changes_actual_IQ_and_receives_its_gradients():
    torch.manual_seed(92);prior=SignatureContextPrior(Core(),124)
    with torch.no_grad():nn.init.normal_(prior.signature_conditioner.weight,std=.1)
    x=torch.randn(2,9,256,requires_grad=True);first=prior(x)
    with torch.no_grad():prior.signature_projection.bias.fill_(.3)
    second=prior(x);assert not torch.allclose(first,second)
    second.square().mean().backward()
    for p in [prior.signature_projection.weight,prior.signature_conditioner.weight,prior.encoder.blocks[1][0].net[0].weight]:
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0
    _,sig=prior.forward_with_signature(x.detach());assert (BOUND*prior.signature_conditioner(sig).tanh()).abs().max()<=BOUND


def test_aux_detaches_teacher_and_inference_has_no_between_record_statistics():
    torch.manual_seed(93);prior=SignatureContextPrior(Core(),125);x=torch.randn(3,9,256)
    with torch.no_grad():nn.init.normal_(prior.signature_conditioner.weight,std=.1)
    output,sig=prior.forward_with_signature(x);target=torch.randn(3,160,requires_grad=True)
    errors=signature_errors((sig,sig*1.1),target);errors.mean().backward();assert target.grad is None and prior.signature_projection.weight.grad.abs().sum()>0
    changed=x.clone();changed[1]*=10
    torch.testing.assert_close(prior(changed)[0],output[0],rtol=1e-6,atol=1e-7)
    torch.testing.assert_close(prior(x[[2,0,1]]),output[[2,0,1]],rtol=1e-6,atol=1e-7)
    with pytest.raises(ValueError):signature_errors((sig,),target)
    with pytest.raises(ValueError):prior(torch.zeros(3,8,256))


def test_matched_aux_only_control_keeps_same_projection_and_skips_it_in_inference():
    torch.manual_seed(94);core=Core();context=SignatureContextPrior(copy.deepcopy(core),126);control=SignatureContextPrior(copy.deepcopy(core),126,decoder_context=False)
    assert all(torch.equal(v,control.signature_projection.state_dict()[k]) for k,v in context.signature_projection.state_dict().items())
    x=torch.randn(2,9,256);calls=[];hook=control.signature_projection.register_forward_hook(lambda *args:calls.append(1))
    try:
        actual=control(x);assert not calls
        training,prediction=control.forward_with_signature(x);assert len(calls)==1 and prediction.shape==(2,160)
    finally:hook.remove()
    torch.testing.assert_close(training,actual,rtol=0,atol=0);torch.testing.assert_close(actual,core(x),rtol=0,atol=0)
    assert control.signature_conditioner is None
