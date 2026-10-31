import copy
import pytest
import torch
from experiments.pamf_dura_efficiency.signature_context_prior import SignatureContextPrior,signature_errors
from experiments.pamf_dura_efficiency.signature_deep_prior import DeepSignaturePrior
from test_dura_signature_context_prior import Core


@pytest.mark.parametrize('mode,added',[('linear',30880),('mlp',56640)])
def test_deep_zero_init_preserves_original_state_SIG_IQ_rng_and_extra_count(mode,added):
    torch.set_num_threads(2);torch.manual_seed(21);original=SignatureContextPrior(Core(),9);before=copy.deepcopy(original);rng=torch.get_rng_state().clone()
    deep=DeepSignaturePrior(original,23,mode);assert torch.equal(rng,torch.get_rng_state())
    assert all(torch.equal(v,deep.state_dict()[k]) for k,v in before.state_dict().items())
    x=torch.randn(2,9,256);a,s=deep.forward_with_signature(x);b,t=before.forward_with_signature(x)
    torch.testing.assert_close(a,b,rtol=0,atol=0);torch.testing.assert_close(s,t,rtol=0,atol=0)
    assert sum(p.numel() for p in deep.deep_signature_projection.parameters())==added


@pytest.mark.parametrize('mode',['linear','mlp'])
def test_deep_path_uses_later_features_and_reaches_actual_decoder(mode):
    torch.set_num_threads(2);torch.manual_seed(22);deep=DeepSignaturePrior(SignatureContextPrior(Core(),9),23,mode)
    last=deep.deep_signature_projection if mode=='linear' else deep.deep_signature_projection[-1]
    with torch.no_grad():last.weight.fill_(.01);deep.signature_conditioner.weight.fill_(.01)
    captured=[];handle=deep.deep_signature_projection.register_forward_pre_hook(lambda module,args:captured.append(args[0].detach().clone()))
    x=torch.randn(2,9,256);out,sig=deep.forward_with_signature(x);handle.remove();assert captured[0].shape==(2,192)
    out.square().mean().backward()
    assert last.weight.grad is not None and last.weight.grad.abs().sum()>0
    assert deep.encoder.blocks[3][0].net[0].weight.grad.abs().sum()>0
    original=sig.detach().clone()
    with torch.no_grad():last.weight.zero_();last.bias.zero_();_,zero=deep.forward_with_signature(x)
    assert (original-zero).abs().max()>0


@pytest.mark.parametrize('mode',['linear','mlp'])
def test_auxiliary_updates_new_zero_output_head_and_detaches_source_reference(mode):
    torch.set_num_threads(2);torch.manual_seed(23);deep=DeepSignaturePrior(SignatureContextPrior(Core(),9),23,mode)
    _,sig=deep.forward_with_signature(torch.randn(2,9,256));target=torch.randn(2,160,requires_grad=True)
    signature_errors([sig,sig],target).mean().backward();last=deep.deep_signature_projection if mode=='linear' else deep.deep_signature_projection[-1]
    assert last.weight.grad is not None and torch.isfinite(last.weight.grad).all() and last.weight.grad.abs().sum()>0 and target.grad is None
    if mode=='mlp':assert torch.equal(deep.deep_signature_projection[0].weight.grad,torch.zeros_like(deep.deep_signature_projection[0].weight))
