import pytest
import torch

from experiments.pamf_phase2_practical.vendor_native.paper_reproduction.mopc_hr_non_exemplar_cil_sei.algorithm import hierarchical_regularization


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA optimization')
def test_cuda_foreach_matches_original_scalar_and_gradient_exactly(monkeypatch):
    gen = torch.Generator(device='cuda').manual_seed(45)
    current = {str(i): torch.randn((i+2, 7), generator=gen, device='cuda', requires_grad=True) for i in range(10)}
    previous = {name: torch.randn(value.shape, generator=gen, device='cuda') for name,value in current.items()}
    reference = next(iter(current.values())).new_zeros(())
    for i, (name, value) in enumerate(current.items()):
        reference = reference + 0.7*(1.0-i/len(current))*torch.sum((value-previous[name].detach())**2)
    reference_grads = torch.autograd.grad(reference, tuple(current.values()))
    calls=[]; original = torch._foreach_sub
    def track(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(torch, '_foreach_sub', track)
    actual = hierarchical_regularization(current, previous, lambda_max=.7)
    actual_grads = torch.autograd.grad(actual, tuple(current.values()))
    assert calls and torch.equal(actual, reference)
    assert all(torch.equal(a,b) for a,b in zip(actual_grads, reference_grads))


def test_cpu_fallback_and_invalid_shapes_remain_unchanged():
    cur={'a':torch.tensor([1.,3.],requires_grad=True),'b':torch.tensor([4.],requires_grad=True)}
    previous={'a':torch.tensor([0.,0.]),'b':torch.tensor([2.])}
    value=hierarchical_regularization(cur,previous,lambda_max=.7)
    assert float(value.detach()) == pytest.approx(.7*(10+2))
    with pytest.raises(ValueError, match='shapes'):
        hierarchical_regularization(cur,{'a':torch.tensor([0.]),'b':previous['b']})


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA source smoke')
def test_source_smoke_compares_penalty_and_gradients_without_model_mutation():
    from experiments.pamf_phase2_practical.group2_c_backbone_fast_math import verify_regularization
    model=torch.nn.Sequential(torch.nn.Linear(8,5),torch.nn.Linear(5,3)).cuda().eval()
    original={k:v.detach().clone() for k,v in model.state_dict().items()}
    result=verify_regularization(model)
    assert result['status']=='PASS' and result['loss_equal'] and result['gradients_equal']
    assert result['source_only'] and not result['query_read']
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in original.items())
