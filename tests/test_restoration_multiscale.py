import pytest
import torch
from experiments.pamf_phase3_restoration.multiscale import scratch_multiscale


def test_scratch_identity_rng_and_all_global_paths_receive_gradients():
    torch.set_num_threads(2);before=torch.get_rng_state();g=scratch_multiscale(4)
    assert torch.equal(before,torch.get_rng_state())
    x=torch.randn(2,2,256);torch.testing.assert_close(g(x),x,atol=0,rtol=0)
    assert g.initialization_report['ancestors']==[]
    with torch.no_grad():g.output.weight.normal_(std=.002)
    (g(x)-torch.randn_like(x)).square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0 for p in g.parameters())


def test_frozen_per_packet_independence_permutation_and_strict_state():
    torch.set_num_threads(2);g=scratch_multiscale(5).eval()
    with torch.no_grad():
        g.output.weight.normal_(std=.002);x=torch.randn(3,2,256);y=g(x)
        torch.testing.assert_close(g(x[[2,0,1]]),y[[2,0,1]],atol=2e-6,rtol=2e-5)
        torch.testing.assert_close(g(x[:1]),y[:1],atol=2e-6,rtol=2e-5)
        other=scratch_multiscale(6).eval();other.load_state_dict(g.state_dict(),strict=True)
        torch.testing.assert_close(other(x),y,atol=0,rtol=0)


def test_only_deployment_iq_input_and_input_gradient_available():
    torch.set_num_threads(2);g=scratch_multiscale(5)
    for x in [torch.randn(2,2,128),torch.randn(2,3,256),torch.randn(2,2,256,dtype=torch.float64)]:
        with pytest.raises(ValueError,match='per-packet'):g(x)
    x=torch.randn(1,2,256,requires_grad=True);g(x).square().sum().backward()
    assert x.grad is not None and x.grad.abs().sum()>0
