import torch
from experiments.phase1_dura_orthogonal.cross_expert_refiner import CrossExpertRefiner
from experiments.phase1_dura_orthogonal.polyphase_cross_expert import PolyphaseCrossExpertRefiner,PolyphaseConv1d


def test_same_initial_state_rng_and_expected_replacements():
    torch.manual_seed(43);baseline=CrossExpertRefiner(width=8);state=torch.get_rng_state()
    torch.manual_seed(43);candidate=PolyphaseCrossExpertRefiner(width=8);after=torch.get_rng_state()
    assert torch.equal(state,after)
    assert baseline.state_dict().keys()==candidate.state_dict().keys()
    assert all(torch.equal(v,candidate.state_dict()[k]) for k,v in baseline.state_dict().items())
    assert sum(isinstance(m,PolyphaseConv1d) for m in candidate.modules())==9


def test_nonzero_full_model_forward_and_gradients_are_close():
    torch.manual_seed(44);baseline=CrossExpertRefiner(width=8);candidate=PolyphaseCrossExpertRefiner(width=8)
    with torch.no_grad():
        for head in (baseline.time_out,baseline.freq_out):
            torch.nn.init.normal_(head.weight,std=.001);torch.nn.init.normal_(head.bias,std=.001)
    candidate.load_state_dict(baseline.state_dict(),strict=True)
    x=torch.randn(2,2,256);a=torch.randn_like(x);d=torch.randn_like(x);p=torch.randn_like(x);v=torch.rand(2)
    y=baseline(x,a,d,p,v);z=candidate(x,a,d,p,v)
    assert not torch.equal(y,a+.5*(p-a))
    torch.testing.assert_close(y,z,rtol=1e-4,atol=1e-5)
    y.square().mean().backward();z.square().mean().backward()
    for (ka,pa),(kb,pb) in zip(baseline.named_parameters(),candidate.named_parameters()):
        assert ka==kb and pa.grad is not None and pb.grad is not None
        torch.testing.assert_close(pa.grad,pb.grad,rtol=2e-4,atol=1e-5)
