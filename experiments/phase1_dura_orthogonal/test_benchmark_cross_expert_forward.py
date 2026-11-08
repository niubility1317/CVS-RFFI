import torch
from experiments.phase1_dura_orthogonal import benchmark_cross_expert_forward as probe
from experiments.phase1_dura_orthogonal.cross_expert_refiner import CrossExpertRefiner


def test_timed_forward_matches_original(monkeypatch):
    monkeypatch.setattr(probe,'timed',lambda fn:(fn(),1.0))
    torch.manual_seed(42)
    model=CrossExpertRefiner(width=8).eval()
    with torch.no_grad():
        for head in (model.time_out,model.freq_out):
            torch.nn.init.normal_(head.weight,std=.001)
            torch.nn.init.normal_(head.bias,std=.001)
    x=torch.randn(3,2,256);anchor=torch.randn_like(x);direction=torch.randn_like(x)
    proposal=torch.randn_like(x);variance=torch.rand(3)
    with torch.no_grad():
        reference=model(x,anchor,direction,proposal,variance)
        actual,phases=probe.forward_timed(model,x,anchor,direction,proposal,variance)
    assert probe.same_bits(actual,reference)
    assert not probe.same_bits(reference,anchor+.5*(proposal-anchor))
    assert set(phases)=={'input_seconds','fft_seconds','time_branch_seconds','frequency_branch_seconds','ifft_finish_seconds'}


def test_bitwise_check_distinguishes_signed_zero():
    assert not probe.same_bits(torch.tensor([0.]),torch.tensor([-0.]))
