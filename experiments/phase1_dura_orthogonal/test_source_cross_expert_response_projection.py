import torch
from experiments.phase1_dura_orthogonal.source_cross_expert_response_projection import project_response


def test_opposing_component_is_removed():
    wave=(torch.tensor([1.,0.]),torch.tensor([0.,2.]))
    response=(torch.tensor([-3.,4.]),torch.tensor([0.,-1.]))
    projected=project_response(wave,response)
    dot=sum((a*b).sum() for a,b in zip(wave,projected))
    torch.testing.assert_close(dot,torch.tensor(0.),atol=1e-6,rtol=0.)
    torch.testing.assert_close(projected[0][1],response[0][1])


def test_agreeing_and_zero_primary_are_unchanged():
    for primary,aux in [(torch.tensor([1.,0.]),torch.tensor([2.,3.])),(torch.zeros(2),torch.tensor([-1.,2.]))]:
        output=project_response((primary,),(aux,))
        assert torch.equal(output[0],aux)
        assert torch.isfinite(output[0]).all()
