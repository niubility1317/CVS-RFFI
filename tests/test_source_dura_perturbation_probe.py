import torch
from experiments.phase1_dura_orthogonal.source_dura_perturbation_probe import EPSILON,perturbations,response_metrics


def test_directions_have_prescribed_relative_rms():
    torch.manual_seed(10)
    clean=torch.randn(3,2,256)
    z=torch.complex(clean[:,0],clean[:,1])
    reference=z.abs().square().mean(1).sqrt()*EPSILON
    for delta in perturbations(clean):
        torch.testing.assert_close(delta.abs().square().mean(1).sqrt(),reference)


def test_identity_and_zero_response_metrics():
    torch.manual_seed(11)
    delta=torch.complex(torch.randn(3,256),torch.randn(3,256))
    identity=response_metrics(delta,delta)
    torch.testing.assert_close(identity,torch.tensor([[1.,0.,1.]]).expand(3,-1))
    zero=response_metrics(delta,torch.zeros_like(delta))
    torch.testing.assert_close(zero,torch.tensor([[0.,1.,0.]]).expand(3,-1))
