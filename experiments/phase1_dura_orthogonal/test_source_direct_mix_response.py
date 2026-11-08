import torch
from experiments.phase1_dura_orthogonal.source_direct_mix_response import outputs
from experiments.phase1_dura_orthogonal.source_dura_perturbation_probe import response_metrics


def test_outputs_include_both_experts_half_and_frozen_gates():
    x=torch.ones(2,2,256)
    context=lambda v:(v*.2,v*.1,torch.ones(len(v)))
    direct=lambda v:v*.8
    gates=(lambda x,a,d,p,v:a,lambda x,a,d,p,v:p)
    a,p,half,g0,g1=outputs(x,context,direct,gates)
    torch.testing.assert_close(half,(a+p)*.5)
    torch.testing.assert_close(g0,a)
    torch.testing.assert_close(g1,p)


def test_response_metrics_identity_and_suppression():
    delta=torch.complex(torch.ones(2,256),torch.zeros(2,256))
    identity=response_metrics(delta,delta)
    suppressed=response_metrics(delta,delta*.25)
    torch.testing.assert_close(identity[:,0],torch.ones(2))
    torch.testing.assert_close(identity[:,1],torch.zeros(2))
    torch.testing.assert_close(suppressed[:,0],torch.full((2,),.25))
    torch.testing.assert_close(suppressed[:,1],torch.full((2,),.5625))
