import torch
from bounded_oracle import oracle_refine

def test_refinement_respects_each_waveform_radius_and_reduces_loss():
    torch.manual_seed(9);x=torch.randn(5,2,16);labels=torch.ones(5,dtype=torch.long);w=torch.randn(32)
    def classifier(v):
        score=v.flatten(1)@w;return torch.stack((-score,score),1)
    result=oracle_refine(classifier,x,labels,.1,steps=8);r=(result-x).flatten(1).norm(dim=1);bound=.1*x.flatten(1).norm(dim=1)
    assert torch.all(r<=bound+1e-5);assert torch.all(torch.nn.functional.cross_entropy(classifier(result),labels,reduction='none')<=torch.nn.functional.cross_entropy(classifier(x),labels,reduction='none')+1e-6)
    torch.testing.assert_close(oracle_refine(classifier,x,labels,0),x)
