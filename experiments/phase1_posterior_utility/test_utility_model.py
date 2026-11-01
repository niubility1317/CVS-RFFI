import torch
from utility_model import build_features,UtilityNet,select_candidate,FEATURES

def test_feature_shape_and_training_gradient():
    torch.manual_seed(2);n=2;z=torch.randn(n,512);states=torch.randn(n,6,512);A=torch.eye(512).repeat(n,1,1);features=build_features(torch.randn(n,160),torch.randn(n,6,160),torch.randn(n,6),torch.randn(n,6,6),torch.log_softmax(torch.randn(n,6),1),torch.randn(n),torch.randn(n,512),A,z,states);assert features.shape==(n,6,FEATURES)
    model=UtilityNet();model(features).square().mean().backward();assert model.net[0].weight.grad.abs().sum()>0

def test_selector_abstains_if_no_disagreement_or_insufficient_confidence():
    scores=torch.tensor([[5.,4.,3.,2.,1.,0.],[-2.,-2.,-2.,-2.,-2.,-2.]]);parent=torch.tensor([0,0]);candidate=torch.tensor([[0,0,0,0,0,0],[1,1,1,1,1,1]])
    assert select_candidate(scores,parent,candidate,.75).tolist()==[-1,-1];candidate[0,1]=1;assert select_candidate(scores,parent,candidate,.75).tolist()==[1,-1]
