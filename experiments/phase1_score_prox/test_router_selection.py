import torch
import pytest
from router_selection import select_confidence,oracle_reachable

def test_margin_and_ties_protect_parent():
    scores=torch.tensor([[.8,.8,.7],[.5,.7,.6],[.8,.85,.9]])
    assert select_confidence(scores,0).tolist()==[0,1,2]
    assert select_confidence(scores,.15).tolist()==[0,1,0]

def test_oracle_is_evaluation_only_union():
    pred=torch.tensor([[1,0,2],[0,1,2],[2,1,0]]);target=torch.tensor([1,1,3])
    assert oracle_reachable(pred,target).tolist()==[True,True,False]
    with pytest.raises(ValueError):select_confidence(torch.zeros(2,1),0)
