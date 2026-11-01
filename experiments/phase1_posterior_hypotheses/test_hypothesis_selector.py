import torch
import pytest
from hypothesis_selector import choose_hypotheses

def test_map_consistency_and_product_are_distinct_and_batched():
    weights=torch.log(torch.tensor([[.7,.2,.1],[.1,.2,.7]]));logits=torch.tensor([[[0.,2.,0.],[0.,3.,0.],[0.,0.,0.]],[[4.,0.,0.],[0.,0.,0.],[4.,0.,0.]]]);r=choose_hypotheses(weights,logits);assert r['map'].tolist()==[0,2];assert r['self'].tolist()==[1,0];assert all(x.shape==(2,) for x in r.values())
    assert r['product'][0].item()==1

def test_incompatible_inputs_rejected():
    with pytest.raises(ValueError):choose_hypotheses(torch.zeros(2,3),torch.zeros(2,3,4))
