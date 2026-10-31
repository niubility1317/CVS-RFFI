import pytest
import torch
from torch import nn
from experiments.pamf_identity_flow.model import IdentityConditionalFlow
from experiments.pamf_phase1_flow import anchor_balanced as a


class Reader(nn.Module):
    def __init__(self):super().__init__();self.head=nn.Linear(6,6,bias=False)
    def features(self,x):return torch.cat((x.mean(2),x.square().mean(2),x[:,:,:64].mean(2)),1)
    def classify_features(self,x):return self.head(x)


class Anchor(nn.Module):
    def __init__(self):super().__init__();self.scale=nn.Parameter(torch.tensor(.8))
    def forward(self,x):return x*self.scale


def test_step_updates_field_only_and_balances_gradient():
    torch.manual_seed(7);field=IdentityConditionalFlow(width=16,layers=1,heads=4)
    f=Reader().eval().requires_grad_(False);dura=Anchor().eval().requires_grad_(False)
    x=torch.randn(3,2,256);target=torch.randn_like(x);before=field.output.weight.detach().clone()
    optimizer=torch.optim.AdamW(field.parameters(),lr=.0003)
    terms,stats=a.step(field,f,dura,x,target,optimizer,torch.randn_like(x))
    assert stats['ref_norm']>0 and 0<=stats['scale']<=1 and torch.isfinite(terms['iq']).all()
    assert not torch.equal(before,field.output.weight)
    assert all(v.grad is None for v in f.parameters()) and all(v.grad is None for v in dura.parameters())


def test_checkpoint_binds_dura_digest(tmp_path):
    field=a.core.make_model();path=tmp_path/'field.pt';a.save(path,field,'a'*40,0,'d'*64)
    a.load(path,'cpu','a'*40,'d'*64,epoch=0)
    with pytest.raises(ValueError):a.load(path,'cpu','a'*40,'e'*64,epoch=0)
