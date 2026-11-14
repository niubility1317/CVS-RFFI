from experiments.pamf_phase2_practical import group2_savedb_candidate_diagnostic as candidate
import pytest
import torch
from torch import nn


def test_candidate_forwards_paper_semantics_and_diagnostic_clip(monkeypatch):
    seen=[]
    monkeypatch.setattr(candidate.cil,'register',lambda anchor,x,y,**kw: seen.append(kw) or 'state')
    prepare={'mopc_hr':'anchor'}
    assert candidate.register_candidate(prepare,'x','y',seed=17,methods=('mopc_hr',),clip=None)=={'mopc_hr':'state'}
    assert candidate.register_candidate(prepare,'x','y',seed=17,methods=('mopc_hr',),clip=1.)=={'mopc_hr':'state'}
    assert seen==[dict(seed=17,mopc_semantics='paper',mopc_max_grad_norm=None),
                  dict(seed=17,mopc_semantics='paper',mopc_max_grad_norm=1.)]


def test_historical_guard_detects_fp32_norm_overflow():
    model=nn.Linear(1,1,bias=False)
    model.weight.grad=torch.tensor([[1e20]])
    with pytest.raises(FloatingPointError,match='gradient_norm'):
        candidate.historical_record([],model,{'loss':torch.tensor(1.)},None,0.)


def test_historical_guard_records_normal_gradient():
    model=nn.Linear(1,1,bias=False);model.weight.grad=torch.tensor([[3.]])
    trace=[];candidate.historical_record(trace,model,{'loss':torch.tensor(1.)},None,0.)
    assert trace[0]['gradient_norm']==pytest.approx(3.)
