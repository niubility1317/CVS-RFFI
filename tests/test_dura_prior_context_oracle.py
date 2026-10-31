import copy
import pytest
import torch
from torch import nn
from experiments.pamf_dura_efficiency.prior_context_oracle import decoder_context,decision,CASES


class Prior(nn.Module):
    def __init__(self):
        super().__init__();self.decoder_context=True;self.signature_conditioner=nn.Linear(160,2)
    def forward(self,x):return self.signature_conditioner(x)


class Model(nn.Module):
    def __init__(self):super().__init__();self.prior=nn.ModuleList([Prior(),Prior()])
    def forward(self,x):return sum(p(x) for p in self.prior)


def test_fixed_weights_oracle_disabled_and_hooks_restore_on_exception():
    torch.manual_seed(18);model=Model();state=copy.deepcopy(model.state_dict());x=torch.randn(3,160);t=torch.randn(3,160);baseline=model(x)
    with decoder_context(model,'disabled',t):assert torch.equal(model(x),torch.zeros(3,2))
    expected=model(t)
    with decoder_context(model,'clean_SIG',t):torch.testing.assert_close(model(x),expected)
    with pytest.raises(RuntimeError):
        with decoder_context(model,'shifted_SIG',t):raise RuntimeError('intentional caller error')
    assert torch.equal(model(x),baseline) and all(torch.equal(v,state[k]) for k,v in model.state_dict().items())
    assert all(not p.signature_conditioner._forward_hooks and not p.signature_conditioner._forward_pre_hooks for p in model.prior)


def test_oracle_is_per_record_and_rejects_unmatched_boundary():
    torch.manual_seed(19);model=Model();x=torch.randn(3,160);t=torch.randn(3,160)
    with decoder_context(model,'clean_SIG',t):all_records=model(x)
    with decoder_context(model,'clean_SIG',t[:1]):one=model(x[:1])
    torch.testing.assert_close(all_records[:1],one)
    with pytest.raises(ValueError):
        with decoder_context(model,'clean_SIG',t[:1]):model(x)
    with pytest.raises(ValueError):
        with decoder_context(model,'unknown',t):pass
    assert all(not p.signature_conditioner._forward_pre_hooks for p in model.prior)


def test_decision_requires_both_views_shift_control_and_RX_protection():
    def cell(n,e=.2):return dict(teacher_correct=n,feature_cosine_error=e)
    group=dict(count=512,cases={k:cell(n) for k,n in zip(CASES,[490,489,494,492])})
    summary={v:{'all':copy.deepcopy(group),'rx:1':copy.deepcopy(group),'scene:clean':copy.deepcopy(group),'scene:urban':copy.deepcopy(group)} for v in ['1','20']}
    assert decision(summary)['status']=='PASS'
    summary['20']['rx:1']['cases']['clean_SIG']['teacher_correct']=489
    assert decision(summary)['status']=='FAIL'
    summary['20']['rx:1']['cases']['clean_SIG']['teacher_correct']=494
    summary['20']['all']['cases']['shifted_SIG']['teacher_correct']=493
    assert decision(summary)['status']=='FAIL' and decision({'1':summary['1']})['status']=='FAIL'
