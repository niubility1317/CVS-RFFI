import pytest
import torch
from experiments.pamf_dura_efficiency.joint_branches import paired_cases,CASES
from experiments.pamf_dura_efficiency.branch_features import BRANCHES


def test_joint_swap_preserves_identity_response_and_replaces_actual_weighted_pa_sig():
    torch.manual_seed(51);p={k:torch.randn(3,160) for k in BRANCHES};q={k:torch.randn(3,160) for k in BRANCHES}
    total=p['identity']+p['PA']+p['response']+p['signature'];clean=q['identity']+q['PA']+q['response']+q['signature']
    before={k:v.clone() for k,v in p.items()};cases=paired_cases(total,p,clean,q)
    assert tuple(cases)==CASES and cases['native'] is total and cases['all_clean'] is clean
    torch.testing.assert_close(cases['PA_signature'],p['identity']+q['PA']+p['response']+q['signature'])
    torch.testing.assert_close(cases['PA_signature']-cases['signature'],q['PA']-p['PA'],atol=1e-6,rtol=2e-6)
    assert all(torch.equal(v,before[k]) for k,v in p.items())
    same=paired_cases(clean,q,clean,q)
    for v in same.values():torch.testing.assert_close(v,clean)


def test_unmatched_or_incomplete_or_nonadditive_components_are_rejected():
    p={k:torch.zeros(2,160) for k in BRANCHES};x=torch.zeros(2,160)
    with pytest.raises(ValueError):paired_cases(x,p,x,{'PA':p['PA']})
    with pytest.raises(ValueError):paired_cases(x,p,x,dict(p,signature=torch.zeros(1,160)))
    with pytest.raises(AssertionError):paired_cases(x+1,p,x,p)
