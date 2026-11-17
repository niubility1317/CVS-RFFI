import torch
from experiments.pamf_repl_uce_guard import plan
from experiments.pamf_repl_uce_guard.projection import guarded_loss
from experiments.pamf_repl_uaux_soft.projection import auxiliary_loss

def test_ce_guard_only_changes_conflicting_ce_and_preserves_supervision():
    p=torch.nn.Parameter(torch.zeros(2))
    labeled=p[0];ref=p[1];other=-p[1];ce=-2*p[0]+3*p[1]
    loss,details,_,stats=guarded_loss(labeled,ref+other+ce,ref,other,[('other.p',p)],alpha=0,beta=0,gamma=.5)
    loss.backward()
    assert torch.allclose(p.grad,torch.tensor([0.,3.]))
    assert stats['other_base_ce_l_dot_before']==-2 and stats['other_base_ce_l_dot_after']==-1
    assert details['other']['dot_after']==-1

def test_zero_ce_guard_matches_existing_auxiliary_rule():
    p=torch.nn.Parameter(torch.zeros(2))
    labeled=2*p[0];ref=p[1];other=-p[0]-p[1];u=ref+other-3*p[0]+p[1]
    a,_,ga,_=auxiliary_loss(labeled,u,ref,other,[('other.p',p)],alpha=.5,beta=.5)
    b,_,gb,_=guarded_loss(labeled,u,ref,other,[('other.p',p)],alpha=.5,beta=.5,gamma=0)
    assert torch.equal(ga[0],gb[0]) and torch.equal(a.detach(),b.detach())

def test_matrix_is_bounded_and_unique():
    rows=plan.candidate_rows()
    assert len(rows)==len({r['row_id'] for r in rows})==16
    assert sum(r['setting']=='ten_class' for r in rows)==8
    assert all(0<=r['gamma']<=1 and r['mode']=='soft' for r in plan.RECIPES.values())
