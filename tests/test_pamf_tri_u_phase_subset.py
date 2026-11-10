import pytest
import torch
from experiments.pamf_tri_u_phase_subset import plan,scope,source


def test_two_arms_have_all_three_seeds_and_complementary_branches():
    rows=plan.candidate_rows()
    assert len(rows)==6 and set(r['model_seed'] for r in rows)=={2026092701,2026092702,2026092703}
    a,b=plan.ARMS.values()
    assert set(a).isdisjoint(b) and set(a)|set(b)=={'time','frequency','pa','path','reference'}
    with pytest.raises(ValueError):plan.configuration({}, {}, '/project','commit')
    with pytest.raises(ValueError):source.parent(dict(model_seed=2026092701,run_id='other',row_id='bad',output_root='/bad'))


def test_only_selected_branches_receive_consistency_gradient():
    torch.manual_seed(3);names=('time','frequency','pa','path','reference')
    teacher={k:torch.randn(4,3) for k in names}
    student={k:torch.randn(4,3,requires_grad=True) for k in names}
    accepted=torch.tensor([True,False,True,False])
    loss=scope.subset_consistency(teacher,student,accepted,('frequency','pa'));loss.backward()
    assert all(student[k].grad is None for k in ('time','path','reference'))
    assert all(student[k].grad is not None and student[k].grad[accepted].abs().sum()>0 for k in ('frequency','pa'))
    assert all(torch.equal(student[k].grad[~accepted],torch.zeros_like(student[k].grad[~accepted])) for k in ('frequency','pa'))
    with pytest.raises(ValueError):scope.subset_consistency(teacher,student,accepted,('unknown',))
