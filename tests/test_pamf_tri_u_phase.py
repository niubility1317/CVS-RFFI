import pytest
import torch
from experiments.pamf_tri_u_phase import objective,plan,source


def test_complex_phase_rotation_preserves_magnitude_and_is_invertible():
    x=torch.tensor([[[1.,2.,-3.],[4.,-5.,6.]],[[2.,0.,1.],[-1.,3.,2.]]])
    angle=torch.tensor([.2,-.3])
    y=objective.rotate_iq(x,angle)
    assert torch.allclose(y.square().sum(1),x.square().sum(1),atol=1e-5)
    assert torch.allclose(objective.rotate_iq(y,-angle),x,atol=1e-5)


def test_branch_consistency_masks_unaccepted_and_updates_student_only():
    torch.manual_seed(5)
    teacher={k:torch.randn(4,3,requires_grad=True) for k in ('time','frequency','pa','path','reference')}
    student={k:torch.randn(4,3,requires_grad=True) for k in teacher}
    mask=torch.tensor([True,False,True,False])
    loss=objective.branch_consistency(teacher,student,mask);loss.backward()
    assert all(v.grad is None for v in teacher.values())
    assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in student.values())
    assert all(torch.equal(v.grad[~mask],torch.zeros_like(v.grad[~mask])) for v in student.values())


def test_plan_is_three_seeds_and_rejects_unregistered_parent():
    assert [r['model_seed'] for r in plan.candidate_rows()]==[2026092701,2026092702,2026092703]
    with pytest.raises(ValueError):plan.configuration({}, {}, '/project', 'commit')
    with pytest.raises(ValueError):source.parent(dict(model_seed=2026092701,run_id='other',row_id='bad',output_root='/bad'))
