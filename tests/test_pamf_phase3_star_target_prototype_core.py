"""Balanced detached targets, opaque nearest neighbors, and strong-view gradients."""
import importlib
import numpy as np
import pytest
import torch
from torch.nn import functional as F


def api():return importlib.import_module('experiments.pamf_phase3_open_world.star_target_prototype_core')


def test_balanced_assignment_is_detached_rows_one_and_uniform_prototype_mass():
    m=api();p=torch.eye(160)[:16].clone().requires_grad_();weak=p.detach().repeat(2,1).requires_grad_();q=m.balanced_assignments(weak,p)
    assert q.shape==(32,16) and not q.requires_grad
    torch.testing.assert_close(q.sum(1),torch.ones(32),rtol=0,atol=2e-7)
    torch.testing.assert_close(q.sum(0),torch.full((16,),2.),rtol=0,atol=2e-6)
    assert float(q.max(1).values.min())>.99
    torch.testing.assert_close(p,torch.eye(160)[:16],rtol=0,atol=0)


def test_parent_neighbors_exclude_self_stable_opaque_ties_and_no_labels():
    m=api();z=torch.eye(160)[torch.tensor([0,0,1])];ids=np.asarray(['b','a','c']);neighbors=m.nearest_neighbors(z,ids)
    assert neighbors.tolist()==[1,0,1]
    order=np.asarray([2,0,1]);again=m.nearest_neighbors(z[order.tolist()],ids[order]);np.testing.assert_array_equal(ids[neighbors.tolist()],ids[order][again.tolist()][np.argsort(order)])
    with pytest.raises(ValueError):m.nearest_neighbors(z,np.asarray(['same','same','other']))


def test_strong_views_and_prototypes_get_gradients_weak_targets_stop_gradient():
    m=api();torch.manual_seed(124);weak=F.normalize(torch.randn(32,160),dim=1).requires_grad_();p=torch.nn.Parameter(F.normalize(torch.randn(16,160),dim=1))
    first=F.normalize(torch.randn(8,160),dim=1).requires_grad_();second=F.normalize(torch.randn(8,160),dim=1).requires_grad_()
    indices=torch.arange(8);neighbors=m.nearest_neighbors(weak.detach());result=m.consistency(weak,first,second,p,indices,neighbors)
    q=m.balanced_assignments(weak,p);logp=[(x@F.normalize(p,dim=1).T/.1).log_softmax(1) for x in (first,second)]
    expected_global=-.5*sum((q[indices]*v).sum(1).mean() for v in logp)
    expected_local=-.5*sum((q[neighbors[indices]]*v).sum(1).mean() for v in logp)
    torch.testing.assert_close(result['global'],expected_global);torch.testing.assert_close(result['local'],expected_local)
    result['loss'].backward();assert weak.grad is None and float(first.grad.norm())>0 and float(second.grad.norm())>0 and float(p.grad.norm())>0


def test_spherical_initialization_fixed_capacity_seed_budget_and_bad_boundaries(subtests):
    m=api();torch.manual_seed(51);z=F.normalize(torch.randn(32,160),dim=1);centers,diag=m.initialize_prototypes(z)
    again,other=m.initialize_prototypes(z);torch.testing.assert_close(centers,again,rtol=0,atol=0)
    assert centers.shape==(16,160) and diag['seed']==2026103113 and diag['iterations']==25 and diag['capacity']==16
    torch.testing.assert_close(centers.norm(dim=1),torch.ones(16),rtol=0,atol=2e-7)
    for defect in ('zero','shape','NaN','self'):
        with subtests.test(defect=defect):
            bad=z.clone();p=centers.clone()
            if defect=='zero':bad[0]=0
            elif defect=='shape':p=p[:15]
            elif defect=='NaN':bad[0,0]=float('nan')
            with pytest.raises(ValueError):
                if defect=='self':m.consistency(z,z[:8],z[:8],centers,torch.arange(8),torch.arange(32))
                else:m.balanced_assignments(bad,p)
