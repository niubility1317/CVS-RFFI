"""Hand arithmetic and source-domain masks for physical representation losses."""
import math

import pytest
import torch
from torch.nn import functional as F

from experiments.pamf_phase3_open_world import star_physical_open_loss as losses


def source_rows():
    y = torch.tensor([0]*4 + [1]*4)
    rx = torch.tensor([1,3,1,3]*2)
    day = torch.tensor([1,2,2,1]*2)
    z = F.one_hot(y, num_classes=64).float()
    return z, y, rx, day


def test_physical_offdiagonal_unbiased_covariance_and_std_hand_arithmetic():
    z0 = torch.zeros(3,64); z0[:,0] = torch.tensor([-1.,0.,1.]); z0[:,1] = z0[:,0]
    z1 = (2*z0).clone(); z0.requires_grad_(); z1.requires_grad_()
    out = losses.physical(z0,z1)
    assert set(out) == {'total','invariance','std','covariance'}
    torch.testing.assert_close(out['invariance'], torch.tensor(1/48), atol=1e-7, rtol=0)
    torch.testing.assert_close(out['std'], torch.tensor(62*.99/64), atol=1e-7, rtol=0)
    torch.testing.assert_close(out['covariance'], torch.tensor(34/64), atol=1e-7, rtol=0)
    torch.testing.assert_close(out['total'], torch.tensor(25/48+25*(62*.99/64)+34/64), atol=3e-6, rtol=0)
    out['total'].backward()
    for z in (z0,z1): assert z.grad is not None and torch.isfinite(z.grad).all() and z.grad.abs().sum() > 0


def test_constant_preunit_features_are_finite_and_invariance_has_both_gradients():
    z0 = torch.zeros(4,64,requires_grad=True); z1 = torch.ones(4,64,requires_grad=True)
    out = losses.physical(z0,z1)
    torch.testing.assert_close(out['std'], torch.tensor(.99), atol=2e-7, rtol=0)
    assert out['covariance'].item() == 0 and out['invariance'].item() == 1
    torch.testing.assert_close(out['total'], torch.tensor(49.75), atol=3e-6, rtol=0)
    out['total'].backward()
    assert torch.isfinite(z0.grad).all() and torch.isfinite(z1.grad).all()
    assert z0.grad.abs().sum() > 0 and z1.grad.abs().sum() > 0
    same = losses.physical(torch.zeros(2,64),torch.zeros(2,64))
    torch.testing.assert_close(same['total'],torch.tensor(24.75),atol=3e-6,rtol=0)


def test_cross_rx_and_day_eligible_positive_only_denominator_hand_arithmetic():
    z,y,rx,day = source_rows(); z.requires_grad_()
    out = losses.cross_rx_supervised(z,y,rx,day)
    # Each row has one eligible same-class positive (score10), four negatives
    # (score0), and two same-class noneligible rows excluded entirely.
    expected = math.log1p(4*math.exp(-10))
    torch.testing.assert_close(out['total'],torch.tensor(expected),atol=5e-7,rtol=0)
    assert out['positive_pair_count'] == 8 and out['anchors'] == 8
    out['total'].backward()
    assert z.grad is not None and torch.isfinite(z.grad).all() and z.grad.abs().sum() > 0


def test_multiple_positives_average_log_probability_and_permutation():
    torch.manual_seed(12)
    z = F.normalize(torch.randn(6,64),dim=1).requires_grad_()
    y = torch.tensor([0,0,0,1,1,1]); rx=torch.tensor([1,3,4,1,3,4]); day=torch.tensor([1,2,3,1,2,3])
    out = losses.cross_rx_supervised(z,y,rx,day)
    s=z@z.T/.1; terms=[]
    for i in range(6):
        positive=[j for j in range(6) if y[j]==y[i] and rx[j]!=rx[i] and day[j]!=day[i]]
        allowed=[j for j in range(6) if j in positive or y[j]!=y[i]]
        terms.append(torch.logsumexp(s[i,allowed],0)-s[i,positive].mean())
    torch.testing.assert_close(out['total'],torch.stack(terms).mean(),atol=3e-7,rtol=0)
    assert out['positive_pair_count']==12 and out['anchors']==6
    order=torch.tensor([4,2,0,5,3,1])
    changed=losses.cross_rx_supervised(z[order],y[order],rx[order],day[order])
    torch.testing.assert_close(out['total'],changed['total'],atol=5e-7,rtol=0)


def test_invalid_preunit_inputs_rejected(subtests):
    good=torch.randn(3,64)
    bad=[(good[:,:63],good[:,:63]),(good[:1],good[:1]),(good.double(),good.double()),
         (good,good[:2]),(torch.full_like(good,float('nan')),good)]
    for i,(z0,z1) in enumerate(bad):
        with subtests.test(case=i),pytest.raises(ValueError): losses.physical(z0,z1)


def test_invalid_source_labels_units_devices_or_missing_cross_domain_positive(subtests):
    z,y,rx,day=source_rows()
    cases=[(z,torch.full_like(y,-1),rx,day,.1),(z,torch.full_like(y,6),rx,day,.1),
        (z,y.float(),rx,day,.1),(z,y,rx.float(),day,.1),(z,y,rx,day.float(),.1),
        (z,y[:-1],rx,day,.1),(z.double(),y,rx,day,.1),(z*2,y,rx,day,.1),
        (torch.full_like(z,float('inf')),y,rx,day,.1),(z,y,rx,day,.2),
        (z,y,torch.ones_like(rx),day,.1),(z,y,rx,torch.ones_like(day),.1),
        (z,torch.empty(y.shape,dtype=torch.int64,device='meta'),rx,day,.1)]
    for i,args in enumerate(cases):
        with subtests.test(case=i),pytest.raises(ValueError): losses.cross_rx_supervised(*args[:4],temperature=args[4])
