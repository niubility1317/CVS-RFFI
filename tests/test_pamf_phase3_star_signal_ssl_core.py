import math
import torch
import pytest
from experiments.pamf_phase3_open_world import star_signal_ssl_core as core


def test_positive_pairs_and_self_exclusion_match_exact_symmetric_cross_entropy():
    z=torch.eye(160)[:4];r=core.instance_contrast(z,z.clone())
    expected=math.log(1+6*math.exp(-10))
    assert float(r['loss'])==pytest.approx(expected,abs=1e-6)
    assert r['negative_count']==6 and r['anchors']==8 and r['positive_cosine']==pytest.approx(1.)
    assert r['loss']<core.instance_contrast(z,z.roll(1,0))['loss']


def test_both_view_gradients_reach_shared_signal_parameters():
    torch.manual_seed(31);encoder=torch.nn.Linear(8,160);x=torch.randn(6,8);y=x+.1*torch.randn_like(x)
    a=encoder(x);b=encoder(y);a.retain_grad();b.retain_grad();result=core.instance_contrast(a,b);result['loss'].backward()
    assert a.grad.norm()>0 and b.grad.norm()>0 and encoder.weight.grad.norm()>0
    assert torch.isfinite(encoder.weight.grad).all()


def test_two_existing_phase_gain_views_preserve_packet_length_and_gain_range_without_RMS():
    x=torch.randn(5,2,256);before=x.clone();g=torch.Generator().manual_seed(100)
    a,b=core.two_views(x,g);torch.testing.assert_close(x,before,rtol=0,atol=0)
    assert a.shape==b.shape==x.shape and not torch.equal(a,b)
    for v in (a,b):
        ratio=v.square().sum((1,2)).sqrt()/x.square().sum((1,2)).sqrt()
        assert bool(((ratio>=.95-1e-6)&(ratio<=1.05+1e-6)).all())
        assert not torch.allclose(v.square().mean((1,2)),torch.ones(len(x)))
    g2=torch.Generator().manual_seed(100);a2,b2=core.two_views(x,g2)
    torch.testing.assert_close(a,a2,rtol=0,atol=0);torch.testing.assert_close(b,b2,rtol=0,atol=0)


@pytest.mark.parametrize('a,b',[(torch.zeros(3,160),torch.ones(3,160)),(torch.ones(1,160),torch.ones(1,160)),
    (torch.ones(3,159),torch.ones(3,159)),(torch.ones(3,160,dtype=torch.float64),torch.ones(3,160)),
    (torch.full((3,160),float('nan')),torch.ones(3,160))])
def test_invalid_views_are_rejected_without_labels_or_K(a,b):
    with pytest.raises(ValueError):core.instance_contrast(a,b)
