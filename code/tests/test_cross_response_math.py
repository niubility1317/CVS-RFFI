import sys
from pathlib import Path
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from cvsrffi.cross_response.statistics import FixedSourceNormalizer, WaveformStatistics, query_cell_targets
from cvsrffi.cross_response.readouts import ResponseReadouts
from cvsrffi.cross_response.predictor import SharedResponsePredictor
from cvsrffi.cross_response.tensor_ops import double_center, grid_decomposition, arithmetic_completion, response_loss, identity_interaction_loss
from cvsrffi.cross_response.decision import decision_margin_loss


def test_centering_completion_and_non_disentangled_additive_counterexample():
    torch.manual_seed(1)
    x = torch.randn(3,4,5,dtype=torch.double)
    e = double_center(x)
    assert torch.allclose(e.mean(0),torch.zeros(4,5,dtype=torch.double),atol=1e-14)
    assert torch.allclose(e.mean(1),torch.zeros(3,5,dtype=torch.double),atol=1e-14)
    assert torch.allclose(x-arithmetic_completion(x),2*e,atol=1e-14)
    d = grid_decomposition(x)
    assert torch.allclose(x,sum(d.values()))
    tx = torch.arange(3.).view(3,1,1).expand(3,4,1)
    rx = torch.arange(4.).view(1,4,1).expand(3,4,1)
    mixed = torch.cat((tx,rx),-1)
    assert double_center(mixed).abs().max()==0
    assert mixed[:,0].var(0).sum()>0 and mixed[0].var(0).sum()>0
    interaction = (tx+1)*(rx+2)
    assert torch.allclose(double_center(interaction),(tx-tx.mean())*(rx-rx.mean()))
    assert identity_interaction_loss(interaction)>0
    # All unordered TX/RX rectangle residuals have the exact scaling factor.
    rectangles = torch.stack([x[a,r]-x[a,s]-x[b,r]+x[b,s]
                              for a in range(3) for b in range(a+1,3)
                              for r in range(4) for s in range(r+1,4)])
    assert torch.allclose(rectangles.square().mean(),8*e.square().mean())


@pytest.mark.parametrize('mode',['additive','bilinear'])
def test_shared_predictor_algebra_and_gradients(mode):
    torch.manual_seed(2)
    model = SharedResponsePredictor(5,8,mode,3).double()
    t,d = torch.randn(3,5,dtype=torch.double,requires_grad=True),torch.randn(4,5,dtype=torch.double,requires_grad=True)
    y = model(t,d)
    assert y.shape==(3,4,8)
    assert torch.allclose(double_center(y),model.centered_interaction(t,d),atol=1e-14)
    loss,_ = response_loss(y,torch.randn_like(y))
    loss.backward()
    assert t.grad.norm()>0 and d.grad.norm()>0
    assert all(p.grad is not None for p in model.parameters())
    assert torch.allclose(model(t[[2,0,1]],d[[1,3,0,2]]),y[[2,0,1]][:,[1,3,0,2]])


def test_response_target_detach_and_na():
    p = torch.randn(1,2,3,requires_grad=True)
    t = torch.randn_like(p,requires_grad=True)
    loss,diag = response_loss(p,t)
    assert diag['interaction_error'] is None and not diag['interaction_available']
    loss.backward()
    assert t.grad is None
    with pytest.raises(ValueError): response_loss(p,t,1)
    with pytest.raises(ValueError): double_center(torch.full((2,2,1),float('nan')))


def test_interaction_reweight_default_and_explicit_control():
    pred,target = torch.randn(3,4,5,requires_grad=True),torch.randn(3,4,5)
    basic,diag = response_loss(pred,target)
    weighted,_ = response_loss(pred,target,interaction_weight=.7)
    assert torch.allclose(basic,(pred-target).square().mean())
    assert torch.allclose(weighted,basic+.7*diag['interaction_error'])
    assert torch.allclose(identity_interaction_loss(pred),double_center(pred).square().sum(-1).mean())


def test_readout_donor_exclusion_detach_and_reuse():
    torch.manual_seed(3)
    read = ResponseReadouts(3,4,2)
    zi,zd = torch.randn(4,4,2,3,requires_grad=True),torch.randn(4,4,2,4,requires_grad=True)
    args = ([2,3],[2,3],[0,1],[0,1])
    t,d = read(zi,zd,*args,detach_identity=True)
    a,b = zi.detach().clone(),zd.detach().clone()
    a[2:,2:] += 1000
    b[2:,2:] -= 1000
    t2,d2 = read(a,b,*args)
    assert torch.equal(t,t2) and torch.equal(d,d2)
    assert torch.allclose(t,read.tx(zi[2:,:2]).mean((1,2)))
    (t.sum()+d.sum()).backward()
    assert zi.grad is None and zd.grad.norm()>0 and zd.grad[2:].abs().sum()==0
    assert read.tx.weight.grad.norm()>0
    with pytest.raises(ValueError): read(zi,zd,[0],[0],[0],[1])


def test_fft_zero_tone_and_per_record_targets():
    stat = WaveformStatistics(bands=8)
    z = torch.zeros(2,2,32,requires_grad=True)
    out = stat(z)
    assert torch.isfinite(out).all() and not out.requires_grad
    assert torch.allclose(out,torch.full_like(out,torch.tensor(1e-8).log()))
    tone = torch.exp(2j*torch.pi*torch.arange(32)/32)
    energy = stat(tone).exp()-1e-8
    assert torch.allclose(energy.sum(),torch.tensor(1.),atol=1e-6)
    assert energy.argmax()==4
    with pytest.raises(ValueError): stat(torch.zeros(2,16))
    with pytest.raises(ValueError): WaveformStatistics()(torch.zeros(2,4))
    with pytest.raises(ValueError): WaveformStatistics()(torch.full((2,32),float('nan')))
    iq = torch.randn(2,3,2,2,32)
    assert torch.equal(query_cell_targets(stat,iq),stat(iq).mean(2))


def test_autocorrelation_covariance_and_event_metadata():
    z = torch.exp(2j*torch.pi*torch.arange(32)/32)
    result = WaveformStatistics('autocorr',lags=(1,2))(z)
    expected = torch.tensor([torch.cos(torch.tensor(2*torch.pi/32)),torch.sin(torch.tensor(2*torch.pi/32)),torch.cos(torch.tensor(4*torch.pi/32)),torch.sin(torch.tensor(4*torch.pi/32))])
    assert torch.allclose(result,expected,atol=1e-6)
    assert WaveformStatistics('autocorr')(torch.zeros(2,16)).abs().sum()==0
    with pytest.raises(ValueError): WaveformStatistics('autocorr')(torch.zeros(2,4))
    iq = torch.tensor([[1.,-1.,1.,-1.],[2.,-2.,2.,-2.]])
    assert torch.equal(WaveformStatistics('iq')(iq),torch.tensor([1.,4.,2.,-3.,4.]))
    event = WaveformStatistics('event')
    with pytest.raises(ValueError): event(iq)
    out = event(iq,{'event_ids':['real-packet-1'],'windows':[(0,4)]})
    assert torch.allclose(out,torch.tensor([5.,5**.5,0.]))


def test_source_normalization_freezes_and_restores():
    normal = FixedSourceNormalizer(2)
    with pytest.raises(RuntimeError): normal(torch.zeros(1,2))
    with pytest.raises(ValueError): normal.fit(torch.ones(3,2),source_role='target')
    source = torch.tensor([[1.,2.],[3.,2.]],requires_grad=True)
    normal.fit(source)
    assert normal.near_zero.tolist()==[False,True]
    assert torch.equal(normal(source),torch.tensor([[-1.,0.],[1.,0.]]))
    assert not normal(source).requires_grad
    with pytest.raises(RuntimeError): normal.fit(source)
    restored = FixedSourceNormalizer(2)
    restored.load_state_dict(normal.state_dict())
    assert torch.equal(normal(source),restored(source)) and restored.sample_count==2


def test_decision_robust_equal_rx_median_single_sided_and_detach():
    # RX0 query margin 1; other RX means 4,6,100 => reference 6, not max or sample mean.
    margins = torch.tensor([1.,4.,4.,6.,6.,100.,100.,100.,100.])
    logits = torch.stack((margins,torch.zeros_like(margins)),1).requires_grad_()
    labels = [0]*9
    rx = [0,1,1,2,2,3,3,3,3]
    loss,d = decision_margin_loss(logits,labels,rx,list(range(9)),[0]*9,min_records=2)
    assert d['reference'][0,1]==6
    # Check gradient of the query alone: reference graph must be detached.
    query_loss = torch.relu(d['reference'][0,1]-(logits[0,0]-logits[0,1])).square()
    query_loss.backward(retain_graph=True)
    assert logits.grad[1:].abs().sum()==0
    logits.grad.zero_()
    loss.backward()
    assert logits.grad[5:].abs().sum()==0  # strongest margins are never pulled down
    assert logits.grad[0,0]<0


def test_decision_excludes_duplicates_wrong_condition_and_bad_refs():
    logits = torch.tensor([[1.,0.],[5.,0.],[5.,0.],[-1.,0.]],requires_grad=True)
    loss,d = decision_margin_loss(logits,[0]*4,[0,1,1,2],['q','same','same','bad'],[0]*4,min_records=2)
    assert d['valid_comparisons']==0 and loss==0
    loss.backward()
    assert logits.grad.abs().sum()==0
    _,d = decision_margin_loss(logits,[0]*4,[0,1,1,2],list(range(4)),[0,1,1,0],min_records=2)
    assert not d['valid_mask'][0].any()
