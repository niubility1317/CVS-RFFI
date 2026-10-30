"""Differentiable same-pass signal path; no real weights or observations."""
from copy import deepcopy
import importlib
import numpy as np
import pytest
import torch
from torch import nn
from test_pamf_phase3_star_reference_open_model import fixture as native_fixture, hooks, Core


def api():return importlib.import_module('experiments.pamf_phase3_open_world.star_signal_open_model')


class CoordinateSinc(nn.Module):
    def __init__(self):
        super().__init__();self.low_coordinate=nn.Parameter(torch.zeros(24,1));self.band_coordinate=nn.Parameter(torch.ones(24,1))
        self.register_buffer('low_origin_hz',torch.full((24,1),100000.));self.register_buffer('band_origin_hz',torch.full((24,1),50000.))
        self.scale_hz=25000.;self.sample_rate=25000000.;self.mode='CENTER_25K'
    def forward(self,x):return x*(1+.01*self.low_coordinate.mean())+.01*self.band_coordinate.mean()*x.square()


class Headless(nn.Module):
    def __init__(self,native):
        super().__init__();self.encoder=deepcopy(native.id_backbone.encoder);del self.encoder.core.id_backbone.cls_head.weight
    def forward_features(self,x):return self.encoder.features(x)


class SignalCore(Core):
    def forward(self,x):return super().forward(self.sinc(x))


def source():
    native,x=native_fixture();reader=native.id_backbone.encoder;core=reader.core.id_backbone
    core.sinc=CoordinateSinc();core.__class__=SignalCore
    # Remove toy-only count padding and replace it to retain the real F0 count.
    del reader.count_padding;reader.count_padding=nn.Parameter(torch.zeros(232993-sum(p.numel() for p in reader.parameters())))
    return native,x


def test_one_pass_zero_last_matches_frozen_preprocessing_without_detaching_student():
    a=api();native,x=source();signal=Headless(native);model=a.SignalOpen(signal).eval();r=signal.encoder
    out=model.features(x);assert set(out)=={'z','u','c','raw_fused160','contributions','reference29'}
    from experiments.pamf_phase3_open_world import star_density_core as g
    np.testing.assert_array_equal(np.asarray(out['u'].detach().tolist(),np.float32),g.unit(np.asarray(out['raw_fused160'].detach().tolist(),np.float32)).astype(np.float32))
    np.testing.assert_allclose(np.asarray(out['c'].detach().tolist(),np.float32),g.contribution_unit(np.asarray(out['contributions'].detach().tolist(),np.float32)),atol=6e-8)
    assert r.core.id_backbone.calls==1 and r.response.calls==1 and r.sequence_pool.calls==1
    assert not out['reference29'][:,27].any() and out['u'].requires_grad and out['c'].requires_grad
    torch.testing.assert_close(out['z'],torch.nn.functional.normalize(out['u'],dim=1),atol=0,rtol=0)
    assert a.parameter_report(model)==dict(signal_feature_parameters=232033,projection_parameters=102688,signal_parameters=334721,combined_active_parameters=567714)
    assert not any(n.endswith('cls_head.weight') for n,p in model.named_parameters())
    out['z'][:,0].sum().backward()
    assert r.core.id_backbone.sinc.low_coordinate.grad.abs().sum()>0
    assert r.core.id_backbone.t2[0].weight.grad.abs().sum()>0
    assert model.projection.network[-1].weight.grad.abs().sum()>0 and not model.projection.network[0].weight.grad.any()


def test_hooks_cleanup_masked_reference_and_eval_state():
    a=api();native,x=source();model=a.SignalOpen(Headless(native));model.train();assert not any(m.training for m in model.modules())
    before=hooks(model);buffers={n:b.clone() for n,b in model.named_buffers()};model(x)
    assert hooks(model)==before
    for n,b in model.named_buffers():torch.testing.assert_close(b,buffers[n],atol=0,rtol=0)
    model.signal.encoder.fail=True
    with pytest.raises(RuntimeError):model(x)
    assert hooks(model)==before
    model.signal.encoder.fail=False;model.signal.encoder.response.unmask=True
    with pytest.raises(ValueError,match='27'):model(x)
    assert hooks(model)==before


def test_center_units_two_groups_physical_decay_and_clip_without_class_head():
    a=api();native,x=source();model=a.SignalOpen(Headless(native)).eval();opt=a.optimizer(model,lr=.0001)
    assert len(opt.param_groups)==2 and opt.param_groups[1]['lr']==pytest.approx(.1)
    assert opt.param_groups[1]['eps']==pytest.approx(1e-11) and opt.param_groups[1]['weight_decay']==0
    q=model.signal.encoder.core.id_backbone.sinc.low_coordinate;before=q.detach().clone()
    # Physical decay must occur even for zero adaptive gradients.
    for p in model.parameters():p.grad=torch.zeros_like(p)
    norm=a.clip(opt,model);assert float(norm)==0;opt.step()
    torch.testing.assert_close(q,before-.0001*.0001*(before+4),atol=1e-12,rtol=1e-6)
    assert opt._sinc_unit_steps==opt._sinc_physical_clip_calls==1
    assert opt._sinc_unit_lr_records[0]['sinc_lr']==pytest.approx(.1)
    with pytest.raises(ValueError):opt.step()


def test_input_errors_and_no_numpy_abi(monkeypatch):
    a=api();native,x=source();model=a.SignalOpen(Headless(native))
    monkeypatch.setattr(torch.Tensor,'numpy',lambda *a,**k:pytest.fail('NumPy ABI forbidden'))
    monkeypatch.setattr(torch,'from_numpy',lambda *a,**k:pytest.fail('NumPy ABI forbidden'))
    assert model(x).shape==(4,160)
    for bad in (x.double(),x[:,:,:255],x*float('nan')):
        with pytest.raises(ValueError):model(bad)
