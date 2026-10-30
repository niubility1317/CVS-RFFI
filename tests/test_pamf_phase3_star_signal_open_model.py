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


class ReturnAuxCore(SignalCore):
    def forward(self,x,y=None,return_aux=False):
        self.calls+=1;x=self.sinc(x);h=self.t2(x)
        pa=self.pa(torch.cat((x.mean(-1),x.square().mean(-1)),1))
        # Real ResidualPhysicalHead.forward always classifies, even though
        # IdentityOnlyCVS.features keeps only the returned joint embedding.
        out=self.cls_head(self.base(h.mean(-1)),None,pa,labels=y,return_emb=True)
        return dict(feat_joint=out[7],logits=out[0]) if return_aux else out[0]


def return_aux_source():
    """Use actual historical head/features method definitions, synthetic weights."""
    import ast,math
    from pathlib import Path
    root=Path(__file__).parents[1]
    def actual(relative,name,**namespace):
        nodes=ast.parse((root/relative).read_text(encoding='utf-8')).body
        node=next(n for n in nodes if isinstance(n,ast.ClassDef) and n.name==name)
        ns=dict(torch=torch,nn=nn,F=torch.nn.functional,math=math,**namespace)
        exec(compile(ast.Module(body=[node],type_ignores=[]),relative,'exec'),ns);return ns[name]
    Head=actual('experiments/cvs_residual_identity/model.py','ResidualPhysicalHead')
    Identity=actual('experiments/cvs_identity_ce/model.py','IdentityOnlyCVS')
    Reference=actual('experiments/cvs_reference_identity/model.py','ReferenceCVS')
    Sequence=actual('local_artifacts/cric_dura_comparison_20261021/frozen_sources/BASE/experiments/cvs_feature_sequence50/features.py',
        'SequenceReference',ReferenceCVS=Reference)
    native,x=source();reader=native.id_backbone.encoder;core=reader.core.id_backbone
    head=Head(160);head.load_state_dict(core.cls_head.state_dict(),strict=True);core.cls_head=head
    core.__class__=ReturnAuxCore;reader.core.__class__=Identity;reader.__class__=Sequence
    return native.eval(),x


def test_return_aux_real_head_discards_logits_but_preserves_features_and_all_gradients():
    train=importlib.import_module('experiments.pamf_phase3_open_world.star_signal_open_train')
    native,x=return_aux_source();reader=native.id_backbone.encoder;head=reader.core.id_backbone.cls_head
    original_state={k:v.detach().clone() for k,v in reader.state_dict().items()};identity=native(x).detach().clone()
    exported=train._export(native);copied=exported.encoder;copy_head=copied.core.id_backbone.cls_head
    assert type(copied) is type(reader) and type(copied).features is type(reader).features
    assert type(copy_head).forward is type(head).forward and type(copy_head).components is type(head).components
    raw=reader.features(x);copy_raw=exported.forward_features(x)
    torch.testing.assert_close(copy_raw,raw,atol=0,rtol=0)
    assert not hasattr(copy_head,'weight') and not any('cls_head.weight' in k for k in exported.state_dict())
    names=[k for k,p in reader.named_parameters() if k!='core.id_backbone.cls_head.weight']
    original=dict(reader.named_parameters());copy=dict(copied.named_parameters());weight=torch.linspace(.1,1.1,160)
    g0=torch.autograd.grad((raw*weight).sum(),[original[k] for k in names],allow_unused=True)
    g1=torch.autograd.grad((copy_raw*weight).sum(),[copy[k] for k in names],allow_unused=True)
    for left,right in zip(g0,g1):
        if left is None:assert right is None
        else:torch.testing.assert_close(right,left,atol=0,rtol=0)
    for prefix in ('core.id_backbone.sinc.','core.id_backbone.t2.','core.id_backbone.pa.','response_projection.','sequence_pool.'):
        assert any(g is not None and bool(g.abs().sum()>0) for k,g in zip(names,g1) if k.startswith(prefix))
    model=api().SignalOpen(exported);out=model.features(x)
    assert set(out)=={'z','u','c','raw_fused160','contributions','reference29'} and not out['reference29'][:,27].any()
    assert api().parameter_report(model)['signal_parameters']==334721
    torch.testing.assert_close(native(x),identity,atol=0,rtol=0)
    for k,v in reader.state_dict().items():torch.testing.assert_close(v,original_state[k],atol=0,rtol=0)


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


def test_export_actual_sequence_reader_boundary_keeps_feature_graph_and_original_identity(monkeypatch):
    """The source loader installs SequenceReference, not x3.CompositeReference.

    Use the native-interface toy with that distinct class boundary; exercise
    production _export without the Headless callback used by other fixtures.
    """
    import ast,sys,types
    from pathlib import Path
    train=importlib.import_module('experiments.pamf_phase3_open_world.star_signal_open_train')
    native,x=source();reader=native.id_backbone.encoder
    SequenceReference=type('SequenceReference',(type(reader),),{'__module__':'experiments.cvs_feature_sequence50.features'})
    reader.__class__=SequenceReference;original_logits=native(x).detach().clone()
    # Execute only the historical export definitions, avoiding unavailable
    # native-release constructors. The exact production guard remains intact.
    old=types.ModuleType('experiments.cvs_rdrop_sinc_units.x3_features');old.nn=nn;old.torch=torch
    old.CompositeReference=type('CompositeReference',(nn.Module,),{})
    text=Path(__file__).parents[1].joinpath('experiments/cvs_rdrop_sinc_units/x3_features.py').read_text(encoding='utf-8')
    nodes=[n for n in ast.parse(text).body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in ('ExportedComposite','export')]
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'<historical-x3-export>','exec'),old.__dict__)
    for name,key,value in (('experiments.cvs_source_identity_readout.input_policy','NoExplicitCFO',type(reader.response)),
            ('experiments.cvs_residual_identity.model','ResidualPhysicalHead',type(reader.core.id_backbone.cls_head))):
        module=types.ModuleType(name);setattr(module,key,value);monkeypatch.setitem(sys.modules,name,module)
    monkeypatch.setitem(sys.modules,old.__name__,old)
    source_state={n:p.detach().clone() for n,p in reader.state_dict().items()};features=type(reader).features
    exported=train._export(native)
    assert type(exported.encoder) is SequenceReference and type(exported.encoder).features is features
    assert sum(p.numel() for p in exported.parameters())==232033
    torch.testing.assert_close(exported.forward_features(x),reader.features(x),atol=0,rtol=0)
    assert hasattr(reader.core.id_backbone.cls_head,'weight') and not hasattr(exported.encoder.core.id_backbone.cls_head,'weight')
    assert not {p.data_ptr() for p in reader.parameters()} & {p.data_ptr() for p in exported.parameters()}
    a=api();model=a.SignalOpen(exported);opt=a.optimizer(model)
    for step in range(2):
        opt.zero_grad(set_to_none=True);model(x)[:,0].sum().backward()
        assert model.signal.encoder.core.id_backbone.sinc.low_coordinate.grad.abs().sum()>0
        assert model.signal.encoder.core.id_backbone.t2[0].weight.grad.abs().sum()>0
        assert model.signal.encoder.response_projection[0].weight.grad.abs().sum()>0
        assert model.signal.encoder.sequence_pool.projection.weight.grad.abs().sum()>0
        if step:assert model.projection.network[0].weight.grad.abs().sum()>0
        a.clip(opt,model);opt.step()
    torch.testing.assert_close(native(x),original_logits,atol=0,rtol=0)
    for n,p in reader.state_dict().items():torch.testing.assert_close(p,source_state[n],atol=0,rtol=0)
