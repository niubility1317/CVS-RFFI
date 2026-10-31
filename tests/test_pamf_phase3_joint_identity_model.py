"""Original cosine head on r03 fused features, with one encoder execution."""
import importlib
import pytest
import torch
from torch.nn import functional as F


def api():return importlib.import_module('experiments.pamf_phase3_open_world.joint_identity_model')


def fixture():
    from test_pamf_phase3_star_signal_open_model import return_aux_source
    from experiments.pamf_phase3_open_world import star_signal_open_model as signal
    torch.set_num_threads(1);native,x=return_aux_source();reader=native.id_backbone.encoder
    parent=signal.SignalOpen(signal.HeadlessSignal(reader)).eval().requires_grad_(False)
    with torch.no_grad():parent.projection.network[-1].weight.normal_(0,.003)
    return native,reader,parent,x


def test_same_pass_native_cosine_and_shared_body_head_gradients():
    a=api();native,reader,parent,x=fixture();before={k:v.clone() for k,v in reader.state_dict().items()};model=a.JointIdentity(parent,reader)
    calls=model.signal.encoder.core.id_backbone.calls;out=model.joint(x);assert set(out)=={'z','logits'} and model.signal.encoder.core.id_backbone.calls==calls+1
    raw=parent.features(x)['raw_fused160'];weight=reader.core.id_backbone.cls_head.weight
    oracle=30*F.linear(F.normalize(raw,dim=1,eps=1e-4),F.normalize(weight,dim=1,eps=1e-4));torch.testing.assert_close(out['logits'],oracle,rtol=0,atol=0)
    head=model.signal.encoder.core.id_backbone.cls_head
    assert head.classify.__self__ is head and head.classify.__func__.__name__=='_discarded_feature_logits'
    F.cross_entropy(out['logits'],torch.arange(len(x))%6).backward()
    assert head.weight.grad.norm()>0 and model.signal.encoder.core.id_backbone.t2[0].weight.grad.norm()>0
    assert all(p.grad is None for n,p in model.named_parameters() if n.startswith(a.SINC_PREFIX))
    assert all(p.grad is None for p in model.projection.parameters())  # Native identity uses raw fused, not z.
    assert a.parameter_report(model)==dict(total_parameters=335681,signal_feature_parameters=232033,projection_parameters=102688,identity_head_parameters=960,trainable_parameters=335633)
    assert sum(k==a.HEAD_WEIGHT for k in model.state_dict())==1 and all(torch.equal(v,reader.state_dict()[k]) for k,v in before.items())
    assert not {p.data_ptr() for p in model.parameters()}&({p.data_ptr() for p in parent.parameters()}|{p.data_ptr() for p in reader.parameters()})


def test_strict_complete_model_and_invalid_inputs_preserve_hooks(subtests):
    a=api();native,reader,parent,x=fixture();model=a.JointIdentity(parent,reader);copy=a.JointIdentity(parent,reader)
    copy.load_state_dict(model.state_dict(),strict=True);torch.testing.assert_close(copy.joint(x)['logits'],model.joint(x)['logits'],rtol=0,atol=0)
    state=dict(model.state_dict());state.pop(a.HEAD_WEIGHT)
    with pytest.raises(RuntimeError):copy.load_state_dict(state,strict=True)
    for bad in (x.double(),x[:,:,:255],x*float('nan')):
        with subtests.test(shape=tuple(bad.shape)),pytest.raises(ValueError):model.joint(bad)
    assert not any(m._forward_hooks for m in model.modules()) and not any(m.training for m in model.modules())

