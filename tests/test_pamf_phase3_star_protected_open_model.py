"""Only synthetic native F0 inputs; protected160 never changes its identity head."""
from copy import deepcopy
import importlib

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from test_pamf_phase3_star_reference_open_model import fixture, hooks


def api():
    return importlib.import_module('experiments.pamf_phase3_open_world.star_protected_open_model')


def test_zero_init_exact_identity_one_native_pass_actual_contributions_and_parameter_counts():
    a=api();model,x=fixture();reader=model.id_backbone.encoder;head=reader.core.id_backbone.cls_head
    before=model(x);params=dict(model.named_parameters());buffers={n:b.clone() for n,b in model.named_buffers()}
    features_method=reader.features;head_method=reader.classify_features
    calls=(reader.core.id_backbone.calls,reader.response.calls,reader.sequence_pool.calls)
    assert a.install(model,residual_enabled=True) is model
    out=a.forward(model,x)
    assert set(out)=={'raw_original_fused160','raw_protected_fused160','protected160','contributions','reference29','logits'}
    torch.testing.assert_close(out['logits'],before,atol=0,rtol=0)
    torch.testing.assert_close(out['raw_original_fused160'],out['raw_protected_fused160'],atol=0,rtol=0)
    torch.testing.assert_close(out['protected160'],F.normalize(out['raw_original_fused160'],dim=1),atol=0,rtol=0)
    c=out['contributions'];assert c.shape==(4,4,160) and out['reference29'].shape==(4,29)
    torch.testing.assert_close(((c[:,0]+c[:,1])+c[:,2])+c[:,3],out['raw_original_fused160'],atol=0,rtol=0)
    assert not out['reference29'][:,27].any()
    assert (reader.core.id_backbone.calls,reader.response.calls,reader.sequence_pool.calls)==tuple(v+1 for v in calls)
    assert model.id_backbone.encoder is reader and reader.core.id_backbone.cls_head is head
    assert reader.features==features_method and reader.classify_features==head_method
    for n,p in params.items():assert dict(model.named_parameters())[n] is p and not p.requires_grad
    for n,b in buffers.items():torch.testing.assert_close(dict(model.named_buffers())[n],b,atol=0,rtol=0)
    report=a.parameter_report(model)
    expected=dict(native_identity_parameters=232993,open_projector_parameters=45312,direction_projection_parameters=10400,
        residual_gain_parameters=160,adapter_parameters=55872,relation_parameters=10305,open_world_parameters=66177,
        active_parameters=299170,native_container_parameters=900722,training_container_parameters=966899,
        unused_native_parameters=667729,trainable_identity_parameters=0,trainable_adapter_parameters=55872,
        trainable_open_world_parameters=66177)
    for k,v in expected.items():assert report[k]==v
    assert not hasattr(model.open_world,'conditioner') and not hasattr(model.open_world,'identity_projection')
    assert not hasattr(reader,'_open_world_original_features')


def test_first_step_gain_and_relation_then_direction_gradients_original_state_stays_frozen():
    a=api();model,x=fixture();reader=model.id_backbone.encoder
    reader.native_bn=nn.BatchNorm1d(160);original=reader.features
    reader.features=lambda values:reader.native_bn(original(values))
    model.eval();identity=model(x).detach().clone();source={n:v.detach().clone() for n,v in model.state_dict().items()}
    a.install(model,True);model.train()  # forward must still keep the complete original F0 in eval mode.
    proto=F.normalize(torch.randn(3,160),dim=1);optimizer=torch.optim.SGD([p for p in model.parameters() if p.requires_grad],lr=.1)
    for step in range(2):
        optimizer.zero_grad(set_to_none=True);out=a.forward(model,x)
        torch.testing.assert_close(out['logits'],identity,atol=0,rtol=0)
        assert not reader.training and not reader.native_bn.training
        a.relation(model,out['protected160'],proto).square().mean().backward()
        assert model.open_world.residual_gain.grad.abs().sum()>0
        assert model.open_world.relation_scorer[0].weight.grad.abs().sum()>0
        direction=model.open_world.direction_projection.weight.grad
        projection=model.open_world.projector[0].weight.grad
        if step==0:assert not direction.any() and not projection.any()
        else:assert direction.abs().sum()>0 and projection.abs().sum()>0
        assert all(p.grad is None for n,p in model.named_parameters() if not n.startswith('open_world.'))
        optimizer.step()
    for n,v in source.items():torch.testing.assert_close(model.state_dict()[n],v,atol=0,rtol=0)
    torch.testing.assert_close(a.forward(model,x)['logits'],identity,atol=0,rtol=0)


def test_bound_original_head_routing_zero_baseline_and_outer_autocast():
    a=api();model,x=fixture();reader=model.id_backbone.encoder;baseline=model(x).detach().clone();a.install(model,True)
    with torch.no_grad():model.open_world.residual_gain.fill_(20);reader.response_gain.zero_()
    baseline=model(x).detach().clone();captured=[]
    original=reader.classify_features
    reader.classify_features=lambda z:(captured.append(z),original(z))[1]
    with torch.autocast(device_type='cpu',dtype=torch.bfloat16):out=a.forward(model,x)
    assert all(v.dtype==torch.float32 for v in out.values())
    torch.testing.assert_close(captured[-1],out['raw_original_fused160'],atol=0,rtol=0)
    torch.testing.assert_close(out['logits'],baseline,atol=0,rtol=0)
    delta=out['raw_protected_fused160']-out['raw_original_fused160']
    assert (delta.norm(dim=1)<=.1*out['raw_original_fused160'].norm(dim=1)+2e-6).all()
    assert delta.abs().sum()>0 and not out['contributions'][:,2].any()
    original_features=reader.features;reader.features=lambda values:original_features(values)*0
    zero=a.forward(model,x)
    assert not zero['raw_original_fused160'].any() and not zero['raw_protected_fused160'].any() and not zero['protected160'].any()


def test_R0_same_initialization_computes_adapter_but_only_relation_is_trainable():
    a=api();r0,x=fixture();r1=deepcopy(r0);rng=torch.get_rng_state().clone()
    a.install(r0,False);a.install(r1,True)
    torch.testing.assert_close(torch.get_rng_state(),rng,atol=0,rtol=0)
    for n,p in r0.open_world.named_parameters():
        torch.testing.assert_close(p,dict(r1.open_world.named_parameters())[n],atol=0,rtol=0)
        assert p.requires_grad==n.startswith('relation_scorer.')
    with torch.no_grad():r0.open_world.residual_gain.fill_(20)
    calls=[];handle=r0.open_world.direction_projection.register_forward_hook(lambda *unused:calls.append(1))
    try:out=a.forward(r0,x)
    finally:handle.remove()
    assert calls==[1]
    torch.testing.assert_close(out['raw_protected_fused160'],out['raw_original_fused160'],atol=0,rtol=0)
    a.relation(r0,out['protected160'],F.normalize(torch.randn(2,160),dim=1)).sum().backward()
    assert all(p.grad is None for n,p in r0.open_world.named_parameters() if not n.startswith('relation_scorer.'))
    assert r0.open_world.relation_scorer[0].weight.grad.abs().sum()>0
    report=a.parameter_report(r0)
    assert report['trainable_adapter_parameters']==0 and report['trainable_open_world_parameters']==10305


def test_relation_shared_and_per_query_prototypes_finite_shapes_permutation_and_no_mutation(subtests):
    a=api();model,x=fixture();a.install(model,True);out=a.forward(model,x);z=out['protected160']
    p=F.normalize(torch.randn(3,160),dim=1);z0=z.detach().clone();p0=p.clone();order=torch.tensor([2,0,1])
    scores=a.relation(model,z,p);assert scores.shape==(4,3) and scores.dtype==torch.float32 and torch.isfinite(scores).all()
    torch.testing.assert_close(a.relation(model,z,p[order]),scores[:,order])
    torch.testing.assert_close(a.relation(model,z,p.expand(4,-1,-1)),scores)
    query_p=p.expand(4,-1,-1).clone().detach().requires_grad_()
    query_scores=a.relation(model,z,query_p)
    expected=torch.cat([a.relation(model,z[i:i+1],query_p[i]) for i in range(4)])
    torch.testing.assert_close(query_scores,expected)
    query_scores.sum().backward()
    assert query_p.grad.shape==(4,3,160) and (query_p.grad.abs().sum((1,2))>0).all()
    torch.testing.assert_close(z,z0,atol=0,rtol=0);torch.testing.assert_close(p,p0,atol=0,rtol=0)
    for bad_z,bad_p in ((z[:,:64],p),(z,p[:,:64]),(z.double(),p),(z,p.double()),
                        (z,torch.full_like(p,float('nan'))),(z,p.expand(3,-1,-1))):
        with subtests.test(shape=tuple(bad_p.shape),dtype=bad_p.dtype),pytest.raises(ValueError):a.relation(model,bad_z,bad_p)


def test_encode_cached_same_pass_features_has_no_native_calls_and_stops_original_gradients(subtests):
    a=api();model,x=fixture();a.install(model,True);reader=model.id_backbone.encoder
    out=a.forward(model,x);calls=(reader.core.id_backbone.calls,reader.response.calls,reader.sequence_pool.calls)
    f=out['raw_original_fused160'].detach().clone().requires_grad_()
    c=out['contributions'].detach().clone().requires_grad_();before_f=f.detach().clone();before_c=c.detach().clone()
    values=a.encode(model,f,c)
    assert set(values)=={'raw_protected_fused160','protected160'}
    for k,v in values.items():torch.testing.assert_close(v,out[k],atol=0,rtol=0)
    assert (reader.core.id_backbone.calls,reader.response.calls,reader.sequence_pool.calls)==calls
    a.relation(model,values['protected160'],F.normalize(torch.randn(2,160),dim=1)).sum().backward()
    assert f.grad is None and c.grad is None and model.open_world.residual_gain.grad.abs().sum()>0
    torch.testing.assert_close(f,before_f,atol=0,rtol=0);torch.testing.assert_close(c,before_c,atol=0,rtol=0)
    for bad_f,bad_c in ((f[:,:159],c),(f,c[:,:3]),(f.double(),c),(f,c.double()),(f,torch.full_like(c,float('nan')))):
        with subtests.test(f=tuple(bad_f.shape),c=tuple(bad_c.shape)),pytest.raises(ValueError):a.encode(model,bad_f,bad_c)


def test_errors_remove_only_call_local_hooks_and_multiple_models_are_isolated(subtests):
    a=api();model,x=fixture();other=deepcopy(model);a.install(model,True);a.install(other,True)
    reader=model.id_backbone.encoder;initial=hooks(model);other_out=a.forward(other,x)
    for flag in ('fail','skip_signature','duplicate_response','unmask'):
        owner=reader.response if flag=='unmask' else reader;setattr(owner,flag,True)
        with subtests.test(flag=flag),pytest.raises((ValueError,RuntimeError)):a.forward(model,x)
        assert hooks(model)==initial and model._protected_open_last is None
        assert other._protected_open_last is other_out
        setattr(owner,flag,False)
    for invalid in (x[:,:,:255],x.double(),torch.full_like(x,float('nan'))):
        with subtests.test(shape=tuple(invalid.shape),dtype=invalid.dtype),pytest.raises(ValueError):a.forward(model,invalid)
        assert hooks(model)==initial and model._protected_open_last is None
    out=a.forward(model,x);saved={k:v.detach().clone() for k,v in out.items()};a.clear(other)
    assert model._protected_open_last is out and other._protected_open_last is None
    a.forward(other,x*.7)
    for k,v in out.items():torch.testing.assert_close(v,saved[k],atol=0,rtol=0)
    with pytest.raises(ValueError):a.install(model,True)
    with pytest.raises(ValueError):a.install(fixture()[0],1)
    with pytest.raises(ValueError):a.install(fixture()[0],True,seed=True)
