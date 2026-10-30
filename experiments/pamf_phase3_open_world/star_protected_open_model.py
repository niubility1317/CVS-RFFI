"""Frozen original-F0 identity and a bounded contribution-conditioned unit160.

The original feature and classification methods remain unchanged. Only the new
adapter and pair relation can train; the R0 control disables adapter updates.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


SEED=2026092701
PARAMETERS=66177
ACTIVE_PARAMETERS=299170
UNIT_EPS=1e-12
CONTRIBUTION_NAMES=('time_frequency','PA','reference','signature')


class ProtectedParts(nn.Module):
    def __init__(self,residual_enabled):
        super().__init__();self.residual_enabled=residual_enabled
        self.projector=nn.Sequential(nn.Linear(640,64),nn.GELU(),nn.Linear(64,64),nn.LayerNorm(64))
        self.direction_projection=nn.Linear(64,160)
        self.residual_gain=nn.Parameter(torch.zeros(160))
        self.relation_scorer=nn.Sequential(nn.Linear(320,32),nn.GELU(),nn.Linear(32,1))
        if not residual_enabled:
            self.projector.requires_grad_(False);self.direction_projection.requires_grad_(False)
            self.residual_gain.requires_grad_(False)


def _parts(model):
    parts=getattr(model,'open_world',None)
    if not isinstance(parts,ProtectedParts):raise ValueError('Install the protected160 module first')
    return parts


def _matrix(value,shape,name):
    if (not torch.is_tensor(value) or tuple(value.shape)!=shape or value.dtype!=torch.float32
            or not bool(torch.isfinite(value).all())):
        raise ValueError('Finite FP32 tensor with the declared shape required: '+name)
    return value


def clear(model):
    for handle in getattr(model,'_protected_open_handles',()):handle.remove()
    model._protected_open_handles=[];model._protected_open_last=None


def install(native_model,residual_enabled=True,seed=SEED):
    if type(residual_enabled) is not bool or type(seed) is not int or seed<=0:
        raise ValueError('Boolean residual policy and positive integer initialization seed required')
    if not isinstance(native_model,nn.Module) or hasattr(native_model,'open_world'):
        raise ValueError('One unmodified original native model required')
    try:
        reader=native_model.id_backbone.encoder;head=reader.core.id_backbone.cls_head
        targets=(head.base_norm,head.pa_norm,reader.response_projection,reader.response,reader.sequence_pool)
        prototype=next(reader.parameters())
    except (AttributeError,StopIteration) as exc:
        raise ValueError('Complete original fusion, gain and masked-reference interfaces required') from exc
    if (not all(isinstance(m,nn.Module) for m in targets)
            or not callable(getattr(reader,'features',None)) or not callable(getattr(reader,'classify_features',None))
            or hasattr(reader,'_open_world_original_features')
            or not torch.is_tensor(getattr(head,'gain',None)) or tuple(head.gain.shape)!=(160,)
            or not torch.is_tensor(getattr(reader,'response_gain',None)) or tuple(reader.response_gain.shape)!=(160,)):
        raise ValueError('Unmodified original feature/head methods and actual 160-dimensional fusion gains required')
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed);parts=ProtectedParts(residual_enabled)
    parts.to(device=prototype.device,dtype=torch.float32)
    for parameter in native_model.parameters():parameter.requires_grad_(False);parameter.grad=None
    native_model.eval();native_model.add_module('open_world',parts)
    clear(native_model);return native_model


def encode(model,raw_original_fused160,contributions):
    """Use legal same-pass cached F0 inputs without a native/head forward.

    Both inputs are fixed teacher quantities. Detach explicitly so callers
    cannot accidentally backpropagate through an upstream original F0 graph.
    """
    parts=_parts(model)
    if not torch.is_tensor(raw_original_fused160) or raw_original_fused160.ndim!=2 or not len(raw_original_fused160):
        raise ValueError('Nonempty original fused160 matrix required')
    n=len(raw_original_fused160);f=_matrix(raw_original_fused160,(n,160),'original fused160')
    c=_matrix(contributions,(n,4,160),'four native contributions')
    if f.device!=c.device or f.device!=parts.residual_gain.device:
        raise ValueError('Original features, contributions and adapter must share a device')
    with torch.autocast(device_type=f.device.type,enabled=False):
        f=f.detach();c=c.detach()
        h=parts.projector(c.flatten(1))
        direction=F.normalize(parts.direction_projection(h),dim=1,eps=UNIT_EPS)
        delta=.1*f.norm(dim=1,keepdim=True)*parts.residual_gain.tanh()*direction
        raw=f+delta if parts.residual_enabled else f+torch.zeros_like(delta)
        protected=F.normalize(raw,dim=1,eps=UNIT_EPS)
    _matrix(raw,(n,160),'protected raw fused160');_matrix(protected,(n,160),'protected unit160')
    return dict(raw_protected_fused160=raw,protected160=protected)


def forward(model,x):
    """One frozen native pass; only the bounded open branch remains trainable."""
    _parts(model);clear(model);handles=[]
    try:
        if (not torch.is_tensor(x) or x.ndim!=3 or not len(x) or tuple(x.shape[1:])!=(2,256)
                or x.dtype!=torch.float32 or not bool(torch.isfinite(x).all())):
            raise ValueError('Finite original native FP32[N,2,256] input required')
        # An outer model.train() must not update original BN buffers/dropout.
        model.training=False
        for name,child in model.named_children():
            if name!='open_world':child.eval()
        reader=model.id_backbone.encoder;head=reader.core.id_backbone.cls_head;captured={};n=len(x)
        targets=(head.base_norm,head.pa_norm,reader.response_projection,reader.response,reader.sequence_pool)
        def record(name,value):
            if name in captured:raise ValueError('Each native contribution must be captured exactly once: '+name)
            captured[name]=value
        for name,module in zip(('identity','physical','reference_projection','reference29','signature'),targets):
            handles.append(module.register_forward_hook(lambda m,inputs,value,key=name:record(key,value)))
        model._protected_open_handles=handles
        with torch.no_grad(),torch.autocast(device_type=x.device.type,enabled=False):
            f=_matrix(reader.features(x),(n,160),'original raw fused160')
            if set(captured)!={'identity','physical','reference_projection','reference29','signature'}:
                raise ValueError('All five same-pass native contribution captures required')
            for name in ('identity','physical','reference_projection','signature'):_matrix(captured[name],(n,160),name)
            reference=_matrix(captured['reference29'],(n,29),'masked reference29')
            if bool(reference[:,27].ne(0).any()):raise ValueError('NoExplicitCFO reference coordinate 27 must remain zero')
            contributions=torch.stack((captured['identity'],head.gain.tanh()*captured['physical'],
                reader.response_gain.tanh()*captured['reference_projection'],captured['signature']),dim=1)
            logits=_matrix(reader.classify_features(f),(n,6),'original default-head logits')
        result=dict(raw_original_fused160=f,**encode(model,f,contributions),contributions=contributions,reference29=reference,logits=logits)
        model._protected_open_last=result;return result
    except BaseException:
        model._protected_open_last=None;raise
    finally:
        for handle in handles:handle.remove()
        model._protected_open_handles=[]


def relation(model,z,prototypes):
    """Score shared [K,160] or query-specific [N,K,160] support prototypes."""
    parts=_parts(model)
    if not torch.is_tensor(z) or z.ndim!=2 or not len(z):raise ValueError('Nonempty unit160 query matrix required')
    n=len(z);_matrix(z,(n,160),'relation queries')
    if (not torch.is_tensor(prototypes) or prototypes.ndim not in (2,3) or not prototypes.shape[-2]
            or prototypes.shape[-1]!=160 or (prototypes.ndim==3 and len(prototypes)!=n)):
        raise ValueError('Shared [K,160] or aligned [N,K,160] support prototypes required')
    k=prototypes.shape[-2];_matrix(prototypes,(k,160) if prototypes.ndim==2 else (n,k,160),'support prototypes')
    if z.device!=prototypes.device or z.device!=parts.residual_gain.device:raise ValueError('Relation inputs and parameters must share a device')
    p=prototypes[None] if prototypes.ndim==2 else prototypes
    with torch.autocast(device_type=z.device.type,enabled=False):
        inputs=torch.cat(((z[:,None]-p).abs(),z[:,None]*p),dim=-1)
        result=parts.relation_scorer(inputs).squeeze(-1)
    return _matrix(result,(n,k),'relation logits')


def parameter_report(model):
    parts=_parts(model)
    count=lambda module:sum(p.numel() for p in module.parameters())
    trainable=lambda module:sum(p.numel() for p in module.parameters() if p.requires_grad)
    identity=count(model.id_backbone.encoder);additional=count(parts);container=count(model)
    adapter=count(parts.projector)+count(parts.direction_projection)+parts.residual_gain.numel()
    return dict(native_identity_parameters=identity,open_projector_parameters=count(parts.projector),
        direction_projection_parameters=count(parts.direction_projection),residual_gain_parameters=parts.residual_gain.numel(),
        adapter_parameters=adapter,relation_parameters=count(parts.relation_scorer),open_world_parameters=additional,
        active_parameters=identity+additional,native_container_parameters=container-additional,
        training_container_parameters=container,unused_native_parameters=container-additional-identity,
        trainable_identity_parameters=trainable(model.id_backbone.encoder),
        trainable_adapter_parameters=trainable(parts.projector)+trainable(parts.direction_projection)+(parts.residual_gain.numel() if parts.residual_gain.requires_grad else 0),
        trainable_open_world_parameters=trainable(parts),residual_enabled=parts.residual_enabled,
        expected_open_world_parameters=PARAMETERS,expected_active_parameters=ACTIVE_PARAMETERS)
