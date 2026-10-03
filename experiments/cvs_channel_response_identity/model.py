"""Explicit compensation response: zero auxiliary output whenever G is identity.

The source attribution found almost duplicate u/v descriptors in the previous
additive branch. Here the auxiliary route receives only the change produced by
G, retaining the original shallow CVS backbone and original single CE head.
"""
import torch
from torch import nn
from experiments.cvs_channel_order_identity.model import BoundedPacketFilter,LocalComplexFeature,CROP
from experiments.cvs_neural_residual_identity.model import NeuralResidualCVS,neural_contract

VARIANTS=('response_mean','response_attention','response_order_attention')
BASE_VARIANT='neural_residual_shallow'


def relative_cross_tokens(reference,change):
    """Phase invariant, signed, linear in the change near zero, no time pooling."""
    r,i=reference.unbind(1);dr,di=change.unbind(1)
    energy=(r.square()+i.square()).mean(-1,keepdim=True).clamp_min(1e-6)
    return torch.cat(((r*dr+i*di)/energy,(r*di-i*dr)/energy),dim=1)


class ResponseBranch(nn.Module):
    def __init__(self,variant):
        super().__init__()
        if variant not in VARIANTS:raise ValueError('Unregistered response architecture')
        self.variant=variant;self.feature=LocalComplexFeature()
        with torch.random.fork_rng(devices=[]):self.compensation=BoundedPacketFilter()
        self.value=nn.Sequential(nn.Conv1d(32 if variant=='response_order_attention' else 16,64,1,bias=False),nn.GELU(),nn.Conv1d(64,64,1,bias=False))
        self.attention=None if variant=='response_mean' else nn.Sequential(nn.Conv1d(8,16,1),nn.GELU(),nn.Conv1d(16,4,1))
        self.project=nn.Linear(64,160,bias=False)

    def forward(self,x):
        gx,coefficients=self.compensation(x)
        u_full=self.feature(x[:,:,None,:]);v_full=self.feature(gx[:,:,None,:])
        u=u_full[...,CROP:-CROP];v=v_full[...,CROP:-CROP]
        delta=v-u
        tokens=relative_cross_tokens(u,delta)
        difference=None
        if self.variant=='response_order_attention':
            w=self.compensation.apply_filter(u_full,coefficients)[...,CROP:-CROP]
            difference=v-w
            tokens=torch.cat((tokens,relative_cross_tokens(u,difference)),dim=1)
        values=self.value(tokens).reshape(len(x),4,16,-1)
        if self.attention is None:
            weights=torch.full((len(x),4,u.shape[-1]),1/u.shape[-1],dtype=x.dtype,device=x.device)
        else:
            power=u.square().sum(1)
            context=torch.log1p(power/power.mean(-1,keepdim=True).clamp_min(1e-6))
            weights=self.attention(context).softmax(-1)
        pooled=(values*weights[:,:,None,:]).sum(-1).flatten(1)
        residual=self.project(pooled)
        return dict(residual=residual,delta=delta,difference=difference,tokens=tokens,values=values,weights=weights,pooled=pooled,u=u,compensated_input=gx)


class ChannelResponseCVS(NeuralResidualCVS):
    def __init__(self,variant):
        if variant not in VARIANTS:raise ValueError('Unregistered response architecture')
        super().__init__(BASE_VARIANT);self.response_variant=variant
        with torch.random.fork_rng(devices=[]):self.response_branch=ResponseBranch(variant)

    def features(self,x):return super().features(x)+self.response_branch(x)['residual']

    def response_parameters(self):return list(self.response_branch.parameters())

    def contract(self):
        c=neural_contract(BASE_VARIANT);b=self.response_branch
        extra=sum(p.numel() for p in b.parameters());total=sum(p.numel() for p in self.parameters())
        c.update(mode=self.response_variant,identity_core='channel_response_identity',base_variant=BASE_VARIANT,
                 base_trainable_parameters=total-extra,new_trainable_parameters=extra,total_parameters=total,total_trainable_parameters=total,
                 response_active=(isinstance(b,ResponseBranch) and isinstance(b.feature,LocalComplexFeature) and isinstance(b.compensation,BoundedPacketFilter) and b.variant==self.response_variant),response_input='phase-invariant signed cross(u,F(Gx)-F(x)), divided by per-packet per-channel mean|u|^2',
                 response_order_active=b.variant=='response_order_attention',response_order_input='cross(u,F(Gx)-G(Fx))' if b.variant=='response_order_attention' else None,
                 response_compensation_active=isinstance(b.compensation,BoundedPacketFilter),
                 response_identity_null=True,response_direct_u_v_readout=False,response_linear_near_zero=True,
                 response_value_channels=[32 if b.variant=='response_order_attention' else 16,64,64],response_value_bias=False,
                 response_attention_active=b.attention is not None,response_heads=4,response_head_dimension=16,
                 response_attention_context='log1p(reference power / per-channel packet mean power)',response_context_channels=[8,16,4] if b.attention is not None else None,
                 response_projection=[64,160],response_valid_crop=CROP,response_valid_tokens=246,response_rho=.25,
                 response_operator='G from original packet; same F for original/compensated inputs; no downsampling inside response branch',
                 response_new_training_strategy=False,response_loss='original single cross_entropy',response_inference_fit=False,response_cross_packet_state=False,
                 initial_function='exact own scratch neural_residual_shallow; G identity, bias-free response values/project; nonzero projection permits G exit gradient',
                 response_interpretation='learned sensitivity to bounded compensation, not calibrated channel inverse, TX recovery, or arbitrary-domain invariance')
        return c

    @torch.no_grad()
    def response_diagnostics(self,x):
        flags=[(m,m.training) for m in self.modules()]
        try:
            self.eval();base=NeuralResidualCVS.features(self,x);a=self.response_branch(x)
            ratio=lambda z,ref:(z.flatten(1).norm(dim=1)/ref.flatten(1).norm(dim=1).clamp_min(1e-12)).mean().item()
            gradient=lambda params:float(torch.stack([p.grad.detach().norm() for p in params if p.grad is not None]).norm()) if any(p.grad is not None for p in params) else None
            b=self.response_branch
            return dict(active=self.contract()['response_active'],variant=self.response_variant,packets=len(x),compensation_input_relative=ratio(a['compensated_input']-x,x),
                        response_feature_relative=ratio(a['delta'],a['u']),response_token_norm=a['tokens'].flatten(1).norm(dim=1).mean().item(),
                        response_pooled_norm=a['pooled'].norm(dim=1).mean().item(),response_projection_relative=ratio(a['residual'],base),
                        response_order_relative=None if a['difference'] is None else ratio(a['difference'],a['u']),
                        attention_entropy=-(a['weights']*a['weights'].clamp_min(1e-12).log()).sum(-1).mean().item(),
                        compensation_exit_gradient_norm=gradient(list(b.compensation.context[-1].parameters())),
                        value_gradient_norm=gradient(list(b.value.parameters())),projection_gradient_norm=gradient(list(b.project.parameters())),
                        attention_gradient_norm=None if b.attention is None else gradient(list(b.attention.parameters())),
                        scope='given source batch only, no fitting or persistent state')
        finally:
            for module,training in flags:module.training=training

    @torch.no_grad()
    def diagnostics(self,x):
        flags=[(m,m.training) for m in self.modules()]
        try:
            self.eval()
            return dict(super().diagnostics(x),channel_response=self.response_diagnostics(x))
        finally:
            for module,training in flags:module.training=training


def build(variant):return ChannelResponseCVS(variant)


def response_contract(variant):
    with torch.random.fork_rng(devices=[]):return build(variant).contract()
