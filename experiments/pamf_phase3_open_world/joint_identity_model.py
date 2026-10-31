"""r03 shared feature body with its same-source original native cosine head."""
from __future__ import annotations
from copy import deepcopy
import torch
from torch import nn
from . import star_signal_open_model as signal

SINC_PREFIX=signal.SINC_PREFIX
HEAD_WEIGHT='signal.encoder.core.id_backbone.cls_head.weight'


class JointIdentity(signal.SignalOpen):
    def __init__(self,parent,original_reader):
        nn.Module.__init__(self)
        if signal.parameter_report(parent)['signal_parameters']!=334721:raise ValueError('Complete source-only r03 signal/projection required')
        copied=deepcopy(parent);self.signal=copied.signal;self.projection=copied.projection
        head=self.signal.encoder.core.id_backbone.cls_head;source=original_reader.core.id_backbone.cls_head
        if hasattr(head,'weight') or source.weight.shape!=(6,160) or source.weight.dtype!=torch.float32 or float(source.scale)!=30.:raise ValueError('Original six-class native cosine W[6,160]/scale30 required')
        head.weight=nn.Parameter(source.weight.detach().to(next(self.parameters()).device).clone());head.scale=source.scale
        # Keep the instance's discarded-logit hook for native return_aux. The
        # final original cosine computation is called explicitly in joint().
        self.requires_grad_(True).eval()
        for name,p in self.named_parameters():
            if name.startswith(SINC_PREFIX):p.requires_grad_(False)
        if parameter_report(self)!=dict(total_parameters=335681,signal_feature_parameters=232033,projection_parameters=102688,identity_head_parameters=960,trainable_parameters=335633):raise ValueError('Complete native joint body/projection/head parameter contract differs')

    def joint(self,x):
        features=self.features(x);head=self.signal.encoder.core.id_backbone.cls_head
        logits=type(head).classify(head,features['raw_fused160'])
        if logits.shape!=(len(x),6) or logits.dtype!=torch.float32 or not bool(torch.isfinite(logits).all()):raise ValueError('Finite original six-class native logits required')
        return dict(z=features['z'],logits=logits)


def parameter_report(model):
    count=lambda m:sum(p.numel() for p in m.parameters())
    return dict(total_parameters=count(model),signal_feature_parameters=count(model.signal)-960,projection_parameters=count(model.projection),
        identity_head_parameters=model.signal.encoder.core.id_backbone.cls_head.weight.numel(),trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad))
