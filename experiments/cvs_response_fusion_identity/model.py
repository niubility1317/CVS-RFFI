"""Constrain compensation-response fusion using the measured norm-cancellation failure."""
import torch
from torch.nn import functional as F
from experiments.cvs_channel_response_identity.model import ChannelResponseCVS
from experiments.cvs_neural_residual_identity.model import NeuralResidualCVS

VARIANTS=('response_span_mean','response_anchor_mean')
EPSILON=1e-4
SPAN_RIDGE=1e-3


def contrast_response(response,weight):
    """Permutation-equivariant ridge projection into centered class-weight span.

    The result is in the span even when the rows are dependent. This is not an
    exact orthogonal projector: ridge stabilizes the small differentiable solve.
    """
    normalized=F.normalize(weight,dim=1,eps=EPSILON)
    contrast=normalized-normalized.mean(0,keepdim=True)
    gram=contrast@contrast.T
    inverse_action=torch.linalg.solve(gram+SPAN_RIDGE*torch.eye(len(weight),device=weight.device,dtype=weight.dtype),contrast)
    return (response@contrast.T)@inverse_action


def anchored_features(base,response):
    """Separate base normalization from a bounded, zero-at-identity response."""
    denominator=(base.square().sum(1,keepdim=True)+response.square().sum(1,keepdim=True)).clamp_min(EPSILON**2).sqrt()
    return F.normalize(base,dim=1,eps=EPSILON)+response/denominator


class ResponseFusionCVS(ChannelResponseCVS):
    def __init__(self,variant):
        if variant not in VARIANTS:raise ValueError('Unregistered response fusion')
        super().__init__('response_mean');self.fusion_variant=variant

    def fusion_components(self,x):
        base=NeuralResidualCVS.features(self,x);response=self.response_branch(x)['residual']
        head=self.core.id_backbone.cls_head
        if self.fusion_variant=='response_span_mean':
            used=contrast_response(response,head.weight);joint=base+used
        else:
            denominator=(base.square().sum(1,keepdim=True)+response.square().sum(1,keepdim=True)).clamp_min(EPSILON**2).sqrt()
            used=response/denominator;joint=F.normalize(base,dim=1,eps=EPSILON)+used
        return dict(base=base,raw_response=response,used_response=used,features=joint)

    def features(self,x):return self.fusion_components(x)['features']

    def classify_features(self,features):
        if self.fusion_variant=='response_span_mean':return super().classify_features(features)
        head=self.core.id_backbone.cls_head
        # Deliberately no second joint normalization: a head-null response must
        # not amplify the original centered logits by shrinking their divisor.
        return head.scale*F.linear(features,F.normalize(head.weight,dim=1,eps=EPSILON))

    def contract(self):
        c=super().contract()
        c.update(mode=self.fusion_variant,identity_core='response_fusion_identity',response_base_variant='response_mean',
            fusion_active=True,fusion_mode=self.fusion_variant,fusion_new_parameters=0,fusion_normalization_epsilon=EPSILON,
            fusion_uses_classifier_weight=self.fusion_variant=='response_span_mean',
            fusion_span_ridge=SPAN_RIDGE if self.fusion_variant=='response_span_mean' else None,
            fusion_span_interpretation='range of centered normalized class weights; ridge projection, not exact orthogonal projection' if self.fusion_variant=='response_span_mean' else None,
            fusion_final_joint_normalization=self.fusion_variant=='response_span_mean',
            fusion_response_norm_bound=1. if self.fusion_variant=='response_anchor_mean' else None,
            fusion_original_classifier_weights_retained=True,fusion_classifier_scale=30.,
            fusion_labels_used_in_forward=False,fusion_cross_packet_state=False,fusion_new_loss=False,fusion_new_training_strategy=False,
            initial_function='exact own scratch response_mean/shallow classifier while G=I; no inherited weights',
            interpretation='Remove measured classifier-null normalization shortcut; not arbitrary-channel invariance, calibrated equalization or guaranteed performance improvement')
        return c

    @torch.no_grad()
    def response_diagnostics(self,x):
        d=super().response_diagnostics(x)
        d.update(variant=self.fusion_variant,raw_branch_variant='response_mean',scope='Raw compensation-response branch before constrained fusion; actual used response is reported separately')
        return d

    @torch.no_grad()
    def fusion_diagnostics(self,x):
        flags=[(m,m.training) for m in self.modules()]
        try:
            self.eval();v=self.fusion_components(x);b=v['base'];r=v['raw_response'];used=v['used_response']
            denom=b.norm(dim=1).clamp_min(1e-12)
            head=self.core.id_backbone.cls_head;w=F.normalize(head.weight,dim=1,eps=EPSILON);w=w-w.mean(0)
            return dict(active=True,variant=self.fusion_variant,packets=len(x),raw_response_relative=float((r.norm(dim=1)/denom).mean()),
                        used_response_norm=float(used.norm(dim=1).mean()),used_response_norm_max=float(used.norm(dim=1).max()),
                        base_feature_norm=float(b.norm(dim=1).mean()),centered_response_logit_norm=float((used@w.T).norm(dim=1).mean()),
                        final_joint_normalization=self.fusion_variant=='response_span_mean',scope='Actual frozen current-batch features; no labels or fitted state')
        finally:
            for module,training in flags:module.training=training

    @torch.no_grad()
    def diagnostics(self,x):return dict(super().diagnostics(x),response_fusion=self.fusion_diagnostics(x))


def build(variant):return ResponseFusionCVS(variant)


def fusion_contract(variant):
    with torch.random.fork_rng(devices=[]):return build(variant).contract()
