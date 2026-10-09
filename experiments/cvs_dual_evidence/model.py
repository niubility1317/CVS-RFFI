"""Two active backbones: full received IQ and within-packet curvature evidence.

Curvature is invariant to a scalar complex gain/constant CFO in the ideal
nonzero-input limit. It does not identify a transmitter or cancel arbitrary
FIR/IQ imbalance/nonlinear receiver distortion. The raw identity path is retained.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from experiments.cvs_phase1_overlay.model import native_modules
native_modules()
from model_dual_cvsincnet import DualCVSincNetDisentangle


def curvature_view(x):
    if x.ndim!=3 or x.shape[1:]!=(2,256):raise ValueError('Expected received IQ[B,2,256]')
    x=x.float();z=torch.complex(x[:,0],x[:,1]);power=z.abs().square().mean(1,keepdim=True)
    z=z/torch.sqrt(power+1e-12);p=z.abs().square();eps=1e-4;values=[]
    for lag in (1,2,4,8):
        center=z[:,lag:-lag];plus=z[:,2*lag:];minus=z[:,:-2*lag]
        pc=p[:,lag:-lag];pp=p[:,2*lag:];pm=p[:,:-2*lag]
        numerator=plus*minus*center.conj().square()
        denominator=torch.sqrt(pp*pm+eps**2)*pc+eps
        phase=numerator/denominator
        amplitude=torch.tanh(.25*(torch.log(pp+eps)+torch.log(pm+eps)-2*torch.log(pc+eps)))
        for value in (phase.real,phase.imag,amplitude):values.append(F.pad(value,(lag,lag)))
    return torch.stack(values,1)


def raw_view(x):return x.float().repeat(1,6,1)/math.sqrt(6.)


class ResidualBlock(nn.Module):
    def __init__(self,cin,cout,stride):
        super().__init__()
        self.body=nn.Sequential(nn.Conv1d(cin,cout,5,stride,padding=2,bias=False),nn.GroupNorm(8,cout),nn.SiLU(),
            nn.Conv1d(cout,cout,3,padding=1,bias=False),nn.GroupNorm(8,cout))
        self.skip=nn.Conv1d(cin,cout,1,stride,bias=False)
    def forward(self,x):return F.silu(self.body(x)+self.skip(x))


class EvidenceBackbone(nn.Module):
    def __init__(self,curvature):
        super().__init__();self.curvature=curvature
        widths=(12,32,64,96,160)
        self.blocks=nn.Sequential(*[ResidualBlock(widths[i],widths[i+1],2) for i in range(4)])
        self.pool=nn.Sequential(nn.Linear(320,160),nn.LayerNorm(160))
    def forward(self,x):
        h=self.blocks(curvature_view(x) if self.curvature else raw_view(x))
        return self.pool(torch.cat([h.mean(-1),torch.sqrt(h.var(-1,unbiased=False)+1e-5)],1))


class ConditionalFusion(nn.Module):
    def __init__(self,interaction):
        super().__init__();self.interaction=interaction
        self.a_norm=nn.LayerNorm(160);self.b_norm=nn.LayerNorm(160)
        self.a=nn.Linear(160,64);self.b=nn.Linear(160,64);self.out=nn.Linear(64,160,bias=False)
        nn.init.zeros_(self.out.weight)
    def forward(self,za,zb):
        a=self.a(self.a_norm(za));b=self.b(self.b_norm(zb))
        gate=1+.5*torch.tanh(a)
        h=F.silu(b)*gate if self.interaction else F.silu(a+b)
        delta=self.out(h)
        self.last=dict(correction_ratio=(delta.detach().norm(dim=1)/za.detach().norm(dim=1).clamp_min(1e-6)).mean(),
            gate_abs_delta=(gate.detach()-1).abs().mean() if self.interaction else za.new_tensor(0.))
        return za+delta


class DualEvidenceModel(DualCVSincNetDisentangle):
    """Retain native public methods and state names; replace unused nuisance CNN."""
    def forward(self,x,y_tx=None,grl_lambda=0.,return_aux=False,domain_labels=None,**kwargs):
        # No TX/domain labels enter the evidence transform or fusion operation.
        if self.id_backbone.mixstyle_on:raise ValueError('MixStyle is outside this fixed baseline')
        branches={};hooks=[];head=self.id_backbone.encoder.core.id_backbone.cls_head
        if return_aux:
            hooks=[head.base_norm.register_forward_hook(lambda m,i,o:branches.update(identity=o)),
                head.pa_norm.register_forward_hook(lambda m,i,o:branches.update(physical=o))]
        try:
            with torch.autocast(device_type=x.device.type,enabled=False):
                za=self.id_backbone.encoder.features(x.float());zb=self.dom_backbone(x.float())
                z=self.evidence_fusion(za,zb);logits=self.id_backbone.encoder.classify_features(z)
        finally:
            for hook in hooks:hook.remove()
        self.last_dual_metrics=dict(self.evidence_fusion.last,secondary_feature_norm=zb.detach().norm(dim=1).mean())
        if not return_aux:return logits
        aid=dict(feat_joint=z,feat_con=z,logits=logits,feat_cls=branches['identity'],
            feat_imp=branches['physical'],feat_pa=branches['physical'],feat_dac=torch.zeros_like(z),base=branches['identity'])
        zero=logits.new_zeros((len(x),self.num_domains))
        out=dict(tx_logits=logits,z_id=z,z_dom=zb,z_dom_raw=zb,dom_logits=zero,adv_dom_logits=zero,
            z_id_key='dual_evidence_fused',z_dom_key='complementary_evidence_not_domain',
            aux_id=aid,aux_dom=dict(feat_joint=zb,feat_imp=zb),representation_mode='dual',
            tx_adv_on_zdom=False,crra_enabled=False,crra_q=None,crra_q_raw=None,crra_condition_tx_adv_logits=None,
            crra_correction_energy=logits.new_zeros(len(x)),crra_gate=logits.new_zeros(len(x)),
            crra_alpha=logits.new_zeros(len(x)),crra_support_distance=logits.new_zeros(len(x)))
        for name,key in [('id_feat_cls','feat_cls'),('id_feat_imp','feat_imp'),('id_feat_pa','feat_pa'),
            ('id_feat_dac','feat_dac'),('id_feat_joint','feat_joint'),('id_feat_con','feat_con'),('id_base','base')]:out[name]=aid[key]
        return out

    def forward_identity_only(self,x,y_tx=None,domain_labels=None,crra_epoch=None):
        return self.forward(x,y_tx=y_tx,domain_labels=domain_labels,return_aux=True)

    def dual_diagnostics(self):
        result={k:float(v) for k,v in getattr(self,'last_dual_metrics',{}).items()}
        for name,module in [('main',self.id_backbone),('secondary',self.dom_backbone),('fusion',self.evidence_fusion)]:
            gradients=[p.grad.detach().float().square().sum() for p in module.parameters() if p.grad is not None]
            result[name+'_gradient_norm']=float(torch.stack(gradients).sum().sqrt()) if gradients else None
        return result


def upgrade(model,arm,seed):
    if arm=='legacy_cosine':return model
    if arm not in ('raw_dual','curvature_dual','curvature_interaction'):raise ValueError('Unknown arm')
    device=next(model.parameters()).device
    # CPU-only private initialization preserves both process CPU and all CUDA RNG.
    state=torch.random.get_rng_state()
    try:
        torch.random.set_rng_state(torch.Generator(device='cpu').manual_seed(int(seed)+17603).get_state())
        secondary=EvidenceBackbone(arm!='raw_dual')
        fusion=ConditionalFusion(arm=='curvature_interaction')
    finally:torch.random.set_rng_state(state)
    model.__class__=DualEvidenceModel
    model.dom_backbone=secondary.to(device);model.evidence_fusion=fusion.to(device);model.evidence_arm=arm
    # Unused legacy heads remain explicitly reported, never mistaken for active losses.
    for name in ('dom_enhancer','dom_head','adv_head'):
        for p in getattr(model,name).parameters():p.requires_grad_(False)
    return model
