"""Independent differentiable headless PAMF signal path and metric residual.

The original six-class identifier is neither stored nor changed by this module.
"""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F
from .star_metric_residual_model import MetricResidual,SEED,_matrix

FEATURE_PARAMETERS=232033
PROJECTION_PARAMETERS=102688
PARAMETERS=334721
SINC_PREFIX='signal.encoder.core.id_backbone.sinc.'


def unit(x):
    """Same FP64 canonical normalization as the frozen source feature cache."""
    norm=x.double().norm(dim=1,keepdim=True)
    if bool(norm.eq(0).any()):raise ValueError('Nonzero signal features required')
    return (x.double()/norm).float()


def contribution_unit(c):
    value=c.double();norm=value.norm(dim=2,keepdim=True)
    value=value/torch.where(norm>0,norm,torch.ones_like(norm))
    return unit(value.flatten(1))


class SignalOpen(nn.Module):
    def __init__(self,signal):
        super().__init__()
        if not isinstance(signal,nn.Module) or not callable(getattr(signal,'forward_features',None)):
            raise ValueError('Actual exported headless composite required')
        if any(n.endswith('cls_head.weight') for n,_ in signal.named_parameters()):
            raise ValueError('Six-class identity W must not enter the open signal branch')
        self.signal=signal;self.projection=MetricResidual(SEED)
        self.to(device=next(signal.parameters()).device,dtype=torch.float32);self.requires_grad_(True);self.eval()

    def train(self,mode=True):
        # eval freezes BN running moments/dropout, while autograd remains enabled.
        return super().train(False)

    def features(self,x):
        if (not torch.is_tensor(x) or x.ndim!=3 or not len(x) or tuple(x.shape[1:])!=(2,256)
                or x.dtype!=torch.float32 or not bool(torch.isfinite(x).all())):
            raise ValueError('Finite FP32[N,2,256] cached P1 IQ required')
        self.eval();reader=self.signal.encoder;head=reader.core.id_backbone.cls_head;n=len(x);captured={};handles=[]
        if x.device!=next(self.parameters()).device:raise ValueError('IQ and signal network must share a device')
        targets=(head.base_norm,head.pa_norm,reader.response_projection,reader.response,reader.sequence_pool)
        def record(key,value):
            if key in captured:raise ValueError('Each native contribution must execute exactly once')
            captured[key]=value
        try:
            for key,module in zip(('base','pa','ref_projection','reference29','signature'),targets):
                handles.append(module.register_forward_hook(lambda m,a,v,key=key:record(key,v)))
            with torch.autocast(device_type=x.device.type,enabled=False):
                raw=_matrix(self.signal.forward_features(x),160,'student fused160')
                if set(captured)!={'base','pa','ref_projection','reference29','signature'}:raise ValueError('Five same-pass native hooks required')
                for k in ('base','pa','ref_projection','signature'):
                    if _matrix(captured[k],160,k).shape[0]!=n:raise ValueError('Aligned native contributions required')
                ref=captured['reference29']
                if ref.shape!=(n,29) or ref.dtype!=torch.float32 or not bool(torch.isfinite(ref).all()) or bool(ref[:,27].ne(0).any()):
                    raise ValueError('Finite masked reference29 with CFO coordinate 27 exactly zero required')
                c_raw=torch.stack((captured['base'],head.gain.tanh()*captured['pa'],reader.response_gain.tanh()*captured['ref_projection'],captured['signature']),1)
                u=unit(raw);c=contribution_unit(c_raw)
                # Deliberately do not call old MetricResidual.forward: it detaches u/c.
                z=F.normalize(F.normalize(u,dim=1)+self.projection.network(F.normalize(c,dim=1)),dim=1,eps=1e-12)
            return dict(z=_matrix(z,160,'signal metric160'),u=u,c=c,raw_fused160=raw,contributions=c_raw,reference29=ref)
        finally:
            for handle in handles:handle.remove()

    def forward(self,x):return self.features(x)['z']


def parameter_report(model):
    count=lambda m:sum(p.numel() for p in m.parameters())
    return dict(signal_feature_parameters=count(model.signal),projection_parameters=count(model.projection),
        signal_parameters=count(model),combined_active_parameters=count(model)+232993)


def optimizer(model,lr=.0001):
    """Headless CENTER_UNITS policy, using the original attach/physical clip."""
    from experiments.cric_dura_comparison import f_optimizer
    named=dict(model.named_parameters());sinc=model.signal.encoder.core.id_backbone.sinc
    names=tuple(SINC_PREFIX+k for k in ('low_coordinate','band_coordinate'))
    if (type(sinc).__name__!='CoordinateSinc' or sinc.mode!='CENTER_25K' or sinc.scale_hz!=25000.
            or sinc.sample_rate!=25000000. or float(lr)!=.0001):raise ValueError('Fixed differentiable CENTER_25K signal Sinc required')
    leaves=[named[k] for k in names];flat=list(model.parameters())
    if any(not p.requires_grad for p in flat) or any(p.shape!=(24,1) for p in leaves):raise ValueError('All headless signal/projection parameters must train')
    ids={id(p) for p in leaves};other=[p for p in flat if id(p) not in ids]
    policy=dict(mode='CENTER_UNITS',multiplier=1000.,base_eps=1e-8,base_weight_decay=.0001,sinc=leaves,names=names,
        scale=25000.,origins=[sinc.low_origin_hz,sinc.band_origin_hz])
    groups=[dict(params=other),dict(params=leaves,lr=lr*1000.,eps=1e-11,weight_decay=0.)]
    return f_optimizer.attach(torch.optim.AdamW(groups,lr=lr,eps=1e-8,weight_decay=.0001),policy)


def clip(optimizer,model):
    from experiments.cric_dura_comparison import f_optimizer
    return f_optimizer.clip_physical(optimizer,model.parameters(),5.)
