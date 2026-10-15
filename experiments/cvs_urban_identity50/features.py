"""Observe the deployed fusion; change only auxiliary gain gradients."""
from contextlib import contextmanager
import torch


def modules(model):
    encoder=model.id_backbone.encoder;core=encoder.core.id_backbone
    if tuple(encoder.sequence_pool.modes)!=('SIG_TIME',) or core.cls_head.weight.shape!=(6,160) or core.cls_head.scale!=30.:
        raise ValueError('Fixed SIG_TIME/default cosine6x160 required')
    return encoder,core


@contextmanager
def capture(model):
    encoder,core=modules(model);values={};handles=[]
    targets=dict(base=core.cls_head.base_norm,PA=core.cls_head.pa_norm,
        reference=encoder.response_projection,signature=encoder.sequence_pool.output_norm)
    for name,module in targets.items():
        def record(module,args,output,name=name):
            if name in values:raise ValueError('Unexpected repeated deployed fusion branch')
            values[name]=output
        handles.append(module.register_forward_hook(record))
    try:yield values
    finally:
        for handle in handles:handle.remove()


def auxiliary_joint(model,output,values):
    encoder,core=modules(model)
    if set(values)!={'base','PA','reference','signature'}:raise ValueError('Actual deployed branch was bypassed')
    z=values['base']+core.cls_head.gain.tanh().detach()*values['PA']
    z=z+encoder.response_gain.tanh().detach()*values['reference']
    z=z+(.2*encoder.sequence_pool.gain_raw.tanh()).detach()*values['signature']
    actual=output['id_feat_joint']
    if actual.shape!=(len(z),160) or not torch.equal(actual.detach(),z.detach()):
        raise ValueError('Auxiliary feature is not the actual deployed joint160')
    return z


def satellite_forward(model,x):
    with capture(model) as values:out=model.forward_identity_only(x)
    return out,auxiliary_joint(model,out,values)


def parameter_groups(model):
    encoder,core=modules(model)
    return dict(t1=list(core.t1.parameters()),t2=list(core.t2.parameters()),
        SIG_local=list(encoder.sequence_pool.views[0].local_projection.parameters()),
        PA_projection=list(core.pa_proj.parameters()),reference_projection=list(encoder.response_projection.parameters()))


def boundary_parameters(model):
    encoder,core=modules(model)
    return dict(default_head=core.cls_head.weight,PA_gain=core.cls_head.gain,
        reference_gain=encoder.response_gain,signature_gain=encoder.sequence_pool.gain_raw)


def frozen_teacher(model,x):
    if any(p.requires_grad for p in model.parameters()):raise ValueError('Own EMA must be frozen')
    modes={m:m.training for m in model.modules()};model.eval()
    try:
        with torch.no_grad():out=model(x,y_tx=None,grl_lambda=1.,return_aux=True,domain_labels=None)
    finally:
        for module,mode in modes.items():module.training=mode
    if out['id_feat_joint'].shape!=(len(x),160):raise ValueError('Own EMA joint160 semantics differ')
    return {k:out[k].detach() for k in ('id_feat_joint','tx_logits')}
