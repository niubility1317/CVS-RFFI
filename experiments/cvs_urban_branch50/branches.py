"""Capture the five actual fusion inputs; preserve student autograd graphs."""
from contextlib import contextmanager
import torch
from experiments.cvs_urban_identity50 import features as joint
BRANCHES=('T','F','P','R','S')


def layout(model):
    encoder,core=joint.modules(model)
    if core.fuse[0].in_features!=321:
        raise ValueError('Fixed full T160/F160/rho1 fusion layout required')
    return encoder,core


@contextmanager
def capture(model):
    encoder,core=layout(model);values={};handles=[]
    def record(name,value):
        if name in values:raise ValueError('Repeated deployed branch capture: '+name)
        if value.ndim!=2:raise ValueError('Actual per-record fusion vectors required')
        values[name]=value
    def fusion(module,args):
        x=args[0]
        if x.ndim!=2 or x.shape[1]!=321:raise ValueError('Actual fuse input layout changed')
        record('T',x[:,:160]);record('F',x[:,160:320]);record('_rho',x[:,320:])
    handles.append(core.fuse.register_forward_pre_hook(fusion))
    for name,module in dict(_base=core.cls_head.base_norm,P=core.cls_head.pa_norm,
            R=encoder.response_projection,S=encoder.sequence_pool.output_norm).items():
        def hook(module,args,out,name=name):record(name,out)
        handles.append(module.register_forward_hook(hook))
    try:yield values
    finally:
        for handle in handles:handle.remove()


def forward(model,x):
    with capture(model) as values:out=model.forward_identity_only(x)
    complete(model,out,values,len(x))
    return out,values


def complete(model,out,values,count):
    if set(values)!={*BRANCHES,'_base','_rho'}:raise ValueError('An actual fusion branch was bypassed')
    if any(values[k].shape!=(count,160) for k in BRANCHES):raise ValueError('Fixed 160-wide branch inputs required')
    if out['tx_logits'].shape!=(count,6) or not torch.isfinite(out['tx_logits']).all():
        raise ValueError('Finite default cosine6 outputs required')
    values['J']=joint.auxiliary_joint(model,out,dict(base=values['_base'],PA=values['P'],
        reference=values['R'],signature=values['S']))


class NativeCapture:
    """Arm only after concat creation; remove hooks before any native U/EMA forward."""
    def __init__(self,model,x):
        self.model=model;self.x=x;self.done=False;self.handles=[]
        self.scope=capture(model);self.values=self.scope.__enter__();self.closed=False
        def before(module,args):
            if not args or args[0] is not x:raise ValueError('Next native forward is not the registered labeled concat')
        def after(module,args,out):
            self.close()
            complete(model,out,self.values,len(x));self.done=True
        self.handles=[model.register_forward_pre_hook(before),model.register_forward_hook(after)]

    def close(self):
        if self.closed:return
        for handle in self.handles:handle.remove()
        self.scope.__exit__(None,None,None);self.closed=True


def frozen_teacher(model,x):
    if any(p.requires_grad for p in model.parameters()):raise ValueError('Own EMA must be frozen')
    modes={m:m.training for m in model.modules()};model.eval()
    try:
        with torch.no_grad():out,values=forward(model,x)
    finally:
        for module,mode in modes.items():module.training=mode
    return out['tx_logits'].detach(),{k:values[k].detach() for k in (*BRANCHES,'J')}


def feature_parameters(model,z):
    """Actual joint160 ancestor parameters; absent per-loss gradients later count as zero."""
    named=[(name,p) for name,p in model.named_parameters() if p.requires_grad]
    gradients=torch.autograd.grad(z.sum(),[p for _,p in named],allow_unused=True,retain_graph=True)
    selected=[(name,p) for (name,p),g in zip(named,gradients) if g is not None]
    head=joint.modules(model)[1].cls_head.weight
    if any(p is head for _,p in selected):raise ValueError('Default class weights cannot be feature extractor parameters')
    if not selected:raise ValueError('Actual joint160 feature graph is detached')
    return selected


@contextmanager
def replace_one(model,clean,key):
    """Operational frozen same-model intervention; it is not a deployable input."""
    encoder,core=layout(model)
    if key not in BRANCHES:raise ValueError('Exactly one registered branch must be replaced')
    if model.training or any(p.requires_grad for p in model.parameters()):
        raise ValueError('Branch replacement is restricted to a frozen diagnostic model')
    target=clean[key]
    if target.requires_grad or target.ndim!=2 or target.shape[1]!=160 or not torch.isfinite(target).all():
        raise ValueError('Detached same-model clean branch required')
    if key in ('T','F'):
        def replace(module,args):
            x=args[0]
            if x.shape!=(len(target),321) or x.dtype!=target.dtype or x.device!=target.device:
                raise ValueError('Frozen branch shape/device/dtype mismatch')
            a=target if key=='T' else x[:,:160]
            b=target if key=='F' else x[:,160:320]
            return (torch.cat((a,b,x[:,320:]),dim=1),*args[1:])
        handle=core.fuse.register_forward_pre_hook(replace)
    else:
        module=dict(P=core.cls_head.pa_norm,R=encoder.response_projection,S=encoder.sequence_pool.output_norm)[key]
        def replace(module,args,out):
            if out.shape!=target.shape or out.dtype!=target.dtype or out.device!=target.device:
                raise ValueError('Frozen branch shape/device/dtype mismatch')
            return target
        handle=module.register_forward_hook(replace)
    try:yield
    finally:handle.remove()
