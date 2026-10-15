"""Actual four-window SIG_TIME groups, before its existing output projection."""
from contextlib import contextmanager
import torch
from torch.nn import functional as F

KEYS=('increment','feature_area','time_area')
PAIRS=tuple((i,j) for i in range(7) for j in range(i+1,7))
UU=tuple(7+k for k,(i,j) in enumerate(PAIRS) if j<6)
UT=tuple(7+k for k,(i,j) in enumerate(PAIRS) if j==6)


def split(descriptor):
    if descriptor.ndim!=2 or descriptor.shape[1]!=112:raise ValueError('Actual SIG_TIME112 required')
    v=descriptor.reshape(-1,4,28)
    return dict(increment=v[:,:,:6],feature_area=v[:,:,UU],time_area=v[:,:,UT])


@contextmanager
def capture(model):
    pool=model.id_backbone.encoder.sequence_pool
    if tuple(pool.modes)!=('SIG_TIME',) or len(pool.views)!=1:raise ValueError('Fixed single SIG_TIME view required')
    view=pool.views[0]
    if view.mode!='SIG_TIME' or not torch.equal(view.descriptor_lift,torch.eye(28,device=view.descriptor_lift.device)):
        raise ValueError('Actual increment7/area21 identity layout required')
    values={}
    def hook(module,args,out):
        if values:raise ValueError('Repeated SIG_TIME forward')
        if out.ndim!=2 or out.shape[1]!=112:raise ValueError('Descriptor layout changed')
        if not torch.allclose(out.reshape(-1,4,28)[:,:,6],out.new_ones((len(out),4)),atol=2e-6,rtol=0):
            raise ValueError('Expected constant time increment at coordinate6')
        values['descriptor']=out
    handle=view.register_forward_hook(hook)
    try:yield values
    finally:handle.remove()


def teacher_scales(descriptors,floor=1e-3):
    if not 0<floor<1:raise ValueError('Positive source scale floor required')
    groups=split(descriptors.detach())
    return {key:float(v.square().mean().sqrt().clamp_min(floor)) for key,v in groups.items()}


def grouped_loss(student,teacher,scales,weight=None):
    if student.shape!=teacher.shape or set(scales)!=set(KEYS):raise ValueError('Matched time-corresponding groups/scales required')
    if any(not 0<float(scales[key])<1e6 for key in KEYS):raise ValueError('Finite positive frozen source scales required')
    a,b=split(student),split(teacher.detach());losses={}
    for key in KEYS:
        # Mean within each semantic group first: 15 area coordinates get no
        # automatic 15/6 advantage. Fixed dt is deliberately absent.
        losses[key]=F.smooth_l1_loss((a[key]-b[key])/scales[key],torch.zeros_like(a[key]),reduction='none').flatten(1).mean(1)
    each=sum(losses.values())/3
    if weight is not None:
        if weight.shape!=each.shape:raise ValueError('One source reliability weight per item required')
        each=each*weight.detach()
    return each.mean(),{key:value.mean() for key,value in losses.items()}
