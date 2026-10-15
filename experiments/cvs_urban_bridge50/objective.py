"""One common joint anchor; the experimental increment is local SIG teaching."""
import math
import torch
from torch.nn import functional as F
from experiments.cvs_urban_bridge.signature import split,KEYS


def weights(logits,y,records):
    if y.ndim!=1 or logits.shape!=(len(y),6) or len(records)!=len(y) or bool(((y<0)|(y>=6)).any()):
        raise ValueError('True labeled source batch and six-class own EMA required')
    if any(r['scenario'] not in ('practical_low_suburban','practical_mid_urban','practical_low_urban') for r in records):
        raise ValueError('Only the declared source training scenes are eligible')
    urban=y.new_tensor([r['scenario'].endswith('_urban') for r in records],dtype=logits.dtype)
    q=logits.new_tensor([r['q'] for r in records])
    if not torch.isfinite(q).all() or bool(((q<0)|(q>1)).any()):raise ValueError('Finite continuous usability in [0,1] required')
    correct=(logits.detach().argmax(1)==y).to(logits.dtype)
    return (urban*correct).detach(),(urban*correct*q).detach()


def direction(student,target,weight):
    if student.shape!=target.shape or student.ndim!=2 or student.shape[1]!=160 or weight.shape!=(len(student),):
        raise ValueError('Actual corresponding joint160 and per-record reliability required')
    a=F.normalize(student,dim=1,eps=1e-4);b=F.normalize(target.detach(),dim=1,eps=1e-4)
    return ((1-(a*b).sum(1).clamp(-1,1))*weight.detach()).mean()


def joint(urban,hq,clean_teacher,weight):
    return .5*(direction(urban,clean_teacher,weight)+direction(hq,clean_teacher,weight))


def local(student,teacher,scales,weight):
    if student.shape!=teacher.shape or weight.shape!=(len(student),) or set(scales)!=set(KEYS):
        raise ValueError('Corresponding raw112 teacher/student and source RMS scales required')
    if any(not math.isfinite(float(scales[k])) or not 1e-3<=float(scales[k])<1e6 for k in KEYS):
        raise ValueError('Frozen finite source RMS scales, floor1e-3, required')
    a,b=split(student),split(teacher.detach());groups={}
    for key in KEYS:
        error=(a[key]-b[key])/scales[key]
        if not torch.isfinite(error).all():raise FloatingPointError('Nonfinite scaled local SIG evidence')
        huber=F.smooth_l1_loss(error,torch.zeros_like(error),reduction='none')
        groups[key]=(huber/(1+huber)).flatten(1).mean(1)
    per_record=sum(groups.values())/3
    return (per_record*weight.detach()).mean(),{k:v.mean() for k,v in groups.items()}
