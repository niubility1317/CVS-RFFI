"""Independent open metric residual; never an identity classifier or F0 child."""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F

SEED=2026092701
PARAMETERS=102688
TEMPERATURE=.1


def _matrix(x,width,name):
    if (not torch.is_tensor(x) or x.ndim!=2 or not len(x) or x.shape[1]!=width or x.dtype!=torch.float32
            or not bool(torch.isfinite(x).all())):raise ValueError('Finite FP32 matrix required: '+name)
    return x


def _unit(x):
    if bool(x.norm(dim=1).eq(0).any()):raise ValueError('Every feature must be nonzero')
    return F.normalize(x,dim=1,eps=1e-12)


class MetricResidual(nn.Module):
    def __init__(self,model_seed=SEED):
        super().__init__()
        if type(model_seed) is not int or model_seed!=SEED:raise ValueError('Fixed initialization seed 2026092701 required')
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(model_seed)
            self.network=nn.Sequential(nn.Linear(640,128),nn.GELU(),nn.Linear(128,160))
            nn.init.zeros_(self.network[-1].weight);nn.init.zeros_(self.network[-1].bias)

    def forward(self,u,c):
        _matrix(u,160,'frozen u160');_matrix(c,640,'frozen c640')
        if len(c)!=len(u) or c.device!=u.device or u.device!=self.network[0].weight.device:raise ValueError('Aligned features and network must share a device')
        with torch.autocast(device_type=u.device.type,enabled=False):
            baseline=_unit(u.detach());contributions=_unit(c.detach())
            z=_unit(baseline+self.network(contributions))
        return _matrix(z,160,'open unit160')


def _metadata(z,rx,day,y=None):
    _matrix(z,160,'open features');n=len(z)
    metadata=(('rx',rx),('day',day))+((('y',y),) if y is not None else ())
    for name,a in metadata:
        if not torch.is_tensor(a) or a.dtype!=torch.int64 or a.shape!=(n,) or a.device!=z.device:raise ValueError('Aligned source integer metadata required: '+name)
    if bool((~torch.isin(rx,torch.tensor([1,3,4,6,8],device=z.device))).any()) or bool((~torch.isin(day,torch.tensor([1,2,3],device=z.device))).any()):
        raise ValueError('Only public original source RX/day labels allowed')
    if y is not None and bool(((y<0)|(y>=6)).any()):raise ValueError('Only visible source-L old-six labels allowed')


def _candidates(z,rx,day):
    _,cells=torch.unique(torch.stack((rx,day),dim=1),dim=0,return_inverse=True)
    cell_count=int(cells.max())+1;onehot=F.one_hot(cells,num_classes=cell_count).to(z.dtype)
    valid=(rx[:,None]!=rx[None])&(day[:,None]!=day[None])
    counts=valid.to(z.dtype)@onehot
    if bool(valid.sum(dim=1).eq(0).any()):raise ValueError('Every anchor needs cross-RX AND cross-day candidates')
    logweight=-counts[:,cells].clamp_min(1).log()
    logits=(_unit(z)@_unit(z).T/TEMPERATURE+logweight).masked_fill(~valid,-torch.inf)
    return valid,onehot,logweight,logits


def _average(values,groups):
    _,indices=torch.unique(groups,dim=0,return_inverse=True);size=int(indices.max())+1
    total=torch.zeros(size,device=values.device,dtype=values.dtype).scatter_add(0,indices,values)
    return (total/torch.bincount(indices,minlength=size).to(values.dtype)).mean()


def supervised_metric(z,y,rx,day):
    _metadata(z,rx,day,y)
    with torch.autocast(device_type=z.device.type,enabled=False):
        valid,onehot,_,logits=_candidates(z,rx,day);positive=valid&(y[:,None]==y[None])
        counts=positive.to(z.dtype)@onehot;domains=(counts>0).sum(dim=1)
        if bool(domains.eq(0).any()):raise ValueError('Each source-L anchor needs a cross-domain same-class positive')
        cells=onehot.argmax(1)
        target=positive.to(z.dtype)/counts[:,cells].clamp_min(1)/domains[:,None]
        logtarget=target.clamp_min(torch.finfo(z.dtype).tiny).log()
        logstudent=logits.log_softmax(dim=1).masked_fill(~valid,0.)
        per=(target*(logtarget-logstudent)).sum(dim=1)
        loss=_average(per,torch.stack((y,rx,day),dim=1))
    return dict(loss=loss,anchors=len(z),positive_pairs=int(positive.sum()),anchor_cells=int(torch.unique(torch.stack((y,rx,day),1),dim=0).shape[0]))


def oe_structure(z,teacher,rx,day):
    _metadata(z,rx,day);_matrix(teacher,160,'frozen OE teacher')
    if teacher.shape!=z.shape or teacher.device!=z.device:raise ValueError('Aligned frozen teacher features required')
    with torch.autocast(device_type=z.device.type,enabled=False):
        valid,_,logweight,student=_candidates(z,rx,day);u=_unit(teacher.detach())
        logits=(u@u.T/TEMPERATURE+logweight).masked_fill(~valid,-torch.inf)
        probability=logits.softmax(1);logteacher=logits.log_softmax(1).masked_fill(~valid,0.)
        logstudent=student.log_softmax(1).masked_fill(~valid,0.)
        per=(probability*(logteacher-logstudent)).sum(dim=1)
        loss=_average(per,torch.stack((rx,day),dim=1))
    return dict(loss=loss,anchors=len(z),anchor_cells=int(torch.unique(torch.stack((rx,day),1),dim=0).shape[0]))
