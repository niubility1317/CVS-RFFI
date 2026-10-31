"""Exact differentiable receiver-held-out updates on legal source banks."""
from __future__ import annotations
import math
import numpy as np
import torch
from torch.func import functional_call
from .source_style_core import NativeOpen, SINC_PREFIX, HEAD_WEIGHT
from . import star_crossset_metric_model as metric
from .star_metric_residual_model import _metadata, _unit, _average

RECEIVERS=(1,3,4,6,8)
SEEDS=dict(train_L=2026103180,train_OE=2026103181,held_L=2026103182,held_OE=2026103183)


def _rows(values,rx,day,y=None):
    if (values.ndim!=1 or values.dtype!=np.int64 or rx.shape!=values.shape or rx.dtype!=np.int64
            or day.shape!=values.shape or day.dtype!=np.int64 or set(rx.tolist())!=set(RECEIVERS)
            or set(day.tolist())!={1,2,3}):
        raise ValueError('Complete source-five-RX/three-day integer metadata required')
    if y is not None and (y.shape!=values.shape or y.dtype!=np.int64 or set(y.tolist())!=set(range(6))):
        raise ValueError('Only legal source-L six-class labels required')


class ReceiverSampler:
    """Matched source streams; stratified coverage prevents empty RX/day sets."""
    def __init__(self,L,OE):
        lm,om=L['metadata'],OE['metadata']
        if set(lm)-{'y','rx_i','day_i','domain'} or set(om)-{'rx_i','day_i','domain'}:
            raise ValueError('OE has only public RX/day/domain; no OE class labels')
        self.lrx=np.asarray(lm['rx_i']);self.lday=np.asarray(lm['day_i']);self.y=np.asarray(lm['y'])
        self.orx=np.asarray(om['rx_i']);self.oday=np.asarray(om['day_i'])
        _rows(np.arange(len(self.y),dtype=np.int64),self.lrx,self.lday,self.y)
        _rows(np.arange(len(self.orx),dtype=np.int64),self.orx,self.oday)
        self.L={(r,c,d):np.flatnonzero((self.lrx==r)&(self.y==c)&(self.lday==d)) for r in RECEIVERS for c in range(6) for d in (1,2,3)}
        self.OE={(r,d):np.flatnonzero((self.orx==r)&(self.oday==d)) for r in RECEIVERS for d in (1,2,3)}
        if any(not len(v) for v in (*self.L.values(),*self.OE.values())):
            raise ValueError('Every known TX/RX/day and public OE RX/day cell must be nonempty')
        self.rng={k:np.random.default_rng(v) for k,v in SEEDS.items()}

    def _L(self,r,count,rng):
        # Cover all six old identities and all three days on every RX.
        selected=[int(rng.choice(self.L[r,c,d])) for c in range(6) for d in (1,2,3)]
        pool=np.flatnonzero(self.lrx==r)
        selected.extend(rng.choice(pool,count-len(selected),replace=True).tolist())
        return np.asarray(selected,np.int64)

    def _OE(self,r,count,rng):
        selected=[int(rng.choice(self.OE[r,d])) for d in (1,2,3)]
        pool=np.flatnonzero(self.orx==r)
        selected.extend(rng.choice(pool,count-len(selected),replace=True).tolist())
        return np.asarray(selected,np.int64)

    def sample(self,step):
        if type(step) is not int or step<1:raise ValueError('Positive source step required')
        held=RECEIVERS[(step-1)%len(RECEIVERS)];train=[r for r in RECEIVERS if r!=held]
        return dict(held_rx=held,
            train_L=np.concatenate([self._L(r,32,self.rng['train_L']) for r in train]),
            train_OE=np.concatenate([self._OE(r,16,self.rng['train_OE']) for r in train]),
            held_L=self._L(held,64,self.rng['held_L']),
            held_OE=self._OE(held,32,self.rng['held_OE']))


def _selection(rows,n,device):
    if rows is None:return torch.arange(n,device=device)
    if not torch.is_tensor(rows) or rows.dtype!=torch.int64 or rows.ndim!=1 or not len(rows) or rows.device!=device or bool(((rows<0)|(rows>=n)).any()):
        raise ValueError('Nonempty legal source anchor indices required')
    return rows


def anchor_losses(z_L,z_OE,y_L,rx_L,day_L,rx_OE,day_OE,teacher_OE,*,L_rows=None,OE_rows=None):
    """Old cross-RX/day objectives, selecting anchors but retaining all candidates.

    Held anchors can share one RX because their candidate banks include the
    other source RXs. Candidate student features remain differentiable.
    """
    _metadata(z_L,rx_L,day_L,y_L);_metadata(z_OE,rx_OE,day_OE)
    if teacher_OE.shape!=z_OE.shape or teacher_OE.dtype!=z_OE.dtype or teacher_OE.device!=z_OE.device or not bool(torch.isfinite(teacher_OE).all()):
        raise ValueError('Aligned frozen source-OE feature teacher required')
    a=_selection(L_rows,len(z_L),z_L.device);o=_selection(OE_rows,len(z_OE),z_OE.device)
    lvalid,onehot,cells,lweight,_=metric._set_prior(rx_L[a],day_L[a],rx_L,day_L,z_L.dtype)
    ovalid,_,_,oweight,_=metric._set_prior(rx_L[a],day_L[a],rx_OE,day_OE,z_L.dtype)
    known,outside=_unit(z_L),_unit(z_OE)
    logits=torch.cat((known[a]@known.T/metric.TEMPERATURE+lweight,known[a]@outside.T/metric.TEMPERATURE+oweight),1)
    valid=torch.cat((lvalid,ovalid),1);positive=lvalid&(y_L[a,None]==y_L[None])
    counts=positive.to(z_L.dtype)@onehot;domains=(counts>0).sum(1)
    if bool(domains.eq(0).any()):raise ValueError('Each L anchor needs same-class cross-RX/day positives')
    target_L=positive.to(z_L.dtype)/counts[:,cells].clamp_min(1)/domains[:,None]
    target=torch.cat((target_L,torch.zeros_like(ovalid,dtype=z_L.dtype)),1)
    logstudent=logits.masked_fill(~valid,-torch.inf).log_softmax(1).masked_fill(~valid,0.)
    per=(target*(target.clamp_min(torch.finfo(z_L.dtype).tiny).log()-logstudent)).sum(1)
    crossset=_average(per,torch.stack((y_L[a],rx_L[a],day_L[a]),1))
    _,groups=torch.unique(torch.stack((rx_OE,day_OE),1),dim=0,return_inverse=True)
    onehot=torch.nn.functional.one_hot(groups,num_classes=int(groups.max())+1).to(z_OE.dtype)
    valid=(rx_OE[o,None]!=rx_OE[None])&(day_OE[o,None]!=day_OE[None])
    if bool(valid.sum(1).eq(0).any()):raise ValueError('Each OE anchor needs cross-RX/day candidates')
    counts=valid.to(z_OE.dtype)@onehot;logweight=-counts[:,groups].clamp_min(1).log()
    teacher=_unit(teacher_OE.detach())
    student=(outside[o]@outside.T/metric.TEMPERATURE+logweight).masked_fill(~valid,-torch.inf)
    logits=(teacher[o]@teacher.T/metric.TEMPERATURE+logweight).masked_fill(~valid,-torch.inf)
    per=(logits.softmax(1)*(logits.log_softmax(1).masked_fill(~valid,0.)-student.log_softmax(1).masked_fill(~valid,0.))).sum(1)
    topology=_average(per,torch.stack((rx_OE[o],day_OE[o]),1))
    return dict(crossset_metric=crossset,OE_KL=topology,L_anchors=len(a),OE_anchors=len(o))


def bilevel_loss(model,train_objective,held_objective,*,alpha=.001,beta=.5):
    """F(theta)+beta G(theta-alpha grad F), retaining the complete Hessian path."""
    if (not isinstance(alpha,(int,float)) or not math.isfinite(alpha) or alpha<0
            or not isinstance(beta,(int,float)) or not math.isfinite(beta) or beta<0):
        raise ValueError('Finite nonnegative meta step and outer weight required')
    all_parameters=dict(model.named_parameters());active={n:p for n,p in all_parameters.items() if p.requires_grad}
    if not active:raise ValueError('Trainable native source parameters required')
    train=train_objective(model)
    if not torch.is_tensor(train) or train.ndim or not bool(torch.isfinite(train)):raise FloatingPointError('Finite scalar source train loss required')
    gradients=torch.autograd.grad(train,tuple(active.values()),create_graph=True,retain_graph=True,allow_unused=True)
    virtual=dict(all_parameters);unused=[];norm2=train.new_zeros(())
    for (name,parameter),gradient in zip(active.items(),gradients):
        if gradient is None:unused.append(name);continue
        if not bool(torch.isfinite(gradient).all()):raise FloatingPointError('Nonfinite virtual source gradient: '+name)
        virtual[name]=parameter-alpha*gradient;norm2=norm2+gradient.detach().double().square().sum()
    # Include nonpersistent buffers in functional state; no optimizer or state
    # loading is used for the virtual update and none of its tensors is detached.
    buffers={n:v.detach().clone() for n,v in model.named_buffers()}
    def call(*args,**kwargs):return functional_call(model,(virtual,buffers),args,kwargs,strict=True)
    held=held_objective(call);total=train+beta*held
    if not torch.is_tensor(held) or held.ndim or not bool(torch.isfinite(total)):raise FloatingPointError('Finite scalar held-out meta loss required')
    diagnostics=dict(full_second_order=True,second_order=True,virtual_updates_committed=False,
        second_order_contribution_active=alpha!=0.,meta_alpha=float(alpha),meta_beta=float(beta),
        inner_gradient_norm=float(norm2.sqrt().cpu()),virtual_delta_norm=float(alpha*norm2.sqrt().cpu()),
        unused_inner_parameters=unused,unused_inner_parameter_count=len(unused),
        trainable_parameter_tensors=len(active),frozen_parameter_tensors=len(all_parameters)-len(active))
    return dict(total=total,train=train,held=held,diagnostics=diagnostics)
