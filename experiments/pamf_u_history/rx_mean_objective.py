"""Ordinary raw PL plus same-base soft consistency under a source RX mean shift."""
from contextlib import contextmanager
from types import MethodType
import numpy as np
import torch
from torch.nn import functional as F
from experiments.cric_dura_comparison.io import read
from .covariance_objective import Exposure as BaseExposure,SourceViews,schedule,combine
from .rx_mean_views import MeanBank,Shift,RECIPE as VIEW_RECIPE

METHODS=('RXMean-U','RXMean-LU')
RECIPE=dict(view=VIEW_RECIPE,positive_threshold=.95,positive_weight=1.,U_scope='all_hidden_U',
    base_schedule='new_U_epoch_even_raw_odd_H',H_schedule='existing_covariance_replica_scene_schedule',
    teacher='current_student_eval_detached_same_base',student='eval_with_gradient_no_dropout',
    L_pairs=64,L_base='same_raw_or_As_native_input',scope_scalar='(nU*U+nL*L)/sum;detach_L_in_U',
    ramp_updates=444,coefficient_rule='min(1,.1*median(primary_norm)/max_scope_median(aux_norm));12train_batches_6raw6H,noV',
    EMA_updates=0,new_inference_parameters=0,gradient_calibration_seed=2026102741)


def soft_loss(logits,probabilities):
    if logits.shape!=probabilities.shape or logits.ndim!=2 or logits.shape[1]!=6 or probabilities.requires_grad:raise ValueError('Six-class detached same-base probability target required')
    return F.kl_div(logits.log_softmax(1),probabilities,reduction='batchmean')


class Exposure(BaseExposure):
    def __init__(self,metadata):
        super().__init__(metadata);self.raw_seen=self.H_seen=0;self.styles={};self.L_seen=0;self.nonzero_U=0

    def state(self):
        s=super().state()
        for key in ('covariance_U_seen','covariance_L_pairs'):s.pop(key)
        s.update(mean_shift_U_seen=self.seen,mean_shift_L_pairs=self.L_seen,raw_base_U=self.raw_seen,H_base_U=self.H_seen,
            student_U_packets=2*self.seen,teacher_U_packets=self.seen+self.H_seen,extra_L_teacher_packets=self.L_seen,
            extra_L_student_packets=self.L_seen,nonzero_shift_U=self.nonzero_U,by_RX_style=self.styles,stored_training_statistics=20)
        return s


def load_bank(c):
    from .rx_mean_source import RUN
    from pathlib import Path
    root=Path(c['project'])/'runs'/RUN/'mean_bank';done=read(root/'complete.json')
    if (done['status']!='RX_MEAN_BANK_FROZEN' or done['recipe']!=VIEW_RECIPE or done['source_contract']!=read(c['source_contract'])
        or done['U_seen']!=56700 or done['model_seed'] is not None or done['V_reads'] or done['target_reads'] or done['F_G_weight_reads']):raise ValueError('Original fullU-only response means required')
    with np.load(root/'bank.npz',allow_pickle=False) as a:return MeanBank(a['means'])


class Objective:
    def __init__(self,c):
        self.c=c;self.history=None;self.cache=None;self.shift=None;self.last={};self.candidates={}

    @contextmanager
    def observe(self,student):
        self.inputs={};self.L_forwards=0;original=student.forward_identity_only
        def forward(model,x,*args,**kwargs):
            if model.training and len(x)==128:
                self.L_forwards+=1
                if self.L_forwards in (2,4):self.inputs[{2:'H',4:'raw'}[self.L_forwards]]=x.detach()
            return original(x,*args,**kwargs)
        student.forward_identity_only=MethodType(forward,student)
        try:yield
        finally:student.forward_identity_only=original

    def plain(self,student,device):
        data=student._cd_U_loader.dataset
        if data.role!='U_s' or 'y' in data.metadata:raise ValueError('Hidden sourceU only')
        if self.history is None:self.history=Exposure(data.metadata)
        if self.cache is None:self.cache=SourceViews(self.c,data)
        if self.shift is None:self.shift=Shift(load_bank(self.c),device)
        try:batch=next(student._cd_U_iter)
        except StopIteration:student._cd_U_iter=iter(student._cd_U_loader);batch=next(student._cd_U_iter)
        self.ids=self.history.indices(batch,data.role);self.batch=batch;x=batch[0].to(device)
        student.eval()
        with torch.no_grad():self.raw_prob=student(x).softmax(1);confidence,pseudo=self.raw_prob.max(1)
        student.train();self.accepted=confidence>=.95
        logits=student.forward_identity_only(x,domain_labels=batch[2].to(device))['tx_logits']
        student._cd_U_seen+=len(x);student._cd_U_accepted+=int(self.accepted.sum())
        value=(F.cross_entropy(logits,pseudo,reduction='none')*self.accepted).mean()
        self.last=dict(RXM_PL_loss=value.detach(),RXM_raw_PL_accepted=int(self.accepted.sum()))
        return value

    def extra(self,student,kwargs):
        if self.L_forwards!=5 or set(self.inputs)!={'H','raw'}:raise ValueError('Original five extraL input forwards required')
        call=student._feature_l_steps+self.c.get('step_offset',0)-4440
        if call!=self.history.calls+1:raise ValueError('Continuous own E20+10+20 source trajectory required')
        mode=self.c.get('_calibration_base_override') or ('raw' if ((call-1)//222)%2==0 else 'H')
        if mode not in ('raw','H') or ('_calibration_base_override' in self.c and not self.c['rx_mean_calibration']):raise ValueError('Fixed raw/H schedule required')
        replica,scene=schedule(call);device=kwargs['labels'].device
        base=self.batch[0].to(device) if mode=='raw' else self.cache.take(self.batch[3],replica,scene,device)
        shifted,styles=self.shift(base,self.batch[3]['rx_i'],self.ids,call)
        mask=(torch.arange(128,device=device)+student._feature_l_steps).remainder(2)!=0
        lbase=self.inputs[mode][mask];lrx=kwargs['extra'][1]['rx_i'].to(device)[mask]
        lshift,lstyles=self.shift(lbase,lrx,torch.arange(64,device=device),call)
        student.eval()
        try:
            with torch.no_grad():
                up=self.raw_prob if mode=='raw' else student(base).softmax(1)
                lp=student(lbase).softmax(1)
            ul=student(shifted);ll=student(lshift)
        finally:student.train()
        u=soft_loss(ul,up);l=soft_loss(ll,lp);n=len(base)
        methods=METHODS if self.c['rx_mean_calibration'] else (self.c['paper_method'],)
        losses={m:combine(u,l,n,64,'LU' if m=='RXMean-LU' else 'U') for m in methods}
        if self.c['rx_mean_calibration']:self.calibration_nodes=(ul,ll)
        self.candidates=losses if self.c['rx_mean_calibration'] else {}
        h=self.history;h.calls+=1;h.seen+=n;h.accepted+=int(self.accepted.sum());h.visits[self.ids]+=1;h.L_seen+=64
        h.raw_seen+=n*(mode=='raw');h.H_seen+=n*(mode=='H');h.by_replica[replica-1]+=n*(mode=='H');h.by_scene[scene]+=n*(mode=='H')
        h.nonzero_U+=int((shifted.detach()-base).flatten(1).abs().amax(1).gt(0).sum())
        for rx in self.batch[3]['rx_i'].unique().tolist():
            chosen=self.batch[3]['rx_i']==rx;r=h.by_rx.setdefault(str(rx),dict(seen=0,accepted=0,mean_shift=0))
            r['seen']+=int(chosen.sum());r['mean_shift']+=int(chosen.sum());r['accepted']+=int((chosen.to(device)&self.accepted).sum())
            for style in styles.unique().tolist():
                count=int((chosen.to(device)&(styles==style)).sum())
                if count:h.styles[str(rx)+'->'+str(style)]=h.styles.get(str(rx)+'->'+str(style),0)+count
        value=losses[self.c['paper_method']];weight=self.c['rx_mean_coefficient']*min(call/444.,1.)
        self.last.update(RXM_loss=value.detach(),RXM_U_loss=u.detach(),RXM_L_loss=l.detach(),RXM_weight=weight,RXM_seen=h.seen,
            RXM_calls=h.calls,RXM_H_base=int(mode=='H'),RXM_L_pairs=h.L_seen,RXM_teacher_U_seen=h.seen+h.H_seen,
            RXM_nonzero_shift_U=h.nonzero_U,**{'RXM_'+k:v for k,v in self.shift.diagnostic.items()})
        return weight*value
