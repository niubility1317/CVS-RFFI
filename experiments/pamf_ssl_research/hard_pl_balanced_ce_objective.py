"""Reweight CE arithmetic on six reused logits; keep every original accepted sample."""
import math
import torch
from torch.nn import functional as F
from .source_balanced_pseudo_ce import contribution_weights
from .hard_pl_native_as_objective import NativeAs
from .hard_pl_guard_objective import Guarded


def ce_reallocation(logits,pseudo,accepted,weights):
    if (len(logits)!=6 or any(z.ndim!=2 or z.shape!=logits[0].shape or z.shape[0]!=len(pseudo) for z in logits)
        or pseudo.ndim!=1 or pseudo.dtype!=torch.long or accepted.dtype!=torch.bool or accepted.shape!=pseudo.shape
        or weights.shape!=pseudo.shape or weights.requires_grad or not bool(torch.isfinite(weights).all())):
        raise ValueError('Six aligned original U logits and detached accepted contribution weights required')
    if bool(((weights<.5-1e-12)|(weights>1.5+1e-12)).any()) or not torch.equal(weights[~accepted],torch.ones_like(weights[~accepted])):
        raise ValueError('Bounded positive CE weights and neutral rejected entries required')
    if abs(float(weights[accepted].double().sum())-int(accepted.sum()))>1e-8*max(1,int(accepted.sum())):
        raise ValueError('Mean accepted CE contribution must remain one')
    ces=[F.cross_entropy(z,pseudo,reduction='none',label_smoothing=.01) for z in logits]
    packet=.25*(ces[0]+ces[1])+.25*(ces[2]+ces[3])+.125*(ces[4]+ces[5])
    mask=accepted.to(packet.dtype);w=weights.to(packet.dtype)
    before=(packet*mask).mean()/1.25;after=(packet*mask*w).mean()/1.25
    delta=(packet*mask*(w-1)).mean()/1.25
    return delta,before,after


class BalancedCE(NativeAs):
    def __init__(self,c):
        super().__init__(c);self.bce_calls=self.bce_captures=self.bce_packets=self.bce_accepted=0
        self.bce_mass=0.;self.bce_min=1.;self.bce_max=1.;self.bce_max_dose_error=0.
        self._bce_logits=self._bce_current=None

    def unlabeled_student_forward(self,student,x,domain,pseudo,accepted,metadata):
        if self._bce_logits is None:raise ValueError('Contribution capture escaped current U loss')
        rx=metadata['rx_i'].to(x.device).long()
        if self._bce_current is None:self._bce_current=(pseudo,accepted,rx)
        elif any(not torch.equal(a,b) for a,b in zip(self._bce_current,(pseudo,accepted,rx))):
            raise ValueError('All six views must retain exactly the same raw pseudo labels,acceptance and RX')
        logits=super().unlabeled_student_forward(student,x,domain,pseudo,accepted,metadata)
        self._bce_logits.append(logits);return logits

    def loss(self,student,device):
        if self._bce_logits is not None:raise ValueError('Nested CE contribution context forbidden')
        self._bce_logits=[];self._bce_current=None
        try:
            value=super().loss(student,device)
            if len(self._bce_logits)!=6 or self._bce_current is None:raise ValueError('Six existing U logits required')
            pseudo,accepted,rx=self._bce_current;weights=contribution_weights(pseudo,accepted,rx)
            delta,before,after=ce_reallocation(self._bce_logits,pseudo,accepted,weights)
            value=value+self.settings['weight']*delta
            a=int(accepted.sum());mass=float(weights[accepted].sum());error=abs(mass-a)
            self.bce_calls+=1;self.bce_captures+=6;self.bce_packets+=6*len(pseudo);self.bce_accepted+=a;self.bce_mass+=mass
            self.bce_max_dose_error=max(self.bce_max_dose_error,error)
            if a:self.bce_min=min(self.bce_min,float(weights[accepted].min()));self.bce_max=max(self.bce_max,float(weights[accepted].max()))
            self.metrics.update(HPL_loss=value.detach(),HPL_base_loss=(value/self.settings['weight']).detach(),
                HPL_BCE_calls=self.bce_calls,HPL_BCE_captures=self.bce_captures,HPL_BCE_packets=self.bce_packets,
                HPL_BCE_extra_CE_kernels=6*self.bce_calls,HPL_BCE_accepted=self.bce_accepted,HPL_BCE_mass=self.bce_mass,
                HPL_BCE_min=self.bce_min,HPL_BCE_max=self.bce_max,HPL_BCE_max_dose_error=self.bce_max_dose_error,
                HPL_BCE_before=before.detach(),HPL_BCE_after=after.detach(),HPL_BCE_delta=delta.detach(),HPL_BCE_CE_normalizer=1.25)
            if self.c.get('hard_pl_smoke'):
                self.smoke_bce_weights=weights;self.smoke_bce_accepted=accepted;self.smoke_bce_pseudo=pseudo;self.smoke_bce_rx=rx
                self.smoke_bce_logits=list(self._bce_logits);self.smoke_bce_delta=delta.detach()
                self.smoke_bce_before=before.detach();self.smoke_bce_after=after.detach()
            return value
        finally:self._bce_logits=self._bce_current=None

    def state_dict(self):
        s=super().state_dict()
        if (self._bce_logits is not None or self._bce_current is not None or self.bce_calls!=self.calls
            or self.bce_captures!=6*self.calls or self.bce_packets!=6*self.seen or self.bce_accepted!=self.accepted
            or not math.isfinite(self.bce_mass) or abs(self.bce_mass-self.accepted)>1e-7*max(1,self.accepted)
            or self.bce_min<.5-1e-12 or self.bce_max>1.5+1e-12 or self.bce_max_dose_error>1e-6):
            raise ValueError('Bounded CE contribution counts,dose or transient context differs')
        s['U_balanced_CE']=dict(recipe=self.c['hard_pl_balanced_ce'],calls=self.bce_calls,captures=self.bce_captures,
            captured_packet_views=self.bce_packets,extra_CE_kernels=6*self.bce_calls,accepted=self.bce_accepted,
            accepted_weight_mass=self.bce_mass,minimum_weight=self.bce_min,maximum_weight=self.bce_max,
            max_batch_dose_error=self.bce_max_dose_error,all_raw_accepted_retained=True,raw_acceptance_changed=False,
            extra_teacher=0,extra_backbone=0,extra_head=0,new_head=False,persistent_context=False,
            EMA_updates=0,prototype_updates=0,U_truth_reads=0,target_reads=0,CE_dose_normalizer=1.25)
        return s


class GuardedBalancedCE(Guarded,BalancedCE):pass
