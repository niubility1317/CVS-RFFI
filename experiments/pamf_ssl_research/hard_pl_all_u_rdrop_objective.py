"""Retain StepNoCap and add label-free RDrop for previously rejected U only."""
import torch
from .hard_pl_step_guard_objective import StepGuardedNativeAs
from .rejected_rdrop import RECIPE,rejected_rdrop


class AllURDrop(StepGuardedNativeAs):
    def __init__(self,c):
        super().__init__(c)
        if c.get('hard_pl_all_u_rdrop')!=RECIPE:raise ValueError('Fixed rejected-U RDrop recipe required')
        self.allrd_calls=self.allrd_seen=self.allrd_rejected=self.allrd_nonphysical_rejected=0
        self.allrd_pairs=self.allrd_prediction_views=0;self.allrd_current=None

    def unlabeled_student_forward(self,student,x,domain,pseudo,accepted,metadata):
        if self.allrd_current is None:raise ValueError('RDrop capture escaped the current U objective')
        value=super().unlabeled_student_forward(student,x,domain,pseudo,accepted,metadata)
        self.allrd_current.append((value,pseudo,accepted));return value

    def loss(self,student,device):
        if self.allrd_current is not None:raise ValueError('One complete current U objective required')
        self.allrd_current=[]
        try:
            zero=super().loss(student,device)
            if self.pending_gradient_loss is None or len(self.allrd_current)!=6:raise ValueError('Complete original U loss and six existing logits required')
            logits=[v[0] for v in self.allrd_current];pseudo=self.allrd_current[0][1];accepted=self.allrd_current[0][2]
            if any(not torch.equal(y,pseudo) or not torch.equal(a,accepted) for _,y,a in self.allrd_current):
                raise ValueError('All U views must keep the same original raw pseudo labels and acceptance')
            step=student._feature_l_steps+self.c.get('step_offset',0)
            physical=(torch.arange(len(accepted),device=accepted.device)+step).remainder(2).bool()
            extra,counts=rejected_rdrop(logits,accepted,physical)
            self.pending_gradient_loss=self.pending_gradient_loss+self.settings['weight']*extra
            self.allrd_calls+=1;self.allrd_seen+=len(accepted);self.allrd_rejected+=counts['rejected_U']
            self.allrd_nonphysical_rejected+=counts['rejected_nonphysical_half'];self.allrd_pairs+=counts['rejected_RDrop_pairs'];self.allrd_prediction_views+=counts['reused_prediction_views']
            self.metrics.update(HPL_loss=self.pending_gradient_loss.detach(),HPL_base_loss=(self.pending_gradient_loss/self.settings['weight']).detach(),
                HPL_ALLRD_calls=self.allrd_calls,HPL_ALLRD_seen=self.allrd_seen,HPL_ALLRD_rejected=self.allrd_rejected,
                HPL_ALLRD_nonphysical_rejected=self.allrd_nonphysical_rejected,HPL_ALLRD_pairs=self.allrd_pairs,
                HPL_ALLRD_prediction_views=self.allrd_prediction_views,HPL_ALLRD_extra_KL_evaluations=3*self.allrd_calls,
                HPL_ALLRD_extra_base_loss=extra.detach())
            if self.c.get('hard_pl_smoke'):
                self.smoke_allrd_counts=counts;self.smoke_allrd_logits=logits
                self.smoke_allrd_mask=accepted.detach().clone();self.smoke_allrd_physical=physical.detach().clone()
                self.smoke_allrd_extra=extra.detach();self.smoke_allrd_extra_requires_grad=extra.requires_grad
            return zero
        finally:self.allrd_current=None

    def state_dict(self):
        state=super().state_dict()
        if (self.allrd_current is not None or self.allrd_calls!=self.calls or self.allrd_seen!=self.seen
                or self.allrd_rejected!=self.seen-self.accepted or not 0<=self.allrd_nonphysical_rejected<=self.allrd_rejected
                or self.allrd_pairs!=2*self.allrd_rejected+self.allrd_nonphysical_rejected or self.allrd_prediction_views!=2*self.allrd_pairs):
            raise ValueError('Exactly one complete rejected-U consistency objective and original mask per source step required')
        state['U_all_RDrop']=dict(recipe=self.c['hard_pl_all_u_rdrop'],calls=self.allrd_calls,seen=self.allrd_seen,
            rejected=self.allrd_rejected,nonphysical_rejected=self.allrd_nonphysical_rejected,pairs=self.allrd_pairs,
            reused_prediction_views=self.allrd_prediction_views,extra_KL_evaluations=3*self.allrd_calls,
            hard_label_or_CE_on_rejected=False,raw_acceptance_changed=False,physical_As_KL_weight=0.,pending_logits=False,
            extra_teacher=0,extra_student=0,extra_head=0,new_head=False,U_truth_reads=0,target_reads=0)
        return state
