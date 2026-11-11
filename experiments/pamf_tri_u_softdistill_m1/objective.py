import torch
from torch.nn import functional as F
from experiments.pamf_physics_branch.objective import select_u_candidates
from experiments.pamf_tri_u_coherence_m1.objective import attenuated
from experiments.pamf_tri_u_coherence_cal_m1.objective import scales
from experiments.pamf_ssl_research.hard_pl_guard_objective import Guarded
from experiments.pamf_ssl_research.hard_pl_receiver_extrapolation_objective import ReceiverExtrapolation


def masked_loss(logits,teacher,accepted,temperature):
    if logits.shape!=teacher.shape or logits.ndim!=2 or accepted.shape!=logits.shape[:1] or accepted.dtype!=torch.bool:
        raise ValueError('Aligned soft teacher and source-U acceptance required')
    p=(teacher.detach()/temperature).softmax(1)
    kl=F.kl_div((logits/temperature).log_softmax(1),p,reduction='none').sum(1)*temperature**2
    return (kl*accepted).mean()


class SoftDistillU(ReceiverExtrapolation):
    def __init__(self,c):
        super().__init__(c);self._soft_teacher=None;self._soft_batch=None;self._raw_input=None
        self.soft_calls=self.soft_selected=self.soft_accepted=0;self.coherence_sum=self.scale_sum=0.

    def validate_unlabeled_batch(self,batch):
        super().validate_unlabeled_batch(batch);visible=batch[3]['tx_label_visible']
        if visible.dtype!=torch.bool or visible.shape!=batch[1].shape or bool(visible.any()):
            raise ValueError('Source U identity labels must remain hidden')

    def accept_pseudo(self,student,x,confidence,pseudo):
        if self._soft_teacher is not None or self._raw_input is not None:raise ValueError('Unconsumed soft teacher')
        modes=[(m,m.training) for m in student.modules()]
        try:
            student.eval()
            with torch.no_grad():self._soft_teacher=student(x).detach();self._raw_input=x.detach()
        finally:
            for module,training in modes:module.training=training
        return super().accept_pseudo(student,x,confidence,pseudo)

    def unlabeled_student_forward(self,student,x,domain,pseudo,accepted,metadata):
        first=self._rxe_forward_index==0
        result=super().unlabeled_student_forward(student,x,domain,pseudo,accepted,metadata)
        if first:
            if self._soft_batch is not None:raise ValueError('One U packet required')
            self._soft_batch=(x.detach(),domain.detach(),accepted.detach())
        return result

    def loss(self,student,device):
        if self._soft_batch is not None or self._soft_teacher is not None or self._raw_input is not None:raise ValueError('Unconsumed soft packet')
        try:
            base=super().loss(student,device);_,domain,accepted=self._soft_batch;x=self._raw_input;teacher=self._soft_teacher
            step=student._feature_l_steps+self.c.get('step_offset',0);recipe=self.c['soft_recipe'];arm=self.c['soft_branch']
            indices=select_u_candidates(accepted,recipe['selected'],202611136+2*step)
            x=x.index_select(0,indices);domain=domain.index_select(0,indices)
            accepted=accepted.index_select(0,indices);teacher=teacher.index_select(0,indices)
            modes=[(m,m.training) for m in student.modules()]
            try:
                student.eval()
                with torch.no_grad():
                    response=student.id_backbone.encoder.response(x)
                    if response.ndim!=2 or response.shape[1]!=29:raise ValueError('Actual reference response required')
                    coherence=response[:,-4];scale=scales(coherence,arm['mode'],.25)
                with attenuated(student,scale,arm['frequency']):
                    logits=student(x)
                extra=masked_loss(logits,teacher,accepted,recipe['temperature'])
            finally:
                for module,training in modes:module.training=training
            self.soft_calls+=1;self.soft_selected+=len(indices);self.soft_accepted+=int(accepted.sum())
            self.coherence_sum+=float(coherence.sum());self.scale_sum+=float(scale.sum())
            self.metrics.update(HPL_phase_U_loss=extra.detach(),HPL_soft_scale_mean=scale.mean().detach())
            return base+arm['weight']*extra
        finally:self._soft_batch=self._soft_teacher=self._raw_input=None

    def state_dict(self):
        s=super().state_dict()
        if self._soft_batch is not None or self._soft_teacher is not None or self._raw_input is not None or self.soft_calls!=self.calls:
            raise ValueError('One soft branch view per U loss')
        s['U_phase_relation']=dict(calls=self.soft_calls,selected=self.soft_selected,accepted=self.soft_accepted,
            extra_eval_forwards=2*self.soft_calls,U_truth_read=False,intervention='soft_branch_logit_distillation',
            coherence_sum=self.coherence_sum,scale_sum=self.scale_sum)
        return s


class GuardedSoftDistillU(Guarded,SoftDistillU):pass
