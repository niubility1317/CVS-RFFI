"""Detached physical branch-view teacher ensemble for the original Tri U loss."""
from contextlib import contextmanager
import torch
from torch.nn import functional as F
from experiments.pamf_branch_training.objective import _actual_paths
from experiments.pamf_physics_branch.objective import select_u_candidates
from experiments.pamf_tri_u_coherence_cal_m1.objective import scales
from experiments.pamf_ssl_research.hard_pl_guard_objective import Guarded
from experiments.pamf_ssl_research.hard_pl_receiver_extrapolation_objective import ReceiverExtrapolation


def ensemble_kl(logits,full,view,accepted,temperature,mix):
    if (logits.ndim!=2 or full.shape!=logits.shape or view.shape!=logits.shape
        or accepted.shape!=logits.shape[:1] or accepted.dtype!=torch.bool
        or temperature<=0 or not 0<=mix<=1):
        raise ValueError('Aligned detached teacher distributions and source-U mask required')
    target=(1-mix)*(full.detach()/temperature).softmax(1)+mix*(view.detach()/temperature).softmax(1)
    kl=F.kl_div((logits/temperature).log_softmax(1),target,reduction='none').sum(1)*temperature**2
    return (kl*accepted).mean()


@contextmanager
def attenuated_teacher(student,scale,branch):
    core,_,reference=_actual_paths(student)
    if branch not in ('ref','ref_freq','ref_pa_main'):raise ValueError('Registered physical branch required')
    modules=[reference]
    if branch=='ref_freq':modules.extend((core.f_proj,core.freq_stats_proj))
    if branch=='ref_pa_main':modules.append(core.pa_proj)
    handles=[]
    def hook(module,inputs,output):
        if not isinstance(output,torch.Tensor) or output.ndim!=2 or len(output)!=len(scale):
            raise ValueError('Aligned physical branch vector required')
        return output*scale.to(output.dtype).unsqueeze(1)
    try:
        handles=[module.register_forward_hook(hook) for module in modules]
        yield
    finally:
        for handle in handles:handle.remove()


class TeacherEnsembleU(ReceiverExtrapolation):
    def __init__(self,c):
        super().__init__(c);self._teacher=None;self._student=None
        self.teacher_calls=self.teacher_selected=self.teacher_accepted=0;self.coherence_sum=self.scale_sum=0.

    def validate_unlabeled_batch(self,batch):
        super().validate_unlabeled_batch(batch);visible=batch[3]['tx_label_visible']
        if visible.dtype!=torch.bool or visible.shape!=batch[1].shape or bool(visible.any()):
            raise ValueError('Source U identity labels must remain hidden')

    def accept_pseudo(self,student,x,confidence,pseudo):
        if self._teacher is not None:raise ValueError('Unconsumed teacher packet')
        arm=self.c['teacher_branch'];modes=[(m,m.training) for m in student.modules()]
        try:
            student.eval()
            with torch.no_grad():
                response=student.id_backbone.encoder.response(x)
                if response.ndim!=2 or response.shape[1]!=29:raise ValueError('Actual reference response required')
                coherence=response[:,-4];scale=scales(coherence,arm['mode'],.25)
                full=student(x)
                with attenuated_teacher(student,scale,arm['branch']):view=student(x)
                self._teacher=(full.detach(),view.detach(),coherence.detach(),scale.detach())
        finally:
            for module,training in modes:module.training=training
        return super().accept_pseudo(student,x,confidence,pseudo)

    def unlabeled_student_forward(self,student,x,domain,pseudo,accepted,metadata):
        first=self._rxe_forward_index==0
        logits=super().unlabeled_student_forward(student,x,domain,pseudo,accepted,metadata)
        if first:
            if self._student is not None:raise ValueError('One original U student view required')
            self._student=(logits,accepted.detach())
        return logits

    def loss(self,student,device):
        if self._teacher is not None or self._student is not None:raise ValueError('Unconsumed U packet')
        try:
            base=super().loss(student,device)
            logits,accepted=self._student;full,view,coherence,scale=self._teacher
            step=student._feature_l_steps+self.c.get('step_offset',0);recipe=self.c['teacher_recipe'];arm=self.c['teacher_branch']
            indices=select_u_candidates(accepted,recipe['selected'],202611146+2*step)
            logits=logits.index_select(0,indices);accepted=accepted.index_select(0,indices)
            full=full.index_select(0,indices);view=view.index_select(0,indices)
            coherence=coherence.index_select(0,indices);scale=scale.index_select(0,indices)
            extra=ensemble_kl(logits,full,view,accepted,recipe['temperature'],arm['mix'])
            self.teacher_calls+=1;self.teacher_selected+=len(indices);self.teacher_accepted+=int(accepted.sum())
            self.coherence_sum+=float(coherence.sum());self.scale_sum+=float(scale.sum())
            self.metrics.update(HPL_phase_U_loss=extra.detach(),HPL_teacher_scale_mean=scale.mean().detach())
            return base+arm['weight']*extra
        finally:self._teacher=self._student=None

    def state_dict(self):
        s=super().state_dict()
        if self._teacher is not None or self._student is not None or self.teacher_calls!=self.calls:
            raise ValueError('One teacher ensemble per U loss')
        s['U_phase_relation']=dict(calls=self.teacher_calls,selected=self.teacher_selected,accepted=self.teacher_accepted,
            extra_eval_forwards=2*self.teacher_calls,U_truth_read=False,intervention='physical_branch_teacher_ensemble',
            coherence_sum=self.coherence_sum,scale_sum=self.scale_sum)
        return s


class GuardedTeacherEnsembleU(Guarded,TeacherEnsembleU):pass
