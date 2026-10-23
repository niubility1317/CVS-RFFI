"""Keep hard pseudo targets/losses;replace only unreliable simulated IQ with exact raw IQ."""
import torch
from .hard_pl_joint_objective import JointNoAsMixStyle
from .hard_pl_guard_objective import Guarded


def view_mask(probabilities,pseudo,accepted,threshold):
    if (probabilities.ndim!=2 or probabilities.shape!=(len(pseudo),6) or accepted.shape!=pseudo.shape
        or pseudo.dtype!=torch.long or accepted.dtype!=torch.bool or not bool(torch.isfinite(probabilities).all())):
        raise ValueError('Finite source As probabilities and exact raw pseudo/acceptance masks required')
    confidence,label=probabilities.max(1)
    return accepted&(confidence>=threshold)&(label==pseudo)


class ReliableAs(JointNoAsMixStyle):
    def __init__(self,c):
        super().__init__(c)
        self.ras_calls=self.ras_generated=self.ras_teacher=self.ras_used=self.ras_fallback=self.ras_raw_accepted=0
        self.ras_confident=self.ras_agree=0;self.ras_active=False

    def mixed_view(self,student,x,ordinary,physical,metadata,epoch,step,pseudo,accepted):
        if self.ras_active or self.style_active or self.joint_active:raise ValueError('As teacher must precede transient student feature/style contexts')
        n=len(x);selected={k:v[physical] if torch.is_tensor(v) and v.ndim and len(v)==n else v for k,v in metadata.items()}
        raw=x[physical];self.ras_active=True;was=student.training
        snapshots={name:buf.detach().clone() for name,buf in student.named_buffers()
            if name.endswith(('running_mean','running_var','num_batches_tracked'))} if self.c.get('hard_pl_smoke') else {}
        try:
            with torch.no_grad():
                simulated=student._cd_practical(raw,selected,epoch,step)
                if simulated.shape!=raw.shape or not bool(torch.isfinite(simulated).all()):raise ValueError('Finite aligned native source As IQ required')
                student.eval();q=student(simulated).softmax(1)
                used=view_mask(q,pseudo[physical],accepted[physical],self.c['hard_pl_reliable_as']['augmented_view_gate'])
                replacement=raw.clone();replacement[used]=simulated[used]
                mixed=ordinary.clone();mixed[physical]=replacement
        finally:student.train(was);self.ras_active=False
        if snapshots and any(not torch.equal(dict(student.named_buffers())[name],value) for name,value in snapshots.items()):
            raise ValueError('Extra current-student eval must not update normalization buffers')
        if not torch.equal(mixed[physical][~used],raw[~used]) or not torch.equal(mixed[physical][used],simulated[used]):
            raise ValueError('As eligibility must retain exact raw fallback and selected simulated IQ')
        self.ras_calls+=1;self.ras_generated+=len(raw);self.ras_teacher+=len(raw);self.ras_used+=int(used.sum())
        self.ras_fallback+=int((~used).sum());self.ras_raw_accepted+=int(accepted[physical].sum())
        self.ras_confident+=int((q.max(1).values>=.95).sum());self.ras_agree+=int((q.argmax(1)==pseudo[physical]).sum())
        if self.c.get('hard_pl_smoke'):
            self.smoke_ras_used=used;self.smoke_ras_pseudo=pseudo[physical].detach().clone()
            self.smoke_ras_raw_accepted=accepted[physical].detach().clone();self.smoke_ras_q=q.detach().clone()
            self.smoke_ras_exact_fallback=True;self.smoke_ras_eval_norm_unchanged=True;self.smoke_ras_training_restored=student.training==was
        return mixed

    def loss(self,student,device):
        value=super().loss(student,device)
        self.physical_student_views=2*self.ras_used;self.raw_As_seen=self.ras_fallback
        self.metrics.update(HPL_physical_student_views=self.physical_student_views,HPL_raw_As_seen=self.raw_As_seen,
            HPL_RAS_calls=self.ras_calls,HPL_RAS_generated=self.ras_generated,HPL_RAS_teacher_As=self.ras_teacher,
            HPL_RAS_used=self.ras_used,HPL_RAS_fallback=self.ras_fallback,HPL_RAS_raw_accepted_half=self.ras_raw_accepted,
            HPL_RAS_confident=self.ras_confident,HPL_RAS_agree=self.ras_agree,HPL_RAS_view_threshold=.95,
            HPL_RAS_all_raw_accepted_retained=1.,HPL_RAS_CE_normalizer=1.25)
        return value

    def state_dict(self):
        state=super().state_dict()
        if (self.ras_active or self.ras_calls!=self.calls or self.ras_generated!=self.seen//2
            or self.ras_teacher!=self.ras_generated or self.physical_seen!=self.ras_generated
            or self.ras_used+self.ras_fallback!=self.ras_generated or not 0<=self.ras_used<=self.ras_raw_accepted<=self.ras_generated
            or self.physical_student_views!=2*self.ras_used or self.raw_As_seen!=self.ras_fallback):
            raise ValueError('Source As generation/teacher/actual use/exactraw fallback counts differ')
        state['U_reliable_As']=dict(recipe=self.c['hard_pl_reliable_as'],calls=self.ras_calls,generated=self.ras_generated,
            teacher_As_views=self.ras_teacher,used=self.ras_used,raw_fallback=self.ras_fallback,raw_accepted_half=self.ras_raw_accepted,
            view_confident=self.ras_confident,view_argmax_agree=self.ras_agree,all_raw_accepted_retained=True,
            extra_student_forwards=0,new_head=False,EMA_updates=0,U_truth_read=False,teacher_eval_norm_updates=0)
        return state


class GuardedReliableAs(Guarded,ReliableAs):pass
