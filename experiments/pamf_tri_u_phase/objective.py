"""Matched source-U phase intervention; no target data or U labels."""
from contextlib import contextmanager
import torch
from torch.nn import functional as F
from experiments.pamf_branch_training.relation import capture_branches
from experiments.pamf_branch_training.objective import _actual_paths
from experiments.pamf_physics_branch.objective import select_u_candidates
from experiments.pamf_repl_view_shuffle.source import installed as original_tri
from experiments.pamf_ssl_research.hard_pl_receiver_extrapolation_objective import ReceiverExtrapolation
from experiments.pamf_ssl_research.hard_pl_guard_objective import Guarded


def rotate_iq(x,angle):
    if x.ndim!=3 or x.shape[1]!=2 or angle.shape!=(len(x),) or angle.device!=x.device:
        raise ValueError('Aligned IQ and one phase per packet required')
    co,si=angle.cos()[:,None],angle.sin()[:,None]
    return torch.stack((co*x[:,0]-si*x[:,1],si*x[:,0]+co*x[:,1]),dim=1)


def branch_consistency(teacher,student,accepted):
    if not teacher or teacher.keys()!=student.keys() or accepted.dtype!=torch.bool:
        raise ValueError('Aligned branches and boolean acceptance required')
    losses=[]
    for name in teacher:
        a,b=teacher[name],student[name]
        if a.shape!=b.shape or a.ndim!=2 or len(a)!=len(accepted):raise ValueError('Aligned branch batches required')
        difference=1-(F.normalize(a.detach(),dim=1,eps=1e-8)*F.normalize(b,dim=1,eps=1e-8)).sum(1)
        losses.append((difference*accepted.to(difference.dtype)).mean())
    return torch.stack(losses).mean()


def phase_branches(student,x,angles,accepted):
    modes=[(m,m.training) for m in student.modules()]
    try:
        student.eval();core,path,reference=_actual_paths(student)
        with torch.no_grad():
            with capture_branches(core,path,reference) as teacher:student(x)
        with capture_branches(core,path,reference) as changed:student(rotate_iq(x,angles))
        return branch_consistency(teacher,changed,accepted)
    finally:
        for module,training in modes:module.training=training


class PhaseU(ReceiverExtrapolation):
    def __init__(self,c):
        super().__init__(c);self._phase_batch=None
        self.phase_calls=self.phase_selected=self.phase_accepted=0

    def validate_unlabeled_batch(self,batch):
        super().validate_unlabeled_batch(batch);visible=batch[3]['tx_label_visible']
        if visible.dtype!=torch.bool or visible.shape!=batch[1].shape or bool(visible.any()):
            raise ValueError('Phase consistency requires hidden-label source U')

    def unlabeled_student_forward(self,student,x,domain,pseudo,accepted,metadata):
        first=self._rxe_forward_index==0
        result=super().unlabeled_student_forward(student,x,domain,pseudo,accepted,metadata)
        if first:
            if self._phase_batch is not None:raise ValueError('One source U packet required')
            self._phase_batch=(x.detach(),accepted.detach())
        return result

    def loss(self,student,device):
        if self._phase_batch is not None:raise ValueError('Unconsumed phase U packet')
        try:
            base=super().loss(student,device);x,accepted=self._phase_batch
            step=student._feature_l_steps+self.c.get('step_offset',0);recipe=self.c['phase_relation']
            selected=select_u_candidates(accepted,recipe['selected'],202611116+2*step)
            generator=torch.Generator(device='cpu').manual_seed(202611118+2*step)
            angles=((2*torch.rand(len(selected),generator=generator)-1)*recipe['max_phase_radians']).to(x.device,x.dtype)
            mask=accepted.index_select(0,selected)
            extra=phase_branches(student,x.index_select(0,selected),angles,mask)
            self.phase_calls+=1;self.phase_selected+=len(selected);self.phase_accepted+=int(mask.sum())
            self.metrics.update(HPL_phase_U_loss=extra.detach(),HPL_phase_U_calls=self.phase_calls,
                                HPL_phase_U_selected=self.phase_selected,HPL_phase_U_accepted=self.phase_accepted)
            return base+recipe['weight']*extra
        finally:self._phase_batch=None

    def state_dict(self):
        s=super().state_dict()
        if self._phase_batch is not None or self.phase_calls!=self.calls:raise ValueError('One phase loss per U objective required')
        s['U_phase_relation']=dict(calls=self.phase_calls,selected=self.phase_selected,accepted=self.phase_accepted,
                                   extra_eval_forwards=2*self.phase_calls,U_truth_read=False)
        return s


class GuardedPhaseU(Guarded,PhaseU):pass


@contextmanager
def installed(c,holder):
    from experiments.pamf_repl_completion import objectives
    previous=objectives.objective;objectives.objective=GuardedPhaseU
    try:
        with original_tri(c,holder,enabled=True):yield
    finally:objectives.objective=previous
