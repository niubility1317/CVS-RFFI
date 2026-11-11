"""Physically measurable per-packet response coherence controls U branch view."""
from contextlib import contextmanager
import torch
from torch.nn import functional as F
from experiments.pamf_branch_training.objective import _actual_paths
from experiments.pamf_physics_branch.objective import select_u_candidates
from experiments.pamf_ssl_research.hard_pl_guard_objective import Guarded
from experiments.pamf_ssl_research.hard_pl_receiver_extrapolation_objective import ReceiverExtrapolation


def scales(coherence,mode,min_scale):
    if coherence.ndim!=1 or not bool(torch.isfinite(coherence).all()) or mode not in ('adaptive','uniform'):
        raise ValueError('Finite per-packet coherence and registered mode required')
    adaptive=min_scale+(1-min_scale)*coherence.clamp(0,1)
    return adaptive if mode=='adaptive' else adaptive.mean().expand_as(adaptive)


@contextmanager
def attenuated(student,scale,frequency):
    core,_,reference=_actual_paths(student)
    modules=[reference]+([core.f_proj,core.freq_stats_proj] if frequency else [])
    handles=[]
    def hook(module,inputs,output):
        if not isinstance(output,torch.Tensor) or output.ndim!=2 or len(output)!=len(scale):
            raise ValueError('Actual per-packet PAMF branch vector required')
        return output*scale.to(output.dtype).unsqueeze(1)
    try:
        handles=[m.register_forward_hook(hook) for m in modules]
        yield
    finally:
        for h in handles:h.remove()


def masked_loss(logits,pseudo,accepted):
    if logits.ndim!=2 or pseudo.shape!=logits.shape[:1] or accepted.shape!=pseudo.shape or accepted.dtype!=torch.bool:
        raise ValueError('Aligned pseudo labels and acceptance required')
    return (F.cross_entropy(logits,pseudo,reduction='none',label_smoothing=.01)*accepted).mean()


class CoherenceU(ReceiverExtrapolation):
    def __init__(self,c):
        super().__init__(c);self._coherence_batch=None
        self.coherence_calls=self.coherence_selected=self.coherence_accepted=0
        self.coherence_sum=self.scale_sum=0.

    def validate_unlabeled_batch(self,batch):
        super().validate_unlabeled_batch(batch);visible=batch[3]['tx_label_visible']
        if visible.dtype!=torch.bool or visible.shape!=batch[1].shape or bool(visible.any()):
            raise ValueError('Source U identity labels must remain hidden')

    def unlabeled_student_forward(self,student,x,domain,pseudo,accepted,metadata):
        first=self._rxe_forward_index==0
        result=super().unlabeled_student_forward(student,x,domain,pseudo,accepted,metadata)
        if first:
            if self._coherence_batch is not None:raise ValueError('One U packet required')
            self._coherence_batch=(x.detach(),domain.detach(),pseudo.detach(),accepted.detach())
        return result

    def loss(self,student,device):
        if self._coherence_batch is not None:raise ValueError('Unconsumed U packet')
        try:
            base=super().loss(student,device);x,domain,pseudo,accepted=self._coherence_batch
            step=student._feature_l_steps+self.c.get('step_offset',0);recipe=self.c['coherence_recipe']
            indices=select_u_candidates(accepted,recipe['selected'],202611126+2*step)
            x=x.index_select(0,indices);domain=domain.index_select(0,indices)
            pseudo=pseudo.index_select(0,indices);accepted=accepted.index_select(0,indices)
            modes=[(m,m.training) for m in student.modules()]
            try:
                student.eval()
                with torch.no_grad():
                    response=student.id_backbone.encoder.response(x)
                    if response.ndim!=2 or response.shape[1]!=29:raise ValueError('Actual 29D reference response required')
                    coherence=response[:,-4];scale=scales(coherence,self.c['coherence_branch']['mode'],recipe['min_scale'])
                with attenuated(student,scale,self.c['coherence_branch']['frequency']):
                    logits=student.forward_identity_only(x,domain_labels=domain)['tx_logits']
                extra=masked_loss(logits,pseudo,accepted)
            finally:
                for module,training in modes:module.training=training
            self.coherence_calls+=1;self.coherence_selected+=len(indices);self.coherence_accepted+=int(accepted.sum())
            self.coherence_sum+=float(coherence.sum());self.scale_sum+=float(scale.sum())
            self.metrics.update(HPL_phase_U_loss=extra.detach(),HPL_coherence_mean=coherence.mean().detach(),
                                HPL_branch_scale_mean=scale.mean().detach())
            return base+recipe['weight']*extra
        finally:self._coherence_batch=None

    def state_dict(self):
        s=super().state_dict()
        if self._coherence_batch is not None or self.coherence_calls!=self.calls:raise ValueError('One U branch view per loss')
        s['U_phase_relation']=dict(calls=self.coherence_calls,selected=self.coherence_selected,
            accepted=self.coherence_accepted,extra_eval_forwards=self.coherence_calls,U_truth_read=False,
            intervention='response_coherence_guided_branch_attenuation',coherence_sum=self.coherence_sum,scale_sum=self.scale_sum)
        return s


class GuardedCoherenceU(Guarded,CoherenceU):pass
