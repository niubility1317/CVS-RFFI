import torch
from experiments.pamf_tri_u_coherence_m1.objective import attenuated
from experiments.pamf_tri_u_coherence_cal_m1.objective import scales
from experiments.pamf_ssl_research.hard_pl_guard_objective import Guarded
from experiments.pamf_ssl_research.hard_pl_receiver_extrapolation_objective import ReceiverExtrapolation


def select_mask(raw,stable,random_control,seed):
    if raw.dtype!=torch.bool or stable.dtype!=torch.bool or raw.shape!=stable.shape:
        raise ValueError('Aligned raw and stable masks required')
    stable=raw&stable
    if not random_control:return stable
    available=raw.nonzero(as_tuple=True)[0];count=int(stable.sum())
    generator=torch.Generator(device='cpu').manual_seed(seed)
    order=torch.randperm(len(available),generator=generator)[:count].to(available.device)
    result=torch.zeros_like(raw);result[available[order]]=True
    return result


class ConsensusU(ReceiverExtrapolation):
    def __init__(self,c):
        super().__init__(c);self.gate_calls=self.offered=self.raw_accepted=self.consensus_accepted=0
        self.scale_sum=self.coherence_sum=0.

    def validate_unlabeled_batch(self,batch):
        super().validate_unlabeled_batch(batch);visible=batch[3]['tx_label_visible']
        if visible.dtype!=torch.bool or visible.shape!=batch[1].shape or bool(visible.any()):
            raise ValueError('Source U identity labels must remain hidden')

    def accept_pseudo(self,student,x,confidence,pseudo):
        raw=super().accept_pseudo(student,x,confidence,pseudo);recipe=self.c['consensus_recipe']
        modes=[(m,m.training) for m in student.modules()]
        try:
            student.eval()
            with torch.no_grad():
                response=student.id_backbone.encoder.response(x)
                if response.ndim!=2 or response.shape[1]!=29:raise ValueError('Actual reference response required')
                coherence=response[:,-4];scale=scales(coherence,recipe['mode'],.25)
                with attenuated(student,scale,recipe['frequency']):
                    weak_conf,weak_pseudo=student(x).softmax(1).max(1)
                stable=(weak_pseudo==pseudo)&(weak_conf>=recipe['weak_threshold'])
                step=student._feature_l_steps+self.c.get('step_offset',0)
                selected=select_mask(raw,stable,recipe['random_control'],202611132+2*step)
        finally:
            for module,training in modes:module.training=training
        self.gate_calls+=1;self.offered+=len(x);self.raw_accepted+=int(raw.sum());self.consensus_accepted+=int(selected.sum())
        self.scale_sum+=float(scale.sum());self.coherence_sum+=float(coherence.sum())
        return selected

    def state_dict(self):
        s=super().state_dict()
        if self.gate_calls!=self.calls or self.offered!=self.seen or not 0<=self.consensus_accepted<=self.raw_accepted<=self.offered:
            raise ValueError('Actual U consensus counts differ')
        s['U_phase_relation']=dict(calls=self.gate_calls,selected=self.offered,accepted=self.consensus_accepted,
            extra_eval_forwards=self.gate_calls,U_truth_read=False,raw_accepted=self.raw_accepted,
            scale_sum=self.scale_sum,coherence_sum=self.coherence_sum,intervention='raw_and_perturbed_prediction_consensus')
        return s


class GuardedConsensusU(Guarded,ConsensusU):pass
