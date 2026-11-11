"""Balance source-U pseudo classes and rank by physical branch sensitivity."""
import torch
from experiments.pamf_tri_u_coherence_m1.objective import attenuated
from experiments.pamf_tri_u_coherence_cal_m1.objective import scales
from experiments.pamf_ssl_research.hard_pl_guard_objective import Guarded
from experiments.pamf_ssl_research.hard_pl_receiver_extrapolation_objective import ReceiverExtrapolation


def js_divergence(full,view):
    if full.ndim!=2 or full.shape!=view.shape or not bool(torch.isfinite(full).all() and torch.isfinite(view).all()):
        raise ValueError('Finite aligned teacher logits required')
    p=full.softmax(1);q=view.softmax(1);m=.5*(p+q)
    return .5*((p*(p.clamp_min(1e-30).log()-m.clamp_min(1e-30).log())).sum(1)
               +(q*(q.clamp_min(1e-30).log()-m.clamp_min(1e-30).log())).sum(1))


def balanced_mask(raw,pseudo,risk,cap):
    if (raw.dtype!=torch.bool or pseudo.dtype!=torch.long or raw.ndim!=1 or raw.shape!=pseudo.shape
        or risk.shape!=raw.shape or not bool(torch.isfinite(risk).all()) or cap not in (1,2)):
        raise ValueError('Aligned pseudo classes, source acceptance and finite ranks required')
    result=torch.zeros_like(raw)
    if not bool(raw.any()):return result
    classes,counts=torch.unique(pseudo[raw],return_counts=True)
    limit=int(counts.min())*cap
    for cls in classes:
        indices=(raw & (pseudo==cls)).nonzero(as_tuple=True)[0]
        keep=min(limit,len(indices));order=torch.argsort(risk[indices],stable=True)[:keep]
        result[indices[order]]=True
    return result


class BalancedU(ReceiverExtrapolation):
    def __init__(self,c):
        super().__init__(c);self.gate_calls=self.offered=self.raw_accepted=self.balanced_accepted=0
        self.coherence_sum=self.scale_sum=self.risk_sum=0.;self.raw_classes=[0]*6;self.kept_classes=[0]*6

    def validate_unlabeled_batch(self,batch):
        super().validate_unlabeled_batch(batch);visible=batch[3]['tx_label_visible']
        if visible.dtype!=torch.bool or visible.shape!=batch[1].shape or bool(visible.any()):
            raise ValueError('Source U identity labels must remain hidden')

    def accept_pseudo(self,student,x,confidence,pseudo):
        raw=super().accept_pseudo(student,x,confidence,pseudo);recipe=self.c['balanced_recipe']
        modes=[(m,m.training) for m in student.modules()]
        try:
            student.eval()
            with torch.no_grad():
                response=student.id_backbone.encoder.response(x)
                if response.ndim!=2 or response.shape[1]!=29:raise ValueError('Actual reference response required')
                coherence=response[:,-4];scale=scales(coherence,recipe['mode'],.25)
                full=student(x)
                if full.ndim!=2 or full.shape[1]!=6:raise ValueError('Six deployed identity logits required')
                with attenuated(student,scale,recipe['frequency']):view=student(x)
                discrepancy=js_divergence(full,view)
                if recipe['rank']=='confidence':risk=-confidence
                elif recipe['rank']=='physical':risk=discrepancy
                elif recipe['rank']=='random':
                    step=student._feature_l_steps+self.c.get('step_offset',0)
                    generator=torch.Generator(device='cpu').manual_seed(202611157+2*step)
                    risk=torch.rand(len(x),generator=generator).to(x.device)
                else:raise ValueError('Unregistered pseudo ranking')
                selected=balanced_mask(raw,pseudo,risk,recipe['cap'])
        finally:
            for module,training in modes:module.training=training
        self.gate_calls+=1;self.offered+=len(x);self.raw_accepted+=int(raw.sum());self.balanced_accepted+=int(selected.sum())
        self.scale_sum+=float(scale.sum());self.coherence_sum+=float(coherence.sum());self.risk_sum+=float(discrepancy.sum())
        for cls in range(6):
            self.raw_classes[cls]+=int((raw & (pseudo==cls)).sum())
            self.kept_classes[cls]+=int((selected & (pseudo==cls)).sum())
        return selected

    def state_dict(self):
        s=super().state_dict()
        if (self.gate_calls!=self.calls or self.offered!=self.seen or not 0<=self.balanced_accepted<=self.raw_accepted<=self.offered
            or sum(self.raw_classes)!=self.raw_accepted or sum(self.kept_classes)!=self.balanced_accepted):
            raise ValueError('Actual balanced U counts differ')
        s['U_phase_relation']=dict(calls=self.gate_calls,selected=self.offered,accepted=self.balanced_accepted,
            extra_eval_forwards=2*self.gate_calls,U_truth_read=False,raw_accepted=self.raw_accepted,
            scale_sum=self.scale_sum,coherence_sum=self.coherence_sum,risk_sum=self.risk_sum,
            raw_classes=self.raw_classes,kept_classes=self.kept_classes,intervention='class_balanced_physical_rank')
        return s


class GuardedBalancedU(Guarded,BalancedU):pass
