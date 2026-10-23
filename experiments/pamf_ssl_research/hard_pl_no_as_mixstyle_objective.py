"""One detached style plan per U batch,shared by all six supervised U passes."""
import torch
from experiments.cric_dura_comparison.f_model import PREFIX
from .hard_pl_mixstyle_objective import HardPLMixStyleObjective,style_hooks
from .u_mixstyle_objective import protected_partners


class NoAsMixStyleObjective(HardPLMixStyleObjective):
    def __init__(self,c):
        super().__init__(c);self.expected_style_forwards=6;self.shared_plan=None;self.style_plan_calls=0

    def unlabeled_student_forward(self,student,x,domain,pseudo,accepted,metadata):
        if self.style_active:raise ValueError('Nested source style context forbidden')
        n=len(x);recipe=self.c['hard_pl_mixstyle']
        if self.style_calls%6==0:
            rx=metadata['rx_i'].to(x.device).long();p=torch.nn.functional.one_hot(pseudo,6).to(x.dtype)
            partner,valid=protected_partners(p,accepted.to(x.dtype),rx,self.c['source_rxs'],1.)
            if bool((valid&(~accepted|~accepted[partner]|(pseudo!=pseudo[partner])|(rx==rx[partner]))).any()):
                raise ValueError('Shared style plan violates accepted-only pseudo identity or source RX')
            alpha=torch.tensor(recipe['alpha'],device=x.device)
            lam=torch.distributions.Beta(alpha,alpha).sample((n,1,1))
            enabled={name:bool(torch.rand((),device=x.device)<recipe['probability_per_level']) for name in recipe['levels']}
            self.shared_plan=(partner,valid,lam,enabled);self.style_plan_calls+=1
        partner,valid,lam,enabled=self.shared_plan
        if len(partner)!=n:raise ValueError('Six aligned U forward views must share one sample plan')
        capture={} if self.c.get('hard_pl_smoke') else None;self.style_active=True
        try:
            with style_hooks(student,PREFIX,recipe,partner,valid,lam,force=bool(self.c.get('hard_pl_smoke')),
                             capture=capture,enabled_by_level=enabled,prepend=True) as applied:
                logits=student.forward_identity_only(x,domain_labels=domain)['tx_logits']
        finally:self.style_active=False
        self.style_calls+=1;self.style_seen+=n;self.style_eligible+=int(valid.sum())
        for name,v in applied.items():self.style_applied[name]+=v
        if capture is not None:self.style_smoke_maps=capture
        return logits

    def loss(self,student,device):
        value=super().loss(student,device)
        self.metrics['HPL_MS_plan_calls']=self.style_plan_calls
        return value

    def state_dict(self):
        value=super().state_dict()
        if self.style_calls%6 or self.style_plan_calls!=self.calls:
            raise ValueError('Exactly one shared style plan per completed six-pass U objective required')
        value['U_style'].update(shared_six_forward_plan=True,plan_calls=self.style_plan_calls,
            contrast_captures_after_style=True)
        return value
