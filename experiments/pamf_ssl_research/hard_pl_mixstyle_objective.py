"""One raw U student forward with transient, identity-protected style hooks."""
from contextlib import contextmanager
import torch
from experiments.cric_dura_comparison.f_model import PREFIX
from .hard_pl_objective import HardPLObjective
from .u_mixstyle_objective import protected_partners,mix_moments


@contextmanager
def style_hooks(backbone,prefix,recipe,partner,valid,lam,*,force=False,capture=None):
    observed=set();handles=[];applied={}
    try:
        for name in recipe['levels']:
            def transform(module,inputs,output,name=name):
                if name in observed or not torch.is_tensor(output) or len(output)!=len(partner):
                    raise ValueError('Exactly one aligned U feature tensor per declared style level required')
                observed.add(name)
                enabled=force or bool(torch.rand((),device=output.device)<recipe['probability_per_level'])
                value=mix_moments(output,partner,valid,lam,recipe['eps']) if enabled else output
                applied[name]=int(valid.sum()) if enabled else 0
                if capture is not None:
                    value.retain_grad();capture[name]=value
                return value
            handles.append(backbone.get_submodule(prefix+name).register_forward_hook(transform))
        yield applied
        if observed!=set(recipe['levels']):raise ValueError('Actual shallow/middle style hooks missing')
    finally:
        for handle in handles:handle.remove()


class HardPLMixStyleObjective(HardPLObjective):
    def __init__(self,c):
        super().__init__(c)
        self.style_calls=self.style_seen=self.style_eligible=0
        self.style_applied={name:0 for name in c['hard_pl_mixstyle']['levels']}
        self.style_active=False

    def loss(self,student,device):
        value=super().loss(student,device)
        self.metrics.update(HPL_MS_calls=self.style_calls,HPL_MS_seen=self.style_seen,HPL_MS_eligible=self.style_eligible,
            **{'HPL_MS_applied_'+name:v for name,v in self.style_applied.items()})
        return value

    def unlabeled_student_forward(self,student,x,domain,pseudo,accepted,metadata):
        if self.style_active:raise ValueError('Nested U style context forbidden')
        rx=metadata['rx_i'].to(x.device).long();n=len(x);recipe=self.c['hard_pl_mixstyle']
        p=torch.nn.functional.one_hot(pseudo,6).to(x.dtype)
        partner,valid=protected_partners(p,accepted.to(x.dtype),rx,self.c['source_rxs'],1.)
        if bool((valid&(~accepted|~accepted[partner]|(pseudo!=pseudo[partner])|(rx==rx[partner]))).any()):
            raise ValueError('Actual style partner violates accepted pseudo identity or source RX')
        alpha=torch.tensor(recipe['alpha'],device=x.device)
        lam=torch.distributions.Beta(alpha,alpha).sample((n,1,1))
        capture={} if self.c.get('hard_pl_smoke') else None;self.style_active=True
        try:
            with style_hooks(student,PREFIX,recipe,partner,valid,lam,
                             force=bool(self.c.get('hard_pl_smoke')),capture=capture) as applied:
                logits=student.forward_identity_only(x,domain_labels=domain)['tx_logits']
        finally:self.style_active=False
        self.style_calls+=1;self.style_seen+=n;self.style_eligible+=int(valid.sum())
        for name,v in applied.items():self.style_applied[name]+=v
        if capture is not None:self.style_smoke_maps=capture
        return logits

    def state_dict(self):
        value=super().state_dict()
        if (self.style_active or self.style_calls!=self.calls or self.style_seen!=self.seen
            or self.style_eligible>self.accepted or any(v>self.style_eligible for v in self.style_applied.values())):
            raise ValueError('Actual U style counters or transient-hook lifetime differ')
        value['U_style']=dict(recipe=self.c['hard_pl_mixstyle'],calls=self.style_calls,seen=self.style_seen,
            eligible=self.style_eligible,applied=dict(self.style_applied),persistent_hooks=False,
            teacher_or_L_style_calls=0,partner_TX_truth_reads=0)
        return value
